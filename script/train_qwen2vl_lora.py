# -*- coding: utf-8 -*-
"""
Qwen2-VL LoRA finetune on multi-frame VQA (SageMaker/T4 friendly)
- Loads Unsloth FIRST for better memory & speed
- 4-bit base weights + LoRA (r=32 by default)
- Safer TrainingArguments (correct `evaluation_strategy`)
- Uniform frame picking across the clip (optionally random / first_k)
- Labels mask so loss only applies to the short answer letter
- Reproducibility: fixed seeds
- Optional CUDA memory fragmentation mitigation via env var

Run (example):
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train_qwen2vl_lora_from_frames_rewrite.py \
  --dataset_json data/train/train_100.json \
  --frames_root data/train_frames_448 \
  --output_dir outputs/qwen2vl7b_unsloth_poc \
  --epochs 2 --bsz 1 --grad_acc 16 \
  --max_images 3 --short_side 336 --num_workers 2 \
  --frame_picker uniform
"""

# --- Import order matters: import unsloth BEFORE transformers/peft ---
from unsloth import FastVisionModel  # noqa: E402

import os, json, glob, random
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from PIL import Image

from transformers import (
    Trainer, TrainingArguments, set_seed,
)
from peft import LoraConfig, get_peft_model

# ------------------------- Config constants -------------------------
LETTER_SET = list("ABCDEFGH")
DEFAULT_MODEL = "unsloth/Qwen2-VL-7B-Instruct"  # fast & 4-bit friendly

# ------------------------- Utils -------------------------
def extract_letter(ans: str) -> str:
    s = (ans or "").strip()
    if s and s[0] in LETTER_SET:
        return s[0]
    if s in LETTER_SET:
        return s
    return s[:1] if s else "A"


def resize_short_side(im: Image.Image, short_side: int = 448) -> Image.Image:
    if short_side <= 0:
        return im
    w, h = im.size
    if min(w, h) == short_side:
        return im
    if w < h:
        nw, nh = short_side, int(h * (short_side / w))
    else:
        nh, nw = short_side, int(w * (short_side / h))
    return im.resize((nw, nh), Image.BICUBIC)


def pick_first_k(paths: List[str], k: int) -> List[str]:
    return paths[:k]


def pick_uniform(paths: List[str], k: int) -> List[str]:
    if len(paths) <= k:
        return paths
    idxs = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]
    return [paths[i] for i in idxs]


def pick_random(paths: List[str], k: int, rng: random.Random) -> List[str]:
    if len(paths) <= k:
        return paths
    return rng.sample(paths, k)


# ------------------------- Dataset -------------------------
class FramesVQADataset(Dataset):
    def __init__(
        self,
        json_path: str,
        frames_root: str,
        processor,
        max_images: int = 4,
        short_side: int = 448,
        indices: List[int] | None = None,
        frame_picker: str = "uniform",
        seed: int = 42,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        all_items = js["data"]
        self.items = [all_items[i] for i in indices] if indices else all_items
        self.frames_root = frames_root
        self.processor = processor
        self.max_images = max_images
        self.short_side = short_side
        self.frame_picker = frame_picker
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def _load_images(self, video_path: str):
        vid_base = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.join(self.frames_root, vid_base)
        paths_all = sorted(glob.glob(os.path.join(folder, "f*.jpg")))
        # choose subset
        if self.max_images <= 0:
            chosen = []
        else:
            if self.frame_picker == "uniform":
                chosen = pick_uniform(paths_all, self.max_images)
            elif self.frame_picker == "random":
                chosen = pick_random(paths_all, self.max_images, self.rng)
            else:  # "first_k"
                chosen = pick_first_k(paths_all, self.max_images)

        imgs = []
        for p in chosen:
            try:
                im = Image.open(p).convert("RGB")
                im = resize_short_side(im, self.short_side)
                imgs.append(im)
            except Exception:
                continue

        if not imgs:
            imgs = [Image.new("RGB", (self.short_side, self.short_side), "white")]
        return imgs

    def __getitem__(self, idx):
        ex = self.items[idx]
        images = self._load_images(ex["video_path"])  # List[PIL.Image]

        q = (ex["question"] or "").strip()
        choices = "\n".join(ex.get("choices", []))
        instruction = "Trả lời CHỈ 1 ký tự (A/B/C/D/...). Không giải thích.\n"
        user_text = f"{instruction}Câu hỏi: {q}\nLựa chọn:\n{choices}\nĐáp án:"

        messages = [
            {
                "role": "user",
                "content": [
                    *({"type": "image", "image": img} for img in images),
                    {"type": "text", "text": user_text},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        label_letter = extract_letter(ex.get("answer", "A"))
        return {"prompt": prompt, "images": images, "label_text": label_letter}


# ------------------------- Collator -------------------------
@dataclass
class Collator:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts = [f["prompt"] for f in features]
        img_batch = [f["images"] for f in features]  # List[List[Image]]
        labels_txt = [f["label_text"] for f in features]

        # Append labels to the text so target tokens are at the end
        full_texts = [t + l for t, l in zip(texts, labels_txt)]
        batch = self.processor(text=full_texts, images=img_batch, return_tensors="pt", padding=True)

        # build label mask to ignore prompt tokens
        prompt_only = self.processor.tokenizer(
            texts, padding=True, return_tensors="pt", add_special_tokens=False
        )
        labels = batch["input_ids"].clone()

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

        for i in range(labels.size(0)):
            plen = (prompt_only["input_ids"][i] != pad_id).sum().item()
            labels[i, :plen] = -100
        batch["labels"] = labels
        return batch


# ------------------------- Model builder -------------------------
def build_model_and_processor(model_name: str = DEFAULT_MODEL, lora_r: int = 32):
    # Load 4-bit VLM in training mode
    model, processor = FastVisionModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=True,
        load_in_8bit=False,
        load_in_16bit=False,
        device_map="auto",
        use_gradient_checkpointing="unsloth",  # memory saver
        max_lora_rank=lora_r,  # exposes rank limit
        trust_remote_code=True,
        fast_inference=False,  # training mode
    )

    # Add LoRA to language blocks
    lora = LoraConfig(
        r=lora_r,
        lora_alpha=2 * lora_r,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.config.use_cache = False  # must disable for training

    return model, processor


# ------------------------- Main -------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--frames_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2vl7b_unsloth")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=8)

    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--short_side", type=int, default=384)
    parser.add_argument("--num_workers", type=int, default=2)

    # Eval memory knobs
    parser.add_argument("--frame_picker", type=str, default="uniform", choices=["uniform", "random", "first_k"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--lora_r", type=int, default=32)

    # Optional: smaller eval than train to avoid OOM
    parser.add_argument("--eval_max_images", type=int, default=None,
                        help="If set, use this many images for eval (<= max_images)")
    parser.add_argument("--eval_short_side", type=int, default=None,
                        help="If set, resize eval images to this short side (<= short_side)")

    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=400)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--evaluation_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])

    args = parser.parse_args()

    # Reproducibility
    set_seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Optional: reduce CUDA fragmentation if not set externally
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model, processor = build_model_and_processor(args.model_name, args.lora_r)

    # Load dataset to split indices
    with open(args.dataset_json, "r", encoding="utf-8") as f:
        total_items = len(json.load(f)["data"])

    idx = list(range(total_items))
    random.Random(args.seed).shuffle(idx)
    cut = int(len(idx) * 0.95)
    train_idx, val_idx = idx[:cut], idx[cut:]

    train_ds = FramesVQADataset(
        args.dataset_json,
        args.frames_root,
        processor,
        max_images=args.max_images,
        short_side=args.short_side,
        indices=train_idx,
        frame_picker=args.frame_picker,
        seed=args.seed,
    )
    val_ds = FramesVQADataset(
        args.dataset_json,
        args.frames_root,
        processor,
        max_images=(args.eval_max_images or args.max_images),
        short_side=(args.eval_short_side or args.short_side),
        indices=val_idx,
        frame_picker=args.frame_picker,
        seed=args.seed,
    )

    collate = Collator(processor)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        eval_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        bf16=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        report_to="none",
        remove_unused_columns=False,  # IMPORTANT for VLM
        gradient_checkpointing=True,  # redundant but explicit
        eval_accumulation_steps=1,
    )

    # Optional compute_metrics for eval accuracy (based on last labeled token)
    import numpy as np
    def compute_metrics_fn(eval_pred):
        preds, labels = eval_pred
        # HF sometimes returns tuple(preds) for CLM; take first
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        # preds: [bs, seq, vocab]
        # labels: [bs, seq]
        logits = np.asarray(preds)
        label_ids = np.asarray(labels)
        # get last supervised token index per sample
        accs = []
        for i in range(label_ids.shape[0]):
            mask = label_ids[i] != -100
            if not mask.any():
                continue
            last_idx = np.where(mask)[0][-1]
            pred_id = logits[i, last_idx].argmax(-1)
            true_id = label_ids[i, last_idx]
            accs.append(float(pred_id == true_id))
        acc = float(np.mean(accs)) if accs else 0.0
        return {"accuracy": acc}

    trainer = Trainer(
        model=model,
        args=targs,
        data_collator=collate,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.evaluation_strategy != "no" else None,
        compute_metrics=compute_metrics_fn if args.evaluation_strategy != "no" else None,
    )

    trainer.train()

    # Save LoRA adapter + processor
    adapter_dir = os.path.join(args.output_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
