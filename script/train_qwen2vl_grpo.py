# -*- coding: utf-8 -*-
"""
GRPO RL for Qwen2-VL-7B on multi-frame VQA (Python 3.12, T4 16GB)
- Unsloth FastVisionModel (fast_inference=True) => tối ưu cho Qwen2-VL
- Dataset đa khung hình: load frames từ data/train_frames_448/<video>/f*.{png|jpg|jpeg|webp}
- Reward shaping: <thinking>...</thinking> + <final>[A-H]</final> + correctness
- Tùy chọn nạp LoRA SFT (đúng kiến trúc Qwen2-VL). Nếu không có, sẽ RL từ base.

CÁCH CHẠY MẪU:
python scripts/train_qwen2vl_grpo.py \
  --dataset_json data/train/train_with_thinking.json \
  --frames_root data/train_frames_448 \
  --output_dir outputs/qwen2vl_rl \
  --run_name grpo_qwen2_ep1 \
  --epochs 1.0 --lr 5e-6 --num_generations 3 \
  --max_images 3 --short_side 384 --frame_picker uniform
# Nếu có LoRA SFT dành cho Qwen2-VL (không phải 2.5), thêm:
#   --sft_adapter /path/to/qwen2-vl-sft-lora
"""

# ==== ENV (tùy chọn) =========================================================
import os
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")  # nếu bạn suy luận bằng vLLM ở nơi khác
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ==== IMPORTS ================================================================
import os, json, glob, random, re
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from unsloth import FastVisionModel
from trl import GRPOConfig, GRPOTrainer

# ==== HẰNG SỐ ===============================================================
BASE_MODEL_2VL = "Qwen/Qwen2-VL-7B-Instruct"
LETTER_SET = list("ABCDEFGH")
THINK_S, THINK_E = "<thinking>", "</thinking>"
FINAL_S, FINAL_E = "<final>", "</final>"
IMG_PATTERNS = ("f*.png", "f*.jpg", "f*.jpeg", "f*.webp")

# ==== TIỆN ÍCH NHỎ ===========================================================
def extract_letter(ans: str) -> str:
    s = (ans or "").strip()
    if s and s[0] in LETTER_SET:
        return s[0]
    if s in LETTER_SET:
        return s
    return s[:1] if s else "A"


def resize_short_side(im: Image.Image, short_side: int = 384) -> Image.Image:
    if short_side <= 0:
        return im
    w, h = im.size
    if w == 0 or h == 0:
        return im
    if w < h:
        nw, nh = short_side, int(h * (short_side / w))
    else:
        nh, nw = short_side, int(w * (short_side / h))
    return im.resize((nw, nh), Image.BICUBIC)


def pick_first_k(paths: List[str], k: int) -> List[str]:
    return paths[:k]


def pick_uniform(paths: List[str], k: int) -> List[str]:
    if k <= 0:
        return []
    if len(paths) <= k:
        return paths
    idxs = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]
    return [paths[i] for i in idxs]


def pick_random(paths: List[str], k: int, rng: random.Random) -> List[str]:
    if len(paths) <= k:
        return paths
    return rng.sample(paths, k)


def _natural_key(p: str):
    # sắp xếp f00, f01, f10 đúng thứ tự
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]

# ==== DATASET ================================================================
class RLMultiFrameDataset(Dataset):
    """
    Đọc file JSON (train_with_thinking), tạo prompt đa-ảnh theo chat template của Qwen2-VL.
    Trả về:
      - prompt: str (đã apply_chat_template, add_generation_prompt=True)
      - images: List[PIL.Image] (K ảnh)   <-- CHUẨN cho Unsloth GRPO
      - answer: str (chữ cái gold)
      - choices_letters: str (tập chữ cái hợp lệ, VD 'ABCD')
    """
    def __init__(
        self,
        json_path: str,
        frames_root: str,
        tokenizer,
        max_images: int = 3,
        short_side: int = 384,
        frame_picker: str = "uniform",
        seed: int = 42,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        self.items = js["data"] if isinstance(js, dict) and "data" in js else js

        self.frames_root = frames_root
        self.tokenizer = tokenizer
        self.max_images = max_images
        self.short_side = short_side
        self.frame_picker = frame_picker
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def _load_images(self, video_path: str) -> List[Image.Image]:
        vid_base = os.path.splitext(os.path.basename(video_path))[0]
        folder = os.path.join(self.frames_root, vid_base)

        paths_all = []
        for pat in IMG_PATTERNS:
            paths_all.extend(glob.glob(os.path.join(folder, pat)))
        if not paths_all:
            return []

        paths_all = sorted(paths_all, key=_natural_key)

        if self.frame_picker == "random":
            chosen = pick_random(paths_all, self.max_images, self.rng)
        elif self.frame_picker == "first_k":
            chosen = pick_first_k(paths_all, self.max_images)
        else:
            chosen = pick_uniform(paths_all, self.max_images)

        imgs = []
        for p in chosen:
            try:
                im = Image.open(p).convert("RGB")
                im = resize_short_side(im, self.short_side)
                imgs.append(im)
            except Exception:
                continue
        return imgs

    def _format_choices(self, raw_choices: List[str]) -> str:
        return "\n".join(raw_choices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.items[idx]
        question = ex.get("question", "")
        choices = ex.get("choices", [])
        answer_letter = extract_letter(ex.get("answer", ""))
        video_path = ex.get("video_path", "")
        imgs = self._load_images(video_path)

        # Fallback để không bao giờ rỗng (tránh lỗi flatten images)
        if len(imgs) == 0:
            imgs = [Image.new("RGB", (self.short_side, self.short_side), (255, 255, 255))]

        # Nội dung văn bản sau các token <image>
        instruction = (
            "Bạn là trợ lý giao thông Việt Nam cho bài toán trắc nghiệm theo video nhiều khung hình.\n"
            f"Câu hỏi: {question}\n"
            f"Các lựa chọn:\n{self._format_choices(choices)}\n\n"
            f"Hãy suy luận ngắn gọn trong {THINK_S}...{THINK_E} "
            f"và CHỈ in DUY NHẤT 1 chữ cái đáp án (A–H) trong {FINAL_S}...{FINAL_E}.\n"
            "Không in thêm bất kỳ nội dung nào ngoài 2 khối trên."
        )

        content = [{"type": "image"} for _ in range(len(imgs))]
        content.append({"type": "text", "text": instruction})
        prompt_chat = [{"role": "user", "content": content}]

        prompt = self.tokenizer.apply_chat_template(
            prompt_chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        choice_letters = "".join([extract_letter(c) for c in choices if c])

        return {
            "prompt": prompt,
            "images": imgs,  # <-- CHỈ cần 1 list[PIL], KHÔNG bọc thêm
            "answer": answer_letter,
            "choices_letters": choice_letters,
        }

# ==== REWARDS ================================================================

def formatting_reward_func(completions, **kwargs):
    """+0.5 nếu có đúng 1 cặp <thinking>, +0.5 nếu có đúng 1 cặp <final>.
       -2.0 nếu có 'addCriterion' dày đặc (anti-gibberish safeguard)."""
    scores = []
    think_pat = rf"{re.escape(THINK_S)}(.*?){re.escape(THINK_E)}"
    final_pat = rf"{re.escape(FINAL_S)}(.*?){re.escape(FINAL_E)}"
    for c in completions:
        s = 0.0
        if len(re.findall(think_pat, c, re.DOTALL)) == 1:
            s += 0.5
        if len(re.findall(final_pat, c, re.DOTALL)) == 1:
            s += 0.5
        if c:
            removal = c.replace("addCriterion", "").replace("\n", "")
            if (len(c) - len(removal)) / max(len(c), 1) >= 0.5:
                s -= 2.0
        scores.append(s)
    return scores


def choice_validity_reward_func(completions, choices_letters, **kwargs):
    """+0.25 nếu chữ cái trong <final> thuộc tập choices_letters, ngược lại -0.5."""
    out = []
    regex = rf"{re.escape(FINAL_S)}\s*([A-H])\s*{re.escape(FINAL_E)}"
    for c, cl in zip(completions, choices_letters):
        m = re.search(regex, c, re.DOTALL)
        if m:
            out.append(0.25 if m.group(1) in cl else -0.5)
        else:
            out.append(-0.5)
    return out


def correctness_reward_func(completions, answer, **kwargs):
    """+2.0 nếu chữ cái trong <final> == đáp án gold, else 0.0."""
    out = []
    regex = rf"{re.escape(FINAL_S)}\s*([A-H])\s*{re.escape(FINAL_E)}"
    for c, a in zip(completions, answer):
        m = re.search(regex, c, re.DOTALL)
        out.append(2.0 if (m and m.group(1) == a) else 0.0)
    return out

# ==== MAIN ==================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--dataset_json", type=str, required=True,
                        help="VD: data/train/train_with_thinking.json")
    parser.add_argument("--frames_root", type=str, required=True,
                        help="VD: data/train_frames_448")

    # base + (optional) SFT adapter for Qwen2-VL
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_2VL)
    parser.add_argument("--sft_adapter", type=str, default="",
                        help="Đường dẫn LoRA SFT Qwen2-VL (nếu có). Bỏ trống để RL từ base.")

    # output
    parser.add_argument("--output_dir", type=str, default="outputs/qwen2vl_rl")
    parser.add_argument("--run_name", type=str, default="grpo_qwen2")

    # vision knobs
    parser.add_argument("--max_images", type=int, default=3)
    parser.add_argument("--short_side", type=int, default=384)
    parser.add_argument("--frame_picker", type=str, choices=["uniform","random","first_k"], default="uniform")

    # train knobs (T4-friendly)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--grad_acc", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=3)
    parser.add_argument("--max_prompt_len", type=int, default=1024)
    parser.add_argument("--max_completion_len", type=int, default=128)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=5)

    args = parser.parse_args()

    # reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # --------- Load Qwen2-VL with Unsloth fast path (4-bit)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=16384,
        load_in_4bit=True,
        fast_inference=False,           # Khuyến nghị cho Qwen2-VL
        gpu_memory_utilization=0.85,
    )

    # --------- Attach LoRA on language blocks (vision frozen)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_rslora=False,
        use_gradient_checkpointing="unsloth",
    )

    # (Optional) load prior SFT LoRA for Qwen2-VL
    if args.sft_adapter and os.path.isdir(args.sft_adapter):
        try:
            model.load_adapter(args.sft_adapter, adapter_name="sft2", is_trainable=True)
            model.set_adapter("sft2")
            print(f"[OK] Loaded Qwen2 SFT adapter: {args.sft_adapter}")
        except Exception as e:
            print(f"[WARN] Failed to load Qwen2 SFT adapter: {e}")

    # --------- Dataset
    train_ds = RLMultiFrameDataset(
        json_path=args.dataset_json,
        frames_root=args.frames_root,
        tokenizer=tokenizer,
        max_images=args.max_images,
        short_side=args.short_side,
        frame_picker=args.frame_picker,
        seed=args.seed,
    )

    # --------- GRPO config (T4-safe)
    training_args = GRPOConfig(
        learning_rate=args.lr,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",

        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_acc,
        num_generations=args.num_generations,

        max_prompt_length=args.max_prompt_len,
        max_completion_length=args.max_completion_len,
        num_train_epochs=args.epochs,

        logging_steps=args.logging_steps,
        log_completions=False,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to="none",
        output_dir=args.output_dir,

        # ổn định hơn
        importance_sampling_level="sequence",
        mask_truncated_completions=False,
        loss_type="dr_grpo",
    )

    # --------- Trainer
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        reward_funcs=[
            formatting_reward_func,
            choice_validity_reward_func,
            correctness_reward_func,
        ],
        train_dataset=train_ds,
    )

    # --------- Train!
    trainer.train()

    # --------- Save LoRA (RL)
    from pathlib import Path
    from peft import PeftModel, get_peft_model_state_dict

    save_dir = Path(args.output_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    lora_dir = save_dir / "lora_adapter_rl_qwen2"
    lora_dir.mkdir(parents=True, exist_ok=True)

    try:
        if hasattr(model, "save_lora"):
            # Một số bản Unsloth hỗ trợ trực tiếp
            model.save_lora(str(lora_dir))
        elif isinstance(model, PeftModel):
            # Chuẩn PEFT
            model.save_pretrained(str(lora_dir))
        else:
            # Trích state_dict thủ công
            state_dict = get_peft_model_state_dict(model)
            torch.save(state_dict, lora_dir / "adapter_model.bin")

            # Ghi adapter_config.json tối thiểu
            peft_cfg = getattr(model, "peft_config", None)
            if peft_cfg and isinstance(peft_cfg, dict):
                first_key = next(iter(peft_cfg))
                cfg = peft_cfg[first_key].to_dict()
            else:
                cfg = {
                    "peft_type": "LORA",
                    "auto_mapping": None,
                    "base_model_name_or_path": args.base_model,
                    "bias": "none",
                    "inference_mode": False,
                    "r": 32,
                    "lora_alpha": 32,
                    "lora_dropout": 0.05,
                    "task_type": "CAUSAL_LM"
                }
            with open(lora_dir / "adapter_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

        print(f"[OK] LoRA adapter (RL) saved to: {lora_dir}")
        try:
            tokenizer.save_pretrained(str(save_dir))
        except Exception as e:
            print(f"[WARN] Tokenizer save failed: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to save LoRA adapter: {e}")


if __name__ == "__main__":
    main()
