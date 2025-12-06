# -*- coding: utf-8 -*-
"""
infer_and_submit.py

Chức năng:
1) (Tùy chọn) Tách frame từ video test nếu thiếu (bỏ qua khi --skip_extract)
2) Suy luận bằng Qwen2-VL-7B + LoRA (4-bit + offload_dir)
3) Xuất submission.csv (giữ đúng thứ tự nếu có sample_order_csv)

Ví dụ chạy khi ĐÃ CÓ frame sẵn:
python a/infer_submit_from_frames.py \
  --test_json data/public_test/public_test.json \
  --frames_root data/test_frames_448 \
  --vl_base unsloth/qwen2-vl-7b-instruct-unsloth-bnb-4bit \
  --vl_adapter outputs/qwen2vl7b_unsloth_poc/lora_adapter \
  --offload_dir .offload_qwen2vl7b \
  --max_images 4 \
  --skip_extract \
  --sample_order_csv data/public_test_sample_submission.csv \
  --submission_out submission.csv

Ví dụ chạy khi CẦN tự tách frame:
python infer_and_submit.py \
  --test_json data/public_test.json \
  --video_root data/public_test/videos \
  --frames_root data/test_frames_448 \
  --vl_base unsloth/qwen2-vl-7b-instruct-unsloth-bnb-4bit \
  --vl_adapter outputs/qwen2vl7b_lora_poc/lora_adapter \
  --offload_dir .offload_qwen2vl7b \
  --num_extract_frames 4 \
  --short_side 448 \
  --max_images 4 \
  --submission_out submission.csv
"""

import os
import re
import cv2
import csv
import json
import glob
import math
import argparse
from typing import List, Dict, Tuple
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel


LETTER_SET = list("ABCDEFGH")


# ====================== Utilities ======================

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def extract_candidate_letters(choices: List[str]) -> List[str]:
    """Lấy các ký tự A–H từ danh sách lựa chọn. Nếu không parse được, cắt theo độ dài choices."""
    letters = []
    for c in choices:
        m = re.match(r"\s*([A-Ha-h])\s*[\.\)]", c or "")
        if m:
            letters.append(m.group(1).upper())
    if not letters:
        letters = LETTER_SET[:len(choices)] if choices else LETTER_SET[:4]
    return letters


def video_to_frames_uniform(video_path: str, out_dir: str, num_frames: int = 4, short_side: int = 448) -> int:
    """
    Tách num_frames khung hình theo vị trí đều (uniform), resize theo short_side, lưu f00.jpg, f01.jpg, ...
    Trả về số frame đã lưu.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return 0

    idxs = [int(total * (i + 1) / (num_frames + 1)) for i in range(num_frames)]
    saved = 0

    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(frame)

        # Resize theo short_side, giữ tỉ lệ
        w, h = im.size
        if short_side > 0 and min(w, h) != short_side:
            if w < h:
                nw, nh = short_side, int(h * (short_side / w))
            else:
                nh, nw = short_side, int(w * (short_side / h))
            im = im.resize((nw, nh), Image.BICUBIC)

        im.save(os.path.join(out_dir, f"f{i:02d}.jpg"), quality=90)
        saved += 1

    cap.release()
    return saved


def list_frame_paths(frames_root: str, video_path: str, max_images: int = 4) -> List[str]:
    """Lấy danh sách đường dẫn frame theo thứ tự tên file."""
    vid = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(frames_root, vid)
    paths = sorted(glob.glob(os.path.join(frame_dir, "f*.jpg")))[:max_images]
    return paths


def load_images(paths: List[str]) -> List[Image.Image]:
    """Đọc list đường dẫn ảnh thành danh sách PIL Image RGB."""
    ims = []
    for p in paths:
        try:
            ims.append(Image.open(p).convert("RGB"))
        except Exception:
            # Bỏ qua ảnh lỗi
            continue
    return ims


# ====================== Vision Scorer ======================

class VisionScorer:
    """
    Qwen2-VL-7B + LoRA (4-bit) cho bài trắc nghiệm hình ảnh.
    Chấm điểm từng đáp án bằng negative loss (higher is better), sau đó softmax-pick.
    """

    def __init__(self, base_id: str, adapter_dir: str,
                 offload_dir: str = None, device: str = "auto", use_fast: bool = False):
        if offload_dir:
            ensure_dir(offload_dir)

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.processor = AutoProcessor.from_pretrained(
            base_id, trust_remote_code=True, use_fast=use_fast
        )
        base = Qwen2VLForConditionalGeneration.from_pretrained(
            base_id,
            quantization_config=bnb,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=offload_dir,
        )
        self.model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False).eval()

        # pad token
        tok = getattr(self.processor, "tokenizer", None)
        if tok is None:
            raise RuntimeError("AutoProcessor không có tokenizer; kiểm tra model base.")
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    @torch.no_grad()
    def score_letters(self, images: List[Image.Image], question: str, choices: List[str]) -> Dict[str, float]:
        """
        Với 1 câu hỏi và các lựa chọn, tính "điểm" cho từng ký tự (A, B, C, ...),
        bằng negative loss khi ép model sinh ngay sau prompt là ký tự đó.
        """
        letters = extract_candidate_letters(choices)
        choices_text = "\n".join(choices) if choices else ""
        instr = "Trả lời CHỈ 1 ký tự (A/B/C/D/...). Không giải thích.\n"
        user_text = f"{instr}Câu hỏi: {question}\n"
        if choices_text:
            user_text += f"Lựa chọn:\n{choices_text}\n"
        user_text += "Đáp án:"

        # Tin nhắn gồm image(s) + text
        content = []
        for im in images:
            content.append({"type": "image", "image": im})
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tiền xử lý prompt-only để biết độ dài prompt (mask nhãn)
        prompt_only = self.processor.tokenizer([prompt], padding=True, return_tensors="pt", add_special_tokens=False)
        plen = (prompt_only["input_ids"][0] != self.pad_id).sum().item()

        # Tính loss cho từng ký tự đáp án
        scores: Dict[str, float] = {}
        for L in letters:
            full = self.processor(text=[prompt + L], images=[images], return_tensors="pt").to(self.model.device)
            labels = full["input_ids"].clone()
            labels[:, :plen] = -100  # Chỉ tính loss cho phần sinh ra (ký tự đáp án)
            loss = self.model(**full, labels=labels).loss.item()
            scores[L] = -loss  # negative loss: lớn hơn là tốt hơn

        return scores


def softmax_pick(scores: Dict[str, float]) -> str:
    if not scores:
        return "A"
    m = max(scores.values())
    exps = {k: math.exp(v - m) for k, v in scores.items()}
    return max(exps.items(), key=lambda kv: kv[1])[0]


# ====================== I/O helpers ======================

def read_test_items(test_json_path: str) -> List[Dict]:
    with open(test_json_path, "r", encoding="utf-8") as f:
        js = json.load(f)
    return js["data"] if isinstance(js, dict) and "data" in js else js


def read_order_ids_from_csv(sample_order_csv: str) -> List[str]:
    """
    Đọc cột 'id' từ CSV mẫu để đảm bảo thứ tự output. Không phụ thuộc pandas.
    """
    ids = []
    try:
        with open(sample_order_csv, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            if "id" in (r.fieldnames or []):
                for row in r:
                    ids.append(str(row["id"]))
    except Exception:
        return []
    return ids


def write_submission_csv(path_out: str, preds: Dict[str, str], order_ids: List[str] = None):
    with open(path_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "answer"])
        if order_ids:
            for i in order_ids:
                w.writerow([i, preds.get(i, "A")])
        else:
            for k, v in preds.items():
                w.writerow([k, v])


# ====================== Main ======================

def main():
    ap = argparse.ArgumentParser(description="Inference Qwen2-VL-7B + LoRA cho bài trắc nghiệm video/frame.")
    # Dữ liệu
    ap.add_argument("--test_json", required=True, help="JSON test có id, question, choices, video_path")
    ap.add_argument("--video_root", default="", help="Thư mục chứa mp4 test (chỉ cần nếu muốn tự tách frame)")
    ap.add_argument("--frames_root", required=True, help="Nơi lưu frame <frames_root>/<video_id>/f*.jpg")
    # Vision model
    ap.add_argument("--vl_base", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--vl_adapter", required=True, help="Đường dẫn thư mục LoRA adapter")
    ap.add_argument("--offload_dir", default=".offload_qwen2vl7b")
    # Tham số trích frame / đọc ảnh
    ap.add_argument("--num_extract_frames", type=int, default=4, help="Số frame tách/1 video nếu thiếu")
    ap.add_argument("--short_side", type=int, default=448, help="Resize short-side khi tách frame")
    ap.add_argument("--max_images", type=int, default=4, help="Số frame đưa vào model")
    ap.add_argument("--skip_extract", action="store_true",
                    help="Bỏ qua bước tách frame. Chỉ chạy inference trên các frame đã có.")
    ap.add_argument("--fail_if_missing_frames", action="store_true",
                    help="Nếu thiếu frame cho video nào thì dừng và báo lỗi.")
    # Output
    ap.add_argument("--submission_out", default="submission.csv")
    ap.add_argument("--sample_order_csv", default="")
    args = ap.parse_args()

    ensure_dir(args.frames_root)

    # 1) Đọc test
    test_items = read_test_items(args.test_json)

    # 2) Chuẩn bị frame
    if args.skip_extract:
        print(">> Step 1/2: Skip extract (sử dụng frame đã có)")
        # Kiểm tra thiếu frame (tùy chọn fail)
        missing = []
        for ex in test_items:
            paths = list_frame_paths(args.frames_root, ex["video_path"], max_images=args.max_images)
            if len(paths) < 1:
                missing.append(os.path.splitext(os.path.basename(ex["video_path"]))[0])
        if missing:
            msg = f"[Cảnh báo] {len(missing)} video chưa có frame trong '{args.frames_root}': ví dụ {missing[:5]}"
            print(msg)
            if args.fail_if_missing_frames:
                raise FileNotFoundError("Thiếu frame cho các video: " + ", ".join(missing[:20]) + ("..." if len(missing) > 20 else ""))
    else:
        print(">> Step 1/2: Extract frames if missing")
        if not args.video_root:
            raise ValueError("Thiếu --video_root. Hãy truyền --video_root hoặc dùng --skip_extract.")
        for ex in tqdm(test_items, desc="Extract"):
            vid_rel = ex["video_path"]  # ví dụ: "public_test/videos/xxx.mp4" hoặc đường tương đối khác
            vid_name = os.path.splitext(os.path.basename(vid_rel))[0]
            out_dir = os.path.join(args.frames_root, vid_name)
            have_frames = os.path.isdir(out_dir) and len(glob.glob(os.path.join(out_dir, "f*.jpg"))) >= 1
            if have_frames:
                continue

            ensure_dir(out_dir)
            # video path thực
            vpath = os.path.join(args.video_root, f"{vid_name}.mp4")
            if not os.path.exists(vpath):
                # fallback: ghép từ 'data' + rel path nếu người dùng để y như train
                alt = os.path.join("data", vid_rel) if not os.path.isabs(vid_rel) else vid_rel
                vpath = alt

            saved = video_to_frames_uniform(
                vpath, out_dir,
                num_frames=args.num_extract_frames,
                short_side=args.short_side
            )
            if saved == 0:
                # tạo 1 frame trắng làm fallback để tránh crash khi không muốn fail
                Image.new("RGB", (args.short_side, args.short_side), "white").save(
                    os.path.join(out_dir, "f00.jpg"), quality=90
                )

    # 3) Load model
    print(">> Step 2/2: Inference (load model)")
    scorer = VisionScorer(
        base_id=args.vl_base,
        adapter_dir=args.vl_adapter,
        offload_dir=args.offload_dir,
        device="auto",
        use_fast=False
    )

    # 4) Dự đoán từng mẫu
    print(">> Infer each item")
    preds: Dict[str, str] = {}
    for ex in tqdm(test_items, desc="Infer"):
        q = (ex.get("question") or "").strip()
        choices = ex.get("choices", [])
        paths = list_frame_paths(args.frames_root, ex["video_path"], max_images=args.max_images)
        ims = load_images(paths)

        if not ims:
            if args.fail_if_missing_frames:
                vid_id = os.path.splitext(os.path.basename(ex["video_path"]))[0]
                raise FileNotFoundError(f"Không tìm thấy frame hợp lệ cho video_id='{vid_id}' trong {args.frames_root}")
            # fallback trắng
            ims = [Image.new("RGB", (args.short_side, args.short_side), "white")]

        scores = scorer.score_letters(ims, q, choices)
        preds[str(ex["id"])] = softmax_pick(scores)

    # 5) Ghi file submission (giữ đúng thứ tự nếu có sample CSV)
    order_ids = read_order_ids_from_csv(args.sample_order_csv) if args.sample_order_csv else []
    write_submission_csv(args.submission_out, preds, order_ids if order_ids else None)

    print(f"✓ Saved {args.submission_out} with {len(preds)} rows.")


if __name__ == "__main__":
    main()
