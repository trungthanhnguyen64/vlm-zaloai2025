# import os, json, numpy as np, pathlib, glob
# from decord import VideoReader, cpu
# from PIL import Image

# JSON_IN = "data/train/train.json"
# VIDEO_ROOT = "data/train/videos"
# FRAMES_ROOT = "data/train_frames_448"
# os.makedirs(FRAMES_ROOT, exist_ok=True)

# with open(JSON_IN, "r", encoding="utf-8") as f:
#     samples = json.load(f)["data"]

# def load_and_sample(video_path, num_frames=3, short_side=448):
#     vr = VideoReader(video_path, ctx=cpu(0))
#     n = len(vr)
#     if n == 0: return []
#     idxs = np.linspace(0, n-1, num_frames).astype(int)
#     batch = vr.get_batch(idxs).asnumpy()  # (N,H,W,3)
#     imgs = []
#     for arr in batch:
#         im = Image.fromarray(arr)
#         w, h = im.size
#         if min(w,h) != short_side:
#             if w < h:
#                 nw, nh = short_side, int(h * (short_side/w))
#             else:
#                 nh, nw = short_side, int(w * (short_side/h))
#             im = im.resize((nw, nh), Image.BICUBIC)
#         imgs.append(im)
#     return imgs

# count_ok = 0
# for ex in samples:
#     vid_rel = ex["video_path"]                    # "train/videos/xxx.mp4"
#     vid_path = os.path.join("data", vid_rel) if not os.path.exists(os.path.join(VIDEO_ROOT, os.path.basename(vid_rel))) \
#                else os.path.join(VIDEO_ROOT, os.path.basename(vid_rel))
#     base = os.path.splitext(os.path.basename(vid_rel))[0]
#     out_dir = os.path.join(FRAMES_ROOT, base)
#     if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "f*.jpg")):
#         count_ok += 1
#         continue
#     os.makedirs(out_dir, exist_ok=True)
#     try:
#         imgs = load_and_sample(vid_path, num_frames=3, short_side=448)
#         for i, im in enumerate(imgs):
#             im.save(os.path.join(out_dir, f"f{i:02d}.jpg"), quality=90)
#         if imgs: count_ok += 1
#     except Exception as e:
#         print("Skip", vid_path, "->", e)

# print("Done. Folders created:", count_ok)
import os, json, numpy as np, glob
from PIL import Image
from decord import VideoReader, cpu
# NOTE: nếu decord hỗ trợ GPU trên máy bạn:
try:
    from decord import gpu as decord_gpu
    HAS_DEC_GPU = True
except Exception:
    HAS_DEC_GPU = False

# =========================
# CẤU HÌNH
# =========================
JSON_IN     = "data/train/train.json"
VIDEO_ROOT  = "data/train/videos"
FRAMES_ROOT = "data/train_frames_hq"



# Số frame trích xuất / video và kích thước cạnh ngắn
NUM_FRAMES = 8
SHORT_SIDE = 512  # 448 -> 512 cho chi tiết tốt hơn; không upscale

# Định dạng output: "png" (khuyến nghị), "webp", hoặc "jpeg"
OUTPUT_FMT = "png"

os.makedirs(FRAMES_ROOT, exist_ok=True)

# =========================
# TIỆN ÍCH
# =========================
def _pil_resample():
    # Tương thích Pillow/Pillow-SIMD
    return getattr(Image, "Resampling", Image).LANCZOS

def resize_keep_quality(im: Image.Image, short_side: int):
    """Chỉ downscale giữ tỉ lệ; không upscale."""
    if short_side is None or short_side <= 0:
        return im
    w, h = im.size
    mn = min(w, h)
    if mn <= short_side:
        return im  # không phóng to
    if w <= h:
        new_w, new_h = short_side, int(h * (short_side / w))
    else:
        new_h, new_w = short_side, int(w * (short_side / h))
    return im.resize((new_w, new_h), _pil_resample())

def save_frame(im: Image.Image, out_path: str, fmt: str = "png"):
    """Lưu với cấu hình 'sạch' cho training."""
    fmt = fmt.lower()
    im = im.convert("RGB")
    if fmt == "png":
        # lossless, dung lượng cao hơn nhưng sạch artefact
        out = os.path.splitext(out_path)[0] + ".png"
        im.save(out, format="PNG", compress_level=3)
    elif fmt == "webp":
        # lossless, file nhỏ hơn PNG, cần Pillow hỗ trợ WebP
        out = os.path.splitext(out_path)[0] + ".webp"
        im.save(out, format="WEBP", lossless=True, quality=100, method=6)
    elif fmt == "jpeg" or fmt == "jpg":
        # JPEG 4:4:4 giảm artefact màu, vẫn có mất mát
        out = os.path.splitext(out_path)[0] + ".jpg"
        im.save(out, format="JPEG", quality=95, subsampling=0, optimize=True)
    else:
        out = out_path
        im.save(out)
    return out

def get_vr(video_path: str):
    """Mở VideoReader với GPU nếu có, fallback CPU."""
    # Nếu decord cài có NVDEC và driver CUDA hợp lệ, dùng GPU(0)
    if HAS_DEC_GPU:
        try:
            return VideoReader(video_path, ctx=decord_gpu(0))
        except Exception:
            pass
    # Fallback CPU
    return VideoReader(video_path, ctx=cpu(0))

def load_and_sample(video_path: str, num_frames=3, short_side=512):
    """Đọc video, sample đều và downscale an toàn."""
    vr = get_vr(video_path)
    n = len(vr)
    if n == 0:
        return []
    # Sample đều trên toàn clip
    idxs = np.linspace(0, n - 1, num_frames).astype(np.int64)
    batch = vr.get_batch(idxs).asnumpy()  # (N, H, W, 3), uint8
    imgs = []
    for arr in batch:
        im = Image.fromarray(arr, mode="RGB")
        im = resize_keep_quality(im, short_side)
        imgs.append(im)
    return imgs

# =========================
# MAIN
# =========================
with open(JSON_IN, "r", encoding="utf-8") as f:
    samples = json.load(f)["data"]

count_ok = 0
for ex in samples:
    # "train/videos/xxx.mp4" trong JSON; cố gắng map vào VIDEO_ROOT
    vid_rel = ex["video_path"]
    candidate = os.path.join("data", vid_rel)
    if os.path.exists(candidate):
        vid_path = candidate
    else:
        vid_path = os.path.join(VIDEO_ROOT, os.path.basename(vid_rel))

    base = os.path.splitext(os.path.basename(vid_rel))[0]
    out_dir = os.path.join(FRAMES_ROOT, base)
    if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "f*.*")):
        count_ok += 1
        continue

    os.makedirs(out_dir, exist_ok=True)
    try:
        imgs = load_and_sample(vid_path, num_frames=NUM_FRAMES, short_side=SHORT_SIDE)
        for i, im in enumerate(imgs):
            # chỉ tên gốc, phần mở rộng quyết định bởi save_frame
            raw_out = os.path.join(out_dir, f"f{i:02d}.jpg")
            save_frame(im, raw_out, fmt=OUTPUT_FMT)
        if imgs:
            count_ok += 1
    except Exception as e:
        print("Skip", vid_path, "->", e)

print("Done. Folders created:", count_ok)
