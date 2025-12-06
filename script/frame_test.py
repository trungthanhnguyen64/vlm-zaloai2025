import os, json, numpy as np, pathlib, glob
from decord import VideoReader, cpu
from PIL import Image

JSON_IN = "data/public_test/public_test.json"
VIDEO_ROOT = "data/public_test/videos"
FRAMES_ROOT = "data/test_frames_448"
os.makedirs(FRAMES_ROOT, exist_ok=True)

with open(JSON_IN, "r", encoding="utf-8") as f:
    samples = json.load(f)["data"]

def load_and_sample(video_path, num_frames=8, short_side=448):
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    if n == 0: return []
    idxs = np.linspace(0, n-1, num_frames).astype(int)
    batch = vr.get_batch(idxs).asnumpy()  # (N,H,W,3)
    imgs = []
    for arr in batch:
        im = Image.fromarray(arr)
        w, h = im.size
        if min(w,h) != short_side:
            if w < h:
                nw, nh = short_side, int(h * (short_side/w))
            else:
                nh, nw = short_side, int(w * (short_side/h))
            im = im.resize((nw, nh), Image.BICUBIC)
        imgs.append(im)
    return imgs

count_ok = 0
for ex in samples:
    vid_rel = ex["video_path"]                    # "train/videos/xxx.mp4"
    vid_path = os.path.join("data", vid_rel) if not os.path.exists(os.path.join(VIDEO_ROOT, os.path.basename(vid_rel))) \
               else os.path.join(VIDEO_ROOT, os.path.basename(vid_rel))
    base = os.path.splitext(os.path.basename(vid_rel))[0]
    out_dir = os.path.join(FRAMES_ROOT, base)
    if os.path.isdir(out_dir) and glob.glob(os.path.join(out_dir, "f*.jpg")):
        count_ok += 1
        continue
    os.makedirs(out_dir, exist_ok=True)
    try:
        imgs = load_and_sample(vid_path, num_frames=8, short_side=448)
        for i, im in enumerate(imgs):
            im.save(os.path.join(out_dir, f"f{i:02d}.jpg"), quality=90)
        if imgs: count_ok += 1
    except Exception as e:
        print("Skip", vid_path, "->", e)

print("Done. Folders created:", count_ok)
