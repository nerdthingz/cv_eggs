import os
import glob
import shutil
import threading
import numpy as np
import pandas as pd
import torch
from PIL import Image
from huggingface_hub import login
from transformers import Sam3Model, Sam3Processor

HF_TOKEN = "" # the token
ROOT = "/kaggle/input/datasets/swayamsingh48/eggs-s"
OUT = "/kaggle/working/sam3_out"
PROMPTS = ["egg", "white egg", "brown egg", "an egg in a hand"]
THRESH = 0.5
FALLBACK_THRESH = 0.25

login(token=HF_TOKEN)

for d in ["masks", "union", "overlay"]:
    os.makedirs(f"{OUT}/{d}", exist_ok=True)

files = []
for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
    files += glob.glob(f"{ROOT}/**/*.{ext}", recursive=True)
files = sorted(set(files))
print("images:", len(files))

n_gpu = max(1, torch.cuda.device_count())
processor = Sam3Processor.from_pretrained("facebook/sam3")
models = [Sam3Model.from_pretrained("facebook/sam3").to(f"cuda:{i}").eval() for i in range(n_gpu)]


def make_id(path):
    parts = os.path.relpath(path, ROOT).split(os.sep)
    return "__".join(parts[:-1] + [os.path.splitext(parts[-1])[0]])


def run(model, device, image, prompt, thresh):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_instance_segmentation(
        outputs,
        threshold=thresh,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]


def process(path, model, device):
    uid = make_id(path)
    npz_path = f"{OUT}/masks/{uid}.npz"
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if os.path.exists(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        sc = d["scores"]
        return [uid, path, int(d["masks"].shape[0]), str(d["prompt"]), float(sc.max()) if len(sc) else 0.0, w, h]

    used, res = None, None
    for p in PROMPTS:
        res = run(model, device, image, p, THRESH)
        if len(res["masks"]) > 0:
            used = p
            break
    if used is None:
        res = run(model, device, image, PROMPTS[0], FALLBACK_THRESH)
        if len(res["masks"]) > 0:
            used = PROMPTS[0] + "@" + str(FALLBACK_THRESH)

    if used is None:
        masks = np.zeros((0, h, w), dtype=np.uint8)
        scores = np.zeros((0,), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        used = "none"
    else:
        masks = res["masks"].cpu().numpy().astype(np.uint8)
        scores = res["scores"].cpu().numpy().astype(np.float32)
        boxes = res["boxes"].cpu().numpy().astype(np.float32)

    np.savez_compressed(npz_path, masks=masks, scores=scores, boxes=boxes, prompt=np.array(used))

    union = (masks.sum(axis=0) > 0).astype(np.uint8) * 255 if len(masks) else np.zeros((h, w), dtype=np.uint8)
    Image.fromarray(union).save(f"{OUT}/union/{uid}.png")

    arr = np.array(image).astype(np.float32)
    m = union > 0
    arr[m] = 0.45 * arr[m] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
    Image.fromarray(arr.astype(np.uint8)).save(f"{OUT}/overlay/{uid}.jpg", quality=90)

    return [uid, path, int(masks.shape[0]), used, float(scores.max()) if len(scores) else 0.0, w, h]


results = [[] for _ in range(n_gpu)]


def work(i):
    device = f"cuda:{i}"
    mine = files[i::n_gpu]
    for k, path in enumerate(mine):
        results[i].append(process(path, models[i], device))
        if k % 25 == 0:
            print(f"gpu{i}: {k}/{len(mine)}", flush=True)


threads = [threading.Thread(target=work, args=(i,)) for i in range(n_gpu)]
for t in threads:
    t.start()
for t in threads:
    t.join()

rows = [r for part in results for r in part]
df = pd.DataFrame(rows, columns=["id", "path", "n_eggs", "prompt_used", "max_score", "width", "height"])
df.to_csv(f"{OUT}/summary.csv", index=False)

print(len(df), "images processed")
print(df["n_eggs"].value_counts().sort_index())
print(df["prompt_used"].value_counts())
print(df["max_score"].describe())

shutil.make_archive("/kaggle/working/sam3_out", "zip", OUT)
print("saved: /kaggle/working/sam3_out.zip")