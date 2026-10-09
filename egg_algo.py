"""Task 2 (classical egg segmentation, no deep learning) + Task 3 (evaluation against SAM 3 masks).
Run once on Kaggle. Parameters are frozen, so dev and eval are each run one time. Results are zipped at the end."""
import os
import glob
import shutil
from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


@dataclass
class Config:
    img_root: str = "/kaggle/input/datasets/swayamsingh48/eggs-s"
    sam_root: str = "/kaggle/input/datasets/swayamsingh48/sam-result"  # Task 1 output (has union/ folder)
    work_dir: str = "/kaggle/working/task2"                            # split.csv from development lives here
    out: str = "/kaggle/working/results_task2_task3"
    work: int = 400                  # long side of the working image (px)
    denoise: int = 5                 # denoising strength
    clahe: float = 2.0               # contrast boost
    k: int = 12                      # number of colour clusters (regions)
    canny: tuple = (40, 120)         # edges used as walls between regions
    barrier: int = 2                 # wall thickness
    offsets: tuple = (-0.3, -0.15, 0.0, 0.15, 0.3)  # seed points: 5x5 grid around the image centre
    thrs: tuple = (10, 18, 28)       # colour distance limits for merging regions with the seed region
    thr_scale: float = 1.302         # multiplier on thrs
    lab_w: tuple = (0.5, 1.0, 1.0)   # L, a, b weights (lightness counts less)
    opens: tuple = (5, 15, 31)       # opening sizes, cut thin bridges to fingers
    min_area: float = 0.02           # egg blob size limits (fraction of image)
    max_area: float = 0.5
    area_mu: float = -2.41           # egg-size prior (log area), from dev SAM masks
    area_sd: float = 0.67
    centre_sd: float = 0.3           # eggs are usually near the centre
    grow: float = 1.147              # enlarge the fitted ellipse
    edge_pow: float = 1.849          # weight of "the outline sits on real edges"
    extra_ratio: float = 0.7         # an extra egg must score >= this * best egg
    min_ell_iou: float = 0.85        # an extra egg must be this ellipse-like
    max_overlap: float = 0.2
    max_eggs: int = 4


def get_split(c):
    f = f"{c.work_dir}/split.csv"
    if os.path.exists(f):  # reuse the split from development
        return pd.read_csv(f)
    print("WARNING: split.csv not found, building a new name-grouped 80/20 split")
    rows = []
    for p in sorted(glob.glob(f"{c.img_root}/**/*.jpg", recursive=True)):
        parts = os.path.relpath(p, c.img_root).split(os.sep)
        uid = "__".join(parts[:-1] + [os.path.splitext(parts[-1])[0]])
        u = f"{c.sam_root}/union/{uid}.png"
        if os.path.exists(u):
            rows.append((uid, p, u, parts[-1].split(".rf.")[0]))  # name before .rf. = original photo
    df = pd.DataFrame(rows, columns=["id", "img", "union", "group"])
    g = np.random.RandomState(42).permutation(df["group"].unique())
    df["split"] = np.where(df["group"].isin(set(g[: int(0.2 * len(g))])), "eval", "dev")
    os.makedirs(c.work_dir, exist_ok=True)
    df.to_csv(f, index=False)
    return df


# ---------------------------------------------------------------- algorithm
def clean(img, c):
    img = cv2.fastNlMeansDenoisingColored(img, None, c.denoise, c.denoise, 5, 15)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(c.clahe, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def cut_regions(img, c):
    """k-means on colour, split into connected pieces, Canny edges act as walls."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lbl, _ = cv2.kmeans(lab.reshape(-1, 3), c.k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
    lbl = lbl.reshape(H, W)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), *c.canny)
    wall = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=c.barrier) > 0
    reg, n = np.zeros((H, W), np.int32), 1
    for k in range(c.k):
        cnt, cc = cv2.connectedComponents(((lbl == k) & ~wall).astype(np.uint8))
        reg[cc > 0] = cc[cc > 0] + n - 1
        n += cnt - 1
    count = np.maximum(np.bincount(reg.ravel(), minlength=n), 1)
    mean = np.stack([np.bincount(reg.ravel(), lab[..., i].ravel(), n) / count for i in range(3)], 1)
    return reg, mean  # region id per pixel, mean colour per region


def edge_map(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.dilate(np.clip(m / (np.percentile(m, 99) + 1e-6), 0, 1), np.ones((5, 5), np.uint8))


def candidate(comp, c, edge):
    """Fit an ellipse to a blob; score = ellipse fit * egg-size prior * edge support (and * centre prior)."""
    H, W = comp.shape
    a = comp.sum() / (H * W)
    if not c.min_area <= a <= c.max_area:
        return None
    cnt = max(cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), (ew, eh), ang = cv2.fitEllipse(cnt)
    fit = np.zeros_like(comp)
    cv2.ellipse(fit, ((cx, cy), (ew, eh), ang), 1, -1)
    ei = (comp & fit).sum() / max((comp | fit).sum(), 1)
    gw, gh = ew * c.grow, eh * c.grow
    out = np.zeros_like(comp)
    cv2.ellipse(out, ((cx, cy), (gw, gh), ang), 1, -1)
    pts = cv2.ellipse2Poly((int(cx), int(cy)), (max(int(gw / 2), 1), max(int(gh / 2), 1)), int(ang), 0, 360, 6)
    es = edge[np.clip(pts[:, 1], 0, H - 1), np.clip(pts[:, 0], 0, W - 1)].mean()
    base = ei * np.exp(-0.5 * ((np.log(a) - c.area_mu) / c.area_sd) ** 2) * (0.05 + es) ** c.edge_pow
    d = np.hypot(cx / W - 0.5, cy / H - 0.5)
    return base * np.exp(-0.5 * (d / c.centre_sd) ** 2), base, ei, out


def grabcut(img, em):
    """Refine an ellipse: core = sure egg, band around the outline = undecided, outside = background."""
    m = em.astype(np.uint8)
    k = max(3, int(0.12 * np.sqrt(m.sum()))) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    gc = np.full(m.shape, cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(m, ker) > 0] = cv2.GC_PR_BGD
    gc[m > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(m, ker) > 0] = cv2.GC_FGD
    try:
        cv2.grabCut(img, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m > 0
    out = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(out)
    if n < 2:
        return m > 0
    comp = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, 1, -1)
    return (filled > 0) if 0.6 < filled.sum() / max(m.sum(), 1) < 1.8 else (m > 0)  # fall back to the ellipse


def segment(d, c):
    img, cl = d["img"], d["clean"]
    H, W = img.shape[:2]
    reg, mean = cut_regions(cl, c)
    edge, wv, cands = edge_map(cl), np.array(c.lab_w, np.float32), []
    for oy in c.offsets:
        for ox in c.offsets:  # each seed point proposes an egg colour
            cy, cx = int(H * (0.5 + oy)), int(W * (0.5 + ox))
            patch = reg[max(cy - 5, 0):cy + 6, max(cx - 5, 0):cx + 6].ravel()
            patch = patch[patch > 0]
            if len(patch) == 0:
                continue
            dist = np.sqrt((((mean - mean[np.bincount(patch).argmax()]) * wv) ** 2).sum(1))
            for t in c.thrs:
                ok = dist < t * c.thr_scale
                ok[0] = False
                m = cv2.morphologyEx(ok[reg].astype(np.uint8), cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c.barrier + 3,) * 2))
                for o in c.opens:
                    _, cc = cv2.connectedComponents(cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o, o))))
                    if cc[cy, cx]:
                        r = candidate((cc == cc[cy, cx]).astype(np.uint8), c, edge)
                        if r:
                            cands.append(r)
    pred = np.zeros((H, W), bool)
    if not cands:
        return pred, 0
    cands.sort(key=lambda x: -x[0])
    first, kept = cands[0], [cands[0]]
    for cd in sorted(cands[1:], key=lambda x: -x[1]):  # further eggs, scored without the centre prior
        if cd[1] < c.extra_ratio * first[1] or len(kept) >= c.max_eggs:
            break
        if cd[2] >= c.min_ell_iou and all((cd[3] & k[3]).sum() / max(min(cd[3].sum(), k[3].sum()), 1) <= c.max_overlap for k in kept):
            kept.append(cd)
    for k in kept:
        pred |= grabcut(img, k[3])
    return pred, len(kept)


# ---------------------------------------------------------------- evaluation
GREEN, RED, BLUE = (0, 200, 0), (0, 0, 230), (230, 110, 0)  # BGR


def text(img, s, y=16):
    cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def panel(img, gt, pred, caption):
    """photo | SAM mask (green) | our mask (red) | agreement map."""
    def tint(m, col):
        o = img.copy()
        o[m] = (img[m] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
        cv2.drawContours(o, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, 2)
        return o
    agree = img.copy()
    for m, col in ((gt & pred, BLUE), (gt & ~pred, GREEN), (pred & ~gt, RED)):
        agree[m] = (img[m] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
    tiles = [img.copy(), tint(gt, GREEN), tint(pred, RED), agree]
    for t, s in zip(tiles, ["photo", "SAM (green)", "ours (red)", "blue=both green=SAM only red=ours only"]):
        text(t, s)
    bar = np.full((24, 4 * img.shape[1], 3), 40, np.uint8)
    text(bar, caption, 17)
    return np.vstack([bar, np.hstack(tiles)])


def prepare(r, c):
    img, gt = cv2.imread(r.img), cv2.imread(r.union, cv2.IMREAD_GRAYSCALE)
    s = c.work / max(img.shape[:2])
    size = (round(img.shape[1] * s), round(img.shape[0] * s))
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    gt = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST) > 127
    if gt.sum() == 0:  # SAM found no egg -> cannot be scored
        return None
    n, _, st, _ = cv2.connectedComponentsWithStats(gt.astype(np.uint8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ring = (cv2.dilate(gt.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0) & ~gt
    return dict(id=r.id, img=img, gt=gt, clean=clean(img, c),
                n_gt=int(sum(st[k, cv2.CC_STAT_AREA] >= 0.005 * gt.size for k in range(1, n))),
                egg_area=float(gt.mean()),  # egg size as a fraction of the image
                contrast=float(np.linalg.norm(lab[gt].mean(0) - lab[ring].mean(0))) if ring.any() else 0.0)  # egg vs surroundings


def evaluate(d, c, split):
    pred, n = segment(d, c)
    gt = d["gt"]
    i, u, p, g = int((pred & gt).sum()), int((pred | gt).sum()), int(pred.sum()), int(gt.sum())
    r = dict(id=d["id"], iou=i / max(u, 1), dice=2 * i / max(p + g, 1), precision=i / max(p, 1), recall=i / max(g, 1),
             n_pred=n, n_gt=d["n_gt"], egg_area=d["egg_area"], contrast=d["contrast"])
    cap = f"{r['id'].split('__')[-1].split('.rf.')[0]} | IoU {r['iou']:.2f}  P {r['precision']:.2f}  R {r['recall']:.2f} | eggs ours/SAM {n}/{d['n_gt']}"
    cv2.imwrite(f"{c.out}/pred_masks/{split}/{d['id']}.png", pred.astype(np.uint8) * 255)
    cv2.imwrite(f"{c.out}/comparison/{split}/{d['id']}.jpg", panel(d["img"], gt, pred, cap))
    return r


def grid(ids, split, path, c):
    ims = [cv2.imread(f"{c.out}/comparison/{split}/{i}.jpg") for i in ids]
    w = max(i.shape[1] for i in ims)
    cv2.imwrite(path, np.vstack([cv2.copyMakeBorder(i, 0, 0, 0, w - i.shape[1], cv2.BORDER_CONSTANT) for i in ims]))


def main(c):
    for s in ("dev", "eval"):
        for d in ("pred_masks", "comparison"):
            os.makedirs(f"{c.out}/{d}/{s}", exist_ok=True)
    df, res = get_split(c), []
    for s in ("dev", "eval"):
        rows = list(df[df["split"] == s].itertuples())
        data = [d for d in Parallel(n_jobs=-1)(delayed(prepare)(r, c) for r in rows) if d]
        r = pd.DataFrame(Parallel(n_jobs=-1)(delayed(evaluate)(d, c, s) for d in data))
        r["split"] = s
        print(f"{s}: {len(r)} scored, {len(rows) - len(r)} skipped (SAM found no egg)")
        res.append(r)
    allr = pd.concat(res, ignore_index=True)
    allr["count_ok"] = allr["n_pred"] == allr["n_gt"]
    allr["too_many"], allr["too_few"] = allr["n_pred"] > allr["n_gt"], allr["n_pred"] < allr["n_gt"]
    summary = allr.groupby("split").agg(
        images=("iou", "size"), mean_iou=("iou", "mean"), median_iou=("iou", "median"), mean_dice=("dice", "mean"),
        precision=("precision", "mean"), recall=("recall", "mean"), iou_above_0_9=("iou", lambda x: (x > 0.9).mean()),
        iou_below_0_5=("iou", lambda x: (x < 0.5).mean()), count_accuracy=("count_ok", "mean"),
        too_many_eggs=("too_many", "sum"), too_few_eggs=("too_few", "sum")).round(4)
    print("\n" + summary.T.to_string())
    for s in ("dev", "eval"):
        b = allr[(allr["split"] == s) & (allr["iou"] < 0.5)]
        print(f"{s} failures (IoU<0.5): {len(b)} | no overlap (<0.1): {(b['iou'] < 0.1).sum()} | "
              f"too small: {((b['recall'] < 0.6) & (b['precision'] > b['recall'])).sum()} | "
              f"too big: {((b['precision'] < 0.6) & (b['recall'] > b['precision'])).sum()}")
    summary.to_csv(f"{c.out}/metrics_summary.csv")
    allr.to_csv(f"{c.out}/per_image_metrics.csv", index=False)
    df.to_csv(f"{c.out}/split.csv", index=False)

    ev = allr[allr["split"] == "eval"].sort_values("iou")
    grid(ev["id"].head(6), "eval", f"{c.out}/eval_worst.jpg", c)
    grid(ev["id"].tail(4), "eval", f"{c.out}/eval_best.jpg", c)

    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    col = allr["split"].map({"dev": "tab:blue", "eval": "tab:orange"})
    for s, k in (("dev", "tab:blue"), ("eval", "tab:orange")):
        ax[0, 0].hist(allr[allr["split"] == s]["iou"], bins=np.linspace(0, 1, 21), alpha=0.6, color=k, label=s)
    ax[0, 0].set(title="IoU distribution", xlabel="IoU", ylabel="images")
    ax[0, 0].legend()
    ax[0, 1].scatter(allr["egg_area"], allr["iou"], c=col, s=14)
    ax[0, 1].set(title="IoU vs egg size", xlabel="egg area / image area", ylabel="IoU")
    ax[0, 2].scatter(allr["contrast"], allr["iou"], c=col, s=14)
    ax[0, 2].set(title="IoU vs egg-to-surroundings colour difference", xlabel="Lab distance", ylabel="IoU")
    ax[1, 0].scatter(allr["recall"], allr["precision"], c=allr["iou"], s=16)
    ax[1, 0].set(title="precision vs recall (colour = IoU)", xlabel="recall", ylabel="precision")
    ct = pd.crosstab(allr["n_gt"], allr["n_pred"])
    ax[1, 1].imshow(ct.values, cmap="Blues")
    ax[1, 1].set(title="egg count: SAM (rows) vs ours (columns)", xticks=range(ct.shape[1]), yticks=range(ct.shape[0]),
                 xticklabels=ct.columns, yticklabels=ct.index)
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax[1, 1].text(j, i, ct.values[i, j], ha="center", va="center")
    ax[1, 2].scatter(allr["iou"], allr["dice"], c=col, s=14)
    ax[1, 2].set(title="Dice vs IoU", xlabel="IoU", ylabel="Dice")
    plt.tight_layout()
    plt.savefig(f"{c.out}/analysis_plots.png", dpi=90)
    plt.close(fig)

    shutil.make_archive("/kaggle/working/egg_task2_task3_results", "zip", c.out)
    print("\nsaved /kaggle/working/egg_task2_task3_results.zip")


if __name__ == "__main__":
    main(Config())