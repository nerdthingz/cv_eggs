import os
from dataclasses import dataclass

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

SRC = "/kaggle/input/datasets/swayamsingh48/egg-vid/Raw and Boiled Egg Spinning - Cool Science Experiment _ Educational Videos by Mocomi_1080p.mp4"
TMP, OUT = "/kaggle/working/tmp.mp4", "/kaggle/working/tracked_eggs.mp4"
STEP, BATCH, MIN_HITS, MAX_MISS, GATE = 2, 32, 3, 10, 1.5


@dataclass
class Config:
    work: int = 400
    denoise: int = 5
    clahe: float = 2.0
    k: int = 12
    canny: tuple = (40, 120)
    barrier: int = 2
    offsets: tuple = tuple(np.linspace(-0.4, 0.4, 7))
    thrs: tuple = (10, 18, 28)
    thr_scale: float = 1.302
    lab_w: tuple = (0.5, 1.0, 1.0)
    opens: tuple = (5, 15, 31)
    min_area: float = 0.003
    max_area: float = 0.5
    area_mu: float = -2.41
    area_sd: float = 1.0
    grow: float = 1.147
    edge_pow: float = 1.849
    extra_ratio: float = 0.4
    min_ell_iou: float = 0.8
    max_overlap: float = 0.2
    max_eggs: int = 4


def clean(img, c):
    img = cv2.fastNlMeansDenoisingColored(img, None, c.denoise, c.denoise, 5, 15)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(c.clahe, (8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def cut_regions(img, c):
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
    return reg, mean


def edge_map(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.dilate(np.clip(m / (np.percentile(m, 99) + 1e-6), 0, 1), np.ones((5, 5), np.uint8))


def candidate(comp, c, edge):
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
    return base, ei, out


def grabcut(img, em):
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
    return (filled > 0) if 0.6 < filled.sum() / max(m.sum(), 1) < 1.8 else (m > 0)


def detect(frame, c):
    cv2.setNumThreads(1)
    H, W = frame.shape[:2]
    s = c.work / max(H, W)
    img = cv2.resize(frame, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    cl = clean(img, c)
    reg, mean = cut_regions(cl, c)
    edge, wv = edge_map(cl), np.array(c.lab_w, np.float32)
    seeds = {}
    for oy in c.offsets:
        for ox in c.offsets:  # seeds cover the whole frame, one per distinct region
            cy, cx = int(h * (0.5 + oy)), int(w * (0.5 + ox))
            p = reg[max(cy - 5, 0):cy + 6, max(cx - 5, 0):cx + 6].ravel()
            p = p[p > 0]
            if len(p):
                seeds.setdefault(np.bincount(p).argmax(), (cy, cx))
    cands = []
    for rid, (cy, cx) in seeds.items():
        dist = np.sqrt((((mean - mean[rid]) * wv) ** 2).sum(1))
        for t in c.thrs:
            ok = dist < t * c.thr_scale
            ok[0] = False
            m = cv2.morphologyEx(ok[reg].astype(np.uint8), cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * c.barrier + 3,) * 2))
            for o in c.opens:
                _, cc = cv2.connectedComponents(cv2.morphologyEx(
                    m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o, o))))
                if cc[cy, cx]:
                    r = candidate((cc == cc[cy, cx]).astype(np.uint8), c, edge)
                    if r:
                        cands.append(r)
    if not cands:
        return []
    cands.sort(key=lambda x: -x[0])
    kept = [cands[0]]
    for cd in cands[1:]:  # greedy NMS: keep strong, ellipse-like, non-overlapping blobs
        if cd[0] < c.extra_ratio * kept[0][0] or len(kept) >= c.max_eggs:
            break
        if cd[1] >= c.min_ell_iou and all(
                (cd[2] & k[2]).sum() / max(min(cd[2].sum(), k[2].sum()), 1) <= c.max_overlap for k in kept):
            kept.append(cd)
    return [grabcut(img, k[2]) for k in kept]


def centroid(m):
    ys, xs = np.nonzero(m)
    return np.array([xs.mean(), ys.mean()], np.float32)


class Track:
    def __init__(s, tid, m):
        s.id, s.hits, s.miss, s.mask, s.area = tid, 1, 0, m, m.sum()
        s.kf = cv2.KalmanFilter(4, 2)  # constant-velocity model on the centroid
        s.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        s.kf.measurementMatrix = np.eye(2, 4, dtype=np.float32)
        s.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        s.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        s.kf.errorCovPost = np.eye(4, dtype=np.float32)
        s.kf.statePost = np.append(centroid(m), [0, 0]).astype(np.float32).reshape(4, 1)

    def predict(s):
        return s.kf.predict()[:2, 0]


def update(tracks, masks, nid):
    preds = [t.predict() for t in tracks]
    cents, areas = [centroid(m) for m in masks], [m.sum() for m in masks]
    cost = np.full((len(tracks), len(masks)), 1e6)
    for i, t in enumerate(tracks):  # cost = centre distance (in egg diameters) + size change
        d = 2 * np.sqrt(t.area / np.pi)
        for j in range(len(masks)):
            dist = np.linalg.norm(preds[i] - cents[j]) / d
            if dist < GATE:
                cost[i, j] = dist + abs(np.log(areas[j] / t.area))
    matched = {}
    if cost.size:
        r, q = linear_sum_assignment(cost)  # Hungarian assignment
        matched = {i: j for i, j in zip(r, q) if cost[i, j] < 1e6}
    for i, t in enumerate(tracks):
        if i in matched:
            j = matched[i]
            t.kf.correct(cents[j].reshape(2, 1))
            t.mask, t.area, t.hits, t.miss = masks[j], areas[j], t.hits + 1, 0
        else:
            t.kf.statePost = t.kf.statePre.copy()  # coast on the prediction
            t.miss += 1
    for j, m in enumerate(masks):  # unmatched detections start new tracks
        if j not in matched.values():
            tracks.append(Track(nid, m))
            nid += 1
    return [t for t in tracks if t.miss <= MAX_MISS], nid


def render(f, tracks):
    H, W = f.shape[:2]
    for t in tracks:
        if t.hits < MIN_HITS or t.miss:
            continue
        m = cv2.resize(t.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        col = tuple(int(x) for x in np.random.RandomState(t.id).randint(60, 255, 3))
        f[m] = (f[m] * 0.5 + np.array(col) * 0.5).astype(np.uint8)
        cv2.drawContours(f, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, col, 3)
        x, y = centroid(m).astype(int)
        for th, cl in ((6, (0, 0, 0)), (2, (255, 255, 255))):
            cv2.putText(f, f"egg {t.id}", (x - 40, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cl, th, cv2.LINE_AA)
    return f


def main():
    c = Config()
    cap = cv2.VideoCapture(SRC)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(TMP, cv2.VideoWriter_fourcc(*"mp4v"), fps / STEP, (W, H))
    tracks, nid, i = [], 1, 0
    while True:
        frames = []
        while len(frames) < BATCH:
            ok, f = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                frames.append(f)
            i += 1
        if not frames:
            break
        for f, ms in zip(frames, Parallel(n_jobs=-1)(delayed(detect)(f, c) for f in frames)):
            tracks, nid = update(tracks, ms, nid)
            vw.write(render(f, tracks))
        print(f"{i}/{total} frames, {nid - 1} ids so far", flush=True)
    vw.release()
    if os.system(f"ffmpeg -y -loglevel error -i {TMP} -c:v libx264 -pix_fmt yuv420p {OUT}") != 0:
        os.replace(TMP, OUT)
    print("saved", OUT)


main()