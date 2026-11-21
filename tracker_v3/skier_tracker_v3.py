"""skier_tracker_v3 — find THE competing skier in a broadcast Big Air clip.

Replaces the Viterbi "pick one box every frame" tracker.  Main ideas:

1. Detect people AND skis (COCO class 30) every STRIDE frames (YOLO11s).
2. Estimate camera motion between detection frames (LK optical flow + RANSAC
   similarity: pan + zoom + roll), ignoring burned-in broadcast graphics.
3. Link detections into short clean tracklets with camera-compensated IoU
   (a tracklet never glides from one object to another).
4. Score every tracklet on positive skier evidence:
     - ground-relative motion: a tracklet whose position on the ground never
       leaves a small radius is fixed to the world (banner, cameraman,
       spectator, judge)  -> rejected outright;
     - skis under the feet (spectators / cameramen / banners have none);
     - detection confidence and how centred the camera keeps it.
5. Per camera shot, dynamic programming picks the best chain of
   non-overlapping tracklets.  "No skier" is always allowed: a weak tracklet
   is worth less than leaving frames empty.
6. Pose is run on an upscaled crop of the chosen box in every frame, so small
   skiers get far better keypoints than in a full-frame pass.

Output (same layout as temporal_tracker.py):
    <out>/<stem>_pose.npy        float32 [T,17,3]  (x, y, conf), zeros = no skier
    <out>/<stem>_pose_meta.json  fps, size, cut_frames, shots, coverage, ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

# ------------------------------------------------------------------ config
DET_IMGSZ = 960
DET_CONF_PERSON = 0.15
DET_CONF_SKIS = 0.08
POSE_CROP = 320
SMALL_W, SMALL_H = 640, 360           # resolution for flow / cut detection
STATIC_RADIUS = 0.6                   # body heights: max ground excursion of a static object
STATIC_MIN_S = 0.6                    # seconds of evidence needed to call it static
MIN_TRACKLET_DETS = 3
SCORE_FLOOR = 0.30                    # tracklets below this are never used
OVERLAP_DETS = 6                      # detection frames two chained tracklets may share


# ------------------------------------------------------------------ helpers
def iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def warp_box(box, A):
    """Apply 2x3 similarity A to a box (centre moves, size scales)."""
    if A is None:
        return box
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    s = float(np.sqrt(abs(np.linalg.det(A[:, :2]))))
    c2 = A[:, :2] @ c + A[:, 2]
    w, h = (box[2] - box[0]) * s, (box[3] - box[1]) * s
    return np.array([c2[0] - w / 2, c2[1] - h / 2, c2[0] + w / 2, c2[1] + h / 2])


def read_small_frames(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    small = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        small.append(cv2.cvtColor(cv2.resize(f, (SMALL_W, SMALL_H), interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY))
    cap.release()
    return small, fps, W, H


def overlay_mask(small):
    """Burned-in graphics = pixels that barely change over the whole clip but
    have texture.  Returned mask is 255 where flow features are allowed."""
    n = len(small)
    idx = np.linspace(0, n - 1, min(40, n)).astype(int)
    stack = np.stack([small[i] for i in idx]).astype(np.float32)
    std = stack.std(axis=0)
    grad = cv2.Laplacian(np.median(stack, axis=0), cv2.CV_32F)
    static_tex = (std < 6) & (np.abs(grad) > 8)
    static_tex = cv2.dilate(static_tex.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    mask = np.where(static_tex, 0, 255).astype(np.uint8)
    return mask, static_tex


def detect_cuts(small, diff_thr=25.0, hist_thr=0.25):
    d = [0.0]; h = [0.0]
    prev = small[0].astype(np.float32)
    ph = cv2.calcHist([small[0]], [0], None, [32], [0, 256]); ph /= ph.sum()
    for g in small[1:]:
        gf = g.astype(np.float32)
        hist = cv2.calcHist([g], [0], None, [32], [0, 256]); hist /= hist.sum()
        d.append(float(np.abs(gf - prev).mean())); h.append(float(0.5 * np.abs(hist - ph).sum()))
        prev, ph = gf, hist
    d = np.array(d); h = np.array(h)
    med = np.median(d) + 1e-6
    cuts = []
    for i in range(1, len(d) - 1):
        peak = d[i] >= d[i - 1] and d[i] >= d[i + 1]
        if peak and d[i] > max(diff_thr, 6 * med) and (h[i] > hist_thr or d[i] > 3 * diff_thr):
            if not cuts or i - cuts[-1] > 8:
                cuts.append(i)
    return cuts


def camera_motion(g0, g1, mask, person_boxes_small):
    """Similarity transform mapping frame g0 -> g1 (small resolution)."""
    m = mask.copy()
    for b in person_boxes_small:
        x0, y0, x1, y1 = [int(v) for v in b]
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0
    pts = cv2.goodFeaturesToTrack(g0, 300, 0.003, 8, mask=m)
    if pts is None or len(pts) < 10:
        return None, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    if ok.sum() < 8:
        return None, 0.0
    A, inl = cv2.estimateAffinePartial2D(pts[ok].reshape(-1, 2), p1[ok].reshape(-1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if A is None:
        return None, 0.0
    return A, float(inl.sum()) / max(len(pts), 1)


def to_full(A_small, sx, sy):
    """Convert a small-resolution similarity into full-resolution coordinates."""
    if A_small is None:
        return None
    S = np.diag([sx, sy, 1.0]); Si = np.diag([1 / sx, 1 / sy, 1.0])
    A3 = np.vstack([A_small, [0, 0, 1]])
    return (S @ A3 @ Si)[:2]


# ------------------------------------------------------------------ stage 1: detection
def run_detection(det_model, path, det_frames, overlay_static, W, H):
    cap = cv2.VideoCapture(str(path))
    want = set(det_frames)
    dets = {}
    t = 0
    sx, sy = SMALL_W / W, SMALL_H / H
    while True:
        ok = cap.grab()
        if not ok:
            break
        if t in want:
            _, f = cap.retrieve()
            r = det_model(f, imgsz=DET_IMGSZ, conf=min(DET_CONF_PERSON, DET_CONF_SKIS),
                          classes=[0, 30], verbose=False)[0]
            persons, skis = [], []
            for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                if int(k) == 0 and c >= DET_CONF_PERSON:
                    # drop people inside burned-in graphics (athlete headshots etc.)
                    cx, cy = int((b[0] + b[2]) / 2 * sx), int((b[1] + b[3]) / 2 * sy)
                    cx = min(max(cx, 0), SMALL_W - 1); cy = min(max(cy, 0), SMALL_H - 1)
                    if overlay_static[cy, cx]:
                        continue
                    persons.append((np.array(b), float(c)))
                elif int(k) == 30 and c >= DET_CONF_SKIS:
                    skis.append((np.array(b), float(c)))
            dets[t] = (persons, skis)
        t += 1
    cap.release()
    return dets


def nms_merge(items, thr=0.5):
    items = sorted(items, key=lambda x: -x[1]); keep = []
    for b, c in items:
        if all(iou(b, k[0]) < thr for k in keep):
            keep.append((b, c))
    return keep


def gap_detection(det_tile, path, frames, dets, overlay_static, W, H):
    """Re-detect in 2x2 overlapping zoomed tiles on frames where no skier was found."""
    if not frames:
        return 0
    want = set(frames); cap = cv2.VideoCapture(str(path)); t = 0; added = 0
    tw, th = int(W * 0.6), int(H * 0.6)
    offs = [(0, 0), (W - tw, 0), (0, H - th), (W - tw, H - th)]
    sx, sy = SMALL_W / W, SMALL_H / H
    last = max(frames)
    while t <= last:
        if not cap.grab():
            break
        if t in want:
            _, f = cap.retrieve()
            persons, skis = dets[t]
            newp, news = list(persons), list(skis)
            for ox, oy in offs:
                r = det_tile(f[oy:oy + th, ox:ox + tw], imgsz=640, conf=min(DET_CONF_PERSON, DET_CONF_SKIS),
                             classes=[0, 30], verbose=False)[0]
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                    b = np.array(b) + np.array([ox, oy, ox, oy])
                    if int(k) == 0 and c >= DET_CONF_PERSON:
                        cx = min(max(int((b[0] + b[2]) / 2 * sx), 0), SMALL_W - 1)
                        cy = min(max(int((b[1] + b[3]) / 2 * sy), 0), SMALL_H - 1)
                        if not overlay_static[cy, cx]:
                            newp.append((b, float(c)))
                    elif int(k) == 30 and c >= DET_CONF_SKIS:
                        news.append((b, float(c)))
            merged = nms_merge(newp)
            added += max(0, len(merged) - len(persons))
            dets[t] = (merged, nms_merge(news))
        t += 1
    cap.release()
    return added


def skis_support(pbox, skis):
    """1 if a skis box sits around the person's feet."""
    h = pbox[3] - pbox[1]; w = pbox[2] - pbox[0]
    foot = np.array([pbox[0] - 0.5 * w, pbox[1] + 0.55 * h, pbox[2] + 0.5 * w, pbox[3] + 0.35 * h])
    best = 0.0
    for sb, sc in skis:
        cx, cy = (sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2
        if foot[0] <= cx <= foot[2] and foot[1] <= cy <= foot[3]:
            best = max(best, min(1.0, sc / 0.3))
    return best


# ------------------------------------------------------------------ stage 2: tracklets
def build_tracklets(det_frames, dets, cam, shot_of):
    """cam[i] = full-res similarity from det_frames[i-1] to det_frames[i] (or None)."""
    tracklets = []          # dict(frames, boxes, conf, skis, shot)
    active = []             # indices into tracklets
    for i, t in enumerate(det_frames):
        persons, skis = dets[t]
        new_shot = i == 0 or shot_of[t] != shot_of[det_frames[i - 1]]
        if new_shot:
            active = []
        A = cam[i] if i > 0 else None
        preds = []
        for ti in active:
            tr = tracklets[ti]
            pb = warp_box(tr["boxes"][-1], A)
            # constant velocity in ground coordinates (last step)
            if len(tr["gvel"]) and tr["last_i"] == i - 1:
                v = tr["gvel"][-1]
                pb = pb + np.array([v[0], v[1], v[0], v[1]])
            preds.append(pb)
        cost = np.full((len(active), len(persons)), 9.0)
        for a, pb in enumerate(preds):
            ph = pb[3] - pb[1]
            for b, (db, dc) in enumerate(persons):
                dh = db[3] - db[1]
                d = np.hypot((pb[0] + pb[2] - db[0] - db[2]) / 2, (pb[1] + pb[3] - db[1] - db[3]) / 2)
                rel = d / max(ph, dh, 8.0)
                size_ratio = max(ph, dh) / max(min(ph, dh), 1.0)
                if rel < 1.2 and size_ratio < 2.0:
                    cost[a, b] = (1 - iou(pb, db)) + 0.5 * rel
        assigned = set()
        if len(active) and len(persons):
            r, c = linear_sum_assignment(cost)
            for a, b in zip(r, c):
                if cost[a, b] < 1.6:
                    tr = tracklets[active[a]]
                    db, dc = persons[b]
                    # ground velocity = observed motion minus camera-predicted motion
                    cam_pred = warp_box(tr["boxes"][-1], A) if tr["last_i"] == i - 1 else None
                    if cam_pred is not None:
                        gv = ((db[0] + db[2]) / 2 - (cam_pred[0] + cam_pred[2]) / 2,
                              (db[1] + db[3]) / 2 - (cam_pred[1] + cam_pred[3]) / 2)
                        tr["gvel"].append(np.array(gv))
                        tr["gsteps"].append((np.array(gv), db[3] - db[1], A is not None))
                    tr["frames"].append(t); tr["boxes"].append(db); tr["conf"].append(dc)
                    tr["skis"].append(skis_support(db, skis)); tr["last_i"] = i
                    assigned.add(b)
        # unmatched detections start new tracklets
        for b, (db, dc) in enumerate(persons):
            if b in assigned:
                continue
            tracklets.append(dict(frames=[t], boxes=[db], conf=[dc], skis=[skis_support(db, skis)],
                                  gvel=[], gsteps=[], last_i=i, shot=shot_of[t]))
        # keep tracklets alive for 2 missed detection frames
        active = [k for k, tr in enumerate(tracklets) if i - tr["last_i"] <= 2 and tr["shot"] == shot_of[t]]
    return tracklets


def score_tracklet(tr, fps, W, H):
    n = len(tr["frames"])
    dur = (tr["frames"][-1] - tr["frames"][0] + 1) / fps
    hs = np.array([b[3] - b[1] for b in tr["boxes"]])
    bh = float(np.median(hs))
    # ground trajectory (only steps with a valid camera estimate)
    g = np.zeros(2); traj = [g.copy()]
    valid_steps = 0
    for gv, h, ok in tr["gsteps"]:
        if ok:
            g = g + gv; valid_steps += 1
        traj.append(g.copy())
    traj = np.array(traj)
    excursion = float(np.max(np.linalg.norm(traj - traj[0], axis=1))) / max(bh, 1.0)
    net = float(np.linalg.norm(traj[-1] - traj[0])) / max(bh, 1.0)
    speed = net / max(dur, 1e-3)
    # scale change relative to the camera zoom (approaching / receding skier)
    static = (dur >= STATIC_MIN_S and valid_steps >= 0.6 * max(len(tr["gsteps"]), 1)
              and excursion < STATIC_RADIUS)
    skis = float(np.mean(tr["skis"]))
    conf = float(np.mean(tr["conf"]))
    centres = np.array([((b[0] + b[2]) / 2 / W - 0.5, (b[1] + b[3]) / 2 / H - 0.5) for b in tr["boxes"]])
    centre = 1.0 - min(1.0, float(np.median(np.linalg.norm(centres, axis=1))) / 0.5)
    known = valid_steps / max(len(tr["gsteps"]), 1)
    move = min(1.0, excursion / 2.0) if known >= 0.5 else max(0.5, min(1.0, excursion / 2.0))
    size = min(1.0, bh / (0.12 * H))
    if static or n < MIN_TRACKLET_DETS:
        s = 0.0
    else:
        s = 0.35 * move + 0.30 * skis + 0.15 * conf + 0.12 * centre + 0.08 * size
    tr.update(score=float(s), static=bool(static), excursion=excursion, speed=speed,
              skis_frac=skis, conf_mean=conf, centre=centre, bh=bh, dur=dur)
    return s


def choose_chain(tracklets, cams_by_frame, fps):
    """Weighted interval scheduling with a plausibility check between links."""
    cand = [k for k, tr in enumerate(tracklets) if tr["score"] >= SCORE_FLOOR]
    cand.sort(key=lambda k: tracklets[k]["frames"][-1])
    best = {}   # k -> (value, prev)
    order = []
    for k in cand:
        tr = tracklets[k]
        own = len(tr["frames"]) * (tr["score"] - SCORE_FLOOR + 0.05)
        bv, bp = own, None
        for j in order:
            tj = tracklets[j]
            if tj["shot"] != tr["shot"] or tj["frames"][0] >= tr["frames"][0]:
                continue
            # allow a short overlap (the same skier is often split into two
            # tracklets that share a few frames around a fast camera move)
            overlap = sum(1 for f in tj["frames"] if f >= tr["frames"][0])
            if overlap > OVERLAP_DETS or tj["frames"][-1] >= tr["frames"][-1]:
                continue
            gap = max(0, tr["frames"][0] - tj["frames"][-1])
            # far-away re-entry right after the previous tracklet is implausible
            bj, bk = tj["boxes"][-1], tr["boxes"][0]
            d = np.hypot((bj[0] + bj[2] - bk[0] - bk[2]) / 2, (bj[1] + bj[3] - bk[1] - bk[3]) / 2)
            rel = d / max(tj["bh"], tr["bh"], 8.0)
            allowed = rel < 1.5 + 0.15 * gap
            pen = 0.0 if allowed else 0.5 * len(tr["frames"])
            v = best[j][0] + own - pen
            if v > bv:
                bv, bp = v, j
        best[k] = (bv, bp)
        order.append(k)
    if not best:
        return []
    # best chain per shot
    chain = []
    for shot in sorted({tracklets[k]["shot"] for k in best}):
        ks = [k for k in best if tracklets[k]["shot"] == shot]
        k = max(ks, key=lambda q: best[q][0])
        seq = []
        while k is not None:
            seq.append(k); k = best[k][1]
        chain.extend(reversed(seq))
    return chain


# ------------------------------------------------------------------ stage 3: pose on crops
def interpolate_track(chain, tracklets, T, stride):
    boxes = [None] * T; src = np.full(T, -1)
    for k in sorted(chain, key=lambda q: tracklets[q]["score"]):   # best written last
        tr = tracklets[k]
        fr, bx = tr["frames"], tr["boxes"]
        for a in range(len(fr)):
            boxes[fr[a]] = bx[a]; src[fr[a]] = k
            if a + 1 < len(fr) and fr[a + 1] - fr[a] <= 3 * stride:
                for t in range(fr[a] + 1, fr[a + 1]):
                    w = (t - fr[a]) / (fr[a + 1] - fr[a])
                    boxes[t] = (1 - w) * bx[a] + w * bx[a + 1]; src[t] = k
        # extend half a stride at both ends
        for t in range(max(0, fr[0] - stride // 2), fr[0]):
            boxes[t] = bx[0]; src[t] = k
        for t in range(fr[-1] + 1, min(T, fr[-1] + stride // 2 + 1)):
            boxes[t] = bx[-1]; src[t] = k
    return boxes, src


def pose_on_crop(pose_model, cr, sc, x0, y0, b, rot):
    """Pose on a (possibly rotated) crop; keypoints mapped back to full frame."""
    h, w = cr.shape[:2]
    img = cr if rot is None else cv2.rotate(cr, rot)
    r = pose_model(img, imgsz=POSE_CROP, conf=0.1, verbose=False)[0]
    if r.keypoints is None or not len(r.boxes):
        return None
    k_all = r.keypoints.data.cpu().numpy().copy()
    bx = r.boxes.xyxy.cpu().numpy()
    def unrot(x, y):
        if rot is None:
            return x, y
        if rot == cv2.ROTATE_180:
            return w - 1 - x, h - 1 - y
        if rot == cv2.ROTATE_90_CLOCKWISE:        # (x,y) -> (h-1-y, x)
            return y, h - 1 - x
        return w - 1 - y, x                       # counter-clockwise
    best, bestv = None, -1
    for j in range(len(bx)):
        k = k_all[j]
        xs, ys = unrot(k[:, 0], k[:, 1])
        k[:, 0] = xs / sc + x0; k[:, 1] = ys / sc + y0
        vis = k[:, 2] > 0.3
        if vis.sum() < 3:
            continue
        kb = np.array([k[vis, 0].min(), k[vis, 1].min(), k[vis, 0].max(), k[vis, 1].max()])
        v = iou(kb, b) + 0.3 * float(r.boxes.conf[j])
        if v > bestv:
            best, bestv = k, v
    if best is None or bestv < 0.15:
        return None
    return best


def run_pose(pose_model, path, boxes, W, H):
    T = len(boxes)
    kps = np.zeros((T, 17, 3), np.float32)
    cap = cv2.VideoCapture(str(path))
    t = 0
    while t < T:
        ok = cap.grab()
        if not ok:
            break
        b = boxes[t]
        if b is not None:
            _, f = cap.retrieve()
            w, h = b[2] - b[0], b[3] - b[1]
            side = max(w, h) * 1.5 + 16
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            crop = f[y0:y1, x0:x1]
            if crop.size:
                sc = POSE_CROP / max(crop.shape[:2])
                cr = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC if sc > 1 else cv2.INTER_AREA)
                best = pose_on_crop(pose_model, cr, sc, x0, y0, b, None)
                # inverted / sideways skier (flips, corks): try rotated crops
                if best is None or np.mean(np.sort(best[:, 2])[-8:]) < 0.5:
                    for rot in (cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                        k = pose_on_crop(pose_model, cr, sc, x0, y0, b, rot)
                        if k is not None and (best is None or np.mean(np.sort(k[:, 2])[-8:]) > np.mean(np.sort(best[:, 2])[-8:]) + 0.05):
                            best = k
                if best is not None:
                    kps[t] = best
        t += 1
    cap.release()
    return kps


# ------------------------------------------------------------------ main entry
def process_clip(path, out_dir, det_model, pose_model, debug_video=None, det_tile=None):
    t0 = time.time()
    path = Path(path)
    small, fps, W, H = read_small_frames(path)
    T = len(small)
    stride = 3 if fps > 45 else 2
    mask, overlay_static = overlay_mask(small)
    cuts = detect_cuts(small)
    bounds = [0] + cuts + [T]
    shot_of = np.zeros(T, int)
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        shot_of[a:b] = s
    # detection frames: every stride, plus the first frame of every shot
    det_frames = sorted(set(range(0, T, stride)) | set(cuts))
    dets = run_detection(det_model, path, det_frames, overlay_static, W, H)
    sx, sy = SMALL_W / W, SMALL_H / H
    cam = [None]
    for i in range(1, len(det_frames)):
        a, b = det_frames[i - 1], det_frames[i]
        if shot_of[a] != shot_of[b]:
            cam.append(None); continue
        pb = [p[0] * np.array([sx, sy, sx, sy]) for p in dets[a][0]]
        pb = [np.array([q[0] - 0.3 * (q[2] - q[0]), q[1] - 0.3 * (q[3] - q[1]),
                        q[2] + 0.3 * (q[2] - q[0]), q[3] + 0.3 * (q[3] - q[1])]) for q in pb]
        A, _ = camera_motion(small[a], small[b], mask, pb)
        cam.append(to_full(A, 1 / sx, 1 / sy))
    del small
    def track():
        trs = build_tracklets(det_frames, dets, cam, shot_of)
        for tr in trs:
            score_tracklet(tr, fps, W, H)
        ch = choose_chain(trs, cam, fps)
        bx, sr = interpolate_track(ch, trs, T, stride)
        return trs, ch, bx, sr
    tracklets, chain, boxes, src = track()
    # pass 2: zoomed tiles where the skier is missing inside a shot that has him elsewhere
    n_gap = 0
    if det_tile is not None:
        shots_with = {shot_of[t] for t in range(T) if boxes[t] is not None}
        gap = [t for t in det_frames if boxes[t] is None and shot_of[t] in shots_with]
        n_gap = len(gap)
        if gap:
            gap_detection(det_tile, path, gap, dets, overlay_static, W, H)
            tracklets, chain, boxes, src = track()
    kps = run_pose(pose_model, path, boxes, W, H)

    stem = path.stem
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{stem}_pose.npy", kps)
    # camera similarity between consecutive detection frames (full-res coords),
    # NaN where unknown -> lets features use ground-relative motion
    cam_arr = np.full((len(det_frames), 2, 3), np.nan, np.float32)
    for i, A in enumerate(cam):
        if A is not None:
            cam_arr[i] = A
    np.savez_compressed(out_dir / f"{stem}_cam.npz", det_frames=np.array(det_frames), cam=cam_arr)
    has_pose = (kps[:, :, 2] > 0.3).sum(1) >= 6
    shots = []
    for s, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        shots.append(dict(start=int(a), end=int(b),
                          tracked=int(sum(boxes[t] is not None for t in range(a, b))),
                          pose=int(has_pose[a:b].sum())))
    meta = dict(tracker="skier_tracker_v3", fps=float(fps), width=W, height=H, num_frames=T,
                cut_frames=[int(c) for c in cuts], n_camera_shots=len(bounds) - 1, shots=shots,
                det_stride=stride, track_frames=int(sum(b is not None for b in boxes)),
                tracked_with_pose=int(has_pose.sum()), pose_coverage=float(has_pose.mean()),
                bbox=[None if b is None else [float(v) for v in b] for b in boxes],
                chain=[dict(start=int(tracklets[k]["frames"][0]), end=int(tracklets[k]["frames"][-1]),
                            score=round(tracklets[k]["score"], 3), skis=round(tracklets[k]["skis_frac"], 2),
                            excursion=round(tracklets[k]["excursion"], 2)) for k in chain],
                n_tracklets=len(tracklets), n_gap_frames_retried=n_gap,
                n_static_rejected=int(sum(tr["static"] for tr in tracklets)),
                seconds=round(time.time() - t0, 1))
    with open(out_dir / f"{stem}_pose_meta.json", "w") as fh:
        json.dump(meta, fh)
    if debug_video:
        render_debug(path, debug_video, kps, boxes, src, tracklets, dets, chain)
    return meta


SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def render_debug(path, out, kps, boxes, src, tracklets, dets, chain):
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    by_frame = {}
    for k, tr in enumerate(tracklets):
        for f, b in zip(tr["frames"], tr["boxes"]):
            by_frame.setdefault(f, []).append((k, b))
    t = 0; last_other = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if t in by_frame:
            last_other = by_frame[t]
        for k, b in last_other:           # every candidate, grey (red if static)
            if k in chain:
                continue
            c = (0, 0, 255) if tracklets[k]["static"] else (160, 160, 160)
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 1)
            cv2.putText(f, f"{tracklets[k]['score']:.2f}", (int(b[0]), int(b[1]) - 3), 0, 0.45, c, 1)
        b = boxes[t]
        if b is not None:
            k = int(src[t]); tr = tracklets[k]
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 3)
            cv2.putText(f, f"SKIER s={tr['score']:.2f} skis={tr['skis_frac']:.2f} exc={tr['excursion']:.1f}",
                        (int(b[0]), max(int(b[1]) - 8, 15)), 0, 0.6, (0, 255, 0), 2)
            kp = kps[t]
            for a, c2 in SKEL:
                if kp[a, 2] > 0.3 and kp[c2, 2] > 0.3:
                    cv2.line(f, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[c2, 0]), int(kp[c2, 1])), (0, 255, 255), 2)
        else:
            cv2.putText(f, "no skier", (20, H - 60), 0, 1.0, (0, 0, 255), 2)
        cv2.putText(f, f"frame {t}", (W - 200, H - 30), 0, 0.8, (255, 255, 255), 2)
        vw.write(f); t += 1
    vw.release()


def load_models(model_dir=None):
    """YOLO11s detector (960 + 640 tiles) and YOLO11s-pose, run through OpenVINO only."""
    from ov_yolo import OVYolo
    d = Path(model_dir) if model_dir else Path(__file__).resolve().parent / "models"
    return (OVYolo(d / "yolo11s_openvino_model", "detect"),
            OVYolo(d / "yolo11s-pose_openvino_model", "pose"),
            OVYolo(d / "yolo11s640_openvino_model", "detect"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="pose_v3")
    ap.add_argument("--debug-dir", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--models", default=None, help="folder with the three *_openvino_model dirs")
    args = ap.parse_args()
    det, pose, tile = load_models(args.models)
    for v in args.videos:
        stem = Path(v).stem
        if args.skip_existing and (Path(args.out) / f"{stem}_pose_meta.json").exists():
            continue
        dbg = str(Path(args.debug_dir) / f"{stem}_v3.mp4") if args.debug_dir else None
        if dbg:
            Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
        try:
            m = process_clip(v, args.out, det, pose, dbg, det_tile=tile)
        except Exception as e:                       # keep going on a broken clip
            print(f"{stem}: FAILED {e!r}", flush=True)
            continue
        print(f"{stem}: cuts={m['cut_frames']} tracked={m['track_frames']}/{m['num_frames']} "
              f"pose={m['pose_coverage']:.2f} chain={[(c['start'], c['end'], c['score']) for c in m['chain']]} "
              f"static_rejected={m['n_static_rejected']} {m['seconds']}s", flush=True)
