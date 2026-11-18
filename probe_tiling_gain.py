"""Probe whether the tiled detector recovers the skier inside a tracking gap.

The early-return bug in ``collect_candidates`` meant the tiled pass was skipped
whenever the full-frame YOLO pass found *no* keypoints.  This script measures
how many clips would benefit from the fix without re-tracking them: for the
longest missing run of a cached clip it samples frames, predicts the skier box
by interpolating the neighbouring tracked boxes, and compares

* full-frame candidates with tiling disabled, and
* 2x2 tiled candidates,

counting only candidates close to the predicted skier box (so a random
spectator or sign does not count as a hit).

Usage:
    python probe_tiling_gain.py --top 20
    python probe_tiling_gain.py --clips oly_beijing_2022_men_010 men_013
    python probe_tiling_gain.py --all --samples 4

Writes ``tiling_gain_probe.csv`` (worst / most-improved first).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from temporal_tracker import (collect_candidates,
                              collect_candidates_tiled, resolve_video)

DATA_DIR = Path("C:/Users/franc/Downloads/AQA-7/AQA-7/Actions/ski_big_air")
POSE_DIR = Path("roi_pose_dataset")
SKI_DIR = Path("ski_dataset")


def _missing_runs(ok):
    runs, start = [], None
    for i, good in enumerate(ok):
        if not good and start is None:
            start = i
        elif good and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ok)))
    return runs


def _load_boxes(stem):
    path = SKI_DIR / f"{stem}_ski.npz"
    if not path.is_file():
        return None
    try:
        bbox = np.load(path)["bbox"]
    except Exception:
        return None
    ok = np.isfinite(bbox).all(axis=1)
    return bbox, ok


def _predicted_box(bbox, ok, t, lo, hi):
    """Linear interpolation between nearest tracked boxes inside [lo, hi).

    Both neighbours must exist: extrapolating from one side across a long gap
    (or, worse, across a camera cut) puts the predicted box nowhere near the
    skier, and then no detector can produce a valid "hit".
    """
    before = [i for i in range(t - 1, lo - 1, -1) if ok[i]]
    after = [i for i in range(t + 1, hi) if ok[i]]
    if not before or not after:
        return None
    t0, t1 = before[0], after[0]
    a = (t - t0) / float(t1 - t0)
    return (1 - a) * bbox[t0] + a * bbox[t1]


def _shot_for(t, shots, n_frames):
    """Shot [a, b) containing frame t; fall back to the whole clip."""
    for s in shots:
        a, b = int(s.get("start", 0)), int(s.get("end", n_frames))
        if a <= t < b:
            return a, b
    return 0, n_frames


def _hits(cands, pred, max_conf_dist=4.0):
    """Candidates consistent with the predicted skier box."""
    if pred is None:
        return 0
    px, py, pw, ph = (float(v) for v in pred)
    pc = np.array([px + pw / 2.0, py + ph / 2.0])
    psize = max(pw, ph, 1.0)
    hits = 0
    for c in cands:
        if c.conf < 0.15:
            continue
        kp = c.keypoints
        if kp is None or int((kp[:, 2] >= 0.3).sum()) < 5:
            continue
        if np.linalg.norm(c.center - pc) > max_conf_dist * psize:
            continue
        ratio = c.size / psize
        if 0.20 <= ratio <= 5.0:
            hits += 1
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_combined.csv")
    ap.add_argument("--dataset", default="competition")
    ap.add_argument("--review", default="roi_pose_dataset/detection_review.csv")
    ap.add_argument("--top", type=int, default=20,
                    help="probe the N worst cached clips")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--samples", type=int, default=6,
                    help="frames sampled inside the longest gap")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--out", default="tiling_gain_probe.csv")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels, dtype={"filename": str})
    labels = labels[labels["dataset"] == args.dataset].copy()
    if args.clips:
        wanted = {Path(str(c)).stem for c in args.clips}
        labels = labels[labels["filename"].map(
            lambda s: Path(str(s)).stem in wanted)]
    elif not args.all:
        review_path = Path(args.review)
        if review_path.is_file():
            review = pd.read_csv(review_path)
            review = review[review["severity"].isin(["fail", "warn"])]
            order = list(review["filename"].astype(str))
            rank = {Path(s).stem: i for i, s in enumerate(order)}
            labels = labels[labels["filename"].map(
                lambda s: Path(str(s)).stem in rank)].copy()
            labels["_rank"] = labels["filename"].map(
                lambda s: rank[Path(str(s)).stem])
            labels = labels.sort_values("_rank").head(args.top)
        else:
            labels = labels.head(args.top)

    import torch
    from ultralytics import YOLO
    torch.set_num_threads(2)
    model = YOLO("yolov8n-pose.pt")
    model.overrides["imgsz"] = int(args.imgsz)

    rows = []
    for pos, (_, row) in enumerate(labels.iterrows(), 1):
        video = row.get("video_path")
        if video is None or pd.isna(video) or not Path(str(video)).is_file():
            video = resolve_video(row["filename"], DATA_DIR)
        if video is None:
            continue
        video = Path(video)
        stem = video.stem
        loaded = _load_boxes(stem)
        if loaded is None:
            continue
        bbox, ok = loaded
        runs = _missing_runs(ok)
        if not runs:
            continue
        gap_lo, gap_hi = max(runs, key=lambda r: r[1] - r[0])
        gap_len = gap_hi - gap_lo
        if gap_len < 10:
            continue
        meta = {}
        meta_path = POSE_DIR / f"{stem}_pose_meta.json"
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}

        cap = cv2.VideoCapture(str(video))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        lo = max(0, gap_lo)
        hi = min(n_frames, gap_hi)
        sample_ts = np.linspace(lo, hi - 1, num=max(1, args.samples), dtype=int)
        sample_ts = sorted(set(int(t) for t in sample_ts))
        shots = meta.get("shots", []) or []
        full_hits = tile_hits = judged = 0
        for t in sample_ts:
            shot_lo, shot_hi = _shot_for(t, shots, n_frames)
            pred = _predicted_box(bbox, ok, t, shot_lo, shot_hi)
            if pred is None:
                # No tracked box on both sides of this frame inside its shot:
                # there is no skier prediction to compare the detectors to.
                continue
            judged += 1
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            good, frame = cap.read()
            if not good:
                continue
            base = collect_candidates(model, frame, conf_thres=0.1,
                                      bbox_pad=0.25, tiling=False)
            tiled = collect_candidates_tiled(model, frame, conf_thres=0.1,
                                             bbox_pad=0.25, grid=(2, 2))
            full_hits += 1 if _hits(base, pred) else 0
            tile_hits += 1 if _hits(tiled, pred) else 0
        cap.release()
        cov = float(ok.mean())
        row_out = {
            "clip": stem,
            "pose_coverage": round(cov, 3),
            "gap_start": int(gap_lo),
            "gap_end": int(gap_hi),
            "gap_frames": int(gap_len),
            "samples": judged,
            "full_hits": int(full_hits),
            "tile_hits": int(tile_hits),
            "tile_gain": int(tile_hits - full_hits),
            "n_shots": int(meta.get("n_camera_shots", 1) or 1),
        }
        rows.append(row_out)
        print(f"[{pos}/{len(labels)}] {stem:42s} cov {cov:4.2f} gap {gap_len:4d} "
              f"full {full_hits}/{judged} tile {tile_hits}/{judged}", flush=True)

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["tile_gain", "gap_frames"], ascending=False)
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out} ({len(out)} rows)")
        print(out.head(20).to_string(index=False))
    else:
        print("no clips probed")


if __name__ == "__main__":
    main()
