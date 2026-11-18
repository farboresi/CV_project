"""Parallel competition re-track (one process per clip, model per worker).

The competition clips are 13-20 s at 59.94 fps and 1280x720, so a single
process takes several minutes per clip.  On an 18-core CPU, four workers with
four torch threads each give a useful wall-clock speedup without changing the
tracker.

Usage:
    python run_retrack_competition_parallel.py --workers 4 --threads 4
    python run_retrack_competition_parallel.py --workers 4 --limit 8
    python run_retrack_competition_parallel.py --workers 4 --all
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("C:/Users/franc/Downloads/AQA-7/AQA-7/Actions/ski_big_air")
OUTPUT_DIR = Path("roi_pose_dataset")
SKI_DIR = Path("ski_dataset")

_MODEL = None
_IMGSZ = 1280


def _init_worker(imgsz, threads, tile_trigger=0):
    global _MODEL, _IMGSZ
    import torch
    torch.set_num_threads(int(threads))
    from ultralytics import YOLO
    import temporal_tracker as tt
    if tile_trigger:
        tt.TILE_TRIGGER_SIZE = int(tile_trigger)
    _MODEL = YOLO("yolov8n-pose.pt")
    _MODEL.overrides["imgsz"] = int(imgsz)
    _IMGSZ = int(imgsz)


def _process_one(item):
    idx, total, row, person_select = item
    from temporal_tracker import (process_video_pose_temporal, resolve_video,
                                  _write_empty_ski)
    filename = str(row["filename"])
    video = Path(str(row["video_path"])) if pd.notna(row.get("video_path")) else None
    if video is None or not video.is_file():
        video = resolve_video(filename, DATA_DIR)
    if video is None:
        return {"filename": filename, "error": "video missing"}
    video = Path(video)
    stem = video.stem
    try:
        pose, boxes, meta = process_video_pose_temporal(
            _MODEL, video, 0, None, max_gap=15, conf_thres=0.1,
            bbox_pad=0.25, track_mode="longest", cut_aware=True,
            person_select=person_select)
    except Exception as exc:
        return {"filename": filename, "error": f"{type(exc).__name__}: {exc}"}

    pose_file = OUTPUT_DIR / f"{stem}_pose.npy"
    pose21_file = OUTPUT_DIR / f"{stem}_pose21.npy"
    ski_file = SKI_DIR / f"{stem}_ski.npz"
    meta_file = OUTPUT_DIR / f"{stem}_pose_meta.json"
    np.save(pose_file, pose)
    pose21 = np.zeros((len(pose), 21, 3), dtype=np.float32)
    pose21[:, :17] = pose
    np.save(pose21_file, pose21)
    _write_empty_ski(ski_file, len(pose), boxes, meta["fps"],
                     meta["width"], meta["height"])
    meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {
        "filename": filename,
        "stem": stem,
        "pose_feature_path": str(pose_file),
        "ski_feature_path": str(ski_file),
        "num_frames": len(pose),
        "target": float(row["target"]) if pd.notna(row.get("target")) else np.nan,
        "track_frames": meta.get("track_frames"),
        "tracked_with_pose": meta.get("tracked_with_pose"),
        "pose_coverage": meta.get("pose_coverage"),
        "n_camera_shots": meta.get("n_camera_shots"),
        "cut_frames": meta.get("cut_frames"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_combined.csv")
    ap.add_argument("--dataset", default="competition",
                    choices=["competition", "aqa7"])
    ap.add_argument("--all", action="store_true",
                    help="process all clips of the dataset, not only scored")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clips", nargs="*", default=None,
                    help="only these clip stems (for targeted re-tracks)")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="0 = 1280 for competition, 640 for AQA")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tile-trigger", type=int, default=0,
                    help="override TILE_TRIGGER_SIZE (px); 0 keeps default")
    ap.add_argument("--person-select", default=None,
                    choices=["longest", "motion", "size", "event"],
                    help="default: motion for competition, event for AQA")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.imgsz:
        args.imgsz = 1280 if args.dataset == "competition" else 640
    if args.person_select is None:
        args.person_select = ("motion" if args.dataset == "competition"
                              else "event")
    if args.out is None:
        args.out = (f"roi_pose_dataset/dataset_processed_features_"
                    f"{args.dataset}.csv")

    labels = pd.read_csv(args.labels, dtype={"filename": str})
    labels["filename"] = labels["filename"].astype(str)
    comp = labels[labels["dataset"] == args.dataset].copy()
    if args.clips:
        wanted = {Path(str(c)).stem for c in args.clips}
        comp = comp[comp["filename"].map(
            lambda s: Path(str(s)).stem in wanted
            or str(s).zfill(3) in wanted)]
    if not args.all:
        comp = comp[comp["target"].notna()]
    if args.limit:
        comp = comp.head(args.limit)
    comp = comp.reset_index(drop=True)
    total = len(comp)
    print(f"{args.dataset} clips: {total} "
          f"(scored {int(comp['target'].notna().sum())}), "
          f"workers={args.workers}, threads={args.threads}, "
          f"imgsz={args.imgsz}, select={args.person_select}, "
          f"tile_trigger={args.tile_trigger}", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SKI_DIR.mkdir(parents=True, exist_ok=True)
    items = [(i + 1, total, comp.iloc[i].to_dict(), args.person_select)
             for i in range(total)]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
            initargs=(args.imgsz, args.threads, args.tile_trigger)) as pool:
        futures = {pool.submit(_process_one, item): item for item in items}
        for done, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            elapsed = time.time() - t0
            if "error" in rec:
                print(f"[{done}/{total}] {rec['filename']}: ERROR {rec['error']} "
                      f"({elapsed / 60:.1f} min)", flush=True)
            else:
                print(f"[{done}/{total}] {rec['filename']}: "
                      f"{rec['tracked_with_pose']}/{rec['num_frames']} pose "
                      f"({rec['pose_coverage']:.1%}), "
                      f"{rec['n_camera_shots']} shots, "
                      f"{elapsed / 60:.1f} min elapsed", flush=True)

    df = pd.DataFrame(records)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nwrote {out} ({len(df)} rows) in "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
