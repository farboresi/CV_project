"""Re-track the AQA clips with the whole-clip, cut-aware tracker.

Run from the ski_big_air folder:
    python run_retrack_aqa.py
    python run_retrack_aqa.py --only-mismatched --person-select event
    python run_retrack_aqa.py --clips 1026 "38" "67" --person-select event

Writes pose arrays and ``<clip>_pose_meta.json`` into ``roi_pose_dataset`` and
a fresh AQA-only summary to ``roi_pose_dataset/dataset_processed_features_aqa.csv``.
The summary is always rebuilt for **all** AQA clips from the cached files, so a
partial re-track does not shrink it.  The combined 427-row summary is never
touched; use ``build_combined_summary.py`` for that.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from dataset_sources import build_labels
from temporal_tracker import process_aqa_dataset_temporal, summarize_cached

DATA_DIR = Path("C:/Users/franc/Downloads/AQA-7/AQA-7/Actions/ski_big_air")
OUTPUT_DIR = Path("roi_pose_dataset")
SKI_DIR = Path("ski_dataset")


def _stem(row) -> str:
    video = row.get("video_path")
    if video is not None and not pd.isna(video):
        return Path(str(video)).stem
    return Path(str(row["filename"])).stem


def _meta_for(stem: str) -> dict:
    p = OUTPUT_DIR / f"{stem}_pose_meta.json"
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person-select", default="event",
                    choices=["longest", "event", "motion", "size"])
    ap.add_argument("--only-mismatched", action="store_true",
                    help="only clips whose cached meta already used another "
                         "person_select policy")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="explicit clip names/stems to re-track")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N selected clips")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the selected clips and exit")
    ap.add_argument("--out-summary",
                    default="roi_pose_dataset/dataset_processed_features_aqa.csv")
    args = ap.parse_args()

    # write=False: keep the combined 427-row labels_combined.csv intact.
    labels = build_labels(DATA_DIR, include_aqa=True,
                          include_competition=False, write=False)
    labels["filename"] = labels["filename"].astype(str)
    print(f"AQA clips: {len(labels)}", flush=True)

    todo = labels
    if args.clips:
        wanted = {Path(str(c)).stem for c in args.clips}
        todo = todo[todo["filename"].map(lambda s: Path(s).stem in wanted)]
        if todo.empty:
            print(f"no clip matches {sorted(wanted)}", flush=True)
            return

    if args.only_mismatched:
        keep = []
        for _, row in todo.iterrows():
            meta = _meta_for(_stem(row))
            if meta.get("person_select") != args.person_select:
                keep.append(row["filename"])
        todo = todo[todo["filename"].isin(keep)]

    if args.limit:
        todo = todo.head(args.limit)
    todo = todo.reset_index(drop=True)

    print(f"clips to re-track: {len(todo)} "
          f"(person_select={args.person_select})", flush=True)
    if args.dry_run:
        for name in todo["filename"].tolist():
            print(f"  {name}")
        return
    if todo.empty:
        print("nothing to do", flush=True)
        return

    t0 = time.time()
    process_aqa_dataset_temporal(
        todo,
        output_dir=OUTPUT_DIR,
        ski_output_dir=SKI_DIR,
        data_dir=DATA_DIR,
        overwrite=True,
        max_gap=15,
        pose_conf=0.1,
        bbox_pad=0.25,
        imgsz=640,
        track_mode="longest",
        cut_aware=True,
        # "event" prefers the segment with the most shoulder rotation and
        # vertical excursion; it fixes the approach-vs-jump case (1026-40)
        # and matches "longest" on the tested AQA clips.
        person_select=args.person_select,
        summary_name=None,  # never clobber the combined summary
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.1f} min", flush=True)

    out = Path(args.out_summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_cached(labels, OUTPUT_DIR, SKI_DIR, DATA_DIR)
    summary.to_csv(out, index=False)
    print(f"wrote {out} ({len(summary)} rows, rebuilt from cache)", flush=True)


if __name__ == "__main__":
    main()
