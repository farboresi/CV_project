"""Re-track competition clips with the whole-clip, cut-aware tracker.

Examples:
    python run_retrack_competition.py --scored-only
    python run_retrack_competition.py --limit 5
    python run_retrack_competition.py --all

The competition labels are read directly from ``labels_combined.csv`` so this
script never rewrites the shared label table.  Pose arrays and meta files go to
``roi_pose_dataset``; the summary is written separately from the AQA one.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from temporal_tracker import (flag_detection_gaps, process_aqa_dataset_temporal,
                              summarize_cached)

DATA_DIR = Path("C:/Users/franc/Downloads/AQA-7/AQA-7/Actions/ski_big_air")
OUTPUT_DIR = Path("roi_pose_dataset")
SKI_DIR = Path("ski_dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels_combined.csv")
    ap.add_argument("--scored-only", action="store_true",
                    help="only clips with a judges' score (91 clips)")
    ap.add_argument("--all", action="store_true",
                    help="process every competition clip (193 clips)")
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: first N clips after filtering")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--person-select", default="motion",
                    choices=["longest", "motion", "size"])
    ap.add_argument("--out-summary",
                    default="roi_pose_dataset/dataset_processed_features_competition.csv")
    ap.add_argument("--summary-only", action="store_true",
                    help="rebuild the summary from cached pose/meta files and "
                         "skip tracking entirely")
    ap.add_argument("--clips", nargs="*", default=None,
                    help="explicit competition clips to score/re-track")
    ap.add_argument("--flag-gaps", nargs="?", const=25, type=int, default=0,
                    metavar="N",
                    help="do not track: rank the cached clips worst-first by "
                         "pose coverage and longest missing run, write the "
                         "review CSV and print the top N")
    ap.add_argument("--review-out",
                    default="roi_pose_dataset/detection_review.csv",
                    help="where --flag-gaps writes its ranking")
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)
    labels["filename"] = labels["filename"].astype(str)
    comp = labels[labels["dataset"] == "competition"].copy()
    all_comp = comp.copy()
    if args.clips:
        wanted = {Path(str(c)).stem for c in args.clips}
        comp = comp[comp["filename"].map(lambda s: Path(str(s)).stem in wanted)]
        # Keep all_comp as the full competition set: the final summary is
        # rebuilt from every cached clip, so a targeted one-clip re-track must
        # not overwrite the shared 193-row summary with a single row.
    # Default to the 91 scored runs: they are the external validation set and
    # a full 193-clip pass is a multi-hour job.
    if not args.all:
        comp = comp[comp["target"].notna()]
    if args.limit:
        comp = comp.head(args.limit)

    if args.flag_gaps:
        review = flag_detection_gaps(all_comp, OUTPUT_DIR, SKI_DIR, DATA_DIR)
        out = Path(args.review_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(out, index=False)
        counts = review["severity"].value_counts().to_dict()
        print(f"wrote {out} ({len(review)} rows)  "
              + "  ".join(f"{k}={counts.get(k, 0)}"
                          for k in ("fail", "warn", "ok", "missing")),
              flush=True)
        show = review[review["severity"].isin(["fail", "warn"])].head(args.flag_gaps)
        if len(show):
            print(f"\n{'clip':38s} {'sev':5s} {'cov':>5s} {'gap':>5s} "
                  f"{'gap@':>12s} {'shot':>10s}", flush=True)
            for _, r in show.iterrows():
                span = (f"{int(r['longest_gap_start'])}-"
                        f"{int(r['longest_gap_end'])}")
                shot = "-" if pd.isna(r["gap_shot"]) else str(r["gap_shot"])
                print(f"{Path(str(r['filename'])).stem[:38]:38s} "
                      f"{r['severity']:5s} {r['pose_coverage']:5.2f} "
                      f"{int(r['longest_gap_frames']):5d} {span:>12s} "
                      f"{shot:>10s}", flush=True)
        return

    if args.summary_only:
        out = Path(args.out_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_cached(all_comp, OUTPUT_DIR, SKI_DIR, DATA_DIR)
        summary.to_csv(out, index=False)
        print(f"wrote {out} ({len(summary)} rows, rebuilt from cache)",
              flush=True)
        return
    print(f"competition clips to process: {len(comp)} "
          f"(scored {int(comp['target'].notna().sum())})", flush=True)

    t0 = time.time()
    summary = process_aqa_dataset_temporal(
        comp,
        output_dir=OUTPUT_DIR,
        ski_output_dir=SKI_DIR,
        data_dir=DATA_DIR,
        overwrite=True,
        max_gap=15,
        pose_conf=0.1,
        bbox_pad=0.25,
        imgsz=args.imgsz,
        track_mode="longest",
        cut_aware=True,
        person_select=args.person_select,
        summary_name=None,  # never clobber the combined summary
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed / 60:.1f} min", flush=True)
    if summary is not None and len(summary):
        out = Path(args.out_summary)
        out.parent.mkdir(parents=True, exist_ok=True)
        summary = summarize_cached(all_comp, OUTPUT_DIR, SKI_DIR, DATA_DIR)
        summary.to_csv(out, index=False)
        print(f"wrote {out} ({len(summary)} rows, rebuilt from cache)", flush=True)


if __name__ == "__main__":
    main()
