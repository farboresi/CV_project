"""Compare YOLO pose models on the worst-tracked clips, and split the failures
into DETECTOR problems vs TRACKER problems.

Why: every cached track in roi_pose_dataset/ was made with yolov8n-pose (nano).
Before fine-tuning YOLO, check (a) whether a bigger off-the-shelf model already
finds the skier, and (b) whether the gaps in the cached tracks are frames where
the detector saw nobody, or frames where a detection existed but the tracker
dropped it.

What it does, per clip and per model:
  * runs the SAME candidate filter the tracker uses
    (temporal_tracker.collect_candidates, tiling included), so a "candidate"
    here is exactly what Viterbi would have been offered;
  * compares every candidate to the cached track position.

Frame categories (relative to the cached nano track):
  tracked             cached track has a pose on this frame
    - agree           model has a candidate on the tracked skier
    - miss            model has no candidate on the tracked skier
  gap_near            no cached pose, but within --near-frames of a tracked
                      frame in the same camera shot, so the skier's position
                      can be interpolated:
    - near_cand       model has a candidate at that position -> recoverable
    - other_cand      candidates exist, none at that position
    - no_cand         model returns nothing
  gap_far             long gap / no reference position:
    - any_cand / no_cand

How to read the result:
  * gap_near / near_cand is high for NANO      -> the detector saw the skier and
                                                  the TRACKER dropped it.
                                                  Retraining YOLO won't help.
  * near_cand rises a lot for m / x vs nano    -> bigger model fixes it; just
                                                  re-track with it.
  * no_cand stays high for every model         -> look at frames/*.jpg: if the
                                                  skier is visible there, that's
                                                  the case for fine-tuning.

Run from CV_Project/ in the cv2026 env, e.g.
    ../Project/cv2026/bin/python compare_pose_models.py --worst 10
    ../Project/cv2026/bin/python compare_pose_models.py \
        --clips xg_aspen_2026_men_002 xg_aspen_2026_men_011 --stride 1
Outputs go to pose_model_compare/: per_frame.csv, per_clip.csv, summary.csv,
frames/<clip>_<frame>.jpg (gap frames where no model found anyone).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

import temporal_tracker as tt

HERE = Path(__file__).resolve().parent
DEFAULT_MODELS = [
    str(HERE / "yolov8n-pose.pt"),
    str(HERE.parent / "Project" / "models" / "yolo11m-pose.pt"),
    "yolo11x-pose.pt",                      # auto-downloads (~120 MB) on first use
]
DEFAULT_POSE_DIRS = [HERE / "roi_pose_dataset", HERE / "roi_pose_dataset_xg2025men"]
DEFAULT_VIDEO_DIRS = [
    HERE / "long_clips",
    HERE / "xg2025men",
    HERE / "YT_clips",
    HERE.parent / "YT_clips",
    HERE.parent / "Project" / "long_clips",
    HERE.parent / "Project" / "AQA-7" / "Actions" / "ski_big_air",
]
KP_CONF = 0.2       # keypoint counts as present
MIN_KP = 3          # frame counts as tracked


# --------------------------------------------------------------------------- #
# cached track
# --------------------------------------------------------------------------- #
def load_track(stem, pose_dirs):
    for d in pose_dirs:
        npy, meta = d / f"{stem}_pose.npy", d / f"{stem}_pose_meta.json"
        if npy.is_file() and meta.is_file():
            return np.load(npy), json.loads(meta.read_text())
    return None, None


def track_geometry(pose):
    """Per-frame centre (x, y) and size of the cached skeleton, NaN if absent."""
    T = len(pose)
    centre = np.full((T, 2), np.nan)
    size = np.full(T, np.nan)
    for t in range(T):
        ok = pose[t, :, 2] > KP_CONF
        if ok.sum() < MIN_KP:
            continue
        xy = pose[t, ok, :2]
        centre[t] = xy.mean(0)
        size[t] = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1.0)
    return centre, size


def shot_id_per_frame(T, cuts):
    ids = np.zeros(T, dtype=int)
    for c in sorted(cuts or []):
        if 0 < c < T:
            ids[c:] += 1
    return ids


def reference_positions(centre, shot_ids, near_frames):
    """Interpolated skier position on untracked frames close to tracked ones
    in the same shot. Returns (ref_xy, is_near)."""
    T = len(centre)
    tracked = ~np.isnan(centre[:, 0])
    ref = centre.copy()
    is_near = np.zeros(T, dtype=bool)
    idx = np.flatnonzero(tracked)
    for t in np.flatnonzero(~tracked):
        same = idx[shot_ids[idx] == shot_ids[t]]
        if len(same) == 0:
            continue
        before, after = same[same < t], same[same > t]
        b = before[-1] if len(before) else None
        a = after[0] if len(after) else None
        if b is not None and a is not None and (t - b) <= near_frames and (a - t) <= near_frames:
            w = (t - b) / (a - b)
            ref[t] = (1 - w) * centre[b] + w * centre[a]
            is_near[t] = True
        else:
            nearest = b if (a is None or (b is not None and t - b <= a - t)) else a
            if nearest is not None and abs(t - nearest) <= near_frames:
                ref[t] = centre[nearest]
                is_near[t] = True
    return ref, is_near


# --------------------------------------------------------------------------- #
# clips
# --------------------------------------------------------------------------- #
def find_video(stem, video_dirs):
    for d in video_dirs:
        p = tt.resolve_video(stem, d)
        if p is not None:
            return p
    return None


def pick_worst(n, pose_dirs, video_dirs):
    rows = []
    for d in pose_dirs:
        for meta_path in d.glob("*_pose_meta.json"):
            stem = meta_path.name[: -len("_pose_meta.json")]
            try:
                cov = json.loads(meta_path.read_text()).get("pose_coverage")
            except Exception:
                continue
            if cov is None or find_video(stem, video_dirs) is None:
                continue
            rows.append((cov, stem))
    rows.sort()
    return [s for _, s in rows[:n]]


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def classify(cands, t, tracked, is_near, ref, ref_size, match_radius):
    if tracked[t] or is_near[t]:
        if not cands:
            hit = False
        else:
            d = [np.linalg.norm(np.asarray(c.center) - ref[t]) for c in cands]
            hit = min(d) <= match_radius * ref_size
        if tracked[t]:
            return "tracked", ("agree" if hit else "miss")
        if hit:
            return "gap_near", "near_cand"
        return "gap_near", ("other_cand" if cands else "no_cand")
    return "gap_far", ("any_cand" if cands else "no_cand")


def run(args):
    pose_dirs = [Path(p) for p in args.pose_dirs]
    video_dirs = [Path(p) for p in args.video_dirs]
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)

    clips = args.clips or pick_worst(args.worst, pose_dirs, video_dirs)
    if not clips:
        raise SystemExit("No clips found (check --pose-dirs / --video-dirs).")
    print(f"{len(clips)} clips: {', '.join(clips)}")

    models = {}
    for m in args.models:
        name = Path(m).stem
        mdl = YOLO(m)
        models[name] = mdl
    print("models:", ", ".join(models))

    rows = []
    for stem in clips:
        pose, meta = load_track(stem, pose_dirs)
        video = find_video(stem, video_dirs)
        if pose is None or video is None:
            print(f"  skip {stem}: track={pose is not None} video={video}")
            continue

        cap = cv2.VideoCapture(str(video))
        vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # cached coordinates may be at a different resolution than this file
        sx = vw / meta.get("width", vw)
        sy = vh / meta.get("height", vh)
        pose = pose.copy()
        pose[..., 0] *= sx
        pose[..., 1] *= sy

        T = len(pose)
        centre, size = track_geometry(pose)
        tracked = ~np.isnan(centre[:, 0])
        shots = shot_id_per_frame(T, meta.get("cut_frames"))
        ref, is_near = reference_positions(centre, shots, args.near_frames)
        med_size = np.nanmedian(size) if tracked.any() else max(vw, vh) * 0.1
        imgsz = args.imgsz or (640 if stem.isdigit() else 1280)
        for mdl in models.values():
            mdl.overrides["imgsz"] = imgsz
            if args.device:
                mdl.overrides["device"] = args.device

        print(f"\n{stem}: {T} frames, cached coverage {meta.get('pose_coverage', float('nan')):.2f}, "
              f"cuts {meta.get('cut_frames')}, imgsz {imgsz}")
        saved = 0
        t = 0
        while t < T:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok:
                break
            ref_size = size[t] if tracked[t] else med_size
            per_model = {}
            for name, mdl in models.items():
                t0 = time.perf_counter()
                cands = tt.collect_candidates(mdl, frame, tiling=not args.no_tiling)
                ms = (time.perf_counter() - t0) * 1000
                group, cat = classify(cands, t, tracked, is_near, ref, ref_size, args.match_radius)
                per_model[name] = cat
                rows.append(dict(clip=stem, frame=t, shot=int(shots[t]), model=name,
                                 group=group, category=cat, n_cands=len(cands),
                                 best_conf=max((c.conf for c in cands), default=0.0),
                                 max_size=max((c.size for c in cands), default=0.0),
                                 ms=ms))
            # save frames where every model came back empty on a gap
            if (not tracked[t] and all(c == "no_cand" for c in per_model.values())
                    and saved < args.max_frames):
                img = frame.copy()
                if is_near[t]:
                    cx, cy = map(int, ref[t])
                    r = int(max(ref_size, 20))
                    cv2.circle(img, (cx, cy), r, (0, 255, 255), 2)
                cv2.putText(img, f"{stem} f{t} all models: no candidate", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imwrite(str(out / "frames" / f"{stem}_{t:05d}.jpg"), img)
                saved += 1
            t += args.stride
        cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_frame.csv", index=False)
    if df.empty:
        raise SystemExit("No frames processed.")

    # per clip x model: fraction of frames in each group/category
    counts = (df.groupby(["clip", "model", "group", "category"]).size()
                .rename("n").reset_index())
    counts["frac_of_group"] = counts["n"] / counts.groupby(["clip", "model", "group"])["n"].transform("sum")
    counts.to_csv(out / "per_clip.csv", index=False)

    # headline summary per model
    summ = []
    for name, g in df.groupby("model", sort=False):
        tr = g[g.group == "tracked"]
        gn = g[g.group == "gap_near"]
        gf = g[g.group == "gap_far"]
        summ.append(dict(
            model=name,
            ms_per_frame=g.ms.median(),
            frames=len(g),
            any_cand=(g.n_cands > 0).mean(),
            tracked_agree=(tr.category == "agree").mean() if len(tr) else np.nan,
            gap_near_frames=len(gn),
            gap_near_found=(gn.category == "near_cand").mean() if len(gn) else np.nan,
            gap_near_empty=(gn.category == "no_cand").mean() if len(gn) else np.nan,
            gap_far_frames=len(gf),
            gap_far_any=(gf.category == "any_cand").mean() if len(gf) else np.nan,
        ))
    summ = pd.DataFrame(summ)
    summ.to_csv(out / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== summary (fractions) ===")
    print(summ.round(3).to_string(index=False))
    first = summ.iloc[0]
    print(f"""
How to read it:
  gap_near_found for {first.model} = {first.gap_near_found:.0%}: on gap frames next to the
    cached track, the nano detector already had the skier -> that share of the gaps
    is a TRACKER problem (Viterbi / gate / cut handling), not detection.
  gap_near_found rising for bigger models -> a model swap recovers those frames.
  gap_near_empty staying high for every model (nobody detected) -> open
    {out / 'frames'} and check by eye whether the skier is visible there;
    if yes, that is the case for fine-tuning.
  ms_per_frame -> cost of re-tracking everything with that model.""")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--clips", nargs="+", help="clip stems; default = --worst N by cached coverage")
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--stride", type=int, default=3, help="process every Nth frame")
    ap.add_argument("--imgsz", type=int, default=0, help="0 = 640 for AQA, 1280 for competition (as in re-track)")
    ap.add_argument("--device", default="", help="e.g. mps, cpu, 0")
    ap.add_argument("--no-tiling", action="store_true")
    ap.add_argument("--near-frames", type=int, default=30,
                    help="gap frames this close to a tracked frame get a reference position")
    ap.add_argument("--match-radius", type=float, default=0.75,
                    help="candidate counts as the skier within this many body sizes")
    ap.add_argument("--max-frames", type=int, default=8, help="empty-gap frames saved per clip")
    ap.add_argument("--pose-dirs", nargs="+", default=[str(p) for p in DEFAULT_POSE_DIRS])
    ap.add_argument("--video-dirs", nargs="+", default=[str(p) for p in DEFAULT_VIDEO_DIRS])
    ap.add_argument("--out", default=str(HERE / "pose_model_compare"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
