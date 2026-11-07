"""Phase 0: build a per-clip QA manifest for the pose / ski tracks.

Scans every label in labels_CV.csv, resolves its video and cached outputs, and
writes one row per clip with coverage, confidence, length and quality flags.

Outputs
-------
clip_manifest.csv          one row per clip
clip_manifest_summary.txt  human-readable summary
clip_manifest_quality.png  coverage / quality plots

Usage
-----
    python build_clip_manifest.py
    python build_clip_manifest.py --labels labels_CV.csv --out qa/clip_manifest.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# QA thresholds
POSE_ANY_CONF = 0.10
BODY_CONF = 0.30
BODY_MIN_KPTS = 8
ANKLE_CONF = 0.20
SKI_CONF = 0.15
LOW_BODY_FRAC = 0.20
LOW_SKI_FRAC = 0.10


def resolve_video(name, data_dir):
    name = str(name)
    for candidate in (f"{name}.mp4", f"{name.zfill(3)}.mp4",
                       f"{name.zfill(3)}.avi", f"{name}.avi"):
        path = Path(data_dir) / candidate
        if path.is_file():
            return path
    return None


def find_cached(stem, directory, suffix):
    stem = str(stem)
    for candidate in (stem, stem.zfill(3)):
        path = Path(directory) / f"{candidate}{suffix}"
        if path.exists():
            return path
    return None


def video_meta(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    meta = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "n_frames_video": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return meta


def safe_frac(mask):
    mask = np.asarray(mask, dtype=bool)
    return float(mask.mean()) if mask.size else 0.0


def body_box(frame_keypoints, pad=0.25, min_conf=0.20):
    """Padded body box from one frame's keypoints (used when skis are absent)."""
    kp = np.asarray(frame_keypoints, dtype=np.float32)
    visible = kp[:, 2] >= min_conf
    if int(visible.sum()) < 4:
        return None
    xs, ys = kp[visible, 0], kp[visible, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_px = pad * max(w, h)
    return (x1 - pad_px, y1 - pad_px, w + 2 * pad_px, h + 2 * pad_px)


def analyse_clip(row, data_dir, pose_dir, ski_dir, splits):
    name = str(row["filename"])
    video = resolve_video(name, data_dir)
    stem = video.stem if video else name

    record = {
        "filename": name,
        "stem": stem,
        "video_path": str(video) if video else "",
        "target": float(row.get("target", np.nan)),
        "fall": int(row.get("fall", 0)) if pd.notna(row.get("fall", np.nan)) else np.nan,
        "split": splits.get(name, ""),
    }

    if video is None:
        record.update({"width": np.nan, "height": np.nan, "fps": np.nan,
                       "n_frames_video": np.nan, "domain": "missing", "res_class": "missing"})
    else:
        meta = video_meta(video) or {}
        record.update(meta)
        record["domain"] = "mp4" if video.suffix.lower() == ".mp4" else "avi"
        record["res_class"] = "HD" if (meta.get("width", 0) >= 960) else "lowres"

    pose_path = find_cached(stem, pose_dir, "_pose.npy")
    pose21_path = find_cached(stem, pose_dir, "_pose21.npy")
    ski_path = find_cached(stem, ski_dir, "_ski.npz")

    record.update({
        "has_pose": pose_path is not None,
        "has_pose21": pose21_path is not None,
        "has_ski": ski_path is not None,
        "n_frames_pose": 0,
        "n_frames_pose21": 0,
        "n_frames_ski": 0,
        "pose_det_frac": 0.0,
        "body8_frac": 0.0,
        "ankle_frac": 0.0,
        "mean_conf_detected": 0.0,
        "bbox_frac": 0.0,
        "ski_any_frac": 0.0,
        "ski_both_frac": 0.0,
        "mean_ski_conf": 0.0,
        "mean_ski_len_rel": 0.0,
        "median_bbox_h_px": np.nan,
        "median_bbox_h_frac": np.nan,
        "pose_bbox_frac": 0.0,
        "pose_median_bbox_h_px": np.nan,
        "pose_median_bbox_h_frac": np.nan,
        "n_tracked_frames": 0,
        "median_ski_len_px": np.nan,
        "pose_motion": np.nan,
        "size_score": 0.0,
    })

    pose_boxes = []
    if pose_path is not None:
        pose = np.load(pose_path, mmap_mode="r")
        if pose.ndim == 3 and pose.shape[1] >= 17:
            record["n_frames_pose"] = int(pose.shape[0])
            conf = np.asarray(pose[:, :17, 2], dtype=np.float32)
            xy_present = conf.max(axis=1) > POSE_ANY_CONF
            # Ignore leading/trailing zero padding from the temporal tracker.
            span = slice(0, len(conf))
            if xy_present.any():
                idx = np.flatnonzero(xy_present)
                span = slice(int(idx[0]), int(idx[-1]) + 1)
            conf_span = conf[span]
            present_span = xy_present[span]
            record["n_tracked_frames"] = int(span.stop - span.start)
            body_count = (conf_span > BODY_CONF).sum(axis=1)
            record["pose_det_frac"] = safe_frac(present_span)
            record["body8_frac"] = safe_frac(body_count >= BODY_MIN_KPTS)
            record["ankle_frac"] = safe_frac(
                (conf_span[:, 15] > ANKLE_CONF) | (conf_span[:, 16] > ANKLE_CONF))
            if present_span.any():
                record["mean_conf_detected"] = float(conf_span[present_span].mean())
            for i in range(span.start, span.stop):
                box = body_box(pose[i, :17])
                if box is not None:
                    pose_boxes.append(box)
            if pose_boxes:
                heights = np.asarray([b[3] for b in pose_boxes], dtype=np.float32)
                record["pose_bbox_frac"] = len(pose_boxes) / max(1, span.stop - span.start)
                record["pose_median_bbox_h_px"] = float(np.median(heights))
                if pd.notna(record.get("height", np.nan)) and record["height"]:
                    record["pose_median_bbox_h_frac"] = float(
                        record["pose_median_bbox_h_px"] / record["height"])

    if pose21_path is not None:
        pose21 = np.load(pose21_path, mmap_mode="r")
        if pose21.ndim == 3:
            record["n_frames_pose21"] = int(pose21.shape[0])

    if ski_path is not None:
        with np.load(ski_path) as ski:
            conf = np.asarray(ski["conf"], dtype=np.float32)
            bbox = np.asarray(ski["bbox"], dtype=np.float32)
            length = np.asarray(ski["length"], dtype=np.float32)
            record["n_frames_ski"] = int(len(conf))
            if len(conf):
                detected = conf > SKI_CONF
                record["ski_any_frac"] = safe_frac(detected.any(axis=1))
                record["ski_both_frac"] = safe_frac(detected.all(axis=1))
                valid = detected & np.isfinite(length)
                if valid.any():
                    record["mean_ski_conf"] = float(conf[valid].mean())
                bbox_ok = np.isfinite(bbox).all(axis=1)
                record["bbox_frac"] = safe_frac(bbox_ok)
                rel = valid & bbox_ok[:, None]
                if rel.any():
                    scale = np.maximum(bbox[:, 2], bbox[:, 3])[:, None]
                    ratios = length[rel] / np.maximum(scale.repeat(2, axis=1)[rel], 1e-6)
                    record["mean_ski_len_rel"] = float(np.median(ratios))
                if bbox_ok.any():
                    heights = bbox[bbox_ok][:, 3]
                    record["median_bbox_h_px"] = float(np.median(heights))
                    if pd.notna(record.get("height", np.nan)) and record["height"]:
                        record["median_bbox_h_frac"] = float(np.median(heights) / record["height"])
                if valid.any():
                    record["median_ski_len_px"] = float(np.median(length[valid]))

    # Relative body motion: a static false detection (a logo, a banner) has
    # almost no keypoint motion inside its own bbox.
    if pose_path is not None and ski_path is not None and "bbox" in locals():
        motion_pose = np.load(pose_path, mmap_mode="r")
        centers = bbox[:, :2] + bbox[:, 2:4] / 2.0
        relative = []
        for i in range(min(len(motion_pose), len(bbox))):
            box = bbox[i]
            if not np.isfinite(box).all() or box[3] <= 1.0:
                continue
            conf_i = np.asarray(motion_pose[i, :17, 2], dtype=np.float32)
            ok_kpts = conf_i > BODY_CONF
            if ok_kpts.sum() < 4:
                continue
            xy = (np.asarray(motion_pose[i, :17, :2], dtype=np.float32) - centers[i]) / box[3]
            relative.append((xy, ok_kpts))
        if len(relative) >= 5:
            stds = []
            for keypoint in range(17):
                values = np.array([xy[keypoint] for xy, ok in relative if ok[keypoint]])
                if len(values) >= 5:
                    stds.append(float(values.std(axis=0).mean()))
            if stds:
                record["pose_motion"] = float(np.mean(stds))

    lengths = [record["n_frames_video"], record["n_frames_pose"],
               record["n_frames_pose21"], record["n_frames_ski"]]
    lengths = [v for v in lengths if pd.notna(v) and v > 0]
    record["length_mismatch"] = bool(
        len(lengths) >= 2 and (max(lengths) - min(lengths)) > 1
    )

    flags = []
    if video is None:
        flags.append("missing_video")
    if pose_path is None:
        flags.append("missing_pose")
    if pose21_path is None:
        flags.append("missing_pose21")
    if ski_path is None:
        flags.append("missing_ski")
    if record["length_mismatch"]:
        flags.append("length_mismatch")
    if record["body8_frac"] < LOW_BODY_FRAC:
        flags.append("low_body")
    if record["ankle_frac"] < 0.10:
        flags.append("low_ankle")
    if record["has_ski"] and record["ski_any_frac"] == 0.0 and record["body8_frac"] >= LOW_BODY_FRAC:
        flags.append("empty_ski_output")
    elif record["has_ski"] and record["ski_both_frac"] < LOW_SKI_FRAC:
        flags.append("low_ski")
    body_h = record["median_bbox_h_px"]
    if not np.isfinite(body_h):
        body_h = record["pose_median_bbox_h_px"]
    body_box_frac = record["bbox_frac"]
    if not record["has_ski"] or body_box_frac <= 0:
        body_box_frac = record["pose_bbox_frac"]

    tiny = bool((np.isfinite(body_h) and body_h < 40.0) or
                (np.isfinite(record["median_ski_len_px"]) and record["median_ski_len_px"] < 12.0))
    static = bool(np.isfinite(record["pose_motion"]) and record["pose_motion"] < 0.02)
    if tiny:
        flags.append("tiny_subject")
    if static:
        flags.append("static_pose")
    record["flags"] = ";".join(flags)

    size_reference = body_h
    if not np.isfinite(size_reference):
        size_reference = record["median_ski_len_px"]
    record["size_score"] = float(np.clip(size_reference / 80.0, 0.0, 1.0)) \
        if np.isfinite(size_reference) else 0.0

    # Body-only quality: skis are not part of the active pipeline.
    record["qa_score"] = float(np.clip(
        0.60 * record["body8_frac"]
        + 0.20 * body_box_frac
        + 0.20 * record["size_score"], 0.0, 1.0
    ))

    # Practical inclusion flag for body-only feature extraction / modelling.
    record["usable"] = bool(
        record["has_pose"]
        and record["n_tracked_frames"] >= 8
        and record["body8_frac"] >= 0.30
        and body_box_frac >= 0.30
        and not tiny
        and not static
    )
    return record


def write_summary(df, path):
    lines = ["Clip manifest summary", "=" * 72, ""]
    lines.append(f"clips: {len(df)}")
    present = {k: int(df[k].sum()) for k in ("has_pose", "has_pose21", "has_ski")}
    lines.append(f"cached outputs: pose={present['has_pose']} pose21={present['has_pose21']} "
                 f"ski={present['has_ski']}")
    lines.append(f"videos missing: {int((df['domain'] == 'missing').sum())}")
    lines.append(f"length mismatches: {int(df['length_mismatch'].sum())}")
    lines.append("")

    cols = ["target", "fall", "pose_det_frac", "body8_frac", "ankle_frac",
            "bbox_frac", "ski_any_frac", "ski_both_frac", "mean_ski_len_rel",
            "median_bbox_h_px", "median_ski_len_px", "size_score", "qa_score"]
    agg = df.groupby("domain")[cols].mean(numeric_only=True)
    counts = df.groupby("domain").size().rename("n")
    table = counts.to_frame().join(agg).round(3)
    lines.append("By domain (means):")
    lines.append(table.to_string())
    lines.append("")

    lines.append("Films by resolution class:")
    lines.append(df.groupby(["res_class", "domain"]).size().to_string())
    lines.append("")

    flags = {}
    for value in df["flags"].fillna(""):
        for flag in str(value).split(";"):
            if flag:
                flags[flag] = flags.get(flag, 0) + 1
    lines.append("Flag counts:")
    for flag, count in sorted(flags.items(), key=lambda item: -item[1]):
        lines.append(f"  {flag:22s} {count:4d}")
    lines.append("")

    empty_ski = int(((df["has_ski"]) & (df["ski_any_frac"] == 0)).sum())
    lines.append(f"ski files with zero detections: {empty_ski}/{int(df['has_ski'].sum())}")
    lines.append("")
    lines.append(f"usable clips (body-only: tracked>=8, body8>=0.30, bbox>=0.30, "
                 f"not tiny/static): {int(df['usable'].sum())}/{len(df)}")
    lines.append(df[df["usable"]].groupby("domain").size().to_string())
    lines.append("")

    lines.append("Worst 10 clips by qa_score:")
    worst = df.nsmallest(10, "qa_score")[
        ["filename", "domain", "body8_frac", "ski_both_frac", "median_bbox_h_px", "qa_score", "flags"]]
    lines.append(worst.to_string(index=False))
    lines.append("")
    lines.append("Best 10 clips by qa_score:")
    best = df.nlargest(10, "qa_score")[
        ["filename", "domain", "body8_frac", "ski_both_frac", "median_bbox_h_px", "qa_score", "flags"]]
    lines.append(best.to_string(index=False))

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def make_plot(df, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("[WARN] matplotlib unavailable; skipping plot:", exc)
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=120)

    ax = axes[0, 0]
    for domain, group in df.groupby("domain"):
        ax.hist(group["body8_frac"], bins=np.linspace(0, 1, 21), alpha=0.55, label=domain)
    ax.set_title("Body coverage (fraction of frames with >=8 confident joints)")
    ax.set_xlabel("body8_frac")
    ax.set_ylabel("clips")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for domain, group in df.groupby("domain"):
        ax.hist(group["ski_both_frac"], bins=np.linspace(0, 1, 21), alpha=0.55, label=domain)
    ax.set_title("Ski coverage (fraction of frames with both skis detected)")
    ax.set_xlabel("ski_both_frac")
    ax.set_ylabel("clips")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    scatter = ax.scatter(df["qa_score"], df["target"], c=df["fall"].fillna(0),
                         cmap="coolwarm", s=24, alpha=0.8)
    ax.set_xlabel("qa_score")
    ax.set_ylabel("target score")
    ax.set_title("Target vs clip QA (colour = fall)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for domain, group in df.groupby("domain"):
        ax.hist(group["median_bbox_h_px"].dropna(), bins=30, alpha=0.55, label=domain)
    ax.axvline(40, color="k", linestyle="--", linewidth=1, label="tiny_subject < 40px")
    ax.set_title("Subject size (median bbox height, px)")
    ax.set_xlabel("median bbox height (px)")
    ax.set_ylabel("clips")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default="labels_CV.csv")
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--pose-dir", default="roi_pose_dataset")
    parser.add_argument("--ski-dir", default="ski_dataset")
    parser.add_argument("--splits", default="splits.csv")
    parser.add_argument("--out", default="clip_manifest.csv")
    parser.add_argument("--summary", default="clip_manifest_summary.txt")
    parser.add_argument("--plot", default="clip_manifest_quality.png")
    args = parser.parse_args()

    labels = pd.read_csv(args.labels)
    splits = {}
    if Path(args.splits).is_file():
        split_df = pd.read_csv(args.splits, dtype={"filename": str})
        splits = dict(zip(split_df["filename"], split_df["split"]))

    records = []
    for _, row in labels.iterrows():
        records.append(analyse_clip(row, args.data_dir, args.pose_dir, args.ski_dir, splits))
        if len(records) % 25 == 0:
            print(f"  scanned {len(records)}/{len(labels)} clips")

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    print(write_summary(df, args.summary))
    make_plot(df, args.plot)
    print(f"\nwrote {args.summary} and {args.plot}")


if __name__ == "__main__":
    main()
