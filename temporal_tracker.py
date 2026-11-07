"""Temporal body-pose tracking for the B1 body-only pipeline.

The per-frame detector used in ``CV_project_rebuilt.ipynb`` sometimes alternates
between a good skier box and a missed frame.  This module adds an offline
temporal layer:

1. collect every YOLO pose candidate for every frame,
2. use Viterbi decoding to choose one consistent skier track across the clip,
3. interpolate short gaps in the track,
4. refine the pose on the tracked/interpolated boxes,
5. interpolate short keypoint gaps.

It is intentionally offline: the whole clip is available, so a confident
detection near the end can correct a weak detection near the beginning.

This module does not use the classical ski detector.  Ski columns are left at
zero for backward compatibility with ``_pose21.npy``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ski_detector import (
    bbox_from_keypoints,
    pose_keypoints_in_roi,
)


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    conf: float
    center: np.ndarray
    size: float


# --------------------------------------------------------------------------- #
# geometry / association
# --------------------------------------------------------------------------- #
def iou_xywh(a, b):
    ax1, ay1, aw, ah = (float(v) for v in a)
    bx1, by1, bw, bh = (float(v) for v in b)
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 1e-9 else 0.0


def _candidate_from_box(box, keypoints, conf, frame_shape, bbox_pad):
    x, y, w, h = (float(v) for v in box)
    bbox = bbox_from_keypoints(keypoints, frame_shape, pad=bbox_pad)
    if bbox is None:
        bbox = (x, y, w, h)
    bbox = tuple(float(v) for v in bbox)
    bx, by, bw, bh = bbox
    center = np.array([bx + bw / 2.0, by + bh / 2.0], dtype=np.float32)
    size = max(bw, bh, 1.0)
    return Candidate(bbox=bbox, keypoints=keypoints, conf=float(conf),
                     center=center, size=float(size))


def collect_candidates(model, frame, conf_thres=0.1, bbox_pad=0.25):
    """Return every detected person as a Candidate for one frame."""
    result = model(frame, verbose=False, conf=conf_thres)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return []

    boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else None
    box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
    xy = result.keypoints.xy.cpu().numpy()
    kconf = (result.keypoints.conf.cpu().numpy()
             if result.keypoints.conf is not None else np.ones(xy.shape[:2], dtype=np.float32))

    out = []
    for i in range(len(xy)):
        kp = np.zeros((xy.shape[1], 3), dtype=np.float32)
        kp[:, :2] = xy[i]
        kp[:, 2] = kconf[i]
        if boxes is not None and len(boxes) > i:
            box = boxes[i]
            conf = float(box_conf[i]) if box_conf is not None else float(kp[:, 2].mean())
        else:
            bbox = bbox_from_keypoints(kp, frame.shape[:2], pad=bbox_pad)
            if bbox is None:
                continue
            box = bbox
            conf = float(kp[:, 2].mean())
        out.append(_candidate_from_box(box, kp, conf, frame.shape[:2], bbox_pad))
    return out


def _transition_cost(a: Candidate, b: Candidate, iou_weight=1.0,
                     distance_weight=1.2, size_weight=0.8):
    scale = max((a.size + b.size) / 2.0, 1.0)
    distance = float(np.linalg.norm(a.center - b.center) / scale)
    aw, ah = float(a.bbox[2]), float(a.bbox[3])
    bw, bh = float(b.bbox[2]), float(b.bbox[3])
    size_cost = abs(math.log(max(bw, 1.0) / max(aw, 1.0))) + \
                abs(math.log(max(bh, 1.0) / max(ah, 1.0)))
    return distance_weight * distance + size_weight * size_cost + \
        iou_weight * (1.0 - iou_xywh(a.bbox, b.bbox))


def _entry_prior(candidate: Candidate, entry_bbox):
    if entry_bbox is None:
        return 0.0
    ex, ey, ew, eh = (float(v) for v in entry_bbox)
    entry_center = np.array([ex + ew / 2.0, ey + eh / 2.0], dtype=np.float32)
    scale = max(max(ew, eh), candidate.size, 1.0)
    center_cost = float(np.linalg.norm(candidate.center - entry_center) / scale)
    size_cost = abs(math.log(max(candidate.bbox[2], 1.0) / max(ew, 1.0))) + \
                abs(math.log(max(candidate.bbox[3], 1.0) / max(eh, 1.0)))
    return 0.6 * center_cost + 0.2 * size_cost


def viterbi_track(candidates, entry_bbox=None, max_gap=15,
                  emit_weight=1.0, miss_penalty=2.5):
    """Choose the best single track through per-frame candidates.

    ``candidates`` is a list indexed by relative frame; each entry is a list of
    Candidate objects.  Returns ``{frame_index: Candidate}`` for the chosen path.
    """
    first = next((t for t, c in enumerate(candidates) if c), None)
    if first is None:
        return {}

    # dp[t][candidate_index] = (cost, previous=(prev_t, prev_index) | None)
    dp = [dict() for _ in candidates]
    for i, cand in enumerate(candidates[first]):
        emission = -math.log(max(cand.conf, 1e-4)) * emit_weight
        dp[first][i] = (emission + _entry_prior(cand, entry_bbox), None)

    for t in range(first + 1, len(candidates)):
        if not candidates[t]:
            continue
        lookback = range(t - 1, max(first, t - max_gap - 1) - 1, -1)
        for i, cand in enumerate(candidates[t]):
            emission = -math.log(max(cand.conf, 1e-4)) * emit_weight
            best_cost, best_prev = float("inf"), None
            for k in lookback:
                if not dp[k]:
                    continue
                gap = t - k - 1
                for j, prev_cand in dp[k].items():
                    cost = prev_cand[0] + _transition_cost(candidates[k][j], cand) + \
                        miss_penalty * gap
                    if cost < best_cost:
                        best_cost, best_prev = cost, (k, j)
            if best_prev is not None:
                dp[t][i] = (best_cost + emission, best_prev)

    last = max((t for t in range(len(dp)) if dp[t]), default=None)
    if last is None:
        return {}

    best_idx = min(dp[last], key=lambda j: dp[last][j][0])
    path = {}
    t, j = last, best_idx
    while True:
        path[t] = candidates[t][j]
        prev = dp[t][j][1]
        if prev is None:
            break
        t, j = prev
    return path


# --------------------------------------------------------------------------- #
# gap filling
# --------------------------------------------------------------------------- #
def interpolate_boxes(path, n_frames, max_gap=15):
    """Fill short gaps between tracked boxes with linear interpolation."""
    if not path:
        return {}
    boxes = {t: np.asarray(c.bbox, dtype=np.float32) for t, c in path.items()}
    known = sorted(boxes)
    filled = dict(boxes)
    for t1, t2 in zip(known[:-1], known[1:]):
        gap = t2 - t1 - 1
        if gap <= 0 or gap > max_gap:
            continue
        b1, b2 = boxes[t1], boxes[t2]
        for t in range(t1 + 1, t2):
            alpha = (t - t1) / float(t2 - t1)
            filled[t] = (1 - alpha) * b1 + alpha * b2
    return filled


def interpolate_keypoints(seq, max_gap=15, min_conf=0.2):
    """Fill per-joint gaps by linear interpolation over short misses."""
    seq = np.asarray(seq, dtype=np.float32).copy()
    n = len(seq)
    for j in range(seq.shape[1]):
        valid = np.flatnonzero(seq[:, j, 2] >= min_conf)
        if len(valid) == 0:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            gap = b - a - 1
            if gap <= 0 or gap > max_gap:
                continue
            for t in range(a + 1, b):
                alpha = (t - a) / float(b - a)
                seq[t, j, :2] = (1 - alpha) * seq[a, j, :2] + alpha * seq[b, j, :2]
                seq[t, j, 2] = min(seq[a, j, 2], seq[b, j, 2]) * 0.6
    return seq


# --------------------------------------------------------------------------- #
# video processing
# --------------------------------------------------------------------------- #
def resolve_video(stem, data_dir):
    stem = str(stem)
    data_dir = Path(data_dir)
    candidates = (
        data_dir / f"{stem}.mp4",
        data_dir / f"{stem.zfill(3)}.avi",
        data_dir / f"{stem}.avi",
    )
    return next((p for p in candidates if p.is_file()), None)


def find_entry_frame(video_path, model, min_conf=0.15, max_scan_frames=150):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    while cap.isOpened() and frame_idx < max_scan_frames:
        ok, frame = cap.read()
        if not ok:
            break
        result = model(frame, verbose=False, conf=min_conf)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.cpu().numpy()
            best = int(np.argmax(confs))
            cx, cy, w, h = result.boxes.xywh.cpu().numpy()[best]
            cap.release()
            return frame_idx, (float(cx - w / 2), float(cy - h / 2), float(w), float(h))
        frame_idx += 1
    cap.release()
    return None, None


def process_video_pose_temporal(model, video_path, entry_idx, entry_bbox,
                                max_gap=15, conf_thres=0.1, bbox_pad=0.25,
                                min_visible=10):
    """Return ``(pose_seq, box_seq, meta)`` for a whole video.

    pose_seq: [T, 17, 3] absolute frame coordinates, zeros where no pose.
    box_seq : [T, 4] padded body boxes, NaN where no track.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Pass 1: collect all candidates from the entry frame onward.
    candidates = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, entry_idx)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        candidates.append(collect_candidates(model, frame, conf_thres=conf_thres,
                                             bbox_pad=bbox_pad))
    cap.release()

    path = viterbi_track(candidates, entry_bbox=entry_bbox, max_gap=max_gap)
    boxes_rel = interpolate_boxes(path, len(candidates), max_gap=max_gap)

    # Pass 2: refine pose on the tracked/interpolated boxes.
    pose_rel = np.zeros((len(candidates), 17, 3), dtype=np.float32)
    box_rel = np.full((len(candidates), 4), np.nan, dtype=np.float32)
    aligner = None
    try:
        from ski_detector import BodyAligner
        aligner = BodyAligner()
    except Exception:
        aligner = None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, entry_idx)
    for t in range(len(candidates)):
        ok, frame = cap.read()
        if not ok:
            break
        if t not in boxes_rel:
            continue
        box = boxes_rel[t]
        box_rel[t] = box
        kp = path[t].keypoints if t in path else np.zeros((17, 3), dtype=np.float32)
        visible = int((kp[:, 2] > 0.3).sum()) if kp.size else 0

        if visible < min_visible:
            refined = pose_keypoints_in_roi(
                model, frame, box, aligner=aligner,
                conf_thres=conf_thres, fallback_full_frame=False)
            if refined.size and int((refined[:, 2] > 0.3).sum()) >= visible:
                kp = refined
        pose_rel[t] = kp[:17]
    cap.release()

    pose_rel = interpolate_keypoints(pose_rel, max_gap=max_gap)

    pose_full = np.zeros((total, 17, 3), dtype=np.float32)
    box_full = np.full((total, 4), np.nan, dtype=np.float32)
    end = min(total, entry_idx + len(pose_rel))
    pose_full[entry_idx:end] = pose_rel[:end - entry_idx]
    box_full[entry_idx:end] = box_rel[:end - entry_idx]

    meta = {
        "entry_frame": int(entry_idx),
        "track_frames": int(len(path)),
        "tracked_with_pose": int((pose_full[:, :, 2].max(axis=1) > 0).sum()),
        "fps": fps,
        "width": width,
        "height": height,
    }
    return pose_full, box_full, meta


COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def _draw_pose(frame, keypoints, threshold=0.3):
    for a, b in COCO_SKELETON:
        if keypoints[a, 2] >= threshold and keypoints[b, 2] >= threshold:
            cv2.line(frame, (int(keypoints[a, 0]), int(keypoints[a, 1])),
                     (int(keypoints[b, 0]), int(keypoints[b, 1])),
                     (0, 255, 255), 2)
    for x, y, c in keypoints:
        if c >= threshold:
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 2, (255, 255, 255), -1)
    return frame


def visualize_temporal_tracking(labels_df, video_index=0, output_dir="ski_visual_checks",
                                save_video=True, pose_model_path="yolov8n-pose.pt",
                                max_gap=15):
    """Render the Viterbi/gap-filled track for one clip."""
    from ultralytics import YOLO

    row = labels_df.iloc[video_index]
    video_file = row.get("video_path")
    if video_file is None or pd.isna(video_file):
        raise ValueError("labels_df needs a valid video_path column")
    video_file = Path(video_file)
    stem = video_file.stem
    target = row.get("target", np.nan)

    model = YOLO(pose_model_path)
    entry_idx, entry_bbox = find_entry_frame(video_file, model)
    if entry_bbox is None:
        print(f"[FAIL] No skier detected during entry scan of {stem}")
        return None
    print(f"\n--- Temporal visualization: {stem} ---")
    print(f"Ground truth score: {target} | entry frame {entry_idx}")

    pose, boxes, meta = process_video_pose_temporal(
        model, video_file, entry_idx, entry_bbox, max_gap=max_gap)
    print(f"Track: {meta['track_frames']} linked frames | "
          f"pose in {meta['tracked_with_pose']}/{len(pose)} frames")

    cap = cv2.VideoCapture(str(video_file))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    contact = []

    for t in range(len(pose)):
        ok, frame = cap.read()
        if not ok:
            break
        display = frame.copy()
        box = boxes[t]
        if np.isfinite(box).all():
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 128, 0), 2)
            cv2.putText(display, "Temporal body box", (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
        if pose[t, :, 2].max() > 0:
            display = _draw_pose(display, pose[t])
        cv2.putText(display, f"frame {t} | score {target}",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if save_video:
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_dir / f"{stem}_temporal_check.mp4"),
                    cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (display.shape[1], display.shape[0]))
            writer.write(display)
        step = max(1, (len(pose) - entry_idx) // 6)
        if t >= entry_idx and (t - entry_idx) % step == 0 and len(contact) < 6:
            contact.append(cv2.resize(display, (480, 270)))

    cap.release()
    if writer is not None:
        writer.release()

    if contact:
        while len(contact) < 6:
            contact.append(np.zeros_like(contact[0]))
        sheet = np.vstack([np.hstack(contact[i:i + 2]) for i in range(0, 6, 2)])
        sheet_path = out_dir / f"{stem}_temporal_contact_sheet.jpg"
        cv2.imwrite(str(sheet_path), sheet)
        print(f"[SAVED] {sheet_path}")
        try:
            from IPython.display import Image, display as ipy_display
            ipy_display(Image(filename=str(sheet_path)))
        except Exception:
            pass
        return sheet_path
    return None


def _write_empty_ski(path, n_frames, boxes, fps, width, height):
    np.savez_compressed(
        path,
        quads=np.full((n_frames, 2, 8), np.nan, dtype=np.float32),
        conf=np.zeros((n_frames, 2), dtype=np.float32),
        angle=np.full((n_frames, 2), np.nan, dtype=np.float32),
        length=np.zeros((n_frames, 2), dtype=np.float32),
        bbox=np.asarray(boxes, dtype=np.float32),
        fps=np.float32(fps),
        width=np.int32(width),
        height=np.int32(height),
    )


def process_aqa_dataset_temporal(labels_df, output_dir, ski_output_dir, data_dir,
                                 pose_model_path="yolov8n-pose.pt", overwrite=True,
                                 max_gap=15, pose_conf=0.1, bbox_pad=0.25):
    """Drop-in replacement for the notebook's ``process_aqa_dataset``."""
    from ultralytics import YOLO

    output_path = Path(output_dir)
    ski_path = Path(ski_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ski_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(pose_model_path)

    records = []
    for idx, row in labels_df.iterrows():
        video_file = row.get("video_path")
        if video_file is None or pd.isna(video_file) or not Path(video_file).is_file():
            video_file = resolve_video(row["filename"], data_dir)
        if video_file is None:
            print(f"[SKIP] {row['filename']}: video missing")
            continue
        video_file = Path(video_file)
        stem = video_file.stem
        target = row.get("target", np.nan)
        pose_file = output_path / f"{stem}_pose.npy"
        pose21_file = output_path / f"{stem}_pose21.npy"
        ski_file = ski_path / f"{stem}_ski.npz"

        if pose_file.exists() and not overwrite:
            n = int(np.load(pose_file).shape[0])
            print(f"[{idx + 1}/{len(labels_df)}] {stem}: [SKIP] pose exists ({n} frames)")
            records.append({
                "filename": row["filename"],
                "pose_feature_path": str(pose_file),
                "ski_feature_path": str(ski_file),
                "num_frames": n,
                "target": target,
            })
            continue

        print(f"[{idx + 1}/{len(labels_df)}] Processing: {stem} (Target Score: {target})")
        entry_idx, entry_bbox = find_entry_frame(video_file, model)
        if entry_bbox is None:
            print(f"  [SKIP] {stem}: no skier detected in entry scan")
            continue
        print(f"  -> skier entry at frame {entry_idx}")

        pose, boxes, meta = process_video_pose_temporal(
            model, video_file, entry_idx, entry_bbox,
            max_gap=max_gap, conf_thres=pose_conf, bbox_pad=bbox_pad)
        np.save(pose_file, pose)

        pose21 = np.zeros((len(pose), 21, 3), dtype=np.float32)
        pose21[:, :17] = pose
        np.save(pose21_file, pose21)
        _write_empty_ski(ski_file, len(pose), boxes, meta["fps"],
                         meta["width"], meta["height"])

        print(f"  -> track {meta['track_frames']} frames, "
              f"pose in {meta['tracked_with_pose']}/{len(pose)} frames")
        records.append({
            "filename": row["filename"],
            "pose_feature_path": str(pose_file),
            "ski_feature_path": str(ski_file),
            "num_frames": int(len(pose)),
            "target": target,
        })

    summary = pd.DataFrame(records)
    summary.to_csv(output_path / "dataset_processed_features.csv", index=False)
    print(f"\nTemporal pipeline finished. Features saved to: "
          f"{output_path / 'dataset_processed_features.csv'}")
    return summary
