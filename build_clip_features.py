"""Phase 1: build clip-level features from the temporal body-pose tracks.

Reads the per-clip cached pose produced by CV_project_rebuilt.ipynb:

    roi_pose_dataset/<stem>_pose.npy   (T, 17, 3)  COCO body keypoints

and writes one row per clip with interpretable body / spin / posture / ROI /
temporal-window features.  With ``--body-only`` (the active B1 pipeline) skis
are neither read nor used: the body box is reconstructed from the keypoints and
leading/trailing tracker padding is trimmed before any statistic is computed.

Metadata and QA columns from ``clip_manifest.csv`` are merged in for filtering
and ablation.

Usage
-----
    python build_clip_features.py --body-only --out pose_features.csv
    python build_clip_features.py --out clip_features_with_ski.csv   # legacy
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)

BODY_CONF = 0.30
SKI_CONF = 0.15


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def add_stats(out, values, name, funcs=("mean", "std", "min", "max", "median")):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    out[f"{name}_n"] = int(x.size)
    if x.size == 0:
        for func in funcs:
            out[f"{name}_{func}"] = np.nan
        return out
    for func in funcs:
        out[f"{name}_{func}"] = float(getattr(np, func)(x))
    return out


def last_n(values, n=8):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return x[-n:] if x.size else x


def circ_diff(a, b, period=180.0):
    d = np.abs(a - b) % period
    return np.minimum(d, period - d)


def joint_angle(a, b, c):
    """Angle at b (degrees) for arrays (..., 2)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    valid = (n1 > 1e-6) & (n2 > 1e-6)
    cos = np.sum(v1 * v2, axis=-1) / np.maximum(n1 * n2, 1e-6)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    ang = np.where(valid, ang, np.nan)
    return ang


def safe_div(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.full(a.shape, np.nan, dtype=float)
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-6)
    out[good] = a[good] / b[good]
    return out


def body_box_from_keypoints(keypoints, pad=0.25, min_conf=0.20):
    """Padded body box derived from one frame's visible keypoints.

    The temporal tracker stores exactly this box in ``_ski.npz['bbox']``, so
    deriving it from the pose makes the body-only pipeline independent of the
    (unreliable) ski detector.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return None
    visible = kp[:, 2] >= min_conf if kp.shape[1] > 2 else np.ones(len(kp), bool)
    if int(visible.sum()) < 4:
        return None
    xs, ys = kp[visible, 0], kp[visible, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    pad_px = float(pad) * max(w, h)
    return (x1 - pad_px, y1 - pad_px, w + 2 * pad_px, h + 2 * pad_px)


# --------------------------------------------------------------------------- #
# per-clip extraction
# --------------------------------------------------------------------------- #
def extract_features(row, body_only=False):
    stem = str(row["stem"])
    pose_dir = Path(row["pose_dir"])
    pose_path = pose_dir / f"{stem}_pose.npy"
    if not pose_path.is_file():
        pose_path = pose_dir / f"{stem}_pose21.npy"
    ski_path = Path(row["ski_dir"]) / f"{stem}_ski.npz"
    if not pose_path.is_file() or (not body_only and not ski_path.is_file()):
        return None

    pose = np.load(pose_path)
    if body_only:
        # Body box is reconstructed from the pose itself; skis are not used.
        boxes = [body_box_from_keypoints(pose[i]) for i in range(len(pose))]
        bbox = np.asarray([
            box if box is not None else (np.nan, np.nan, np.nan, np.nan)
            for box in boxes
        ], dtype=np.float32)
        ski_conf = np.zeros((len(pose), 2), dtype=np.float32)
        ski_angle_npz = np.full((len(pose), 2), np.nan, dtype=np.float32)
        ski_length = np.zeros((len(pose), 2), dtype=np.float32)
        ski_quads = np.full((len(pose), 2, 8), np.nan, dtype=np.float32)
    else:
        with np.load(ski_path) as cached:
            bbox = np.asarray(cached["bbox"], dtype=np.float32)
            ski_conf = np.asarray(cached["conf"], dtype=np.float32)
            ski_angle_npz = np.asarray(cached["angle"], dtype=np.float32)
            ski_length = np.asarray(cached["length"], dtype=np.float32)
            ski_quads = np.asarray(cached["quads"], dtype=np.float32)

    n = min(len(pose), len(bbox), len(ski_conf))
    if n < 4:
        return None
    pose, bbox = pose[:n], bbox[:n]
    ski_conf, ski_angle_npz, ski_length = ski_conf[:n], ski_angle_npz[:n], ski_length[:n]
    ski_quads = ski_quads[:n]

    # The temporal tracker leaves leading zero frames before the skier enters.
    # Trim them (and any trailing empties) so every mean/fraction describes the
    # skier's actual tracked span, not the empty padding.
    video_frames = n
    body_present = np.asarray(pose[:, :17, 2], dtype=np.float32).max(axis=1) > 0
    if body_present.any():
        idx = np.flatnonzero(body_present)
        entry_offset, last = int(idx[0]), int(idx[-1])
    else:
        entry_offset, last = 0, n - 1
    pose = pose[entry_offset:last + 1]
    bbox = bbox[entry_offset:last + 1]
    ski_conf = ski_conf[entry_offset:last + 1]
    ski_angle_npz = ski_angle_npz[entry_offset:last + 1]
    ski_length = ski_length[entry_offset:last + 1]
    ski_quads = ski_quads[entry_offset:last + 1]
    n = len(pose)
    if n < 4:
        return None

    xy = np.asarray(pose[:, :, :2], dtype=float)
    conf = np.asarray(pose[:, :, 2], dtype=float)
    body_conf = conf[:, :17]

    width = float(row.get("width", np.nan))
    height = float(row.get("height", np.nan))
    fps = float(row.get("fps", 30.0)) if pd.notna(row.get("fps", np.nan)) else 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    bbox_ok = np.isfinite(bbox).all(axis=1) & (bbox[:, 3] > 1.0)
    scale = np.where(bbox_ok, bbox[:, 3], np.nan)

    out = {
        "filename": row["filename"],
        "stem": stem,
        "domain": row.get("domain", ""),
        "res_class": row.get("res_class", ""),
        "session_group": "s_1026" if str(row["filename"]).startswith("1026") else f"s_{stem}",
        "target": float(row["target"]),
        "fall": int(row["fall"]) if pd.notna(row.get("fall", np.nan)) else np.nan,
        "split": row.get("split", ""),
        "usable": bool(row.get("usable", False)),
        "qa_score": float(row.get("qa_score", np.nan)),
        "n_frames": int(n),
        "video_frames": int(video_frames),
        "entry_offset": int(entry_offset),
        "fps": fps,
        "duration": float(n / fps),
        "width": width,
        "height": height,
    }

    # ---- coverage / quality (also present in the manifest) -----------------
    out.update({
        "body8_frac": float(((body_conf > BODY_CONF).sum(axis=1) >= 8).mean()),
        "pose_det_frac": float((body_conf.max(axis=1) > 0.1).mean()),
        "ankle_frac": float(((conf[:, 15] > 0.2) | (conf[:, 16] > 0.2)).mean()),
        "bbox_frac": float(bbox_ok.mean()),
        "pose_conf_mean": float(body_conf[body_conf.max(axis=1) > 0.1].mean())
        if (body_conf.max(axis=1) > 0.1).any() else 0.0,
    })
    if not body_only:
        out.update({
            "ski_any_frac": float((ski_conf > SKI_CONF).any(axis=1).mean()),
            "ski_both_frac": float((ski_conf > SKI_CONF).all(axis=1).mean()),
            "ski_conf_mean": float(ski_conf[ski_conf > SKI_CONF].mean())
            if (ski_conf > SKI_CONF).any() else 0.0,
            "median_ski_len_px": float(row.get("median_ski_len_px", np.nan)),
        })
    if np.isfinite(height) and height > 0:
        out["bbox_h_frac_mean"] = float(np.nanmean(scale / height))
        out["bbox_h_px_mean"] = float(np.nanmean(scale))
    else:
        out["bbox_h_frac_mean"] = np.nan
        out["bbox_h_px_mean"] = np.nan
    body_h = float(np.nanmedian(scale)) if np.isfinite(scale).any() else np.nan
    out["median_bbox_h_px"] = body_h
    out["median_bbox_h_frac"] = (body_h / height
                                 if np.isfinite(body_h) and np.isfinite(height) and height > 0
                                 else np.nan)
    out["size_score"] = (float(np.clip(body_h / 80.0, 0.0, 1.0))
                         if np.isfinite(body_h) else 0.0)

    # ---- body axis / spin --------------------------------------------------
    shoulder = np.nanmean(xy[:, [5, 6]], axis=1)
    hip = np.nanmean(xy[:, [11, 12]], axis=1)
    axis_ok = ((conf[:, 5] > BODY_CONF) & (conf[:, 6] > BODY_CONF) &
               (conf[:, 11] > BODY_CONF) & (conf[:, 12] > BODY_CONF))
    body_vec = shoulder - hip
    raw_angle = np.degrees(np.arctan2(body_vec[:, 0], -body_vec[:, 1]))
    raw_angle = np.where(axis_ok, raw_angle, np.nan)

    # unwrap over valid frames (interpolate tiny gaps so the spin is continuous)
    unwrapped = np.full(n, np.nan)
    valid = np.isfinite(raw_angle)
    if valid.sum() >= 2:
        idx = np.flatnonzero(valid)
        filled = np.interp(np.arange(n), idx, raw_angle[valid])
        unwrapped[:] = np.unwrap(np.radians(filled))
        unwrapped[~valid] = np.nan
    delta = np.diff(unwrapped)
    delta_valid = np.isfinite(delta)
    spin_rate = np.where(delta_valid, np.degrees(delta), np.nan)
    spin_rate_full = np.concatenate([[np.nan], spin_rate])

    if np.isfinite(unwrapped).sum() >= 2:
        first, last = unwrapped[np.isfinite(unwrapped)][0], unwrapped[np.isfinite(unwrapped)][-1]
        out["spin_net_deg"] = float(np.degrees(last - first))
        out["spin_abs_deg"] = float(np.nansum(np.abs(np.degrees(delta))))
        signs = np.sign(delta[delta_valid])
        out["spin_reversals"] = float(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0.0
    else:
        out["spin_net_deg"] = np.nan
        out["spin_abs_deg"] = np.nan
        out["spin_reversals"] = np.nan
    out["spin_rotations_net"] = out["spin_net_deg"] / 360.0
    out["spin_rotations_abs"] = out["spin_abs_deg"] / 360.0

    add_stats(out, raw_angle, "body_angle", ("mean", "std", "min", "max"))
    add_stats(out, np.abs(raw_angle), "abs_body_angle", ("mean", "std", "max"))
    add_stats(out, np.cos(np.radians(raw_angle)), "uprightness", ("mean", "std", "min"))
    add_stats(out, spin_rate, "spin_rate", ("mean", "std", "max", "median"))
    out["inversion_frac"] = float(np.nanmean(np.abs(raw_angle) > 90)) if valid.any() else np.nan
    out["horizontal_frac"] = float(np.nanmean(np.abs(raw_angle) > 60)) if valid.any() else np.nan

    # ---- limb angles / tuck ------------------------------------------------
    knee = np.nanmean(np.stack([
        joint_angle(xy[:, 11], xy[:, 13], xy[:, 15]),
        joint_angle(xy[:, 12], xy[:, 14], xy[:, 16]),
    ]), axis=0)
    knee[(conf[:, 13] < BODY_CONF) | (conf[:, 14] < BODY_CONF)] = np.nan
    elbow = np.nanmean(np.stack([
        joint_angle(xy[:, 5], xy[:, 7], xy[:, 9]),
        joint_angle(xy[:, 6], xy[:, 8], xy[:, 10]),
    ]), axis=0)
    elbow[(conf[:, 7] < BODY_CONF) | (conf[:, 8] < BODY_CONF)] = np.nan
    add_stats(out, knee, "knee_angle", ("mean", "std", "min", "max"))
    add_stats(out, elbow, "elbow_angle", ("mean", "std", "min", "max"))

    ankle_y = np.nanmean(xy[:, [15, 16], 1], axis=1)
    shoulder_y = np.nanmean(xy[:, [5, 6], 1], axis=1)
    out["body_vertical_extent_rel"] = float(np.nanmean(safe_div(ankle_y - shoulder_y, scale))
                                            ) if np.isfinite(scale).any() else np.nan
    out["hip_to_ankle_rel"] = float(np.nanmean(safe_div(np.nanmean(xy[:, [11, 12], 1], axis=1) - ankle_y, scale))
                                    ) if np.isfinite(scale).any() else np.nan

    # ---- wrists / tuck (body-only) ----------------------------------------
    wrist_ankle = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(scale[i]):
            continue
        wrists = [j for j in (9, 10) if conf[i, j] > BODY_CONF]
        ankles = [j for j in (15, 16) if conf[i, j] > 0.20]
        if wrists and ankles:
            wrist_ankle[i] = min(np.linalg.norm(xy[i, w] - xy[i, a])
                                 for w in wrists for a in ankles) / scale[i]
    add_stats(out, wrist_ankle, "wrist_ankle_rel", ("mean", "std", "min", "max"))
    out["tuck_frac"] = float(np.nanmean(wrist_ankle < 0.30)) if np.isfinite(wrist_ankle).any() else np.nan

    # ---- body-only limb / posture features ---------------------------------
    shoulder = np.nanmean(xy[:, [5, 6]], axis=1)
    hip = np.nanmean(xy[:, [11, 12]], axis=1)
    wrist = np.nanmean(xy[:, [9, 10]], axis=1)
    knee_mid = np.nanmean(xy[:, [13, 14]], axis=1)
    head = np.nanmean(xy[:, [3, 4]], axis=1)

    hand_shoulder = safe_div(np.linalg.norm(wrist - shoulder, axis=1), scale)
    hand_hip = safe_div(np.linalg.norm(wrist - hip, axis=1), scale)
    hand_knee = safe_div(np.linalg.norm(wrist - knee_mid, axis=1), scale)
    hand_head = safe_div(np.linalg.norm(wrist - head, axis=1), scale)
    ankle_sep = safe_div(np.linalg.norm(xy[:, 15] - xy[:, 16], axis=1), scale)
    shoulder_width = safe_div(np.linalg.norm(xy[:, 5] - xy[:, 6], axis=1), scale)
    hip_width = safe_div(np.linalg.norm(xy[:, 11] - xy[:, 12], axis=1), scale)
    knee_sep = safe_div(np.linalg.norm(xy[:, 13] - xy[:, 14], axis=1), scale)
    torso_len = safe_div(np.linalg.norm(shoulder - hip, axis=1), scale)

    add_stats(out, hand_shoulder, "hand_shoulder_rel", ("mean", "std", "min", "max"))
    add_stats(out, hand_hip, "hand_hip_rel", ("mean", "std", "min", "max"))
    add_stats(out, hand_knee, "hand_knee_rel", ("mean", "std", "min", "max"))
    add_stats(out, hand_head, "hand_head_rel", ("mean", "std", "min", "max"))
    add_stats(out, ankle_sep, "ankle_sep_rel", ("mean", "std", "min", "max"))
    add_stats(out, shoulder_width, "shoulder_width_rel", ("mean", "std"))
    add_stats(out, hip_width, "hip_width_rel", ("mean", "std"))
    add_stats(out, knee_sep, "knee_sep_rel", ("mean", "std", "min", "max"))
    add_stats(out, torso_len, "torso_len_rel", ("mean", "std", "min", "max"))

    out["arm_down_frac"] = float(np.nanmean(hand_shoulder < 0.20)) \
        if np.isfinite(hand_shoulder).any() else np.nan
    out["arms_up_frac"] = float(np.nanmean(wrist[:, 1] < shoulder[:, 1])) \
        if np.isfinite(wrist).any() and np.isfinite(shoulder).any() else np.nan
    out["grab_proxy_frac"] = float(np.nanmean(hand_knee < 0.12)) \
        if np.isfinite(hand_knee).any() else np.nan

    for name, idxs in (("head", (0, 1, 2, 3, 4)), ("arms", (5, 6, 7, 8, 9, 10)),
                       ("legs", (11, 12, 13, 14, 15, 16)),
                       ("hands", (9, 10)), ("feet", (15, 16))):
        out[f"vis_{name}_frac"] = float((conf[:, idxs] > BODY_CONF).any(axis=1).mean())

    wrist_ski = np.full(n, np.nan)
    cross = np.full(n, np.nan)
    if not body_only:
        # ---- ski tip/tail points -------------------------------------------
        # Taken from ski_dataset (quads), not pose21: the ski keypoints in cached
        # pose21 files can be stale or empty.
        quad_pts = np.asarray(ski_quads, dtype=float).reshape(n, 2, 4, 2)
        ski_finite = np.isfinite(quad_pts).all(axis=(2, 3))
        ski_ok = (ski_conf > SKI_CONF) & ski_finite
        ski_tip = (quad_pts[:, :, 0] + quad_pts[:, :, 1]) / 2.0      # (n, 2, 2)
        ski_tail = (quad_pts[:, :, 2] + quad_pts[:, :, 3]) / 2.0
        ski_center = (ski_tip + ski_tail) / 2.0

        for i in range(n):
            if not np.isfinite(scale[i]):
                continue
            wrists = [j for j in (9, 10) if conf[i, j] > BODY_CONF]
            ski_points = []
            for slot in range(2):
                if ski_ok[i, slot]:
                    ski_points.extend([ski_tip[i, slot], ski_tail[i, slot]])
            if wrists and ski_points:
                wrist_ski[i] = min(np.linalg.norm(xy[i, w] - p)
                                   for w in wrists for p in ski_points) / scale[i]
        add_stats(out, wrist_ski, "wrist_ski_rel", ("mean", "std", "min", "max"))
        out["grab_frac"] = float(np.nanmean(wrist_ski < 0.12)) \
            if np.isfinite(wrist_ski).any() else np.nan

        # ---- ski geometry --------------------------------------------------
        left_vec = ski_tip[:, 0] - ski_tail[:, 0]
        right_vec = ski_tip[:, 1] - ski_tail[:, 1]
        angle_l = np.degrees(np.arctan2(left_vec[:, 1], left_vec[:, 0])) % 180.0
        angle_r = np.degrees(np.arctan2(right_vec[:, 1], right_vec[:, 0])) % 180.0
        ski_ok_l = ski_ok[:, 0]
        ski_ok_r = ski_ok[:, 1]
        angle_l = np.where(ski_ok_l, angle_l, np.nan)
        angle_r = np.where(ski_ok_r, angle_r, np.nan)
        both = ski_ok_l & ski_ok_r

        cross = np.where(both, circ_diff(angle_l, angle_r), np.nan)
        rel_l = np.where(ski_ok_l & axis_ok, circ_diff(angle_l, raw_angle), np.nan)
        rel_r = np.where(ski_ok_r & axis_ok, circ_diff(angle_r, raw_angle), np.nan)

        add_stats(out, angle_l, "ski_angle_left", ("mean", "std", "min", "max"))
        add_stats(out, angle_r, "ski_angle_right", ("mean", "std", "min", "max"))
        add_stats(out, cross, "ski_cross_angle", ("mean", "std", "min", "max", "median"))
        add_stats(out, rel_l, "ski_body_angle_left", ("mean", "std"))
        add_stats(out, rel_r, "ski_body_angle_right", ("mean", "std"))
        out["ski_cross_frac_gt30"] = float(np.nanmean(cross > 30)) \
            if np.isfinite(cross).any() else np.nan
        out["ski_asymmetry_deg"] = float(np.nanmean(np.abs(angle_l - angle_r))) \
            if both.any() else np.nan

        center_l = ski_center[:, 0]
        center_r = ski_center[:, 1]
        sep = np.where(both, np.linalg.norm(center_l - center_r, axis=1), np.nan)
        add_stats(out, safe_div(sep, scale), "ski_separation_rel", ("mean", "std", "min", "max"))
        add_stats(out, safe_div(ski_length, scale[:, None]), "ski_length_rel", ("mean", "std", "min", "max"))
        add_stats(out, ski_length, "ski_length_px", ("mean", "std", "max"))

        # ski angular velocity and tip speed
        both_l = ski_ok_l[1:] & ski_ok_l[:-1]
        both_r = ski_ok_r[1:] & ski_ok_r[:-1]
        d_angle_l = np.where(both_l, circ_diff(angle_l[1:], angle_l[:-1]), np.nan)
        d_angle_r = np.where(both_r, circ_diff(angle_r[1:], angle_r[:-1]), np.nan)
        ski_ang_vel = np.nanmean(np.stack([d_angle_l, d_angle_r]), axis=0)
        add_stats(out, ski_ang_vel, "ski_ang_vel_deg", ("mean", "std", "max"))
        tip_l = ski_tip[:, 0]
        tip_r = ski_tip[:, 1]
        tip_speed = np.nanmean(np.stack([
            safe_div(np.linalg.norm(np.diff(tip_l, axis=0), axis=1), scale[1:]),
            safe_div(np.linalg.norm(np.diff(tip_r, axis=0), axis=1), scale[1:]),
        ]), axis=0)
        add_stats(out, tip_speed, "ski_tip_speed_rel", ("mean", "std", "max"))

    # ---- ROI / tracking ----------------------------------------------------
    if bbox_ok.any():
        cx = bbox[:, 0] + bbox[:, 2] / 2.0
        cy = bbox[:, 1] + bbox[:, 3] / 2.0
        center = np.stack([cx, cy], axis=1)
        step = np.linalg.norm(np.diff(center, axis=0), axis=1)
        step = safe_div(step, scale[1:])
        roi_speed = np.concatenate([[np.nan], step])
        mean_center = np.nanmean(center[bbox_ok], axis=0)
        range_ = np.linalg.norm(center[bbox_ok] - mean_center, axis=1)
        out["roi_speed_rel_mean"] = float(np.nanmean(roi_speed))
        out["roi_speed_rel_max"] = float(np.nanmax(roi_speed))
        out["roi_range_rel"] = float(np.nanmax(range_) / np.nanmean(scale[bbox_ok]))
        out["roi_x_norm_mean"] = float(np.nanmean(cx[bbox_ok] / width)) if width else np.nan
        out["roi_y_norm_mean"] = float(np.nanmean(cy[bbox_ok] / height)) if height else np.nan
        out["roi_x_norm_std"] = float(np.nanstd(cx[bbox_ok] / width)) if width else np.nan
        out["roi_y_norm_std"] = float(np.nanstd(cy[bbox_ok] / height)) if height else np.nan
        aspect = safe_div(bbox[:, 2], bbox[:, 3])
        out["roi_aspect_mean"] = float(np.nanmean(aspect))
        out["roi_aspect_std"] = float(np.nanstd(aspect))
    else:
        for key in ("roi_speed_rel_mean", "roi_speed_rel_max", "roi_range_rel",
                    "roi_x_norm_mean", "roi_y_norm_mean", "roi_x_norm_std",
                    "roi_y_norm_std", "roi_aspect_mean", "roi_aspect_std"):
            out[key] = np.nan
        roi_speed = np.full(n, np.nan)

    # ---- temporal windows --------------------------------------------------
    t = np.linspace(0.0, 1.0, n)
    early, late = t <= 0.33, t >= 0.67
    signals = {
        "body_angle": raw_angle,
        "abs_body_angle": np.abs(raw_angle),
        "spin_rate": spin_rate_full,
        "wrist_ankle": wrist_ankle,
        "hand_shoulder": hand_shoulder,
        "hand_knee": hand_knee,
        "ankle_sep": ankle_sep,
        "knee_angle": knee,
        "elbow_angle": elbow,
        "roi_speed": roi_speed,
        "bbox_h": np.divide(bbox[:, 3], height, out=np.full(n, np.nan), where=height > 0),
    }
    if not body_only:
        signals["wrist_ski"] = wrist_ski
        signals["ski_cross"] = cross
    for name, values in signals.items():
        add_stats(out, values[early], f"early_{name}", ("mean", "std"))
        add_stats(out, values[late], f"late_{name}", ("mean", "std"))
        last = last_n(values, 8)
        out[f"last8_{name}_mean"] = float(np.mean(last)) if last.size else np.nan

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="clip_manifest.csv")
    parser.add_argument("--pose-dir", default="roi_pose_dataset")
    parser.add_argument("--ski-dir", default="ski_dataset")
    parser.add_argument("--out", default="clip_features.csv")
    parser.add_argument("--body-only", action="store_true",
                        help="use only 17 body keypoints; skis are not read or used")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    manifest["pose_dir"] = args.pose_dir
    manifest["ski_dir"] = args.ski_dir

    rows, skipped = [], []
    for i, row in manifest.iterrows():
        features = extract_features(row, body_only=args.body_only)
        if features is None:
            skipped.append(str(row["filename"]))
        else:
            rows.append(features)
        if (i + 1) % 25 == 0:
            print(f"  extracted {i + 1}/{len(manifest)}")

    df = pd.DataFrame(rows)
    if "session_group" in df.columns:
        df["session_group"] = df["session_group"].astype(str)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(df)} rows x {len(df.columns)} columns")
    if skipped:
        print(f"skipped {len(skipped)} clips: {skipped[:10]}")
    print(f"usable clips in feature table: {int(df['usable'].sum())}/{len(df)}")


if __name__ == "__main__":
    main()
