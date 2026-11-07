"""Event-level features: take-off / apex / landing segmentation + rotation episode.

Why
---
``build_clip_features.py`` summarises whole clips and fixed windows (``early_*``,
``late_*``, ``last8_*``).  A big-air jump is a ~1 s event inside a ~3 s clip, so
those means blur take-off, trick and landing together.  This module detects
*when* the jump happens and re-expresses the same per-frame signals relative to
the detected events.

Detection
---------
Two segmenters, both from data already on disk (no video decode):

1. **Rotation episode (default anchor).**  ``|spin rate|`` (smoothed, spikes
   robustly clipped) is thresholded at a fraction of its peak; the longest run
   is the trick episode, i.e. the airborne phase.  Body rotation measured in the
   image is invariant to camera translation and zoom, which is what makes this
   usable here.  Detected on ~94% of clips.

2. **Ballistic arc (fallback, diagnosed).**  A parabola is fitted to the
   Gaussian-smoothed bbox-centre track after a cubic detrend, and take-off /
   landing are read where the residual returns to baseline.  This was tried
   first and **does not locate the jump**: the operator pans to follow the
   skier, so the image-space vertical track is dominated by camera motion
   (measured: median airtime 0.48 s against a ~1 s expected flight, and zero
   overlap with the rotation episode).  It is kept only as a comparison and is
   flagged separately via ``ev_arc_ok``.

Boundaries are coarse: +/- 3 frames (~0.1 s).  Clips where neither segmenter
fires get ``ev_ok = 0`` and NaN phase features (not a fake zero).

Outputs
-------
    event_features.csv    one row per clip, event timing + phase-sliced features
    event_windows.csv     t_takeoff / t_apex / t_landing in seconds (inspection)

Usage
-----
    python build_event_features.py --body-only --out pose_event_features.csv
    python build_event_features.py --plot 4        # overlay check on N clips
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from build_clip_features import (BODY_CONF, SKI_CONF, body_box_from_keypoints,
                                 circ_diff, joint_angle, safe_div)

warnings.filterwarnings("ignore", category=RuntimeWarning)

SMOOTH_SIGMA = 3.0          # frames (~0.1 s at 30 fps) for the position track
ARC_R2_MIN = 0.35           # parabola must explain this much of the window
ARC_CONTRAST_MIN = 0.10     # ... and this much MORE than a straight line does
ARC_Z_MIN = 2.5             # curvature / its standard error
ARC_MIN_HALF = 4            # minimum half-width of the arc window (frames)
ARC_MAX_SECONDS = 0.9       # cap on one-sided growth (a jump is ~1 s total)
ARC_MIN_AMP_REL = 0.25      # fitted excursion / median bbox height
ROT_FRAC = 0.30             # rotation threshold as a fraction of peak rate
LAND_WIN = 5                # frames either side for the landing/impact windows
EPS = 1e-9


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _smooth(x, sigma=SMOOTH_SIGMA):
    """Gaussian smooth that ignores NaN (nan-aware convolution)."""
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if not good.any():
        return np.full_like(x, np.nan)
    r = max(1, int(round(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    # np.convolve(..., mode="same") returns max(len(x), len(k)); slice explicitly
    # so short clips keep their original length.
    n = len(x)
    start = (len(k) - 1) // 2
    num = np.convolve(np.where(good, x, 0.0), k, mode="full")[start:start + n]
    den = np.convolve(good.astype(float), k, mode="full")[start:start + n]
    out = np.where(den > 1e-6, num / np.maximum(den, EPS), np.nan)
    out[den < 0.25] = np.nan          # keep edges unsupported by data as NaN
    return out


def _nan_stat(x, fn):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else np.nan


def _moments(values, name, out, funcs=("mean", "std", "max", "min")):
    for f in funcs:
        out[f"{name}_{f}"] = _nan_stat(values, getattr(np, f))


# --------------------------------------------------------------------------- #
# per-frame signals (same definitions as build_clip_features.py)
# --------------------------------------------------------------------------- #
def clip_signals(row, pose_dir="roi_pose_dataset", ski_dir="ski_dataset",
                 body_only=False):
    stem = str(row["stem"])
    pose_path = Path(pose_dir) / f"{stem}_pose.npy"
    if not pose_path.is_file():
        pose_path = Path(pose_dir) / f"{stem}_pose21.npy"
    ski_path = Path(ski_dir) / f"{stem}_ski.npz"
    if not pose_path.is_file() or (not body_only and not ski_path.is_file()):
        return None

    pose = np.load(pose_path)
    if body_only:
        boxes = [body_box_from_keypoints(pose[i]) for i in range(len(pose))]
        bbox = np.asarray([
            box if box is not None else (np.nan, np.nan, np.nan, np.nan)
            for box in boxes
        ], dtype=np.float32)
        ski_conf = np.zeros((len(pose), 2), dtype=np.float32)
        ski_quads = np.full((len(pose), 2, 8), np.nan, dtype=np.float32)
    else:
        with np.load(ski_path) as cached:
            bbox = np.asarray(cached["bbox"], dtype=np.float32)
            ski_conf = np.asarray(cached["conf"], dtype=np.float32)
            ski_quads = np.asarray(cached["quads"], dtype=np.float32)

    n = min(len(pose), len(bbox), len(ski_conf))
    if n < 12:
        return None
    pose, bbox, ski_conf, ski_quads = pose[:n], bbox[:n], ski_conf[:n], ski_quads[:n]

    # Remove leading/trailing empty padding written by the temporal tracker.
    body_present = np.asarray(pose[:, :17, 2], dtype=np.float32).max(axis=1) > 0
    entry_offset = 0
    if body_present.any():
        idx = np.flatnonzero(body_present)
        entry_offset, last = int(idx[0]), int(idx[-1])
        pose = pose[entry_offset:last + 1]
        bbox = bbox[entry_offset:last + 1]
        ski_conf = ski_conf[entry_offset:last + 1]
        ski_quads = ski_quads[entry_offset:last + 1]
        n = len(pose)
        if n < 12:
            return None

    xy = np.asarray(pose[:, :, :2], dtype=float)
    conf = np.asarray(pose[:, :, 2], dtype=float)

    fps = float(row["fps"]) if pd.notna(row.get("fps", np.nan)) else 30.0
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    frame_h = float(row["height"]) if pd.notna(row.get("height", np.nan)) else np.nan

    bbox_ok = np.isfinite(bbox).all(axis=1) & (bbox[:, 3] > 1.0)
    h_px = np.where(bbox_ok, bbox[:, 3], np.nan)
    cy_px = bbox[:, 1] + bbox[:, 3] / 2.0
    cy_px = np.where(bbox_ok, cy_px, np.nan)
    med_h = _nan_stat(h_px, np.median)

    # --- body axis + spin ---------------------------------------------------
    shoulder = np.nanmean(xy[:, [5, 6]], axis=1)
    hip = np.nanmean(xy[:, [11, 12]], axis=1)
    axis_ok = ((conf[:, 5] > BODY_CONF) & (conf[:, 6] > BODY_CONF) &
               (conf[:, 11] > BODY_CONF) & (conf[:, 12] > BODY_CONF))
    body_vec = shoulder - hip
    raw_angle = np.degrees(np.arctan2(body_vec[:, 0], -body_vec[:, 1]))
    raw_angle = np.where(axis_ok, raw_angle, np.nan)

    unwrapped = np.full(n, np.nan)
    valid_a = np.isfinite(raw_angle)
    if valid_a.sum() >= 2:
        idx = np.flatnonzero(valid_a)
        filled = np.interp(np.arange(n), idx, raw_angle[valid_a])
        unwrapped[:] = np.unwrap(np.radians(filled))
        unwrapped[~valid_a] = np.nan
    d = np.diff(unwrapped)
    spin = np.concatenate([[np.nan], np.where(np.isfinite(d), np.degrees(d) * fps, np.nan)])

    # --- limb angles / grabs (as in build_clip_features) --------------------
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

    quad_pts = np.asarray(ski_quads, dtype=float).reshape(n, 2, 4, 2)
    ski_finite = np.isfinite(quad_pts).all(axis=(2, 3))
    ski_ok = (ski_conf > SKI_CONF) & ski_finite
    ski_tip = (quad_pts[:, :, 0] + quad_pts[:, :, 1]) / 2.0
    ski_tail = (quad_pts[:, :, 2] + quad_pts[:, :, 3]) / 2.0

    wrist_ankle = np.full(n, np.nan)
    wrist_ski = np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(h_px[i]):
            continue
        wrists = [j for j in (9, 10) if conf[i, j] > BODY_CONF]
        ankles = [j for j in (15, 16) if conf[i, j] > 0.20]
        ski_points = []
        for slot in range(2):
            if ski_ok[i, slot]:
                ski_points.extend([ski_tip[i, slot], ski_tail[i, slot]])
        if wrists and ankles:
            wrist_ankle[i] = min(np.linalg.norm(xy[i, w] - xy[i, a])
                                 for w in wrists for a in ankles) / h_px[i]
        if wrists and ski_points:
            wrist_ski[i] = min(np.linalg.norm(xy[i, w] - p)
                               for w in wrists for p in ski_points) / h_px[i]

    left_vec = ski_tip[:, 0] - ski_tail[:, 0]
    right_vec = ski_tip[:, 1] - ski_tail[:, 1]
    angle_l = np.where(ski_ok[:, 0],
                       np.degrees(np.arctan2(left_vec[:, 1], left_vec[:, 0])) % 180.0, np.nan)
    angle_r = np.where(ski_ok[:, 1],
                       np.degrees(np.arctan2(right_vec[:, 1], right_vec[:, 0])) % 180.0, np.nan)
    both = ski_ok[:, 0] & ski_ok[:, 1]
    cross = np.where(both, circ_diff(angle_l, angle_r), np.nan)

    # --- motion -------------------------------------------------------------
    speed = np.full(n, np.nan)
    if bbox_ok.sum() >= 2:
        step = np.linalg.norm(np.diff(np.stack([bbox[:, 0] + bbox[:, 2] / 2,
                                                cy_px], axis=1), axis=0), axis=1)
        speed[1:] = safe_div(step, h_px[1:]) * fps
    dscale = np.full(n, np.nan)
    if bbox_ok.sum() >= 2:
        dscale[1:] = safe_div(np.diff(h_px), h_px[1:]) * fps
    vy = np.full(n, np.nan)
    if bbox_ok.sum() >= 2:
        vy[1:] = safe_div(np.diff(cy_px), h_px[1:]) * fps

    return {
        "n": n, "fps": fps, "frame_h": frame_h, "med_h": med_h,
        "entry_offset": entry_offset,
        "track_ok": bbox_ok, "cy_px": cy_px, "h_px": h_px,
        "vy": vy, "speed": speed, "dscale": dscale,
        "spin": spin, "abs_spin": np.abs(spin),
        "abs_body": np.abs(raw_angle),
        "knee": knee, "elbow": elbow,
        "wrist_ski": wrist_ski, "wrist_ankle": wrist_ankle,
        "ski_cross": cross,
        "bbox_h_rel": safe_div(h_px, frame_h) if np.isfinite(frame_h) and frame_h > 0
        else np.full(n, np.nan),
    }


# --------------------------------------------------------------------------- #
# event detection
# --------------------------------------------------------------------------- #
def _fit_win(t, y, ok, apex, L, R):
    """Parabolic LS fit on [apex-L, apex+R] with a curvature-contrast test.

    ``r2`` alone cannot separate a ballistic arc from a smooth camera pan: a
    quadratic fits any 3 s drift well.  The discriminator is how much better the
    parabola is than a straight line on the *same* window, plus the curvature
    over its own standard error.
    """
    lo, hi = max(0, apex - L), min(len(y), apex + R + 1)
    idx = np.arange(lo, hi)
    idx = idx[ok[idx] & np.isfinite(y[idx])]
    if idx.size < 6:
        return None
    tt, yy = t[idx], y[idx]
    if np.ptp(tt) < 3.0 / 30.0:
        return None
    X = np.column_stack([tt ** 2, tt, np.ones_like(tt)])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    r2 = 1.0 - float(resid @ resid) / ss_tot
    lin = np.polyfit(tt, yy, 1)
    r2_lin = 1.0 - float(((yy - np.polyval(lin, tt)) ** 2).sum()) / ss_tot
    dof = max(1, len(yy) - 3)
    s2 = float(resid @ resid) / dof
    try:
        cov = s2 * np.linalg.inv(X.T @ X)
        curv_z = float(beta[0] / np.sqrt(max(cov[0, 0], EPS)))
    except np.linalg.LinAlgError:
        curv_z = 0.0
    return {"coef": beta, "r2": r2, "r2_lin": r2_lin, "curv_z": abs(curv_z)}


def detect_arc(cy_px, track_ok, fps):
    """Ballistic window on the smoothed vertical track.

    Returns dict with apex/take-off/landing frames, fit r2, curvature, or None.
    """
    n = len(cy_px)
    y = _smooth(cy_px, SMOOTH_SIGMA)
    ok = track_ok & np.isfinite(y)
    if ok.sum() < 14:
        return None
    t = np.arange(n) / fps
    max_half = max(ARC_MIN_HALF + 2, int(round(ARC_MAX_SECONDS * fps)))

    idx = np.flatnonzero(ok)
    if len(idx) < 14:
        return None

    # The camera pans, tilts and zooms while following the skier, and all three
    # are smooth over the clip.  A cubic detrend removes them; what is left is
    # the short (~1 s) ballistic bump of the jump itself.  Take-off and landing
    # are then the points where that residual returns to its baseline, which
    # bounds the window by the data instead of by an r2 threshold (a quadratic
    # fits any smooth pan well, so r2 alone cannot find the event).
    cu = np.polyfit(t[idx], y[idx], 3)
    yd = np.where(ok, y - np.polyval(cu, t), np.nan)

    best = None
    for mode in (-1, 1):                      # -1: apex is the track minimum
        s = _smooth(yd * mode, 2.0)           # > 0 above the baseline at the apex
        if not np.isfinite(s).any():
            continue
        apex = int(np.nanargmax(s))
        peak = float(s[apex])
        if not np.isfinite(peak) or peak <= 0:
            continue

        # Stop only after 3 consecutive frames below the baseline band, so a
        # single noisy dip does not truncate the flight to a few frames.
        thr = 0.15 * peak
        L, miss = 0, 0
        for k in range(1, max_half + 1):
            i = apex - k
            if i < 0:
                break
            if not np.isfinite(s[i]) or s[i] <= thr:
                miss += 1
                if miss >= 3:
                    break
                continue
            miss = 0
            L = k
        R, miss = 0, 0
        for k in range(1, max_half + 1):
            i = apex + k
            if i >= n:
                break
            if not np.isfinite(s[i]) or s[i] <= thr:
                miss += 1
                if miss >= 3:
                    break
                continue
            miss = 0
            R = k
        if L + R < 4:
            continue

        fin = _fit_win(t, yd, ok, apex, L, R)
        if fin is None or np.sign(fin["coef"][0]) != -mode:
            continue
        coef = fin["coef"]
        apex_v = float(-coef[1] / (2 * coef[0])) if abs(coef[0]) > EPS else t[apex]
        apex_frame = int(np.clip(round(apex_v * fps), apex - 2, apex + 2))
        cand = {
            "mode": mode, "apex": apex_frame, "to": apex - L, "ld": apex + R,
            "r2": float(fin["r2"]), "contrast": float(fin["r2"] - fin["r2_lin"]),
            "curv_z": float(fin["curv_z"]), "curv": float(coef[0]),
            "amp_rel": float(abs(peak)),
            "quality": float(peak) * float(fin["r2"]),
        }
        if best is None or cand["quality"] > best["quality"]:
            best = cand

    return best


def detect_rotation(abs_spin, fps):
    """Longest contiguous run of high |spin rate| (the trick), in frames."""
    n = len(abs_spin)
    s = _smooth(abs_spin, 2.0)
    good = np.isfinite(s)
    if not good.any():
        return None
    s = np.where(good, s, 0.0)
    peak = float(np.nanpercentile(s, 98))
    if peak < 90.0:                       # <0.25 rev/s: no real rotation seen
        return None
    mask = s > ROT_FRAC * peak
    if not mask.any():
        return None
    # longest run
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, n - 1))
    a, b = max(runs, key=lambda ab: ab[1] - ab[0])
    if b - a < 3:
        return None
    return {"start": a, "end": b, "peak": peak,
            "net_deg": float(np.nansum(s[a:b + 1]) / fps)}


def _overlap(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return inter / union if union > 0 else 0.0


# --------------------------------------------------------------------------- #
# feature assembly
# --------------------------------------------------------------------------- #
PHASE_SIGNALS = (
    "speed", "dscale", "abs_spin", "abs_body", "wrist_ski", "wrist_ankle",
    "knee", "elbow", "ski_cross", "bbox_h_rel",
)


def extract_events(row, pose_dir="roi_pose_dataset", ski_dir="ski_dataset",
                   anchor="auto", body_only=False):
    sig = clip_signals(row, pose_dir, ski_dir, body_only=body_only)
    if sig is None:
        return None
    n, fps, med_h = sig["n"], sig["fps"], sig["med_h"]

    out = {
        "filename": row["filename"], "stem": str(row["stem"]),
        "target": float(row["target"]),
        "fall": int(row["fall"]) if pd.notna(row.get("fall", np.nan)) else np.nan,
        "n_frames": int(n), "fps": float(fps),
        "entry_offset": int(sig["entry_offset"]),
        "med_bbox_h_px": float(med_h) if np.isfinite(med_h) else np.nan,
        "ev_track_frac": float(sig["track_ok"].mean()),
        "ev_pace_mean": _nan_stat(sig["speed"], np.mean),
    }

    arc = detect_arc(sig["cy_px"], sig["track_ok"], fps)
    rot = detect_rotation(sig["abs_spin"], fps)

    # ---- arcs -------------------------------------------------------------
    ok_arc = False
    if arc is not None and np.isfinite(med_h) and med_h > 0:
        amp_rel = arc["amp_rel"] / med_h
        arc["amp_rel_norm"] = float(amp_rel)
        air = (arc["ld"] - arc["to"]) / fps
        ok_arc = bool(arc["r2"] >= ARC_R2_MIN and amp_rel >= ARC_MIN_AMP_REL
                      and 0.35 <= air <= 2.0)
        out.update({
            "ev_arc_contrast": arc["contrast"],
            "ev_arc_curv_z": arc["curv_z"],
            "ev_arc_airtime_s": float(air),
            "ev_arc_r2": arc["r2"],
            "ev_arc_curv_rel": float(arc["curv"] * med_h),   # px/s^2 per bbox height
            "ev_arc_amp_rel": float(amp_rel),
            "ev_arc_mode": int(arc["mode"]),
        })
        # boundary kinematics, normalised by bbox height
        vy_s = _smooth(sig["vy"], 2.0)
        v_at = lambda i: float(vy_s[i]) if 0 <= i < n and np.isfinite(vy_s[i]) else np.nan
        v_to, v_ld = v_at(arc["to"]), v_at(arc["ld"])
        out["ev_takeoff_vy"] = v_to
        out["ev_landing_vy"] = v_ld
        out["ev_vy_symmetry"] = (abs(v_to) / abs(v_ld)
                                 if np.isfinite(v_to) and np.isfinite(v_ld) and abs(v_ld) > EPS
                                 else np.nan)
    else:
        for k in ("ev_arc_r2", "ev_arc_curv_rel", "ev_arc_amp_rel", "ev_arc_mode",
                  "ev_arc_airtime_s", "ev_takeoff_vy", "ev_landing_vy",
                  "ev_vy_symmetry", "ev_arc_contrast", "ev_arc_curv_z"):
            out[k] = np.nan

    # ---- rotation episode -------------------------------------------------
    if rot is not None:
        out.update({
            "ev_rot_start_rel": rot["start"] / max(1, n - 1),
            "ev_rot_end_rel": rot["end"] / max(1, n - 1),
            "ev_rot_dur_s": (rot["end"] - rot["start"]) / fps,
            "ev_rot_frac": (rot["end"] - rot["start"] + 1) / n,
            "ev_rot_peak_dps": float(rot["peak"]),
            "ev_rot_net_deg": rot["net_deg"],
            "ev_rot_arc_iou": (_overlap(rot["start"], rot["end"], arc["to"], arc["ld"])
                               if arc is not None else np.nan),
        })
    else:
        for k in ("ev_rot_start_rel", "ev_rot_end_rel", "ev_rot_dur_s", "ev_rot_frac",
                  "ev_rot_peak_dps", "ev_rot_net_deg", "ev_rot_arc_iou"):
            out[k] = np.nan

    # ---- choose the air window --------------------------------------------
    # Rotation is the anchor: it is camera-invariant and fires on ~94% of clips.
    # The ballistic arc only helps on static-camera shots, so it is a fallback.
    win, src = None, "none"
    if anchor in ("auto", "rot") and rot is not None and rot["end"] > rot["start"]:
        pad = 2
        win = {"to": max(0, rot["start"] - pad), "ld": min(n - 1, rot["end"] + pad),
               "apex": (rot["start"] + rot["end"]) // 2}
        src = "rot"
    elif anchor in ("auto", "arc") and arc is not None:
        win = {"to": max(0, arc["to"]), "ld": min(n - 1, arc["ld"]),
               "apex": int(arc["apex"])}
        src = "arc"

    if win is not None:
        out["ev_to_rel"] = win["to"] / max(1, n - 1)
        out["ev_ld_rel"] = win["ld"] / max(1, n - 1)
        out["ev_apex_rel"] = win["apex"] / max(1, n - 1)
        out["ev_airtime_s"] = (win["ld"] - win["to"]) / fps
        out["ev_air_frac"] = (win["ld"] - win["to"] + 1) / n
    else:
        for k in ("ev_to_rel", "ev_ld_rel", "ev_apex_rel", "ev_airtime_s",
                  "ev_air_frac"):
            out[k] = np.nan

    # ---- phase slicing ----------------------------------------------------
    if win is not None:
        air_mask = np.zeros(n, bool)
        air_mask[max(0, win["to"]):min(n, win["ld"] + 1)] = True
        ground = sig["track_ok"] & ~air_mask
        for name in PHASE_SIGNALS:
            v = np.asarray(sig[name], dtype=float)
            va, vg = v[air_mask], v[ground]
            _moments(va, f"{name}_air", out, ("mean", "std", "max"))
            out[f"{name}_ground_mean"] = _nan_stat(vg, np.mean)
            out[f"{name}_air_minus_ground"] = (
                out[f"{name}_air_mean"] - out[f"{name}_ground_mean"]
                if np.isfinite(out[f"{name}_air_mean"]) and
                np.isfinite(out[f"{name}_ground_mean"]) else np.nan)
        # ---- landing window ------------------------------------------------
        lo, hi = max(0, win["ld"] - LAND_WIN), min(n, win["ld"] + LAND_WIN + 1)
        vy_abs = np.abs(sig["vy"])
        _moments(vy_abs[lo:hi], "ev_land_dvy", out, ("max", "mean"))
        _moments(sig["dscale"][lo:hi], "ev_land_dscale", out, ("max", "mean"))
        _moments(sig["dscale"][lo:hi], "ev_land_dscale_abs", out, ("max",))
        out["ev_land_dscale_abs_max"] = _nan_stat(np.abs(sig["dscale"][lo:hi]), np.max)
        out["ev_land_spin_std"] = _nan_stat(sig["spin"][lo:hi], np.std)
        out["ev_land_knee_min"] = _nan_stat(sig["knee"][lo:hi], np.min)
        _moments(sig["speed"][lo:hi], "ev_land_speed", out, ("mean",))
        # ---- take-off window -----------------------------------------------
        lo2, hi2 = max(0, win["to"] - LAND_WIN), min(n, win["to"] + 1)
        _moments(sig["speed"][lo2:hi2], "ev_to_speed", out, ("mean", "max"))
        out["ev_to_knee_min"] = _nan_stat(sig["knee"][lo2:hi2], np.min)
        out["ev_to_body_abs"] = _nan_stat(sig["abs_body"][lo2:hi2], np.mean)
    else:
        for name in PHASE_SIGNALS:
            for s in ("mean", "std", "max"):
                out[f"{name}_air_{s}"] = np.nan
            out[f"{name}_ground_mean"] = np.nan
            out[f"{name}_air_minus_ground"] = np.nan
        for k in ("ev_land_dvy_max", "ev_land_dvy_mean", "ev_land_dscale_max",
                  "ev_land_dscale_mean", "ev_land_dscale_abs_max", "ev_land_spin_std",
                  "ev_land_knee_min", "ev_land_speed_mean", "ev_to_speed_mean",
                  "ev_to_speed_max", "ev_to_knee_min", "ev_to_body_abs"):
            out[k] = np.nan

    out["ev_ok"] = int(win is not None)
    out["ev_arc_ok"] = int(ok_arc)
    out["ev_rot_ok"] = int(rot is not None)
    out["ev_anchor"] = {"none": 0, "arc": 1, "rot": 2}[src]
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="clip_manifest.csv")
    ap.add_argument("--pose-dir", default="roi_pose_dataset")
    ap.add_argument("--ski-dir", default="ski_dataset")
    ap.add_argument("--out", default="event_features.csv")
    ap.add_argument("--windows", default="event_windows.csv")
    ap.add_argument("--anchor", choices=("auto", "rot", "arc"), default="auto",
                    help="event anchor: rotation (default), ballistic arc, or either")
    ap.add_argument("--plot", type=int, default=0,
                    help="overlay detected events on this many clips")
    ap.add_argument("--body-only", action="store_true",
                    help="use only body keypoints; skis are not read or used")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows, windows, plotted = [], [], []
    for i, row in manifest.iterrows():
        feats = extract_events(row, args.pose_dir, args.ski_dir, args.anchor,
                               body_only=args.body_only)
        if feats is None:
            continue
        rows.append(feats)
        if args.plot and np.isfinite(feats.get("ev_airtime_s", np.nan)):
            plotted.append(feats)

    df = pd.DataFrame(rows)
    if args.body_only:
        # Keep the table body-only: no ski / grab-from-ski columns.
        ski_cols = [c for c in df.columns if "ski" in c.lower()]
        df = df.drop(columns=ski_cols)
    df.to_csv(args.out, index=False)

    w = df[["filename", "n_frames", "fps", "ev_apex_rel", "ev_to_rel", "ev_ld_rel",
            "ev_airtime_s", "ev_arc_r2", "ev_arc_amp_rel", "ev_ok"]].copy()
    w["t_to_s"] = w["ev_to_rel"] * (w["n_frames"] - 1) / w["fps"]
    w["t_apex_s"] = w["ev_apex_rel"] * (w["n_frames"] - 1) / w["fps"]
    w["t_ld_s"] = w["ev_ld_rel"] * (w["n_frames"] - 1) / w["fps"]
    w.to_csv(args.windows, index=False)

    print(f"wrote {args.out}: {len(df)} rows x {len(df.columns)} columns")
    print(f"wrote {args.windows}")
    print(f"  phase window : {int(df['ev_ok'].sum())}/{len(df)} "
          f"(rotation anchor {int((df['ev_anchor'] == 2).sum())}, "
          f"arc anchor {int((df['ev_anchor'] == 1).sum())})")
    print(f"  arc accepted : {int(df['ev_arc_ok'].sum())}/{len(df)}")
    print(f"  rotation     : {int(df['ev_rot_ok'].sum())}/{len(df)}")
    good = df[df["ev_ok"] == 1]
    if len(good):
        print(f"  airtime s    : median {good['ev_airtime_s'].median():.2f} "
              f"[{good['ev_airtime_s'].quantile(.05):.2f}, "
              f"{good['ev_airtime_s'].quantile(.95):.2f}]")
        print(f"  arc r2       : median {good['ev_arc_r2'].median():.2f}")
        print(f"  amp (bbox h) : median {good['ev_arc_amp_rel'].median():.2f}")
        print(f"  apex rel pos : median {good['ev_apex_rel'].median():.2f}")
        iou = good["ev_rot_arc_iou"].dropna()
        if len(iou):
            print(f"  arc/rotation IoU: median {iou.median():.2f}")

    if args.plot:
        _plot(plotted[:args.plot], manifest, args.pose_dir, args.ski_dir)


def _plot(items, manifest, pose_dir, ski_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for feats in items:
        row = manifest[manifest["filename"] == feats["filename"]].iloc[0]
        sig = clip_signals(row, pose_dir, ski_dir)
        n, fps = sig["n"], sig["fps"]
        t = np.arange(n) / fps
        y = _smooth(sig["cy_px"], SMOOTH_SIGMA)
        fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(t, sig["cy_px"], ".", ms=2, alpha=0.4, label="cy px")
        axes[0].plot(t, y, label="smoothed")
        axes[1].plot(t, sig["abs_spin"], alpha=0.4, label="|spin|")
        axes[1].plot(t, _smooth(sig["abs_spin"], 2.0), label="smoothed")
        axes[2].plot(t, sig["speed"], alpha=0.4)
        axes[2].plot(t, sig["dscale"], alpha=0.4, label="dscale")
        for ax in axes:
            for rel, c in ((feats["ev_to_rel"], "tab:green"),
                           (feats["ev_apex_rel"], "tab:orange"),
                           (feats["ev_ld_rel"], "tab:red")):
                if np.isfinite(rel):
                    ax.axvline(rel * (n - 1) / fps, color=c, lw=1.2)
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("cy")
        axes[1].set_ylabel("spin deg/s")
        axes[2].set_ylabel("speed / dscale")
        axes[2].set_xlabel("time (s)")
        fig.suptitle(f"{feats['filename']}  airtime={feats['ev_airtime_s']:.2f}s "
                     f"r2={feats['ev_arc_r2']:.2f} (green=takeoff orange=apex red=landing)",
                     fontsize=9)
        fig.tight_layout()
        name = f"_ev_{str(feats['filename']).replace('.', '_')}.png"
        fig.savefig(name, dpi=80)
        plt.close(fig)
        print("wrote", name)


if __name__ == "__main__":
    main()
