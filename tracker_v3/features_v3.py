"""features_v3 — clip-level jump features from skier_tracker_v3 output.

For every clip:  <pose_dir>/<stem>_pose.npy, _pose_meta.json, _cam.npz

All angles are corrected for camera roll (from the per-frame camera similarity
transforms), and all distances are in body heights, so the features do not
depend on how the broadcast camera pans or zooms.

Feature groups
  trick window  : the 1.5 s window with the most body rotation (the jump)
                  -> in-plane rotation (flips / corks), shoulder-swap count
                  (spins), inverted fraction, ground-relative vertical range
  landing/fall  : last part of the run after the trick window
                  -> torso tilt, lying-down box shape, hip height, upright frac
  quality       : pose coverage, tracked fraction, number of camera shots
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONF = 0.3
NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 0, 5, 6, 11, 12, 13, 14, 15, 16
TRICK_WIN_S = 1.5


def _mid(k, a, b):
    if k[a, 2] >= CONF and k[b, 2] >= CONF:
        return (k[a, :2] + k[b, :2]) / 2
    if k[a, 2] >= CONF:
        return k[a, :2]
    if k[b, 2] >= CONF:
        return k[b, :2]
    return None


def camera_roll_and_ref(meta, cam_npz, T):
    """Per-frame cumulative camera roll (radians) and a map image->shot reference
    coordinates (for ground-relative positions).  Linear in time between
    detection frames."""
    roll = np.zeros(T); M = [np.eye(3) for _ in range(T)]
    if cam_npz is None:
        return roll, M
    df = cam_npz["det_frames"]; cam = cam_npz["cam"]
    cuts = set(meta.get("cut_frames") or [])
    cur = np.eye(3); cur_roll = 0.0
    prev_f = 0
    vals = {}
    for i, f in enumerate(df):
        f = int(f)
        new_shot = any(prev_f < c <= f for c in cuts) or i == 0
        A = cam[i]
        if new_shot:
            cur = np.eye(3); cur_roll = 0.0
        elif not np.isnan(A).any():
            A3 = np.vstack([A, [0, 0, 1]])
            cur = A3 @ cur                                # ref -> frame f
            cur_roll += float(np.arctan2(A[1, 0], A[0, 0]))
        vals[f] = (cur.copy(), cur_roll)
        prev_f = f
    keys = sorted(vals)
    j = 0
    for t in range(T):
        while j + 1 < len(keys) and keys[j + 1] <= t:
            j += 1
        Mt, r = vals[keys[j]]
        M[t] = Mt; roll[t] = r
    return roll, M


def clip_features(pose, meta, cam_npz):
    T = len(pose); fps = float(meta.get("fps") or 59.94)
    H = float(meta.get("height") or 720)
    cuts = sorted(int(c) for c in (meta.get("cut_frames") or []))
    shot_id = np.searchsorted(np.array(cuts + [10 ** 9]), np.arange(T), side="right")
    roll, M = camera_roll_and_ref(meta, cam_npz, T)

    valid = (pose[:, :, 2] >= CONF).sum(1) >= 6
    theta = np.full(T, np.nan)       # torso angle, 0 = upright, roll-corrected
    swap = np.full(T, np.nan)        # signed shoulder width / torso length
    inv = np.full(T, np.nan)         # 1 if shoulders below hips (world frame)
    bh = np.full(T, np.nan)          # body height in px
    gy = np.full(T, np.nan)          # ground-relative vertical position (ref coords)
    aspect = np.full(T, np.nan)
    hip_low = np.full(T, np.nan)
    for t in range(T):
        k = pose[t]
        if not valid[t]:
            continue
        sh, hp = _mid(k, L_SH, R_SH), _mid(k, L_HIP, R_HIP)
        vis = k[:, 2] >= CONF
        xs, ys = k[vis, 0], k[vis, 1]
        h = max(ys.max() - ys.min(), xs.max() - xs.min(), 4.0)
        bh[t] = h
        aspect[t] = (xs.max() - xs.min()) / max(ys.max() - ys.min(), 1.0)
        c = np.array([xs.mean(), ys.mean(), 1.0])
        try:
            cref = np.linalg.solve(M[t], c)
            s = np.sqrt(abs(np.linalg.det(M[t][:2, :2])))
            gy[t] = cref[1] / max(h / max(s, 1e-3), 1.0)     # in body heights
        except np.linalg.LinAlgError:
            pass
        if sh is not None and hp is not None:
            v = sh - hp
            # camera roll from chained similarity transforms drifts (RANSAC
            # failures), and broadcast cameras are level -> use image angle
            ang = np.arctan2(v[0], -v[1])                     # image y points down
            theta[t] = np.arctan2(np.sin(ang), np.cos(ang))
            tl = max(np.linalg.norm(v), 1.0)
            inv[t] = float(np.cos(theta[t]) < 0)
            if k[L_SH, 2] >= CONF and k[R_SH, 2] >= CONF:
                swap[t] = (k[L_SH, 0] - k[R_SH, 0]) / tl
        an = _mid(k, L_ANK, R_ANK)
        if hp is not None and an is not None:
            hip_low[t] = (an[1] - hp[1]) / h                 # ~0.5 standing, ~0 lying

    # smooth the torso angle (circular, 5 valid frames) and the swap signal
    def smooth(arr, circ):
        out = arr.copy(); idx = np.where(~np.isnan(arr))[0]
        for j, t in enumerate(idx):
            nb = idx[max(0, j - 2):j + 3]
            nb = nb[(np.abs(nb - t) <= 6) & (shot_id[nb] == shot_id[t])]
            if circ:
                out[t] = np.arctan2(np.median(np.sin(arr[nb])), np.median(np.cos(arr[nb])))
            else:
                out[t] = np.median(arr[nb])
        return out
    theta = smooth(theta, True); swap = smooth(swap, False)
    # per-frame rotation energy (only between close valid frames in the same shot)
    dtheta = np.zeros(T); dswap = np.zeros(T)
    last = None
    for t in range(T):
        if np.isnan(theta[t]):
            continue
        if last is not None and t - last <= 4 and shot_id[t] == shot_id[last]:
            d = theta[t] - theta[last]
            dtheta[t] = abs(np.arctan2(np.sin(d), np.cos(d)))
            if not np.isnan(swap[t]) and not np.isnan(swap[last]):
                dswap[t] = abs(swap[t] - swap[last])
        last = t
    # ignore jitter
    dtheta[dtheta < 0.08] = 0
    energy = dtheta / np.pi + dswap
    W = max(3, int(TRICK_WIN_S * fps))
    cs = np.concatenate([[0], np.cumsum(energy)])
    win_e = cs[W:] - cs[:-W] if T > W else np.array([cs[-1]])
    w0 = int(np.argmax(win_e)); w1 = min(T, w0 + W)

    f = {}
    f["n_frames"] = T; f["duration_s"] = T / fps
    f["pose_frac"] = float(valid.mean())
    f["tracked_frac"] = float(meta.get("track_frames", 0)) / max(T, 1)
    f["n_shots"] = len(cuts) + 1
    # ---- trick window
    tw = slice(w0, w1)
    f["trick_energy"] = float(win_e[w0]) if len(win_e) else 0.0
    f["trick_pose_frac"] = float(valid[tw].mean())
    th_w = theta[tw]; ok = ~np.isnan(th_w)
    if ok.sum() > 3:
        un = np.unwrap(th_w[ok])
        f["trick_inplane_turns"] = float((un.max() - un.min()) / (2 * np.pi))
    else:
        f["trick_inplane_turns"] = np.nan
    # rotation episode length: longest run of above-threshold smoothed energy
    e_s = np.convolve(energy, np.ones(9) / 9, mode="same")
    thr = 0.3 * (np.percentile(e_s, 99) + 1e-6)
    run = best_run = 0
    for v in e_s:
        run = run + 1 if v > thr else 0
        best_run = max(best_run, run)
    f["rot_episode_s"] = best_run / fps
    sw = swap[tw]; sw = sw[~np.isnan(sw)]
    n_sw = 0; sgn = 0
    for v in sw:
        if sgn == 0:
            sgn = 1 if v >= 0 else -1
        elif sgn == 1 and v < -0.25:
            n_sw += 1; sgn = -1
        elif sgn == -1 and v > 0.25:
            n_sw += 1; sgn = 1
    f["trick_swap_turns"] = n_sw / 2.0
    f["trick_swap_std"] = float(np.std(sw)) if len(sw) > 2 else np.nan
    iv = inv[tw]; iv = iv[~np.isnan(iv)]
    f["trick_inverted_frac"] = float(iv.mean()) if len(iv) else np.nan
    th = theta[tw]; th = th[~np.isnan(th)]
    f["trick_max_tilt"] = float(np.max(np.abs(th))) if len(th) else np.nan
    f["trick_pos"] = (w0 + w1) / 2 / T
    # whole-clip rotation (robust, no window)
    f["clip_inverted_frac"] = float(np.nanmean(inv)) if np.isfinite(inv).any() else np.nan
    f["clip_rot_energy_p95"] = float(np.percentile(energy[valid], 95)) if valid.any() else np.nan
    # ---- landing / fall: from end of trick window to end of the run
    tail_idx = np.where(valid[w1:])[0] + w1
    if len(tail_idx) < 5:                       # fall back to last 25% of tracked frames
        vi = np.where(valid)[0]
        tail_idx = vi[int(0.75 * len(vi)):]
    f["tail_frames"] = len(tail_idx)
    if len(tail_idx):
        thv = np.abs(theta[tail_idx]); thv = thv[~np.isnan(thv)]
        f["tail_tilt_mean"] = float(np.mean(thv)) if len(thv) else np.nan
        f["tail_tilt_p90"] = float(np.percentile(thv, 90)) if len(thv) else np.nan
        f["tail_lying_frac"] = float(np.mean(thv > np.deg2rad(55))) if len(thv) else np.nan
        a = aspect[tail_idx]; a = a[~np.isnan(a)]
        f["tail_aspect_p90"] = float(np.percentile(a, 90)) if len(a) else np.nan
        hl = hip_low[tail_idx]; hl = hl[~np.isnan(hl)]
        f["tail_hip_low_p10"] = float(np.percentile(hl, 10)) if len(hl) else np.nan
        f["tail_hip_low_mean"] = float(np.mean(hl)) if len(hl) else np.nan
        iv2 = inv[tail_idx]; iv2 = iv2[~np.isnan(iv2)]
        f["tail_inverted_frac"] = float(np.mean(iv2)) if len(iv2) else np.nan
        # immediate landing (first 1 s after the trick)
        land = tail_idx[tail_idx < w1 + int(1.0 * fps)]
        tl = np.abs(theta[land]); tl = tl[~np.isnan(tl)]
        f["land_tilt_max"] = float(np.max(tl)) if len(tl) else np.nan
        f["land_rot_energy"] = float(energy[land].sum()) if len(land) else np.nan
        hb = bh[tail_idx]; hb = hb[~np.isnan(hb)]
        f["tail_height_drop"] = float(np.min(hb) / np.median(bh[valid])) if len(hb) else np.nan
    # missing skier right after the trick often = crash in the snow spray
    after = slice(w1, min(T, w1 + int(2 * fps)))
    f["post_trick_missing"] = float(1 - valid[after].mean()) if w1 < T else np.nan
    return f


def build(pose_dir, scores_csv, out_csv):
    pose_dir = Path(pose_dir)
    sc = pd.read_csv(scores_csv, dtype={"clip_id": str})
    sc = sc[sc["dataset"] == "competition"]
    rows = []
    for _, r in sc.iterrows():
        stem = r["clip_id"]
        p, m, c = (pose_dir / f"{stem}_pose.npy", pose_dir / f"{stem}_pose_meta.json",
                   pose_dir / f"{stem}_cam.npz")
        if not p.exists():
            continue
        meta = json.loads(m.read_text())
        if "failed" in meta:
            continue
        cam = np.load(c) if c.exists() else None
        f = clip_features(np.load(p), meta, cam)
        f.update(clip_id=stem, competition=r["competition"], target=r["official_score_100"],
                 fall=int(r["fall"]), women=int("women" in r["competition"]))
        rows.append(f)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose-dir", default="pose_v3")
    ap.add_argument("--scores", default="scores/all_clips_scores.csv")
    ap.add_argument("--out", default="features_v3.csv")
    a = ap.parse_args()
    d = build(a.pose_dir, a.scores, a.out)
    print(d.shape)
