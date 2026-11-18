"""Temporal body-pose tracking for the B1 body-only pipeline.

The per-frame detector used in ``CV_project_rebuilt.ipynb`` sometimes alternates
between a good skier box and a missed frame.  This module adds an offline
temporal layer:

1. collect every YOLO pose candidate for every frame,
2. grow one track per detected person and keep the track present longest
   (``track_mode="longest"``), or use the original entry-box Viterbi path
   (``track_mode="entry"``),
3. inside a camera shot, require physically plausible and temporally smooth
   motion: a hard association gate, a central-square prior, and a second DP
   pass with a Savitzky-Golay smoothed trajectory prior,
4. keep every camera shot (a cut is the only allowed position jump),
5. interpolate short gaps in the track (never across a cut),
6. refine the pose on the tracked/interpolated boxes,
7. interpolate short keypoint gaps (never across a cut).

It is intentionally offline: the whole clip is available, so a confident
detection near the end can correct a weak detection near the beginning.

This module does not use the classical ski detector.  Ski columns are left at
zero.  Only ``<clip>_pose.npy`` is written; the redundant padded
``<clip>_pose21.npy`` copy is no longer produced.
"""
from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ski_detector import (
    bbox_from_keypoints,
    pose_keypoints_in_roi,
)

# --------------------------------------------------------------------------- #
# physical association gate
# --------------------------------------------------------------------------- #
# A tracked box is a physical object: between two consecutive frames inside one
# camera shot it can only move by a bounded amount.  The bound scales with the
# body size (a close-up can move more pixels than a distant skier) with a small
# frame-width floor.  Camera cuts are exempt: the skier legitimately jumps to a
# new position when the broadcast changes camera.
ENABLE_GATE = True           # master switch (A/B testing)
MAX_GATE_GAP = 15            # hard gate applies across all trackable gaps (up to max_gap)
MAX_STEP_BODY = 1.5          # centre move per frame, in body sizes
MAX_STEP_FRAME = 0.05        # minimum allowance, fraction of frame width
MAX_STEP_FRAME_CAP = 0.35    # hard cap per frame, fraction of frame width
MAX_GAP_FRAME_CAP = 0.60     # hard cap for any gap, fraction of frame width
MAX_SIZE_LOG_STEP = 1.0      # |log(size ratio)| allowed per frame
MAX_SIZE_LOG_CAP = 2.0       # hard cap on the size ratio over a gap
PERSON_STABILITY_FRAC = 0.8  # motion/size selection needs this fraction of
                             # the longest track's frames

# --- within-shot travel direction -------------------------------------------
# A skier travels in one ground-relative direction inside a camera shot; a
# spectator walking back and forth, or a track that jumps between two people,
# reverses.  Camera pans are removed with the phase-correlation shifts before
# the sign is measured.  The rule is soft: it only overrides a longer track
# when a direction-consistent alternative has a reasonable length.
ENABLE_DIRECTION_PRIOR = True     # within-shot travel is one-directional
DIRECTION_MIN_CONSISTENCY = 0.60  # |net dx| / sum|dx| for a moving track
DIRECTION_MIN_MOTION = 1.0        # body sizes of ground dx before judging
DIRECTION_MIN_FRAMES = 15         # short tracks are not judged (jump fragments)
DIRECTION_LENGTH_FRAC = 0.50      # alternative must be this long vs longest

# --- pose plausibility / static-object prior --------------------------------
# YOLO pose fires on signs, shadows, course markers and helmet/face close-ups.
# A real skier has several body joints, usually some lower-body joints, a
# keypoint cloud that fills the box, and keypoints near the box centre.  The
# penalty is soft: it only changes the choice when a better candidate exists.
ENABLE_POSE_QUALITY = True
POSE_QUALITY_WEIGHT = 0.8
POSE_MIN_JOINTS = 6
POSE_MIN_LOWER = 1
POSE_MAX_BOX_OFFSET = 0.60
POSE_MIN_SPREAD = 0.25

# --- event-aware track selection --------------------------------------------
# A long approach segment can be longer than the jump segment.  The jump has
# much more shoulder-line rotation (flip/spin) and usually a larger vertical
# excursion, so "event" selection scores rotation first, then vertical range.
EVENT_LENGTH_FRAC = 0.40      # candidate must be this long vs the longest
EVENT_ROTATION_WEIGHT = 1.0   # deg/frame
EVENT_VERTICAL_WEIGHT = 1.0   # body sizes

# --- cut detection: texture test --------------------------------------------
# The brightness-histogram test misses cuts between two visually similar wide
# shots, which is exactly what broadcast competition clips do (a snowy run-out
# looks like another snowy run-out).  A real camera change also destroys the
# local image features: almost no ORB descriptor matches survive.  Inside a
# shot, even a fast pan keeps 40-70% of them.  The texture test is only run
# when the frame pair already differs measurably, and it is skipped when the
# frames have too few features to judge (motion blur, dark sky).
CUT_TEXTURE_MIN_DIFF = 12.0        # mean abs diff that triggers the test
CUT_TEXTURE_MATCH_MIN = 0.10       # match ratio below this = different scene
CUT_TEXTURE_MIN_FEATURES = 20      # too few features -> cannot judge
CUT_TEXTURE_SIZE = (320, 180)      # resolution used for feature matching

# --- shot-level skier presence ----------------------------------------------
# Kept as a diagnostic: the tracker still writes every shot, but feature
# scripts can drop crowd/spectator/partial shots using this score.
SKIER_SCORE_MIN = 0.45

# --- smooth-motion / centre prior -------------------------------------------
CENTER_PRIOR_WEIGHT = 3.0    # cost per normalised unit outside the square
CENTER_SQUARE_RADIUS = 0.5   # central 50% square is free; linear penalty outside
SMOOTH_WINDOW = 9            # Savitzky-Golay window (odd, frames)
SMOOTH_POLYORDER = 2
PRIOR_WEIGHT = 0.6           # weight of the smoothed-trajectory prior
SMOOTH_REFINE_MIN = 0.8      # keep the refined path if it keeps this fraction

# --- appearance consistency (soft, off by default) --------------------------
ENABLE_APPEARANCE = False    # compute/store colour descriptors in pass 1
APPEARANCE_WEIGHT = 0.0      # pairwise cost weight; 0 = off
APPEARANCE_MAX_GAP = 30      # frame intervals over which appearance is trusted
APPEARANCE_AB_BINS = 8       # Lab a/b histogram bins per body half
APPEARANCE_L_BINS = 8        # centred-L histogram bins per body half
APPEARANCE_MIN_PIXELS = 64   # minimum masked pixels for a valid descriptor

# --- candidate validity ------------------------------------------------------
# Broadcast overlays often contain a head-and-shoulders portrait that YOLO
# scores as a person.  Such a graphic has face/shoulder keypoints but no hips,
# knees or ankles, and can dominate a track because it is large and static.
# A strict "must have lower-body keypoints" rule also drops real skiers whose
# legs are occluded, so the filter is restricted to large boxes touching a
# frame edge with all visible keypoints concentrated in the upper part.
DROP_EDGE_PORTRAITS = True
PORTRAIT_EDGE_MARGIN = 0.02   # fraction of frame width/height
PORTRAIT_MIN_SIZE_FRAC = 0.15  # of frame width
PORTRAIT_MAX_KEYPOINTS = 9    # head/shoulder graphics have at most this many
MIN_LOWER_KEYPOINTS = 1       # hips/knees/ankles visible at min_kp_conf

# --- tiling for small / distant skiers ------------------------------------- #
# When the skier is far from camera (wide broadcast shot) the bounding box can
# be only 10-15 px tall at the native 1280 px inference size, below YOLO's
# reliable detection floor.  Splitting the frame into an overlapping grid and
# running YOLO on each tile upscales the skier to a detectable size without
# changing the model weights.  Tiling only fires when the full-frame pass finds
# no candidates (or only very small ones), so the extra compute is paid only
# when needed.
ENABLE_TILING      = True    # master switch; set False for AQA (skier already close)
TILE_TRIGGER_SIZE  = 50      # px: tile when every full-frame candidate is smaller
TILE_MIN_CANDS     = 1       # also tile when fewer than this many cands found
TILE_GRID          = (2, 2)  # (rows, cols); 2×2 quadrants are usually enough
TILE_OVERLAP       = 0.20    # fractional overlap so detections near tile edges survive
TILE_NMS_IOU       = 0.45    # IoU threshold to suppress same-person tiled duplicates

# --- same-shot tail recovery ------------------------------------------------ #
# One camera shot can contain the athlete twice: the in-run and, after the
# jump, the landing ride-out.  Only one Viterbi path is kept per shot, so the
# later segment is dropped whenever the earlier one is longer.  A second,
# confident segment in the uncovered tail of the shot is tracked as well.  The
# confidence floor is what separates a real athlete (median conf 0.5-0.9) from
# the floodlight / tower / banner false positives that sit at 0.1-0.35.
TAIL_RECOVERY_MIN_FRAMES = 45     # uncovered tail frames needed before we look
TAIL_RECOVERY_MIN_CONF = 0.40     # median detection confidence of the tail
TAIL_RECOVERY_MIN_BODY8 = 0.50    # fraction of tail frames with 8 body joints

try:
    from scipy.signal import savgol_filter, medfilt
except Exception:            # pragma: no cover - scipy is normally available
    savgol_filter = None
    medfilt = None


@dataclass
class Candidate:
    bbox: tuple[float, float, float, float]
    keypoints: np.ndarray
    conf: float
    center: np.ndarray
    size: float
    appearance: np.ndarray | None = None


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


def _looks_like_edge_portrait(candidate, frame_shape, min_kp_conf=0.25):
    """True for a large edge-anchored, upper-body-only overlay portrait.

    Real skiers can also miss lower-body keypoints (occlusion, spray, small
    scale), so the rule is deliberately narrow: the box must touch a frame
    edge, be large relative to the frame, have no hips/knees/ankles, and have
    only a few visible keypoints (head/shoulder graphics).
    """
    if not DROP_EDGE_PORTRAITS:
        return False
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = (float(v) for v in candidate.bbox)
    margin_x = PORTRAIT_EDGE_MARGIN * frame_w
    margin_y = PORTRAIT_EDGE_MARGIN * frame_h
    # Broadcast portraits sit at the left/right edge (occasionally the top).
    # A real skier close-up often touches the bottom edge, so that edge is not
    # used: dropping such a box would lose the athlete, not an overlay.
    touches_edge = (x <= margin_x or y <= margin_y
                    or x + w >= frame_w - margin_x)
    if not touches_edge:
        return False
    if max(w, h) < PORTRAIT_MIN_SIZE_FRAC * frame_w:
        return False
    visible = candidate.keypoints[:, 2] >= min_kp_conf
    if int(visible.sum()) < 3:
        return False
    if int(visible.sum()) > PORTRAIT_MAX_KEYPOINTS:
        return False
    if int(visible[11:17].sum()) >= MIN_LOWER_KEYPOINTS:
        return False
    return True


def _appearance_descriptor(frame, bbox, keypoints=None):
    """Return an illumination-tolerant colour descriptor for one person.

    The person crop is masked with the visible pose hull (or a central ellipse
    when too few joints are reliable), converted to Lab, split into upper and
    lower body halves, and summarised by ``a/b`` histograms plus an
    exposure-normalised ``L`` histogram.  The upper/lower split keeps "black
    top / white pants" distinct from the reverse, which a single global
    histogram cannot do.
    """
    if frame is None or frame.size == 0:
        return None
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = (float(v) for v in bbox)
    x0 = max(0, int(math.floor(x)))
    y0 = max(0, int(math.floor(y)))
    x1 = min(frame_w, int(math.ceil(x + w)))
    y1 = min(frame_h, int(math.ceil(y + h)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    crop = frame[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    if keypoints is not None and len(keypoints) >= 3:
        visible = keypoints[:, 2] >= 0.25
        if int(visible.sum()) >= 6:
            pts = keypoints[visible, :2].copy()
            pts[:, 0] -= x0
            pts[:, 1] -= y0
            pts = np.round(pts).astype(np.int32)
            pts[:, 0] = np.clip(pts[:, 0], 0, crop_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, crop_h - 1)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(mask, hull, 255)
    if int(mask.sum()) == 0:
        cv2.ellipse(mask, (crop_w // 2, crop_h // 2),
                    (max(2, int(0.45 * crop_w)),
                     max(2, int(0.48 * crop_h))),
                    0, 0, 360, 255, -1)
    # Include clothing just outside the joint hull.
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    valid = mask > 0
    if int(valid.sum()) < APPEARANCE_MIN_PIXELS:
        return None

    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    mean_l = float(lab[:, :, 0][valid].mean())
    min_half = max(8, APPEARANCE_MIN_PIXELS // 2)
    feats = []
    for rows in np.array_split(np.arange(crop_h), 2):
        if len(rows) == 0:
            return None
        y0h, y1h = int(rows[0]), int(rows[-1]) + 1
        half_mask = valid[y0h:y1h]
        if int(half_mask.sum()) < min_half:
            return None
        pixels = lab[y0h:y1h][half_mask]
        hist_ab, _, _ = np.histogram2d(
            pixels[:, 1], pixels[:, 2],
            bins=(APPEARANCE_AB_BINS, APPEARANCE_AB_BINS),
            range=((0, 256), (0, 256)))
        hist_l, _ = np.histogram(
            pixels[:, 0] - mean_l,
            bins=APPEARANCE_L_BINS, range=(-128, 128))
        hist_ab = hist_ab.ravel().astype(np.float32)
        hist_l = hist_l.astype(np.float32)
        hist_ab /= max(float(hist_ab.sum()), 1.0)
        hist_l /= max(float(hist_l.sum()), 1.0)
        feats.append(np.concatenate([hist_ab, hist_l]))
    desc = np.concatenate(feats).astype(np.float32)
    total = float(desc.sum())
    if total <= 0:
        return None
    return desc / total


def collect_candidates(model, frame, conf_thres=0.1, bbox_pad=0.25,
                       min_keypoints=3, min_kp_conf=0.25,
                       max_bbox_frac=0.95, min_aspect=0.15, max_aspect=5.0,
                       tiling=None):
    """Return every plausible person as a Candidate for one frame.

    Broadcast clips and some AQA clips contain spectators, officials and large
    background structures.  A raw YOLO "person" box with no visible keypoints is
    usually a false positive, so candidates with too few confident keypoints or
    an implausible box shape are dropped before tracking.
    """
    if tiling is None:
        tiling = ENABLE_TILING
    result = model(frame, verbose=False, conf=conf_thres)[0]
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        # No full-frame detection at all.  This is exactly the case tiling
        # exists for: a distant skier can be below YOLO's full-frame
        # detection floor yet clearly visible once a tile is upscaled.  The
        # early return here used to skip the tiled pass, so wide shots with
        # no full-frame candidates never got a second chance.
        if tiling:
            return _apply_tiling(model, frame, [],
                                 conf_thres=conf_thres, bbox_pad=bbox_pad)
        return []

    boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else None
    box_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else None
    xy = result.keypoints.xy.cpu().numpy()
    kconf = (result.keypoints.conf.cpu().numpy()
             if result.keypoints.conf is not None else np.ones(xy.shape[:2], dtype=np.float32))

    height, width = frame.shape[:2]
    out = []
    for i in range(len(xy)):
        kp = np.zeros((xy.shape[1], 3), dtype=np.float32)
        kp[:, :2] = xy[i]
        kp[:, 2] = kconf[i]
        if int((kp[:, 2] >= min_kp_conf).sum()) < min_keypoints:
            continue
        if boxes is not None and len(boxes) > i:
            box = boxes[i]
            conf = float(box_conf[i]) if box_conf is not None else float(kp[:, 2].mean())
        else:
            bbox = bbox_from_keypoints(kp, frame.shape[:2], pad=bbox_pad)
            if bbox is None:
                continue
            box = bbox
            conf = float(kp[:, 2].mean())
        cand = _candidate_from_box(box, kp, conf, frame.shape[:2], bbox_pad)
        bw, bh = float(cand.bbox[2]), float(cand.bbox[3])
        if bw < 6.0 or bh < 6.0:
            continue
        if bw > max_bbox_frac * width or bh > max_bbox_frac * height:
            continue
        aspect = bh / max(bw, 1.0)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        if _looks_like_edge_portrait(cand, frame.shape, min_kp_conf):
            continue
        if ENABLE_APPEARANCE:
            cand.appearance = _appearance_descriptor(
                frame, cand.bbox, cand.keypoints)
        out.append(cand)
    if tiling:
        out = _apply_tiling(model, frame, out, conf_thres=conf_thres,
                            bbox_pad=bbox_pad)
    return out


def _merge_candidates_nms(a_list, b_list, iou_thr=TILE_NMS_IOU):
    """Merge two Candidate lists, deduplicating by IoU.

    Candidates from ``b_list`` that overlap an existing entry by more than
    ``iou_thr`` are suppressed (or replace the existing entry if they have
    higher detection confidence).  Used to combine full-frame and tiled
    detections without double-counting the same person.
    """
    merged = list(a_list)
    for b in b_list:
        replaced = False
        for i, m in enumerate(merged):
            if iou_xywh(b.bbox, m.bbox) >= iou_thr:
                if b.conf > m.conf:
                    merged[i] = b          # keep the more confident detection
                replaced = True
                break
        if not replaced:
            merged.append(b)
    return merged


def collect_candidates_tiled(model, frame, conf_thres=0.1, bbox_pad=0.25,
                              min_keypoints=3, min_kp_conf=0.25,
                              min_aspect=0.15, max_aspect=5.0,
                              grid=TILE_GRID, overlap=TILE_OVERLAP):
    """Run YOLO on a grid of overlapping tiles; return full-frame Candidates.

    Each tile sub-image is passed to the model at the model's configured imgsz,
    upscaling a distant skier to a detectable apparent size.  Detections are
    translated back to full-frame pixel coordinates so they can be merged with
    normal full-frame candidates via :func:`_merge_candidates_nms`.

    The portrait filter and global size cap are re-applied in full-frame space
    so a tile-local detection that maps to an edge portrait is still suppressed.
    """
    frame_h, frame_w = frame.shape[:2]
    rows, cols = grid
    tile_h = frame_h / rows
    tile_w = frame_w / cols
    pad_h = int(tile_h * overlap)
    pad_w = int(tile_w * overlap)

    out = []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, int(r * tile_h) - pad_h)
            y1 = min(frame_h, int((r + 1) * tile_h) + pad_h)
            x0 = max(0, int(c * tile_w) - pad_w)
            x1 = min(frame_w, int((c + 1) * tile_w) + pad_w)
            if y1 - y0 < 32 or x1 - x0 < 32:
                continue
            tile = frame[y0:y1, x0:x1]
            # collect_candidates applies filters relative to the tile, which is
            # intentional: a person-sized box inside the tile should pass even
            # if it would be tiny in the full frame.
            tile_cands = collect_candidates(
                model, tile, conf_thres=conf_thres, bbox_pad=bbox_pad,
                min_keypoints=min_keypoints, min_kp_conf=min_kp_conf,
                max_bbox_frac=0.95, min_aspect=min_aspect, max_aspect=max_aspect,
                tiling=False)
            for cand in tile_cands:
                tx, ty, tw, th = cand.bbox
                # Translate box and keypoints to full-frame coordinates.
                fx, fy = tx + x0, ty + y0
                new_kp = cand.keypoints.copy()
                new_kp[:, 0] += x0
                new_kp[:, 1] += y0
                new_center = np.array([fx + tw / 2, fy + th / 2], dtype=np.float32)
                full_cand = Candidate(
                    bbox=(fx, fy, tw, th),
                    keypoints=new_kp,
                    conf=cand.conf,
                    center=new_center,
                    size=cand.size,
                    appearance=cand.appearance,
                )
                # Re-apply full-frame portrait filter and global size cap.
                if _looks_like_edge_portrait(full_cand, frame.shape):
                    continue
                if tw > 0.95 * frame_w or th > 0.95 * frame_h:
                    continue
                out.append(full_cand)
    return out


def _apply_tiling(model, frame, cands, conf_thres=0.1, bbox_pad=0.25):
    """Return a merged candidate list that includes tiled detections if needed.

    Tiling is triggered when either:
    * no full-frame candidates were found at all (including the case where the
      full-frame model returned no keypoints and ``collect_candidates`` calls
      this function directly from its early return), or
    * the best full-frame detection is smaller than ``TILE_TRIGGER_SIZE`` px
      AND fewer than ``TILE_MIN_CANDS`` candidates were found.

    Both conditions indicate the skier is too small / distant for a reliable
    full-frame YOLO pass.
    """
    if not ENABLE_TILING:
        return cands
    too_few  = len(cands) < TILE_MIN_CANDS
    too_small = cands and max(c.size for c in cands) < TILE_TRIGGER_SIZE
    if not (too_few or too_small):
        return cands
    tiled = collect_candidates_tiled(model, frame, conf_thres=conf_thres,
                                     bbox_pad=bbox_pad)
    return _merge_candidates_nms(cands, tiled)


def _appearance_cost(a: Candidate, b: Candidate):
    """Histogram-intersection distance between two person descriptors.

    Returns 0 when either descriptor is unavailable, so an unknown appearance
    never acts as a hard penalty.
    """
    if a.appearance is None or b.appearance is None:
        return 0.0
    if not np.any(a.appearance) or not np.any(b.appearance):
        return 0.0
    inter = float(np.minimum(a.appearance, b.appearance).sum())
    return float(np.clip(1.0 - inter, 0.0, 1.0))


def _transition_cost(a: Candidate, b: Candidate, iou_weight=1.0,
                     distance_weight=1.2, size_weight=0.8,
                     appearance_weight=0.0):
    scale = max((a.size + b.size) / 2.0, 1.0)
    distance = float(np.linalg.norm(a.center - b.center) / scale)
    aw, ah = float(a.bbox[2]), float(a.bbox[3])
    bw, bh = float(b.bbox[2]), float(b.bbox[3])
    size_cost = abs(math.log(max(bw, 1.0) / max(aw, 1.0))) + \
                abs(math.log(max(bh, 1.0) / max(ah, 1.0)))
    cost = distance_weight * distance + size_weight * size_cost + \
        iou_weight * (1.0 - iou_xywh(a.bbox, b.bbox))
    if appearance_weight:
        cost += appearance_weight * _appearance_cost(a, b)
    return cost


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


def _crosses_cut(t0, t1, cuts):
    """True when the transition t0 -> t1 spans a camera cut."""
    return bool(cuts) and any(int(t0) < int(c) <= int(t1) for c in cuts)


def _gate_ok(a: Candidate, b: Candidate, steps=1, frame_width=None):
    """True when moving from ``a`` to ``b`` is physically plausible.

    ``steps`` is the number of frame intervals between the two detections
    (1 for consecutive frames).  The centre bound grows with the body size and
    with the number of missed frames, and is capped by a fraction of the frame
    width so a spectator cannot be linked to the skier in one step.
    """
    steps = max(int(steps), 1)
    size = max((a.size + b.size) / 2.0, 1.0)
    limit = MAX_STEP_BODY * size
    if frame_width:
        fw = float(frame_width)
        limit = max(limit, MAX_STEP_FRAME * fw)
        limit = min(limit, MAX_STEP_FRAME_CAP * fw)
        limit = min(limit * steps, MAX_GAP_FRAME_CAP * fw)
    else:
        limit = limit * steps
    if float(np.linalg.norm(a.center - b.center)) > limit:
        return False
    ratio = max(b.size, 1.0) / max(a.size, 1.0)
    return abs(math.log(ratio)) <= min(MAX_SIZE_LOG_STEP * steps,
                                      MAX_SIZE_LOG_CAP)


def _size_ok(a: Candidate, b: Candidate):
    """True when the apparent body size is consistent within one shot."""
    ratio = max(b.size, 1.0) / max(a.size, 1.0)
    return abs(math.log(ratio)) <= MAX_SIZE_LOG_CAP


def _center_penalty(candidate, frame_width, frame_height):
    """Penalty for a candidate outside the central square (0 inside).

    The normalised offsets are in ``[0, 1]`` from the frame centre to an edge.
    ``max(nx, ny)`` makes the zero-cost region a true square, not a circle:
    with ``CENTER_SQUARE_RADIUS = 0.5`` the central 50% of the frame is free.
    """
    if not frame_width or not frame_height:
        return 0.0
    nx = abs(float(candidate.center[0]) - float(frame_width) / 2.0) \
        / max(float(frame_width) / 2.0, 1.0)
    ny = abs(float(candidate.center[1]) - float(frame_height) / 2.0) \
        / max(float(frame_height) / 2.0, 1.0)
    return max(0.0, max(nx, ny) - CENTER_SQUARE_RADIUS)


def _prior_cost(candidate, t, prior):
    """Cost of deviating from the smoothed trajectory prediction at frame t."""
    if prior is None or t not in prior:
        return 0.0
    px, py, pw, ph = prior[t]
    scale = max(candidate.size, 1.0)
    distance = math.hypot(float(candidate.center[0]) - px,
                          float(candidate.center[1]) - py) / scale
    size_cost = abs(math.log(max(candidate.bbox[2], 1.0) / max(pw, 1.0))) + \
                abs(math.log(max(candidate.bbox[3], 1.0) / max(ph, 1.0)))
    return distance + 0.5 * size_cost


def _pose_quality_penalty(candidate, frame_width=None, frame_height=None):
    """Soft cost for a partial or implausible body pose.

    A real skier has several visible joints, usually at least one hip/knee/
    ankle, and the keypoint cloud fills the box.  Signs, shadows, course
    markers and helmet/face close-ups have few joints, a tiny spread, or
    keypoints far from the box centre.
    """
    kp = candidate.keypoints
    if kp is None or len(kp) < 17:
        return 0.3
    conf = kp[:17, 2]
    visible = conf >= 0.3
    n_body = int(visible.sum())
    if n_body == 0:
        return 0.6
    n_lower = int(visible[11:17].sum())
    n_torso = int(visible[[5, 6, 11, 12]].sum())
    xy = kp[visible, :2]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    pose_center = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0],
                           dtype=np.float32)
    offset = float(np.linalg.norm(pose_center - candidate.center)
                   / max(candidate.size, 1.0))
    spread = float(max(x1 - x0, y1 - y0) / max(candidate.size, 1.0))

    penalty = 0.45 * max(0.0, 1.0 - n_body / 8.0)
    if n_lower < POSE_MIN_LOWER:
        penalty += 0.35
    if n_torso < 3:
        penalty += 0.15
    if offset > POSE_MAX_BOX_OFFSET:
        penalty += 0.6 * min(1.0, (offset - POSE_MAX_BOX_OFFSET) / 0.6)
    if spread < POSE_MIN_SPREAD:
        penalty += 0.5 * (1.0 - spread / POSE_MIN_SPREAD)
    if frame_width and frame_height:
        x, y, w, h = candidate.bbox
        margin_x = 0.02 * frame_width
        margin_y = 0.02 * frame_height
        edge = (x <= margin_x or y <= margin_y
                or x + w >= frame_width - margin_x)
        if edge and n_body < 8:
            penalty += 0.15
    return float(min(penalty, 1.5))


def _emission_cost(candidate, t, frame_width, frame_height, prior,
                   emit_weight, center_prior):
    cost = -math.log(max(candidate.conf, 1e-4)) * emit_weight
    if center_prior:
        cost += CENTER_PRIOR_WEIGHT * _center_penalty(
            candidate, frame_width, frame_height)
    if ENABLE_POSE_QUALITY:
        cost += POSE_QUALITY_WEIGHT * _pose_quality_penalty(
            candidate, frame_width, frame_height)
    if prior is not None:
        cost += PRIOR_WEIGHT * _prior_cost(candidate, t, prior)
    return cost


def smooth_path(path, n_frames, frame_width, frame_height, cuts=None,
                window=SMOOTH_WINDOW, polyorder=SMOOTH_POLYORDER):
    """Return ``{t: (cx, cy, w, h)}`` from a temporal smoother.

    Each camera shot is smoothed independently, so a cut never averages two
    camera views.  Gaps are filled by linear interpolation before the
    Savitzky-Golay filter, which gives the DP a prediction for every frame of
    the shot.
    """
    if not path:
        return {}
    cuts = sorted({int(c) for c in (cuts or []) if 0 < int(c) < n_frames})
    bounds = [0] + cuts + [n_frames]
    out = {}
    for a, b in zip(bounds[:-1], bounds[1:]):
        pts = [(int(t), path[t]) for t in path if a <= t < b]
        if not pts:
            continue
        pts.sort()
        ts = np.asarray([t for t, _ in pts], dtype=float)
        cx = np.asarray([p.center[0] for _, p in pts], dtype=float)
        cy = np.asarray([p.center[1] for _, p in pts], dtype=float)
        lw = np.log(np.maximum([p.bbox[2] for _, p in pts], 1.0))
        lh = np.log(np.maximum([p.bbox[3] for _, p in pts], 1.0))
        grid = np.arange(a, b, dtype=float)
        cx = np.interp(grid, ts, cx)
        cy = np.interp(grid, ts, cy)
        lw = np.interp(grid, ts, lw)
        lh = np.interp(grid, ts, lh)
        n = len(grid)
        w = min(int(window), n if n % 2 == 1 else n - 1)
        if len(pts) >= 3 and w >= 5:
            # A short median prefilter removes one-frame outliers before the
            # Savitzky-Golay smoothing, so a single bad detection cannot drag
            # the trajectory prediction away from the real path.
            if medfilt is not None and n >= 5:
                cx = medfilt(cx, 5)
                cy = medfilt(cy, 5)
                lw = medfilt(lw, 5)
                lh = medfilt(lh, 5)
            po = min(int(polyorder), w - 1)
            if savgol_filter is not None:
                cx = savgol_filter(cx, w, po)
                cy = savgol_filter(cy, w, po)
                lw = savgol_filter(lw, w, po)
                lh = savgol_filter(lh, w, po)
            else:  # simple centred moving average fallback
                k = np.ones(w, dtype=float) / w
                cx = np.convolve(cx, k, mode="same")
                cy = np.convolve(cy, k, mode="same")
                lw = np.convolve(lw, k, mode="same")
                lh = np.convolve(lh, k, mode="same")
        for i, t in enumerate(grid.astype(int)):
            out[int(t)] = (float(cx[i]), float(cy[i]),
                           float(np.exp(lw[i])), float(np.exp(lh[i])))
    return out


def _split_path_on_gate(path, frame_width=None, cuts=None):
    """Split a path wherever a transition violates the physical gate."""
    keys = sorted(path)
    if not keys:
        return []
    segments, cur = [], {keys[0]: path[keys[0]]}
    for t0, t1 in zip(keys[:-1], keys[1:]):
        # The hard position gate applies to consecutive frames and short gaps;
        # long gaps are only checked for an implausible size change.
        steps = t1 - t0
        if (not ENABLE_GATE) or _crosses_cut(t0, t1, cuts):
            ok = True
        elif steps <= MAX_GATE_GAP + 1:
            ok = _gate_ok(path[t0], path[t1], steps, frame_width)
        else:
            ok = _size_ok(path[t0], path[t1])
        if ok:
            cur[t1] = path[t1]
        else:
            segments.append(cur)
            cur = {t1: path[t1]}
    segments.append(cur)
    return segments


def _merge_path_segments(segments, frame_width=None, cuts=None):
    """Stitch gated segments together where the inter-segment gap is plausible.

    After :func:`_split_path_on_gate` breaks a path at impossible links, we try
    to re-join adjacent segments.  Three cases:

    1. The gap **crosses a detected camera cut** — always merge (position jump is
       expected at a cut).
    2. The gap is **short** (≤ ``MAX_GATE_GAP + 1`` frames) and passes the
       physical gate — merge.
    3. The gap is **long** but the apparent body sizes are consistent — merge.

    If a segment cannot be merged into the growing result *and* it is longer than
    the accumulated path, we switch to it (the later segment is the skier).
    If none of the above, the unattachable segment is simply dropped.

    The result always preserves a path that is internally gate-consistent.
    """
    if not segments:
        return {}
    merged = dict(segments[0])
    for seg in segments[1:]:
        if not merged or not seg:
            if seg:
                merged = dict(seg)
            continue
        last_t  = max(merged)
        first_t = min(seg)
        last_cand  = merged[last_t]
        first_cand = seg[first_t]
        crosses = _crosses_cut(last_t, first_t, cuts)
        if crosses:
            # Camera cut: position jump is legitimate — always stitch.
            merged.update(seg)
        else:
            steps = first_t - last_t
            if steps <= MAX_GATE_GAP + 1:
                ok = _gate_ok(last_cand, first_cand, steps, frame_width)
            else:
                ok = _size_ok(last_cand, first_cand)
            if ok:
                merged.update(seg)
            elif len(seg) > len(merged):
                # Later segment is longer — switch to it rather than growing a
                # shorter orphan.
                merged = dict(seg)
            # else: drop the unattachable shorter segment
    return merged


def viterbi_track(candidates, entry_bbox=None, max_gap=15,
                  emit_weight=1.0, miss_penalty=2.5, seed=None,
                  frame_width=None, frame_height=None, cuts=None, stats=None,
                  prior=None, center_prior=True):
    """Choose the best single track through per-frame candidates.

    ``candidates`` is a list indexed by relative frame; each entry is a list of
    Candidate objects.  Returns ``{frame_index: Candidate}`` for the chosen path.

    ``seed=(t0, i0)`` starts the path at candidate ``i0`` of frame ``t0``.  This
    is used by :func:`choose_person_track` to grow one track per person instead
    of always starting from the entry box.

    ``frame_width`` enables the physical association gate: transitions that move
    the box further than a body-size-scaled bound are rejected.  Transitions
    that cross a camera cut in ``cuts`` are exempt.
    """
    if seed is not None:
        first, seed_i = seed
        if first >= len(candidates) or not candidates[first]:
            return {}
        if seed_i >= len(candidates[first]):
            return {}
    else:
        first = next((t for t, c in enumerate(candidates) if c), None)
        if first is None:
            return {}

    # dp[t][candidate_index] = (cost, previous=(prev_t, prev_index) | None)
    dp = [dict() for _ in candidates]
    for i, cand in enumerate(candidates[first]):
        if seed is not None and i != seed_i:
            continue
        emission = _emission_cost(cand, first, frame_width, frame_height,
                                  prior, emit_weight, center_prior)
        entry_cost = 0.0 if seed is not None else _entry_prior(cand, entry_bbox)
        dp[first][i] = (emission + entry_cost, None)

    # ----------------------------------------------------------------------- #
    # Forward Viterbi pass (t from first + 1 to len(candidates) - 1)
    # ----------------------------------------------------------------------- #
    for t in range(first + 1, len(candidates)):
        if not candidates[t]:
            continue
        lookback = range(t - 1, max(first, t - max_gap - 1) - 1, -1)
        for i, cand in enumerate(candidates[t]):
            emission = _emission_cost(cand, t, frame_width, frame_height,
                                      prior, emit_weight, center_prior)
            best_cost, best_prev = float("inf"), None
            for k in lookback:
                if not dp[k]:
                    continue
                gap = t - k - 1
                for j, prev_cand in dp[k].items():
                    # The hard physical gate applies to consecutive frames and
                    # to short gaps (up to MAX_GATE_GAP missed frames).  Across
                    # a long gap the skier has had time to move, so only the
                    # apparent body size is constrained there.
                    steps = t - k
                    if ENABLE_GATE and not _crosses_cut(k, t, cuts):
                        if stats is not None:
                            stats["gate_checks"] = (
                                stats.get("gate_checks", 0) + 1)
                        if steps <= MAX_GATE_GAP + 1:
                            ok = _gate_ok(candidates[k][j], cand, steps,
                                          frame_width)
                        else:
                            ok = _size_ok(candidates[k][j], cand)
                        if not ok:
                            if stats is not None:
                                stats["gate_rejects"] = (
                                    stats.get("gate_rejects", 0) + 1)
                            continue
                    app_w = 0.0
                    if (APPEARANCE_WEIGHT > 0
                            and not _crosses_cut(k, t, cuts)
                            and steps <= APPEARANCE_MAX_GAP):
                        app_w = APPEARANCE_WEIGHT
                    cost = prev_cand[0] + _transition_cost(
                        candidates[k][j], cand,
                        appearance_weight=app_w) + miss_penalty * gap
                    if cost < best_cost:
                        best_cost, best_prev = cost, (k, j)
            if best_prev is not None:
                dp[t][i] = (best_cost + emission, best_prev)

    last = max((t for t in range(len(dp)) if dp[t]), default=None)
    if last is None:
        return {}

    best_idx = min(dp[last], key=lambda j: dp[last][j][0])
    forward_path = {}
    t, j = last, best_idx
    while True:
        forward_path[t] = candidates[t][j]
        prev = dp[t][j][1]
        if prev is None:
            break
        t, j = prev

    # ----------------------------------------------------------------------- #
    # Backward Viterbi pass (t from first - 1 down to 0)
    # ----------------------------------------------------------------------- #
    # When a person is seeded at frame t0 > 0 (e.g. from rank_seed_tracks
    # sampling every 2 seconds), forward-only Viterbi loses all frames before
    # t0.  Running backward Viterbi from the same seed candidate recovers the
    # athlete's trajectory backwards to their true entry frame.
    backward_path = {}
    if seed is not None and first > 0:
        dp_back = [dict() for _ in range(first + 1)]
        dp_back[first][seed_i] = (0.0, None)
        for t in range(first - 1, -1, -1):
            if not candidates[t]:
                continue
            lookahead = range(t + 1, min(first + 1, t + max_gap + 2))
            for i, cand in enumerate(candidates[t]):
                emission = _emission_cost(cand, t, frame_width, frame_height,
                                          prior, emit_weight, center_prior)
                best_cost, best_next = float("inf"), None
                for k in lookahead:
                    if not dp_back[k]:
                        continue
                    gap = k - t - 1
                    steps = k - t
                    for j, next_cand in dp_back[k].items():
                        if ENABLE_GATE and not _crosses_cut(t, k, cuts):
                            if stats is not None:
                                stats["gate_checks"] = (
                                    stats.get("gate_checks", 0) + 1)
                            if steps <= MAX_GATE_GAP + 1:
                                ok = _gate_ok(cand, candidates[k][j], steps,
                                              frame_width)
                            else:
                                ok = _size_ok(cand, candidates[k][j])
                            if not ok:
                                if stats is not None:
                                    stats["gate_rejects"] = (
                                        stats.get("gate_rejects", 0) + 1)
                                continue
                        app_w = 0.0
                        if (APPEARANCE_WEIGHT > 0
                                and not _crosses_cut(t, k, cuts)
                                and steps <= APPEARANCE_MAX_GAP):
                            app_w = APPEARANCE_WEIGHT
                        cost = next_cand[0] + _transition_cost(
                            cand, candidates[k][j],
                            appearance_weight=app_w) + miss_penalty * gap
                        if cost < best_cost:
                            best_cost, best_next = cost, (k, j)
                if best_next is not None:
                    dp_back[t][i] = (best_cost + emission, best_next)

        earliest = min((t for t in range(first) if dp_back[t]), default=None)
        if earliest is not None:
            best_idx = min(dp_back[earliest], key=lambda j: dp_back[earliest][j][0])
            t, j = earliest, best_idx
            while t < first:
                backward_path[t] = candidates[t][j]
                next_ptr = dp_back[t][j][1]
                if next_ptr is None:
                    break
                t, j = next_ptr

    path = {**backward_path, **forward_path}

    # Safety net: the DP gate should already prevent impossible links, but if a
    # path still contains one (e.g. an unusual gap), split it.  Rather than
    # simply discarding all but the seed segment (old behaviour), try to stitch
    # the segments back together via _merge_path_segments: segments separated by
    # a camera cut or a gate-plausible gap are re-joined; segments that cannot
    # be joined are dropped or replaced only if they are longer.
    segments = _split_path_on_gate(path, frame_width=frame_width, cuts=cuts)
    if len(segments) > 1:
        if stats is not None:
            stats["gate_splits"] = stats.get("gate_splits", 0) + len(segments) - 1
        path = _merge_path_segments(segments, frame_width=frame_width, cuts=cuts)
        # If the seed frame was lost during merging (should be very rare),
        # fall back to the seed-containing segment.
        if seed is not None and seed[0] not in path:
            seed_seg = next((s for s in segments if seed[0] in s), segments[0])
            path = dict(seed_seg)
    return path


def _same_person(a: Candidate, b: Candidate, rel_dist=0.75) -> bool:
    """True when two candidates are close enough to be the same person."""
    distance = float(np.linalg.norm(a.center - b.center))
    return distance < rel_dist * max(a.size, b.size, 1.0)


def _path_summary(path, camera_shifts=None):
    """Basic motion/size summary for a candidate path.

    When ``camera_shifts`` is provided, it also calculates motion relative to
    the ground (subtracting camera background translation), separating stationary
    spectators from a moving skier even when the camera pans with the skier.
    """
    keys = sorted(path)
    if not keys:
        return None
    ordered = [path[t] for t in keys]
    centers = np.asarray([c.center for c in ordered], dtype=np.float32)
    sizes = np.asarray([c.size for c in ordered], dtype=np.float32)
    median_size = float(np.median(sizes)) if sizes.size else 1.0

    # Image-plane motion (raw pixel coordinates)
    diffs = np.diff(centers, axis=0) if len(ordered) > 1 else np.zeros((0, 2), dtype=np.float32)
    image_motion = (float(np.sum(np.linalg.norm(diffs, axis=1)))
                    / max(median_size, 1.0)) if len(diffs) else 0.0
    vy_range = (float(centers[:, 1].max() - centers[:, 1].min())
                if len(keys) else 0.0)

    # Shoulder-line rotation rate: high for a flip/spin, low for an approach.
    angles = []
    for t, cand in zip(keys, ordered):
        kp = cand.keypoints
        if kp is None or len(kp) < 7:
            continue
        if kp[5, 2] >= 0.3 and kp[6, 2] >= 0.3:
            dx = float(kp[6, 0] - kp[5, 0])
            dy = float(kp[6, 1] - kp[5, 1])
            angles.append((t, math.degrees(math.atan2(dy, dx))))
    rotation_total = 0.0
    rotation_steps = 0
    for (t0, a0), (t1, a1) in zip(angles[:-1], angles[1:]):
        if t1 - t0 - 1 > 3:
            continue
        step = (a1 - a0 + 180.0) % 360.0 - 180.0
        if abs(step) <= 90.0:  # reject shoulder-axis swaps
            rotation_total += abs(step)
            rotation_steps += 1
    rotation_energy = rotation_total / max(len(keys), 1)
    rotation_support = rotation_steps / max(len(keys) - 1, 1)

    # Ground-relative motion (camera-compensated)
    # A spectator static on the ground has ground_motion ≈ 0 even if the camera pans.
    # A skier moving at speed has ground_motion >> 0 even if kept in frame center.
    ground_dx = np.zeros(0, dtype=np.float32)
    if camera_shifts is not None and len(diffs) > 0:
        ground_diffs = []
        ground_positions = [centers[0].copy()]
        n_shifts = len(camera_shifts)
        for idx in range(len(diffs)):
            t0, t1 = keys[idx], keys[idx + 1]
            cam_shift = np.zeros(2, dtype=np.float32)
            for tau in range(t0, min(t1, n_shifts)):
                cam_shift += camera_shifts[tau]
            g_d = diffs[idx] - cam_shift
            ground_diffs.append(g_d)
            ground_positions.append(ground_positions[-1] + g_d)
        ground_diffs = np.asarray(ground_diffs, dtype=np.float32)
        ground_positions = np.asarray(ground_positions, dtype=np.float32)
        ground_motion = (float(np.sum(np.linalg.norm(ground_diffs, axis=1)))
                         / max(median_size, 1.0))
        g_vy_range = float(ground_positions[:, 1].max() - ground_positions[:, 1].min())
        ground_vy_range_rel = g_vy_range / max(median_size, 1.0)
        ground_dx = ground_diffs[:, 0]
    else:
        ground_motion = image_motion
        ground_vy_range_rel = vy_range / max(median_size, 1.0)
        ground_dx = diffs[:, 0] if len(diffs) else np.zeros(0, dtype=np.float32)

    # Signed horizontal travel: a skier keeps one ground-relative direction
    # inside a shot; a reversing spectator or an identity switch does not.
    if ground_dx.size:
        significant = np.abs(ground_dx) > 0.05 * max(median_size, 1.0)
        if significant.any():
            net_dx = float(np.sum(ground_dx[significant]))
            abs_dx = float(np.sum(np.abs(ground_dx[significant])))
            direction_consistency = abs(net_dx) / max(abs_dx, 1e-6)
            ground_dir = int(np.sign(net_dx))
        else:
            net_dx = abs_dx = 0.0
            direction_consistency = 0.0
            ground_dir = 0
    else:
        net_dx = abs_dx = 0.0
        direction_consistency = 0.0
        ground_dir = 0

    return {
        "frames": len(keys),
        "span": max(keys) - min(keys) + 1,
        "median_size": median_size,
        "motion": image_motion,
        "ground_motion": ground_motion,
        "vy_range_rel": vy_range / max(median_size, 1.0),
        "ground_vy_range_rel": ground_vy_range_rel,
        "ground_dx_rel": net_dx / max(median_size, 1.0),
        "ground_abs_dx_rel": abs_dx / max(median_size, 1.0),
        "direction_consistency": direction_consistency,
        "ground_dir": ground_dir,
        "rotation_energy": rotation_energy,
        "rotation_support": rotation_support,
    }


def _direction_ok(summary):
    """True when a track has enough motion to judge and keeps one direction.

    Tracks with little ground-relative horizontal motion (a standing athlete,
    a close-up) are not judged.  A reversing track is only rejected when a
    direction-consistent alternative exists (the caller enforces the length
    floor).
    """
    if not summary:
        return True
    if summary.get("frames", 0) < DIRECTION_MIN_FRAMES:
        return True
    if summary.get("ground_abs_dx_rel", 0.0) < DIRECTION_MIN_MOTION:
        return True
    return summary.get("direction_consistency", 0.0) >= DIRECTION_MIN_CONSISTENCY


def _prefer_direction_consistent(pool, min_track):
    """Drop reversing tracks when a comparable one-direction track exists."""
    if not ENABLE_DIRECTION_PRIOR or len(pool) <= 1:
        return pool
    max_frames = max(item[1]["frames"] for item in pool)
    min_len = max(int(min_track),
                  int(round(DIRECTION_LENGTH_FRAC * max_frames)))
    consistent = [item for item in pool
                  if item[1]["frames"] >= min_len and _direction_ok(item[1])]
    return consistent if consistent else pool


def rank_seed_tracks(candidates, fps=30.0, max_gap=15, seed_seconds=2.0,
                     max_seeds=12, frame_width=None, frame_height=None,
                     cuts=None, stats=None, center_prior=True,
                     camera_shifts=None):
    """Return ``[(path, summary), ...]`` for every person seed in a shot.

    Seeds are sampled throughout the clip/shot, not only at its beginning: the
    skier can enter late, especially in a broadcast clip that starts with a
    crowd or an intro.  ``seed_seconds`` is the sampling stride; every sampled
    frame contributes any person not already represented by an earlier seed.
    """
    first = next((t for t, c in enumerate(candidates) if c), None)
    if first is None:
        return []

    seeds = []
    # everyone visible in the first detected frame
    for i, cand in enumerate(candidates[first]):
        if len(seeds) >= max_seeds:
            break
        seeds.append((first, i, cand))

    # Seed every new contiguous candidate cluster.  A long detection gap
    # (e.g. the skier is lost during the approach and re-acquired in the air)
    # must not be hidden by the "same person" seed deduplication: the first
    # cluster can be a 2-frame false positive while the real 60-frame track
    # starts after the gap.  One seed per cluster is enough; the periodic
    # sampling below adds more.
    last_seen = first
    for t in range(first + 1, len(candidates)):
        if not candidates[t]:
            continue
        if t - last_seen > max_gap + 1:
            best = max(range(len(candidates[t])),
                       key=lambda i: (candidates[t][i].conf,
                                      candidates[t][i].size))
            if len(seeds) < max_seeds:
                seeds.append((t, best, candidates[t][best]))
        last_seen = t

    # then sample the rest of the shot/clip at a fixed time stride, adding
    # anyone new.  This lets a late-entering skier become a seed.  Deduplicate
    # only against seeds close in time: the same image position 2 s later is
    # often a different phase of the jump (or a different person).
    stride = max(1, int(round(seed_seconds * fps)))
    for t in range(first + stride, len(candidates), stride):
        if len(seeds) >= max_seeds:
            break
        for i, cand in enumerate(candidates[t]):
            if any(abs(t - s[0]) <= stride
                   and _same_person(cand, s[2], rel_dist=1.5)
                   for s in seeds):
                continue
            seeds.append((t, i, cand))
            if len(seeds) >= max_seeds:
                break

    out = []
    for t0, i0, _ in seeds:
        path = viterbi_track(candidates, entry_bbox=None, max_gap=max_gap,
                             seed=(t0, i0), frame_width=frame_width,
                             frame_height=frame_height, cuts=cuts, stats=stats,
                             center_prior=center_prior)
        if not path:
            continue
        summary = _path_summary(path, camera_shifts=camera_shifts)
        if summary is not None:
            summary["seed"] = (t0, i0)
            keys = sorted(path)
            if frame_width and frame_height:
                penalties = [_center_penalty(path[k], frame_width, frame_height)
                             for k in keys]
                summary["center_penalty_mean"] = float(np.mean(penalties))
                summary["center_penalty_median"] = float(np.median(penalties))
            else:
                summary["center_penalty_mean"] = 0.0
                summary["center_penalty_median"] = 0.0
            out.append((path, summary))
    return out


def choose_person_track(candidates, fps=30.0, max_gap=15, seed_seconds=2.0,
                         max_seeds=12, frame_width=None, frame_height=None,
                         cuts=None, stats=None, select="longest", min_track=5,
                         center_prior=True, smooth=True, return_ranked=False,
                         camera_shifts=None):
    """Grow one track per detected person and keep the best one.

    Broadcast clips can contain spectators, officials or other athletes.  The
    old ``viterbi_track`` started from the entry box, which could be a spectator
    and then drift.  Here every person sampled throughout the clip/shot is a
    seed; each seed produces a full track, and one is selected:

    * ``select="longest"``: most frames among active movers (or all if none);
    * ``select="motion"``: largest ground-relative travel / vertical range,
      allowing relaxed length floors for late-entering skiers;
    * ``select="size"``: largest person, with stability floor.

    Stationary spectators (zero motion on the ground) are separated from moving
    athletes even when the broadcast camera is panning to track the skier.
    """
    ranked = rank_seed_tracks(
        candidates, fps=fps, max_gap=max_gap, seed_seconds=seed_seconds,
        max_seeds=max_seeds, frame_width=frame_width,
        frame_height=frame_height, cuts=cuts, stats=stats,
        center_prior=center_prior, camera_shifts=camera_shifts)
    if not ranked:
        return ({}, []) if return_ranked else {}

    # Separate candidates who are moving relative to the ground from stationary
    # people (spectators, judges, coaches standing on the snow).
    STATIC_GROUND_SPEED_THR = 0.15  # body sizes per frame relative to ground

    active_items = []
    for item in ranked:
        summary = item[1]
        span = max(summary.get("span", 1), 1)
        ground_speed = summary.get("ground_motion", summary["motion"]) / span
        if ground_speed >= STATIC_GROUND_SPEED_THR:
            active_items.append(item)

    # When active movers exist, establish the length floor from active movers,
    # NOT from stationary spectators who stood in the frame for the whole shot.
    benchmark_pool = active_items if active_items else ranked
    max_frames = max(item[1]["frames"] for item in benchmark_pool)
    min_frames_std = max(int(min_track),
                         int(round(PERSON_STABILITY_FRAC * max_frames)))

    if select == "motion":
        # Candidate must either meet the standard stability floor or, if they are
        # a late-entering skier with significant ground motion (ballistic jump),
        # meet a relaxed floor (0.50 of the active mover's span).
        min_frames_relaxed = max(int(min_track), int(round(0.50 * max_frames)))
        pool = []
        for item in ranked:
            n_f = item[1]["frames"]
            g_mot = item[1].get("ground_motion", item[1]["motion"])
            if n_f >= min_frames_std:
                pool.append(item)
            elif n_f >= min_frames_relaxed and (g_mot / max(n_f, 1)) >= 0.25:
                pool.append(item)

        key = lambda item: (item[1].get("ground_vy_range_rel", item[1]["vy_range_rel"]),
                            -item[1].get("center_penalty_mean", 0.0),
                            item[1].get("ground_motion", item[1]["motion"]),
                            item[1]["frames"], item[1]["median_size"])
    elif select == "event":
        min_frames_event = max(int(min_track),
                               int(round(EVENT_LENGTH_FRAC * max_frames)))
        pool = [item for item in ranked
                if item[1]["frames"] >= min_frames_event]
        key = lambda item: (
            EVENT_ROTATION_WEIGHT * item[1].get("rotation_energy", 0.0),
            EVENT_VERTICAL_WEIGHT * item[1].get(
                "ground_vy_range_rel", item[1]["vy_range_rel"]),
            item[1]["frames"],
            -item[1].get("center_penalty_mean", 0.0))
    elif select == "size":
        pool = [item for item in benchmark_pool if item[1]["frames"] >= min_frames_std]
        key = lambda item: (item[1]["median_size"],
                            -item[1].get("center_penalty_mean", 0.0),
                            item[1]["frames"], item[1]["motion"])
    else:
        # "longest" selector prefers active movers over static spectators if available
        pool = benchmark_pool
        key = lambda item: (item[1]["frames"], item[1]["span"],
                            -item[1].get("center_penalty_mean", 0.0),
                            item[1].get("ground_motion", item[1]["motion"]),
                            item[1]["median_size"])
    if not pool:
        pool = ranked
        key = lambda item: (item[1]["frames"], item[1]["span"],
                            -item[1].get("center_penalty_mean", 0.0),
                            item[1]["motion"], item[1]["median_size"])
    # Within a shot the skier keeps one ground-relative direction.  A track
    # that reverses is usually a spectator or an identity switch; replace it
    # only when a comparable one-direction track exists.
    pool = _prefer_direction_consistent(pool, min_track)
    best_path, best_summary = max(pool, key=key)

    # Second pass: smooth the chosen trajectory and re-run the DP with that
    # prediction as a prior.  This enforces smooth motion in space and time,
    # not just a bounded one-frame displacement.
    if smooth and frame_width and frame_height:
        prior = smooth_path(best_path, len(candidates), frame_width,
                            frame_height, cuts)
        refined = viterbi_track(
            candidates, entry_bbox=None, max_gap=max_gap,
            seed=best_summary["seed"], frame_width=frame_width,
            frame_height=frame_height, cuts=cuts, stats=stats,
            prior=prior, center_prior=center_prior)
        if refined and len(refined) >= SMOOTH_REFINE_MIN * len(best_path):
            best_path = refined
    return (best_path, ranked) if return_ranked else best_path


# Backward-compatible alias; the selector used to be longest-only.
choose_longest_track = choose_person_track


# --------------------------------------------------------------------------- #
# cross-shot identity helpers (Fix 5)
# --------------------------------------------------------------------------- #
# After picking the best person per shot independently, we verify that the
# selections across shots are likely the same athlete.  Two signals are used:
#
#   1. Centre penalty — the camera operator keeps the skier central; a path with
#      a high mean centre penalty while other shots have low penalties is a red
#      flag.  We only flag a shot when the *reference* shot is well-centred
#      (penalty < REF threshold) and the candidate exceeds the BAD threshold,
#      so wide shots where the skier drifts off-centre are not penalised.
#
#   2. Appearance — when ENABLE_APPEARANCE=True, the Lab colour histograms of
#      the selected paths are compared via histogram-intersection distance.  A
#      large distance (> APP threshold) suggests a different person.
#
# When a shot is flagged, its alternative seed tracks (from rank_seed_tracks)
# are tested in order; the first compatible alternative replaces the primary
# selection.  If no alternative passes, the primary selection is kept and the
# warning is recorded in the meta.

_CROSS_SHOT_CENTER_REF_THR = 0.15   # ref must be this well-centred to trigger check
_CROSS_SHOT_CENTER_BAD_THR = 0.25   # candidate flagged above this penalty (outer 25% of frame)
_CROSS_SHOT_APP_THR        = 0.45   # histogram-intersection distance threshold
_CROSS_SHOT_MIN_FRAMES     = 5      # skip shots with fewer tracked frames


def _shot_descriptor(path, frame_width=None, frame_height=None):
    """Compact identity descriptor for a tracked path.

    Returns a dict with centre-penalty statistics and (optionally) an averaged
    appearance histogram.  Returns ``None`` for an empty path.
    """
    if not path:
        return None
    cands = [path[k] for k in sorted(path)]
    if frame_width and frame_height:
        penalties = [_center_penalty(c, frame_width, frame_height) for c in cands]
        cp_mean   = float(np.mean(penalties))
        cp_median = float(np.median(penalties))
    else:
        cp_mean = cp_median = 0.0
    # Average appearance histograms if they were computed (ENABLE_APPEARANCE=True).
    app_descs = [c.appearance for c in cands if c.appearance is not None]
    app_mean  = None
    if app_descs:
        stacked = np.stack(app_descs, axis=0).astype(np.float32)
        app_mean = stacked.mean(axis=0)
        s = float(app_mean.sum())
        if s > 0:
            app_mean /= s
    return {
        "cp_mean":   cp_mean,
        "cp_median": cp_median,
        "app_mean":  app_mean,
        "n_frames":  len(cands),
    }


def _shots_compatible(ref_desc, cand_desc):
    """Return True when ``cand_desc`` is likely the same athlete as ``ref_desc``.

    The check is intentionally lenient: a shot is only flagged when both a
    clear centre-penalty discrepancy *and* (if appearance is available) a large
    colour distance point to a different identity.  Erring on the side of
    keeping the primary selection avoids spurious re-selections in edge cases.
    """
    if ref_desc is None or cand_desc is None:
        return True  # cannot say they differ — be conservative
    # Centre check
    centre_bad = (ref_desc["cp_mean"] < _CROSS_SHOT_CENTER_REF_THR
                  and cand_desc["cp_mean"] > _CROSS_SHOT_CENTER_BAD_THR)
    if not centre_bad:
        return True  # centre is fine — compatible
    # Centre looks suspicious; confirm with appearance if available.
    r_app = ref_desc.get("app_mean")
    c_app = cand_desc.get("app_mean")
    if r_app is not None and c_app is not None:
        inter = float(np.minimum(r_app, c_app).sum())
        if 1.0 - inter > _CROSS_SHOT_APP_THR:
            return False   # both centre AND colour disagree → flag
        return True        # colour agrees despite centre offset → keep
    # Appearance unavailable: flag on centre penalty alone.
    return False


def _path_pose_fracs(path):
    """Fraction of path frames with a full body and with lower-body joints."""
    if not path:
        return 0.0, 0.0
    body8, lower = [], []
    for cand in path.values():
        kp = cand.keypoints
        if kp is None or len(kp) < 17:
            body8.append(False)
            lower.append(False)
            continue
        visible = kp[:17, 2] >= 0.3
        body8.append(int(visible.sum()) >= 8)
        lower.append(int(visible[11:17].sum()) >= 1)
    return float(np.mean(body8)), float(np.mean(lower))


def _skier_score(summary, body8_frac, lower_body_frac, crowd_frac,
                 frame_height):
    """Heuristic 0-1 score that a tracked person is the athlete, not a spectator.

    It combines jump evidence (shoulder rotation, vertical excursion), pose
    completeness, one-direction travel, central framing, subject size and a
    penalty for shots that are full of people.  It is a diagnostic/selection
    score, not a trained classifier.
    """
    rotation = float(summary.get("rotation_energy", 0.0))
    vy = float(summary.get("ground_vy_range_rel",
                           summary.get("vy_range_rel", 0.0)))
    motion = min(1.0, rotation / 5.0 + vy / 4.0)
    pose = 0.6 * body8_frac + 0.4 * lower_body_frac
    if summary.get("ground_abs_dx_rel", 0.0) > 1.0:
        direction = float(summary.get("direction_consistency", 0.0))
    else:
        direction = 0.5  # standing/close-up: no evidence either way
    center = 1.0 - min(1.0, float(summary.get("center_penalty_mean", 0.0))
                       / 0.5)
    size = min(1.0, float(summary.get("median_size", 0.0))
               / max(0.15 * frame_height, 1.0))
    crowd = 1.0 - min(1.0, crowd_frac)
    return float(np.clip(
        0.35 * motion + 0.25 * pose + 0.15 * direction
        + 0.10 * center + 0.10 * size + 0.05 * crowd, 0.0, 1.0))


def _shot_info_entry(a, b, path_rel, candidates, frame_width, frame_height,
                     camera_shifts=None, cross_shot_check=True,
                     was_warned=False):
    """Build one shot record (absolute path + diagnostics) for a tracked shot."""
    if not path_rel:
        return {}, None
    path = {t + a: cand for t, cand in path_rel.items()}
    keys = sorted(path)
    centers = np.asarray([path[k].center for k in keys], dtype=np.float32)
    sizes = np.asarray([path[k].size for k in keys], dtype=np.float32)
    median_size = float(np.median(sizes)) if sizes.size else 1.0
    vy_motion = (float(np.sum(np.abs(np.diff(centers[:, 1]))))
                 / max(median_size, 1.0)) if len(keys) > 1 else 0.0
    y_range = (float(centers[:, 1].max() - centers[:, 1].min())
               if len(keys) else 0.0)
    x_range = (float(centers[:, 0].max() - centers[:, 0].min())
               if len(keys) else 0.0)
    penalties = [_center_penalty(path[k], frame_width, frame_height)
                 for k in keys]
    shot_shifts = None
    if camera_shifts is not None and len(camera_shifts) >= b - 1:
        shot_shifts = camera_shifts[a:b - 1]
    dir_summary = _path_summary(path_rel, camera_shifts=shot_shifts) or {}
    body8_frac, lower_body_frac = _path_pose_fracs(path)
    shot_cands = [c for c in candidates[a:b] if c]
    crowd_frac = (min(1.0, float(np.median([len(c) for c in shot_cands]))
                      / 6.0) if shot_cands else 0.0)
    skier_score = _skier_score(dir_summary, body8_frac, lower_body_frac,
                               crowd_frac, frame_height or 1)
    record = {
        "start": int(a), "end": int(b),
        "frames": int(len(path)), "span": int(max(keys) - min(keys) + 1),
        "median_size": median_size, "vy_motion": vy_motion,
        "y_range": y_range, "x_range": x_range,
        "vy_range_rel": y_range / max(median_size, 1.0),
        "ground_vy_range_rel": float(
            dir_summary.get("ground_vy_range_rel",
                            y_range / max(median_size, 1.0))),
        "rotation_energy": float(
            dir_summary.get("rotation_energy", 0.0)),
        "rotation_support": float(
            dir_summary.get("rotation_support", 0.0)),
        "ground_dx_rel": float(dir_summary.get("ground_dx_rel", 0.0)),
        "ground_abs_dx_rel": float(dir_summary.get("ground_abs_dx_rel", 0.0)),
        "direction_consistency": float(
            dir_summary.get("direction_consistency", 0.0)),
        "ground_dir": int(dir_summary.get("ground_dir", 0)),
        "direction_ok": bool(_direction_ok(dir_summary)),
        "body8_frac": float(body8_frac),
        "lower_body_frac": float(lower_body_frac),
        "crowd_frac": float(crowd_frac),
        "skier_score": float(skier_score),
        "skier_likely": bool(skier_score >= SKIER_SCORE_MIN),
        "center_penalty_mean": (float(np.mean(penalties))
                                if penalties else 0.0),
        "center_penalty_median": (float(np.median(penalties))
                                  if penalties else 0.0),
        "identity_checked": bool(cross_shot_check),
        "identity_warning": bool(was_warned),
    }
    return path, record


def _recover_shot_tail(candidates, a, b, path_rel, fps, max_gap,
                       frame_width=None, frame_height=None, stats=None,
                       center_prior=True, smooth=True, camera_shifts=None):
    """Track a second, confident person segment in a shot's uncovered tail.

    Returns a shot-relative ``{t: Candidate}`` for the tail segment, or ``{}``
    when the tail is too short, too weak or has no usable pose.  See
    ``TAIL_RECOVERY_*`` for the gates.
    """
    if not path_rel:
        return {}
    t0 = max(path_rel) + 1                    # first uncovered frame in shot
    if b - (a + t0) < TAIL_RECOVERY_MIN_FRAMES:
        return {}
    sub_shifts = None
    if camera_shifts is not None:
        lo, hi = a + t0, b - 1
        if hi - lo > 1 and len(camera_shifts) >= hi:
            sub_shifts = camera_shifts[lo:hi]
    tail_rel = choose_person_track(
        candidates[a + t0:b], fps=fps, max_gap=max_gap, select="longest",
        min_track=TAIL_RECOVERY_MIN_FRAMES, frame_width=frame_width,
        frame_height=frame_height, stats=stats, center_prior=center_prior,
        smooth=smooth, camera_shifts=sub_shifts)
    if not tail_rel or len(tail_rel) < TAIL_RECOVERY_MIN_FRAMES:
        return {}
    confs = np.asarray([c.conf for c in tail_rel.values()], dtype=np.float32)
    if float(np.median(confs)) < TAIL_RECOVERY_MIN_CONF:
        return {}
    body8, _lower = _path_pose_fracs(tail_rel)
    if body8 < TAIL_RECOVERY_MIN_BODY8:
        return {}
    return {t + t0: c for t, c in tail_rel.items()}


def choose_shot_tracks(candidates, cuts, fps=30.0, max_gap=15,
                       shot_select="all", person_select="longest",
                       min_track=5, max_seeds=12,
                       frame_width=None, frame_height=None, stats=None,
                       center_prior=True, smooth=True, cross_shot_check=True,
                       camera_shifts=None):
    """Track one person in every camera shot and return their union.

    A hard camera cut is the only place where a box may jump: the broadcast
    changes viewpoint, scale and often the image position of the same skier.
    The clip is therefore split at cuts and :func:`choose_person_track` runs
    independently inside each shot.  Unlike the old behaviour, **no shot is
    discarded**: every shot with a plausible person contributes to the output.
    This keeps the whole clip instead of selecting a "best" temporal window.

    ``shot_select`` is accepted for backward compatibility but ignored; the
    output always contains all shots.  ``person_select`` chooses the person
    inside each shot (see :func:`choose_person_track`).  ``frame_width`` /
    ``frame_height`` enable the physical gate and the central prior, and
    ``stats`` collects gate diagnostics.

    When ``cross_shot_check=True`` (default), the per-shot selections are
    compared against the reference shot (the one with the most tracked frames).
    A shot whose selected person has a high centre penalty and (if
    ``ENABLE_APPEARANCE=True``) a large colour distance from the reference is
    re-tried against its next-best seed tracks.  Warnings are recorded in
    ``stats["identity_warnings"]`` and in the per-shot info.
    """
    clean = sorted({int(c) for c in cuts if 0 < int(c) < len(candidates)})
    bounds = [0] + clean + [len(candidates)]

    # Phase 1: pick the best person in every shot; keep the ranked alternatives
    # so the cross-shot identity check can try re-selecting when needed.
    shot_data = []   # list of (a, b, path_rel, ranked_all)
    tails = {}       # (a, b) -> absolute [start, end) of a recovered tail
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        shot_shifts = None
        if camera_shifts is not None and len(camera_shifts) >= b - 1:
            shot_shifts = camera_shifts[a:b - 1]
        path_rel, ranked = choose_person_track(
            candidates[a:b], fps=fps, max_gap=max_gap,
            max_seeds=max_seeds, select=person_select,
            min_track=min_track, frame_width=frame_width,
            frame_height=frame_height, stats=stats,
            center_prior=center_prior, smooth=smooth, return_ranked=True,
            camera_shifts=shot_shifts)
        # The athlete can reappear later in the same shot (the landing after the
        # jump).  That segment is not reachable by the gate, so it is tracked as
        # its own path instead of being thrown away with the rest of the shot.
        tail_rel = _recover_shot_tail(
            candidates, a, b, path_rel, fps, max_gap,
            frame_width=frame_width, frame_height=frame_height, stats=stats,
            center_prior=center_prior, smooth=smooth,
            camera_shifts=camera_shifts)
        if tail_rel:
            tails[(a, b)] = (int(a + min(tail_rel)), int(b))
            path_rel = {**path_rel, **tail_rel}
        shot_data.append((a, b, path_rel, ranked))

    # Phase 2: cross-shot identity check.
    # Reference = shot with the most tracked frames (usually the jump shot).
    # Any shot whose descriptor is incompatible with the reference is re-tried
    # against its alternative seed tracks (first-pass paths, no smooth refinement).
    identity_warnings = []
    if cross_shot_check and len(shot_data) > 1:
        ref_i = max(range(len(shot_data)),
                    key=lambda i: len(shot_data[i][2]))
        ref_desc = _shot_descriptor(shot_data[ref_i][2], frame_width, frame_height)
        for i, (a, b, path_rel, ranked) in enumerate(shot_data):
            if i == ref_i or len(path_rel) < _CROSS_SHOT_MIN_FRAMES:
                continue
            desc = _shot_descriptor(path_rel, frame_width, frame_height)
            if _shots_compatible(ref_desc, desc):
                continue
            # Try alternatives in rank order.
            reselected = False
            for alt_path_rel, _alt_sum in ranked[1:]:
                if len(alt_path_rel) < _CROSS_SHOT_MIN_FRAMES:
                    continue
                alt_desc = _shot_descriptor(alt_path_rel, frame_width, frame_height)
                if _shots_compatible(ref_desc, alt_desc):
                    shot_data[i] = (a, b, alt_path_rel, ranked)
                    reselected = True
                    break
            warning = {
                "shot_index": i,
                "frame_range": [int(a), int(b)],
                "cp_mean": desc["cp_mean"] if desc else None,
                "ref_cp_mean": ref_desc["cp_mean"] if ref_desc else None,
                "reselected": reselected,
            }
            identity_warnings.append(warning)
        if identity_warnings and stats is not None:
            stats["identity_warnings"] = identity_warnings

    # Phase 3: combine shots into one path and build shot-info records.
    combined = {}
    shots = []
    for a, b, path_rel, _ranked in shot_data:
        idx_in_data = len(shots)
        was_warned = any(w["shot_index"] == idx_in_data
                         for w in identity_warnings)
        path, record = _shot_info_entry(
            a, b, path_rel, candidates, frame_width, frame_height,
            camera_shifts=camera_shifts, cross_shot_check=cross_shot_check,
            was_warned=was_warned)
        if record is None:
            continue
        if (a, b) in tails:
            record["tail_recovered"] = [int(tails[(a, b)][0]),
                                        int(tails[(a, b)][1])]
        combined.update(path)
        shots.append(record)
    info = []
    for s in shots:
        d = dict(s)
        d["selected"] = True  # every shot is kept; kept for old readers
        info.append(d)
    return combined, info


# --------------------------------------------------------------------------- #
# gap filling
# --------------------------------------------------------------------------- #
def interpolate_boxes(path, n_frames, max_gap=15, cuts=None):
    """Fill short gaps between tracked boxes with linear interpolation.

    A gap that spans a camera cut is left empty: interpolating between two
    camera views would invent a box that exists in neither view.
    """
    if not path:
        return {}
    boxes = {t: np.asarray(c.bbox, dtype=np.float32) for t, c in path.items()}
    known = sorted(boxes)
    filled = dict(boxes)
    for t1, t2 in zip(known[:-1], known[1:]):
        gap = t2 - t1 - 1
        if gap <= 0 or gap > max_gap:
            continue
        if _crosses_cut(t1, t2, cuts):
            continue
        b1, b2 = boxes[t1], boxes[t2]
        for t in range(t1 + 1, t2):
            alpha = (t - t1) / float(t2 - t1)
            filled[t] = (1 - alpha) * b1 + alpha * b2
    return filled


def interpolate_keypoints(seq, max_gap=15, min_conf=0.2, cuts=None):
    """Fill per-joint gaps by linear interpolation over short misses.

    Gaps across a camera cut are not filled, for the same reason as
    :func:`interpolate_boxes`.
    """
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
            if _crosses_cut(a, b, cuts):
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


def estimate_camera_shifts(small_frames, frame_width=None, frame_height=None):
    """Estimate inter-frame camera background shifts using phase correlation.

    Returns an array of shape (N - 1, 2) giving (dx, dy) background shifts
    in full-frame pixel coordinates.  A positive shift means the ground features
    moved in that direction in the image (e.g. camera panned in the opposite
    direction).  Used to separate ground-relative motion from camera panning.
    """
    if not small_frames or len(small_frames) < 2:
        return np.zeros((0, 2), dtype=np.float32)
    n = len(small_frames)
    sh_h, sh_w = small_frames[0].shape[:2]
    scale_x = (float(frame_width) / float(sh_w)) if frame_width and sh_w else 1.0
    scale_y = (float(frame_height) / float(sh_h)) if frame_height and sh_h else 1.0

    shifts = np.zeros((n - 1, 2), dtype=np.float32)
    for t in range(n - 1):
        f0 = small_frames[t].astype(np.float64)
        f1 = small_frames[t + 1].astype(np.float64)
        shift, resp = cv2.phaseCorrelate(f0, f1)
        if resp >= 0.2:  # confident background translation
            shifts[t, 0] = float(shift[0]) * scale_x
            shifts[t, 1] = float(shift[1]) * scale_y
    return shifts


def _texture_match_ratio(matcher, prev_desc, desc, min_features):
    """Fraction of ORB descriptors of two frames that match each other.

    Returns ``None`` when either frame has too few features to judge, so a
    blurred or underexposed frame never counts as evidence of a cut.
    """
    if prev_desc is None or desc is None:
        return None
    if len(prev_desc) < min_features or len(desc) < min_features:
        return None
    try:
        matches = matcher.knnMatch(prev_desc, desc, k=2)
    except cv2.error:
        return None
    good = [m for m, n in (p for p in matches if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    return len(good) / float(max(len(prev_desc), len(desc)))


def detect_cuts(frames, diff_thr=35.0, corr_thr=0.5, min_gap=8,
                black_thr=20.0, texture_thr=CUT_TEXTURE_MATCH_MIN,
                texture_diff_floor=CUT_TEXTURE_MIN_DIFF):
    """Return frame indices where a hard camera cut occurs.

    Two independent tests are used, because broadcast competition footage
    defeats either one alone:

    * **brightness**: a large mean-absolute difference *and* a large drop in
      brightness-histogram correlation.  This catches the obvious cuts.
    * **texture**: almost no ORB feature matches survive a camera change, while
      a fast pan or zoom keeps most of them.  This catches the cuts between two
      visually similar wide shots (snow against snow), where the brightness
      histograms barely move.  It only runs when the frame pair already differs
      by ``texture_diff_floor``, and it abstains when either frame has too few
      features (motion blur, night sky).

    Transitions to or from a near-black frame are ignored: many AQA clips open
    with a black leader and that is not a camera change.
    """
    cuts = []
    prev = None
    prev_mean = None
    orb = matcher = None
    prev_desc = None
    if texture_thr:
        orb = cv2.ORB_create(nfeatures=500)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    for i, frame in enumerate(frames):
        mid = cv2.resize(frame, CUT_TEXTURE_SIZE)
        small = cv2.resize(mid, (160, 90))
        desc = (orb.detectAndCompute(mid, None)[1]
                if orb is not None else None)
        mean = float(small.mean())
        if prev is not None:
            diff = float(np.mean(np.abs(small.astype(np.float32)
                                        - prev.astype(np.float32))))
            h1 = cv2.calcHist([prev], [0], None, [32], [0, 256])
            h2 = cv2.calcHist([small], [0], None, [32], [0, 256])
            cv2.normalize(h1, h1)
            cv2.normalize(h2, h2)
            corr = float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
            allowed = (prev_mean > black_thr and mean > black_thr
                       and (not cuts or i - cuts[-1] >= min_gap))
            is_cut = diff > diff_thr and corr < corr_thr
            if (not is_cut and allowed and texture_thr
                    and diff > texture_diff_floor):
                ratio = _texture_match_ratio(matcher, prev_desc, desc,
                                             CUT_TEXTURE_MIN_FEATURES)
                is_cut = ratio is not None and ratio < texture_thr
            if is_cut and allowed:
                cuts.append(i)
        prev = small
        prev_mean = mean
        prev_desc = desc
    return cuts


def process_video_pose_temporal(model, video_path, entry_idx, entry_bbox,
                                max_gap=15, conf_thres=0.1, bbox_pad=0.25,
                                min_visible=10, track_mode="entry",
                                cut_aware=True, cut_diff_thr=35.0,
                                cut_corr_thr=0.5, cut_min_gap=8,
                                shot_select="all", person_select="longest",
                                center_prior=True, smooth=True):
    """Return ``(pose_seq, box_seq, meta)`` for a whole video.

    pose_seq: [T, 17, 3] absolute frame coordinates, zeros where no pose.
    box_seq : [T, 4] padded body boxes, NaN where no track.

    ``track_mode="entry"`` keeps the original behaviour (one Viterbi path seeded
    by the entry box).  ``track_mode="longest"`` builds a track per person
    sampled throughout the clip and keeps the one present for the most frames,
    which avoids locking onto a spectator.

    ``cut_aware`` detects hard camera cuts.  A cut is the only place where a
    box may jump (the broadcast changes viewpoint, scale and image position),
    so the clip is tracked independently inside each shot and **all** shots are
    written to the output.  There is no primary-shot selection: the whole clip
    is kept.  Inside a shot the motion must be smooth in space and time, and a
    central-square prior favours the person the cameraman keeps in frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Pass 1: collect all candidates from the entry frame onward, and detect
    # hard camera cuts on a small grayscale frame.
    candidates = []
    small_frames = []
    cut_frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, entry_idx)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if not width or not height:
            height, width = frame.shape[:2]
        candidates.append(collect_candidates(model, frame, conf_thres=conf_thres,
                                             bbox_pad=bbox_pad))
        if cut_aware:
            small_frames.append(cv2.cvtColor(cv2.resize(frame, (160, 90)),
                                             cv2.COLOR_BGR2GRAY))
            # Cuts are detected on a larger grayscale frame: the texture test
            # needs enough detail to tell a cut from a fast pan.
            cut_frames.append(cv2.cvtColor(
                cv2.resize(frame, CUT_TEXTURE_SIZE), cv2.COLOR_BGR2GRAY))
    cap.release()

    cuts = (detect_cuts(cut_frames, diff_thr=cut_diff_thr,
                        corr_thr=cut_corr_thr, min_gap=cut_min_gap)
            if cut_aware else [])

    camera_shifts = (estimate_camera_shifts(small_frames, width, height)
                     if cut_aware and small_frames else None)

    shot_info = []
    gate_stats = {}
    if track_mode == "longest" and cut_aware and cuts:
        path, shot_info = choose_shot_tracks(
            candidates, cuts, fps=fps, max_gap=max_gap,
            shot_select=shot_select, person_select=person_select,
            frame_width=width, frame_height=height, stats=gate_stats,
            center_prior=center_prior, smooth=smooth,
            camera_shifts=camera_shifts)
    elif track_mode == "longest":
        path = choose_person_track(candidates, fps=fps, max_gap=max_gap,
                                    select=person_select,
                                    frame_width=width, frame_height=height,
                                    stats=gate_stats,
                                    center_prior=center_prior, smooth=smooth,
                                    camera_shifts=camera_shifts)
        # Single-shot clips still get one shot record so the skier score and
        # the rotation/vertical diagnostics are available everywhere.
        if path:
            _p, rec = _shot_info_entry(
                0, len(candidates), path, candidates, width, height,
                camera_shifts=camera_shifts, cross_shot_check=False)
            if rec is not None:
                shot_info = [rec]
    else:
        path = viterbi_track(candidates, entry_bbox=entry_bbox,
                             max_gap=max_gap, frame_width=width,
                             frame_height=height,
                             cuts=cuts if cut_aware else None,
                             stats=gate_stats, center_prior=center_prior)
    boxes_rel = interpolate_boxes(path, len(candidates), max_gap=max_gap,
                                  cuts=cuts if cut_aware else None)

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

    pose_rel = interpolate_keypoints(pose_rel, max_gap=max_gap,
                                     cuts=cuts if cut_aware else None)

    pose_full = np.zeros((total, 17, 3), dtype=np.float32)
    box_full = np.full((total, 4), np.nan, dtype=np.float32)
    end = min(total, entry_idx + len(pose_rel))
    pose_full[entry_idx:end] = pose_rel[:end - entry_idx]
    box_full[entry_idx:end] = box_rel[:end - entry_idx]

    # Fill a missing box from the keypoints so box and pose stay aligned even
    # when the tracker had a long gap.  The pose is the more reliable signal
    # once it has been refined inside the ROI.
    box_from_pose = 0
    for t in range(total):
        if np.all(np.isfinite(box_full[t])):
            continue
        kp = pose_full[t]
        vis = kp[:, 2] >= 0.3
        if int(vis.sum()) < 3:
            continue
        xy = kp[vis, :2]
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        pad_x = 0.10 * (x1 - x0 + 1.0)
        pad_y = 0.05 * (y1 - y0 + 1.0)
        box_full[t] = [x0 - pad_x, y0 - pad_y,
                       (x1 - x0) + 2 * pad_x, (y1 - y0) + 2 * pad_y]
        box_from_pose += 1

    # Box/pose alignment: a large offset means the tracked box and the refined
    # skeleton describe different regions (partial body, static object, drift).
    offsets = []
    for t in range(total):
        b = box_full[t]
        if not np.all(np.isfinite(b)) or b[3] <= 1:
            continue
        kp = pose_full[t]
        vis = kp[:, 2] >= 0.3
        if int(vis.sum()) < 3:
            continue
        xy = kp[vis, :2]
        pose_center = (xy.min(axis=0) + xy.max(axis=0)) / 2.0
        box_center = b[:2] + b[2:] / 2.0
        offsets.append(float(np.linalg.norm(pose_center - box_center) / b[3]))
    offsets = np.asarray(offsets, dtype=np.float32)
    box_pose_offset_median = (float(np.median(offsets))
                              if offsets.size else None)
    box_pose_offset_frac_gt_0_5 = (float((offsets > 0.5).mean())
                                   if offsets.size else None)

    # No primary-shot selection: the output keeps every shot.  Coverage is
    # therefore measured over the whole tracked range, not over a chosen window.
    primary_span = int(len(candidates))
    tracked_with_pose = int((pose_full[:, :, 2].max(axis=1) > 0).sum())
    pose_conf = pose_full[:, :17, 2]
    body_joints = (pose_conf >= 0.3).sum(axis=1)
    lower_joints = (pose_conf[:, 11:17] >= 0.3).sum(axis=1)
    valid_frames = pose_full[:, :, 2].max(axis=1) > 0
    if valid_frames.any():
        body8_frac = float((body_joints[valid_frames] >= 8).mean())
        lower_body_frac = float((lower_joints[valid_frames] >= 1).mean())
    else:
        body8_frac = 0.0
        lower_body_frac = 0.0
    pose_quality_mean = float(np.clip(
        0.6 * body8_frac + 0.4 * lower_body_frac, 0.0, 1.0))

    meta = {
        "entry_frame": int(entry_idx),
        "track_mode": track_mode,
        "track_frames": int(len(path)),
        "track_span": (int(max(path) - min(path) + 1) if path else 0),
        "tracked_with_pose": tracked_with_pose,
        "primary_span": primary_span,
        "pose_coverage": (float(tracked_with_pose) / max(primary_span, 1)),
        "body8_frac": body8_frac,
        "lower_body_frac": lower_body_frac,
        "pose_quality_mean": pose_quality_mean,
        "pose_quality_enabled": bool(ENABLE_POSE_QUALITY),
        "pose_quality_weight": float(POSE_QUALITY_WEIGHT),
        "box_from_pose_frames": int(box_from_pose),
        "box_pose_offset_median": box_pose_offset_median,
        "box_pose_offset_frac_gt_0_5": box_pose_offset_frac_gt_0_5,
        "skier_score_best": (max(float(s.get("skier_score", 0.0))
                                 for s in shot_info) if shot_info else None),
        "skier_shot_frac": (
            sum(1 for s in shot_info if s.get("skier_likely"))
            / len(shot_info)) if shot_info else None,
        "skier_frame_frac": (
            sum(int(s.get("frames", 0)) for s in shot_info
                if s.get("skier_likely"))
            / max(sum(int(s.get("frames", 0)) for s in shot_info), 1))
        if shot_info else None,
        "skier_score_min": float(SKIER_SCORE_MIN),
        "n_shots": int(len(shot_info)) if shot_info else 1,
        "n_camera_shots": int(len(cuts) + 1) if cuts else 1,
        "cut_frames": [int(c + entry_idx) for c in cuts],
        "shots": shot_info,
        "shot_mode": "all",
        "primary_shot": ("all" if shot_info
                         else "0-" + str(len(candidates))),
        "fps": fps,
        "width": width,
        "height": height,
        "person_select": person_select,
        "person_stability_frac": PERSON_STABILITY_FRAC,
        "direction_min_consistency": DIRECTION_MIN_CONSISTENCY,
        "direction_min_motion": DIRECTION_MIN_MOTION,
        "direction_length_frac": DIRECTION_LENGTH_FRAC,
        "center_prior": bool(center_prior),
        "center_prior_weight": CENTER_PRIOR_WEIGHT,
        "center_square_radius": CENTER_SQUARE_RADIUS,
        "smooth": bool(smooth),
        "smooth_window": SMOOTH_WINDOW,
        "smooth_polyorder": SMOOTH_POLYORDER,
        "prior_weight": PRIOR_WEIGHT,
        "appearance_enabled": bool(ENABLE_APPEARANCE),
        "appearance_weight": float(APPEARANCE_WEIGHT),
        "appearance_max_gap": int(APPEARANCE_MAX_GAP),
        "tiling_enabled": bool(ENABLE_TILING),
        "tiling_trigger_size": int(TILE_TRIGGER_SIZE),
        "camera_motion_compensated": bool(camera_shifts is not None and len(camera_shifts) > 0),
        "identity_warnings": gate_stats.get("identity_warnings", []),
        "gate_checks": int(gate_stats.get("gate_checks", 0)),
        "gate_rejects": int(gate_stats.get("gate_rejects", 0)),
        "gate_splits": int(gate_stats.get("gate_splits", 0)),
        "gate_reject_frac": float(
            gate_stats.get("gate_rejects", 0)
            / max(gate_stats.get("gate_checks", 0), 1)),
        "gate_config": {
            "enabled": ENABLE_GATE,
            "max_gate_gap": MAX_GATE_GAP,
            "max_step_body": MAX_STEP_BODY,
            "max_step_frame": MAX_STEP_FRAME,
            "max_step_frame_cap": MAX_STEP_FRAME_CAP,
            "max_gap_frame_cap": MAX_GAP_FRAME_CAP,
            "max_size_log_step": MAX_SIZE_LOG_STEP,
            "max_size_log_cap": MAX_SIZE_LOG_CAP,
        },
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
                                 max_gap=15, pose_conf=0.1, bbox_pad=0.25,
                                 imgsz=640, track_mode="longest",
                                 cut_aware=True, shot_select="all",
                                 person_select="longest",
                                 summary_name="dataset_processed_features.csv"):
    """Drop-in replacement for the notebook's ``process_aqa_dataset``.

    ``imgsz`` is the YOLO inference size.  The AQA clips are small (320x240 /
    640x480) and work at 640, but the broadcast competition clips are 1280x720
    wide shots where the skier is much smaller; 1280 recovers far more frames.

    ``track_mode="longest"`` builds one track per person sampled throughout the
    clip and keeps the track present for the most frames.  Use it when a
    spectator or official can be picked up by the entry scan.

    ``cut_aware`` splits the clip at hard camera cuts and tracks inside each
    shot.  Unlike the old behaviour, **every shot is written**: there is no
    ``shot_select`` window.  The parameter is kept for backward compatibility
    and ignored.  A per-clip ``<clip>_pose_meta.json`` records the cut frames,
    per-shot statistics, the gate diagnostics and the smooth/centre prior
    configuration.
    """
    from ultralytics import YOLO

    output_path = Path(output_dir)
    ski_path = Path(ski_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ski_path.mkdir(parents=True, exist_ok=True)
    model = YOLO(pose_model_path)
    model.overrides["imgsz"] = int(imgsz)

    records = []
    for pos, (_, row) in enumerate(labels_df.iterrows(), 1):
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
        ski_file = ski_path / f"{stem}_ski.npz"

        if pose_file.exists() and not overwrite:
            n = int(np.load(pose_file).shape[0])
            print(f"[{pos}/{len(labels_df)}] {stem}: [SKIP] pose exists ({n} frames)")
            records.append({
                "filename": row["filename"],
                "pose_feature_path": str(pose_file),
                "ski_feature_path": str(ski_file),
                "num_frames": n,
                "target": target,
            })
            continue

        print(f"[{pos}/{len(labels_df)}] Processing: {stem} (Target Score: {target})")
        if track_mode == "longest":
            # Do not trust the entry scan: it can pick a spectator.  Start at
            # frame 0 and let choose_person_track select the person.
            entry_idx, entry_bbox = 0, None
        else:
            entry_idx, entry_bbox = find_entry_frame(video_file, model)
            if entry_bbox is None:
                print(f"  [SKIP] {stem}: no skier detected in entry scan")
                continue
            print(f"  -> skier entry at frame {entry_idx}")

        pose, boxes, meta = process_video_pose_temporal(
            model, video_file, entry_idx, entry_bbox,
            max_gap=max_gap, conf_thres=pose_conf, bbox_pad=bbox_pad,
            track_mode=track_mode, cut_aware=cut_aware,
            shot_select=shot_select, person_select=person_select)
        np.save(pose_file, pose)

        _write_empty_ski(ski_file, len(pose), boxes, meta["fps"],
                         meta["width"], meta["height"])
        meta_file = output_path / f"{stem}_pose_meta.json"
        meta_file.write_text(json.dumps(meta, indent=1), encoding="utf-8")

        print(f"  -> [{meta['track_mode']}] track {meta['track_frames']} frames "
              f"(span {meta['track_span']}, "
              f"{meta.get('n_camera_shots', meta['n_shots'])} camera shot(s), "
              f"all shots kept), "
              f"pose in {meta['tracked_with_pose']} frames "
              f"({100 * meta['pose_coverage']:.0f}% of the clip), "
              f"gate {meta.get('gate_rejects', 0)}/"
              f"{meta.get('gate_checks', 0)}")
        records.append({
            "filename": row["filename"],
            "pose_feature_path": str(pose_file),
            "ski_feature_path": str(ski_file),
            "num_frames": int(len(pose)),
            "track_mode": meta["track_mode"],
            "track_frames": meta["track_frames"],
            "n_shots": meta["n_shots"],
            "n_camera_shots": meta.get("n_camera_shots", meta["n_shots"]),
            "primary_shot": meta["primary_shot"],
            "primary_span": meta["primary_span"],
            "pose_coverage": round(meta["pose_coverage"], 3),
            "gate_checks": meta.get("gate_checks", 0),
            "gate_rejects": meta.get("gate_rejects", 0),
            "gate_reject_frac": round(meta.get("gate_reject_frac", 0.0), 4),
            "gate_splits": meta.get("gate_splits", 0),
            "target": target,
        })

    summary = pd.DataFrame(records)
    if summary_name:
        summary.to_csv(output_path / summary_name, index=False)
        print(f"\nTemporal pipeline finished. Features saved to: "
              f"{output_path / summary_name}")
    return summary


def summarize_cached(labels_df, output_dir, ski_output_dir, data_dir=None):
    """Rebuild the feature summary from cached pose/meta files, no tracking."""
    output_path = Path(output_dir)
    ski_path = Path(ski_output_dir)
    records = []
    for _, row in labels_df.iterrows():
        video = row.get("video_path")
        if video is not None and not pd.isna(video) and Path(str(video)).is_file():
            stem = Path(str(video)).stem
        else:
            resolved = resolve_video(row["filename"], data_dir) if data_dir else None
            stem = Path(str(resolved)).stem if resolved else Path(str(row["filename"])).stem
        pose_file = output_path / f"{stem}_pose.npy"
        ski_file = ski_path / f"{stem}_ski.npz"
        meta_file = output_path / f"{stem}_pose_meta.json"
        meta = {}
        if meta_file.is_file():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
        if pose_file.is_file():
            num_frames = int(np.load(pose_file, mmap_mode="r").shape[0])
        else:
            num_frames = int(meta.get("primary_span", 0) or 0)
        records.append({
            "filename": row["filename"],
            "pose_feature_path": str(pose_file),
            "ski_feature_path": str(ski_file),
            "num_frames": num_frames,
            "track_mode": meta.get("track_mode"),
            "track_frames": meta.get("track_frames", 0),
            "n_shots": meta.get("n_shots", 0),
            "n_camera_shots": meta.get("n_camera_shots", meta.get("n_shots", 0)),
            "primary_shot": meta.get("primary_shot"),
            "primary_span": meta.get("primary_span"),
            "pose_coverage": round(float(meta.get("pose_coverage", 0.0)), 3),
            "gate_checks": meta.get("gate_checks", 0),
            "gate_rejects": meta.get("gate_rejects", 0),
            "gate_reject_frac": round(float(meta.get("gate_reject_frac", 0.0)), 4),
            "gate_splits": meta.get("gate_splits", 0),
            "target": row.get("target", np.nan),
        })
    return pd.DataFrame(records)


def _missing_runs(ok):
    """Return ``[start, end)`` runs where ``ok`` is False."""
    runs = []
    start = None
    for i, good in enumerate(ok):
        if not good and start is None:
            start = i
        elif good and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(ok)))
    return runs


def flag_detection_gaps(labels_df, output_dir, ski_output_dir, data_dir=None,
                        min_coverage=0.60, max_gap_frames=60,
                        warn_coverage=0.85, warn_gap_frames=30):
    """Rank cached clips by how badly the tracker lost the athlete.

    Reads only the cached ``_pose_meta.json`` / ``_ski.npz`` / ``_pose.npy``
    files, so the whole set takes a few seconds and nothing is re-tracked.
    ``severity`` is ``fail`` when the clip tracks under ``min_coverage`` of its
    frames or has one missing run of at least ``max_gap_frames``, ``warn`` for
    the softer thresholds, and ``missing`` when the clip was never processed.
    ``gap_shot`` is the camera shot the longest missing run falls in, which is
    the part of the clip worth looking at.
    """
    output_path = Path(output_dir)
    ski_path = Path(ski_output_dir)
    records = []
    for _, row in labels_df.iterrows():
        video = row.get("video_path")
        if video is not None and not pd.isna(video) and Path(str(video)).is_file():
            stem = Path(str(video)).stem
        else:
            resolved = resolve_video(row["filename"], data_dir) if data_dir else None
            stem = Path(str(resolved)).stem if resolved else Path(str(row["filename"])).stem
        meta_file = output_path / f"{stem}_pose_meta.json"
        if not meta_file.is_file():
            records.append({"filename": row["filename"], "severity": "missing",
                            "num_frames": 0, "pose_coverage": 0.0,
                            "track_frames": 0, "longest_gap_frames": 0,
                            "longest_gap_start": -1, "longest_gap_end": -1,
                            "n_gaps": 0, "gap_frac": 0.0, "gap_shot": None,
                            "n_camera_shots": 0})
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        ski_file = ski_path / f"{stem}_ski.npz"
        bbox = None
        if ski_file.is_file():
            try:
                with np.load(ski_file) as data:
                    bbox = np.asarray(data["bbox"], dtype=float)
            except (OSError, ValueError, KeyError):
                bbox = None
        if bbox is not None and len(bbox):
            num_frames = len(bbox)
            ok = np.isfinite(bbox[:, 0]) & (bbox[:, 2] > 0)
            runs = _missing_runs(ok)
            gap_frac = float((~ok).mean())
        else:
            num_frames = int(meta.get("primary_span", 0) or 0)
            runs = []
            gap_frac = 1.0 - float(meta.get("pose_coverage", 0.0) or 0.0)
        longest = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)
        coverage = round(float(meta.get("pose_coverage", 0.0) or 0.0), 3)
        gap_len = longest[1] - longest[0]
        gap_shot = None
        for shot in meta.get("shots", []) or []:
            s, e = int(shot.get("start", 0)), int(shot.get("end", 0))
            if runs and s <= longest[0] < e:
                gap_shot = f"{s}-{e}"
                break
        if coverage < min_coverage or gap_len >= max_gap_frames:
            severity = "fail"
        elif coverage < warn_coverage or gap_len >= warn_gap_frames:
            severity = "warn"
        else:
            severity = "ok"
        records.append({
            "filename": row["filename"],
            "severity": severity,
            "num_frames": num_frames,
            "pose_coverage": coverage,
            "track_frames": meta.get("track_frames", 0),
            "longest_gap_frames": gap_len,
            "longest_gap_start": longest[0],
            "longest_gap_end": longest[1] - 1,
            "n_gaps": len(runs),
            "gap_frac": round(gap_frac, 3),
            "gap_shot": gap_shot,
            "n_camera_shots": meta.get("n_camera_shots", meta.get("n_shots", 0)),
        })
    review = pd.DataFrame(records)
    order = {"fail": 0, "warn": 1, "ok": 2, "missing": 3}
    review["_order"] = review["severity"].map(order).fillna(9).astype(int)
    review = review.sort_values(["_order", "pose_coverage", "longest_gap_frames"],
                                ascending=[True, True, False])
    return review.drop(columns=["_order"]).reset_index(drop=True)
