"""Classical-CV ski detector for the AQA big-air clips.

Skis are long, thin, near-parallel structures attached to the boots.  This
module finds them inside a tracked skier bounding box without any labels.

Pipeline
--------
1. estimate the body axis (hips -> shoulders) and smooth it over time,
2. crop a square ROI around the ankles, rotated so the body is upright,
3. CLAHE + LSD line segments, merged into full-length ski axes,
4. keep the best nearly-parallel pair near the boots, with a body-relative
   angle prior and a temporal prior,
5. when the pair is lost, propagate the previous skis with optical flow
   before decaying confidence.

Everything is resolution-independent: sizes are fractions of the skier box, so
it works on the 320x240 .avi clips as well as the 1920x1080 .mp4s.
"""
from __future__ import annotations

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _seg_len(line):
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _seg_angle(line):
    return float(np.degrees(np.arctan2(line[3] - line[1], line[2] - line[0])) % 180.0)


def _ang_diff(a, b):
    d = abs(a - b) % 180.0
    return float(min(d, 180.0 - d))


def _point_line_dist(point, line):
    p1x, p1y, p2x, p2y = (float(v) for v in line[:4])
    dx, dy = p2x - p1x, p2y - p1y
    norm = np.hypot(dx, dy)
    if norm < 1e-6:
        return float("inf")
    px, py = float(point[0]) - p1x, float(point[1]) - p1y
    return abs(dx * py - dy * px) / norm


def _projection_interval(line, origin, direction):
    t1 = float(np.dot(np.asarray(line[:2]) - origin, direction))
    t2 = float(np.dot(np.asarray(line[2:]) - origin, direction))
    return (min(t1, t2), max(t1, t2))


def _blend_angle(prev_deg, new_deg, weight):
    """Circular interpolation so 179 deg and 1 deg do not average to 90."""
    a = np.radians(prev_deg) * 2.0
    b = np.radians(new_deg) * 2.0
    v = weight * np.array([np.cos(a), np.sin(a)]) + (1.0 - weight) * np.array([np.cos(b), np.sin(b)])
    if np.hypot(*v) < 1e-6:
        return float(new_deg)
    return float((np.degrees(np.arctan2(v[1], v[0])) / 2.0) % 180.0)


def make_roi_transform(center, side, out_size, up=None):
    """Forward/inverse affine mapping a square frame ROI to an ``out_size`` crop.

    ``up`` is the image-space body-up direction; it is mapped to (0, -1) in the
    crop so the skier is always upright.  Pass ``None`` for an axis-aligned crop.
    Returns (M, Minv, scale) where M maps frame -> crop and Minv maps back.
    """
    center = np.asarray(center, dtype=np.float64)
    side = float(max(side, 32.0))
    out_size = float(out_size)
    scale = out_size / side
    c_out = np.array([out_size / 2.0, out_size / 2.0])

    if up is None:
        R = np.eye(2, dtype=np.float64)
    else:
        up = np.asarray(up, dtype=np.float64)
        norm = np.hypot(up[0], up[1])
        up = up / norm if norm > 1e-6 else np.array([0.0, -1.0])
        ux, uy = up
        # rotation that maps `up` to (0, -1)
        R = np.array([[-uy, ux], [-ux, -uy]], dtype=np.float64)

    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = scale * R
    M[:, 2] = c_out - scale * (R @ center)

    Minv = np.zeros((2, 3), dtype=np.float64)
    Minv[:2, :2] = R.T / scale
    Minv[:, 2] = center - (R.T / scale) @ c_out
    return M.astype(np.float32), Minv.astype(np.float32), scale


def apply_transform(points, M):
    """Apply a 2x3 affine to an (N, 2) array."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(pts, np.asarray(M, dtype=np.float32)).reshape(-1, 2)


def warp_roi(image, M, out_size, border_value=114):
    out = int(out_size)
    return cv2.warpAffine(image, np.asarray(M, dtype=np.float32), (out, out),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border_value)


# --------------------------------------------------------------------------- #
# body orientation
# --------------------------------------------------------------------------- #
class BodyAligner:
    """Temporally smoothed hips->shoulders axis, used to keep the skier upright."""

    def __init__(self, smooth=0.8):
        self.smooth = float(smooth)
        self.up = None
        self.missing = 0

    def reset(self):
        self.up = None
        self.missing = 0

    def update(self, keypoints):
        """Update from a [17, 3] COCO array in frame coordinates. Returns the axis."""
        vector = None
        if keypoints is not None and len(keypoints) >= 17:
            kp = np.asarray(keypoints, dtype=np.float64)
            conf = kp[:, 2] if kp.shape[1] > 2 else np.ones(len(kp))
            shoulders, hips = kp[[5, 6], :2], kp[[11, 12], :2]
            sh_ok = bool((conf[[5, 6]] > 0.25).all())
            hi_ok = bool((conf[[11, 12]] > 0.25).all())
            if sh_ok and hi_ok:
                vector = shoulders.mean(axis=0) - hips.mean(axis=0)
            elif hi_ok and conf[0] > 0.25:                       # head -> hips
                vector = kp[0, :2] - hips.mean(axis=0)
            elif hi_ok:                                          # knees -> hips
                knees = kp[[13, 14], :2]
                knee_ok = conf[[13, 14]] > 0.25
                if knee_ok.any():
                    vector = hips.mean(axis=0) - knees[knee_ok].mean(axis=0)

        if vector is not None and np.hypot(vector[0], vector[1]) > 1e-3:
            vector = vector / np.hypot(vector[0], vector[1])
            if self.up is None:
                self.up = vector
            else:
                # a sudden 180 deg jump is almost always a bad keypoint, not a flip
                weight = self.smooth if np.dot(self.up, vector) > -0.3 else 0.25
                blended = weight * self.up + (1.0 - weight) * vector
                norm = np.hypot(blended[0], blended[1])
                if norm > 1e-6:
                    self.up = blended / norm
            self.missing = 0
        else:
            self.missing += 1

        if self.up is None:
            self.up = np.array([0.0, -1.0])
        return self.up


# --------------------------------------------------------------------------- #
# detector
# --------------------------------------------------------------------------- #
class SkiDetectorCV:
    """Detects up to two skis per frame inside a tracked skier ROI.

    Parameters
    ----------
    crop_scale : square crop side, in multiples of the skier bbox size
    work_size : the crop is resized to this many pixels before line detection
    min_len_frac : shortest accepted LSD segment, as a fraction of the crop
    max_angle_diff : maximum angle between the two skis (degrees)
    smooth : temporal EMA weight for angle/centre/length (0 = no smoothing)
    max_missing : frames a detection is carried over when the pair is lost
    align_body : rotate the ROI so the skier is upright and apply a body-relative
        angle prior (skis are horizontal in that frame)
    angle_prior_deg : max deviation from horizontal for a ski line (aligned mode)
    flow_propagation : use Lucas-Kanade to carry skis through missed detections
    require_pose_anchor : refuse to guess when no reliable ankle is available
    """

    def __init__(self, crop_scale=2.6, work_size=480, min_len_frac=0.055,
                 max_angle_diff=14.0, smooth=0.65, max_missing=10,
                 align_body=False, angle_prior_deg=70.0,
                 flow_propagation=True, require_pose_anchor=True):
        self.crop_scale = float(crop_scale)
        self.work_size = int(work_size)
        self.min_len_frac = float(min_len_frac)
        self.max_angle_diff = float(max_angle_diff)
        self.smooth = float(smooth)
        self.max_missing = int(max_missing)
        self.align_body = bool(align_body)
        self.angle_prior_deg = float(angle_prior_deg)
        self.flow_propagation = bool(flow_propagation)
        self.require_pose_anchor = bool(require_pose_anchor)
        self.aligner = BodyAligner()
        self._lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
        self._unit = 60.0
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self):
        self.prev = None          # list of ski dicts (frame coords), last frame
        self.missing = 0
        self._prev_gray = None
        self.aligner.reset()

    def _empty(self):
        self._prev_gray = None
        return {"skis": [], "ankle": None, "pose_ok": False, "aligned": False}

    # -- main entry point --------------------------------------------------
    def detect(self, frame, bbox, keypoints=None):
        """Return {'skis': [...], 'ankle': (x, y), 'pose_ok': bool}.

        bbox is (x, y, w, h) and keypoints is the optional [17, 3] COCO array
        in full-frame pixel coordinates.  Each ski dict has: quad (4,2),
        tip (2,), tail (2,), angle (deg), length (px), conf (0-1), side.
        """
        x, y, w, h = (float(v) for v in bbox[:4])
        unit = max(w, h)
        H, W = frame.shape[:2]
        if unit < 12.0 or not np.isfinite(unit):
            return self._empty()
        self._unit = unit

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ankle, pose_ok = self._ankle_anchor(x, y, w, h, keypoints, H)
        up = self.aligner.update(keypoints)
        aligned = bool(self.align_body and pose_ok)
        if self.require_pose_anchor and not pose_ok:
            return self._flow_or_temporal(gray_full, ankle, pose_ok, aligned=False)

        # square, optionally body-aligned ROI centred on the boots
        side = float(np.clip(self.crop_scale * unit, 48.0, 2.0 * max(H, W)))
        M, Minv, scale = make_roi_transform(ankle, side, self.work_size,
                                            up if aligned else None)
        small = warp_roi(gray_full, M, self.work_size)
        small = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(small)
        unit_c = unit * scale
        ankle_c = np.array([self.work_size / 2.0, self.work_size / 2.0])

        lines = self._detect_lines(small, self.min_len_frac * self.work_size)
        lines = self._merge_collinear(lines)

        # body-relative prior: in the aligned frame skis are roughly horizontal
        if aligned:
            lines = [l for l in lines
                     if min(_seg_angle(l), 180.0 - _seg_angle(l)) <= self.angle_prior_deg]

        # skis pass close to the boots
        lines = [l for l in lines if _point_line_dist(ankle_c, l) < 0.30 * unit_c]
        pair = self._best_pair(lines, ankle_c, unit_c)
        if pair is None:
            return self._flow_or_temporal(gray_full, ankle, pose_ok, aligned)

        pair = [self._refine(line, lines, unit_c) for line in pair]
        pair = [p for p in pair if p is not None]
        if len(pair) < 2:
            return self._flow_or_temporal(gray_full, ankle, pose_ok, aligned)

        skis = [self._line_to_ski(line, ankle_c, Minv) for line in pair]
        skis = [s for s in skis if s is not None]
        if len(skis) < 2:
            return self._flow_or_temporal(gray_full, ankle, pose_ok, aligned)

        skis = self._assign_sides(skis, ankle, keypoints)
        skis = self._smooth(skis)
        self.prev, self.missing = skis, 0
        self._prev_gray = gray_full
        return {"skis": skis, "ankle": ankle, "pose_ok": pose_ok, "aligned": aligned}

    # -- anchors / detector internals -------------------------------------
    @staticmethod
    def _ankle_anchor(x, y, w, h, keypoints, frame_h):
        if keypoints is not None and len(keypoints) >= 17:
            kp = np.asarray(keypoints, dtype=np.float64)
            conf = kp[:, 2] if kp.shape[1] > 2 else np.ones(len(kp))
            ankles = kp[[15, 16], :2]
            ok = conf[[15, 16]] > 0.20
            if ok.any():
                return ankles[ok].mean(axis=0), True
            # lower legs blurred / cropped: extrapolate the ankle from hip->knee
            knees, hips = kp[[13, 14], :2], kp[[11, 12], :2]
            knee_ok = conf[[13, 14]] > 0.20
            if knee_ok.any():
                side = int(np.argmax(np.where(knee_ok, conf[[13, 14]], -1)))
                if conf[[11, 12][side]] > 0.20:
                    ankle_est = knees[side] + 1.0 * (knees[side] - hips[side])
                    ankle_est[1] = min(max(ankle_est[1], 0.0), frame_h - 1.0)
                    return ankle_est, True
        return np.array([x + w / 2.0, min(y + h, frame_h - 1.0)]), False

    def _detect_lines(self, gray, min_len):
        result = self._lsd.detect(gray)[0]
        if result is None:
            lines = np.zeros((0, 4), dtype=np.float32)
        else:
            lines = np.asarray(result, dtype=np.float32).reshape(-1, 4)
        if len(lines):
            lengths = np.linalg.norm(lines[:, 2:] - lines[:, :2], axis=1)
            lines = lines[lengths >= min_len]
        if len(lines) < 2:
            edges = cv2.Canny(gray, 40, 120)
            hough = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=25,
                                    minLineLength=int(min_len), maxLineGap=4)
            if hough is not None:
                extra = np.asarray(hough, dtype=np.float32).reshape(-1, 4)
                lines = np.vstack([lines, extra]) if len(lines) else extra
        return lines

    @staticmethod
    def _merge_collinear(lines, angle_tol=6.0, dist_tol=2.5, gap_frac=0.15):
        items = [np.asarray(l, dtype=np.float64) for l in lines]
        changed = True
        while changed and len(items) > 1:
            changed = False
            used = [False] * len(items)
            merged = []
            for i in range(len(items)):
                if used[i]:
                    continue
                base = items[i]
                used[i] = True
                base_len = _seg_len(base)
                for j in range(i + 1, len(items)):
                    if used[j]:
                        continue
                    other = items[j]
                    if _ang_diff(_seg_angle(base), _seg_angle(other)) > angle_tol:
                        continue
                    mid = (other[:2] + other[2:]) / 2.0
                    if _point_line_dist(mid, base) > dist_tol:
                        continue
                    angle = np.radians(_seg_angle(base))
                    direction = np.array([np.cos(angle), np.sin(angle)])
                    t0, t1 = _projection_interval(other, base[:2], direction)
                    gap = max(0.0, max(0.0, t0) - min(base_len, t1))
                    if gap > gap_frac * max(base_len, _seg_len(other)):
                        continue
                    pts = np.vstack([base[:2], base[2:], other[:2], other[2:]])
                    t = pts @ direction
                    base = np.concatenate([pts[np.argmin(t)], pts[np.argmax(t)]])
                    base_len = _seg_len(base)
                    used[j] = True
                    changed = True
                merged.append(base)
            items = merged
        return items

    def _best_pair(self, lines, ankle_c, unit_c):
        if len(lines) < 2:
            return None

        prev_angle, prev_center = None, None
        if self.prev and self.prev[0].get("_confidence_frames", 0) <= self.max_missing:
            theta = np.radians(np.array([s["angle"] for s in self.prev]) * 2.0)
            prev_angle = float((np.degrees(np.arctan2(np.sin(theta).mean(), np.cos(theta).mean())) / 2.0) % 180.0)
            prev_center = np.mean([s["_center"] for s in self.prev], axis=0)

        best, best_score = None, -np.inf
        for i in range(len(lines)):
            l1 = lines[i]
            a1, len1 = _seg_angle(l1), _seg_len(l1)
            if not (0.10 * unit_c <= len1 <= 1.25 * unit_c):
                continue
            for j in range(i + 1, len(lines)):
                l2 = lines[j]
                a2, len2 = _seg_angle(l2), _seg_len(l2)
                if not (0.10 * unit_c <= len2 <= 1.25 * unit_c):
                    continue
                if _ang_diff(a1, a2) > self.max_angle_diff:
                    continue

                sep = 0.5 * (_point_line_dist((l2[:2] + l2[2:]) / 2.0, l1) +
                             _point_line_dist((l1[:2] + l1[2:]) / 2.0, l2))
                if sep < max(1.5, 0.05 * unit_c) or sep > 0.30 * unit_c:
                    continue

                d_ankle = max(_point_line_dist(ankle_c, l1), _point_line_dist(ankle_c, l2))
                if d_ankle > 0.22 * unit_c:
                    continue

                centre = ((l1[:2] + l1[2:]) / 2.0 + (l2[:2] + l2[2:]) / 2.0) / 2.0
                if np.hypot(*(centre - ankle_c)) > 0.45 * unit_c:
                    continue

                angle = np.radians(0.5 * (a1 + a2))
                direction = np.array([np.cos(angle), np.sin(angle)])
                i1 = _projection_interval(l1, ankle_c, direction)
                i2 = _projection_interval(l2, ankle_c, direction)
                overlap = min(i1[1], i2[1]) - max(i1[0], i2[0])
                if overlap <= 0.10 * min(len1, len2):
                    continue

                score = (len1 + len2) - 6.0 * sep - 10.0 * d_ankle + 0.6 * overlap
                if prev_angle is not None and prev_center is not None:
                    score += 0.9 * (len1 + len2) * np.cos(np.radians(_ang_diff(prev_angle, 0.5 * (a1 + a2))))
                    score -= 0.004 * float(np.hypot(*(centre - prev_center)))
                if score > best_score:
                    best_score, best = score, (l1, l2)
        return best

    @staticmethod
    def _refine(line, all_lines, unit_c):
        """Merge all collinear fragments near `line` into one full-length ski axis."""
        angle = np.radians(_seg_angle(line))
        direction = np.array([np.cos(angle), np.sin(angle)])
        normal = np.array([-direction[1], direction[0]])
        mid = (np.asarray(line[:2], dtype=np.float64) + np.asarray(line[2:], dtype=np.float64)) / 2.0

        members = []
        for other in all_lines:
            if _ang_diff(_seg_angle(other), _seg_angle(line)) > 8.0:
                continue
            off = max(abs(np.dot(np.asarray(other[:2]) - mid, normal)),
                      abs(np.dot(np.asarray(other[2:]) - mid, normal)))
            if off > max(2.0, 0.035 * unit_c):
                continue
            t_mid = 0.5 * (np.dot(np.asarray(other[:2]) - mid, direction) +
                           np.dot(np.asarray(other[2:]) - mid, direction))
            if abs(t_mid) > 0.85 * unit_c:
                continue
            members.append(other)
        if not members:
            members = [line]

        points = np.vstack([np.asarray([m[:2], m[2:]], dtype=np.float64) for m in members])
        t = (points - mid) @ direction
        lo = max(float(t.min()), -1.25 * unit_c)
        hi = min(float(t.max()), 1.25 * unit_c)
        if hi - lo < 4.0:
            return None
        return np.concatenate([mid + direction * lo, mid + direction * hi])

    def _line_to_ski(self, line, ankle_c, Minv):
        length_c = _seg_len(line)
        if length_c < 4.0:
            return None
        p1 = np.asarray(line[:2], dtype=np.float64)
        p2 = np.asarray(line[2:], dtype=np.float64)
        # ski tip = the end farther from the boot, tail = the end closer to it
        if np.hypot(*(p1 - ankle_c)) >= np.hypot(*(p2 - ankle_c)):
            tip_c, tail_c = p1, p2
        else:
            tip_c, tail_c = p2, p1
        tip = apply_transform(tip_c[None, :], Minv)[0]
        tail = apply_transform(tail_c[None, :], Minv)[0]
        return self._make_ski(tip, tail)

    def _make_ski(self, tip, tail):
        tip = np.asarray(tip, dtype=np.float64)
        tail = np.asarray(tail, dtype=np.float64)
        length = float(np.hypot(*(tip - tail)))
        if length < 2.0:
            return None
        direction = (tip - tail) / length
        angle = float((np.degrees(np.arctan2(direction[1], direction[0]))) % 180.0)
        width = float(np.clip(0.035 * length, 2.0, 8.0))
        normal = np.array([-direction[1], direction[0]]) * width / 2.0
        quad = np.array([tip + normal, tip - normal, tail - normal, tail + normal], dtype=np.float32)
        return {
            "quad": quad,
            "tip": tip.astype(np.float32),
            "tail": tail.astype(np.float32),
            "angle": angle,
            "length": length,
            "conf": self._confidence({"length": length}),
            "side": None,
            "_center": ((tip + tail) / 2.0),
            "_confidence_frames": 0,
        }

    @staticmethod
    def _assign_sides(skis, ankle, keypoints):
        if keypoints is not None and len(keypoints) >= 17:
            kp = np.asarray(keypoints, dtype=np.float64)
            conf = kp[[15, 16], 2] if kp.shape[1] > 2 else np.ones(2)
            if (conf > 0.25).all():
                ankle_l, ankle_r = kp[15, :2], kp[16, :2]
                d_l = [np.hypot(*(s["_center"] - ankle_l)) for s in skis]
                d_r = [np.hypot(*(s["_center"] - ankle_r)) for s in skis]
                if d_l[0] + d_r[1] <= d_l[1] + d_r[0]:
                    skis[0]["side"], skis[1]["side"] = "left", "right"
                else:
                    skis[0]["side"], skis[1]["side"] = "right", "left"
                return skis
        order = np.argsort([s["_center"][0] for s in skis])
        skis[int(order[0])]["side"] = "left"
        skis[int(order[1])]["side"] = "right"
        return skis

    def _smooth(self, skis):
        """EMA on angle/centre/length, matching this frame's skis to the last ones."""
        if not self.prev or len(self.prev) < 2:
            for s in skis:
                s["conf"] = self._confidence(s)
            return skis

        prev_by_side = {s["side"]: s for s in self.prev if s.get("side")}
        if len(prev_by_side) < 2:
            for s in skis:
                s["conf"] = self._confidence(s)
            return skis

        w = self.smooth
        for ski in skis:
            old = prev_by_side.get(ski["side"])
            if old is None:
                ski["conf"] = self._confidence(ski)
                continue
            angle = _blend_angle(old["angle"], ski["angle"], w)
            centre = w * old["_center"] + (1.0 - w) * ski["_center"]
            length = w * old["length"] + (1.0 - w) * ski["length"]
            direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
            tip = centre + direction * length / 2.0
            tail = centre - direction * length / 2.0
            normal = np.array([-direction[1], direction[0]]) * float(np.clip(0.035 * length, 2.0, 8.0)) / 2.0
            ski.update({
                "angle": angle,
                "length": length,
                "tip": tip.astype(np.float32),
                "tail": tail.astype(np.float32),
                "_center": centre,
                "quad": np.array([tip + normal, tip - normal, tail - normal, tail + normal],
                                 dtype=np.float32),
                "conf": self._confidence({"length": length}),
            })
        return skis

    def _confidence(self, ski):
        unit = getattr(self, "_unit", 60.0)
        return float(np.clip(0.35 + 0.30 * min(1.0, ski["length"] / (0.55 * unit)), 0.0, 0.9))

    # -- temporal fallbacks ------------------------------------------------
    def _flow_or_temporal(self, gray, ankle, pose_ok, aligned):
        """Propagate the previous skis with optical flow; decay if that fails."""
        if (self.flow_propagation and self.prev and self._prev_gray is not None
                and self._prev_gray.shape == gray.shape and self.missing < self.max_missing):
            moved = self._propagate_with_flow(self._prev_gray, gray)
            if moved:
                self.missing += 1
                out = []
                for ski in moved:
                    copy = dict(ski)
                    copy["conf"] = float(ski.get("conf", 0.4) * (0.92 ** self.missing))
                    copy["_confidence_frames"] = self.missing
                    out.append(copy)
                self.prev = out
                self._prev_gray = gray
                return {"skis": out, "ankle": ankle, "pose_ok": pose_ok, "aligned": aligned}

        result = self._temporal_fallback(ankle, pose_ok)
        result["aligned"] = aligned
        self._prev_gray = gray
        return result

    def _propagate_with_flow(self, prev_gray, gray):
        points = []
        for ski in self.prev:
            points.extend([ski["tip"], ski["tail"], ski["_center"]])
        p0 = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)

        lk = dict(winSize=(21, 21), maxLevel=3,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
        p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk)
        if p1 is None:
            return None
        p0_back, status_back, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, p1, None, **lk)
        if p0_back is None:
            return None

        fb_error = np.linalg.norm(p0.reshape(-1, 2) - p0_back.reshape(-1, 2), axis=1)
        displacement = np.linalg.norm(p1.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)
        max_step = max(6.0, 0.6 * max(s["length"] for s in self.prev))
        valid = ((status.ravel() == 1) & (status_back.ravel() == 1) &
                 (fb_error < 2.0) & (displacement < max_step))

        moved, any_ok = [], False
        for k, ski in enumerate(self.prev):
            base = 3 * k
            tip_i, tail_i, centre_i = base, base + 1, base + 2
            tip_ok, tail_ok, centre_ok = valid[tip_i], valid[tail_i], valid[centre_i]
            new_tip = np.asarray(ski["tip"], dtype=np.float64)
            new_tail = np.asarray(ski["tail"], dtype=np.float64)

            if tip_ok and tail_ok:
                new_tip = p1[tip_i, 0].astype(np.float64)
                new_tail = p1[tail_i, 0].astype(np.float64)
                any_ok = True
            elif tip_ok or tail_ok or centre_ok:
                index = tip_i if tip_ok else (tail_i if tail_ok else centre_i)
                flow = p1[index, 0].astype(np.float64) - p0[index, 0].astype(np.float64)
                new_tip, new_tail = new_tip + flow, new_tail + flow
                any_ok = True
            else:
                moved.append(dict(ski))
                continue

            rebuilt = self._make_ski(new_tip, new_tail)
            if rebuilt is None:
                moved.append(dict(ski))
                continue
            rebuilt["side"] = ski.get("side")
            rebuilt["conf"] = ski.get("conf", 0.4)
            moved.append(rebuilt)

        return moved if any_ok else None

    def _temporal_fallback(self, ankle, pose_ok):
        if self.prev and self.missing < self.max_missing:
            self.missing += 1
            decay = 0.65 ** self.missing
            out = []
            for ski in self.prev:
                copy = dict(ski)
                copy["conf"] = float(ski.get("conf", 0.5) * decay)
                copy["_confidence_frames"] = self.missing
                out.append(copy)
            self.prev = out
            return {"skis": out, "ankle": ankle, "pose_ok": pose_ok}
        self.prev, self.missing = None, 0
        return {"skis": [], "ankle": ankle, "pose_ok": pose_ok}


# --------------------------------------------------------------------------- #
# notebook helpers
# --------------------------------------------------------------------------- #
def bbox_from_keypoints(keypoints, frame_shape=None, pad=0.25, min_conf=0.2):
    """Return a padded (x, y, w, h) box around the visible body keypoints.

    The tracker box is often tighter than the actual body (feet/head clipped).
    Deriving the box from the pose keypoints and padding it keeps the ROI around
    the whole skier.  Returns ``None`` when fewer than four keypoints are usable.
    """
    kp = np.asarray(keypoints, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return None
    visible = kp[:, 2] >= min_conf if kp.shape[1] > 2 else np.ones(len(kp), dtype=bool)
    if int(visible.sum()) < 4:
        return None

    xs, ys = kp[visible, 0], kp[visible, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    pad_px = float(pad) * max(w, h)
    x1, x2 = x1 - pad_px, x2 + pad_px
    y1, y2 = y1 - pad_px, y2 + pad_px

    if frame_shape is not None:
        frame_h, frame_w = frame_shape[:2]
        x1, x2 = max(0.0, x1), min(float(frame_w), x2)
        y1, y2 = max(0.0, y1), min(float(frame_h), y2)

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def pose_keypoints_full_frame(model, frame, prev_bbox=None, conf_thres=0.1,
                              bbox_pad=0.25):
    """Run YOLO pose on the full frame and return ``([17,3], padded bbox)``.

    With a single skier, this is more robust than cropping a tight tracker ROI:
    the person stays in view even when the tracker drifts.  ``prev_bbox`` is used
    only to keep the same person when several people are detected.
    """
    result = model(frame, verbose=False, conf=conf_thres)[0]
    empty = np.zeros((17, 3), dtype=np.float32)
    if result.keypoints is None or len(result.keypoints.xy) == 0:
        return empty, None

    boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else None
    if boxes is not None and len(boxes) > 0:
        if prev_bbox is not None:
            target = np.array([prev_bbox[0] + prev_bbox[2] / 2.0,
                               prev_bbox[1] + prev_bbox[3] / 2.0])
            person = int(np.argmin(np.linalg.norm(boxes[:, :2] - target, axis=1)))
        else:
            person = int(np.argmax(result.boxes.conf.cpu().numpy()))
    else:
        person = 0

    xy = result.keypoints.xy[person].cpu().numpy()
    kconf = (result.keypoints.conf[person].cpu().numpy()
             if result.keypoints.conf is not None else np.ones(len(xy), dtype=np.float32))
    keypoints = np.zeros((len(xy), 3), dtype=np.float32)
    keypoints[:, :2] = xy
    keypoints[:, 2] = kconf

    bbox = bbox_from_keypoints(keypoints, frame.shape[:2], pad=bbox_pad)
    if bbox is None and boxes is not None and len(boxes) > 0:
        cx, cy, w, h = boxes[person]
        bbox = (cx - w / 2.0, cy - h / 2.0, w, h)
    return keypoints, bbox


def pose_keypoints_detected_roi(model, frame, prev_bbox=None, aligner=None,
                                conf_thres=0.1, bbox_pad=0.25, min_visible=4):
    """Detect the person on the full frame, then refine pose on their ROI crop.

    This removes reliance on CSRT tracking: each frame's person box comes from
    YOLO itself, so the ROI cannot drift off the skier.  The extra ROI pass keeps
    the low-resolution AQA clips (320x240) upscaled, which is where a full-frame
    pose pass loses the smaller joints.

    Returns ``([17,3] keypoints, padded (x, y, w, h) body box)``.
    """
    kp_full, bbox = pose_keypoints_full_frame(
        model, frame, prev_bbox=prev_bbox, conf_thres=conf_thres, bbox_pad=bbox_pad)
    if bbox is None:
        return kp_full, None

    kp = pose_keypoints_in_roi(
        model, frame, bbox, aligner=aligner, conf_thres=conf_thres,
        fallback_full_frame=False)
    visible = int((kp[:, 2] > 0.3).sum()) if kp.size else 0
    if visible < min_visible:
        kp = kp_full

    body_bbox = bbox_from_keypoints(kp, frame.shape[:2], pad=bbox_pad)
    if body_bbox is None:
        body_bbox = bbox
    return kp, body_bbox


def pose_keypoints_in_roi(model, frame, bbox, aligner=None, out_size=640, margin=0.30,
                          conf_thres=0.1, fallback_full_frame=True):
    """Run YOLO pose on the tracked skier ROI and return [17, 3] frame coords.

    When ``aligner`` is given the crop is rotated so the skier is upright
    (using the temporally smoothed body axis) and the keypoints are mapped back
    through the inverse affine.  Otherwise the original aspect-preserving crop
    is used.

    ``conf_thres`` is intentionally lower than the Ultralytics default (0.25)
    because the 320x240 AQA clips produce very low person confidences.  If the
    ROI crop yields no pose (common for small, blurry skiers), the function can
    fall back to a full-frame pose pass and keeps the person closest to the
    tracked bbox centre.
    """
    x, y, w, h = (float(v) for v in bbox[:4])
    H, W = frame.shape[:2]
    keypoints = np.zeros((17, 3), dtype=np.float32)

    def _full_frame_fallback():
        result = model(frame, verbose=False, conf=conf_thres)[0]
        if result.keypoints is None or len(result.keypoints.xy) == 0:
            return np.zeros((17, 3), dtype=np.float32)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xywh.cpu().numpy()
            target = np.array([x + w / 2.0, y + h / 2.0])
            person = int(np.argmin(np.linalg.norm(boxes[:, :2] - target, axis=1)))
        else:
            person = 0
        xy = result.keypoints.xy[person].cpu().numpy()
        kconf = (result.keypoints.conf[person].cpu().numpy()
                 if result.keypoints.conf is not None else np.ones(len(xy), dtype=np.float32))
        out = np.zeros((len(xy), 3), dtype=np.float32)
        out[:, :2] = xy
        out[:, 2] = kconf
        return out

    if aligner is not None and aligner.up is not None:
        center = np.array([x + w / 2.0, y + h / 2.0])
        side = float(np.clip(1.35 * np.hypot(w, h), 48.0, 2.0 * max(H, W)))
        M, Minv, _ = make_roi_transform(center, side, out_size, aligner.up)
        crop = warp_roi(frame, M, out_size)
        result = model(crop, verbose=False, conf=conf_thres)[0]
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            # pick the person closest to the crop centre (not a spectator)
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xywh.cpu().numpy()
                centres = boxes[:, :2]
                person = int(np.argmin(np.linalg.norm(centres - out_size / 2.0, axis=1)))
            else:
                person = 0
            xy = result.keypoints.xy[person].cpu().numpy()
            kconf = (result.keypoints.conf[person].cpu().numpy()
                     if result.keypoints.conf is not None else np.ones(len(xy), dtype=np.float32))
            keypoints = np.zeros((len(xy), 3), dtype=np.float32)
            keypoints[:, :2] = apply_transform(xy, Minv)
            keypoints[:, 2] = kconf
    else:
        x1 = max(0, int(x - w * margin))
        y1 = max(0, int(y - h * margin))
        x2 = min(W, int(x + w * (1 + margin)))
        y2 = min(H, int(y + h * (1 + margin)))
        crop = frame[y1:y2, x1:x2]
        ch, cw = crop.shape[:2]
        if ch > 0 and cw > 0:
            scale = min(float(out_size) / cw, float(out_size) / ch)
            nw, nh = int(cw * scale), int(ch * scale)
            padded = np.full((out_size, out_size, 3), 114, dtype=np.uint8)
            pad_x, pad_y = (out_size - nw) // 2, (out_size - nh) // 2
            padded[pad_y:pad_y + nh, pad_x:pad_x + nw] = cv2.resize(crop, (nw, nh))
            result = model(padded, verbose=False, conf=conf_thres)[0]
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                crop_kpts = result.keypoints.xy[0].cpu().numpy()
                kconf = result.keypoints.conf[0].cpu().numpy()
                keypoints = np.zeros((len(crop_kpts), 3), dtype=np.float32)
                keypoints[:, 0] = (crop_kpts[:, 0] - pad_x) / scale + x1
                keypoints[:, 1] = (crop_kpts[:, 1] - pad_y) / scale + y1
                keypoints[:, 2] = kconf

    if fallback_full_frame and keypoints[:, 2].max() <= 0:
        keypoints = _full_frame_fallback()

    if aligner is not None:
        body = keypoints[:17] if len(keypoints) >= 17 else keypoints
        aligner.update(body if len(body) and body[:, 2].max() > 0 else None)
    return keypoints


def pose21_keypoints_in_roi(model, frame, bbox, aligner=None, out_size=640):
    """Same as pose_keypoints_in_roi, but for a 21-keypoint ski-pose model.

    Returns [21, 3] in frame coordinates.  The body axis is taken from the
    first 17 keypoints, so the crop can be aligned with the previous pose.
    """
    keypoints = pose_keypoints_in_roi(model, frame, bbox, aligner=aligner, out_size=out_size)
    if len(keypoints) >= 21:
        return keypoints[:21]
    out = np.zeros((21, 3), dtype=np.float32)
    out[:len(keypoints)] = keypoints
    return out


def skis_to_keypoints(ski_result):
    """Pseudo keypoints 17..20: L tip, L tail, R tip, R tail -> [4, 3]."""
    out = np.zeros((4, 3), dtype=np.float32)
    for ski in ski_result.get("skis", []):
        base = 0 if ski["side"] == "left" else 2
        out[base] = [ski["tip"][0], ski["tip"][1], ski["conf"]]
        out[base + 1] = [ski["tail"][0], ski["tail"][1], ski["conf"]]
    return out


def attach_ski_keypoints(pose17, ski_result):
    """[17, 3] COCO pose + 4 pseudo ski keypoints -> [21, 3]."""
    return np.vstack([np.asarray(pose17, dtype=np.float32), skis_to_keypoints(ski_result)])


def draw_skis(frame, ski_result, conf_threshold=0.12):
    """Draw ski quads (magenta), tips (green) and their angle."""
    for ski in ski_result.get("skis", []):
        if ski["conf"] < conf_threshold:
            continue
        quad = np.asarray(ski["quad"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [quad], isClosed=True, color=(255, 0, 255), thickness=2)
        tip = tuple(np.asarray(ski["tip"], dtype=np.int32))
        cv2.circle(frame, tip, 4, (0, 255, 0), -1)
        cv2.putText(frame, f"{ski['side'][0].upper()} {ski['angle']:.0f}deg",
                    (tip[0] + 5, tip[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 0, 255), 1, cv2.LINE_AA)
    return frame
