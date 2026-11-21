# tracker_v3 — skier tracking, features and models (rebuilt 2026-11-19)

Replaces `temporal_tracker.py` (Viterbi "pick one person every frame") for the
217 competition clips in `long_clips/`.

## Why the old tracker failed
It always had to output *somebody*. When the athlete was hidden (behind the
kicker), tiny (wide landing shot) or blurred, it took the only confident
person in view: the skier drawing on the Pacifico banner, the cameraman next to
the landing, a spectator in the crowd.  Its ground-motion check was computed
over the whole stitched path, so the athlete's flight hid the static pieces.

## New method (`skier_tracker_v3.py`)
1. **Detect people and skis** (YOLO11s, COCO classes 0 + 30) every 3rd frame
   (60 fps clips) / 2nd frame (30 fps).
2. **Camera motion** between detection frames: LK optical flow + RANSAC
   similarity (pan + zoom), ignoring burned-in broadcast graphics (pixels that
   never change over the clip).
3. **Tracklets**: detections linked with camera-compensated IoU + Hungarian
   matching.  A tracklet never glides from one object to another.
4. **Score each tracklet** on positive skier evidence:
   * fixed to the world (ground-relative position never leaves 0.6 body
     heights for ≥ 0.6 s) → rejected outright (banner, cameraman, judges);
   * skis detected under the feet;
   * confidence, centring, size.
5. **Chain selection per camera shot** (weighted interval scheduling).  "No
   skier" is always allowed: a weak tracklet is worth less than an empty frame.
6. **Second pass** on frames inside a shot where no skier was found: re-detect
   in 2×2 zoomed tiles (small skiers in wide shots).
7. **Pose on an upscaled crop** of the chosen box (YOLO11s-pose, 320 px), with
   rotated crops (90/180/270°) when the skier is upside down.

Runs with **OpenVINO only** (`ov_yolo.py`, no PyTorch): ~50 s per clip on 2 CPU
cores.  Models are in `models/` (exported YOLO11s / YOLO11s-pose).

```
python tracker_v3/skier_tracker_v3.py long_clips/*.mp4 --out pose_v3 [--debug-dir dbg]
python tracker_v3/features_v3.py --pose-dir pose_v3 --out tracker_v3/features_v3.csv
python tracker_v3/train_v3.py --new tracker_v3/features_v3.csv --out tracker_v3/results_v3
```
`--debug-dir` writes a video per clip: green = chosen skier (with tracklet
score, skis fraction, ground excursion), grey = other candidates, red = rejected
as static.

## Outputs (`pose_v3/`)
* `<clip>_pose.npy` — float32 [T,17,3] COCO keypoints in full-frame pixels; zeros
  where no skier.  Same layout as `roi_pose_dataset/`.
* `<clip>_pose_meta.json` — fps, size, `cut_frames`, `shots`, per-frame `bbox`,
  the chosen tracklet chain, coverage.
* `<clip>_cam.npz` — camera similarity between consecutive detection frames.

## Features (`features_v3.py`)
* **Trick window** = the 1.5 s with the most body rotation: in-plane torso
  rotation (flips/corks), shoulder-swap count (spins), inverted fraction,
  max tilt, rotation-episode length.
* **Landing / fall** (after the trick window): torso tilt, fraction lying,
  hip height above ankles, box aspect, landing tilt, body-height drop,
  skier missing right after the trick.
* **Quality**: pose / tracked fraction, shots, duration.

## Evaluation (`train_v3.py`)
Leave-one-competition-out (7 folds) — every prediction is on an event never
seen in training.  Same models on the OLD features (`air_phase_features.csv` +
`pose_features_blind.csv`) for comparison, plus a pose-only ablation.
