# Jump-descriptor features (experimental, body-only)

Generated: 2026-11-09

This is a **new, compact feature table** built alongside the existing
`pose_features.csv`.  The goal is to replace a 205-column bag of
`mean/median/min/max/sd` statistics with a small set of physically meaningful
jump descriptors: **rotation episode, rotations, direction, tuck/grab proxy,
take-off, landing, airtime**.

It is deliberately honest about what 2D body pose can and cannot measure.  Every
unreliable quantity carries a source/confidence flag, and the features are
**not yet validated** against manual annotation.

## Outputs

| file | what it is |
|---|---|
| `build_jump_features.py` | extractor (pose-only by default, optional camera compensation) |
| `jump_features.csv` | 217 clips × 78 columns (~66 numeric) |
| `make_annotation_template.py` | builds a stratified 50-clip manual-annotation template |
| `jump_annotation_template.csv` | blank annotation sheet (fill the `ann_*` columns) |
| `validate_jump_features.py` | compares a filled annotation sheet with the extractor |
| `jump_feature_validation.md` | written by the validator |

## Feature schema

### Event / rotation

| feature | meaning |
|---|---|
| `ev_found`, `ev_anchor`, `ev_conf` | event found; anchor 1=rotation, 2=camera arc, 3=motion fallback; confidence |
| `rot_start_s`, `rot_end_s`, `rot_dur_s`, `rot_frac` | rotation-episode timing |
| `rot_peak_dps`, `rot_mean_abs_dps` | in-plane rotation rate |
| `rot_abs_turns` | absolute in-plane rotations (turns) in the episode |
| `rot_net_turns` | signed in-plane rotations (front/back flip sign) |
| `rot_turns_round` | `rot_abs_turns` rounded to 0.5 turn |
| `rot_support`, `rot_conf` | fraction of valid frames; 0–1 confidence |

### Airtime / height

| feature | meaning |
|---|---|
| `airtime_s` | ballistic flight time, **only** with `--camera-comp` |
| `airtime_src` | 0 = none, 2 = camera-compensated arc |
| `flight_height_rel` | arc amplitude / median body height (camera-comp only) |
| `arc_r2`, `cam_success` | fit quality and camera-registration success |
| `time_span_s`, `time_span_frac` | chosen analysis-window duration / fraction |
| `to_rel`, `apex_rel`, `ld_rel` | take-off / apex / landing position in the clip |

### Direction

| feature | meaning |
|---|---|
| `travel_dir` | −1 = skier moves left, +1 = right |
| `flip_dir` | sign of `rot_net_turns` if abs ≥ 0.25 turn, else 0 |
| `flip_vs_travel` | +1 = front flip, −1 = back flip, 0 = unknown |
| `takeoff_body_travel_rel`, `landing_body_travel_rel` | body-axis angle in the travel frame (deg) |
| `takeoff_uprightness`, `landing_uprightness` | abs(cos(body angle)) |
| `landing_dir_class` | 0 unknown, 1 upright/forward, 2 inverted/backward, 3 leaning/switch |

### Take-off / air / landing / tuck

| feature | meaning |
|---|---|
| `to_knee_p10`, `to_body_abs_mean`, `to_speed_mean` | take-off window |
| `air_spin_abs_mean`, `air_spin_abs_max`, `air_body_abs_mean` | air phase |
| `air_knee_p10`, `air_elbow_p10`, `air_wrist_ankle_min`, `air_scale_std` | air posture |
| `land_knee_p10`, `land_knee_flex` | landing flexion (robust 10th percentile) |
| `land_body_abs_max` | maximum body-axis deviation at landing |
| `land_residual_spin_abs` | rotation still present at touchdown |
| `land_scale_delta_abs_max`, `land_vy_abs_max`, `land_speed_mean` | landing impact proxies |
| `tuck_proxy_any`, `tuck_proxy_frac`, `tuck_proxy_duration_s`, `tuck_proxy_phase` | body-only tuck/grab **proxy** (not a grab type) |
| `approach_speed_mean`, `approach_speed_max` | approach motion |

## Reliability tiers — read this before modelling

**Comparatively trustworthy**
* `time_span_s` / `rot_dur_s` — event window is found on 217/217 clips.
* `approach_speed_mean`.
* `land_knee_*`, `land_body_abs_max`, `land_vy_abs_max` — pose geometry in a
  short landing window.

**Proxy only**
* `airtime_s` / `flight_height_rel` — require `--camera-comp`; a smoke test on
  6 clips produced 3 accepted arcs (0.40–0.83 s, 1.56–1.96 body heights,
  r² 0.94–0.99).  This is promising but unvalidated.
* `rot_abs_turns` — in-plane rotations only.  Yaw/cork rotations are invisible
  in a 2D projection, and the tracked axis can flip by 180° when the body
  inverts.  The current distribution (median 0.24, max 2.21 turns; 113/217 clips
  round to 0) almost certainly **under-counts** real tricks.
* `tuck_proxy_*` — hands near knees; this cannot distinguish a tuck from a real
  ski/board grab.

**Not reliable enough to trust yet**
* `flip_dir` / `flip_vs_travel` — 207/217 clips get 0 (unknown).
* `landing_dir_class` — 195/217 clips are class 1; only 2 inverted.
* `airtime_s` without camera compensation — deliberately NaN.

## Current run summary (pose-only, no camera compensation)

```
rows 217   columns 78   numeric 66
ev_anchor: rotation 179, motion fallback 38, camera arc 0
rot_abs_turns: median 0.24, p90 0.92, max 2.21
rot_conf:       median 0.32, 10% at 0.0
airtime_s:      0 clips (camera compensation disabled)
```

Redundancy is much lower than the old table:

| table | columns | pairs with \|r\|>0.99 | pairs >0.95 |
|---|---:|---:|---:|
| `pose_features.csv` | 205 | 162 | 294 |
| `jump_features.csv` | 78 | 4 | 9 |

## Univariate check against the target

| feature | n | Spearman with target | p |
|---|---:|---:|---:|
| `approach_speed_mean` | 217 | **+0.215** | 0.001 |
| `air_minus_ground_body` | 215 | +0.132 | 0.053 |
| `air_wrist_ankle_min` | 214 | +0.074 | 0.284 |
| `landing_uprightness` | 216 | +0.070 | 0.303 |
| `rot_abs_turns` | 216 | −0.020 | 0.775 |
| `land_knee_flex` | 210 | −0.028 | 0.686 |
| `tuck_proxy_frac` | 217 | +0.008 | 0.909 |

This is expected and should not be hidden: **most semantic jump features are
weakly correlated with the score**.  The label is dominated by fall vs clean
(fall mean 10.5 vs clean 56.7), and the dataset has little rotation-difficulty
variance.  `air_minus_ground_body` is the only feature with a notable fall
association (r = −0.21, p = 0.002), which is consistent with "more body-axis
change in the air" being related to unsuccessful landings.

## Validation workflow (required before regression)

1. Run:
   ```powershell
   python make_annotation_template.py --n 50 --features jump_features.csv
   ```
2. Fill the `ann_*` columns in a copy named `jump_annotation_filled.csv`:
   take-off / apex / landing time, airtime, rotations, flip, take-off/landing
   direction, tuck, landing outcome.  Set `annotated=1`.
3. Run:
   ```powershell
   python validate_jump_features.py
   ```
4. Keep only features whose numeric error / categorical accuracy is acceptable.
   Expect `rot_abs_turns` to need a correction factor or a different modality.

## Integration with modelling

* Keep `pose_features.csv` as the legacy reference table.
* Use `jump_features.csv` as the **interpretable semantic branch**.
* Use the frozen `r3d_18` branch for appearance/motion, especially rotations
  and trick type that 2D pose cannot see.
* Evaluate with `splits_grouped.csv` only; never tune on the 1026 session.
* Prefer a small model (ridge / shallow tree) on `jump_features.csv`; the table
  is designed to be small enough to interpret, not to win a feature-count race.

## Reproduce

```powershell
python build_jump_features.py --out jump_features.csv
python build_jump_features.py --camera-comp --out jump_features_cam.csv
python make_annotation_template.py --n 50 --features jump_features.csv
python validate_jump_features.py
```

## Limitations

1. **Monocular 2D pose.**  No depth, no yaw, no true 3D body orientation.
2. **Rotation counting is the weakest link.**  The current estimate is in-plane
   and conservative; validate before quoting "N spins".
3. **Direction is a heuristic.**  Forward/backward is defined relative to the
   image-plane travel sign; it is not a true ski-direction label.
4. **Airtime needs camera compensation**, which requires video decoding and can
   fail on textureless snow (6-clip smoke test: registration success 0.42–0.99).
5. **No grab-type ground truth.**  `tuck_proxy_*` is a body-only proxy.
6. **Fall label remains the dominant signal.**  These descriptors are unlikely
   to beat a constant-mean score model on their own; their value is
   interpretability, error analysis and combination with the visual branch.
