# Phase 1 report — clip features and grouped regression baselines

Date: 2026-11-06

## Deliverables

| File | What it is |
|---|---|
| `build_clip_features.py` | Builds `clip_features.csv` from `pose21` + `ski.npz` |
| `clip_features.csv` | 234 clips × 201 columns (body, spin, grab, ski, ROI, quality) |
| `train_clip_regression.py` | Grouped-CV baselines, ablations, external test, fall-aware model |
| `clip_baseline_results.csv` | Metrics per feature set / model / split |
| `clip_baseline_folds.csv` | Per-fold metrics (stability) |
| `clip_baseline_oof.csv` | OOF predictions for the best model of each feature set |
| `clip_baseline_summary.txt` | Readable results report |
| `clip_baseline_mae.png` | MAE comparison plot |
| `build_grouped_splits.py` | Builds the leak-free split |
| `splits_grouped.csv` | 140 train / 35 val / 59 test (1026), with 5 GroupKFold ids |

Features include: body axis angle and unwrapped spin, shoulder/hip/ankle
geometry, knee/elbow angles, wrist-to-ankle and wrist-to-ski distances (grabs),
ski angles / crossing / separation / length / angular velocity / tip speed,
ROI speed/position/aspect, early / late / last-8-frame aggregates, coverage and
subject-size quality metrics.

## Evaluation design

* **Usable clips:** 187/234 (body8 ≥ 0.30, bbox ≥ 0.30, ski_any ≥ 0.05, not
  tiny/static).
* **CV:** 136 usable clips, GroupKFold(5) on `session_group`, **excluding the
  1026 session**.
* **External test:** all 51 usable `1026 (1)(2)-*` segments held out entirely
  (they are consecutive pieces of one recording).
* **Models:** train-mean dummy, Ridge (winsorised), RandomForest, HistGradientBoosting.
* **Ablations:** body only (68 features), body+ski (120), core (134), core+quality (154).

## Results

| Split | Best model | MAE | Dummy MAE | Δ |
|---|---|---|---|---|
| Grouped CV (136) | two-stage expected value | 20.98 | 21.38 | +0.40 |
| Grouped CV (136) | best non-two-stage (RF, body+ski) | 21.38 | 21.38 | 0.00 |
| External 1026 (51) | Ridge, body only | 21.07 | 21.10 | +0.03 |

* Best CV Spearman ≈ 0.12; best external Spearman ≈ 0.26 (but MAE unchanged).
* Per-fold MAE std is 2.3–3.1 points, so a 0.4-point “improvement” is noise.
* Adding ski features does **not** improve over body-only features.
* Adding quality metadata does **not** improve over core features.
* `handcrafted_features.csv` alone, and merged with the new features, also fails
  to beat the dummy under grouped CV.
* The previous fusion OOF MAE of 19.48 is reproducible, but the 1026 segments
  are spread across its folds and the gain over the constant mean is only ~1.3
  points (Spearman 0.28).

## Why the signal is weak

1. **The target is dominated by fall vs clean.** Falls average 10.4, clean runs
   55.1. Knowing only the fall label and using conditional means gives
   **MAE 12.96**, vs 21.4 for a constant. But a fall classifier from the
   clip-level pose/ski features reaches only **AUC 0.59**.
2. **The existing 3D-CNN fall classifier reaches AUC ≈ 0.80** on its saved test
   predictions, so visual appearance carries much more fall signal than
   aggregate pose/ski geometry.
3. **Label duplication limits the effective sample.** Only 102 distinct targets
   exist across the 187 usable clips; in the full set, 170/234 clips share a
   target value with another clip. The 1026 session has 59 near-consecutive
   segments from one recording, and its target correlates −0.41 with segment
   index (session drift).
4. **Aggregate clip features throw away timing.** A jump is a short event inside
   a ~3 s clip; means/quantiles over the whole clip blur the takeoff, trick and
   landing. Even good per-frame signal would be diluted.

## Phase 1b: label / duplicate audit

See `LABEL_AUDIT_REPORT.md`. Summary:

* 0 exact duplicate files.
* Same-target pairs are only marginally more similar than different-target
  pairs (visual +0.036, frame +0.023, pose +0.028); a true duplicate would be
  near 1.0.
* Only 4 candidate repeat-run pairs exist (62/63, 145/160, 171/173, 136/137).
* Consecutive 1026 segments do not overlap; they are distinct jumps.
* **Repeated targets are score ties, not repeated samples.** Keep
  `splits_grouped.csv`; no target-based regrouping is needed.

## Recommendation

Do **not** invest next in a 21-keypoint ski model or a 21-node ST-GCN as a
score predictor. The label structure is sound; the bottleneck is the lack of a
reliable fall/landing signal from pose alone.

Recommended order:

1. **Build a two-stage visual model:** fall/landing classifier → clean-run score
   regressor. Use the existing 3D-CNN features (fall AUC ≈ 0.80) rather than
   pose aggregates. Predict `P(fall)`, then blend `P(fall)·fall_mean +
   (1−P(fall))·clean_score`.
2. **Use pose/ski as auxiliary inputs to that model**, not as the primary
   representation: ski crossing/grab timing around the landing, plus body
   rotation. The clip-level ablation says these add nothing on their own.
3. **Only then** revisit a temporal pose model (ST-GCN/TCN) with the corrected
   grouped splits and ski edges.

## Reproduce

```bash
python build_clip_manifest.py
python refresh_ski_outputs.py --overwrite
python build_clip_features.py
python build_grouped_splits.py
python train_clip_regression.py
```
