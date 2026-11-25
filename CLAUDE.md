# CV_Project — Action Quality Assessment for Big Air Skiing

MSc Data Science, FCUP 2026/27. Group of 2 (Mariana + teammate farboresi).
Deadline: **2026-12-01 23:59**. Deliverables: OneDrive code link, presentation, optional ≤2-page PDF.

**Goal:** given a short ski Big Air clip, predict the judges' score (0–100) and binary fall flag.

No git repository — this file and `PROJECT_STATUS.md` are the only change history.
Read `PROJECT_STATUS.md` before quoting any number from a report or notebook.

---

## Datasets

| Source | Clips | Score range | Notes |
|---|---|---|---|
| AQA-7 ski_big_air Split-4 | 175 | 0–50 integers, rescaled to 0–100 | 320×240, 103 frames @ ~30 fps, `.avi` |
| Competition (various) | 217 | 0–100 continuous | 1280×720, broadcast, 1–4 camera cuts, ~59.94 fps |

Competition sub-events (`scores/all_clips_scores.csv`): `xg_aspen_2023_women` (40),
`oly_beijing_2022_women` (36), `oly_beijing_2022_men` (36), **`xg_aspen_2025_men` (31)**,
`xg_aspen_2025_women` (29), `xg_aspen_2026_men` (27), `xg_aspen_2026_women` (18).

The 31 `xg_aspen_2025_men` clips (b-prefixed: `xg_aspen_2025_men_b001` … `b050`) are
the previously separate "xg2025men" dataset — they are now fully part of `roi_pose_dataset/`
and `scores/all_clips_scores.csv`. **`labels_xg2025men.csv` is no longer needed.**

### Canonical score source

`scores/all_clips_scores.csv` — **392 rows**, columns:
`dataset, clip_id, video_path, competition, athlete, attempt, fall, official_score, score_scale, official_score_100, aqa_split4`

Use `official_score_100` as the regression target and `fall` as the classification target.

**Loading pattern** (both feature notebooks use this):
```python
sc = pd.read_csv("scores/all_clips_scores.csv", dtype={"clip_id": str})
sc = sc.rename(columns={"clip_id": "filename", "official_score_100": "target"})
sc["filename"] = sc["filename"].str.replace("^aqa_", "", regex=True)
# AQA pose files are named '001'/'002'… scores CSV uses 'aqa_001'/'aqa_002'
```

**Do not use `labels_CV.csv` targets for modelling** — their provenance is unclear.
Use `labels_CV.csv` only as a filename/split lookup for the old notebooks.

### Label files

| File | What it is |
|---|---|
| `scores/all_clips_scores.csv` | **Canonical — single source of truth** for all 392 scored clips |
| `labels_CV.csv` | 234-clip legacy file; targets unreliable, only used by old notebooks for splits |
| `labels_xg2025men.csv` | Superseded — xg2025men scores are now in `scores/all_clips_scores.csv` |
| `splits_grouped.csv` | **The only split to report from** — leak-free GroupKFold, whole 1026 session held out |
| `splits.csv` | Old random split — session-leaked, do not use for reporting |

### Video locations

| Clips | Path |
|---|---|
| AQA-7 | `../Project/AQA-7/Actions/ski_big_air/` (`.avi`, 320×240) |
| Competition (AQA+comp) | `long_clips/` (`.mp4`) |
| xg2025men | `xg2025men/` (`.mp4`, 31 clips, also tracked and in `roi_pose_dataset/`) |
| YT clips (1026 session) | `YT_clips/` (`1026 (1)(2)-N.mp4`, 59 clips) |

---

## Pose cache — current state (2026-11-18)

**`roi_pose_dataset/` — 458 pose files, all with meta JSON**

| Group | Count | Coverage |
|---|---|---|
| AQA-7 (`001`–`175`) | 175 | varies, 55 clips below 0.6 |
| xg2025men b-prefixed | 31 | median **1.00**, all ≥ 0.97 — excellent |
| Other competition | 193 | varies |
| YT 1026 clips | 59 | varies, unscored |

392 of the 458 clips have scores in `scores/all_clips_scores.csv`.
The 66 unmatched (59 YT/1026 + 7 old non-b xg2025men clips) get `target=NaN` and are
skipped by the correlation/validation cells automatically.

`roi_pose_dataset_xg2025men/` and `ski_dataset_xg2025men/` — **superseded**,
the b-prefixed files have been merged into `roi_pose_dataset/`.

---

## Code organisation

### Tracking and pose extraction

| File | Role |
|---|---|
| `temporal_tracker.py` | **Core tracker.** Viterbi-based whole-clip, cut-aware YOLO-pose tracker. Writes `_pose.npy` [T,17,3] + `_pose_meta.json`. All features depend on this. |
| `run_retrack_aqa.py` | Re-tracks all AQA clips → `roi_pose_dataset/`. |
| `run_retrack_competition.py` | Re-tracks competition clips (single-process). `--flag-gaps` for coverage audit. |
| `run_retrack_competition_parallel.py` | Parallel re-track of competition clips. |
| `run_retrack_xg2025men.py` | Re-tracks xg2025men → `roi_pose_dataset_xg2025men/`. Superseded now that files are merged. |
| `run_retrack_xg2025men_parallel.py` | Parallel variant. Superseded. |
| `ski_detector.py` | Classical ski detector — not used in active pipeline. |

### Feature extraction

| File | Role |
|---|---|
| `feature_extraction_v2.ipynb` | **Active.** Air-phase detection → air-window spin + cork → fall signs. Single label source. Output: `air_phase_features.csv`. |
| `feature_extraction_blind.ipynb` | Whole-clip features: shoulder x-swap, cut-safe spin, cork (inversion), amplitude, landing jitter. Output: `pose_features_blind.csv`. |
| `build_clip_features.py --body-only` | 205 body/spin/posture features → `pose_features.csv`. Teammate pipeline. |
| `build_event_features.py` | Take-off/apex/landing segmentation → `pose_event_features.csv`. |
| `build_clip_manifest.py` | Per-clip QA flags → `clip_manifest.csv`. |
| `build_grouped_splits.py` | Builds `splits_grouped.csv` (leak-free split). |

### Training and evaluation

| File | Role |
|---|---|
| `train_clip_regression.py` | Tabular baselines on `pose_features.csv` → `pose_baseline_*.csv`, `checkpoints/pose_regression_model.joblib`. |
| `train_r3d_heads_grouped.py` | Frozen r3d_18 (512-D) heads on `splits_grouped.csv` → `GROUPED_EVAL_REPORT.md`. |
| `eval_event_features.py` | Event features ablation → `pose_event_eval.md`. |
| `predict_pose_regression.py` | Inference → `pose_predictions.csv`. |

### Notebooks

| Notebook | What it does |
|---|---|
| `feature_extraction_v2.ipynb` | **Active feature notebook.** Air phase first, then spin/fall. Single label source (`scores/all_clips_scores.csv`). |
| `feature_extraction_blind.ipynb` | Whole-clip complementary features. Same single label source. |
| `CV_project_rebuilt-updated.ipynb` *(hyphen)* | Research notebook: r3d_18 cache, MLP heads, pose GRU/ST-GCN. Still reads `splits.csv` — numbers are leaky; use for methods only. |
| `CV_project_xg2025men.ipynb` | Full pipeline on xg2025men only. Superseded now that those clips are in the main pose dataset. |
| `CV_project_rebuilt_updated.ipynb` *(underscore)* | Older copy — superseded. |
| `CV_project.ipynb` | Early draft — superseded. |

---

## feature_extraction_v2.ipynb — what it extracts

The v2 notebook is the main new feature work. It is simpler and more reliable than v1:

**Why v1 (feature_extraction_blind.ipynb) had poor results:**
- Whole-clip spin was corrupted: `np.unwrap` treated camera-angle changes at cuts as body rotation
- Grab detection conflates tuck with grab — can't disambiguate without ski keypoints (dropped)
- Only competition clips were scored (AQA naming mismatch — now fixed in both notebooks)

**v2 approach:**
1. **Air phase detection** — per camera shot, finds the shot with the highest centroid lift.
   The broadcast cuts to the wide shot at peak height, so apex-at-frame-0 of a shot is valid.
   Output: `air_start`, `air_end`, `air_duration_s`, `apex_y_norm` (lift in body-heights).
2. **Air-phase spin** — shoulder x-swap restricted to the air window.
   ρ +0.43 vs manual total turns (validated on 100 annotated clips).
   Output: `air_swap_turns`, `air_swap_rate_hz`, `air_inverted_frac` (cork proxy).
3. **Fall signs** — in the last 20% of the clip (after landing):
   - `head_collapse_frac` — nose within 30% body-height of ankle level (body horizontal)
   - `body_height_drop` — min body height in tail / median whole-clip
   - `sudden_stop` — tail centroid speed / whole-clip mean (falls end abruptly)
4. **Validation cell** — explicitly checks official score by fall flag per dataset,
   feature means for clean vs fall, and Spearman ρ vs score. Saves `score_by_fall.png`.

---

## Pipeline — run in this order

```
1. TRACK        (complete — 458 clips in roi_pose_dataset/)

2. FEATURES     feature_extraction_v2.ipynb        → air_phase_features.csv   (primary)
                feature_extraction_blind.ipynb      → pose_features_blind.csv  (supplementary)
                build_clip_features.py --body-only  → pose_features.csv        (teammate)

3. SPLIT        build_grouped_splits.py
                → splits_grouped.csv  ← the only split to report from

4. BASELINES    train_clip_regression.py
                → pose_baseline_*.csv, checkpoints/pose_regression_model.joblib

5. RGB          train_r3d_heads_grouped.py
                → GROUPED_EVAL_REPORT.md

6. EVENTS       eval_event_features.py
                → pose_event_eval.md

7. INFERENCE    predict_pose_regression.py
                → pose_predictions.csv
```

---

## Key results (from `GROUPED_EVAL_REPORT.md`)

All numbers are **leak-free** (grouped split, 1026 session held out).

| Model | Test MAE | Constant-mean floor | ρ |
|---|---|---|---|
| Frozen r3d_18 score head | 24.12 [20.63, 27.76] | 22.19 [18.82, 25.38] | −0.014 |
| Tabular RF (core+quality) | ~19.6 (external 1026, n=51) | ~21.1 | ~0.43 |
| Tabular RF (grouped CV) | 21.6 | 21.2 | — |

**Conclusion:** no model reliably beats the constant mean on unseen sessions.

Strongest signals from annotations (`jump_annotation_filled.csv`, n=100):

| Feature | ρ with score |
|---|---|
| fall_flag | −0.625 |
| total_rotation (spin+flip) | +0.358 |
| spin_turns | +0.326 |
| is_cork | +0.252 |
| airtime | +0.161 |
| has_grab | −0.039 |

Competition falls score 20.4 mean vs 76.6 for clean — a 56-point gap.
This is the dominant signal; all other features operate within that constraint.

---

## The split leak — read before reporting anything

`splits.csv` is a random clip split. 40 of the 59 segments of the `1026 (1)(2)-*` recording
are in its training set. Every result from notebooks using `splits.csv` is session-contaminated.

**Always use `splits_grouped.csv`** — the whole 1026 session is the held-out test set.

---

## Stale — do not cite these numbers

- `splits.csv` family — session-leaked
- `clip_features.csv`, `event_features.csv` — legacy ski-inclusive tables
- `CV_project_rebuilt_updated.ipynb` (underscore), `CV_project.ipynb` — superseded notebooks
- `checkpoints_3dcnn/{best_head,best_model,...}_fold*.pth` — trained on leaked split
- `fused_test_predictions.npz` — fusion is a no-op (fall gate never fires)
- `PHASE1_REPORT.md`, `LABEL_AUDIT_REPORT.md` numbers — from leaky split
- `labels_xg2025men.csv` — superseded by `scores/all_clips_scores.csv`
- `roi_pose_dataset_xg2025men/`, `ski_dataset_xg2025men/` — superseded, files merged into main dataset

---

## Decisions / invariants

| Topic | Decision |
|---|---|
| Score source | `scores/all_clips_scores.csv` only — single source for all 392 clips |
| AQA naming fix | Strip `aqa_` prefix on load: `sc["filename"].str.replace("^aqa_", "", regex=True)` |
| Split | `splits_grouped.csv` only; never report from `splits.csv` |
| Person selection | AQA: `track_mode="longest"`; competition: `person_select="motion"` |
| Camera cuts | Never interpolate or differentiate across a cut (`cut_frames` in `_pose_meta.json`) |
| Speed features | Use p95/max, not mean — oly_beijing clips have 16–30% duplicate frames |
| Air window | Use per-shot centroid Y lift (v2 notebook); apex at frame 0 of shot is valid |
| Shoulder x-swap | Total-rotation proxy (ρ +0.43 vs total turns), restricted to air window in v2 |
| Cork detection | `air_inverted_frac` (shoulders below hips during air); ρ +0.25 from annotations |
| Grab detection | Dropped — wrist-ankle proximity conflates tuck with grab |
| Flip detection | ρ −0.07 vs score — not worth a separate detector |
| Model bar | No model shipped unless it beats the train-mean dummy |
| AQA imgsz | 640; competition imgsz 1280 |

---

## Watch-outs

1. `scores/all_clips_scores.csv` uses `clip_id` (not `filename`) and `official_score_100` (not `target`) — always rename on load.
2. AQA pose file stems are `001`, `002` etc. but the scores CSV uses `aqa_001`, `aqa_002` — strip the prefix.
3. 59 YT/1026 clips have pose files but no scores in `scores/all_clips_scores.csv` — they get `target=NaN` and are excluded from correlation cells automatically.
4. `DATA_DIR` is hardcoded to `C:/Users/franc/...` in older notebook cells — fix to `Path.cwd() / ...` on macOS.
5. The 1026 session is out-of-distribution: OOF MAE 16.99 vs grouped-test 22.55; target drifts with segment index (ρ ≈ −0.41). Do not tune on it.
6. 55 clips in `roi_pose_dataset/` have coverage below 0.6 — features will be noisy for those clips; `air_detected=0` is the expected outcome for most of them.
