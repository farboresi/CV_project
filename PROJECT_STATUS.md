# Project status — what is current, what is stale

Last updated: 2026-11-17 (fail-clip re-track in progress; duplicate-frame finding
recorded and deliberately deferred).

There is **no git repository** here, so nothing below is recoverable from history.
Read this before quoting any number from a report or a notebook.

## Latest: tracker v2, full re-track, competition evaluation (2026-11-16)

Tracker changes applied and validated on a full re-track of all 234 AQA clips
and the 91 scored competition clips:

* **Within-shot travel direction** (`ENABLE_DIRECTION_PRIOR=True`): a skier
  keeps one ground-relative horizontal direction inside a camera shot.  The
  prior is soft: a reversing track is replaced only when a one-direction track
  of at least 50% of its length exists.  A/B probes: `probe_direction.py`.
* **Pose-quality emission penalty** (`ENABLE_POSE_QUALITY=True`,
  `POSE_QUALITY_WEIGHT=0.8`): soft cost for few joints, missing lower body,
  keypoint cloud far from the box centre, or tiny spread.  Static objects and
  helmet/face close-ups are suppressed without a hard rule.
* **Event-aware track selection** (`person_select="event"` for AQA): scores
  shoulder rotation energy + vertical excursion with a 40% length floor.  Fixes
  `1026 (1)(2)-40` (approach 0-27 -> jump 68-155); identical to `longest` on
  the tested AQA clips (`001/005/021/048/058/096/106/160`, `1026-33/-59`).
* **Shot-level skier score** (`_skier_score`): rotation + vertical + pose
  completeness + direction + centre + size - crowd, stored per shot and as
  `skier_score_best` / `skier_shot_frac` in every meta file.
* **Box from pose**: missing boxes are filled from the keypoint cloud and
  `box_pose_offset_median` / `box_pose_offset_frac_gt_0_5` are recorded.
* **Single-shot clips now get a shot record**, so the score/diagnostics exist
  for every clip, not only cut clips.
* Seed-dedup fix and edge-portrait filter are in this run (see below).

Re-track results (234 AQA, 91 scored competition; `retrack_aqa_event.log`,
`retrack_comp_parallel.log`):

* AQA coverage on the previously weak clips: `006` 0.65, `013` 0.38, `016`
  0.67, `027` 0.25, `007` 0.65, `050` 0.42; `048` 0.84, `106` 0.76, `058` 0.98,
  `160` 1.00.
* Competition coverage: mean 0.77 (median 0.76, min 0.26); every clip has a
  track and a skier score.
* **Targeted tiling/upscaling** (item 8) works: re-running the worst clips with
  `--tile-trigger 120` fixed `1026 (1)(2)-3` 0.27 -> 0.76, `1026 (1)(2)-47`
  0.22 -> 0.54, `72` 0.18 -> 0.81, `38` 0.04 -> 0.17, `65` 0.04 -> 0.18, `67`
  0.06 -> 0.17, and `xg_aspen_2026_men_050` 0.21 -> 0.86.  `1026 (1)(2)-6`
  needed `imgsz=1280` + tiling: 0.02 -> 0.53.  `97`/`101` needed 1280 +
  tiling: 0.22 -> 0.85/0.69.  `1026 (1)(2)-34` is *worse* at 1280 (0.41 ->
  0.20), so resolution is not monotonically better.
* `audit_tracks.py` scans all 427 cached tracks -> `track_audit.csv`,
  `track_audit_shots.csv`, `retrack_priority.csv`.  Final flags: 3
  low-coverage (`65`, `38`, `67`), 9 partial-body, 3 missed-jump (`7`, `78`,
  `98`), 2 box/pose misalignment, 1 tiny-subject, 0 wrong-person, 88 reversal
  (soft; long reversing shots, mostly crowd/wide).  The ranked re-track list
  carries `suggest_imgsz` / `suggest_tile` / `suggest_select`.

Feature rebuild after the re-track:

* `jump_features.csv` 234 rows (all AQA clips after the tiling fixes).
* `jump_features_cal_leakfree.csv`: airtime calibration OOF MAE 0.229 s
  (test 1026 0.386 s).
* `train_jump_regression.py`: no model beats the train-mean dummy
  (`core`/RF 21.70 vs 21.11 OOF; external test 22.00 vs 22.50).  Do not ship
  the jump regression.
* `simple_motion_features.csv`: 427 rows.
  - AQA: best `jump_core_control`/RF 20.41 vs 21.46 dummy (+4.9%, rho +0.20).
  - Competition (91 scored, 3 events): best `all_simple`/Ridge **21.13 vs
    22.50 dummy (+6.1%)**; jump-table control 21.69.  Strongest univariate
    signals after coverage control: `sh_std_deg` +0.34, `body_std_deg` +0.31,
    `sp_net_deg` -0.23.  Twist is *not* predictive on competition (-0.08),
    unlike AQA (+0.24).
  - Competition folds are now grouped by `session_group`; the earlier shuffled
    5-fold result (19.58 MAE) was event leakage.
* `cork_features.csv` 426 rows; best cork OOF AUC 0.678 (not deployable).

Air-window finding: the rotation episode is only ~30% of the air phase
(median air/rot ratio 3.56).  Fixed padding reaches median IoU 0.65; per-clip
optimal padding reaches 0.97, but no observable predicts the pad.  Pose-only
acceleration and ORB-stabilised ballistic fits both under-segment the flight
(median IoU 0.04 and 0.31).  **Do not trust `air_*` features as full-air-phase
measurements; the calibrated airtime and the rotation window are the reliable
quantities.**  Probes: `probe_air_window*.py`, `probe_air_arc.py`,
`probe_air_dynamics.py`.

Combined manifest: `clip_manifest_combined.csv` (427 rows) via
`build_manifest_combined.py`; `jump_features_combined.csv` (427 rows).
AQA keeps its grouped split; competition is `split="external"` with
`event_fold` 0-2 over the three scored events.


## The one thing that matters

`splits.csv` is a **random clip split**, and 40 of the 59 segments of the
`1026 (1)(2)-*` recording are in its training set (7 in val, 12 in test). Every
score / AUC figure produced by `CV_project_rebuilt-updated.ipynb` is therefore
measured partly on a recording the model has already seen.

The leak-free frame is **`splits_grouped.csv`**: the whole 1026 session is the
test set (59 clips, 3 falls), and train+val (166 clips) is split into 5
GroupKFold folds. Report from that file only.

| | leaked (`splits.csv`) | leak-free (`splits_grouped.csv`) |
|---|---|---|
| score head test MAE | 19.41 | **24.12** [20.63, 27.76] |
| score head test rho | 0.285 | **−0.014** [−0.28, +0.24] |
| constant-mean floor | 20.89 | **22.19** [18.82, 25.38] |
| margin over floor | −1.48 (better) | **+1.93 (worse)** |
| fall AUC on test | 0.799 (6 falls) | 0.619 (3 falls) — unusable CI |

Reproduce with `python train_r3d_heads_grouped.py` (≈2 min: reads the frozen
r3d_18 caches, decodes no video). Output: `GROUPED_EVAL_REPORT.md`,
`checkpoints_3dcnn/grouped_predictions.npz`.

**Conclusion:** on an unseen session the frozen r3d_18 features do not beat
predicting the training mean. The earlier "beats the dummy" result was the
session leak. End-of-clip windows remain worse than the floor (22.62 vs 22.19).

## Canonical pipeline

Run in this order. Steps 1–3 produce the pose cache and feature table, 4 the
split, 5–8 the models and diagnostics.  The active branch is **body-only B1**:
skis are not used for pose, tracking or features.

1. `CV_project_rebuilt.ipynb` — temporal body-pose ingest
   (`temporal_tracker.py` → `roi_pose_dataset/`).
2. `build_clip_manifest.py` — `clip_manifest.csv`: body-only quality flags and
   `usable` (skis are diagnostic only).
3. `build_clip_features.py --body-only --out pose_features.csv` — 205 body /
   spin / posture / ROI / window features per clip, no ski columns.
4. `build_grouped_splits.py` — `splits_grouped.csv` (**the** split).
5. `train_clip_regression.py` — tabular baselines → `pose_baseline_*`, then a
   final fit → `checkpoints/pose_regression_model.joblib` + feature contract.
6. `train_r3d_heads_grouped.py` — RGB score/fall/two-stage heads, leak-free.
7. `build_event_features.py --body-only --out pose_event_features.csv` —
   take-off / apex / landing / trick-phase segmentation. Anchored on the
   **rotation episode**, not on a ballistic arc fit (a ballistic fit on the bbox
   track fails here — see `pose_event_eval.md`).
8. `eval_event_features.py` — `pose_event_eval.md`: do the event/phase features
   beat the aggregate ones on `splits_grouped.csv`?
9. `predict_pose_regression.py` — load the saved bundle and write
   `pose_predictions.csv` for inference.

`refresh_ski_outputs.py` is **not** part of the active body-only pipeline. It is
kept only for reproducing the legacy ski table (`clip_features.csv`).

`CV_project_rebuilt-updated.ipynb` is the research notebook (all the modelling
exploration). It still reads `splits.csv`, so its printed numbers are not
reportable; it is useful for the methods and diagnostics only.

## Current

| artifact | what it is |
|---|---|
| `labels_CV.csv` | 234 clips, `target` 0–100, `fall` 0/1 (32 falls) — single label source |
| `splits_grouped.csv` | the only split to report from (built from `pose_features.csv`) |
| `clip_manifest.csv` | body-only quality flags; 41/234 clips flagged unusable |
| `pose_features.csv` | **active** body-only tabular features, 205 columns, no ski features |
| `pose_baseline_results.csv`, `pose_baseline_summary.txt` | fresh tabular baselines on the temporal body pose |
| `checkpoints/pose_regression_model.joblib` | final body-only regression pipeline (auto-selected, fit on train+val) |
| `checkpoints/pose_regression_features.json` | feature contract + training metadata for inference |
| `predict_pose_regression.py` | loads the bundle and writes `pose_predictions.csv` |
| `pose_event_features.csv` | **active** event / phase features, body-only, no ski columns |
| `pose_event_windows.csv` | detected take-off / apex / landing times per clip, for inspection |
| `pose_event_eval.md` | body-only event features vs the aggregate table, same folded protocol |
| `clip_features.csv`, `event_features.csv` | legacy 21-keypoint + ski tables, kept for reference only |
| `GROUPED_EVAL_REPORT.md` | **current headline numbers**, with bootstrap CIs |
| `checkpoints_3dcnn/grouped_predictions.npz` | OOF + grouped-test predictions behind that report |
| `checkpoints_3dcnn/r3d18_feature_cache_v1.npz` | 234×512 full-clip features (frozen r3d_18) |
| `checkpoints_3dcnn/r3d18_window_cache_v1.npz` | 468×512 last-8 / last-4 window features |
| `PHASE1_REPORT.md`, `LABEL_AUDIT_REPORT.md` | still accurate as *method* write-ups; their numbers were computed on the leaked split or on the tabular CV, and are superseded by the table above |

## Stale — do not cite these numbers

* `splits.csv`, `splits_10pct_backup.csv`, `splits_train/val/test.csv`,
  `splits_*_fold*.csv` — the old random split family (four different test sets:
  45, 51, 59, 20 clips). `build_grouped_splits.py` replaces all of it.
* `clip_baseline_*`, `clip_features.csv`, `event_features.csv`,
  `EVENT_EVAL_REPORT.md` — legacy ski-inclusive feature/baseline artifacts.
  The active equivalents are `pose_baseline_*`, `pose_features.csv`,
  `pose_event_features.csv`, and `pose_event_eval.md`.
* `checkpoints_3dcnn/{best_head,best_model,score_reg,fall_clf,end_head,fall_head}_fold*.pth`
  — trained against `splits.csv`, i.e. contaminated. Kept for reference only.
* `checkpoints_3dcnn/{pred_full_aug,pred_end_only,pred_aug+end,oof_blend_arrays,final_low_arrays}.npz`
  — from an older 203-clip / 183-OOF / 20-test frame, not comparable to anything above.
* `fused_test_predictions.npz` — its "learned" correction moves MAE 19.41 → 19.34
  (+0.07) with rho 0.285 → 0.287, and the fall gate never fired (mean fall
  probability 0.01 on test). The fusion stage is a no-op, not a result.
* `train_ski_pose.py`, `ski_pose_dataset*/`, `roi_pose_dataset_21kpt/`,
  `yolo*-21.yaml` — abandoned 21-keypoint experiment (pose mAP50 ≈ 0.0017).
* `CV_project.ipynb`, `CV_project_rebuilt_updated.ipynb` (underscore),
  `backup_before_fixes/`, `*.bak*`, `*.bak_pre_splitfix` — copies with no owner.
  `CV_project_rebuilt-updated.ipynb.bak_pre_splitfix` is the notebook as it was
  before the split-constant fix below.

## Fixed 2026-11-17 — tiling was skipped on exactly the frames it was meant for

`collect_candidates` returned `[]` as soon as the full-frame YOLO pass produced
no keypoints, **before** `_apply_tiling` ran.  `ENABLE_TILING` therefore never
fired on frames where the detector found nobody, which is precisely the
distant/wide-shot case the 2x2 tiled pass exists to recover.  Tiling only ran
when the full-frame pass found a person but all candidates were smaller than
`TILE_TRIGGER_SIZE` — the opposite of the hard case.

Fix: the early-return path in `temporal_tracker.collect_candidates` now calls
`_apply_tiling(model, frame, [], ...)`.  Original kept as
`temporal_tracker.py.bak_pre_tilingfix` (there is no git here).

Measured on `oly_beijing_2022_men_010` (330 frames, 1280x720, cuts `44, 304`):

* full-frame YOLO returns **0 candidates on 154 of the 260** frames in the wide
  shot `44-304`; a 2x2 tile pass recovers the skier on most of those frames
  (e.g. frame 220: full-frame 0, tiled box at `(525, 362)` 38x70, conf 0.20);
* cached pass-1 track in that shot: **33 frames -> 172 frames**; the longest
  remaining gap drops to 15 frames, so `interpolate_boxes(max_gap=15)` fills
  all 330 boxes instead of leaving the jump empty;
* the clip-level cached track was 97 frames / 34% pose coverage.  Full re-track
  with the fix (`retrack_gapfix_men010.log`): **236 track frames and 330/330
  posed frames (100% coverage)**, with the wide shot `44-304` at 172 tracked
  frames (was 33).  Visual check: boxes and skeletons ride the athlete through
  the in-run, the jump and the landing
  (`ski_visual_checks/gapfix_oly_beijing_2022_men_010.jpg`).

A second, independent wide-shot clip, `oly_beijing_2022_men_013`, goes from
**39% to 100% pose coverage** (track 102 -> 211 frames; wide shot `9-215`
22 -> 131 tracked frames).  Visual check:
`ski_visual_checks/gapfix_oly_beijing_2022_men_013.jpg`.

The probe also ranks the remaining cached gaps.  On clips re-tracked before the
fix, tiling recovers the skier inside the longest missing run on
`xg_aspen_2025_men_009` 6/6 sampled frames, `xg_aspen_2025_men_016` 5/6,
`oly_beijing_2022_women_010` 4/6, and `xg_aspen_2023_women_078` 1/6
(`tiling_gain_probe_bracketed.csv`).  **Caution:** the probe predicts the box
from the old track, so a hit proves the tiled detector can follow *that*
subject, not that the subject is the athlete.

`xg_aspen_2025_men_009` re-tracked to 100% coverage (88% -> 100%, track 666
frames), but the visual check
(`ski_visual_checks/gapfix_xg_aspen_2025_men_009.jpg`) shows the track riding
foreground people/signage in several wide/outro shots.  Coverage is not
identity correctness: visually check unscored clips before using them.

Also fixed: `run_retrack_competition.py --clips ...` used to rebuild the shared
`dataset_processed_features_competition.csv` from only the selected clips,
overwriting 193 rows with 1.  The summary rebuild now always uses the full
competition label set.

`probe_tiling_gain.py` measures the same effect on cached clips without
re-tracking: it samples the longest missing run, interpolates the skier box
from the neighbouring tracked boxes, and counts full-frame vs 2x2-tiled
candidates near that box.  Output: `tiling_gain_probe.csv`.

**Retrack bookkeeping:** the fail-clip loop starts a new Python process per
chunk, so a chunk only picks up a code edit if it starts after the edit.  Chunk
8 had already started when this fix landed, so chunks 1-8 were re-tracked with
the old code; chunks 9-12 (if the loop is still running) import the fixed code.
Do not compare those groups as one homogeneous run.  Chunks 1-8 need a targeted
re-run, worst `tiling_gain_probe.csv` rows first.

## In progress / deferred (2026-11-17)

* **Fail-clip re-track is running.** The 119 `fail` clips from
  `roi_pose_dataset/detection_review.csv` are being re-tracked in 12 chunks of
  10 (`retrack_fail.log`, PowerShell, `--all --imgsz 1280`, `--clips` from
  `fails.txt`). Chunks 1-3 are complete (~25-30 min/chunk); chunk 4 is mid-run.
  **The CSV written before the run is the only before-baseline** — there is no
  `roi_pose_dataset_backup`, every meta has been overwritten.
  Provisional means: chunk 1 0.36 -> 0.63, chunk 2 0.46 -> 0.74,
  chunk 3 0.54 -> 0.70.
  Two things to check when it finishes, both unexplained so far:
  `xg_aspen_2026_men_011` came back **worse** (0.58 -> 0.48; cuts `[404, 961]`,
  shot 0-404 keeps only 20 frames at `median_size` 60 px), and
  `xg_aspen_2023_women_080` / `_025` / `_046` / `_068` / `_020` re-tracked to
  the *same* coverage as the cached run, so for those clips the old numbers
  were real rather than stale-cache artefacts.
* **Duplicate source frames — deferred, do not act on yet.** 16-30% of frames
  in the `oly_beijing_2022_*` clips are byte-identical copies of their
  predecessor (17% = every 6th frame, i.e. 25 fps padded to 30; 26-30% in
  `men_071`, `women_008`, `women_003`), while almost every `xg_aspen_*` clip is
  clean. Measured consequences: (a) `pose_coverage` and `longest_gap_frames`
  divide by *video* frames, so duplicated clips read low — on distinct frames
  `women_008` goes 0.71 -> 1.00 and `men_071` 0.48 -> 0.65, but
  corr(dup%, coverage) is only -0.10 over 22 sampled clips, so this did **not**
  drive the fail list; (b) every mean-per-frame column (`*_mean` speed/spin,
  `*_pace`) is diluted by exactly the duplication rate while `*_max`/`*_p95`
  are unaffected (verified against a 0%-dup control clip), so those columns
  carry a source-dependent scale error across the two competitions; (c)
  seconds-based features (airtime, durations, arc timing) are unaffected and
  the tracker gate is neutral (a duplicated frame is a zero-displacement step
  that always passes `MAX_STEP_BODY`). A cheap fix would be a distinct-frame
  coverage column plus dedup in the `*_mean` features; not started.

## Known open issues

1. **Notebook writes outside the repo.** `DATA_DIR` is hardcoded to
   `C:/Users/franc/Downloads/AQA-7/AQA-7/Actions/ski_big_air` in most cells, so
   notebook checkpoints land there while the scripts here use project-relative
   paths. The two sets are not the same directory.
2. **Body-pose features are weak on CV.** The active tables are body-only
   (`pose_features.csv`, `pose_event_features.csv`); `_pose21.npy` ski columns
   are intentionally zero. Tabular RF does not beat the train-mean dummy on the
   grouped CV folds (21.6 vs 21.2 MAE), although on the external 1026 session
   ridge/RF reach ~19.3–20.0 MAE with rho ~0.35–0.40. Treat the signal as weak
   and do not tune on the 1026 test set.
3. **The 1026 session is out of distribution**, not just leaked: OOF MAE 16.99
   vs grouped-test 22.55. Its target drifts with segment index (rho ≈ −0.41), so
   even a clean split cannot make this an easy test set.
4. **Test set has 3 falls**, so no fall metric on it is meaningful. Quote OOF
   (fall AUC 0.841) and say it is out-of-fold, or pool folds with CIs.
5. **`VideoRegressionDataset` decodes every frame** at 224² into one tensor and
   the RGB path trains on all 234 clips, including the 41 flagged unusable.
6. **`pose_features.csv` has 225 rows, not 234.** Nine clips have fewer than
   four tracked body frames and cannot yield features; they are all unusable,
   so `splits_grouped.csv` now has 225 clips (59 in the 1026 test set).

## Fixed 2026-11-06

`SPLIT_FRACTIONS` was `(0.70, 0.10, 0.150)` in cell 3 while `make_splits`
assigns test as the remainder (0.20) and cell 8A checked for 0.20. The 4.2pp
gap meant cell 3 warned and **rewrote the shared `splits.csv` on every run**.
Both cells now use `(0.70, 0.10, 0.20)` and neither rewrites an existing split
file — they warn and keep it. The `assert` message ("train/test leakage") was
also corrected: it only checks filename overlap, so it never proved the 1026
session was held out.

## Deleted 2026-11-06

Only unreferenced or exactly-duplicated files were removed; nothing a live
script or notebook cell reads was touched. No git here, so this is the record.

* **duplicate notebooks** — 2 item(s), 1.1 MB
* **notebook backups** — 6 item(s), 6.5 MB
* **old split files** — 24 item(s), 0.0 MB
* **superseded checkpoints** — 11 item(s), 506.8 MB
* **abandoned 21-keypoint experiment** — 11 item(s), 437.1 MB
* **superseded feature caches** — 8 item(s), 3.1 MB
* **stale fusion outputs** — 3 item(s), 0.0 MB
* **scratch** — 2 item(s), 0.2 MB

Total freed: 955 MB.

Kept deliberately:
* `CV_database.zip` (428 MB) — **not** a redundant copy: 81 of its 234 video
  names do not exist on disk, so it is the only copy of those under the
  original names.
* `splits.csv` — still read by 19 places in the research notebook; deleting it
  would make the notebook silently regenerate a leaky split.
* `checkpoints/base*.pt`, `checkpoints/stgcn_fold*.pt`, `stgcn_*.csv`,
  `pose_cache/`, `ski_visual_checks/`, `baseline_history.csv`,
  `filtered_labels.csv` — written/read by cells that still run.
* `best_head_fold*.pth`, `score_reg_fold*.pth`, `fall_clf_fold*.pth`,
  `end_head_fold*.pth`, `fall_head_fold*.pth` — contaminated by the leaky
  split, but small and are the evidence behind the earlier numbers.
* `handcrafted_features.csv` — cited by `PHASE1_REPORT.md`.
* `target.csv`, `rigorous_labels.csv`, `youtube.csv` — tiny label/video
  provenance; in a project whose main problem is labels, history is cheap.
* `CV_project_rebuilt-updated.ipynb.bak_pre_splitfix` — the only snapshot of
  the notebook before the `SPLIT_FRACTIONS` fix.
* `_vis_proto.py` — unreferenced, but it is code, not an artifact; left for you.

## Experimental — jump descriptors (2026-11-12)

`jump_features.csv` (217 × 78) is a new, compact semantic branch built by
`build_jump_features.py`; it is **not** part of the active pipeline yet.
Validated on 50 manually annotated clips (`jump_feature_validation.md`):

* raw **airtime** from the camera-comp arc is not usable (MAE 1.50 s).  A Ridge
  calibration on pose metadata reaches LOO MAE 0.23 s, but the honest number is
  lower: OOF 0.15 s on train+val and **0.51 s on the external 1026 session**
  (`build_airtime_leakfree.py`).  Dataset-specific.
* **flips** stay in the pose branch: `rot_abs_turns` MAE 0.88 / ρ 0.24; the
  best frozen r3d_18 head is worse (0.94 / 0.17).
* **spins** are unsolved.  Pose MAE 2.25 / ρ 0.04.  The frozen r3d_18 head
  looked good in LOO (1.14 / 0.39) but that was session-leaky; under grouped CV
  it does not beat the train-mean baseline (OOF 1.29 vs 1.22; 1026 test 2.25 vs
  1.65).  See `VISUAL_SPIN_MODEL.md`.
* **landing geometry** is weak: no landing feature is significant after
  multiple comparisons (`LANDING_FEATURE_EVAL.md`).  The 6-feature air-phase
  fall screen (AUC ~0.76 on random folds) drops to OOF AUC 0.67 under grouped
  CV.
* **grouped-CV score models do not beat the constant mean** (`JUMP_MODEL_EVAL.md`):
  best OOF MAE 20.86 vs 21.45 (`core`/RF), external 22.05 vs 22.33 with
  overlapping bootstrap intervals.  Spearman rises to +0.14..+0.23, so there
  may be weak ranking signal but not a deployable score.
* direction/tuck proxies are majority-class predictors; do not model them.
* trick axis (`cork`/`straight`/`none`) has no signal in the frozen features.

### Low-effort shoulder/spine branch (2026-11-13)
`build_simple_motion_features.py` → `simple_motion_features.csv` (262 clips:
234 AQA + 28 competition poses present at the time).  Each frame gives two lines:
the shoulder line and the spine (shoulder-mid → hip-mid).  Their signed
frame-to-frame rotations are combined as `0.5*(d_shoulder + d_spine)` (the
"body" signal); the angle between the lines is the torso **twist**.  Movement
summaries are total/net rotation, peak/mean angular speed and active fraction.
Angles are aspect-corrected, long gaps are interpolated/smoothed (5 frames), and
differencing never crosses a known camera cut (`*_pose_meta.json` or
`aqa_cut_clips.csv`).

Result (`SIMPLE_MOTION_FEATURES.md`, 5-fold CV on the 234 AQA clips):

* the professor's hypothesis "more movement → higher score" is **not**
  supported raw: `body_total_deg` ρ = -0.08 and `sh_total_deg` ρ = -0.05 on clean
  clips.  This is a tracking-coverage confound: `body_valid_frac` correlates
  +0.78 with movement and -0.18 with the score.  After controlling for coverage
  the partial ρ is +0.15..+0.20.
* the stable simple signal is **shoulder–spine twist**: `twist_mean_abs_deg`
  ρ = +0.27 (partial +0.22), `twist_mean_dps` +0.21, `twist_std_deg` +0.21.
* grouped-CV score models are only marginally better than the constant mean:
  best simple 21.11 MAE (`shoulder_motion`/RF) and 21.15 for a single
  `twist_mean_abs_deg` Ridge, vs 21.46 dummy.  The complex jump table is 20.06.
  So the branch is cheap and interpretable but **not sufficient alone**.
* re-test on the scored competition clips when their poses finish; the
  judges' scores are less dominated by fall/clean.

### Ellipse / pendulum spin probe (2026-11-13)

`probe_ellipse_spin.py` tests the professor's idea that a rotating shoulder
vector traces a compressed ellipse whose phase gives the spin.  On the 49
annotated clips with a usable air window:

* the ellipse **shape** separates the rotation type: `ell_ecc`
  (sqrt(minor/major), 0 = line, 1 = circle) correlates **-0.33** with spin
  turns (p=0.02) and **+0.25** with flip turns (p=0.09).  Flatter shoulder
  ellipses go with spins, rounder ones with flips.
* naive phase unwrapping does **not** work (`ell_net_turns` rho +0.08,
  p=0.60); peak/zero-crossing counts are noise-dominated.
* **a parametric constant-rate ellipse fit does work**:
  `x(t) = cx + R(phi).[a cos(wt+th0), b sin(wt+th0)]` fitted by robust least
  squares gives the rotation frequency `orb_hz`.  `orb_hz` correlates
  **+0.33** with manual spin turns (p=0.02) and **+0.45** with total turns
  (p=0.001); controlling for airtime, the partial rho is **+0.34**.  On the
  23 clips with `orb_r2 > 0.2`, `orb_hz` vs spin is **+0.47** (p=0.02).
  This is the first positive spin signal from the pose branch (the old
  `rot_abs_turns` spin rho was about 0.04).
* caveats: n=49, the orbit `r2` is usually 0.1-0.4, and the fit gives a
  ranking/frequency, not an exact turn count.  Needs grouped CV and the
  competition clips before any modelling claim.
* next: add orbit features to the feature table on the automatic air window,
  then grouped-CV the spin/turn target.  The fit is low-dimensional, so this
  stays in the "low-effort" spirit.

### Torso / flip probe (2026-11-13)

`probe_torso_flip.py` applies the same orbit fit to the spine (`5-6` mid to
`11-12` mid) and the two torso edges (`5->11`, `6->12`).  It does **not**
transfer as cleanly:

* **flip count is not recovered**: `flip_hz` vs manual flips rho +0.05
  (p=0.72), `spine_turns` +0.04; even on the 20 clips with flips > 0 the
  correlation is ~0.  So torso ellipse frequency is not a flip counter.
* **direction is unvalidated**: only 3-4 front flips in the 100-clip sheet.
  `sign(omega)` gets 86% "accuracy" but that is just the back-flip majority;
  `sign(omega)*travel_dir` gets 64%.  Would need a balanced sample.
* **cork has a weak signal**: left-edge frequency is higher for cork
  (1.08 vs 0.75 Hz, p=0.04, rank-biserial +0.38), the single-orbit fit is
  worse for cork (`flip_r2` 0.52 vs 0.67, p=0.06), and the left/right edge
  phase-reversal gap is *lower* for cork (0.08 vs 0.17, p=0.007) - opposite
  to the naive expectation, so treat as exploratory.
* why flips are harder: the body is non-rigid, shoulder/hip keypoints can
  swap under inversion (180 deg jumps), and the torso vector is short.  The
  spin fit worked because foreshortening carries the phase; for an in-plane
  flip the direction carries it, and flexion/camera view corrupt it.
* next: run Procrustes/Kabsch on **all 17 keypoints** (or a long head-ankle
  axis) for the flip rate; use the orbit residual and edge disagreement as
  cork features; re-test direction only on a balanced set.

### Whole-skeleton Procrustes flip probe (2026-11-13)

`probe_skel_rotation.py` fits the best confidence-weighted 2D rotation between
consecutive frames using all 17 keypoints, over the annotated air window
(50 clips):

* **flip count is not improved.**  `skel_net_turns` rho +0.25 (p=0.087),
  raw MAE 0.78; the existing `rot_net_turns` on the same clips is rho +0.26,
  MAE 0.69.  More joints do not add information beyond the old body-axis
  rotation.  On the 21 clips with flips > 0 the correlation is ~+0.1.
* **direction fails.**  `sign(net rotation)` is 57% accurate on the 23
  back/front clips (coin flip); the existing `flip_dir` is worse.  The sheet
  has only 3 front flips, so this cannot be validated.
* **cork works best so far.**  Corked tricks have more net rotation
  (0.69 vs 0.52 turns, p=0.018, rb +0.45), more absolute rotation
  (3.67 vs 2.42, p=0.024, rb +0.43) and a higher non-rigidity residual
  (`skel_resid` 0.49 vs 0.37, p=0.007, rb +0.51).  That is consistent with
  cork being an off-axis, non-rigid combination of spin and flip.
* caveats: n=50, one annotation sheet, air windows are manual; `skel_max_dps`
  is noise-dominated (a few 150 deg/frame swaps), so use residuals and
  medians, not maxima.  Treat cork as exploratory until grouped CV.

### Shoulder x-order swap probe (2026-11-13)

`probe_shoulder_swap.py` counts hysteretic sign changes of `x5 - x6` (and
`x11 - x12` as a sanity check).  A full 360-degree turn gives **two** swaps, so
`turns = swaps / 2`.  On the 49 annotated clips:

* `swap_turns` vs **total** manual turns rho **+0.43** (p=0.002); vs spin
  alone +0.20 (p=0.18) and vs flip alone +0.22 (p=0.13).  So it is a generic
  total-rotation proxy, not a spin-specific counter.
* the shoulder length at the crossing was meant to separate spin (shoulders
  overlap -> short) from flip (line stays long), but the split did not hold:
  `swap_spin_turns` vs spin +0.14, `swap_flip_turns` vs flip +0.21.
* `swap_hip_agree` is 1.0 on most clips, so most swaps are real body rotations
  rather than left/right keypoint label swaps.
* vs the AQA score: rho +0.14 (p=0.33), not significant on this subset.
* counts under-estimate (e.g. 4 manual spins -> 2.5 swaps), so it is a ranking
  proxy, not an exact counter.

The five swap features are now in `cork_features.csv` / `cork_features_manual.csv`
(`swap_turns`, `swap_rate_hz`, `swap_spin_turns`, `swap_flip_turns`,
`swap_hip_agree`).

### Vertical-order swap probe (2026-11-13)

`probe_vertical_swap.py` applies the same hysteretic swap count to the vertical
order of the shoulder/hip midpoints and of the head/ankles.  The hope was a
**flip-specific** counter (a flip inverts the body, a pure spin does not).  On
the 49 annotated clips it does **not** isolate flips:

| signal | vs spin | vs flip | vs total |
|---|---:|---:|---:|
| shoulder/hip y-swaps | +0.21 | +0.13 | **+0.34** (p=0.016) |
| head/hip y-swaps | +0.22 | +0.06 | **+0.33** (p=0.023) |
| head/ankle y-swaps | +0.19 | +0.19 | **+0.36** (p=0.011) |
| hip/ankle y-swaps | +0.14 | +0.16 | +0.28 (p=0.054) |

So it is another **total-rotation** proxy, not a flip counter.  Reasons: a
corked spin tilts the body and swaps the projected vertical order; the torso
and head/feet are often nearly horizontal during a flip, where the sign is
noisy; and fast flips fall into pose gaps.  The x-order shoulder swap remains
the better cheap total-rotation feature, and flips stay with the weak
skeleton/orbit estimates.

**Using them separately vs together:** the x-swap alone gives rho +0.43 with
total turns; the y-swap alone +0.34.  The equal average `swap_total_turns`
improves the rank correlation to +0.46 (shoulder/hip y) or +0.49
(head/ankle y), but LOO-calibrated MAE stays at ~0.97-0.99 (x alone ~0.96), so
the gain is marginal.  The **ratio** `y / (x + y)` does **not** separate spin
from flip: pure-spin clips average 0.18-0.20, pure-flip clips 0.14-0.16, and
the AUC for flip presence is 0.50-0.54.  Use the x-swap alone (simplest) or
the equal average as a slightly more robust total-rotation estimate; do not
use the ratio as a spin/flip classifier.

`v_spine_turns`, `v_headhip_turns`, `v_headank_turns`, `v_hipank_turns` and
`swap_total_turns` are now in `cork_features.csv` /
`cork_features_manual.csv`.

### Cork classifier (2026-11-13)

`build_cork_features.py` → `cork_features.csv` (289 clips × 44 columns) /
`cork_features_manual.csv` (290 × 44).  It builds an **automatic** air window
per clip (169 from the jump event `to_rel`/`ld_rel`, 109 from the longest
confident pose run) and computes:

* whole-skeleton Procrustes features (`skel_rotation.py`): net/absolute turns,
  mean/p95 angular speed, non-rigidity residual, reversal fraction, coverage;
* constant-rate ellipse fits on the spine and shoulder vectors
  (`spine_orb_*`, `shoulder_orb_*`), the angle between their axes and the
  spin/flip frequency ratio.

`eval_cork_classifier.py` → `CORK_CLASSIFIER.md` (target: manual
`ann_spin_axis`, cork vs straight; grouped folds; 1026 external):

* 78 annotated cork/straight clips have features (55 cork, 23 straight);
  49 are in the CV folds and 29 in the 1026 test (25/5).
* **automatic-window OOF AUC (leak-free folds):** `all`/RF **0.676**
  (balanced acc 0.658), `core3_orbit`/RF 0.642, `core3_norm_window`/RF 0.616,
  `core3`/RF **0.534**; logistic 0.50-0.58.
* **manual-window check** (`cork_features_manual.csv`,
  `CORK_CLASSIFIER_MANUAL.md`) is similar: `all`/RF 0.659 (0.642),
  `core3_norm_window`/RF **0.704**, `core3_norm`/RF 0.602, `core3`/RF 0.556.
  So window quality is not the main driver; the simple 3-feature model is at
  chance either way.
* **uncertainty:** bootstrap 95% CI for `all`/RF is [0.515, 0.824] and label
  permutation p=0.050; for `core3`/RF AUC 0.534, p=0.347 → weak, not
  deployable.  The 23-feature RF is overfit-prone on 49 clips.
* external 1026 (`core3_norm_window`/RF) AUC 0.783 on 25 cork vs 5 straight is
  not evidence.
* **verdict: weak, borderline, not deployable.**  The earlier "manual window
  kills it" conclusion was wrong: the first comparison used a leaky CV where
  the 1026 rows (`fold=-1`) were included in training.  After the fix, manual
  and automatic windows agree.  The blockers are only 23 straight annotations
  and a small, overfit-prone feature set.  Next: expand the sheet, run
  repeated grouped CV with nested feature selection, and keep `skel_resid`
  as the primary candidate.

### Snow-spray branch (2026-11-12)

`build_spray_features.py` → `spray_features.csv` (234 × 50) measures the powder
thrown around the skis at impact.  It masks the skier with the pose keypoints
and tracks snow texture / bright-pixel fraction / background residual in a band
around the feet.  Windows are oracle-free (late 60–100% vs air 15–55%).

* **50 annotations:** `spray_tex_all_ratio` r = +0.43 with annotated landing
  falls; pose + spray leave-session-out AUC 0.80 / acc 0.77 vs pose 0.75 / 0.71.
  But this sample is stratified and over-represents falls.
* **Full dataset (193 usable, 27 falls):** no classifier gain — OOF AUC 0.665
  with pose + spray vs 0.663 for pose alone.  The 1026 session has only 2 falls,
  so the external split says nothing.
* The most repeatable single variable in the main sessions is
  `spray_resid_burst_dur_s` (r = −0.33): clean landings keep a longer residual
  burst.  The texture ratio is weak (r ≈ +0.13).
* The measure has **no camera-motion compensation**, so panning inflates it.
  This is the first thing to fix before trusting it.

Reports: `SPRAY_FEATURE_EVAL.md`, `SPRAY_FALL_EVAL.md`.

The leak-free table `jump_features_cal_leakfree.csv` is the correct input for
any grouped-CV experiment; `jump_features_cal.csv` is for descriptive use only.

See `JUMP_FEATURES_REPORT.md` for the schema and the full result.

## Competition clips (2026-11-13)

`competition_data/` adds 193 broadcast Big Air runs from 7 competitions
(Olympics 2022, X Games 2023/2025/2026) with per-run `score_100`, `fall` and
`session_group`.  91 of them have published scores; the rest are being filled.

* `dataset_sources.py` builds one label table from `labels_CV.csv` and
  `competition_data/long_labels.csv` (`labels_combined.csv`, 427 clips), resolves
  videos, keeps `session_group`, and allows unscored clips (pose extraction does
  not need the target).
* `CV_project_rebuilt.ipynb` now uses it.  `INCLUDE_AQA` /
  `INCLUDE_COMPETITION` select sources; `MAX_CLIPS_PER_SOURCE` is a smoke-test
  limit; `OVERWRITE_POSE=False` resumes.
* The competition clips are 1280x720 wide shots.  At the old YOLO `imgsz=640`
  the tracker lost the skier (e.g. 132/839 frames); at **`imgsz=1280`** the same
  clip tracked 755/839 frames and pose covers 839/839.  The notebook therefore
  uses `POSE_IMGSZ_AQA=640` and `POSE_IMGSZ_COMPETITION=1280`.
* **Spectators/officials:** `track_mode="longest"` is the default for **both**
  AQA and competition.  It collects every plausible person (dropping boxes with
  too few keypoints or an implausible shape), samples person seeds throughout
  the clip, and keeps the best track.  The old entry-box Viterbi could start on
  a spectator and stay there.  Existing AQA pose arrays need a one-time re-track
  (`OVERWRITE_POSE_AQA=True`).
* **Camera cuts:** broadcast competition clips cut 1-4 times per run (start
  close-up, wide jump, landing close-up), so no single track can cross the clip.
  `CUT_AWARE=True` detects hard cuts and tracks inside each shot, **keeping every
  shot**.  There is no primary-shot / temporal-window selection anymore;
  `shot_select` is accepted for compatibility but ignored.  Black leaders are
  not counted as cuts.  A per-clip `<clip>_pose_meta.json` records cut frames,
  per-shot stats and the gate/prior configuration.  AQA is not cut-free either:
  13 of 234 clips have a real mid-clip cut (`048`, `058`, `096`, `113`, and nine
  `1026 (1)(2)-*` broadcast clips); all other AQA clips are single-shot.  See
  `competition_data/README.md`; `aqa_cut_clips.csv` lists the affected AQA clips.
* **Why the old shot selector was removed:** `"frames"` picked whichever shot
  tracked the skier longest, often the landing/interview shot rather than the
  jump.  On `xg_aspen_2023_women_061` it chose a 13-frame window (17% coverage);
  `"jump"` / `"motion"` found the jump but still discarded the rest of the clip.
  Keeping every shot removes this failure mode.  On `xg_aspen_2026_men_002` the
  skier is too small/dark in the wide jump shot at 1280, so that clip still
  needs better detection (tiling/upscaling) rather than a selector.
* Cost: roughly 2-4 minutes per competition clip at 1280 on CPU; the full
  competition set is an overnight run.  Start with `MAX_CLIPS_PER_SOURCE=2`.
* **Physical association gate (2026-11-14):** `temporal_tracker.py` now rejects
  a candidate link when the box would move further than physics allows inside
  one camera shot:
  - consecutive frames and short gaps (up to **3 missed frames**): centre move
    ≤ **1.5 body sizes per frame** (with a 5% frame-width floor and a 35% cap)
    and apparent size change ≤ `e^1.0` per frame;
  - longer gaps: only the apparent-size cap (`e^2.0`, i.e. 7.4×) is applied;
  - transitions spanning a detected camera cut are always allowed.
  A post-Viterbi split is the safety net.  Each `<clip>_pose_meta.json` records
  `gate_checks`, `gate_rejects`, `gate_reject_frac`, `gate_splits` and the
  `gate_config`.  `probe_track_gate.py` → `track_gate_probe.csv` compares gate
  vs no-gate on the candidate paths (pass 1 only, no pose refinement).
  Validation:
  - `xg_aspen_2023_women_061`: no-gate path had consecutive switches at
    0→1 (3.80 bodies, size 186→341) and 94→95 (3.32 bodies, size 380→85);
    the gate path is 228 vs 230 frames, both removed.
  - `xg_aspen_2025_men_012`: no-gate had a 478→479 switch (3.89 bodies,
    size 45→490); the gate path is 562 vs 571 frames and the switch is gone.
  - `xg_aspen_2026_men_026`: unchanged; its largest jumps are across 9–13-frame
    gaps (~0.9 frame width, ~1.2–1.8 body/frame), which are physically
    plausible and therefore allowed.
  - `oly_beijing_2022_men_026`: full pipeline unchanged (67 tracked frames,
    pose coverage 79.2% of the selected shot; 1781/15563 candidate links
    rejected, 0 path splits).  *This is before the whole-clip change below.*
  - six AQA clips (`160`, `106`, `149`, `161`, `21`, `56`): chosen path
    unchanged; the gate only rejects implausible alternative candidates.
* **Person selection without "longest" (2026-11-14):** the gate removes identity
  switches, but it does not choose *which* person is the skier.
  `choose_person_track` (alias `choose_longest_track`) now takes
  `select="longest" | "motion" | "size"`.  `"motion"` picks the
  largest vertical travel relative to body size, i.e. the jump, among tracks
  with at least **80% of the longest track's frames** (`PERSON_STABILITY_FRAC`).
  The floor is necessary: without it a sparse 37-frame path in
  `oly_beijing_2022_men_026` had a higher apparent vertical range (7.47) than
  the real 66-frame skier path (6.97) and won.  With the floor:
  - `oly_beijing_2022_men_026`: longest 67 frames / 79.2% coverage; motion
    66 frames / **82.6%** coverage.
  - `xg_aspen_2023_women_061`: motion 224 vs longest 228 frames, slightly
    higher vertical range (1.41 vs 1.36).
  - `xg_aspen_2025_men_012`: motion and longest pick the same 562-frame path.
  - AQA `106`: longest 22 frames / 69.9% coverage; motion 22 / 66.0%, so AQA
    keeps `"longest"`.
  Defaults in the notebook: `PERSON_SELECT_AQA="longest"`,
  `PERSON_SELECT_COMPETITION="motion"`.  `size` is available but picks a
  close-up person on AQA `106`, so it is not recommended.
* **Whole-clip tracking, smooth-motion prior, centre prior (2026-11-15):** the
  detector no longer selects a temporal window.
  - `choose_shot_tracks` tracks every camera shot independently and returns
    the **union** of all shots.  `shot_select` is accepted for backward
    compatibility but ignored; there is no primary-shot selection.
  - Seeds are sampled every `seed_seconds` through the shot, not only in the
    first two seconds, so a late-entering skier can become a seed.
  - **Centre prior:** `_center_penalty` is a true central square
    (`CENTER_SQUARE_RADIUS=0.5`, i.e. the central 50% is free) with
    `CENTER_PRIOR_WEIGHT=3.0`.  It enters the DP emission cost and is a
    tie-breaker when choosing the person inside a shot.
  - **Smooth-motion prior:** pass 1 produces a path; a 5-frame median filter
    followed by a Savitzky-Golay filter (`SMOOTH_WINDOW=9`,
    `SMOOTH_POLYORDER=2`) convolves it into a smooth trajectory prediction;
    pass 2 re-runs the DP with that prediction (`PRIOR_WEIGHT=0.6`).  Each
    shot is smoothed independently, so a camera cut is never bridged.
  - `interpolate_boxes` / `interpolate_keypoints` also skip gaps that span a
    camera cut.  Coverage is measured over the full tracked range, not over a
    selected shot.
  - Validation (new pipeline, `person_select="motion"` for competition):
    - `oly_beijing_2022_men_026`: 3 shots; old primary `50-257` (66 frames,
      171 pose, 82.6%); new all shots 173 raw / **282 pose, 89.5%**.  Visual
      check follows the same skier from close-up → wide jump → landing
      close-up.
    - `xg_aspen_2023_women_061`: old primary `274-718` (13 frames, 76 pose,
      17.1%); new 560 raw / **599 pose, 60.5%** across 4 shots.  Visual check
      follows the skier through close-up skis, wide jump, landing and the
      final close-up.
    - `xg_aspen_2026_men_001`: 4 shots, **837/839 pose frames (100%)**; visual
      check follows the skier from the intro group shot through the jump and
      the wide landing shot.
    - AQA with real cuts: `048` 87/103 (84.5%), `058` 101/103 (98.1%),
      `096` 103/103 (100%), `113` 103/103 (100%); both shots kept.
    - AQA without cuts: `160` 100%; `106` 78/103 (75.7%, was 69.9%);
      `021` 93/103 (90.3%, was 71.8%).
  - Cached pose arrays still need a full re-track:
    `OVERWRITE_POSE_AQA=True` / `OVERWRITE_POSE_COMPETITION=True`.
* **Edge-portrait filter (2026-11-15):** broadcast clips often contain a
  head-and-shoulders athlete graphic at the left/right edge.  YOLO scores it as
  a person (face + shoulders, no hips/legs), and it can dominate a track because
  it is large and static.  `_looks_like_edge_portrait` drops a candidate only if
  it touches the left/right/top edge, is at least 15% of the frame width, has at
  most 9 visible keypoints, and has no hips/knees/ankles.  A/B
  (`probe_portrait_filter.py` → `portrait_filter_probe.csv`):
  - `xg_aspen_2026_men_026`: 932 → 664 candidates, 268 portraits dropped; the
    unfiltered path spent ~45 frames on the portrait, the filtered path follows
    the skier (386 raw frames; visual check
    `ski_visual_checks/portrait_fixed_xg026.jpg`).
  - AQA cost: `048` 87→86 pose frames, `106` 78→76; all other tested clips
    unchanged (`058`, `096`, `113`, `160`, `021`).  Competition `oly`,
    `xg_aspen_2023_women_061` and `xg_aspen_2026_men_001` unchanged.
* **Soft appearance prior — tested, not enabled (2026-11-15):**
  `_appearance_descriptor` builds Lab `a/b` + exposure-normalised `L`
  histograms for upper/lower body halves inside the pose hull;
  `_appearance_cost` is a histogram-intersection distance used as an optional
  pairwise DP term (`APPEARANCE_WEIGHT`, `APPEARANCE_MAX_GAP=30`, skipped
  across cuts).  A/B (`probe_appearance.py`, `_track_visual.py`):
  - weights 0.1/0.25/0.3 leave the path unchanged on
    `xg_aspen_2023_women_061` (990 frames, 4 shots);
  - weight 0.5 switches the last ~30 frames from the skier close-up to a
    spectator in the crowd (the skier's spray changes the histogram; the crowd
    person matches better).  On `oly` and `xg_aspen_2026_men_026` it only drops
    a few frames (jaccard 0.985, same person 100%).
  - Conclusion: no gain.  `ENABLE_APPEARANCE=False` / `APPEARANCE_WEIGHT=0.0`;
    the code is kept for future descriptor work (torso-only or multi-reference
    appearance), not for production tracking.

* **Cut detection on visually similar wide shots (2026-11-16):** the brightness test
  (`diff > 35` **and** histogram correlation `< 0.5`) misses a cut between two snowy
  wide shots, which is the normal competition structure.  `detect_cuts` now also runs
  a texture test: ORB descriptor match ratio `< CUT_TEXTURE_MATCH_MIN (0.10)` means a
  different scene, evaluated only when the frame pair already differs by
  `> CUT_TEXTURE_MIN_DIFF (12)` mean-abs, and skipped when either frame has fewer than
  `CUT_TEXTURE_MIN_FEATURES (20)` features (motion blur, night sky) so a whip-pan
  cannot fake a cut.  `process_video_pose_temporal` now keeps a 320x180 grayscale pass
  for it; camera shifts still use the 160x90 pass.  On `xg_aspen_2023_women_066`
  (869 frames) the cuts are `42, 349, 837` (was `42, 837`); ORB gives 0.027 at 349 vs
  0.40-0.80 inside a shot, and nothing fires in the 540-620 whip-pan.
* **`xg_aspen_2023_women_066` is still not fixed (2026-11-16).**  After the cut fix it
  is 4 shots / 416 tracked frames / 503 pose frames (58%), i.e. unchanged in coverage,
  and the jump is still missing.  Cause, measured with the candidate collector:
  - the skier **is** detected 42-470 (approach) and 620-837 (landing ride-out,
    conf 0.5-0.86);
  - on ~80% of frames 440-620 (mid-air, inverted silhouette, then a whip-pan) YOLO
    returns **no candidate at all**; 2x2 tiling and `imgsz=1920` do not recover it;
  - the only candidates in that stretch are floodlight towers (conf 0.11-0.34), which
    the DP chains into the athlete: approach -> tower -> landing becomes one 241-frame
    "chimera" (body8_frac 0.99, because YOLO invents a full pose on a tower).
  A selector, gate or shot-bookkeeping change cannot fix that; it needs a different
  candidate source (dark-silhouette blob detector for night runs, or an
  ROI-upscaled pass centred on the jump).  Note the *leading* close-up (0-42) is
  already its own shot after the cut at 42, so "skip the useless intro" does not
  recover the jump - the dropout is inside the 349-837 shot.

* **Same-shot tail recovery (2026-11-16):** `choose_shot_tracks` kept exactly one
  Viterbi path per camera shot, so a run whose athlete appears twice inside one shot
  (in-run, then the landing ride-out after the jump) lost the later segment whenever
  the earlier one was longer.  `_recover_shot_tail` now tracks the uncovered tail of a
  shot as a second segment when it covers `>= TAIL_RECOVERY_MIN_FRAMES (45)` frames, its
  median detection confidence is `>= TAIL_RECOVERY_MIN_CONF (0.40)`, and
  `>= TAIL_RECOVERY_MIN_BODY8 (0.50)` of its frames have 8 body keypoints.  The shot
  record gains `tail_recovered: [start, end]`.  The confidence floor is what keeps the
  floodlight/tower/banner false positives (conf 0.11-0.34) out of the output; a real
  athlete sits at 0.5-0.9.
  - Validation, `xg_aspen_2023_women_056` (869 frames, cuts `56, 651, 831`): shot 2
    `56-651` recovered its tail `[414, 651]`, so coverage goes 612 -> **849 pose
    frames (70% -> 98%)**, track 532 -> 628 frames.  The boxes now sit on the skier
    from the takeoff (500-530, previously undetected) through the landing ride-out
    (548-608), checked frame by frame.
  - Validation, `xg_aspen_2023_women_066`: shot 3 `349-837` (the mid-air shot) recovers
    its tail `[498, 837]`, so coverage goes 416 -> **842 pose frames (58% -> 97%)**,
    track 416 -> 654 frames.  Checked frame by frame: the recovered boxes ride the
    red-suited athlete down the landing (560-820); no floodlight/tower segment is
    appended, so the 0.40 confidence floor holds.
  - The tail pass costs no extra detection (it reuses the pass-1 candidates); the
    extra wall time is pose refinement on the newly boxed frames.  On this CPU an
    869-frame clip takes ~25-50 min once most frames carry a box.
  - Still open on 056: frames 394-413 (skier behind the ramp) remain empty, and the
    few frames where he is a blurred sliver mid-jump are only partly covered - the
    genuinely undetectable stretch (no YOLO candidates at any tiling) needs the
    alternative candidate source described above.

* **Detection-failure triage (2026-11-16):** `flag_detection_gaps` (in
  `temporal_tracker.py`) reads only the cached meta/`_ski.npz` files, so it costs
  seconds for the whole set and tracks nothing.  For every clip it reports the missing
  runs of the bbox track: `pose_coverage`, `longest_gap_frames` plus its `start`/`end`,
  `n_gaps`, `gap_frac`, the camera shot the longest gap sits in (`gap_shot`, i.e. the
  part worth watching) and a `severity` - `fail` under 0.60 coverage or one missing run
  of 60+ frames, `warn` under 0.85 / 30+ frames, `missing` when the clip has no cache.
  Driven by
  `py -3.11 run_retrack_competition.py --all --flag-gaps [N]`, which writes
  `roi_pose_dataset/detection_review.csv` (worst first) and prints the top N.  Current
  state of the 193 cached competition clips: **119 fail, 7 warn, 67 ok**; the largest
  dropouts are inside single long camera shots (e.g. `xg_aspen_2026_men_003` gap
  169-818 of its `0-819` shot), which is exactly the case the tail recovery above
  addresses.

* **A failing clip's cache can be stale - rebuild it before diagnosing (2026-11-16).**
  `xg_aspen_2026_men_003` looked like "`detect_cuts` never finds the camera change and
  then the tracker loses the skier when he gets small": its cached meta says
  `cuts [819]` (2 shots) while `detect_cuts` on the same frames returns `338, 819`
  (3 shots).  Re-tracked with the current code the clip goes 249 -> **730 pose frames
  (28% -> 81%)**, track 218 -> 602, and `bbox` runs become `0-168` + `338-898`; the
  boxes sit on the athlete through the jump, the landing ride-out and the closing
  close-up (checked frame by frame).  So before blaming the detector, re-run the clip:
  a meta file can predate both the cut fix and the tail recovery above.
  - What is genuinely left in this clip is `169-337`, the wide night in-run where the
    athlete is 20-30 px: the probe returns 0 raw / 0-1 tiled candidates at frames 240,
    300, 400, 430, 445, 455.  That is the same missing candidate source as 066.
* **A third cut test was tried and rejected (2026-11-16).**  Phase-correlation peak
  collapse (a cut has no single global translation) combined with a single-frame diff
  prominence test was measured on the 40 worst cached clips: it only shifted one
  existing cut by a frame (`oly_beijing_2022_men_083`, 35 -> 34) and added one
  spurious cut inside a flashing sequence (`xg_aspen_2026_women_007`, 1526).  Not
  worth the regression risk, so `detect_cuts` keeps its brightness + texture tests.

Pose outputs go to `roi_pose_dataset/<clip_id>_pose.npy` and `_pose21.npy`.
Downstream feature scripts still read `clip_manifest.csv` / `splits_grouped.csv`,
which remain AQA-only; extending them to the competition clips is the next step.

## Decisions / invariants (2026-11-15)

These are settled; do not reopen without new data.

| topic | decision |
|---|---|
| Feature philosophy | Compact semantic extractor (`build_jump_features.py`), not generic stat tables; emit confidence/source flags |
| Spin vs flip | Never sum; spin = yaw/vertical axis, flip = somersault/lateral axis |
| Airtime | Only usable via calibrated Ridge (`airtime_cal_s`); use `jump_features_cal_leakfree.csv` for grouped CV |
| Person selection | `track_mode="longest"` default; `PERSON_SELECT_AQA="longest"`, `PERSON_SELECT_COMPETITION="motion"`; camera-compensated ground-motion separates moving skiers from stationary spectators; relaxed stability floor (0.50) for late-entering skiers |
| Shot selection | None — `choose_shot_tracks` keeps all shots and returns the union; `shot_select` accepted for compatibility only; `primary_span` = full tracked range |
| Seed sampling | Every 2 s throughout each shot (not only first 2 s) |
| Interpolation across cuts | Never — `interpolate_boxes`/`interpolate_keypoints` skip cut-spanning gaps |
| Centre prior | Soft: central 50% square free, linear penalty outside, `CENTER_PRIOR_WEIGHT=3.0`; enters Viterbi emission and person-selection tie-breaker |
| Smooth-motion prior | 5-frame median + Savitzky–Golay (`SMOOTH_WINDOW=9`, `SMOOTH_POLYORDER=2`); second DP pass `PRIOR_WEIGHT=0.6`; each shot smoothed independently |
| Physical gate | ≤1.5 body sizes/frame (scaled with steps up to `MAX_GATE_GAP=15`); cuts exempt; post-split stitching via `_merge_path_segments` |
| Bidirectional tracking (Point 1) | `viterbi_track` tracks both forward (`t0 → T`) and backward (`t0 → 0`) from seed; recovers athlete trajectory before seed sampling frame |
| Tiling (Point 2) | `ENABLE_TILING=True`: 2×2 grid, 20% overlap, NMS merge; triggers when full-frame cands <1 or size <50 px |
| Cross-shot identity (Point 5) | `cross_shot_check=True`: compares per-shot centre and colour against reference shot; re-selects alternative seeds if inconsistent |
| AQA re-track | Full re-track required (tracker changed); `OVERWRITE_POSE_AQA=True` |
| 3D pose | Not pursued: monocular yaw worst-recovered, low pose coverage, only ~50 annotations |
| Spin frequency | Ellipse parametric orbit fit (`orbit_fit.py`) works; naive phase unwrap and peak counting do not |
| Cork | Not deployable: corrected AUC borderline (all/RF 0.676, core3 0.534), permutation p≥0.05; expand straight annotations before retrying |
| Shoulder x-swap | Total-rotation proxy (rho +0.43 vs total turns), not a spin counter |
| Vertical y-swap | Total-rotation proxy (rho +0.34), not flip-specific; ratio spin/flip AUC 0.50–0.54 → useless |
| Appearance prior | `ENABLE_APPEARANCE=False`; weights ≤0.3 have no effect, 0.5 hurts on spray-change shots; kept for future torso-only/multi-reference work |
| Minimum model bar | No model shipped unless it beats the train-mean dummy |
| Automatic air window | IoU 0.32 / recall 0.35 vs manual — do not trust window-dependent features until fixed |

## Next move (2026-11-15)

1. **Re-track everything** — run pipeline cell 3 with
   `OVERWRITE_POSE_AQA=True` / `OVERWRITE_POSE_COMPETITION=True`.  Cached
   pose arrays are still from the old (single-shot) tracker.
2. **Rebuild feature tables** — `build_jump_features.py`,
   `build_cork_features.py`, `build_simple_motion_features.py`, then the
   evaluations (`eval_cork_classifier.py`, `eval_simple_motion.py`).
3. **(Optional) Appearance prior prototype** — `ENABLE_APPEARANCE=False` gate +
   `probe_appearance.py`; A/B on `xg_aspen_2023_women_061`,
   `xg_aspen_2026_men_026`, `oly_beijing_2022_men_026`, AQA `106`/`021`.
4. **Finish competition pose extraction** — `INCLUDE_AQA=False`,
   `MAX_CLIPS_PER_SOURCE=0`; then `build_simple_motion_features.py` and
   `eval_simple_motion.py --dataset competition`.
5. **Extend manifests and feature scripts** — `clip_manifest.csv` /
   `splits_grouped.csv` and feature scripts to competition clips, splitting by
   `session_group`.
6. **Fix automatic air-window estimator** — target IoU > 0.7 vs manual windows
   before trusting any window-dependent feature (spin orbit, cork,
   airtime calibration).
7. **Expand cork/straight sheet** — ≥50 straight clips; then repeated grouped
   CV (5×5) + permutation with nested feature selection; keep `skel_resid` as
   primary candidate.
8. **Balanced flip-direction set** — ≥20 front flips before attempting direction
   classification.

## Important context

- `labels_combined.csv` — 427 clips: 234 AQA + 193 competition; 325 scored; 66 falls.
- Automatic air window vs manual: median IoU only 0.32, recall 0.35 — limits
  airtime/flip/spin/cork features based on the automatic window.
- `xg_aspen_2026_men_002`: skier too small/dark in wide jump shot (~13 px
  vertical range) — needs tiling/upscaling, not a person-selector fix.
- AQA cuts: 71 black-leader false positives ignored; 13 real cuts in
  `aqa_cut_clips.csv` (048, 058, 096, 113, nine 1026-session clips).
- Full-clip poses now include intro/outro shots — feature scripts that divide by
  total frame count will give different numbers; use `cut_frames`, `shots`, and
  air windows instead.
- Temp scripts `_swap_compare.py` and `_vis_proto.py` remain in the folder;
  `_vis_proto.py` is unreferenced.
- No git repository — all changes are uncommitted and unrecoverable from
  history.  Read this file before quoting any number.

