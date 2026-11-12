# Competition clips with real judges' scores

193 live Big Air ski jumps cut from 7 full broadcasts (one clip per run, replays removed),
plus the official AQA-7 ski scores. The videos themselves are **not in git** (901 MB):
they go in `long_clips/` next to this folder (shared via OneDrive) — see `.gitignore`.

| Competition (`session_group`) | Clips | Scores filled |
|---|---|---|
| `oly_beijing_2022_men` | 36 | 36 (official, Wikipedia/Olympedia) |
| `oly_beijing_2022_women` | 36 | 36 (official) |
| `xg_aspen_2023_women` | 40 | 19 (read off the broadcast scoreboard) |
| `xg_aspen_2025_women` | 29 | 0 — to fill |
| `xg_aspen_2025_men` | 7 | 0 — ~25 more runs still being reviewed |
| `xg_aspen_2026_men` | 27 | 0 — to fill |
| `xg_aspen_2026_women` | 18 | 0 — to fill |

## Files

- **`long_labels.csv`** — one row per clip in `long_clips/<clip_id>.mp4`:
  `clip_id, athlete, attempt, fall, score, session_group, video, start, end, score_100, score_source`
  - `score` = per-run score **as published**; `score_100` = rescaled to 0–100.
    X Games 2023 scored each run out of **50**; Olympics and X Games 2025/26 out of **100**.
  - Always use the **per-run** score, never the competition total (totals are built differently:
    Olympics = best 2 runs in different directions, X Games 2023 = best 2 runs, X Games 2025/26 = best single run).
  - `fall` = 1 if the skier went down on the landing (not a crash later in the run-out).
    Check: every Olympic run scored < 30 is labelled as a fall.
  - `session_group` = competition. **Split by this column** (GroupKFold / hold out a whole competition):
    the same jump must never be in train and test. Some X Games 2026 jumps also exist among our older Shorts.
  - `video,start,end` = source broadcast id and the run's time window in it (seconds).
- **`aqa7_ski_scores.csv`** — the official AQA-7 judges' scores for the 175 ski clips (Split-4, 0–50,
  `aqa_score_100` = ×2), next to the hand-made `labels_CV.csv` target for the same clip.
- `official_scores.csv` — the per-run scores found online/on broadcast (source of `score`).
- `scores_to_fill.csv` — clips still without a score; type the published score in `score`, then run
  `pipeline/merge_scores.py`.
- `review_labels.csv` — the manual review of the 442 auto-detected candidates (keep/drop, athlete, attempt, fall, notes).
- `long_videos.csv` — the 7 YouTube broadcasts (id, competition, title, URL).
- `pipeline/` — scripts that produced the clips (download → camera-cut detection → YOLO person
  features → run segmentation → review page → cutting → run-count check → score merge).
  They were run from `Project/long_cuts/` on Mariana's machine; paths are relative to that layout.
