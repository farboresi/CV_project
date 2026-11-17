# Clips + official judges' scores

`all_clips_scores.csv` — one row per clip, **392 clips, all with an official judges' score**:
175 AQA-7 ski clips + 217 live runs cut from 7 full broadcasts (Olympics 2022, X Games 2023/2025/2026).

Videos are not in git: competition clips are in `long_clips/<clip_id>.mp4` (shared via OneDrive),
AQA-7 clips are `AQA-7/Actions/ski_big_air/NNN.avi`.

| Column | Meaning |
|---|---|
| `dataset` | `competition` or `aqa7` |
| `clip_id`, `video_path` | clip name and video file |
| `competition` | event — **split by this** (hold out whole events) so a jump never appears in train and test |
| `athlete`, `attempt` | competition clips only |
| `fall` | 1 = went down on the landing |
| `official_score`, `score_scale` | score as published, out of 50 (AQA-7, X Games 2023) or 100 |
| `official_score_100` | **target to use**: score on 0–100 |
| `aqa_split4` | AQA-7 official train/test split |
