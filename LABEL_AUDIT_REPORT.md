# Label / near-duplicate audit

Date: 2026-11-06

## Question

The target table has 120 distinct values for 234 clips (56 values are shared by
170 clips). Are these repeated scores caused by duplicate footage, or are they
genuine ties from a coarse/discrete judge score?

## Method

`audit_clip_duplicates.py` compares every pair of clips using:

1. **md5** of the video file — exact duplicates / re-encodes.
2. **Visual descriptor** — 16 uniformly sampled frames, 32×32 grey, mean-centred,
   L2-normalised, averaged.
3. **Frame-level best match** — max similarity between any sampled frame of two
   clips; catches a time-shifted duplicate that the averaged descriptor misses.
4. **Pose-trajectory descriptor** — cached 17 COCO joints resampled to 32 bins,
   hip-centred and bbox-scaled.
5. A separate check of consecutive `1026 (1)(2)-*` segments for temporal overlap.

Outputs: `clip_duplicate_audit.csv`, `clip_duplicate_pairs.csv`,
`clip_duplicate_summary.txt`, `clip_duplicate_similarity.png`.

## Findings

**No exact duplicates.** md5 finds 0 duplicate files.

**Repeated targets are only slightly more similar than different targets.**

| Pair type | Visual sim | Frame match | Pose sim |
|---|---|---|---|
| Same target (228 pairs) | 0.492 | 0.636 | 0.381 |
| Different target (27,033 pairs) | 0.456 | 0.613 | 0.353 |
| Lift | +0.036 | +0.023 | +0.028 |

A true duplicate would sit near 1.0. Frame-level match reaches ≥0.995 for only
3 pairs, and only **one** of those shares a target (`62` vs `63`). The
similarity that does exist is dominated by the shared snow/venue background:
the top visual pairs are mostly different-target clips from the same camera.

**The 1026 session is not overlapping footage.** Consecutive segments show
frame matches of 0.01–0.10 at cuts (and lower at camera changes), so the 59
segments are distinct jumps from one recording, not repeated windows.

**Candidate repeat runs are rare.** Applying `same_target & frame_match ≥ 0.95 &
pose_sim ≥ 0.5`, plus the `62/63` pair (`frame_match 0.999`), gives only:

| Pair | Target | Visual | Frame match | Pose |
|---|---|---|---|---|
| 62 / 63 | 87.622 | 0.973 | 0.999 | 0.458 |
| 145 / 160 | 38.340 | 0.981 | 0.972 | 0.652 |
| 171 / 173 | 4.400 | 0.987 | 0.968 | 0.865 |
| 136 / 137 | 66.555 | 0.955 | 0.967 | 0.632 |

Even if all four were the same jump, that is 8 clips out of 234 — too few to
change the grouped-CV conclusions. `clip_duplicate_audit.csv` keeps their
nearest-neighbour scores so a sensitivity run can exclude them.

## Conclusion

Repeated targets are **not** repeated samples. They are consistent with a
discrete scoring scale / repeated judge averages, and the dataset remains 234
independent clips for modelling. `splits_grouped.csv` is adequate; no
target-based regrouping is needed.

## Practical follow-up

* Keep `splits_grouped.csv` (1026 held out, no session leaks).
* Optionally re-run the baseline excluding the four candidate pairs as a
  sensitivity check; expect unchanged conclusions.
* The main modelling bottleneck remains the **fall vs clean** signal, not
  duplicate samples.
