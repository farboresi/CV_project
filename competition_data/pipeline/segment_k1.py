"""Re-segment X Games 2025 men (K1gPvmthfWw, night event) with looser rules.

The default segmenter dropped this broadcast: in-air shots are against a black sky (rejected
as 'black frame') and landings are filmed close-up (big person box, rejected as interview).
Here: no brightness rule, person boxes up to 0.75 of the frame height count as action, gaps up to
2.5 s are bridged. Candidates overlapping runs already kept in review_labels.csv are skipped.
Output: candidates_k1.csv (new ids xg_aspen_2025_men_b###)
"""
import csv, json, re, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent; VID = "K1gPvmthfWw"; COMP = "xg_aspen_2025_men"
MAXH, GAP, MIN_D, MAX_D = 0.75, 2.5, 6, 45
info = json.loads((HERE.parent / "long_videos" / f"{VID}.info.json").read_text())
chap = [(c["start_time"], c["end_time"], c["title"]) for c in info["chapters"]]
SKIP = re.compile(r"introduc|format", re.I)   # NOT awards: that chapter is mislabelled and covers minutes 16-62 of competition
fr = {float(r["t"]): (int(r["n_person"]), float(r["max_h"]), float(r["blue"])) for r in csv.DictReader(open(HERE / f"{VID}_frames.csv"))}
ts = sorted(fr)
act = [t for t in ts if fr[t][1] < MAXH and fr[t][2] < 0.5]
segs, cur = [], [act[0]]
for t in act[1:]:
    if t - cur[-1] <= GAP: cur.append(t)
    else: segs.append(cur); cur = [t]
segs.append(cur)
kept = [(float(r["start"]), float(r["end"])) for r in csv.DictReader(open(HERE / "review_labels.csv", encoding="utf-8-sig"))
        if r["video"] == VID and r["status"] == "keep"]
out, k = [], 0
for s in segs:
    a, b = s[0], s[-1] + 0.5
    if not MIN_D <= b - a <= MAX_D: continue
    ch = next((c for c in chap if c[0] <= a < c[1]), None)
    if ch and SKIP.search(ch[2]): continue
    if a > 3690: continue                                    # actual closing
    if st.median(fr[t][0] for t in s) > 3: continue
    if any(min(b, e) - max(a, s0) > 3 for s0, e in kept): continue      # already have it
    k += 1
    out.append(dict(clip_id=f"{COMP}_b{k:03d}", video=VID, competition=COMP, start=round(a, 1), end=round(b, 1),
                    dur=round(b - a, 1), skier_seen=round(sum(fr[t][0] > 0 for t in s) / len(s), 2), replay_flag=0,
                    chapter=ch[2] if ch else ""))
with open(HERE / "candidates_k1.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
print(len(out), "new candidates;", len(kept), "already kept")
for r in out: print(r["clip_id"], r["start"], r["end"], r["dur"], r["chapter"])
