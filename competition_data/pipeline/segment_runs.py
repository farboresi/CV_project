"""Turn per-frame features into candidate jump clips (one per live run).

A frame is 'action' when no large person is on screen (max_h < 0.45: the skier is small on
the hill, not an interview/close-up), the frame is not the X Games blue wipe, and it is not black.
Action frames are merged allowing 1 s gaps; stretches of 5-40 s with few people (not crowd shots)
become candidates. A candidate bracketed by blue wipes on both sides is flagged as a replay.
Segments inside non-competition chapters (anthem, interviews, lifestyle...) are dropped.
Output: long_cuts/candidates.csv
"""
import csv, json, re, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent; VID = HERE.parent / "long_videos"
comp = {r["id"]: r["competition"] for r in csv.DictReader(open(HERE.parent / "long_videos_local.csv"), delimiter=";")}
SKIP = re.compile(r"opening|anthem|introduc|lifestyle|interview|conclusion|break|format|awards|closing", re.I)
rows_out = []
for f in sorted(HERE.glob("*_frames.csv")):
    vid = f.stem.replace("_frames", "")
    info = json.loads((VID / f"{vid}.info.json").read_text())
    chapters = [(c["start_time"], c["end_time"], c["title"]) for c in (info.get("chapters") or [])]
    fr = {}
    for r in csv.DictReader(open(f)):
        fr[float(r["t"])] = (int(r["n_person"]), float(r["max_h"]), float(r["blue"]), float(r["bright"]))
    ts = sorted(fr)
    wipe = [t for t in ts if fr[t][2] >= 0.5]
    act = [t for t in ts if fr[t][1] < 0.45 and fr[t][2] < 0.5 and fr[t][3] > 35]
    segs, cur = [], [act[0]]
    for t in act[1:]:
        if t - cur[-1] <= 1.5: cur.append(t)
        else: segs.append(cur); cur = [t]
    segs.append(cur)
    k = 0
    for s in segs:
        a, b = s[0], s[-1] + 0.5
        if not 5 <= b - a <= 40: continue
        ch = next((c for c in chapters if c[0] <= a < c[1]), None)
        if ch and SKIP.search(ch[2]): continue
        ns = [fr[t][0] for t in s]
        if st.median(ns) > 3: continue                      # crowd shot
        seen = sum(n > 0 for n in ns) / len(ns)
        if seen < 0.3: continue                             # skier almost never detected
        before = any(a - 8 <= w <= a + 1 for w in wipe); after = any(b - 1.5 <= w <= b + 2.5 for w in wipe)
        k += 1
        rows_out.append(dict(clip_id=f"{comp[vid]}_{k:03d}", video=vid, competition=comp[vid], start=round(a, 1), end=round(b, 1),
                             dur=round(b - a, 1), skier_seen=round(seen, 2), replay_flag=int(before and after),
                             chapter=ch[2] if ch else ""))
with open(HERE / "candidates.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0])); w.writeheader(); w.writerows(rows_out)
from collections import Counter
print(len(rows_out), "candidates;", sum(r["replay_flag"] for r in rows_out), "flagged replay")
for c, n in Counter(r["competition"] for r in rows_out).items():
    print(f"  {c:24s} {n:4d}  replay-flagged {sum(r['replay_flag'] for r in rows_out if r['competition']==c)}")
