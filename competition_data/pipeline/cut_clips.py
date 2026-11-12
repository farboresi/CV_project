"""Cut the reviewed live jumps into clips + a labels file.

Input:  long_cuts/review_labels.csv   (exported from review/index.html)
Output: long_clips/<clip_id>.mp4      one clip per kept jump (H.264, 720p, no audio, +0.5 s padding)
        long_clips/long_labels.csv    clip_id,athlete,attempt,fall,score,session_group,video,start,end
session_group = competition, so a whole competition stays on one side of the split.
Notes from the review are applied first (see apply_note): hand-noted end times get RIDE_AWAY
seconds added, missing live runs noted on dropped rows become clips.
Already-cut clips are skipped. Usage: python3 cut_clips.py [time_budget_s]
"""
import csv, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; OUT = ROOT / "long_clips"; OUT.mkdir(exist_ok=True)
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
import re
RIDE_AWAY = 3.0   # seconds added after a hand-noted end time (notes mark the landing, not the ride-away)
TS = r"(\d{1,2}(?::\d{2}){1,2}(?:\.\d+)?)"

def to_s(t):
    s = 0.0
    for p in t.split(":"): s = s * 60 + float(p)
    return s

NAME_FIX = {"colby stevensen": "Colby Stevenson", "oliver magnusson": "Oliwer Magnusson"}   # typos -> official spelling

def apply_note(r):
    """Read review notes: 'actual end: 0:16:41' -> end = that + RIDE_AWAY;
    'actual start: ...' -> start; 'LIVE MISSING 6:37-6:55' on a dropped row -> new kept clip;
    'replay after 12s' -> end = start + 12."""
    note = r.get("note", "") or ""
    r["athlete"] = NAME_FIX.get(" ".join(r.get("athlete", "").lower().split()), r.get("athlete", "").strip())
    m = re.search(r"start\D{0,3}" + TS, note, re.I)
    if m: r["start"] = str(to_s(m.group(1)))
    m = re.search(r"end\D{0,3}" + TS, note, re.I)
    if m: r["end"] = str(to_s(m.group(1)) + RIDE_AWAY)
    m = re.search(r"replay after\s*(\d+(?:\.\d+)?)\s*s", note, re.I)
    if m: r["end"] = str(float(r["start"]) + float(m.group(1)))
    m = re.search(TS + r"\s*[-–]\s*" + TS, note)          # any 'a-b' range = the live run's start-end
    if m and r["status"] == "keep":
        r["start"], r["end"] = str(to_s(m.group(1))), str(to_s(m.group(2)) + RIDE_AWAY)
    m = re.search(r"missing\D{0,3}" + TS + r"\s*[-–]\s*" + TS, note, re.I)
    if m and r["status"] != "keep":
        r.update(status="keep", start=str(to_s(m.group(1))), end=str(to_s(m.group(2)) + RIDE_AWAY),
                 clip_id=r["clip_id"] + "_live")
    if note: print(f"note {r['clip_id']:28s} -> {r['status']:4s} {float(r['start']):8.1f}-{float(r['end']):8.1f}  ({note})")
    return r

rows = [apply_note(r) for r in csv.DictReader(open(HERE / "review_labels.csv", encoding="utf-8-sig"))]
rows = [r for r in rows if r["status"] == "keep"]
# a run recovered from a 'LIVE MISSING' note that overlaps a clip already kept is a duplicate
rows = [r for r in rows if not (r["clip_id"].endswith("_live") and any(
    o is not r and o["video"] == r["video"] and min(float(o["end"]), float(r["end"])) - max(float(o["start"]), float(r["start"])) > 3
    for o in rows))]
EXTRA = HERE / "review_xg2025men_labels.csv"            # second review page (X Games 2025 men re-detection)
if EXTRA.exists():
    rows += [r for r in (apply_note(r) for r in csv.DictReader(open(EXTRA, encoding="utf-8-sig"))) if r["status"] == "keep"]
with open(OUT / "long_labels.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["clip_id", "athlete", "attempt", "fall", "score", "session_group", "video", "start", "end"])
    for r in rows:
        w.writerow([r["clip_id"], r["athlete"], r["attempt"], r["fall"], r["score"], r["competition"], r["video"], r["start"], r["end"]])
t0 = time.time(); done = 0
for r in rows:
    out = OUT / f"{r['clip_id']}.mp4"
    if out.exists(): done += 1; continue
    if time.time() - t0 > BUDGET: break
    a = max(0, float(r["start"]) - 0.5); d = float(r["end"]) - float(r["start"]) + 1.0
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", str(ROOT / "long_videos" / f"{r['video']}.mp4"),
                    "-t", f"{d:.2f}", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-y", str(out)], check=True)
    done += 1
print(f"{done}/{len(rows)} clips in long_clips/ ; labels in long_clips/long_labels.csv")
