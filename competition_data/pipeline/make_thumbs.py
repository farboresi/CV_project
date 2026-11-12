"""Thumbnail strip per candidate: 4 frames across the run + a crop of the top-left name/attempt
overlay (taken mid-run, where the athlete bug is on screen). Resumable with a time budget."""
import csv, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; OUT = HERE / "review" / "thumbs"; VID = HERE.parent / "long_videos"
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 140
t0 = time.time()
CAND = sys.argv[2] if len(sys.argv) > 2 else "candidates.csv"
for r in csv.DictReader(open(HERE / CAND)):
    out = OUT / f"{r['clip_id']}.jpg"
    if out.exists(): continue
    if time.time() - t0 > BUDGET: print("paused"); sys.exit(0)
    a, b = float(r["start"]), float(r["end"]); src = str(VID / f"{r['video']}.mp4")
    ins, n = [], 0
    for fr in (0.15, 0.4, 0.65, 0.9):
        ins += ["-ss", f"{a + (b - a) * fr:.2f}", "-i", src]
    ins += ["-ss", f"{a + (b - a) * 0.5:.2f}", "-i", src]
    fc = ";".join(f"[{i}:v]trim=end_frame=1,scale=256:144[v{i}]" for i in range(4))
    fc += ";[4:v]trim=end_frame=1,crop=iw*0.3:ih*0.22:0:0,scale=-2:144[v4];[v0][v1][v2][v3][v4]hstack=inputs=5"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *ins, "-filter_complex", fc,
                    "-frames:v", "1", "-q:v", "4", "-y", str(out)], check=False)
print("ALL DONE")
