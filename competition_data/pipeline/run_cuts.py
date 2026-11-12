"""Resumable camera-cut detection over the long broadcasts, in 10-minute segments.

Each call works until ~TIME_BUDGET seconds, then stops; re-run to continue.
Output: long_cuts/<id>_cuts.csv  (time_s, score)  - one row per detected camera cut
        long_cuts/progress.json    seconds processed per video
"""
import json, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VID = HERE.parent / "long_videos"
SEG = 600
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 140
prog_f = HERE / "progress.json"
prog = json.loads(prog_f.read_text()) if prog_f.exists() else {}
t0 = time.time()
for mp4 in sorted(VID.glob("*.mp4")):
    vid = mp4.stem
    dur = json.loads((VID / f"{vid}.info.json").read_text())["duration"]
    out = HERE / f"{vid}_cuts.csv"
    if not out.exists():
        out.write_text("time_s,score\n")
    while prog.get(vid, 0) < dur:
        if time.time() - t0 > BUDGET:
            prog_f.write_text(json.dumps(prog, indent=1)); print("paused", prog); sys.exit(0)
        start = prog.get(vid, 0)
        tmp = HERE / "_seg.txt"
        subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-threads", "4", "-ss", str(start), "-t", str(SEG),
                        "-i", str(mp4), "-an", "-vf",
                        f"scale=192:-2,select='gt(scene,0.30)',metadata=print:file={tmp}", "-f", "null", "-"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        txt = tmp.read_text() if tmp.exists() else ""
        rows = []
        for m in re.finditer(r"pts_time:([0-9.]+)\s*\n\s*lavfi\.scene_score=([0-9.]+)", txt):
            rows.append(f"{start + float(m.group(1)):.3f},{float(m.group(2)):.3f}")
        with out.open("a") as f:
            f.writelines(r + "\n" for r in rows)
        prog[vid] = start + SEG
        prog_f.write_text(json.dumps(prog, indent=1))
print("ALL DONE", prog)
