"""Resumable per-frame features at 2 fps over the long broadcasts.

For every sampled frame: number of people (YOLO11n, conf>0.35), height of the largest
person box (fraction of frame height), blue-wipe fraction (X Games transition), brightness.
Output: long_cuts/<id>_frames.csv ; progress in long_cuts/frames_progress.json
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np, onnxruntime as ort, cv2

HERE = Path(__file__).resolve().parent
VID = HERE.parent / "long_videos"
MODEL = HERE.parent / "models" / "yolo11n.onnx"
FPS, W, H, SEG = 2, 640, 360, 300
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 140
only = sys.argv[2] if len(sys.argv) > 2 else None

sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def detect(rgb):
    pad = np.full((640, 640, 3), 114, np.uint8); pad[140:500] = rgb       # letterbox 640x360 -> 640x640
    x = pad.transpose(2, 0, 1)[None].astype(np.float32) / 255
    out = sess.run(None, {inp: x})[0][0]          # (84, 8400): cx,cy,w,h + 80 class scores
    conf = out[4]                                 # class 0 = person
    k = conf > 0.35
    if not k.any(): return 0, 0.0
    b = out[:4, k].T; c = conf[k]
    boxes = [[float(cx - w / 2), float(cy - h / 2), float(w), float(h)] for cx, cy, w, h in b]
    idx = cv2.dnn.NMSBoxes(boxes, c.tolist(), 0.35, 0.5)
    idx = np.array(idx).flatten()
    return len(idx), float(max(boxes[i][3] for i in idx) / H)

prog_f = HERE / "frames_progress.json"
prog = json.loads(prog_f.read_text()) if prog_f.exists() else {}
t0 = time.time()
for mp4 in sorted(VID.glob("*.mp4")):
    vid = mp4.stem
    if only and vid != only: continue
    dur = json.loads((VID / f"{vid}.info.json").read_text())["duration"]
    out = HERE / f"{vid}_frames.csv"
    while prog.get(vid, 0) < dur:
        if time.time() - t0 > BUDGET:
            prog_f.write_text(json.dumps(prog, indent=1)); print("paused", prog); sys.exit(0)
        start = prog.get(vid, 0)
        p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(start), "-t", str(SEG), "-i", str(mp4),
                            "-an", "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                           capture_output=True, check=True)
        fr = np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W, 3)
        rows = []
        for i, f in enumerate(fr):
            n, hmax = detect(f)
            r, g, b = (f[::4, ::4].astype(np.int16)[..., j] for j in range(3))
            blue = float(((b > 150) & (b - r > 60)).mean())
            rows.append(f"{start + i / FPS:.1f},{n},{hmax:.3f},{blue:.3f},{f.mean():.0f}\n")
        if start == 0: out.write_text("t,n_person,max_h,blue,bright\n")
        with out.open("a") as fh: fh.writelines(rows)
        prog[vid] = start + SEG
        prog_f.write_text(json.dumps(prog, indent=1))
print("ALL DONE", prog)
