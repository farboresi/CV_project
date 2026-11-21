"""Contact sheet of a tracked clip from the saved v3 outputs (no re-tracking)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SKEL = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def sheet(video, pose_dir, out, step=None, cols=6, w=400):
    stem = Path(video).stem
    meta = json.load(open(Path(pose_dir) / f"{stem}_pose_meta.json"))
    kps = np.load(Path(pose_dir) / f"{stem}_pose.npy")
    T = meta["num_frames"]; step = step or max(1, T // 36)
    cap = cv2.VideoCapture(str(video)); tiles = []
    for t in range(5, T, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t); ok, f = cap.read()
        if not ok:
            break
        b = meta["bbox"][t]
        if b:
            cv2.rectangle(f, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 4)
            k = kps[t]
            for a, c in SKEL:
                if k[a, 2] > 0.3 and k[c, 2] > 0.3:
                    cv2.line(f, (int(k[a, 0]), int(k[a, 1])), (int(k[c, 0]), int(k[c, 1])), (0, 255, 255), 3)
        else:
            cv2.putText(f, "no skier", (20, 690), 0, 1.6, (0, 0, 255), 4)
        cv2.putText(f, str(t), (1150, 700), 0, 1.4, (255, 255, 255), 4)
        tiles.append(cv2.resize(f, (w, int(w * f.shape[0] / f.shape[1]))))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    sheet(sys.argv[1], sys.argv[2], sys.argv[3])
