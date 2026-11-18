import json
from pathlib import Path

import pandas as pd

labels = pd.read_csv("labels_combined.csv", dtype={"filename": str})
stems = ["006", "013", "016", "027", "007", "050", "048", "106", "058",
         "160", "1026 (1)(2)-40", "1026 (1)(2)-42", "1026 (1)(2)-33"]


def fmt(v, nd=2):
    return "None" if v is None else round(float(v), nd)


for stem in stems:
    row = labels[labels["video_path"].map(lambda p: Path(str(p)).stem == stem)]
    if row.empty:
        print(stem, "no label")
        continue
    p = Path("roi_pose_dataset") / f"{stem}_pose_meta.json"
    if not p.is_file():
        print(stem, "no meta")
        continue
    m = json.loads(p.read_text())
    shots = [(s["start"], s["end"], fmt(s.get("skier_score")),
              s.get("skier_likely")) for s in m.get("shots", [])]
    print(f"{stem:>18s} track {m.get('track_frames')} "
          f"pose {m.get('tracked_with_pose')}/{m.get('primary_span')} "
          f"cov {fmt(m.get('pose_coverage'))} shots {m.get('n_camera_shots')} "
          f"score {fmt(m.get('skier_score_best'))} "
          f"off {fmt(m.get('box_pose_offset_median'))} "
          f"b8 {fmt(m.get('body8_frac'))} lb {fmt(m.get('lower_body_frac'))} "
          f"info {shots}")
