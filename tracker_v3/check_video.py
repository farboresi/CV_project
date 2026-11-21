"""check_video.py — run the v3 skier tracker on ONE video and see the result.

Usage (from the CV_Project folder):
    python tracker_v3/check_video.py long_clips/xg_aspen_2025_men_b002.mp4
    python tracker_v3/check_video.py xg_aspen_2025_men_b002          # clip name is enough
    python tracker_v3/check_video.py                                  # asks for it

Writes to  check_output/<clip>/ :
    <clip>_v3.mp4      annotated video  (green = chosen skier + skeleton,
                       grey = other people, red = rejected as fixed to the ground,
                       "no skier" when nobody qualifies)
    <clip>_sheet.jpg   contact sheet, ~36 frames
    <clip>_pose.npy / _pose_meta.json / _cam.npz   tracker output
and prints tracking stats, the jump features and the predicted score / fall.

Models: uses the OpenVINO models in tracker_v3/models (needs `pip install openvino`).
If openvino is missing it falls back to ultralytics + yolo11s.pt / yolo11s-pose.pt.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import skier_tracker_v3 as S  # noqa: E402
import features_v3 as F       # noqa: E402
from qa_sheet import sheet    # noqa: E402

PROJECT = HERE.parent


def find_video(arg: str) -> Path:
    p = Path(arg).expanduser()
    if p.is_file():
        return p
    stem = p.stem if p.suffix else p.name
    for folder in ("long_clips", "train_check", "YT_clips", "xg2025men"):
        for ext in (".mp4", ".avi", ".mov"):
            q = PROJECT / folder / f"{stem}{ext}"
            if q.is_file():
                return q
    hits = list(PROJECT.rglob(f"{stem}.mp4"))
    if hits:
        return hits[0]
    sys.exit(f"Video not found: {arg}")


def load_models():
    try:
        return S.load_models(HERE / "models")                    # OpenVINO (fast)
    except Exception as e:                                      # noqa: BLE001
        print(f"[info] OpenVINO models not usable ({e!r}); falling back to ultralytics")
        from ultralytics import YOLO
        return YOLO("yolo11s.pt"), YOLO("yolo11s-pose.pt"), YOLO("yolo11s.pt")


def predict(feats: dict, stem: str):
    """Score / fall prediction from models trained on features_v3.csv."""
    import pandas as pd
    fcsv = HERE / "features_v3.csv"
    if not fcsv.exists():
        return None
    train = pd.read_csv(fcsv)
    # a clip that is part of the training set gets its honest out-of-fold
    # prediction (model trained without its whole competition)
    oof = HERE / "results_v3_oof.csv"
    if oof.exists() and stem in set(train["clip_id"]):
        o = pd.read_csv(oof)
        o = o[(o["features"] == "NEW v3") & (o["clip_id"] == stem)]
        if len(o):
            r = o.iloc[0]
            return dict(score=float(r["pred_hgb"]), p_fall=float(r["pfall_rf"]),
                        true_score=float(r["target"]), true_fall=int(r["fall"]),
                        note="out-of-fold (model never saw this competition)")
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    cols = [c for c in train.columns if c not in ("clip_id", "competition", "target", "fall")]
    x = np.array([[feats.get(c, np.nan) for c in cols]], float)
    reg = make_pipeline(SimpleImputer(strategy="median"),
                        HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                      min_samples_leaf=10, random_state=0))
    clf = make_pipeline(SimpleImputer(strategy="median"),
                        RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                               class_weight="balanced", random_state=0))
    reg.fit(train[cols], train["target"]); clf.fit(train[cols], train["fall"])
    return dict(score=float(reg.predict(x)[0]), p_fall=float(clf.predict_proba(x)[0, 1]),
                note="model trained on all 217 clips")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="path or clip name, e.g. xg_aspen_2025_men_b002")
    ap.add_argument("--out", default=str(PROJECT / "check_output"))
    ap.add_argument("--no-video", action="store_true", help="skip the annotated video (faster)")
    ap.add_argument("--open", action="store_true", help="open the results when done (macOS)")
    a = ap.parse_args()
    video = find_video(a.video or input("Video path or clip name: ").strip().strip("'\""))
    stem = video.stem
    out = Path(a.out) / stem
    out.mkdir(parents=True, exist_ok=True)
    print(f"Video : {video}\nOutput: {out}\nTracking... (~1 min)")

    det, pose, tile = load_models()
    dbg = None if a.no_video else out / f"{stem}_v3.mp4"
    meta = S.process_clip(video, out, det, pose, str(dbg) if dbg else None, det_tile=tile)
    sheet_path = out / f"{stem}_sheet.jpg"
    sheet(video, out, str(sheet_path))

    T = meta["num_frames"]
    print("\n=== Tracking ===")
    print(f"frames {T}  ({T / meta['fps']:.1f} s @ {meta['fps']:.1f} fps)   camera cuts at {meta['cut_frames']}")
    print(f"skier boxed in {meta['track_frames']}/{T} frames, pose in {meta['pose_coverage']:.0%}")
    print(f"tracklets {meta['n_tracklets']}, rejected as static {meta['n_static_rejected']}")
    print("chosen tracklets (start-end, score, skis, ground excursion):")
    for c in meta["chain"]:
        print(f"   {c['start']:5d}-{c['end']:<5d} s={c['score']:.2f} skis={c['skis']:.2f} exc={c['excursion']:.1f}")
    gaps, t = [], 0
    bx = meta["bbox"]
    while t < T:
        if bx[t] is None:
            u = t
            while u < T and bx[u] is None:
                u += 1
            if u - t >= 15:
                gaps.append(f"{t}-{u - 1}")
            t = u
        else:
            t += 1
    print(f"'no skier' stretches (>=15 frames): {', '.join(gaps) or 'none'}")

    cam = np.load(out / f"{stem}_cam.npz")
    feats = F.clip_features(np.load(out / f"{stem}_pose.npy"), meta, cam)
    print("\n=== Jump features ===")
    for k in ("trick_inplane_turns", "trick_swap_turns", "trick_inverted_frac", "rot_episode_s",
              "tail_tilt_mean", "tail_lying_frac", "tail_hip_low_p10", "tail_aspect_p90", "land_tilt_max"):
        v = feats.get(k)
        print(f"   {k:22s} {v:.2f}" if v is not None and np.isfinite(v) else f"   {k:22s} n/a")
    json.dump({k: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)
               for k, v in feats.items()}, open(out / f"{stem}_features.json", "w"), indent=1)

    p = predict(feats, stem)
    if p:
        print("\n=== Prediction ===")
        print(f"   score  {p['score']:.1f} / 100      P(fall) {p['p_fall']:.2f}   [{p['note']}]")
        if "true_score" in p:
            print(f"   actual {p['true_score']:.1f}            fall = {p['true_fall']}")
    print(f"\nSaved: {sheet_path}" + (f"\n       {dbg}" if dbg else ""))
    if a.open and sys.platform == "darwin":
        subprocess.run(["open", str(sheet_path)] + ([str(dbg)] if dbg else []))


if __name__ == "__main__":
    main()
