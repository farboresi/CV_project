"""Does the detected take-off / apex / landing structure carry usable signal?

Leak-free protocol, identical for every feature set:

* dev  = ``splits_grouped.csv`` rows with split != "test", evaluated by the
  GroupKFold ``fold`` column already in the file (whole sessions per fold)
* test = the held-out 1026 session
* models are the notebook's tabular ones (dummy / ridge / hgb), same hyperparams

Feature sets compared:
    base              aggregate columns from ``pose_features.csv`` ("core")
    events            all recovered event columns
    base+events       both
    events:arc        ballistic-arc timing/geometry only
    events:rot        rotation-episode columns only
    events:phase      air-vs-ground phase slices + landing/take-off windows
    events:meta       tracking-quality proxies only (the confound check)
    events:no-meta    everything except those proxies

The active tables are body-only: no ski columns are read or reported.

Usage
-----
    python eval_event_features.py
    python eval_event_features.py --features pose_features.csv \
        --events pose_event_features.csv --out pose_event_eval.md
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from train_clip_regression import make_model, feature_groups

warnings.filterwarnings("ignore", category=RuntimeWarning)

META = {"target", "fall", "n_frames", "fps", "med_bbox_h_px", "width", "height",
        "ev_anchor"}   # ev_anchor is an ordinal code for which segmenter fired
QUALITY_PROXY = {"ev_track_frac", "ev_pace_mean", "ev_ok", "ev_rot_ok"}
N_BOOT = 2000
SEED = 42


def bootstrap_ci(y, p, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y, float), np.asarray(p, float)
    vals = [float(np.mean(np.abs(p[i] - y[i])))
            for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def scores(y, p):
    out = {"mae": float(np.mean(np.abs(p - y))),
           "rmse": float(np.sqrt(np.mean((p - y) ** 2)))}
    out["rho"] = (float(spearmanr(y, p)[0])
                  if len(y) > 2 and np.std(p) > 1e-12 else np.nan)
    return out


def evaluate(cols, dev, te, y_dev, y_te, kinds=("dummy", "ridge", "hgb")):
    """OOF over the grouped dev folds, then fit on all dev and score the test set."""
    rows = []
    for kind in kinds:
        oof = np.full(len(dev), np.nan)
        for fold in sorted(dev["fold"].unique()):
            tr = (dev["fold"] != fold).to_numpy()
            m = make_model(kind)
            m.fit(dev.loc[tr, cols].astype(float), y_dev[tr])
            oof[~tr] = m.predict(dev.loc[~tr, cols].astype(float))
        m = make_model(kind)
        m.fit(dev[cols].astype(float), y_dev)
        pred = m.predict(te[cols].astype(float))
        lo, hi = bootstrap_ci(y_te, pred)
        o, t = scores(y_dev, oof), scores(y_te, pred)
        rows.append({"model": kind, "n_feat": len(cols),
                     "oof_mae": o["mae"], "oof_rho": o["rho"],
                     "test_mae": t["mae"], "test_ci_lo": lo, "test_ci_hi": hi,
                     "test_rho": t["rho"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="pose_features.csv")
    ap.add_argument("--events", default="pose_event_features.csv")
    ap.add_argument("--splits", default="splits_grouped.csv")
    ap.add_argument("--out", default="pose_event_eval.md")
    args = ap.parse_args()

    base = pd.read_csv(args.features, dtype={"filename": str})
    ev = pd.read_csv(args.events, dtype={"filename": str})
    sp = pd.read_csv(args.splits, dtype={"filename": str})

    clash = ["target", "fall", "session_group", "split", "usable"]
    base = base.drop(columns=[c for c in clash if c in base.columns])
    ev = ev.drop(columns=[c for c in clash + ["n_frames", "fps", "stem"]
                          if c in ev.columns])
    df = sp.merge(base, on="filename", how="left").merge(ev, on="filename", how="left")
    df = df[df["usable"].astype(bool)].reset_index(drop=True)
    df["session_group"] = df["session_group"].astype(str)

    # feature groups must be taken from the aggregate table alone, otherwise the
    # event columns are absorbed into "base" and the comparison is meaningless
    base_cols = feature_groups(base)["core"]
    numeric = list(ev.select_dtypes(include=[np.number]).columns)
    ev_cols = [c for c in numeric if c not in META]

    arc_cols = [c for c in ev_cols if c.startswith(
        ("ev_arc", "ev_apex", "ev_to_", "ev_ld_", "ev_air", "ev_takeoff",
         "ev_landing", "ev_vy"))]
    rot_cols = [c for c in ev_cols if c.startswith("ev_rot")]
    phase_cols = [c for c in ev_cols if c not in arc_cols and c not in rot_cols
                  and c not in QUALITY_PROXY]
    sets = {
        "base": base_cols,
        "events": ev_cols,
        "base+events": base_cols + ev_cols,
        "events:arc": arc_cols,
        "events:rot": rot_cols,
        "events:phase": phase_cols,
        "events:meta": [c for c in ev_cols if c in QUALITY_PROXY],
        "events:no-meta": [c for c in ev_cols if c not in QUALITY_PROXY],
    }
    print(f"clips={len(df)} | " + " | ".join(f"{k}={len(v)}" for k, v in sets.items()))

    dev = df[df["split"] != "test"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    y_dev, y_te = dev["target"].to_numpy(float), te["target"].to_numpy(float)

    rows = []
    for tag, cols in sets.items():
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue
        for r in evaluate(cols, dev, te, y_dev, y_te):
            rows.append({"set": tag, **r})

    res = pd.DataFrame(rows)
    print("\n" + res.round(3).to_string(index=False))

    floor = float(np.mean(np.abs(y_dev.mean() - y_te)))
    det = df[df["ev_ok"] == 1]

    lines = ["# Event features: take-off / apex / landing, leak-free evaluation", "",
             f"* clips: {len(df)} usable (dev {len(dev)}, test {len(te)} from the 1026 session)",
             f"* base features: {len(base_cols)} aggregate columns from `pose_features.csv`",
             f"* event features: {len(ev_cols)} columns from `pose_event_features.csv`",
             "* identical folds, seeds and models across feature sets; test MAE carries a "
             f"{N_BOOT}-resample bootstrap 95% CI", "",
             f"Constant-floor test MAE (train mean): **{floor:.2f}**", "",
             "## Model comparison", "",
             "| feature set | model | #feat | dev OOF MAE | dev OOF rho | test MAE | "
             "test 95% CI | test rho |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rho = "-" if not np.isfinite(r["test_rho"]) else f"{r['test_rho']:+.3f}"
        lines.append(
            f"| {r['set']} | {r['model']} | {r['n_feat']} | {r['oof_mae']:.2f} | "
            f"{r['oof_rho']:+.3f} | {r['test_mae']:.2f} | "
            f"[{r['test_ci_lo']:.2f}, {r['test_ci_hi']:.2f}] | {rho} |")

    arcs = df[df["ev_arc_ok"] == 1]
    lines += ["", "## Detector diagnostics", "",
              f"* clips with a phase window: {len(det)}/{len(df)} "
              f"(rotation anchor {int((df['ev_anchor'] == 2).sum())}, "
              f"arc anchor {int((df['ev_anchor'] == 1).sum())})",
              f"* rotation episode found: {int(df['ev_rot_ok'].sum())}/{len(df)} clips",
              f"* ballistic arc accepted: {len(arcs)}/{len(df)} clips"]
    if len(det):
        lines += [f"* trick window duration: median {det['ev_airtime_s'].median():.2f} s "
                  f"(p05 {det['ev_airtime_s'].quantile(.05):.2f}, "
                  f"p95 {det['ev_airtime_s'].quantile(.95):.2f})",
                  f"* window centre position in clip: median "
                  f"{det['ev_apex_rel'].median():.2f}"]
    if len(arcs):
        iou = arcs["ev_rot_arc_iou"].dropna()
        lines += [f"* arc fit r2: median {arcs['ev_arc_r2'].median():.2f}, curvature "
                  f"contrast median {arcs['ev_arc_contrast'].median():.2f}"
                  + (f", arc/rotation IoU median {iou.median():.2f}" if len(iou) else ""),
                  ""
                  "The ballistic arc is **not** used as the anchor. On these clips the "
                  "operator pans to follow the skier, so the bbox vertical track is "
                  "dominated by camera motion: the fitted arc lands on the pan rather "
                  "than the jump (median airtime 0.48 s against a ~1 s expected flight, "
                  "and ~0 overlap with the rotation episode). `build_event_features.py "
                  "--anchor arc` reproduces it; the phases above are anchored on "
                  "rotation, which is invariant to camera translation and zoom."]

    lines += ["", "### Event features vs the fall label (a fall is a failed landing)", "",
              "| feature | mean clean | mean fall | Spearman rho | p |", "|---|---|---|---|---|"]
    f_all = df["fall"].to_numpy(int)
    for c in ("ev_airtime_s", "ev_arc_r2", "ev_arc_amp_rel", "ev_land_dscale_abs_max",
              "ev_land_dvy_max", "ev_land_knee_min", "ev_land_spin_std",
              "ev_rot_dur_s", "ev_rot_net_deg", "ev_rot_arc_iou", "knee_air_min",
              "speed_air_max", "ev_vy_symmetry"):
        if c not in df.columns:
            continue
        v = df[c].to_numpy(float)
        ok = np.isfinite(v)
        if ok.sum() < 20:
            continue
        rho, p = spearmanr(v[ok], f_all[ok])
        lines.append(f"| `{c}` | {np.nanmean(v[f_all == 0]):.2f} | "
                     f"{np.nanmean(v[f_all == 1]):.2f} | {rho:+.3f} | {p:.3f} |")

    def get(tag, kind="hgb"):
        r = res[(res["set"] == tag) & (res["model"] == kind)]
        return r.iloc[0] if len(r) else None

    bd, be, eo = get("base"), get("base+events"), get("events")
    lines += ["", "## Verdict", ""]
    if bd is not None and be is not None and eo is not None:
        lines += [
            f"* base (hgb): test MAE {bd['test_mae']:.2f} "
            f"[{bd['test_ci_lo']:.2f}, {bd['test_ci_hi']:.2f}], OOF {bd['oof_mae']:.2f}",
            f"* events only (hgb): test MAE {eo['test_mae']:.2f} "
            f"[{eo['test_ci_lo']:.2f}, {eo['test_ci_hi']:.2f}], OOF {eo['oof_mae']:.2f}",
            f"* base+events (hgb): test MAE {be['test_mae']:.2f} "
            f"[{be['test_ci_lo']:.2f}, {be['test_ci_hi']:.2f}] "
            f"(delta vs base {be['test_mae'] - bd['test_mae']:+.2f}), "
            f"OOF {be['oof_mae']:.2f} (delta {be['oof_mae'] - bd['oof_mae']:+.2f})",
        ]
    lines += ["", f"Deltas smaller than the bootstrap CI width are noise at {len(te)} "
              f"test clips; OOF deltas over {len(dev)} dev clips are the more stable "
              "comparison.", ""]

    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
