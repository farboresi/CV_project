"""train_v3 — score + fall models, leave-one-competition-out.

Every prediction is made on a competition the model never saw in training
(7 folds: oly_beijing_2022_men/women, xg_aspen_2023_women, 2025_men/women,
2026_men/women).  Imputation / scaling are fit inside each training fold.

Compares
  * dummy (training-fold mean)
  * OLD features (air_phase_features.csv + pose_features_blind.csv from the old tracker)
  * NEW features (features_v3.csv from skier_tracker_v3)
with ridge / random forest / gradient boosting, a fall classifier, and a
two-stage model:  score = P(fall)*mean_fall_score + (1-P(fall))*clean_regressor.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0


def regressors():
    return {
        "ridge": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)),
        "rf": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    RandomForestRegressor(n_estimators=400, min_samples_leaf=5, max_features=0.5,
                                                          random_state=SEED, n_jobs=2)),
        "hgb": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                     HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200,
                                                                   min_samples_leaf=10, random_state=SEED)),
    }


def classifiers():
    return {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),
        "rf": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                    RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced",
                                                           random_state=SEED, n_jobs=2)),
    }


def boot_ci(err, n=2000):
    rng = np.random.default_rng(SEED)
    m = [np.mean(rng.choice(err, len(err))) for _ in range(n)]
    return np.percentile(m, [2.5, 97.5])


def evaluate(df, feats, name):
    y = df["target"].values; fall = df["fall"].values; g = df["competition"].values
    X = df[feats].values.astype(float)
    groups = sorted(set(g))
    res = {}
    oof = {k: np.zeros(len(df)) for k in ["dummy", *regressors(), "two_stage"]}
    pf_oof = {k: np.zeros(len(df)) for k in classifiers()}
    for grp in groups:
        te = g == grp; tr = ~te
        oof["dummy"][te] = y[tr].mean()
        for k, mk in regressors().items():
            m = mk().fit(X[tr], y[tr]); oof[k][te] = m.predict(X[te])
        for k, mk in classifiers().items():
            c = mk().fit(X[tr], fall[tr]); pf_oof[k][te] = c.predict_proba(X[te])[:, 1]
        # two-stage
        c = classifiers()["logreg"]().fit(X[tr], fall[tr]); p = c.predict_proba(X[te])[:, 1]
        clean = tr & (fall == 0)
        rc = regressors()["ridge"]().fit(X[clean], y[clean])
        oof["two_stage"][te] = p * y[tr & (fall == 1)].mean() + (1 - p) * rc.predict(X[te])
    for k, pred in oof.items():
        err = np.abs(pred - y)
        lo, hi = boot_ci(err)
        rho = spearmanr(pred, y).correlation if k != "dummy" else np.nan
        per = {grp: float(err[g == grp].mean()) for grp in groups}
        res[f"score/{k}"] = dict(mae=float(err.mean()), ci=[float(lo), float(hi)], spearman=float(rho), per_group=per)
    for k, p in pf_oof.items():
        res[f"fall/{k}"] = dict(auc=float(roc_auc_score(fall, p)),
                                bal_acc=float(balanced_accuracy_score(fall, p > 0.5)))
    return res, oof, pf_oof


def table(results):
    rows = []
    for setname, res in results.items():
        for k, v in res.items():
            if k.startswith("score/"):
                rows.append(dict(features=setname, task="score", model=k[6:], MAE=round(v["mae"], 2),
                                 CI95=f"[{v['ci'][0]:.1f}, {v['ci'][1]:.1f}]", spearman=round(v["spearman"], 3)))
            else:
                rows.append(dict(features=setname, task="fall", model=k[5:], AUC=round(v["auc"], 3),
                                 bal_acc=round(v["bal_acc"], 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", default="features_v3.csv")
    ap.add_argument("--old-air", default="air_phase_features.csv")
    ap.add_argument("--old-blind", default="pose_features_blind.csv")
    ap.add_argument("--out", default="results_v3")
    a = ap.parse_args()

    new = pd.read_csv(a.new)
    new_feats = [c for c in new.columns if c not in ("clip_id", "competition", "target", "fall")]
    # old features for the same clips
    oa = pd.read_csv(a.old_air).rename(columns={"filename": "clip_id"})
    ob = pd.read_csv(a.old_blind).rename(columns={"filename": "clip_id"})
    drop = {"dataset", "target", "fall", "n_frames", "fps"}
    oa = oa[[c for c in oa.columns if c not in drop]]
    ob = ob[[c for c in ob.columns if c not in drop | {"pose_coverage", "n_camera_shots"}]]
    old = new[["clip_id", "competition", "target", "fall", "women"]].merge(oa, on="clip_id", how="left").merge(
        ob, on="clip_id", how="left")
    old_feats = [c for c in old.columns if c not in ("clip_id", "competition", "target", "fall")
                 and old[c].dtype != object]

    results = {}; preds = {}
    # ablation: drop anything about how the clip was cut / tracked (length, shots,
    # coverage, gender) so only body-pose evidence is left
    meta_cols = {"n_frames", "duration_s", "n_shots", "tracked_frac", "pose_frac", "trick_pos",
                 "tail_frames", "women", "trick_pose_frac", "post_trick_missing"}
    pose_only = [c for c in new_feats if c not in meta_cols]
    sets = [("OLD tracker+features", old, old_feats), ("NEW v3", new, new_feats),
            ("NEW v3 pose-only", new, pose_only)]
    for name, d, fs in sets:
        r, oof, pf = evaluate(d, fs, name)
        results[name] = r
        preds[name] = pd.DataFrame({"clip_id": d["clip_id"], "competition": d["competition"],
                                    "target": d["target"], "fall": d["fall"],
                                    **{f"pred_{k}": v for k, v in oof.items()},
                                    **{f"pfall_{k}": v for k, v in pf.items()}})
    t = table(results)
    pd.set_option("display.width", 200)
    print(f"n clips = {len(new)}, falls = {int(new['fall'].sum())}, competitions = {new['competition'].nunique()}")
    print(t.to_string(index=False))
    t.to_csv(f"{a.out}_table.csv", index=False)
    json.dump(results, open(f"{a.out}.json", "w"), indent=1)
    pd.concat([p.assign(features=k) for k, p in preds.items()]).to_csv(f"{a.out}_oof.csv", index=False)
