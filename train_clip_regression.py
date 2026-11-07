"""Phase 2: grouped-CV tabular baselines for AQA-7 target regression.

Active input
------------
``pose_features.csv`` — the body-only table produced by
``build_clip_features.py --body-only``.  Ski columns are absent; the extra
non-ROI group is the arm/tuck/grab-proxy block, labelled ``body+arms``.

Evaluation design
-----------------
* CV: GroupKFold on ``session_group`` over the usable clips, **excluding the
  1026 session**.  Each 1026 clip is a segment of one long recording, so it is
  held out entirely as an external test set (train on other sessions -> test on
  1026).  This is the honest hard split.
* Feature ablations: body only, body + arms, full core, core + quality metadata.
* Every model sees identical folds.  ``dummy`` (train mean) is the reference.
* Metrics: MAE / RMSE / Spearman / R2 overall, plus MAE split by fall status and
  Spearman on clean (non-fall) clips.
* Optional two-stage fall-aware regressor: OOF fall probability is added as a
  feature, and an expected-value model mixes a fall mean and a clean regressor.

Outputs
-------
    pose_baseline_results.csv      one row per (feature set, model, split)
    pose_baseline_folds.csv        per-fold metrics
    pose_baseline_oof.csv          OOF predictions (best feature sets)
    pose_baseline_summary.txt      readable report
    pose_baseline_mae.png          MAE comparison plot
    checkpoints/pose_regression_model.joblib    final fitted pipeline
    checkpoints/pose_regression_features.json   feature contract + metadata

The final model is fitted on all usable non-1026 clips (train+val) after the
evaluation.  Auto-selection uses the best CV MAE among body / body+arms / core;
the quality-only set is excluded so the deployed model does not learn QA
shortcuts.

Usage
-----
    python train_clip_regression.py
    python train_clip_regression.py --features pose_features.csv
    python train_clip_regression.py --features clip_features.csv   # legacy
    python train_clip_regression.py --no-save-model                # evaluation only
"""
from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
BOX_SESSION = "s_1026"


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to train-fold quantiles (robust to ratio blow-ups)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_model(kind):
    if kind == "dummy":
        return DummyRegressor(strategy="mean")
    if kind == "ridge":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("win", Winsorizer()),
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=30.0)),
        ])
    if kind == "rf":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", RandomForestRegressor(
                n_estimators=600, min_samples_leaf=4, max_features=0.25,
                n_jobs=-1, random_state=SEED)),
        ])
    if kind == "hgb":
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("m", HistGradientBoostingRegressor(
                max_iter=400, learning_rate=0.04, max_leaf_nodes=7,
                min_samples_leaf=12, l2_regularization=5.0,
                random_state=SEED)),
        ])
    raise ValueError(kind)


def make_forest_classifier():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("m", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=10, l2_regularization=5.0, random_state=SEED)),
    ])


# --------------------------------------------------------------------------- #
# feature groups
# --------------------------------------------------------------------------- #
def feature_groups(df):
    numeric = list(df.select_dtypes(include=[np.number]).columns)
    meta = {"target", "fall"}
    quality = {
        "body8_frac", "pose_det_frac", "ankle_frac", "bbox_frac",
        "pose_conf_mean", "bbox_h_frac_mean", "bbox_h_px_mean",
        "median_bbox_h_px", "median_bbox_h_frac", "size_score", "qa_score",
        "n_frames", "duration", "fps", "width", "height",
        "video_frames", "entry_offset",
    }
    counts = {c for c in numeric if c.endswith("_n")}
    core = [c for c in numeric if c not in meta | quality | counts]

    def is_ski(c):
        return ("ski" in c) or c.startswith("grab") or c.startswith("tuck")

    def is_roi(c):
        return c.startswith("roi") or "bbox_h" in c

    body = [c for c in core if not is_ski(c) and not is_roi(c)]
    body_extra = [c for c in core if not is_roi(c)]
    # With the active body-only table there are no ski columns: the extra group
    # is the arm/grab/tuck block instead.
    extra_tag = "body+ski" if any("ski" in c for c in core) else "body+arms"
    return {
        "body": body,
        extra_tag: body_extra,
        "core": core,
        "core+quality": core + [c for c in quality if c in numeric],
    }


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def metrics(y, p, fall=None):
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "spearman": float(spearmanr(y, p).correlation) if np.std(p) > 1e-9 else np.nan,
        "pearson": float(pearsonr(y, p)[0]) if np.std(p) > 1e-9 else np.nan,
        "r2": float(r2_score(y, p)),
    }
    if fall is not None:
        fall = np.asarray(fall).astype(int)
        for name, mask in (("fall", fall == 1), ("clean", fall == 0)):
            if mask.sum() >= 3:
                out[f"mae_{name}"] = float(mean_absolute_error(y[mask], p[mask]))
            else:
                out[f"mae_{name}"] = np.nan
        if (fall == 0).sum() >= 8 and np.std(p[fall == 0]) > 1e-9:
            out["spearman_clean"] = float(spearmanr(y[fall == 0], p[fall == 0]).correlation)
        else:
            out["spearman_clean"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate_cv(train_df, features, tag, n_splits=5):
    train_df = train_df.reset_index(drop=True)
    X = train_df[features].astype(float)
    y = train_df["target"].to_numpy()
    fall = train_df["fall"].to_numpy()
    groups = train_df["session_group"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    kinds = ["dummy", "ridge", "rf", "hgb"]
    oof = {k: np.full(len(train_df), np.nan) for k in kinds}
    fold_rows = []
    for fold, (tr, va) in enumerate(folds):
        for kind in kinds:
            model = make_model(kind)
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            oof[kind][va] = pred
            fold_rows.append({"set": tag, "model": kind, "fold": fold,
                              "split": "cv", **metrics(y[va], pred, fall[va])})

    rows = [{"set": tag, "model": kind, "split": "cv",
             **metrics(y, p, fall)} for kind, p in oof.items()]
    return rows, fold_rows, oof


def evaluate_external(train_df, test_df, features, tag):
    Xtr = train_df[features].astype(float)
    ytr = train_df["target"].to_numpy()
    Xte = test_df[features].astype(float)
    yte = test_df["target"].to_numpy()
    fall_te = test_df["fall"].to_numpy()
    rows, preds = [], {}
    for kind in ["dummy", "ridge", "rf", "hgb"]:
        model = make_model(kind)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        preds[kind] = pred
        rows.append({"set": tag, "model": kind, "split": "external-1026",
                     **metrics(yte, pred, fall_te)})
    return rows, preds


def fall_two_stage(train_df, features, n_splits=5):
    """OOF fall probability + expected-value regression."""
    df = train_df.reset_index(drop=True)
    X = df[features].astype(float)
    y = df["target"].to_numpy()
    fall = df["fall"].to_numpy().astype(int)
    groups = df["session_group"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    p_fall = np.full(len(df), np.nan)
    clean_pred = np.full(len(df), np.nan)
    fall_mean = np.nan
    for tr, va in folds:
        clf = make_forest_classifier()
        clf.fit(X.iloc[tr], fall[tr])
        p_fall[va] = clf.predict_proba(X.iloc[va])[:, 1]

        clean_mask = fall[tr] == 0
        if clean_mask.sum() >= 10:
            reg = make_model("rf")
            reg.fit(X.iloc[tr][clean_mask], y[tr][clean_mask])
            clean_pred[va] = reg.predict(X.iloc[va])
            fall_mean = float(np.mean(y[tr][~clean_mask])) if (~clean_mask).any() else float(np.mean(y[tr]))

    expected = p_fall * fall_mean + (1.0 - p_fall) * clean_pred
    auc = roc_auc_score(fall, p_fall) if len(np.unique(fall)) > 1 else np.nan
    rows = [
        {"set": "two-stage", "model": "fall-prob", "split": "cv",
         "n": len(df), "mae": np.nan, "rmse": np.nan, "spearman": np.nan,
         "pearson": np.nan, "r2": np.nan, "auc_fall": float(auc)},
        {"set": "two-stage", "model": "clean-only rf", "split": "cv",
         **metrics(y, clean_pred, fall), "auc_fall": float(auc)},
        {"set": "two-stage", "model": "expected-value", "split": "cv",
         **metrics(y, expected, fall), "auc_fall": float(auc)},
    ]
    return rows, p_fall, expected


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def write_summary(results, folds, path, feature_counts, n_cv, n_ext):
    lines = ["Clip-level regression baselines", "=" * 78, ""]
    lines.append(f"CV clips (no 1026 session): {n_cv}")
    lines.append(f"external 1026 clips: {n_ext}")
    lines.append("")
    lines.append("Feature sets: " + ", ".join(f"{k}={v}" for k, v in feature_counts.items()))
    lines.append("")
    order = ["body", "body+ski", "body+arms", "core", "core+quality", "two-stage"]
    for tag in [t for t in order if t in set(results["set"])]:
        sub = results[(results["set"] == tag)]
        lines.append(f"--- {tag} " + "-" * (70 - len(tag)))
        cols = ["model", "split", "n", "mae", "mae_clean", "mae_fall",
                "spearman", "spearman_clean", "r2"]
        if "auc_fall" in sub.columns and sub["auc_fall"].notna().any():
            cols.append("auc_fall")
        show = sub[[c for c in cols if c in sub.columns]].copy()
        show = show.round(3)
        lines.append(show.to_string(index=False))
        lines.append("")

    cv = results[results.split == "cv"]
    if len(cv):
        best = cv.sort_values("mae").iloc[0]
        dummy = cv[cv.model == "dummy"]
        dummy_mae = float(dummy["mae"].mean()) if len(dummy) else np.nan
        lines.append(f"best CV MAE: {best['mae']:.2f} ({best['set']} / {best['model']})")
        lines.append(f"mean dummy MAE: {dummy_mae:.2f} "
                     f"-> improvement {dummy_mae - best['mae']:+.2f}")

    ext = results[results.split == "external-1026"]
    if len(ext):
        best_ext = ext.sort_values("mae").iloc[0]
        dummy_ext = ext[ext.model == "dummy"]
        dummy_ext_mae = float(dummy_ext["mae"].mean()) if len(dummy_ext) else np.nan
        lines.append(f"best external MAE: {best_ext['mae']:.2f} "
                     f"({best_ext['set']} / {best_ext['model']})")
        lines.append(f"external dummy MAE: {dummy_ext_mae:.2f} "
                     f"-> improvement {dummy_ext_mae - best_ext['mae']:+.2f}")
    lines.append("")
    lines.append("Interpretation guide: if best MAE is within +/-1 point of the dummy,")
    lines.append("the current features carry no reliable clip-level signal.")
    lines.append("")
    fold_std = folds.groupby(["set", "model"])["mae"].agg(["mean", "std"]).round(2)
    lines.append("Per-fold MAE stability (CV):")
    lines.append(fold_std.to_string())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print("[WARN] matplotlib unavailable:", exc)
        return

    sub = results[(results.split == "cv") & (results.model != "dummy")]
    sub = sub[sub.set != "two-stage"]
    if sub.empty:
        return
    tags = list(dict.fromkeys(sub["set"]))
    models = ["ridge", "rf", "hgb"]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    width = 0.8 / len(models)
    x = np.arange(len(tags))
    for i, model in enumerate(models):
        vals = [sub[(sub.set == t) & (sub.model == model)]["mae"].mean() for t in tags]
        ax.bar(x + i * width, vals, width, label=model)
    dummy = results[(results.split == "cv") & (results.model == "dummy")]["mae"].mean()
    ax.axhline(dummy, color="k", linestyle="--", linewidth=1, label="dummy mean")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(tags, rotation=15)
    ax.set_ylabel("CV MAE")
    ax.set_title("Clip-level regression MAE by feature set (1026 held out)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# final model + feature contract
# --------------------------------------------------------------------------- #
AUTO_FINAL_SETS = ("body", "body+arms", "core")


def choose_final_model(results_df, requested_model="auto", requested_set="auto"):
    """Pick the final (feature_set, model) pair.

    ``auto`` uses the best CV MAE among the non-dummy, non-quality-only sets.
    Quality metadata (``core+quality``) is excluded from auto selection so the
    deployed model does not learn clip-QA shortcuts.
    """
    cv = results_df[(results_df["split"] == "cv")
                    & (results_df["model"] != "dummy")
                    & (results_df["set"] != "two-stage")].copy()
    if cv.empty:
        raise SystemExit("no CV results available to choose a final model")

    if requested_set == "auto":
        cv = cv[cv["set"].isin(AUTO_FINAL_SETS)]
    else:
        cv = cv[cv["set"] == requested_set]
    if requested_model != "auto":
        cv = cv[cv["model"] == requested_model]
    if cv.empty:
        raise SystemExit(f"no CV result for model={requested_model!r}, "
                         f"set={requested_set!r}")
    best = cv.sort_values("mae").iloc[0]
    return str(best["set"]), str(best["model"]), best


def save_final_model(cv_df, groups, results_df, tag, kind, model_path,
                     contract_path, feature_table, requested_model,
                     requested_set):
    cols = groups[tag]
    X = cv_df[cols].astype(float)
    y = cv_df["target"].to_numpy(dtype=float)

    model = make_model(kind)
    model.fit(X, y)

    model_path = Path(model_path)
    contract_path = Path(contract_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    row = results_df[(results_df["set"] == tag)
                     & (results_df["model"] == kind)
                     & (results_df["split"] == "cv")].iloc[0]
    dummy_rows = results_df[(results_df["set"] == tag)
                            & (results_df["model"] == "dummy")
                            & (results_df["split"] == "cv")]
    dummy_mae = float(dummy_rows["mae"].iloc[0]) if len(dummy_rows) else np.nan
    margin = (float(row["mae"]) - dummy_mae) if np.isfinite(dummy_mae) else None

    contract = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "feature_table": str(feature_table),
        "feature_columns": list(cols),
        "n_features": int(len(cols)),
        "feature_set": tag,
        "model_kind": kind,
        "selection": {"requested_model": requested_model,
                      "requested_set": requested_set,
                      "policy": "best CV MAE; quality-only sets excluded from auto"},
        "metrics_cv": {
            "n": int(row["n"]),
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "spearman": float(row["spearman"]) if pd.notna(row["spearman"]) else None,
            "r2": float(row["r2"]),
            "dummy_mae": dummy_mae,
            "margin_over_dummy": margin,
        },
        "training": {
            "n_clips": int(len(cv_df)),
            "sessions": sorted(cv_df["session_group"].astype(str).unique().tolist()),
            "filenames": cv_df["filename"].astype(str).tolist(),
            "held_out_session": BOX_SESSION,
            "target_mean": float(np.mean(y)),
            "target_std": float(np.std(y, ddof=1)) if len(y) > 1 else 0.0,
            "target_min": float(np.min(y)),
            "target_max": float(np.max(y)),
            "falls": int(cv_df["fall"].fillna(0).astype(int).sum()),
        },
        "environment": {
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "input_requirements": {
            "file": "CSV with one row per clip",
            "required_columns": ["filename"],
            "feature_columns_must_exist": True,
            "missing_values": "imputed with the training median inside the model",
            "do_not_use": ["target", "fall", "session_group", "split", "usable"],
        },
        "notes": "Body-only pose regression. Do not train or tune on the 1026 "
                 "session; it is the external test set.",
        "selection_warning": (
            "CV MAE is worse than the train-mean dummy; predictions are weak "
            "and should be treated as a baseline, not a reliable scorer."
            if margin is not None and margin > 0 else None),
    }
    contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return contract


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="pose_features.csv")
    parser.add_argument("--results", default="pose_baseline_results.csv")
    parser.add_argument("--folds", default="pose_baseline_folds.csv")
    parser.add_argument("--oof", default="pose_baseline_oof.csv")
    parser.add_argument("--summary", default="pose_baseline_summary.txt")
    parser.add_argument("--plot", default="pose_baseline_mae.png")
    parser.add_argument("--save-model", default="checkpoints/pose_regression_model.joblib")
    parser.add_argument("--save-contract",
                        default="checkpoints/pose_regression_features.json")
    parser.add_argument("--final-model", default="auto",
                        choices=("auto", "ridge", "rf", "hgb", "dummy"))
    parser.add_argument("--final-feature-set", default="auto",
                        choices=("auto", "body", "body+arms", "core", "core+quality"))
    parser.add_argument("--no-save-model", action="store_true",
                        help="evaluate only; do not fit or write the final bundle")
    args = parser.parse_args()

    df = pd.read_csv(args.features, dtype={"session_group": str})
    required = {"filename", "target", "fall", "session_group", "usable"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{args.features} is missing required columns: "
                         f"{sorted(missing)}")
    df["session_group"] = df["session_group"].astype(str)
    ski_cols = [c for c in df.columns if "ski" in c.lower()]
    if ski_cols:
        print(f"[WARN] legacy table with {len(ski_cols)} ski columns; the active "
              f"pipeline expects a body-only table from "
              f"`build_clip_features.py --body-only`.")
    else:
        print(f"body-only feature table: {len(df.columns)} columns, no ski features")
    groups = feature_groups(df)
    print("feature sets:", {k: len(v) for k, v in groups.items()})

    usable = df[df["usable"]].copy()
    cv_df = usable[usable["session_group"] != BOX_SESSION].reset_index(drop=True)
    ext_df = usable[usable["session_group"] == BOX_SESSION].reset_index(drop=True)
    print(f"usable={len(usable)} | cv={len(cv_df)} | external-1026={len(ext_df)}")

    results, fold_rows = [], []
    oof_best = {}
    for tag, cols in groups.items():
        print(f"\n[{tag}] {len(cols)} features")
        rows, folds, oof = evaluate_cv(cv_df, cols, tag)
        results += rows
        fold_rows += folds
        best = min(rows, key=lambda r: r["mae"])["model"]
        oof_best[tag] = (best, oof[best])
        ext_rows, _ = evaluate_external(cv_df, ext_df, cols, tag)
        results += ext_rows

    # fall-aware two-stage diagnostic on the full core set
    print("\n[two-stage fall-aware]")
    ts_rows, _, _ = fall_two_stage(cv_df, groups["core"])
    results += ts_rows

    results_df = pd.DataFrame(results)
    folds_df = pd.DataFrame(fold_rows)
    results_df.to_csv(args.results, index=False)
    folds_df.to_csv(args.folds, index=False)

    oof_out = cv_df[["filename", "session_group", "target", "fall"]].copy()
    for tag, (model, pred) in oof_best.items():
        oof_out[f"{tag}__{model}"] = pred
    oof_out.to_csv(args.oof, index=False)

    write_summary(results_df, folds_df, Path(args.summary),
                  {k: len(v) for k, v in groups.items()}, len(cv_df), len(ext_df))
    make_plot(results_df, args.plot)

    print("\n", results_df.round(3).to_string(index=False))
    print(f"\nwrote {args.results}, {args.folds}, {args.oof}, {args.summary}, {args.plot}")

    if not args.no_save_model:
        tag, kind, best = choose_final_model(
            results_df, requested_model=args.final_model,
            requested_set=args.final_feature_set)
        contract = save_final_model(
            cv_df, groups, results_df, tag, kind,
            model_path=args.save_model, contract_path=args.save_contract,
            feature_table=args.features,
            requested_model=args.final_model,
            requested_set=args.final_feature_set)
        print(f"\nfinal model: {kind} on `{tag}` ({len(groups[tag])} features)")
        print(f"  CV MAE {contract['metrics_cv']['mae']:.3f} "
              f"(dummy {contract['metrics_cv']['dummy_mae']:.3f})")
        print(f"  trained on {contract['training']['n_clips']} clips / "
              f"{len(contract['training']['sessions'])} sessions")
        print(f"  wrote {args.save_model}")
        print(f"  wrote {args.save_contract}")
        if contract.get("selection_warning"):
            print(f"  [WARN] {contract['selection_warning']}")


if __name__ == "__main__":
    main()
