"""Run inference with the saved body-only pose regression bundle.

The bundle is written by ``train_clip_regression.py``:

    checkpoints/pose_regression_model.joblib
    checkpoints/pose_regression_features.json

The JSON contract lists the exact feature columns and the training metadata.
Input is any CSV that has those columns plus a ``filename`` column (normally
``pose_features.csv``).

Usage
-----
    python predict_pose_regression.py --features pose_features.csv
    python predict_pose_regression.py --features pose_features.csv `
        --out pose_predictions.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Winsorizer is defined in train_clip_regression.py; importing it is required for
# joblib to unpickle the saved pipeline.
from train_clip_regression import Winsorizer  # noqa: F401


def load_bundle(model_path, contract_path):
    model_path = Path(model_path)
    contract_path = Path(contract_path)
    if not model_path.is_file():
        raise SystemExit(f"model not found: {model_path}")
    if not contract_path.is_file():
        raise SystemExit(f"feature contract not found: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    model = joblib.load(model_path)
    return model, contract


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/pose_regression_model.joblib")
    ap.add_argument("--contract", default="checkpoints/pose_regression_features.json")
    ap.add_argument("--features", default="pose_features.csv")
    ap.add_argument("--out", default="pose_predictions.csv")
    ap.add_argument("--no-clip", action="store_true",
                    help="do not clip predictions to the 0-100 score range")
    args = ap.parse_args()

    model, contract = load_bundle(args.model, args.contract)
    cols = contract["feature_columns"]

    df = pd.read_csv(args.features)
    missing = [c for c in ["filename"] + cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{args.features} is missing {len(missing)} required column(s): "
            f"{missing[:10]}{' ...' if len(missing) > 10 else ''}")

    X = df[cols].astype(float)
    pred = np.asarray(model.predict(X), dtype=float)
    if not args.no_clip:
        pred = np.clip(pred, 0.0, 100.0)

    train_names = set(contract["training"].get("filenames", []))
    in_training = df["filename"].astype(str).isin(train_names).to_numpy()

    out = pd.DataFrame({"filename": df["filename"], "prediction": pred,
                        "in_training": in_training})
    for col in ("target", "fall", "usable", "session_group"):
        if col in df.columns:
            out[col] = df[col].to_numpy()

    out.to_csv(args.out, index=False)

    print(f"model    : {contract['model_kind']} on `{contract['feature_set']}` "
          f"({contract['n_features']} features)")
    print(f"trained  : {contract['training']['n_clips']} clips / "
          f"{len(contract['training']['sessions'])} sessions")
    print(f"CV MAE   : {contract['metrics_cv']['mae']:.3f} "
          f"(dummy {contract['metrics_cv']['dummy_mae']:.3f})")
    print(f"wrote    : {args.out} ({len(out)} rows)")

    if in_training.any():
        print(f"[NOTE] {int(in_training.sum())}/{len(out)} input clips were used "
              f"to fit this model; their metrics are in-sample.")

    if "target" in out.columns:
        def score(mask, label):
            if int(mask.sum()) < 3:
                return
            yy = out.loc[mask, "target"].to_numpy(dtype=float)
            pp = pred[mask]
            mae = float(np.mean(np.abs(yy - pp)))
            rho = (float(spearmanr(yy, pp).correlation)
                   if len(yy) > 2 and np.std(pp) > 1e-12 else float("nan"))
            print(f"{label:14s}: MAE {mae:.3f} | rho {rho:.3f} | n {int(mask.sum())}")

        score(np.ones(len(out), dtype=bool), "on input")
        out_of_sample = ~in_training
        if "usable" in out.columns:
            out_of_sample = out_of_sample & out["usable"].fillna(False).astype(bool)
        score(out_of_sample, "out-of-sample")
        if "session_group" in out.columns:
            is_1026 = out["session_group"].astype(str) == "s_1026"
            if "usable" in out.columns:
                is_1026 = is_1026 & out["usable"].fillna(False).astype(bool)
            score(is_1026.to_numpy(), "1026 session")


if __name__ == "__main__":
    main()
