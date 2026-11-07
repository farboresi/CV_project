"""Build a leak-free clip split for AQA-7 modelling.

The old ``splits.csv`` was a random clip split, so the 59 segments of the
``1026 (1)(2)-*`` recording were scattered across train and test.  This script
groups by recording session and:

* holds the whole 1026 session out as the external **test** set;
* splits the remaining sessions into train/val with ``GroupShuffleSplit``;
* adds 5 GroupKFold ``fold`` ids over the train+val clips for cross-validation.

Output
------
    splits_grouped.csv   filename, target, fall, session_group, split, fold

Usage
-----
    python build_grouped_splits.py
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

TEST_GROUP = "s_1026"
SEED = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="pose_features.csv")
    parser.add_argument("--out", default="splits_grouped.csv")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.features, dtype={"session_group": str})
    df["session_group"] = df["session_group"].astype(str)
    df = df[["filename", "target", "fall", "session_group", "usable"]].copy()

    is_test = df["session_group"] == TEST_GROUP
    test = df[is_test].copy()
    dev = df[~is_test].reset_index(drop=True).copy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=SEED)
    train_idx, val_idx = next(splitter.split(
        dev, groups=dev["session_group"].to_numpy()))
    dev["split"] = "train"
    dev.loc[val_idx, "split"] = "val"

    dev["fold"] = -1
    dev_folds = GroupKFold(n_splits=args.folds).split(
        dev, groups=dev["session_group"].to_numpy())
    for fold, (_, fold_val) in enumerate(dev_folds):
        dev.loc[fold_val, "fold"] = fold

    test["split"] = "test"
    test["fold"] = -1
    out = pd.concat([dev, test], ignore_index=True)
    out = out.sort_values(["split", "session_group", "filename"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)

    print(f"wrote {args.out}: {len(out)} clips")
    print(out.groupby("split").agg(
        clips=("filename", "size"),
        groups=("session_group", "nunique"),
        usable=("usable", "sum"),
        target_mean=("target", "mean"),
        falls=("fall", "sum"),
    ).round(2).to_string())

    # leak checks
    spans = out.groupby("session_group")["split"].nunique()
    print(f"\nsessions spanning >1 split: {int((spans > 1).sum())}")
    if (spans > 1).any():
        print(spans[spans > 1].to_string())

    print("\nfold sizes (train+val only):")
    print(out[out.split != "test"].groupby("fold").size().to_string())


if __name__ == "__main__":
    main()
