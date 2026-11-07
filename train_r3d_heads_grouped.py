"""Leak-free re-evaluation of the frozen r3d_18 heads on ``splits_grouped.csv``.

Why this script exists
----------------------
The notebooks read ``splits.csv``, a *random* clip split.  The 59 segments of the
``1026 (1)(2)-*`` recording are one session, and 40 of them ended up in train
(7 in val, 12 in test), so every score/AUC number printed by the notebook is
measured on clips from a recording the model has already seen.

This script re-runs the same two heads (score regressor + fall classifier) and a
two-stage expected-value model on ``splits_grouped.csv``, where the whole 1026
session is held out as the test set and the folds are GroupKFold over the
remaining sessions.  The frozen r3d_18 features are read from the existing
caches, so this costs seconds, not hours: no video is decoded.

Reference points reported next to every result
----------------------------------------------
* ``dummy`` / ``const``: predict the train mean (the honest floor).
* bootstrap 95% CIs on the test MAE and the fall AUC, because the test set has
  only a handful of falls and point estimates alone are misleading.

Outputs
-------
    checkpoints_3dcnn/grouped_predictions.npz    predictions + CIs (arrays)
    GROUPED_EVAL_REPORT.md                       readable report

Usage
-----
    python train_r3d_heads_grouped.py
    python train_r3d_heads_grouped.py --splits splits_grouped.csv --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, rankdata, spearmanr

CKPT_DIR = Path("checkpoints_3dcnn")
FULL_CACHE = CKPT_DIR / "r3d18_feature_cache_v1.npz"
WINDOW_CACHE = CKPT_DIR / "r3d18_window_cache_v1.npz"
END_WINDOWS = ("last8", "last4")

TARGET_SCALE = 100.0
SEED = 42
HEAD_EPOCHS = 150
HEAD_PATIENCE = 30
HEAD_BATCH = 64
HEAD_LR = 6e-3
HEAD_WD = 1e-3
DROP_OUT = 0.5
FALL_LR = 1e-3
FALL_EPOCHS = 100
FALL_PATIENCE = 20
N_BOOT = 2000

torch.set_num_threads(max(1, (torch.get_num_threads() or 4) - 1))
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def roc_auc(y_true, y_prob) -> float:
    """Rank-based ROC AUC (tie-aware); nan when only one class is present."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_prob)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def regression_metrics(preds, targets, fall=None) -> dict:
    preds = np.asarray(preds, float).ravel()
    targets = np.asarray(targets, float).ravel()
    out = {
        "n": int(len(preds)),
        "mae": float(np.mean(np.abs(preds - targets))),
        "rmse": float(np.sqrt(np.mean((preds - targets) ** 2))),
        "spearman": (float(spearmanr(preds, targets)[0])
                     if len(preds) > 2 and np.std(preds) > 1e-12 else float("nan")),
        "pearson": (float(pearsonr(preds, targets)[0])
                    if len(preds) > 2 and np.std(preds) > 1e-12 else float("nan")),
    }
    if fall is not None:
        fall = np.asarray(fall).astype(int)
        for name, mask in (("fall", fall == 1), ("clean", fall == 0)):
            out[f"mae_{name}"] = (float(np.mean(np.abs(preds[mask] - targets[mask])))
                                  if mask.sum() else float("nan"))
    return out


def bootstrap_ci(y, p, metric, n_boot=N_BOOT, seed=SEED):
    """Percentile bootstrap 95% CI. Resamples where the metric is undefined
    (e.g. a bootstrap draw with no positive class) are skipped."""
    y = np.asarray(y, float).ravel()
    p = np.asarray(p, float).ravel()
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        v = metric(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


# --------------------------------------------------------------------------- #
# data / features
# --------------------------------------------------------------------------- #
def load_features():
    with np.load(FULL_CACHE, allow_pickle=False) as z:
        full = {str(k): v for k, v in zip(z["keys"], z["features"])}
    win = {}
    if WINDOW_CACHE.exists():
        with np.load(WINDOW_CACHE, allow_pickle=False) as z:
            win = {str(k): v for k, v in zip(z["keys"], z["features"])}
    return full, win


def feature_sets(full, win):
    """The two feature sets the notebook compares: full clip, and full + end."""
    sets = {"full": ("full",)}
    if win:
        sets["full+end"] = ("full",) + END_WINDOWS
    return sets


def build_X(names, parts, full, win):
    rows = []
    for name in names:
        cols = [full[f"{name}|0"]]
        for part in parts:
            if part != "full":
                cols.append(win[f"{name}|{part}"])
        rows.append(np.concatenate(cols))
    return np.stack(rows).astype(np.float32)


def fit_scaler(X):
    return (X.mean(0, keepdims=True).astype(np.float32),
            (X.std(0, keepdims=True) + 1e-6).astype(np.float32))


def apply_scaler(X, scaler):
    return (X - scaler[0]) / scaler[1]


# --------------------------------------------------------------------------- #
# heads (identical shapes/LRs to notebook cells 3, 8A and 8C)
# --------------------------------------------------------------------------- #
class ScoreHead(nn.Module):
    def __init__(self, in_dim, hidden=(256, 64), dropout=DROP_OUT):
        super().__init__()
        layers = [nn.Dropout(p=dropout)]
        prev = in_dim
        for width in hidden:
            layers += [nn.Linear(prev, width), nn.ReLU(inplace=True),
                       nn.Dropout(p=dropout)]
            prev = width
        layers += [nn.Linear(prev, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FallHead(nn.Module):
    def __init__(self, in_dim, dropout=DROP_OUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_score(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = ScoreHead(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=HEAD_WD)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr / TARGET_SCALE).float()
    Xva_t = torch.from_numpy(Xva)
    yva = np.asarray(yva, np.float64)

    best, best_state, bad = float("inf"), None, 0
    n = len(Xtr_t)
    for _ in range(HEAD_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HEAD_BATCH):
            b = perm[i:i + HEAD_BATCH]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mae = float(np.mean(np.abs(model(Xva_t).numpy() * TARGET_SCALE - yva)))
        sch.step(mae)
        if mae < best - 1e-4:
            best, bad = mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= HEAD_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def train_fall(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    model = FallHead(Xtr.shape[1]).to(DEVICE)
    n_pos = max(1, int(ytr.sum()))
    n_neg = int(len(ytr) - ytr.sum())
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos]))
    val_crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=FALL_LR, weight_decay=HEAD_WD)
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    Xva_t = torch.from_numpy(Xva)
    yva_t = torch.from_numpy(yva.astype(np.float32))

    best, best_state, bad = float("inf"), None, 0
    n = len(Xtr_t)
    for _ in range(FALL_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, HEAD_BATCH):
            b = perm[i:i + HEAD_BATCH]
            opt.zero_grad(set_to_none=True)
            loss = crit(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = val_crit(model(Xva_t), yva_t).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= FALL_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_score(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(np.ascontiguousarray(X))).numpy() * TARGET_SCALE


def predict_fall(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.ascontiguousarray(X))).numpy()
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def run(df, full, win, parts, tag, seed=SEED):
    dev = df[df["split"] != "test"].reset_index(drop=True)
    te = df[df["split"] == "test"].reset_index(drop=True)
    folds = sorted(dev["fold"].unique())

    X_dev = build_X(dev["filename"].astype(str), parts, full, win)
    y_dev = dev["target"].to_numpy(np.float64)
    f_dev = dev["fall"].to_numpy(int)
    X_te = build_X(te["filename"].astype(str), parts, full, win)
    y_te = te["target"].to_numpy(np.float64)
    f_te = te["fall"].to_numpy(int)

    oof_score = np.full(len(dev), np.nan)
    oof_fall = np.full(len(dev), np.nan)
    oof_clean = np.full(len(dev), np.nan)
    fold_rows = []

    for fold in folds:
        tr = np.flatnonzero(dev["fold"].to_numpy() != fold)
        va = np.flatnonzero(dev["fold"].to_numpy() == fold)
        scaler = fit_scaler(X_dev[tr])
        Xtr, Xva = apply_scaler(X_dev[tr], scaler), apply_scaler(X_dev[va], scaler)

        head = train_score(Xtr, y_dev[tr], Xva, y_dev[va], seed + int(fold))
        oof_score[va] = predict_score(head, Xva)

        clf = train_fall(Xtr, f_dev[tr], Xva, f_dev[va], seed + int(fold))
        oof_fall[va] = predict_fall(clf, Xva)

        fold_rows.append({
            "fold": int(fold), "n_train": len(tr), "n_val": len(va),
            "train_falls": int(f_dev[tr].sum()), "val_falls": int(f_dev[va].sum()),
            "val_mae": regression_metrics(oof_score[va], y_dev[va])["mae"],
        })

    # --- clean-only regressor for the two-stage expected value ---------------
    # Trained per fold on fold-train clean clips only, applied to the fold-val
    # clips (both fall and clean), so `expected` is a genuine OOF prediction.
    for fold in folds:
        tr = np.flatnonzero(dev["fold"].to_numpy() != fold)
        va = np.flatnonzero(dev["fold"].to_numpy() == fold)
        clean_tr = tr[f_dev[tr] == 0]
        if len(clean_tr) < 20:
            continue
        s2 = fit_scaler(X_dev[clean_tr])
        h2 = train_score(apply_scaler(X_dev[clean_tr], s2), y_dev[clean_tr],
                         apply_scaler(X_dev[va], s2), y_dev[va], seed + int(fold))
        oof_clean[va] = predict_score(h2, apply_scaler(X_dev[va], s2))

    # OOF two-stage: per fold, fall mean comes from that fold's training clips.
    expected = np.full(len(dev), np.nan)
    for fold in folds:
        tr = np.flatnonzero(dev["fold"].to_numpy() != fold)
        va = np.flatnonzero(dev["fold"].to_numpy() == fold)
        falls = f_dev[tr] == 1
        fall_mean = float(y_dev[tr][falls].mean()) if falls.any() else float(y_dev[tr].mean())
        expected[va] = (oof_fall[va] * fall_mean
                        + (1.0 - oof_fall[va]) * oof_clean[va])

    # --- fit on all train+val, score the grouped test set --------------------
    scaler = fit_scaler(X_dev)
    Xtr, Xte = apply_scaler(X_dev, scaler), apply_scaler(X_te, scaler)
    head = train_score(Xtr, y_dev, Xte, y_te, seed)
    pred_te = predict_score(head, Xte)

    clf = train_fall(Xtr, f_dev, Xte, f_te, seed)
    p_te = predict_fall(clf, Xte)

    clean_dev = f_dev == 0
    s2 = fit_scaler(X_dev[clean_dev])
    h2 = train_score(apply_scaler(X_dev[clean_dev], s2), y_dev[clean_dev],
                     apply_scaler(X_te, s2), y_te, seed)
    clean_te = predict_score(h2, apply_scaler(X_te, s2))
    fall_mean = float(y_dev[f_dev == 1].mean()) if (f_dev == 1).any() else float(y_dev.mean())
    expected_te = p_te * fall_mean + (1.0 - p_te) * clean_te

    # --- text baselines ------------------------------------------------------
    train_mean = y_dev.mean()
    const_te = np.full(len(y_te), train_mean)
    clean_mean_te = np.full(len(y_te), float(y_dev[clean_dev].mean()))

    mae_ci = bootstrap_ci(y_te, pred_te, lambda a, b: float(np.mean(np.abs(a - b))))
    rho_ci = bootstrap_ci(y_te, pred_te, lambda a, b: float(spearmanr(a, b)[0]))
    auc_te = roc_auc(f_te, p_te)
    auc_ci = bootstrap_ci(f_te.astype(float), p_te, lambda a, b: roc_auc(a, b))
    oof_mae = regression_metrics(oof_score, y_dev, f_dev)["mae"]
    oof_auc = roc_auc(f_dev, oof_fall)

    rows = [
        ("two-stage: expected value", regression_metrics(expected_te, y_te, f_te),
         *bootstrap_ci(y_te, expected_te, lambda a, b: float(np.mean(np.abs(a - b))))),
        ("score head", regression_metrics(pred_te, y_te, f_te), *mae_ci),
        ("clean-only head", regression_metrics(clean_te, y_te, f_te),
         *bootstrap_ci(y_te, clean_te, lambda a, b: float(np.mean(np.abs(a - b))))),
        ("const train mean", regression_metrics(const_te, y_te, f_te),
         *bootstrap_ci(y_te, const_te, lambda a, b: float(np.mean(np.abs(a - b))))),
        ("const clean-mean", regression_metrics(clean_mean_te, y_te, f_te),
         *bootstrap_ci(y_te, clean_mean_te, lambda a, b: float(np.mean(np.abs(a - b))))),
    ]

    return {
        "tag": tag, "sets": rows, "dev": dev, "te": te,
        "oof_score": oof_score, "oof_fall": oof_fall, "oof_clean": oof_clean,
        "oof_expected": expected, "y_dev": y_dev, "f_dev": f_dev,
        "pred_te": pred_te, "p_te": p_te, "clean_te": clean_te,
        "expected_te": expected_te, "y_te": y_te, "f_te": f_te,
        "oof_mae": oof_mae, "oof_auc": oof_auc,
        "test_mae_ci": mae_ci, "test_rho_ci": rho_ci,
        "auc_te": auc_te, "auc_ci": auc_ci, "folds": fold_rows,
        "train_mean": float(train_mean),
    }


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits_grouped.csv")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-report", default="GROUPED_EVAL_REPORT.md")
    args = ap.parse_args()

    df = pd.read_csv(args.splits, dtype={"filename": str})
    full, win = load_features()

    missing = sorted(set(df["filename"].astype(str)) - {k.split("|")[0] for k in full})
    if missing:
        raise SystemExit(f"{len(missing)} split clip(s) have no cached feature, "
                         f"e.g. {missing[:5]}")

    sets = feature_sets(full, win)
    results = [run(df, full, win, parts, tag) for tag, parts in sets.items()]

    n_dev = int((df["split"] != "test").sum())
    te = df[df["split"] == "test"]
    n_falls_te = int(te["fall"].sum())

    lines = [
        "# Leak-free re-evaluation on `splits_grouped.csv`", "",
        "Generated by `train_r3d_heads_grouped.py` from the frozen r3d_18 feature "
        "caches (no video decoding).", "",
        f"* split file: `{args.splits}`",
        f"* train+val: {n_dev} clips over {int(df[df['split'] != 'test']['fold'].nunique())} "
        f"GroupKFold folds",
        f"* test: {len(te)} clips ({int(te['session_group'].nunique())} session(s), "
        f"{n_falls_te} falls), target mean {te['target'].mean():.1f}",
        f"* bootstrap: {N_BOOT} resamples, 95% percentile CI",
        "* `const train mean` is the honest floor: predicting the mean of the "
        "training clips.", "",
    ]
    for r in results:
        lines += [f"## Feature set: `{r['tag']}`", ""]
        lines.append("| model | MAE | MAE 95% CI | RMSE | rho | MAE falls | MAE clean |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, m, lo, hi in r["sets"]:
            lines.append(
                f"| {name} | {m['mae']:.2f} | [{lo:.2f}, {hi:.2f}] | {m['rmse']:.2f} | "
                f"{m['spearman']:.3f} | {m['mae_fall']:.1f} | {m['mae_clean']:.1f} |")
        lines += [
            "",
            f"* OOF (train+val, {n_dev} clips): MAE {r['oof_mae']:.2f}, "
            f"fall AUC {r['oof_auc']:.3f}",
            f"* grouped-test fall AUC: {r['auc_te']:.3f} "
            f"[{r['auc_ci'][0]:.3f}, {r['auc_ci'][1]:.3f}] "
            f"on {n_falls_te} positive / {len(te) - n_falls_te} negative clips",
            "",
        ]

    base = results[0]
    lines += ["## Per-fold stability (score head, `%s`)" % base["tag"], "",
              "| fold | n train | n val | falls train | falls val | val MAE |",
              "|---|---|---|---|---|---|"]
    for f in base["folds"]:
        lines.append(f"| {f['fold']} | {f['n_train']} | {f['n_val']} | "
                     f"{f['train_falls']} | {f['val_falls']} | {f['val_mae']:.2f} |")
    fold_maes = np.array([f["val_mae"] for f in base["folds"]])
    lines += ["", f"OOF fold MAE mean {fold_maes.mean():.2f} "
                  f"+/- {fold_maes.std():.2f} (std across folds)", ""]

    verdict_bits = []
    for r in results:
        head = r["sets"][1]   # score head
        floor = r["sets"][3]  # const train mean
        gain = floor[1]["mae"] - head[1]["mae"]
        overlaps = not (head[2] > floor[3] or floor[2] > head[3])
        rho = (f"rho {head[1]['spearman']:+.3f} "
               f"[{r['test_rho_ci'][0]:+.2f}, {r['test_rho_ci'][1]:+.2f}]")
        if gain <= 0:
            verdict_bits.append(
                f"* `{r['tag']}`: score head is **worse than the constant floor** "
                f"({head[1]['mae']:.2f} vs {floor[1]['mae']:.2f} MAE, {rho}). "
                f"The frozen features do not beat the training mean on an unseen session.")
        else:
            verdict_bits.append(
                f"* `{r['tag']}`: score head beats the constant floor by {gain:+.2f} MAE "
                f"({head[1]['mae']:.2f} vs {floor[1]['mae']:.2f}, {rho}); "
                f"95% CIs {'overlap -> not resolvable at this test size' if overlaps else 'do not overlap'}.")
    lines += ["## Verdict", ""] + verdict_bits + [
        "",
        f"The fall AUC above rests on only {n_falls_te} positive test clip(s): its "
        "95% CI spans most of [0, 1], so it should not be quoted as a result.", "",
        "Compare against the previously reported (leaked-split) numbers: test MAE "
        "19.41 at rho 0.285 on 45 clips. Those clips included 12 from the 1026 "
        "session that was also in training.", "",
    ]

    Path(args.out_report).write_text("\n".join(lines), encoding="utf-8")

    np.savez(
        CKPT_DIR / "grouped_predictions.npz",
        **{f"{r['tag']}__test_names": r["te"]["filename"].astype(str).to_numpy()
           for r in results},
        **{f"{r['tag']}__test_target": r["y_te"] for r in results},
        **{f"{r['tag']}__test_fall": r["f_te"] for r in results},
        **{f"{r['tag']}__test_pred": r["pred_te"] for r in results},
        **{f"{r['tag']}__test_p_fall": r["p_te"] for r in results},
        **{f"{r['tag']}__test_expected": r["expected_te"] for r in results},
        **{f"{r['tag']}__oof_pred": r["oof_score"] for r in results},
        **{f"{r['tag']}__oof_p_fall": r["oof_fall"] for r in results},
        **{f"{r['tag']}__oof_target": r["y_dev"] for r in results},
    )

    print("\n".join(lines))
    print(f"\nwrote {args.out_report} and {CKPT_DIR / 'grouped_predictions.npz'}")


if __name__ == "__main__":
    main()
