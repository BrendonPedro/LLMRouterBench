#!/usr/bin/env python3
import copy, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_cv_eval as cv

OUT = cv.OUT / "ordinal_router_mlp64"
OUT.mkdir(parents=True, exist_ok=True)

# Original: 0=DeepSeek win, -1=tie, 1=Qwen win
# Ordered: 0=DeepSeek better, 1=tie, 2=Qwen better
ORD_MAP = {0: 0, -1: 1, 1: 2}
LABELS = [0, 1, 2]

HIDDEN = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH = 256
MAX_EPOCHS = 300
PATIENCE = 20


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class NominalMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 3),
        )

    def forward(self, x):
        return self.net(x)


class OrdinalMLP(nn.Module):
    """
    Shared 384 -> 64 ReLU representation plus one latent preference score
    and two ordered thresholds.

    Classes:
      0 DeepSeek better
      1 Tie
      2 Qwen better
    """
    def __init__(self, d):
        super().__init__()
        self.hidden = nn.Linear(d, HIDDEN)
        self.score = nn.Linear(HIDDEN, 1)

        # threshold 2 is constrained above threshold 1
        self.t1 = nn.Parameter(torch.tensor(-0.5))
        self.raw_gap = nn.Parameter(torch.tensor(0.0))

    def latent(self, x):
        h = F.relu(self.hidden(x))
        return self.score(h).squeeze(1)

    def thresholds(self):
        t1 = self.t1
        t2 = self.t1 + F.softplus(self.raw_gap)
        return t1, t2

    def cumulative_logits(self, x):
        s = self.latent(x)
        t1, t2 = self.thresholds()
        # logits for P(Y > 0), P(Y > 1)
        return torch.stack([s - t1, s - t2], dim=1)

    def predict_class(self, x):
        logits = self.cumulative_logits(x)
        p_gt0 = torch.sigmoid(logits[:, 0])
        p_gt1 = torch.sigmoid(logits[:, 1])

        probs = torch.stack([
            1.0 - p_gt0,
            p_gt0 - p_gt1,
            p_gt1
        ], dim=1)
        return probs.argmax(dim=1)


def metric_bundle(y, p):
    per = f1_score(y, p, labels=LABELS, average=None, zero_division=0)
    return {
        "macro_f1": f1_score(y, p, labels=LABELS,
                             average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, p),
        "accuracy": accuracy_score(y, p),
        "ordinal_mae": mean_absolute_error(y, p),
        "f1_deepseek": per[0],
        "f1_tie": per[1],
        "f1_qwen": per[2],
    }


def batches(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    for start in range(0, len(idx), BATCH):
        b = idx[start:start+BATCH]
        yield X[b], y[b]


def train_nominal(Xtr, ytr, Xval, yval, seed):
    seed_all(seed)
    model = NominalMLP(Xtr.shape[1])

    counts = np.bincount(ytr, minlength=3)
    weights = len(ytr) / (3.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32)

    opt = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best, best_f1, stale = None, -1.0, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in batches(Xtr, ytr, seed + epoch):
            xb = torch.from_numpy(xb)
            yb = torch.from_numpy(yb).long()

            opt.zero_grad()
            loss = F.cross_entropy(
                model(xb), yb, weight=weights
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(torch.from_numpy(Xval)).argmax(1).numpy()

        f1 = f1_score(yval, pv, average="macro", zero_division=0)
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best)
    return model, epoch + 1


def train_ordinal(Xtr, ytr, Xval, yval, seed):
    seed_all(seed)
    model = OrdinalMLP(Xtr.shape[1])

    # cumulative targets:
    # DS  = [0,0]
    # Tie = [1,0]
    # QW  = [1,1]
    Y = np.column_stack([ytr > 0, ytr > 1]).astype(np.float32)

    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    pos_weight = torch.tensor(
        neg / np.maximum(pos, 1),
        dtype=torch.float32
    )

    opt = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    best, best_f1, stale = None, -1.0, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(len(ytr))

        for start in range(0, len(order), BATCH):
            b = order[start:start+BATCH]
            xb = torch.from_numpy(Xtr[b])
            yb = torch.from_numpy(Y[b])

            opt.zero_grad()
            logits = model.cumulative_logits(xb)
            loss = F.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pos_weight
            )
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model.predict_class(
                torch.from_numpy(Xval)
            ).numpy()

        f1 = f1_score(yval, pv, average="macro", zero_division=0)
        if f1 > best_f1 + 1e-6:
            best_f1 = f1
            best = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best)
    return model, epoch + 1


def main():
    pool = cv.load_full_pool_with_folds().copy()
    pool["y3"] = pool["target"].map(ORD_MAP).astype(int)

    cache = cv.load_embedding_cache(
        cv.MINILM_CACHE, cv.MINILM_MODEL
    )

    print(f"pool={len(pool)} folds={sorted(pool.fold.unique())}")
    print(f"MiniLM cached={len(cache)}")
    print("class counts:", pool.y3.value_counts().sort_index().to_dict())
    print()

    rows, preds = [], []

    for fold in sorted(pool.fold.unique()):
        train = pool[pool.fold != fold].copy()
        test = pool[pool.fold == fold].copy()

        norm = Normalizer(norm="l2")
        Xall = norm.fit_transform(
            cv.embedding_matrix(train["query"].tolist(), cache)
        ).astype(np.float32)
        Xte = norm.transform(
            cv.embedding_matrix(test["query"].tolist(), cache)
        ).astype(np.float32)

        yall = train.y3.to_numpy()
        yte = test.y3.to_numpy()

        tr_idx, va_idx = train_test_split(
            np.arange(len(train)),
            test_size=0.10,
            random_state=42 + int(fold),
            stratify=yall,
        )

        Xtr, Xval = Xall[tr_idx], Xall[va_idx]
        ytr, yval = yall[tr_idx], yall[va_idx]

        models = []

        t0 = time.perf_counter()
        nom, nom_epochs = train_nominal(
            Xtr, ytr, Xval, yval, 42 + int(fold)
        )
        nom_train = time.perf_counter() - t0
        nom.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            p_nom = nom(torch.from_numpy(Xte)).argmax(1).numpy()
        nom_inf = time.perf_counter() - t0
        models.append(("Nominal MLP-64", p_nom,
                       nom_train, nom_inf, nom_epochs))

        t0 = time.perf_counter()
        ordm, ord_epochs = train_ordinal(
            Xtr, ytr, Xval, yval, 142 + int(fold)
        )
        ord_train = time.perf_counter() - t0
        ordm.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            p_ord = ordm.predict_class(
                torch.from_numpy(Xte)
            ).numpy()
        ord_inf = time.perf_counter() - t0
        models.append(("Ordinal MLP-64", p_ord,
                       ord_train, ord_inf, ord_epochs))

        p_tie = np.ones(len(test), dtype=int)
        models.insert(0, ("Always Tie", p_tie, 0.0, 0.0, 0))

        for name, p, tr_sec, inf_sec, epochs in models:
            m = metric_bundle(yte, p)

            rows.append({
                "fold": fold, "method": name,
                **m,
                "epochs": epochs,
                "train_sec": tr_sec,
                "infer_sec": inf_sec,
                "n_test": len(test),
            })

            print(
                f"fold {fold} | {name:16s} "
                f"F1={m['macro_f1']:.4f} "
                f"DS={m['f1_deepseek']:.4f} "
                f"Tie={m['f1_tie']:.4f} "
                f"Qwen={m['f1_qwen']:.4f} "
                f"epochs={epochs}"
            )

            for i, idx in enumerate(test.index):
                preds.append({
                    "row_index": idx,
                    "fold": fold,
                    "dataset": test.loc[idx, "dataset"],
                    "target": int(yte[i]),
                    "method": name,
                    "prediction": int(p[i]),
                })

        print()

    fdf = pd.DataFrame(rows)
    pdf = pd.DataFrame(preds)

    fdf.to_csv(OUT / "folds.csv", index=False)
    pdf.to_csv(OUT / "predictions.csv", index=False)

    summary = []

    print("=" * 72)
    print("POOLED HELD-OUT RESULTS")
    print("=" * 72)

    for method in ["Always Tie", "Nominal MLP-64", "Ordinal MLP-64"]:
        d = pdf[pdf.method == method]
        y = d.target.to_numpy()
        p = d.prediction.to_numpy()
        m = metric_bundle(y, p)

        fd = fdf[fdf.method == method]

        summary.append({
            "method": method,
            **m,
            "fold_macro_f1_mean": fd.macro_f1.mean(),
            "fold_macro_f1_std": fd.macro_f1.std(ddof=1),
            "train_sec_total": fd.train_sec.sum(),
        })

        print()
        print(method)
        print(f"  Macro-F1:          {m['macro_f1']:.4f}")
        print(f"  Balanced accuracy: {m['balanced_accuracy']:.4f}")
        print(f"  Accuracy:          {m['accuracy']:.4f}")
        print(f"  Ordinal MAE:       {m['ordinal_mae']:.4f}")
        print(f"  F1 DeepSeek:       {m['f1_deepseek']:.4f}")
        print(f"  F1 Tie:            {m['f1_tie']:.4f}")
        print(f"  F1 Qwen:           {m['f1_qwen']:.4f}")
        print(
            f"  Fold Macro-F1:     "
            f"{fd.macro_f1.mean():.4f} +/- "
            f"{fd.macro_f1.std(ddof=1):.4f}"
        )
        print("  Confusion matrix [DS, Tie, Qwen]:")
        print(confusion_matrix(y, p, labels=LABELS))

    sdf = pd.DataFrame(summary)
    sdf.to_csv(OUT / "summary.csv", index=False)

    nom = sdf[sdf.method == "Nominal MLP-64"].iloc[0]
    ordm = sdf[sdf.method == "Ordinal MLP-64"].iloc[0]

    delta = ordm.macro_f1 - nom.macro_f1

    print()
    print("=" * 72)
    print("PREDECLARED COMPARISON")
    print("=" * 72)
    print(f"Ordinal - nominal Macro-F1 delta: {delta:+.4f}")
    print(
        "Criterion: ordinal >= nominal +0.02 Macro-F1 "
        "and non-zero F1 for both winner classes."
    )

    promising = (
        delta >= 0.02
        and ordm.f1_deepseek > 0
        and ordm.f1_qwen > 0
    )
    print(f"PROMISING = {bool(promising)}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
