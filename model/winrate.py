"""A win-probability channel, fit properly rather than reused from the net.

The duration network carries a `win_head`, but it scores **0.4926** held-out
accuracy — worse than a coin flip. That is not a training failure, it is
structural: the champion main-effect channel sums both teams equally, so the
model is symmetric in the two sides and *cannot* express "blue wins". Shipping
that head as a "win rate" objective would surface noise as a recommendation.

The fix is a design that can represent a side: **antisymmetric** champion
features, blue minus red. Under that specification champion identity predicts
the winning side at AUC ~0.529 — small, but 4.5 standard errors above chance.

Fits an ensemble of bootstrap ridge models so the UI's pessimism (value minus
ensemble spread) keeps its meaning for this objective too.

    PYTHONPATH=. python model/winrate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import data as D

ROOT = Path(__file__).resolve().parents[1]
FOLDS = 10
MEMBERS = 5
LAMBDA = 300.0


def design(enc, n: int, n_champ: int) -> np.ndarray:
    """Blue champions minus red champions — the only way to encode a side."""
    X = np.zeros((n, n_champ))
    for side, sign in ((0, 1.0), (1, -1.0)):
        for j in range(5):
            X[np.arange(n), enc["picks"][:, side, j]] += sign
    X[:, 0] = 0
    return X


def fit(X: np.ndarray, y: np.ndarray, idx: np.ndarray, lam: float = LAMBDA):
    mu = float(y[idx].mean())
    w = np.linalg.solve(
        X[idx].T @ X[idx] + lam * np.eye(X.shape[1]), X[idx].T @ (y[idx] - mu)
    )
    return w, mu


def auc(pred: np.ndarray, y: np.ndarray) -> float:
    pos, neg = pred[y == 1], pred[y == 0]
    if not len(pos) or not len(neg):
        return 0.5
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main() -> None:
    rows = D.load_matches(ROOT / "data")
    n = len(rows)
    vocab = D.build_vocab(rows)
    enc = D.encode(rows, vocab)
    n_champ = len(vocab)
    y = enc["blue_win"].astype(np.float64)
    X = design(enc, n, n_champ)

    print(f"{n} games, blue win rate {y.mean():.4f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, FOLDS)
    aucs, accs = [], []
    for fold in folds:
        tr = np.setdiff1d(perm, fold)
        w, mu = fit(X, y, tr)
        p = X[fold] @ w + mu
        aucs.append(auc(p, y[fold]))
        accs.append(float(((p > 0.5).astype(float) == y[fold]).mean()))
    aucs, accs = np.array(aucs), np.array(accs)
    print(
        f"10-fold CV: AUC {aucs.mean():.4f} +- {aucs.std(ddof=1) / np.sqrt(FOLDS):.4f}"
        f"   accuracy {accs.mean():.4f}"
    )
    print("  (the network's own win_head scores 0.4926 accuracy — worse than chance)")

    # Bootstrap ensemble, so pessimism means the same thing for this objective.
    members = []
    for m in range(MEMBERS):
        boot = np.random.default_rng(100 + m).choice(n, size=n, replace=True)
        w, mu = fit(X, y, boot)
        members.append({"w": w.tolist(), "mu": mu})

    out = {
        "lambda": LAMBDA,
        "members": members,
        "cv": {
            "auc": float(aucs.mean()),
            "auc_se": float(aucs.std(ddof=1) / np.sqrt(FOLDS)),
            "accuracy": float(accs.mean()),
        },
        "note": (
            "Antisymmetric ridge on champion identity (blue minus red). Predicts "
            "P(this side wins). AUC ~0.53 — real but weak."
        ),
    }
    path = ROOT / "artifacts" / "win_model.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({MEMBERS} bootstrap members, {n_champ} coefficients each)")


if __name__ == "__main__":
    main()
