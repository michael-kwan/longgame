"""Does gradient boosting beat ridge on this problem?

Ridge on a bag-of-champions is the incumbent (DESIGN.md §9). Trees are a
reasonable thing to try — they could find composition effects that an additive
model cannot, e.g. "two tanks *and* a scaling carry" behaving differently from
the sum of its parts.

Three framings are compared under identical 10-fold CV:

1. **raw** — XGBoost straight on the 173-wide champion count matrix. This is the
   obvious thing to do and the wrong thing to do: trees split one feature at a
   time, and a champion appearing in ~5% of games gives almost no split value.
2. **encoded** — champion effects are target-encoded *inside each training fold*
   (a ridge fit on the training portion only), then aggregated per team into a
   handful of dense features. Trees can actually use these, and any non-additive
   composition effect has somewhere to show up.
3. **hybrid** — ridge prediction plus XGBoost fit on ridge's residuals. If
   interactions exist beyond the additive part, this is where they surface.

Leakage matters more than usual here: the effect being measured is ~2% of
variance, so any target encoding fit outside the fold would manufacture a win.

    PYTHONPATH=. python model/gbdt.py
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from pathlib import Path

from model import data as D

FOLDS = 10
XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_lambda=2.0,
    n_jobs=8,
    verbosity=0,
)


def load():
    rows = D.load_matches(Path("data"))
    n = len(rows)
    y = np.array([r["duration_s"] / 60 for r in rows], dtype=np.float64)
    vocab = D.build_vocab(rows)
    enc = D.encode(rows, vocab)
    n_champ = len(vocab)

    counts = np.zeros((n, n_champ))
    idx = np.repeat(np.arange(n), 10)
    counts[idx, enc["picks"].reshape(n, 10).ravel()] += 1
    counts[:, 0] = 0

    ctx = np.column_stack([enc["tier"], enc["queue"]]).astype(float)
    return y, counts, ctx, enc, n_champ


def ridge_weights(counts, y, tr, lam=300.0):
    mu = y[tr].mean()
    A = counts[tr].T @ counts[tr] + lam * np.eye(counts.shape[1])
    w = np.linalg.solve(A, counts[tr].T @ (y[tr] - mu))
    return w, mu


def team_features(enc, w, rows_idx):
    """Aggregate in-fold champion effects per team — dense, tree-friendly."""
    picks = enc["picks"][rows_idx]           # (n, 2, 5)
    eff = w[picks]                           # (n, 2, 5)
    feats = []
    for side in (0, 1):
        e = eff[:, side, :]
        feats += [e.sum(1), e.mean(1), e.std(1), e.min(1), e.max(1)]
    both = eff.reshape(len(rows_idx), 10)
    feats += [both.sum(1), both.std(1), both.max(1) - both.min(1),
              eff[:, 0, :].sum(1) - eff[:, 1, :].sum(1)]
    return np.column_stack(feats)


def r2(pred, actual):
    return 1.0 - ((pred - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def main() -> None:
    y, counts, ctx, enc, n_champ = load()
    n = len(y)
    print(f"{n} games, duration sd {y.std():.2f} min, {n_champ - 1} champions\n")

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    folds = np.array_split(perm, FOLDS)

    scores = {k: [] for k in ("ridge", "raw", "encoded", "hybrid", "hybrid_raw")}

    for fold in folds:
        te = fold
        tr = np.setdiff1d(perm, fold)

        w, mu = ridge_weights(counts, y, tr)
        ridge_tr = counts[tr] @ w + mu
        ridge_te = counts[te] @ w + mu
        scores["ridge"].append(r2(ridge_te, y[te]))

        # 1. raw champion counts
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(np.hstack([counts[tr], ctx[tr]]), y[tr])
        scores["raw"].append(r2(m.predict(np.hstack([counts[te], ctx[te]])), y[te]))

        # 2. in-fold target-encoded team aggregates
        Ftr = np.hstack([team_features(enc, w, tr), ctx[tr]])
        Fte = np.hstack([team_features(enc, w, te), ctx[te]])
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Ftr, y[tr])
        scores["encoded"].append(r2(m.predict(Fte), y[te]))

        # 3. boosting on ridge residuals, over the aggregates
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Ftr, y[tr] - ridge_tr)
        scores["hybrid"].append(r2(ridge_te + m.predict(Fte), y[te]))

        # 3b. same, but over raw champion counts — the fair test of whether any
        # interaction signal survives once additive effects are removed.
        Rtr = np.hstack([counts[tr], ctx[tr]])
        Rte = np.hstack([counts[te], ctx[te]])
        m = xgb.XGBRegressor(**XGB_PARAMS)
        m.fit(Rtr, y[tr] - ridge_tr)
        scores["hybrid_raw"].append(r2(ridge_te + m.predict(Rte), y[te]))

    print(f"{'model':<34} {'CV R^2':>10}   {'SE':>7}")
    print("-" * 55)
    for name, label in [
        ("ridge", "ridge, bag-of-champions"),
        ("raw", "XGBoost, raw champion counts"),
        ("encoded", "XGBoost, in-fold encoded aggregates"),
        ("hybrid", "ridge + XGBoost on residuals (agg)"),
        ("hybrid_raw", "ridge + XGBoost on residuals (raw)"),
    ]:
        v = np.array(scores[name])
        print(f"{label:<34} {v.mean():>+10.4f}   {v.std(ddof=1) / np.sqrt(FOLDS):>7.4f}")

    # Paired comparison — same folds, so the difference has its own (smaller) SE.
    print()
    base = np.array(scores["ridge"])
    for name in ("raw", "encoded", "hybrid", "hybrid_raw"):
        d = np.array(scores[name]) - base
        se = d.std(ddof=1) / np.sqrt(FOLDS)
        verdict = "better" if d.mean() > 2 * se else "worse" if d.mean() < -2 * se else "no difference"
        print(f"  {name:<10} vs ridge: {d.mean():+.4f} +- {se:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
