"""Do champion PAIRS carry duration signal beyond the sum of the two champions?

`matchups.py` tested one narrow case: same-role, opposing sides (lane matchups).
This tests the two broader classes:

* **synergy**  — both champions on the SAME team, any roles. "Do these two
  together stall a game more than each does alone?"
* **adversarial** — champions on OPPOSITE teams, any roles.

Two independent methods, because each has a different failure mode:

1. **Permutation test.** Strip additive per-champion effects, group residuals by
   pair, and compare the weighted mean-square of the group means against a null
   built by shuffling residuals across games. Sensitive to *any* pair structure,
   but only over pairs with enough samples.
2. **Factorization machine.** The canonical model for pairwise effects in sparse
   data: each champion gets a rank-k vector and the prediction adds the dot
   product of every co-occurring pair. Unlike trees it does not need each pair
   observed many times — it shares strength across pairs. If interactions exist
   at all, an FM should find them where ridge cannot.

    PYTHONPATH=. python model/interactions.py
"""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np

from model import data as D

FOLDS = 10
MIN_PAIR = 30
PERMUTATIONS = 300


def load():
    rows = D.load_matches(Path("data"))
    n = len(rows)
    y = np.array([r["duration_s"] / 60 for r in rows], dtype=np.float64)
    vocab = D.build_vocab(rows)
    enc = D.encode(rows, vocab)
    return rows, y, enc, len(vocab)


def additive_residuals(enc, y, n_champ, lam=300.0):
    """Residuals after removing per-champion main effects (fit on all data)."""
    n = len(y)
    X = np.zeros((n, n_champ))
    X[np.repeat(np.arange(n), 10), enc["picks"].reshape(n, 10).ravel()] += 1
    X[:, 0] = 0
    mu = y.mean()
    w = np.linalg.solve(X.T @ X + lam * np.eye(n_champ), X.T @ (y - mu))
    return y - (X @ w + mu)


def pair_lists(enc, kind: str) -> list[list[tuple[int, int]]]:
    picks = enc["picks"]
    out = []
    for i in range(len(picks)):
        blue, red = picks[i, 0], picks[i, 1]
        pairs = []
        if kind == "synergy":
            for team in (blue, red):
                for a in range(5):
                    for b in range(a + 1, 5):
                        x, z = int(team[a]), int(team[b])
                        if x and z:
                            pairs.append((min(x, z), max(x, z)))
        else:  # adversarial
            for x in blue:
                for z in red:
                    x, z = int(x), int(z)
                    if x and z:
                        pairs.append((min(x, z), max(x, z)))
        out.append(pairs)
    return out


def permutation_test(pairs, resid, rng, min_pair=MIN_PAIR, perms=PERMUTATIONS):
    def stat(vals):
        g = collections.defaultdict(list)
        for ps, r in zip(pairs, vals):
            for p in ps:
                g[p].append(r)
        keep = [(np.mean(v), len(v)) for v in g.values() if len(v) >= min_pair]
        if not keep:
            return None, 0
        m = np.array([k[0] for k in keep])
        w = np.array([k[1] for k in keep])
        return float(np.average(m**2, weights=w)), len(keep)

    obs, n_groups = stat(resid)
    if obs is None:
        return None
    null = np.array([stat(rng.permutation(resid))[0] for _ in range(perms)])
    return {"obs": obs, "null": null.mean(), "sd": null.std(),
            "z": (obs - null.mean()) / null.std(), "groups": n_groups}


class InteractionFM:
    """Interaction-only factorization machine, fit to ridge's residuals.

    Ridge already solves the additive part optimally, so the FM is given only
    what ridge missed and carries no linear term of its own. That isolates the
    question (is there pairwise structure?) instead of making a hand-rolled SGD
    optimiser re-derive main effects, which it does badly.

    Prediction for a draft with champion set S:  sum_{i<j in S} <v_i, v_j>
    computed in O(k|S|) via the standard identity.
    """

    def __init__(self, n_features, k=8, lr=3e-3, epochs=15, reg=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.k, self.lr, self.epochs, self.reg = k, lr, epochs, reg
        self.V = rng.normal(0, 0.01, (n_features, k))
        self.rng = rng

    def _interaction(self, idx):
        Vs = self.V[idx]
        s = Vs.sum(axis=0)
        return 0.5 * float(s @ s - (Vs * Vs).sum()), s

    def fit(self, index_lists, target):
        order = np.arange(len(target))
        for epoch in range(self.epochs):
            self.rng.shuffle(order)
            lr = self.lr / (1.0 + 0.3 * epoch)      # decay keeps SGD stable
            for i in order:
                idx = index_lists[i]
                pred, s = self._interaction(idx)
                err = np.clip(pred - target[i], -20.0, 20.0)
                grad = s[None, :] - self.V[idx]      # d/dV_i of the interaction
                step = lr * (err * grad + self.reg * self.V[idx])
                np.clip(step, -0.05, 0.05, out=step)
                self.V[idx] -= step
            if not np.isfinite(self.V).all():        # bail rather than emit NaN
                self.V[:] = 0.0
                return self
        return self

    def predict(self, index_lists):
        return np.array([self._interaction(idx)[0] for idx in index_lists])


def r2(pred, actual):
    return 1.0 - ((pred - actual) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def main() -> None:
    rows, y, enc, n_champ = load()
    n = len(y)
    print(f"{n} games, duration sd {y.std():.2f} min\n")

    resid = additive_residuals(enc, y, n_champ)
    rng = np.random.default_rng(0)

    print(f"Permutation test on pair residuals (pairs with >= {MIN_PAIR} games, "
          f"{PERMUTATIONS} shuffles):")
    for kind in ("synergy", "adversarial"):
        res = permutation_test(pair_lists(enc, kind), resid, rng)
        if res is None:
            print(f"  {kind:12} no pairs met the threshold")
            continue
        verdict = "SIGNAL" if res["z"] > 2 else "nothing"
        print(f"  {kind:12} {res['groups']:6d} pairs   observed {res['obs']:.4f} vs "
              f"null {res['null']:.4f}+-{res['sd']:.4f}   z = {res['z']:+.2f}   -> {verdict}")

    # --- ridge, then an interaction-only FM on what ridge missed
    print("\nFactorization machine on ridge residuals (10-fold CV):")
    idx_lists = [[int(c) for c in enc["picks"][i].reshape(10) if c] for i in range(n)]
    X = np.zeros((n, n_champ))
    X[np.repeat(np.arange(n), 10), enc["picks"].reshape(n, 10).ravel()] += 1
    X[:, 0] = 0

    perm = np.random.default_rng(0).permutation(n)
    folds = np.array_split(perm, FOLDS)
    # Sweep the interaction penalty as well as the rank: an undertuned FM
    # overfits and "proves" the wrong thing. If the best configuration is the one
    # that shrinks the interactions away, that is the answer.
    configs = [(4, 0.1), (8, 0.1), (8, 1.0), (8, 10.0), (16, 10.0)]
    ridge_scores, fm_scores = [], {c: [] for c in configs}

    for fold in folds:
        tr = np.setdiff1d(perm, fold)
        mu = y[tr].mean()
        w = np.linalg.solve(X[tr].T @ X[tr] + 300 * np.eye(n_champ), X[tr].T @ (y[tr] - mu))
        base_tr, base_te = X[tr] @ w + mu, X[fold] @ w + mu
        ridge_scores.append(r2(base_te, y[fold]))
        resid_tr = y[tr] - base_tr
        for cfg in configs:
            k, reg = cfg
            fm = InteractionFM(n_champ, k=k, reg=reg, seed=1).fit(
                [idx_lists[i] for i in tr], resid_tr)
            pred = base_te + fm.predict([idx_lists[i] for i in fold])
            fm_scores[cfg].append(r2(pred, y[fold]))

    rs = np.array(ridge_scores)
    print(f"  {'ridge alone':<26} R2 = {rs.mean():+.4f} +- {rs.std(ddof=1)/np.sqrt(FOLDS):.4f}")
    for cfg in configs:
        k, reg = cfg
        v = np.array(fm_scores[cfg])
        d = v - rs
        se = d.std(ddof=1) / np.sqrt(FOLDS)
        verdict = "better" if d.mean() > 2 * se else "worse" if d.mean() < -2 * se else "no difference"
        label = f"+ FM rank {k}, reg {reg:g}"
        print(f"  {label:<26} R2 = {v.mean():+.4f} +- "
              f"{v.std(ddof=1)/np.sqrt(FOLDS):.4f}   delta {d.mean():+.4f} +- {se:.4f} -> {verdict}")


if __name__ == "__main__":
    main()
