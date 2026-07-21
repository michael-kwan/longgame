"""Does player experience or lobby imbalance predict game duration?

Two features a match record cannot supply, joined in from `ingest.profiles`:

* **experience** — ranked games played, per player. The intuition is that
  experienced lobbies surrender more decisively (shorter) or grind more
  (longer); either way it should show up.
* **lobby imbalance** — per-player tier/division/LP converted to a single elo
  scale. A match record only carries one lobby-average tier, which cannot express
  a spread. Stomps end fast, so the *spread* is the interesting part, not the mean.

Everything is measured as a delta against the champion-only ridge under identical
folds, so the comparison is paired and the noise mostly cancels.

    PYTHONPATH=. python model/experience.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from model import data as D

FOLDS = 10
COVERAGE_LEVELS = (3, 5, 8)
TIER_ORDER = {t: i for i, t in enumerate(D.TIERS)}


def elo_score(tier, division, lp) -> float | None:
    """Flatten tier/division/LP onto one axis. ~400 points per tier."""
    if tier is None or tier not in TIER_ORDER:
        return None
    div = division if isinstance(division, int) and 1 <= division <= 4 else 4
    base = TIER_ORDER[tier] * 400
    # Master+ has no divisions; LP just keeps climbing.
    if TIER_ORDER[tier] >= TIER_ORDER["MASTER"]:
        return base + (lp or 0)
    return base + (4 - div) * 100 + (lp or 0)


def load_profiles(data_dir: Path) -> dict[str, dict]:
    path = data_dir / "profiles.db"
    if not path.exists():
        raise SystemExit("no data/profiles.db — run `python -m ingest.profiles` first")
    db = sqlite3.connect(path)
    out = {}
    for puuid, solo, flex, tier, div, lp, level in db.execute(
        "SELECT puuid, solo_games, flex_games, tier, division, lp, level "
        "FROM profiles WHERE ok=1"
    ):
        out[puuid] = {
            "games": (solo or 0) + (flex or 0),
            "elo": elo_score(tier, div, lp),
            "level": level or 0,
        }
    return out


def build(rows, profiles, min_covered: int):
    """Per-game features from whichever players we have profiles for."""
    feats, mask = [], []
    for r in rows:
        g, e, lv = [], [], []
        for p in r["blue_picks"] + r["red_picks"]:
            pr = profiles.get(p["puuid"])
            if not pr:
                continue
            g.append(pr["games"])
            lv.append(pr["level"])
            if pr["elo"] is not None:
                e.append(pr["elo"])
        if len(g) < min_covered:
            feats.append(np.zeros(8))
            mask.append(False)
            continue
        g = np.array(g, dtype=float)
        lg = np.log1p(g)
        e_arr = np.array(e, dtype=float) if e else np.array([0.0])
        feats.append(np.array([
            lg.mean(), lg.std(), lg.min(), lg.max(),
            np.mean(lv),
            e_arr.mean() / 400.0,
            e_arr.std() / 400.0,                    # lobby imbalance
            (e_arr.max() - e_arr.min()) / 400.0,    # worst-case gap
        ]))
        mask.append(True)
    return np.array(feats), np.array(mask)


def cv_r2(X, y, perm, folds, lam):
    scores = []
    for fold in folds:
        tr = np.setdiff1d(perm, fold)
        mu = y[tr].mean()
        w = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]),
                            X[tr].T @ (y[tr] - mu))
        pred = X[fold] @ w + mu
        scores.append(1 - ((pred - y[fold]) ** 2).sum() / ((y[fold] - y[fold].mean()) ** 2).sum())
    return np.array(scores)


def main() -> None:
    rows = D.load_matches(Path("data"))
    profiles = load_profiles(Path("data"))
    vocab = D.build_vocab(rows)
    enc = D.encode(rows, vocab)
    n_all = len(rows)
    print(f"{n_all} games, {len(profiles)} player profiles")

    for min_covered in COVERAGE_LEVELS:
        F, mask = build(rows, profiles, min_covered)
        n = int(mask.sum())
        if n < 500:
            print(f"\n>= {min_covered}/10 players covered: only {n} games, skipping")
            continue
        y = np.array([r["duration_s"] / 60 for r in rows])[mask]
        champs = np.zeros((n_all, len(vocab)))
        champs[np.repeat(np.arange(n_all), 10), enc["picks"].reshape(n_all, 10).ravel()] += 1
        champs[:, 0] = 0
        champs = champs[mask]
        F = F[mask]
        F = (F - F.mean(0)) / (F.std(0) + 1e-9)

        perm = np.random.default_rng(0).permutation(n)
        folds = np.array_split(perm, FOLDS)

        print(f"\n>= {min_covered}/10 players covered: {n} games "
              f"({100 * n / n_all:.1f}% of the set)")
        pick = lambda cands: max(cands, key=lambda v: v.mean())
        base = pick([cv_r2(champs, y, perm, folds, l) for l in (100, 300, 1000)])
        exp_only = pick([cv_r2(F, y, perm, folds, l) for l in (1, 10, 100)])
        both = pick([cv_r2(np.hstack([champs, F]), y, perm, folds, l)
                     for l in (100, 300, 1000)])

        def line(name, v, ref=None):
            se = v.std(ddof=1) / np.sqrt(FOLDS)
            extra = ""
            if ref is not None:
                d = v - ref
                dse = d.std(ddof=1) / np.sqrt(FOLDS)
                verdict = ("better" if d.mean() > 2 * dse else
                           "worse" if d.mean() < -2 * dse else "no difference")
                extra = f"   delta {d.mean():+.4f} +- {dse:.4f} -> {verdict}"
            print(f"    {name:<34} R2 = {v.mean():+.4f} +- {se:.4f}{extra}")

        line("champions only", base)
        line("experience + elo only", exp_only)
        line("champions + experience + elo", both, base)


if __name__ == "__main__":
    main()
