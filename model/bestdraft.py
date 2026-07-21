"""Is the best full draft more than the best five individual picks?

The data cannot answer this directly — every full draft in the set is unique, so
there is no replication to estimate a composition-level effect from. What can be
answered is whether the *model* has any combination structure worth searching:

* **greedy** — fill each role with the champion that scores best given what is
  already locked, one role at a time.
* **hill climbing** — from many random starts, repeatedly swap whichever single
  role improves the total most, until nothing improves.

If hill climbing never beats greedy, the model's optimum is separable: the best
full draft is exactly the best pick per role, and there is nothing combinatorial
to exploit. Given that pairwise interactions measured null (`interactions.py`),
that is the expected answer — this checks the model agrees with the data.

    PYTHONPATH=. python model/bestdraft.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from model import data as D
from model.net import DraftNet

ROOT = Path(__file__).resolve().parents[1]
SUPPORT_FLOOR = 30
RESTARTS = 40


def load_ensemble():
    meta = json.loads((ROOT / "artifacts" / "meta.json").read_text())
    cfg = meta["config"]
    n_champ = len(meta["vocab"]["champions"]) + 1
    nets = []
    for i in range(cfg["members"]):
        path = ROOT / "artifacts" / f"member{i}.pt"
        if not path.exists():
            continue
        m = DraftNet(n_champ, dim=cfg["dim"], hidden=cfg["hidden"])
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        nets.append(m)
    return nets, meta, np.array(meta["role_playrates"])


def batch_eval(nets, allies, roles, tier, queue, centers):
    """E[duration] for a batch of ally drafts; enemy unknown, no bans."""
    b = len(allies)
    n_slot = max(len(a) for a in allies) if allies else 0
    ally_c = torch.zeros(b, 5, dtype=torch.long)
    ally_r = torch.full((b, 5), D.ROLE_UNK, dtype=torch.long)
    ally_m = torch.zeros(b, 5)
    for i, (a, rr) in enumerate(zip(allies, roles)):
        for j, (c, r) in enumerate(zip(a, rr)):
            ally_c[i, j], ally_r[i, j], ally_m[i, j] = c, r, 1.0
    batch = {
        "ally_c": ally_c, "ally_r": ally_r, "ally_m": ally_m,
        "enemy_c": torch.zeros(b, 5, dtype=torch.long),
        "enemy_r": torch.full((b, 5), D.ROLE_UNK, dtype=torch.long),
        "enemy_m": torch.zeros(b, 5),
        "ban_c": torch.zeros(b, 10, dtype=torch.long),
        "ban_m": torch.zeros(b, 10),
        "tier": torch.full((b,), tier, dtype=torch.long),
        "queue": torch.full((b,), queue, dtype=torch.long),
    }
    total = torch.zeros(b)
    with torch.no_grad():
        for m in nets:
            logits, _ = m(batch)
            total += (logits.softmax(-1) * centers).sum(-1)
    return (total / len(nets)).numpy()


def main() -> None:
    nets, meta, play = load_ensemble()
    champs = meta["vocab"]["champions"]
    centers = torch.tensor(D.BIN_CENTERS)
    tier = D.TIERS.index("EMERALD")
    queue = D.QUEUES.index("FLEXRANKED")

    legal = {r: [c for c in range(1, len(champs) + 1) if play[c][r] >= SUPPORT_FLOOR]
             for r in range(len(D.ROLES))}
    print(f"{len(champs)} champions; legal per role: "
          + ", ".join(f"{D.ROLES[r]} {len(legal[r])}" for r in legal))

    def score(picks):
        """picks: dict role -> champion index."""
        roles = sorted(picks)
        return batch_eval(nets, [[picks[r] for r in roles]], [roles], tier, queue, centers)[0]

    def score_many(cands, picks, role):
        roles = sorted(set(picks) | {role})
        allies, rr = [], []
        for c in cands:
            p = dict(picks); p[role] = c
            allies.append([p[r] for r in roles]); rr.append(roles)
        return batch_eval(nets, allies, rr, tier, queue, centers)

    # ---- greedy
    picks = {}
    for role in range(len(D.ROLES)):
        cands = [c for c in legal[role] if c not in picks.values()]
        vals = score_many(cands, picks, role)
        picks[role] = cands[int(vals.argmax())]
    greedy = dict(picks)
    greedy_score = score(greedy)

    # ---- hill climbing from random restarts
    rng = np.random.default_rng(0)
    best, best_score = dict(greedy), greedy_score
    beat_greedy = 0
    for t in range(RESTARTS):
        cur = {r: int(rng.choice(legal[r])) for r in range(len(D.ROLES))}
        if len(set(cur.values())) < 5:
            continue
        cur_score = score(cur)
        for _ in range(30):
            improved = False
            for role in range(len(D.ROLES)):
                others = {v for k, v in cur.items() if k != role}
                cands = [c for c in legal[role] if c not in others]
                vals = score_many(cands, {k: v for k, v in cur.items() if k != role}, role)
                j = int(vals.argmax())
                if vals[j] > cur_score + 1e-6:
                    cur[role], cur_score, improved = cands[j], float(vals[j]), True
            if not improved:
                break
        if cur_score > greedy_score + 1e-4:
            beat_greedy += 1
        if cur_score > best_score:
            best, best_score = dict(cur), cur_score

    name = lambda p: ", ".join(f"{D.ROLES[r]}={champs[p[r] - 1]}" for r in sorted(p))
    print(f"\ngreedy (best pick per role, in order):")
    print(f"  {name(greedy)}")
    print(f"  E[duration] = {greedy_score:.3f} min")
    print(f"\nhill climbing, {RESTARTS} random restarts:")
    print(f"  best found  = {best_score:.3f} min")
    print(f"  {name(best)}")
    print(f"  restarts that beat greedy: {beat_greedy}/{RESTARTS}")
    print(f"  gain over greedy: {best_score - greedy_score:+.4f} min")

    # ---- what is the spread across drafts at all?
    sample, roles_all = [], []
    for _ in range(400):
        p = {r: int(rng.choice(legal[r])) for r in range(len(D.ROLES))}
        if len(set(p.values())) < 5:
            continue
        roles = sorted(p)
        sample.append([p[r] for r in roles]); roles_all.append(roles)
    vals = batch_eval(nets, sample, roles_all, tier, queue, centers)
    worst_picks = {}
    for role in range(len(D.ROLES)):
        cands = [c for c in legal[role] if c not in worst_picks.values()]
        v = score_many(cands, worst_picks, role)
        worst_picks[role] = cands[int(v.argmin())]
    print(f"\nrandom legal drafts: mean {vals.mean():.2f} min, sd {vals.std():.2f}, "
          f"range [{vals.min():.2f}, {vals.max():.2f}]")
    print(f"greedy-worst draft:  {score(worst_picks):.2f} min")
    print(f"best-minus-worst span: {best_score - score(worst_picks):.2f} min")


if __name__ == "__main__":
    main()
