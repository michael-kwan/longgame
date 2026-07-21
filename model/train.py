"""Train the duration value model.

Offline Monte Carlo value learning (DESIGN.md §1): regress the observed terminal
duration onto partial-draft states. An ensemble is trained because §6 needs
disagreement as a pessimism signal — argmax otherwise hunts for the model's own
extrapolation errors.

    python -m model.train --members 5 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data as D
from .net import DraftNet, losses

ROOT = Path(__file__).resolve().parents[1]


def to_torch(batch: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items()}


def baselines(enc: dict[str, np.ndarray], train_idx, val_idx) -> dict:
    """Numbers the model has to beat before it is worth believing (§9)."""
    train_min = enc["duration"][train_idx]
    val_min = enc["duration"][val_idx]

    bins = D.duration_bin(train_min)
    prior = np.bincount(bins, minlength=D.N_BINS).astype(np.float64)
    prior /= prior.sum()
    val_bins = D.duration_bin(val_min)
    nll = -np.log(np.clip(prior[val_bins], 1e-9, None)).mean()

    mean_pred = train_min.mean()

    # Ridge on a bag-of-champions over the full draft. This is the "dumb
    # baseline" of DESIGN.md §9 and it is genuinely competitive — the net has to
    # beat it at 10 picks revealed or it has not earned its complexity.
    ridge = _ridge_full_draft(enc, train_idx, val_idx)

    return {
        "prior_nll": float(nll),
        "mean_mae": float(np.abs(val_min - mean_pred).mean()),
        "mean_minutes": float(mean_pred),
        "std_minutes": float(val_min.std()),
        "ridge_r2": ridge["r2"],
        "ridge_se": ridge["se"],
        "ridge_mae": ridge["mae"],
        "ridge_lambda": ridge["lam"],
    }


def _ridge_full_draft(enc, train_idx, val_idx, folds: int = 10) -> dict:
    """Cross-validated, because a single split cannot measure an effect this small.

    At ~12k matches the per-fold R2 ranges over roughly [+0.008, +0.029]; two runs
    on near-identical data produced +0.009 and +0.017 from single splits alone.
    Reporting a point estimate without the standard error invites reading noise
    as progress.
    """
    n = len(enc["duration"])
    n_champ = int(enc["picks"].max()) + 1
    counts = np.zeros((n, n_champ), dtype=np.float64)
    rows = np.repeat(np.arange(n), 10)
    counts[rows, enc["picks"].reshape(n, 10).ravel()] += 1
    counts[:, 0] = 0  # padding column

    y = enc["duration"].astype(np.float64)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    eye = np.eye(n_champ)

    best = {"r2": -np.inf, "se": None, "mae": np.inf, "lam": None}
    for lam in (100, 300, 1000, 3000):
        r2s, maes = [], []
        for fold in np.array_split(perm, folds):
            tr = np.setdiff1d(perm, fold, assume_unique=False)
            mu = y[tr].mean()
            w = np.linalg.solve(counts[tr].T @ counts[tr] + lam * eye,
                                counts[tr].T @ (y[tr] - mu))
            pred = counts[fold] @ w + mu
            ss_res = ((pred - y[fold]) ** 2).sum()
            ss_tot = ((y[fold] - y[fold].mean()) ** 2).sum()
            r2s.append(1.0 - ss_res / ss_tot)
            maes.append(np.abs(pred - y[fold]).mean())
        r2s = np.array(r2s)
        if r2s.mean() > best["r2"]:
            best = {
                "r2": float(r2s.mean()),
                "se": float(r2s.std(ddof=1) / np.sqrt(folds)),
                "mae": float(np.mean(maes)),
                "lam": lam,
            }
    return best


def evaluate(model, enc, idx, centers, device, seed=1234, force_k=None) -> dict:
    model.eval()
    rng = np.random.default_rng(seed)
    nlls, maes, preds, actuals, wins, win_correct = [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(idx), 4096):
            chunk = idx[start:start + 4096]
            batch = to_torch(D.sample_states(enc, chunk, rng, force_k=force_k), device)
            logits, win_logit = model(batch)
            logp = torch.log_softmax(logits, dim=-1)
            nlls.append(-logp.gather(1, batch["y_bin"][:, None]).squeeze(1).cpu().numpy())
            em = (logits.softmax(-1) * centers).sum(-1)
            preds.append(em.cpu().numpy())
            actuals.append(batch["y_min"].cpu().numpy())
            maes.append((em - batch["y_min"]).abs().cpu().numpy())
            wins.append(torch.sigmoid(win_logit).cpu().numpy())
            win_correct.append(
                ((win_logit > 0).float() == batch["y_win"]).float().cpu().numpy()
            )
    preds, actuals = np.concatenate(preds), np.concatenate(actuals)
    ss_res = float(((preds - actuals) ** 2).sum())
    ss_tot = float(((actuals - actuals.mean()) ** 2).sum())
    return {
        "nll": float(np.concatenate(nlls).mean()),
        "mae": float(np.concatenate(maes).mean()),
        "r2": 1.0 - ss_res / ss_tot,
        "win_acc": float(np.concatenate(win_correct).mean()),
        "pred_spread": float(preds.std()),
    }


def train_member(enc, train_idx, val_idx, vocab, args, seed, device, centers):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = DraftNet(len(vocab), dim=args.dim, hidden=args.hidden, dropout=args.dropout).to(device)

    # The champion main-effect table is a ridge regression living inside the net,
    # and ridge's optimal penalty here is nothing like the trunk's. Give it (and
    # the embedding tables, which are equally data-starved per parameter) their
    # own decay groups.
    effect_params, embed_params, trunk_params = [], [], []
    for name, p in model.named_parameters():
        if name.startswith("champ_effect"):
            effect_params.append(p)
        elif name.startswith(("champ.", "role.", "tier.", "queue.")):
            embed_params.append(p)
        else:
            trunk_params.append(p)
    # The effect table is zero-initialised and sees a much smaller gradient than
    # the trunk, so it needs its own (larger) learning rate or it stays inert.
    opt = torch.optim.AdamW(
        [
            {"params": effect_params, "weight_decay": args.effect_wd,
             "lr": args.lr * args.effect_lr_mult},
            {"params": embed_params, "weight_decay": args.embed_wd, "lr": args.lr},
            {"params": trunk_params, "weight_decay": args.wd, "lr": args.lr},
        ],
        lr=args.lr,
    )

    steps_per_epoch = max(1, len(train_idx) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=[args.lr * args.effect_lr_mult, args.lr, args.lr],
        total_steps=total_steps,
    )

    # Bootstrap resample so ensemble members disagree where data is thin —
    # that disagreement is the pessimism signal used at inference (§6).
    member_idx = rng.choice(train_idx, size=len(train_idx), replace=True)

    # Keep the best-validation weights, not the last epoch's. Without this the
    # ensemble ships its most overfit checkpoint.
    # Select on MAE, not NLL: the product ranks candidates by E[duration], and
    # the two metrics diverge here — NLL plateaus while MAE is still improving.
    best = {"score": float("inf"), "state": None, "report": None, "epoch": 0}

    for epoch in range(args.epochs):
        model.train()
        perm = rng.permutation(len(member_idx))
        running = 0.0
        for b in range(steps_per_epoch):
            chunk = member_idx[perm[b * args.batch:(b + 1) * args.batch]]
            batch = to_torch(D.sample_states(enc, chunk, rng), device)
            logits, win_logit = model(batch)
            loss, dur_loss, _ = losses(logits, win_logit, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(dur_loss.detach())
        val = evaluate(model, enc, val_idx, centers, device)
        flag = ""
        if val[args.select] < best["score"] - 1e-6:
            best = {
                "score": val[args.select],
                "state": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "report": val,
                "epoch": epoch + 1,
            }
            flag = "  *"
        print(
            f"  seed {seed} epoch {epoch + 1}/{args.epochs} "
            f"train_nll={running / steps_per_epoch:.4f} "
            f"val_nll={val['nll']:.4f} mae={val['mae']:.2f} r2={val['r2']:+.4f} "
            f"win_acc={val['win_acc']:.3f}{flag}",
            flush=True,
        )
        if epoch + 1 - best["epoch"] >= args.patience:
            print(f"  seed {seed} early stop (best epoch {best['epoch']})", flush=True)
            break

    model.load_state_dict(best["state"])
    return model, best["report"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--effect-wd", type=float, default=0.5)
    ap.add_argument("--embed-wd", type=float, default=0.1)
    ap.add_argument("--effect-lr-mult", type=float, default=20.0)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--select", default="mae", choices=["mae", "nll"])
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--queues", default="all",
                    help="all | SOLORANKED | FLEXRANKED | comma-separated")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train] device={device}")

    rows = D.load_matches(args.data_dir)
    if args.queues != "all":
        wanted = set(args.queues.split(","))
        before = len(rows)
        rows = [r for r in rows if r["queue"] in wanted]
        print(f"[train] queue filter {sorted(wanted)}: {before} -> {len(rows)} matches")
    print(f"[train] {len(rows)} unique matches")
    if len(rows) < 3000:
        print(f"[train] WARNING: {len(rows)} matches is far too few for {len(set()) or 173} "
              f"champions — expect the model to learn essentially nothing.")
    vocab = D.build_vocab(rows)
    enc = D.encode(rows, vocab)
    print(f"[train] {len(vocab) - 1} champions seen")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    n_val = int(len(rows) * args.val_frac)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    print(f"[train] train={len(train_idx)} val={len(val_idx)}")

    base = baselines(enc, train_idx, val_idx)
    print(
        f"[train] baseline: prior_nll={base['prior_nll']:.4f} "
        f"mean_mae={base['mean_mae']:.2f} min "
        f"(mean={base['mean_minutes']:.1f}, sd={base['std_minutes']:.1f})"
    )
    print(
        f"[train] ridge (full draft, bag-of-champions, lambda={base['ridge_lambda']}): "
        f"R^2 {base['ridge_r2']:+.4f} +- {base['ridge_se']:.4f} (10-fold CV)  "
        f"MAE {base['ridge_mae']:.3f} min  <- the net must beat this at 10 picks"
    )

    centers = torch.from_numpy(D.BIN_CENTERS).to(device)
    args.out.mkdir(parents=True, exist_ok=True)

    members, reports = [], []
    for m in range(args.members):
        print(f"[train] member {m + 1}/{args.members}")
        model, val = train_member(enc, train_idx, val_idx, vocab, args, 100 + m, device, centers)
        members.append(model)
        reports.append(val)
        torch.save(model.state_dict(), args.out / f"member{m}.pt")

    mean_nll = float(np.mean([r["nll"] for r in reports]))
    mean_mae = float(np.mean([r["mae"] for r in reports]))
    mean_r2 = float(np.mean([r["r2"] for r in reports]))
    print("\n[train] ===== summary =====")
    print(f"  prior NLL   {base['prior_nll']:.4f}  ->  model {mean_nll:.4f} "
          f"({100 * (base['prior_nll'] - mean_nll) / base['prior_nll']:+.1f}%)")
    print(f"  mean MAE    {base['mean_mae']:.2f}  ->  model {mean_mae:.2f} min")
    print(f"  R^2         {mean_r2:+.4f}   (DESIGN.md expects 0.02-0.06)")
    print(f"  ridge R^2   {base['ridge_r2']:+.4f} +- {base['ridge_se']:.4f}  (full draft, 10-fold CV)")
    print(f"  NOTE: the net's numbers above come from a single {args.val_frac:.0%} split, so they "
          f"carry roughly +-0.01 of noise.\n        Differences smaller than that are not real.")
    print(f"  win acc     {np.mean([r['win_acc'] for r in reports]):.3f}")

    print("\n[train] signal by draft phase (ensemble member 0):")
    phases = {}
    for k in (0, 2, 4, 6, 8, 10):
        rep = evaluate(members[0], enc, val_idx, centers, device, force_k=k)
        phases[k] = rep
        print(f"    {k:2d} picks revealed:  R^2 {rep['r2']:+.4f}   MAE {rep['mae']:.3f}   "
              f"spread of predictions {rep['pred_spread']:.3f} min")

    meta = {
        "n_matches": len(rows),
        "phases": {str(k): v for k, v in phases.items()},
        "vocab": vocab.to_json(),
        "baseline": base,
        "members": reports,
        "summary": {"nll": mean_nll, "mae": mean_mae, "r2": mean_r2},
        "config": {"dim": args.dim, "hidden": args.hidden, "members": args.members,
                   "dropout": args.dropout},
        "role_playrates": D.role_playrates(rows, vocab).tolist(),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (args.out / "meta.json").write_text(json.dumps(meta))
    print(f"[train] wrote {args.out}/meta.json and {args.members} checkpoints")


if __name__ == "__main__":
    main()
