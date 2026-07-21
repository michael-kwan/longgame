"""DeepSets value model over partial drafts.

Why DeepSets rather than a transformer:

* Permutation invariance is not an approximation here, it is the correct
  inductive bias — DESIGN.md §4 treats the draft as a set because pick order is
  unrecoverable. Sum-pooling gets that for free and cannot leak a spurious
  order signal.
* Partial drafts are native: pooling over a mask handles 0..5 revealed picks
  with no padding semantics to get wrong.
* It is ~100k parameters and the forward pass is an embedding lookup, two sums
  and three matmuls — implementable in plain JS for the static page, with no
  external inference runtime (which the CSP would block anyway).

Heads: duration distribution (the reward model) and win probability (auxiliary,
and half the causal story per §2 — balanced games run long).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import BIN_CENTERS, N_BINS, N_ROLES, QUEUES, TIERS


class DraftNet(nn.Module):
    def __init__(self, n_champions: int, dim: int = 32, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.dim = dim
        self.champ = nn.Embedding(n_champions, dim, padding_idx=0)
        self.role = nn.Embedding(N_ROLES, dim)
        self.tier = nn.Embedding(len(TIERS), 16)
        self.queue = nn.Embedding(len(QUEUES), 8)

        # The champion table is most of the capacity and sees the least data per
        # parameter, so it is initialised small and decayed hard by the optimiser.
        nn.init.normal_(self.champ.weight, std=0.02)
        with torch.no_grad():
            self.champ.weight[0].zero_()

        # ally, enemy, ally*enemy interaction, bans, counts(3), tier, queue
        in_dim = dim * 4 + 3 + 16 + 8
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.duration_head = nn.Linear(hidden, N_BINS)
        self.win_head = nn.Linear(hidden, 1)

        nn.init.zeros_(self.duration_head.bias)
        nn.init.zeros_(self.win_head.bias)

        # Explicit additive channel: one scalar per champion, summed over every
        # champion in the game, shifting the distribution along the duration
        # axis. This is the ridge main-effects baseline embedded in the net —
        # without it the MLP has to rediscover the easiest signal through a
        # bottleneck, and empirically it fails to (it lost to plain ridge).
        # The MLP is then free to model interactions on top.
        self.champ_effect = nn.Embedding(n_champions, 1, padding_idx=0)
        nn.init.zeros_(self.champ_effect.weight)
        centers = torch.tensor(BIN_CENTERS, dtype=torch.float32)
        self.register_buffer("bin_dir", (centers - centers.mean()) / centers.std())

    def _pool(self, champ_ids, role_ids, mask):
        # Mean, not sum: with sum-pooling the feature scale grows as the draft
        # fills (and the ally*enemy term grows quadratically), so the trunk sees
        # a moving distribution and washes out champion identity exactly when
        # the most is known. Counts are supplied separately as features.
        h = self.champ(champ_ids) + self.role(role_ids)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1)
        return pooled / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        ally = self._pool(batch["ally_c"], batch["ally_r"], batch["ally_m"])
        enemy = self._pool(batch["enemy_c"], batch["enemy_r"], batch["enemy_m"])
        ban_mask = batch["ban_m"]
        bans = (self.champ(batch["ban_c"]) * ban_mask.unsqueeze(-1)).sum(dim=1)
        bans = bans / ban_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        counts = torch.stack(
            [
                batch["ally_m"].sum(1) / 5.0,
                batch["enemy_m"].sum(1) / 5.0,
                batch["ban_m"].sum(1) / 10.0,
            ],
            dim=1,
        )
        feats = torch.cat(
            [
                ally,
                enemy,
                ally * enemy,   # explicit comp-vs-comp interaction
                bans,
                counts,
                self.tier(batch["tier"]),
                self.queue(batch["queue"]),
            ],
            dim=1,
        )
        h = self.trunk(feats)

        # Champion main effects: every champion in the game, either side.
        effect = (
            (self.champ_effect(batch["ally_c"]).squeeze(-1) * batch["ally_m"]).sum(1)
            + (self.champ_effect(batch["enemy_c"]).squeeze(-1) * batch["enemy_m"]).sum(1)
        )
        logits = self.duration_head(h) + effect.unsqueeze(1) * self.bin_dir
        return logits, self.win_head(h).squeeze(-1)


def losses(
    logits: torch.Tensor,
    win_logit: torch.Tensor,
    batch: dict[str, torch.Tensor],
    win_weight: float = 0.3,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    duration = F.cross_entropy(logits, batch["y_bin"], label_smoothing=label_smoothing)
    win = F.binary_cross_entropy_with_logits(win_logit, batch["y_win"])
    return duration + win_weight * win, duration, win


@torch.no_grad()
def expected_minutes(logits: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (logits.softmax(dim=-1) * centers).sum(dim=-1)
