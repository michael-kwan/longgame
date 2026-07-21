"""Turn crawled matches into masked partial-draft training examples.

Set completion, per DESIGN.md §4: real pick order is unrecoverable, so a finished
draft is treated as a set and we train on random reveals of it. Every match
yields many states.

Ranked SR draft reveals all 10 bans first, then picks alternate B,R,R,B,B,R,R,B,B,R.
That fixes *how many* champions each side has revealed at step k; *which* ones is
unknown, so we sample uniformly within each side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PICK_SIDES = "BRRBBRRBBR"
# Blue/red revealed counts after k picks, k = 0..10.
REVEAL_COUNTS = [
    (PICK_SIDES[:k].count("B"), PICK_SIDES[:k].count("R")) for k in range(11)
]

# Duration bins in minutes; open at both ends. Chosen to straddle the 15/20-min
# surrender spikes and give resolution through the interesting 25-40 range.
EDGES_MIN = [18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]
N_BINS = len(EDGES_MIN) + 1
BIN_CENTERS = np.array(
    [16.0] + [(EDGES_MIN[i] + EDGES_MIN[i + 1]) / 2 for i in range(len(EDGES_MIN) - 1)] + [49.0],
    dtype=np.float32,
)

ROLES = ["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"]
ROLE_UNK = len(ROLES)          # enemy roles are not known during a draft
N_ROLES = len(ROLES) + 1

TIERS = [
    "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD",
    "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER",
]
QUEUES = ["SOLORANKED", "FLEXRANKED"]

PAD = 0  # champion index 0 is reserved for "empty slot"


class Vocab:
    def __init__(self, champions: list[str]):
        self.champions = list(champions)
        self.index = {c: i + 1 for i, c in enumerate(self.champions)}

    def __len__(self) -> int:
        return len(self.champions) + 1  # + PAD

    def get(self, name) -> int:
        return self.index.get(name, PAD)

    def to_json(self) -> dict:
        return {
            "champions": self.champions,
            "roles": ROLES,
            "tiers": TIERS,
            "queues": QUEUES,
            "edges_min": EDGES_MIN,
            "bin_centers": BIN_CENTERS.tolist(),
        }


def load_matches(data_dir: Path) -> list[dict]:
    shards = sorted((data_dir / "matches").glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards in {data_dir / 'matches'}")
    rows: list[dict] = []
    seen: set[str] = set()
    for shard in shards:
        for row in pq.read_table(shard).to_pylist():
            if row["game_id"] in seen:
                continue
            seen.add(row["game_id"])
            rows.append(row)
    return rows


def build_vocab(rows: list[dict]) -> Vocab:
    names: set[str] = set()
    for r in rows:
        for p in r["blue_picks"] + r["red_picks"]:
            if p["champion"]:
                names.add(p["champion"])
        names.update(b for b in r["blue_bans"] + r["red_bans"] if b)
    return Vocab(sorted(names))


def encode(rows: list[dict], vocab: Vocab) -> dict[str, np.ndarray]:
    """Pack matches into flat arrays. Blue is always index 0, red index 1."""
    n = len(rows)
    picks = np.zeros((n, 2, 5), dtype=np.int32)
    roles = np.full((n, 2, 5), ROLE_UNK, dtype=np.int32)
    bans = np.zeros((n, 10), dtype=np.int32)
    tier = np.zeros(n, dtype=np.int32)
    queue = np.zeros(n, dtype=np.int32)
    duration = np.zeros(n, dtype=np.float32)
    blue_win = np.zeros(n, dtype=np.float32)

    role_index = {r: i for i, r in enumerate(ROLES)}
    tier_index = {t: i for i, t in enumerate(TIERS)}
    queue_index = {q: i for i, q in enumerate(QUEUES)}

    for i, r in enumerate(rows):
        for side, key in ((0, "blue_picks"), (1, "red_picks")):
            for j, p in enumerate(r[key][:5]):
                picks[i, side, j] = vocab.get(p["champion"])
                roles[i, side, j] = role_index.get(p["position"], ROLE_UNK)
        all_bans = [b for b in (r["blue_bans"] or []) + (r["red_bans"] or []) if b][:10]
        for j, b in enumerate(all_bans):
            bans[i, j] = vocab.get(b)
        tier[i] = tier_index.get(r["tier"], TIERS.index("EMERALD"))
        queue[i] = queue_index.get(r["queue"], 0)
        duration[i] = r["duration_s"] / 60.0
        blue_win[i] = 1.0 if r["blue_win"] else 0.0

    return {
        "picks": picks, "roles": roles, "bans": bans, "tier": tier,
        "queue": queue, "duration": duration, "blue_win": blue_win,
    }


def duration_bin(minutes: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.array(EDGES_MIN, dtype=np.float32), minutes, side="right")


def sample_states(
    enc: dict[str, np.ndarray],
    idx: np.ndarray,
    rng: np.random.Generator,
    ally_role_p: float = 0.95,
    enemy_role_p: float = 0.15,
    force_k: int | None = None,
) -> dict[str, np.ndarray]:
    """Draw one random partial-draft view per match in `idx`.

    Half of ban-phase states are sampled too, so the model is calibrated before
    any champion is locked.
    """
    b = len(idx)
    picks, roles, bans = enc["picks"][idx], enc["roles"][idx], enc["bans"][idx]

    # Ally side is random, so the model learns a side-symmetric function and we
    # get a free 2x augmentation.
    ally_side = rng.integers(0, 2, size=b)
    enemy_side = 1 - ally_side
    ar = np.arange(b)

    ally_c = picks[ar, ally_side]      # (b,5)
    ally_r = roles[ar, ally_side]
    enemy_c = picks[ar, enemy_side]
    enemy_r = roles[ar, enemy_side]

    if force_k is None:
        in_ban_phase = rng.random(b) < 0.2
        k = rng.integers(0, 11, size=b)             # picks revealed
        n_bans = np.where(in_ban_phase, rng.integers(0, 11, size=b), 10)
        k = np.where(in_ban_phase, 0, k)
    else:
        # Fixed draft phase, for measuring how signal accumulates.
        k = np.full(b, force_k)
        n_bans = np.full(b, 10)

    counts = np.array(REVEAL_COUNTS)[k]             # (b,2) blue,red
    ally_n = np.where(ally_side == 0, counts[:, 0], counts[:, 1])
    enemy_n = np.where(ally_side == 0, counts[:, 1], counts[:, 0])

    # Which champions on each side are revealed: uniform within the side.
    def reveal(counts_per_row: np.ndarray) -> np.ndarray:
        order = rng.random((b, 5)).argsort(axis=1)
        rank = order.argsort(axis=1)
        return rank < counts_per_row[:, None]

    ally_m = reveal(ally_n)
    enemy_m = reveal(enemy_n)
    ban_m = np.arange(10)[None, :] < n_bans[:, None]

    # Enemy roles are almost never known mid-draft; ally roles almost always are.
    ally_r = np.where(rng.random((b, 5)) < ally_role_p, ally_r, ROLE_UNK)
    enemy_r = np.where(rng.random((b, 5)) < enemy_role_p, enemy_r, ROLE_UNK)

    ally_win = np.where(ally_side == 0, enc["blue_win"][idx], 1.0 - enc["blue_win"][idx])

    return {
        "ally_c": (ally_c * ally_m).astype(np.int64),
        "ally_r": np.where(ally_m, ally_r, ROLE_UNK).astype(np.int64),
        "ally_m": ally_m.astype(np.float32),
        "enemy_c": (enemy_c * enemy_m).astype(np.int64),
        "enemy_r": np.where(enemy_m, enemy_r, ROLE_UNK).astype(np.int64),
        "enemy_m": enemy_m.astype(np.float32),
        "ban_c": (bans[ar] * ban_m).astype(np.int64),
        "ban_m": ban_m.astype(np.float32),
        "tier": enc["tier"][idx].astype(np.int64),
        "queue": enc["queue"][idx].astype(np.int64),
        "y_bin": duration_bin(enc["duration"][idx]).astype(np.int64),
        "y_min": enc["duration"][idx].astype(np.float32),
        "y_win": ally_win.astype(np.float32),
    }


def role_playrates(rows: list[dict], vocab: Vocab) -> np.ndarray:
    """Champion x role play counts — the legal-action mask of DESIGN.md §4."""
    table = np.zeros((len(vocab), len(ROLES)), dtype=np.int32)
    role_index = {r: i for i, r in enumerate(ROLES)}
    for r in rows:
        for p in r["blue_picks"] + r["red_picks"]:
            ci, ri = vocab.get(p["champion"]), role_index.get(p["position"])
            if ci and ri is not None:
                table[ci, ri] += 1
    return table
