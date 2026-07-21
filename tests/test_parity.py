"""The page's JS inference must agree with PyTorch, or the UI lies silently.

Builds a few draft states, runs them through both implementations, and
requires expected-duration agreement to well under the model's own noise floor.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.data import N_BINS, ROLE_UNK  # noqa: E402
from model.net import DraftNet  # noqa: E402

BUNDLE = ROOT / "artifacts" / "bundle.json"
STATES = Path(__file__).parent / "fixtures" / "parity_states.json"

pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(), reason="run `python -m model.export` first"
)


def build_states(n_champions: int) -> list[dict]:
    """Empty board, ban phase, mid draft, and a full draft."""
    return [
        {"allyC": [], "allyR": [], "enemyC": [], "enemyR": [], "bans": [], "tier": 5, "queue": 0},
        {"allyC": [], "allyR": [], "enemyC": [], "enemyR": [],
         "bans": [3, 11, 27, 40, 55], "tier": 7, "queue": 1},
        {"allyC": [7, 19], "allyR": [0, 2], "enemyC": [31], "enemyR": [ROLE_UNK],
         "bans": [3, 11, 27, 40, 55, 61, 70, 88, 90, 101], "tier": 5, "queue": 0},
        {"allyC": [7, 19, 22, 44, 60], "allyR": [0, 1, 2, 3, 4],
         "enemyC": [31, 33, 47, 52, 66], "enemyR": [ROLE_UNK] * 5,
         "bans": [3, 11, 27, 40, 55, 61, 70, 88, 90, 101], "tier": 9, "queue": 0},
    ]


def torch_predict(bundle, states) -> list[dict]:
    cfg = bundle["config"]
    centers = np.array(bundle["vocab"]["bin_centers"], dtype=np.float32)
    results = []

    nets = []
    for i in range(len(bundle["members"])):
        path = ROOT / "artifacts" / f"member{i}.pt"
        model = DraftNet(cfg["n_champions"], dim=cfg["dim"], hidden=cfg["hidden"])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        nets.append(model)

    for st in states:
        def pad(ids, fill=0, n=5):
            return ids + [fill] * (n - len(ids))

        batch = {
            "ally_c": torch.tensor([pad(st["allyC"])]),
            "ally_r": torch.tensor([pad(st["allyR"], ROLE_UNK)]),
            "ally_m": torch.tensor([[1.0] * len(st["allyC"]) + [0.0] * (5 - len(st["allyC"]))]),
            "enemy_c": torch.tensor([pad(st["enemyC"])]),
            "enemy_r": torch.tensor([pad(st["enemyR"], ROLE_UNK)]),
            "enemy_m": torch.tensor([[1.0] * len(st["enemyC"]) + [0.0] * (5 - len(st["enemyC"]))]),
            "ban_c": torch.tensor([pad(st["bans"], 0, 10)]),
            "ban_m": torch.tensor([[1.0] * len(st["bans"]) + [0.0] * (10 - len(st["bans"]))]),
            "tier": torch.tensor([st["tier"]]),
            "queue": torch.tensor([st["queue"]]),
        }
        means, probs_sum, win = [], np.zeros(N_BINS), 0.0
        with torch.no_grad():
            for model in nets:
                logits, win_logit = model(batch)
                p = torch.softmax(logits, dim=-1).numpy()[0]
                means.append(float((p * centers).sum()))
                probs_sum += p / len(nets)
                win += float(torch.sigmoid(win_logit)) / len(nets)
        results.append(
            {"mean": float(np.mean(means)), "std": float(np.std(means)),
             "win": win, "probs": probs_sum.tolist(), "means": means}
        )
    return results


def test_js_matches_torch():
    bundle = json.loads(BUNDLE.read_text())
    states = build_states(bundle["config"]["n_champions"])
    STATES.parent.mkdir(parents=True, exist_ok=True)
    STATES.write_text(json.dumps(states))

    proc = subprocess.run(
        ["node", str(Path(__file__).parent / "parity.mjs")],
        capture_output=True, text=True, check=True,
    )
    js = json.loads(proc.stdout)
    ref = torch_predict(bundle, states)

    assert len(js) == len(ref)
    for i, (a, b) in enumerate(zip(js, ref)):
        # GELU is the tanh approximation in JS, exact erf in torch, so allow a
        # small tolerance — but far tighter than the model's own MAE.
        assert abs(a["mean"] - b["mean"]) < 0.02, f"state {i}: E[dur] {a['mean']} vs {b['mean']}"
        assert abs(a["std"] - b["std"]) < 0.02, f"state {i}: spread {a['std']} vs {b['std']}"
        assert abs(a["win"] - b["win"]) < 0.01, f"state {i}: win {a['win']} vs {b['win']}"
        assert np.abs(np.array(a["probs"]) - np.array(b["probs"])).max() < 0.005


def test_states_differ():
    """Guards against a degenerate model that ignores its input."""
    bundle = json.loads(BUNDLE.read_text())
    ref = torch_predict(bundle, build_states(bundle["config"]["n_champions"]))
    spread = max(r["mean"] for r in ref) - min(r["mean"] for r in ref)
    assert spread > 0.05, f"model output barely varies across drafts ({spread:.4f} min)"
