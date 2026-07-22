"""Bundle the trained ensemble into a single self-contained HTML page.

The page runs inference in plain JavaScript — no ONNX/TF runtime, no CDN, no
network at all. That is partly a constraint (artifact CSP blocks external
scripts; file:// blocks fetch) and partly the point: a ~90k-parameter DeepSets
forward pass is an embedding lookup, two sums and three matmuls.

    python -m model.export
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch

from .data import N_BINS
from .net import DraftNet

ROOT = Path(__file__).resolve().parents[1]

# Fixed order; the JS decoder slices the flat buffer using this list.
TENSOR_ORDER = [
    "champ.weight",
    "role.weight",
    "tier.weight",
    "queue.weight",
    "trunk.0.weight",
    "trunk.0.bias",
    "trunk.3.weight",
    "trunk.3.bias",
    "duration_head.weight",
    "duration_head.bias",
    "win_head.weight",
    "win_head.bias",
    "champ_effect.weight",
    "bin_dir",
]


def flatten(state: dict[str, torch.Tensor]) -> tuple[bytes, list]:
    shapes, chunks = [], []
    for name in TENSOR_ORDER:
        t = state[name].detach().cpu().float().numpy().astype(np.float32)
        shapes.append([name, list(t.shape)])
        chunks.append(t.ravel())
    return np.concatenate(chunks).tobytes(), shapes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--web", type=Path, default=ROOT / "web")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "index.html")
    ap.add_argument("--roster", type=Path, default=None)
    args = ap.parse_args()

    meta = json.loads((args.artifacts / "meta.json").read_text())
    vocab = meta["vocab"]
    n_champions = len(vocab["champions"]) + 1
    cfg = meta["config"]

    members, shapes = [], None
    for i in range(cfg["members"]):
        path = args.artifacts / f"member{i}.pt"
        if not path.exists():
            continue
        model = DraftNet(n_champions, dim=cfg["dim"], hidden=cfg["hidden"])
        model.load_state_dict(torch.load(path, map_location="cpu"))
        blob, shapes = flatten(model.state_dict())
        members.append(base64.b64encode(blob).decode("ascii"))
    if not members:
        raise SystemExit("no member checkpoints found — run `python -m model.train` first")

    win_path = args.artifacts / "win_model.json"
    win_model = json.loads(win_path.read_text()) if win_path.exists() else None

    roster_path = args.roster or (args.artifacts / "roster.json")
    roster = json.loads(roster_path.read_text()) if roster_path.exists() else None

    bundle = {
        "config": {**cfg, "n_bins": N_BINS, "n_champions": n_champions},
        "vocab": vocab,
        "shapes": shapes,
        "members": members,
        "role_playrates": meta["role_playrates"],
        "stats": {
            "n_matches": meta["n_matches"],
            "baseline": meta["baseline"],
            "summary": meta["summary"],
            "phases": meta.get("phases", {}),
            "trained_at": meta["trained_at"],
        },
        "roster": roster,
        "win_model": win_model,
    }

    template = (args.web / "template.html").read_text()
    css = (args.web / "style.css").read_text()
    js = (args.web / "infer.js").read_text() + "\n" + (args.web / "app.js").read_text()

    html = (
        template
        .replace("/*STYLE*/", css)
        .replace("/*BUNDLE*/", json.dumps(bundle, separators=(",", ":")))
        .replace("/*APP*/", js)
    )
    (args.artifacts / "bundle.json").write_text(json.dumps(bundle, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    size_mb = len(html.encode()) / 1e6
    print(
        f"[export] {args.out} ({size_mb:.2f} MB, {len(members)} ensemble members, "
        f"{n_champions - 1} champions, trained on {meta['n_matches']} matches)"
    )


if __name__ == "__main__":
    main()
