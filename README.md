# longgame

Champion select recommendations that maximise **game duration** instead of win rate.

Give it the current draft — bans, both sides' picks — and a role, and it ranks the
champions that make the longest game. Output is a full distribution over game
length, not just a point estimate, and it updates turn by turn.

The end product is a **single self-contained HTML file** that runs the model in
plain JavaScript with no network calls, no CDN and no inference runtime.

See [DESIGN.md](DESIGN.md) for why it's built this way. The short version:

- **It is barely an RL problem.** Horizon is ≤ 10 decisions, reward is terminal and
  directly observed, and there is no simulator. It collapses to learning
  `V(board) = E[duration | partial draft]` offline, then acting greedily.
- **The draft is a set, not a sequence.** Real pick order is not recoverable from
  match data (Riot doesn't expose it either), so the model is permutation-invariant
  by construction and trains on random reveals of finished drafts.
- **The enemy is not an adversary.** They optimise win rate, you optimise duration.
  Not opposed, so this is expectimax over a learned behaviour policy, not minimax.

## Quick start

```bash
uv venv .venv && uv pip install --python .venv/bin/python requests pyarrow pandas pytest torch

.venv/bin/python -m ingest.crawl --region NA --target 150000 --workers 10
.venv/bin/python -m model.train  --members 5 --epochs 14
.venv/bin/python -m model.roster --region NA "You#NA1" "Jgl#NA1" "Mid#NA1" "Adc#NA1" "Sup#NA1"
.venv/bin/python -m model.export

open web/index.html
```

Each step is resumable and independent; the crawler can keep running while you train.

## Layout

| Path | What |
|---|---|
| `ingest/dsl.py` | Parser for OP.GG's compact class-DSL response format |
| `ingest/opgg.py` | MCP client — sessions, retries, verified field selections |
| `ingest/records.py` | Match normalisation and filtering rules |
| `ingest/crawl.py` | Snowball crawler (SQLite state, Parquet out, raw archived) |
| `model/data.py` | Partial-draft sampling, duration binning, vocab |
| `model/net.py` | DeepSets value model |
| `model/train.py` | Ensemble training + baselines |
| `model/roster.py` | Five Riot IDs → champion pools, roles, elo |
| `model/export.py` | Bundles weights + UI into one HTML file |
| `web/infer.js` | JS port of the forward pass (parity-tested against PyTorch) |
| `model/gbdt.py` | XGBoost vs ridge — does nonlinearity help? (no) |
| `model/matchups.py` | Do lane matchups carry signal? (no) |
| `model/ceiling.py` | How much is predictable at draft time? (~2%) |
| `tests/` | Parser, filtering, and JS↔PyTorch parity |

## Data

No Riot API key. Everything comes from the OP.GG MCP server
(`https://mcp-api.op.gg/mcp`), which returns duration, both sides' bans, all ten
champions with positions, elo band and queue in one call per match.

Discovery snowballs: every match detail hands back ten summoner identities, so a
small seed of champion leaderboards expands indefinitely. Measured ~3.3 matches/s
sustained at 10 workers.

Please keep concurrency modest — it's an unmetered public endpoint. The crawler
archives every raw response to `data/raw-*.jsonl.gz` so a schema change never
forces a re-crawl.

## Honest expectations

The draft explains only a small share of duration variance; most of it is how the
game actually gets played. Realistic R² is 0.02–0.06, which is worth **+1 to +3
minutes** of expected duration and a visible shift in `P(game > 35 min)`.

Two things follow from that, both handled in the code but worth internalising:

1. **`argmax` is dangerous here.** It doesn't just surface the best champion, it
   hunts for wherever the model over-predicts. Ensemble pessimism
   (`mean − λ·spread`), a data-support floor, and roster-pool restriction all
   exist to fight this. The `spread` column in the UI is often larger than the gap
   between the top few candidates — that is the honest signal, not a bug.
2. **Check the baselines.** `model/train.py` prints a predict-the-mean MAE and a
   prior-distribution NLL every run. If the model isn't beating those, it hasn't
   learned anything, regardless of how confident the page looks.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

`tests/test_parity.py` runs the shipped `web/infer.js` under Node and requires it
to agree with PyTorch to well inside the model's own noise floor — a silent
mismatch there would make the page confidently wrong.
