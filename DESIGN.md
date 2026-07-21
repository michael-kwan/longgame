# longgame — a draft policy that maximizes game duration

## Goal

Turn-by-turn champion select recommendations. Given the current board (bans + picks
on both sides) and a target role, return the champion that maximizes expected game
**duration** — not win rate.

---

## 1. Framing: this is not (mostly) an RL problem

The instinct is "RL policy, self-play, AlphaZero." Resist it. Look at the structure:

- Horizon is **≤ 10 decisions** (5 picks if you control your whole team, 1 in soloq).
- Reward is **terminal and scalar** (game duration), observed directly in data.
- The state is **fully observed** and small (20 champion slots + context).
- There is **no simulator**. You cannot roll out a LoL game. Every bit of dynamics
  has to be learned from logged matches anyway.

So the whole problem reduces to learning one thing:

> **V(board) = E[duration | this partial draft]**

That's Monte Carlo policy evaluation on logged data. Once you have `V`, acting greedily
w.r.t. it is *one step of policy improvement* — which is already most of the available
gain in a horizon this short. That is legitimately RL (offline MC value learning +
greedy improvement), it just doesn't need policy gradients, self-play, or a replay buffer.

**Plan: build the value model first. Add search only if it measurably beats greedy.**

### The opponent is not an adversary

They're optimizing win rate; you're optimizing duration. Those aren't opposed, so this
is **not** minimax. Model the enemy team (and, in soloq, your own teammates) as a
**stochastic behavior policy** learned from data, and take an **expectation** over their
picks. Expectimax, not minimax.

---

## 2. What actually makes games long

Worth holding in your head, because it predicts what the model should learn and gives
you a sanity check on its outputs.

Duration is bimodal-ish with mass spikes at ~15 min and ~20 min (surrender timers) and
a long right tail. So maximizing duration decomposes into roughly:

1. **Don't get stomped** → avoid the 15-min FF.
2. **Don't stomp** → avoid closing early.
3. **Be structurally bad at ending** → no siege, weak objective/tower pressure, scaling
   carries, tanks, poke, disengage, waveclear that stalls rather than pushes.

(1) and (2) together mean the policy will implicitly steer toward **balanced win
probability** — a coinflip game runs long. That's why the win-probability head below
is more than an auxiliary task; it's half the causal story.

---

## 3. Data — OP.GG MCP (no Riot API key needed)

**Verified against the live endpoint.** `lol_get_summoner_game_detail` returns
everything the model needs, in one call per match:

```
game_length_second              ← the label
teams[].key                     ← BLUE / RED
teams[].banned_champions_names  ← bans, per side
teams[].participants[].champion_name
teams[].participants[].position ← TOP/JUNGLE/MID/ADC/SUPPORT
teams[].game_stat.is_win
average_tier_info.tier          ← elo band
game_type                       ← SOLORANKED / FLEXRANKED / ...
created_at
teams[].participants[].summoner.{game_name, tagline, puuid}
```

### Crawl strategy: snowball

That last field is what makes this viable. Each match detail hands back **10 new
summoner identities**, so discovery is self-sustaining:

```
seed: lol_list_champion_leaderboard (top master+ per champion, ~170 champs)
  → lol_list_summoner_matches (20 game ids + created_at per summoner)
    → lol_get_summoner_game_detail (full draft + duration)
      → 10 new summoners ────────┐
      └──────────────────────────┘  loop
```

Dedupe on `game_id`; keep a visited set of puuids and a frontier queue. Snowballing
also drifts you off the master+ seed into surrounding elo bands naturally — sample the
frontier by `average_tier_info.tier` to hit whatever distribution you want.

### Measured throughput

| Concurrency | Rate | Errors |
|---|---|---|
| 4 workers | 1.25 matches/s | 0 |
| 12 workers | 4.22 matches/s | 0 |

~15k matches/hour at 12 workers. **150k matches ≈ 10 hours.** No auth, no key, no
quota. Compare Riot's personal dev key: 100 req/2 min *and* it expires every 24 hours.

Be a good citizen — this is an unmetered public endpoint. Cap at ~8–12 workers, back
off on any 429/5xx, cache aggressively, and never re-fetch a `game_id` you already have.

### What you give up vs. Riot match-v5

| Lost | Impact | Workaround |
|---|---|---|
| `gameVersion` (patch) | Low | Map `created_at` → patch via a date table |
| Ban `pickTurn` order | **None** | §4 is order-agnostic by design |
| `gameEndedInEarlySurrender` | Low | Drop duration < 300s (remakes are obvious — saw a 66s game in sampling) |
| Clean ladder enumeration (`league-v4`) | **None** | Snowball is strictly better coverage |
| Timeline / per-minute data | Low | Not needed for a draft-only model |

The one real cost is **dependence on an unofficial third-party endpoint**. Its schema
can change without notice. Mitigate by archiving raw responses to disk on ingest so a
schema break never forces a re-crawl.

### Response format note
Responses come back in OP.GG's compact class-DSL, not JSON:

```
GameDetail("eq3Kyi...=",1466,"SOLORANKED",...,[Team("BLUE",["Malphite",...],...)])
```

It's token-efficient for LLM use but needs a small parser for bulk ingest. Write one
tolerant parser in `ingest/parse.py` and unit-test it against archived fixtures — this
is the most likely silent-breakage point in the pipeline.

### Scope for v1
`SOLORANKED`, NA + KR, whatever elo the snowball reaches, last ~6 patches.

---

## 4. State, action, mask

### The order problem, dissolved

Since real pick order is unrecoverable, treat the draft as **set completion**, not a
sequence. `V` conditions on *what is on the board now*, not on how it got there. Train
by randomly masking subsets of each finished draft — every match becomes ~20 training
states. The turn structure (whose pick is next, how many remain) is known at *inference*
from the draft format; it never has to be recovered from data.

### State
| Component | Encoding |
|---|---|
| 10 bans | champ id + `type=BAN` + side + ban turn (order known) |
| 10 picks | champ id + `type=PICK` + side + role-or-`UNK` |
| empty slots | `MASK` token |
| context | patch, queue, region, avg tier |

Your own picks carry a known role; enemy picks carry `UNK`. Randomly drop roles during
training so the model is calibrated for that asymmetry at inference.

### Action space
~170 champions, masked down to legal:
- not banned, not already picked
- **playable in the target role** — champ×role playrate ≥ threshold at this patch/tier
- **has data support** — ≥ N historical games in that role (see §6 on why this matters)

---

## 5. Model

One shared trunk, four heads. A small transformer over the 22 slot tokens.

```
tokens: [CLS] [CTX] [ban×10] [pick×10]
trunk:  6 layers, d=256, 8 heads   (~10M params incl. champion embeddings)
```

Heads off `[CLS]`:

| Head | Output | Purpose |
|---|---|---|
| `p_duration` | softmax over ~20 duration bins (2-min bins, 15→50+) | **the reward model** |
| `p_win` | blue-side win probability | auxiliary; encodes the "balance" half of §2 |
| `p_next` | softmax over champions | **behavior policy** — enemy/teammate model + search prior |
| `p_surrender` | P(game ends by surrender) | interpretability, and a useful sub-objective |

Predicting the **full duration distribution** rather than a scalar is the key choice:
it costs nothing and lets you swap objectives without retraining —
`E[duration]`, `P(duration > 35)`, or a CVaR/quantile target, all read off the same
softmax. My guess is `P(duration > T)` is what you actually want.

Augmentation: mirror blue/red sides.

---

## 6. Acting — and the trap

### Baseline policy (build this first)
For each legal champion `c`: evaluate `V(board + c)`. Pick the argmax. That's ~170
forward passes = one batch. Milliseconds.

### The trap: offline distribution shift
`V` is trained on human drafts. `argmax` deliberately walks *off* that distribution,
straight into states the model has never seen — and neural nets extrapolate
confidently and wrongly. Un-defended, this policy will recommend Yuumi top into Darius
and promise you a 45-minute game. This is the single most likely way the project
produces impressive-looking garbage.

Three defenses, all cheap, use all three:
1. **Pessimism** — ensemble of 5 value heads; score = `mean − λ·std`.
2. **Support constraint** — hard-mask champions with thin data in that role/matchup.
3. **Behavior regularization** — BCQ-style, only consider `c` where
   `p_next(c) ≥ τ · max_c' p_next(c')`. Keeps candidates on the manifold.

### If greedy isn't enough
Sampled expectimax: for each candidate, roll the remaining draft — `p_next` for slots
you don't control, your own policy for slots you do — N≈64 rollouts, average the
terminal duration. Depth ≤ 9, fully batchable. Only bother if it beats greedy on §7.

---

## 7. Evaluation

Ordered by how much I'd trust them.

1. **Matched-pair natural experiment** *(the good one)*. Find real games sharing a
   board state and role where different champions were actually picked. Compare their
   realized durations. This is model-free counterfactual evidence that the model's
   preferences track reality.
2. **Agreement backtest.** Bucket held-out games by whether the actual pick matched the
   model's top-k. Compare mean duration across buckets.
3. **Calibration** of `p_duration` on held-out data — reliability curves per bin. A
   miscalibrated distribution head silently breaks any quantile objective.
4. **Beat the dumb baselines** before believing anything: per-champion mean duration,
   and a hand-built comp-feature model (tank count, scaling index, siege, waveclear).
   If the transformer can't beat those, it hasn't learned interactions.
5. **Online**: play games with it, log durations, A/B against your own instincts.

### Expectation setting
The draft explains a **small** fraction of duration variance — realistically R² ≈ 0.02–0.06.
Duration sd is ~7–8 min. So a good policy is worth maybe **+1 to +3 min** in expectation,
and more visibly a shift in `P(game > 35 min)` — call it 35% → 45%. That's a real,
usable effect. It is not a 50-minute-games-on-demand button. Decide now that this is
the bar, so the result can be judged honestly later.

Also: in soloq you control **one** pick out of five. Same machinery — those turns are
just expectations instead of maxima — but the achievable lift shrinks accordingly.

---

## 8. Build order

| # | Milestone | Output |
|---|---|---|
| 1 | Ingest — snowball crawler + DSL parser → Parquet | ~150k matches, resumable, raw archived |
| 2 | EDA — duration distribution, FF spikes, per-champ marginals | confirms §2, gives baseline #4 |
| 3 | Champ×role playrate table | the legal-action mask |
| 4 | Baseline duration model (GBDT on comp features) | the number to beat |
| 5 | Transformer trunk + `p_duration` + `p_win` | the value model |
| 6 | Greedy policy + pessimism/support guards | **usable recommender** |
| 7 | Matched-pair eval | do we believe it? |
| 8 | `p_next` head + expectimax | only if 7 says the model is real |
| 9 | CLI / live draft input | ergonomics |

Ship 1–7 before touching 8. Milestone 6 is already the product.

---

## 9. Other OP.GG tools worth using

Beyond ingest (§3), the same server covers inference-time needs:

- `lol_list_lane_meta_champions` — champion×role playrates for the legal-action mask (§4),
  instead of deriving it from the crawl.
- `lol_list_champions` / `lol_list_items` — canonical champion id ↔ name mapping.
- `lol_get_champion_analysis` — sanity-check that the model's picks are coherent for the
  current patch.

The Riot API key in `tft-playerbase/.env` stays as a fallback if OP.GG's schema breaks.

---

## Open questions

- **Soloq (1 pick) or full-team control (5 picks)?** Changes achievable lift a lot, not
  the architecture.
- **Objective: `E[duration]` or `P(duration > T)`?** The distributional head defers this,
  but it should drive how the eval is framed.
- **Rank band?** Low elo has longer, messier games and different FF behavior; high elo
  drafts are cleaner. Affects both data volume and what "long" means.
