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

Un-defended, this policy will recommend Yuumi top into Darius and promise you a
45-minute game. **Low sample size is the source of that error, but `argmax` is what
makes it fatal** — two separate problems:

1. **Sparse support → extrapolation.** The model has never seen Yuumi top. It doesn't
   respond with "unsure"; it interpolates from champion embeddings and outputs a
   confident number that is unconstrained by data.
2. **`argmax` actively hunts for those errors.** Even if prediction error were small
   and zero-mean everywhere, maximizing over ~170 candidates systematically selects the
   ones the model *over*-predicts. The optimizer's curse: the winner of a max is
   disproportionately likely to be an overestimate. So error doesn't average out — the
   policy seeks it.

That second point is why "just collect more data" doesn't fix it. You need to make the
policy *prefer* candidates it has evidence for:

1. **Pessimism** — ensemble of 5 value heads; score = `mean − λ·std`. Directly cancels
   the optimizer's curse: candidates the ensemble disagrees on get penalized.
2. **Support constraint** — hard-mask champions with thin data in that role/matchup.
3. **Behavior regularization** — BCQ-style, only consider `c` where
   `p_next(c) ≥ τ · max_c' p_next(c')`. Keeps candidates on the manifold.

Note that the roster feature (§7) does most of this work for free: restricting picks to
champions your players actually play collapses the action space from ~170 to ~10 per
slot, and every survivor is by construction well-supported. Keep the guards anyway —
they're cheap and they cover the enemy-side rollouts too.

### If greedy isn't enough
Sampled expectimax: for each candidate, roll the remaining draft — `p_next` for slots
you don't control, your own policy for slots you do — N≈64 rollouts, average the
terminal duration. Depth ≤ 9, fully batchable. Only bother if it beats greedy on §7.

---

## 7. The roster — your five players

You control all five picks and you know who's playing. That's a much stronger setting
than soloq, and it changes the action space more than the model.

### Input
Five Riot IDs, assigned to roles. **Verified live** via `lol_get_summoner_profile`:

```
league_stats[].{game_type, tier_info.{tier,division,lp}}    ← elo band, per queue
ranked_most_champions.my_champion_stats[].{champion_name, play, win, game_second}
```

Real response for a master-tier mid:
`Orianna(455 games) Blitzcrank(58) Thresh(38) Xerath(25) Ziggs(22) Lee Sin(3) …`

### Three things this buys

**1. A per-slot action mask.** Champion pool with play counts, thresholded at ~N≥5
games. Action space per slot drops from ~170 to ~10. This is both the feature you asked
for *and* the strongest defense against §6's failure mode.

**2. Elo conditioning.** Feed the roster's average tier into the context token so `V` is
evaluated in the right band. Games are meaningfully longer and messier at low elo.

**3. A per-player duration prior — free, and worth more than it looks.** `game_second /
play` per champion gives each player's *actual average game length*. Some players simply
play long games: they don't surrender, they scale, they farm. That's a player-level
random effect that the draft-only model structurally cannot see. Add it as a context
feature (roster mean + spread). My guess is this is a bigger single lever than any
individual champion choice.

### Two wrinkles

- **`ranked_most_champions` has no role split.** The pool is season-long champion counts
  with no position attached. Get roles from `lol_list_summoner_matches`
  (`participants[].position`), but that caps at 20 matches — thin. Merge the two: use
  the season pool for counts, the recent history for role assignment, and fall back to
  a champion's modal role from `lol_list_lane_meta_champions` when a player's own
  history is silent.
- **Flex rank is often null.** The sampled player had `FLEXRANKED → TierInfo(null,null,null)`
  despite being master in soloq. Fall back to soloq tier when flex is missing.

### Queue scope
Ranked 5s only: `game_type ∈ {SOLORANKED, FLEXRANKED}`, drop ARAM/normals/Arena.
Flex alone is far too thin to train on, so **train on both with a queue token** and
evaluate on flex. Drafts are drafts; the queue token absorbs the behavioral difference
(premades coordinate, surrender less, and play longer).

---

## 8. Output: watching the distribution move

You want `E[duration]` *and* the distribution, updating as the draft proceeds. The
distributional head (§5) gives this directly — it's the same softmax, read at each state.

Per turn, for the current board:

- **Ranked candidate list** for the slot on the clock: champion, `E[duration]`,
  `Δ` vs. the current board, and the pessimism-adjusted score actually used to rank.
- **The distribution itself** — the ~20-bin softmax, drawn as a density.
- **The trajectory** — one density per completed draft step, as a ridgeline: how the
  distribution sharpens and shifts from ban phase to final pick. Overlay a running
  `E[duration]` line.

The ridgeline is the interesting artifact. Early bans barely move it; a scaling-carry
pick should visibly shift mass past 35 minutes and thin out the 15-minute surrender
spike. If it *doesn't* do that, the model is wrong and you'll see it immediately —
which makes this a debugging tool, not just a display.

---

## 9. Evaluation

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

Controlling all five picks (§7) sits at the good end of that range, since every turn is
a maximization rather than an expectation over a teammate you can't influence.

---

## 10. Build order

| # | Milestone | Output |
|---|---|---|
| 1 | Ingest — snowball crawler + DSL parser → Parquet | ~150k matches, resumable, raw archived |
| 2 | EDA — duration distribution, FF spikes, per-champ marginals | confirms §2, gives baseline #4 |
| 3 | Champ×role playrate table | the legal-action mask |
| 4 | Baseline duration model (GBDT on comp features) | the number to beat |
| 5 | Transformer trunk + `p_duration` + `p_win` | the value model |
| 6 | Roster resolver — 5 Riot IDs → pools, roles, elo, duration priors | per-slot action masks |
| 7 | Greedy policy + pessimism/support guards | **usable recommender** |
| 8 | Turn-by-turn UI — ranked picks + ridgeline | the thing you actually look at |
| 9 | Matched-pair eval | do we believe it? |
| 10 | `p_next` head + expectimax | only if 9 says the model is real |

Ship 1–9 before touching 10. Milestone 8 is already the product.

---

## 11. Other OP.GG tools worth using

Beyond ingest (§3), the same server covers inference-time needs:

- `lol_list_lane_meta_champions` — champion×role playrates for the legal-action mask (§4),
  instead of deriving it from the crawl.
- `lol_list_champions` / `lol_list_items` — canonical champion id ↔ name mapping.
- `lol_get_champion_analysis` — sanity-check that the model's picks are coherent for the
  current patch.

The Riot API key in `tft-playerbase/.env` stays as a fallback if OP.GG's schema breaks.

---

## Results so far (12.5k matches)

Built and running end to end. What the numbers actually say:

| | R² (held out) | MAE |
|---|---|---|
| Predict the mean | 0 | 4.97 min |
| Ridge, bag-of-champions, full draft only | **+0.0168** | 4.93 min |
| DraftNet at 10 picks revealed | +0.0130 | 4.96 min |
| DraftNet averaged over all draft phases | +0.0004 | 4.95 min |

Signal by draft phase (picks revealed → R²):
`0: −0.009 · 2: −0.009 · 4: +0.001 · 6: +0.010 · 8: +0.009 · 10: +0.013`

Three honest readings:

1. **The signal is real but late.** On an empty board the model is *worse* than
   predicting the average; it only earns its keep once most of the draft is
   visible. Prediction spread widens from 0.56 to 1.13 min over the same range,
   so late-draft recommendations differ by about a minute — consistent with the
   "+1 to +3 min" ceiling, at the low end.
2. **Ridge still edges it out on full drafts.** The net does the broader job
   (every partial state, full distribution, ensemble spread) but has not yet
   beaten the linear baseline where they compete directly. Per §9 that is the bar,
   so this is a known gap, not a rounding error. `model/train.py` prints the ridge
   number every run so it can't be quietly forgotten.
3. **It is data-limited, not architecture-limited.** Ridge R² rose from +0.011 at
   10.7k matches to +0.017 at 11.7k — still climbing steeply. The crawl is the
   lever; 100k+ is where the design's 0.02–0.06 estimate should be tested.

Two findings worth keeping:

- **The MLP alone lost to plain ridge.** It needed an explicit per-champion
  additive channel (`champ_effect`) to recover main effects, plus its own learning
  rate — at the shared rate the table stayed inert (weights sd 0.026, predictions
  barely varying). With `--effect-lr-mult 20` the spread went to 1.13 min.
- **Sum-pooling was actively harmful.** Feature scale grew as the draft filled,
  so prediction spread *shrank* with more information — backwards. Mean-pooling
  fixed the direction.

## Settled

- Not RL — offline value learning + greedy improvement (§1).
- Set completion, not sequence modeling (§4).
- Enemy modeled as a win-rate-optimizing behavior policy, expectimax not minimax (§2).
- Ranked 5s only; full control of all five picks (§7).
- Report `E[duration]` and the full distribution per turn (§8).
- Roster set by five Riot IDs → champion pools, roles, elo band (§7).

## Still open

- **Region(s) to crawl.** Affects volume and meta. NA only, or NA+KR+EUW?
- **Elo band to train on.** The roster's band is what matters at inference, but training
  wide with a tier token probably beats training narrow. Worth an ablation.
- **Off-pool escape hatch?** Strict pool masking is safe but may block a genuinely great
  pick a player could learn. Option: surface off-pool picks in a separate, clearly
  flagged list rather than mixing them into the ranking.
