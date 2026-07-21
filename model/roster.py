"""Resolve five Riot IDs into champion pools, roles and elo band.

A static page cannot call OP.GG directly (no CORS headers, and the artifact CSP
blocks cross-origin requests anyway), so the roster is resolved here and baked
into the page at export time.

DESIGN.md §7: `ranked_most_champions` gives season-long counts with no role
attached, so roles come from recent match history, falling back to the
champion's modal role in the training data.

    python -m model.roster --region NA "Faker#KR1" "Player2#NA1" ...
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from ingest.dsl import ParseError, dig, parse
from ingest.opgg import OpggClient, OpggError

ROOT = Path(__file__).resolve().parents[1]

PROFILE_FIELDS = [
    "data.summoner.league_stats[].game_type",
    "data.summoner.league_stats[].tier_info.tier",
    "data.summoner.league_stats[].tier_info.division",
    "data.summoner.ranked_most_champions.my_champion_stats[].champion_name",
    "data.summoner.ranked_most_champions.my_champion_stats[].play",
    "data.summoner.ranked_most_champions.my_champion_stats[].win",
    "data.summoner.ranked_most_champions.my_champion_stats[].game_second",
]

HISTORY_FIELDS = [
    "data.game_history[].participants[].champion_name",
    "data.game_history[].participants[].position",
    "data.game_history[].game_type",
]


def _tier(parsed) -> str | None:
    """Prefer flex rank; fall back to solo, which is often the only one set."""
    stats = dig(parsed, "data", "summoner", "league_stats", default=[]) or []
    by_queue = {}
    for s in stats:
        if isinstance(s, dict):
            tier = dig(s, "tier_info", "tier")
            if tier:
                by_queue[s.get("game_type")] = tier
    return by_queue.get("FLEXRANKED") or by_queue.get("SOLORANKED")


def resolve(client: OpggClient, riot_id: str, region: str, min_games: int) -> dict:
    game_name, _, tag_line = riot_id.partition("#")
    if not tag_line:
        raise ValueError(f"{riot_id!r} is not a Riot ID (expected Name#TAG)")

    profile = parse(
        client.call(
            "lol_get_summoner_profile",
            {
                "game_name": game_name,
                "tag_line": tag_line,
                "region": region,
                "desired_output_fields": PROFILE_FIELDS,
            },
        )
    ).value

    stats = dig(profile, "data", "summoner", "ranked_most_champions",
                "my_champion_stats", default=[]) or []
    pool = []
    for s in stats:
        if not isinstance(s, dict):
            continue
        plays = s.get("play") or 0
        if plays < min_games or not s.get("champion_name"):
            continue
        seconds = s.get("game_second") or 0
        pool.append(
            {
                "champion": s["champion_name"],
                "games": plays,
                "wins": s.get("win") or 0,
                # Per-player, per-champion average game length: the player-level
                # duration prior of DESIGN.md §7.
                "avg_minutes": round(seconds / plays / 60.0, 2) if plays else None,
            }
        )
    pool.sort(key=lambda p: -p["games"])

    # Roles from recent history (capped at 20 games by the API).
    roles: dict[str, Counter] = {}
    try:
        hist = parse(
            client.call(
                "lol_list_summoner_matches",
                {
                    "game_name": game_name,
                    "tag_line": tag_line,
                    "region": region,
                    "limit": 20,
                    "desired_output_fields": HISTORY_FIELDS,
                },
            )
        ).value
        for game in dig(hist, "data", "game_history", default=[]) or []:
            for p in (game.get("participants") or []) if isinstance(game, dict) else []:
                champ, pos = p.get("champion_name"), p.get("position")
                if champ and pos:
                    roles.setdefault(champ, Counter())[pos] += 1
    except (OpggError, ParseError):
        pass

    for entry in pool:
        counts = roles.get(entry["champion"])
        entry["role"] = counts.most_common(1)[0][0] if counts else None

    # Weight by games played, not by distinct champions: a one-trick mid with
    # two off-role support games is a mid, not a support.
    role_games: Counter = Counter()
    for counts in roles.values():
        role_games.update(counts)
    main_role = role_games.most_common(1)
    return {
        "riot_id": riot_id,
        "region": region,
        "tier": _tier(profile),
        "main_role": main_role[0][0] if main_role else None,
        "pool": pool,
        "avg_minutes": (
            round(sum(p["avg_minutes"] * p["games"] for p in pool if p["avg_minutes"])
                  / max(1, sum(p["games"] for p in pool if p["avg_minutes"])), 2)
            if pool else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve a 5-player roster from OP.GG")
    ap.add_argument("riot_ids", nargs="+", help="Name#TAG, up to five")
    ap.add_argument("--region", default="NA")
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts" / "roster.json")
    args = ap.parse_args()

    client = OpggClient()
    players = []
    for riot_id in args.riot_ids[:5]:
        try:
            info = resolve(client, riot_id, args.region, args.min_games)
        except (OpggError, ParseError, ValueError) as exc:
            print(f"[roster] {riot_id}: FAILED ({exc})")
            continue
        players.append(info)
        top = ", ".join(f"{p['champion']}({p['games']})" for p in info["pool"][:6])
        print(
            f"[roster] {riot_id:28} tier={info['tier'] or '-':13} "
            f"role={info['main_role'] or '-':8} avg={info['avg_minutes'] or '-'}min  {top}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"players": players}, indent=1))
    print(f"[roster] wrote {args.out} ({len(players)} players)")


if __name__ == "__main__":
    main()
