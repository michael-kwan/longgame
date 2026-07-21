"""Normalise parsed OP.GG responses into match rows.

One row per match, both sides' drafts intact. Filtering rules follow DESIGN.md §3:
ranked 5s only, remakes dropped by duration since OP.GG has no early-surrender flag.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import pyarrow as pa

from .dsl import dig

RANKED_QUEUES = {"SOLORANKED", "FLEXRANKED"}
MIN_DURATION_S = 300   # below this is a remake, not a game
MAX_DURATION_S = 5400  # 90 min; anything beyond is a data error

_PLAYER = pa.struct(
    [
        ("champion", pa.string()),
        ("position", pa.string()),
        ("puuid", pa.string()),
        ("game_name", pa.string()),
        ("tagline", pa.string()),
    ]
)

SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("region", pa.string()),
        ("created_at", pa.string()),
        ("duration_s", pa.int32()),
        ("queue", pa.string()),
        ("tier", pa.string()),
        ("blue_win", pa.bool_()),
        ("blue_bans", pa.list_(pa.string())),
        ("red_bans", pa.list_(pa.string())),
        ("blue_picks", pa.list_(_PLAYER)),
        ("red_picks", pa.list_(_PLAYER)),
    ]
)


class Summoner(NamedTuple):
    game_name: str
    tagline: str
    puuid: str | None = None


class MatchRef(NamedTuple):
    game_id: str
    created_at: str


def match_refs(parsed: Any) -> list[MatchRef]:
    """Extract (game_id, created_at) pairs for ranked games from a match list."""
    history = dig(parsed, "data", "game_history", default=[]) or []
    refs = []
    for game in history:
        if not isinstance(game, dict):
            continue
        if game.get("game_type") not in RANKED_QUEUES:
            continue
        game_id, created_at = game.get("id"), game.get("created_at")
        if isinstance(game_id, str) and isinstance(created_at, str):
            refs.append(MatchRef(game_id, created_at))
    return refs


def _players(team: dict) -> list[dict]:
    out = []
    for p in team.get("participants") or []:
        if not isinstance(p, dict):
            continue
        summoner = p.get("summoner") if isinstance(p.get("summoner"), dict) else {}
        out.append(
            {
                "champion": p.get("champion_name"),
                "position": p.get("position"),
                "puuid": summoner.get("puuid"),
                "game_name": summoner.get("game_name"),
                "tagline": summoner.get("tagline"),
            }
        )
    return out


def summoners_in(parsed: Any) -> list[Summoner]:
    """Every identifiable player in a game detail — the snowball frontier."""
    found = []
    for team in dig(parsed, "data", "game_detail", "teams", default=[]) or []:
        if not isinstance(team, dict):
            continue
        for p in _players(team):
            if p["game_name"] and p["tagline"]:
                found.append(Summoner(p["game_name"], p["tagline"], p["puuid"]))
    return found


def game_row(parsed: Any, region: str) -> dict | None:
    """Build a match row, or None if the game is unusable.

    Rejects: non-ranked queues, remakes, malformed drafts, missing team split.
    """
    detail = dig(parsed, "data", "game_detail")
    if not isinstance(detail, dict):
        return None

    queue = detail.get("game_type")
    if queue not in RANKED_QUEUES:
        return None

    duration = detail.get("game_length_second")
    if not isinstance(duration, int) or not MIN_DURATION_S <= duration <= MAX_DURATION_S:
        return None

    teams = detail.get("teams")
    if not isinstance(teams, list) or len(teams) != 2:
        return None

    by_side: dict[str, dict] = {}
    for team in teams:
        if isinstance(team, dict) and team.get("key") in ("BLUE", "RED"):
            by_side[team["key"]] = team
    if len(by_side) != 2:
        return None

    blue, red = by_side["BLUE"], by_side["RED"]
    blue_picks, red_picks = _players(blue), _players(red)
    if len(blue_picks) != 5 or len(red_picks) != 5:
        return None
    if any(p["champion"] is None for p in blue_picks + red_picks):
        return None

    def bans(team: dict) -> list[str]:
        raw = team.get("banned_champions_names") or []
        return [b for b in raw if isinstance(b, str)]

    return {
        "game_id": detail.get("id"),
        "region": region,
        "created_at": detail.get("created_at"),
        "duration_s": duration,
        "queue": queue,
        "tier": dig(detail, "average_tier_info", "tier"),
        "blue_win": dig(blue, "game_stat", "is_win"),
        "blue_bans": bans(blue),
        "red_bans": bans(red),
        "blue_picks": blue_picks,
        "red_picks": red_picks,
    }
