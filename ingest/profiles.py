"""Fetch per-player profiles for everyone in the crawled match set.

Adds the features a match record does not carry: how many ranked games the
player has actually played, their tier and LP. `average_tier_info` on a match is
a single lobby-level number; this gives all ten players individually, which is
what any "experience" or "lobby imbalance" feature needs.

Ordered by how often a player appears in the dataset, so a partial run still
covers the games most likely to have full coverage.

    python -m ingest.profiles --region NA --limit 4000 --workers 10
"""

from __future__ import annotations

import argparse
import collections
import signal
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .dsl import ParseError, dig, parse
from .opgg import OpggClient, OpggError

ROOT = Path(__file__).resolve().parents[1]

PROFILE_FIELDS = [
    "data.summoner.league_stats[].game_type",
    "data.summoner.league_stats[].win",
    "data.summoner.league_stats[].lose",
    "data.summoner.league_stats[].tier_info.tier",
    "data.summoner.league_stats[].tier_info.division",
    "data.summoner.league_stats[].tier_info.lp",
    "data.summoner.level",
]

_stop = False


def _sigint(signum, frame):
    global _stop
    if _stop:
        sys.exit(130)
    _stop = True
    print("\n[profiles] stopping after this batch", flush=True)


def player_counts(data_dir: Path) -> list[tuple[str, str, str, int]]:
    """(puuid, game_name, tagline, appearances) ordered by appearances."""
    import pyarrow.parquet as pq

    seen_games: set[str] = set()
    counts: collections.Counter = collections.Counter()
    ids: dict[str, tuple[str, str]] = {}
    for shard in sorted((data_dir / "matches").glob("*.parquet")):
        for row in pq.read_table(shard).to_pylist():
            if row["game_id"] in seen_games:
                continue
            seen_games.add(row["game_id"])
            for p in row["blue_picks"] + row["red_picks"]:
                if p["puuid"] and p["game_name"] and p["tagline"]:
                    counts[p["puuid"]] += 1
                    ids[p["puuid"]] = (p["game_name"], p["tagline"])
    return [(pu, *ids[pu], c) for pu, c in counts.most_common()]


def parse_profile(parsed) -> dict:
    stats = dig(parsed, "data", "summoner", "league_stats", default=[]) or []
    out = {"solo_games": 0, "flex_games": 0, "tier": None, "division": None, "lp": None,
           "level": dig(parsed, "data", "summoner", "level")}
    for s in stats:
        if not isinstance(s, dict):
            continue
        wins, losses = s.get("win") or 0, s.get("lose") or 0
        games = wins + losses
        if s.get("game_type") == "SOLORANKED":
            out["solo_games"] = games
            if dig(s, "tier_info", "tier"):
                out["tier"] = dig(s, "tier_info", "tier")
                out["division"] = dig(s, "tier_info", "division")
                out["lp"] = dig(s, "tier_info", "lp")
        elif s.get("game_type") == "FLEXRANKED":
            out["flex_games"] = games
            if out["tier"] is None and dig(s, "tier_info", "tier"):
                out["tier"] = dig(s, "tier_info", "tier")
                out["division"] = dig(s, "tier_info", "division")
                out["lp"] = dig(s, "tier_info", "lp")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="NA")
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, _sigint)

    db = sqlite3.connect(args.data_dir / "profiles.db")
    db.execute(
        """CREATE TABLE IF NOT EXISTS profiles (
               puuid TEXT PRIMARY KEY, game_name TEXT, tagline TEXT,
               solo_games INTEGER, flex_games INTEGER, tier TEXT,
               division INTEGER, lp INTEGER, level INTEGER, ok INTEGER)"""
    )
    db.commit()

    have = {r[0] for r in db.execute("SELECT puuid FROM profiles").fetchall()}
    players = [p for p in player_counts(args.data_dir) if p[0] not in have][: args.limit]
    print(f"[profiles] {len(have)} already stored, fetching {len(players)}", flush=True)

    client = OpggClient()

    def fetch(rec):
        puuid, name, tag, _ = rec
        try:
            raw = client.call(
                "lol_get_summoner_profile",
                {"game_name": name, "tag_line": tag, "region": args.region,
                 "desired_output_fields": PROFILE_FIELDS},
            )
            return puuid, name, tag, parse_profile(parse(raw).value), 1
        except (OpggError, ParseError):
            return puuid, name, tag, {}, 0

    started, done = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        for i in range(0, len(players), 200):
            if _stop:
                break
            batch = players[i:i + 200]
            rows = []
            for puuid, name, tag, info, ok in pool.map(fetch, batch):
                rows.append((puuid, name, tag, info.get("solo_games"), info.get("flex_games"),
                             info.get("tier"), info.get("division"), info.get("lp"),
                             info.get("level"), ok))
            db.executemany(
                "INSERT OR REPLACE INTO profiles VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )
            db.commit()
            done += len(batch)
            rate = done / max(time.time() - started, 1e-6)
            print(f"[profiles] {done}/{len(players)}  {rate:.1f} profiles/s", flush=True)

    total = db.execute("SELECT COUNT(*) FROM profiles WHERE ok=1").fetchone()[0]
    print(f"[profiles] done, {total} profiles stored", flush=True)


if __name__ == "__main__":
    main()
