"""Snowball crawler: seed champion leaderboards, then expand through co-players.

Every game detail yields 10 new summoner identities, so discovery is
self-sustaining (DESIGN.md §3). Crawl state lives in SQLite so runs are
resumable; raw responses are archived so a schema change never forces a
re-crawl.

    python -m ingest.crawl --region NA --target 150000 --workers 10
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import signal
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from . import records
from .dsl import ParseError, dig, parse
from .opgg import (
    CHAMPION_LIST_FIELDS,
    GAME_DETAIL_FIELDS,
    LEADERBOARD_FIELDS,
    MATCH_LIST_FIELDS,
    OpggClient,
    OpggError,
)

SHARD_ROWS = 2000

_stop = False


def _slugs(key: str) -> list[str]:
    """Candidate leaderboard identifiers for an internal champion key."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", key).upper()
    flat = key.upper()
    return [snake] if snake == flat else [snake, flat]


def _handle_sigint(signum, frame):
    global _stop
    if _stop:
        sys.exit(130)
    _stop = True
    print("\n[crawl] stopping after this batch — press Ctrl-C again to force", flush=True)


class Store:
    """SQLite crawl state + Parquet output. Main thread only."""

    def __init__(self, root: Path, region: str):
        self.root = root
        self.region = region
        self.matches_dir = root / "matches"
        self.matches_dir.mkdir(parents=True, exist_ok=True)
        self.raw_path = root / f"raw-{region}.jsonl.gz"
        self.db = sqlite3.connect(root / "crawl.db")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS summoners (
                key TEXT PRIMARY KEY, game_name TEXT, tagline TEXT,
                region TEXT, done INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_pending ON summoners(done);
            CREATE TABLE IF NOT EXISTS matches (game_id TEXT PRIMARY KEY);
            """
        )
        self.db.commit()
        self._buffer: list[dict] = []

    # -- frontier ----------------------------------------------------------

    def add_summoners(self, people) -> int:
        rows = [
            (f"{self.region}/{p.game_name}#{p.tagline}", p.game_name, p.tagline, self.region)
            for p in people
        ]
        cur = self.db.executemany(
            "INSERT OR IGNORE INTO summoners(key,game_name,tagline,region) VALUES (?,?,?,?)",
            rows,
        )
        self.db.commit()
        return cur.rowcount

    def pending(self, limit: int) -> list[tuple[str, str, str]]:
        return self.db.execute(
            "SELECT key,game_name,tagline FROM summoners WHERE done=0 ORDER BY rowid LIMIT ?",
            (limit,),
        ).fetchall()

    def mark_done(self, keys) -> None:
        self.db.executemany("UPDATE summoners SET done=1 WHERE key=?", [(k,) for k in keys])
        self.db.commit()

    # -- matches -----------------------------------------------------------

    def unseen(self, refs) -> list[records.MatchRef]:
        out = []
        for ref in refs:
            hit = self.db.execute(
                "SELECT 1 FROM matches WHERE game_id=?", (ref.game_id,)
            ).fetchone()
            if not hit:
                out.append(ref)
        return out

    def record_match(self, game_id: str, row: dict | None, raw: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO matches(game_id) VALUES (?)", (game_id,))
        with gzip.open(self.raw_path, "at", encoding="utf-8") as fh:
            fh.write(json.dumps({"game_id": game_id, "raw": raw}) + "\n")
        if row:
            self._buffer.append(row)
            if len(self._buffer) >= SHARD_ROWS:
                self.flush()

    def flush(self) -> None:
        if not self._buffer:
            self.db.commit()
            return
        table = pa.Table.from_pylist(self._buffer, schema=records.SCHEMA)
        shard = self.matches_dir / f"part-{int(time.time() * 1000)}.parquet"
        pq.write_table(table, shard, compression="zstd")
        print(f"[crawl] wrote {len(self._buffer)} rows -> {shard.name}", flush=True)
        self._buffer.clear()
        self.db.commit()

    def counts(self) -> tuple[int, int, int]:
        seen = self.db.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
        done = self.db.execute("SELECT COUNT(*) FROM summoners WHERE done=1").fetchone()[0]
        todo = self.db.execute("SELECT COUNT(*) FROM summoners WHERE done=0").fetchone()[0]
        return seen, done, todo

    def stored_rows(self) -> int:
        total = len(self._buffer)
        for shard in self.matches_dir.glob("*.parquet"):
            total += pq.ParquetFile(shard).metadata.num_rows
        return total


def seed(client: OpggClient, store: Store, region: str, n_champions: int) -> None:
    """Prime the frontier from per-champion master+ leaderboards."""
    if store.pending(1):
        print("[crawl] frontier already populated, skipping seed", flush=True)
        return
    text = client.call("lol_list_champions", {"desired_output_fields": CHAMPION_LIST_FIELDS})
    champions = [
        c["key"]
        for c in (dig(parse(text).value, "data", "champions", default=[]) or [])
        if isinstance(c, dict) and isinstance(c.get("key"), str)
    ]
    if not champions:
        raise OpggError("could not list champions to seed from")
    picks = champions[:n_champions]
    print(f"[crawl] seeding from {len(picks)} champion leaderboards", flush=True)

    def fetch(key: str):
        # The leaderboard wants UPPER_SNAKE_CASE, but the mapping from the
        # internal key is not perfectly regular (KogMaw -> KOGMAW, not KOG_MAW),
        # so try the snake form then the flat form. Seeds are fungible; a
        # champion that resolves to neither is simply skipped.
        for slug in _slugs(key):
            try:
                raw = client.call(
                    "lol_list_champion_leaderboard",
                    {
                        "region": region,
                        "champion": slug,
                        "desired_output_fields": LEADERBOARD_FIELDS,
                    },
                )
            except (OpggError, ParseError):
                continue
            try:
                entries = dig(parse(raw).value, "leaderboard", default=[]) or []
            except ParseError:
                continue
            return [
                records.Summoner(s["game_name"], s["tagline"])
                for e in entries
                if isinstance(e, dict) and isinstance(s := e.get("summoner"), dict)
                and s.get("game_name") and s.get("tagline")
            ]
        return []

    with ThreadPoolExecutor(6) as pool:
        for people in pool.map(fetch, picks):
            store.add_summoners(people)
    print(f"[crawl] frontier: {store.counts()[2]} summoners", flush=True)


def run(region: str, target: int, workers: int, data_dir: Path, seed_champions: int) -> None:
    client = OpggClient()
    store = Store(data_dir, region)
    seed(client, store, region, seed_champions)

    started = time.time()
    start_rows = store.stored_rows()

    def list_matches(person):
        key, game_name, tagline = person
        try:
            raw = client.call(
                "lol_list_summoner_matches",
                {
                    "game_name": game_name,
                    "tag_line": tagline,
                    "region": region,
                    "limit": 20,
                    "desired_output_fields": MATCH_LIST_FIELDS,
                },
            )
            return key, records.match_refs(parse(raw).value)
        except (OpggError, ParseError):
            return key, []

    def fetch_detail(ref):
        try:
            raw = client.call(
                "lol_get_summoner_game_detail",
                {
                    "region": region,
                    "game_id": ref.game_id,
                    "created_at": ref.created_at,
                    "desired_output_fields": GAME_DETAIL_FIELDS,
                },
            )
            return ref, raw, parse(raw).value
        except (OpggError, ParseError):
            return ref, None, None

    with ThreadPoolExecutor(workers) as pool:
        while not _stop:
            rows_now = store.stored_rows()
            if rows_now >= target:
                print(f"[crawl] reached target of {target} matches", flush=True)
                break

            people = store.pending(workers * 3)
            if not people:
                print("[crawl] frontier exhausted", flush=True)
                break

            refs: list[records.MatchRef] = []
            seen_batch = set()
            for key, found in pool.map(list_matches, people):
                for ref in found:
                    if ref.game_id not in seen_batch:
                        seen_batch.add(ref.game_id)
                        refs.append(ref)
            store.mark_done([p[0] for p in people])

            fresh = store.unseen(refs)
            kept = 0
            discovered: list[records.Summoner] = []
            for ref, raw, parsed in pool.map(fetch_detail, fresh):
                if raw is None:
                    continue
                row = records.game_row(parsed, region)
                store.record_match(ref.game_id, row, raw)
                kept += row is not None
                # This is the snowball: every game detail hands back 10 players.
                discovered.extend(records.summoners_in(parsed))
            added = store.add_summoners(discovered)
            store.flush()

            elapsed = max(time.time() - started, 1e-6)
            total = store.stored_rows()
            rate = (total - start_rows) / elapsed
            seen, done, todo = store.counts()
            print(
                f"[crawl] +{kept}/{len(fresh)} kept | rows={total} seen={seen} "
                f"crawled={done} frontier={todo} (+{added}) | {rate:.2f} matches/s",
                flush=True,
            )

    store.flush()
    seen, done, todo = store.counts()
    print(
        f"[crawl] done. rows={store.stored_rows()} game_ids_seen={seen} "
        f"summoners_crawled={done} frontier={todo}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Snowball-crawl ranked LoL drafts from OP.GG")
    ap.add_argument("--region", default="NA")
    ap.add_argument("--target", type=int, default=150_000, help="stop at this many kept matches")
    ap.add_argument("--workers", type=int, default=10, help="concurrent requests (be polite)")
    ap.add_argument("--seed-champions", type=int, default=60)
    ap.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_sigint)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    run(args.region, args.target, args.workers, args.data_dir, args.seed_champions)


if __name__ == "__main__":
    main()
