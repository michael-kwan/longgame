"""Thin client for the OP.GG MCP server (streamable HTTP transport).

One MCP session per thread, reused across calls. Retries with backoff on
throttling and transport errors. See DESIGN.md §3 for why this is the data
source and what the politeness budget is.
"""

from __future__ import annotations

import json
import random
import threading
import time

import requests

ENDPOINT = "https://mcp-api.op.gg/mcp"
PROTOCOL_VERSION = "2025-06-18"
_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


class OpggError(RuntimeError):
    """The server returned an error for a tool call."""


class OpggClient:
    def __init__(self, timeout: float = 60.0, max_retries: int = 4):
        self.timeout = timeout
        self.max_retries = max_retries
        self._http = requests.Session()
        self._local = threading.local()

    # -- transport ---------------------------------------------------------

    def _post(self, body: dict, session_id: str | None = None) -> requests.Response:
        headers = dict(_HEADERS)
        if session_id:
            headers["mcp-session-id"] = session_id
        return self._http.post(
            ENDPOINT, data=json.dumps(body), headers=headers, timeout=self.timeout
        )

    def _new_session(self) -> str:
        resp = self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "longgame", "version": "0.1"},
                },
            }
        )
        resp.raise_for_status()
        session_id = resp.headers.get("mcp-session-id")
        if not session_id:
            raise OpggError("server did not return an mcp-session-id")
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, session_id)
        return session_id

    def _session(self) -> str:
        session_id = getattr(self._local, "session_id", None)
        if session_id is None:
            session_id = self._new_session()
            self._local.session_id = session_id
        return session_id

    def _drop_session(self) -> None:
        self._local.session_id = None

    @staticmethod
    def _decode(resp: requests.Response) -> dict:
        text = resp.text
        if "text/event-stream" in resp.headers.get("content-type", ""):
            for line in text.splitlines():
                if line.startswith("data:"):
                    return json.loads(line[5:].strip())
            raise OpggError("event-stream response contained no data frame")
        return json.loads(text)

    # -- tool calls --------------------------------------------------------

    def call(self, tool: str, arguments: dict) -> str:
        """Invoke a tool, returning its text content. Retries transient failures."""
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            if attempt:
                # Exponential backoff with jitter; be gentle on a free endpoint.
                time.sleep(min(30.0, 2.0**attempt) * (0.5 + random.random()))
            try:
                resp = self._post(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": tool, "arguments": arguments},
                    },
                    self._session(),
                )
                if resp.status_code in (400, 404, 409):
                    # Stale or unrecognised session: rebuild and retry.
                    self._drop_session()
                    last_error = OpggError(f"session rejected ({resp.status_code})")
                    continue
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = OpggError(f"http {resp.status_code}")
                    continue
                resp.raise_for_status()
                payload = self._decode(resp)
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last_error = exc
                self._drop_session()
                continue

            if "error" in payload:
                raise OpggError(f"{tool}: {payload['error']}")
            result = payload.get("result", {})
            content = result.get("content") or []
            text = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            if result.get("isError"):
                raise OpggError(f"{tool}: {text[:300]}")
            return text

        raise OpggError(f"{tool}: giving up after {self.max_retries} attempts: {last_error}")


# -- field selections ------------------------------------------------------
# `desired_output_fields` is a CLOSED SET on the server side; these are the
# exact paths verified against the live API. Requesting an unknown path
# silently drops the field, so changes here need re-verification.

CHAMPION_LIST_FIELDS = ["data.champions[].key", "data.champions[].name"]

LEADERBOARD_FIELDS = [
    "leaderboard[].summoner.game_name",
    "leaderboard[].summoner.tagline",
]

MATCH_LIST_FIELDS = [
    "data.game_history[].id",
    "data.game_history[].created_at",
    "data.game_history[].game_type",
    "data.game_history[].game_length_second",
]

GAME_DETAIL_FIELDS = [
    "data.game_detail.id",
    "data.game_detail.created_at",
    "data.game_detail.game_type",
    "data.game_detail.game_length_second",
    "data.game_detail.average_tier_info.tier",
    "data.game_detail.teams[].key",
    "data.game_detail.teams[].banned_champions_names",
    "data.game_detail.teams[].game_stat.is_win",
    "data.game_detail.teams[].participants[].champion_name",
    "data.game_detail.teams[].participants[].position",
    "data.game_detail.teams[].participants[].summoner.game_name",
    "data.game_detail.teams[].participants[].summoner.tagline",
    "data.game_detail.teams[].participants[].summoner.puuid",
]
