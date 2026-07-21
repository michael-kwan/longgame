"""Parser for OP.GG's compact class-DSL response format.

Responses look like:

    class LolGetSummonerGameDetail: data
    class Data: game_detail
    class GameDetail: id,game_length_second,teams
    class Team: key,banned_champions_names

    LolGetSummonerGameDetail(Data(GameDetail("abc=",1466,[Team("BLUE",["Malphite"])])))

Header lines declare positional field names per class; the body is a single
expression. We zip the two together into plain nested dicts.

This is the most likely silent-breakage point in the pipeline (§3 of DESIGN.md),
so the parser is deliberately tolerant: unknown classes and arity mismatches are
recorded as warnings rather than raised, and callers decide what to do.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_HEADER = re.compile(r"^class\s+(\w+)\s*:\s*(.*)$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_IDENT = re.compile(r"[A-Za-z_]\w*")

_LITERALS = {"true": True, "false": False, "null": None, "none": None, "None": None}


class ParseError(ValueError):
    """The response was not parseable as OP.GG DSL."""


@dataclass
class ParseResult:
    value: Any
    schema: dict[str, list[str]]
    warnings: list[str] = field(default_factory=list)


def parse(text: str) -> ParseResult:
    """Parse a full OP.GG DSL response into nested dicts/lists."""
    schema, body = _split_header(text)
    if not body.strip():
        raise ParseError("no body expression found")
    p = _Parser(body, schema)
    value = p.parse_value()
    p.skip_ws()
    if not p.at_end():
        p.warn(f"trailing content at offset {p.i}: {body[p.i:p.i + 40]!r}")
    return ParseResult(value=value, schema=schema, warnings=p.warnings)


def _split_header(text: str) -> tuple[dict[str, list[str]], str]:
    schema: dict[str, list[str]] = {}
    lines = text.splitlines()
    i = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _HEADER.match(stripped)
        if not m:
            break
        name, raw_fields = m.group(1), m.group(2)
        schema[name] = [f.strip() for f in raw_fields.split(",") if f.strip()]
    else:
        i = len(lines)
    return schema, "\n".join(lines[i:])


class _Parser:
    def __init__(self, s: str, schema: dict[str, list[str]]):
        self.s = s
        self.i = 0
        self.n = len(s)
        self.schema = schema
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        if len(self.warnings) < 50:
            self.warnings.append(msg)

    def at_end(self) -> bool:
        return self.i >= self.n

    def skip_ws(self) -> None:
        while self.i < self.n and self.s[self.i] in " \t\r\n":
            self.i += 1

    def parse_value(self) -> Any:
        self.skip_ws()
        if self.at_end():
            raise ParseError("unexpected end of input")
        c = self.s[self.i]
        if c == '"':
            return self.parse_string()
        if c == "[":
            return self.parse_list()
        if c == "-" or c.isdigit():
            return self.parse_number()
        m = _IDENT.match(self.s, self.i)
        if m:
            name = m.group(0)
            self.i = m.end()
            self.skip_ws()
            if self.i < self.n and self.s[self.i] == "(":
                return self.parse_object(name)
            if name in _LITERALS:
                return _LITERALS[name]
            self.warn(f"bare identifier {name!r} at offset {m.start()}")
            return name
        raise ParseError(f"unexpected character {c!r} at offset {self.i}")

    def parse_string(self) -> str:
        start = self.i
        self.i += 1  # opening quote
        while self.i < self.n:
            c = self.s[self.i]
            if c == "\\":
                self.i += 2
                continue
            if c == '"':
                self.i += 1
                raw = self.s[start:self.i]
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    # Lenient fallback: strip quotes, unescape the common pairs.
                    return raw[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            self.i += 1
        raise ParseError(f"unterminated string starting at offset {start}")

    def parse_number(self) -> int | float:
        m = _NUMBER.match(self.s, self.i)
        if not m:
            raise ParseError(f"bad number at offset {self.i}")
        self.i = m.end()
        raw = m.group(0)
        if any(ch in raw for ch in ".eE"):
            return float(raw)
        return int(raw)

    def _parse_items(self, close: str) -> list[Any]:
        items: list[Any] = []
        self.i += 1  # opening bracket
        self.skip_ws()
        if self.i < self.n and self.s[self.i] == close:
            self.i += 1
            return items
        while True:
            items.append(self.parse_value())
            self.skip_ws()
            if self.at_end():
                raise ParseError(f"unterminated group, expected {close!r}")
            c = self.s[self.i]
            if c == ",":
                self.i += 1
                continue
            if c == close:
                self.i += 1
                return items
            raise ParseError(f"expected ',' or {close!r} at offset {self.i}, got {c!r}")

    def parse_list(self) -> list[Any]:
        return self._parse_items("]")

    def parse_object(self, name: str) -> Any:
        args = self._parse_items(")")
        fields = self.schema.get(name)
        if fields is None:
            self.warn(f"no header for class {name!r}")
            return {"__class__": name, "__args__": args}
        if len(args) > len(fields):
            self.warn(
                f"class {name!r}: {len(args)} args but {len(fields)} declared fields"
            )
            obj = dict(zip(fields, args))
            obj["__extra__"] = args[len(fields):]
            return obj
        if len(args) < len(fields):
            self.warn(
                f"class {name!r}: {len(args)} args but {len(fields)} declared fields"
            )
        return dict(zip(fields, args))


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Walk a nested dict by key path, returning `default` on any miss."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
