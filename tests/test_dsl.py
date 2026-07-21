from pathlib import Path

import pytest

from ingest import records
from ingest.dsl import ParseError, dig, parse

FIXTURES = Path(__file__).parent / "fixtures"


def test_parses_nested_objects_and_lists():
    text = """class Root: a,b
class Inner: x

Root([Inner(1),Inner(2)],"hi")"""
    r = parse(text)
    assert r.value == {"a": [{"x": 1}, {"x": 2}], "b": "hi"}
    assert not r.warnings


@pytest.mark.parametrize(
    "literal,expected",
    [("true", True), ("false", False), ("null", None), ("-3", -3), ("1.5", 1.5), ("2e3", 2000.0)],
)
def test_scalar_literals(literal, expected):
    assert parse(f"class Root: v\n\nRoot({literal})").value == {"v": expected}


def test_strings_may_contain_delimiters():
    # Summoner names are user-controlled: commas, parens and quotes must not
    # break the parse.
    text = 'class Root: name,tag\n\nRoot("a,b)c[d\\"e","NA1")'
    assert parse(text).value == {"name": 'a,b)c[d"e', "tag": "NA1"}


def test_apostrophes_in_champion_names():
    text = 'class Root: c\n\nRoot(["Vel\'Koz","Cho\'Gath","Kai\'Sa"])'
    assert parse(text).value["c"] == ["Vel'Koz", "Cho'Gath", "Kai'Sa"]


def test_empty_list():
    assert parse("class Root: v\n\nRoot([])").value == {"v": []}


def test_missing_trailing_fields_are_tolerated():
    r = parse("class Root: a,b,c\n\nRoot(1)")
    assert r.value == {"a": 1}
    assert r.warnings


def test_extra_args_are_preserved_not_dropped():
    r = parse("class Root: a\n\nRoot(1,2,3)")
    assert r.value["a"] == 1
    assert r.value["__extra__"] == [2, 3]
    assert r.warnings


def test_unknown_class_is_kept_for_debugging():
    r = parse("class Root: a\n\nRoot(Mystery(7))")
    assert r.value["a"] == {"__class__": "Mystery", "__args__": [7]}
    assert r.warnings


def test_unterminated_input_raises():
    with pytest.raises(ParseError):
        parse('class Root: a\n\nRoot("abc')


def test_empty_body_raises():
    with pytest.raises(ParseError):
        parse("class Root: a\n")


def test_dig_returns_default_on_miss():
    assert dig({"a": {"b": 1}}, "a", "b") == 1
    assert dig({"a": {"b": 1}}, "a", "z", default="fallback") == "fallback"
    assert dig(None, "a", default=42) == 42


# -- against real captured responses ---------------------------------------


def test_game_detail_fixture_round_trips():
    r = parse((FIXTURES / "game_detail.txt").read_text())
    assert not r.warnings, r.warnings
    row = records.game_row(r.value, region="NA")
    assert row is not None
    assert row["duration_s"] == 1466
    assert row["queue"] == "SOLORANKED"
    assert row["tier"] == "MASTER"
    assert row["blue_win"] is True
    assert len(row["blue_picks"]) == 5
    assert len(row["red_picks"]) == 5
    assert {p["position"] for p in row["blue_picks"]} == {
        "TOP", "JUNGLE", "MID", "ADC", "SUPPORT"
    }
    assert "Malphite" in row["blue_bans"]


def test_summoners_extracted_for_snowball():
    r = parse((FIXTURES / "game_detail.txt").read_text())
    people = records.summoners_in(r.value)
    assert len(people) == 10
    assert all(p.game_name and p.tagline for p in people)


def test_match_list_fixture_yields_ranked_refs():
    r = parse((FIXTURES / "match_list.txt").read_text())
    assert not r.warnings, r.warnings
    refs = records.match_refs(r.value)
    assert refs
    assert all(ref.game_id and ref.created_at for ref in refs)


# -- filtering rules -------------------------------------------------------


def _detail(**overrides):
    base = {
        "id": "g1",
        "created_at": "2026-07-21T09:33:36+09:00",
        "game_type": "SOLORANKED",
        "game_length_second": 1800,
        "average_tier_info": {"tier": "GOLD"},
        "teams": [
            {
                "key": side,
                "banned_champions_names": ["Malphite"],
                "game_stat": {"is_win": side == "BLUE"},
                "participants": [
                    {
                        "champion_name": f"C{i}",
                        "position": pos,
                        "summoner": {"game_name": f"p{i}", "tagline": "NA1", "puuid": f"u{i}"},
                    }
                    for i, pos in enumerate(["TOP", "JUNGLE", "MID", "ADC", "SUPPORT"])
                ],
            }
            for side in ("BLUE", "RED")
        ],
    }
    base.update(overrides)
    return {"data": {"game_detail": base}}


def test_accepts_a_well_formed_ranked_game():
    assert records.game_row(_detail(), "NA") is not None


@pytest.mark.parametrize("duration", [0, 66, 299, 5401])
def test_rejects_remakes_and_impossible_durations(duration):
    assert records.game_row(_detail(game_length_second=duration), "NA") is None


@pytest.mark.parametrize("queue", ["ARAM", "NORMAL", "ARENA", "URF"])
def test_rejects_non_ranked_queues(queue):
    assert records.game_row(_detail(game_type=queue), "NA") is None


def test_accepts_flex():
    assert records.game_row(_detail(game_type="FLEXRANKED"), "NA") is not None


def test_rejects_incomplete_team():
    d = _detail()
    d["data"]["game_detail"]["teams"][0]["participants"].pop()
    assert records.game_row(d, "NA") is None


def test_rejects_missing_champion():
    d = _detail()
    d["data"]["game_detail"]["teams"][0]["participants"][0]["champion_name"] = None
    assert records.game_row(d, "NA") is None


def test_rejects_garbage():
    assert records.game_row({"data": {}}, "NA") is None
    assert records.game_row(None, "NA") is None
