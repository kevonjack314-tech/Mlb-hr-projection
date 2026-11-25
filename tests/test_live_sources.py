"""Tests: the ESPN live adapters for football and basketball.

The parsing is tested against captured payload SHAPES rather than the network,
so the suite stays offline and deterministic. One network-gated test actually
hits ESPN, and skips cleanly when there is no egress.
"""
import pandas as pd
import pytest

from src import espn
from src.espn import _num, parse_stat_categories


# --------------------------------------------------------------------------- #
# Payload parsing
# --------------------------------------------------------------------------- #
def test_num_survives_espn_string_formats():
    """ESPN ships numbers as strings, with commas and '--' for empty."""
    assert _num("3,095") == 3095.0
    assert _num("4.6") == pytest.approx(4.6)
    assert _num("--") is None and _num("") is None and _num(None) is None
    assert _num("--", 0.0) == 0.0
    assert _num(float("nan")) is None


def test_stat_categories_are_namespaced_by_category():
    """`gamesPlayed` appears in every category. Reading the wrong one silently
    rescales every per-game rate, so the keys carry their category."""
    payload = {"categories": [
        {"name": "rushing", "names": ["gamesPlayed", "rushingTouchdowns"],
         "statistics": [{"stats": ["17", "2"]}]},
        {"name": "receiving", "names": ["gamesPlayed", "receivingTargets"],
         "statistics": [{"stats": ["10", "0"]}]},
    ]}
    flat = parse_stat_categories(payload)
    assert flat["rushing.gamesPlayed"] == "17"
    assert flat["receiving.gamesPlayed"] == "10"
    assert parse_stat_categories(None) == {}
    assert parse_stat_categories({}) == {}


# --------------------------------------------------------------------------- #
# Scoreboard -> implied team totals
# --------------------------------------------------------------------------- #
def _fake_scoreboard(spread=-10.0, total=50.5, status="STATUS_SCHEDULED"):
    return {"events": [{
        "id": "1", "date": "2026-11-29T18:00Z",
        "weather": {"temperature": 53, "displayValue": "Rain"},
        "competitions": [{
            "status": {"type": {"name": status}},
            "venue": {"fullName": "Paycor Stadium", "indoor": False},
            "odds": [{"overUnder": total, "spread": spread}],
            "competitors": [
                {"homeAway": "home", "team": {"abbreviation": "CIN", "id": "4",
                                              "displayName": "Cincinnati Bengals"}},
                {"homeAway": "away", "team": {"abbreviation": "NO", "id": "18",
                                              "displayName": "New Orleans Saints"}},
            ]}]}]}


def test_the_market_total_splits_into_implied_team_totals(monkeypatch):
    """CIN -10 on a 50.5 total is 30.25 / 20.25 — the favourite gets the points,
    and the two halves still add back to the posted total."""
    monkeypatch.setattr(espn, "fetch_json", lambda *a, **k: _fake_scoreboard())
    espn.scoreboard.cache_clear()
    g = espn.scoreboard("nfl", "2026-11-29")[0]
    espn.scoreboard.cache_clear()
    assert g["home_implied"] == 30.25 and g["away_implied"] == 20.25
    assert g["home_implied"] + g["away_implied"] == pytest.approx(g["game_total"])
    assert g["game"] == "NO @ CIN" and g["temp_f"] == 53.0


def test_a_game_with_no_posted_line_has_no_implied_total(monkeypatch):
    sb = _fake_scoreboard()
    sb["events"][0]["competitions"][0]["odds"] = []
    monkeypatch.setattr(espn, "fetch_json", lambda *a, **k: sb)
    espn.scoreboard.cache_clear()
    g = espn.scoreboard("nfl", "2026-11-29")[0]
    espn.scoreboard.cache_clear()
    assert g["home_implied"] is None and g["game_total"] is None


def test_a_failed_fetch_returns_no_games_rather_than_raising(monkeypatch):
    monkeypatch.setattr(espn, "fetch_json", lambda *a, **k: None)
    espn.scoreboard.cache_clear()
    assert espn.scoreboard("nfl", "2026-11-29") == ()
    assert espn.scoreboard("curling", "2026-11-29") == ()
    espn.scoreboard.cache_clear()


# --------------------------------------------------------------------------- #
# Roster filtering
# --------------------------------------------------------------------------- #
def test_players_who_will_not_dress_are_dropped(monkeypatch):
    """Projecting a touchdown for someone on IR is worse than omitting him."""
    monkeypatch.setattr(espn, "fetch_json", lambda *a, **k: {"athletes": [
        {"position": "offense", "items": [
            {"id": "1", "displayName": "Healthy Back",
             "position": {"abbreviation": "RB"}},
            {"id": "2", "displayName": "Hurt Back",
             "position": {"abbreviation": "RB"},
             "injuries": [{"status": "Out"}]},
        ]},
        {"position": "injuredReserveOrOut", "items": [
            {"id": "3", "displayName": "On IR",
             "position": {"abbreviation": "WR"}}]},
        {"position": "practiceSquad", "items": [
            {"id": "4", "displayName": "PS Guy",
             "position": {"abbreviation": "WR"}}]},
    ]})
    espn.roster.cache_clear()
    names = [p["name"] for p in espn.roster("nfl", "4")]
    espn.roster.cache_clear()
    assert names == ["Healthy Back"]


# --------------------------------------------------------------------------- #
# NBA stat packing
# --------------------------------------------------------------------------- #
def test_made_attempted_field_is_unpacked():
    """ESPN packs makes and attempts into one string: '2.4-6.1'."""
    from src.nba.sources import _player_rates, _split_made_attempted
    assert _split_made_attempted("2.4-6.1") == (2.4, 6.1)
    assert _split_made_attempted(None) == (None, None)
    assert _split_made_attempted("garbage") == (None, None)

    rates = _player_rates({
        "averages.avgMinutes": "33.4",
        "averages.avgThreePointFieldGoalsMade-avgThreePointFieldGoalsAttempted": "2.6-7.0",
        "averages.threePointFieldGoalPct": "34.5",
        "averages.gamesPlayed": "18",
    })
    assert rates["minutes_season"] == 33.4
    # A percentage arrives as 34.5, not 0.345 — reading it raw would make every
    # shooter a 3450% shooter.
    assert rates["fg3_pct"] == pytest.approx(0.345, abs=1e-3)
    assert rates["fg3a_per36"] == pytest.approx(7.0 / 33.4 * 36, abs=0.05)


def test_a_player_with_no_minutes_is_not_a_three_point_bet():
    from src.nba.sources import _player_rates
    assert _player_rates({"averages.avgMinutes": "12.0"}) is None   # no attempts
    assert _player_rates({}) is None


# --------------------------------------------------------------------------- #
# Fallback contract
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mod", ["src.nfl.sources", "src.nba.sources"])
def test_a_dead_feed_falls_back_to_the_synthetic_slate(monkeypatch, mod):
    """Same guard baseball uses: degrade the board, never take the page down,
    and never let a modelled slate claim to be live."""
    import importlib
    m = importlib.import_module(mod)
    monkeypatch.setattr(m.espn, "scoreboard", lambda *a, **k: ())
    df, source, notes = m.get_slate("2026-11-29", prefer_live=True)
    assert df is not None and not df.empty
    assert source.startswith("DEMO")
    assert any("synthetic" in n.lower() for n in notes)


@pytest.mark.parametrize("mod", ["src.nfl.sources", "src.nba.sources"])
def test_an_exception_mid_build_still_degrades_cleanly(monkeypatch, mod):
    import importlib
    m = importlib.import_module(mod)

    def boom(*a, **k):
        raise RuntimeError("feed exploded")

    monkeypatch.setattr(m.espn, "scoreboard", boom)
    df, source, notes = m.get_slate("2026-11-29", prefer_live=True)
    assert source.startswith("DEMO") and not df.empty
    assert any("failed" in n.lower() for n in notes)


@pytest.mark.parametrize("mod", ["src.nfl.sources", "src.nba.sources"])
def test_prefer_live_false_never_touches_the_network(monkeypatch, mod):
    import importlib
    m = importlib.import_module(mod)

    def boom(*a, **k):
        raise AssertionError("network touched with prefer_live=False")

    monkeypatch.setattr(m.espn, "fetch_json", boom)
    df, source, _ = m.get_slate("2026-11-29", prefer_live=False)
    assert source.startswith("DEMO") and not df.empty


# --------------------------------------------------------------------------- #
# The real thing (skipped without egress)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sport,date_iso", [("nfl", "2026-11-29"),
                                            ("nba", "2026-11-25")])
def test_live_feed_end_to_end(sport, date_iso):
    from src.multisport import build_slate, prob_col
    if not espn.scoreboard(sport, date_iso):
        pytest.skip("no network egress to ESPN")
    board, source, _ = build_slate(sport, date_iso, prefer_live=True,
                                   with_notes=True)
    assert source.startswith("LIVE")
    assert len(board) > 50 and board["game"].nunique() >= 2
    p = pd.to_numeric(board[prob_col(sport)], errors="coerce")
    assert p.between(0, 0.97).all() and p.max() > 0.15
    assert board["player"].nunique() == len(board)
