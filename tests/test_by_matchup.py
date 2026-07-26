"""Top 2 HR picks per team, per game."""

import datetime as dt

from src.demo import build_demo_slate
from src.model import score_slate


def test_two_picks_per_team_every_game():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    picks = (df.sort_values("hr_prob_game", ascending=False)
             .groupby(["game", "team"]).head(2))
    # Every team in every game has at most 2 and (for full rosters) exactly 2.
    counts = picks.groupby(["game", "team"]).size()
    assert (counts <= 2).all()
    assert (counts == 2).mean() > 0.9        # nearly all teams have a full 2

    # Coverage: every (game, team) present in the slate appears in the board.
    all_gt = set(map(tuple, df[["game", "team"]].drop_duplicates().values))
    pick_gt = set(map(tuple, picks[["game", "team"]].drop_duplicates().values))
    assert all_gt == pick_gt

    # Each team's picks really are its two best by HR probability. Compare the
    # PROBABILITIES, not player identity: exact ties happen on a real board and
    # either tied bat is an equally correct pick.
    for (game, team), grp in df.groupby(["game", "team"]):
        best2 = sorted(grp["hr_prob_game"].nlargest(2).tolist(), reverse=True)
        got = sorted(picks[(picks["game"] == game) & (picks["team"] == team)]
                     ["hr_prob_game"].tolist(), reverse=True)
        assert got == best2


def test_both_teams_present_per_game():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    for game, grp in df.groupby("game"):
        assert grp["team"].nunique() == 2      # away + home
