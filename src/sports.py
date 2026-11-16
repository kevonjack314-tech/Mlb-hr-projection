"""Sport registry — what the app needs to know to run a sport.

The MLB app answers one question: which bats are most likely to do the rare,
high-odds thing tonight? Nothing about the pipeline around that question is
baseball-specific — the slate, the per-player probability, the calibration
curve fit on graded outcomes, the parlay construction and the grading loop all
work the same way whatever the sport is. What changes is the EVENT being
predicted and the data that predicts it.

So each sport registers a spec here, and the shared machinery reads the spec
instead of hard-coding baseball.

The event each sport targets is chosen to be the same SHAPE of bet — a rare,
repeatable, skill-driven binary outcome priced at long odds, which is what
makes a parlay model worth having:

    MLB  home run            ~4% per plate appearance
    NFL  anytime touchdown   ~2.3 offensive TDs per team per game to share out
    NBA  3+ made threes      a volume-and-accuracy threshold, not a lottery
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


@dataclass(frozen=True)
class SportSpec:
    key: str                  # "mlb" | "nfl" | "nba"
    label: str                # shown in the sidebar picker
    emoji: str
    event: str                # "home run"
    event_short: str          # "HR"
    prop_label: str           # the bet as a book would write it
    odds_api_key: str         # The Odds API sport key
    # Where this sport's graded record lives. MLB's files predate the registry
    # and stay at the top level so the existing 63-day record keeps working.
    data_subdir: str = ""
    # Roughly how often the event happens, for sanity-checking a model that has
    # wandered off: a slate whose median player is at 40% to homer is broken.
    base_rate: float = 0.0
    player_noun: str = "player"
    lineup_noun: str = "lineup"
    notes: str = ""
    tier_bands: dict = field(default_factory=dict)

    @property
    def data_dir(self) -> str:
        return os.path.join(_DATA_DIR, self.data_subdir) if self.data_subdir else _DATA_DIR

    def path(self, filename: str) -> str:
        return os.path.join(self.data_dir, filename)


MLB = SportSpec(
    key="mlb", label="MLB Home Runs", emoji="⚾",
    event="home run", event_short="HR", prop_label="1+ Home Run",
    odds_api_key="baseball_mlb",
    base_rate=0.12, player_noun="hitter", lineup_noun="lineup",
    notes="Rare power event off a ~4% per-PA rate, ~4.1 PAs a game.",
    tier_bands={"Star": (200, 450), "Mid": (500, 700), "Under": (700, 2500)},
)

NFL = SportSpec(
    key="nfl", label="NFL Anytime TD", emoji="🏈",
    event="touchdown", event_short="TD", prop_label="Anytime Touchdown",
    odds_api_key="americanfootball_nfl", data_subdir="nfl",
    base_rate=0.22, player_noun="skill player", lineup_noun="depth chart",
    notes=("Football's home run: about 2.3 offensive touchdowns per team per "
           "game, shared out by goal-line and red-zone role."),
    tier_bands={"Star": (-200, 150), "Mid": (150, 400), "Under": (400, 2000)},
)

NBA = SportSpec(
    key="nba", label="NBA 3+ Threes", emoji="🏀",
    event="3+ made threes", event_short="3PM", prop_label="3+ Made Threes",
    odds_api_key="basketball_nba", data_subdir="nba",
    base_rate=0.35, player_noun="shooter", lineup_noun="rotation",
    notes=("A volume-and-accuracy threshold rather than a lottery: minutes x "
           "three-point attempt rate x accuracy, against pace and perimeter "
           "defense."),
    tier_bands={"Star": (-250, 120), "Mid": (120, 300), "Under": (300, 1200)},
)

SPORTS = {s.key: s for s in (MLB, NFL, NBA)}
DEFAULT_SPORT = "mlb"


def get_sport(key: str | None) -> SportSpec:
    """Look up a sport, falling back to MLB rather than raising.

    The app's sport picker, the workflows and the scripts all funnel through
    here, so an unknown key degrades to the sport that definitely has data
    instead of taking the page down.
    """
    return SPORTS.get(str(key or "").lower(), SPORTS[DEFAULT_SPORT])


def ensure_data_dir(spec: SportSpec) -> str:
    os.makedirs(spec.data_dir, exist_ok=True)
    return spec.data_dir
