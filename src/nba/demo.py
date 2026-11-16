"""Deterministic synthetic NBA slate — the app works with zero network.

Mirrors the baseball and football demo generators: profiles are hashed from
the player's name and the slate date, so a board is reproducible and the tests
have something stable to assert against.

The archetypes exist because three-point props live or die on the interaction
between MINUTES and ATTEMPT RATE, and the interesting cases are the ones where
those two disagree — the bench gunner who takes nine threes per 36 but only
plays 17 minutes, against the starter who plays 36 and takes four.
"""

from __future__ import annotations

import hashlib
import random

import pandas as pd

from .teams import TEAMS

# archetype -> (minutes, 3PA per 36, 3P%, season 3PA)
_ARCHETYPES = {
    "sniper":      ((32, 38), (9.0, 12.0), (0.375, 0.425), (300, 560)),
    "star":        ((33, 38), (5.5, 8.5), (0.340, 0.390), (250, 450)),
    "wing3d":      ((26, 34), (5.5, 8.0), (0.355, 0.405), (200, 380)),
    "gunner":      ((14, 22), (9.5, 13.0), (0.330, 0.390), (120, 260)),
    "stretchbig":  ((24, 32), (4.0, 6.5), (0.340, 0.385), (150, 300)),
    "slasher":     ((28, 35), (2.0, 4.0), (0.300, 0.350), (80, 190)),
    "rimbig":      ((22, 30), (0.2, 1.2), (0.200, 0.300), (5, 40)),
    "guard2":      ((22, 30), (4.5, 7.0), (0.330, 0.380), (140, 280)),
    "bench":       ((12, 20), (3.0, 6.0), (0.310, 0.370), (60, 150)),
}
_SLOTS = ["sniper", "star", "wing3d", "slasher", "rimbig",
          "guard2", "stretchbig", "gunner", "bench"]

_FIRST = ["Jalen", "Tyrese", "Anthony", "Paolo", "Franz", "Cade", "Scottie",
          "Evan", "Keegan", "Jabari", "Bennedict", "Jaden", "Shaedon", "Amen",
          "Brandon", "Devin", "Austin", "Cason", "Donovan", "Trey", "Max"]
_LAST = ["Mathurin", "Sharpe", "Banchero", "Wagner", "Cunningham", "Barnes",
         "Mobley", "Murray", "Williams", "Thompson", "Edwards", "Sengun",
         "Giddey", "Wallace", "Hield", "Johnson", "Reaves", "Castle", "Dort"]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _name(team: str, slot: int, seed: str) -> str:
    h = _seed(team, slot, seed, "name")
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h // 89) % len(_LAST)]}"


def _profile(team: str, arch: str, slot: int, seed: str) -> dict:
    rng = random.Random(_seed(team, slot, seed, "prof"))
    mins, rate, pct, att = _ARCHETYPES[arch]
    m = round(rng.uniform(*mins), 1)
    return {
        "archetype": arch,
        "minutes_l5": m,
        "minutes_season": round(m + rng.uniform(-2.5, 2.5), 1),
        "minutes_stdev_l5": round(rng.uniform(1.5, 7.0), 1),
        "fg3a_per36": round(rng.uniform(*rate), 2),
        "fg3_pct": round(rng.uniform(*pct), 4),
        "fg3a_season": rng.randint(*att),
        # Most nights nobody is out; when someone is, the backup's minutes jump.
        "teammate_out_minutes_bump": (round(rng.uniform(3.0, 8.0), 1)
                                      if rng.random() < 0.15 else 0.0),
    }


def _unique_namer(seed: str):
    """Hand out distinct names for one slate.

    Hashing a name per player can collide, and two different players sharing a
    name is not cosmetic here: the parlay builder de-dupes legs by player, so a
    collision silently drops a leg or stacks the same name twice.
    """
    pool = [f"{f} {l}" for f in _FIRST for l in _LAST]
    rng = random.Random(_seed(seed, "names"))
    rng.shuffle(pool)
    it = iter(pool)
    return lambda *_: next(it)


def build_demo_slate(date_iso: str, n_games: int = 6) -> pd.DataFrame:
    teams = sorted(TEAMS)
    rng = random.Random(_seed(date_iso, "slate"))
    rng.shuffle(teams)
    namer = _unique_namer(date_iso)
    rows = []
    for g in range(min(n_games, len(teams) // 2)):
        away, home = teams[2 * g], teams[2 * g + 1]
        grng = random.Random(_seed(date_iso, home, away, "game"))
        home_pace = round(grng.uniform(95.0, 104.0), 1)
        away_pace = round(grng.uniform(95.0, 104.0), 1)
        total = round(grng.uniform(215.5, 241.5) * 2) / 2
        for side, team, opp, pace, opp_pace in (
                ("away", away, home, away_pace, home_pace),
                ("home", home, away, home_pace, away_pace)):
            trng = random.Random(_seed(date_iso, team, "team"))
            b2b = trng.random() < 0.18
            opp_3pa = round(trng.uniform(31.0, 41.5), 1)
            opp_pct = round(trng.uniform(0.335, 0.392), 4)
            team_3pa = round(trng.uniform(30.0, 43.0), 1)
            for slot, arch in enumerate(_SLOTS):
                rows.append({
                    "player": namer(),
                    "team": team, "opponent": opp, "team_name": TEAMS[team],
                    "game": f"{away} @ {home}",
                    "home_team": home, "is_home": side == "home",
                    "game_total": total,
                    "team_pace": pace, "opp_pace": opp_pace,
                    "team_3pa_per_game": team_3pa,
                    "opp_3pa_allowed": opp_3pa, "opp_3p_pct_allowed": opp_pct,
                    "back_to_back": b2b,
                    "days_rest": 0 if b2b else trng.randint(1, 3),
                    "data_quality": "modeled",
                    **_profile(team, arch, slot, date_iso),
                })
    return pd.DataFrame(rows)
