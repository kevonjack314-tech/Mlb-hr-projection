"""Deterministic synthetic NFL slate — the app works with zero network.

Mirrors src/demo.py for baseball: every player's usage profile is generated
from a hash of their name plus the slate date, so the same player gets the same
profile within a date and the board is reproducible. Role archetypes are what
matter for touchdowns, so the generator builds depth charts around them: a
bell-cow back, a goal-line back who barely plays otherwise, a target-hog WR1,
a red-zone tight end, and so on.
"""

from __future__ import annotations

import hashlib
import random

import pandas as pd

from .teams import TEAMS, is_indoor

# (name, position, archetype) per team slot. Archetypes drive the usage split.
_ARCHETYPES = {
    "bellcow":   {"snap": (0.70, 0.85), "rush": (0.55, 0.72), "i5": (0.45, 0.70),
                  "tgt": (0.08, 0.15), "rz_tgt": (0.05, 0.11)},
    "committee": {"snap": (0.40, 0.58), "rush": (0.28, 0.45), "i5": (0.15, 0.35),
                  "tgt": (0.06, 0.13), "rz_tgt": (0.04, 0.09)},
    "goalline":  {"snap": (0.15, 0.30), "rush": (0.10, 0.22), "i5": (0.40, 0.65),
                  "tgt": (0.01, 0.04), "rz_tgt": (0.01, 0.04)},
    "wr1":       {"snap": (0.85, 0.95), "rush": (0.0, 0.02), "i5": (0.0, 0.03),
                  "tgt": (0.22, 0.31), "rz_tgt": (0.18, 0.30)},
    "wr2":       {"snap": (0.72, 0.88), "rush": (0.0, 0.02), "i5": (0.0, 0.02),
                  "tgt": (0.14, 0.21), "rz_tgt": (0.09, 0.17)},
    "slot":      {"snap": (0.60, 0.80), "rush": (0.0, 0.03), "i5": (0.0, 0.02),
                  "tgt": (0.12, 0.19), "rz_tgt": (0.06, 0.12)},
    "te_rz":     {"snap": (0.70, 0.88), "rush": (0.0, 0.0), "i5": (0.0, 0.02),
                  "tgt": (0.12, 0.19), "rz_tgt": (0.16, 0.26)},
    "te2":       {"snap": (0.30, 0.50), "rush": (0.0, 0.0), "i5": (0.0, 0.01),
                  "tgt": (0.04, 0.09), "rz_tgt": (0.03, 0.08)},
    "qb_rush":   {"snap": (0.98, 1.0), "rush": (0.08, 0.18), "i5": (0.10, 0.30),
                  "tgt": (0.0, 0.0), "rz_tgt": (0.0, 0.0)},
}

_SLOTS = [("RB", "bellcow"), ("RB", "committee"), ("RB", "goalline"),
          ("WR", "wr1"), ("WR", "wr2"), ("WR", "slot"),
          ("TE", "te_rz"), ("TE", "te2"), ("QB", "qb_rush")]

_FIRST = ["Marcus", "Jalen", "Trey", "Deon", "Kyren", "Isiah", "Brock", "Tank",
          "Rome", "Malik", "Xavier", "Chase", "Devon", "Rashee", "Caleb",
          "Jaxon", "Quentin", "Darnell", "Keon", "Tyree", "Amari", "Bo"]
_LAST = ["Harrison", "Brooks", "Nabers", "Odunze", "Bowers", "Worthy", "McBride",
         "Gibbs", "Hall", "Walker", "Spears", "Flowers", "Downs", "Mims",
         "Legette", "Coleman", "Thomas", "Pearsall", "Corum", "Allen", "Wright"]


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _name(team: str, slot: int, seed: str) -> str:
    h = _seed(team, slot, seed, "name")
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h // 97) % len(_LAST)]}"


def _profile(team: str, pos: str, arch: str, slot: int, seed: str) -> dict:
    rng = random.Random(_seed(team, slot, seed, "prof"))
    a = _ARCHETYPES[arch]
    snap = round(rng.uniform(*a["snap"]), 3)
    rush = round(rng.uniform(*a["rush"]), 3)
    i5 = round(rng.uniform(*a["i5"]), 3)
    tgt = round(rng.uniform(*a["tgt"]), 3)
    rz = round(rng.uniform(*a["rz_tgt"]), 3)
    return {
        "position": pos, "archetype": arch, "snap_share": snap,
        "rush_share": rush, "inside5_share": i5 if rush > 0.02 else None,
        "tgt_share": tgt, "rz_tgt_share": rz if tgt > 0.02 else None,
        "ypc": round(rng.uniform(3.4, 5.4), 2) if pos in ("RB", "QB") else None,
        "yprr": round(rng.uniform(1.0, 2.5), 2) if pos in ("WR", "TE") else None,
        "season_tds": rng.randint(0, 11),
        "td_rate_prior": round(rng.uniform(0.08, 0.52), 3),
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


def build_demo_slate(date_iso: str, n_games: int = 7) -> pd.DataFrame:
    """A believable Sunday slate: n_games matchups, nine skill players a side."""
    teams = sorted(TEAMS)
    rng = random.Random(_seed(date_iso, "slate"))
    rng.shuffle(teams)
    namer = _unique_namer(date_iso)
    rows = []
    for g in range(min(n_games, len(teams) // 2)):
        away, home = teams[2 * g], teams[2 * g + 1]
        grng = random.Random(_seed(date_iso, home, away, "game"))
        total = round(grng.uniform(37.5, 52.5) * 2) / 2
        spread = round(grng.uniform(-9.5, 9.5) * 2) / 2      # home spread
        # Implied team totals from the market line, the way a book derives them.
        home_total = round(total / 2 - spread / 2, 2)
        away_total = round(total / 2 + spread / 2, 2)
        indoors = is_indoor(home)
        wind = 0.0 if indoors else round(grng.uniform(2, 20), 1)
        temp = 70.0 if indoors else round(grng.uniform(24, 82), 1)
        opp_rz = round(grng.uniform(0.48, 0.68), 3)
        for side, team, opp, implied in (("away", away, home, away_total),
                                         ("home", home, away, home_total)):
            for slot, (pos, arch) in enumerate(_SLOTS):
                prof = _profile(team, pos, arch, slot, date_iso)
                prng = random.Random(_seed(team, slot, date_iso, "opp"))
                rows.append({
                    "player": namer(),
                    "team": team, "opponent": opp,
                    "team_name": TEAMS[team][0],
                    "game": f"{away} @ {home}",
                    "home_team": home, "is_home": side == "home",
                    "game_total": total, "spread": spread,
                    "implied_team_total": implied,
                    "roof": TEAMS[home][2], "wind_mph": wind, "temp_f": temp,
                    "opp_rz_td_rate": opp_rz,
                    "opp_td_allowed_pos": round(prng.uniform(0.25, 1.25), 2),
                    "data_quality": "modeled",
                    **prof,
                })
    return pd.DataFrame(rows)
