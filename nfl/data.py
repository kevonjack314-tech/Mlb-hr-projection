"""NFL analytics data — teams, player archetypes, previous-season baselines,
and player-vs-team history.

Pure analytics (no betting framework): the model runs on
  • previous-season per-game production (yards, TDs, usage) — deterministic
    modeled baselines now, with a `nfl_data_py` hook to overlay REAL prior-season
    numbers (same pattern the MLB app used before going live),
  • team defense strength + the SCHEME they run (see schemes.py),
  • each player's archetype (who they are stylistically), and
  • what the player has done against THIS team before.

Roster tuple: (name, pos, tier 1-5, archetype).
"""

from __future__ import annotations

import hashlib
import random

import pandas as pd

# team -> (name, offense strength 1-5, defense strength 1-5, dome?)
TEAMS = {
    "KC":  ("Chiefs", 5, 4, False), "BUF": ("Bills", 5, 4, False),
    "BAL": ("Ravens", 5, 4, False), "DET": ("Lions", 5, 3, True),
    "SF":  ("49ers", 4, 4, False),  "PHI": ("Eagles", 4, 4, False),
    "CIN": ("Bengals", 4, 2, False), "MIA": ("Dolphins", 4, 3, False),
    "DAL": ("Cowboys", 4, 3, True), "HOU": ("Texans", 4, 3, True),
    "GB":  ("Packers", 4, 3, False), "LAR": ("Rams", 4, 3, True),
    "NYJ": ("Jets", 3, 4, False),   "PIT": ("Steelers", 3, 4, False),
    "CLE": ("Browns", 2, 4, False), "JAX": ("Jaguars", 3, 2, False),
    "LAC": ("Chargers", 3, 3, True), "ATL": ("Falcons", 3, 3, True),
    "MIN": ("Vikings", 3, 3, True), "SEA": ("Seahawks", 3, 3, False),
    "TB":  ("Buccaneers", 3, 3, False), "IND": ("Colts", 3, 2, True),
    "NO":  ("Saints", 2, 3, True),  "ARI": ("Cardinals", 3, 2, False),
    "LV":  ("Raiders", 2, 2, True), "DEN": ("Broncos", 2, 4, False),
    "CHI": ("Bears", 3, 3, False),  "WAS": ("Commanders", 3, 2, False),
    "NYG": ("Giants", 2, 3, False), "NE":  ("Patriots", 2, 3, False),
    "TEN": ("Titans", 2, 2, False), "CAR": ("Panthers", 2, 2, False),
}

ROSTERS = {
    "KC":  [("Patrick Mahomes", "QB", 5, "pocket_passer"),
            ("Isiah Pacheco", "RB", 3, "zone_runner"),
            ("Rashee Rice", "WR", 4, "yac_creator"),
            ("Xavier Worthy", "WR", 3, "deep_threat"),
            ("Hollywood Brown", "WR", 3, "deep_threat"),
            ("Travis Kelce", "TE", 4, "seam_te")],
    "BUF": [("Josh Allen", "QB", 5, "dual_threat_qb"),
            ("James Cook", "RB", 4, "zone_runner"),
            ("Khalil Shakir", "WR", 3, "slot_tech"),
            ("Keon Coleman", "WR", 3, "alpha_x"),
            ("Curtis Samuel", "WR", 2, "yac_creator"),
            ("Dalton Kincaid", "TE", 3, "seam_te")],
    "BAL": [("Lamar Jackson", "QB", 5, "dual_threat_qb"),
            ("Derrick Henry", "RB", 5, "workhorse_power"),
            ("Zay Flowers", "WR", 4, "yac_creator"),
            ("Rashod Bateman", "WR", 3, "alpha_x"),
            ("Nelson Agholor", "WR", 2, "deep_threat"),
            ("Mark Andrews", "TE", 4, "rz_te")],
    "DET": [("Jared Goff", "QB", 4, "pocket_passer"),
            ("Jahmyr Gibbs", "RB", 5, "receiving_back"),
            ("David Montgomery", "RB", 4, "workhorse_power"),
            ("Amon-Ra St. Brown", "WR", 5, "yac_creator"),
            ("Jameson Williams", "WR", 3, "deep_threat"),
            ("Sam LaPorta", "TE", 4, "seam_te")],
    "SF":  [("Brock Purdy", "QB", 4, "pocket_passer"),
            ("Christian McCaffrey", "RB", 5, "receiving_back"),
            ("Deebo Samuel", "WR", 4, "yac_creator"),
            ("Brandon Aiyuk", "WR", 4, "alpha_x"),
            ("Jauan Jennings", "WR", 2, "slot_tech"),
            ("George Kittle", "TE", 4, "seam_te")],
    "PHI": [("Jalen Hurts", "QB", 5, "dual_threat_qb"),
            ("Saquon Barkley", "RB", 5, "workhorse_power"),
            ("A.J. Brown", "WR", 5, "alpha_x"),
            ("DeVonta Smith", "WR", 4, "slot_tech"),
            ("Jahan Dotson", "WR", 2, "deep_threat"),
            ("Dallas Goedert", "TE", 3, "yac_creator")],
    "CIN": [("Joe Burrow", "QB", 5, "pocket_passer"),
            ("Chase Brown", "RB", 3, "zone_runner"),
            ("Ja'Marr Chase", "WR", 5, "alpha_x"),
            ("Tee Higgins", "WR", 4, "alpha_x"),
            ("Andrei Iosivas", "WR", 2, "rz_te"),
            ("Mike Gesicki", "TE", 2, "seam_te")],
    "MIA": [("Tua Tagovailoa", "QB", 4, "pocket_passer"),
            ("De'Von Achane", "RB", 4, "receiving_back"),
            ("Raheem Mostert", "RB", 3, "zone_runner"),
            ("Tyreek Hill", "WR", 5, "deep_threat"),
            ("Jaylen Waddle", "WR", 4, "yac_creator"),
            ("Jonnu Smith", "TE", 2, "seam_te")],
    "DAL": [("Dak Prescott", "QB", 4, "pocket_passer"),
            ("Rico Dowdle", "RB", 3, "zone_runner"),
            ("CeeDee Lamb", "WR", 5, "alpha_x"),
            ("Brandin Cooks", "WR", 3, "deep_threat"),
            ("Jalen Tolbert", "WR", 2, "deep_threat"),
            ("Jake Ferguson", "TE", 3, "rz_te")],
    "HOU": [("C.J. Stroud", "QB", 4, "pocket_passer"),
            ("Joe Mixon", "RB", 4, "workhorse_power"),
            ("Nico Collins", "WR", 5, "alpha_x"),
            ("Stefon Diggs", "WR", 4, "slot_tech"),
            ("Tank Dell", "WR", 3, "deep_threat"),
            ("Dalton Schultz", "TE", 3, "rz_te")],
    "GB":  [("Jordan Love", "QB", 4, "pocket_passer"),
            ("Josh Jacobs", "RB", 4, "workhorse_power"),
            ("Jayden Reed", "WR", 4, "slot_tech"),
            ("Romeo Doubs", "WR", 3, "alpha_x"),
            ("Christian Watson", "WR", 3, "deep_threat"),
            ("Tucker Kraft", "TE", 3, "yac_creator")],
    "LAR": [("Matthew Stafford", "QB", 4, "pocket_passer"),
            ("Kyren Williams", "RB", 4, "zone_runner"),
            ("Puka Nacua", "WR", 5, "alpha_x"),
            ("Cooper Kupp", "WR", 4, "slot_tech"),
            ("Demarcus Robinson", "WR", 2, "deep_threat"),
            ("Colby Parkinson", "TE", 2, "seam_te")],
    "NYJ": [("Aaron Rodgers", "QB", 4, "pocket_passer"),
            ("Breece Hall", "RB", 4, "receiving_back"),
            ("Garrett Wilson", "WR", 4, "alpha_x"),
            ("Mike Williams", "WR", 3, "alpha_x"),
            ("Allen Lazard", "WR", 2, "rz_te"),
            ("Tyler Conklin", "TE", 2, "seam_te")],
    "PIT": [("Russell Wilson", "QB", 3, "pocket_passer"),
            ("Najee Harris", "RB", 3, "workhorse_power"),
            ("Jaylen Warren", "RB", 3, "receiving_back"),
            ("George Pickens", "WR", 4, "alpha_x"),
            ("Calvin Austin", "WR", 2, "deep_threat"),
            ("Pat Freiermuth", "TE", 3, "rz_te")],
    "CLE": [("Deshaun Watson", "QB", 2, "pocket_passer"),
            ("Nick Chubb", "RB", 4, "workhorse_power"),
            ("Amari Cooper", "WR", 4, "alpha_x"),
            ("Jerry Jeudy", "WR", 3, "slot_tech"),
            ("Elijah Moore", "WR", 2, "yac_creator"),
            ("David Njoku", "TE", 3, "rz_te")],
    "JAX": [("Trevor Lawrence", "QB", 4, "pocket_passer"),
            ("Travis Etienne", "RB", 4, "receiving_back"),
            ("Brian Thomas Jr.", "WR", 4, "deep_threat"),
            ("Christian Kirk", "WR", 3, "slot_tech"),
            ("Gabe Davis", "WR", 2, "deep_threat"),
            ("Evan Engram", "TE", 3, "yac_creator")],
    "LAC": [("Justin Herbert", "QB", 4, "pocket_passer"),
            ("J.K. Dobbins", "RB", 3, "zone_runner"),
            ("Ladd McConkey", "WR", 4, "slot_tech"),
            ("Joshua Palmer", "WR", 2, "alpha_x"),
            ("Quentin Johnston", "WR", 2, "deep_threat"),
            ("Will Dissly", "TE", 2, "rz_te")],
    "ATL": [("Kirk Cousins", "QB", 4, "pocket_passer"),
            ("Bijan Robinson", "RB", 5, "receiving_back"),
            ("Tyler Allgeier", "RB", 2, "workhorse_power"),
            ("Drake London", "WR", 4, "alpha_x"),
            ("Darnell Mooney", "WR", 3, "deep_threat"),
            ("Kyle Pitts", "TE", 3, "seam_te")],
    "MIN": [("Sam Darnold", "QB", 3, "pocket_passer"),
            ("Aaron Jones", "RB", 4, "zone_runner"),
            ("Justin Jefferson", "WR", 5, "alpha_x"),
            ("Jordan Addison", "WR", 3, "deep_threat"),
            ("Jalen Nailor", "WR", 2, "deep_threat"),
            ("T.J. Hockenson", "TE", 3, "seam_te")],
    "SEA": [("Geno Smith", "QB", 3, "pocket_passer"),
            ("Kenneth Walker", "RB", 4, "workhorse_power"),
            ("DK Metcalf", "WR", 4, "alpha_x"),
            ("Tyler Lockett", "WR", 3, "slot_tech"),
            ("Jaxon Smith-Njigba", "WR", 3, "slot_tech"),
            ("Noah Fant", "TE", 2, "seam_te")],
    "TB":  [("Baker Mayfield", "QB", 3, "pocket_passer"),
            ("Rachaad White", "RB", 3, "receiving_back"),
            ("Bucky Irving", "RB", 3, "zone_runner"),
            ("Mike Evans", "WR", 4, "alpha_x"),
            ("Chris Godwin", "WR", 4, "slot_tech"),
            ("Cade Otton", "TE", 2, "rz_te")],
    "IND": [("Anthony Richardson", "QB", 3, "dual_threat_qb"),
            ("Jonathan Taylor", "RB", 5, "workhorse_power"),
            ("Michael Pittman", "WR", 4, "alpha_x"),
            ("Josh Downs", "WR", 3, "slot_tech"),
            ("Alec Pierce", "WR", 2, "deep_threat"),
            ("Kylen Granson", "TE", 2, "seam_te")],
    "NO":  [("Derek Carr", "QB", 3, "pocket_passer"),
            ("Alvin Kamara", "RB", 4, "receiving_back"),
            ("Chris Olave", "WR", 4, "alpha_x"),
            ("Rashid Shaheed", "WR", 3, "deep_threat"),
            ("Cedrick Wilson", "WR", 1, "slot_tech"),
            ("Juwan Johnson", "TE", 2, "rz_te")],
    "ARI": [("Kyler Murray", "QB", 4, "dual_threat_qb"),
            ("James Conner", "RB", 4, "workhorse_power"),
            ("Marvin Harrison Jr.", "WR", 4, "alpha_x"),
            ("Michael Wilson", "WR", 2, "alpha_x"),
            ("Greg Dortch", "WR", 2, "slot_tech"),
            ("Trey McBride", "TE", 4, "seam_te")],
    "LV":  [("Gardner Minshew", "QB", 2, "pocket_passer"),
            ("Zamir White", "RB", 2, "workhorse_power"),
            ("Davante Adams", "WR", 5, "alpha_x"),
            ("Jakobi Meyers", "WR", 3, "slot_tech"),
            ("Tre Tucker", "WR", 2, "deep_threat"),
            ("Brock Bowers", "TE", 4, "seam_te")],
    "DEN": [("Bo Nix", "QB", 3, "pocket_passer"),
            ("Javonte Williams", "RB", 3, "workhorse_power"),
            ("Courtland Sutton", "WR", 4, "alpha_x"),
            ("Josh Reynolds", "WR", 2, "alpha_x"),
            ("Marvin Mims", "WR", 2, "deep_threat"),
            ("Greg Dulcich", "TE", 1, "seam_te")],
    "CHI": [("Caleb Williams", "QB", 3, "dual_threat_qb"),
            ("D'Andre Swift", "RB", 3, "receiving_back"),
            ("DJ Moore", "WR", 4, "yac_creator"),
            ("Keenan Allen", "WR", 4, "slot_tech"),
            ("Rome Odunze", "WR", 3, "alpha_x"),
            ("Cole Kmet", "TE", 3, "rz_te")],
    "WAS": [("Jayden Daniels", "QB", 4, "dual_threat_qb"),
            ("Brian Robinson", "RB", 3, "workhorse_power"),
            ("Austin Ekeler", "RB", 3, "receiving_back"),
            ("Terry McLaurin", "WR", 4, "alpha_x"),
            ("Noah Brown", "WR", 2, "alpha_x"),
            ("Zach Ertz", "TE", 2, "rz_te")],
    "NYG": [("Daniel Jones", "QB", 2, "dual_threat_qb"),
            ("Devin Singletary", "RB", 3, "zone_runner"),
            ("Malik Nabers", "WR", 4, "alpha_x"),
            ("Wan'Dale Robinson", "WR", 2, "slot_tech"),
            ("Darius Slayton", "WR", 2, "deep_threat"),
            ("Theo Johnson", "TE", 1, "seam_te")],
    "NE":  [("Drake Maye", "QB", 3, "dual_threat_qb"),
            ("Rhamondre Stevenson", "RB", 3, "workhorse_power"),
            ("DeMario Douglas", "WR", 2, "slot_tech"),
            ("Kendrick Bourne", "WR", 2, "yac_creator"),
            ("Ja'Lynn Polk", "WR", 2, "alpha_x"),
            ("Hunter Henry", "TE", 3, "rz_te")],
    "TEN": [("Will Levis", "QB", 2, "pocket_passer"),
            ("Tony Pollard", "RB", 3, "zone_runner"),
            ("Tyjae Spears", "RB", 3, "receiving_back"),
            ("Calvin Ridley", "WR", 4, "alpha_x"),
            ("DeAndre Hopkins", "WR", 3, "alpha_x"),
            ("Chigoziem Okonkwo", "TE", 2, "seam_te")],
    "CAR": [("Bryce Young", "QB", 2, "pocket_passer"),
            ("Chuba Hubbard", "RB", 3, "zone_runner"),
            ("Diontae Johnson", "WR", 3, "slot_tech"),
            ("Adam Thielen", "WR", 3, "slot_tech"),
            ("Xavier Legette", "WR", 2, "deep_threat"),
            ("Tommy Tremble", "TE", 1, "rz_te")],
}

ARCHETYPE_LABEL = {
    "alpha_x": "Alpha X receiver", "deep_threat": "Deep threat",
    "slot_tech": "Slot technician", "yac_creator": "YAC creator",
    "workhorse_power": "Workhorse power back", "zone_runner": "Zone runner",
    "receiving_back": "Receiving back", "seam_te": "Seam-stretching TE",
    "rz_te": "Red-zone TE", "dual_threat_qb": "Dual-threat QB",
    "pocket_passer": "Pocket passer",
}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def prev_season_baseline(name: str, pos: str, tier: int, season: int) -> dict:
    """Previous-season per-game production (modeled, deterministic).

    Hook: replace with real numbers from `nfl_data_py.import_weekly_data(
    [season-1])` aggregated per player — the shapes match 1:1.
    """
    rng = random.Random(_seed(name, season - 1, "prevszn"))
    b = {"prev_games": rng.randint(13, 17)}
    if pos == "QB":
        b.update(pass_ypg=round({2: 195, 3: 222, 4: 248, 5: 272}.get(tier, 210)
                                + rng.uniform(-15, 15), 1),
                 rush_ypg=round(rng.uniform(8, 40 if tier >= 4 else 22), 1),
                 rec_ypg=0.0, rec_pg=0.0,
                 prev_tds=rng.randint(3, 9),           # rushing TDs
                 rz_share=round(rng.uniform(0.10, 0.28), 3))
    elif pos == "RB":
        b.update(rush_ypg=round({1: 22, 2: 38, 3: 55, 4: 74, 5: 92}[tier]
                                + rng.uniform(-8, 8), 1),
                 rec_ypg=round(rng.uniform(8, 34 if tier >= 4 else 22), 1),
                 rec_pg=round(rng.uniform(1.5, 4.8), 1), pass_ypg=0.0,
                 prev_tds=rng.randint(max(1, tier * 2 - 2), tier * 3 + 2),
                 rz_share=round({1: .05, 2: .10, 3: .16, 4: .24, 5: .30}[tier]
                                + rng.uniform(-.03, .03), 3))
    else:
        base = {1: 22, 2: 38, 3: 54, 4: 72, 5: 92}[tier] * (0.85 if pos == "TE" else 1)
        b.update(rec_ypg=round(base + rng.uniform(-8, 8), 1),
                 rec_pg=round(base / 12.5, 1), rush_ypg=0.0, pass_ypg=0.0,
                 prev_tds=rng.randint(max(1, tier - 1), tier + 4),
                 rz_share=round({1: .04, 2: .08, 3: .12, 4: .17, 5: .22}[tier]
                                + rng.uniform(-.02, .02), 3))
    return b


def vs_team_history(name: str, opp: str) -> dict:
    """Career production vs this specific opponent (modeled, deterministic).

    Hook: aggregate real games from nfl_data_py play-by-play once wired.
    """
    rng = random.Random(_seed(name, opp, "vshist"))
    games = rng.randint(0, 6)
    if games == 0:
        return {"vs_games": 0, "vs_ypg_mult": 1.0, "vs_tds": 0}
    return {"vs_games": games,
            "vs_ypg_mult": round(max(0.65, min(1.40, rng.gauss(1.0, 0.18))), 2),
            "vs_tds": rng.randint(0, max(1, games))}


def _matchups_for_week(week: int) -> list[tuple[str, str]]:
    abbrs = list(TEAMS.keys())
    rng = random.Random(_seed("nflweek", week))
    rng.shuffle(abbrs)
    return [(abbrs[i], abbrs[i + 1]) for i in range(0, len(abbrs), 2)]


def build_week_slate(week: int, season: int = 2026) -> pd.DataFrame:
    """One row per skill player for the week's games, with prev-season baselines,
    archetype, opponent scheme context, and vs-team history attached."""
    rng = random.Random(_seed(season, week, "env"))
    rows = []
    for away, home in _matchups_for_week(week):
        h_off, h_def = TEAMS[home][1], TEAMS[home][2]
        a_off, a_def = TEAMS[away][1], TEAMS[away][2]
        total = round(38 + 2.2 * (h_off + a_off) - 1.4 * (h_def + a_def)
                      + rng.uniform(-3, 3) + 6, 1)
        spread = round(-(h_off - a_off) * 1.8 - 2.0 + rng.uniform(-2, 2), 1)
        dome = TEAMS[home][3]
        wind = 0.0 if dome else round(max(0.0, rng.gauss(8, 5)), 0)
        for team, opp, implied, opp_def, is_home in (
                (home, away, round(total / 2 - spread / 2, 1), a_def, True),
                (away, home, round(total / 2 + spread / 2, 1), h_def, False)):
            for name, pos, tier, arch in ROSTERS[team]:
                rows.append({
                    "player": name, "pos": pos, "team": team, "opponent": opp,
                    "archetype": arch, "tier": tier,
                    "game": f"{away} @ {home}", "is_home": is_home,
                    "team_implied": implied, "game_total": total,
                    "opp_def": opp_def, "wind_mph": wind, "dome": dome,
                    **prev_season_baseline(name, pos, tier, season),
                    **vs_team_history(name, opp),
                })
    return pd.DataFrame(rows)


def live_week_slate(week: int, season: int):
    """Hook for nfl_data_py (nflverse) — real rosters, real prior-season stats,
    real vs-team history. Returns None until wired (season starts September)."""
    return None
