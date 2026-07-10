"""Team scheme profiles + archetype-vs-scheme interactions.

The core idea the app runs on: WHAT a defense plays determines WHO eats.
Each team gets a defensive scheme profile (coverage shell, man/blitz rates,
front style) and an offensive system. Each skill player gets an archetype.
The interaction matrix below turns "Ja'Marr Chase vs a press-man blitz team"
into a quantified boost — and a plain-language reason.

Scheme profiles are bundled, editable data (best-effort tendencies — swap in
charting data like man/zone rates from nflverse participation when wired).
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Defensive schemes
# --------------------------------------------------------------------------- #
DEF_SCHEMES = {
    "press_man_blitz": {
        "label": "Press-man, blitz-heavy",
        "desc": "High man-coverage rate with frequent pressure. Lives and dies "
                "on winning one-on-ones outside; vulnerable to alpha receivers "
                "and quick game / screens when the blitz comes.",
    },
    "two_high_zone": {
        "label": "Two-high zone shell",
        "desc": "Quarters/Cover-2 base that takes away deep shots and forces "
                "checkdowns. Light boxes invite the run; YAC and underneath "
                "targets thrive, vertical threats get capped.",
    },
    "single_high_man": {
        "label": "Single-high (Cover 1/3)",
        "desc": "One deep safety, corners on islands, loaded run box. Seams and "
                "deep posts open up — but running into the box is a grind.",
    },
    "soft_zone_bend": {
        "label": "Soft zone, bend-don't-break",
        "desc": "Low blitz, deep-drop zones. Gives up the underneath all day: "
                "slot receivers, checkdown backs and YAC creators feast; "
                "explosives are rare.",
    },
    "attacking_front": {
        "label": "Attacking front, gap-shooters",
        "desc": "Penetrating defensive line that blows up runs at the mercy of "
                "over-pursuit. Screens and play-action shots punish it; "
                "straight-ahead run volume struggles.",
    },
}

# team -> defensive scheme key (bundled, editable)
TEAM_DEF = {
    "KC": "two_high_zone", "BUF": "two_high_zone", "BAL": "press_man_blitz",
    "DET": "attacking_front", "SF": "single_high_man", "PHI": "attacking_front",
    "CIN": "soft_zone_bend", "MIA": "press_man_blitz", "DAL": "attacking_front",
    "HOU": "press_man_blitz", "GB": "two_high_zone", "LAR": "two_high_zone",
    "NYJ": "press_man_blitz", "PIT": "single_high_man", "CLE": "press_man_blitz",
    "JAX": "soft_zone_bend", "LAC": "two_high_zone", "ATL": "soft_zone_bend",
    "MIN": "press_man_blitz", "SEA": "single_high_man", "TB": "single_high_man",
    "IND": "two_high_zone", "NO": "single_high_man", "ARI": "soft_zone_bend",
    "LV": "soft_zone_bend", "DEN": "press_man_blitz", "CHI": "two_high_zone",
    "WAS": "soft_zone_bend", "NYG": "attacking_front", "NE": "two_high_zone",
    "TEN": "single_high_man", "CAR": "soft_zone_bend",
}

# team -> offensive system (colors the volume story, shown in matchup text)
TEAM_OFF = {
    "KC": "Spread quick-game", "BUF": "Spread RPO", "BAL": "Option run + play-action",
    "DET": "Wide zone + heavy play-action", "SF": "Wide zone YAC machine",
    "PHI": "RPO + QB run", "CIN": "Dropback vertical", "MIA": "Speed motion outside zone",
    "DAL": "Dropback spread", "HOU": "Vertical play-action", "GB": "West coast motion",
    "LAR": "Condensed play-action", "NYJ": "West coast quick-game",
    "PIT": "Run-first play-action", "CLE": "Wide zone play-action",
    "JAX": "Spread dropback", "LAC": "Run-heavy play-action", "ATL": "Outside zone",
    "MIN": "Play-action shots", "SEA": "Spread tempo", "TB": "Dropback spread",
    "IND": "RPO tempo", "NO": "Quick-game screens", "ARI": "Spread QB-run",
    "LV": "West coast", "DEN": "Outside zone quick-game", "CHI": "Spread RPO",
    "WAS": "Spread QB-run", "NYG": "West coast", "NE": "Under-center play-action",
    "TEN": "Vertical dropback", "CAR": "Quick-game west coast",
}

# --------------------------------------------------------------------------- #
# Archetype × defensive-scheme interaction matrix
#   archetype -> scheme -> (yds_mult, td_mult, reason)
# Neutral = (1.0, 1.0). Kept in ±15% bands — schemes tilt games, not decide them.
# --------------------------------------------------------------------------- #
INTERACTIONS = {
    "alpha_x": {   # true #1 outside receiver
        "press_man_blitz": (1.13, 1.15, "alpha outside receivers feast on press-man islands"),
        "single_high_man": (1.07, 1.08, "one-on-ones outside with a single-high safety"),
        "two_high_zone": (0.93, 0.92, "two-high shells bracket the alpha and cap explosives"),
        "soft_zone_bend": (1.02, 0.97, "volume is there underneath but the end zone shrinks"),
        "attacking_front": (1.04, 1.05, "pressure forces quick throws to the best hand"),
    },
    "deep_threat": {
        "press_man_blitz": (1.10, 1.12, "blitzes leave no safety help — one step and it's gone"),
        "single_high_man": (1.09, 1.10, "deep posts beat single-high all day"),
        "two_high_zone": (0.86, 0.85, "two deep safeties erase the vertical game"),
        "soft_zone_bend": (0.90, 0.90, "soft zones never let it get over the top"),
        "attacking_front": (1.05, 1.06, "play-action shots punish run-crashing fronts"),
    },
    "slot_tech": {
        "press_man_blitz": (1.04, 1.03, "quick separation beats man inside when the blitz comes"),
        "two_high_zone": (1.08, 1.04, "soft middle of two-high is the slot's office"),
        "soft_zone_bend": (1.12, 1.05, "underneath zones concede the slot all game"),
        "single_high_man": (0.96, 0.96, "physical slot coverage with help inside"),
        "attacking_front": (1.02, 1.0, "hot routes go to the slot vs pressure"),
    },
    "yac_creator": {
        "soft_zone_bend": (1.13, 1.08, "catch-and-run space all day vs soft zone"),
        "two_high_zone": (1.09, 1.04, "underneath completions turn into YAC vs two-high"),
        "press_man_blitz": (1.02, 1.02, "one broken tackle beats man with no help"),
        "single_high_man": (1.0, 1.0, ""),
        "attacking_front": (1.06, 1.03, "screens weaponize the YAC threat vs gap-shooters"),
    },
    "workhorse_power": {
        "two_high_zone": (1.10, 1.10, "light two-high boxes are a power back's dream"),
        "soft_zone_bend": (1.06, 1.06, "bend-don't-break concedes the ground game"),
        "single_high_man": (0.92, 0.94, "loaded single-high boxes make it a grind"),
        "press_man_blitz": (0.98, 1.0, ""),
        "attacking_front": (0.90, 0.93, "gap-shooting fronts blow up downhill runs"),
    },
    "zone_runner": {
        "two_high_zone": (1.08, 1.07, "light boxes + zone tracks = chunk runs"),
        "attacking_front": (0.93, 0.95, "penetration kills the zone track before it starts"),
        "single_high_man": (0.94, 0.95, "extra hat in the box vs single-high"),
        "soft_zone_bend": (1.05, 1.04, ""),
        "press_man_blitz": (1.0, 1.0, ""),
    },
    "receiving_back": {
        "press_man_blitz": (1.11, 1.06, "checkdowns and screens beat the blitz"),
        "soft_zone_bend": (1.08, 1.03, "free checkdowns all game vs soft zone"),
        "attacking_front": (1.07, 1.03, "screens punish over-pursuing fronts"),
        "two_high_zone": (1.05, 1.0, "two-high forces the ball underneath to the back"),
        "single_high_man": (1.02, 1.0, ""),
    },
    "seam_te": {
        "single_high_man": (1.10, 1.10, "the seam is open vs single-high"),
        "two_high_zone": (1.06, 1.04, "the TE works the soft middle between shells"),
        "press_man_blitz": (1.02, 1.04, "mismatch vs a nickel or backer in man"),
        "soft_zone_bend": (1.03, 1.0, ""),
        "attacking_front": (0.98, 1.0, ""),
    },
    "rz_te": {       # red-zone target TE
        "press_man_blitz": (1.0, 1.10, "back-shoulder fades vs man in the red zone"),
        "single_high_man": (1.0, 1.06, ""),
        "two_high_zone": (1.0, 1.04, "zone softens inside the 10"),
        "soft_zone_bend": (1.0, 1.02, ""),
        "attacking_front": (1.0, 1.0, ""),
    },
    "dual_threat_qb": {
        "press_man_blitz": (1.08, 1.10, "man coverage turns its back — the QB takes off"),
        "attacking_front": (1.05, 1.06, "upfield rush lanes invite scrambles"),
        "two_high_zone": (1.04, 1.03, "two-high spies get stressed by QB runs"),
        "single_high_man": (1.02, 1.03, ""),
        "soft_zone_bend": (0.98, 0.98, ""),
    },
    "pocket_passer": {
        "soft_zone_bend": (1.07, 1.04, "all-day clean pocket vs low-blitz zone"),
        "two_high_zone": (1.0, 0.97, "takes the checkdowns two-high gives"),
        "press_man_blitz": (0.93, 0.94, "pressure + man is the hard mode for pocket QBs"),
        "single_high_man": (1.02, 1.02, ""),
        "attacking_front": (0.95, 0.96, ""),
    },
}

DEFAULT_INTERACTION = (1.0, 1.0, "")


def scheme_boost(archetype: str, def_scheme: str):
    """(yds_mult, td_mult, reason) for a player archetype vs a defensive scheme."""
    return INTERACTIONS.get(archetype, {}).get(def_scheme, DEFAULT_INTERACTION)


def def_scheme_of(team: str) -> str:
    return TEAM_DEF.get(team, "two_high_zone")


def scheme_label(key: str) -> str:
    return DEF_SCHEMES.get(key, {}).get("label", key)


def scheme_desc(key: str) -> str:
    return DEF_SCHEMES.get(key, {}).get("desc", "")
