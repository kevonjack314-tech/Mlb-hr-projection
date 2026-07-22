"""Synthetic but realistic slate generator.

This module guarantees the app is fully functional with zero network access. It
builds a believable slate of games, probable pitchers, and hitter Statcast
profiles. Player metrics are generated deterministically from a name seed and a
"power tier" so the same player always gets the same profile within a date, and
stars (tier 5) produce elite barrel/EV numbers while contact-only hitters (tier 1)
produce low-power profiles.

When live data is available (see sources.py) the real numbers replace these. The
names below are illustrative current-era hitters used to make the demo readable;
they are not guaranteed to reflect today's exact active rosters.
"""

from __future__ import annotations

import hashlib
import random
from datetime import date

import numpy as np
import pandas as pd

from .lineup import demo_spot_for_index

# team_abbr -> (probable pitcher name, throws L/R, pitcher quality tier 1-5,
#               groundball-or-flyball lean: 'GB'/'NEU'/'FB')
# Hitters: (name, bats L/R/S, power tier 1-5, primary position)
TEAMS = {
    "NYY": {
        "name": "New York Yankees",
        "hitters": [
            ("Aaron Judge", "R", 5, "RF"), ("Juan Soto", "L", 5, "RF"),
            ("Giancarlo Stanton", "R", 5, "DH"), ("Anthony Rizzo", "L", 3, "1B"),
            ("Gleyber Torres", "R", 3, "2B"), ("Jazz Chisholm Jr.", "L", 4, "3B"),
            ("Austin Wells", "L", 3, "C"), ("Anthony Volpe", "R", 3, "SS"),
            ("Alex Verdugo", "L", 2, "LF"), ("Trent Grisham", "L", 3, "CF"),
        ],
    },
    "BOS": {
        "name": "Boston Red Sox",
        "hitters": [
            ("Rafael Devers", "L", 5, "3B"), ("Triston Casas", "L", 4, "1B"),
            ("Tyler O'Neill", "R", 4, "LF"), ("Jarren Duran", "L", 3, "CF"),
            ("Wilyer Abreu", "L", 3, "RF"), ("Trevor Story", "R", 3, "SS"),
            ("Masataka Yoshida", "L", 2, "DH"), ("Connor Wong", "R", 2, "C"),
            ("Ceddanne Rafaela", "R", 3, "2B"), ("Vaughn Grissom", "R", 2, "2B"),
        ],
    },
    "TOR": {
        "name": "Toronto Blue Jays",
        "hitters": [
            ("Vladimir Guerrero Jr.", "R", 5, "1B"), ("Bo Bichette", "R", 3, "SS"),
            ("George Springer", "R", 3, "RF"), ("Daulton Varsho", "L", 3, "CF"),
            ("Alejandro Kirk", "R", 2, "C"), ("Justin Turner", "R", 2, "DH"),
            ("Ernie Clement", "R", 2, "3B"), ("Davis Schneider", "R", 3, "2B"),
            ("Addison Barger", "L", 3, "RF"), ("Spencer Horwitz", "L", 2, "1B"),
        ],
    },
    "BAL": {
        "name": "Baltimore Orioles",
        "hitters": [
            ("Gunnar Henderson", "L", 5, "SS"), ("Adley Rutschman", "S", 4, "C"),
            ("Anthony Santander", "S", 4, "RF"), ("Ryan Mountcastle", "R", 3, "1B"),
            ("Cedric Mullins", "L", 3, "CF"), ("Jordan Westburg", "R", 3, "3B"),
            ("Colton Cowser", "L", 3, "LF"), ("Ryan O'Hearn", "L", 3, "DH"),
            ("Jackson Holliday", "L", 3, "2B"), ("Heston Kjerstad", "L", 3, "DH"),
        ],
    },
    "TB": {
        "name": "Tampa Bay Rays",
        "hitters": [
            ("Yandy Diaz", "R", 3, "1B"), ("Brandon Lowe", "L", 4, "2B"),
            ("Isaac Paredes", "R", 4, "3B"), ("Randy Arozarena", "R", 4, "LF"),
            ("Josh Lowe", "L", 3, "RF"), ("Christopher Morel", "R", 4, "DH"),
            ("Jonny DeLuca", "R", 2, "CF"), ("Ben Rortvedt", "L", 1, "C"),
            ("Jose Caballero", "R", 2, "SS"), ("Curtis Mead", "R", 2, "DH"),
        ],
    },
    "CLE": {
        "name": "Cleveland Guardians",
        "hitters": [
            ("Jose Ramirez", "S", 5, "3B"), ("Josh Naylor", "L", 4, "1B"),
            ("Steven Kwan", "L", 2, "LF"), ("David Fry", "R", 3, "C"),
            ("Lane Thomas", "R", 3, "RF"), ("Andres Gimenez", "L", 2, "2B"),
            ("Bo Naylor", "L", 3, "C"), ("Brayan Rocchio", "S", 2, "SS"),
            ("Will Brennan", "L", 2, "CF"), ("Kyle Manzardo", "L", 3, "DH"),
        ],
    },
    "DET": {
        "name": "Detroit Tigers",
        "hitters": [
            ("Riley Greene", "L", 4, "CF"), ("Kerry Carpenter", "L", 4, "RF"),
            ("Spencer Torkelson", "R", 4, "1B"), ("Colt Keith", "L", 3, "2B"),
            ("Matt Vierling", "R", 3, "3B"), ("Jake Rogers", "R", 3, "C"),
            ("Parker Meadows", "L", 3, "CF"), ("Wenceel Perez", "S", 2, "DH"),
            ("Trey Sweeney", "L", 2, "SS"), ("Zach McKinstry", "L", 2, "3B"),
        ],
    },
    "KC": {
        "name": "Kansas City Royals",
        "hitters": [
            ("Bobby Witt Jr.", "R", 5, "SS"), ("Salvador Perez", "R", 4, "C"),
            ("Vinnie Pasquantino", "L", 3, "1B"), ("Maikel Garcia", "R", 2, "3B"),
            ("MJ Melendez", "L", 3, "LF"), ("Hunter Renfroe", "R", 3, "RF"),
            ("Michael Massey", "L", 2, "2B"), ("Kyle Isbel", "L", 2, "CF"),
            ("Tommy Pham", "R", 2, "DH"), ("Yuli Gurriel", "R", 2, "1B"),
        ],
    },
    "MIN": {
        "name": "Minnesota Twins",
        "hitters": [
            ("Carlos Correa", "R", 4, "SS"), ("Royce Lewis", "R", 4, "3B"),
            ("Byron Buxton", "R", 5, "CF"), ("Matt Wallner", "L", 4, "RF"),
            ("Trevor Larnach", "L", 3, "LF"), ("Max Kepler", "L", 3, "RF"),
            ("Ryan Jeffers", "R", 3, "C"), ("Edouard Julien", "L", 3, "2B"),
            ("Jose Miranda", "R", 3, "1B"), ("Willi Castro", "S", 2, "2B"),
        ],
    },
    "CWS": {
        "name": "Chicago White Sox",
        "hitters": [
            ("Luis Robert Jr.", "R", 4, "CF"), ("Andrew Vaughn", "R", 3, "1B"),
            ("Eloy Jimenez", "R", 3, "DH"), ("Andrew Benintendi", "L", 2, "LF"),
            ("Gavin Sheets", "L", 3, "RF"), ("Paul DeJong", "R", 3, "SS"),
            ("Korey Lee", "R", 2, "C"), ("Lenyn Sosa", "R", 2, "2B"),
            ("Miguel Vargas", "R", 3, "3B"), ("Nicky Lopez", "L", 1, "2B"),
        ],
    },
    "HOU": {
        "name": "Houston Astros",
        "hitters": [
            ("Yordan Alvarez", "L", 5, "DH"), ("Kyle Tucker", "L", 5, "RF"),
            ("Jose Altuve", "R", 4, "2B"), ("Alex Bregman", "R", 4, "3B"),
            ("Yainer Diaz", "R", 3, "C"), ("Jeremy Pena", "R", 3, "SS"),
            ("Chas McCormick", "R", 3, "LF"), ("Jake Meyers", "R", 2, "CF"),
            ("Mauricio Dubon", "R", 2, "2B"), ("Jon Singleton", "L", 2, "1B"),
        ],
    },
    "SEA": {
        "name": "Seattle Mariners",
        "hitters": [
            ("Julio Rodriguez", "R", 5, "CF"), ("Cal Raleigh", "S", 4, "C"),
            ("Mitch Garver", "R", 3, "DH"), ("Randy Arozarena", "R", 4, "LF"),
            ("Luke Raley", "L", 3, "RF"), ("Josh Rojas", "L", 2, "3B"),
            ("J.P. Crawford", "L", 2, "SS"), ("Dylan Moore", "R", 2, "2B"),
            ("Victor Robles", "R", 2, "RF"), ("Dominic Canzone", "L", 2, "LF"),
        ],
    },
    "TEX": {
        "name": "Texas Rangers",
        "hitters": [
            ("Corey Seager", "L", 5, "SS"), ("Marcus Semien", "R", 4, "2B"),
            ("Adolis Garcia", "R", 4, "RF"), ("Wyatt Langford", "R", 4, "LF"),
            ("Nathaniel Lowe", "L", 3, "1B"), ("Josh Jung", "R", 3, "3B"),
            ("Jonah Heim", "S", 3, "C"), ("Evan Carter", "L", 3, "CF"),
            ("Leody Taveras", "S", 2, "CF"), ("Ezequiel Duran", "R", 2, "3B"),
        ],
    },
    "LAA": {
        "name": "Los Angeles Angels",
        "hitters": [
            ("Mike Trout", "R", 5, "CF"), ("Taylor Ward", "R", 4, "LF"),
            ("Anthony Rendon", "R", 2, "3B"), ("Logan O'Hoppe", "R", 3, "C"),
            ("Nolan Schanuel", "L", 2, "1B"), ("Zach Neto", "R", 3, "SS"),
            ("Jo Adell", "R", 4, "RF"), ("Mickey Moniak", "L", 3, "CF"),
            ("Luis Rengifo", "S", 2, "2B"), ("Brandon Drury", "R", 3, "DH"),
        ],
    },
    "OAK": {
        "name": "Athletics",
        "hitters": [
            ("Brent Rooker", "R", 4, "DH"), ("Lawrence Butler", "L", 4, "RF"),
            ("Tyler Soderstrom", "L", 3, "1B"), ("Shea Langeliers", "R", 3, "C"),
            ("JJ Bleday", "L", 3, "CF"), ("Seth Brown", "L", 3, "LF"),
            ("Zack Gelof", "R", 3, "2B"), ("Max Schuemann", "R", 1, "SS"),
            ("Miguel Andujar", "R", 2, "LF"), ("Jacob Wilson", "R", 2, "SS"),
        ],
    },
    "ATL": {
        "name": "Atlanta Braves",
        "hitters": [
            ("Ronald Acuna Jr.", "R", 5, "RF"), ("Matt Olson", "L", 5, "1B"),
            ("Austin Riley", "R", 4, "3B"), ("Marcell Ozuna", "R", 5, "DH"),
            ("Ozzie Albies", "S", 3, "2B"), ("Sean Murphy", "R", 3, "C"),
            ("Michael Harris II", "L", 3, "CF"), ("Jarred Kelenic", "L", 3, "LF"),
            ("Orlando Arcia", "R", 2, "SS"), ("Travis d'Arnaud", "R", 3, "C"),
        ],
    },
    "PHI": {
        "name": "Philadelphia Phillies",
        "hitters": [
            ("Bryce Harper", "L", 5, "1B"), ("Kyle Schwarber", "L", 5, "DH"),
            ("Trea Turner", "R", 4, "SS"), ("Nick Castellanos", "R", 4, "RF"),
            ("J.T. Realmuto", "R", 3, "C"), ("Alec Bohm", "R", 3, "3B"),
            ("Bryson Stott", "L", 3, "2B"), ("Brandon Marsh", "L", 3, "LF"),
            ("Johan Rojas", "R", 2, "CF"), ("Cristian Pache", "R", 1, "CF"),
        ],
    },
    "NYM": {
        "name": "New York Mets",
        "hitters": [
            ("Pete Alonso", "R", 5, "1B"), ("Francisco Lindor", "S", 4, "SS"),
            ("Brandon Nimmo", "L", 3, "LF"), ("Mark Vientos", "R", 4, "3B"),
            ("Starling Marte", "R", 3, "RF"), ("Jeff McNeil", "L", 2, "2B"),
            ("Francisco Alvarez", "R", 4, "C"), ("Tyrone Taylor", "R", 2, "CF"),
            ("J.D. Martinez", "R", 4, "DH"), ("Jose Iglesias", "R", 2, "2B"),
        ],
    },
    "MIA": {
        "name": "Miami Marlins",
        "hitters": [
            ("Jazz Chisholm Jr.", "L", 4, "CF"), ("Jake Burger", "R", 4, "3B"),
            ("Jesus Sanchez", "L", 3, "RF"), ("Bryan De La Cruz", "R", 3, "LF"),
            ("Josh Bell", "S", 3, "1B"), ("Nick Fortes", "R", 2, "C"),
            ("Otto Lopez", "R", 2, "2B"), ("Xavier Edwards", "S", 1, "SS"),
            ("Connor Norby", "R", 3, "2B"), ("Dane Myers", "R", 2, "CF"),
        ],
    },
    "WSH": {
        "name": "Washington Nationals",
        "hitters": [
            ("CJ Abrams", "L", 3, "SS"), ("Dylan Crews", "R", 3, "RF"),
            ("James Wood", "L", 4, "LF"), ("Keibert Ruiz", "S", 2, "C"),
            ("Joey Meneses", "R", 2, "1B"), ("Luis Garcia Jr.", "L", 3, "2B"),
            ("Jacob Young", "R", 1, "CF"), ("Jose Tena", "L", 2, "3B"),
            ("Juan Yepez", "R", 3, "DH"), ("Andres Chaparro", "R", 3, "1B"),
        ],
    },
    "CHC": {
        "name": "Chicago Cubs",
        "hitters": [
            ("Seiya Suzuki", "R", 4, "RF"), ("Ian Happ", "S", 3, "LF"),
            ("Cody Bellinger", "L", 4, "CF"), ("Dansby Swanson", "R", 3, "SS"),
            ("Christopher Morel", "R", 4, "DH"), ("Nico Hoerner", "R", 2, "2B"),
            ("Michael Busch", "L", 3, "1B"), ("Isaac Paredes", "R", 4, "3B"),
            ("Pete Crow-Armstrong", "L", 3, "CF"), ("Miguel Amaya", "R", 2, "C"),
        ],
    },
    "MIL": {
        "name": "Milwaukee Brewers",
        "hitters": [
            ("Christian Yelich", "L", 4, "LF"), ("William Contreras", "R", 4, "C"),
            ("Willy Adames", "R", 4, "SS"), ("Rhys Hoskins", "R", 4, "1B"),
            ("Jackson Chourio", "R", 4, "RF"), ("Garrett Mitchell", "L", 3, "CF"),
            ("Sal Frelick", "L", 2, "RF"), ("Brice Turang", "L", 2, "2B"),
            ("Joey Ortiz", "R", 3, "3B"), ("Gary Sanchez", "R", 3, "C"),
        ],
    },
    "STL": {
        "name": "St. Louis Cardinals",
        "hitters": [
            ("Nolan Arenado", "R", 4, "3B"), ("Paul Goldschmidt", "R", 4, "1B"),
            ("Willson Contreras", "R", 4, "C"), ("Nolan Gorman", "L", 4, "2B"),
            ("Alec Burleson", "L", 3, "RF"), ("Lars Nootbaar", "L", 3, "LF"),
            ("Brendan Donovan", "L", 2, "2B"), ("Masyn Winn", "R", 3, "SS"),
            ("Jordan Walker", "R", 4, "RF"), ("Ivan Herrera", "R", 3, "C"),
        ],
    },
    "PIT": {
        "name": "Pittsburgh Pirates",
        "hitters": [
            ("Bryan Reynolds", "S", 4, "LF"), ("Oneil Cruz", "L", 5, "SS"),
            ("Ke'Bryan Hayes", "R", 2, "3B"), ("Andrew McCutchen", "R", 3, "DH"),
            ("Jack Suwinski", "L", 3, "CF"), ("Rowdy Tellez", "L", 3, "1B"),
            ("Joey Bart", "R", 3, "C"), ("Isiah Kiner-Falefa", "R", 2, "2B"),
            ("Henry Davis", "R", 3, "RF"), ("Nick Gonzales", "R", 2, "2B"),
        ],
    },
    "CIN": {
        "name": "Cincinnati Reds",
        "hitters": [
            ("Elly De La Cruz", "S", 5, "SS"), ("Spencer Steer", "R", 4, "1B"),
            ("Jonathan India", "R", 3, "2B"), ("Jeimer Candelario", "S", 3, "3B"),
            ("Tyler Stephenson", "R", 3, "C"), ("Will Benson", "L", 3, "RF"),
            ("TJ Friedl", "L", 2, "CF"), ("Jake Fraley", "L", 3, "RF"),
            ("Santiago Espinal", "R", 1, "2B"), ("Stuart Fairchild", "R", 2, "CF"),
        ],
    },
    "LAD": {
        "name": "Los Angeles Dodgers",
        "hitters": [
            ("Shohei Ohtani", "L", 5, "DH"), ("Mookie Betts", "R", 5, "RF"),
            ("Freddie Freeman", "L", 5, "1B"), ("Teoscar Hernandez", "R", 4, "LF"),
            ("Will Smith", "R", 4, "C"), ("Max Muncy", "L", 4, "3B"),
            ("Gavin Lux", "L", 2, "2B"), ("Tommy Edman", "S", 2, "CF"),
            ("Andy Pages", "R", 3, "CF"), ("Chris Taylor", "R", 2, "LF"),
        ],
    },
    "SD": {
        "name": "San Diego Padres",
        "hitters": [
            ("Fernando Tatis Jr.", "R", 5, "RF"), ("Manny Machado", "R", 4, "3B"),
            ("Xander Bogaerts", "R", 3, "2B"), ("Jake Cronenworth", "L", 3, "1B"),
            ("Jackson Merrill", "L", 4, "CF"), ("Luis Arraez", "L", 1, "2B"),
            ("Kyle Higashioka", "R", 3, "C"), ("Jurickson Profar", "S", 3, "LF"),
            ("Donovan Solano", "R", 2, "1B"), ("David Peralta", "L", 2, "LF"),
        ],
    },
    "SF": {
        "name": "San Francisco Giants",
        "hitters": [
            ("Matt Chapman", "R", 4, "3B"), ("Heliot Ramos", "R", 3, "LF"),
            ("Jung Hoo Lee", "L", 2, "CF"), ("Michael Conforto", "L", 3, "RF"),
            ("Wilmer Flores", "R", 3, "1B"), ("Tyler Fitzgerald", "R", 3, "SS"),
            ("Patrick Bailey", "S", 2, "C"), ("Mike Yastrzemski", "L", 3, "RF"),
            ("LaMonte Wade Jr.", "L", 2, "1B"), ("Thairo Estrada", "R", 2, "2B"),
        ],
    },
    "ARI": {
        "name": "Arizona Diamondbacks",
        "hitters": [
            ("Ketel Marte", "S", 4, "2B"), ("Corbin Carroll", "L", 4, "RF"),
            ("Christian Walker", "R", 4, "1B"), ("Eugenio Suarez", "R", 4, "3B"),
            ("Lourdes Gurriel Jr.", "R", 3, "LF"), ("Joc Pederson", "L", 4, "DH"),
            ("Gabriel Moreno", "R", 2, "C"), ("Geraldo Perdomo", "S", 2, "SS"),
            ("Jake McCarthy", "L", 2, "CF"), ("Pavin Smith", "L", 2, "1B"),
        ],
    },
    "COL": {
        "name": "Colorado Rockies",
        "hitters": [
            ("Ryan McMahon", "L", 3, "3B"), ("Brenton Doyle", "R", 3, "CF"),
            ("Ezequiel Tovar", "R", 3, "SS"), ("Charlie Blackmon", "L", 2, "RF"),
            ("Elias Diaz", "R", 2, "C"), ("Michael Toglia", "S", 4, "1B"),
            ("Nolan Jones", "L", 4, "RF"), ("Jake Cave", "L", 2, "LF"),
            ("Hunter Goodman", "R", 3, "C"), ("Sam Hilliard", "L", 3, "LF"),
        ],
    },
}

# A pool of probable-pitcher archetypes: (throws, quality tier 1-5, batted-ball lean)
# Flyball pitchers (FB) surrender more HR; groundball (GB) suppress them.
_PITCHER_NAMES = [
    "Logan Webb", "Zack Wheeler", "Tyler Glasnow", "Corbin Burnes", "Pablo Lopez",
    "Tarik Skubal", "Cole Ragans", "Seth Lugo", "Sonny Gray", "Aaron Nola",
    "Framber Valdez", "Hunter Greene", "Garrett Crochet", "Bailey Ober", "Reese Olson",
    "Bryce Miller", "Jared Jones", "Spencer Schwellenbach", "Kutter Crawford", "Ranger Suarez",
    "Nathan Eovaldi", "Jack Flaherty", "Yusei Kikuchi", "Michael King", "Dylan Cease",
    "Luis Castillo", "Freddy Peralta", "Sean Manaea", "Jose Berrios", "Kevin Gausman",
]


def _seed_int(*parts: str) -> int:
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16)


def _hitter_profile(name: str, bats: str, tier: int, slate_seed: str) -> dict:
    """Generate a deterministic, realistic Statcast profile for a hitter."""
    rng = random.Random(_seed_int(name, slate_seed, "hitter"))

    # Tier-anchored means (MLB-realistic ranges).
    barrel = {1: 4.5, 2: 7.0, 3: 9.5, 4: 13.0, 5: 17.0}[tier] + rng.uniform(-1.5, 1.5)
    hard_hit = {1: 33, 2: 38, 3: 43, 4: 48, 5: 53}[tier] + rng.uniform(-3, 3)
    avg_ev = {1: 86.5, 2: 88.0, 3: 89.5, 4: 91.0, 5: 92.5}[tier] + rng.uniform(-1.0, 1.0)
    max_ev = {1: 104, 2: 107, 3: 110, 4: 113, 5: 116}[tier] + rng.uniform(-2, 2)
    la = rng.uniform(8, 19)  # sweet-spot launch ~ 8-19 deg
    xwoba = {1: 0.300, 2: 0.320, 3: 0.340, 4: 0.365, 5: 0.395}[tier] + rng.uniform(-0.02, 0.02)
    k_pct = {1: 18, 2: 21, 3: 23, 4: 25, 5: 26}[tier] + rng.uniform(-3, 3)
    # Swing-and-miss (whiff%) = swings that miss / swings. Power hitters tend to
    # swing-and-miss more; it tracks K% but is a distinct, swing-level signal.
    whiff_pct = {1: 18.0, 2: 21.0, 3: 24.0, 4: 27.0, 5: 30.0}[tier] + rng.uniform(-3.5, 3.5)
    whiff_pct = max(10.0, min(40.0, whiff_pct + (k_pct - {1: 18, 2: 21, 3: 23, 4: 25, 5: 26}[tier]) * 0.5))
    # Plate discipline: chase (O-Swing%) tracks whiff; zone-contact (Z-Contact%)
    # is the inverse. Fly-ball% rises with power/launch and is a real HR driver.
    chase_pct = max(16.0, min(40.0, 22.0 + (whiff_pct - 24.0) * 0.6 + rng.uniform(-3.0, 3.0)))
    zone_contact_pct = max(72.0, min(96.0, 100.0 - whiff_pct * 0.55 + rng.uniform(-3.0, 3.0)))
    fb_pct = {1: 30.0, 2: 33.0, 3: 36.0, 4: 39.0, 5: 42.0}[tier] + (la - 13.0) * 0.6 + rng.uniform(-4.0, 4.0)
    fb_pct = max(20.0, min(50.0, fb_pct))
    # Batted-ball distribution: GB% + LD% + FB% ≈ 100. Line drives ~21% league.
    ld_pct = max(14.0, min(28.0, 21.0 + rng.uniform(-3.0, 3.0)))
    gb_pct = max(28.0, min(58.0, 100.0 - fb_pct - ld_pct))
    # Pull% (~40% league) rises with power; HR/FB is the fly-ball -> HR conversion
    # rate (~12-13% league), strongly tied to raw power tier.
    pull_pct = max(30.0, min(52.0, 38.0 + (tier - 3) * 2.0 + rng.uniform(-4.0, 4.0)))
    hr_fb = {1: 6.0, 2: 9.0, 3: 12.0, 4: 16.0, 5: 20.0}[tier] + (barrel - {1: 4.5, 2: 7.0, 3: 9.5, 4: 13.0, 5: 17.0}[tier]) * 0.4 + rng.uniform(-2.0, 2.0)
    hr_fb = max(3.0, min(28.0, hr_fb))
    # Statcast expected power: xISO (= xSLG - xBA) and xSLG, contact-quality based.
    xiso = {1: 0.110, 2: 0.140, 3: 0.175, 4: 0.215, 5: 0.255}[tier] + rng.uniform(-0.025, 0.025)
    xiso = max(0.070, min(0.300, xiso))
    xba_est = max(0.210, min(0.300, 0.250 + rng.uniform(-0.03, 0.03)))
    xslg = round(xba_est + xiso, 3)
    # Real ISO (slugging - avg) tracks xISO; Sweet-Spot% (8-32 deg launch share).
    iso = round(max(0.080, min(0.320, xiso + rng.uniform(-0.03, 0.03))), 3)
    sweet_spot_pct = round(max(24.0, min(42.0, 30.0 + (la - 13.0) * 0.4 + rng.uniform(-3.0, 3.0))), 1)
    # Barrels per plate appearance (%): barrel rate scaled by how often the bat
    # puts a ball in play (BBE/PA ~ 0.6). A premier season-long HR predictor.
    brl_pa = round(max(1.5, min(13.0, barrel * rng.uniform(0.55, 0.68))), 1)
    # Sprint speed (ft/s) — athletic context (not a HR driver; shown for color).
    sprint_speed = round(max(23.0, min(30.5, rng.gauss(27.0, 1.3))), 1)
    # Performance vs pitch families (wOBA-like). Most hitters handle fastballs
    # best; breaking/offspeed separate the disciplined from the exploitable.
    vs_fb = round(max(0.250, min(0.440, xwoba + rng.uniform(-0.010, 0.055))), 3)
    vs_br = round(max(0.225, min(0.420, xwoba + rng.uniform(-0.060, 0.020))), 3)
    vs_os = round(max(0.225, min(0.420, xwoba + rng.uniform(-0.050, 0.030))), 3)

    # Season HR/PA anchored to tier; PA accrued over the season.
    pa = rng.randint(180, 480)
    hr_per_pa = {1: 0.018, 2: 0.028, 3: 0.038, 4: 0.052, 5: 0.068}[tier] + rng.uniform(-0.006, 0.006)
    hr_per_pa = max(0.005, hr_per_pa)
    season_hr = max(0, int(round(hr_per_pa * pa)))

    # Recent form: a "heat" factor that shifts recent HR rate vs season.
    # Capped to stay realistic — even scorching stretches rarely exceed ~0.10 HR/PA.
    heat = rng.gauss(1.0, 0.40)
    heat = max(0.25, min(2.0, heat))
    hr7 = min(0.11, max(0.0, hr_per_pa * heat * rng.uniform(0.6, 1.5)))
    hr15 = min(0.10, max(0.0, hr_per_pa * heat * rng.uniform(0.7, 1.35)))
    hr30 = min(0.095, max(0.0, hr_per_pa * (0.5 * heat + 0.5) * rng.uniform(0.8, 1.2)))

    return {
        "barrel_pct": round(barrel, 1),
        "hard_hit_pct": round(hard_hit, 1),
        "avg_ev": round(avg_ev, 1),
        "max_ev": round(max_ev, 1),
        "launch_angle": round(la, 1),
        "xwoba": round(xwoba, 3),
        "k_pct": round(max(10.0, k_pct), 1),
        "whiff_pct": round(whiff_pct, 1),
        "contact_pct": round(100.0 - whiff_pct, 1),
        "chase_pct": round(chase_pct, 1),
        "zone_contact_pct": round(zone_contact_pct, 1),
        "fb_pct": round(fb_pct, 1),
        "gb_pct": round(gb_pct, 1),
        "ld_pct": round(ld_pct, 1),
        "pull_pct": round(pull_pct, 1),
        "hr_fb": round(hr_fb, 1),
        "xiso": round(xiso, 3),
        "xslg": xslg,
        "iso": iso,
        "sweet_spot_pct": sweet_spot_pct,
        "brl_pa": brl_pa,
        "sprint_speed": sprint_speed,
        "vs_fb": vs_fb,
        "vs_br": vs_br,
        "vs_os": vs_os,
        "pa": pa,
        "season_hr": season_hr,
        "hr_per_pa": round(hr_per_pa, 4),
        "hr_rate_7": round(hr7, 4),
        "hr_rate_15": round(hr15, 4),
        "hr_rate_30": round(hr30, 4),
        "power_tier": tier,
    }


def _pitcher_profile(team_abbr: str, slate_seed: str) -> dict:
    rng = random.Random(_seed_int(team_abbr, slate_seed, "pitcher"))
    name = _PITCHER_NAMES[_seed_int(team_abbr, slate_seed) % len(_PITCHER_NAMES)]
    throws = rng.choice(["L", "R", "R", "R"])  # ~25% LHP
    tier = rng.randint(1, 5)
    lean = rng.choices(["GB", "NEU", "FB"], weights=[0.3, 0.4, 0.3])[0]
    # HR/9 inversely related to quality, amplified by flyball lean.
    base_hr9 = {1: 1.7, 2: 1.4, 3: 1.2, 4: 1.0, 5: 0.85}[tier]
    lean_adj = {"GB": -0.25, "NEU": 0.0, "FB": 0.30}[lean]
    hr9 = max(0.4, base_hr9 + lean_adj + rng.uniform(-0.15, 0.15))
    barrel_allowed = {1: 11.0, 2: 9.5, 3: 8.0, 4: 6.8, 5: 5.5}[tier] + rng.uniform(-1, 1)
    # Meatball (middle-middle) rate: worse pitchers groove more; league ~5%.
    meatball = {1: 6.4, 2: 5.6, 3: 5.0, 4: 4.4, 5: 3.8}[tier] + rng.uniform(-0.5, 0.5)
    # Fastball velo: season baseline, with a mostly-flat last-start delta and
    # an occasional dead-arm dip (fatigue) that raises HR risk.
    velo_base = round(rng.uniform(91.0, 97.0), 1)
    velo_delta = round(rng.choices([rng.uniform(-0.4, 0.4), rng.uniform(-2.2, -1.0)],
                                   weights=[0.8, 0.2])[0], 1)
    # 3rd-time-through wOBA lift: league ~+0.02-0.03, worse arms fade harder.
    tto_penalty = round({1: 0.045, 2: 0.035, 3: 0.028, 4: 0.020, 5: 0.012}[tier]
                        + rng.uniform(-0.010, 0.010), 3)
    gb_pct = {"GB": 52, "NEU": 44, "FB": 36}[lean] + rng.uniform(-3, 3)
    fb_pct = {"GB": 28, "NEU": 36, "FB": 44}[lean] + rng.uniform(-3, 3)
    # Predictable-FB tendency in hitter's counts (league ~55%); auto-fastball
    # arms sit higher and let good FB hitters cheat.
    hitter_count_fb = round(rng.uniform(42.0, 72.0), 1)
    # Pitch mix (% fastball / breaking / offspeed), summing to exactly 100.
    pmix_fb = rng.randint(44, 64)
    pmix_br = min(rng.randint(20, 38), 95 - pmix_fb)
    pmix_os = 100 - pmix_fb - pmix_br
    return {
        "pitcher_name": name,
        "pitcher_throws": throws,
        "pitcher_tier": tier,
        "pitcher_lean": lean,
        "pitcher_hr9": round(hr9, 2),
        "pitcher_barrel_pct_allowed": round(barrel_allowed, 1),
        "sp_meatball_pct": round(meatball, 2),
        "sp_velo_base": velo_base,
        "sp_velo_last": round(velo_base + velo_delta, 1),
        "sp_velo_delta": velo_delta,
        "sp_tto_penalty": tto_penalty,
        "sp_hitter_count_fb": hitter_count_fb,
        "pitcher_gb_pct": round(gb_pct, 1),
        "pitcher_fb_pct": round(fb_pct, 1),
        "pitcher_mix_fb": pmix_fb,
        "pitcher_mix_br": pmix_br,
        "pitcher_mix_os": pmix_os,
    }


# Rotating game pairings so different dates produce different (deterministic) slates.
def _matchups_for_date(d: date) -> list[tuple[str, str]]:
    abbrs = list(TEAMS.keys())
    rng = random.Random(_seed_int(d.isoformat(), "matchups"))
    rng.shuffle(abbrs)
    # Pick an even number of teams (10-16 games).
    n_games = rng.randint(10, len(abbrs) // 2)
    selected = abbrs[: n_games * 2]
    pairs = []
    for i in range(0, len(selected), 2):
        away, home = selected[i], selected[i + 1]
        pairs.append((away, home))
    return pairs


def build_demo_slate(game_date: date) -> pd.DataFrame:
    """Build a full per-hitter slate DataFrame for the given date (synthetic)."""
    slate_seed = game_date.isoformat()
    rng = random.Random(_seed_int(slate_seed, "weather"))
    rows = []

    for away, home in _matchups_for_date(game_date):
        home_pitcher = _pitcher_profile(home, slate_seed)  # faced by AWAY hitters
        away_pitcher = _pitcher_profile(away, slate_seed)  # faced by HOME hitters

        # Synthetic weather per game (the park is the home park).
        temp = round(rng.uniform(55, 95), 0)
        wind_speed = round(rng.uniform(0, 18), 0)
        wind_dir = round(rng.uniform(0, 359), 0)
        humidity = round(rng.uniform(30, 80), 0)

        for side, team, opp, opp_pitcher in (
            ("away", away, home, home_pitcher),
            ("home", home, away, away_pitcher),
        ):
            team_info = TEAMS[team]
            for idx, (name, bats, tier, pos) in enumerate(team_info["hitters"]):
                prof = _hitter_profile(name, bats, tier, slate_seed)
                row = {
                    "player": name,
                    "team": team,
                    "team_name": team_info["name"],
                    "bats": bats,
                    "position": pos,
                    "lineup_spot": demo_spot_for_index(idx),
                    "opponent": opp,
                    "home_team": home,
                    "is_home": side == "home",
                    "game": f"{away} @ {home}",
                    "data_quality": "modeled",
                    "temp_f": temp,
                    "wind_mph": wind_speed,
                    "wind_dir_deg": wind_dir,
                    "humidity_pct": humidity,
                }
                row.update(prof)
                row.update(opp_pitcher)
                rows.append(row)

    df = pd.DataFrame(rows)
    return df
