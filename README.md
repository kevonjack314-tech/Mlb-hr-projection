# ⚾ MLB Home Run Projection Tool

A detailed, transparent dashboard that ranks hitters by **home-run upside** for any
MLB date. It blends Statcast-style batted-ball quality, recent form, the opposing
pitcher matchup, ballpark factors, and live weather into a composite **HR Score**,
a per-game **HR probability**, and three specialized rankings — **Longshots**,
**Consistent HR Hitters**, and **Sneaky HR Chances**.

> **For research & entertainment only.** HR probabilities are model estimates, not
> guarantees or betting advice.

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (default `http://localhost:8501`).

- **Date picker** in the sidebar — defaults to today, works for any date.
- **"Try live data"** toggle — pulls the real schedule, probable pitchers, rosters
  and weather. If the network is unavailable, it falls back to a deterministic
  **synthetic demo slate** so the app always works.

---

## What the app shows

### Tabs / pages
1. **🚀 Best Longshots** — high-upside, boom-or-bust bats ranked by an
   *explosiveness* score (max EV + barrel% + favorable park/weather). Good for
   +EV HR props and DFS tournaments. Includes vig-free fair odds.
2. **🎯 Consistent HR Hitters** — reliable, high-floor power: steady hard
   contact, low strikeouts, season HR pace, EV & xwOBA, weighted by sample size.
3. **🕵️ Sneaky HR Chances** — under-the-radar value: favorable matchup vs a
   hittable arm, hidden park/weather edge, or a hot streak not yet reflected in
   the season line. Each player shows *why* they're sneaky.
4. **📊 All Combined + Best Metrics** — the master table for the full slate with
   every column, sortable/filterable, plus a Top-20 overall leaderboard.

### Features
- **Leaderboard cards** with key metrics and a one-line rationale per player.
- **Bar charts** of the top bats by each score.
- **Sort & filter** by team, position, handedness, platoon advantage, and minimum
  plate appearances.
- **CSV export** on every view.
- **Tooltips & a metric glossary** in the sidebar; **methodology** with the live
  weight tables (pulled straight from the code so docs never drift).

---

## Data sources

| Layer | Live source | Offline fallback |
|------|-------------|------------------|
| Schedule, probable pitchers, rosters/lineups | **MLB StatsAPI** (`statsapi.mlb.com`, no key) | Synthetic slate (`src/demo.py`) |
| Weather (temp, wind speed/direction, humidity) | **Open-Meteo** forecast at park lat/lon (no key) | Synthetic / neutral |
| Park factors & dimensions | Bundled `data/park_factors.csv` (Statcast/ESPN-calibrated) | same (bundled) |
| Hitter Statcast (barrel%, EV, xwOBA…) | Hook for `pybaseball` Statcast leaderboards (stub) | Deterministic modeled profiles |

The live path is fully **defensive**: any failed fetch downgrades just that piece
(or the whole slate) to the synthetic path and records a note in the **Data
provenance** panel, so you always know exactly where each number came from.

> **Wiring real Statcast metrics:** implement `src/sources.py::_statcast_lookup`
> to return a season profile per player (e.g. via `pybaseball`'s
> `statcast_batter_exitvelo_barrels`). Nothing else in the pipeline needs to change.

---

## Methodology (show your work)

Every raw input is normalized to a **0–100 sub-score** against *fixed league
reference ranges* (≈5th–95th percentile of qualified hitters), so scores are
comparable **across dates**, not just within one slate. The reference ranges live
in `src/model.py::REF`.

### 1. Composite **HR Score** (0–100)

A weighted blend of five sub-scores (`HR_SCORE_WEIGHTS`):

| Component | Weight | What it captures |
|-----------|:------:|------------------|
| Power quality (Statcast) | **0.34** | Barrel%, Hard-Hit%, xwOBA, Max EV, Avg EV |
| Season HR rate | **0.16** | Season-long HR / PA |
| Recent form | **0.16** | Last 7 / 15 / 30-day HR rate |
| Matchup | **0.16** | Opposing pitcher + platoon edge |
| Environment | **0.18** | Park factor + weather |

The **power-quality** sub-score is itself a weighted blend
(`POWER_QUALITY_WEIGHTS`): Barrel% **0.35**, Hard-Hit% **0.20**, xwOBA **0.20**,
Max EV **0.15**, Avg EV **0.10**. Barrel rate gets the most weight because it is
the single best public predictor of home-run output.

**Recent form** weights the windows (`RECENT_FORM_WEIGHTS`): 7-day **0.50**,
15-day **0.30**, 30-day **0.20** — the hottest, most recent signal counts most.

### 2. **HR probability** (≥1 HR in the game)

We build an adjusted per-PA HR rate, then compound it over expected plate
appearances:

```
quality_implied = LEAGUE_HR_PER_PA × (0.5 + power_quality/100)      # 0.5×–1.5× league
base_rate       = 0.55·(season HR/PA) + 0.25·(recent HR/PA, capped) + 0.20·quality_implied
p_adj           = clip(base_rate × matchup_mult × env_mult, 0.002, 0.085)
P(≥1 HR)        = 1 − (1 − p_adj)^PA          # PA ≈ 4.1
xHR             = p_adj × PA
fair_odds       = vig-free American odds from P(≥1 HR)
```

`LEAGUE_HR_PER_PA = 0.034`. The recent rate is capped so a small-sample heater
can't blow up the estimate; `p_adj` is capped at 0.085/PA, which puts the very
best spots around a realistic **~30%** chance of a homer in the game and the
league-average bat near ~10–13%.

### 3. **Matchup multiplier**

From the probable pitcher: HR/9 allowed, barrel% allowed, ground-ball/fly-ball
lean (FB arms give up more HR), and the hitter's **platoon edge**. Switch hitters
are evaluated from the side they'll bat against that pitcher; a homer-prone fly-ball
pitcher faced with a platoon advantage is the juiciest matchup.

### 4. **Environment** — park + weather

`env_mult = park × wind × temperature × humidity`, each centered at 1.0:

- **Park factor** — handedness-aware (e.g. Yankee Stadium's short right porch
  inflates LHB HR; Fenway suppresses RHB HR). Bundled, Statcast/ESPN-calibrated.
- **Wind** — the wind vector is resolved against each park's home-plate→center-field
  bearing (`orientation_deg`). A wind blowing **out** adds carry (~1.5%/mph of the
  out component, with a small pull-side cross-wind credit); **in** knocks balls
  down. Roofed/closed games are neutral.
- **Temperature** — warmer, thinner air carries: ~1% per ~3.5 °F around a 70 °F
  baseline.
- **Humidity** — a small second-order carry effect (humid air is slightly less
  dense).

### 5. Specialized scores

- **Longshot** = `0.45·MaxEV + 0.25·Barrel + 0.20·Env + 0.10·Matchup`, then
  nudged by a **variance bonus** (higher K% = more boom-or-bust) and a **chalk
  penalty** (already-high-probability bats aren't true "longshots").
- **Consistency** = `0.28·HardHit + 0.22·Contact(low-K) + 0.20·SeasonHR +
  0.15·AvgEV + 0.15·xwOBA`, scaled by a sample-size confidence factor.
- **Sneaky** = `0.30·Matchup + 0.25·Env + 0.25·FormGap + 0.20·UnderRadar`, where
  *FormGap* rewards bats heating up beyond their season line and *UnderRadar*
  favors lower-profile hitters who still have real batted-ball pop.

---

## Assumptions & limitations

- **Expected PA** is a flat ~4.1 for every starter (lineup-spot adjustments are a
  natural next step).
- When a posted lineup isn't available yet, the tool uses the team's **active
  roster** (up to 13 position players), so some listed bats may not start.
- The bundled **player pool** in `demo.py` is illustrative current-era hitters for
  the offline demo; it is not guaranteed to match today's exact active rosters.
- **Park factors** are static seasonal estimates; refresh `data/park_factors.csv`
  annually.
- Statcast batted-ball metrics are **modeled** until you wire a live feed via
  `_statcast_lookup` (see above).

---

## Project layout

```
.
├── app.py                  # Streamlit UI: tabs, filters, charts, CSV export
├── requirements.txt
├── data/
│   └── park_factors.csv    # 30-park HR factors, dimensions, lat/lon, roof
├── src/
│   ├── parks.py            # park / wind / temp / humidity multipliers
│   ├── model.py            # composite scoring + probability (weights as constants)
│   ├── demo.py             # deterministic synthetic slate (offline fallback)
│   └── sources.py          # live MLB StatsAPI + Open-Meteo, defensive fallback
└── .streamlit/config.toml  # dark theme
```

## Disclaimer

This tool is for educational and entertainment purposes. Projections are
probabilistic estimates derived from public data and modeling assumptions; they
are **not** financial or betting advice.
