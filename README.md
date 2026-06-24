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

## Deploy

The repo is deploy-ready for any of these:

- **Streamlit Community Cloud** (easiest): point [share.streamlit.io](https://share.streamlit.io)
  at this repo, branch `main`, main file `app.py`. Auto-redeploys on push.
- **Docker** (Render / Railway / Fly.io / anywhere): a `Dockerfile` is included
  and honors `$PORT`.
  ```bash
  docker build -t mlb-hr .
  docker run -p 8501:8501 mlb-hr           # then open http://localhost:8501
  ```
- **Procfile** platforms (Render/Heroku-style) use the bundled `Procfile`.

For **live data** set the network egress / outbound access to reach the hosts in
the allowlist below, and optionally set `ODDS_API_KEY` for live HR odds. CI
(`.github/workflows/ci.yml`) runs the tests on every push/PR; a daily workflow
keeps the lineup HR log fresh.

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
5. **🧾 Lineups** — today's **batting orders (1–9)** for both teams in each game,
   next to the **opposing starter's HRs allowed by lineup spot over their last 5
   games** (an **SP HRs@Spot** column flags which order positions take that pitcher
   deep). Live from posted lineups + Statcast box scores; updates with the date.
6. **📈 HR Trends & Backtest** — analyzes **every home run over the trailing ~month**
   (configurable lookback): a browsable **stat sheet of previous HR hitters with
   their lineup spot** (filter by spot/team/player, CSV export), the *shared
   profile* of who went deep (how HR hitters out-index the field on barrel%, EV,
   max EV, park, platoon), **model calibration** (actual vs. predicted HR rate by
   decile), the hottest HR parks, **HRs-by-lineup-spot**, a **Profile Match %** for
   today's bats, and the **Top-5 list in each category**.
6. **🎰 Parlay Builder** — builds **1–5 leg HR parlays with roles, not names** (the
   ULX formula): an **⚓ Anchor** (highest-confidence bat, bats 3-5), **💰 Value**
   bats (underpriced profiles, 6-7), and **🚀 Deep-Space Longshots** (overlooked
   ceiling, 7-9), diversified across games, archetypes **and lineup spots** and
   graded on an 11-point checklist with a 🟢/🟡/🔴 light. Shows combined odds, model
   win %, and **EV**, plus a "build your own" mode. Strategies: ULX role-based,
   Safest, Best-value (edge), Boom.
7. **💎 Value Finder** — ranks the biggest **model-vs-book edges** (positive
   **Edge%** = +EV), filterable by role / probability / live-only, with a one-click
   value parlay.

### Lineup spot (with a recurring HR-by-spot log)

Every hitter carries a **lineup spot (1–9)** — live from the posted batting order
when available, else estimated. It feeds the model three ways:
- **Expected PA by spot** (top of order bats more, ~4.6 → 3.7 PA) folds directly
  into the per-game HR probability.
- **ULX role fit** — the parlay builder fits Anchors to 3-5, Value to 6-7,
  Longshots to 7-9, and checks "different lineup spots".
- **Recurring HR-by-spot log** — `data/lineup_hr_log.csv` accumulates one row per
  hitter-day (date, player, spot, HR), de-duped, growing as the date advances. A
  **seed log (~30 days)** ships in the repo so there's history out of the box, and
  it keeps growing via:
  - the **GitHub Actions schedule** (`.github/workflows/update-lineup-log.yml`),
    which runs daily, appends the prior day's HRs and commits the log (runners
    have open network, so it logs **real** games), and
  - `scripts/update_lineup_log.py` for manual/local runs (or the `/loop` skill).

  Per-player "HRs from today's spot" and the league HR-by-spot chart come from this
  log and nudge parlay selection. (In the bundled seed, HR/game peaks at spots
  3–4 — exactly where real middle-of-the-order power lives.)

### HR odds

Every hitter gets **Book Odds** to hit ≥1 HR: **live** from a sportsbook (The Odds
API, market `batter_home_runs`) when an `ODDS_API_KEY` env var is set and the host
is allowlisted, otherwise a **model-implied** market price (the vig-free fair price
shaded by a typical HR-prop hold). **Edge%** = model HR% − book-implied HR%
(positive = +EV); it powers the Best-value parlay strategy and the ticket EV.

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
| Hitter Statcast (barrel%, EV, max EV, LA) | **Baseball Savant** exit-velo & barrels leaderboard via `pybaseball` | Deterministic modeled profiles |
| Season counting stats (PA, HR, K%, xwOBA) | **FanGraphs** season batting via `pybaseball` | Modeled profiles |
| **Contact%** & swing-and-miss **Whiff%** (= 100 − Contact%) | **FanGraphs** plate discipline via `pybaseball` | Modeled profiles |
| **Chase%** (O-Swing%), **Zone-Contact%** (Z-Contact%), **Fly-Ball%** (FB%) | **FanGraphs** discipline + batted-ball via `pybaseball` | Modeled profiles |
| **Ground-Ball%**, **Line-Drive%**, **Pull%**, **HR/FB** | **FanGraphs** batted-ball via `pybaseball` | Modeled profiles |
| **xISO** (= xSLG − xBA) & **xSLG** (Statcast expected power) | **Baseball Savant** expected stats via `pybaseball` | Modeled profiles |
| **Barrel/PA%** (barrels per plate appearance) | **Baseball Savant** exit-velo/barrels via `pybaseball` | Modeled profiles |
| **Sprint speed** (ft/s, athletic context) | **Baseball Savant** sprint-speed via `pybaseball` | Modeled profiles |
| **vs-pitch-type** (vs FB / breaking / offspeed wOBA) | **Baseball Savant** Statcast date-range (woba by pitch family) | Modeled |
| **Pitcher peripherals** (HR/9, GB%, FB%, barrels allowed) | **FanGraphs** pitching via `pybaseball` | Modeled |
| **Pitcher pitch mix** (FB / breaking / offspeed usage) | **Baseball Savant** Statcast date-range (pitch_type usage) | Modeled |
| Recent form (7/15/30-day HR rate) | **Baseball Savant** Statcast date-range pull, aggregated by batter id | Modeled recent rates |

Real Statcast/FanGraphs metrics are merged onto the real slate **per player**: each
hitter starts from a modeled profile, then real season metrics and real recent-form
HR rates overlay wherever they resolve (matched by MLBAM id, then by normalized
name). So the slate is **real where live data exists and modeled only to fill
gaps** — and the **Data** column / provenance panel tells you exactly which rows
are `real` vs `modeled`, plus the overall coverage %.

The live path is fully **defensive**: any failed fetch downgrades just that piece
(or the whole slate) to the synthetic path and records a note in the **Data
provenance** panel, so you always know exactly where each number came from.

### ⚠️ Enabling live data (network egress allowlist)

`pip install -r requirements.txt` installs `pybaseball`, which is all that's
needed for live data **on an open network**. In a sandboxed environment (e.g.
Claude Code on the web) outbound access is governed by an **egress allowlist**;
until these hosts are added, every fetch returns `host_not_allowed` and the app
runs on the synthetic demo slate:

```
statsapi.mlb.com          # schedule, probable pitchers, rosters/lineups
baseballsavant.mlb.com    # Statcast batted-ball metrics + recent-form events
www.fangraphs.com         # season PA / HR / K% / xwOBA
api.open-meteo.com        # weather (temp, wind, humidity)
api.the-odds-api.com      # live HR prop odds (optional; needs ODDS_API_KEY)
```

Add them in your environment's **network egress settings** (see
<https://code.claude.com/docs/en/claude-code-on-the-web>). With the hosts allowed
and `pybaseball` installed, flip the sidebar **"Try live data"** toggle on and the
source badge changes to **🟢 LIVE (real Statcast)**.

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

**Expected HR & regression.** A season **xHR** is computed from batted-ball quality
(Barrels/PA, with a fly-ball term) × PA. The **HR − xHR** gap flags over- and
under-performers; a negative gap (fewer HR than the contact deserves) is a
positive-regression "due" signal that feeds the **Sneaky** score. *Sprint speed is
shown as athletic context only — it has no measurable effect on HR power, so xHR is
deliberately **not** sprint-adjusted.*

**Pitch-type matchup.** The hitter's wOBA vs fastballs / breaking / offspeed is
weighted by the probable pitcher's **pitch mix** into a pitch-arsenal edge that
folds into the matchup multiplier and score. Both sides are **live**: the batter
splits and the pitcher's pitch-type usage are aggregated from one shared Statcast
date-range pull (also reused for recent form), and the pitcher's HR/9, GB%, FB% and
barrels-allowed come from FanGraphs.

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

### 5. Trailing-month backtest & profile matching

The **HR Trends & Backtest** tab gathers every HR in a date window (default ~30
days ending on the selected date) with each hitter's profile and game context,
then:

- **Shared-profile analysis** — averages the HR hitters' metrics and reports the
  **lift vs. the slate baseline** (e.g. HR hitters carry higher barrel% and season
  HR/PA and skew toward platoon advantage).
- **Calibration / validation** — bins hitter-games by predicted game HR probability
  and plots **actual vs. predicted** HR rate per decile; a good model tracks the
  diagonal. (With live Statcast this is genuine out-of-sample validation; in the
  offline simulation, outcomes are drawn from the model's own probabilities, so it
  illustrates the pipeline rather than proving accuracy.)
- **Profile Match %** — a Gaussian-kernel similarity between each current hitter and
  the trailing-month **HR-hitter centroid** across barrel%, hard-hit%, EV, max EV,
  launch angle, whiff%, fly-ball%, pull%, HR/FB, **xISO**, and park factor. A blended
  **Calibrated** score = `0.85·HR Score + 0.15·Profile Match` and a **Top-5 list per
  category** are produced from it.
- **Trend strength (recency weighting)** — the centroid is **recency-weighted** with a
  configurable half-life (default 10 days): an HR from `h` days ago counts half as
  much as one today, so the match tracks *what's going deep now*. A **"what's
  shifting" table** compares HR hitters' last-7-day averages vs. the full window
  (e.g. "Pull% / HR/FB / xISO trending up"), and those shifts steer today's ranks.

Data path: LIVE pulls actual HR events from Baseball Savant
(`pybaseball.statcast`); OFFLINE simulates outcomes deterministically from the
modeled slates so the whole analysis runs without network.

### 6. Specialized scores

- **Swing-and-miss signal** = `0.6·Whiff% + 0.4·K%` (normalized). Real **Whiff%**
  (swing-and-miss rate = 100 − Contact%) is the primary input; K% is the fallback.
- **Fly-ball multiplier** on the HR probability: `1 + (FB% − 35)/35 · 0.5`, clipped
  to `[0.85, 1.18]`. Fly balls are the raw material of home runs, so an
  above-average **FB%** earns a direct HR-rate boost (and below-average a haircut).
- **HR/FB multiplier** on the HR probability: `1 + (HR/FB − 12.5)/12.5 · 0.35`,
  clipped to `[0.88, 1.15]`. HR-per-fly-ball is the fly-ball→HR conversion rate — a
  direct read on game power applied to balls in the air.
- **Longshot** = `0.32·MaxEV + 0.20·Barrel + 0.13·FlyBall + 0.10·HR/FB +
  0.08·Pull% + 0.10·Env + 0.07·Matchup` — the ceiling rewards air-ball power and
  pull tendency (pulled fly balls clear the wall most often) — then nudged by a
  **variance bonus** = `f(0.7·swing-and-miss + 0.3·Chase%)` and a **chalk penalty**.
- **Consistency** = `0.28·HardHit + 0.22·ContactFloor + 0.20·SeasonHR +
  0.15·AvgEV + 0.15·xwOBA`, where **ContactFloor = 0.6·(100 − swing-and-miss) +
  0.4·Zone-Contact%** (in-zone contact is the cleanest repeatable-contact signal),
  scaled by a sample-size confidence factor.
- **Sneaky** = `0.26·Matchup + 0.22·Env + 0.22·FormGap + 0.16·UnderRadar +
  0.14·Regression`, where *FormGap* rewards bats heating up beyond their season
  line, *UnderRadar* favors lower-profile hitters with real batted-ball pop, and
  *Regression* rewards hitters sitting **below their xHR** (due to bounce back).

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
- The recent-form pull (Statcast date-range) is heavy on first call; it is cached
  in-process and on disk (`pybaseball` cache) and refreshed via the sidebar button.
- Pitcher matchup peripherals (HR/9, FB%, barrels allowed) are currently modeled;
  the hitter Statcast, season stats, and recent-form HR rates are **live** when the
  hosts above are allowlisted.

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
│   ├── statcast.py         # real Statcast/FanGraphs season + recent-form pulls
│   ├── odds.py             # live HR odds (The Odds API) + model-implied fallback
│   ├── parlay.py           # ULX role-based 1-5 leg HR parlay generator
│   ├── lineup.py           # lineup-spot expected-PA, role fit, recurring HR log
│   ├── pitchers.py         # probable-SP HRs allowed by lineup spot (last 5 games)
│   ├── learn.py            # self-calibration: HR rate by model rating
│   ├── history.py          # trailing-month HR backtest, profile match, top-5
│   └── sources.py          # live MLB StatsAPI + Open-Meteo, merges real metrics
└── .streamlit/config.toml  # dark theme
```

## Disclaimer

This tool is for educational and entertainment purposes. Projections are
probabilistic estimates derived from public data and modeling assumptions; they
are **not** financial or betting advice.
