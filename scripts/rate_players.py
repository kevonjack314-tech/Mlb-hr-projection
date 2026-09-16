#!/usr/bin/env python3
"""Rate a SPECIFIC list of bats with the full live model, then split them into
diversified parlays.

The parlay builder normally picks from the whole slate. This answers the other
question — "here are the bats I already like, what does the model think, and
how should they be grouped?" — by running the identical live pipeline and then
restricting everything to the named players.

Usage:
    python scripts/rate_players.py [YYYY-MM-DD] "Name +odds, Name +odds, ..."

Odds are optional; when given, the model's edge is measured against YOUR price
rather than the book's current one. Needs open network (GitHub Actions runners
have it; sandboxes may not).
"""
import datetime as dt
import itertools
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history import (  # noqa: E402
    add_profile_similarity, build_hr_history, hr_profile_centroid,
)
from src.learn import attach_calibrated_prob, hit_rate_by_score  # noqa: E402
from src.lineup import attach_spot_signal, player_spot_hr  # noqa: E402
from src.model import score_slate  # noqa: E402
from src.odds import attach_odds, format_american  # noqa: E402
from src.pitchers import attach_sp_spot_signal, sp_spot_counts_for  # noqa: E402
from src.sources import get_slate  # noqa: E402
from src.statcast import normalize_name  # noqa: E402
from src.trends import attach_trend_signals  # noqa: E402

N_LEGS = 3
N_TICKETS = 3


def parse_request(text: str):
    """'Name +350, Other +310' -> [(name_key, display, american_or_None), ...]"""
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.search(r"([+-]\d{2,5})\s*$", chunk)
        odds = int(m.group(1)) if m else None
        name = chunk[:m.start()].strip() if m else chunk
        out.append((normalize_name(name), name, odds))
    return out


def dec(american) -> float:
    a = float(american)
    return a / 100.0 + 1.0 if a > 0 else 100.0 / abs(a) + 1.0


def to_american(d: float) -> int:
    return round((d - 1.0) * 100) if d >= 2 else -round(100.0 / (d - 1.0))


def _fmt(v, nd=1):
    try:
        f = float(v)
        return f"{f:.{nd}f}" if f == f else "—"
    except (TypeError, ValueError):
        return "—"


def best_tickets(rows, n_tickets=N_TICKETS, n_legs=N_LEGS):
    """Split the rated bats into tickets, covering the slip before repeating.

    Three rules, all from the model's own construction logic:
      * no two legs from the same GAME on one ticket — a single bad pitching
        matchup or a cold night at that park must not be able to kill a whole
        ticket by itself;
      * cover every bat on the slip before any bat is used twice, so nothing
        the user liked is silently dropped;
      * among equally-covering options, take the highest model win probability.

    Coverage has to be optimised ACROSS the whole set, not one ticket at a
    time: picking the strongest ticket first can strand two bats who share a
    game and therefore cannot legally finish a ticket together. Small slips are
    solved exactly; anything large enough to blow up falls back to greedy.
    """
    pool = sorted(rows, key=lambda r: -r["p"])
    combos = [c for c in itertools.combinations(pool, n_legs)
              if len({x["game"] for x in c}) == n_legs]
    if not combos:
        return []

    def prob(c):
        p = 1.0
        for x in c:
            p *= x["p"]
        return p

    def score(tickets):
        covered = set()
        for t in tickets:
            covered |= {x["name"] for x in t}
        return (len(covered), sum(math.log(prob(t)) for t in tickets))

    n_sets = math.comb(len(combos), n_tickets) if len(combos) >= n_tickets else 0
    if 0 < n_sets <= 250_000:
        return list(max(itertools.combinations(combos, n_tickets), key=score))

    chosen, used = [], set()
    for _ in range(n_tickets):
        taken = {tuple(sorted(x["name"] for x in t)) for t in chosen}
        best = max(
            (c for c in combos
             if tuple(sorted(x["name"] for x in c)) not in taken),
            key=lambda c: (len({x["name"] for x in c} - used), prob(c)),
            default=None)
        if best is None:
            break
        chosen.append(best)
        used |= {x["name"] for x in best}
    return chosen


def main() -> None:
    date_iso = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    want = parse_request(sys.argv[2] if len(sys.argv) > 2 else "")
    if not want:
        print("No players given."), sys.exit(1)
    game_date = dt.date.fromisoformat(date_iso)

    df, source, notes = get_slate(game_date, prefer_live=True)
    print(f"DATE: {date_iso}\nSLATE SOURCE: {source}")
    for n in notes[:4]:
        print(" -", n)
    if not str(source).startswith("LIVE"):
        print("\n!! NOT LIVE DATA — refusing to rate players off a synthetic slate.")
        sys.exit(1)

    scored = score_slate(df)
    start_iso = (game_date - dt.timedelta(days=30)).isoformat()
    events, slate_hist, h_src, _ = build_hr_history(start_iso, date_iso, prefer_live=True)
    print(f"HISTORY SOURCE: {h_src} ({len(events)} HR events)")

    scored = add_profile_similarity(
        scored, hr_profile_centroid(events, end_date_iso=date_iso, half_life_days=10))
    scored = attach_spot_signal(scored, player_spot_hr(slate_hist))
    scored = attach_calibrated_prob(scored, hit_rate_by_score(slate_hist))
    has_pid = "pitcher_id" in scored.columns
    pairs = tuple((g, n, (grp["pitcher_id"].iloc[0] if has_pid else None))
                  for (g, n), grp in scored.groupby(["game", "pitcher_name"]))
    scored = attach_sp_spot_signal(scored, sp_spot_counts_for(pairs, date_iso, True))
    scored = attach_trend_signals(scored, events, game_date.strftime("%A"))
    scored = attach_odds(scored, date_iso, use_live=True)
    scored["_key"] = scored["player"].map(normalize_name)
    live_odds = (bool(scored["odds_is_live"].any())
                 if "odds_is_live" in scored.columns else False)
    odds_label = ("LIVE book prices" if live_odds else
                  "MODEL-IMPLIED (tier-banded) — NOT real market prices")
    print(f"GAMES: {scored['game'].nunique()} | HITTERS: {len(scored)} | "
          f"ODDS: {odds_label}\n")

    rows, missing = [], []
    for key, display, odds in want:
        hit = scored[scored["_key"] == key]
        if hit.empty:
            missing.append(display)
            continue
        r = hit.iloc[0]
        p = float(r["hr_prob_game"])
        my_dec = dec(odds) if odds else None
        rows.append({
            "name": r["player"], "game": r["game"], "p": p, "row": r,
            "my_odds": odds,
            "edge": (p * my_dec - 1.0) * 100 if my_dec else None,
        })

    rows.sort(key=lambda x: -x["p"])
    print("=" * 96)
    print("MODEL RATING — your bats, ranked")
    print("=" * 96)
    book_hdr = "BOOK" if live_odds else "BOOK*"
    print(f"{'#':>2} {'PLAYER':22} {'HR%':>6} {'SCORE':>6} {'ULX':>10} {'SPOT':>4} "
          f"{'BRL%':>5} {'HR/FB':>6} {'MAXEV':>6} {book_hdr:>6} {'YOURS':>7} "
          f"{'EDGE':>8}")
    for i, x in enumerate(rows, 1):
        r = x["row"]
        spot = r.get("lineup_spot")
        spot_txt = str(int(spot)) if spot == spot and spot is not None else "—"
        mine = f"+{x['my_odds']}" if x["my_odds"] else "—"
        edge = f"{x['edge']:+.1f}%" if x["edge"] is not None else "—"
        print(f"{i:>2} {x['name'][:22]:22} {x['p']*100:>5.1f}% "
              f"{float(r.get('hr_score', 0)):>6.0f} "
              f"{str(r.get('ulx_grade', '—')):>10} {spot_txt:>4} "
              f"{_fmt(r.get('barrel_pct')):>5} {_fmt(r.get('hr_fb')):>6} "
              f"{_fmt(r.get('max_ev')):>6} "
              f"{format_american(r.get('book_odds')):>6} {mine:>7} {edge:>8}")
    for x in rows:
        r = x["row"]
        if r.get("rationale"):
            print(f"   {x['name']}: {r['rationale']}")
    if missing:
        print(f"\n!! NOT ON TODAY'S SLATE (scratched, not posted, or name mismatch): "
              f"{', '.join(missing)}")
    if len(rows) < N_LEGS:
        print("\nToo few rated bats to build tickets.")
        return

    print("\n" + "=" * 96)
    print(f"{N_TICKETS} x {N_LEGS}-LEG TICKETS — model-optimised, no two legs from one game")
    print("=" * 96)
    for i, ticket in enumerate(best_tickets(rows), 1):
        p = 1.0
        d_model = 1.0
        d_mine = 1.0
        has_all_prices = all(x["my_odds"] for x in ticket)
        print(f"\nTICKET {i}")
        for x in ticket:
            r = x["row"]
            p *= x["p"]
            d_model *= 1.0 / max(x["p"], 1e-6)
            if has_all_prices:
                d_mine *= dec(x["my_odds"])
            book = format_american(r.get("book_odds"))
            mine = f"+{x['my_odds']}" if x["my_odds"] else "—"
            tag = "book" if live_odds else "book*"
            print(f"   {x['name']:22} {x['p']*100:>5.1f}%  yours {mine:>6}  "
                  f"{tag} {book:>6}  {x['game']}")
        line = (f"   TICKET: model win {p*100:.2f}%  fair {to_american(1/p):+d}")
        if has_all_prices:
            ev = (p * d_mine - 1.0) * 100
            line += (f"  |  your price {to_american(d_mine):+d}  "
                     f"$10 pays ${10*(d_mine-1):,.2f}  EV {ev:+.1f}%")
        print(line)
    print("\nEV is vs the prices on YOUR slip, using the model's probability — "
          "it does not depend on the book column.")
    print("Positive = the model thinks your price is longer than the risk deserves.")
    if not live_odds:
        print("* BOOK prices are MODEL-IMPLIED tier-band fallbacks, not real "
              "market prices — no live odds feed was available. Ignore that "
              "column for shopping; set ODDS_API_KEY for real prices.")


if __name__ == "__main__":
    main()
