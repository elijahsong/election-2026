"""Kalshi odds for the congress-track submission, and the toss-up optimizer (public Kalshi API only; no orders).

    uv run python src/kalshi/run.py                                         # the latest cached snapshot
    uv run python src/kalshi/run.py data/raw/kalshi/events_<stamp>.json     # a specific cached snapshot
    uv run python src/kalshi/run.py --refresh                               # pull live prices, cache them, run
    uv run python src/kalshi/run.py --no-risk                               # skip the portfolio risk step

Prices are cached: every pull is saved as data/raw/kalshi/events_<UTC stamp>.json, and a run without --refresh reads
the latest saved snapshot and never calls the API (it stops if there is none). Rerunning after a new submission
therefore re-prices it against the same quotes; only --refresh moves the prices.

Inputs
  data/processed/full congress track template.csv   the submission (notebooks/full-forecast/full_forecast_2026.ipynb)
  data/reference/track_overrides_2026.csv           the retired probability overrides (a record; no longer applied)
  data/processed/house/candidates_2026.csv          party for California's person markets
Outputs (data/processed/kalshi/)
  kalshi_markets_2026.csv       one row per Kalshi market of a template race: quote, price, kalshi_prob
  kalshi_odds_2026.csv          the submission's 470 rows beside Kalshi's price for each predicted winner
  tossups_2026.csv              the toss-up races: submission, Kalshi, blend, proposed pick, expected Brier scores
  proposed_overrides_2026.csv   track_overrides_2026.csv with the blend's rows: informational only, since the
                                submission no longer applies probability overrides
  kelly_bets_2026.csv           recommended positions, sized on the submission's probabilities
  kalshi_odds_log_2026.csv      every pull's markets, appended with pulled_at (a rerun of a snapshot replaces its
                                rows), so past odds can be read back without calling the API
  portfolio_risk_2026.csv       the Kelly bets' profit distribution over every simulation, in the main forecast and
                                with the generic ballot off by GB_SHIFTS: expected profit, P(loss), percentiles, worst
  portfolio_stress_bets_2026.csv  each bet's stake and expected profit in each of those runs

Portfolio risk (scenarios.generic_ballot.portfolio_risk, shared with notebooks/full-forecast/): the forecast is rerun
here, with production settings (house.forecast, senate.forecast, the production Senate blend weights) and
data/reference/margin_adjustments_2026.csv, unshifted and shifted by each of GB_SHIFTS. The unshifted run must
reproduce the submission exactly; if it doesn't (e.g. the notebook used custom Senate weights), the risk step stops
and the full-forecast notebook's section 9 is the one to use. About 10 seconds.
Raw
  data/raw/kalshi/events_<UTC stamp>.json   the matched House and Senate events as fetched

Method and settings: kalshi/odds.py (matching, prices) and kalshi/optimizer.py (toss-ups, blend, Kelly).
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from congress import adjustments
from house import forecast as house_forecast
from kalshi import odds, optimizer
from scenarios import generic_ballot
from senate import forecast as senate_forecast
from senate import stage2

TRACK_PATH = ROOT / "data/processed/full congress track template.csv"
OVERRIDES_PATH = ROOT / "data/reference/track_overrides_2026.csv"
CANDIDATES_PATH = ROOT / "data/processed/house/candidates_2026.csv"
RAW_DIR = ROOT / "data/raw/kalshi"
OUT_DIR = ROOT / "data/processed/kalshi"
LOG_PATH = OUT_DIR / "kalshi_odds_log_2026.csv"

TOSSUP_LOW, TOSSUP_HIGH = 0.4, 0.6   # either the submission or Kalshi puts the submitted candidate in this range
MARKET_WEIGHT = 0.5                  # Kalshi's weight in the blended belief; judgment, not tuned
BANKROLL = 1000.0                    # dollars
KELLY_FRACTION = 0.25                # quarter Kelly
MIN_EDGE = 0.02                      # belief minus cost (price + fee) per $1 contract
GB_SHIFTS = {"R+4": -4.0, "R+2": -2.0, "D+2": 2.0, "D+4": 4.0}  # generic-ballot stress tests (points, D - R)


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path | None:
    """The newest cached events_<UTC stamp>.json (the stamps sort by time), or None."""
    snapshots = sorted(raw_dir.glob("events_*.json"))
    return snapshots[-1] if snapshots else None


def load_events(argv: list[str], raw_dir: Path = RAW_DIR) -> tuple[list[dict], str]:
    """(events, as_of): a cached snapshot, the one named in argv or else the latest in raw_dir; live from the API only
    with --refresh, which caches the pull first."""
    if "--refresh" not in argv:
        path = Path(argv[0]) if argv else latest_snapshot(raw_dir)
        if path is None:
            raise SystemExit(f"No cached Kalshi snapshot in {raw_dir}; run with --refresh to pull one")
        print(f"Kalshi prices from the cached snapshot {path.name} (no API call)")
        return json.loads(path.read_text()), path.stem.removeprefix("events_")
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    events = [e for e in odds.fetch_open_events() if odds.race_id_from_title(e["title"])]
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"events_{stamp}.json"
    path.write_text(json.dumps(events))
    print(f"-> {path} ({len(events)} House and Senate events)")
    return events, stamp


def log_pull(markets: pd.DataFrame, as_of: str) -> None:
    """Append this pull's markets to the odds log, replacing any earlier rows from the same snapshot."""
    pull = markets.assign(pulled_at=as_of)[["pulled_at", *markets.columns]]
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH)
        pull = pd.concat([log[log["pulled_at"] != as_of], pull], ignore_index=True)
    pull.to_csv(LOG_PATH, index=False)
    print(f"-> {LOG_PATH} ({pull['pulled_at'].nunique()} pulls, {len(pull)} rows)")


def risk_step(track: pd.DataFrame, table: pd.DataFrame, bets: pd.DataFrame) -> None:
    """The portfolio's profit in every simulation of the main forecast and each GB_SHIFTS stress test; stops unless
    the rerun forecast reproduces the submission."""
    house_inputs, senate_inputs = house_forecast.load_inputs(), senate_forecast.load_inputs()
    margin = adjustments.load_margin_adjustments(template=house_inputs.template)
    house_runs, senate_runs = generic_ballot.scenario_runs(
        GB_SHIFTS, house_inputs=house_inputs, senate_inputs=senate_inputs,
        senate_weights=stage2.WEIGHTS, house_adjustments=adjustments.by_race(margin, "House"),
        senate_adjustments=adjustments.by_race(margin, "Senate"))
    rerun = pd.concat([house_runs["adjusted"].track, senate_runs["adjusted"].track]).set_index("race_id")
    rerun["win_probability"] = rerun["win_probability"].round(6)
    submitted = track.set_index("race_id")
    columns = ["predicted_winner_party", "candidate_name", "win_probability"]
    if not rerun.loc[submitted.index, columns].equals(submitted[columns]):
        raise SystemExit("Risk step stopped: the production forecast with margin_adjustments_2026.csv doesn't "
                         "reproduce the submission (custom Senate weights, or a stale submission?). Use section 9 of "
                         "notebooks/full-forecast/full_forecast_2026.ipynb, or rerun it first.")
    risk, by_bet, _ = generic_ballot.portfolio_risk(bets, table, track, house_runs, senate_runs)
    for path in generic_ballot.write_portfolio_risk(risk, by_bet, OUT_DIR):
        print(f"-> {path}")
    worst = risk["expected profit"].idxmin()
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"\nPortfolio risk ({len(bets)} bets, ${bets['stake'].sum():,.2f} staked; profit per simulation, "
              f"model probabilities):")
        print(risk.drop(columns=["stake", "best simulation"]).round(3).to_string())
        print(f"Worst scenario {worst}: expected ${risk.loc[worst, 'expected profit']:,.0f}, "
              f"P(loss) {risk.loc[worst, 'P(loss)']:.1%}, 5th percentile ${risk.loc[worst, '5th percentile']:,.0f}, "
              f"worst simulation ${risk.loc[worst, 'worst simulation']:,.0f}")


def main(argv: list[str]) -> None:
    events, as_of = load_events([a for a in argv if a != "--no-risk"])
    track = pd.read_csv(TRACK_PATH)
    markets = odds.fill_party_from_candidates(odds.race_markets(events), pd.read_csv(CANDIDATES_PATH))
    markets = markets[markets["race_id"].isin(track["race_id"])]
    table = odds.track_odds(track, markets)

    unmatched = table.loc[table["match"] == "none", "race_id"].tolist()
    print(f"Kalshi {as_of}: {len(table) - len(unmatched)}/{len(table)} submitted candidates matched to a market; "
          f"unmatched: {unmatched}")
    disagree = table[table["picks_agree"] == False]
    print(f"Kalshi's favorite differs from the submitted pick in {len(disagree)} races: "
          + ", ".join(disagree["race_id"]))

    races = optimizer.tossups(table, low=TOSSUP_LOW, high=TOSSUP_HIGH)
    picks = optimizer.blend_picks(races, markets, market_weight=MARKET_WEIGHT)
    proposals = optimizer.proposed_overrides(picks, pd.read_csv(OVERRIDES_PATH), market_weight=MARKET_WEIGHT,
                                             as_of=as_of)
    bets = optimizer.kelly_bets(races, markets, bankroll=BANKROLL, kelly_fraction=KELLY_FRACTION,
                                min_edge=MIN_EDGE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in [("kalshi_markets_2026", markets), ("kalshi_odds_2026", table), ("tossups_2026", picks),
                        ("proposed_overrides_2026", proposals), ("kelly_bets_2026", bets)]:
        frame.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"-> {OUT_DIR / name}.csv ({len(frame)} rows)")
    log_pull(markets, as_of)

    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_rows", 100):
        print(f"\n{len(picks)} toss-ups (submission <= {TOSSUP_HIGH} or Kalshi in [{TOSSUP_LOW}, {TOSSUP_HIGH}]), "
              f"market weight {MARKET_WEIGHT}:")
        print(picks[["race_id", "predicted_winner_party", "candidate_name", "win_probability", "kalshi_winner_prob",
                     "blend_prob", "proposed_party", "proposed_candidate", "proposed_probability",
                     "expected_brier_submitted", "expected_brier_proposed"]].round(3).to_string(index=False))
        print(f"\nKelly bets ({KELLY_FRACTION} Kelly, ${BANKROLL:,.0f} bankroll, edge >= {MIN_EDGE}):")
        print(bets[["race_id", "market_ticker", "candidate", "side", "price", "belief", "edge", "kelly",
                    "kelly_contracts", "contracts", "stake", "expected_profit"]].round(3).to_string(index=False))
        print(f"Total stake ${bets['stake'].sum():,.2f}, expected profit ${bets['expected_profit'].sum():,.2f} "
              f"(under the submission's probabilities)")
    if "--no-risk" not in argv:
        risk_step(track, table, bets)


if __name__ == "__main__":
    main(sys.argv[1:])
