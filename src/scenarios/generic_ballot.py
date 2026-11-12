"""Generic-ballot scenarios: what a shifted generic ballot does to the value of Kalshi positions.

The forecasts themselves come from the production pipelines with a `shift`:

    house.forecast.run(house_inputs, shift=-4.0)
    senate.forecast.run(senate_inputs, shift=-4.0)

The shift moves 2026's gen_margin_raw (D+8.7) and each Stage 1 re-predicts, so a race moves by the model's own
pass-through (congress/adjustments.py, and the two forecast modules for what moves and what doesn't). Every run uses
its chamber's production seed, so the error draws are identical and runs differ only by the shift. Chamber summaries
and flipped calls are congress.report.chamber_summary and flipped_races.

Kalshi. The prices are one saved pull (data/processed/kalshi/kalshi_markets_2026.csv), held fixed across scenarios:
the question is what the model's probabilities say the same prices are worth.

- Fixed bets: a given set of bets (e.g. kelly_bets_2026.csv) valued under each scenario's probabilities
  (kalshi.optimizer.market_beliefs, expected_profit).
- Re-sized: kalshi.optimizer.tossups and kelly_bets rerun on each scenario's probabilities, as run.py does.

notebooks/scenarios/generic_ballot_shift.ipynb runs it; notebooks/full-forecast/ has the shifts for the submission.

Portfolio risk (portfolio_risk): a fixed portfolio, sized on the submission, priced in every simulation of every
generic-ballot run (scenario_runs), with the picks held at the submission's (kalshi/portfolio.py has the method).
Shared by the full forecast's section 9 and src/kalshi/run.py, so both write the same
data/processed/kalshi/portfolio_risk_2026.csv and portfolio_stress_bets_2026.csv.
"""

from dataclasses import dataclass

import pandas as pd

from congress import report
from house import forecast as house_forecast
from kalshi import odds, optimizer, portfolio
from senate import forecast as senate_forecast


@dataclass
class KalshiScenario:
    odds: pd.DataFrame       # odds.track_odds: the scenario's track beside Kalshi's prices
    beliefs: pd.Series       # P(YES) per market under the scenario (optimizer.market_beliefs)
    bets: pd.DataFrame       # optimizer.kelly_bets re-sized on the scenario's probabilities


def kalshi_scenario(track: pd.DataFrame, markets: pd.DataFrame, *, low, high, bankroll, kelly_fraction, min_edge
                    ) -> KalshiScenario:
    """A scenario's House + Senate track rows priced against `markets`, and its own Kelly bets (as run.py sizes
    them)."""
    table = odds.track_odds(track, markets)
    beliefs = optimizer.market_beliefs(table, markets)
    bets = optimizer.kelly_bets(optimizer.tossups(table, low=low, high=high), markets, bankroll=bankroll,
                                kelly_fraction=kelly_fraction, min_edge=min_edge)
    return KalshiScenario(table, beliefs, bets)


def kalshi_summary(kalshi: dict[str, KalshiScenario], fixed_bets: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario: the fixed bets' expected profit under its probabilities, and its own re-sized bets'
    count, stake and expected profit (all in dollars, under that scenario's probabilities)."""
    rows = {}
    for name, k in kalshi.items():
        fixed = optimizer.expected_profit(fixed_bets, k.beliefs)
        rows[name] = {
            "fixed bets: stake": fixed["stake"].sum(),
            "fixed bets: expected profit": fixed["expected_profit"].sum(),
            "fixed bets: losing in expectation": int((fixed["edge"] < 0).sum()),
            "fixed bets: unpriced": int(fixed["belief"].isna().sum()),
            "re-sized bets: n": len(k.bets),
            "re-sized bets: stake": k.bets["stake"].sum(),
            "re-sized bets: expected profit": k.bets["expected_profit"].sum(),
        }
    return pd.DataFrame(rows).T


def scenario_runs(shifts: dict[str, float], *, house_inputs, senate_inputs, senate_weights, house_adjustments,
                  senate_adjustments, baseline: str = "adjusted") -> tuple[dict, dict]:
    """({name: House Forecast}, {name: Senate Forecast}) for `baseline` (no shift) and each of `shifts`, most
    Republican first, every run with the same weights and adjustments."""
    named = sorted({baseline: 0.0, **shifts}.items(), key=lambda item: item[1])
    house = {name: house_forecast.run(house_inputs, shift=shift, adjustments=house_adjustments) for name, shift in named}
    senate = {name: senate_forecast.run(senate_inputs, weights=senate_weights, shift=shift,
                                        adjustments=senate_adjustments) for name, shift in named}
    return house, senate


def portfolio_risk(bets: pd.DataFrame, odds_table: pd.DataFrame, picks: pd.DataFrame, house_runs: dict,
                   senate_runs: dict, *, seed: int = house_forecast.SEED) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """(risk: one row per run, the profit distribution (portfolio.summarize) with its generic ballot; by_bet: each
    bet's stake and expected profit in each run; profits: {run: (simulations, bets) profit}). picks (the
    submission) fix who each bet is on; seed draws the same-party top-two candidates."""
    profits = {}
    for name, house in house_runs.items():
        race_ids, outcomes = report.pick_outcomes(house, senate_runs[name], picks, seed=seed)
        profits[name] = portfolio.profit_draws(bets, odds_table, race_ids, outcomes)
    stake = bets["stake"].sum()
    risk = pd.DataFrame({name: portfolio.summarize(p.sum(axis=1), stake=stake) for name, p in profits.items()}).T
    risk.insert(0, "generic ballot (D - R)", [house_runs[name].generic_ballot for name in risk.index])
    by_bet = pd.DataFrame({name: p.mean(axis=0) for name, p in profits.items()},
                          index=pd.MultiIndex.from_frame(bets[["race_id", "candidate", "side"]]))
    by_bet.insert(0, "stake", bets["stake"].to_numpy())
    return risk, by_bet, profits


def write_portfolio_risk(risk: pd.DataFrame, by_bet: pd.DataFrame, out_dir) -> list:
    """Write portfolio_risk_2026.csv and portfolio_stress_bets_2026.csv (6 decimals); return their paths."""
    paths = [out_dir / "portfolio_risk_2026.csv", out_dir / "portfolio_stress_bets_2026.csv"]
    risk.rename_axis("scenario").reset_index().round(6).to_csv(paths[0], index=False)
    by_bet.reset_index().round(6).to_csv(paths[1], index=False)
    return paths
