"""The volatility of a fixed Kalshi portfolio: its profit in every simulated election.

The portfolio is optimizer.kelly_bets, sized on the submission's probabilities. Every bet is on the submitted
candidate's market or, in a two-market race, on the other one (optimizer.race_beliefs), so a bet's payout in a
simulation depends only on whether the submitted candidate won there (congress.report.pick_outcomes):

    YES on the submitted candidate, or NO on the other one    pays $1 if the submitted candidate wins
    NO on the submitted candidate, or YES on the other one    pays $1 if the submitted candidate loses

A bet's profit in a simulation is contracts * (payout - cost), cost being kelly_bets' price plus fee per contract.
Summed over bets, that gives one portfolio profit per simulation. Its mean is the portfolio's expected profit (kelly_bets'
expected_profit, up to the same-party top-two draws), and its spread is the volatility. The national error in each
simulation is the generic-ballot uncertainty, so nothing is added for it.

Stress tests: the same bets and submitted candidates, priced against a generic-ballot scenario's draws.

Caveats: House and Senate are simulated separately, with independent national errors, so the combined spread
understates how the two chambers' bets move together (the stress tests move both). Fills are assumed at the quoted
top of book, with no slippage.
"""

import numpy as np
import pandas as pd


def bet_payoffs(bets: pd.DataFrame, odds_table: pd.DataFrame) -> pd.DataFrame:
    """bets with pays_if_pick_wins: whether each bet pays when the race's submitted candidate wins."""
    submitted = bets["race_id"].map(odds_table.set_index("race_id")["kalshi_market"])
    on_submitted = bets["market_ticker"].eq(submitted)
    return bets.assign(pays_if_pick_wins=on_submitted == bets["side"].eq("yes"))


def profit_draws(bets: pd.DataFrame, odds_table: pd.DataFrame, race_ids: list[str],
                 outcomes: np.ndarray) -> np.ndarray:
    """(n_sims, n_bets) profit of each bet in each simulation, in dollars; outcomes from pick_outcomes."""
    payoffs = bet_payoffs(bets, odds_table)
    column = {race_id: j for j, race_id in enumerate(race_ids)}
    pick_wins = outcomes[:, [column[r] for r in payoffs["race_id"]]]
    pays = np.where(payoffs["pays_if_pick_wins"].to_numpy(), pick_wins, ~pick_wins)
    return payoffs["contracts"].to_numpy() * (pays - payoffs["cost"].to_numpy())


def summarize(profit: np.ndarray, stake: float) -> dict:
    """The portfolio profit distribution (profit: one total per simulation)."""
    return {
        "expected profit": profit.mean(),
        "sd": profit.std(),
        "P(loss)": (profit < 0).mean(),
        "5th percentile": np.quantile(profit, 0.05),
        "1st percentile": np.quantile(profit, 0.01),
        "worst simulation": profit.min(),
        "best simulation": profit.max(),
        "stake": stake,
        "expected return on stake": profit.mean() / stake,
    }
