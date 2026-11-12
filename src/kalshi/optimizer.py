"""Two uses of Kalshi prices in toss-up races: blended submission picks, and Kelly bet sizing.

Toss-ups (tossups): races where either side puts the submitted candidate between `low` and `high` (default 40-60%):
the submission's win_probability is at most `high` (it is always >= 0.5), or Kalshi's kalshi_winner_prob is in
[low, high]. This is "the model or Kalshi puts the Democrat at 40-60%", stated per submitted candidate so that it also
covers independent-vs-R and top-two same-party races.

1. Submission picks (blend_picks)
   Belief for the submitted candidate: q = (1 - market_weight) * p + market_weight * k, a linear opinion pool of the
   submission's probability p and Kalshi's k. A report r on that candidate has expected Brier score
   q(1 - r)^2 + (1 - q)r^2 under q, minimized at r = q, so the proposal reports the side with q >= 0.5:
       q >= 0.5   the submitted candidate, at q
       q <  0.5   Kalshi's most likely other candidate (kalshi_opponent), at 1 - q (a flip)
   market_weight = 0.5 is a judgment, not a fit: the repo has no Kalshi history to tune it on.
   p is the submission as filed, so in the five manually overridden races it is the override, not the model.

   proposed_overrides writes the result in data/reference/track_overrides_2026.csv's format. Each toss-up gets a row;
   existing overrides of other races are kept. model_party / model_candidate / model_probability are the model's own
   values (an existing override's, else the submission's). It is informational only: the submission no longer
   applies probability overrides (manual judgment is a margin adjustment, src/congress/adjustments.py), and
   track_overrides_2026.csv is kept only as the record of the retired overrides.

2. Bet sizing (kelly_bets)
   Treats the submission's probability as the truth. Options per toss-up: buy YES or NO on the submitted candidate's
   market (belief p), and, in a two-market race, on the opponent's market (belief 1 - p). Buying at price P costs
   c = P + fee_rate * P * (1 - P) per $1 contract (Kalshi's quadratic taker fee, multiplier 1 on these series; the
   per-order round-up to the cent is ignored). Edge = belief - c; the full-Kelly fraction of a binary contract is
   edge / (1 - c). Each race keeps its best option if its edge is at least min_edge; its kelly_stake is
   kelly_fraction * Kelly * bankroll, scaled down together if the stakes would exceed the bankroll. contracts is
   kelly_contracts capped at the size quoted at that price (top of book; deeper levels cost more and aren't priced).
   Races are sized one at a time, ignoring their correlation through the national environment; fractional Kelly and
   the bankroll cap are the only guards against that. Recommendations only: nothing here places an order.

3. Valuing bets under other beliefs (market_beliefs, expected_profit)
   market_beliefs gives P(YES) per market from a submission's odds table, by the same rule kelly_bets uses.
   expected_profit re-scores a fixed set of bets under those beliefs, e.g. a generic-ballot scenario's forecast
   (src/scenarios/generic_ballot.py).
"""

import numpy as np
import pandas as pd

OVERRIDE_COLUMNS = ["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability",
                    "model_party", "model_candidate", "model_probability", "reason"]


def tossups(odds: pd.DataFrame, low: float = 0.4, high: float = 0.6) -> pd.DataFrame:
    model_close = odds["win_probability"] <= high
    market_close = odds["kalshi_winner_prob"].between(low, high)
    priced = odds["kalshi_winner_prob"].notna()  # a race without a matched market can't be blended or bet
    return odds[(model_close | market_close) & priced].assign(model_close=model_close, market_close=market_close)


def kalshi_opponent(race: pd.Series, markets: pd.DataFrame) -> pd.Series:
    """The most likely other candidate in the race's Kalshi event."""
    event = markets[(markets["event_ticker"] == race["kalshi_event"])
                    & (markets["market_ticker"] != race["kalshi_market"])]
    return event.loc[event["kalshi_prob"].idxmax()]


def expected_brier(belief: float, report: float) -> float:
    return belief * (1 - report) ** 2 + (1 - belief) * report ** 2


def blend_picks(races: pd.DataFrame, markets: pd.DataFrame, market_weight: float) -> pd.DataFrame:
    """Each race with the blended belief q and the pick and probability that minimize expected Brier under q."""
    out = races.copy()
    out["blend_prob"] = (1 - market_weight) * out["win_probability"] + market_weight * out["kalshi_winner_prob"]
    out["flip"] = out["blend_prob"] < 0.5
    opponents = pd.DataFrame([kalshi_opponent(race, markets) for _, race in out.iterrows()], index=out.index)
    out["proposed_party"] = np.where(out["flip"], opponents["party"], out["predicted_winner_party"])
    out["proposed_candidate"] = np.where(out["flip"], opponents["candidate"], out["candidate_name"])
    out["proposed_probability"] = np.where(out["flip"], 1 - out["blend_prob"], out["blend_prob"]).round(6)
    out["expected_brier_submitted"] = [expected_brier(q, p) for q, p in zip(out["blend_prob"], out["win_probability"])]
    out["expected_brier_proposed"] = [expected_brier(q, q) for q in out["blend_prob"]]
    return out


def proposed_overrides(picks: pd.DataFrame, overrides: pd.DataFrame, market_weight: float, as_of: str
                       ) -> pd.DataFrame:
    """overrides with a row per blended pick, its model_* columns set to the model's own values."""
    existing = overrides.set_index("race_id")
    rows = []
    for _, race in picks.iterrows():
        if race["race_id"] in existing.index:
            model = existing.loc[race["race_id"], ["model_party", "model_candidate", "model_probability"]].tolist()
            source = "override"
        else:
            model = [race["predicted_winner_party"], race["candidate_name"], race["win_probability"]]
            source = "model"
        reason = (f"Kalshi blend ({as_of}): submitted ({source}) {race['predicted_winner_party']} "
                  f"{race['win_probability']:.3f}, Kalshi {race['kalshi_winner_prob']:.3f}, "
                  f"market weight {market_weight}")
        rows.append([race["race_id"], race["chamber"], race["proposed_party"], race["proposed_candidate"],
                     race["proposed_probability"], *model, reason])
    new = pd.DataFrame(rows, columns=OVERRIDE_COLUMNS)
    kept = overrides[~overrides["race_id"].isin(new["race_id"])]
    return pd.concat([kept, new], ignore_index=True)[OVERRIDE_COLUMNS]


def race_beliefs(race: pd.Series, event: pd.DataFrame) -> dict[str, float]:
    """{market_ticker: P(YES)} for one race: win_probability on the submitted candidate's market and, in a
    two-market race, 1 - win_probability on the other; any other market of the event is left unpriced."""
    beliefs = {race["kalshi_market"]: race["win_probability"]}
    if len(event) == 2:
        other = event.loc[event["market_ticker"] != race["kalshi_market"], "market_ticker"].item()
        beliefs[other] = 1 - race["win_probability"]
    return beliefs


def market_beliefs(races: pd.DataFrame, markets: pd.DataFrame) -> pd.Series:
    """P(YES) per market_ticker (race_beliefs) over every race matched to a Kalshi market."""
    beliefs = {}
    for _, race in races[races["kalshi_market"].notna()].iterrows():
        beliefs |= race_beliefs(race, markets[markets["event_ticker"] == race["kalshi_event"]])
    return pd.Series(beliefs, name="belief_yes", dtype=float)


def expected_profit(bets: pd.DataFrame, beliefs: pd.Series) -> pd.DataFrame:
    """bets (kelly_bets' columns) valued under other beliefs (market_beliefs): each bet's belief and edge
    (belief - cost per contract) and expected_profit = contracts * edge. A bet on a market the beliefs leave unpriced
    gets NaN."""
    yes = bets["market_ticker"].map(beliefs)
    belief = np.where(bets["side"] == "yes", yes, 1 - yes)
    out = bets[["race_id", "market_ticker", "candidate", "side", "price", "cost", "contracts", "stake"]].copy()
    out["belief"] = belief
    out["edge"] = out["belief"] - out["cost"]
    out["expected_profit"] = out["contracts"] * out["edge"]
    return out


def kelly_bets(races: pd.DataFrame, markets: pd.DataFrame, bankroll: float, kelly_fraction: float, min_edge: float,
               fee_rate: float = 0.07) -> pd.DataFrame:
    """The best bet per toss-up race, by Kelly fraction, with the submission's probability as the truth."""
    options = []
    for _, race in races.iterrows():
        event = markets[markets["event_ticker"] == race["kalshi_event"]]
        beliefs = race_beliefs(race, event)
        for _, m in event[event["market_ticker"].isin(beliefs)].iterrows():
            p = beliefs[m["market_ticker"]]
            for side, belief, price, size in [("yes", p, m["yes_ask"], m["yes_ask_size"]),
                                              ("no", 1 - p, m["no_ask"], m["yes_bid_size"])]:
                if not 0 < price < 1:
                    continue
                cost = price + fee_rate * price * (1 - price)
                options.append({"race_id": race["race_id"], "market_ticker": m["market_ticker"],
                                "candidate": m["candidate"], "side": side, "price": price, "cost": cost,
                                "size_at_price": size, "belief": belief, "edge": belief - cost,
                                "kelly": (belief - cost) / (1 - cost)})
    best = (pd.DataFrame(options).sort_values("kelly", ascending=False).groupby("race_id").head(1)
            .query("edge >= @min_edge").reset_index(drop=True))
    stake = kelly_fraction * best["kelly"] * bankroll
    best["kelly_stake"] = stake * min(1.0, bankroll / stake.sum()) if len(best) else stake
    best["kelly_contracts"] = np.floor(best["kelly_stake"] / best["cost"]).astype(int)
    best["contracts"] = np.minimum(best["kelly_contracts"], np.floor(best["size_at_price"])).astype(int)  # top of book
    best["stake"] = best["contracts"] * best["cost"]
    best["expected_profit"] = best["contracts"] * best["edge"]
    return best
