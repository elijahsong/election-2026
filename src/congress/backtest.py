"""Backtest of the full forecast on 2020, 2022 and 2024, scored as ratings and calls (notebooks/full-forecast/
full_forecast_backtest.ipynb).

Each test year is forecast the way 2026 is, from earlier elections only:
  House   Stage 1's rolling out-of-sample predictions (each year fit on 2006 through the election before it), and the
          H0 error model fit on the earlier years' residuals (from 2018), simulated with the race rules
          (house.simulation.backtest). Margin adjustments, where given, are added to the Stage 1 margin first
  Senate  the Stage 2 blend's out-of-sample centers (production weights) and the S-M0 errors fit on earlier years
          (from 2016) (senate.simulation.calibration.backtest). Independent-vs-R races are not in a historical panel
  Top two same-party races, predicted by top_two.predict_top_two from the primary and fundraising (no polls)

Race rows (house_races, senate_races) share congress.report's columns where they can: race_id, chamber, race_type,
dem_candidate, gop_candidate, incumbent (D, R or open), margin_model, margin_adjustment, margin (the simulated
center), actual (the observed two-party margin, D - R; blank where none is valid), p_dem, p_gop, p_other,
predicted_winner_party and party_probability (the most likely side), rating (report.rate), winner_party (D, R or I),
called_right. Political errors are actual - margin: positive means the Democrats did better than forecast.

Kalshi (2024 only; data/processed/kalshi/kalshi_2024_eve.csv from src/kalshi/history.py): each race's election-eve
prices beside the model, and the run.py strategy (toss-ups, quarter Kelly) replayed on them, scored on the results.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import house.regression as hr
from congress import report
from house import forecast as house_forecast
from house import simulation
from house.regression.data import ROOT
from kalshi import odds, optimizer
from senate import forecast as senate_forecast
from senate.simulation import calibration as senate_calibration

YEARS = (2020, 2022, 2024)
MEDSL_HOUSE_PATH = ROOT / "data/raw/1976-2024-house.tab"            # candidate names (MEDSL November returns)
MEDSL_SENATE_PATH = ROOT / "data/raw/medsl_senate/1976-2024-senate-state.csv"
TOP_TWO_HISTORY_PATH = ROOT / "data/processed/house/toptwosameparty_history.csv"
KALSHI_2024_PATH = ROOT / "data/processed/kalshi/kalshi_2024_eve.csv"
HOUSE_KEY = ["year", "state", "district"]
SENATE_KEY = ["year", "state", "class"]
PARTY_CODE = {"D": "D", "R": "R", "other": "I"}
INCUMBENT = {1: "D", -1: "R", 0: "open"}


def candidate_names(path, key: dict[str, str], party: str, years=YEARS) -> pd.DataFrame:
    """Each party's top November candidate (title case) from a MEDSL returns file: the columns of `key` ({ours:
    MEDSL's}), dem_candidate, gop_candidate. Write-ins are dropped."""
    m = pd.read_csv(path, dtype=str)
    m = m[m["year"].isin([str(y) for y in years]) & (m["stage"].str.upper() == "GEN")
          & (m["writein"].str.upper() != "TRUE") & m[party].isin(["DEMOCRAT", "REPUBLICAN"])]
    m = m.assign(votes=pd.to_numeric(m["candidatevotes"]), year=m["year"].astype(int))
    top = m.sort_values("votes", ascending=False).drop_duplicates([*key.values(), party])
    wide = top.pivot(index=list(key.values()), columns=party, values="candidate").reset_index()
    wide = wide.rename(columns={**{v: k for k, v in key.items()}, "DEMOCRAT": "dem_candidate",
                                "REPUBLICAN": "gop_candidate"})
    for column in ("dem_candidate", "gop_candidate"):
        wide[column] = wide[column].str.title()
    return wide


def house_names() -> pd.DataFrame:
    names = candidate_names(MEDSL_HOUSE_PATH, {"year": "year", "state": "state_po", "district": "district"}, "party")
    return names.assign(district=names["district"].astype(int))


def senate_names() -> pd.DataFrame:
    names = candidate_names(MEDSL_SENATE_PATH, {"year": "year", "state": "state_po", "special": "special"},
                            "party_simplified")
    return names.assign(special=names["special"].str.upper() == "TRUE")


def finish(races: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Add the predicted side, its probability, the rating and whether the call was right."""
    probs = races[["p_dem", "p_gop", "p_other"]]
    races["predicted_winner_party"] = probs.idxmax(axis=1).map({"p_dem": "D", "p_gop": "R", "p_other": "I"})
    races["party_probability"] = probs.max(axis=1)
    races["rating"] = report.rate(races["party_probability"], races["predicted_winner_party"], thresholds)
    races["called_right"] = (races["predicted_winner_party"] == races["winner_party"]).where(
        races["winner_party"].notna())
    return races


# --- House -----------------------------------------------------------------------------------------------------

@dataclass
class HouseInputs:
    predictors: pd.DataFrame
    targets: pd.DataFrame
    races: pd.DataFrame        # historical_races: lineups, winners, observed margins
    run: simulation.Stage1Inputs
    fallback: simulation.Stage1Inputs
    names: pd.DataFrame


def load_house_inputs() -> HouseInputs:
    """The production configurations' rolling Stage 1 runs (2018-2024), the race lineups and candidate names."""
    import warnings
    with warnings.catch_warnings(action="ignore"):
        predictors, targets = hr.load_predictors(), hr.load_targets()
    return HouseInputs(
        predictors=predictors, targets=targets,
        races=simulation.historical_races(targets, simulation.load_targets_raw()),
        run=simulation.stage1_inputs(house_forecast.CONFIG, predictors=predictors, targets=targets),
        fallback=simulation.stage1_inputs(house_forecast.FALLBACK_CONFIG, predictors=predictors, targets=targets),
        names=house_names())


def adjust(predictions: pd.DataFrame, adjustments: pd.DataFrame | None) -> pd.DataFrame:
    """predictions with adjustments' margin_adjustment (HOUSE_KEY rows, points) added to prediction."""
    if adjustments is None or adjustments.empty:
        return predictions
    points = predictions[HOUSE_KEY].merge(adjustments, on=HOUSE_KEY, how="left")["margin_adjustment"].fillna(0)
    return predictions.assign(prediction=predictions["prediction"] + points.to_numpy())


def house_backtest(inputs: HouseInputs, *, adjustments: pd.DataFrame | None = None, years=YEARS,
                   n_sims: int = house_forecast.N_SIMS, seed: int = house_forecast.SEED) -> simulation.Calibration:
    """house.simulation.backtest with the production settings, margins adjusted first. The error model is fit on the
    unadjusted residuals (oos), as in 2026."""
    return simulation.backtest(
        inputs.run.oos, test_years=list(years), races=inputs.races,
        predictions=adjust(inputs.run.predictions, adjustments),
        fallback_predictions=adjust(inputs.fallback.predictions, adjustments), nu=house_forecast.NU, n_sims=n_sims,
        seed=seed, house_size=house_forecast.HOUSE_SEATS, national_prior_sd=house_forecast.NATIONAL_PRIOR_SD,
        national_prior_elections=house_forecast.NATIONAL_PRIOR_ELECTIONS)


def house_races(calibration: simulation.Calibration, inputs: HouseInputs, adjustments: pd.DataFrame | None,
                thresholds: dict = report.RATING_THRESHOLDS) -> pd.DataFrame:
    d = pd.concat([r.districts for r in calibration.years.values()], ignore_index=True)
    points = adjustments if adjustments is not None else pd.DataFrame(columns=[*HOUSE_KEY, "margin_adjustment"])
    d = d.merge(points, on=HOUSE_KEY, how="left").merge(inputs.names, on=HOUSE_KEY, how="left")
    d = d.merge(inputs.predictors[[*HOUSE_KEY, "inc_signed"]], on=HOUSE_KEY, how="left")
    adjustment = d["margin_adjustment"].fillna(0.0).where(d["prediction"].notna(), 0.0)
    races = pd.DataFrame({
        "year": d["year"], "race_id": [simulation.race_id(s, n) for s, n in zip(d["state"], d["district"])],
        "chamber": "House", "race_type": d["race_type"], "dem_candidate": d["dem_candidate"],
        "gop_candidate": d["gop_candidate"], "incumbent": d["inc_signed"].map(INCUMBENT),
        "margin_model": d["prediction"] - adjustment, "margin_adjustment": adjustment, "margin": d["prediction"],
        "actual": d["actual"], "p_dem": d["p_dem"], "p_gop": d["p_gop"], "p_other": d["p_other"],
        "winner_party": d["winner_party"].map(PARTY_CODE).where(d["winner_resolved"]),
        "treatment": d["treatment"]})
    return finish(races, thresholds)


def house_seats(calibration: simulation.Calibration) -> pd.DataFrame:
    """Per year: the simulated D seats (median, 80% interval), P(D 218+), and what happened."""
    rows = {}
    for year, r in calibration.years.items():
        seats = r.house_seats["D"]
        actual = int((r.districts["winner_party"] == "D").sum())
        rows[year] = {"median D seats": float(np.median(seats)),
                      "D seats 80%": f"{np.quantile(seats, 0.1):.0f}–{np.quantile(seats, 0.9):.0f}",
                      f"P(D {report.HOUSE_MAJORITY}+)": float((seats >= report.HOUSE_MAJORITY).mean()),
                      "actual D seats": actual, "D majority": actual >= report.HOUSE_MAJORITY}
    return pd.DataFrame(rows).T


# --- Senate ----------------------------------------------------------------------------------------------------

def senate_races(*, years=YEARS, n_sims: int = senate_forecast.N_SIMS, seed: int = senate_forecast.SEED,
                 thresholds: dict = report.RATING_THRESHOLDS) -> pd.DataFrame:
    """The production Senate backtest (S1 + blend + S-M0; every test year from 2018 drawn as 03_monte_carlo.ipynb
    does), as race rows for `years`."""
    oos = senate_calibration.blend_out_of_sample(senate_forecast.CONFIG)
    scored = senate_calibration.backtest(oos, model=senate_forecast.MODEL, n_sims=n_sims, seed=seed)["races"]
    d = scored[scored["year"].isin(years)].merge(
        oos[[*SENATE_KEY, "special", "center", "actual", "inc_signed"]], on=SENATE_KEY, validate="one_to_one")
    d = d.assign(special=d["special"].astype(bool)).merge(senate_names(), on=["year", "state", "special"], how="left",
                                                          validate="one_to_one")
    races = pd.DataFrame({
        "year": d["year"], "race_id": d["state"] + "-SEN" + np.where(d["special"].astype(bool), "-S", ""),
        "chamber": "Senate", "race_type": np.where(d["polling_available"], "polled", "unpolled"),
        "dem_candidate": d["dem_candidate"], "gop_candidate": d["gop_candidate"],
        "incumbent": d["inc_signed"].map(INCUMBENT), "margin_model": d["center"], "margin_adjustment": 0.0,
        "margin": d["center"], "actual": d["actual"], "p_dem": d["p_dem"], "p_gop": 1 - d["p_dem"], "p_other": 0.0,
        "winner_party": np.where(d["actual"] > 0, "D", "R"), "treatment": "simulated"})
    return finish(races, thresholds)


# --- Scoring ---------------------------------------------------------------------------------------------------

def errors(rows: pd.DataFrame) -> pd.Series:
    """MAE, RMSE and bias (actual - margin; + = Democrats beat the forecast) over rows with an observed margin."""
    e = (rows["actual"] - rows["margin"]).dropna()
    return pd.Series({"margin n": len(e), "MAE": e.abs().mean(), "RMSE": np.sqrt((e ** 2).mean()), "bias (D+)": e.mean()})


def scorecard(races: pd.DataFrame) -> pd.DataFrame:
    """Per chamber and year (and pooled): calls, misses, Brier, seats called vs won, and the margin errors over all
    simulated races and over the ones not rated Safe."""
    groups = [*races.groupby(["chamber", "year"])] + [((c, "pooled"), g) for c, g in races.groupby("chamber")]
    rows = {}
    for (chamber, year), g in groups:
        resolved = g[g["winner_party"].notna()]
        contested = g[g["treatment"] == "simulated"]
        close = contested[~contested["rating"].str.startswith("Safe")]
        rows[(chamber, year)] = {
            "races": len(resolved), "called right": int(resolved["called_right"].sum()),
            "accuracy": resolved["called_right"].mean(), "misses": int((~resolved["called_right"].astype(bool)).sum()),
            "not Safe": len(close), "misses among not Safe": int((~close["called_right"].astype(bool)).sum()),
            "Brier": ((resolved["p_dem"] - (resolved["winner_party"] == "D")) ** 2).mean(),
            "expected D wins": resolved["p_dem"].sum(), "called D wins": int((resolved["predicted_winner_party"] == "D").sum()),
            "actual D wins": int((resolved["winner_party"] == "D").sum()),
            **errors(contested).add_prefix("all: "), **errors(close).add_prefix("not Safe: "),
        }
    return pd.DataFrame(rows).T.rename_axis(["chamber", "year"])


def rating_scorecard(races: pd.DataFrame) -> pd.DataFrame:
    """Per chamber and rating: how many races, the favorite's average probability, how often the favorite won, and
    which party won them."""
    resolved = races[races["winner_party"].notna()]
    table = resolved.groupby(["chamber", "rating"]).apply(lambda g: pd.Series({
        "races": len(g), "favorite's probability": g["party_probability"].mean(),
        "favorite won": g["called_right"].mean(), "misses": int((~g["called_right"].astype(bool)).sum()),
        "D won": int((g["winner_party"] == "D").sum()), "R won": int((g["winner_party"] == "R").sum()),
        "I won": int((g["winner_party"] == "I").sum())}), include_groups=False)
    order = {r: i for i, r in enumerate(report.RATING_ORDER)}
    return table.sort_index(key=lambda idx: idx.map(order) if idx.name == "rating" else idx)


def by_group(races: pd.DataFrame, column: str) -> pd.DataFrame:
    """Margin errors and call accuracy of the simulated races, by `column` (e.g. incumbent)."""
    contested = races[races["treatment"] == "simulated"]
    return contested.groupby(["chamber", column]).apply(lambda g: pd.Series({
        **errors(g), "accuracy": g["called_right"].mean()}), include_groups=False)


MISS_COLUMNS = ["year", "chamber", "race_id", "rating", "predicted_winner_party", "party_probability", "winner_party",
                "dem_candidate", "gop_candidate", "incumbent", "margin_model", "margin_adjustment", "margin", "actual"]


def misses(races: pd.DataFrame) -> pd.DataFrame:
    """Every resolved race whose favorite lost, most confident first."""
    wrong = races[races["winner_party"].notna() & ~races["called_right"].astype(bool)]
    return wrong.sort_values("party_probability", ascending=False)[MISS_COLUMNS].reset_index(drop=True)


# --- Same-party top two ----------------------------------------------------------------------------------------

def top_two_backtest(history: pd.DataFrame, *, weights_no_poll: dict, points_per_log: float, base_sd: float,
                     split_sd: float, nu: float) -> pd.DataFrame:
    """predict_top_two on each historical race (no polls), beside the November result. A race without fundraising
    (fundraising_missing) is predicted from the primary alone."""
    settings = {"points_per_log": points_per_log, "base_sd": base_sd, "split_sd": split_sd, "nu": nu}
    full = history[~history["fundraising_missing"]]
    bare = history[history["fundraising_missing"]].assign(individual_contribution_advantage=0.0)
    predicted = pd.concat([
        simulation.predict_top_two(full, weights_with_poll=weights_no_poll, weights_no_poll=weights_no_poll, **settings),
        simulation.predict_top_two(bare, weights_with_poll={"primary": 1.0}, weights_no_poll={"primary": 1.0},
                                   **settings)], ignore_index=True)
    winners = history[history["won"]].set_index(HOUSE_KEY)["candidate_name"]
    leader_share = history[history["primary_rank"] == 1].set_index(HOUSE_KEY)["general_share_top_two"]
    out = predicted.join(winners.rename("winner"), on=HOUSE_KEY).join(
        (2 * leader_share - 100).rename("actual_margin"), on=HOUSE_KEY)
    out["race_id"] = [simulation.race_id(s, d) for s, d in zip(out["state"], out["district"])]
    out["predicted"] = out["leader"].where(out["p_leader"] >= 0.5, out["runner_up"])
    out["called_right"] = out["predicted"] == out["winner"]
    out["error"] = out["actual_margin"] - out["mean"]
    out["inside_80"] = out["actual_margin"].between(out["interval_80_low"], out["interval_80_high"])
    return out.sort_values(HOUSE_KEY).reset_index(drop=True)


# --- Kalshi 2024 -----------------------------------------------------------------------------------------------

def kalshi_markets_2024(path=KALSHI_2024_PATH) -> pd.DataFrame:
    """The cached 2024 eve markets with race_id and run.py's market columns (yes_ask, no_ask = 1 - bid). Depth isn't
    known historically, so the quoted sizes are left unlimited. An event with no eve price (no candle before Election
    Day) is dropped."""
    eve = pd.read_csv(path)
    eve = eve[eve.groupby("event_ticker")["eve_price"].transform(lambda p: p.notna().any())]
    house = eve["chamber"] == "House"
    race_id = np.where(house, [simulation.race_id(s, int(d)) if h else "" for s, d, h in
                               zip(eve["state"], eve["district"].fillna(0), house)],
                       eve["state"] + "-SEN" + np.where(eve["special"].fillna(False).astype(bool), "-S", ""))
    bid, ask = eve["eve_yes_bid"], eve["eve_yes_ask"]
    return eve.assign(race_id=race_id, yes_bid=bid, yes_ask=ask.where((ask > 0) & (ask < 1)),
                      no_ask=(1 - bid).where((bid > 0) & (bid < 1)), yes_ask_size=np.inf, yes_bid_size=np.inf,
                      kalshi_prob=eve["eve_prob"], price=eve["eve_price"], volume=eve["volume_through_eve"])


def kalshi_compare(races: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Each 2024 race Kalshi listed, the model's p_dem beside Kalshi's eve P(D) (its D market's normalized price, or
    one minus the R market's when there's no D market), and the result."""
    by_party = markets.pivot_table(index="race_id", columns="party", values="kalshi_prob", aggfunc="sum")
    kalshi_p_dem = by_party.get("D", 1 - by_party.get("R"))
    out = races[races["year"] == 2024].merge(kalshi_p_dem.rename("kalshi_p_dem"), left_on="race_id", right_index=True)
    out["dem_won"] = out["winner_party"] == "D"
    out["model_minus_kalshi"] = out["p_dem"] - out["kalshi_p_dem"]
    return out


def bins(table: pd.DataFrame, column: str, edges=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)) -> pd.DataFrame:
    """Races grouped by `column` (a P(D)): how many, the mean of each forecast, and how often Democrats won."""
    group = pd.cut(table[column], list(edges), include_lowest=True)
    return table.groupby(group, observed=True).agg(races=("race_id", "size"), kalshi_p_dem=("kalshi_p_dem", "mean"),
                                                   model_p_dem=("p_dem", "mean"), dem_won=("dem_won", "mean"))


def kalshi_strategy_2024(races: pd.DataFrame, markets: pd.DataFrame, *, low: float, high: float, bankroll: float,
                         kelly_fraction: float, min_edge: float) -> pd.DataFrame:
    """run.py's strategy replayed on 2024: the model's race calls priced against Kalshi's eve quotes, toss-ups bet at
    `kelly_fraction` Kelly, each bet's expected profit (model probabilities) and realized profit (the result)."""
    races = races[(races["year"] == 2024) & races["race_id"].isin(markets["race_id"])]
    name = np.where(races["predicted_winner_party"] == "D", races["dem_candidate"].fillna(""),
                    races["gop_candidate"].fillna(""))
    track = pd.DataFrame({"race_id": races["race_id"], "chamber": races["chamber"],
                          "predicted_winner_party": races["predicted_winner_party"], "candidate_name": name,
                          "win_probability": races["party_probability"]}).reset_index(drop=True)
    table = odds.track_odds(track, markets[markets["race_id"].isin(track["race_id"])])
    bets = optimizer.kelly_bets(optimizer.tossups(table, low=low, high=high), markets, bankroll=bankroll,
                                kelly_fraction=kelly_fraction, min_edge=min_edge)
    result = bets["market_ticker"].map(markets.set_index("market_ticker")["result"])
    pays = (bets["side"] == "yes") == (result == "yes")
    return bets.assign(result=result, realized_profit=bets["contracts"] * (pays.astype(float) - bets["cost"]))


# --- Which adjusted candidates are running again ----------------------------------------------------------------

NAME_SUFFIXES = {"JR", "SR", "II", "III", "IV"}


def name_key(name: str) -> str:
    """FIRST LAST, uppercase ASCII, punctuation, middle names and suffixes dropped ("BRIAN K. FITZPATRICK" ->
    "BRIAN FITZPATRICK")."""
    import re
    import unicodedata
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().upper()
    tokens = [t for t in re.sub(r"[^A-Z\s-]", " ", text).split() if t not in NAME_SUFFIXES]
    return f"{tokens[0]} {tokens[-1]}" if tokens else ""


def returning_candidates(rows: pd.DataFrame, path=MEDSL_HOUSE_PATH) -> pd.DataFrame:
    """rows (year, state, district, party: the party of the candidate an adjustment is about) with that candidate's
    name, the earlier years they were on this seat's regular November ballot (any party label, so Minnesota's DFL
    counts), and returning: whether there is one. Special elections don't count."""
    m = pd.read_csv(path, dtype=str)
    m = m[(m["stage"] == "GEN") & (m["special"] != "TRUE") & (m["writein"] != "TRUE")]
    m = m.assign(year=m["year"].astype(int), district=m["district"].astype(int), votes=pd.to_numeric(m["candidatevotes"]),
                 key=m["candidate"].map(name_key),
                 party_group=np.select([m["party"].fillna("").str.contains("DEMOCRAT"),
                                        m["party"].fillna("").str.contains("REPUBLICAN")], ["D", "R"], ""))
    out = []
    for row in rows.to_dict("records"):
        seat = m[(m["state_po"] == row["state"]) & (m["district"] == row["district"])]
        ballot = seat[(seat["year"] == row["year"]) & (seat["party_group"] == row["party"])]
        candidate = ballot.sort_values("votes", ascending=False)["candidate"].iloc[0] if len(ballot) else None
        earlier = sorted(seat.loc[(seat["year"] < row["year"]) & (seat["key"] == name_key(candidate)), "year"].unique())
        out.append({**row, "candidate": candidate.title() if candidate else None,
                    "earlier_runs": ", ".join(map(str, earlier)), "returning": bool(earlier)})
    return pd.DataFrame(out)
