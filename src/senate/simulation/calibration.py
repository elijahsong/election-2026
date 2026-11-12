"""Chronological calibration backtest of the Senate Monte Carlo on the Stage 2 blend's out-of-sample predictions.

For each test year Y, the error SDs are estimated from the blend residuals of EARLIER years only (from 2016, the first
year with expert ratings), Y is simulated around its own out-of-sample blend predictions, and the draws are scored.

The panel is Y's scored races: observed D-vs-R contests only (imputed, third-party-runner-up and independent-vs-R
races, and AK 2022, are never predicted). Seat counts are therefore D wins on that MATCHED PANEL, never Senate totals,
and the simulated and actual counts cover identical races.

Metrics, per test year and pooled over races:
    brier, brier_competitive  mean (p_dem - dem_won)^2; p_dem = share of draws > 0. Competitive = ex ante,
                              |partisan_lean + gb_margin_2p_norm| <= 10, known before the election
    coverage_<l>, width_<l>   share of actual margins inside the central l interval of the draws, and its mean width
    seat_percentile           the actual panel D-win count's mid-rank in the draws, P(draw < actual) + P(= actual) / 2
"""

import numpy as np
import pandas as pd

import senate.regression as sr
from senate import stage1, stage2

from .errors import estimate_error_sds, load_regions, simulate_margins

FIRST_RESIDUAL_YEAR = 2016  # expert ratings start in 2016
TEST_YEARS = (2018, 2020, 2022, 2024)
LEVELS = (0.5, 0.8, 0.9, 0.95)


def blend_out_of_sample(config=stage1.SENATE_S1, years=range(2016, 2026, 2), data=None, polls=None,
                        ratings=None, weights=stage2.WEIGHTS) -> pd.DataFrame:
    """The Stage 2 blend's out-of-sample predictions for `years` (Stage 1 fit on earlier years only, blended with
    `weights`), with center (= prediction), competitive_ex_ante and inc_signed."""
    data = sr.load_data() if data is None else data
    if polls is None or ratings is None:
        polls, ratings = stage2.load_inputs()
    rolling = stage1.rolling_predictions(config, eval_years=list(years), data=data)
    oos = stage2.blend(pd.concat([r.predictions for r in rolling.values()], ignore_index=True), polls, ratings,
                       weights=weights)
    oos = oos.merge(data[[*sr.KEY, "partisan_lean", "gb_margin_2p_norm", "inc_signed"]], on=sr.KEY, how="left",
                    validate="one_to_one")
    oos["competitive_ex_ante"] = (oos["partisan_lean"] + oos["gb_margin_2p_norm"]).abs() <= sr.EX_ANTE_THRESHOLD
    return oos.assign(center=oos["prediction"], config=config.name, region=oos["state"].map(load_regions()))


def score_races(sims: np.ndarray, panel: pd.DataFrame) -> pd.DataFrame:
    """One row per race: p_dem, dem_won, the competitive flag, and inside_<l> / width_<l> for each level."""
    actual = panel["actual"].to_numpy()
    out = panel[[*sr.KEY, "polling_available", "competitive_ex_ante"]].assign(
        p_dem=(sims > 0).mean(axis=0), dem_won=actual > 0)
    for level in LEVELS:
        lo, hi = np.quantile(sims, [(1 - level) / 2, (1 + level) / 2], axis=0)
        out[f"inside_{level:g}"] = (actual >= lo) & (actual <= hi)
        out[f"width_{level:g}"] = hi - lo
    return out


def summarize(races: pd.DataFrame) -> pd.Series:
    sq = (races["p_dem"] - races["dem_won"]) ** 2
    comp = races["competitive_ex_ante"]
    return pd.Series({"n": len(races), "brier": sq.mean(), "n_competitive": int(comp.sum()),
                      "brier_competitive": sq[comp].mean(),
                      **{f"coverage_{l:g}": races[f"inside_{l:g}"].mean() for l in LEVELS},
                      **{f"width_{l:g}": races[f"width_{l:g}"].mean() for l in LEVELS}})


def seat_summary(sims: np.ndarray, panel: pd.DataFrame) -> pd.Series:
    """Matched-panel D wins: the draws' median and 80/90% intervals, the actual count and its mid-rank percentile."""
    seats = (sims > 0).sum(axis=1)
    actual = int((panel["actual"] > 0).sum())
    q = np.quantile(seats, [0.05, 0.1, 0.5, 0.9, 0.95], method="inverted_cdf")
    return pd.Series({"panel_races": len(panel), "actual_dem_wins": actual, "median": q[2], "p10": q[1], "p90": q[3],
                      "p05": q[0], "p95": q[4],
                      "seat_percentile": (seats < actual).mean() + (seats == actual).mean() / 2})


def backtest(oos: pd.DataFrame, model: str = "S-M0", test_years=TEST_YEARS, n_sims: int = 20_000,
             seed: int = 0) -> dict[str, pd.DataFrame]:
    """{"metrics": per test year and pooled, "seats": per test year, "sds": the SDs each year was simulated with,
    "races": per race}. Each test year's errors are estimated from years FIRST_RESIDUAL_YEAR .. Y - 2 only."""
    if oos["config"].nunique() != 1:
        raise ValueError("oos mixes Stage 1 configurations")
    rng = np.random.default_rng(seed)
    races, seats, sds_rows = [], {}, {}
    for year in test_years:
        train = oos[(oos["year"] >= FIRST_RESIDUAL_YEAR) & (oos["year"] < year)]
        if train.empty or train["year"].max() >= year:
            raise ValueError(f"No earlier residuals for {year}")
        sds, _ = estimate_error_sds(train, model)
        panel = oos[oos["year"] == year].reset_index(drop=True)
        sims = simulate_margins(panel, sds, n_sims, rng)
        races.append(score_races(sims, panel))
        seats[year] = seat_summary(sims, panel)
        sds_rows[year] = {**sds, "train_years": f"{train['year'].min()}-{train['year'].max()}"}
    races = pd.concat(races, ignore_index=True)
    metrics = pd.DataFrame({**{y: summarize(g) for y, g in races.groupby("year")}, "pooled": summarize(races)}).T
    return {"metrics": metrics, "seats": pd.DataFrame(seats).T, "sds": pd.DataFrame(sds_rows).T, "races": races}


def polling_group_diagnostics(oos: pd.DataFrame) -> pd.DataFrame:
    """For S-M1: polled vs unpolled races' counts, signed bias, MAE, and competitive share. Polling availability is
    correlated with competitiveness, so the two race scales also differ by how close the races are."""
    groups = oos.groupby(np.where(oos["polling_available"], "polled", "unpolled"))
    return groups.apply(lambda g: pd.Series({
        "n": len(g), "bias": g["residual"].mean(), "mae": g["residual"].abs().mean(),
        "sd": g["residual"].std(), "n_competitive_ex_ante": int(g["competitive_ex_ante"].sum()),
        "bias_competitive": g.loc[g["competitive_ex_ante"], "residual"].mean(),
        "mean_abs_prediction": g["prediction"].abs().mean()}), include_groups=False)
