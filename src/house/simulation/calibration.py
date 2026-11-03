"""Chronological calibration backtest for the hierarchical simulation, and the metrics it reports.

For each test election, the Stage 2 error SDs are estimated (hierarchical.estimate_error_sds) from the out-of-sample
Stage 1 residuals of EARLIER elections only. Every one of the election's races is then simulated under the race rules
(races.simulate_races: the D-vs-R races from their own Stage 1 out-of-sample margins, the others by their lineup), and
the draws are scored against what happened. Nothing here depends on the Stage 1 specification: `oos` (out_of_sample)
carries the residuals of whichever configuration produced it, and backtest refuses a table that mixes configurations.

Only pre-election information decides how a race is simulated: its lineup (races.historical_races) and Stage 1's
margin. The winner and the actual margin are read only to score the draws.

Metrics, per test election, from that election's simulation draws only. Scoring covers the races with a resolved
winner (all 435 except a voided election, NC-9 2018); a race's p_dem is the share of draws Democrats win it:

    Brier (all races)     mean (p_dem - dem_won)^2 over every resolved race, the rule-decided ones included
    contested Brier       the same over the simulated D-vs-R races
    competitive Brier     the same over competitive_ex_ante races (|partisan_lean + eve generic ballot| <= 10, all
                          known before the election; Stage 1's flag, so D-vs-R races only)
    coverage_<level>      share of observed two-party margins inside the central <level> interval of their draws, over
                          the simulated races with a valid observed margin (no imputed target) only
    width_<level>         mean width of that interval, points
    panel_dem_seats       draws of the number of D wins among the resolved races (the matched panel)
    panel_actual_dem_seats  D wins among the SAME races; its percentile in the draws is mid-rank:
                          P(draw < actual) + P(draw = actual) / 2. house_seats has every party's seats over all 435
    seat median/intervals quantiles of panel_dem_seats by the inverted CDF (seat_quantile), so each is a seat count
                          that was actually drawn; the table and the seat plot both read them from seat_summary
    reliability           districts binned by p_dem (RELIABILITY_EDGES): mean p_dem vs the share D won, per bin
    call                  the most likely ballot party (D, R or other; call_party); a wrong call is one whose winner
                          differs. competitive_stats tabulates calls against winners (the confusion matrix) with
                          accuracy, Brier, and coverage for all races, the contested ones, and the competitive_ex_ante
                          and competitive_ex_post subsets. ex_post (|actual margin| <= 10) uses the result, so it
                          describes, never forecasts.

Stage 2 training years. By default each test year trains on every earlier election in oos: the main backtest scores
2020-2024 on Stage 1 residuals from 2018 on. backtest's train_years fixes them instead, for the separate 2018 test:
Stage 2 fit on the 2012-2016 Stage 1 out-of-sample residuals and scored on 2018. It is kept apart from the main
backtest, and never feeds it or the 2026 fit, because pre-2018 district errors are far larger (residual SD about
10-11 points in 2012-2016, against 6-7 from 2018 on, whether Stage 1 is fit on the years before or on every other
year), so it tests how Stage 2 does when its training elections come from a noisier period, not the production setup.

Simulated and observed seat counts always cover identical races: the resolved ones. A race's winner is targets.csv's
winner_party (D, R, or 3P, counted as other).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from house.regression.schema import KEY

from .factors import (
    DEMOGRAPHIC_PRIOR_ELECTIONS, REGIONAL_PRIOR_ELECTIONS, attach_factors, estimate_factor_sds,
)
from .hierarchical import t_scale
from .races import DEM, GOP, OTHER, attach_predictions, simulate_races

FLAGS = ["competitive_ex_ante", "competitive_ex_post"]
# Win-probability bins for reliability: the two outer bins hold the safe seats, the inner four the races in play
RELIABILITY_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


def out_of_sample(rolling: dict, scored_only: bool = True) -> pd.DataFrame:
    """Stage 1's rolling out-of-sample predictions ({year: Stage1Result}, stage1.rolling_predictions) stacked into one
    row per district-year: the predictions' columns (KEY, prediction, actual, residual, ...), Stage 1's competitive
    flags, and config (the Stage 1 configuration's name). By default rows without an actual are dropped, leaving the
    scored residuals; scored_only=False keeps every prediction (e.g. the unscored races.lineup_targets rows)."""
    frames = []
    for result in rolling.values():
        flags = result.scored[[*KEY, *FLAGS]]
        frames.append(result.predictions.merge(flags, on=KEY, how="left", validate="one_to_one")
                      .assign(config=result.config.name))
    out = pd.concat(frames, ignore_index=True)
    if scored_only:  # every scored row has its flags
        return out[out["actual"].notna()].astype({flag: bool for flag in FLAGS}).reset_index(drop=True)
    return out.astype({flag: "boolean" for flag in FLAGS})


def win_probability(sims: np.ndarray) -> np.ndarray:
    """Per district (column), the share of draws with a Democratic margin above 0."""
    return (sims > 0).mean(axis=0)


def brier(probability, outcome) -> float:
    """Mean squared difference between probabilities and 0/1 outcomes."""
    probability, outcome = np.asarray(probability, dtype=float), np.asarray(outcome, dtype=float)
    return float(np.mean((probability - outcome) ** 2))


def central_interval(sims: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    """(lower, upper) per district: the central `level` interval of its draws."""
    lower, upper = np.quantile(sims, [(1 - level) / 2, (1 + level) / 2], axis=0)
    return lower, upper


def seat_percentile(draws: np.ndarray, actual: int) -> float:
    """Mid-rank percentile of `actual` among `draws`, in [0, 100]."""
    return float(100 * ((draws < actual).mean() + 0.5 * (draws == actual).mean()))


def seat_quantile(draws: np.ndarray, q):
    """Quantile(s) of integer seat draws by the inverted CDF: the smallest drawn count with at least a share q of the
    draws at or below it. Unlike linear interpolation, it never lands between two seat counts."""
    return np.quantile(draws, q, method="inverted_cdf")


def seat_summary(draws: np.ndarray, actual: int | None = None, levels=(0.8, 0.95)) -> dict:
    """The seat numbers the calibration table and plot_seat_distribution report, from one set of seat draws:
    {"median": int, "intervals": {level: (lower, upper)}, "actual": int | None, "percentile": float | None}."""
    intervals = {level: tuple(int(x) for x in seat_quantile(draws, [(1 - level) / 2, (1 + level) / 2]))
                 for level in levels}
    return {"median": int(seat_quantile(draws, 0.5)), "intervals": intervals, "actual": actual,
            "percentile": None if actual is None else seat_percentile(draws, actual)}


def call_party(districts: pd.DataFrame) -> pd.Series:
    """The simulation's call per race: the ballot party (D, R or other) that wins the most draws."""
    return districts[["p_dem", "p_gop", "p_other"]].set_axis(["D", "R", "other"], axis=1).idxmax(axis=1)


def coverage(districts: pd.DataFrame, level: float) -> tuple[float, float]:
    """(share of observed margins inside the central `level` interval, mean interval width in points), over the races
    with both an interval (simulated) and a valid observed margin."""
    pct = round(100 * level)
    d = districts[districts["actual"].notna() & districts[f"lower_{pct}"].notna()]
    inside = (d["actual"] >= d[f"lower_{pct}"]) & (d["actual"] <= d[f"upper_{pct}"])
    return float(inside.mean()), float((d[f"upper_{pct}"] - d[f"lower_{pct}"]).mean())


def scored(districts: pd.DataFrame, subset: str = "all") -> pd.DataFrame:
    """The races scored in `subset`: every resolved race ("all"), the simulated ones ("contested"), or a
    competitive flag's ("competitive_ex_ante", "competitive_ex_post")."""
    d = districts[districts["winner_resolved"]]
    if subset == "all":
        return d
    if subset == "contested":
        return d[d["treatment"] == "simulated"]
    return d[d[subset].eq(True)]


def reliability(districts: pd.DataFrame, edges=RELIABILITY_EDGES) -> pd.DataFrame:
    """Districts (p_dem, dem_won) binned by simulated Democratic win probability: per bin, n, forecast (mean p_dem),
    and observed (share D won). Calibrated probabilities put observed near forecast in every bin."""
    districts = districts[districts["winner_resolved"]] if "winner_resolved" in districts else districts
    bins = pd.cut(districts["p_dem"], edges, include_lowest=True)
    return (districts.groupby(bins, observed=True)
            .agg(n=("p_dem", "size"), forecast=("p_dem", "mean"), observed=("dem_won", "mean"))
            .rename_axis("p_dem bin"))


@dataclass
class YearCalibration:
    """One test election: its Stage 2 fit (from earlier elections only) and its scored draws, for every race."""
    year: int
    training_years: list[int]      # the elections whose Stage 1 residuals fit this year's Stage 2 SDs
    error_sd: dict[str, float]
    error_scale: dict[str, float]
    districts: pd.DataFrame        # one row per race: KEY, race_type, treatment, prediction_source, prediction,
    #                                actual, competitive flags, p_dem, p_gop, p_other, winner_party, winner_resolved,
    #                                dem_won, lower/upper_<pct> (simulated races only)
    panel_dem_seats: np.ndarray    # (n_sims,) simulated D wins among the resolved races (the matched panel)
    house_seats: dict[str, np.ndarray]  # (n_sims,) seats per ballot party (D, R, other) over every race
    house_size: int
    error_details: pd.DataFrame | None = None  # factors.estimate_factor_sds' details: how each SD was estimated
    winners: np.ndarray | None = None          # (n_sims, races) int8 party codes, districts' row order (group_seats)
    mean_error: np.ndarray | None = None       # (n_sims,) simulated error averaged over the simulated races

    @property
    def panel_actual_dem_seats(self) -> int:
        """D wins among the resolved races."""
        return int(self.districts.loc[self.districts["winner_resolved"], "dem_won"].sum())

    @property
    def panel_size(self) -> int:
        """Races with a resolved winner (simulated and observed seats are counted over these alike)."""
        return int(self.districts["winner_resolved"].sum())

    @property
    def unresolved(self) -> list[str]:
        """The races without an observed winner, as "ST-district"."""
        d = self.districts[~self.districts["winner_resolved"]]
        return [f"{state}-{district}" for state, district in zip(d["state"], d["district"])]

    def seat_summary(self, levels=(0.8, 0.95)) -> dict:
        """seat_summary of this election's panel_dem_seats against its panel_actual_dem_seats."""
        return seat_summary(self.panel_dem_seats, self.panel_actual_dem_seats, levels)

    @property
    def mean_signed_error(self) -> float:
        """mean(actual - prediction) over the races with a valid observed margin: the realized national direction."""
        d = self.districts[self.districts["actual"].notna()]
        return float((d["actual"] - d["prediction"]).mean())


def _subset_brier(districts: pd.DataFrame, subset: str) -> float:
    d = scored(districts, subset)
    return brier(d["p_dem"], d["dem_won"])


@dataclass
class Calibration:
    config: str
    levels: tuple[float, ...]
    years: dict[int, YearCalibration]

    @property
    def components(self) -> tuple[str, ...]:
        """The error components simulated: national, state, district, then region and/or demographic."""
        return tuple(next(iter(self.years.values())).error_sd)

    def table(self) -> pd.DataFrame:
        """The compact table: one row per test election. Seat columns come from YearCalibration.seat_summary, the
        same numbers plot_seat_distribution draws, over the resolved races ("434 of 435" when one is unresolved)."""
        rows = {}
        for y, year in self.years.items():
            d = year.districts
            seats = year.seat_summary(self.levels)
            row = {"stage 2 training": ", ".join(map(str, year.training_years)),
                   "matched panel": f"{year.panel_size} of {year.house_size}",
                   "unresolved": ", ".join(year.unresolved),
                   "Brier (all races)": _subset_brier(d, "all"),
                   "contested Brier": _subset_brier(d, "contested"),
                   "competitive Brier": _subset_brier(d, "competitive_ex_ante")}
            row |= {f"{round(100 * level)}% coverage": coverage(d, level)[0] for level in self.levels}
            row["median panel_dem_seats"] = seats["median"]
            row |= {f"{round(100 * level)}% panel seat interval": f"{lower}–{upper}"
                    for level, (lower, upper) in seats["intervals"].items()}
            row |= {"panel_actual_dem_seats": seats["actual"], "actual-seat percentile": seats["percentile"]}
            rows[y] = row
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("test year")

    def diagnostics(self) -> pd.DataFrame:
        """Races by treatment and prediction source, coverage sample sizes, interval widths, the fitted SDs, and each
        election's realized national direction."""
        rows = {}
        for y, year in self.years.items():
            d = year.districts
            row = {**{f"n {t}": int((d["treatment"] == t).sum())
                      for t in ["simulated", "fixed", "third_party", "same_party"]},
                   "n fallback": int(d["prediction_source"].str.endswith("_fallback").sum()),
                   "n observed margins": int(d["actual"].notna().sum()),
                   "n competitive": int(d["competitive_ex_ante"].eq(True).sum()),
                   "mean_signed_error": year.mean_signed_error,
                   **{f"{level} sd": sd for level, sd in year.error_sd.items()},
                   **{f"{level} scale": scale for level, scale in year.error_scale.items()}}
            if year.mean_error is not None:  # national-level overlap of the factors (factors.py docstring)
                row["sim sd of mean error"] = float(year.mean_error.std())
            row |= {f"{round(100 * level)}% mean width": coverage(d, level)[1] for level in self.levels}
            rows[y] = row
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("test year")

    def wrong_calls(self) -> pd.DataFrame:
        """Every resolved race whose call (call_party) missed the winner, most confident misses first within a year:
        KEY, race_type, called and won (D / R / other), the win probabilities, Stage 1's prediction and the actual
        margin (blank for races decided by rule), and the competitive flags."""
        frames = []
        for year in self.years.values():
            d = scored(year.districts).assign(called=lambda x: call_party(x), won=lambda x: x["winner_party"])
            wrong = d[d["called"] != d["won"]]
            confidence = wrong[["p_dem", "p_gop", "p_other"]].max(axis=1)
            frames.append(wrong.assign(confidence=confidence).sort_values("confidence", ascending=False))
        columns = [*KEY, "race_type", "called", "won", "p_dem", "p_gop", "p_other", "prediction", "actual", *FLAGS]
        return pd.concat(frames, ignore_index=True)[columns]

    def competitive_stats(self) -> pd.DataFrame:
        """Calls against winners, by (test year, subset): every resolved race, the contested (simulated) ones,
        competitive_ex_ante (known before the election), and competitive_ex_post (|actual margin| <= 10, hindsight).
        The confusion-matrix cells (called D / won D, ...), wrong calls, accuracy, Brier, and interval coverage."""
        rows = {}
        for y, year in self.years.items():
            for subset in ["all", "contested", *FLAGS]:
                s = scored(year.districts, subset)
                called, won = call_party(s), s["winner_party"]
                row = {"n": len(s)}
                row |= {f"called {c}, won {w}": int(((called == c) & (won == w)).sum())
                        for c in ["D", "R"] for w in ["D", "R"]}
                row["any other"] = int(((called == "other") | (won == "other")).sum())
                row |= {"wrong calls": int((called != won).sum()), "accuracy": float((called == won).mean()),
                        "Brier": brier(s["p_dem"], s["dem_won"])}
                row |= {f"{round(100 * level)}% coverage": coverage(s, level)[0] for level in self.levels}
                rows[(y, subset)] = row
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["test year", "subset"])

    def seats(self) -> pd.DataFrame:
        """Median simulated D seats against the actual count, per test election, over the resolved races: the
        difference, the actual count's percentile, and the D seats the race calls add up to."""
        rows = {}
        for y, year in self.years.items():
            seats = year.seat_summary(self.levels)
            called = int((call_party(scored(year.districts)) == "D").sum())
            rows[y] = {"matched panel": f"{year.panel_size} of {year.house_size}",
                       "unresolved": ", ".join(year.unresolved),
                       "median panel_dem_seats": seats["median"], "panel_actual_dem_seats": seats["actual"],
                       "median − actual": seats["median"] - seats["actual"],
                       "actual-seat percentile": seats["percentile"],
                       "called D seats": called, "called − actual": called - seats["actual"]}
        return pd.DataFrame.from_dict(rows, orient="index").rename_axis("test year")

    def reliability(self, edges=RELIABILITY_EDGES) -> pd.DataFrame:
        """reliability per test election, indexed by (test year, p_dem bin), over the resolved races."""
        return pd.concat({y: reliability(year.districts, edges) for y, year in self.years.items()},
                         names=["test year"])

    def pooled(self) -> pd.Series:
        """Race-level metrics pooled over every test election (seat counts are not pooled)."""
        d = pd.concat([year.districts for year in self.years.values()], ignore_index=True)
        out = {"races scored": int(d["winner_resolved"].sum()), "Brier (all races)": _subset_brier(d, "all"),
               "contested Brier": _subset_brier(d, "contested"),
               "competitive Brier": _subset_brier(d, "competitive_ex_ante"),
               "observed margins": int(d["actual"].notna().sum())}
        for level in self.levels:
            pct = round(100 * level)
            out[f"{pct}% coverage"], out[f"{pct}% mean width"] = coverage(d, level)
        return pd.Series(out, name=self.config)


def group_seats(calibration: Calibration, column: str, groups: pd.DataFrame | None = None) -> pd.DataFrame:
    """Seat calibration within groups of races: per (test year, group), the resolved races, the median simulated D
    seats and central intervals (seat_summary), the actual D seats among the same races, its mid-rank percentile, and
    whether each interval holds it. A group's seats move together only as far as the error model correlates its races,
    so too few actual counts inside their intervals across groups means the joint distribution is too narrow.

    column names the grouping: a column of the districts (e.g. competitive_ex_ante), or one of `groups` (KEY + column,
    e.g. every district-year's region or college tercile), which replaces any column of that name the districts carry.
    The same grouping scores every hierarchy alike."""
    rows = {}
    for y, year in calibration.years.items():
        d = year.districts[KEY].copy()
        if groups is None:
            d[column] = year.districts[column].to_numpy()
        else:
            d = d.merge(groups[[*KEY, column]], on=KEY, how="left", validate="one_to_one")
        resolved = year.districts["winner_resolved"].to_numpy()
        dem_won = year.districts["dem_won"].to_numpy()
        for group in sorted(d[column].dropna().unique()):
            columns = resolved & d[column].eq(group).to_numpy()
            seats = seat_summary((year.winners[:, columns] == DEM).sum(axis=1), int(dem_won[columns].sum()),
                                 calibration.levels)
            row = {"races": int(columns.sum()), "median D seats": seats["median"], "actual D seats": seats["actual"],
                   "actual-seat percentile": seats["percentile"]}
            for level, (lower, upper) in seats["intervals"].items():
                pct = round(100 * level)
                row |= {f"{pct}% seat interval": f"{lower}–{upper}",
                        f"inside {pct}%": lower <= seats["actual"] <= upper, f"{pct}% seat width": upper - lower}
            rows[(y, group)] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["test year", column])


def backtest(oos: pd.DataFrame, test_years, *, races: pd.DataFrame, predictions: pd.DataFrame, nu: float,
             n_sims: int, seed: int, house_size: int, national_prior_sd: float, national_prior_elections: float,
             levels: tuple[float, ...] = (0.8, 0.95), train_years=None,
             fallback_predictions: pd.DataFrame | None = None, regions: pd.Series | None = None,
             loadings: pd.DataFrame | None = None, regional_prior_elections: float = REGIONAL_PRIOR_ELECTIONS,
             demographic_prior_elections: float = DEMOGRAPHIC_PRIOR_ELECTIONS) -> Calibration:
    """Calibrate the race-level simulation chronologically: for each test year, estimate the error SDs from the
    elections before it in `oos` (out_of_sample), simulate every race of the year (races.simulate_races), and score
    the draws.

    races is races.historical_races (every year's lineups, winners, observed margins). predictions has the
    configuration's Stage 1 predictions for the simulated races (out_of_sample(rolling, scored_only=False), so the
    races.lineup_targets rows are included); fallback_predictions, Model A's, fills races the configuration can't
    predict. train_years, when given, fixes the Stage 2 training elections for every test year (the separate 2018
    test uses 2012-2016); each must be in oos and precede every test year. By default a test year trains on every
    earlier election in oos.

    Each test year draws from np.random.default_rng([seed, test_year]), so its result doesn't depend on which other
    years or configurations run, and is reproduced exactly by the same seed. house_size labels the panels
    (e.g. "434 of 435").

    The error model (factors.py): H0 by default. regions (factors.load_regions) adds the political-region shock,
    loadings (factors.demographic_loadings) the demographic one, each fit by factors.estimate_factor_sds on the same
    training residuals with its zero prior (regional_ / demographic_prior_elections). With neither, every number is
    the H0 backtest's."""
    configs = oos["config"].unique()
    if len(configs) != 1:
        raise ValueError(f"oos mixes Stage 1 configurations: {list(configs)}; calibrate each separately")
    if oos.duplicated(KEY).any():
        raise ValueError("Duplicate district-years in oos")
    if train_years is not None:
        train_years = sorted(int(y) for y in train_years)
        missing = sorted(set(train_years) - set(oos["year"]))
        if missing:
            raise ValueError(f"train_years {missing} have no rows in oos")
        if max(train_years) >= min(test_years):
            raise ValueError(f"train_years {train_years} must all precede every test year {list(test_years)}")
    years = {}
    for test_year in test_years:
        train = oos[oos["year"].isin(train_years)] if train_years is not None else oos[oos["year"] < test_year]
        year_races = races[races["year"] == test_year].reset_index(drop=True)
        if train.empty:
            raise ValueError(f"No election before {test_year} to estimate its error scales from")
        if year_races.empty:
            raise ValueError(f"No races for {test_year}")
        sds, details = estimate_factor_sds(train, regions=regions, loadings=loadings,
                                           national_prior_sd=national_prior_sd,
                                           national_prior_elections=national_prior_elections,
                                           regional_prior_elections=regional_prior_elections,
                                           demographic_prior_elections=demographic_prior_elections)
        scales = {level: t_scale(sd, nu) for level, sd in sds.items()}

        # how each race is simulated uses only its lineup and Stage 1's margin; winners and actuals score it after
        in_year = predictions[predictions["year"] == test_year]
        fallback = None if fallback_predictions is None else \
            fallback_predictions[fallback_predictions["year"] == test_year]
        point = attach_predictions(year_races, in_year, primary_source=str(configs[0]), fallback=fallback)
        if regions is not None or loadings is not None:
            point = attach_factors(point, regions=regions, loadings=loadings)
        draws = simulate_races(point, scales, nu, n_sims, np.random.default_rng([seed, test_year]))

        districts = point.merge(in_year[[*KEY, *FLAGS]], on=KEY, how="left", validate="one_to_one")
        districts["p_dem"], districts["p_gop"], districts["p_other"] = (draws.win_probability(code)
                                                                        for code in (DEM, GOP, OTHER))
        districts["winner_party"] = districts["winner_party"].replace({"3P": "other"})
        districts["dem_won"] = districts["winner_party"].eq("D")
        simulated = (districts["treatment"] == "simulated").to_numpy()
        for level in levels:
            pct = round(100 * level)
            lower, upper = np.full(len(districts), np.nan), np.full(len(districts), np.nan)
            lower[simulated], upper[simulated] = central_interval(draws.margins[:, simulated], level)
            districts[f"lower_{pct}"], districts[f"upper_{pct}"] = lower, upper
        years[test_year] = YearCalibration(
            year=test_year, training_years=sorted(int(y) for y in train["year"].unique()), error_sd=sds,
            error_scale=scales, districts=districts,
            panel_dem_seats=draws.seats(DEM, columns=districts["winner_resolved"]),
            house_seats={name: draws.seats(code) for name, code in [("D", DEM), ("R", GOP), ("other", OTHER)]},
            house_size=house_size, error_details=details, winners=draws.winners,
            mean_error=(draws.margins[:, simulated] - point["prediction"].to_numpy()[simulated]).mean(axis=1))
    return Calibration(config=str(configs[0]), levels=tuple(levels), years=years)
