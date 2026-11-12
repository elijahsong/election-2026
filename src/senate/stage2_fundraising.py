"""EXPERIMENTAL: a fundraising signal on top of the Senate Stage 2 blend (notebooks/senate/stage2/
05_fundraising_adjustment.ipynb). Production (stage2.blend, forecast.py) never imports this module; stage1's
specifications and stage2.WEIGHTS are unchanged.

Fundraising advantage, Democratic-positive, per race:

    fund_adv = log((D receipts + STABILIZER) / (R receipts + STABILIZER)),   STABILIZER = $100,000

Receipts are each nominee's two-year receipts from data/processed/senate_fec.csv (data-processing/fec_pre_general.py
and fec_finance.py). That is normally the pre-general (12G) report, through about 20 days before Election Day, or the
October quarterly without one. The cutoff rule is by date, not by source: a nominee's amount is used only if its
report's coverage ends before Election Day. That keeps 2026's candidate summaries (through mid-October). The one
exception is a nominee with $0 and no coverage dates at all: they filed nothing even by December 31, so $0 was also
the reported figure on Election Day. The stabilizer keeps a near-zero opponent from giving an extreme ratio. It was
fixed in advance (small next to the median nominee's $5-7M), not tuned.

Missing data (fund_status). fund_adv is NaN unless the status is "ok". A NaN means no fundraising signal: dF = 0 under
approach B, and approach A keeps the production three-component blend exactly.
    independent_race    an independent-vs-R contest (Orman KS 2014, McMullin UT 2022, Osborn NE 2024 and 2026): no D
                        nominee, and the independent's money is never used as a Democrat's. Not scored in any year
    no_nominee          a party had no nominee (uncontested by D or R; imputed targets, never scored)
    unresolved          a party's nominee isn't known (2026 only)
    no_fec_record       a nominee without a usable FEC record (the source's 0 is unknown, not a known $0)
    post_election_only  a nominee's only amount is a report that ends on or after Election Day: the full-cycle
                        summary through December 31, where no pre-general report was found (mostly 2006-2016)
Sanders and King (dem_is_independent) fill the D slot, as they do in the targets.

Approach A, a fourth blend component. margin_fundraising = a + b * fund_adv: an OLS fit of the actual D margin on
fund_adv over the observed races of 2006 through Y-2 (fit_fundraising_margin), a standalone fundraising forecast in
margin points. Weights are WEIGHTS_4. A race without expert ratings renormalizes the other weights, as stage2.blend
does.

Approach B, a capped adjustment to the production blend:

    M_new = M_baseline + dF,   dF = clip(slope_Y * fund_adv, -cap, +cap),   cap = CAPS[case]

The caps are 0.5 points with polls and 1.0 without. They are fixed and never tuned. slope_Y is the OLS slope of the
production blend's out-of-sample residuals (actual - blend) on fund_adv over CALIBRATION_START through Y-2
(fit_adjustment_slope). The fitted intercept absorbs the blend's average bias and is discarded, so dF = 0 at even
money. 2012-2014 have no expert ratings, so for those years the "production blend" is stage2.blend's renormalized
fallback (fundamentals and polls only).

Fundamentals-level diagnostic: the same slope fitted on the Stage 1 residuals (actual - fundamentals), added to
the fundamentals uncapped. It shows whether a gain at the regression level carries through the blend.

Every calibration is chronological: year Y's fits use years before Y only.
"""

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import senate.regression as sr
from senate import stage1, stage2

ROOT = Path(__file__).resolve().parents[2]
FEC_PATH = ROOT / "data/processed/senate_fec.csv"
KEY = sr.KEY

STABILIZER = 100_000.0  # dollars added to each side of the log ratio
CALIBRATION_START = 2012  # the first year with out-of-sample Stage 1 (and so blend) residuals
COMPONENTS_4 = [*stage2.COMPONENTS, "fundraising"]
WEIGHTS_4 = {
    "polling": {"fundamentals": 0.35, "polling": 0.54, "experts": 0.09, "fundraising": 0.02},
    "no_polling": {"fundamentals": 0.60, "polling": 0.00, "experts": 0.35, "fundraising": 0.05},
}
CAPS = {"polling": 0.5, "no_polling": 1.0}  # margin points


def election_day(year: int) -> date:
    """The Tuesday after the first Monday of November."""
    nov1 = date(year, 11, 1)
    return nov1 + timedelta(days=(0 - nov1.weekday()) % 7 + 1)


def fundraising_table(fec: pd.DataFrame | None = None, stabilizer: float = STABILIZER) -> pd.DataFrame:
    """One row per race of senate_fec.csv: KEY, the two nominees' receipts and report end dates, fund_status and
    fund_adv (module docstring)."""
    fec = pd.read_csv(FEC_PATH) if fec is None else fec
    status = {p: fec[f"{p}_match_status"] for p in ["dem", "gop"]}
    missing = {p: fec[f"{p}_fec_missing"].astype("boolean").fillna(False).astype(bool) for p in ["dem", "gop"]}
    eday = pd.to_datetime(fec["year"].map(election_day))
    pre_election = pd.Series(True, index=fec.index)
    for p in ["dem", "gop"]:
        end = pd.to_datetime(fec[f"{p}_coverage_end_date"])
        never_filed = end.isna() & fec[f"{p}_receipts"].eq(0)  # no report even by Dec 31: $0 on Election Day too
        pre_election &= (end < eday) | never_filed
    out = fec[KEY].copy()
    out["fund_status"] = np.select(  # first match wins
        [fec["ind_name"].notna(),
         (status["dem"] == "no_nominee") | (status["gop"] == "no_nominee"),
         (status["dem"] == "nominee_unresolved") | (status["gop"] == "nominee_unresolved"),
         missing["dem"] | missing["gop"],
         ~pre_election],
        ["independent_race", "no_nominee", "unresolved", "no_fec_record", "post_election_only"], default="ok")
    ok = out["fund_status"] == "ok"
    for p in ["dem", "gop"]:
        out[f"{p}_receipts"] = fec[f"{p}_receipts"].astype(float).where(ok)
        out[f"{p}_report_end"] = pd.to_datetime(fec[f"{p}_coverage_end_date"]).where(ok)
    if out.loc[ok, ["dem_receipts", "gop_receipts"]].lt(0).any().any():
        raise ValueError("Negative pre-election receipts")
    out["fund_adv"] = np.log((out["dem_receipts"] + stabilizer) / (out["gop_receipts"] + stabilizer))
    return out


def _before(rows: pd.DataFrame, year: int, column: str) -> pd.DataFrame:
    train = rows[(rows["year"] < year) & rows["fund_adv"].notna() & rows[column].notna()]
    if train.empty:
        raise ValueError(f"No calibration rows before {year}")
    return train


def fit_fundraising_margin(margin_rows: pd.DataFrame, year: int):
    """Approach A's mapping: OLS actual ~ fund_adv over margin_rows (observed races, columns year, actual, fund_adv)
    before `year`."""
    return smf.ols("actual ~ fund_adv", data=_before(margin_rows, year, "actual")).fit()


def fit_adjustment_slope(residual_rows: pd.DataFrame, year: int, residual: str = "residual"):
    """Approach B's mapping: OLS residual ~ fund_adv over residual_rows (year, fund_adv, residual) before `year`.
    Only the slope is used."""
    return smf.ols(f"{residual} ~ fund_adv", data=_before(residual_rows, year, residual)).fit()


def capped_adjustment(fund_adv: pd.Series, slope: float, polling_available: pd.Series, caps: dict = CAPS):
    """(dF, the uncapped slope * fund_adv). dF is clipped to +-caps["polling"] with polls and +-caps["no_polling"]
    without, and is 0 where fund_adv is missing."""
    uncapped = slope * fund_adv
    cap = np.where(polling_available, caps["polling"], caps["no_polling"])
    return uncapped.clip(-cap, cap).fillna(0.0), uncapped


def four_component_blend(baseline: pd.DataFrame, margin_fundraising: pd.Series, weights: dict = WEIGHTS_4):
    """Approach A's prediction. baseline is stage2.blend's output. A race without margin_fundraising keeps its
    baseline prediction; a race without ratings renormalizes the weights of the components it has."""
    for case, w in weights.items():
        if set(w) != set(COMPONENTS_4) or min(w.values()) < 0 or not np.isclose(sum(w.values()), 1.0):
            raise ValueError(f"{case} weights must cover {COMPONENTS_4}, be non-negative and sum to 1; got {w}")
    margins = baseline[[f"margin_{c}" for c in stage2.COMPONENTS]].assign(margin_fundraising=margin_fundraising)
    available = margins.notna().to_numpy()
    w = pd.DataFrame([weights[case] for case in baseline["weight_case"]], index=baseline.index)[COMPONENTS_4]
    w = w.where(available, 0.0)
    w = w.div(w.sum(axis=1), axis=0)
    prediction = (margins.fillna(0).to_numpy() * w.to_numpy()).sum(axis=1)
    return pd.Series(prediction, index=baseline.index).where(margin_fundraising.notna(), baseline["prediction"])


def add_variants(rows: pd.DataFrame, history: pd.DataFrame, margin_rows: pd.DataFrame, year: int,
                 caps: dict = CAPS, weights: dict = WEIGHTS_4) -> tuple[pd.DataFrame, dict]:
    """rows: one year's production blend with fund_adv. history: earlier years' blend rows (with residual,
    residual_fundamentals, fund_adv). margin_rows: observed races (year, actual, fund_adv). Returns (rows with the
    variants' predictions, the year's calibration). Everything is fitted on years before `year`."""
    slope_fit = fit_adjustment_slope(history, year, "residual")
    fund_fit = fit_adjustment_slope(history, year, "residual_fundamentals")
    margin_fit = fit_fundraising_margin(margin_rows, year)
    out = rows.copy()
    out["margin_fundraising"] = margin_fit.predict(out).where(out["fund_adv"].notna())
    out["prediction_A"] = four_component_blend(out, out["margin_fundraising"], weights)
    out["delta_F"], out["delta_F_uncapped"] = capped_adjustment(out["fund_adv"], slope_fit.params["fund_adv"],
                                                                out["polling_available"], caps)
    cap = np.where(out["polling_available"], caps["polling"], caps["no_polling"])
    out["at_cap"] = out["fund_adv"].notna() & (out["delta_F_uncapped"].abs() >= cap)
    out["prediction_B"] = out["prediction"] + out["delta_F"]
    out["fundamentals_adj"] = out["margin_fundamentals"] + (fund_fit.params["fund_adv"] * out["fund_adv"]).fillna(0)
    calibration = {
        "year": year,
        "B_slope": slope_fit.params["fund_adv"], "B_slope_se": slope_fit.bse["fund_adv"], "B_n": int(slope_fit.nobs),
        "B_years": f"{int(history['year'].min())}-{year - 2}",
        "S1_slope": fund_fit.params["fund_adv"], "S1_slope_se": fund_fit.bse["fund_adv"],
        "A_intercept": margin_fit.params["Intercept"], "A_slope": margin_fit.params["fund_adv"],
        "A_n": int(margin_fit.nobs), "A_years": f"{int(margin_rows['year'].min())}-{year - 2}",
    }
    return out, calibration


def margin_history(data: pd.DataFrame, fund: pd.DataFrame) -> pd.DataFrame:
    """Observed D-vs-R races (every year with targets) with actual and fund_adv: approach A's training rows."""
    rows = data.loc[data["observed"], [*KEY, sr.TARGET]].rename(columns={sr.TARGET: "actual"})
    return rows.merge(fund[[*KEY, "fund_adv"]], on=KEY, how="left", validate="one_to_one")


def production_blend(stage1_predictions: pd.DataFrame, polls: pd.DataFrame, ratings: pd.DataFrame,
                     fund: pd.DataFrame) -> pd.DataFrame:
    """stage2.blend with the production WEIGHTS, plus fund_adv, fund_status and residual_fundamentals."""
    out = stage2.blend(stage1_predictions, polls, ratings)
    out = out.merge(fund[[*KEY, "fund_adv", "fund_status"]], on=KEY, how="left", validate="one_to_one")
    out["residual_fundamentals"] = out["actual"] - out["margin_fundamentals"]
    return out


def history(last_year: int, data: pd.DataFrame, polls: pd.DataFrame, ratings: pd.DataFrame, fund: pd.DataFrame,
            config=stage1.SENATE_S1) -> pd.DataFrame:
    """The production blend, out of sample, for CALIBRATION_START..last_year (Stage 1 fit on earlier years only)."""
    rolling = stage1.rolling_predictions(config, eval_years=range(CALIBRATION_START, last_year + 1, 2), data=data)
    return production_blend(pd.concat([r.predictions for r in rolling.values()], ignore_index=True), polls, ratings,
                            fund)


def backtest(eval_years, data: pd.DataFrame, polls: pd.DataFrame, ratings: pd.DataFrame, fund: pd.DataFrame,
             config=stage1.SENATE_S1, caps: dict = CAPS, weights: dict = WEIGHTS_4):
    """(per-race table for eval_years, per-year calibration table). Each eval year's production blend is adjusted
    with fits on earlier years' out-of-sample blends."""
    blends = history(max(eval_years), data, polls, ratings, fund, config)
    margins = margin_history(data, fund)
    results = [add_variants(blends[blends["year"] == y], blends, margins, y, caps, weights) for y in eval_years]
    return (pd.concat([r for r, _ in results], ignore_index=True),
            pd.DataFrame([c for _, c in results]).set_index("year"))


def forecast(data: pd.DataFrame, polls: pd.DataFrame, ratings: pd.DataFrame, fund: pd.DataFrame,
             year: int = stage1.FORECAST_YEAR, config=stage1.SENATE_S1, caps: dict = CAPS,
             weights: dict = WEIGHTS_4):
    """(year's production blend with each variant, the calibration), every fit on years before `year`."""
    rows = production_blend(stage1.forecast(config, year, data=data).predictions, polls, ratings, fund)
    return add_variants(rows, history(year - 2, data, polls, ratings, fund, config), margin_history(data, fund),
                        year, caps, weights)


def scores_by(rows: pd.DataFrame, data: pd.DataFrame, predictions: dict[str, str], by: str) -> pd.DataFrame:
    """stage2.score_table within each value of rows[by] (e.g. year, polling_available), stacked."""
    return pd.concat({value: stage2.score_table(g, data, predictions) for value, g in rows.groupby(by)},
                     names=[by, "variant"])
