"""Experimental lagged Senate performance predictor (a controlled ablation; production S1 is unchanged).

    r_{s,t}  = M^actual_{s,t} - M^S1_{s,t}               a past race's residual against fundamentals
    L_{s,t}  = 0.75 r_{s,recent} + 0.25 r_{s,previous}   the state's two most recent PRIOR Senate elections

Construction, all fixed in advance:
- Residuals (oos_residuals): actual minus S1's rolling out-of-sample prediction, i.e. S1 fit only on the years before
  the residual's own election (senate.stage1.rolling_predictions). So no residual uses its own result, and none used
  for a target year t uses anything from t or later. The first year with one is 2010: a fit on 2006 alone can't
  separate the generic ballot (constant within a year) from the intercept. Only observed D-vs-R races have
  residuals; imputed, third-party and excluded races (EXCLUDED_RACES, AK 2022) don't.
- Elections are state-years, ordered by year. Both classes and specials count. A state-year with two races (a regular
  election and a special on the same ballot) is one election, whose residual is the mean of its two races' residuals,
  and it requires both.
- Lags (lagged_performance): for a target race in year t, recent and previous are the state's last two Senate
  state-years before t (never t itself, so a same-day special is not a lag of the regular race). L needs both
  residuals. If either is unavailable, L is missing, with lag_missing_reason; it is never set to 0, and an older
  election is never substituted.

The S1 + L experiment (fit_predict_lagged): S1's formula plus lagged_performance, with inc_signed still fixed at +2.5
and one extra coefficient (the persistence coefficient) estimated by the same equal-year WLS. Two treatments of a
missing L (missing_lag):
- "exclude": training rows are observed races of earlier years that have L, and a scored race without L keeps S1's
  own prediction (lag_fallback).
- "neutral": a missing L is set to 0, an explicit no-signal value flagged lag_neutral, so the model trains on every
  observed race (from 2006, as production S1) and predicts every race. lagged_performance itself stays NaN where
  missing; the 0 is applied only inside the fit.
"""

import numpy as np
import pandas as pd

from .backtest import fit, fixed_offset
from .data import KEY, TARGET

WEIGHTS = (0.75, 0.25)  # recent, previous; fixed, not tuned
FIRST_RESIDUAL_YEAR = 2010
LAG = "lagged_performance"
INCUMBENT_LAG = "incumbent_lag"


def oos_residuals(stage1_module, config, data: pd.DataFrame, years) -> pd.DataFrame:
    """KEY and residual (actual - S1's rolling out-of-sample prediction) for every scored race of `years`."""
    rolling = stage1_module.rolling_predictions(config, eval_years=list(years), data=data)
    return pd.concat([r.predictions[[*KEY, "residual"]] for r in rolling.values()], ignore_index=True)


def state_year_residuals(races: pd.DataFrame, residuals: pd.DataFrame) -> pd.DataFrame:
    """One row per (year, state) with a Senate race in `races` (every race of senate_incumbency.csv): its residual (the
    mean over its races, NaN unless every race has one) and why it's missing."""
    joined = races[KEY].merge(residuals, on=KEY, how="left", validate="one_to_one")
    grouped = joined.groupby(["year", "state"])["residual"]
    out = pd.DataFrame({"residual": grouped.mean(), "n_races": grouped.size(), "n_missing": grouped.apply(
        lambda r: int(r.isna().sum()))}).reset_index()
    out.loc[out["n_missing"] > 0, "residual"] = np.nan
    out["missing_reason"] = np.select(
        [out["year"] < FIRST_RESIDUAL_YEAR, out["n_missing"] > 0],
        [f"before {FIRST_RESIDUAL_YEAR} (no out-of-sample fundamentals)",
         "a race without an observed D-vs-R residual (imputed, third-party or excluded)"], None)
    return out


def lagged_performance(races: pd.DataFrame, state_years: pd.DataFrame, weights=WEIGHTS) -> pd.DataFrame:
    """For every race in `races`: recent_year, previous_year, r_recent, r_previous, lagged_performance (L, NaN unless
    both residuals exist) and lag_missing_reason."""
    elections = state_years.sort_values(["state", "year"])
    by_state = {s: g for s, g in elections.groupby("state")}
    rows = []
    for race in races[KEY].itertuples(index=False):
        prior = by_state[race.state]
        prior = prior[prior["year"] < race.year].tail(2)  # the two most recent earlier elections
        row = {"year": race.year, "state": race.state, "class": race[2], "recent_year": np.nan,
               "previous_year": np.nan, "r_recent": np.nan, "r_previous": np.nan, LAG: np.nan,
               "lag_missing_reason": None}
        if len(prior) < 2:
            row["lag_missing_reason"] = f"only {len(prior)} earlier Senate election(s) in the data"
        else:
            previous, recent = prior.iloc[0], prior.iloc[1]
            row.update(recent_year=recent["year"], previous_year=previous["year"], r_recent=recent["residual"],
                       r_previous=previous["residual"])
            if pd.isna(recent["residual"]) or pd.isna(previous["residual"]):
                missing = recent if pd.isna(recent["residual"]) else previous
                row["lag_missing_reason"] = f"{int(missing['year'])}: {missing['missing_reason']}"
            else:
                row[LAG] = weights[0] * recent["residual"] + weights[1] * previous["residual"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["lag_available"] = out[LAG].notna()
    return out


def incumbent_lag(races: pd.DataFrame, nominees: pd.DataFrame, residuals: pd.DataFrame,
                  weights=WEIGHTS) -> pd.DataFrame:
    """The incumbent-specific lag, for every race in `races` (KEY and inc_signed).

    The incumbent is the race's Democratic nominee when inc_signed is +1, the Republican when -1 (senate_fec.csv's
    dem_candidate_id / gop_candidate_id, the FEC candidate ID, which a senator keeps across elections). Their earlier
    Senate races are the races of earlier years in which the same ID was a D or R nominee. The two most recent of those
    are considered, and each counts if it has a residual (residuals: the out-of-sample D-positive residuals of
    oos_residuals, so an R incumbent who over-performed has a negative one):
        two residuals   incumbent_lag = 0.75 r_recent + 0.25 r_previous            flag "two elections"
        one residual    incumbent_lag = that residual                              flag "one election (<year>)"
        none            incumbent_lag NaN (the fit uses 0 with missing_lag="neutral"), with the reason in the flag:
                        an open seat (appointees are open), no earlier race of the ID in the data (e.g. a win in an
                        off-cycle special), or earlier races without residuals (before 2010, or not an observed
                        D-vs-R race)
    Challengers and open-seat candidates never get a lag."""
    ids = races[[*KEY, "inc_signed"]].merge(
        nominees[[*KEY, "dem_name", "dem_candidate_id", "gop_name", "gop_candidate_id"]], on=KEY, how="left",
        validate="one_to_one")
    dem, gop = ids["inc_signed"] == 1, ids["inc_signed"] == -1
    ids["incumbent_id"] = ids["dem_candidate_id"].where(dem, ids["gop_candidate_id"].where(gop))
    ids["incumbent_name"] = ids["dem_name"].where(dem, ids["gop_name"].where(gop))
    appearances = pd.concat([ids[[*KEY, col]].rename(columns={col: "candidate_id"})
                             for col in ["dem_candidate_id", "gop_candidate_id"]]).dropna()
    appearances = appearances.merge(residuals[[*KEY, "residual"]], on=KEY, how="left", validate="many_to_one")
    by_id = {cid: g.sort_values("year") for cid, g in appearances.groupby("candidate_id")}

    rows = []
    for race in ids.itertuples(index=False):
        row = {"year": race.year, "state": race.state, "class": race[2], "incumbent_id": race.incumbent_id,
               "incumbent_name": race.incumbent_name, "inc_prev_years": None, "inc_r_recent": np.nan,
               "inc_r_previous": np.nan, INCUMBENT_LAG: np.nan, "inc_lag_n": 0}
        if pd.isna(race.incumbent_id):
            row["inc_lag_flag"] = "neutral: open seat (no incumbent)"
        else:
            history = by_id.get(race.incumbent_id)
            earlier = history[history["year"] < race.year].tail(2) if history is not None else pd.DataFrame()
            usable = earlier.dropna(subset=["residual"]) if len(earlier) else earlier
            row["inc_prev_years"] = ",".join(str(int(y)) for y in earlier["year"]) if len(earlier) else None
            if len(earlier) == 0:
                row["inc_lag_flag"] = "neutral: no earlier race of the incumbent in the data"
            elif len(usable) == 0:
                row["inc_lag_flag"] = "neutral: earlier races without a residual"
            elif len(usable) == 1:
                r = usable.iloc[-1]
                row.update({INCUMBENT_LAG: r["residual"], "inc_r_recent": r["residual"], "inc_lag_n": 1,
                            "inc_lag_flag": f"one election ({int(r['year'])})"})
            else:
                previous, recent = usable.iloc[0], usable.iloc[1]
                row.update({INCUMBENT_LAG: weights[0] * recent["residual"] + weights[1] * previous["residual"],
                            "inc_r_recent": recent["residual"], "inc_r_previous": previous["residual"],
                            "inc_lag_n": 2, "inc_lag_flag": "two elections"})
        rows.append(row)
    out = pd.DataFrame(rows)
    out["incumbent_lag_available"] = out[INCUMBENT_LAG].notna()
    return out


def fit_predict_lagged(data: pd.DataFrame, s1_formula: str, fixed: dict, s1_predictions: pd.DataFrame,
                       eval_years, equal_year: bool = True, with_lag: bool = True,
                       missing_lag: str = "exclude", lag: str = LAG,
                       available: str = "lag_available") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(predictions, coefficients by eval year) of S1 + L. data must have lagged_performance. s1_predictions are S1's
    rolling predictions for eval_years (stage1.rolling_predictions).

    missing_lag says what a race without L does:
        "exclude"  it isn't trained on, and keeps S1's own prediction (lag_fallback). Training starts in 2014
        "neutral"  L is set to 0 (no persistence signal; flagged lag_neutral) for training and prediction, so the
                   model trains on every observed race from 2006, the same rows as production S1
    with_lag=False is the matched-sample control for "exclude": S1's formula refit on exactly the same training rows,
    which separates L's effect from the shorter training sample. lag / available name the lag column and its
    availability flag (incumbent_lag / incumbent_lag_available for the incumbent-specific lag)."""
    if missing_lag not in ("exclude", "neutral"):
        raise ValueError(f"missing_lag must be 'exclude' or 'neutral'; got {missing_lag!r}")
    neutral = missing_lag == "neutral"
    data = data.assign(lag_neutral=~data[available])
    if neutral:
        data[lag] = data[lag].where(data[available], 0.0)
    formula = f"{s1_formula} + {lag}" if with_lag else s1_formula
    frames, coefficients = [], {}
    for year in eval_years:
        usable = data["observed"] & (data["year"] < year) & (True if neutral else data[available])
        train = data[usable]
        if train["year"].nunique() < 2:
            raise ValueError(f"{year}: S1 + L needs at least two training years with L")
        model = fit(formula, train, equal_year=equal_year, fixed=fixed)
        coefficients[year] = pd.concat([model.params, pd.Series(fixed, dtype=float),
                                        pd.Series({"n_train": len(train),
                                                   "n_train_with_L": int(train[available].sum()),
                                                   "train_years": f"{train['year'].min()}-{train['year'].max()}"})])
        s1 = s1_predictions[s1_predictions["year"] == year]
        rows = s1[KEY].merge(data, on=KEY, how="left", validate="one_to_one")
        lagged = model.predict(rows) + fixed_offset(rows, fixed)
        use_model = np.ones(len(rows), dtype=bool) if neutral else rows[available].to_numpy()
        out = s1.copy()
        out["prediction_s1"] = out["prediction"]
        out["prediction"] = np.where(use_model, lagged.to_numpy(), out["prediction_s1"].to_numpy())
        out["lag_fallback"] = ~use_model
        out["lag_neutral"] = rows["lag_neutral"].to_numpy() if neutral else False
        out["residual"] = out["actual"] - out["prediction"]
        frames.append(out)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(coefficients).T.rename_axis("eval_year")
