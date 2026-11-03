"""Expanding-window backtests: each eval year is fitted on earlier years only and predicted out of sample.

`backtest` fits a formula by OLS, or WLS with equal-year and/or recency weights, optionally with fixed (imposed)
coefficients. `baseline` scores a fixed expression with no fitting (e.g. uniform swing). Both return a Backtest whose
predictions share one layout, so `metrics` scores them alike.
"""

from dataclasses import dataclass

import pandas as pd
import statsmodels.formula.api as smf

from .data import KEY, TARGET

PREDICTION_COLUMNS = [*KEY, "special", "inc_signed", "partisan_lean", "gb_margin_2p_norm", TARGET]


@dataclass
class Backtest:
    name: str
    formula: str
    predictions: pd.DataFrame  # KEY, eval_year, prediction, actual, residual (actual - prediction), ...
    coefficients: pd.DataFrame  # eval_year x term, fixed terms included (empty for a baseline)
    n_train: dict


def training_weights(train: pd.DataFrame, equal_year: bool = False, half_life: float | None = None) -> pd.Series:
    """Per-row weights, normalized to mean 1 (all 1 for OLS):
    equal_year  1 / (rows in the row's year), so each training year carries the same total weight
    half_life   2 ** (-(latest training year - year) / half_life): a year half_life years older counts half"""
    weights = pd.Series(1.0, index=train.index)
    if equal_year:
        weights /= train.groupby("year")["year"].transform("size")
    if half_life is not None:
        weights *= 2.0 ** (-(train["year"].max() - train["year"]) / half_life)
    return weights / weights.mean()


def training_rows(data: pd.DataFrame, eval_year: int, train_imputed: bool = False,
                  first_year: int = 2006) -> pd.DataFrame:
    """The rows a window may train on: years from first_year up to, not including, eval_year, with an
    observed target, plus imputed targets if train_imputed."""
    usable = data["observed"] | (data["imputed"] if train_imputed else False)
    return data[usable & (data["year"] >= first_year) & (data["year"] < eval_year)]


def fixed_offset(rows: pd.DataFrame, fixed: dict | None) -> pd.Series:
    """The imposed terms' contribution, sum(value * rows[term]) over `fixed` (0 without any)."""
    return sum((value * rows[term] for term, value in (fixed or {}).items()), pd.Series(0.0, index=rows.index))


def fit(formula: str, train: pd.DataFrame, equal_year: bool = False, half_life: float | None = None,
        fixed: dict | None = None):
    """smf.wls of formula on train (OLS when every weight is 1). A fixed term (e.g. {"inc_signed": 2.5}) is
    imposed: its contribution is subtracted from the target before fitting, so formula must not include it."""
    train = train.assign(**{TARGET: train[TARGET] - fixed_offset(train, fixed)})
    return smf.wls(formula, data=train, weights=training_weights(train, equal_year, half_life)).fit()


def _scored(rows: pd.DataFrame, eval_year: int, prediction: pd.Series) -> pd.DataFrame:
    out = rows[PREDICTION_COLUMNS].assign(eval_year=eval_year, prediction=prediction.to_numpy())
    out = out.rename(columns={TARGET: "actual"})
    out["residual"] = out["actual"] - out["prediction"]
    return out


def backtest(data: pd.DataFrame, formula: str, eval_years, name: str, equal_year: bool = False,
             half_life: float | None = None, fixed: dict | None = None, train_imputed: bool = False,
             first_year: int = 2006) -> Backtest:
    """Fit `formula` on each eval year's earlier years and predict that year's observed races."""
    predictions, coefficients, n_train = [], {}, {}
    for year in eval_years:
        train = training_rows(data, year, train_imputed, first_year)
        model = fit(formula, train, equal_year, half_life, fixed)
        test = data[(data["year"] == year) & data["observed"]]
        prediction = model.predict(test) + fixed_offset(test, fixed)
        if prediction.isna().any():
            raise ValueError(f"{name} {year}: predictors missing for {test.loc[prediction.isna(), KEY].values}")
        predictions.append(_scored(test, year, prediction))
        coefficients[year] = pd.concat([model.params, pd.Series(fixed or {}, dtype=float)])
        n_train[year] = int(model.nobs)
    return Backtest(name, formula, pd.concat(predictions, ignore_index=True),
                    pd.DataFrame(coefficients).T.rename_axis("eval_year"), n_train)


def baseline(data: pd.DataFrame, expression: str, eval_years, name: str) -> Backtest:
    """A prediction with no fitted parameters, `expression` evaluated on each eval year's observed races."""
    predictions = [_scored(test, year, test.eval(expression))
                   for year in eval_years for test in [data[(data["year"] == year) & data["observed"]]]]
    return Backtest(name, expression, pd.concat(predictions, ignore_index=True), pd.DataFrame(), {})


def drop_one_cycle(data: pd.DataFrame, formula: str, eval_year: int, **fit_options) -> pd.DataFrame:
    """The coefficients of eval_year's training window, refitted with each training year left out in turn
    (row "none": all years). Large swings mean one cycle drives an estimate."""
    train = training_rows(data, eval_year)
    rows = {"none": fit(formula, train, **fit_options).params}
    for year in sorted(train["year"].unique()):
        rows[year] = fit(formula, train[train["year"] != year], **fit_options).params
    return pd.DataFrame(rows).T.rename_axis("dropped_year")
