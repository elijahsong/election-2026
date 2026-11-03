"""Senate Stage 1 production configurations: two fixed specifications, fitted fresh on each training window.

    senate_s0   no incumbency term:   adj_margin_2p_norm ~ partisan_lean + gen_margin_raw
    senate_s1   fixed incumbency:     adj_margin_2p_norm ~ partisan_lean + gen_margin_raw + 2.5 * inc_signed

In S1, inc_signed's coefficient is imposed at +2.5 points of Democratic margin (+1 D incumbent, -1 R, 0 open; appointed
incumbents open, King and Sanders +1), not estimated. Every other coefficient of both is estimated by equal-year WLS.
Neither includes fundraising, race-level polling or expert ratings: those are post-Stage 1 inputs for the Senate.

The specifications come from notebooks/senate/stage1/01_reg_senate_fundamentals.ipynb (expanding-window backtest
2012-2022). A fitted incumbency coefficient (12-16 points early in the sample, falling) lost to uniform swing out of
sample, so incumbency is either left out (S0) or imposed at the House Stage 1 value (S1). Nothing here selects
features or weights.

Methodological choices, all fixed here:

- Estimation: smf.wls with weights 1 / n_t, n_t = the training races of the row's election year (normalized to mean
  1), so every training year carries the same total weight; recomputed from each window's own training rows. The
  fixed term is subtracted from the target before fitting and added back to the prediction
  (senate.regression.backtest.fit and fixed_offset).
- Rows: a race is (year, state, class), so a regular election and a special in one state-year are separate rows.
  Training and scored rows are observed D-vs-R targets (race_type dr, dr_jungle, dr_multi, dr_rcv; not imputed).
  Imputed targets (uncontested and third-party-runner-up races) and senate.regression.EXCLUDED_RACES (AK 2022) are
  never trained on or predicted in years with targets. A year without targets (2026) predicts every race.
- Windows: expanding, from TRAIN_START (2006) through the election before the predicted one. fit_predict refuses any
  training year at or after the predicted year.
- 2024 wasn't used to choose the specifications. It is used here deliberately, as the last rolling eval year and in
  training for 2026, as the House does with its holdout.

Usage (a notebook: put src/ on sys.path first):

    from senate import stage1

    rolling = stage1.rolling_predictions(stage1.SENATE_S1)  # {2012: Stage1Result, ..., 2024: ...}
    oos = pd.concat([r.predictions for r in rolling.values()], ignore_index=True)  # Stage 2's out-of-sample table
    forecast = stage1.forecast(stage1.SENATE_S1)  # trained on 2006-2024, predicts 2026

Stage1Result.predictions has one row per predicted race: year, state, class, special, train_start, train_end,
prediction, actual, residual (actual - prediction; NA without an observed target, e.g. 2026). Stage1Result.scored adds
the House Stage 2 flags competitive_ex_ante (|partisan_lean + gb_margin_2p_norm| <= 10) and competitive_ex_post
(|actual| <= 7.5). 2026's predictions include races without a Democratic nominee (NE: Osborn, an independent; ID, SD),
where the D-vs-R margin is the model's, not a contest on the ballot; their flags are in polls.csv and
expert-ratings.csv (quality_flags).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

import senate.regression as sr

TARGET = sr.TARGET
KEY = sr.KEY
TRAIN_START = 2006  # the first training year of every window
ROLLING_EVAL_YEARS = (2012, 2014, 2016, 2018, 2020, 2022, 2024)  # out-of-sample years kept for Stage 2
FORECAST_YEAR = 2026
FIXED_INC = 2.5  # S1's imposed incumbency coefficient, the House Stage 1 value
PREDICTION_COLUMNS = [*KEY, "special", "train_start", "train_end", "prediction", "actual", "residual"]


@dataclass(frozen=True)
class Stage1Config:
    """A fixed Stage 1 specification: a patsy formula (target ~ estimated terms), the coefficients imposed rather
    than estimated (not in the formula), and the weighting."""
    name: str
    formula: str
    fixed: Mapping[str, float] = MappingProxyType({})
    equal_year: bool = True
    train_start: int = TRAIN_START

    def __post_init__(self):
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))  # read-only


SENATE_S0 = Stage1Config("senate_s0", f"{TARGET} ~ partisan_lean + gen_margin_raw")
SENATE_S1 = Stage1Config("senate_s1", f"{TARGET} ~ partisan_lean + gen_margin_raw", fixed={"inc_signed": FIXED_INC})
CONFIGS = MappingProxyType({c.name: c for c in [SENATE_S0, SENATE_S1]})


@dataclass
class Stage1Result:
    config: Stage1Config
    predictions: pd.DataFrame  # one row per predicted race (module docstring)
    model: object  # the fitted statsmodels result: .summary(), .params, .bse
    metrics: pd.DataFrame | None  # n, mae, rmse, bias, accuracy_pct per subset; None for 2026
    scored: pd.DataFrame | None  # predictions with an actual, plus the competitive flags; None for 2026

    @property
    def coefficients(self) -> pd.Series:
        """Every coefficient, the fixed ones included."""
        return pd.concat([self.model.params, pd.Series(dict(self.config.fixed), dtype=float)])


def expanding_window(config: Stage1Config, year: int) -> list[int]:
    """Every election year from config.train_start up to, not including, `year`."""
    return list(range(config.train_start, year, 2))


def fit_predict(config: Stage1Config, train_years, predict_year: int, *,
                data: pd.DataFrame | None = None) -> Stage1Result:
    """Fit `config` on train_years and predict predict_year. data defaults to senate.regression.load_data() (pass it
    to skip reloading). A predict_year with targets predicts and scores its observed races; one without (2026)
    predicts every race."""
    train_years = sorted(int(y) for y in train_years)
    if max(train_years) >= predict_year:
        raise ValueError(f"Training years {train_years} reach {predict_year} or later; train on earlier years only")
    data = sr.load_data() if data is None else data
    train = data[data["year"].isin(train_years) & data["observed"]]
    model = sr.fit(config.formula, train, equal_year=config.equal_year, fixed=dict(config.fixed))

    year_rows = data[data["year"] == predict_year]
    has_targets = year_rows["observed"].any()
    rows = year_rows[year_rows["observed"]] if has_targets else year_rows
    prediction = model.predict(rows) + sr.fixed_offset(rows, dict(config.fixed))
    if prediction.isna().any():
        missing = rows.loc[prediction.isna(), KEY].values.tolist()
        raise ValueError(f"{config.name} {predict_year}: predictors missing for {missing}")
    predictions = rows.assign(train_start=train_years[0], train_end=train_years[-1], prediction=prediction,
                              actual=rows[TARGET] if has_targets else float("nan"))
    predictions["residual"] = predictions["actual"] - predictions["prediction"]

    metrics = scored = None
    if has_targets:
        masks = sr.subsets(predictions)
        scored = predictions[[*PREDICTION_COLUMNS, "partisan_lean", "gb_margin_2p_norm"]].assign(
            competitive_ex_ante=masks["ex_ante"], competitive_ex_post=masks["ex_post"])
        metrics = pd.DataFrame({name: sr.metrics.score(predictions[mask]) for name, mask in
                                {"all": masks["all"], "competitive_ex_ante": masks["ex_ante"],
                                 "competitive_ex_post": masks["ex_post"]}.items()}).T
    return Stage1Result(config, predictions[PREDICTION_COLUMNS].reset_index(drop=True), model, metrics, scored)


def rolling_predictions(config: Stage1Config, eval_years=ROLLING_EVAL_YEARS, *,
                        data: pd.DataFrame | None = None) -> dict[int, Stage1Result]:
    """{eval year: fit_predict on its expanding window}: out-of-sample predictions, each year fit on earlier years
    only."""
    data = sr.load_data() if data is None else data
    return {year: fit_predict(config, expanding_window(config, year), year, data=data) for year in eval_years}


def forecast(config: Stage1Config, year: int = FORECAST_YEAR, *, data: pd.DataFrame | None = None) -> Stage1Result:
    """The production forecast: fit on every election from config.train_start before `year` (2006-2024 for 2026)."""
    return fit_predict(config, expanding_window(config, year), year, data=data)
