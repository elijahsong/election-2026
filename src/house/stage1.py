"""House Stage 1 production configurations: two fixed specifications, fitted fresh on each training window.

    house_model_a             Model A alone
    house_ensemble_ab_85_15   0.85 * Model A + 0.15 * Model B

Model A, polling (inc_signed's coefficient imposed at +2.5; every other coefficient is estimated):

    adj_margin_2p_norm ~ partisan_lean + gen_margin_raw + 2.5 * inc_signed + partisan_lean:inc_signed

Model B, fundamentals (every coefficient estimated):

    adj_margin_2p_norm ~ partisan_lean + fundraising_advantage + pres_party_signed:is_midterm
                         + pres_party_signed:cpi_region_log_yoy

The specification was chosen on the 2018-2022 backtest (notebooks/house/01_reg_ensemble_compare.ipynb) and checked
once on the 2024 holdout (01_reg_final_2024_holdout.ipynb); nothing here selects features or weights. Model A then
dropped gb_momentum_30d_raw (2026-11-02) for parsimony: it is national, nearly collinear with gen_margin_raw (r = 0.92
across elections), and its coefficient swung by 2-3 points when one training election was dropped, for a small
accuracy edge (about 0.1 MAE, mostly 2018). See 01_reg_model_a_no_momentum.ipynb and
01_reg_generic_ballot_smoothing.ipynb, which were run with momentum still in Model A.

Methodological choices, all fixed by the configurations below:

- Estimation: both models by equal-year WLS (experiments.house.run.run_wls with equal_year=True). A training race is
  weighted by 1 / n_t, n_t = the valid training races of its election year (after exclusions), so every training year
  carries the same total weight. Weights are recomputed from each window's own training rows.
- Signs: inc_signed is +1 for a Democratic incumbent, -1 for a Republican one, 0 for an open seat. pres_party_signed is
  +1 under a Democratic president, -1 under a Republican one (PRES_PARTY_SIGNED); is_midterm is 1 in midterm years.
  Both are derived here from the year (add_derived_predictors), since modeling.csv doesn't hold them.
- Rows: imputed (uncontested) targets are never trained on or predicted in years with targets (run_wls's
  evaluation.prepare_data). Every component of a configuration requires the predictor columns of all its components,
  so A and B are fitted and predicted on the same districts and blend row by row. house_model_a requires only A's
  columns, so it can keep districts the ensemble drops (e.g. a missing fundraising_advantage).
- Windows: expanding, from TRAIN_START (2006) through the election before the predicted one. fit_predict refuses any
  training year at or after the predicted year, so no evaluation year informs its own fit.
- 2024: the holdout (config.HOLDOUT_YEARS) was used once to validate the frozen specification, so these production runs
  use it deliberately (USE_HOLDOUT): as the 2024 rolling window's eval year, and in training for 2026.

Usage (a notebook in notebooks/house: put src/ and the repository root on sys.path first, as the other notebooks do):

    from house import stage1

    rolling = stage1.rolling_predictions(stage1.HOUSE_ENSEMBLE_AB_85_15)  # {2018: Stage1Result, ..., 2024: ...}
    oos = pd.concat([r.predictions for r in rolling.values()], ignore_index=True)  # Stage 2's out-of-sample table
    forecast = stage1.forecast(stage1.HOUSE_ENSEMBLE_AB_85_15)  # trained on 2006-2024, predicts 2026

Each Stage1Result's predictions have one row per predicted district: year, state, district, train_start, train_end,
prediction, actual, residual (actual - prediction; NA without an observed target, e.g. 2026), and prediction_<model>
for each component. 2026's gen_margin_raw is FiftyPlusOne's election-eve margin, transcribed by hand.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

import house.regression as hr
from experiments.house import evaluation
from experiments.house.run import RegressionResult, run_wls
from house.regression.schema import KEY

TARGET = "adj_margin_2p_norm"
TRAIN_START = 2006  # the first training year of every window
ROLLING_EVAL_YEARS = (2018, 2020, 2022, 2024)  # out-of-sample years kept for Stage 2 calibration
FORECAST_YEAR = 2026
USE_HOLDOUT = True  # see the module docstring: 2024 is used deliberately in production

# Known before every election: the White House party, +1 D, -1 R
PRES_PARTY_SIGNED = MappingProxyType({2006: -1, 2008: -1, 2010: 1, 2012: 1, 2014: 1, 2016: 1, 2018: -1, 2020: -1,
                                      2022: 1, 2024: 1, 2026: -1})


def add_derived_predictors(predictors: pd.DataFrame) -> pd.DataFrame:
    """predictors plus pres_party_signed (PRES_PARTY_SIGNED of the year) and is_midterm (1 when year % 4 == 2)."""
    unknown = sorted(set(predictors["year"]) - set(PRES_PARTY_SIGNED))
    if unknown:
        raise ValueError(f"No PRES_PARTY_SIGNED entry for {unknown}")
    return predictors.assign(pres_party_signed=predictors["year"].map(PRES_PARTY_SIGNED),
                             is_midterm=(predictors["year"] % 4 == 2).astype(int))


@dataclass(frozen=True)
class Component:
    """One regression: a patsy formula (target ~ terms) and the coefficients imposed rather than estimated."""
    name: str
    formula: str
    fixed: Mapping[str, float] = MappingProxyType({})

    def __post_init__(self):
        object.__setattr__(self, "fixed", MappingProxyType(dict(self.fixed)))  # read-only


@dataclass(frozen=True)
class Stage1Config:
    """A fixed Stage 1 specification: (component, blend weight) pairs, weights summing to 1, fitted by WLS."""
    name: str
    components: tuple[tuple[Component, float], ...]
    equal_year: bool = True
    train_start: int = TRAIN_START

    def __post_init__(self):
        weights = [w for _, w in self.components]
        if not all(w > 0 for w in weights) or not math.isclose(sum(weights), 1.0):
            raise ValueError(f"Blend weights must be positive and sum to 1; got {weights}")


MODEL_A = Component(
    "A_polling",
    f"{TARGET} ~ partisan_lean + gen_margin_raw + inc_signed + partisan_lean:inc_signed",
    fixed={"inc_signed": 2.5},
)
MODEL_B = Component(
    "B_fundamentals",
    f"{TARGET} ~ partisan_lean + fundraising_advantage + pres_party_signed:is_midterm"
    " + pres_party_signed:cpi_region_log_yoy",
)

HOUSE_MODEL_A = Stage1Config("house_model_a", ((MODEL_A, 1.0),))
HOUSE_ENSEMBLE_AB_85_15 = Stage1Config("house_ensemble_ab_85_15", ((MODEL_A, 0.85), (MODEL_B, 0.15)))
CONFIGS = MappingProxyType({c.name: c for c in [HOUSE_MODEL_A, HOUSE_ENSEMBLE_AB_85_15]})


@dataclass
class Stage1Result:
    config: Stage1Config
    predictions: pd.DataFrame  # one row per predicted district (module docstring)
    components: dict[str, RegressionResult]  # each component's run: .model, .coefficients, .config["weights"], ...
    metrics: pd.DataFrame | None  # the blend scored like a single model (evaluation.evaluate_predictions); None for 2026
    scored: pd.DataFrame | None  # one row per valid target, with competitive-subset flags; None for 2026

    @property
    def coefficients(self) -> pd.DataFrame:
        """coef per term (rows) and component (columns); fixed coefficients included."""
        return pd.DataFrame({name: r.coefficients["coef"] for name, r in self.components.items()})


def expanding_window(config: Stage1Config, year: int) -> list[int]:
    """Every election year from config.train_start up to, not including, `year`."""
    return list(range(config.train_start, year, 2))


def fit_predict(config: Stage1Config, train_years, predict_year: int, *, predictors: pd.DataFrame | None = None,
                targets: pd.DataFrame | None = None) -> Stage1Result:
    """Fit every component of `config` on train_years and predict predict_year, then blend.

    predictors and targets default to hr.load_predictors() and hr.load_targets() (pass them to skip reloading);
    add_derived_predictors is applied here. A predict_year with targets is scored; one without (2026) is predicted only.
    """
    train_years = sorted(int(y) for y in train_years)
    if max(train_years) >= predict_year:
        raise ValueError(f"Training years {train_years} reach {predict_year} or later; train on earlier years only")
    predictors = add_derived_predictors(hr.load_predictors() if predictors is None else predictors)
    targets = hr.load_targets() if targets is None else targets
    has_targets = predict_year in set(targets["year"])
    shared = evaluation.shared_columns([c.formula for c, _ in config.components], predictors.columns)

    runs = {
        component.name: run_wls(
            component.formula,
            train_years,
            eval_years=[predict_year] if has_targets else [],
            predict_years=[] if has_targets else [predict_year],
            equal_year=config.equal_year,
            fixed=dict(component.fixed),
            name=f"{config.name}.{component.name}",
            predictors=predictors,
            targets=targets,
            use_holdout=USE_HOLDOUT,
            require_complete=shared,
        )
        for component, _ in config.components
    }
    predictions = blend(config, runs).assign(train_start=train_years[0], train_end=train_years[-1])
    predictions = predictions[[*KEY, "train_start", "train_end", "prediction", "actual", "residual",
                               *(f"prediction_{c.name}" for c, _ in config.components)]]
    metrics = scored = None
    if has_targets:
        metrics, _, scored = evaluation.evaluate_predictions(predictions[[*KEY, "prediction"]], targets, predictors,
                                                             TARGET)
    return Stage1Result(config, predictions, runs, metrics, scored)


def blend(config: Stage1Config, runs: dict[str, RegressionResult]) -> pd.DataFrame:
    """KEY, prediction_<component> for each component, prediction = sum(weight * prediction_<component>), actual, and
    residual = actual - prediction, for the non-training rows the components predicted (which must be the same)."""
    frames = {c.name: runs[c.name].predictions.query("role != 'train'").set_index(KEY) for c, _ in config.components}
    first = next(iter(frames.values()))
    for name, frame in frames.items():
        if not frame.index.equals(first.index):
            raise ValueError(f"{name} predicted different districts than {next(iter(frames))}")
    out = first[["actual"]].copy()
    for component, _ in config.components:
        out[f"prediction_{component.name}"] = frames[component.name]["prediction"]
    out["prediction"] = sum(weight * out[f"prediction_{c.name}"] for c, weight in config.components)
    out["residual"] = out["actual"] - out["prediction"]
    return out.reset_index()


def rolling_predictions(config: Stage1Config, eval_years=ROLLING_EVAL_YEARS, *,
                        predictors: pd.DataFrame | None = None,
                        targets: pd.DataFrame | None = None) -> dict[int, Stage1Result]:
    """{eval year: fit_predict on its expanding window}: out-of-sample predictions, each year fit on earlier years only."""
    predictors = hr.load_predictors() if predictors is None else predictors
    targets = hr.load_targets() if targets is None else targets
    return {year: fit_predict(config, expanding_window(config, year), year, predictors=predictors, targets=targets)
            for year in eval_years}


def forecast(config: Stage1Config, year: int = FORECAST_YEAR, *, predictors: pd.DataFrame | None = None,
             targets: pd.DataFrame | None = None) -> Stage1Result:
    """The production forecast: fit on every election from config.train_start before `year` (2006-2024 for 2026)."""
    return fit_predict(config, expanding_window(config, year), year, predictors=predictors, targets=targets)
