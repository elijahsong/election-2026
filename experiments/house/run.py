"""run_ols / run_wls: fit a formula by OLS or WLS on the training years, predict the evaluation and prediction years,
and score.

Notebook (from notebooks/house, put the repository root on sys.path first):

    from experiments.house.run import run_ols
    result = run_ols("adj_margin_2p_norm ~ gen_margin_raw + partisan_lean + inc_signed",
                     train_years=list(range(2006, 2022, 2)), eval_years=[2022], name="classic_ols")  # save=False
    result.model.summary(); result.metrics; result.coverage; result.scored; result.predictions

CLI (from the repository root; saves by default, --no-save to skip):

    python -m experiments.house.run --formula "adj_margin_2p_norm ~ gen_margin_raw + partisan_lean + inc_signed" \\
        --train-years 2006-2020 --eval-years 2022 --name classic_ols

The formula is the model: any statsmodels (patsy) formula, with interactions (a:b, a*b), categoricals (C(state)), and
transformations (np.log(x), I(x**2)). Its left-hand side is the target column of targets.csv. Imputed targets are
never trained on or scored: evaluation scores valid (observed, non-imputed, eligible race type) targets, and the
competitive subsets of them, per eval year (see evaluation.py). HOLDOUT_YEARS (2024) need use_holdout=True /
--use-holdout, in training too. predict_years (e.g. 2026) need no targets and get no metrics.

Each run_<estimator>'s body is the whole method, one smf.<estimator>(...).fit() line: _fit_and_score does the same data
work around it for every estimator (prepare the rows, predict, score, record). Another estimator gets its own
run_<estimator> with the same shape. run_wls weights the training rows by training_weights (equal-year, recency, and
election-type weighting), computed within each fit from the training rows alone. run_ols_fixed imposes coefficients
(e.g. inc_signed = 5): it fits OLS on the target minus the fixed terms, re-estimating every other coefficient, and adds
the fixed terms back to the fitted values and predictions.
"""

import argparse
from dataclasses import dataclass

import numpy as np  # also lets formulas use np.log(...): patsy resolves names in this module
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

import house.regression as hr
from house.regression.config import (
    COMPETITIVE_SUBSETS,
    COMPETITIVE_THRESHOLD,
    EVAL_RACE_TYPES,
)

from . import evaluation, tracking


@dataclass
class RegressionResult:
    model: object  # native statsmodels RegressionResults: .summary(), .params, .resid (actual - fitted), ...
    predictions: pd.DataFrame  # year, state, district, role (train/eval/predict), prediction, actual, residual
    metrics: pd.DataFrame | None  # by (subset, year): n, mae, rmse, bias, winner counts; None without eval_years
    coverage: pd.DataFrame | None  # by eval year: districts, exclusions by reason, n per subset
    scored: pd.DataFrame | None  # one row per valid eval target: actual, predicted, residual, winners, subset flags
    coefficients: pd.DataFrame  # one row per term: coef, std_err, t, p_value, ci_low, ci_high
    fit: dict  # nobs, r2, r2_adj, ... (metrics.json "fit")
    counts: dict  # sample sizes, exclusions by "role:reason", row hashes (metrics.json "counts")
    excluded: pd.DataFrame  # every left-out row: year, state, district, role, reason
    config: dict  # what was run, with provenance (config.json)
    run_id: str | None = None  # set when saved


def run_ols(formula: str, train_years, eval_years=(), predict_years=(), *, name: str | None = None,
            predictors: pd.DataFrame | None = None, targets: pd.DataFrame | None = None, use_holdout: bool = False,
            require_complete=(), competitive: dict = COMPETITIVE_SUBSETS, race_types=EVAL_RACE_TYPES,
            save: bool = False, results_dir=tracking.RESULTS_DIR) -> RegressionResult:
    """Fit `formula` by OLS on train_years, predict eval_years and predict_years, and score the eval_years.

    predictors and targets default to hr.load_predictors() and hr.load_targets(); pass them to skip reloading.
    require_complete lists extra columns a row must have (finite) to be used, so runs with different formulas can
    share exactly the same rows. competitive ({name: (column, threshold)}) and race_types (the eligible race types)
    define the evaluation subsets; they default to config.COMPETITIVE_SUBSETS and config.EVAL_RACE_TYPES and don't
    affect training. name labels the run (and its folder). save=True writes it to results/house; a result can also be
    saved later with tracking.save_run.
    """
    return _fit_and_score(
        lambda train: smf.ols(formula, data=train).fit(),
        "ols", {}, formula, train_years, eval_years, predict_years, name=name, predictors=predictors,
        targets=targets, use_holdout=use_holdout, require_complete=require_complete, competitive=competitive,
        race_types=race_types, save=save, results_dir=results_dir)


def training_weights(train: pd.DataFrame, *, equal_year: bool = False, half_life: float | None = None,
                     midterm_weight: float = 1.0) -> pd.Series:
    """WLS weights for the training rows, from their election years alone, normalized to mean 1:

        equal_year=True   w *= 1 / n_t, n_t = the training rows of year t (after exclusions), so every training year
                          carries the same total weight
        half_life=H       w *= 2 ** (-(T - t) / H), T = the latest training year: a year H years older counts half.
                          Using the eval year as T instead only rescales w, which the normalization removes
        midterm_weight=m  w *= m in midterm years (t % 4 == 2: 2006, 2010, ...), 1 in presidential years

    The factors multiply. With every one off (the defaults) w = 1, so WLS is OLS. Normalizing doesn't change WLS
    estimates; it keeps the weights readable (mean 1 within each training window)."""
    if half_life is not None and not half_life > 0:
        raise ValueError(f"half_life must be a positive number of years; got {half_life!r}")
    if not midterm_weight > 0:
        raise ValueError(f"midterm_weight must be positive; got {midterm_weight!r}")
    year = train["year"]
    weights = pd.Series(1.0, index=train.index)
    if equal_year:
        weights /= year.map(year.value_counts())
    if half_life is not None:
        weights *= 2.0 ** (-(year.max() - year) / half_life)
    weights *= np.where(year % 4 == 2, midterm_weight, 1.0)
    return weights / weights.mean()


def run_wls(formula: str, train_years, eval_years=(), predict_years=(), *, equal_year: bool = False,
            half_life: float | None = None, midterm_weight: float = 1.0, fixed: dict | None = None,
            name: str | None = None, predictors: pd.DataFrame | None = None, targets: pd.DataFrame | None = None,
            use_holdout: bool = False, require_complete=(), competitive: dict = COMPETITIVE_SUBSETS,
            race_types=EVAL_RACE_TYPES, save: bool = False, results_dir=tracking.RESULTS_DIR) -> RegressionResult:
    """run_ols, but fit by WLS with training_weights(train, equal_year=..., half_life=..., midterm_weight=...),
    computed from each fit's own training rows (after exclusions). The weights are in result.config["weights"]; with
    every weighting off it equals run_ols. fixed ({column: coefficient}) imposes coefficients as in run_ols_fixed: the
    weighted fit is then of the adjusted target, and the fixed terms are added back to the predictions."""
    weights = {"equal_year": equal_year, "half_life": half_life, "midterm_weight": midterm_weight}
    predictors = hr.load_predictors() if predictors is None else predictors
    fixed, fit_formula = _fixed_formula(formula, predictors.columns, fixed or {})
    return _fit_and_score(
        lambda train: smf.wls(fit_formula, data=train, weights=training_weights(train, **weights)).fit(),
        "wls", {"weights": weights, **({"fixed": fixed, "fit_formula": fit_formula} if fixed else {})}, formula,
        train_years, eval_years, predict_years, name=name, predictors=predictors, targets=targets,
        use_holdout=use_holdout, require_complete=require_complete, competitive=competitive, race_types=race_types,
        save=save, results_dir=results_dir, fixed=fixed)


def run_ols_fixed(formula: str, train_years, eval_years=(), predict_years=(), *, fixed: dict,
                  name: str | None = None, predictors: pd.DataFrame | None = None, targets: pd.DataFrame | None = None,
                  use_holdout: bool = False, require_complete=(), competitive: dict = COMPETITIVE_SUBSETS,
                  race_types=EVAL_RACE_TYPES, save: bool = False, results_dir=tracking.RESULTS_DIR) -> RegressionResult:
    """run_ols with coefficients imposed rather than estimated: fixed = {column: coefficient}, e.g.
    {"inc_signed": 5.0}, for columns that are terms of the formula's right-hand side.

    With target y, the fit is OLS of y - sum(c * column) on the right-hand side less the fixed columns (written out as
    result.config["fit_formula"], e.g. "I(adj_margin_2p_norm - 5.0 * inc_signed) ~ gen_margin_raw + partisan_lean +
    inc_signed - inc_signed"), so every other coefficient is re-estimated. sum(c * column) is added back to the fitted
    values and predictions, which are scored against y as usual, on the same rows as run_ols (the fixed columns still
    must be complete). result.coefficients lists each fixed coefficient with blank standard errors; fit statistics
    (R2, ...) are those of the adjusted-target regression. run_wls(..., fixed=...) does the same by WLS."""
    predictors = hr.load_predictors() if predictors is None else predictors
    fixed, fit_formula = _fixed_formula(formula, predictors.columns, fixed)
    return _fit_and_score(
        lambda train: smf.ols(fit_formula, data=train).fit(),
        "ols", {"fixed": fixed, "fit_formula": fit_formula}, formula, train_years, eval_years, predict_years,
        name=name, predictors=predictors, targets=targets, use_holdout=use_holdout, require_complete=require_complete,
        competitive=competitive, race_types=race_types, save=save, results_dir=results_dir, fixed=fixed)


def _fixed_formula(formula: str, columns, fixed: dict) -> tuple[dict, str]:
    """(fixed as floats, the formula to fit): `formula` unchanged without fixed coefficients, else the target minus
    the fixed terms on the left and the fixed columns removed on the right (run_ols_fixed)."""
    if not fixed:
        return {}, formula
    target, variables = evaluation.formula_variables(formula, columns)
    fixed = {column: float(c) for column, c in fixed.items()}
    unknown = [column for column in fixed if column not in variables]
    if unknown or not all(np.isfinite(c) for c in fixed.values()):
        raise ValueError(f"fixed needs finite coefficients of columns in the formula ({variables}); got {fixed}")
    rhs = formula.split("~", 1)[1].strip()
    return fixed, (f"I({target} - {' - '.join(f'{c!r} * {column}' for column, c in fixed.items())}) ~ {rhs}"
                   + "".join(f" - {column}" for column in fixed))


def _fit_and_score(fit_model, estimator: str, estimator_config: dict, formula: str, train_years, eval_years,
                   predict_years, *, name, predictors, targets, use_holdout, require_complete, competitive, race_types,
                   save, results_dir, fixed: dict | None = None) -> RegressionResult:
    """The data work around one estimator: prepare the training / evaluation rows (evaluation.prepare_data), fit
    `fit_model(train)` (a statsmodels results object), predict, score, and record; see run_ols for the arguments.
    fixed ({column: coefficient}, run_ols_fixed) adds sum(c * column) to the fitted values and predictions and lists
    the fixed coefficients in the coefficients table."""
    fixed = fixed or {}
    predictors = hr.load_predictors() if predictors is None else predictors
    targets = hr.load_targets() if targets is None else targets
    target, variables = evaluation.formula_variables(formula, predictors.columns)
    train, eval_data, excluded = evaluation.prepare_data(
        predictors, targets, target, [*variables, *require_complete], train_years, eval_years, predict_years,
        use_holdout=use_holdout)

    model = fit_model(train)

    if model.nobs != len(train):
        raise ValueError(f"The formula dropped {len(train) - int(model.nobs)} of {len(train)} training rows "
                         "(a transformation gave missing values)")

    def offset(rows: pd.DataFrame) -> pd.Series:  # the fixed terms, 0 without any
        return sum((c * rows[column] for column, c in fixed.items()), pd.Series(0.0, index=rows.index))

    predictions = evaluation.assemble_predictions(train, model.fittedvalues + offset(train), eval_data,
                                                  model.predict(eval_data) + offset(eval_data), target)
    evaluated = predictions[predictions["role"] == "eval"]
    metrics, coverage, scored = (evaluation.evaluate_predictions(evaluated, targets, predictors, target,
                                                                 competitive=competitive, race_types=race_types)
                                 if len(evaluated) else (None, None, None))

    ci = model.conf_int()
    coefficients = pd.DataFrame({"coef": model.params, "std_err": model.bse, "t": model.tvalues,
                                 "p_value": model.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
    fixed_rows = pd.DataFrame({"coef": fixed}, index=list(fixed)).reindex(columns=coefficients.columns)
    coefficients = pd.concat([coefficients, fixed_rows]) if fixed else coefficients
    coefficients.index.name = "term"
    fit = {"nobs": int(model.nobs), "df_model": float(model.df_model), "df_resid": float(model.df_resid),
           "r2": float(model.rsquared), "r2_adj": float(model.rsquared_adj), "resid_std": float(np.sqrt(model.scale)),
           "aic": float(model.aic), "bic": float(model.bic), "condition_number": float(model.condition_number),
           "rank": int(model.model.rank)}
    config = {
        "run_id": None,
        "estimator": estimator,
        **estimator_config,
        "formula": formula,
        "target": target,
        "variables": variables,  # the predictor columns the formula uses
        "name": name,
        "train_years": [int(y) for y in train_years],
        "eval_years": [int(y) for y in eval_years],
        "predict_years": [int(y) for y in predict_years],
        "use_holdout": use_holdout,
        "require_complete": list(require_complete),
        "race_types": list(race_types),
        "competitive": {name: [column, threshold] for name, (column, threshold) in competitive.items()},
        "statsmodels": statsmodels.__version__,
        **tracking.provenance(predictors, targets),
    }
    result = RegressionResult(model, predictions, metrics, coverage, scored, coefficients, fit,
                              evaluation.counts(predictions, excluded), excluded, config)
    if save:
        tracking.save_run(result, results_dir)
    return result


def _list(text: str) -> list[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def _years(text: str) -> list[int]:
    """Comma-separated years; "A-B" is every election year from A through B, e.g. 2006-2020."""
    years = []
    for part in _list(text):
        start, _, end = part.partition("-")
        years += range(int(start), int(end) + 1, 2) if end else [int(start)]
    return years


def parse_args(argv=None) -> dict:
    """run_ols's keyword arguments from the command line."""
    parser = argparse.ArgumentParser(prog="python -m experiments.house.run", description=__doc__.split("\n")[0])
    parser.add_argument("--formula", required=True, help='e.g. "adj_margin_2p_norm ~ gen_margin_raw + inc_signed"')
    parser.add_argument("--train-years", required=True, type=_years, help="e.g. 2006-2020 or 2008,2012,2016")
    parser.add_argument("--eval-years", type=_years, default=[], help="scored against actual targets")
    parser.add_argument("--predict-years", type=_years, default=[], help="predicted only; no targets needed")
    parser.add_argument("--name", help="labels the run and its folder, e.g. classic_ols")
    parser.add_argument("--use-holdout", action="store_true", help="allow HOLDOUT_YEARS (2024), in training too")
    parser.add_argument("--require-complete", type=_list, default=[],
                        help="extra columns a row needs, to share rows across formulas (comma-separated)")
    parser.add_argument("--competitive-threshold", type=float, default=COMPETITIVE_THRESHOLD,
                        help="points: a race is competitive when abs(column) <= this, for every subset in "
                             "config.COMPETITIVE_SUBSETS (default %(default)s)")
    parser.add_argument("--no-save", dest="save", action="store_false")
    args = vars(parser.parse_args(argv))
    threshold = args.pop("competitive_threshold")
    args["competitive"] = {name: (column, threshold) for name, (column, _) in COMPETITIVE_SUBSETS.items()}
    return args


def main(argv=None) -> RegressionResult:
    result = run_ols(**parse_args(argv))
    print(tracking.describe(result), end="\n\n")
    print(result.model.summary(), end="\n\n")
    if result.metrics is None:
        print("No eval_years scored", end="\n\n")
    else:
        print(result.coverage.to_string(), end="\n\n")
        print(result.metrics.round(3).to_string(), end="\n\n")
    print(f"Saved results/house/runs/{result.run_id}" if result.run_id else "Not saved (--no-save)")
    return result


if __name__ == "__main__":
    main()
