"""run_ols: fit an OLS formula on the training years, predict the evaluation and prediction years, and score.

Notebook (from notebooks/house, put the repository root on sys.path first):

    from experiments.house.run import run_ols
    result = run_ols("adj_margin_2p_norm ~ gen_margin_raw + partisan_lean + inc_signed",
                     train_years=list(range(2006, 2022, 2)), eval_years=[2022], name="classic_ols")  # save=False
    result.model.summary(); result.metrics; result.predictions

CLI (from the repository root; saves by default, --no-save to skip):

    python -m experiments.house.run --formula "adj_margin_2p_norm ~ gen_margin_raw + partisan_lean + inc_signed" \\
        --train-years 2006-2020 --eval-years 2022 --name classic_ols

The formula is the model: any statsmodels (patsy) formula, with interactions (a:b, a*b), categoricals (C(state)), and
transformations (np.log(x), I(x**2)). Its left-hand side is the target column of targets.csv. Imputed targets are
never trained on; evaluation scores non_imputed (primary), imputed, and all rows from the same predictions (see
evaluation.py). HOLDOUT_YEARS (2024) need use_holdout=True / --use-holdout, in training too. predict_years (e.g. 2026)
need no targets and get no metrics.

run_ols's body is the whole method: shared helpers (evaluation, tracking) do the data work around one
smf.ols(...).fit(). Another estimator gets its own run_<estimator> with the same shape.
"""

import argparse
from dataclasses import dataclass

import numpy as np  # also lets formulas use np.log(...): patsy resolves names in this module
import pandas as pd
import statsmodels
import statsmodels.formula.api as smf

import house.regression as hr

from . import evaluation, tracking


@dataclass
class OLSResult:
    model: object  # the native statsmodels RegressionResults: .summary(), .params, .pvalues, .resid, ...
    predictions: pd.DataFrame  # year, state, district, role (train/eval/predict), prediction, actual, is_imputed, residual
    metrics: pd.DataFrame | None  # n, mae, rmse, bias for non_imputed / imputed / all; None without eval_years
    coefficients: pd.DataFrame  # one row per term: coef, std_err, t, p_value, ci_low, ci_high
    fit: dict  # nobs, r2, r2_adj, ... (metrics.json "fit")
    counts: dict  # sample sizes, exclusions by "role:reason", row hashes (metrics.json "counts")
    excluded: pd.DataFrame  # every left-out row: year, state, district, role, reason
    config: dict  # what was run, with provenance (config.json)
    run_id: str | None = None  # set when saved


def run_ols(formula: str, train_years, eval_years=(), predict_years=(), *, name: str | None = None,
            predictors: pd.DataFrame | None = None, targets: pd.DataFrame | None = None, use_holdout: bool = False,
            require_complete=(), save: bool = False, results_dir=tracking.RESULTS_DIR) -> OLSResult:
    """Fit `formula` by OLS on train_years, predict eval_years and predict_years, and score the eval_years.

    predictors and targets default to hr.load_predictors() and hr.load_targets(); pass them to skip reloading.
    require_complete lists extra columns a row must have (finite) to be used, so runs with different formulas can
    share exactly the same rows. name labels the run (and its folder). save=True writes it to results/house; a result
    can also be saved later with tracking.save_run.
    """
    predictors = hr.load_predictors() if predictors is None else predictors
    targets = hr.load_targets() if targets is None else targets
    target, variables = evaluation.formula_variables(formula, predictors.columns)
    train, eval_data, excluded = evaluation.prepare_data(
        predictors, targets, target, [*variables, *require_complete], train_years, eval_years, predict_years,
        use_holdout=use_holdout)

    model = smf.ols(formula, data=train).fit()

    if model.nobs != len(train):
        raise ValueError(f"The formula dropped {len(train) - int(model.nobs)} of {len(train)} training rows "
                         "(a transformation gave missing values)")

    predictions = evaluation.assemble_predictions(train, model.fittedvalues, eval_data, model.predict(eval_data),
                                                  target)
    metrics = evaluation.evaluate_predictions(predictions)

    ci = model.conf_int()
    coefficients = pd.DataFrame({"coef": model.params, "std_err": model.bse, "t": model.tvalues,
                                 "p_value": model.pvalues, "ci_low": ci[0], "ci_high": ci[1]})
    coefficients.index.name = "term"
    fit = {"nobs": int(model.nobs), "df_model": float(model.df_model), "df_resid": float(model.df_resid),
           "r2": float(model.rsquared), "r2_adj": float(model.rsquared_adj), "resid_std": float(np.sqrt(model.scale)),
           "aic": float(model.aic), "bic": float(model.bic), "condition_number": float(model.condition_number),
           "rank": int(model.model.rank)}
    config = {
        "run_id": None,
        "estimator": "ols",
        "formula": formula,
        "target": target,
        "variables": variables,  # the predictor columns the formula uses
        "name": name,
        "train_years": [int(y) for y in train_years],
        "eval_years": [int(y) for y in eval_years],
        "predict_years": [int(y) for y in predict_years],
        "use_holdout": use_holdout,
        "require_complete": list(require_complete),
        "statsmodels": statsmodels.__version__,
        **tracking.provenance(predictors, targets),
    }
    result = OLSResult(model, predictions, metrics, coefficients, fit, evaluation.counts(predictions, excluded),
                       excluded, config)
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
    parser.add_argument("--no-save", dest="save", action="store_false")
    return vars(parser.parse_args(argv))


def main(argv=None) -> OLSResult:
    result = run_ols(**parse_args(argv))
    print(tracking.describe(result), end="\n\n")
    print(result.model.summary(), end="\n\n")
    print(result.metrics.round(3).to_string() if result.metrics is not None else "No eval_years scored", end="\n\n")
    print(f"Saved results/house/runs/{result.run_id}" if result.run_id else "Not saved (--no-save)")
    return result


if __name__ == "__main__":
    main()
