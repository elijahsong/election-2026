"""Out-of-sample scores of Backtest predictions: MAE, RMSE, signed bias and winner accuracy, overall and on the
competitive races.

Competitive subsets follow the House Stage 1 definitions:
  ex_ante  |partisan_lean + gb_margin_2p_norm| <= EX_ANTE_THRESHOLD: the uniform-swing expectation, known before
           the election
  ex_post  |actual margin| <= EX_POST_THRESHOLD
bias = mean(actual - prediction): positive means Democrats outperformed the model.
"""

import numpy as np
import pandas as pd

EX_ANTE_THRESHOLD = 10.0
EX_POST_THRESHOLD = 7.5


def subsets(predictions: pd.DataFrame, ex_ante: float = EX_ANTE_THRESHOLD,
            ex_post: float = EX_POST_THRESHOLD) -> dict[str, pd.Series]:
    return {
        "all": pd.Series(True, index=predictions.index),
        "ex_ante": (predictions["partisan_lean"] + predictions["gb_margin_2p_norm"]).abs() <= ex_ante,
        "ex_post": predictions["actual"].abs() <= ex_post,
    }


def score(rows: pd.DataFrame) -> pd.Series:
    residual = rows["residual"]
    called = (rows["prediction"] != 0) & (rows["actual"] != 0)
    correct = np.sign(rows.loc[called, "prediction"]) == np.sign(rows.loc[called, "actual"])
    return pd.Series({"n": len(rows), "mae": residual.abs().mean(), "rmse": np.sqrt((residual ** 2).mean()),
                      "bias": residual.mean(), "accuracy_pct": 100 * correct.mean() if called.any() else np.nan})


def compare(runs, by_year: bool = False, **thresholds) -> pd.DataFrame:
    """One row per run (and eval year if by_year), columns (subset, metric), pooled over the eval years."""
    table = {}
    for run in runs:
        masks = subsets(run.predictions, **thresholds)
        groups = run.predictions.groupby("eval_year") if by_year else [("pooled", run.predictions)]
        for year, rows in groups:
            table[(run.name, year)] = pd.concat({s: score(rows[masks[s].loc[rows.index]]) for s in masks})
    out = pd.DataFrame(table).T.rename_axis(["model", "eval_year"])
    return out.droplevel("eval_year") if not by_year else out


def residuals_by(runs, column: str) -> pd.DataFrame:
    """Mean residual (bias) and MAE of each run by the values of a predictions column, e.g. eval_year or special."""
    return pd.concat({run.name: run.predictions.groupby(column)["residual"].agg(
        n="size", bias="mean", mae=lambda r: r.abs().mean()) for run in runs}, names=["model"])
