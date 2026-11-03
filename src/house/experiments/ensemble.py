"""Two-model blends: how correlated two models' errors are, and how a weighted average of their predictions scores.

A blend of results a and b predicts (1 - weight) * a + weight * b for every district both predicted in the eval year,
and is scored with evaluation.evaluate_predictions exactly like a single model, so its metrics and subsets match the
models'. Nothing is refit. choose_weight picks the weight for an eval year from earlier windows only, so no eval year
informs its own weight.
"""

import pandas as pd

from house.regression.schema import KEY

from . import evaluation


def eval_predictions(result) -> pd.DataFrame:
    """KEY + prediction of a result's eval rows."""
    return result.predictions.loc[result.predictions["role"] == "eval", [*KEY, "prediction"]]


def blend(a, b, weight: float) -> pd.DataFrame:
    """KEY + prediction = (1 - weight) * a's prediction + weight * b's, for the eval rows of results a and b, which must
    predict exactly the same districts."""
    merged = eval_predictions(a).merge(eval_predictions(b), on=KEY, how="outer", suffixes=("_a", "_b"),
                                       indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError(f"{int((merged['_merge'] != 'both').sum())} districts predicted by only one model; give "
                         "both the same require_complete")
    return merged[KEY].assign(prediction=(1 - weight) * merged["prediction_a"] + weight * merged["prediction_b"])


def blend_scores(a, b, weights, targets: pd.DataFrame, predictors: pd.DataFrame, target: str, competitive: dict,
                 race_types) -> pd.DataFrame:
    """The blend's metrics for each weight, pooled over the results' eval years, indexed by (weight, subset)."""
    rows = {}
    for weight in weights:
        metrics, _, _ = evaluation.evaluate_predictions(blend(a, b, weight), targets, predictors, target,
                                                        competitive=competitive, race_types=race_types)
        for subset in metrics.index.get_level_values("subset").unique():
            rows[(float(weight), subset)] = metrics.loc[(subset, "all")]
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["weight", "subset"])


def choose_weight(scores: dict, eval_year: int, subset: str, metric: str,
                  higher_is_better: bool = False) -> tuple[float, list]:
    """(weight, the windows it was chosen on) for `eval_year`: the weight with the best mean `metric` on `subset` over
    the windows of `scores` ({window: blend_scores table}) before eval_year, never eval_year itself or later. Ties go to
    the smaller weight (more on model a)."""
    earlier = sorted(window for window in scores if window < eval_year)
    if not earlier:
        raise ValueError(f"No window before {eval_year} to choose its weight on")
    mean = sum(scores[window].xs(subset, level="subset")[metric] for window in earlier) / len(earlier)
    best = mean.max() if higher_is_better else mean.min()
    return float(mean.index[mean == best].min()), earlier


def error_correlation(pairs: list, subsets) -> pd.DataFrame:
    """For (a, b) result pairs (one per window; several are pooled), on the valid districts both scored, per subset:
    n, the correlation of the two models' residuals (actual - predicted), and each model's bias and MAE."""
    merged = pd.concat([a.scored.merge(b.scored[[*KEY, "residual"]], on=KEY, suffixes=("_a", "_b"),
                                       validate="one_to_one") for a, b in pairs], ignore_index=True)
    rows = {}
    for subset in subsets:
        rows_in = merged if subset == evaluation.VALID else merged[merged[subset]]
        a, b = rows_in["residual_a"], rows_in["residual_b"]
        rows[subset] = {"n": len(rows_in), "corr": a.corr(b), "bias_a": a.mean(), "bias_b": b.mean(),
                        "mae_a": a.abs().mean(), "mae_b": b.abs().mean()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("subset")
