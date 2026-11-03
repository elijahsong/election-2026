"""Error metrics on predicted vs. actual margins, from the residual = actual - predicted: a positive bias (mean
residual) means Democrats outperformed the forecast, a negative one that Republicans did. Nulls raise rather than being
dropped: subset the rows explicitly before scoring.

Winners come from the sign of a two-party margin, D positive. A margin of exactly 0 is a "tie" and is never assigned to
a party: winner_counts leaves a row out of the classification (its denominator, n_classified) when either the actual
or the predicted margin is a tie, and counts those rows separately.
"""

import numpy as np
import pandas as pd


def _residuals(actual, predicted) -> np.ndarray:
    if pd.isna(actual).any() or pd.isna(predicted).any():
        raise ValueError("Null actual or predicted values; select the rows to score explicitly")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError(f"Need equal, non-empty lengths; got {actual.shape} actual and {predicted.shape} predicted")
    return actual - predicted


def mae(actual, predicted) -> float:
    return float(np.abs(_residuals(actual, predicted)).mean())


def rmse(actual, predicted) -> float:
    return float(np.sqrt((_residuals(actual, predicted) ** 2).mean()))


def bias(actual, predicted) -> float:
    """Mean residual, actual - predicted: positive means Democrats outperformed the forecast."""
    return float(_residuals(actual, predicted).mean())


def winner(margin) -> pd.Series:
    """"D" for a positive margin, "R" for a negative one, "tie" for exactly 0. Nulls raise."""
    margin = pd.Series(margin, dtype=float)
    if margin.isna().any():
        raise ValueError("Null margins; select the rows to classify explicitly")
    return pd.Series(np.select([margin > 0, margin < 0], ["D", "R"], "tie"), index=margin.index)


def winner_counts(actual, predicted) -> dict:
    """Winner classification by margin sign, over the rows classified (neither margin a tie; the ties are counted):

        missed_races      wrong predicted winner; missed_pct = 100 * missed_races / n_classified
        accuracy_pct      100 * correct / n_classified (= 100 - missed_pct)
        precision_D_pct   100 * (predicted D and the D won) / predicted D
        precision_R_pct   100 * (predicted R and the R won) / predicted R

    plus the 2x2 confusion counts actual_<D|R>_pred_<D|R>. A percentage with a zero denominator is NaN, so empty input
    gives zero counts and NaN percentages."""
    actual, predicted = winner(actual).to_numpy(), winner(predicted).to_numpy()
    if actual.shape != predicted.shape:
        raise ValueError(f"Need equal lengths; got {actual.shape} actual and {predicted.shape} predicted")
    classified = (actual != "tie") & (predicted != "tie")
    missed = int((classified & (actual != predicted)).sum())
    n = int(classified.sum())
    def pct(part: int, whole: int) -> float:
        return 100 * part / whole if whole else np.nan

    out = {"n_classified": n, "n_actual_tie": int((actual == "tie").sum()),
           "n_predicted_tie": int((predicted == "tie").sum()), "missed_races": missed,
           "missed_pct": pct(missed, n), "accuracy_pct": pct(n - missed, n)}
    for p in "DR":
        called = classified & (predicted == p)
        out[f"precision_{p}_pct"] = pct(int((called & (actual == p)).sum()), int(called.sum()))
    for a in "DR":
        for p in "DR":
            out[f"actual_{a}_pred_{p}"] = int(((actual == a) & (predicted == p)).sum())
    return out
