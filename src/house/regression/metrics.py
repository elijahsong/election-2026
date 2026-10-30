"""Error metrics on predicted vs. actual margins. Error = predicted - actual, so a positive bias over-predicts the
Democratic margin. Nulls raise rather than being dropped: subset the rows explicitly before scoring."""

import numpy as np
import pandas as pd


def _errors(actual, predicted) -> np.ndarray:
    if pd.isna(actual).any() or pd.isna(predicted).any():
        raise ValueError("Null actual or predicted values; select the rows to score explicitly")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.size == 0:
        raise ValueError(f"Need equal, non-empty lengths; got {actual.shape} actual and {predicted.shape} predicted")
    return predicted - actual


def mae(actual, predicted) -> float:
    return float(np.abs(_errors(actual, predicted)).mean())


def rmse(actual, predicted) -> float:
    return float(np.sqrt((_errors(actual, predicted) ** 2).mean()))


def bias(actual, predicted) -> float:
    """Mean signed error, predicted - actual."""
    return float(_errors(actual, predicted).mean())
