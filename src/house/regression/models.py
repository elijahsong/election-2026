"""Estimator wrappers with a common interface: `model.fit(X, y)` returns the model, `model.predict(X)` a Series,
`model.coefficients()` a DataFrame of fitted terms, and `model.fit_stats()` a dict of fit statistics (the last two are
what experiment tracking records).

To add an estimator (WLS, Ridge, Lasso, XGBoost, ...), write a class with that interface and add it to MODELS. Any
model-specific preprocessing belongs inside its `fit`, so it is fitted on training rows only: e.g. a sklearn Pipeline
whose StandardScaler is fitted in `fit` and reused in `predict`. Feature values reach every model in their original
units.
"""

import pandas as pd
import statsmodels.api as sm


class OLSModel:
    """Ordinary least squares via statsmodels; after fitting, `results` is the statsmodels RegressionResults."""

    def __init__(self, *, add_constant: bool):
        self.add_constant = add_constant
        self.features: list[str] | None = None
        self.results = None

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        return sm.add_constant(X, has_constant="add") if self.add_constant else X

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "OLSModel":
        self.features = list(X.columns)
        self.results = sm.OLS(y, self._design(X), missing="raise").fit()
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.results is None:
            raise RuntimeError("Fit the model before predicting")
        if list(X.columns) != self.features:
            raise ValueError(f"Prediction columns {list(X.columns)} differ from the fitted columns {self.features}")
        return pd.Series(self.results.predict(self._design(X)).to_numpy(), index=X.index, name="prediction")

    def coefficients(self) -> pd.DataFrame:
        """One row per term: coef, std_err, t, p_value."""
        if self.results is None:
            raise RuntimeError("Fit the model before reading its coefficients")
        r = self.results
        return pd.DataFrame({"coef": r.params, "std_err": r.bse, "t": r.tvalues, "p_value": r.pvalues})

    def fit_stats(self) -> dict:
        if self.results is None:
            raise RuntimeError("Fit the model before reading its fit statistics")
        r = self.results
        return {"nobs": int(r.nobs), "r2": r.rsquared, "r2_adj": r.rsquared_adj, "resid_std": r.scale ** 0.5,
                "aic": r.aic, "bic": r.bic}


MODELS = {"ols": OLSModel}


def make_model(model_type: str, **params):
    """An unfitted model of `model_type` (a MODELS key), built with `params`."""
    if model_type not in MODELS:
        raise ValueError(f"Unknown model_type {model_type!r}; expected one of {sorted(MODELS)}")
    return MODELS[model_type](**params)
