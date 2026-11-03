import dataclasses

import numpy as np
import pandas as pd
import pytest
from patsy import ModelDesc

from house import stage1
from house.stage1 import HOUSE_ENSEMBLE_AB_85_15, HOUSE_MODEL_A, MODEL_A, MODEL_B, TARGET

YEARS = list(range(2006, 2028, 2))  # 2026 has predictors only


def synthetic_tables(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SYNTHETIC rows, not election data: 6, 9, or 12 districts per year (so equal-year weights differ from equal
    weights), a target linear in every Model A and B predictor plus noise, and no 2026 targets."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, year in enumerate(YEARS):
        for d in range(1, 7 + 3 * (i % 3)):
            rows.append({"year": year, "state": "CA", "district": d, "partisan_lean": rng.uniform(-30, 30),
                         "gen_margin_raw": rng.uniform(-8, 8), "inc_signed": int(rng.choice([-1, 0, 1])),
                         "fundraising_advantage": rng.uniform(-3, 3), "cpi_region_log_yoy": rng.uniform(0, 8)})
    predictors = pd.DataFrame(rows)
    predictors["gb_margin_2p_norm"] = predictors["gen_margin_raw"]  # the default ex-ante competitive subset uses it
    p = predictors
    signal = (1 + 1.0 * p["partisan_lean"] + 1.5 * p["gen_margin_raw"] + 6 * p["inc_signed"]
              + 0.1 * p["partisan_lean"] * p["inc_signed"] + 2 * p["fundraising_advantage"])
    targets = p[["year", "state", "district"]].assign(**{TARGET: signal + rng.normal(0, 3, len(p))})
    targets = targets[targets["year"] < 2026].copy()
    targets["margin_2p_norm"] = targets[TARGET]
    targets["imputation_method"], targets["is_imputed"], targets["race_type"] = None, False, "dr"
    targets["winner_party"] = np.where(targets[TARGET] > 0, "D", "R")
    return predictors, targets


def rhs_terms(formula: str) -> set[str]:
    return {term.name() for term in ModelDesc.from_formula(formula).rhs_termlist}


def test_predictor_definitions_are_fixed():
    assert rhs_terms(MODEL_A.formula) == {"Intercept", "partisan_lean", "gen_margin_raw", "inc_signed",
                                          "partisan_lean:inc_signed"}  # no gb_momentum_30d_raw since 2026-11-02
    assert dict(MODEL_A.fixed) == {"inc_signed": 2.5}
    assert rhs_terms(MODEL_B.formula) == {"Intercept", "partisan_lean", "fundraising_advantage",
                                          "pres_party_signed:is_midterm", "pres_party_signed:cpi_region_log_yoy"}
    assert dict(MODEL_B.fixed) == {}
    assert HOUSE_MODEL_A.components == ((MODEL_A, 1.0),)
    assert HOUSE_ENSEMBLE_AB_85_15.components == ((MODEL_A, 0.85), (MODEL_B, 0.15))
    assert HOUSE_MODEL_A.equal_year and HOUSE_ENSEMBLE_AB_85_15.equal_year
    assert set(stage1.CONFIGS) == {"house_model_a", "house_ensemble_ab_85_15"}

    with pytest.raises(dataclasses.FrozenInstanceError):
        HOUSE_ENSEMBLE_AB_85_15.components = ((MODEL_A, 1.0),)
    with pytest.raises(TypeError):
        MODEL_A.fixed["inc_signed"] = 5.0
    with pytest.raises(ValueError, match="sum to 1"):
        stage1.Stage1Config("bad", ((MODEL_A, 0.85), (MODEL_B, 0.25)))


def test_derived_predictors_are_signed_by_the_presidents_party():
    derived = stage1.add_derived_predictors(pd.DataFrame({"year": [2018, 2020, 2022, 2024, 2026]}))
    assert derived["pres_party_signed"].tolist() == [-1, -1, 1, 1, -1]  # Trump, Trump, Biden, Biden, Trump
    assert derived["is_midterm"].tolist() == [1, 0, 1, 0, 1]
    with pytest.raises(ValueError, match="PRES_PARTY_SIGNED"):
        stage1.add_derived_predictors(pd.DataFrame({"year": [2028]}))


def test_model_a_is_equal_year_wls_with_incumbency_fixed_at_2_5():
    predictors, targets = synthetic_tables()
    result = stage1.fit_predict(HOUSE_MODEL_A, range(2006, 2018, 2), 2018, predictors=predictors, targets=targets)
    run = result.components["A_polling"]
    assert run.config["weights"]["equal_year"] and run.config["fixed"] == {"inc_signed": 2.5}
    assert result.coefficients.loc["inc_signed", "A_polling"] == 2.5 and "inc_signed" not in run.model.params

    # by hand: weight each training race by 1 / (its year's training races) and regress y - 2.5 * inc_signed
    def design(rows):
        return np.column_stack([np.ones(len(rows)), rows["partisan_lean"], rows["gen_margin_raw"],
                                rows["partisan_lean"] * rows["inc_signed"]])

    train = predictors[predictors["year"] < 2018].merge(targets, on=["year", "state", "district"])
    w = 1 / train["year"].map(train["year"].value_counts())
    assert train["year"].value_counts().nunique() == 3  # unequal years, so the weights matter
    y = train[TARGET] - 2.5 * train["inc_signed"]
    beta = np.linalg.lstsq(design(train) * np.sqrt(w.to_numpy())[:, None], y * np.sqrt(w), rcond=None)[0]
    assert np.allclose(run.model.params.to_numpy(), beta)
    unweighted = np.linalg.lstsq(design(train), y, rcond=None)[0]
    assert not np.allclose(beta, unweighted)

    evaluated = predictors[predictors["year"] == 2018]
    assert np.allclose(result.predictions["prediction"], design(evaluated) @ beta + 2.5 * evaluated["inc_signed"])
    assert np.allclose(result.predictions["residual"], result.predictions["actual"] - result.predictions["prediction"])

    # signed incumbency: with partisan_lean = 0 (no interaction), a D incumbent adds +2.5 and an R incumbent -2.5
    by_status = {}
    for inc in [1, 0, -1]:
        changed = predictors.copy()
        row = (changed["year"] == 2018) & (changed["district"] == 1)
        changed.loc[row, ["partisan_lean", "inc_signed"]] = [0.0, inc]
        out = stage1.fit_predict(HOUSE_MODEL_A, range(2006, 2018, 2), 2018, predictors=changed, targets=targets)
        by_status[inc] = out.predictions.loc[out.predictions["district"] == 1, "prediction"].item()
    assert by_status[1] - by_status[0] == pytest.approx(2.5) and by_status[-1] - by_status[0] == pytest.approx(-2.5)


def test_windows_are_chronological_and_never_see_their_own_results():
    predictors, targets = synthetic_tables()
    rolling = stage1.rolling_predictions(HOUSE_ENSEMBLE_AB_85_15, predictors=predictors, targets=targets)
    assert list(rolling) == [2018, 2020, 2022, 2024]
    for year, result in rolling.items():
        for run in result.components.values():
            assert run.config["train_years"] == list(range(2006, year, 2))
        assert result.predictions["year"].eq(year).all() and result.predictions["train_end"].eq(year - 2).all()
        assert result.predictions["actual"].notna().all() and result.metrics is not None

    # 2022's results change only the windows that train on 2022
    shifted = targets.copy()
    shifted.loc[shifted["year"] == 2022, TARGET] += 50
    again = stage1.rolling_predictions(HOUSE_ENSEMBLE_AB_85_15, [2022, 2024], predictors=predictors, targets=shifted)
    assert np.array_equal(again[2022].predictions["prediction"], rolling[2022].predictions["prediction"])
    assert not np.allclose(again[2024].predictions["prediction"], rolling[2024].predictions["prediction"])

    with pytest.raises(ValueError, match="train on earlier years only"):
        stage1.fit_predict(HOUSE_MODEL_A, [2006, 2008, 2010], 2010, predictors=predictors, targets=targets)

    final = stage1.forecast(HOUSE_ENSEMBLE_AB_85_15, predictors=predictors, targets=targets)
    assert final.components["A_polling"].config["train_years"] == list(range(2006, 2026, 2))
    assert final.predictions["year"].eq(2026).all() and len(final.predictions) == (predictors["year"] == 2026).sum()
    assert final.predictions["actual"].isna().all() and final.metrics is None


def test_ensemble_is_exactly_85_a_plus_15_b_on_shared_rows():
    predictors, targets = synthetic_tables()
    predictors.loc[(predictors["year"] == 2020) & (predictors["district"] == 2), "fundraising_advantage"] = np.nan
    data = {"predictors": predictors, "targets": targets}
    with pytest.warns(UserWarning, match="fundraising_advantage"):
        ensemble = stage1.fit_predict(HOUSE_ENSEMBLE_AB_85_15, range(2006, 2020, 2), 2020, **data)
    p = ensemble.predictions
    assert np.array_equal(p["prediction"], 0.85 * p["prediction_A_polling"] + 0.15 * p["prediction_B_fundamentals"])
    assert np.array_equal(p["residual"], p["actual"] - p["prediction"])
    assert ensemble.coefficients.loc["inc_signed", "A_polling"] == 2.5
    assert ensemble.coefficients["B_fundamentals"].notna().sum() == 5  # Intercept + 4 estimated terms

    # A alone doesn't need fundraising_advantage, so it keeps the district the ensemble drops
    alone = stage1.fit_predict(HOUSE_MODEL_A, range(2006, 2020, 2), 2020, **data)
    assert len(alone.predictions) == len(p) + 1
    assert np.array_equal(alone.predictions["prediction"], alone.predictions["prediction_A_polling"])
