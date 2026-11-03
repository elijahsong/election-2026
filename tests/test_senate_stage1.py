from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import senate.regression as sr

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed/senate"
FORMULA = f"{sr.TARGET} ~ partisan_lean + gen_margin_raw"


def synthetic(seed=0):
    # SYNTHETIC races, 2006-2014; 2010 AA has a regular (class 1) and a special (class 2) with different predictors
    rng = np.random.default_rng(seed)
    rows = [{"year": y, "state": s, "class": 1, "special": 0} for y in range(2006, 2016, 2) for s in "ABCDEF"]
    rows.append({"year": 2010, "state": "A", "class": 2, "special": 1})
    df = pd.DataFrame(rows)
    df["state"] = df["state"] + "A"
    df["partisan_lean"] = rng.normal(0, 20, len(df))
    df["gen_margin_raw"] = df["year"].map({2006: 8, 2008: 10, 2010: -7, 2012: 4, 2014: -6}).astype(float)
    df["inc_signed"] = rng.choice([-1, 0, 1], len(df))
    df["gb_margin_2p_norm"] = df["gen_margin_raw"]
    df[sr.TARGET] = df["partisan_lean"] + df["gen_margin_raw"] + 3 * df["inc_signed"] + rng.normal(0, 3, len(df))
    return df.assign(observed=True, imputed=False)


def test_windows_never_see_the_eval_year_or_later():
    data = synthetic()
    assert sr.training_rows(data, 2012)["year"].max() == 2010
    base = sr.backtest(data, formula=FORMULA, eval_years=[2010, 2012], name="base")
    # rewriting every target from 2012 on changes nothing about the 2010 or 2012 windows' fits
    shifted = data.assign(**{sr.TARGET: data[sr.TARGET].where(data["year"] < 2012, 99.0)})
    moved = sr.backtest(shifted, formula=FORMULA, eval_years=[2010, 2012], name="moved")
    pd.testing.assert_frame_equal(base.coefficients, moved.coefficients)


def test_equal_year_weights_give_each_year_the_same_total():
    train = synthetic()  # 2010 has 7 rows, the other years 6
    totals = sr.training_weights(train, equal_year=True).groupby(train["year"]).sum()
    assert np.allclose(totals, totals.iloc[0]) and sr.training_weights(train).eq(1).all()


def test_predictions_stay_aligned_with_their_race_and_add_back_fixed_terms():
    data = synthetic()
    run = sr.backtest(data, formula=FORMULA, eval_years=[2010], name="fixed", fixed={"inc_signed": 3.0})
    pred = run.predictions.set_index(sr.KEY)
    assert {(2010, "AA", 1), (2010, "AA", 2)} <= set(pred.index)
    coef = run.coefficients.loc[2010]
    rows = data[data["year"] == 2010].set_index(sr.KEY)
    expected = coef["Intercept"] + coef["partisan_lean"] * rows["partisan_lean"] \
        + coef["gen_margin_raw"] * rows["gen_margin_raw"] + 3.0 * rows["inc_signed"]
    assert np.allclose(pred.loc[rows.index, "prediction"], expected)
    assert np.allclose(pred["residual"], pred["actual"] - pred["prediction"])


@pytest.mark.skipif(not (PROCESSED / "modeling.csv").exists(), reason="Senate outputs not built")
def test_real_two_race_state_years_are_separate_rows():
    data = sr.load_data()
    mn = data[(data["year"] == 2018) & (data["state"] == "MN")].set_index("class")
    assert mn["special"].to_dict() == {1: 0, 2: 1} and mn[sr.TARGET].notna().all()
    assert not data.duplicated(sr.KEY).any()
    ak = data.set_index(sr.KEY).loc[(2022, "AK", 3)]
    assert ak["excluded"] and not ak["observed"] and not ak["imputed"]
