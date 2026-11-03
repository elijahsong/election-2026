from pathlib import Path

import numpy as np
import pytest

import senate.regression as sr
from senate import stage1

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed/senate"
pytestmark = pytest.mark.skipif(not (PROCESSED / "modeling.csv").exists(), reason="Senate outputs not built")


@pytest.fixture(scope="module")
def data():
    return sr.load_data()


def test_s1_imposes_2_5_points_of_incumbency_and_s0_none(data):
    s0 = stage1.fit_predict(stage1.SENATE_S0, [2006, 2008, 2010], 2012, data=data)
    s1 = stage1.fit_predict(stage1.SENATE_S1, [2006, 2008, 2010], 2012, data=data)
    assert "inc_signed" not in s0.coefficients and s1.coefficients["inc_signed"] == 2.5
    rows = data[(data["year"] == 2012) & data["observed"]].reset_index(drop=True)
    estimated = s1.model.predict(rows).to_numpy()  # the fitted terms only
    assert np.allclose(s1.predictions["prediction"] - estimated, 2.5 * rows["inc_signed"])


def test_windows_are_chronological_and_refuse_the_predicted_year(data):
    assert stage1.expanding_window(stage1.SENATE_S1, 2024) == list(range(2006, 2024, 2))
    with pytest.raises(ValueError, match="earlier years only"):
        stage1.fit_predict(stage1.SENATE_S1, [2006, 2024], 2024, data=data)
    rolling = stage1.rolling_predictions(stage1.SENATE_S1, eval_years=[2018], data=data)
    assert (rolling[2018].predictions["train_end"] == 2016).all()


def test_rolling_rows_are_scored_races_and_2026_predicts_every_race(data):
    result = stage1.fit_predict(stage1.SENATE_S0, stage1.expanding_window(stage1.SENATE_S0, 2022), 2022, data=data)
    assert (2022, "AK", 3) not in set(map(tuple, result.predictions[sr.KEY].values))  # EXCLUDED_RACES
    assert result.scored[["competitive_ex_ante", "competitive_ex_post"]].dtypes.eq(bool).all()
    forecast = stage1.forecast(stage1.SENATE_S1, data=data)
    assert len(forecast.predictions) == 35 and forecast.predictions["actual"].isna().all()
    assert not forecast.predictions.duplicated(sr.KEY).any() and forecast.metrics is None
