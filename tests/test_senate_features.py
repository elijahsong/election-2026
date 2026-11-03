import pandas as pd
import pytest

import house_features
import senate_acs
import senate_schema


def test_state_rows_compare_with_the_nation_and_drop_self_comparisons():
    national = pd.Series({f: 10.0 for f in senate_acs.FIELDS} | {"median_income": 50_000.0})
    states = pd.DataFrame({f: [12.0, 10.0] for f in senate_acs.FIELDS} | {"median_income": [55_000.0, 50_000.0]},
                          index=pd.Index(["CA", "WY"], name="state"))
    out = senate_acs.state_rows(2026, national, states)
    assert list(out.columns) == senate_acs.COLUMNS and not any(c.startswith("rel_state") for c in out)
    assert out["acs_vintage"].eq("2024_1yr").all()  # 2026 borrows the 2024 1-year vintage, as the House does
    assert out["rel_college"].tolist() == [2.0, 0.0] and out.loc[1, "log_rel_income"] == 0.0


def test_fec_features_with_the_senate_key():
    fec = pd.DataFrame({"year": [2024, 2024], "state": ["MO", "MO"], "class": [1, 3],
                        "fundraising_advantage": [14.0, 0.5], "spending_advantage": [-9.0, 0.2],
                        "small_dollar_share_diff": [0.1, None], "dem_fec_missing": False, "gop_fec_missing": False,
                        "dem_match_status": ["no_fec_record", "name_full"], "gop_match_status": "name_full"})
    out = house_features.fec_features(fec, key=senate_schema.KEY, known_gaps=[])
    assert out["fundraising_advantage"].tolist() == [house_features.FEC_LOG_CAP, 0.5]
    assert out.loc[0, "spending_advantage"] == -house_features.FEC_LOG_CAP
    blanked = house_features.fec_features(fec, key=senate_schema.KEY, known_gaps=[(2024, "MO", 1)])
    assert blanked.loc[0, house_features.FEC_FEATURES].isna().all() and blanked.loc[1, "fundraising_advantage"] == 0.5
    with pytest.raises(ValueError, match="no_fec_record"):
        house_features.fec_features(fec, key=senate_schema.KEY, known_gaps=[(2024, "MO", 3)])


def test_senate_modeling_columns_are_the_house_ones_less_state_comparisons():
    house = list(pd.read_csv("data/processed/house/modeling.csv", nrows=0).columns)
    expected = [{"district": "class"}.get(c, c) for c in house if not c.startswith("rel_state")
                and c != "log_rel_state_income"]
    assert list(senate_schema.MODELING) == expected
