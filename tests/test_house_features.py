import pandas as pd
import pytest

import house_features as hf


def test_tables_join_and_keep_missing_generic_ballot_blank():
    keys = pd.DataFrame({"year": [2024, 2024, 2026], "state": ["AK", "NC", "AK"], "district": [0, 1, 0]})
    gb = pd.DataFrame({"election": [2024], "cutoff_date": ["2024-11-04"], "margin_raw": [1.5],
                       "dem_estimate": [45.0], "gop_estimate": [43.5], "und_estimate": [11.5]})
    lean = keys.assign(partisan_lean=[-10.0, 2.0, -12.0], cook_pvi=0.0)
    inc = keys.assign(inc_signed=[-1, 1, 0], status="incumbent")
    acs_supplement = keys.assign(acs_vintage="2024_1yr", college=[30.0, 35.0, 31.0], rel_state_housing_burden=0.0,
                                 extra_after="dropped")

    basic = hf.basic_features(gb, lean, inc)
    nat = hf.nat_political_features(gb)
    modeling = hf.modeling_table(basic, hf.acs_features(acs_supplement), nat)

    assert list(basic.columns) == ["year", "state", "district", "gen_margin_raw", "partisan_lean", "inc_signed"]
    assert list(nat.columns) == ["year", "gb_cutoff_date", "gb_margin_raw", "gb_dem_pct", "gb_gop_pct", "gb_und_pct"]
    assert "extra_after" not in modeling and len(modeling) == 3
    assert modeling.loc[modeling["year"] == 2024, "gen_margin_raw"].tolist() == [1.5, 1.5]
    assert modeling.loc[modeling["year"] == 2026, ["gen_margin_raw", "gb_dem_pct"]].isna().all().all()

    with pytest.raises(ValueError, match="lacks 1 keys"):
        hf.basic_features(gb, lean, inc.iloc[:2])
