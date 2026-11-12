import pandas as pd
import pytest

import house_features as hf


def test_tables_join_and_keep_missing_generic_ballot_blank(monkeypatch):
    keys = pd.DataFrame({"year": [2024, 2024, 2026], "state": ["AK", "NC", "AK"], "district": [0, 1, 0]})
    gb = pd.DataFrame({"election": [2024], "cutoff_date": ["2024-11-04"], "margin_raw": [1.5],
                       "dem_estimate": [45.0], "gop_estimate": [43.5], "und_estimate": [11.5]})
    lean = keys.assign(partisan_lean=[-10.0, 2.0, -12.0], cook_pvi=0.0)
    inc = keys.assign(inc_signed=[-1, 1, 0], status="incumbent")
    acs_supplement = keys.assign(acs_vintage="2024_1yr", college=[30.0, 35.0, 31.0], rel_state_housing_burden=0.0,
                                 rel_state_hispanic=0.0, extra_after="dropped")

    basic = hf.basic_features(gb, lean, inc)
    nat = hf.nat_political_features(gb)
    fec = keys.assign(fundraising_advantage=[14.0, -0.5, None], spending_advantage=[-9.0, 0.3, None],
                      small_dollar_share_diff=[0.1, -0.2, None], dem_fec_missing=False, gop_fec_missing=[True, False, False],
                      dem_match_status="name_full", gop_match_status=["no_nominee", "name_full", "nominee_unresolved"],
                      dem_name="dropped")
    monkeypatch.setattr(hf, "FEC_KNOWN_GAPS", [])
    fec_f = hf.fec_features(fec)
    prices_f = keys.assign(census_region=["west", "south", "west"], cpi_region_yoy=[3.1, 2.4, None])
    war_f = keys.assign(war_lag=[0.0, 3.5, 0.0], war_lag_available=[False, True, False])
    modeling = hf.modeling_table(basic, hf.acs_features(acs_supplement), fec_f, prices_f, war_f, nat)
    assert modeling["war_lag"].tolist() == [0.0, 3.5, 0.0]

    assert list(basic.columns) == ["year", "state", "district", "gen_margin_raw", "partisan_lean", "inc_signed"]
    assert list(nat.columns) == ["year", "gb_cutoff_date", "gb_margin_raw", "gb_dem_pct", "gb_gop_pct", "gb_und_pct"]
    assert "extra_after" not in modeling and "dem_name" not in modeling and len(modeling) == 3
    assert modeling["fundraising_advantage"].tolist()[:2] == [hf.FEC_LOG_CAP, -0.5]  # capped
    assert modeling["spending_advantage"].tolist()[:2] == [-hf.FEC_LOG_CAP, 0.3]
    assert modeling.loc[modeling["year"] == 2024, "gen_margin_raw"].tolist() == [1.5, 1.5]
    assert modeling.loc[modeling["year"] == 2026, ["gen_margin_raw", "gb_dem_pct"]].isna().all().all()
    assert modeling["cpi_region_yoy"].tolist()[:2] == [3.1, 2.4] and pd.isna(modeling.loc[2, "cpi_region_yoy"])

    with pytest.raises(ValueError, match="lacks 1 keys"):
        hf.basic_features(gb, lean, inc.iloc[:2])

