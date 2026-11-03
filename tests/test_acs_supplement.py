import math

import pandas as pd
import pytest
from conftest import COUNTS

import acs
import acs_supplement

COLLEGE_CELLS = acs.ACS_FIELDS[2018]["college"].num  # C15002 (1-year), shared by 2010 and 2018


@pytest.fixture
def acs_csv(fake_api, tmp_path):
    """A 2010 + 2018 acs.csv built through acs.py, plus the fake API state for setting reference values."""
    path = tmp_path / "acs.csv"
    for year in (2010, 2018):
        state = fake_api(year)
        acs.build_acs_year(year, COUNTS, path)

    # nation: 25% college, $60,000 income, 6% unemployment (B23025 only, so 2010's C23001 national rate stays 5%)
    us = state["us:1"][0]
    us.update({v: "30000" for v in COLLEGE_CELLS}, B19013_001E="60000", B23025_005E="21000")
    # nation: renters' burdened bins doubled, so (8,000 + 8,000 + 12,000) / (13,000 + 18,000 + 27,000) = 48.28%
    us.update({v: "2000" for v in acs.BURDEN_CELLS["renters"][acs.BURDENED]})
    # California: 30% college, $70,000 income; every other state matches its districts exactly
    ca = next(r for r in state["state:*"] if r["state"] == "06")
    ca.update({v: "36000" for v in COLLEGE_CELLS}, B19013_001E="70000")
    ca.update({f"{t}03002_012E": "175000" for t in "BC"})  # 25% Hispanic (districts: 140,000 / 700,000 = 20%)
    return path


def test_relative_columns(acs_csv, tmp_path):
    out = acs_supplement.build_supplement(acs_csv, tmp_path / "supplement.csv")
    assert list(out.columns) == acs.COLUMNS + acs_supplement.SUPPLEMENT_COLUMNS
    assert out["year"].value_counts().to_dict() == {2010: 435, 2018: 435}
    pd.testing.assert_frame_equal(out[acs.COLUMNS], pd.read_csv(acs_csv, dtype={"state": str}))

    y18 = out[out["year"] == 2018]
    assert (y18["rel_college"] == 8.33).all()  # 33.33 - 25.00
    assert (y18["log_rel_income"] == round(math.log(65_432 / 60_000), 4)).all()
    assert (y18["rel_unemployment"] == -1.0).all()
    assert (out.loc[out["year"] == 2010, "rel_unemployment"] == 0.0).all()
    assert (out["rel_housing_burden"] == round(44.44 - 48.28, 2)).all()  # district 44.44% vs nation 48.28%

    ca, rest = out[out["state"] == "CA"], out[out["state"] != "CA"]
    assert (ca["rel_state_college"] == 3.33).all()
    assert (ca["log_rel_state_income"] == round(math.log(65_432 / 70_000), 4)).all()
    assert (ca["rel_state_hispanic"] == -5.0).all()  # 20.00 - 25.00
    # districts equal to their state (including every at-large one) compare as exactly zero, never -0.0
    for col in ["rel_state_college", "log_rel_state_income", "rel_state_unemployment", "rel_state_housing_burden",
                "rel_state_hispanic"]:
        assert (rest[col] == 0).all() and not rest[col].map(lambda x: math.copysign(1, x) < 0).any()


def test_rebuild_is_idempotent(acs_csv, tmp_path):
    out_path = tmp_path / "supplement.csv"
    acs_supplement.build_supplement(acs_csv, out_path)
    first, source = out_path.read_bytes(), acs_csv.read_bytes()
    acs_supplement.build_supplement(acs_csv, out_path)
    assert out_path.read_bytes() == first and acs_csv.read_bytes() == source


def test_log_ratio_matches_log_and_is_zero_at_parity():
    x = pd.Series([1, 59_999, 60_000, 60_001, 250_001])
    assert acs_supplement._log_ratio(x, 60_000).tolist() == pytest.approx([math.log(v / 60_000) for v in x], rel=1e-12)
    assert acs_supplement._log_ratio(x, 60_000)[2] == 0.0


def test_rejects_output_over_input(acs_csv):
    with pytest.raises(ValueError, match="separate file"):
        acs_supplement.build_supplement(acs_csv, acs_csv)


def test_rejects_wrong_schema(acs_csv, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.read_csv(acs_csv).drop(columns="poverty").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="schema"):
        acs_supplement.build_supplement(bad, tmp_path / "out.csv")


def test_rejects_unexpected_vintage(acs_csv, tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(acs_csv)
    df.loc[df["year"] == 2018, "acs_vintage"] = "2017_1yr"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="expected 2018_1yr"):
        acs_supplement.build_supplement(bad, tmp_path / "out.csv")


@pytest.fixture
def borrowed_csv(acs_csv, tmp_path, monkeypatch):
    """acs_csv plus a 2026 copy of the 2018 rows (half labelled rebuilt), with 2026 borrowing 2018's vintage."""
    monkeypatch.setattr(acs_supplement, "REFERENCE_YEAR", {2026: 2018})
    df = pd.read_csv(acs_csv)
    y26 = df[df["year"] == 2018].assign(year=2026)
    y26.loc[y26.index[::2], "acs_vintage"] = "2018_1yr_rebuilt"
    path = tmp_path / "with_2026.csv"
    pd.concat([df, y26]).to_csv(path, index=False)
    return path


def test_borrowed_vintage_year_uses_reference_values(borrowed_csv, tmp_path):
    out = acs_supplement.build_supplement(borrowed_csv, tmp_path / "out.csv")
    cols = ["state", "district", *acs_supplement.SUPPLEMENT_COLUMNS]
    y18, y26 = (out.loc[out["year"] == y, cols].reset_index(drop=True) for y in (2018, 2026))
    pd.testing.assert_frame_equal(y26, y18)
    assert set(out.loc[out["year"] == 2026, "acs_vintage"]) == {"2018_1yr", "2018_1yr_rebuilt"}


def test_rebuilt_label_only_allowed_for_borrowed_years(borrowed_csv, tmp_path):
    bad = tmp_path / "bad.csv"
    df = pd.read_csv(borrowed_csv)
    df.loc[df["year"] == 2018, "acs_vintage"] = "2018_1yr_rebuilt"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="expected 2018_1yr$"):
        acs_supplement.build_supplement(bad, tmp_path / "out.csv")

    df = pd.read_csv(borrowed_csv)
    df.loc[df["year"] == 2026, "acs_vintage"] = "2016-2020_5yr"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="expected 2018_1yr or 2018_1yr_rebuilt"):
        acs_supplement.build_supplement(bad, tmp_path / "out.csv")


def test_missing_reference_estimate_raises(acs_csv, fake_api, tmp_path):
    fake_api(2018)["state:*"][0]["B19013_001E"] = "-666666666"
    with pytest.raises(ValueError, match="AL: B19013_001E"):
        acs_supplement.build_supplement(acs_csv, tmp_path / "out.csv")
