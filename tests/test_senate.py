import json
from pathlib import Path

import pandas as pd
import pytest

import senate_county_results as county
import senate_pvi
import senate_targets
from incumbency import regular_class
from senate_schema import AT_LARGE, OUTPUTS, check_output, read_output

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/senate"
built = pytest.mark.skipif(not (PROCESSED / "county_results.csv").exists(), reason="Senate outputs not built")


def test_pvi_weights_the_two_presidential_years_before_the_target_year(monkeypatch):
    # SYNTHETIC presidential results and national baseline
    national = pd.Series({2000: 50.0, 2004: 48.0, 2008: 54.0})
    results = pd.DataFrame({"year": [2000, 2004, 2008] * 2, "state": ["AA"] * 3 + ["BB"] * 3,
                            "dem_2p_norm": [60.0, 58.0, 99.0, 40.0, 44.0, 0.0]})
    monkeypatch.setattr(senate_pvi, "YEARS", [2008])
    out, dropped = senate_pvi.build_pvi(results, national, ["state"])
    out = out.set_index("state")
    assert senate_pvi.prior_pres_years(2008, national.index) == (2004, 2000)  # never 2008 itself
    assert (out.loc["AA", "prev_pres_year"], out.loc["AA", "anteprev_pres_year"]) == (2004, 2000)
    # AA: 0.75 * (58 - 48) + 0.25 * (60 - 50) = 10; BB: 0.75 * (44 - 48) + 0.25 * (40 - 50) = -5.5
    assert out.loc["AA", "pvi"] == pytest.approx(10.0) and out.loc["BB", "pvi"] == pytest.approx(-5.5)
    assert (out["partisan_lean"] == 2 * out["pvi"]).all()
    assert dropped == []


def test_county_pvi_leaves_out_a_county_missing_from_one_presidential_year(monkeypatch):
    national = pd.Series({2000: 50.0, 2004: 50.0})
    results = pd.DataFrame({"year": [2000, 2004, 2004], "state": "CO", "county_id": ["08001", "08001", "08014"],
                            "county_fips": ["08001", "08001", "08014"], "county_name": ["Adams", "Adams", "Broomfield"],
                            "dem_2p_norm": [55.0, 57.0, 52.0]})  # SYNTHETIC
    monkeypatch.setattr(senate_pvi, "YEARS", [2006])
    out, dropped = senate_pvi.build_pvi(results, national, ["state", "county_id", "county_fips"])
    assert out["county_fips"].tolist() == ["08001"] and len(dropped) == 1


def test_surname_handles_last_first_suffixes_and_punctuation():
    names = pd.Series(["MURKOWSKI, LISA", "Robert P Casey Jr", "BOB CASEY JR.", "Kevin J. O'Connor", "ANGUS S. KING, JR."])
    assert county.surname(names).tolist() == ["MURKOWSKI", "CASEY", "CASEY", "OCONNOR", "KING"]


@built
def test_outputs_match_their_schemas():
    for name in OUTPUTS:
        assert check_output(read_output(name), name) == [], name


@built
def test_classes_follow_the_calendar_and_specials_are_other_classes():
    raw = read_output("targets_raw.csv")
    regular = raw[~raw["is_special_election"]]
    assert (regular["class"] == regular["year"].map(regular_class)).all()
    special = raw[raw["is_special_election"]]
    assert (special["class"] != special["year"].map(regular_class)).all()


@built
def test_main_tables_use_the_election_day_round():
    raw = read_output("targets_raw.csv").set_index(["year", "state", "class"])
    supp = read_output("targets_supp.csv")
    # GA 2020 regular: Perdue led on November 3; Ossoff won the January runoff
    ga = raw.loc[(2020, "GA", 2)]
    assert (ga["dem_votes"], ga["gop_votes"], ga["winner_party"]) == (2374519, 2462617, "D")
    assert "election_day_leader_lost" in ga["quality_flags"]
    runoff = supp[(supp["year"] == 2020) & (supp["state"] == "GA") & (supp["class"] == 2) & (supp["contest"] == "runoff")]
    assert runoff["election_date"].item() == "2021-01-05"
    # LA 2014: the official November jungle round, not MEDSL's runoff-plus-1,000,000 counts
    la = raw.loc[(2014, "LA", 2)]
    assert la["race_type"] == "dr_jungle" and la["gop_votes"] == 819777  # Cassidy 603,048 + Maness + Clements
    # ME 2012: King caucuses with Democrats, so he's counted as the Democrat and Dill (D) as third party
    me = raw.loc[(2012, "ME", 1)]
    assert (me["dem_votes"], me["race_type"], me["winner_party"]) == (370580, "dr", "D")


@built
def test_imputation_never_changes_the_observed_votes():
    raw = read_output("targets_raw.csv").set_index(["year", "state", "class"])
    targets = read_output("targets.csv").set_index(["year", "state", "class"])
    votes = ["dem_votes", "gop_votes", "3p_votes", "total_votes"]
    pd.testing.assert_frame_equal(targets[votes], raw.loc[targets.index, votes])
    imputed = targets["imputation_method"].notna()
    assert targets.loc[imputed, "adj_margin_2p_norm"].notna().all()


@built
def test_at_large_rows_equal_statewide():
    state_pvi = read_output("state_pvi.csv").set_index(["year", "state"])
    county_pvi = read_output("county_pvi.csv")
    at_large = county_pvi[county_pvi["is_at_large"]].set_index(["year", "state"])
    assert set(at_large.index.get_level_values("state")) == set(AT_LARGE)
    pd.testing.assert_series_equal(at_large["pvi"], state_pvi.loc[at_large.index, "pvi"])
    assert not county_pvi.loc[~county_pvi["is_at_large"], "state"].isin(AT_LARGE).any()

    raw = read_output("targets_raw.csv").set_index(["year", "state", "class"])
    results = read_output("county_results.csv")
    rows = results[results["is_at_large"]].set_index(["year", "state", "class"])
    assert set(rows.index) == set(raw[raw.index.get_level_values("state").isin(AT_LARGE)].index)
    for c in ["dem_votes", "gop_votes", "3p_votes", "total_votes"]:
        assert (rows[c].astype("int64") == raw.loc[rows.index, c].astype("int64")).all()


@built
def test_corrected_heideman_races_match_statewide():
    # Kansas City, MO is tabulated apart from its counties; leaving it out drops ~100k D votes from Jackson County.
    # Heideman's NH 2016, NY 2016, SC 2016, DE 2018 and NJ 2018 rows were 27k-210k votes short; AR and ID 2022 lack a
    # county. After the corrections no contest differs from the statewide result.
    votes = ["dem_votes", "gop_votes"]
    for county_file, state_file, key in [("county_results.csv", "targets_raw.csv", ["year", "class"]),
                                         ("pres_county_votes.csv", "pres_state_votes.csv", ["year"])]:
        counties = read_output(county_file)
        statewide = read_output(state_file)
        sums = counties[counties["state"] == "MO"].groupby(key)[votes].sum()
        diff = sums - statewide[statewide["state"] == "MO"].set_index(key)[votes]
        assert (diff.abs() < 1000).all().all(), diff
    counties = read_output("county_results.csv")
    statewide = read_output("targets_raw.csv").set_index(["year", "state", "class"])
    for year, state, class_, _ in county.OPENELECTIONS_ROUNDS:
        rows = counties[(counties["year"] == year) & (counties["state"] == state) & (counties["class"] == class_)]
        non_county = county.NON_COUNTY_VOTES.get((year, state, class_), (0, 0, 0))[:2]
        diff = rows[votes].sum() - (statewide.loc[(year, state, class_), votes] - non_county)
        assert (diff.abs() <= 5).all(), (year, state, diff)
    flagged = counties[counties["quality_flags"].fillna("").str.contains(county.SUM_DIFFERS_FLAG)]
    assert flagged.empty, flagged[["year", "state", "class"]].drop_duplicates()


@built
def test_state_pvi_recomputes_from_the_presidential_files():
    pres = read_output("pres_state_votes.csv").set_index(["year", "state"])
    national = pd.read_csv(ROOT / "data/interim/presidential.csv").set_index("pres_year")["dem_pct_2p_norm"]
    pvi = read_output("state_pvi.csv").set_index(["year", "state"])
    for year, state, prev, ante in [(2026, "OH", 2024, 2020), (2008, "VA", 2004, 2000), (2018, "TX", 2016, 2012)]:
        votes = pres.loc[[(prev, state), (ante, state)]]
        share = 100 * votes["dem_votes"] / (votes["dem_votes"] + votes["gop_votes"])
        expected = 0.75 * (share.iloc[0] - national[prev]) + 0.25 * (share.iloc[1] - national[ante])
        assert pvi.loc[(year, state), "pvi"] == pytest.approx(expected)


@built
def test_nat_political_f_is_the_house_file():
    house = pd.read_csv(ROOT / "data/processed/house/nat_political_f.csv")
    pd.testing.assert_frame_equal(read_output("nat_political_f.csv"), house)


def test_runoff_dates_cover_every_runoff_in_the_fixes():
    # every round fix that makes a runoff, and every LA official runoff, has a date
    dates = json.loads(senate_targets.ELECTION_DATES_PATH.read_text())
    assert all(str(y) in dates for y in range(senate_targets.FIRST_YEAR, senate_targets.LAST_YEAR + 1, 2))
    for (year, state, special), stages in senate_targets.ROUND_FIXES.items():
        if "runoff" in stages.values():
            assert (year, state, special) in senate_targets.RUNOFF_DATES
    for (year, state), (_, runoff) in senate_targets.LA_OFFICIAL.items():
        if runoff:
            assert (year, state, False) in senate_targets.RUNOFF_DATES

