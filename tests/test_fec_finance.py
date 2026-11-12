import numpy as np
import pandas as pd
import pytest

import fec_finance
import nominees_2026


@pytest.mark.parametrize("nominee, fec, expected", [
    # suffixes, titles, initials, nicknames and middle names in use
    ("RALPH R. SCOTT, JR.", "SCOTT, RALPH R MR. JR.", "full"),
    ("OSSOFF, THOMAS JONATHAN (JON)", "OSSOFF, T. JONATHAN", "full"),
    ("TOM GRAVES", "GRAVES, JOHN THOMAS MR. JR.", "full"),
    # compound and hyphenated surnames, and a misspelled one
    ("Mario Díaz-Balart", "DIAZ-BALART, MARIO", "full"),
    ("BILL BASCRELL JR", "PASCRELL, WILLIAM J. HON. JR.", "fuzzy"),
    # a conflicting middle initial, or an initial against a different first name, is surname-only
    ("Dan S. Sullivan", "SULLIVAN, DANIEL S", "full"),
    ("Dan S. Sullivan", "SULLIVAN, DAN J", "surname"),
    ("GUS M BILIRAKIS", "BILIRAKIS, MICHAEL", "surname"),
    # filing quirks
    ("NICK LALOTA", "NICK, LALOTA", "variant"),
    ("DOUG LA MALFA", "LAMALFA, DOUG", "full"),
    ("Beth Van Duyne", "VAN DUYNE, ELIZABETH ANN", "full"),
    ("ASHLEY HINSON", "ARENHOLZ, ASHLEY HINSON", "variant"),
    # MEDSL's mis-decoded UTF-8
    ("JESÃ\u009aS G â\u0080\u009cCHUYâ\u0080\u009d GARCÃ\u008dA", "GARCIA, JESUS", "full"),
    ("JOHN SMITH", "JONES, JOHN", None),
])
def test_match_name(nominee, fec, expected):
    assert fec_finance.match_name(nominee, fec) == expected


def records(*rows):
    """FEC records with the columns pick() reads: (candidate_id, fec_name, receipts, coverage_end_date)."""
    df = pd.DataFrame(rows, columns=["candidate_id", "fec_name", "receipts", "coverage_end_date"])
    for field in fec_finance.FINANCE:
        if field not in df:
            df[field] = 0.0
    df["coverage_end_date"] = pd.to_datetime(df["coverage_end_date"])
    return df


def test_pick_duplicate_registrations():
    # one person's two IDs: the only one with filings is used
    pool = records(("H1", "GRAVES, JOHN THOMAS", 100.0, "2018-12-31"), ("H2", "GRAVES, TOM", 0.0, None))
    outcome, rows = fec_finance.pick("TOM GRAVES", pool)
    assert outcome == "duplicate" and rows["candidate_id"].tolist() == ["H1"]
    # identical totals on both: the first by ID
    pool = records(("H9", "COOKE, REBECCA", 50.0, "2024-12-31"), ("H3", "COOKE, REBECCA", 50.0, "2024-12-31"))
    assert fec_finance.pick("REBECCA COOKE", pool)[1]["candidate_id"].tolist() == ["H3"]
    # one name, two people with different filings: ambiguous, so nothing is used
    pool = records(("H1", "ESTES, RON", 100.0, "2018-12-31"), ("H2", "ESTES, RON M.", 5.0, "2018-09-30"))
    assert fec_finance.pick("RON ESTES", pool)[0] == "ambiguous"


def test_pick_surname_only_must_be_unique():
    # a surname-only match (MIKE KELLY files as GEORGE J KELLY JR) is used only when it's the pool's only one
    pool = records(("H1", "KELLY, GEORGE J JR", 1.0, None), ("H2", "KELLY, ANNE", 2.0, None))
    assert fec_finance.pick("MIKE KELLY", pool)[0] == "ambiguous"
    assert fec_finance.pick("MIKE KELLY", pool.iloc[:1])[0] == "surname"
    assert fec_finance.pick("MIKE KELLY", pool.iloc[:1], full_only=True)[0] == "none"


def test_read_webl(tmp_path):
    house = "H6AK00144|BENSON, DIANE E|C|1|DEM|198951.6|0|192486.89|0|0|6464.71|1121.23|0|0|0|0|8018.3|187564.17|" \
            "AK|00||W||L|67|0|10000|12/31/2006|1000|0"
    president = "P60000001|DOE, JANE" + house[house.index("|C|"):]
    path = tmp_path / "webl06.txt"
    path.write_text(house + "\n" + president + "\n")
    out = fec_finance.read_webl(path, 2006)
    assert out["candidate_id"].tolist() == ["H6AK00144"]  # the presidential row is dropped
    row = out.iloc[0]
    assert (row["receipts"], row["disbursements"], row["individual_contributions"]) == (198951.6, 192486.89, 187564.17)
    assert (row["cash_on_hand"], row["debt"], row["district"], row["gen_election"]) == (6464.71, 8018.3, 0, "L")
    assert row["coverage_end_date"] == pd.Timestamp("2006-12-31")
    assert np.isnan(row["individual_unitemized_contributions"]) and pd.isna(row["coverage_start_date"])
    path.write_text(house + "|extra\n")
    with pytest.raises(ValueError, match="fields"):
        fec_finance.read_webl(path, 2006)


def test_add_features():
    df = pd.DataFrame({
        "dem_receipts": [99.0, 0.0, 1.0], "gop_receipts": [9.0, 0.0, 1.0],
        "dem_disbursements": [9.0, 0.0, 1.0], "gop_disbursements": [99.0, 0.0, 1.0],
        "dem_individual_contributions": [100.0, 0.0, 10.0],
        "dem_individual_unitemized_contributions": [25.0, 0.0, 30.0],
        "gop_individual_contributions": [50.0, 10.0, 10.0], "gop_individual_unitemized_contributions": [10.0, 5.0, 5.0],
    })
    out = fec_finance.add_features(df)
    assert list(out.columns) == list(df.columns) + fec_finance.FEATURES
    assert out["fundraising_advantage"].tolist()[:2] == pytest.approx([np.log(100) - np.log(10), 0.0])
    assert out["spending_advantage"].iloc[0] == pytest.approx(np.log(10) - np.log(100))
    assert out["small_dollar_share_dem"].iloc[0] == 0.25 and out["small_dollar_share_gop"].tolist() == [0.2, 0.5, 0.5]
    assert out["small_dollar_share_diff"].iloc[0] == pytest.approx(0.05)
    # a zero denominator leaves the share, and so the difference, undefined
    assert np.isnan(out["small_dollar_share_dem"].iloc[1]) and np.isnan(out["small_dollar_share_diff"].iloc[1])
    # a share outside [0, 1] (30 unitemized of 10 in all) is dropped and flagged, and the FEC amount is kept
    assert np.isnan(out["small_dollar_share_dem"].iloc[2]) and np.isnan(out["small_dollar_share_diff"].iloc[2])
    assert out["small_dollar_share_flag"].tolist()[2] == "dem_share_outside_0_1"
    assert out["small_dollar_share_flag"].iloc[:2].isna().all()
    assert out["dem_individual_unitemized_contributions"].iloc[2] == 30.0


HOUSE_WIKITEXT = """==Alabama==
{|class="wikitable sortable"
!rowspan=2 class="unsortable" |Candidates<ref name=AL>{{cite web |url=https://sos.example/al |title=x}}</ref>
|-
!{{ushr|AL|2|X}}
|nowrap |{{plainlist}}
*{{Party stripe|Democratic Party (US)}}[[Shomari Figures]] (Democratic)<ref name=AL/>
*{{Party stripe|Republican Party (US)}}[[Rhett Marques (politician)|Rhett Marques]] (Republican)
{{endplainlist}}
|-
==Alaska==
{|class="wikitable sortable"
!rowspan=2 class="unsortable" |Candidates<ref name=AL/>
|-
!{{ushr|AK|AL|X}}
|{{plainlist}}
*{{Party stripe|Republican Party (US)}}{{Party stripe|Independent}}T.&nbsp;P. Nykoriak (Republican)
*{{Party stripe|Democratic Party (US)}}{{Aye}} '''[[Mary Peltola]]''' (Democratic)
{{endplainlist}}
|}
==See also==
"""


def test_parse_house():
    out = nominees_2026.parse_house(HOUSE_WIKITEXT, 123)
    assert out[["state", "district", "party", "name"]].values.tolist() == [
        ["AL", 2, "D", "Shomari Figures"], ["AL", 2, "R", "Rhett Marques"],
        ["AK", 0, "R", "T. P. Nykoriak"], ["AK", 0, "D", "Mary Peltola"]]
    assert out["source"].tolist()[0] == "Wikipedia rev 123; cites https://sos.example/al"
    assert out["source"].tolist()[2] == "Wikipedia rev 123; cites https://sos.example/al"  # a reused named ref


def test_candidates_in_handles_an_item_on_the_plainlist_line():
    block = "| nowrap | {{Plainlist |*{{Party stripe|Republican Party (US)}}[[Mike Collins (politician)|Mike Collins]] " \
            "(Republican)\n*{{Party stripe|Democratic Party (United States)}}[[Jon Ossoff]] (Democratic)\n}}"
    assert nominees_2026.candidates_in(block) == [("Mike Collins", "Republican"), ("Jon Ossoff", "Democratic")]
    with pytest.raises(ValueError, match="Unparsed"):
        nominees_2026.candidates_in("*{{Party stripe|Green Party (US)}}Nobody")


def test_independents_fill_the_dem_slot_only_when_they_caucus_with_democrats(monkeypatch):
    # SYNTHETIC candidates for two real INDEPENDENTS_AS_DEM / INDEPENDENT_CANDIDATES keys
    cands = pd.DataFrame([
        {"year": 2012, "state": "ME", "class": 1, "party": "other", "name": "KING, ANGUS S., JR.", "votes": 3.0},
        {"year": 2012, "state": "ME", "class": 1, "party": "D", "name": "DILL, CYNTHIA ANN", "votes": 1.0},
        {"year": 2012, "state": "ME", "class": 1, "party": "R", "name": "SUMMERS, CHARLES", "votes": 2.0},
        {"year": 2022, "state": "UT", "class": 3, "party": "other", "name": "MCMULLIN, EVAN", "votes": 2.0},
        {"year": 2022, "state": "UT", "class": 3, "party": "R", "name": "LEE, MIKE", "votes": 3.0},
    ])
    monkeypatch.setattr(fec_finance, "INDEPENDENTS_AS_DEM", {
        k: v for k, v in fec_finance.INDEPENDENTS_AS_DEM.items() if k == ("senate", 2012, "ME", 1)})
    monkeypatch.setattr(fec_finance, "INDEPENDENT_CANDIDATES", {
        k: v for k, v in fec_finance.INDEPENDENT_CANDIDATES.items() if k == ("senate", 2022, "UT", 3)})
    slots = fec_finance.choose_slots(cands, ["year", "state", "class"], "senate").set_index(["state", "slot"])
    assert slots.loc[("ME", "dem"), ["name", "preset_status", "displaced"]].tolist() == [
        "KING, ANGUS S., JR.", "independent_as_dem", "DILL, CYNTHIA ANN"]
    assert ("ME", "ind") not in slots.index
    # McMullin is kept in the ind slot, and the dem slot stays empty (no D nominee)
    assert slots.loc[("UT", "dem"), "preset_status"] == "no_nominee"
    assert slots.loc[("UT", "ind"), ["name", "preset_status"]].tolist() == ["MCMULLIN, EVAN", "independent"]


def test_suspect_records_flag_empty_money_in_won_or_close_races(monkeypatch):
    key = ["year", "state", "district"]
    df = pd.DataFrame({"year": [2024, 2024, 2024, 2024], "state": ["CA", "OH", "TX", "WA"], "district": [31, 1, 2, 10]})
    for slot in ["dem", "gop"]:
        df[f"{slot}_name"], df[f"{slot}_candidate_id"] = "X", "H0"
        df[f"{slot}_match_status"] = "name_full"
        df[f"{slot}_receipts"] = 1e6
        df[f"{slot}_coverage_end_date"] = "2024-12-31"
    # CA: the D winner with $0 and no coverage; OH: an R losing by 9 with a record ending in 2023; TX: an R losing by
    # 40 with $0 (a blowout, not flagged); WA: an empty record the review found genuine
    df.loc[0, ["dem_receipts", "dem_coverage_end_date"]] = [0.0, None]
    df.loc[1, ["gop_receipts", "gop_coverage_end_date"]] = [40.0, "2023-09-30"]
    df.loc[2, ["gop_receipts", "gop_coverage_end_date"]] = [0.0, None]
    df.loc[3, ["gop_receipts", "gop_coverage_end_date"]] = [0.0, None]
    results = df[key].assign(winner_party="D", margin_2p_norm=[19.5, 9.2, 40.0, 5.0])
    monkeypatch.setattr(fec_finance, "REVIEWED_LOW_FUNDING", {("house", 2024, "WA", 10, "R"): "reviewed"})
    out = fec_finance.suspect_records(df, results, "house", key)
    assert list(zip(out["state"], out["party"])) == [("CA", "D"), ("OH", "R")]


def test_2026_louisiana_slots_go_to_the_top_fundraiser():
    # SYNTHETIC FEC records and slots: two LA races with several candidates per party, and a CA one
    fec = records(("H1", "DAY, JOHN DAVID", 222_000.0, "2026-10-14"), ("H2", "GONZALEZ, PRISCILLA", 0.0, None),
                  ("H3", "CABLE, CONRAD", 100.0, "2026-10-14"), ("H4", "GROMLICH, MATTHEW", 100.0, "2026-10-14"),
                  ("H5", "WIENER, SCOTT", 5_800_000.0, "2026-10-14"), ("H6", "CHAN, CONNIE", 2_400_000.0, "2026-10-14"))
    fec = fec.assign(year=2026, office="H", state=["LA", "LA", "LA", "LA", "CA", "CA"], district=[3, 3, 4, 4, 11, 11])
    slots = pd.DataFrame([
        {"year": 2026, "state": "LA", "district": 3, "slot": "dem", "preset_status": "nominee_unresolved",
         "options": ["John Day", "Priscilla Gonzalez", "Caleb Walker"], "name": None, "displaced": None},  # Walker: no record
        {"year": 2026, "state": "LA", "district": 4, "slot": "dem", "preset_status": "nominee_unresolved",
         "options": ["Conrad Cable", "Matt Gromlich"], "name": None, "displaced": None},  # tied
        {"year": 2026, "state": "CA", "district": 11, "slot": "dem", "preset_status": "nominee_unresolved",
         "options": ["Scott Wiener", "Connie Chan"], "name": None, "displaced": None},  # not a top-fundraiser state
    ])
    out = fec_finance.resolve_2026_by_receipts(slots, "house", fec).set_index(["state", "district"])
    assert out.loc[("LA", 3), "name"] == "John Day" and pd.isna(out.loc[("LA", 3), "preset_status"])
    assert out.loc[("LA", 3), "displaced"].startswith("2026 top fundraiser chosen among John Day $222,000")
    assert out.loc[("LA", 4), "preset_status"] == "nominee_unresolved"  # a tie stays unresolved
    assert out.loc[("CA", 11), "preset_status"] == "nominee_unresolved"
    assert fec_finance.resolve_2026_by_receipts(slots, "senate", fec).equals(slots)


def test_an_independent_fills_the_gop_slot_as_the_republican_proxy(monkeypatch):
    # SYNTHETIC candidates for the real INDEPENDENTS_AS_GOP key (CA-6 2026): an independent and a Democrat, no Republican
    cands = pd.DataFrame([
        {"year": 2026, "state": "CA", "district": 6, "party": "other", "name": "Kevin Kiley", "votes": np.nan},
        {"year": 2026, "state": "CA", "district": 6, "party": "D", "name": "Richard Pan", "votes": np.nan},
    ])
    monkeypatch.setattr(fec_finance, "INDEPENDENTS_AS_DEM", {})
    monkeypatch.setattr(fec_finance, "INDEPENDENT_CANDIDATES", {})
    slots = fec_finance.choose_slots(cands, ["year", "state", "district"], "house").set_index("slot")
    assert slots.loc["gop", ["name", "preset_status"]].tolist() == ["Kevin Kiley", "independent_as_gop"]
    assert slots.loc["dem", "name"] == "Richard Pan" and "ind" not in slots.index


def test_pre_election_amounts_replace_the_cycle_totals_and_say_so():
    # SYNTHETIC: two matched nominees, one with a pre-general record, one without
    m = pd.DataFrame({"year": [2022, 2022], "state": "AA", "district": [1, 2], "candidate_id": ["H1", "H2"],
                      **{f: [100.0, 50.0] for f in fec_finance.AMOUNTS},
                      "coverage_start_date": ["2021-01-01", "2021-01-01"],
                      "coverage_end_date": ["2022-12-31", "2022-12-31"]}).set_index(["year", "state", "district"])
    pre = pd.DataFrame({"year": [2022], "candidate_id": ["H1"], "report_type": ["12G"],
                        "coverage_end_date": pd.to_datetime(["2022-10-19"]),
                        **{f: [80.0] for f in fec_finance.AMOUNTS}}).set_index(["year", "candidate_id"])
    out = fec_finance.pre_election(m, pre)
    assert out["receipts"].tolist() == [80.0, 50.0] and out["cycle_receipts"].tolist() == [100.0, 50.0]
    assert out["fec_timing"].tolist() == ["pre_general_12G", "candidate_summary"]
    assert out["coverage_end_date"].iloc[0] == pd.Timestamp("2022-10-19")
