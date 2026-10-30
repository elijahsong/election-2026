import numpy as np
import pandas as pd
import pytest

import fec_finance
import nominees_2026


@pytest.mark.parametrize("nominee, fec, expected", [
    # suffixes, titles, initials, nicknames and middle names in use
    ("RALPH R. SCOTT, JR.", "SCOTT, RALPH R MR. JR.", "full"),
    ("OSSOFF, THOMAS JONATHAN (JON)", "OSSOFF, T. JONATHAN", "full"),
    ("Jon Ossoff", "OSSOFF, T. JONATHAN", "full"),
    ("BILL BASCRELL JR", "PASCRELL, WILLIAM J. HON. JR.", "fuzzy"),
    ("TOM GRAVES", "GRAVES, JOHN THOMAS MR. JR.", "full"),
    # compound and hyphenated surnames
    ("CHRIS VAN HOLLEN", "VAN HOLLEN, CHRISTOPHER J", "full"),
    ("Mario Díaz-Balart", "DIAZ-BALART, MARIO", "full"),
    # a conflicting middle initial, or an initial against a different first name, is surname-only
    ("Dan S. Sullivan", "SULLIVAN, DANIEL S", "full"),
    ("Dan S. Sullivan", "SULLIVAN, DAN J", "surname"),
    ("GUS M BILIRAKIS", "BILIRAKIS, MICHAEL", "surname"),
    ("SYLVIA R GARCIA", "GARCIA, ROEL", "surname"),
    # filing quirks
    ("NICK LALOTA", "NICK, LALOTA", "variant"),
    ("DOUG LA MALFA", "LAMALFA, DOUG", "full"),
    ("Beth Van Duyne", "VAN DUYNE, ELIZABETH ANN", "full"),
    ("STEPHANIE HERSETH SANDLIN", "HERSETH SANDLIN, STEPHANIE M", "full"),
    ("ASHLEY HINSON", "ARENHOLZ, ASHLEY HINSON", "variant"),
    # a misspelled surname, and MEDSL's mis-decoded UTF-8
    ("JEFF DENHEM", "DENHAM, JEFF", "fuzzy"),
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
        "dem_receipts": [99.0, 0.0], "gop_receipts": [9.0, 0.0],
        "dem_disbursements": [9.0, 0.0], "gop_disbursements": [99.0, 0.0],
        "dem_individual_contributions": [100.0, 0.0], "dem_individual_unitemized_contributions": [25.0, 0.0],
        "gop_individual_contributions": [50.0, 10.0], "gop_individual_unitemized_contributions": [10.0, 5.0],
    })
    out = fec_finance.add_features(df)
    assert list(out.columns) == list(df.columns) + fec_finance.FEATURES
    assert out["fundraising_advantage"].tolist() == pytest.approx([np.log(100) - np.log(10), 0.0])
    assert out["spending_advantage"].iloc[0] == pytest.approx(np.log(10) - np.log(100))
    assert out["small_dollar_share_dem"].iloc[0] == 0.25 and out["small_dollar_share_gop"].tolist() == [0.2, 0.5]
    assert out["small_dollar_share_diff"].iloc[0] == pytest.approx(0.05)
    # a zero denominator leaves the share, and so the difference, undefined
    assert np.isnan(out["small_dollar_share_dem"].iloc[1]) and np.isnan(out["small_dollar_share_diff"].iloc[1])
    assert out["small_dollar_share_flag"].isna().all()


def test_add_features_drops_and_flags_shares_outside_0_1():
    df = pd.DataFrame({"dem_receipts": [1.0], "gop_receipts": [1.0], "dem_disbursements": [1.0],
                       "gop_disbursements": [1.0], "dem_individual_contributions": [10.0],
                       "dem_individual_unitemized_contributions": [30.0], "gop_individual_contributions": [10.0],
                       "gop_individual_unitemized_contributions": [5.0]})
    out = fec_finance.add_features(df)
    assert np.isnan(out["small_dollar_share_dem"].iloc[0]) and out["small_dollar_share_gop"].iloc[0] == 0.5
    assert np.isnan(out["small_dollar_share_diff"].iloc[0])
    assert out["small_dollar_share_flag"].iloc[0] == "dem_share_outside_0_1"
    assert out["dem_individual_unitemized_contributions"].iloc[0] == 30.0  # the FEC amount is kept


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
