"""Race-level FEC candidate finance for House and Senate general elections, 2006-2026.

  data/interim/house_fec_raw.csv    one row per year, state, district: each major party's nominee and FEC totals
  data/interim/senate_fec_raw.csv   one row per year, state, class (special marks a special election)
  data/processed/house_fec.csv      the interim columns plus the five features in add_features
  data/processed/senate_fec.csv
  data/interim/fec_match_review.csv every party slot that isn't a clean name match, with the FEC records considered

FEC sources (data/raw/fec_candidate_summaries; README.md there):
  2008-2026  candidate_summary_YYYY.csv, FEC's candidate summary file (with a header, 50 columns)
  2006       webl06.txt, FEC's older "current campaigns for House and Senate" file (pipe-delimited, no header, 30
             columns, office from the first letter of the candidate ID). It has no itemized / unitemized split and no
             coverage start date, so those fields are blank for 2006.
  2008-2024  webl{yy}.txt is a fallback for nominees with no candidate summary record (e.g. Spartz, IN-5 2024), and a
             cross-check of receipts. {party}_fec_file records which file a slot's record came from.
  Each summary covers the two-year cycle through December 31 of the election year (2026: through mid-October), so
  for 2006-2024 it includes money raised and spent after the election.

Pre-election amounts (data/interim/fec_pre_general.csv, fec_pre_general.py): the matched nominee's totals are replaced
  by their amounts as of their pre-general report (12G, through about 20 days before the election, filed before it),
  or their October quarterly without one: House, cycle-to-date as filed (since the previous November); Senate, the
  last two years of its six-year cycle-to-date. coverage_start_date is blank for these. A Senate amount above 105%
  of the slot's two-year summary total (the six-year cycle wasn't cut to two years: no year-end report), or below 0
  (the year-end report wasn't on the same basis, e.g. Boxer, CA 2010), falls back to the summary. That was public before Election Day, and it is the same window
  as 2026's summary. {party}_fec_timing says which: pre_general_12G, pre_general_Q3, or candidate_summary (no
  pre-general report found; the record's own coverage, after the election for 2006-2024, so a look-ahead that the run
  reports). {party}_cycle_receipts keeps the summary's receipts for the cross-check against webl.

Nominees (one D and one R slot per race; read-only from the repo's sources):
  House 2006-2024   MEDSL via house_targets (read_source, label_contests, candidates): the November / first round
  Senate 2006-2024  CAGE via incumbency.senate_candidates, which also gives class and special
  2026              data/raw/nominees_2026/{house,senate}.csv, built by nominees_2026.py from pinned Wikipedia revisions
  A party's slot is its leading named general-election candidate by votes (jungle, top-two, RCV and party write-ins
  can give a party more than one; {party}_candidate_count says how many). In 2026 there are no votes yet: a party's only
  candidate fills the slot; with several, the one FEC flags as the incumbent does. Failing that, in
  TOP_FUNDRAISER_STATES_2026 (Louisiana) the party's top FEC fundraiser does (resolve_2026_by_receipts); elsewhere the
  slot is nominee_unresolved.
  INDEPENDENTS_AS_DEM puts the independents who caucus with Democrats (Sanders, King) in the dem slot, with
  dem_is_independent True. INDEPENDENTS_AS_GOP puts an independent in the gop slot as the Republican proxy (Kiley,
  CA-6 2026); only the model slot changes, and his ballot party stays independent. INDEPENDENT_CANDIDATES (Orman, McMullin, Osborn, Buchanan) stay independents: their FEC
  totals are in a third slot, ind_, and the dem slot is the race's D nominee (or no_nominee). Party labels are
  otherwise strict: an independent is never a D or R nominee.

Matching: no crosswalk from either source to FEC IDs exists, so each slot's nominee is matched by name (match_name)
  within the year, office and state, in this order of pools: (1) the same district (House) and party, (2) any party
  label (name_party_differs), (3) House only, another district of the state, full names only (name_other_district:
  FEC's district can be stale after redistricting). Candidate summary records are tried before webl-only ones.
  MATCH_OVERRIDES settles reviewed cases by hand. Nothing is matched on fundraising.

match_status, per slot:
  name_full               surname and first name (or initial / nickname) agree, with no conflicting middle initial
  name_variant            full after allowing for a filing quirk (match_name): reversed order, spacing, maiden name
  name_surname            surname only, unique in its pool
  name_fuzzy              first name and a misspelled (80%+ similar) surname, unique; same district and party only
  name_party_differs      a full (or variant) match, but FEC lists another party
  name_other_district     a full-name match in another district of the state
  duplicate_registration  several FEC records of one person; the only one with filings, or one of identical records
  override                MATCH_OVERRIDES
  fec_api                 API_CORRECTIONS: the cycle totals from the FEC API, for a nominee missing from both bulk files
                          or whose only record there is empty
  independent_as_dem      INDEPENDENTS_AS_DEM
  independent_as_gop      INDEPENDENTS_AS_GOP
  independent             INDEPENDENT_CANDIDATES (the ind slot)
  no_fec_record           the nominee is known but has no FEC record
  ambiguous               several different FEC records match equally well; none is used
  no_nominee              the party had no general-election candidate
  nominee_unresolved      several candidates of the party and no way to pick one: 2026 without a single incumbent
                          (no votes yet) outside TOP_FUNDRAISER_STATES_2026, or tied leaders

Missing data: amounts are 0 when there is no nominee or no usable FEC record (no_nominee, no_fec_record, ambiguous),
  with {party}_fec_missing True. They are blank for nominee_unresolved, whose nominee isn't known, and every ind_
  column is blank in races without an INDEPENDENT_CANDIDATES entry. Names are kept only
  where the source names a nominee; IDs and dates are never filled in. A blank in a matched FEC record stays blank.
"""

import argparse
import difflib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "house"))
import acs
import house_targets
import incumbency

ROOT = Path(__file__).resolve().parents[1]
FEC_DIR = ROOT / "data/raw/fec_candidate_summaries"
NOMINEES_2026_DIR = ROOT / "data/raw/nominees_2026"
TARGETS_RAW_PATH = ROOT / "data/processed/house/targets_raw.csv"
SENATE_TARGETS_RAW_PATH = ROOT / "data/processed/senate/targets_raw.csv"
SENATE_INCUMBENCY_PATH = ROOT / "data/interim/senate_incumbency.csv"
DISTRICT_COUNTS_PATH = ROOT / "data/reference/district_counts.json"
PRE_GENERAL_PATH = ROOT / "data/interim/fec_pre_general.csv"  # fec_pre_general.py
SENATE_PRE_GENERAL_TOLERANCE = 1.05  # a Senate pre-general amount above 105% of its two-year summary falls back
INTERIM_DIR = ROOT / "data/interim"
PROCESSED_DIR = ROOT / "data/processed"

YEARS = list(range(2006, 2027, 2))
WEBL_ONLY_YEAR = 2006  # the only year without a candidate summary file
PARTIES = {"dem": "D", "gop": "R"}
SLOT_PARTY = {**PARTIES, "ind": "I"}  # the party code of every slot, the ind slot included

# canonical name -> candidate summary column; the same nine fields are kept for each party
FINANCE = {
    "receipts": "Total_Receipt",
    "disbursements": "Total_Disbursement",
    "individual_contributions": "Individual_Contribution",
    "individual_unitemized_contributions": "Individual_Unitemized_Contribution",
    "individual_itemized_contributions": "Individual_Itemized_Contribution",
    "cash_on_hand": "Cash_On_Hand_COP",
    "debt": "Debt_Owed_By_Committee",
    "coverage_start_date": "Coverage_Start_Date",
    "coverage_end_date": "Coverage_End_Date",
}
AMOUNTS = [f for f in FINANCE if not f.endswith("_date")]
DATES = ["coverage_start_date", "coverage_end_date"]
SUMMARY_ID_COLUMNS = ["Cand_Id", "Cand_Name", "Cand_Office", "Cand_Office_St", "Cand_Office_Dist",
                      "Cand_Party_Affiliation", "Cand_Incumbent_Challenger_Open_Seat"]
# webl{yy}.txt columns, in order (FEC "current campaigns for House and Senate" file description)
WEBL_COLUMNS = ["CAND_ID", "CAND_NAME", "CAND_ICI", "PTY_CD", "CAND_PTY_AFFILIATION", "TTL_RECEIPTS",
                "TRANS_FROM_AUTH", "TTL_DISB", "TRANS_TO_AUTH", "COH_BOP", "COH_COP", "CAND_CONTRIB", "CAND_LOANS",
                "OTHER_LOANS", "CAND_LOAN_REPAY", "OTHER_LOAN_REPAY", "DEBTS_OWED_BY", "TTL_INDIV_CONTRIB",
                "CAND_OFFICE_ST", "CAND_OFFICE_DISTRICT", "SPEC_ELECTION", "PRIM_ELECTION", "RUN_ELECTION",
                "GEN_ELECTION", "GEN_ELECTION_PRECENT", "OTHER_POL_CMTE_CONTRIB", "POL_PTY_CONTRIB", "CVG_END_DT",
                "INDIV_REFUNDS", "CMTE_REFUNDS"]
# canonical name -> webl column; the itemized / unitemized split and the coverage start date aren't in the file
WEBL_FINANCE = {"receipts": "TTL_RECEIPTS", "disbursements": "TTL_DISB", "individual_contributions": "TTL_INDIV_CONTRIB",
                "cash_on_hand": "COH_COP", "debt": "DEBTS_OWED_BY", "coverage_end_date": "CVG_END_DT"}
WEBL_ICI = {"I": "INCUMBENT", "C": "CHALLENGER", "O": "OPEN"}
FEC_PARTY = {"DEM": "D", "DFL": "D", "REP": "R"}

# Independents who caucus with Democrats, counted as the Democratic side: each replaces the dem slot (status
# independent_as_dem, dem_is_independent True), and any D nominee it displaces is reported.
# key: (chamber, year, state, district or class) -> the name as the source spells it
INDEPENDENTS_AS_DEM = {
    # Sanders caucuses with Democrats and won the D primary each time, declining the nomination; no D on the ballot
    **{("senate", year, "VT", 1): "SANDERS, BERNARD (BERNIE)" for year in (2006, 2012, 2018, 2024)},
    # King caucuses with Democrats; the D nominee was a distant third each time
    **{("senate", year, "ME", 1): "KING, ANGUS S., JR." for year in (2012, 2018, 2024)},
}
# 2026 House races where a party with several candidates and no single incumbent is represented by its top FEC
# fundraiser (resolve_2026_by_receipts). Louisiana's November ballot lists every party's candidates in one round, and
# for 2006-2024 the slot is the party's leading vote-getter; before any votes, the top fundraiser is the stand-in
# known in advance (decision 2026-11-02; LA-3 to LA-6). Not applied to California's same-party top-two races
# (CA-11, CA-14, CA-40), which have no opposite-party candidate and are left unresolved for now.
TOP_FUNDRAISER_STATES_2026 = {"LA"}
# Independents who fill the gop slot as the Republican proxy for the regression and fundraising features (status
# independent_as_gop). Only the model slot: their ballot party stays independent, and Stage 2 counts their win as
# OTHER (src/house/simulation/races.py). key as above
INDEPENDENTS_AS_GOP = {
    # Kiley, the incumbent (inc_signed -1, house_incumbency), runs as an independent against Richard Pan (D), with no
    # Republican on the ballot (decision 2026-11-02)
    ("house", 2026, "CA", 6): "Kevin Kiley",
}
INDEPENDENT_STATUSES = ("independent_as_dem", "independent_as_gop", "independent")
# Other leading independents, who didn't commit to caucusing with Democrats: never the D nominee, so the dem slot
# keeps the race's Democrat (or no_nominee). Their FEC totals are kept in the ind slot (status independent).
# key as above
INDEPENDENT_CANDIDATES = {
    ("senate", 2014, "KS", 2): "ORMAN, GREGORY JOHN (GREG)",  # the D nominee (Taylor) withdrew to clear the field
    ("senate", 2022, "UT", 3): "MCMULLIN, EVAN",  # Utah Democrats endorsed him and ran no nominee
    ("senate", 2024, "NE", 1): "OSBORN, DAN",  # no D nominee
    ("senate", 2026, "NE", 2): "Dan Osborn",  # no D nominee
    # Rosendale (R) 56.6, Buchanan (I) 21.9, Ronning (D) 20.2: the independent was the main opposition
    ("house", 2022, "MT", 2): "GARY BUCHANAN",
}
# CAGE's Maine class I races of 2012 and 2018 are swapped (its 2012 rows are Brakey and Ringelstein, the 2018 race; its
# 2018 rows Summers and Dill, the 2012 race; King is in both). Each pair of (year, state, class) races trades years
CAGE_SWAPPED_RACES = [((2012, "ME", 1), (2018, "ME", 1))]
# Hand-settled matches after reviewing fec_match_review.csv (each was "ambiguous"): (chamber, year, state, district or
# class, party) -> FEC candidate ID, or None when the nominee is confirmed to have no FEC record. Each must be used.
MATCH_OVERRIDES = {
    # two different people of one name
    ("house", 2006, "FL", 9, "R"): "H6FL09070",  # Gus Bilirakis; H2FL09012 is his retiring father Michael's committee
    ("house", 2018, "KS", 4, "R"): "H8KS04112",  # incumbent Ron Estes; H8KS04187 is Ron M. Estes, his primary opponent
    # one person's linked registrations whose totals differ slightly; the record covering the cycle is used
    ("house", 2024, "FL", 16, "D"): "H4FL16161",  # same receipts; the 2024 registration (H2FL13139 was for FL-13)
    ("house", 2024, "OR", 3, "R"): "H0OR03174",  # same receipts; covers 2023-2024 (H4OR03218 starts in 2024)
    ("house", 2024, "RI", 1, "R"): "H4RI01141",  # identical but for $0.99 of debt; the 2024 registration
    ("house", 2026, "NV", 3, "R"): "H6NV03204",  # same individual totals, receipts 0.6% apart; the 2026 registration
    ("house", 2026, "VA", 11, "R"): "H6VA11249",  # the only one through 10/14/2026; two others stop on 8/27/2026
    # FEC's name has no comma and is in LAST FIRST order ("GREEN RONALD NEAL")
    ("house", 2012, "OR", 3, "R"): "H2OR03105",
    # MEDSL's "ELIZABETH H ETSY" (transposed letters, below the fuzzy threshold) is incumbent Elizabeth Esty
    ("house", 2016, "CT", 5, "D"): "H2CT05131",
    # FEC files John "JD" Den Dulk as "DULK, JOHN DEN"
    ("house", 2006, "CA", 9, "R"): "H6CA09080",
    # MEDSL's "JOHN ABNEY" is incumbent John Abney Culberson, the surname dropped
    ("house", 2008, "TX", 7, "R"): "H0TX07055",
    # found by suspect_records (fec_api_totals.py --review): the name match picked another, empty record
    # incumbent Jason Smith; his record's party is UNK, so the surname match took Phillip S. Smith (H6MO08134, $0)
    ("house", 2016, "MO", 8, "R"): "H4MO08162",
    # incumbent Mike Kelly, filed as GEORGE J KELLY JR; "KELLY, MIKE" (H4PA03117) is an empty registration
    ("house", 2016, "PA", 3, "R"): "H0PA03271",
    # MEDSL's "GABBY SAUCEDO" is Gabriela Saucedo Mercer, filed as "MERCER, GABRIELA SAUCEDO MRS"
    ("house", 2014, "AZ", 3, "R"): "H2AZ07146",
    # Ben Clark, filed as "CLARK, ROBERT BENJAMIN III" for NC-4 before the 2022 redraw; H2NC09225 is an empty one
    ("house", 2022, "NC", 9, "D"): "H2NC04150",
}

# Nominees missing from both bulk files for their cycle, or matched there only to an empty record (every amount 0, no
# coverage dates), filled from the FEC API's candidate totals for that cycle
# (/v1/candidate/{id}/totals/?cycle=YEAR&election_full=false, the same two-year definitions as the bulk files).
# fec_api_totals.py fetches them into API_TOTALS_PATH; a correction applies only to a slot left no_fec_record or matched
# to an empty record, and one whose fetch failed leaves the slot as it was (reported). key as in MATCH_OVERRIDES -> FEC
# candidate ID
API_CORRECTIONS = {
    ("house", 2018, "MD", 6, "D"): "H6MD08549",  # David Trone
    ("house", 2020, "CA", 50, "R"): "H0CA48024",  # Darrell Issa
    ("house", 2020, "CA", 53, "D"): "H8CA49074",  # Sara Jacobs
    ("house", 2020, "FL", 19, "R"): "H2FL14186",  # Byron Donalds
    ("house", 2022, "MD", 4, "D"): "H2MD04232",  # Glenn Ivey
    ("house", 2022, "TX", 3, "R"): "H2TX00064",  # Keith Self
    ("house", 2024, "IN", 3, "R"): "H0IN03198",  # Marlin Stutzman
    ("house", 2024, "MD", 4, "D"): "H2MD04232",  # Glenn Ivey
    ("house", 2024, "NY", 4, "D"): "H2NY04244",  # Laura Gillen
    ("house", 2024, "TX", 3, "R"): "H2TX00064",  # Keith Self
    ("house", 2024, "VA", 5, "R"): "H0VA07133",  # John McGuire
    # Lucas Kunce: candidate_summary_2024 has his ID with every amount 0 and no coverage dates; not in webl24
    ("senate", 2024, "MO", 1, "D"): "S2MO00528",
    # found by suspect_records (fec_api_totals.py --review)
    ("house", 2018, "NV", 3, "R"): "H2NV04045",  # Danny Tarkanian: in neither 2018 bulk file
    ("house", 2020, "AZ", 2, "R"): "H8AZ02201",  # Brandon Martin: in neither 2020 bulk file
    ("house", 2024, "CA", 31, "D"): "H8CA39174",  # Gil Cisneros: candidate_summary_2024 has his ID with every amount 0
    ("house", 2008, "NM", 2, "R"): "H2NM02134",  # Ed Tinsley: candidate_summary_2008 has his ID with every amount 0
    ("house", 2024, "OH", 1, "R"): "H4OH01109",  # Orlando Sonza: candidate_summary_2024 has his ID with every amount 0
}
# Corrections whose FEC candidate totals are empty or leave out the campaign, though its principal committee filed for
# the cycle: fec_api_totals.py takes that committee's two-year totals instead (/v1/committee/{id}/totals/?cycle=YEAR).
# Checked 2026-11-02 against each candidate's committee list; the receipts are the committee's for the cycle.
# candidate ID -> committee ID
API_COMMITTEE_TOTALS = {
    # Kunce: Heartland Patriots, his principal committee for 2022-2026, is designated U (unauthorized) for 2024, so
    # his 2024 candidate totals are 0 ($23.2M receipts, 2023-01-01 to 2024-12-31)
    "S2MO00528": "C00771972",
    # Gillen: her 2024 candidate totals ($4.89, January 2023) are her 2022 committee's wind-down; the 2024 campaign
    # is Gillen for NY ($6.83M, 2023-04-01 to 2024-12-31)
    "H2NY04244": "C00840165",
    # no candidate totals at all; each campaign's principal committee (P) for the cycle:
    "H2MD04232": "C00792283",  # Glenn F. Ivey for Congress: $1.90M 2022, $0.94M 2024
    "H2TX00064": "C00791293",  # Self for Congress: $0.59M 2022, $0.74M 2024
    "H0IN03198": "C00838110",  # Stutzman for Congress: $1.39M 2024
    "H0VA07133": "C00856831",  # McGuire for Virginia: $1.84M 2024 (its $7,000 recount committee is left out)
    "H8CA39174": "C00850420",  # Cisneros for Congress: $6.48M 2024
    # Tinsley for Congress: a principal House campaign committee in Capitan, NM, first filed 2007-11-21, that FEC links
    # to no candidate ID; $2.40M 2008 (his ID's only 2008 committee is a joint fundraising one)
    "H2NM02134": "C00441394",
    # Sonza: FEC lists no campaign committee for him, only Southwest Ohio Leadership PAC (a leadership PAC, linked to
    # his ID, first filed 2023-07-10; $1.17M 2024). Counted as his campaign's money by decision (2026-11-02), though
    # leadership PAC receipts aren't campaign receipts
    "H4OH01109": "C00845065",
}
API_TOTALS_PATH = FEC_DIR / "api/api_totals.csv"

# Report-only check (suspect_records): a slot whose FEC money looks empty (receipts under SUSPECT_RECEIPTS, with no
# coverage dates or coverage ending before July of the election year) though its nominee won or finished within
# SUSPECT_MARGIN points (2006-2024, from targets_raw.csv). Kunce and Gillen looked like this: a campaign committee
# FEC's candidate totals leave out, fixed with API_CORRECTIONS / API_COMMITTEE_TOTALS. REVIEWED_LOW_FUNDING lists
# slots checked against the FEC API's committee list and found genuinely low-funded, so they're no longer reported.
SUSPECT_RECEIPTS = 50_000
SUSPECT_MARGIN = 20
# key as in MATCH_OVERRIDES -> what the review (2026-11-02) found
REVIEWED_LOW_FUNDING: dict[tuple, str] = {
    **dict.fromkeys([
        ("house", 2006, "OH", 16, "D"),  # Thomas Shaw (H6OH16128)
        ("house", 2012, "OH", 2, "D"),  # William R. Smith
        ("house", 2012, "OH", 14, "D"),  # Dale Virgil Blanchard
        ("house", 2016, "NC", 5, "D"),  # Josh Brannon (Gregory Brannon, NC-2 R, is another person)
        ("house", 2016, "NJ", 11, "D"),  # Joseph M. Wenzel, under either of his IDs
        ("house", 2016, "WA", 10, "R"),  # Jim Postma
        ("house", 2020, "IN", 1, "R"),  # Mark Leyva
        ("house", 2022, "CA", 31, "R"),  # Daniel Bocic Martinez
        ("house", 2024, "CA", 31, "R"),  # Daniel Bocic Martinez, under either of his IDs
        ("house", 2024, "MA", 9, "R"),  # Dan Sullivan
        ("house", 2024, "NC", 7, "D"),  # Marlando Pridgen, under either of his IDs
        ("house", 2024, "NC", 8, "D"),  # Justin Dues
        ("senate", 2012, "MS", 1, "D"),  # Albert N. Gore Jr.
    ], "no committee with totals for the cycle"),
    ("house", 2024, "WA", 10, "R"): "Don Hewett: his one committee took in $40 and stopped filing after 2023-09-30",
    ("senate", 2008, "SC", 2, "D"): "Bob Conley: his one committee took in $17,105 and filed through 2008-06-30",
}

# Name parsing: words dropped as titles or suffixes, and first-name nicknames that aren't a prefix of the formal name
# (prefixes like CHRIS / CHRISTOPHER are matched by rule)
NAME_NOISE = {"JR", "SR", "II", "III", "IV", "V", "MR", "MRS", "MS", "MISS", "DR", "HON", "REV", "PHD", "MD", "ESQ",
              "SEN", "REP"}
# surname particles: in "First ... Last" names they belong to the surname (CHRIS VAN HOLLEN, DOUG LA MALFA)
SURNAME_PARTICLES = {"VAN", "VON", "DE", "DEL", "DELA", "DA", "DI", "DU", "DES", "LA", "LE", "ST"}
NICKNAMES = {
    "BILL": "WILLIAM", "BILLY": "WILLIAM", "WILL": "WILLIAM", "BOB": "ROBERT", "BOBBY": "ROBERT", "ROB": "ROBERT",
    "JIM": "JAMES", "JIMMY": "JAMES", "MIKE": "MICHAEL", "TOM": "THOMAS", "TOMMY": "THOMAS", "DICK": "RICHARD",
    "RICK": "RICHARD", "RICKY": "RICHARD", "CHUCK": "CHARLES", "JACK": "JOHN", "JOHNNY": "JOHN", "TED": "EDWARD",
    "NED": "EDWARD", "PEGGY": "MARGARET", "LIZ": "ELIZABETH", "BETH": "ELIZABETH", "BETSY": "ELIZABETH",
    "HANK": "HENRY", "JERRY": "GERALD", "LARRY": "LAWRENCE", "TONY": "ANTHONY", "PATTY": "PATRICIA",
    "SANDY": "SANDRA", "KATIE": "KATHERINE", "KATHY": "KATHERINE", "SUE": "SUSAN", "JOE": "JOSEPH", "DAVE": "DAVID",
    "NATE": "NATHAN", "STEVE": "STEPHEN", "NICK": "NICHOLAS",
}


# ---------------------------------------------------------------- FEC files

def amounts(values: pd.Series) -> pd.Series:
    """Numbers from text: blank is NaN (unknown, never 0); anything else that doesn't parse raises."""
    text = values.str.strip()
    return pd.to_numeric(text.where(text != ""), errors="raise")


def dates(values: pd.Series) -> pd.Series:
    text = values.str.strip()
    return pd.to_datetime(text.where(text != ""), format="%m/%d/%Y", errors="raise")


def house_district(values: pd.Series, office: pd.Series) -> pd.Series:
    return pd.to_numeric(values.str.strip().where(office == "H"), errors="raise").astype("Int64")


def read_candidate_summary(path: Path, year: int) -> pd.DataFrame:
    """A candidate summary file's House and Senate records, in the canonical columns."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(SUMMARY_ID_COLUMNS + list(FINANCE.values())) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name} lacks columns {sorted(missing)}")
    raw = raw[raw["Cand_Office"].isin(["H", "S"])]
    out = pd.DataFrame({
        "year": year, "candidate_id": raw["Cand_Id"], "fec_name": raw["Cand_Name"], "office": raw["Cand_Office"],
        "state": raw["Cand_Office_St"], "district": house_district(raw["Cand_Office_Dist"], raw["Cand_Office"]),
        "fec_party": raw["Cand_Party_Affiliation"], "fec_ici": raw["Cand_Incumbent_Challenger_Open_Seat"],
        "fec_file": path.name})
    for field, column in FINANCE.items():
        out[field] = dates(raw[column]) if field in DATES else amounts(raw[column])
    return out.reset_index(drop=True)


def read_webl(path: Path, year: int) -> pd.DataFrame:
    """A webl file's House and Senate records, in the canonical columns, plus FEC's own general-election result (W/L,
    filled through 2012) as gen_election. Raises unless every line has the 30 fields."""
    widths = {line.count("|") + 1 for line in path.read_text().splitlines()}
    if widths != {len(WEBL_COLUMNS)}:
        raise ValueError(f"{path.name}: lines with {sorted(widths)} fields, expected {len(WEBL_COLUMNS)}")
    raw = pd.read_csv(path, sep="|", header=None, names=WEBL_COLUMNS, dtype=str, keep_default_na=False,
                      quoting=3)  # csv.QUOTE_NONE: names can contain quotes
    raw = raw.assign(office=raw["CAND_ID"].str[0])
    raw = raw[raw["office"].isin(["H", "S"])]
    out = pd.DataFrame({
        "year": year, "candidate_id": raw["CAND_ID"], "fec_name": raw["CAND_NAME"], "office": raw["office"],
        "state": raw["CAND_OFFICE_ST"], "district": house_district(raw["CAND_OFFICE_DISTRICT"], raw["office"]),
        "fec_party": raw["CAND_PTY_AFFILIATION"], "fec_ici": raw["CAND_ICI"].map(WEBL_ICI).fillna(""),
        "fec_file": path.name})
    for field in FINANCE:
        column = WEBL_FINANCE.get(field)
        if column is None:
            out[field] = pd.NaT if field in DATES else np.nan
        else:
            out[field] = dates(raw[column]) if field in DATES else amounts(raw[column])
    out["gen_election"] = raw["GEN_ELECTION"].to_numpy()
    return out.reset_index(drop=True)


def read_fec(directory: Path = FEC_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(primary, webl): each year's primary records (candidate summary, or webl06 for 2006), and every webl record."""
    primary, webl = [], []
    for year in YEARS:
        summary = directory / f"candidate_summary_{year}.csv"
        webl_path = directory / f"webl{year % 100:02d}.txt"
        if year != 2026:
            webl.append(read_webl(webl_path, year))
        primary.append(webl[-1].drop(columns="gen_election") if year == WEBL_ONLY_YEAR
                       else read_candidate_summary(summary, year))
    primary, webl = pd.concat(primary, ignore_index=True), pd.concat(webl, ignore_index=True)
    for name, df in [("primary", primary), ("webl", webl)]:
        if df.duplicated(["year", "candidate_id"]).any():
            raise ValueError(f"{name} FEC records repeat a candidate ID within a year")
    return primary, webl


def schema_report(directory: Path = FEC_DIR) -> str:
    summary_columns = {y: list(pd.read_csv(directory / f"candidate_summary_{y}.csv", nrows=0).columns)
                       for y in YEARS if y != WEBL_ONLY_YEAR}
    differing = [y for y, cols in summary_columns.items() if cols != summary_columns[2026]]
    return "\n".join([
        f"candidate_summary 2008-2026: {len(summary_columns[2026])} columns, header row; years whose columns differ "
        f"from 2026: {differing or 'none'}",
        f"webl 2006-2024: no header, {len(WEBL_COLUMNS)} pipe-delimited fields in every line, office from the ID; "
        "no itemized / unitemized split and no coverage start date (2006 leaves them blank)",
    ])


# ---------------------------------------------------------------- nominees

def house_candidates_historical() -> pd.DataFrame:
    """Every named House general-election candidate 2006-2024 (MEDSL, November / first round): party D, R (from
    house_targets' party lines) or other."""
    df = house_targets.label_contests(house_targets.read_source())
    cands = house_targets.candidates(df[(df["year"] >= YEARS[0]) & (df["state"] != "DC")])
    cands = cands[(cands["contest"] == "first_round") & cands["is_candidate"]]
    party = cands["bucket"].map({"dem": "D", "gop": "R"}).fillna("other")
    return pd.DataFrame({"year": cands["year"], "state": cands["state"], "district": cands["district"],
                         "party": party, "name": cands["candidate"], "votes": cands["votes"]})


def senate_candidates_historical(cage: pd.DataFrame) -> pd.DataFrame:
    """Every CAGE Senate general-election candidate 2006-2024 (runoff finalists where CAGE has the runoff)."""
    s = incumbency.senate_candidates(cage)
    s = s[s["year"] < 2026].copy()
    for a, b in CAGE_SWAPPED_RACES:
        in_a, in_b = [(s["year"] == y) & (s["state"] == st) & (s["class"] == c) for y, st, c in (a, b)]
        if not in_a.any() or not in_b.any():
            raise ValueError(f"CAGE_SWAPPED_RACES {a}, {b}: a race has no rows")
        s.loc[in_a, "year"], s.loc[in_b, "year"] = b[0], a[0]
    return pd.DataFrame({"year": s["year"], "state": s["state"], "class": s["class"].astype(int),
                         "special": s["special"], "party": s["party"].where(s["party"].isin(["D", "R"]), "other"),
                         "name": s["name_snyder"], "votes": s["candidatevotes"]})


def candidates_2026(chamber: str) -> pd.DataFrame:
    df = pd.read_csv(NOMINEES_2026_DIR / f"{chamber}.csv")
    return df.drop(columns=["party_label", "source"]).assign(votes=np.nan)


# ---------------------------------------------------------------- names

def name_parts(name: str) -> tuple[list[str], list[str]]:
    """(surname words, given words) of "LAST, FIRST MIDDLE (NICK), JR." or "First Middle Last Jr." names, in upper
    case without accents or punctuation. A nickname in parentheses or quotes is added to the given words."""
    try:  # MEDSL has some UTF-8 names decoded as Latin-1 ("JESÃ\x9aS G ... GARCÃ\x8dA"); undo that where it applies
        name = name.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().upper()
    text = re.sub(r"[\u2018\u2019]", "'", text)
    nicknames = " ".join(a or b for a, b in re.findall(r"\(([^)]*)\)|\"([^\"]*)\"", text))
    text = re.sub(r"\([^)]*\)|\"[^\"]*\"", " ", text)

    def words(s: str) -> list[str]:
        s = re.sub(r"['.]", "", s)
        return [w for w in re.sub(r"[^A-Z ]", " ", s).split() if w not in NAME_NOISE]

    if "," in text and words(text.split(",", 1)[1]):
        last, given = text.split(",", 1)
        surname, given = words(last), words(given)
    else:
        text = text.split(",", 1)[0] if "," in text else text  # "JOHN SMITH, JR."
        parts = [w for w in re.sub(r"['.]", "", text).split() if re.sub(r"[^A-Z]", "", w) not in NAME_NOISE]
        start = len(parts) - 1  # a hyphenated last word is one compound surname; particles before it join it
        while start > 1 and parts[start - 1] in SURNAME_PARTICLES:
            start -= 1
        surname = words(" ".join(parts[start:])) if parts else []
        given = words(" ".join(parts[:start]))
    return surname, given + words(nicknames)


def same_first_name(a: str, b: str) -> bool:
    """Two given words name the same person: equal, an initial of the other, a 3+ letter prefix, or a NICKNAMES pair."""
    if a == b:
        return True
    if len(a) == 1 or len(b) == 1:
        return a[0] == b[0]
    short, long = sorted([a, b], key=len)
    return (len(short) >= 3 and long.startswith(short)) or NICKNAMES.get(a) == b or NICKNAMES.get(b) == a


def match_name(nominee: str, fec: str) -> str | None:
    """'full', 'variant', 'surname', 'fuzzy' or None: how well an FEC record's name matches a nominee's (see
    match_parts).

    'variant' is a full match only after allowing for one of two filing quirks: the FEC name in FIRST, LAST order
    (NICK, LALOTA), or the nominee's surname filed as a middle name (ASHLEY HINSON / ARENHOLZ, ASHLEY HINSON), where
    the first given words must still agree.

    'fuzzy' is a misspelled surname (MEDSL's JEFF DENHEM, JANICE D SCAKOWSKY): the first given words agree and the
    surnames are at least 80% similar (difflib ratio). pick uses it only in the same district and party."""
    quality = match_parts(name_parts(nominee), name_parts(fec))
    if quality == "full":
        return quality
    (n_last, n_given), (f_last, f_given) = name_parts(nominee), name_parts(fec)
    if n_given and f_given and same_first_name(n_given[0], f_given[0]) and \
            (n_last[-1] in f_given[1:] or f_last[-1] in n_given[1:]):
        return "variant"
    if "," in fec:
        last, given = fec.split(",", 1)
        if match_parts(name_parts(nominee), name_parts(f"{given}, {last}")) == "full":
            return "variant"
    if quality is None and n_last and f_last and n_given and f_given and given_names_agree(n_given, f_given) \
            and difflib.SequenceMatcher(None, "".join(n_last), "".join(f_last)).ratio() >= 0.8:
        return "fuzzy"
    return quality


def given_names_agree(a: list[str], b: list[str]) -> bool:
    """The first given words name the same person, or any two spelled-out given words do (a middle name in use)."""
    return same_first_name(a[0], b[0]) or any(same_first_name(x, y) for x in a for y in b if len(x) > 1 and len(y) > 1)


def match_parts(nominee: tuple[list[str], list[str]], fec: tuple[list[str], list[str]]) -> str | None:
    """'full', 'surname' or None for two parsed names (name_parts).

    The surnames agree when either's last surname word is one of the other's (so DIAZ-BALART and SEKULA GIBBS match
    their shorter forms), or when they're equal without spaces (LA MALFA / LAMALFA). The match is full when the first given words name the same person, or
    any two spelled-out given words do (a middle name or nickname in use: T. JONATHAN / JON OSSOFF), and the two
    don't give conflicting middle initials (DAN S. vs DAN J. SULLIVAN). An initial counts only against the other first
    given word, so GUS M. BILIRAKIS doesn't match MICHAEL BILIRAKIS."""
    (n_last, n_given), (f_last, f_given) = nominee, fec
    if not n_last or not f_last or (n_last[-1] not in f_last and f_last[-1] not in n_last
                                     and "".join(n_last) != "".join(f_last)):
        return None
    if not n_given or not f_given:
        return "surname"
    agree = given_names_agree(n_given, f_given)
    # middle names, leaving out a word that is part of the other's surname (STEPHANIE HERSETH SANDLIN)
    n_mid, f_mid = [w for w in n_given if w not in f_last], [w for w in f_given if w not in n_last]
    middle_conflict = len(n_mid) > 1 and len(f_mid) > 1 and same_first_name(n_mid[0], f_mid[0]) \
        and n_mid[1][0] != f_mid[1][0] and not any(same_first_name(n_mid[1], b) for b in f_mid) \
        and not any(same_first_name(a, f_mid[1]) for a in n_mid)
    return "full" if agree and not middle_conflict else "surname"


def pick(nominee: str, pool: pd.DataFrame, full_only: bool = False) -> tuple[str, pd.DataFrame]:
    """(outcome, records) for one pool of FEC records: outcome is 'full', 'variant', 'surname', 'fuzzy' (match_name),
    'duplicate', 'ambiguous' or 'none', and records the matching rows (one row unless ambiguous).

    Full and variant matches win over surname-only ones, which win over fuzzy ones (never used when full_only). A
    match must be unique, except for one person's duplicate registrations (see one_person)."""
    if pool.empty:
        return "none", pool
    quality = pool["fec_name"].map(lambda f: match_name(nominee, f))
    full, surname, fuzzy = (pool[quality.isin(["full", "variant"])], pool[quality == "surname"],
                            pool[quality == "fuzzy"])
    if len(full) == 1:
        return quality[full.index].item(), full
    if len(full) > 1:
        return one_person(full)
    if full_only:
        return "none", pool.iloc[:0]
    if len(surname):
        return ("surname", surname) if len(surname) == 1 else one_person(surname)
    if len(fuzzy):
        return ("fuzzy", fuzzy) if len(fuzzy) == 1 else one_person(fuzzy)
    return "none", pool.iloc[:0]


def one_person(records: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """('duplicate', one record) when several matching records are one person's registrations (every two of their
    names fully match), and either only one has filings (a coverage end date) or they all carry identical totals
    (FEC often links one committee to each ID; the first by ID is used). Else ('ambiguous', records)."""
    same_person = all(match_name(a, b) in ("full", "variant")
                      for a, b in itertools.combinations(records["fec_name"], 2))
    filed = records[records["coverage_end_date"].notna()]
    if same_person and len(filed) == 1:
        return "duplicate", filed
    considered = filed if len(filed) else records
    if same_person and len(considered[list(FINANCE)].astype(str).drop_duplicates()) == 1:
        return "duplicate", considered.sort_values("candidate_id").iloc[:1]
    return "ambiguous", records


def match_slot(nominee: str, party: str, race: dict, chamber: str, fec: pd.DataFrame, webl_only: pd.DataFrame,
               at_large: bool, taken: set | None = None) -> tuple[str, pd.DataFrame]:
    """(match_status, records) for one nominee, trying each pool (see the module docstring) in candidate summary
    records, then webl-only ones. records is one row for a match, the candidates considered when ambiguous.

    Pools (1) and (2) are used unless `taken` is given; then only pool (3), another district of the state, among the
    records whose (year, candidate_id) isn't in `taken` (already some race's match)."""
    office = "H" if chamber == "house" else "S"
    for records in (fec, webl_only):
        here = records[(records["year"] == race["year"]) & (records["office"] == office)
                       & (records["state"] == race["state"])]
        if office == "H":
            districts = {0, 1} if at_large else {race["district"]}
            in_district = here["district"].isin(districts).fillna(False)
            others, here = here[~in_district], here[in_district]
        same_party = here["fec_party"].str.upper().map(FEC_PARTY) == party
        if taken is None:
            # a different party label needs the full name: surname-only, it pairs opponents (MIKE vs JOE ROSS)
            pools = [("name_full", here[same_party], False), ("name_party_differs", here[~same_party], True)]
        else:
            free = ~pd.Series(list(zip(others["year"], others["candidate_id"])), index=others.index).isin(taken)
            others = others[free & (others["fec_party"].str.upper().map(FEC_PARTY) == party)]
            pools = [("name_other_district", others, True)]
        for status, pool, full_only in pools:
            outcome, rows = pick(nominee, pool, full_only)
            if outcome == "ambiguous":
                return "ambiguous", rows
            if outcome != "none":
                if status == "name_full":
                    status = {"full": "name_full", "variant": "name_variant", "surname": "name_surname",
                              "fuzzy": "name_fuzzy", "duplicate": "duplicate_registration"}[outcome]
                return status, rows
    return "no_fec_record", fec.iloc[:0]


# ---------------------------------------------------------------- races

def choose_slots(cands: pd.DataFrame, key: list[str], chamber: str) -> pd.DataFrame:
    """One row per race and party slot (dem, gop), plus an ind slot for each INDEPENDENT_CANDIDATES race: the chosen
    candidate's name (blank for no_nominee and nominee_unresolved), candidate_count, and preset_status
    (independent_as_dem / independent_as_gop / independent / no_nominee / nominee_unresolved, or blank when the name
    is to be matched).
    2026 slots with several candidates carry their names in `options`."""
    rows = []
    for values, race in cands.groupby(key):
        race_key = dict(zip(key, values))
        for slot, party in PARTIES.items():
            mine = race[race["party"] == party].sort_values("votes", ascending=False)
            row = {**race_key, "slot": slot, "candidate_count": len(mine), "name": None, "preset_status": None,
                   "options": None}
            if mine.empty:
                row["preset_status"] = "no_nominee"
            elif len(mine) == 1:
                row["name"] = mine["name"].iloc[0]
            elif race_key["year"] == 2026:
                row["preset_status"], row["options"] = "nominee_unresolved", list(mine["name"])
            elif mine["votes"].iloc[0] == mine["votes"].iloc[1]:
                row["preset_status"], row["options"] = "nominee_unresolved", list(mine["name"])  # tied leaders
            else:
                row["name"] = mine["name"].iloc[0]
            rows.append(row)
    slots = pd.DataFrame(rows)

    # INDEPENDENTS_AS_DEM / _AS_GOP: the named independent takes the dem / gop slot; INDEPENDENT_CANDIDATES: an ind slot
    slots["displaced"] = None
    seat_col = "district" if chamber == "house" else "class"
    independents = []
    for table, entries in [("INDEPENDENTS_AS_DEM", INDEPENDENTS_AS_DEM), ("INDEPENDENTS_AS_GOP", INDEPENDENTS_AS_GOP),
                           ("INDEPENDENT_CANDIDATES", INDEPENDENT_CANDIDATES)]:
        for (ch, year, state, seat), name in entries.items():
            if ch != chamber:
                continue
            race = cands[(cands["year"] == year) & (cands["state"] == state) & (cands[seat_col] == seat)]
            if ((race["name"] == name) & (race["party"] == "other")).sum() != 1:
                raise ValueError(f"{table} {(ch, year, state, seat)}: {name!r} isn't one non-major-party candidate "
                                 "of that race")
            if table == "INDEPENDENT_CANDIDATES":
                independents.append({"year": year, "state": state, seat_col: seat, "slot": "ind", "candidate_count": 1,
                                     "name": name, "preset_status": "independent", "options": None, "displaced": None})
                continue
            slot = "dem" if table == "INDEPENDENTS_AS_DEM" else "gop"
            hit = (slots["year"] == year) & (slots["state"] == state) & (slots[seat_col] == seat) \
                & (slots["slot"] == slot)
            slots.loc[hit, "displaced"] = slots.loc[hit, "name"]
            slots.loc[hit, ["name", "preset_status", "options"]] = [name, f"independent_as_{slot}", None]
    return pd.concat([slots, pd.DataFrame(independents, columns=slots.columns)], ignore_index=True)


def resolve_2026_by_incumbent(slots: pd.DataFrame, chamber: str, fec: pd.DataFrame) -> pd.DataFrame:
    """A 2026 slot with several candidates goes to the one whose FEC record (a full-name match in the race) is flagged
    INCUMBENT, when exactly one is."""
    out = slots.copy()
    office = "H" if chamber == "house" else "S"
    for i, row in out[out["preset_status"].eq("nominee_unresolved") & (out["year"] == 2026)].iterrows():
        pool = fec[(fec["year"] == 2026) & (fec["office"] == office) & (fec["state"] == row["state"])]
        if office == "H":
            pool = pool[pool["district"] == row["district"]]
        incumbents = []
        for name in row["options"]:
            outcome, records = pick(name, pool, full_only=True)
            if outcome in ("full", "variant", "duplicate") and (records["fec_ici"] == "INCUMBENT").all():
                incumbents.append(name)
        if len(incumbents) == 1:
            out.loc[i, ["name", "preset_status"]] = [incumbents[0], None]
            out.at[i, "options"] = None
            out.loc[i, "displaced"] = "2026 incumbent chosen among " + "; ".join(row["options"])
    return out


def resolve_2026_by_receipts(slots: pd.DataFrame, chamber: str, fec: pd.DataFrame) -> pd.DataFrame:
    """A 2026 House slot in TOP_FUNDRAISER_STATES_2026 still unresolved after resolve_2026_by_incumbent goes to the
    candidate whose FEC record (a full-name match in the district, as there) has the most receipts. A candidate
    without such a record counts as no receipts. A tie, or no candidate with positive receipts, leaves the slot
    unresolved."""
    out = slots.copy()
    if chamber != "house":
        return out
    todo = out["preset_status"].eq("nominee_unresolved") & (out["year"] == 2026) \
        & out["state"].isin(TOP_FUNDRAISER_STATES_2026)
    for i, row in out[todo].iterrows():
        pool = fec[(fec["year"] == 2026) & (fec["office"] == "H") & (fec["state"] == row["state"])
                   & (fec["district"] == row["district"])]
        receipts = {}
        for name in row["options"]:
            outcome, records = pick(name, pool, full_only=True)
            matched = outcome in ("full", "variant", "duplicate")
            receipts[name] = float(records["receipts"].fillna(0).iloc[0]) if matched else 0.0
        top = max(receipts.values())
        leaders = [name for name, amount in receipts.items() if amount == top]
        if top > 0 and len(leaders) == 1:
            out.loc[i, ["name", "preset_status"]] = [leaders[0], None]
            out.at[i, "options"] = None
            out.loc[i, "displaced"] = "2026 top fundraiser chosen among " + "; ".join(
                f"{name} ${amount:,.0f}" for name, amount in sorted(receipts.items(), key=lambda kv: -kv[1]))
    return out


def match_slots(slots: pd.DataFrame, chamber: str, fec: pd.DataFrame, webl_only: pd.DataFrame,
                counts_by_year: dict) -> tuple[pd.DataFrame, list[dict]]:
    """slots plus match_status and the matched record's fields; and review rows for every slot that isn't a clean
    name_full match."""
    seat = "district" if chamber == "house" else "class"
    at_large_states = {year: {s for s, n in acs.normalize_counts(counts_by_year, year).items() if n == 1}
                       for year in YEARS}
    used_overrides, matched, review = set(), [], []
    for row in slots.to_dict("records"):
        at_large = chamber == "house" and row["state"] in at_large_states[row["year"]]
        override_key = (chamber, row["year"], row["state"], row[seat], SLOT_PARTY[row["slot"]])
        records = fec.iloc[:0]
        if row["preset_status"] in ("no_nominee", "nominee_unresolved"):
            status = row["preset_status"]
        elif override_key in MATCH_OVERRIDES:
            used_overrides.add(override_key)
            candidate_id = MATCH_OVERRIDES[override_key]
            records = pd.concat([fec, webl_only])
            records = records[(records["year"] == row["year"]) & (records["candidate_id"] == candidate_id)]
            status = "override" if candidate_id is not None else "no_fec_record"
            if candidate_id is not None and len(records) != 1:
                raise ValueError(f"MATCH_OVERRIDES {override_key}: {candidate_id} matches {len(records)} records")
        else:
            independent = row["preset_status"] in INDEPENDENT_STATUSES
            party = "other" if independent else PARTIES[row["slot"]]
            status, records = match_slot(row["name"], party, row, chamber, fec, webl_only, at_large)
            if independent and len(records) == 1:
                status = row["preset_status"]
        matched.append((row, status, records, at_large))

    # second pass, House only: nominees still without a record may match one filed in another district of the
    # state (stale after redistricting), if no race has already matched it
    if chamber == "house":
        taken = {(r["year"], recs["candidate_id"].iloc[0]) for r, st, recs, _ in matched
                 if len(recs) == 1 and st != "ambiguous"}
        for i, (row, status, records, at_large) in enumerate(matched):
            if status == "no_fec_record" and row["name"] is not None and \
                    (chamber, row["year"], row["state"], row[seat], SLOT_PARTY[row["slot"]]) not in MATCH_OVERRIDES:
                party = "other" if row["preset_status"] in INDEPENDENT_STATUSES \
                    else PARTIES[row["slot"]]
                status, records = match_slot(row["name"], party, row, chamber, fec, webl_only, at_large, taken)
                if len(records) == 1 and status != "ambiguous":
                    taken.add((row["year"], records["candidate_id"].iloc[0]))
                matched[i] = (row, status, records, at_large)

    # API_CORRECTIONS: a nominee still without a record takes their FEC API cycle totals, matched by ID and year
    api = read_api_totals()
    for i, (row, status, records, at_large) in enumerate(matched):
        key = (chamber, row["year"], row["state"], row[seat], SLOT_PARTY[row["slot"]])
        if key in API_CORRECTIONS:
            empty_record = len(records) == 1 and records["coverage_end_date"].isna().all() \
                and (records["receipts"] == 0).all()
            if status != "no_fec_record" and not empty_record:
                raise ValueError(f"API_CORRECTIONS {key}: the slot is {status}, not no_fec_record or an empty record")
            found = api[(api["year"] == row["year"]) & (api["candidate_id"] == API_CORRECTIONS[key])]
            if len(found) == 1:
                matched[i] = (row, "fec_api", found.assign(office="H" if chamber == "house" else "S",
                                                            state=row["state"]), at_large)

    out = []
    for row, status, records, _ in matched:
        record = records.iloc[0].to_dict() if len(records) == 1 and status != "ambiguous" else {}
        out.append({**row, "match_status": status, "candidate_id": record.get("candidate_id"),
                    "fec_name": record.get("fec_name"), "fec_party": record.get("fec_party"),
                    "fec_district": record.get("district"), "fec_file": record.get("fec_file"),
                    **{f: record.get(f) for f in FINANCE}})
        if status != "name_full" or row["displaced"] is not None:
            considered = records if len(records) else fec.iloc[:0]
            review.append({"chamber": chamber, "year": row["year"], "state": row["state"], "seat": row[seat],
                           "party": SLOT_PARTY[row["slot"]], "nominee": row["name"], "match_status": status,
                           "candidate_count": row["candidate_count"],
                           "note": row["displaced"] or ("; ".join(row["options"]) if row["options"] else None),
                           "fec_records": "; ".join(f"{r.candidate_id} {r.fec_name} ({r.fec_party}, {r.fec_file})"
                                                    for r in considered.itertuples()) or None})
    unused = {k for k in MATCH_OVERRIDES if k[0] == chamber} - used_overrides
    if unused:
        raise ValueError(f"MATCH_OVERRIDES never used: {sorted(unused)}")
    return pd.DataFrame(out), review


def read_api_totals(path: Path = API_TOTALS_PATH) -> pd.DataFrame:
    """fec_api_totals.py's records (year, candidate_id, the FINANCE fields), or none if it hasn't been run."""
    columns = ["year", "candidate_id", "fec_name", "fec_party", "district", "fec_file", *FINANCE]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"candidate_id": str}, parse_dates=DATES)
    return df.assign(fec_name="FEC API candidate totals", fec_party=None, district=None, fec_file=path.name)[columns]


def read_pre_general(path: Path = PRE_GENERAL_PATH) -> pd.DataFrame:
    """fec_pre_general.py's amounts by (year, candidate_id)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run data-processing/fec_pre_general.py first")
    df = pd.read_csv(path, dtype={"candidate_id": str}, parse_dates=["coverage_end_date"])
    return df.set_index(["year", "candidate_id"])


def pre_election(m: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """m (one slot's matched rows) with its FINANCE fields replaced by the pre-general amounts where the candidate
    has them, fec_timing saying which, and cycle_receipts keeping the summary's receipts."""
    m = m.copy()
    m["cycle_receipts"] = m["receipts"]
    ids = pd.MultiIndex.from_arrays([m.index.get_level_values("year"), m["candidate_id"]])
    found = ids.isin(pre.index) & m["candidate_id"].notna().to_numpy()
    # Senate: two-year money as of the pre-general report can't exceed the full two-year total. Above it (by
    # SENATE_PRE_GENERAL_TOLERANCE) the six-year cycle wasn't cut to two years (no year-end report found): fall back
    pre_receipts = pre["receipts"].reindex(ids).to_numpy()
    senate = m["candidate_id"].fillna("").str.startswith("S").to_numpy()
    over = senate & ((pre_receipts > SENATE_PRE_GENERAL_TOLERANCE * m["receipts"].astype(float).to_numpy())
                     | (pre_receipts < 0))  # negative: the year-end report wasn't on the same cycle-to-date basis
    found &= ~over
    rows = pre.reindex(ids[found])
    for field in AMOUNTS:
        m[field] = m[field].astype(float)
        m.loc[found, field] = rows[field].to_numpy()
    m["coverage_start_date"] = pd.to_datetime(m["coverage_start_date"])
    m["coverage_end_date"] = pd.to_datetime(m["coverage_end_date"])
    m.loc[found, "coverage_start_date"] = pd.NaT  # cycle-to-date (fec_pre_general.py): the cycle's own start
    m.loc[found, "coverage_end_date"] = rows["coverage_end_date"].to_numpy()
    timing = pd.Series(np.where(m["candidate_id"].notna(), "candidate_summary", None), index=m.index, dtype=object)
    timing[found] = ("pre_general_" + rows["report_type"]).to_numpy()
    m["fec_timing"] = timing
    return m


def wide(matched: pd.DataFrame, races: pd.DataFrame, key: list[str], pre: pd.DataFrame) -> pd.DataFrame:
    """One row per race: race columns, then each slot's (dem, gop, ind) identifiers, match fields and finance fields
    (pre-election, pre_election), applying the missing-data policy. The ind columns are blank in races without an
    ind slot."""
    out = races.copy()
    zero = {"no_nominee", "no_fec_record", "ambiguous"}
    for slot in SLOT_PARTY:
        m = pre_election(matched[matched["slot"] == slot].set_index(key), pre)
        part = pd.DataFrame(index=m.index)
        part["name"] = m["name"]
        part["candidate_id"] = m["candidate_id"]
        part["match_status"] = m["match_status"]
        if slot == "dem":
            part["is_independent"] = m["preset_status"] == "independent_as_dem"  # INDEPENDENTS_AS_DEM
        part["fec_missing"] = m["candidate_id"].isna()
        part["candidate_count"] = m["candidate_count"]
        part["fec_file"] = m["fec_file"]
        part["fec_timing"] = m["fec_timing"]
        part["cycle_receipts"] = m["cycle_receipts"].astype(float)
        for field in FINANCE:
            part[field] = m[field]
        for field in AMOUNTS:
            part[field] = part[field].astype(float).where(~m["match_status"].isin(zero), 0.0)
        for field in DATES:
            part[field] = pd.to_datetime(part[field])
        out = out.join(part.add_prefix(f"{slot}_"), on=key, validate="one_to_one")
    return out.astype({"ind_fec_missing": "boolean", "ind_candidate_count": "Int64"})


def build_chamber(chamber: str, races: pd.DataFrame, cands: pd.DataFrame, key: list[str], fec: pd.DataFrame,
                  webl_only: pd.DataFrame, counts_by_year: dict,
                  pre: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """(interim table, long slot table, review rows) for one chamber."""
    race_keys = pd.MultiIndex.from_frame(races[key])
    cand_keys = pd.MultiIndex.from_frame(cands[key])
    if not cand_keys.isin(race_keys).all():
        raise ValueError(f"{chamber}: candidates for races not in the race list: "
                         f"{sorted(set(cand_keys[~cand_keys.isin(race_keys)]))[:5]}")
    # races without any candidate rows still get two no_nominee slots
    empty = races.loc[~race_keys.isin(cand_keys), key].assign(party="none", name=None, votes=np.nan)
    slots = choose_slots(pd.concat([cands, empty], ignore_index=True), key, chamber)
    slots = resolve_2026_by_incumbent(slots, chamber, fec)
    slots = resolve_2026_by_receipts(slots, chamber, fec)
    matched, review = match_slots(slots, chamber, fec, webl_only, counts_by_year)
    return wide(matched, races, key, pre), matched, review


def house_races(counts_by_year: dict) -> pd.DataFrame:
    """2006-2024 from targets_raw.csv (with its race_type), and 2026's districts (district_counts.json)."""
    raw = pd.read_csv(TARGETS_RAW_PATH, usecols=["year", "state", "district", "race_type"])
    historical = raw[raw["year"] >= YEARS[0]]
    districts = sorted(incumbency.expected_districts(acs.normalize_counts(counts_by_year, 2026)))
    future = pd.DataFrame([{"year": 2026, "state": s, "district": d, "race_type": None} for s, d in districts])
    return pd.concat([historical, future], ignore_index=True).sort_values(["year", "state", "district"],
                                                                        ignore_index=True)


def senate_races(cands_historical: pd.DataFrame) -> pd.DataFrame:
    """2006-2024 from CAGE's races (checked against senate_incumbency.csv), and 2026 from senate_2026.csv."""
    historical = cands_historical[["year", "state", "class", "special"]].drop_duplicates()
    future = incumbency.read_senate_2026()[["state", "class", "special"]].assign(year=2026)
    races = pd.concat([historical, future], ignore_index=True)
    reference = pd.read_csv(SENATE_INCUMBENCY_PATH, usecols=["year", "state", "class", "special"])
    if set(map(tuple, races.to_numpy())) != set(map(tuple, reference[races.columns].to_numpy())):
        raise ValueError("Senate races differ from senate_incumbency.csv")
    return races.sort_values(["year", "state", "class"], ignore_index=True)


# ---------------------------------------------------------------- features

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """The interim table plus the five features and small_dollar_share_flag. A share is undefined (NaN) when its
    party's individual contributions are 0 or unknown, or when FEC's unitemized and total individual amounts give a
    share outside [0, 1] (flagged, e.g. dem_share_outside_0_1; the amounts are kept). The difference is then NaN too."""
    out = df.copy()
    with np.errstate(invalid="ignore"):  # log1p of a (source) amount below -1 is NaN; share_report lists them
        out["fundraising_advantage"] = np.log1p(out["dem_receipts"]) - np.log1p(out["gop_receipts"])
        out["spending_advantage"] = np.log1p(out["dem_disbursements"]) - np.log1p(out["gop_disbursements"])
    out["small_dollar_share_dem"] = (out["dem_individual_unitemized_contributions"]
                                     / out["dem_individual_contributions"].where(out["dem_individual_contributions"] != 0))
    out["small_dollar_share_gop"] = (out["gop_individual_unitemized_contributions"]
                                     / out["gop_individual_contributions"].where(out["gop_individual_contributions"] != 0))
    flags = []
    for slot in PARTIES:
        share = out[f"small_dollar_share_{slot}"]
        outside = (share < 0) | (share > 1)
        flags.append(pd.Series(np.where(outside, f"{slot}_share_outside_0_1", None), index=out.index))
        out[f"small_dollar_share_{slot}"] = share.where(~outside)
    out["small_dollar_share_diff"] = out["small_dollar_share_dem"] - out["small_dollar_share_gop"]
    out["small_dollar_share_flag"] = pd.concat(flags, axis=1).apply(lambda r: ";".join(r.dropna()) or None, axis=1)
    return out


FEATURES = ["fundraising_advantage", "spending_advantage", "small_dollar_share_dem", "small_dollar_share_gop",
            "small_dollar_share_diff", "small_dollar_share_flag"]


# ---------------------------------------------------------------- checks and report

def check_structure(name: str, df: pd.DataFrame, matched: pd.DataFrame, key: list[str], fec: pd.DataFrame,
                    webl_only: pd.DataFrame, chamber: str) -> None:
    """Raise on a repeated race key, an FEC record used twice in a year, or a matched record whose year, office or
    state isn't its race's."""
    if df.duplicated(key).any():
        raise ValueError(f"{name}: repeated keys {df.loc[df.duplicated(key, keep=False), key].to_numpy()[:5]}")
    if df[key].isna().any().any():
        raise ValueError(f"{name}: blank key values (an unresolved class?)")
    used = matched.dropna(subset=["candidate_id"])
    repeated = used[used.duplicated(["year", "candidate_id"], keep=False)]
    if len(repeated):
        raise ValueError(f"{name}: FEC records assigned to two slots:\n{repeated[[*key, 'slot', 'name', 'candidate_id']]}")
    used = used[used["match_status"] != "fec_api"]  # API rows are matched by ID to their slot's race
    records = pd.concat([fec, webl_only]).drop_duplicates(["year", "candidate_id"]).set_index(["year", "candidate_id"])
    found = records.loc[list(zip(used["year"], used["candidate_id"]))]
    office = "H" if chamber == "house" else "S"
    if (found["office"] != office).any() or (found["state"].to_numpy() != used["state"].to_numpy()).any():
        raise ValueError(f"{name}: a matched FEC record's office or state differs from its race's")


def coverage_table(df: pd.DataFrame) -> pd.DataFrame:
    """By year: races, then per party the known nominees, matched records and the match rate among known nominees."""
    rows = {}
    for year, g in df.groupby("year"):
        row = {"races": len(g)}
        for slot in PARTIES:
            status = g[f"{slot}_match_status"]
            known = ~status.isin(["no_nominee", "nominee_unresolved"])
            row[f"{slot}_nominees"] = int(known.sum())
            row[f"{slot}_unresolved"] = int((status == "nominee_unresolved").sum())
            row[f"{slot}_matched"] = int((known & g[f"{slot}_candidate_id"].notna()).sum())
            row[f"{slot}_rate"] = round(row[f"{slot}_matched"] / max(row[f"{slot}_nominees"], 1), 3)
        rows[year] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def value_report(df: pd.DataFrame) -> list[str]:
    """Source anomalies in the matched records, reported and left as they are."""
    lines = []
    for slot in PARTIES:
        p = f"{slot}_"
        matched = df[df[p + "candidate_id"].notna()]
        negative = {f: int((matched[p + f] < 0).sum()) for f in AMOUNTS if (matched[p + f] < 0).any()}
        summary = matched[matched[p + "fec_file"].str.startswith("candidate_summary")]
        blank = {f: int(summary[p + f].isna().sum()) for f in AMOUNTS if summary[p + f].isna().any()}
        backwards = int((matched[p + "coverage_start_date"] > matched[p + "coverage_end_date"]).sum())
        parts = matched[p + "individual_itemized_contributions"] + matched[p + "individual_unitemized_contributions"]
        split_off = int(((parts - matched[p + "individual_contributions"]).abs() > 1).sum())
        no_dates = int(matched[p + "coverage_end_date"].isna().sum())
        lines.append(f"{slot}: negative values {negative or 'none'}; blank amounts in candidate summary records "
                     f"{blank or 'none'}; coverage start after end {backwards}; no coverage dates (no filings) "
                     f"{no_dates}; itemized + unitemized differs from individual by > $1: {split_off}")
    return lines


def share_report(processed: pd.DataFrame) -> str:
    """Shares outside [0, 1], and log features left NaN by a negative source amount (both are FEC values, kept)."""
    out = []
    for feature, field in [("fundraising_advantage", "receipts"), ("spending_advantage", "disbursements")]:
        inputs = processed[[f"dem_{field}", f"gop_{field}"]]
        undefined = processed[processed[feature].isna() & inputs.notna().all(axis=1)]
        out.append(f"{feature}: NaN from a negative {field} amount in {len(undefined)} rows"
                   + (": " + "; ".join(f"{r.year} {r.state} {r[2]} D {r[3]:,.2f} R {r[4]:,.2f}" for r in
                                       undefined[["year", "state", processed.columns[2], f"dem_{field}",
                                                  f"gop_{field}"]].itertuples(index=False)) if len(undefined) else ""))
    for slot in PARTIES:
        col = f"small_dollar_share_{slot}"
        flagged = processed[processed["small_dollar_share_flag"].fillna("").str.contains(f"{slot}_share")]
        out.append(f"{col}: defined {int(processed[col].notna().sum())}; set to NaN as outside [0, 1]: {len(flagged)}"
                   + (": " + "; ".join(f"{r.year} {r.state}-{r[2]}" for r in flagged.itertuples(index=False))
                      if len(flagged) else ""))
    return "\n".join(out)


def timing_report(df: pd.DataFrame) -> str:
    """Per year, the matched D and R slots by fec_timing, and how much of the summary's receipts the pre-general
    amounts are (median ratio, slots with both)."""
    rows = {}
    for year, g in df.groupby("year"):
        timing = pd.concat([g[f"{s}_fec_timing"] for s in PARTIES]).value_counts()
        ratio = pd.concat([g[f"{s}_receipts"] / g[f"{s}_cycle_receipts"].where(g[f"{s}_cycle_receipts"] > 0)
                           for s in PARTIES])
        rows[year] = {**timing.to_dict(), "median pre-general / summary receipts": round(ratio.median(), 3)}
    table = pd.DataFrame.from_dict(rows, orient="index").fillna(0)
    return "FEC timing of the D and R slots (candidate_summary after 2024 is pre-election; before, a look-ahead):\n" \
        + table.to_string()


def cross_check(df: pd.DataFrame, webl: pd.DataFrame, chamber: str) -> list[str]:
    """Receipts against the webl file for the same ID (2008-2024), and FEC's own general-election flags (2006-2012)."""
    lines = []
    w = webl.set_index(["year", "candidate_id"])
    office = "H" if chamber == "house" else "S"
    used = set()
    for slot in PARTIES:
        p = f"{slot}_"
        m = df[df[p + "candidate_id"].notna() & (df["year"] < 2026)]
        ids = list(zip(m["year"], m[p + "candidate_id"]))
        used |= set(ids)
        in_webl = pd.Series([i in w.index for i in ids], index=m.index)
        both = m[in_webl & (m["year"] != WEBL_ONLY_YEAR)]
        webl_receipts = w.loc[list(zip(both["year"], both[p + "candidate_id"])), "receipts"].to_numpy()
        differs = int((np.abs(both[p + "cycle_receipts"].to_numpy() - webl_receipts) > 1).sum())
        flagged = m[in_webl & (m["year"] <= 2012)]
        gen = w.loc[list(zip(flagged["year"], flagged[p + "candidate_id"])), "gen_election"].to_numpy()
        lines.append(f"{slot}: receipts differ from webl (> $1) for {differs} of {len(both)} matched 2008-2024 records; "
                     f"2006-2012 matched records FEC doesn't mark as general-election candidates: "
                     f"{int((gen == '').sum())} of {len(flagged)}")
    gen_candidates = webl[(webl["office"] == office) & webl["gen_election"].isin(["W", "L"])
                          & webl["fec_party"].str.upper().map(FEC_PARTY).notna()]
    missed = gen_candidates[[k not in used for k in zip(gen_candidates["year"], gen_candidates["candidate_id"])]]
    lines.append(f"FEC D/R general-election candidates (W/L flag, 2006-2012) not in any slot: {len(missed)} of "
                 f"{len(gen_candidates)}" + (": " + "; ".join(f"{r.year} {r.state}-{r.district} {r.fec_name}"
                                                             for r in missed.head(15).itertuples()) if len(missed) else ""))
    return lines


def suspect_records(df: pd.DataFrame, results: pd.DataFrame, chamber: str, key: list[str]) -> pd.DataFrame:
    """Slots whose FEC money looks empty though the nominee won or finished within SUSPECT_MARGIN (see
    SUSPECT_RECEIPTS), less REVIEWED_LOW_FUNDING. results: key, winner_party and margin_2p_norm."""
    df = df.merge(results[[*key, "winner_party", "margin_2p_norm"]], on=key, how="inner", validate="one_to_one")
    rows = []
    for slot, party in PARTIES.items():
        end = pd.to_datetime(df[f"{slot}_coverage_end_date"])
        flagged = (~df[f"{slot}_match_status"].isin(["no_nominee", "nominee_unresolved"])
                   & (df[f"{slot}_receipts"].fillna(0) < SUSPECT_RECEIPTS)
                   & (end.isna() | (end < pd.to_datetime(df["year"].astype(str) + "-07-01")))
                   & ((df["winner_party"] == party) | (df["margin_2p_norm"].abs() < SUSPECT_MARGIN)))
        hits = df.loc[flagged, [*key, f"{slot}_name", f"{slot}_match_status", f"{slot}_candidate_id",
                                f"{slot}_receipts", "winner_party", "margin_2p_norm"]]
        rows.append(hits.set_axis([*key, "name", "match_status", "candidate_id", "receipts", "winner_party",
                                   "margin_2p_norm"], axis=1).assign(party=party))
    out = pd.concat(rows, ignore_index=True)
    reviewed = [(chamber, *k[:3], p) in REVIEWED_LOW_FUNDING for k, p in
                zip(out[key].itertuples(index=False, name=None), out["party"])]
    return out[~pd.Series(reviewed, index=out.index, dtype=bool)].sort_values(key, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cage", default=incumbency.CAGE_PATH, type=Path)
    args = parser.parse_args()
    counts_by_year = json.loads(DISTRICT_COUNTS_PATH.read_text())

    print(schema_report())
    fec, webl = read_fec()
    webl_only = webl[~pd.MultiIndex.from_frame(webl[["year", "candidate_id"]]).isin(
        pd.MultiIndex.from_frame(fec[["year", "candidate_id"]]))].drop(columns="gen_election")
    print(f"FEC House/Senate records: {len(fec)} primary ({fec.groupby('year').size().to_dict()}), "
          f"{len(webl_only)} webl-only fallback records")

    cage = pd.read_csv(incumbency.download_cage(args.cage), sep="\t", low_memory=False)
    senate_hist = senate_candidates_historical(cage)
    chambers = {
        "house": (house_races(counts_by_year), pd.concat([house_candidates_historical(), candidates_2026("house")],
                                                          ignore_index=True), ["year", "state", "district"]),
        "senate": (senate_races(senate_hist), pd.concat([senate_hist, candidates_2026("senate")], ignore_index=True),
                   ["year", "state", "class"]),
    }
    pre = read_pre_general()
    review_rows, outputs = [], {}
    for chamber, (races, cands, key) in chambers.items():
        if chamber == "senate":
            cands = cands.drop(columns="special")  # the race list carries it
        interim, matched, review = build_chamber(chamber, races, cands, key, fec, webl_only, counts_by_year, pre)
        check_structure(chamber, interim, matched, key, fec, webl_only, chamber)
        processed = add_features(interim)
        if list(processed.columns) != list(interim.columns) + FEATURES or len(processed) != len(interim):
            raise ValueError(f"{chamber}: processed columns or rows don't extend the interim table")
        outputs[chamber] = (interim, processed)
        review_rows += review

        print(f"\n===== {chamber}: {len(interim)} races")
        print(coverage_table(interim).to_string())
        status = pd.concat([interim["dem_match_status"], interim["gop_match_status"]])
        print("match_status: " + ", ".join(f"{k} {v}" for k, v in status.value_counts().items()))
        for line in value_report(interim) + cross_check(interim, webl, chamber):
            print(line)
        print(share_report(processed))
        print(timing_report(interim))
        results = pd.read_csv(TARGETS_RAW_PATH if chamber == "house" else SENATE_TARGETS_RAW_PATH)
        suspects = suspect_records(interim, results, chamber, key)
        print(f"Possibly empty FEC records of nominees who won or finished within {SUSPECT_MARGIN} points "
              f"({len(suspects)}; review each, then add API_COMMITTEE_TOTALS or REVIEWED_LOW_FUNDING):")
        if len(suspects):
            print(suspects.to_string(index=False))

    review_df = pd.DataFrame(review_rows)
    api_used = review_df[review_df["match_status"] == "fec_api"]
    print(f"\nAPI_CORRECTIONS applied: {len(api_used)} of {len(API_CORRECTIONS)}"
          + "".join(f"\n  not applied (fetch failed or not run; slot stays no_fec_record): {k}"
                    for k in API_CORRECTIONS if (k[1], k[2], k[3], k[4]) not in
                    set(zip(api_used["year"], api_used["state"], api_used["seat"], api_used["party"]))))
    targets = pd.read_csv(TARGETS_RAW_PATH, usecols=["year", "state", "district", "winner_party"])
    unmatched = review_df[(review_df["chamber"] == "house") & (review_df["match_status"] == "no_fec_record")].merge(
        targets, left_on=["year", "state", "seat"], right_on=["year", "state", "district"])
    winners = unmatched[unmatched["party"] == unmatched["winner_party"]]
    print(f"\nHouse nominees without an FEC record who won their race ({len(winners)}; zero-filled, likely gaps in "
          "FEC's files): " + "; ".join(f"{r.year} {r.state}-{r.seat} {r.party} {r.nominee}" for r in winners.itertuples()))
    print("\nReview (data/interim/fec_match_review.csv): " + ", ".join(
        f"{k} {v}" for k, v in review_df["match_status"].value_counts().items()))
    for status in ["ambiguous", "nominee_unresolved", "name_surname", "name_party_differs", "name_other_district",
                   "independent_as_dem", "independent_as_gop", "independent"]:
        rows = review_df[review_df["match_status"] == status]
        if len(rows):
            print(f"-- {status} ({len(rows)}):")
            print(rows.drop(columns="match_status").to_string(index=False, max_colwidth=90))

    for chamber, (interim, processed) in outputs.items():
        acs.write_csv_atomic(interim, INTERIM_DIR / f"{chamber}_fec_raw.csv")
        acs.write_csv_atomic(processed, PROCESSED_DIR / f"{chamber}_fec.csv")
    acs.write_csv_atomic(review_df, INTERIM_DIR / "fec_match_review.csv")
    print(f"\n-> {INTERIM_DIR}/{{house,senate}}_fec_raw.csv, {PROCESSED_DIR}/{{house,senate}}_fec.csv, "
          f"{INTERIM_DIR}/fec_match_review.csv")


if __name__ == "__main__":
    main()
