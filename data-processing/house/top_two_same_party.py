"""2026 same-party top-two House races (dem_vs_dem, gop_vs_gop): each finalist's June primary result and fundraising.

  data/processed/house/toptwosameparty.csv
      one row per general-election candidate of those races, key year, state, district, candidate_id

Inputs:
  data/processed/house/candidates_2026.csv   (candidates_2026.py) the races and their two finalists
  data/raw/ca_sos_primary_2026/76-us-rep.xlsx  the Secretary of State's certified Statement of Vote for the June 2,
                                               2026 primary, U.S. Representative by county; downloaded here if absent,
                                               with its URL and sha256 in SOURCE.json
  data/raw/fec_candidate_summaries/candidate_summary_2026.csv  FEC's 2026 candidate summary (cycle to date)
  data/reference/top_two_same_party_polls_2026.csv  hand-entered polls, one row per finalist by registry name;
                                               poll_share -1 means no poll, or one judged not usable

Columns:
  candidate_id ... race_type         from the registry; fec_status is FEC's INCUMBENT / CHALLENGER / OPEN
  primary_ballot_name                the name on the primary ballot (PRIMARY_NAME_ALIASES where it differs)
  primary_votes, primary_rank        the candidate's certified votes, summed over counties, and rank in the district
  primary_share                      100 * votes / all votes cast for the office in the district
  primary_share_top_two              100 * votes / the two finalists' votes
  primary_dem_share, primary_gop_share  race-level: 100 * all D (R) candidates' votes / all votes, the voters
                                     whose party has no finalist (or a second one) in November
  poll_share, poll_count, pollster, poll_end_date  the hand-entered poll; blank (never 0) where -1
  poll_share_top_two                 100 * poll_share / the two finalists' poll shares (undecideds dropped)
  receipts ... cash_on_hand          FEC totals through fec_coverage_end_date (after the primary, so not a June
                                     snapshot); self_funding = candidate contributions + candidate loans
  fundraising_advantage              log1p(own receipts) - log1p(opponent's), fec_finance.py's formula with the
                                     opponent in place of the other party; individual_contribution_advantage the same
                                     on individual contributions (which leave out self-funding and loans)

Checks: every finalist matches exactly one primary candidate, one FEC record and one poll row, and the two finalists
are the district's top two in the primary.
"""

import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # data-processing/, for fec_finance
import acs
import fec_finance

ROOT = HERE.parents[1]
REGISTRY_PATH = ROOT / "data/processed/house/candidates_2026.csv"
FEC_2026_PATH = ROOT / "data/raw/fec_candidate_summaries/candidate_summary_2026.csv"
RAW_DIR = ROOT / "data/raw/ca_sos_primary_2026"
SOV_URL = "https://elections.cdn.sos.ca.gov/sov/2026-primary/sov/76-us-rep.xlsx"
SOV_PATH = RAW_DIR / "76-us-rep.xlsx"
POLLS_PATH = ROOT / "data/reference/top_two_same_party_polls_2026.csv"
OUTPUT_PATH = ROOT / "data/processed/house/toptwosameparty.csv"
NO_POLL = -1  # poll_share in POLLS_PATH: no poll, or one judged not usable
YEAR = 2026
PRIMARY_DATE = "2026-06-02"
SAME_PARTY_TYPES = ["dem_vs_dem", "gop_vs_gop"]
# (state, district, registry name) -> primary ballot name, where the two sources spell a finalist differently
PRIMARY_NAME_ALIASES = {
    ("CA", 29, "Luz Rivas"): "Luz Maria Rivas",
    ("CA", 29, "Angelica Dueñas"): "Angélica María Dueñas",
}
SOV_PARTIES = {"DEM": "D", "REP": "R"}  # everything else (NPP, GRN, write-ins, ...) is other
COLUMNS = ["year", "state", "district", "candidate_id", "candidate_name", "party", "race_type", "fec_status",
           "primary_date", "primary_ballot_name", "primary_votes", "primary_rank", "primary_share",
           "primary_share_top_two", "primary_dem_share", "primary_gop_share", "primary_total_votes",
           "poll_share", "poll_share_top_two", "poll_count", "pollster", "poll_end_date", "receipts", "individual_contributions", "self_funding", "cash_on_hand", "fec_coverage_end_date",
           "fundraising_advantage", "individual_contribution_advantage", "source"]


def download_sov() -> None:
    """Save the certified workbook unmodified and record its URL and sha256."""
    request = urllib.request.Request(SOV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as resp:
        raw = resp.read()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SOV_PATH.write_bytes(raw)
    source = {"url": SOV_URL, "sha256": hashlib.sha256(raw).hexdigest(),
              "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "built_by": "data-processing/house/top_two_same_party.py"}
    (RAW_DIR / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")


def read_sov(path: Path) -> pd.DataFrame:
    """One row per primary candidate: district, primary_ballot_name, sov_party, primary_votes.

    The workbook's "Working Data" sheet has one column per candidate: row 0 the district (on its first candidate
    only), row 1 the name, row 2 the party, then one row per county, and each district's "District Totals" and
    "Percent" rows, which are skipped. (The formatted sheet is avoided: its overflow blocks shift the county labels
    by a row.)"""
    sheet = pd.read_excel(path, sheet_name="Working Data", header=None)
    counties = sheet.iloc[3:]
    counties = counties[counties[1].notna() & ~counties[1].astype(str).str.strip().isin(["District Totals", "Percent"])]
    return pd.DataFrame({
        "district": sheet.iloc[0, 2:].ffill().astype(int).values,
        "primary_ballot_name": sheet.iloc[1, 2:].values,
        "sov_party": sheet.iloc[2, 2:].values,
        "primary_votes": counties.iloc[:, 2:].apply(pd.to_numeric).sum().astype(int).values,
    })


def primary_shares(sov: pd.DataFrame) -> pd.DataFrame:
    """sov plus primary_rank, primary_share, primary_total_votes and the race's primary_dem_share / gop_share."""
    out = sov.copy()
    by_district = out.groupby("district")["primary_votes"]
    out["primary_total_votes"] = by_district.transform("sum")
    out["primary_share"] = 100 * out["primary_votes"] / out["primary_total_votes"]
    out["primary_rank"] = by_district.rank(ascending=False, method="min").astype(int)
    party = out["sov_party"].map(SOV_PARTIES)
    for p, column in [("D", "primary_dem_share"), ("R", "primary_gop_share")]:
        out[column] = 100 * out["primary_votes"].where(party == p, 0).groupby(out["district"]).transform("sum") \
            / out["primary_total_votes"]
    return out


def read_fec(path: Path) -> pd.DataFrame:
    """The FEC fields used here, by candidate_id."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("Cand_Id")
    return pd.DataFrame({
        "fec_status": raw["Cand_Incumbent_Challenger_Open_Seat"],
        "receipts": fec_finance.amounts(raw["Total_Receipt"]),
        "individual_contributions": fec_finance.amounts(raw["Individual_Contribution"]),
        "self_funding": fec_finance.amounts(raw["Cand_Contribution"]) + fec_finance.amounts(raw["Cand_Loan"]),
        "cash_on_hand": fec_finance.amounts(raw["Cash_On_Hand_COP"]),
        "fec_coverage_end_date": fec_finance.dates(raw["Coverage_End_Date"]).dt.date,
    })


def advantage(amount: pd.Series, race: pd.Series) -> pd.Series:
    """log1p(own) - log1p(opponent's) in a two-candidate race."""
    own = np.log1p(amount)
    return own - (own.groupby(race).transform("sum") - own)


def read_polls(path: Path) -> pd.DataFrame:
    """The hand-entered polls, with NO_POLL as a blank poll_share."""
    polls = pd.read_csv(path)
    polls["poll_share"] = polls["poll_share"].where(polls["poll_share"] != NO_POLL)
    return polls


def build(registry: pd.DataFrame, sov: pd.DataFrame, fec: pd.DataFrame, polls: pd.DataFrame) -> pd.DataFrame:
    finalists = registry[(registry["year"] == YEAR) & registry["race_type"].isin(SAME_PARTY_TYPES)].copy()
    if set(finalists["state"]) != {"CA"} or (finalists.groupby("district").size() != 2).any():
        raise ValueError("expected exactly two finalists per same-party race, all in CA (the only primary read here)")
    finalists["primary_ballot_name"] = [PRIMARY_NAME_ALIASES.get((s, d, n), n) for s, d, n in
                                        zip(finalists["state"], finalists["district"], finalists["candidate_name"])]
    out = finalists.merge(primary_shares(sov), on=["district", "primary_ballot_name"], how="left", validate="1:1")
    if out["primary_votes"].isna().any():
        missing = out.loc[out["primary_votes"].isna(), ["district", "primary_ballot_name"]]
        raise ValueError(f"finalists not on the primary ballot (add to PRIMARY_NAME_ALIASES):\n{missing}")
    if (out.groupby("district")["primary_rank"].apply(sorted).map(tuple) != (1, 2)).any():
        raise ValueError("a race's finalists aren't its primary's top two")
    missing_fec = set(out["candidate_id"]) - set(fec.index)
    if out["candidate_id"].isna().any() or missing_fec:
        raise ValueError(f"finalists without an FEC record: {sorted(missing_fec)}")
    out = out.join(fec, on="candidate_id")
    out = out.merge(polls, on=["year", "state", "district", "candidate_name"], how="left", validate="1:1",
                    indicator=True)
    if (out["_merge"] != "both").any() or len(polls) != len(out):
        raise ValueError(f"{POLLS_PATH.name} needs exactly one row per finalist, by registry name")
    out["poll_share_top_two"] = 100 * out["poll_share"] / out.groupby("district")["poll_share"].transform("sum",
                                                                                                         min_count=2)

    out["primary_share_top_two"] = 100 * out["primary_votes"] / out.groupby("district")["primary_votes"].transform("sum")
    out["fundraising_advantage"] = advantage(out["receipts"], out["district"])
    out["individual_contribution_advantage"] = advantage(out["individual_contributions"], out["district"])
    out["primary_date"] = PRIMARY_DATE
    out["source"] = f"CA SoS certified Statement of Vote ({SOV_URL}); FEC {FEC_2026_PATH.name}"
    return out[COLUMNS].sort_values(["state", "district", "primary_rank"]).reset_index(drop=True)


def main() -> None:
    if not SOV_PATH.exists():
        download_sov()
    out = build(pd.read_csv(REGISTRY_PATH), read_sov(SOV_PATH), read_fec(FEC_2026_PATH), read_polls(POLLS_PATH))
    print(f"{len(out)} finalists in {out['district'].nunique()} same-party races")
    print(out[["district", "candidate_name", "fec_status", "primary_share", "primary_share_top_two", "poll_share",
               "poll_share_top_two", "receipts",
               "individual_contributions", "fundraising_advantage", "individual_contribution_advantage"]]
          .round(2).to_string(index=False))
    acs.write_csv_atomic(out, OUTPUT_PATH)
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
