"""Same-party top-two House races of 2020, 2022 and 2024 (dem_vs_dem, gop_vs_gop in California and Washington): each
finalist's primary result, fundraising and November result, for the full-forecast backtest of the top-two model
(house/simulation/top_two.py).

  data/processed/house/toptwosameparty_history.csv
      one row per finalist, key year, state, district; the columns top_two.predict_top_two reads (candidate_name,
      primary_rank, primary_share, primary_share_top_two, individual_contribution_advantage, poll_share_top_two),
      plus primary_votes, individual_contributions, general_votes, general_share_top_two, won, source

Races: historical_races' same_party treatment in CA and WA. Louisiana's same-party races (LA-6 2022, LA-4 2024) are
left out: their November ballot is the all-candidate election itself, with no primary to predict from.

Finalists are the two November candidates (write-ins dropped). Sources:
  2020, 2022   FEC Federal Elections (data/raw/fec_federal_elections): PRIMARY VOTES and GENERAL VOTES per
               candidate, and the district's primary total from its "District Votes:" row; FEC ID links fundraising
  2024         FEC hasn't published 2024, so: the CA Secretary of State's certified Statement of Vote for the March 5
               primary, U.S. Representative by district (data/raw/ca_sos_primary_2024/79-us-rep-congress.xlsx), the
               WA Secretary of State's August 6 primary export (data/raw/wa_sos_primary_2024/20240806_allstate.csv),
               and MEDSL's 2024 general results (data/raw/1976-2024-house.tab). Both downloaded here if absent, with
               URL and sha256 in SOURCE.json. Names are matched on the last name within the district, and every
               finalist must match exactly one primary candidate and one FEC record
Fundraising: individual contributions as of each finalist's pre-general report (data/interim/fec_pre_general.csv,
fec_pre_general.py), so only money raised before the election counts; individual_contribution_advantage =
log1p(own) - log1p(opponent's), as for 2026 (whose summary file already ends in mid-October).
No polls: poll_share_top_two is blank, so predict_top_two uses its no-poll weights.
A finalist without a pre-general record who raised under $5,000 (FEC's reporting threshold, so they filed no
reports) takes the summary's amount; otherwise (Sara Jacobs, CA-53 2020, in neither file) the race's fundraising is
blank, flagged fundraising_missing, and the backtest predicts it from the primary alone.

primary_share is 100 * votes / every vote cast for the office in the district's primary; primary_share_top_two is
100 * votes / the two finalists' votes; primary_rank is 1 for the finalist with more primary votes.
"""

import hashlib
import json
import re
import sys
import unicodedata
import urllib.request
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src")]
import top_two_same_party as tt

import house.regression as hr
from house import simulation

YEARS = (2020, 2022, 2024)
STATES = ("CA", "WA")
FEC_DIR = ROOT / "data/raw/fec_federal_elections"
FEC_SHEETS = {2020: "13. US House Results by State", 2022: "8. US House Results by State"}
SUMMARY_PATH = ROOT / "data/raw/fec_candidate_summaries/candidate_summary_{year}.csv"  # 2024 name matching only
PRE_GENERAL_PATH = ROOT / "data/interim/fec_pre_general.csv"
FILING_THRESHOLD = 5000  # FEC: a candidate who raises or spends less files no reports
MEDSL_PATH = ROOT / "data/raw/1976-2024-house.tab"
DOWNLOADS = {
    "ca": (ROOT / "data/raw/ca_sos_primary_2024/79-us-rep-congress.xlsx",
           "https://elections.cdn.sos.ca.gov/sov/2024-primary/sov/79-us-rep-congress.xlsx"),
    "wa": (ROOT / "data/raw/wa_sos_primary_2024/20240806_allstate.csv",
           "https://results.vote.wa.gov/results/20240806/export/20240806_allstate.csv"),
}
OUTPUT_PATH = ROOT / "data/processed/house/toptwosameparty_history.csv"
KEY = ["year", "state", "district"]
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def last_name(name: str) -> str:
    """Lowercase ASCII last name; "Last, First" or "First Last", suffixes and quotes dropped."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    name = name.split(",")[0] if "," in name else name
    tokens = [t for t in re.sub(r"[^a-z\s-]", " ", name).split() if t not in SUFFIXES]
    return tokens[-1] if tokens else ""


def download(path: Path, url: str) -> None:
    """Download url to path if absent, and record its URL, sha256 and download time in SOURCE.json."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            path.write_bytes(response.read())
    source = path.parent / "SOURCE.json"
    if not source.exists():
        downloaded = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
        source.write_text(json.dumps({"url": url, "file": path.name, "downloaded": downloaded,
                                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, indent=2))


def same_party_races() -> pd.DataFrame:
    with warnings.catch_warnings(action="ignore"):
        races = simulation.historical_races(hr.load_targets(), simulation.load_targets_raw())
    keep = races["year"].isin(YEARS) & races["state"].isin(STATES) & (races["treatment"] == "same_party")
    return races.loc[keep, [*KEY, "race_type"]].reset_index(drop=True)


def fec_finalists(year: int, races: pd.DataFrame) -> pd.DataFrame:
    """2020, 2022: each race's two November candidates from FEC Federal Elections, with primary votes and the
    district's primary total."""
    with warnings.catch_warnings(action="ignore"):
        df = pd.read_excel(FEC_DIR / f"federalelections{year}.xlsx", FEC_SHEETS[year], dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    df = df[df["STATE ABBREVIATION"].isin(STATES)].copy()
    df["district"] = pd.to_numeric(df["DISTRICT"].str.extract(r"^(\d+)")[0], errors="coerce").astype("Int64")
    totals = df[df["TOTAL VOTES"].fillna("").str.strip() == "District Votes:"]
    totals = totals.assign(primary_total=pd.to_numeric(totals["PRIMARY VOTES"])).groupby(
        ["STATE ABBREVIATION", "district"])["primary_total"].first()
    rows = []
    for race in races[races["year"] == year].itertuples(index=False):
        d = df[(df["STATE ABBREVIATION"] == race.state) & (df["district"] == race.district) & df["FEC ID"].notna()]
        general = pd.to_numeric(d["GENERAL VOTES"], errors="coerce")
        finalists = d[general.notna() & (general > 0)]
        if len(finalists) != 2:
            raise ValueError(f"{year} {race.state}-{race.district}: {len(finalists)} FEC general candidates")
        for _, f in finalists.iterrows():
            rows.append({"year": year, "state": race.state, "district": race.district,
                         "candidate_name": f["CANDIDATE NAME"].strip(), "candidate_id": f["FEC ID"].strip(),
                         "primary_votes": float(f["PRIMARY VOTES"]),
                         "primary_total": float(totals[(race.state, race.district)]),
                         "general_votes": float(f["GENERAL VOTES"]), "source": f"FEC Federal Elections {year}"})
    return pd.DataFrame(rows)


def read_ca_sov(path: Path) -> pd.DataFrame:
    """The CA by-district Statement of Vote: district, candidate, votes (each candidate's District Totals)."""
    raw = pd.read_excel(path, header=None)
    rows, district, names = [], None, None
    for _, row in raw.iterrows():
        first = str(row[0]).strip() if pd.notna(row[0]) else ""
        if m := re.match(r"(\d+)\w* Congressional District", first):
            district, names = int(m[1]), None
        elif not first and pd.notna(row[1]) and names is None:
            names = {c: str(v).replace("\n", " ").replace("*", "").strip() for c, v in row.items() if c and pd.notna(v)}
        elif first == "District Totals":
            rows += [{"district": district, "candidate": n, "votes": float(row[c])} for c, n in names.items()]
            names = None
    return pd.DataFrame(rows)


def read_wa_primary(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    m = raw["Race"].str.extract(r"Congressional District (\d+) - U.S. Representative")
    raw = raw[m[0].notna()].assign(district=m[0].dropna().astype(int))
    return pd.DataFrame({"district": raw["district"], "candidate": raw["Candidate"].str.replace("&quot;", '"'),
                         "votes": raw["Votes"].astype(float)})


def match_one(name: str, candidates: pd.Series, what: str) -> int:
    hits = candidates.index[candidates.map(last_name) == last_name(name)]
    if len(hits) != 1:
        raise ValueError(f"{name}: {len(hits)} matches in {what}")
    return hits[0]


def finalists_2024(races: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """2024: the November candidates from MEDSL, their primary votes from the CA / WA files, and their FEC IDs."""
    for path, url in DOWNLOADS.values():
        download(path, url)
    primaries = {"CA": read_ca_sov(DOWNLOADS["ca"][0]), "WA": read_wa_primary(DOWNLOADS["wa"][0])}
    medsl = pd.read_csv(MEDSL_PATH, dtype=str)
    medsl = medsl[(medsl["year"] == "2024") & (medsl["stage"] == "GEN") & (medsl["writein"] != "TRUE")]
    medsl = medsl.assign(district=medsl["district"].astype(int))
    rows = []
    for race in races[races["year"] == 2024].itertuples(index=False):
        general = medsl[(medsl["state_po"] == race.state) & (medsl["district"] == race.district)]
        primary = primaries[race.state].query("district == @race.district")
        fec = summary[(summary["Cand_Office_St"] == race.state) & (summary["district"] == race.district)]
        if len(general) != 2:
            raise ValueError(f"2024 {race.state}-{race.district}: {len(general)} MEDSL general candidates")
        for _, g in general.iterrows():
            p = primary.loc[match_one(g["candidate"], primary["candidate"], f"{race.state}-{race.district} primary")]
            f = fec.loc[match_one(g["candidate"], fec["Cand_Name"], f"{race.state}-{race.district} FEC")]
            rows.append({"year": 2024, "state": race.state, "district": race.district,
                         "candidate_name": p["candidate"], "candidate_id": f["Cand_Id"],
                         "primary_votes": p["votes"], "primary_total": primary["votes"].sum(),
                         "general_votes": float(g["candidatevotes"]),
                         "source": f"{race.state} SOS 2024 primary; MEDSL 2024 general"})
    return pd.DataFrame(rows)


def read_summary(year: int) -> pd.DataFrame:
    raw = pd.read_csv(str(SUMMARY_PATH).format(year=year), dtype=str, keep_default_na=False)
    raw = raw[raw["Cand_Office"] == "H"]
    return raw.assign(district=pd.to_numeric(raw["Cand_Office_Dist"], errors="coerce"))


def build() -> pd.DataFrame:
    races = same_party_races()
    finalists = pd.concat([fec_finalists(2020, races), fec_finalists(2022, races),
                           finalists_2024(races, read_summary(2024))], ignore_index=True)
    money = pd.read_csv(PRE_GENERAL_PATH, dtype={"candidate_id": str})
    out = finalists.merge(money[["year", "candidate_id", "individual_contributions"]], on=["year", "candidate_id"],
                          how="left", validate="one_to_one")
    # no pre-general report: under FEC's $5,000 reporting threshold the summary's tiny amount is used as is
    summary = pd.concat([tt.read_fec(str(SUMMARY_PATH).format(year=y)).assign(year=y) for y in YEARS])
    small = summary.reset_index().rename(columns={"Cand_Id": "candidate_id"}).set_index(["year", "candidate_id"])[
        "individual_contributions"]
    small = small[small < FILING_THRESHOLD]
    keys = list(zip(out["year"], out["candidate_id"]))
    out["individual_contributions"] = out["individual_contributions"].fillna(
        pd.Series([small.get(k) for k in keys], index=out.index, dtype=float))
    race = out["year"].astype(str) + out["state"] + out["district"].astype(str)
    out["fundraising_missing"] = out["individual_contributions"].isna().groupby(race).transform("any")
    missing = out.loc[out["individual_contributions"].isna(), ["year", "state", "district", "candidate_name"]]
    if len(missing):
        print(f"No pre-general FEC record (race kept, fundraising blank): "
              f"{missing.values.tolist()}")
    if (out.groupby(race).size() != 2).any() or len(out) != 2 * len(races):
        raise ValueError("Every race needs exactly two finalists")
    out["primary_share"] = 100 * out["primary_votes"] / out["primary_total"]
    out["primary_share_top_two"] = 100 * out["primary_votes"] / out["primary_votes"].groupby(race).transform("sum")
    out["primary_rank"] = out.groupby(race)["primary_votes"].rank(ascending=False, method="first").astype(int)
    out["individual_contribution_advantage"] = tt.advantage(out["individual_contributions"], race)
    out["poll_share_top_two"] = np.nan  # no polls in the backtest
    out["general_share_top_two"] = 100 * out["general_votes"] / out["general_votes"].groupby(race).transform("sum")
    out["won"] = out["general_votes"] == out["general_votes"].groupby(race).transform("max")
    out = out.merge(races, on=KEY, validate="many_to_one")
    return out[[*KEY, "race_type", "candidate_name", "candidate_id", "primary_votes", "primary_total", "primary_share",
                "primary_share_top_two", "primary_rank", "individual_contributions",
                "individual_contribution_advantage", "fundraising_missing", "poll_share_top_two", "general_votes",
                "general_share_top_two", "won", "source"]].sort_values([*KEY, "primary_rank"]).reset_index(drop=True)


def main() -> None:
    out = build()
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"-> {OUTPUT_PATH} ({len(out)} finalists, {len(out) // 2} races)")
    print(out[[*KEY, "candidate_name", "primary_share", "primary_share_top_two", "individual_contribution_advantage",
               "general_share_top_two", "won"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
