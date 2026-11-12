"""Each House and Senate candidate's FEC totals as of the pre-general report: what was public before Election Day.

The candidate summary files (fec_finance.py's source) give two-year cycle totals through December 31 of the election
year for 2006-2024, so they include money raised and spent after the election. 2026's summary runs through mid-October
2026. This builds the pre-election equivalent for every year, so training and prediction see the same window.

  data/interim/fec_pre_general.csv   one row per year and candidate_id: committee_id, report_type, coverage_end_date,
                                     the FINANCE amounts as of that report, and the inputs behind them

Method, per candidate and cycle Y:
  committee        one of the candidate's campaign committees for Y, the first of these that filed the report
                   (committee_source): the candidate master's principal campaign committee (cn, CAND_PCC); the
                   linkage file's principal, then authorized, committees (ccl, CMTE_DSGN P / A, FEC_ELECTION_YR Y);
                   a committee the committee master links to the candidate (cm, CAND_ID); last, for a needed
                   candidate still without a report who raised at least LOOKUP_MIN_RECEIPTS in the cycle, the FEC
                   API's own list of their committees (the bulk files can be stale: Cisneros, CA-31 2024, and
                   Gillen, NY-4 2024, ran from new committees they don't list). One committee per candidate, so
                   transfers between a candidate's committees are never counted twice
  report           its pre-general report (12G: coverage through about 20 days before the election, filed before it);
                   without one, its October quarterly (Q3, through September 30 of Y), fetched only for the candidates
                   that need it: fec_finance.py's nominee slots (data/interim/{house,senate}_fec_raw.csv) and the
                   top-two finalists (data/processed/house/toptwosameparty_history.csv). Most recent amendment only
  amount           FEC's Form 3 Column B, "election cycle to date": the cycle starts after the seat's previous
                   general election.
                   House: that is about two years, so the amount is the report's cycle-to-date as filed (from the
                   previous November to the report), the same definition in every year, 2026 included. e.g.
                   Fitzpatrick (PA-1) 2022: 12G $4,294,970.77 through Oct 19, 2022; the full-cycle summary, through
                   Dec 31, has $4,560,231.74.
                   Senate: the cycle is six years, so the committee's cycle-to-date at Dec 31, Y-2 (its year-end
                   report for Y-2, if any) is subtracted, leaving the last two years:
                       amount = report's cycle-to-date - year-end(Y-2) cycle-to-date
                   e.g. Ron Johnson (WI) 2022: 12G $32,043,111.96 over 2017-2022, less $1,658,558.56 at Dec 31, 2020
  point in time    cash_on_hand (cash_on_hand_end_period) and debt (debts_owed_by_committee) as of the report
  unusable         a report without cycle-to-date receipts (common for Senate paper filings before 2019) is skipped,
                   so the candidate falls to their next committee or report, or to no record

  FINANCE field                          API field (Form 3)
  receipts                               total_receipts_ytd
  disbursements                          total_disbursements_ytd
  individual_contributions               total_individual_contributions_ytd
  individual_unitemized_contributions    individual_unitemized_contributions_ytd
  individual_itemized_contributions      individual_itemized_contributions_ytd
  cash_on_hand                           cash_on_hand_end_period (not differenced)
  debt                                   debts_owed_by_committee (not differenced)

Sources (data/raw/fec_pre_general/, SOURCE.json lists every file and request, key redacted):
  ccl{yy}.zip, cn{yy}.zip, cm{yy}.zip  https://www.fec.gov/files/bulk-downloads/{Y}/: the candidate-committee
                              linkage file, the candidate master and the committee master
  api/{type}_{cycle}_p{n}.json  GET /v1/reports/house-senate/?cycle=&report_type=&report_year=&is_amended=false, every
                              page; only the fields used here are kept. 12G for cycle Y, YE for cycle Y-2
  api/Q3_{Y}_needed_*_b{n}_p{p}.json  the October quarterlies of the committees that need them, 100 committee_ids per
                              request (api/committee/{id}_Q3_{Y}.json: the same, fetched one committee at a time
                              earlier, read where present)
  api/YE_{Y-2}_senate_b{n}_p{p}.json  the year-end reports of Y's Senate committees, batched the same way (the bulk
                              YE_{cycle}_p{n}.json pages, cached for the early cycles, are used where complete)
The key allows 60 requests an hour; batching keeps a full rebuild from scratch to about 12 bulk 12G pages per cycle.
Pages are fetched WORKERS at a time (the API answers slowly, well under its rate limit).
Each file is fetched once and reread on later runs. The key is FEC_API_KEY (else DEMO_KEY, which is far too slow).

    python data-processing/fec_pre_general.py     # then fec_finance.py, house_features.py, ...
On a fresh build without the interim slot files, run fec_finance.py once (it needs this file, so: fec_pre_general.py,
fec_finance.py, fec_pre_general.py again for the Q3 fallbacks, fec_finance.py).
"""

import hashlib
import io
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/fec_pre_general"
API_DIR = RAW_DIR / "api"
SOURCE_PATH = RAW_DIR / "SOURCE.json"
OUTPUT_PATH = ROOT / "data/interim/fec_pre_general.csv"
API = "https://api.open.fec.gov/v1/reports/house-senate/"
COMMITTEE_API = "https://api.open.fec.gov/v1/committee/{committee_id}/reports/"
CANDIDATE_API = "https://api.open.fec.gov/v1/candidate/{candidate_id}/committees/"
LOOKUP_MIN_RECEIPTS = 5000  # FEC's reporting threshold: below it a candidate files no reports
NEEDED_PATHS = [ROOT / "data/interim/house_fec_raw.csv", ROOT / "data/interim/senate_fec_raw.csv",
                ROOT / "data/processed/house/toptwosameparty_history.csv"]
WORKERS = 4
LOCK = threading.Lock()
BULK_URL = "https://www.fec.gov/files/bulk-downloads/{year}/{kind}{yy}.zip"
COMMITTEE_SOURCES = ["candidate master CAND_PCC", "linkage principal", "linkage authorized",
                     "committee master CAND_ID", "FEC API candidate committees"]
YEARS = list(range(2006, 2027, 2))
CCL_COLUMNS = ["CAND_ID", "CAND_ELECTION_YR", "FEC_ELECTION_YR", "CMTE_ID", "CMTE_TP", "CMTE_DSGN", "LINKAGE_ID"]
CUMULATIVE = {
    "receipts": "total_receipts_ytd",
    "disbursements": "total_disbursements_ytd",
    "individual_contributions": "total_individual_contributions_ytd",
    "individual_unitemized_contributions": "individual_unitemized_contributions_ytd",
    "individual_itemized_contributions": "individual_itemized_contributions_ytd",
}
POINT_IN_TIME = {"cash_on_hand": "cash_on_hand_end_period", "debt": "debts_owed_by_committee"}
KEEP = ["committee_id", "report_type", "report_year", "cycle", "coverage_start_date", "coverage_end_date",
        "receipt_date", "file_number", *CUMULATIVE.values(), *POINT_IN_TIME.values()]


def log_source(entry: dict) -> None:
    with LOCK:
        log = json.loads(SOURCE_PATH.read_text()) if SOURCE_PATH.exists() else []
        log.append(entry)
        SOURCE_PATH.write_text(json.dumps(log, indent=1))


def fetch_json(path: Path, url: str, keep: list[str] = KEEP) -> dict:
    """The API response cached at path (only the `keep` fields of each result), fetched first if absent."""
    if path.exists():
        return json.loads(path.read_text())
    key = os.environ.get("FEC_API_KEY", "DEMO_KEY")
    raw = json.loads(get(f"{url}&api_key={key}"))
    data = {"pagination": raw["pagination"], "results": [{k: r.get(k) for k in keep} for r in raw["results"]]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # a page is either complete or absent
    log_source({"file": str(path.relative_to(RAW_DIR)), "url": url + "&api_key=<redacted>",
                "fetched": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return data


def to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=KEEP)
    for column in [*CUMULATIVE.values(), *POINT_IN_TIME.values()]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in ["coverage_start_date", "coverage_end_date", "receipt_date"]:
        df[column] = pd.to_datetime(df[column], errors="coerce")
    # several "latest" versions can remain (e.g. an original and its amendment both unflagged): keep the last filed
    return df.sort_values(["receipt_date", "file_number"]).drop_duplicates(
        ["committee_id", "report_type", "coverage_end_date"], keep="last")


def get(url: str, retries: int = 8) -> bytes:
    """The response body. A rate limit (429) waits as long as the API's Retry-After says, as often as it takes (the
    key allows a fixed number of requests an hour); other server errors retry with backoff."""
    request = urllib.request.Request(url, headers={"User-Agent": "election26-fec"})
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code == 429:
                wait = int(error.headers.get("Retry-After") or 60) + 5
                print(f"rate limited; waiting {wait} s", flush=True)
                time.sleep(wait)
                continue
            attempt += 1
            if error.code not in (500, 502, 503, 504) or attempt >= retries:
                raise
            time.sleep(min(60, 2 ** attempt * 2))


def bulk_file(kind: str, year: int) -> pd.DataFrame:
    """An FEC bulk file for `year` (ccl, cn or cm), downloaded once into RAW_DIR."""
    yy = f"{year % 100:02d}"
    path = RAW_DIR / f"{kind}{yy}.zip"
    if not path.exists():
        url = BULK_URL.format(year=year, kind=kind, yy=yy)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(get(url))
        log_source({"file": path.name, "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "fetched": datetime.now(UTC).isoformat(timespec="seconds")})
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
        return pd.read_csv(io.BytesIO(z.read(name)), sep="|", header=None, dtype=str)


def candidate_committees(year: int) -> pd.DataFrame:
    """candidate_id, committee_id, rank: each House and Senate candidate's campaign committees for `year`, in the
    order they are tried (COMMITTEE_SOURCES)."""
    ccl = bulk_file("ccl", year).set_axis(CCL_COLUMNS, axis=1)
    ccl = ccl[ccl["FEC_ELECTION_YR"] == str(year)]
    cn = bulk_file("cn", year)
    cm = bulk_file("cm", year)
    sources = [
        cn[[0, 9]].set_axis(["candidate_id", "committee_id"], axis=1),                          # cn CAND_PCC
        ccl.loc[ccl["CMTE_DSGN"] == "P", ["CAND_ID", "CMTE_ID"]].set_axis(["candidate_id", "committee_id"], axis=1),
        ccl.loc[ccl["CMTE_DSGN"] == "A", ["CAND_ID", "CMTE_ID"]].set_axis(["candidate_id", "committee_id"], axis=1),
        cm.loc[cm[8].isin(["P", "A"]), [14, 0]].set_axis(["candidate_id", "committee_id"], axis=1),  # cm CAND_ID
    ]
    out = pd.concat([s.assign(rank=i) for i, s in enumerate(sources)], ignore_index=True).dropna()
    out = out[out["candidate_id"].str[0].isin(["H", "S"])]
    return out.sort_values("rank").drop_duplicates(["candidate_id", "committee_id"]).reset_index(drop=True)


def reports(report_type: str, cycle: int, report_year: int | None = None) -> pd.DataFrame:
    """Every House/Senate committee's latest report of this type and cycle, one cached JSON file per API page."""
    def page(n: int) -> dict:
        params = {"cycle": cycle, "report_type": report_type, "is_amended": "false", "per_page": 100, "page": n,
                  "sort": "committee_id", **({"report_year": report_year} if report_year else {})}
        name = f"{report_type}_{cycle}{'_' + str(report_year) if report_year else ''}_p{n}.json"
        return fetch_json(API_DIR / name, f"{API}?{urllib.parse.urlencode(params)}")

    first = page(1)
    with ThreadPoolExecutor(WORKERS) as pool:
        rest = list(pool.map(page, range(2, first["pagination"]["pages"] + 1)))
    return to_frame([r for data in [first, *rest] for r in data["results"]])


def bulk_cached(report_type: str, cycle: int, report_year: int) -> bool:
    """Whether every page of a bulk report listing is already cached."""
    first = API_DIR / f"{report_type}_{cycle}_{report_year}_p1.json"
    if not first.exists():
        return False
    pages = json.loads(first.read_text())["pagination"]["pages"]
    return all((API_DIR / f"{report_type}_{cycle}_{report_year}_p{n}.json").exists() for n in range(1, pages + 1))


def batched(report_type: str, year: int, committee_ids, name: str, batch: int = 100) -> list[dict]:
    """The latest `report_type` reports of `year` for these committees, 100 committee_ids per request (cached as
    api/{report_type}_{year}_{name}_b{n}_p{page}.json). Raises if the API ignored the committee filter."""
    ids = sorted(set(committee_ids))
    chunks = [ids[i:i + batch] for i in range(0, len(ids), batch)]

    def one(n: int) -> list[dict]:
        rows, page, pages = [], 1, 1
        while page <= pages:
            params = [("cycle", year), ("report_type", report_type), ("report_year", year), ("is_amended", "false"),
                      ("per_page", 100), ("page", page), *(("committee_id", c) for c in chunks[n])]
            data = fetch_json(API_DIR / f"{report_type}_{year}_{name}_b{n}_p{page}.json",
                              f"{API}?{urllib.parse.urlencode(params)}")
            if not {r["committee_id"] for r in data["results"]} <= set(chunks[n]):
                raise ValueError("the API ignored the committee_id filter")
            rows += data["results"]
            pages, page = data["pagination"]["pages"], page + 1
        return rows

    with ThreadPoolExecutor(WORKERS) as pool:
        return [r for rows in pool.map(one, range(len(chunks))) for r in rows]


def senate_year_end(committee_ids, year: int) -> pd.DataFrame:
    """The year-end (Dec 31, `year`) reports of these Senate committees: from the bulk pages where all are cached,
    else batched."""
    if bulk_cached("YE", year, year):
        return reports("YE", year, report_year=year)
    ids = sorted(set(committee_ids))
    return to_frame(batched("YE", year, ids, f"senate_{hashlib.sha256(','.join(ids).encode()).hexdigest()[:8]}"))


def committee_q3(committee_ids, year: int) -> pd.DataFrame:
    """The October quarterly (Q3 of `year`) of each committee: a committee already cached on its own
    (api/committee/{id}_Q3_{year}.json) is read from there, the rest are fetched batched."""
    ids = sorted(set(committee_ids))
    single = {c: API_DIR / "committee" / f"{c}_Q3_{year}.json" for c in ids}
    rows = [r for c, path in single.items() if path.exists() for r in json.loads(path.read_text())["results"]]
    rest = [c for c, path in single.items() if not path.exists()]
    if rest:
        rows += batched("Q3", year, rest, f"needed_{hashlib.sha256(','.join(rest).encode()).hexdigest()[:8]}")
    return to_frame(rows)


def needed_candidates() -> pd.DataFrame:
    """year, candidate_id, cycle_receipts of every nominee slot (receipts from the slot files' {slot}_cycle_receipts,
    where present) and top-two finalist: the candidates fallbacks are fetched for."""
    frames = []
    for path in NEEDED_PATHS:
        if not path.exists():
            print(f"(no {path.name}: its candidates get no fallback this run)")
            continue
        df = pd.read_csv(path, dtype=str, low_memory=False)
        for column in [c for c in df.columns if c.endswith("candidate_id")]:
            receipts = column.replace("candidate_id", "cycle_receipts")
            frames.append(pd.DataFrame({"year": df["year"], "candidate_id": df[column],
                                        "cycle_receipts": pd.to_numeric(df[receipts]) if receipts in df else float("inf")}))
    out = pd.concat(frames, ignore_index=True).dropna(subset=["candidate_id"])
    out = out.sort_values("cycle_receipts", ascending=False).drop_duplicates(["year", "candidate_id"])
    return out.assign(year=out["year"].astype(int))


def api_committees(candidate_ids, year: int) -> pd.DataFrame:
    """candidate_id, committee_id: the FEC API's principal and authorized committees of each candidate for `year`,
    one cached request per candidate."""
    def one(candidate_id: str) -> list[dict]:
        params = {"cycle": year, "per_page": 50}
        data = fetch_json(API_DIR / "candidate" / f"{candidate_id}_{year}.json",
                          f"{CANDIDATE_API.format(candidate_id=candidate_id)}?{urllib.parse.urlencode(params)}",
                          keep=["committee_id", "designation"])
        return [{"candidate_id": candidate_id, "committee_id": r["committee_id"]}
                for r in data["results"] if r["designation"] in ("P", "A")]

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = [r for found in pool.map(one, sorted(set(candidate_ids))) for r in found]
    return pd.DataFrame(rows, columns=["candidate_id", "committee_id"])


def pre_general(year: int, needed: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate of `year` with a pre-general (or, where needed, October quarterly) report: the first
    committee, in candidate_committees' order, that filed a 12G, else the first that filed a Q3."""
    committees = candidate_committees(year)
    g12 = reports("12G", year)
    g12 = g12[g12["coverage_end_date"].dt.year == year].sort_values("coverage_end_date").drop_duplicates(
        "committee_id", keep="last")
    need = needed[needed["year"] == year]
    covered = set(committees.loc[committees["committee_id"].isin(g12["committee_id"]), "candidate_id"])
    lookup = need.loc[(need["cycle_receipts"] >= LOOKUP_MIN_RECEIPTS) & ~need["candidate_id"].isin(covered),
                      "candidate_id"]
    found = api_committees(lookup, year).assign(rank=COMMITTEE_SOURCES.index("FEC API candidate committees"))
    committees = pd.concat([committees, found], ignore_index=True).drop_duplicates(["candidate_id", "committee_id"])
    has_12g = committees["committee_id"].isin(g12["committee_id"])
    without = ~committees["candidate_id"].isin(committees.loc[has_12g, "candidate_id"])
    wanted = committees["candidate_id"].isin(needed.loc[needed["year"] == year, "candidate_id"])
    q3 = committee_q3(committees.loc[wanted & without, "committee_id"], year)
    q3 = q3[q3["coverage_end_date"].dt.year == year].sort_values("coverage_end_date").drop_duplicates(
        "committee_id", keep="last")
    chosen = pd.concat([g12, q3], ignore_index=True).drop_duplicates("committee_id", keep="first")  # 12G first
    chosen = chosen[chosen[CUMULATIVE["receipts"]].notna()]  # a summary without cycle-to-date (paper-era) is unusable
    out = committees.merge(chosen, on="committee_id", how="inner")
    out["preference"] = (out["report_type"] != "12G").astype(int)
    out = out.sort_values(["candidate_id", "preference", "rank"]).drop_duplicates("candidate_id")
    out["committee_source"] = out["rank"].map(dict(enumerate(COMMITTEE_SOURCES)))
    senate_committees = out.loc[out["candidate_id"].str.startswith("S"), "committee_id"]
    prior = senate_year_end(senate_committees, year - 2)
    prior = prior[prior["coverage_end_date"] == pd.Timestamp(f"{year - 2}-12-31")].drop_duplicates(
        "committee_id", keep="last").set_index("committee_id")
    prior = prior[prior.index.isin(senate_committees)]  # the House amounts are cycle-to-date as filed
    for field, column in CUMULATIVE.items():
        before = out["committee_id"].map(prior[column]).fillna(0.0)
        out[f"{field}_cycle_to_date"] = out[column]
        out[f"{field}_before_cycle"] = before
        out[field] = out[column] - before
    for field, column in POINT_IN_TIME.items():
        out[field] = out[column]
    out["prior_year_end_found"] = out["committee_id"].isin(prior.index)
    return out.assign(year=year)[["year", "candidate_id", "committee_id", "committee_source", "report_type",
                                  "coverage_end_date", "prior_year_end_found", *CUMULATIVE, *POINT_IN_TIME,
                                  *(f"{f}_cycle_to_date" for f in CUMULATIVE),
                                  *(f"{f}_before_cycle" for f in CUMULATIVE)]].reset_index(drop=True)


def main() -> None:
    needed = needed_candidates()
    frames = []
    for year in YEARS:
        frame = pre_general(year, needed)
        frames.append(frame)
        print(f"{year}: {len(frame)} candidates; " + ", ".join(
            f"{t} {n}" for t, n in frame["report_type"].value_counts().items())
            + f"; prior year-end subtracted for {int(frame['prior_year_end_found'].sum())}", flush=True)
    out = pd.concat(frames, ignore_index=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"-> {OUTPUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
