"""Fetch FEC API cycle totals for fec_finance.API_CORRECTIONS (nominees missing from both FEC bulk files).

  data/raw/fec_candidate_summaries/api/{candidate_id}_{year}.json  each API response, unmodified
  data/raw/fec_candidate_summaries/api/api_totals.csv              year, candidate_id and fec_finance.FINANCE fields
  data/raw/fec_candidate_summaries/api/SOURCE.json                 each used request: URL (key redacted), time, sha256

One request per correction: https://api.open.fec.gov/v1/candidate/{id}/totals/?cycle=YEAR&election_full=false, the
two-year cycle, as in the bulk candidate summary file (checked on Donalds 2022, which is in both: every field agrees).
A candidate in fec_finance.API_COMMITTEE_TOTALS gets its committee's two-year totals instead
(/v1/committee/{committee_id}/totals/?cycle=YEAR; the same fields), cached as {committee_id}_{year}.json.
Only responses with data are kept: each is cached and logged in SOURCE.json, and a rerun reuses it instead of asking
again. A failed request (e.g. DEMO_KEY's rate limit) or one with no result is printed, not kept, and writes no row, so
fec_finance.py leaves that slot as no_fec_record; a rerun tries it again. The key is FEC_API_KEY if set, else FEC's
public DEMO_KEY; requests carry no personal information.

  python data-processing/fec_api_totals.py --review

prints, for every slot fec_finance.suspect_records flags in data/interim/{house,senate}_fec_raw.csv, the nominee's FEC
committees with their totals for the cycle (or, with no FEC record, the API's candidates of that name, state and
office), for deciding between API_COMMITTEE_TOTALS and REVIEWED_LOW_FUNDING. It writes nothing.
"""

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

import fec_finance

API_DIR = fec_finance.API_TOTALS_PATH.parent
URL = "https://api.open.fec.gov/v1/candidate/{candidate_id}/totals/"
COMMITTEE_URL = "https://api.open.fec.gov/v1/committee/{committee_id}/totals/"
USER_AGENT = "Election26-research/0.1"
# fec_finance.FINANCE field -> API totals field
API_FIELDS = {
    "receipts": "receipts",
    "disbursements": "disbursements",
    "individual_contributions": "individual_contributions",
    "individual_unitemized_contributions": "individual_unitemized_contributions",
    "individual_itemized_contributions": "individual_itemized_contributions",
    "cash_on_hand": "last_cash_on_hand_end_period",
    "debt": "last_debts_owed_by_committee",
    "coverage_start_date": "coverage_start_date",
    "coverage_end_date": "coverage_end_date",
}


def fetch(candidate_id: str, year: int, key: str) -> tuple[dict | None, dict]:
    """(the one totals result, or None, and the request's log entry), from the cached response if present. Only a
    response with exactly one result is cached."""
    committee_id = fec_finance.API_COMMITTEE_TOTALS.get(candidate_id)
    if committee_id is None:
        url = URL.format(candidate_id=candidate_id) + "?" + urllib.parse.urlencode({"cycle": year, "election_full": "false"})
    else:
        url = COMMITTEE_URL.format(committee_id=committee_id) + "?" + urllib.parse.urlencode({"cycle": year})
    path = API_DIR / f"{committee_id or candidate_id}_{year}.json"
    log = {"candidate_id": candidate_id, "committee_id": committee_id, "cycle": year,
           "url": url + "&api_key=<redacted>", "file": path.name}
    if path.exists():
        raw = path.read_bytes()
    else:
        request = urllib.request.Request(url + "&" + urllib.parse.urlencode({"api_key": key}),
                                         headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                raw = resp.read()
        except Exception as error:  # reported, and the slot keeps its no_fec_record status
            return None, {**log, "status": f"failed: {error}"}
        results = json.loads(raw)["results"]
        if len(results) != 1:
            return None, {**log, "status": f"{len(results)} results, expected 1"}
        API_DIR.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        log["retrieved_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log["sha256"] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)["results"][0], {**log, "status": "ok"}


def api_get(path: str, query: dict, key: str, retries: int = 10) -> list[dict]:
    """The results of one FEC API GET (no caching); [] for a 404, which the API gives for a committee without totals.
    A 429 (rate limit) waits a minute and retries."""
    url = f"https://api.open.fec.gov/v1/{path}?" + urllib.parse.urlencode({**query, "api_key": key})
    for _ in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}),
                                        timeout=60) as r:
                return json.loads(r.read())["results"]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return []
            if error.code != 429:
                raise
            time.sleep(60)
    raise RuntimeError(f"FEC API still rate-limited after {retries} tries: {path}")


def committee_lines(candidate_id: str, cycle: int, key: str) -> list[str]:
    """One line per committee of the candidate with totals for the cycle: ID, name, designation, receipts, coverage."""
    lines = []
    for committee in api_get(f"candidate/{candidate_id}/committees/", {"per_page": 50}, key):
        for t in api_get(f"committee/{committee['committee_id']}/totals/", {"cycle": cycle}, key):
            lines.append(f"    {committee['committee_id']} {committee['name']} ({t.get('committee_designation')}): "
                         f"${t['receipts']:,.0f}, {t['coverage_start_date'][:10]} to {t['coverage_end_date'][:10]}")
    return lines or ["    no committee with totals for the cycle"]


def review(key: str) -> None:
    """Print the FEC API's committees (or same-name candidates) of every fec_finance.suspect_records slot."""
    for chamber, seat, results_path in [("house", "district", fec_finance.TARGETS_RAW_PATH),
                                        ("senate", "class", fec_finance.SENATE_TARGETS_RAW_PATH)]:
        interim = pd.read_csv(fec_finance.INTERIM_DIR / f"{chamber}_fec_raw.csv")
        suspects = fec_finance.suspect_records(interim, pd.read_csv(results_path), chamber,
                                               ["year", "state", seat])
        for r in suspects.to_dict("records"):
            print(f"{chamber} {r['year']} {r['state']}-{r[seat]} {r['party']} {r['name']} ({r['match_status']}, "
                  f"{r['candidate_id']}, ${r['receipts']:,.0f}; winner {r['winner_party']}, "
                  f"margin {r['margin_2p_norm']:+.1f})")
            if isinstance(r["candidate_id"], str):
                print("\n".join(committee_lines(r["candidate_id"], r["year"], key)))
            # other FEC candidates of the surname, state and office in the cycle: the matched ID may be a stale one
            office = "H" if chamber == "house" else "S"
            surname = str(r["name"]).replace(",", " ").split()[0 if "," in str(r["name"]) else -1]
            found = api_get("candidates/search/", {"q": surname, "state": r["state"], "office": office,
                                                   "cycle": r["year"], "per_page": 20}, key)
            for c in found:
                if c["candidate_id"] != r["candidate_id"]:
                    print(f"  other candidate {c['candidate_id']} {c['name']} ({c.get('party')}, "
                          f"district {c.get('district')})")
                    print("\n".join(committee_lines(c["candidate_id"], r["year"], key)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--review", action="store_true", help="print the FEC committees of suspect_records slots")
    args = parser.parse_args()
    key = os.environ.get("FEC_API_KEY", "DEMO_KEY")
    if args.review:
        review(key)
        return
    previous = {}
    if (API_DIR / "SOURCE.json").exists():  # keep each cached response's original retrieval time
        previous = {e["file"]: e for e in json.loads((API_DIR / "SOURCE.json").read_text())["requests"]}
    rows, requests = [], []
    for (chamber, year, state, seat, party), candidate_id in fec_finance.API_CORRECTIONS.items():
        result, log = fetch(candidate_id, year, key)
        log = {"slot": f"{chamber} {year} {state}-{seat} {party}", **log}
        if "retrieved_utc" not in log and log["file"] in previous:
            log["retrieved_utc"] = previous[log["file"]].get("retrieved_utc")
        print(f"{log['slot']} {candidate_id}: {log['status']}")
        if result is not None:
            requests.append(log)
        if result is not None:
            row = {"year": year, "candidate_id": candidate_id}
            for field, api_field in API_FIELDS.items():
                value = result[api_field]
                row[field] = value[:10] if field in fec_finance.DATES and value else value
            rows.append(row)
    pd.DataFrame(rows, columns=["year", "candidate_id", *API_FIELDS]).to_csv(fec_finance.API_TOTALS_PATH, index=False)
    (API_DIR / "SOURCE.json").write_text(json.dumps({
        "endpoint": "https://api.open.fec.gov/developers/#/candidate/get_candidate__candidate_id__totals_",
        "built_by": "data-processing/fec_api_totals.py", "requests": requests}, indent=2) + "\n")
    print(f"-> {fec_finance.API_TOTALS_PATH}: {len(rows)} of {len(fec_finance.API_CORRECTIONS)} corrections")


if __name__ == "__main__":
    main()
