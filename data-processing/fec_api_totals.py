"""Fetch FEC API cycle totals for fec_finance.API_CORRECTIONS (nominees missing from both FEC bulk files).

  data/raw/fec_candidate_summaries/api/{candidate_id}_{year}.json  each API response, unmodified
  data/raw/fec_candidate_summaries/api/api_totals.csv              year, candidate_id and fec_finance.FINANCE fields
  data/raw/fec_candidate_summaries/api/SOURCE.json                 each used request: URL (key redacted), time, sha256

One request per correction: https://api.open.fec.gov/v1/candidate/{id}/totals/?cycle=YEAR&election_full=false, the
two-year cycle, as in the bulk candidate summary file (checked on Donalds 2022, which is in both: every field agrees).
Only responses with data are kept: each is cached and logged in SOURCE.json, and a rerun reuses it instead of asking
again. A failed request (e.g. DEMO_KEY's rate limit) or one with no result is printed, not kept, and writes no row, so
fec_finance.py leaves that slot as no_fec_record; a rerun tries it again. The key is FEC_API_KEY if set, else FEC's
public DEMO_KEY; requests carry no personal information.
"""

import hashlib
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import pandas as pd

import fec_finance

API_DIR = fec_finance.API_TOTALS_PATH.parent
URL = "https://api.open.fec.gov/v1/candidate/{candidate_id}/totals/"
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
    query = {"cycle": year, "election_full": "false"}
    url = URL.format(candidate_id=candidate_id) + "?" + urllib.parse.urlencode(query)
    path = API_DIR / f"{candidate_id}_{year}.json"
    log = {"candidate_id": candidate_id, "cycle": year, "url": url + "&api_key=<redacted>", "file": path.name}
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


def main() -> None:
    key = os.environ.get("FEC_API_KEY", "DEMO_KEY")
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
