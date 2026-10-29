import json
import urllib.parse
from pathlib import Path

import pytest

import acs

COUNTS = json.loads((Path(__file__).parents[1] / "data/reference/district_counts.json").read_text())
USPS_TO_FIPS = {v: k for k, v in acs.STATE_FIPS.items()}

# Synthetic per-district estimates chosen so every derived field has an exact, hand-checkable value. Collapsed C cells
# are the sums of their B cells, as in real ACS data, so 1-year (C) and 5-year (B) vintages give identical fields.
VALUES = {
    **{f"{t}03002_{n:03d}E": x for t in "BC" for n, x in ((1, 700_000), (3, 420_000), (4, 91_000), (12, 140_000))},
    "B15002_001E": 480_000, **{f"B15002_{n:03d}E": 20_000 for n in (15, 16, 17, 18, 32, 33, 34, 35)},
    "C15002_001E": 480_000, "C15002_008E": 20_000, "C15002_009E": 60_000, "C15002_016E": 20_000, "C15002_017E": 60_000,
    "B19013_001E": 65_432, "B01002_001E": 38.4, "B07003_001E": 690_000, "B07003_004E": 600_000,
    "B05002_001E": 700_000, "B05002_013E": 70_000, "C05002_001E": 700_000, "C05002_008E": 70_000,
    "B23025_003E": 350_000, "B23025_005E": 17_500,
    **{f"{t}17001_{n:03d}E": x for t in "BC" for n, x in ((1, 680_000), (2, 81_600))},
    "B25010_001E": 2.55,
}
# C23001 cells run male then female, each as 5 under-65 age groups followed by 2 aged-65+ groups
for i, (lf, unemployed) in enumerate(zip(acs.UNEMPLOYMENT_C23001.den, acs.UNEMPLOYMENT_C23001.num)):
    under_65 = i % 7 < 5
    VALUES[lf], VALUES[unemployed] = (30_000, 1_500) if under_65 else (7_500, 375)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def district_rows(year: int) -> list[dict]:
    """One synthetic API row per district, plus the DC, Puerto Rico, and unassigned (ZZ) rows Census returns."""
    congress = ordinal(acs.BOUNDARY_CONGRESS[year])
    rows = []
    for state, n in COUNTS[str(year)].items():
        for d in [0] if n == 1 else range(1, n + 1):
            name = "(at Large)" if d == 0 else str(d)
            rows.append({"NAME": f"Congressional District {name} ({congress} Congress), {state}",
                         "state": USPS_TO_FIPS[state], "congressional district": f"{d:02d}"})
    rows.append({"NAME": f"Delegate District (at Large) ({congress} Congress), District of Columbia",
                 "state": "11", "congressional district": "98"})
    rows.append({"NAME": f"Resident Commissioner District (at Large) ({congress} Congress), Puerto Rico",
                 "state": "72", "congressional district": "98"})
    rows.append({"NAME": f"Congressional Districts not defined ({congress} Congress), California",
                 "state": "06", "congressional district": "ZZ"})
    for row in rows:
        row.update({v: str(x) for v, x in VALUES.items()})
    return rows


def reference_rows() -> dict[str, list[dict]]:
    """Synthetic national and state rows (including DC and Puerto Rico), defaulting to the district VALUES."""
    values = {v: str(x) for v, x in VALUES.items()}
    states = [{"NAME": usps, "state": fips, **values} for fips, usps in [*acs.STATE_FIPS.items(), ("11", "DC"), ("72", "PR")]]
    return {"us:1": [{"NAME": "United States", "us": "1", **values}], "state:*": states}


@pytest.fixture
def fake_api(monkeypatch):
    """Route acs._get_json to synthetic metadata and data; load(year) returns the mutable state dict."""
    monkeypatch.setenv("CENSUS_API", "test-key")
    state = {"rows": None, "labels": {}, **reference_rows()}

    def get_json(url):
        parsed = urllib.parse.urlparse(url)
        if "/groups/" in parsed.path:
            acs_year = int(parsed.path.split("/")[2])
            table = parsed.path.rsplit("/", 1)[-1].removesuffix(".json")
            variables = {
                v: {"label": "Estimate!!" + state["labels"].get(v, label.format(acs_year=acs_year)),
                    "concept": acs.TABLE_CONCEPTS[table]}
                for v, label in acs.EXPECTED_LABELS.items() if v.startswith(table + "_")
            }
            return {"variables": variables}
        query = urllib.parse.parse_qs(parsed.query)
        geo = query["for"][0]
        rows, geo_cols = {"us:1": (state["us:1"], ["us"]), "state:*": (state["state:*"], ["state"])}.get(
            geo, (state["rows"], ["state", "congressional district"]))
        cols = [*query["get"][0].split(","), *geo_cols]
        return [cols] + [[row[c] for c in cols] for row in rows]

    def load(year):
        state["rows"] = district_rows(year)
        return state

    monkeypatch.setattr(acs, "_get_json", get_json)
    return load
