"""Build the 2026 general-election candidate lists for the House and Senate from Wikipedia, at pinned revisions.

  data/raw/nominees_2026/house.csv   year, state, district, party, party_label, name, source
  data/raw/nominees_2026/senate.csv  year, state, class, special, party, party_label, name, source

The repo has no other 2026 nominee source (house_incumbency.csv and senate_2026.csv name only the running incumbents),
and every 2026 primary is over. Wikipedia's "2026 United States House of Representatives elections" and "2026 United
States Senate elections" list each race's general-election candidates, as "Name (Party)". Each House state table cites
its official source, usually the state's certified candidate list, and that URL is kept in `source`.

Sourcing: each page is read at the revision pinned in PAGES. The API response is cached unmodified in
data/raw/nominees_2026/, so a rerun parses the same text offline. SOURCE.json records each title, revision, timestamp,
URL and sha256. Moving to a newer revision means changing PAGES and reviewing the report again.

party is D (Democratic, DFL, Democratic-NPL), R (Republican), or other. Every listed candidate is kept, including third
parties and several candidates of one party (top-two, top-four). fec_finance.py decides who fills a party's slot.

Checks: the House rows must cover exactly 2026's 435 districts (data/reference/district_counts.json) and the Senate
rows exactly the races in data/raw/incumbency_2026/senate_2026.csv. Every race needs at least one candidate. These are
reported, not blocking: running incumbents (house_incumbency.csv, Cook's names) missing from the candidate list, and
races with more than one candidate of a major party.
"""

import hashlib
import json
import re
import sys
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "house"))
import acs
import incumbency

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/raw/nominees_2026"
HOUSE_INCUMBENCY_PATH = ROOT / "data/interim/house/house_incumbency.csv"
API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "Election26-research/0.1 (nominee list for a forecasting model)"
YEAR = 2026
# chamber -> (page title, pinned revision id), both retrieved 2026-10-30
PAGES = {
    "house": ("2026 United States House of Representatives elections", 1382821364),
    "senate": ("2026 United States Senate elections", 1382860314),
}
DEM_LABELS = {"Democratic", "DFL", "Democratic-NPL"}
GOP_LABELS = {"Republican"}
STATE_CODES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
    "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
    "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
HOUSE_COLUMNS = ["year", "state", "district", "party", "party_label", "name", "source"]
SENATE_COLUMNS = ["year", "state", "class", "special", "party", "party_label", "name", "source"]


def fetch_wikitext(chamber: str, directory: Path = RAW_DIR) -> tuple[str, dict]:
    """(wikitext, provenance) of PAGES[chamber] at its pinned revision, from the cached API response if present."""
    title, revid = PAGES[chamber]
    path = directory / f"{chamber}_wikipedia_rev{revid}.json"
    url = API_URL + "?" + urllib.parse.urlencode({
        "action": "query", "prop": "revisions", "revids": revid, "rvprop": "ids|timestamp|content",
        "rvslots": "main", "format": "json", "formatversion": 2})
    if not path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=120) as resp:
            raw = resp.read()
        directory.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    raw = path.read_bytes()
    page = json.loads(raw)["query"]["pages"][0]
    revision = page["revisions"][0]
    if page["title"] != title or revision["revid"] != revid:
        raise ValueError(f"{path}: expected {title!r} revision {revid}, got {page['title']!r} {revision['revid']}")
    provenance = {"title": title, "revid": revid, "revision_timestamp": revision["timestamp"], "api_url": url,
                  "page_url": f"https://en.wikipedia.org/w/index.php?oldid={revid}", "file": path.name,
                  "sha256": hashlib.sha256(raw).hexdigest()}
    return revision["slots"]["main"]["content"], provenance


def clean(text: str) -> str:
    """Wikitext as plain text: references, bold, templates and HTML entities dropped, links replaced by their label."""
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.S)
    text = text.replace("&nbsp;", " ").replace("'''", "")
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", text)
    while re.search(r"\{\{[^{}]*\}\}", text):
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return " ".join(text.split())


def party_code(label: str) -> str:
    return "D" if label in DEM_LABELS else "R" if label in GOP_LABELS else "other"


def candidates_in(block: str) -> list[tuple[str, str]]:
    """(name, party label) for each "*{{Party stripe|...}}Name (Party)" item in one race's wikitext; raises unless
    every such item is parsed."""
    block = re.sub(r"\*\s*(\{\{[Pp]arty stripe)", r"\n*\1", block)  # an item can follow "{{Plainlist |" on its line
    out = []
    for line in block.splitlines():
        if line.startswith("*") and "party stripe" in line.lower():
            match = re.fullmatch(r"\*\s*(.+?)\s*\(([^()]+)\)", clean(line))
            if not match:
                raise ValueError(f"Unparsed candidate line: {line!r}")
            out.append((match.group(1), match.group(2)))
    items = len(re.findall(r"^\*\s*\{\{[Pp]arty stripe", block, re.M))  # a fusion candidate has two stripes
    if len(out) != items:
        raise ValueError(f"Parsed {len(out)} of {items} candidate items:\n{block[:500]}")
    return out


def cited_url(section: str, wikitext: str) -> str:
    """The URL cited on a House state table's Candidates header, following a reused named reference."""
    match = re.search(r"Candidates\s*<ref([^>]*?)(/?)>(.*?)(?:</ref>|$)", section, re.S | re.M)
    if not match:
        return ""
    if match.group(2) == "/":  # <ref name=X/>: find where X is defined
        name = re.search(r"name\s*=\s*\"?([^\"/>]+)", match.group(1)).group(1).strip()
        definition = re.search(rf"<ref name\s*=\s*\"?{re.escape(name)}\"?\s*>(.*?)</ref>", wikitext, re.S)
        body = definition.group(1) if definition else ""
    else:
        body = match.group(3)
    url = re.search(r"url\s*=\s*([^|}\s]+)", body)
    return url.group(1) if url else ""


def parse_house(wikitext: str, revid: int) -> pd.DataFrame:
    """One row per candidate in the per-state tables ("==Alabama==" through the territories)."""
    body = wikitext[wikitext.index("==Alabama=="):wikitext.index("==See also==")]
    rows = []
    for section in re.split(r"(?m)^(?===[^=])", body):
        url = cited_url(section, wikitext)
        markers = list(re.finditer(r"\{\{ushr\|([A-Z]{2})\|(\w+)\|X\}\}", section))
        for marker, following in zip(markers, markers[1:] + [None]):
            state, district = marker.group(1), marker.group(2)
            block = section[marker.end():following.start() if following else len(section)]
            source = f"Wikipedia rev {revid}" + (f"; cites {url}" if url else "")
            for name, label in candidates_in(block):
                rows.append({"year": YEAR, "state": state, "district": 0 if district == "AL" else int(district),
                             "party": party_code(label), "party_label": label, "name": name, "source": source})
    out = pd.DataFrame(rows, columns=HOUSE_COLUMNS)
    return out[out["state"].isin(acs.STATE_FIPS.values())].reset_index(drop=True)  # drops DC and the territories


def parse_senate(wikitext: str, revid: int, races: pd.DataFrame) -> pd.DataFrame:
    """One row per candidate in the special-election table and the regular "Elections leading to the next Congress"
    table; class and special come from senate_2026.csv's row for that state and kind of race."""
    rows = []
    for heading, special in [("=== Special elections during the preceding Congress ===", 1),
                             ("=== Elections leading to the next Congress ===", 0)]:
        start = wikitext.index(heading)
        table = wikitext[start:wikitext.index("\n|}", start)]
        for block in re.split(r"\n\|-", table):
            header = re.search(r"^!\s*\[\[2026 United States Senate (?:special )?election in ([^|\]]+)", block, re.M)
            if not header:
                continue
            state = STATE_CODES[header.group(1)]
            race = races[(races["state"] == state) & (races["special"] == special)]
            if len(race) != 1:
                raise ValueError(f"Senate {state} special={special}: {len(race)} rows in senate_2026.csv, expected 1")
            for name, label in candidates_in(block):
                rows.append({"year": YEAR, "state": state, "class": int(race["class"].item()), "special": special,
                             "party": party_code(label), "party_label": label, "name": name,
                             "source": f"Wikipedia rev {revid}"})
    return pd.DataFrame(rows, columns=SENATE_COLUMNS)


def check(house: pd.DataFrame, senate: pd.DataFrame, races: pd.DataFrame, counts_by_year: dict) -> None:
    """Raise unless the House rows cover exactly 2026's districts and the Senate rows exactly senate_2026.csv's races."""
    districts = house.drop_duplicates(["state", "district"])
    incumbency.check_districts(districts, YEAR, acs.normalize_counts(counts_by_year, YEAR))
    found = set(zip(senate["state"], senate["class"], senate["special"]))
    expected = set(zip(races["state"], races["class"], races["special"]))
    if found != expected:
        raise ValueError(f"Senate races differ from senate_2026.csv: missing {sorted(expected - found)}, "
                         f"extra {sorted(found - expected)}")


def surname(name: str) -> str:
    """Last word of a "First ... Last" name, accents and a Jr./Sr./roman-numeral suffix dropped, upper case."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    words = [w for w in re.sub(r"[^A-Za-z\- ]", " ", ascii_name).upper().split()
             if w not in {"JR", "SR", "II", "III", "IV"}]
    return words[-1] if words else ""


def report(house: pd.DataFrame, senate: pd.DataFrame, races: pd.DataFrame) -> str:
    lines = [f"House: {len(house)} candidates in {len(house.drop_duplicates(['state', 'district']))} districts; "
             f"Senate: {len(senate)} candidates in {len(senate.drop_duplicates(['state', 'class', 'special']))} races"]
    for name, df, key in [("House", house, ["state", "district"]), ("Senate", senate, ["state", "class", "special"])]:
        major = df[df["party"].isin(["D", "R"])]
        n = major.groupby(key)["party"].value_counts().unstack(fill_value=0).reindex(columns=["D", "R"], fill_value=0)
        n = n.reindex(pd.MultiIndex.from_frame(df[key].drop_duplicates()), fill_value=0)
        lines.append(f"{name} races by D/R candidate count: " + ", ".join(
            f"{d}D-{r}R {k}" for (d, r), k in n.value_counts().sort_index().items()))
        several = n[(n["D"] > 1) | (n["R"] > 1)]
        lines.append(f"{name} races with more than one candidate of a major party ({len(several)}): "
                     + ", ".join("-".join(map(str, k)) for k in several.index))
        lacking = n[(n["D"] == 0) | (n["R"] == 0)]
        lines.append(f"{name} races without a D or an R candidate ({len(lacking)}): " + ", ".join(
            f"{'-'.join(map(str, k))} ({'no D' if d == 0 else ''}{' and ' if d == 0 and r == 0 else ''}"
            f"{'no R' if r == 0 else ''})" for k, (d, r) in lacking[["D", "R"]].iterrows()))

    # running incumbents (Cook's 2026 list, via house_incumbency.csv) who aren't listed as candidates
    inc = pd.read_csv(HOUSE_INCUMBENCY_PATH).query("year == @YEAR and inc == 1")
    listed = house.assign(surname=house["name"].map(surname)).groupby(["state", "district"])["surname"].agg(set)
    missing = [f"{r.state}-{r.district} {r.incumbent_names}" for r in inc.itertuples()
               if surname(r.incumbent_names) not in listed.get((r.state, r.district), set())]
    lines.append(f"House incumbents Cook lists as running (house_incumbency.csv inc = 1) whose surname isn't among their district's candidates ({len(missing)}): "
                 + "; ".join(missing))
    open_seats = races[races["inc"] == 0]
    lines.append("Senate open seats per senate_2026.csv: " + ", ".join(open_seats["state"]))
    return "\n".join(lines)


def main() -> None:
    counts_by_year = json.loads((ROOT / "data/reference/district_counts.json").read_text())
    races = incumbency.read_senate_2026()
    house_text, house_source = fetch_wikitext("house")
    senate_text, senate_source = fetch_wikitext("senate")
    house = parse_house(house_text, house_source["revid"])
    senate = parse_senate(senate_text, senate_source["revid"], races)
    check(house, senate, races, counts_by_year)
    print(report(house, senate, races))
    acs.write_csv_atomic(house, RAW_DIR / "house.csv")
    acs.write_csv_atomic(senate, RAW_DIR / "senate.csv")
    source = {"house": house_source, "senate": senate_source,
              "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "built_by": "data-processing/nominees_2026.py"}
    (RAW_DIR / "SOURCE.json").write_text(json.dumps(source, indent=2) + "\n")
    print(f"-> {RAW_DIR / 'house.csv'}, {RAW_DIR / 'senate.csv'}, {RAW_DIR / 'SOURCE.json'}")


if __name__ == "__main__":
    main()
