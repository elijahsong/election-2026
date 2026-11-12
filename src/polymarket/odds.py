"""Polymarket prices for every race in the congress-track submission, from the public Gamma API (no account, no key).

    uv run python src/polymarket/odds.py      pull live events, cache them in data/raw/polymarket/, print coverage

Source: GET https://gamma-api.polymarket.com/events?tag_slug=midterms&closed=false, paged by offset (fetch_events).
Each pull is cached as data/raw/polymarket/events_<UTC stamp>.json, holding only the race-winner events; load_events
reads the latest snapshot and calls the API only with refresh=True.

Events (race_id_from_title), by title:
    "<ST>-<DD> House Election Winner"                 one market per party   -> "<ST>-<DD>" ("<ST>-AL" at large)
    "<ST>-<DD> House Election Winner (Individual)"    one market per person  (California's same-party top-two races;
    "... (by individual)"                                                       also "(by individual)")
    "<State> Senate Election Winner"                  one market per party   -> "<ST>-SEN"
Other titles (margins, primaries, counties, "First Round Outright Winner ...") are ignored.

Markets: only active, open ones (each event also lists inactive placeholders: "Other", "A", "Person A", ...).
- party: the "(D)" / "(R)" / "(I)" suffix of groupItemTitle ("Mike Lawler (R)"), else the question's wording
  ("Will the Democratic Party win ...").
- candidate: groupItemTitle without the suffix.
- price: the midpoint of bestBid and bestAsk when both are quoted (0 < bid <= ask < 1), else lastTradePrice, else
  the YES outcomePrice. polymarket_prob is the price over the sum of its event's prices, so a race sums to 1, as in
  kalshi/odds.py.

Join (track_odds): the submission's predicted candidate gets polymarket_odd, Polymarket's probability for them:
- a same-party top-two race: the individual event's market of the same last name;
- otherwise the party event's market of the predicted party, else the same last name (Kiley, CA-6, whom the submission
  codes I);
- no market: blank, with polymarket_note saying why.
"""

import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
API = "https://gamma-api.polymarket.com"
TAG = "midterms"
RAW_DIR = ROOT / "data/raw/polymarket"

STATE_ABBR = {
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
HOUSE_TITLE = re.compile(r"(?P<st>[A-Z]{2})-(?P<district>\d{2}|AL) House Election Winner"
                         r"(?P<individual> \((?:by )?individual\))?", re.IGNORECASE)
SENATE_TITLE = re.compile(r"(?P<state>[A-Z][a-z]+(?: [A-Z][a-z]+)*) Senate Election Winner")
SUFFIX = re.compile(r"\s*\((D|R|I)\)\s*$")
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
TITLES = {"sen", "rep", "gov", "dr", "mr", "mrs", "ms"}


def _get_json(url: str, retries: int = 5):
    request = urllib.request.Request(url, headers={"User-Agent": "election26-polymarket-odds"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:  # rate limit or a transient server error: back off and retry
            if error.code not in (429, 500, 502, 503) or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def race_id_from_title(title: str) -> tuple[str, bool] | None:
    """(race_id, individual) for a race-winner event title, else None."""
    title = title.strip()
    house = HOUSE_TITLE.fullmatch(title)
    if house:
        district = "AL" if house["district"].upper() == "AL" else str(int(house["district"]))
        return f"{house['st'].upper()}-{district.zfill(2) if district != 'AL' else 'AL'}", bool(house["individual"])
    senate = SENATE_TITLE.fullmatch(title)
    if senate and senate["state"] in STATE_ABBR:
        return f"{STATE_ABBR[senate['state']]}-SEN", False
    return None


def fetch_events(api: str = API, tag: str = TAG) -> list[dict]:
    """Every open event with the tag, then only the race-winner ones."""
    events, offset = [], 0
    while True:
        page = _get_json(f"{api}/events?tag_slug={tag}&closed=false&limit=100&offset={offset}")
        events += page
        if len(page) < 100:
            return [e for e in events if race_id_from_title(e["title"])]
        offset += 100


def latest_snapshot(raw_dir: Path = RAW_DIR) -> Path | None:
    snapshots = sorted(raw_dir.glob("events_*.json"))
    return snapshots[-1] if snapshots else None


def load_events(refresh: bool = False, raw_dir: Path = RAW_DIR) -> tuple[list[dict], str]:
    """(events, as_of): the latest cached snapshot, or with refresh a live pull, cached first."""
    if not refresh:
        path = latest_snapshot(raw_dir)
        if path is None:
            raise FileNotFoundError(f"No cached Polymarket snapshot in {raw_dir}; pull one with refresh=True")
        return json.loads(path.read_text()), path.stem.removeprefix("events_")
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%MZ")
    events = fetch_events()
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"events_{stamp}.json").write_text(json.dumps(events))
    return events, stamp


def market_party(market: dict) -> str | None:
    suffix = SUFFIX.search(market.get("groupItemTitle") or "")
    if suffix:
        return suffix.group(1)
    question = (market.get("question") or "").lower()
    for word, party in [("democrat", "D"), ("republican", "R"), ("independent", "I")]:
        if word in question:
            return party
    return None


def market_price(market: dict) -> float:
    """Midpoint of bid and ask when both are quoted, else the last trade, else the YES outcome price."""
    bid, ask = market.get("bestBid"), market.get("bestAsk")
    if bid is not None and ask is not None and 0 < float(bid) <= float(ask) < 1:
        return (float(bid) + float(ask)) / 2
    if market.get("lastTradePrice"):
        return float(market["lastTradePrice"])
    prices = market.get("outcomePrices")
    return float(json.loads(prices)[0]) if prices else np.nan


def race_markets(events: list[dict]) -> pd.DataFrame:
    """One row per active market of a race-winner event: race_id, individual, event_slug, candidate, party, bid, ask,
    last, price, polymarket_prob (price over its event's total)."""
    rows = []
    for event in events:
        matched = race_id_from_title(event["title"])
        if not matched:
            continue
        race_id, individual = matched
        for m in event.get("markets", []):
            if not m.get("active") or m.get("closed"):
                continue
            rows.append({"race_id": race_id, "individual": individual, "event_slug": event["slug"],
                         "candidate": SUFFIX.sub("", m.get("groupItemTitle") or "").strip(),
                         "party": market_party(m), "question": m.get("question"), "bid": m.get("bestBid"),
                         "ask": m.get("bestAsk"), "last": m.get("lastTradePrice"), "price": market_price(m)})
    out = pd.DataFrame(rows)
    duplicated = out.drop_duplicates(["race_id", "individual", "event_slug"]).duplicated(["race_id", "individual"])
    if duplicated.any():
        raise ValueError("More than one open winner event for a race: "
                         f"{out.drop_duplicates(['race_id', 'individual', 'event_slug'])[duplicated.to_numpy()]}")
    out["polymarket_prob"] = out["price"] / out.groupby(["race_id", "individual"])["price"].transform("sum")
    return out


def name_words(name: str) -> list[str]:
    """Lower-case ASCII words of a name without titles (Sen.) and suffixes (Jr.)."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return [w for w in re.sub(r"[^a-z\- ]", " ", text).split() if w not in NAME_SUFFIXES | TITLES]


def last_name(name: str) -> str:
    words = name_words(name)
    return words[-1] if words else ""


def by_name(pool: pd.DataFrame, name: str) -> pd.DataFrame:
    """pool's markets for the candidate: the same last name, and if that's ambiguous (two Sullivans on Alaska's Senate
    ballot) the same first name too."""
    match = pool[pool["candidate"].map(last_name) == last_name(name)]
    if len(match) > 1:
        first = (name_words(name) or [""])[0]
        match = match[match["candidate"].map(lambda c: (name_words(c) or [""])[0]) == first]
    return match


def track_odds(submission: pd.DataFrame, markets: pd.DataFrame, same_party: set[str]) -> pd.DataFrame:
    """race_id, polymarket_odd (Polymarket's probability for the submitted candidate) and polymarket_note, one row per
    submission row. same_party: the race ids of same-party top-two races, matched to their individual events."""
    rows = []
    for pick in submission.itertuples(index=False):
        individual = pick.race_id in same_party
        pool = markets[(markets["race_id"] == pick.race_id) & (markets["individual"] == individual)]
        odd, note = np.nan, None
        if pool.empty:
            note = "no open Polymarket winner event" + (" (individual)" if individual else "")
        else:
            named = by_name(pool, pick.candidate_name)
            by_party = pool[pool["party"] == pick.predicted_winner_party]
            match = named if individual else (by_party if len(by_party) == 1 else named)
            if len(match) == 1:
                odd = float(match["polymarket_prob"].iloc[0])
            else:
                note = f"{len(match)} markets match {pick.predicted_winner_party} / {pick.candidate_name}"
        rows.append({"race_id": pick.race_id, "polymarket_odd": odd, "polymarket_note": note})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    events, as_of = load_events(refresh=True)
    markets = race_markets(events)
    print(f"-> {RAW_DIR / f'events_{as_of}.json'}: {len(events)} race-winner events, {len(markets)} active markets")
    print(markets.groupby("individual")["race_id"].nunique().rename("races").to_string())
    sys.exit(0)
