"""Kalshi prices for every race in the congress-track submission, from the public market-data API (no account, no key).

Source: GET https://api.elections.kalshi.com/trade-api/v2/events?status=open&with_nested_markets=true, paged by cursor
(fetch_open_events). Every open event on the exchange comes back (about 20,000, ~100 pages); race_markets keeps the
House and Senate winner events.

Race matching (race_id_from_title) uses the event title, not the ticker. Tickers are irregular: House races are split
across KXHOUSERACE-*, HOUSE*, KXHOUSE*, KXCAELECTION-* and KXCA11PERSON-*, and SENATELA-26 is the Kentucky race.
Titles are regular:
    "<ST>-<DD> House winner?"   -> "<ST>-<DD>"   (at-large seats: "<ST>-AL", or "Alaska House winner?" -> "AK-AL")
    "<State> Senate winner?"    -> "<ST>-SEN"    (2028 events are titled "... winner? (2028)" and don't match)
track_odds requires exactly one event for each template race.

Party of a market (market_party): the last part of the market subtitle ("Democratic party:: Democratic party",
"Dan Osborn:: Independent"), else the ticker suffix (D, DEM, R, GOP, I, IND). California's person markets
(KXCAELECTION, KXCA11PERSON) carry neither; fill_party_from_candidates takes theirs from
data/processed/house/candidates_2026.csv (model_party), matched on last name within the district.

Price (market_price): the midpoint of the YES bid and ask when both are quoted, else the last trade. kalshi_prob is
each market's price divided by the sum over its event, so a race's probabilities sum to 1 (this removes the
overround; in a three-way race the minor candidate keeps its share).

Join (track_odds): the submission's predicted winner is matched to the event's one market of the same party, else by
last name (top-two same-party races, and Kevin Kiley, whom the submission codes R and Kalshi codes Independent).
kalshi_winner_prob is Kalshi's probability for the submitted candidate; kalshi_favorite is Kalshi's most likely
candidate in the race.
"""

import json
import re
import time
import unicodedata
import urllib.error
import urllib.request

import pandas as pd

API = "https://api.elections.kalshi.com/trade-api/v2"

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
HOUSE_TITLE = re.compile(r"(?:(?P<st>[A-Z]{2})-(?P<district>\d{2}|AL)|(?P<state>[A-Z][a-z]+(?: [A-Z][a-z]+)*)) "
                         r"House winner\?")
SENATE_TITLE = re.compile(r"(?P<state>[A-Z][a-z]+(?: [A-Z][a-z]+)*) Senate winner\?")
PARTY_WORDS = {"Democratic": "D", "Republican": "R", "Independent": "I"}
PARTY_SUFFIXES = {"D": "D", "DEM": "D", "R": "R", "GOP": "R", "I": "I", "IND": "I"}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def _get_json(url: str, retries: int = 5) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "election26-kalshi-odds"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:  # rate limit or a transient server error: back off and retry
            if error.code not in (429, 500, 502, 503) or attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def fetch_open_events(api: str = API) -> list[dict]:
    """Every open Kalshi event with its markets nested."""
    events, cursor = [], ""
    while True:
        page = _get_json(f"{api}/events?status=open&limit=200&with_nested_markets=true"
                         + (f"&cursor={cursor}" if cursor else ""))
        events += page["events"]
        cursor = page.get("cursor")
        if not cursor:
            return events


def race_id_from_title(title: str) -> str | None:
    """The template race_id an event title names, or None for any other event."""
    title = title.strip()
    if m := HOUSE_TITLE.fullmatch(title):
        if m["st"]:
            return f"{m['st']}-{m['district']}"
        return f"{STATE_ABBR[m['state']]}-AL" if m["state"] in STATE_ABBR else None
    if (m := SENATE_TITLE.fullmatch(title)) and m["state"] in STATE_ABBR:
        return f"{STATE_ABBR[m['state']]}-SEN"
    return None


def market_party(market: dict) -> str | None:
    label = (market.get("subtitle") or "").split("::")[-1]
    for word, code in PARTY_WORDS.items():
        if word in label:
            return code
    return PARTY_SUFFIXES.get(market["ticker"].rsplit("-", 1)[-1])


def market_price(yes_bid: float, yes_ask: float, last_price: float) -> float:
    """Midpoint of the YES quote when both sides are quoted, else the last trade."""
    if 0 < yes_bid <= yes_ask < 1:
        return (yes_bid + yes_ask) / 2
    return last_price


def last_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    tokens = [t for t in re.sub(r"[^a-z\s-]", " ", ascii_name).split() if t not in NAME_SUFFIXES]
    return tokens[-1] if tokens else ""


def race_markets(events: list[dict]) -> pd.DataFrame:
    """One row per market of each House and Senate winner event, with its party, quote, price and kalshi_prob."""
    rows = []
    for event in events:
        race_id = race_id_from_title(event["title"])
        if race_id is None:
            continue
        for m in event.get("markets", []):
            rows.append({
                "race_id": race_id,
                "event_ticker": event["event_ticker"],
                "market_ticker": m["ticker"],
                "candidate": m.get("yes_sub_title"),
                "party": market_party(m),
                "status": m.get("status"),
                "yes_bid": float(m.get("yes_bid_dollars") or 0),
                "yes_ask": float(m.get("yes_ask_dollars") or 0),
                "no_ask": float(m.get("no_ask_dollars") or 0),
                "yes_ask_size": float(m.get("yes_ask_size_fp") or 0),
                "yes_bid_size": float(m.get("yes_bid_size_fp") or 0),  # the size available to buy NO at no_ask
                "last_price": float(m.get("last_price_dollars") or 0),
                "volume": float(m.get("volume_fp") or 0),
                "open_interest": float(m.get("open_interest_fp") or 0),
            })
    markets = pd.DataFrame(rows)
    markets["price"] = [market_price(*q) for q in markets[["yes_bid", "yes_ask", "last_price"]].itertuples(index=False)]
    markets["kalshi_prob"] = markets["price"] / markets.groupby("event_ticker")["price"].transform("sum")
    return markets


def fill_party_from_candidates(markets: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """markets with a missing party filled from the 2026 House candidate list, matched on last name in the district."""
    cands = candidates.assign(
        race_id=candidates["state"] + "-" + candidates["district"].map(lambda d: "AL" if d == 0 else f"{d:02d}"),
        last=candidates["candidate_name"].map(last_name))
    cands = cands[cands["model_party"].isin(["D", "R"])].drop_duplicates(["race_id", "last"])
    party = cands.set_index(["race_id", "last"])["model_party"]
    out = markets.copy()
    missing = out["party"].isna()
    keys = zip(out.loc[missing, "race_id"], out.loc[missing, "candidate"].fillna("").map(last_name))
    out.loc[missing, "party"] = [party.get(key) for key in keys]
    return out


def _match_market(race: pd.Series, event: pd.DataFrame) -> tuple[pd.Series | None, str]:
    same_party = event[event["party"] == race["predicted_winner_party"]]
    if len(same_party) == 1:
        return same_party.iloc[0], "party"
    same_name = event[event["candidate"].fillna("").map(last_name) == last_name(race["candidate_name"])]
    if len(same_name) == 1:
        return same_name.iloc[0], "name"
    return None, "none"


def track_odds(track: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """The submission's rows, in order, beside Kalshi's price for each predicted winner."""
    events = markets[markets["race_id"].isin(track["race_id"])].groupby("race_id")["event_ticker"].unique()
    missing = sorted(set(track["race_id"]) - set(events.index))
    repeated = events[events.map(len) > 1]
    if missing or len(repeated):
        raise ValueError(f"Kalshi events: none for {missing}; several for {repeated.to_dict()}")

    rows = []
    for _, race in track.iterrows():
        event = markets[markets["event_ticker"] == events[race["race_id"]][0]]
        market, match = _match_market(race, event)
        favorite = event.loc[event["kalshi_prob"].idxmax()]
        row = {"kalshi_event": favorite["event_ticker"], "match": match,
               "kalshi_favorite": favorite["candidate"], "kalshi_favorite_party": favorite["party"],
               "kalshi_favorite_prob": favorite["kalshi_prob"], "kalshi_event_volume": event["volume"].sum()}
        if market is not None:
            row |= {"kalshi_market": market["market_ticker"], "kalshi_candidate": market["candidate"],
                    "kalshi_party": market["party"], "kalshi_winner_prob": market["kalshi_prob"],
                    "yes_bid": market["yes_bid"], "yes_ask": market["yes_ask"],
                    "names_agree": last_name(market["candidate"] or "") == last_name(race["candidate_name"]),
                    "picks_agree": market["market_ticker"] == favorite["market_ticker"]}
        rows.append(row)
    return pd.concat([track.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
