"""Kalshi's 2024 House and Senate race markets and their election-eve prices, for the full-forecast backtest.

    uv run python src/kalshi/history.py      # fetch what isn't cached, then build the table (public API, no key)

Kalshi listed no election markets before the 2024 cycle, so 2020 and 2022 have no Kalshi prices.

Cache (data/raw/kalshi/history_2024/): every API response is saved as JSON and reread on later runs, so the API is
called once per item. A 404 (no market for that race) is cached too.
    events/<event_ticker>.json        GET /events/<ticker>: does the race have an event
    markets/<event_ticker>.json       GET /historical/markets?event_ticker=: its settled markets (party, result)
    candles/<market_ticker>.json      GET /historical/markets/<ticker>/candlesticks: daily candles, Oct 20 - Nov 6

Event tickers, probed for every 2024 race: HOUSE<ST><district>-24 (at-large: <ST>AL, e.g. HOUSEAKAL-24),
SENATE<ST>-24, and SENATE<ST>S-24 for a special election (NE 2024).

Election-eve price: the daily candle that ends at the start of Election Day (end_period_ts at or before
2024-11-05 05:00 UTC, midnight Eastern), so it holds the last full day of trading before the polls opened. The price
is the midpoint of that candle's closing YES bid and ask when both are quoted (odds.market_price), else its closing
trade, else the last earlier close. eve_prob normalizes the event's prices to sum to 1, as odds.race_markets does.

Output: data/processed/kalshi/kalshi_2024_eve.csv, one row per market: chamber, state, district (House) or class and
special (Senate), event_ticker, market_ticker, candidate, party (D, R or I, from the market's custom_strike), result
(yes / no), eve_yes_bid, eve_yes_ask, eve_price, eve_prob, eve_candle_end, volume_through_eve.
"""

import json
import sys
import time
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import house.regression as hr
import senate.regression as sr
from kalshi import odds

CACHE = ROOT / "data/raw/kalshi/history_2024"
OUTPUT_PATH = ROOT / "data/processed/kalshi/kalshi_2024_eve.csv"
YEAR = 2024
CANDLE_START = int(datetime(2024, 10, 20, tzinfo=UTC).timestamp())
CANDLE_END = int(datetime(2024, 11, 6, tzinfo=UTC).timestamp())
EVE_END = int(datetime(2024, 11, 5, 5, tzinfo=UTC).timestamp())  # midnight Eastern, start of Election Day


def cached(path: Path, url: str) -> dict | None:
    """The JSON at url, from path if saved; None (also saved) for a 404."""
    if path.exists():
        return json.loads(path.read_text())
    try:
        data = odds._get_json(url)
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        data = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    time.sleep(0.05)
    return data


def market_party(label: str | None) -> str | None:
    """D, R or I from the market's custom_strike Party ("Democratic", "Democratic (DFL)", "Republican", ...)."""
    if not label:
        return None
    return "D" if label.startswith("Democrat") else "R" if label.startswith("Republican") else "I"


def race_events() -> pd.DataFrame:
    """Every 2024 race with the event ticker to probe."""
    house = hr.load_targets().query("year == @YEAR")[["state", "district"]].drop_duplicates()
    house = house.assign(chamber="House", event_ticker=[
        f"HOUSE{s}{'AL' if d == 0 else d}-24" for s, d in zip(house["state"], house["district"])])
    senate = sr.load_data().query("year == @YEAR")[["state", "class", "special"]]
    senate = senate.assign(chamber="Senate", event_ticker=[
        f"SENATE{s}{'S' if sp else ''}-24" for s, sp in zip(senate["state"], senate["special"])])
    return pd.concat([house, senate], ignore_index=True)


def eve_quote(candles: list[dict]) -> dict:
    """The election-eve candle's closing bid, ask and price (module docstring)."""
    before = [c for c in candles if c["end_period_ts"] <= EVE_END]
    if not before:
        return {}
    eve = before[-1]
    bid = float((eve.get("yes_bid") or {}).get("close") or 0)
    ask = float((eve.get("yes_ask") or {}).get("close") or 0)
    trades = [float(c["price"]["close"]) for c in before if (c.get("price") or {}).get("close") is not None]
    price = odds.market_price(bid, ask, trades[-1] if trades else float("nan"))
    return {"eve_yes_bid": bid, "eve_yes_ask": ask, "eve_price": price,
            "eve_candle_end": datetime.fromtimestamp(eve["end_period_ts"], UTC).isoformat(),
            "volume_through_eve": sum(float(c.get("volume") or 0) for c in before)}


def main() -> None:
    races = race_events()
    rows = []
    for i, race in enumerate(races.to_dict("records")):
        ticker = race["event_ticker"]
        event = cached(CACHE / "events" / f"{ticker}.json", f"{odds.API}/events/{ticker}")
        if event is None:
            continue
        listing = cached(CACHE / "markets" / f"{ticker}.json", f"{odds.API}/historical/markets?event_ticker={ticker}")
        for m in listing["markets"]:
            candles = cached(CACHE / "candles" / f"{m['ticker']}.json",
                             f"{odds.API}/historical/markets/{m['ticker']}/candlesticks?start_ts={CANDLE_START}"
                             f"&end_ts={CANDLE_END}&period_interval=1440")
            party = (m.get("custom_strike") or {}).get("Party")
            rows.append({**{k: race.get(k) for k in ["chamber", "state", "district", "class", "special"]},
                         "event_ticker": ticker, "market_ticker": m["ticker"], "candidate": m.get("yes_sub_title"),
                         "party": market_party(party), "result": m.get("result"),
                         **eve_quote(candles["candlesticks"] if candles else [])})
        if i % 50 == 0:
            print(f"{i}/{len(races)} races probed, {len(rows)} markets")
    markets = pd.DataFrame(rows)
    markets["eve_prob"] = markets["eve_price"] / markets.groupby("event_ticker")["eve_price"].transform("sum")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    markets.to_csv(OUTPUT_PATH, index=False)
    print(f"-> {OUTPUT_PATH}: {markets['event_ticker'].nunique()} races, {len(markets)} markets, "
          f"{markets['eve_price'].notna().sum()} with an eve price")


if __name__ == "__main__":
    main()
