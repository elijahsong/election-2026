import numpy as np
import pandas as pd
import pytest

from kalshi import odds, optimizer


def market(ticker, candidate, subtitle, bid, ask, last=0.5, size=100):
    return {"ticker": ticker, "yes_sub_title": candidate, "subtitle": subtitle, "status": "active",
            "yes_bid_dollars": str(bid), "yes_ask_dollars": str(ask), "no_ask_dollars": str(round(1 - bid, 4)),
            "yes_ask_size_fp": str(size), "yes_bid_size_fp": str(size), "last_price_dollars": str(last)}


# SYNTHETIC events in Kalshi's shape: a D-vs-R House race, a California person market, a Senate race, and noise
EVENTS = [
    {"event_ticker": "KXHOUSERACE-PA10-26", "title": "PA-10 House winner?", "markets": [
        market("KXHOUSERACE-PA10-26-D", "Janelle Stelson", "Democratic party:: Democratic party", 0.60, 0.62),
        market("KXHOUSERACE-PA10-26-R", "Scott Perry", "Republican party:: Republican party", 0.38, 0.40)]},
    {"event_ticker": "KXCAELECTION-2607", "title": "CA-07 House winner? ", "markets": [
        market("KXCAELECTION-2607-DMAT", "Doris Matsui", "Doris Matsui", 0.70, 0.72),
        market("KXCAELECTION-2607-MVAN", "Mai Vang", "Mai Vang", 0.0, 0.0, last=0.30)]},
    {"event_ticker": "SENATELA-26", "title": "Kentucky Senate winner?", "markets": [
        market("SENATELA-26-R", "Andy Barr", "Republican party:: Republican party", 0.95, 0.97),
        market("SENATELA-26-D", "Charles Booker", "", 0.03, 0.05)]},
    {"event_ticker": "SENATEOH-28", "title": "Ohio Senate winner? (2028)", "markets": []},
]
TRACK = pd.DataFrame({
    "race_id": ["PA-10", "CA-07", "KY-SEN"], "chamber": ["House", "House", "Senate"],
    "predicted_winner_party": ["R", "D", "R"], "candidate_name": ["Scott Perry", "Mai Vang", "Andy Barr"],
    "win_probability": [0.55, 0.55, 0.99]})


def test_titles_not_tickers_name_the_race():
    assert odds.race_id_from_title("CA-11 House winner? ") == "CA-11"
    assert odds.race_id_from_title("Alaska House winner?") == "AK-AL"
    assert odds.race_id_from_title("Kentucky Senate winner?") == "KY-SEN"  # ticker SENATELA-26
    assert odds.race_id_from_title("Ohio Senate winner? (2028)") is None
    assert odds.race_id_from_title("Wisconsin Senate District 30 winner? (2028)") is None


def test_track_odds_prices_parties_and_matching():
    markets = odds.race_markets(EVENTS)
    assert markets["race_id"].tolist() == ["PA-10", "PA-10", "CA-07", "CA-07", "KY-SEN", "KY-SEN"]
    assert markets["party"].fillna("?").tolist() == ["D", "R", "?", "?", "R", "D"]  # the last from the ticker suffix
    vang = markets.set_index("candidate").loc["Mai Vang"]
    assert vang["price"] == 0.30  # no quote: the last trade
    assert vang["kalshi_prob"] == pytest.approx(0.30 / (0.71 + 0.30))

    candidates = pd.DataFrame({"state": ["CA", "CA"], "district": [7, 7], "candidate_name": ["Doris Matsui", "Mai Vang"],
                               "model_party": ["D", "D"]})
    markets = odds.fill_party_from_candidates(markets, candidates)
    table = odds.track_odds(TRACK, markets).set_index("race_id")
    assert table.loc["PA-10", "kalshi_winner_prob"] == pytest.approx(0.39 / (0.61 + 0.39))
    assert table.loc["CA-07", "match"] == "name"  # two D markets: the party is ambiguous
    assert table.loc["CA-07", "kalshi_favorite"] == "Doris Matsui"
    assert not table.loc["CA-07", "picks_agree"] and table.loc["KY-SEN", "picks_agree"]


def test_blend_flips_and_keeps_the_models_values_for_the_stale_check():
    markets = odds.race_markets(EVENTS[:2])
    markets["party"] = markets["party"].fillna("D")
    races = optimizer.tossups(odds.track_odds(TRACK.iloc[:2], markets))
    picks = optimizer.blend_picks(races, markets, market_weight=0.5).set_index("race_id")
    assert picks.loc["PA-10", "blend_prob"] == pytest.approx(0.5 * 0.55 + 0.5 * 0.39)
    assert picks.loc["PA-10", ["proposed_party", "proposed_candidate"]].tolist() == ["D", "Janelle Stelson"]
    assert picks.loc["PA-10", "proposed_probability"] == pytest.approx(1 - picks.loc["PA-10", "blend_prob"], abs=1e-6)
    assert picks.loc["PA-10", "expected_brier_proposed"] < picks.loc["PA-10", "expected_brier_submitted"]

    overrides = pd.DataFrame([["CA-07", "House", "D", "Mai Vang", 0.55, "D", "Doris Matsui", 0.55, "judgment"],
                              ["WI-03", "House", "D", "Rebecca Cooke", 0.59, "R", "Derrick Van Orden", 0.59, "x"]],
                             columns=optimizer.OVERRIDE_COLUMNS)
    proposed = optimizer.proposed_overrides(picks.reset_index(), overrides, market_weight=0.5, as_of="t").set_index(
        "race_id")
    assert proposed.loc["CA-07", "model_candidate"] == "Doris Matsui"  # the override's model values, not the track's
    assert proposed.loc["PA-10", ["model_party", "model_probability"]].tolist() == ["R", 0.55]
    assert proposed.loc["WI-03", "reason"] == "x"  # an override outside the toss-ups is kept


def test_kelly_takes_the_best_side_net_of_fees_and_caps_at_the_book():
    markets = odds.race_markets(EVENTS[:1])
    races = optimizer.tossups(odds.track_odds(TRACK.iloc[:1], markets))
    bets = optimizer.kelly_bets(races, markets, bankroll=1000, kelly_fraction=0.25, min_edge=0.02)
    bet = bets.iloc[0]
    # belief 0.55 on Perry: YES on R at 0.40 and NO on D at 0.40 cost the same, beating the D side
    cost = 0.40 + 0.07 * 0.40 * 0.60
    assert bet["candidate"] in ("Scott Perry", "Janelle Stelson") and bet["cost"] == pytest.approx(cost)
    assert bet["kelly"] == pytest.approx((0.55 - cost) / (1 - cost))
    assert bet["kelly_contracts"] == int(0.25 * bet["kelly"] * 1000 / cost) and bet["contracts"] == 100


def test_run_reads_the_latest_cached_snapshot_and_never_calls_the_api(tmp_path, monkeypatch):
    import json

    from kalshi import run

    def no_api():
        raise AssertionError("the API was called")

    monkeypatch.setattr(odds, "fetch_open_events", no_api)
    for stamp, title in [("2026-11-02T0900Z", "old"), ("2026-11-03T0912Z", "new")]:
        (tmp_path / f"events_{stamp}.json").write_text(json.dumps([{"title": title}]))
    events, as_of = run.load_events([], raw_dir=tmp_path)
    assert as_of == "2026-11-03T0912Z" and events == [{"title": "new"}]
    with pytest.raises(SystemExit):
        run.load_events([], raw_dir=tmp_path / "empty")


def test_portfolio_bets_pay_by_side_and_market():
    from kalshi import portfolio

    odds_table = pd.DataFrame({"race_id": ["PA-10"], "kalshi_market": ["PA10-R"]})  # the submitted candidate's market
    bets = pd.DataFrame({"race_id": "PA-10", "market_ticker": ["PA10-R", "PA10-R", "PA10-D", "PA10-D"],
                         "side": ["yes", "no", "yes", "no"], "contracts": 10, "cost": 0.4})
    pick_wins = np.array([[True], [False]])  # simulation 1: the submitted candidate wins; simulation 2: loses
    profit = portfolio.profit_draws(bets, odds_table, ["PA-10"], pick_wins)
    win, lose = 10 * (1 - 0.4), -10 * 0.4
    assert profit.tolist() == [[win, lose, lose, win], [lose, win, win, lose]]


def test_history_eve_price_uses_no_candle_from_election_day_on():
    from kalshi import history

    def candle(end, bid, ask, close):
        return {"end_period_ts": end, "yes_bid": {"close": str(bid)}, "yes_ask": {"close": str(ask)},
                "price": {"close": str(close)}, "volume": "10"}

    eve = history.EVE_END
    candles = [candle(eve - 2 * 86400, 0.40, 0.44, 0.41), candle(eve, 0.50, 0.54, 0.52),
               candle(eve + 86400, 0.98, 0.99, 0.99)]  # the result is known by then
    quote = history.eve_quote(candles)
    assert quote["eve_price"] == pytest.approx(0.52) and quote["volume_through_eve"] == 20
    assert history.market_party("Democratic (DFL)") == "D" and history.market_party("Independent") == "I"
