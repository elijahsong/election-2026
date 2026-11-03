from pathlib import Path

import pandas as pd
import pytest

import senate_polls
import senate_ratings
from senate_schema import read_output

PROCESSED = Path(__file__).resolve().parents[1] / "data/processed/senate"
built = pytest.mark.skipif(not (PROCESSED / "expert-ratings.csv").exists(), reason="polls and ratings not built")


def _race(state, cls=2, special=False, flags=None):
    # SYNTHETIC 2024 race (Election Day 2024-11-05, cutoff the day before)
    return {"year": 2024, "state": state, "class": cls, "special": special, "election_id": f"2024_{state}_{cls}",
            "election_date": "2024-11-05", "cutoff_date": "2024-11-04", "quality_flags": flags}


def _poll(state, end, margin, pollster="P", stage="general", duplicate=False):
    return {"year": 2024, "state": state, "class": 2, "stage": stage, "end_date": end, "poll_margin_dem": margin,
            "pollster": pollster, "is_duplicate": duplicate}


def test_poll_average_uses_the_30_days_before_election_day_and_keeps_unpolled_races_blank():
    races = pd.DataFrame([_race("AA"), _race("BB")])
    clean = pd.DataFrame([
        _poll("AA", "2024-10-05", 50.0),               # 31 days before: outside the window
        _poll("AA", "2024-10-06", 4.0, "X"),           # 30 days before: in
        _poll("AA", "2024-11-04", -2.0, "Y"),          # the day before: in
        _poll("AA", "2024-11-05", 90.0),               # Election Day itself: out
        _poll("AA", "2024-10-20", 70.0, duplicate=True),
        _poll("AA", "2024-10-21", None),               # no D-R margin
        _poll("AA", "2024-11-20", 60.0, stage="runoff"),
    ])
    out = senate_polls.aggregate_polls(clean, races).set_index("state")
    assert out.loc["AA", "poll_margin_dem"] == pytest.approx(1.0)  # simple mean of 4 and -2
    assert (out.loc["AA", "n_polls"], out.loc["AA", "n_pollsters"]) == (2, 2)
    assert out.loc["AA", "latest_poll_date"] == "2024-11-04" and out.loc["AA", "window_start"] == "2024-10-06"
    assert pd.isna(out.loc["BB", "poll_margin_dem"]) and out.loc["BB", "n_polls"] == 0
    assert not out.loc["BB", "poll_available"]

    # a cycle in progress: polls that ended after --as-of aren't used
    early = senate_polls.aggregate_polls(clean, races, as_of="2024-10-31").set_index("state")
    assert early.loc["AA", "cutoff_date"] == "2024-10-31" and early.loc["AA", "poll_margin_dem"] == 4.0


def test_rating_scores_are_democratic_complements_and_the_consensus_is_40_30_30():
    assert senate_ratings.prob_dem(pd.Series(["Solid R", "Tilt D", "Tossup", "Likely D"])).tolist() == \
        [0.01, 0.60, 0.50, 0.90]
    for rating, gop in senate_ratings.RATING_GOP_SCORE.items():
        mirrored = rating.replace(" D", " X").replace(" R", " D").replace(" X", " R")
        assert senate_ratings.RATING_GOP_SCORE[mirrored] == pytest.approx(1 - gop)
    assert sum(senate_ratings.WEIGHTS.values()) == pytest.approx(1.0)

    races = pd.DataFrame([_race("AA"), _race("BB"), _race("CC")]).drop(columns=["election_date", "cutoff_date"])
    rated = pd.DataFrame({"year": 2024, "state": ["AA", "BB"], "class": 2, "source": "test",
                          "sabato_rating": ["Lean D", "Solid R"], "cook_rating": ["Tossup", None],
                          "ie_rating": ["Tilt R", "Likely R"]})
    out = senate_ratings.build_ratings(rated, races).set_index("state")
    assert out.loc["AA", "expert_consensus_dem"] == pytest.approx(0.40 * 0.75 + 0.30 * 0.50 + 0.30 * 0.40)
    # Cook missing: Sabato and Inside Elections renormalized to 4/7 and 3/7, flagged
    assert out.loc["BB", "expert_consensus_dem"] == pytest.approx((0.40 * 0.01 + 0.30 * 0.10) / 0.70)
    assert out.loc["BB", "n_experts"] == 2 and not out.loc["BB", "coverage_complete"]
    assert out.loc["BB", "quality_flags"] == "missing_cook"
    assert pd.isna(out.loc["CC", "expert_consensus_dem"]) and out.loc["CC", "n_experts"] == 0


@built
def test_polls_split_two_race_state_years_by_rcp_table():
    clean = read_output("polls_clean.csv")
    # OK 2014: the same Sooner Poll asked both races; Inhofe (class 2) led 63-28, Lankford (class 3) 59-28
    sooner = clean[(clean["year"] == 2014) & (clean["state"] == "OK") & (clean["end_date"] == "2014-10-29")]
    assert sooner.set_index("class")["gop_pct"].to_dict() == {2: 63.0, 3: 59.0}
    # GA 2020: Perdue-Ossoff only, November and the January 2021 runoff
    ga = clean[(clean["year"] == 2020) & (clean["state"] == "GA")]
    assert set(ga["class"]) == {2} and ga.groupby("stage").size().to_dict() == {"general": 25, "runoff": 13}


@built
def test_poll_margins_are_democratic_positive_and_averages_never_see_election_day():
    clean = read_output("polls_clean.csv")
    margin = 100 * (clean["dem_pct"] - clean["gop_pct"]) / (clean["dem_pct"] + clean["gop_pct"])
    has = clean["poll_margin_dem"].notna()
    assert (clean.loc[has, "poll_margin_dem"] - margin[has]).abs().max() < 1e-9
    assert not clean.loc[clean["is_duplicate"], "duplicate_of"].isna().any()

    polls = read_output("polls.csv")
    dated = polls[polls["latest_poll_date"].notna()]
    assert (dated["latest_poll_date"] <= dated["cutoff_date"]).all()
    # 2026 comes only from the hand-entered FiftyPlusOne averages, never mixed with RCP polls
    assert set(polls.loc[polls["year"] == 2026, "poll_source"].dropna()) <= {"FiftyPlusOne average "
                                                                             "(data/reference/polls_2026_fiftyplusone"
                                                                             ".csv, manual entry)"}
    assert (polls["cutoff_date"] < polls["election_date"]).all()
    assert (polls.loc[~polls["poll_available"], "n_polls"] == 0).all()
    assert polls.loc[~polls["poll_available"], "poll_margin_dem"].isna().all()


@built
def test_expert_ratings_recompute_and_carry_no_outcomes():
    ratings = read_output("expert-ratings.csv")
    assert not {"probwin_outcome", "correct", "brier", "act_val", "winner_party"} & set(ratings.columns)
    for p in senate_ratings.PROVIDERS:
        rated = ratings[f"{p}_rating"].notna()
        gop = ratings.loc[rated, f"{p}_rating"].map(senate_ratings.RATING_GOP_SCORE)
        assert ((ratings.loc[rated, f"{p}_prob_dem"] + gop - 1).abs() < 1e-9).all()
    full = ratings["coverage_complete"]
    expected = 0.40 * ratings["sabato_prob_dem"] + 0.30 * ratings["cook_prob_dem"] + 0.30 * ratings["ie_prob_dem"]
    assert (ratings.loc[full, "expert_consensus_dem"] - expected[full]).abs().max() < 1e-12
    nebraska = ratings[(ratings["year"] == 2026) & (ratings["state"] == "NE")]
    assert nebraska["quality_flags"].str.contains("independent_vs_gop").all()
