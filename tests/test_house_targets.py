import math

import pandas as pd

import house_targets as ht
from house.regression.schema import SUPP_KEY, TARGET_COLUMNS, TARGETS_SCHEMA, TARGETS_SUPP_SCHEMA, validate_schema

COLUMNS = ["year", "state", "district", "stage", "special", "runoff", "candidate", "party", "writein",
           "candidatevotes", "totalvotes", "unofficial", "version"]


def rows(year, state, district, candidates, special=False, runoff="FALSE", stage="GEN", total=None):
    """SYNTHETIC source rows, not election data: one per (candidate, party, votes[, writein])."""
    total = total if total is not None else sum(c[2] for c in candidates)
    return [(year, state, district, stage, special, runoff, c[0], c[1], c[3] if len(c) > 3 else False, c[2], total,
             False, 20250910) for c in candidates]


def build(*contests, everything=False):
    """targets_raw indexed by key; with everything=True, (targets_raw, targets, supp), the first two indexed by key."""
    raw, targets, supp, _ = ht.build_targets(pd.DataFrame([r for c in contests for r in c], columns=COLUMNS))
    raw, targets = raw.set_index(ht.KEY), targets.set_index(ht.KEY)
    return (raw, targets, supp) if everything else raw


def test_votes_aggregate_across_fusion_lines_and_skip_nonvotes():
    out = build(rows(2024, "NY", 1, [("A", "DEMOCRAT", 600), ("A", "WORKING FAMILIES", 100), ("B", "REPUBLICAN", 250),
                                     ("B", "CONSERVATIVE", 50), ("C", "GREEN", 30), ("WRITEIN", None, 5),
                                     ("BLANK", None, 40)]))
    row = out.loc[(2024, "NY", 1)]
    assert (row["dem_votes"], row["gop_votes"], row["3p_votes"], row["total_votes"]) == (700, 300, 35, 1035)
    assert (row["dem_candidate_count"], row["gop_candidate_count"], row["3p_candidate_count"]) == (1, 1, 1)
    assert row["nonvote_votes"] == 40 and row["race_type"] == "dr" and row["winner_party"] == "D"
    assert math.isclose(row["margin_raw"], 100 * 400 / 1035)
    assert math.isclose(row["margin_2p_norm"], 40.0)
    assert row["adj_margin_2p_norm"] == row["margin_2p_norm"]
    assert row["is_imputed"] is False and pd.isna(row["imputation_method"]) and pd.isna(row["dem_votes_imputed"])
    assert not row["needs_imputation"] and not row["is_nonstandard_race"]


def test_targets_blank_without_a_reported_dr_contest():
    out = build(
        rows(2024, "CA", 1, [("A", "DEMOCRAT", 600), ("B", "DEMOCRAT", 400)]),
        rows(2024, "FL", 1, [("C", "REPUBLICAN", 1)], total=-1),
        rows(2024, "TX", 1, [("D", "REPUBLICAN", 900), ("E", "LIBERTARIAN", 100)]),
    )
    assert out["race_type"].tolist() == ["dem_vs_dem", "gop_uncontested", "gop_vs_3p"]
    assert out["adj_margin_2p_norm"].isna().all() and out["is_imputed"].isna().all()
    assert out["needs_imputation"].all() and out["is_nonstandard_race"].all()
    fl = out.loc[(2024, "FL", 1)]
    assert pd.isna(fl["gop_votes"]) and pd.isna(fl["margin_2p_norm"]) and fl["winner_party"] == "R"
    assert fl["quality_flags"] == "votes_not_reported" and fl["source_totalvotes"] == -1
    assert out.loc[(2024, "CA", 1), "margin_2p_norm"] == 100  # observed, but not a target


def test_one_row_per_key_using_the_november_round_and_every_contest_in_supp():
    out, targets, supp = build(
        rows(2014, "LA", 4, [("A", "DEMOCRAT", 500), ("B", "REPUBLICAN", 300), ("C", "REPUBLICAN", 250)]),
        rows(2014, "LA", 4, [("A", "DEMOCRAT", 480), ("C", "REPUBLICAN", 520)], runoff="TRUE"),
        rows(2014, "NY", 6, [("D", "DEMOCRAT", 400), ("E", "REPUBLICAN", 600)]), everything=True)
    assert len(out) == 2
    row = out.loc[(2014, "LA", 4)]
    assert (row["dem_votes"], row["gop_votes"], row["race_type"]) == (500, 550, "dr_jungle")
    assert not row["needs_imputation"] and row["is_nonstandard_race"]  # targets_raw keeps the jungle target...
    jungle = targets.loc[(2014, "LA", 4)]  # ...targets.csv leaves it to imputation, with the votes kept
    assert pd.isna(jungle["adj_margin_2p_norm"]) and jungle["needs_imputation"] and jungle["gop_votes"] == 550
    assert targets.loc[(2014, "NY", 6), "adj_margin_2p_norm"] == -20
    assert row["winner_party"] == "R" and row["quality_flags"] == "runoff_in_source"  # winner from the runoff
    assert list(out.reset_index().columns[:len(TARGET_COLUMNS)]) == TARGET_COLUMNS
    assert validate_schema(out.reset_index(), schema=TARGETS_SCHEMA).ok
    # supp: both contests of the two-contest key only, each with its own results; the runoff isn't a jungle round
    assert supp[SUPP_KEY].values.tolist() == [[2014, "LA", 4, "first_round"], [2014, "LA", 4, "runoff"]]
    assert supp["gop_votes"].tolist() == [550, 520] and supp["race_type"].tolist() == ["dr_jungle", "dr"]
    assert supp["winner_party"].tolist() == ["R", "R"]
    assert validate_schema(supp, schema=TARGETS_SUPP_SCHEMA, key=SUPP_KEY).ok


def test_cross_endorsed_candidate_counts_for_their_larger_line_in_targets_only():
    raw, targets, _ = build(rows(2024, "NY", 9, [("A", "DEMOCRAT", 600), ("A", "REPUBLICAN", 200), ("B", "GREEN", 50)]),
                            everything=True)
    before, after = raw.loc[(2024, "NY", 9)], targets.loc[(2024, "NY", 9)]
    assert before["race_type"] == "cross_endorsed" and pd.isna(before["winner_party"])
    assert (after["race_type"], after["winner_party"], after["dem_candidate_count"]) == ("dem_vs_3p", "D", 1)
    assert after["quality_flags"] == "cross_endorsed_assigned" and pd.isna(after["adj_margin_2p_norm"])
