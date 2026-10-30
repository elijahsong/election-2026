import pandas as pd
import pytest

import house_impute_targets_jungle_rcv as imp
from test_house_targets import build, rows


def test_jungle_and_rcv_targets_are_their_observed_first_round_party_totals():
    raw, targets, _ = build(
        rows(2014, "LA", 5, [("A", "REPUBLICAN", 300), ("B", "REPUBLICAN", 200), ("C", "DEMOCRAT", 400),
                             ("D", "LIBERTARIAN", 100)]),
        rows(2022, "AK", 0, [("E", "DEMOCRAT", 480), ("F", "REPUBLICAN", 520)]),
        rows(2014, "LA", 6, [("G", "REPUBLICAN", 700), ("H", "REPUBLICAN", 300)]),
        rows(2014, "TX", 1, [("I", "DEMOCRAT", 600), ("J", "REPUBLICAN", 400)]),
        everything=True)
    out = imp.impute_jungle_rcv(targets.reset_index()).set_index(imp.KEY)

    jungle, rcv = out.loc[(2014, "LA", 5)], out.loc[(2022, "AK", 0)]
    assert (jungle["race_type"], rcv["race_type"]) == ("dr_jungle", "dr_rcv")
    assert jungle["adj_margin_2p_norm"] == pytest.approx(100 * (400 - 500) / 900)
    for row in (jungle, rcv):
        assert row["adj_margin_2p_norm"] == row["margin_2p_norm"]
        assert row["is_imputed"] is False and not row["needs_imputation"]
        assert pd.isna(row["imputation_method"]) and pd.isna(row["dem_votes_imputed"])
        assert pd.isna(row["gop_votes_imputed"])
    assert not imp.check_against_raw(out.reset_index(), raw.reset_index())

    # other races are untouched: gop_vs_gop stays blank, dr keeps its observed target
    pd.testing.assert_series_equal(out.loc[(2014, "LA", 6)], targets.loc[(2014, "LA", 6)])
    pd.testing.assert_series_equal(out.loc[(2014, "TX", 1)], targets.loc[(2014, "TX", 1)])


def test_rerunning_gives_the_same_frame():
    _, targets, _ = build(rows(2014, "LA", 5, [("A", "REPUBLICAN", 300), ("C", "DEMOCRAT", 400)]), everything=True)
    once = imp.impute_jungle_rcv(targets.reset_index())
    pd.testing.assert_frame_equal(imp.impute_jungle_rcv(once), once)
