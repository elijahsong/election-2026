import pandas as pd
import pytest

import house_impute_targets_uncontested as imp
from test_house_targets import build, rows

KEY = imp.KEY
SHAVE = pd.DataFrame({"year": [2014], "shave_shift": [0.02]})  # SYNTHETIC national SHAVE: margin_2p_norm +4


def contests():
    """SYNTHETIC races, not election data, with their SYNTHETIC lean (D two-party share in a neutral year)."""
    races = [
        (rows(2014, "NC", 1, [("A", "DEMOCRAT", 600), ("B", "REPUBLICAN", 400)]), 0.55),  # dr, residual +0.05
        (rows(2014, "NC", 2, [("C", "DEMOCRAT", 300), ("D", "REPUBLICAN", 500)]), 0.40),  # dr, residual -0.025
        (rows(2014, "NC", 3, [("E", "DEMOCRAT", 700), ("F", "LIBERTARIAN", 100)]), 0.70),  # dem_vs_3p
        (rows(2014, "NC", 4, [("G", "DEMOCRAT", 900)]), None),  # dem_uncontested without a lean
        (rows(2014, "NC", 5, [("H", "DEMOCRAT", 900)]), 0.995),  # dem_uncontested, share clipped to 1
        (rows(2014, "NC", 6, [("I", "DEMOCRAT", 800), ("J", "REPUBLICAN", 50, True)]), 0.10),  # dr_writein
        (rows(2014, "LA", 1, [("K", "REPUBLICAN", 300), ("L", "REPUBLICAN", 200), ("M", "DEMOCRAT", 400)]), 0.40),
        (rows(2014, "LA", 2, [("N", "REPUBLICAN", 700), ("O", "REPUBLICAN", 300)]), 0.333),  # gop_vs_gop
        (rows(2014, "VT", 0, [("P", "DEMOCRAT", 600), ("Q", "PROGRESSIVE", 100)]), 0.65),  # dem_vs_3p, at-large
    ]
    raw, targets, _ = build(*(r for r, _ in races), everything=True)
    lean = pd.DataFrame([(*r[0][:3], lean) for r, lean in races if lean is not None], columns=[*KEY, "lean"])
    return raw.reset_index(), targets.reset_index(), lean


def test_uncontested_targets_come_from_the_state_mean_residual():
    raw, targets, lean = contests()
    out = imp.impute_uncontested(targets, lean, SHAVE).set_index(KEY)
    nc3, la2 = out.loc[(2014, "NC", 3)], out.loc[(2014, "LA", 2)]
    assert (nc3["race_type"], la2["race_type"]) == ("dem_vs_3p", "gop_vs_gop")

    # NC: turnout (1000 + 800) / 2 = 900, adjustment (0.05 - 0.025) / 2 = 0.0125, share 0.7125
    assert (nc3["dem_votes_imputed"], nc3["gop_votes_imputed"]) == (641, 259)
    assert nc3["adj_margin_2p_norm"] == pytest.approx(100 * (641 - 259) / 900)
    assert nc3["is_imputed"] is True and nc3["imputation_method"] == imp.IMPUTATION_METHOD
    assert nc3["needs_imputation"]
    # share 0.995 + 0.0125 is clipped to 1
    assert (out.loc[(2014, "NC", 5), "dem_votes_imputed"], out.loc[(2014, "NC", 5), "gop_votes_imputed"]) == (900, 0)
    assert out.loc[(2014, "NC", 5), "adj_margin_2p_norm"] == 100

    # LA: the jungle race's party totals are the reference: residual 400 / 900 - 0.40, turnout 900
    share = 0.333 + 400 / 900 - 0.40
    assert (la2["dem_votes_imputed"], la2["gop_votes_imputed"]) == (round(900 * share), 900 - round(900 * share))

    # the original columns stay as they were, and every vote column still matches targets_raw.csv
    imputed = ["adj_margin_2p_norm", "dem_votes_imputed", "gop_votes_imputed", "is_imputed", "imputation_method",
               "quality_flags"]
    pd.testing.assert_frame_equal(out.reset_index().drop(columns=imputed), targets.drop(columns=imputed),
                                  check_dtype=False)
    assert not imp.check_votes_unchanged(out.reset_index(), raw)


def test_rows_without_a_reference_fall_back_to_the_national_shave():
    _, targets, lean = contests()
    vt = imp.impute_uncontested(targets, lean, SHAVE).set_index(KEY).loc[(2014, "VT", 0)]
    # share 0.65 + 4 / 200 = 0.67: margin +34, no vote counts without a turnout reference
    assert vt["adj_margin_2p_norm"] == pytest.approx(34)
    assert vt["is_imputed"] is True and vt["imputation_method"] == imp.FALLBACK_METHOD
    assert pd.isna(vt["dem_votes_imputed"]) and pd.isna(vt["gop_votes_imputed"])
    assert vt["quality_flags"] == imp.NO_REFERENCE_FLAG and vt["needs_imputation"]


def test_rows_without_a_lean_or_shave_stay_blank_and_flagged():
    _, targets, lean = contests()
    out = imp.impute_uncontested(targets, lean, SHAVE.iloc[:0]).set_index(KEY)
    for key, flags in [((2014, "VT", 0), f"{imp.NO_REFERENCE_FLAG};{imp.NO_SHAVE_FLAG}"),
                       ((2014, "NC", 4), imp.NO_LEAN_FLAG)]:
        row = out.loc[key]
        assert pd.isna(row["adj_margin_2p_norm"]) and pd.isna(row["is_imputed"]) and pd.isna(row["dem_votes_imputed"])
        assert row["quality_flags"] == flags and row["needs_imputation"]
    assert out.loc[(2014, "NC", 3), "imputation_method"] == imp.IMPUTATION_METHOD  # the primary method needs no SHAVE


def test_read_shave_uses_the_two_party_margin(tmp_path):
    path = tmp_path / "shave.csv"
    # SYNTHETIC: dem_2p_norm and gop_2p_norm are ignored, only margin_2p_norm is used
    pd.DataFrame({"year": [2014], "dem_2p_norm": [99.0], "gop_2p_norm": [1.0], "margin_2p_norm": [4.0]}) \
        .to_csv(path, index=False)
    assert imp.read_shave(path).to_dict("records") == [{"year": 2014, "shave_shift": 0.02}]


def test_me2_rows_without_first_round_counts_are_not_references():
    _, targets, _ = build(rows(2018, "ME", 1, [("A", "DEMOCRAT", 600), ("B", "REPUBLICAN", 400)]),
                          rows(2018, "ME", 2, [("C", "DEMOCRAT", 100), ("D", "REPUBLICAN", 900)]), everything=True)
    targets = targets.reset_index()
    assert targets["quality_flags"].fillna("").str.contains(imp.NOT_FIRST_ROUND_FLAG).tolist() == [False, True]
    lean = targets[KEY].assign(lean=0.5)
    states = imp.state_references(targets, lean)
    assert states[["n_references", "turnout_hat"]].values.tolist() == [[1, 1000]]
    assert states["state_adjustment"].item() == pytest.approx(0.1)


def test_observed_races_are_untouched_and_dr_writein_is_not_a_reference():
    _, targets, lean = contests()
    out = imp.impute_uncontested(targets, lean, SHAVE).set_index(KEY)
    for key in [(2014, "NC", 1), (2014, "NC", 6), (2014, "LA", 1)]:
        pd.testing.assert_series_equal(out.loc[key], targets.set_index(KEY).loc[key], check_dtype=False)
    assert out.loc[(2014, "NC", 6), "race_type"] == "dr_writein"
    # with NC-6 (residual 800 / 850 - 0.10) as a reference, NC-3 would differ from the two-reference result
    assert out.loc[(2014, "NC", 3), "dem_votes_imputed"] == 641


def test_rerunning_gives_the_same_frame():
    _, targets, lean = contests()
    once = imp.impute_uncontested(targets, lean, SHAVE)
    pd.testing.assert_frame_equal(imp.impute_uncontested(once, lean, SHAVE), once)
    assert once.loc[once["quality_flags"].notna(), "quality_flags"].str.count(";").eq(0).all()
