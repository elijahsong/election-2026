import math

import numpy as np
import pandas as pd
import pytest

from experiments.house import evaluation

TARGET = "adj_margin_2p_norm"
COMPETITIVE = {"ex_ante": ("partisan_lean", 10), "ex_post": ("margin_2p_norm", 10), "none": ("partisan_lean", 0)}
RACE_TYPES = ["dr", "dr_rcv"]


def synthetic():
    """SYNTHETIC 2022 districts, NC-1..8. NC-1..4 are valid, with residuals (actual - predicted) -2, -4, -10, +8
    (MAE 6, RMSE sqrt(46), bias -2); NC-2 is an RCV race the D won after trailing (winner conflict), NC-4 is predicted exactly 0 (a tie).
    NC-5 is imputed, NC-6 has no target, NC-7 is an ineligible race type, and NC-8 is not predicted."""
    rows = [  # district, race_type, actual, is_imputed, winner_party, prediction, partisan_lean
        (1, "dr", 4.0, False, "D", 6.0, 3.0),
        (2, "dr_rcv", -2.0, False, "D", 2.0, -5.0),
        (3, "dr", -30.0, False, "R", -20.0, -25.0),
        (4, "dr", 8.0, False, "D", 0.0, 12.0),
        (5, "dem_uncontested", 40.0, True, "D", 30.0, 35.0),
        (6, "no_major_party", np.nan, None, None, 1.0, 0.0),
        (7, "dem_vs_dem", 20.0, False, "D", 15.0, 1.0),
        (8, "dr", 5.0, False, "D", np.nan, 2.0),
    ]
    df = pd.DataFrame(rows, columns=["district", "race_type", TARGET, "is_imputed", "winner_party", "prediction",
                                     "partisan_lean"]).assign(year=2022, state="NC")
    targets = df[["year", "state", "district", TARGET, "is_imputed", "race_type", "winner_party"]].assign(
        margin_2p_norm=df[TARGET])
    predictions = df.loc[df["prediction"].notna(), ["year", "state", "district", "prediction"]]
    return predictions, targets, df[["year", "state", "district", "partisan_lean"]]


def evaluate():
    predictions, targets, predictors = synthetic()
    return evaluation.evaluate_predictions(predictions, targets, predictors, TARGET, competitive=COMPETITIVE,
                                           race_types=RACE_TYPES)


def test_only_observed_non_imputed_eligible_targets_are_scored():
    _, coverage, scored = evaluate()
    assert scored["district"].tolist() == [1, 2, 3, 4]
    assert coverage.loc[2022].to_dict() == {
        "n_districts": 8, "missing_target": 1, "imputed_target": 1, "race_type:dem_vs_dem": 1, "no_prediction": 1,
        "n_valid": 4, "ex_ante": 2, "ex_ante_undetermined": 0, "ex_post": 3, "ex_post_undetermined": 0,
        "none": 0, "none_undetermined": 0,
    }


def test_regression_and_winner_metrics():
    metrics, _, scored = evaluate()
    valid = metrics.loc[("valid", 2022)]
    assert (valid["n"], valid["mae"], valid["bias"]) == (4, 6, -2)  # Republicans outperformed the forecast
    assert valid["rmse"] == pytest.approx(math.sqrt(46))
    assert scored["residual"].tolist() == [-2, -4, -10, 8]  # actual - predicted
    # NC-4's predicted tie is left out of the classification, not assigned to a party
    assert valid[["n_classified", "n_actual_tie", "n_predicted_tie", "missed_races"]].tolist() == [3, 0, 1, 1]
    assert valid["missed_pct"] == pytest.approx(100 / 3)
    # classified: NC-1 D/D, NC-2 R/D, NC-3 R/R; of 2 predicted D, 1 won; of 1 predicted R, 1 won
    assert valid[["accuracy_pct", "precision_D_pct", "precision_R_pct"]].tolist() == pytest.approx([200 / 3, 50, 100])
    assert valid[["actual_D_pred_D", "actual_D_pred_R", "actual_R_pred_D", "actual_R_pred_R"]].tolist() == [1, 0, 1, 1]
    assert valid["n_winner_conflict"] == 1 and scored.loc[scored["winner_conflict"], "district"].tolist() == [2]
    assert scored["correct_winner"].tolist() == [True, False, True, pd.NA]
    assert evaluation.confusion_matrix(scored).to_numpy().tolist() == [[1, 0], [1, 1]]
    assert evaluation.confusion_matrix(scored, percent=True).to_numpy().ravel() == pytest.approx([100 / 3, 0, 100 / 3, 100 / 3])
    assert scored["outcome"].tolist() == ["correct", "gop_miss", "correct", "tie"]
    assert metrics.loc[("valid", "all")].equals(valid.rename(("valid", "all")))  # one year: pooled = per year

    empty = metrics.loc[("none", 2022)]
    assert empty["n"] == 0 and empty[["mae", "rmse", "bias", "missed_pct", "accuracy_pct"]].isna().all()


def test_competitive_subsets_and_worst_misses():
    metrics, _, scored = evaluate()
    assert scored.loc[scored["ex_ante"], "district"].tolist() == [1, 2]  # abs(partisan_lean) <= 10
    assert scored.loc[scored["ex_post"], "district"].tolist() == [1, 2, 4]  # abs(margin_2p_norm) <= 10
    assert metrics.loc[("ex_ante", 2022), "mae"] == 3
    predictions, targets, predictors = synthetic()
    tighter = evaluation.evaluate_predictions(predictions, targets, predictors, TARGET,
                                              competitive={"ex_post": ("margin_2p_norm", 3.5)}, race_types=RACE_TYPES)[2]
    assert tighter.loc[tighter["ex_post"], "district"].tolist() == [2]  # a float threshold: only abs(-2) <= 3.5
    # an expression of columns, from the predictors and the targets: abs(lean + actual) is 7, -7, -55, 20 -> NC-1, NC-2
    summed = evaluation.evaluate_predictions(predictions, targets, predictors, TARGET, race_types=RACE_TYPES,
                                             competitive={"sum": ("partisan_lean + margin_2p_norm", 7)})[2]
    assert summed.loc[summed["sum"], "district"].tolist() == [1, 2]
    assert summed["partisan_lean + margin_2p_norm"].tolist() == [7, -7, -55, 20]
    with pytest.raises(ValueError, match="non-negative number"):
        evaluation.evaluate_predictions(predictions, targets, predictors, TARGET,
                                        competitive={"ex_post": ("margin_2p_norm", "7")}, race_types=RACE_TYPES)

    assert evaluation.worst_misses(scored, n=2)["district"].tolist() == [3, 4]
    assert evaluation.worst_misses(scored, n=5, subset="ex_ante")["district"].tolist() == [2, 1]
    sample = evaluation.competitive_examples(scored, "ex_post", n=2, seed=0)
    assert len(sample) == 2 and sample.equals(evaluation.competitive_examples(scored, "ex_post", n=2, seed=0))
    with pytest.raises(ValueError, match="Not in the ex_ante subset"):
        evaluation.competitive_examples(scored, "ex_ante", n=2, seed=0, districts=[(2022, "NC", 3)])

    report = evaluation.outcome_report(scored, "ex_post").splitlines()
    assert report[report.index("PREDICTED CORRECTLY (1)") + 1].startswith("* NC-01 (D)")
    assert "* NC-02 (R): predicted margin: +2.00, actual: -2.00, residual: -4.00" in report  # a GOP miss
    assert "NOT CLASSIFIED: a margin of exactly 0 (1)" in report  # NC-4's predicted tie

    states = evaluation.state_summary(scored, ["ex_ante"])
    assert states.loc["NC", ["n_districts", "mean_residual", "n_missed", "n_gop_miss", "n_ex_ante", "missed_ex_ante"]
                      ].tolist() == [4, -2, 1, 1, 2, 1]
    assert states.loc["NC", "pct_missed"] == pytest.approx(100 / 3)  # the predicted tie isn't classified
    assert states.loc["NC", "pct_missed_ex_ante"] == 50
