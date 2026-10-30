"""Reusable model specifications for experiments: estimator, its parameters, base predictors, and target.

Years, training inputs, and feature changes are not set here; they are arguments of run.run_experiment (or CLI flags),
so one spec serves every experiment. A spec names its base predictors with "feature_set" (a house.regression
FEATURE_SETS key) or "features" (an explicit list). Add a spec here to make it available as --model <name>.

house.regression.MODEL_CONFIGS are complete configs with years; these are the year-free model parts.
"""

from house.regression.config import CLASSIC_OLS

MODEL_CONFIGS = {
    "classic_ols": CLASSIC_OLS,  # OLS with a constant on gen_margin_raw, partisan_lean, inc_signed
    "enhanced_ols": {**CLASSIC_OLS, "feature_set": "enhanced"},  # + college, hispanic
}
