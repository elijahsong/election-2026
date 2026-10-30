"""User-editable Stage 1 settings: feature lists, holdout years, and named model configs.

Nothing here is chosen or changed automatically. The values below are placeholders; edit them here, add new named
configs, or copy one in a notebook and change the copy (e.g. {**config, "train_years": [...]}).

A config names its predictors with either "feature_set" (a FEATURE_SETS key) or "features" (an explicit list), and
its outcome with "target" (a target column from targets.csv; adj_margin_2p_norm is the canonical one). The
predictors (modeling.csv) and targets (targets.csv) stay separate until backtest.run_experiment joins them.
"""

from .schema import COLUMNS, TRANCHES

# Any predictor column of data/processed/house/modeling.csv can be listed. schema.TRANCHES lists each tranche's
# predictors: basic_f, acs_f, and nat_political_f (e.g. TRANCHES["basic_f"] + ["college"]).
FEATURE_SETS = {
    "classic": TRANCHES["basic_f"],  # gen_margin_raw, partisan_lean, inc_signed
    "enhanced": TRANCHES["basic_f"] + ["college", "hispanic"],
}

# Reserved against accidental use: run_experiment refuses to train or evaluate on these years unless the config sets
# "use_holdout": True.
HOLDOUT_YEARS = [2024]

# ANSI color codes for describe_config, printed bold: model_type by MODEL_COLORS and tranche names by TRANCHE_COLORS
# (default color when missing), and the year lists by YEAR_COLORS
MODEL_COLORS = {"ols": "36"}  # cyan
YEAR_COLORS = {"train_years": "32", "eval_years": "33"}  # green, yellow
TRANCHE_COLORS = {"basic_f": "35", "acs_f": "34", "nat_political_f": "31"}  # magenta, blue, red

CLASSIC_OLS = {
    "model_type": "ols",  # a models.MODELS key
    "model_params": {"add_constant": True},
    "feature_set": "classic",
    "target": "adj_margin_2p_norm",
}

MODEL_CONFIGS = {
    "classic_ols": {
        **CLASSIC_OLS,
        "train_years": [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020],
        "eval_years": [2022],
    },
    "classic_ols_holdout": {
        **CLASSIC_OLS,
        "train_years": [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022],
        "eval_years": [2024],
        "use_holdout": True,
    },
    "classic_ols_2026": {  # production forecast: no 2026 targets needed
        **CLASSIC_OLS,
        "train_years": [2006, 2008, 2010, 2012, 2014, 2016, 2018, 2020, 2022, 2024],
        "eval_years": [2026],
        "use_holdout": True,
    },
}


def resolve_features(config: dict) -> list[str]:
    """The config's predictor list, from "features" or "feature_set" (exactly one)."""
    if ("features" in config) == ("feature_set" in config):
        raise ValueError('Give exactly one of "features" or "feature_set"')
    return list(config["features"]) if "features" in config else list(FEATURE_SETS[config["feature_set"]])


def _bold(text: str, code: str) -> str:
    return f"\033[1;{code}m{text}\033[0m"


def describe_config(config: dict, color: bool = True) -> str:
    """The config for printing: the model settings, a rule, then the data settings (features resolved and grouped by
    tranche, holdout status spelled out) and any other keys. Colored by MODEL_COLORS, YEAR_COLORS and TRANCHE_COLORS
    unless color=False."""
    tranches = {}
    for name in resolve_features(config):
        column = COLUMNS.get(name)
        tranches.setdefault(column.tranche if column else "unregistered", []).append(name)
    model = {name: str(config[name]) for name in ["model_type", "model_params", "feature_set", "target"]
             if name in config}
    data = {
        "train_years": ", ".join(map(str, config["train_years"])),
        "eval_years": ", ".join(map(str, config["eval_years"])),
        "features": "\n".join(f"{tranche}: [{', '.join(names)}]" for tranche, names in tranches.items()),
        "use_holdout": f"{config.get('use_holdout', False)} (HOLDOUT_YEARS: {HOLDOUT_YEARS})",
    }
    data |= {name: str(value) for name, value in config.items() if name not in model and name not in data}
    width = max(map(len, model | data))
    rule = "=" * min(max(width + 2 + len(line) for value in (model | data).values() for line in value.split("\n")), 80)
    if color:
        model["model_type"] = _bold(model["model_type"], MODEL_COLORS.get(config["model_type"], "39"))
        for name, code in YEAR_COLORS.items():
            data[name] = _bold(data[name], code)
        data["features"] = "\n".join(f"{_bold(tranche, TRANCHE_COLORS.get(tranche, '39'))}: [{', '.join(names)}]"
                                     for tranche, names in tranches.items())

    def lines(values: dict) -> list[str]:
        return [f"{name:<{width}}  " + value.replace("\n", "\n" + " " * (width + 2)) for name, value in values.items()]

    return "\n".join([*lines(model), rule, *lines(data)])


def check_holdout(config: dict) -> None:
    """Raise if the config uses a HOLDOUT_YEARS year without "use_holdout": True."""
    used = sorted(set(HOLDOUT_YEARS) & {*config["train_years"], *config["eval_years"]})
    if used and not config.get("use_holdout", False):
        raise ValueError(f"{used} reserved in config.HOLDOUT_YEARS; set \"use_holdout\": True in the config to use them")
