"""Lagged Split Ticket WAR: each race's WAR signal from its nominees' earlier races (an experiment; production unchanged).

Inputs are data/processed/split_ticket_war.csv and split_ticket_war_people.csv (data-processing/split_ticket_war.py),
and the nominee slots of house_fec.csv / senate_fec.csv, whose FEC IDs identify each race's D and R nominees in every
year, 2026 included.

    w_r     = war_dem_r - mean(war_dem over the chamber's WAR races in r's year)     centered WAR of past race r
    v_{r,c} = s_{c,r} * s_{c,now} * w_r                                              seen from the current race
    L       = v_recent                                  one earlier year with WAR
              0.75 v_recent + 0.25 v_previous           two or more (the two most recent years)
              missing (war_lag_available False)         none; with_war_lag puts a neutral 0 into the fit

Choices, all fixed in advance:
- Whose races: every earlier WAR race in which either current nominee (D or R slot, by person_id) was a usable D or
  R nominee, in the same chamber. Senate candidates' House races don't count: the two chambers' WAR come from
  different baselines, and FEC IDs differ by office.
- Sign: war_dem is D positive. s is +1 in the D slot and -1 in the R slot, so a Republican's past over-performance
  counts against the Democrat now, and a party switcher's (Van Drew, D in 2018, R since) flips sign.
- A rematch (both nominees in the same earlier race) counts that race once. Two different earlier races in the same
  year (each nominee ran separately) count as that year's mean, so a year is one observation.
- Centering: Senate WAR averages about D+2 in every cycle (Split Ticket fits the Senate baseline on the House races of
  the cycle), a national offset that isn't about the candidate; centering by chamber-year removes it. House WAR
  already averages within 0.25 points of 0 each year, so centering barely changes it.
- No leakage: only years before the current race, so a race's own WAR never enters its lag. What Split Ticket's
  revised WAR for a past year may itself contain is a limitation of the source (data-processing/README.md).
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
WAR_PATH = ROOT / "data/processed/split_ticket_war.csv"
PEOPLE_PATH = ROOT / "data/processed/split_ticket_war_people.csv"
FEC_PATHS = {"House": ROOT / "data/processed/house_fec.csv", "Senate": ROOT / "data/processed/senate_fec.csv"}
KEYS = {"House": ["year", "state", "district"], "Senate": ["year", "state", "class"]}
WEIGHTS = (0.75, 0.25)  # most recent, second most recent; fixed, not tuned
LAG = "war_lag"
AVAILABLE = "war_lag_available"
SIDES = {"dem": 1, "gop": -1}


def load_war(path=WAR_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def load_people(path=PEOPLE_PATH) -> dict[str, str]:
    """{candidate_id: person_id} for the FEC IDs that belong to one person (split_ticket_war_people.csv)."""
    people = pd.read_csv(path)
    return dict(zip(people["candidate_id"], people["person_id"]))


def appearances(war: pd.DataFrame, chamber: str) -> pd.DataFrame:
    """One row per usable (WAR race, side) of the chamber: year, geography, side, person_id, war_dem and war_centered
    (war_dem minus the mean of the chamber's WAR races that year, all of them, usable or not)."""
    rows = war[war["chamber"] == chamber].copy()
    rows["war_centered"] = rows["war_dem"] - rows.groupby("year")["war_dem"].transform("mean")
    frames = [rows.loc[rows[f"{side}_usable"], ["year", "geography", "war_dem", "war_centered"]]
              .assign(side=side, person_id=rows.loc[rows[f"{side}_usable"], f"{side}_person_id"])
              for side in SIDES]
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["person_id", "year"]).any():
        raise ValueError(f"{chamber}: a person with two usable WAR races in one year")
    return out


def race_nominees(fec: pd.DataFrame, chamber: str, people: dict[str, str]) -> pd.DataFrame:
    """KEY, dem_person_id, gop_person_id and names for every race of house_fec.csv / senate_fec.csv (2006-2026)."""
    out = fec[[*KEYS[chamber], "dem_name", "gop_name"]].copy()
    for side in SIDES:
        out[f"{side}_person_id"] = fec[f"{side}_candidate_id"].map(lambda c: people.get(c, c))
    return out


def lagged_war(nominees: pd.DataFrame, apps: pd.DataFrame, key: list[str], weights=WEIGHTS) -> pd.DataFrame:
    """For every race in nominees: war_lag (NaN without an earlier WAR year), war_lag_available, war_n_years (0, 1,
    2: the years used), war_recent_year / war_previous_year and their values, war_from (whose races: dem, gop, both
    (a rematch, or each in its own race), party_switch), and war_flag, a one-line description."""
    by_person = {pid: g for pid, g in apps.groupby("person_id")}
    rows = []
    for race in nominees.to_dict("records"):  # records, not itertuples: "class" is a keyword
        values = []
        for side, sign_now in SIDES.items():
            pid = race[f"{side}_person_id"]
            history = by_person.get(pid) if isinstance(pid, str) else None
            if history is None:
                continue
            earlier = history[history["year"] < race["year"]]
            for past in earlier.itertuples(index=False):
                values.append({"year": past.year, "geography": past.geography, "from": side,
                               "switch": SIDES[past.side] != sign_now,
                               "value": SIDES[past.side] * sign_now * past.war_centered})
        row = {k: race[k] for k in key}
        row.update({LAG: np.nan, "war_n_years": 0, "war_recent_year": np.nan, "war_recent": np.nan,
                    "war_previous_year": np.nan, "war_previous": np.nan, "war_from": None, "war_party_switch": False})
        if not values:
            row["war_flag"] = "neutral: no earlier WAR race of either nominee"
        else:
            values = pd.DataFrame(values)
            per_race = values.groupby(["year", "geography"]).agg(value=("value", "mean"), sides=("from", "nunique"))
            per_year = per_race.groupby("year")["value"].mean().sort_index(ascending=False)
            recent = per_year.iloc[:2]
            row.update({"war_n_years": len(recent), "war_recent_year": recent.index[0], "war_recent": recent.iloc[0],
                        "war_party_switch": bool(values["switch"].any()),
                        "war_from": "both" if values["from"].nunique() > 1 else values["from"].iloc[0]})
            if len(recent) == 1:
                row[LAG] = recent.iloc[0]
                row["war_flag"] = f"one year ({recent.index[0]})"
            else:
                row.update({"war_previous_year": recent.index[1], "war_previous": recent.iloc[1],
                            LAG: weights[0] * recent.iloc[0] + weights[1] * recent.iloc[1]})
                row["war_flag"] = f"two years ({recent.index[0]}, {recent.index[1]})"
        rows.append(row)
    out = pd.DataFrame(rows)
    out[AVAILABLE] = out[LAG].notna()
    return out


def chamber_lags(chamber: str, weights=WEIGHTS) -> pd.DataFrame:
    """lagged_war for every race of the chamber's nominee table (2006-2026), from the processed WAR files."""
    fec = pd.read_csv(FEC_PATHS[chamber], low_memory=False)
    nominees = race_nominees(fec, chamber, load_people())
    lags = lagged_war(nominees, appearances(load_war(), chamber), KEYS[chamber], weights)
    return nominees.merge(lags, on=KEYS[chamber], validate="one_to_one")


def with_war_lag(frame: pd.DataFrame, lags: pd.DataFrame, key: list[str], cap: float | None = None) -> pd.DataFrame:
    """frame plus war_lag (a neutral 0 where missing, so no race is dropped), war_lag_available and war_flag. Every
    frame row must have a lags row. cap, if given, clips war_lag to [-cap, cap] (a sensitivity check)."""
    out = frame.merge(lags[[*key, LAG, AVAILABLE, "war_flag", "war_from"]], on=key, how="left",
                      validate="one_to_one", indicator=True)
    if (out["_merge"] != "both").any():
        missing = out.loc[out["_merge"] != "both", key].head().values.tolist()
        raise ValueError(f"Races without a WAR lag row: {missing}")
    out[LAG] = out[LAG].fillna(0.0)
    if cap is not None:
        out[LAG] = out[LAG].clip(-cap, cap)
    return out.drop(columns="_merge")


def house_war_features(lags: pd.DataFrame) -> pd.DataFrame:
    """data/processed/house/war_f.csv, the modeling.csv tranche: KEY, war_lag (the neutral 0 where missing) and
    war_lag_available, one row per district-year of house_fec.csv."""
    key = KEYS["House"]
    out = lags[[*key, LAG, AVAILABLE]].copy()
    out[LAG] = out[LAG].fillna(0.0)
    return out.sort_values(key, ignore_index=True)


def score(rows: pd.DataFrame, predictions: dict[str, str], subsets: dict[str, pd.Series],
          actual: str = "actual") -> pd.DataFrame:
    """MAE, bias (actual - prediction; negative = too Democratic) and winners called, for each prediction column
    (predictions: {label: column}) on each subset of rows ({name: boolean mask aligned with rows}); one row per
    (subset, label). A called race needs a nonzero prediction and a nonzero actual."""
    table = []
    for subset, mask in subsets.items():
        part = rows[mask.to_numpy()]
        for label, column in predictions.items():
            err = part[actual] - part[column]
            called = (part[column] != 0) & (part[actual] != 0)
            table.append({"subset": subset, "model": label, "n": len(part), "mae": err.abs().mean(),
                          "bias": err.mean(), "winners_called_pct": 100 * (
                              np.sign(part.loc[called, column]) == np.sign(part.loc[called, actual])).mean()})
    return pd.DataFrame(table).set_index(["subset", "model"])
