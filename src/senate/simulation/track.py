"""The congress-track submission's Senate rows: race_id, chamber, predicted_winner_party, candidate_name,
win_probability (template: data/reference/full congress track template.csv, race_id "<state>-SEN").

Candidate names (resolve_candidates), per 2026 race:
    D and R   the model's nominee slots in data/processed/senate_fec.csv (dem_name, gop_name): the candidates Stage 1
              and Stage 2 were built for. This matters in AK, whose ballot has two Dan Sullivans; the slot is the
              incumbent, Dan S. Sullivan
    I         for an independent-vs-R race: senate_fec.csv's ind_name (NE: Dan Osborn), else the only "Independent"
              on the ballot (data/raw/nominees_2026/senate.csv). Several independents and no slot (ID: Achilles and
              Fleming) leave it unresolved (independent_resolution says why); track_table refuses to name an
              unresolved candidate as a predicted winner

Party and probability (track_table), from the simulated margins (> 0: the non-Republican wins):
    predicted_winner_party  D (a Democratic nominee's race) or I (an independent-only race) when that side wins more
                            than half the draws, else R. Coded as the template does: D, R, I
    win_probability         the predicted winner's share of draws
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TRACK_TEMPLATE_PATH = ROOT / "data/reference/full congress track template.csv"
TRACK_OUTPUT_PATH = ROOT / "data/processed/senate/congress_track_senate_2026.csv"
NOMINEES_PATH = ROOT / "data/processed/senate_fec.csv"
BALLOT_PATH = ROOT / "data/raw/nominees_2026/senate.csv"
KEY = ["year", "state", "class"]


def resolve_candidates(point: pd.DataFrame, nominees: pd.DataFrame, ballot: pd.DataFrame) -> pd.DataFrame:
    """point (one row per race) plus dem_candidate, gop_candidate, ind_candidate and independent_resolution."""
    slots = nominees[[*KEY, "dem_name", "gop_name", "ind_name"]]
    out = point.merge(slots, on=KEY, how="left", validate="one_to_one")
    out = out.rename(columns={"dem_name": "dem_candidate", "gop_name": "gop_candidate", "ind_name": "ind_candidate"})
    independents = ballot[ballot["party_label"] == "Independent"].groupby(KEY)["name"].agg(list)
    resolution = []
    for i, row in out.iterrows():
        if row["seat_type"] != "independent_only":
            resolution.append(None)
        elif pd.notna(row["ind_candidate"]):
            resolution.append("senate_fec.csv ind_name")
        else:
            names = independents.get(tuple(row[KEY]), [])
            if len(names) == 1:
                out.loc[i, "ind_candidate"] = names[0]
                resolution.append("the only Independent on the ballot")
            else:
                resolution.append(f"unresolved: {len(names)} independents on the ballot ({', '.join(names)})")
    out["independent_resolution"] = resolution
    if out.loc[out["seat_type"] == "dem_nominee", ["dem_candidate", "gop_candidate"]].isna().any().any():
        raise ValueError("A D-vs-R race lacks a Democratic or Republican nominee slot")
    return out


def track_table(point: pd.DataFrame, sims: np.ndarray, template: pd.DataFrame) -> pd.DataFrame:
    """The template's Senate rows, in its order, from resolve_candidates' point and the (n_sims, n_races) margins."""
    p_non_r = (sims > 0).mean(axis=0)
    challenger = np.where(point["seat_type"] == "independent_only", "I", "D")
    non_r_wins = p_non_r > 0.5
    party = np.where(non_r_wins, challenger, "R")
    name = np.select([party == "R", party == "I"], [point["gop_candidate"], point["ind_candidate"]],
                     point["dem_candidate"])
    out = pd.DataFrame({"race_id": point["state"] + "-SEN", "chamber": "Senate", "predicted_winner_party": party,
                        "candidate_name": name,
                        "win_probability": np.where(non_r_wins, p_non_r, 1 - p_non_r).round(6)})
    if pd.isna(out["candidate_name"]).any():
        missing = out.loc[pd.isna(out["candidate_name"]), "race_id"].tolist()
        raise ValueError(f"A predicted winner has no resolved candidate: {missing}")
    senate = template.loc[template["chamber"] == "Senate", ["race_id"]]
    if set(senate["race_id"]) != set(out["race_id"]) or len(out) != len(senate):
        raise ValueError("The forecast's races don't match the template's Senate rows")
    return senate.merge(out, on="race_id", how="left", validate="1:1")
