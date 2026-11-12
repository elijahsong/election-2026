"""Merge, validate and write the congress-track submission from the House and Senate 2026 track rows.

Called from the production entry point, notebooks/full-forecast/full_forecast_2026.ipynb:

    import congress_track
    congress_track.write_submission({"House": house_rows, "Senate": senate_rows})

Inputs
  data/reference/full congress track template.csv   race_id, chamber (435 House, 35 Senate rows), blank predictions
  each chamber's rows                                race_id, chamber, predicted_winner_party, candidate_name,
                                                     win_probability (house.forecast / senate.forecast track)
Output (identical files)
  data/processed/full_congress_track_2026.csv        the template's rows and order, every prediction filled
  data/processed/full congress track template.csv    the same file under the template's name, for submission

merge refuses to write unless each chamber covers exactly the template's rows for that chamber, once each, with a
party (D, R or I), a candidate name and a win probability between 0.5 and 1. write_submission rounds probabilities to
6 decimals, so a rerun of the same forecast writes byte-identical files.

Manual judgment enters before the simulation, as margin adjustments (src/congress/adjustments.py), never here: this
module changes no value. The earlier probability overrides (data/reference/track_overrides_2026.csv) are retired and
no longer applied.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "data/reference/full congress track template.csv"
OUTPUT_PATH = ROOT / "data/processed/full_congress_track_2026.csv"
SUBMISSION_PATH = ROOT / "data/processed/full congress track template.csv"  # the template's name; never the input
COLUMNS = ["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability"]
PROBABILITY_DECIMALS = 6


def merge(template: pd.DataFrame, chambers: dict[str, pd.DataFrame]) -> pd.DataFrame:
    problems = []
    for chamber, rows in chambers.items():
        expected = template.loc[template["chamber"] == chamber, "race_id"]
        mismatch = set(rows["race_id"]) != set(expected) or (rows["chamber"] != chamber).any()
        if rows["race_id"].duplicated().any() or mismatch:
            problems.append(f"{chamber}: rows don't match the template's {len(expected)} {chamber} races")
        if rows[COLUMNS[2:]].isna().any().any():
            problems.append(f"{chamber}: blank party, candidate or probability")
        if not rows["predicted_winner_party"].isin(["D", "R", "I"]).all():
            problems.append(f"{chamber}: party codes other than D, R, I")
        if not rows["win_probability"].between(0.5, 1).all():
            problems.append(f"{chamber}: win probabilities outside [0.5, 1]")
    if set(template["chamber"]) != set(chambers):
        problems.append(f"template chambers {sorted(set(template['chamber']))} != {sorted(chambers)}")
    if problems:
        raise ValueError("\n".join(problems))
    filled = pd.concat(chambers.values(), ignore_index=True)[COLUMNS]
    out = template[["race_id", "chamber"]].merge(filled, on=["race_id", "chamber"], how="left", validate="1:1")
    assert len(out) == len(template) and out.notna().all().all()
    return out


def write_submission(chambers: dict[str, pd.DataFrame], template: pd.DataFrame | None = None,
                     paths=(OUTPUT_PATH, SUBMISSION_PATH)) -> pd.DataFrame:
    """merge the chambers' rows into the template (read from TEMPLATE_PATH unless given), round the probabilities,
    write each of `paths`, and return the submission."""
    template = pd.read_csv(TEMPLATE_PATH) if template is None else template
    out = merge(template, chambers)
    out["win_probability"] = out["win_probability"].round(PROBABILITY_DECIMALS)
    for path in paths:
        out.to_csv(path, index=False)
    return out
