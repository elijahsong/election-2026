"""Fill the full congress-track template with the House and Senate 2026 forecasts.

Inputs
  data/reference/full congress track template.csv   race_id, chamber (435 House, 35 Senate rows), blank predictions
  data/processed/house/congress_track_house_2026.csv   House rows: the H0 simulation on Stage 1's AB 85/15 ensemble
                                                       (notebooks/house/stage2/02_monte_carlo.ipynb)
  data/processed/senate/congress_track_senate_2026.csv Senate rows: S-M0 (notebooks/senate/stage2/03_monte_carlo.ipynb)
  data/reference/track_overrides_2026.csv             manual changes for the submission (apply_overrides)
Output
  data/processed/full_congress_track_2026.csv          the template's rows and order, every prediction filled, with
                                                       the overrides applied
  data/processed/full congress track template.csv      the same file under the template's name, for submission

Each chamber's file must cover exactly the template's rows for that chamber, with a party (D, R or I), a candidate
name and a win probability between 0.5 and 1 (the predicted winner's share of draws). The script refuses to write
otherwise. It changes no model values; only the rows of the overrides file differ from the chamber files.

Overrides (apply_overrides): each row names a race, the submitted party, candidate and probability, the model's
values it replaces (model_party, model_candidate, model_probability), and a reason. An override whose model values
no longer match the current forecast is stale and stops the script, so a re-run model never silently keeps an old
manual change. The chamber files (the model's own forecasts) are never edited.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "data/reference/full congress track template.csv"
CHAMBER_PATHS = {
    "House": ROOT / "data/processed/house/congress_track_house_2026.csv",
    "Senate": ROOT / "data/processed/senate/congress_track_senate_2026.csv",
}
OVERRIDES_PATH = ROOT / "data/reference/track_overrides_2026.csv"
OUTPUT_PATH = ROOT / "data/processed/full_congress_track_2026.csv"
SUBMISSION_PATH = ROOT / "data/processed/full congress track template.csv"  # the template's name; never the input
COLUMNS = ["race_id", "chamber", "predicted_winner_party", "candidate_name", "win_probability"]


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


def apply_overrides(track: pd.DataFrame, overrides: pd.DataFrame) -> pd.DataFrame:
    """track with each override's party, candidate and probability, after checking its model values are current."""
    out = track.set_index("race_id")
    for row in overrides.itertuples(index=False):
        current = out.loc[row.race_id]
        if (current["chamber"], current["predicted_winner_party"], current["candidate_name"]) != \
                (row.chamber, row.model_party, row.model_candidate) or \
                abs(current["win_probability"] - row.model_probability) > 1e-6:
            raise ValueError(f"Stale override for {row.race_id}: the model now says "
                             f"{current[['predicted_winner_party', 'candidate_name', 'win_probability']].tolist()}")
        if row.predicted_winner_party not in ("D", "R", "I") or not 0.5 <= row.win_probability <= 1:
            raise ValueError(f"Override for {row.race_id}: invalid party or probability")
        out.loc[row.race_id, COLUMNS[2:]] = [row.predicted_winner_party, row.candidate_name, row.win_probability]
    return out.reset_index()[COLUMNS]


def main() -> None:
    template = pd.read_csv(TEMPLATE_PATH)
    model = merge(template, {chamber: pd.read_csv(path) for chamber, path in CHAMBER_PATHS.items()})
    overrides = pd.read_csv(OVERRIDES_PATH)
    out = apply_overrides(model, overrides)
    print(f"{len(overrides)} overrides: " + "; ".join(
        f"{r.race_id} {r.model_party} {r.model_probability:.3f} -> {r.predicted_winner_party} {r.candidate_name} "
        f"{r.win_probability:.3f}" for r in overrides.itertuples()))
    for path in (OUTPUT_PATH, SUBMISSION_PATH):
        out.to_csv(path, index=False)
        print(f"-> {path} ({len(out)} rows)")
    print(out.groupby(["chamber", "predicted_winner_party"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
