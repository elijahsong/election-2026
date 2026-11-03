"""Senate expert ratings from Sabato's Crystal Ball, Cook Political Report and Inside Elections, one row per race,
with a weighted consensus: data/processed/senate/expert-ratings.csv. Run after incumbency.py and fec_finance.py.

Sources
  2016-2024  data/raw/senate_ratings_historical/output.csv, JHK Forecasts' forecast archive (forecast sabato, cook,
             inside; office senate). Only forecast, year, office, s_id, probwin and rating are read: the archive's
             outcome columns (probwin_outcome, correct, brier, act_val, ...) are never loaded.
  2026       data/reference/ratings_2026.csv, entered by hand (state, class, cook_rating, sabato_rating, ie_rating).
Neither gives the date of the ratings, so as_of_date is blank. JHK's are the final pre-election ratings; the 2026
file is as entered.

expert-ratings.csv
  year, state, class, special, election_id  the race (senate_polls.race_table); every race 2016-2026
  <provider>_rating    sabato, cook, ie: the rating as published (Inside Elections' Tilt categories kept)
  <provider>_prob_dem  1 - RATING_GOP_SCORE[rating]. JHK's probwin is the Republican's score, so the Democratic score
                       is always its complement. HEURISTIC ENCODINGS OF ORDINAL RATINGS, NOT CALIBRATED WIN
                       PROBABILITIES, and not vote margins
  expert_consensus_dem sum(WEIGHTS[p] * <p>_prob_dem) / sum(WEIGHTS[p]) over the providers that rated the race:
                       0.40 Sabato, 0.30 Cook, 0.30 Inside Elections when all three did; renormalized otherwise. The
                       weights only combine the three providers; they aren't weights against polls or fundamentals
  n_experts            providers that rated the race; coverage_complete is n_experts == 3 (missing_<provider> flags
                       name the gaps)
  quality_flags        race_table's nominee exceptions. independent_vs_gop (UT 2022, NE 2024 class 1, NE 2026):
                       the Republican's opponent is an independent, so 1 - prob_gop is the probability a
                       non-Republican wins, not a Democrat. independent_counted_as_dem: King and Sanders, as in
                       targets.csv. no_dem_nominee (AR 2020, ID and SD 2026): no Democrat either, so
                       1 - prob_gop is again the probability a non-Republican wins. no_gop_nominee (CA)
  source               JHK archive or manual entry
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from senate_polls import ELECTION_DATES_PATH, INCUMBENCY_PATH, NOMINEES_PATH, race_table
from senate_schema import EXPERT_RATINGS, KEY, OUTPUT_DIR, write_outputs

ROOT = Path(__file__).resolve().parents[2]
JHK_PATH = ROOT / "data/raw/senate_ratings_historical/output.csv"
RATINGS_2026_PATH = ROOT / "data/reference/ratings_2026.csv"
YEARS = range(2016, 2027, 2)

# The Republican's score for each rating, as in JHK's probwin (checked against every archive row)
RATING_GOP_SCORE = {
    "Solid D": 0.01, "Likely D": 0.10, "Lean D": 0.25, "Tilt D": 0.40, "Tossup": 0.50,
    "Tilt R": 0.60, "Lean R": 0.75, "Likely R": 0.90, "Solid R": 0.99,
}
# provider -> (JHK forecast name, consensus weight)
PROVIDERS = {"sabato": ("sabato", 0.40), "cook": ("cook", 0.30), "ie": ("inside", 0.30)}
WEIGHTS = {p: w for p, (_, w) in PROVIDERS.items()}

JHK_SOURCE = "JHK Forecasts forecast archive (data/raw/senate_ratings_historical/output.csv)"
MANUAL_SOURCE = "manual entry (data/reference/ratings_2026.csv)"


def prob_dem(rating: pd.Series) -> pd.Series:
    """1 - the rating's Republican score; blank where there's no rating."""
    unknown = set(rating.dropna()) - set(RATING_GOP_SCORE)
    if unknown:
        raise ValueError(f"Ratings without a score: {sorted(unknown)}")
    return (1 - rating.map(RATING_GOP_SCORE)).round(2)


def jhk_ratings(jhk: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """year, state, class and one <provider>_rating column per provider, from the archive's senate rows. s_id is the
    state, or "<state>-special" / "NES" (NE 2024) for a special, matched to the state-year's regular or special
    race."""
    names = {name: p for p, (name, _) in PROVIDERS.items()}
    df = jhk[(jhk["office"] == "senate") & jhk["forecast"].isin(names)].copy()
    off = (df["probwin"] - df["rating"].map(RATING_GOP_SCORE)).abs() > 1e-9
    if off.any():
        raise ValueError(f"probwin isn't the rating's Republican score in {int(off.sum())} rows, e.g. "
                         f"{df.loc[off, ['forecast', 'year', 's_id', 'rating', 'probwin']].head(3).values.tolist()}")
    df["provider"] = df["forecast"].map(names)
    df["state"] = df["s_id"].str[:2]
    df["special"] = df["s_id"].str.endswith("-special") | (df["s_id"] == "NES")
    df = df.merge(races[["year", "state", "class", "special"]], on=["year", "state", "special"], how="left",
                  validate="many_to_one")
    if df["class"].isna().any():
        raise ValueError(f"JHK races not in senate_incumbency.csv: "
                         f"{df.loc[df['class'].isna(), ['year', 's_id']].drop_duplicates().values.tolist()}")
    if df.duplicated([*KEY, "provider"]).any():
        raise ValueError("A provider rates a race twice in the JHK archive")
    df["class"] = df["class"].astype(int)
    wide = df.pivot(index=KEY, columns="provider", values="rating").add_suffix("_rating").reset_index()
    return wide.assign(source=JHK_SOURCE)


def manual_ratings(ratings: pd.DataFrame, year: int) -> pd.DataFrame:
    if ratings.duplicated(["state", "class"]).any():
        raise ValueError(f"A race is rated twice in {RATINGS_2026_PATH.name}")
    return ratings.assign(year=year, source=MANUAL_SOURCE)


def build_ratings(rated: pd.DataFrame, races: pd.DataFrame) -> pd.DataFrame:
    """One row per race in YEARS, with each provider's rating and score, the consensus and its coverage."""
    out = races[races["year"].isin(YEARS)].merge(rated, on=KEY, how="left", validate="one_to_one")
    unmatched = rated.merge(out[KEY], on=KEY, how="left", indicator=True)
    if (unmatched["_merge"] != "both").any():
        raise ValueError(f"Rated races not in senate_incumbency.csv: "
                         f"{unmatched.loc[unmatched['_merge'] != 'both', KEY].values.tolist()}")
    for p in PROVIDERS:
        out[f"{p}_prob_dem"] = prob_dem(out[f"{p}_rating"])
    probs = out[[f"{p}_prob_dem" for p in PROVIDERS]].set_axis(list(PROVIDERS), axis=1)
    weights = probs.notna() * pd.Series(WEIGHTS)
    out["expert_consensus_dem"] = (probs.fillna(0) * weights).sum(axis=1) / weights.sum(axis=1).where(
        weights.sum(axis=1) > 0)
    out["n_experts"] = probs.notna().sum(axis=1).astype("int64")
    out["coverage_complete"] = out["n_experts"] == len(PROVIDERS)
    out["as_of_date"] = None  # no source dates its ratings
    missing = probs.isna().apply(lambda row: [f"missing_{p}" for p in row.index[row]], axis=1)
    out["quality_flags"] = [";".join(([f] if isinstance(f, str) else []) + m) or None
                            for f, m in zip(out["quality_flags"], missing)]
    return out.sort_values(KEY, ignore_index=True)[list(EXPERT_RATINGS)]


def report(ratings: pd.DataFrame) -> None:
    summary = ratings.groupby("year").agg(races=("n_experts", "size"), complete=("coverage_complete", "sum"),
                                          any_rating=("n_experts", lambda n: int((n > 0).sum())))
    print("expert-ratings.csv coverage by year:\n" + summary.to_string())
    incomplete = ratings[~ratings["coverage_complete"]]
    print(f"incomplete coverage ({len(incomplete)}): " + (", ".join(incomplete["election_id"]) or "none"))
    flagged = ratings[ratings["quality_flags"].notna()]
    print("flags: " + "; ".join(f"{e} {f}" for e, f in zip(flagged["election_id"], flagged["quality_flags"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=OUTPUT_DIR, type=Path)
    args = parser.parse_args()

    races = race_table(pd.read_csv(INCUMBENCY_PATH), pd.read_csv(NOMINEES_PATH),
                       json.loads(ELECTION_DATES_PATH.read_text()))
    jhk = pd.read_csv(JHK_PATH, usecols=["forecast", "year", "office", "s_id", "probwin", "rating"])
    rated = pd.concat([jhk_ratings(jhk, races), manual_ratings(pd.read_csv(RATINGS_2026_PATH), 2026)],
                      ignore_index=True)
    ratings = build_ratings(rated, races)
    report(ratings)
    write_outputs({"expert-ratings.csv": ratings}, args.output_dir)


if __name__ == "__main__":
    sys.exit(main())
