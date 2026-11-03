"""The 2026 House candidate registry: one row per general-election candidate, with each race's type and method.

  data/processed/house/candidates_2026.csv
      year, state, district, candidate_id, candidate_name, party, party_label, model_party, ballot_status,
      race_type, election_method, as_of_date, data_flags, source

Built from data/raw/nominees_2026/house.csv (nominees_2026.py, a pinned Wikipedia revision that cites each state's
candidate list), fec_finance.py's reviewed matches (data/interim/house_fec_raw.csv) and FEC's 2026 candidate summary.
Nothing is invented: a candidate is listed only if the source lists them, and an ID only if an FEC record matches.

  candidate_id     the FEC candidate ID: fec_finance.py's match when the candidate fills its race's dem or gop slot
                   (reviewed there: overrides, surname and nickname matches), else a full-name match in the district
                   (fec_finance.pick with full_only). Blank when none or several match (data_flags no_fec_record /
                   fec_ambiguous)
  party            the ballot party: D (Democratic, DFL, Democratic-NPL), R (Republican) or other; party_label keeps
                   the source's label (Libertarian, Independent, Green, ...), so independents and third parties are
                   never folded into a major party
  model_party      the slot the candidate fills in the model: party, except fec_finance.INDEPENDENTS_AS_GOP, an
                   independent who is the Republican proxy for the regression and fundraising (Kiley, CA-6;
                   data_flags independent_as_gop). The ballot party stays party: Stage 2 counts his win as OTHER
  ballot_status    listed_general_election: listed among the race's general-election candidates by the source. The
                   source doesn't distinguish certified from pending candidates, so no stronger status is claimed
  race_type        house_targets.race_types (the historical targets' taxonomy) on the model_party counts per race:
                   dr / dr_multi / dr_rcv / dr_jungle, dem_vs_3p, dem_uncontested, dem_vs_dem, gop_vs_gop, ...
  election_method  rcv (RCV_STATES), jungle (JUNGLE_STATES: every party's candidates on one November ballot, as the
                   source lists them), top_two (TOP_TWO_STATES), standard otherwise
  as_of_date       the date of the pinned Wikipedia revision (SOURCE.json)

Checks: exactly 435 districts (district_counts.json), each with at least one candidate, and every
INDEPENDENTS_AS_GOP entry found once as a non-major-party candidate of its race.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # data-processing/, for fec_finance
import acs
import fec_finance
import house_targets
import incumbency

ROOT = HERE.parents[1]
NOMINEES_PATH = ROOT / "data/raw/nominees_2026/house.csv"
SOURCE_PATH = ROOT / "data/raw/nominees_2026/SOURCE.json"
FEC_2026_PATH = ROOT / "data/raw/fec_candidate_summaries/candidate_summary_2026.csv"
FEC_SLOTS_PATH = ROOT / "data/interim/house_fec_raw.csv"  # fec_finance.py
DISTRICT_COUNTS_PATH = ROOT / "data/reference/district_counts.json"
OUTPUT_PATH = ROOT / "data/processed/house/candidates_2026.csv"
YEAR = 2026
RCV_STATES = {"AK", "ME"}       # ranked-choice general elections
JUNGLE_STATES = {"LA"}          # all-candidate November round (several candidates per party on the ballot)
TOP_TWO_STATES = {"CA", "WA"}   # top-two primaries: the general can pit two candidates of one party
BALLOT_STATUS = "listed_general_election"
COLUMNS = ["year", "state", "district", "candidate_id", "candidate_name", "party", "party_label", "model_party",
           "ballot_status", "race_type", "election_method", "as_of_date", "data_flags", "source"]


def election_method(state: str) -> str:
    if state in RCV_STATES:
        return "rcv"
    if state in JUNGLE_STATES:
        return "jungle"
    return "top_two" if state in TOP_TWO_STATES else "standard"


def model_parties(nominees: pd.DataFrame) -> pd.Series:
    """party, with fec_finance.INDEPENDENTS_AS_GOP's independents as R."""
    out = nominees["party"].copy()
    for (chamber, year, state, district), name in fec_finance.INDEPENDENTS_AS_GOP.items():
        if chamber != "house" or year != YEAR:
            continue
        hit = (nominees["state"] == state) & (nominees["district"] == district) & (nominees["name"] == name) \
            & (nominees["party"] == "other")
        if hit.sum() != 1:
            raise ValueError(f"INDEPENDENTS_AS_GOP {(state, district)}: {name!r} isn't one non-major-party candidate")
        out[hit] = "R"
    return out


def race_types(registry: pd.DataFrame) -> pd.Series:
    """Each candidate's race_type: house_targets.race_types on the race's model_party counts."""
    key = ["state", "district"]
    counts = (registry.groupby(key)["model_party"].value_counts().unstack(fill_value=0)
              .reindex(columns=["D", "R", "other"], fill_value=0)
              .rename(columns={"D": "dem", "R": "gop", "other": "3p"}).assign(cross=0))
    states = counts.index.get_level_values("state")
    by_race = house_targets.race_types(counts, writein_only=pd.Series(False, index=counts.index),
                                       rcv=pd.Series(states.isin(RCV_STATES), index=counts.index),
                                       jungle=pd.Series(states.isin(JUNGLE_STATES), index=counts.index))
    return registry.join(by_race.rename("race_type"), on=key)["race_type"]


def slot_ids(fec_slots: pd.DataFrame) -> dict:
    """{(state, district, name): candidate_id} for 2026's dem and gop slots that fec_finance.py matched."""
    out = {}
    rows = fec_slots[fec_slots["year"] == YEAR]
    for slot in ["dem", "gop"]:
        matched = rows[rows[f"{slot}_candidate_id"].notna()]
        out |= dict(zip(zip(matched["state"], matched["district"], matched[f"{slot}_name"]),
                        matched[f"{slot}_candidate_id"]))
    return out


def fec_ids(registry: pd.DataFrame, fec: pd.DataFrame, slots: dict, at_large: set) -> tuple[pd.Series, pd.Series]:
    """(candidate_id, flag) per candidate: fec_finance.py's slot match, else a full-name FEC match in the district,
    else blank and a flag."""
    ids, flags = [], []
    for row in registry.itertuples():
        if (row.state, row.district, row.candidate_name) in slots:
            ids.append(slots[(row.state, row.district, row.candidate_name)])
            flags.append(None)
            continue
        districts = {0, 1} if row.state in at_large else {row.district}
        pool = fec[(fec["office"] == "H") & (fec["state"] == row.state) & fec["district"].isin(districts)]
        outcome, records = fec_finance.pick(row.candidate_name, pool, full_only=True)
        if outcome in ("full", "variant", "duplicate"):
            ids.append(records["candidate_id"].iloc[0])
            flags.append(None)
        else:
            ids.append(None)
            flags.append("fec_ambiguous" if outcome == "ambiguous" else "no_fec_record")
    return pd.Series(ids, index=registry.index), pd.Series(flags, index=registry.index)


def build_registry(nominees: pd.DataFrame, fec: pd.DataFrame, fec_slots: pd.DataFrame, as_of_date: str,
                   at_large: set) -> pd.DataFrame:
    """The registry from nominees_2026's House rows, FEC's 2026 records (read_candidate_summary), and
    fec_finance.py's slot matches (house_fec_raw.csv)."""
    out = nominees.rename(columns={"name": "candidate_name"}).copy()
    out["model_party"] = model_parties(nominees)
    out["ballot_status"] = BALLOT_STATUS
    out["race_type"] = race_types(out)
    out["election_method"] = out["state"].map(election_method)
    out["as_of_date"] = as_of_date
    out["candidate_id"], fec_flag = fec_ids(out, fec, slot_ids(fec_slots), at_large)
    proxy = ["independent_as_gop" if m != p else None for m, p in zip(out["model_party"], out["party"])]
    out["data_flags"] = [";".join(f for f in flags if isinstance(f, str)) or None for flags in zip(fec_flag, proxy)]
    return out[COLUMNS].sort_values(["state", "district", "party", "candidate_name"]).reset_index(drop=True)


def main() -> None:
    nominees = pd.read_csv(NOMINEES_PATH)
    as_of_date = json.loads(SOURCE_PATH.read_text())["house"]["revision_timestamp"][:10]
    counts = acs.normalize_counts(json.loads(DISTRICT_COUNTS_PATH.read_text()), YEAR)
    expected = set(incumbency.expected_districts(counts))
    races = set(zip(nominees["state"], nominees["district"]))
    if races != expected:
        raise ValueError(f"nominees cover {len(races)} districts, not 2026's {len(expected)}: missing "
                         f"{sorted(expected - races)[:5]}, extra {sorted(races - expected)[:5]}")
    fec = fec_finance.read_candidate_summary(FEC_2026_PATH, YEAR)
    at_large = {s for s, n in counts.items() if n == 1}
    registry = build_registry(nominees, fec, pd.read_csv(FEC_SLOTS_PATH), as_of_date, at_large)

    by_race = registry.drop_duplicates(["state", "district"])
    print(f"{len(registry)} candidates in {len(by_race)} districts, as of {as_of_date}")
    print("race_type: " + ", ".join(f"{k} {v}" for k, v in by_race["race_type"].value_counts().items()))
    print("election_method: " + ", ".join(f"{k} {v}" for k, v in by_race["election_method"].value_counts().items()))
    flags = registry["data_flags"].dropna().str.split(";").explode()
    print("data_flags: " + ", ".join(f"{k} {v}" for k, v in flags.value_counts().items()))
    major = registry[registry["party"].isin(["D", "R"]) & registry["data_flags"].fillna("").str.contains("fec")]
    print(f"Major-party candidates without a unique FEC ID ({len(major)}):")
    print(major[["state", "district", "party", "candidate_name", "data_flags"]].to_string(index=False))
    acs.write_csv_atomic(registry, OUTPUT_PATH)
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
