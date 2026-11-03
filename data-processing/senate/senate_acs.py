"""Statewide ACS demographics for the Senate: every acs.py field for the 50 states, one row per election year.

```bash
export CENSUS_API=<your key>
python data-processing/senate/senate_acs.py   # -> data/interim/senate/acs_state.csv (then senate_features.py)
```

Each year uses the House's ACS vintage (acs.YEAR_TO_ACS_VINTAGE: the 2007 1-year for 2006, the 2016-2020 5-year for
2020; 2026 borrows 2024 1-year, acs_supplement.REFERENCE_YEAR), and the same variables, formulas and rounding: the
values are acs_supplement.fetch_reference's state rows, which the House pipeline compares districts with. The
national comparisons (rel_college, log_rel_income, rel_unemployment, rel_housing_burden) are acs_supplement's; its
rel_state_* columns would compare a state with itself, so they're left out.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "house"))
import acs
import acs_supplement

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = ROOT / "data/interim/senate/acs_state.csv"
YEARS = range(2006, 2027, 2)
FIELDS = acs.COLUMNS[acs.COLUMNS.index("acs_vintage") + 1:]  # white .. housing_burden
RELATIVE = ["rel_college", "log_rel_income", "rel_unemployment", "rel_housing_burden"]
COLUMNS = ["year", "state", "acs_vintage", *FIELDS, *RELATIVE]


def state_rows(year: int, national: pd.Series, states: pd.DataFrame) -> pd.DataFrame:
    """One election year's state rows: the fields, then the national comparisons."""
    rows = states.rename_axis("state").reset_index().assign(
        year=year, acs_vintage=acs.acs_vintage_label(acs_supplement.reference_year(year)))
    return acs_supplement.add_relative_columns(rows, national, states)[COLUMNS]


def main() -> None:
    key = acs.census_key()
    fetched = {}  # reference year -> (national, states), so 2026's borrowed 2024 vintage is fetched once
    parts = []
    for year in YEARS:
        ref = acs_supplement.reference_year(year)
        if ref not in fetched:
            fetched[ref] = acs_supplement.fetch_reference(ref, key, FIELDS)
        parts.append(state_rows(year, *fetched[ref]))
        print(f"{year}: {acs.acs_vintage_label(ref)}")
    out = pd.concat(parts, ignore_index=True).sort_values(["year", "state"], ignore_index=True)
    if out[FIELDS + RELATIVE].isna().any().any():
        raise ValueError("Missing ACS values")
    acs.write_csv_atomic(out, OUTPUT_PATH)
    print(f"-> {OUTPUT_PATH} ({len(out)} rows, {out['state'].nunique()} states)")


if __name__ == "__main__":
    main()
