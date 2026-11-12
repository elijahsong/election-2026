"""Parse the pasted MS Senate special county results table and estimate outstanding votes.

Just for fun; not part of the forecast pipeline.

    python extra/ms_special_counties.py extra/ms-special-counties-20261107.txt

Writes <input>.csv next to the input.

Source quirks:
- Columns are separated by runs of spaces; county names can contain one space ("Pearl River").
- Margins are rounded percentage points ("Colom +61", "Hyde-Smith +0.11"); stored signed,
  Colom (D) positive.
- "% In" is a rounded estimate, and ">95%" is open-ended. We keep a range:
  ">95%" -> 95 to 100; a plain "93%" -> 93 to 93.

Outstanding votes per county = votes / pct_in - votes, at each end of the range. The
"projected net" columns assume the outstanding votes split like the county's votes so far;
see the notebook for why that probably understates Colom.
"""

import re
import sys
from pathlib import Path

import pandas as pd

DEM, REP = "Colom", "Hyde-Smith"


def parse_margin(text):
    name, amount = text.rsplit(" +", 1)
    return float(amount) if name == DEM else -float(amount)


def parse_pct_in(text):
    """'>95%' -> (95, 100), '93%' -> (93, 93)."""
    pct = float(text.strip(">%"))
    return (pct, 100.0) if text.startswith(">") else (pct, pct)


def load_counties(path):
    lines = Path(path).read_text().splitlines()[2:]  # skip the two header lines
    rows = [re.split(r"\s{2,}", line.strip()) for line in lines if line.strip()]
    raw = pd.DataFrame(rows, columns=["county", "margin", "votes", "pct_in"])

    df = pd.DataFrame({"county": raw["county"]})
    df["margin_pct"] = raw["margin"].map(parse_margin)
    df["votes"] = raw["votes"].str.replace(",", "").astype(int)
    df["pct_in"] = raw["pct_in"]
    pct_range = raw["pct_in"].map(parse_pct_in)
    df["pct_in_low"] = pct_range.str[0]
    df["pct_in_high"] = pct_range.str[1]

    # Fewer votes are outstanding if more of the county is in, so the high % gives the low count.
    df["outstanding_low"] = df["votes"] / (df["pct_in_high"] / 100) - df["votes"]
    df["outstanding_high"] = df["votes"] / (df["pct_in_low"] / 100) - df["votes"]
    df["projected_net_low"] = df["outstanding_low"] * df["margin_pct"] / 100
    df["projected_net_high"] = df["outstanding_high"] * df["margin_pct"] / 100
    return df


if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    counties = load_counties(in_path)
    out_path = in_path.with_suffix(".csv")
    counties.to_csv(out_path, index=False)
    print(counties.sort_values("outstanding_high", ascending=False).head(15).round(0).to_string())
    print(f"\nWrote {len(counties)} counties to {out_path}")
