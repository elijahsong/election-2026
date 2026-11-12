"""Turn the scraped MS Senate special vote-batch table into a tidy batch dataframe.

Just for fun; not part of the forecast pipeline.

    python extra/ms_special_batches.py extra/ms-special-20261106-1900pst.csv

Writes <input>_batches.csv next to the input.

Source quirks:
- Rows are newest first; "Last updated" is relative ("4h ago") to the snapshot time in
  the filename (e.g. 20261106-1900pst), or just "Nov. 3" for election-night batches,
  which therefore get no clock time.
- Margins are text like "Colom +76", "Hyde-Smith +1,776", or "Even". We store them
  signed, Colom (D) positive.
- "Est. % in" of "<1%" is stored as 0.5.
- The table only gives each batch's total votes and margin, not the candidates' votes.
  We get them from:
      other = OTHER_SHARE * votes
      dem   = (votes - other + margin) / 2
      gop   = (votes - other - margin) / 2
  About half the batches have votes and margin of different parity, so some votes went
  to other candidates or write-ins, but the table can't say how many. OTHER_SHARE = 0
  treats the race as two-way, so a batch's dem/gop votes can end in .5.
"""

import re
import sys
from pathlib import Path

import pandas as pd

DEM, REP = "Colom", "Hyde-Smith"
OTHER_SHARE = 0.0  # assumed share of each batch going to other candidates / write-ins


def parse_margin(text):
    """'Colom +2,574' -> 2574, 'Hyde-Smith +98' -> -98, 'Even' -> 0."""
    if text == "Even":
        return 0
    name, amount = text.rsplit(" +", 1)
    amount = int(amount.replace(",", ""))
    return amount if name == DEM else -amount if name == REP else None


def parse_pct_in(text):
    return 0.5 if text == "<1%" else float(text.rstrip("%"))


def snapshot_time(path):
    """'ms-special-20261106-1900pst.csv' -> Timestamp('2026-11-06 19:00')."""
    stamp = re.search(r"(\d{8}-\d{4})pst", Path(path).name).group(1)
    return pd.to_datetime(stamp, format="%Y%m%d-%H%M")


def load_batches(path):
    raw = pd.read_csv(path, thousands=",")
    raw = raw.iloc[::-1].reset_index(drop=True)  # oldest batch first

    df = pd.DataFrame({"batch": range(1, len(raw) + 1)})
    df["last_updated"] = raw["Last updated"]
    df["hours_ago"] = raw["Last updated"].str.extract(r"(\d+)h ago")[0].astype(float)
    df["approx_time_pst"] = snapshot_time(path) - pd.to_timedelta(df["hours_ago"], unit="h")

    # This batch
    df["new_votes"] = raw["New votes"]
    df["batch_margin"] = raw["Vote batch margin"].map(parse_margin)
    df["other_votes"] = OTHER_SHARE * df["new_votes"]
    df["dem_votes"] = (df["new_votes"] - df["other_votes"] + df["batch_margin"]) / 2
    df["gop_votes"] = (df["new_votes"] - df["other_votes"] - df["batch_margin"]) / 2
    df["dem_pct"] = 100 * df["dem_votes"] / df["new_votes"]
    df["gop_pct"] = 100 * df["gop_votes"] / df["new_votes"]
    df["batch_margin_pct"] = df["dem_pct"] - df["gop_pct"]

    # Racewide, after this batch
    df["cumulative_votes"] = df["new_votes"].cumsum()
    df["cumulative_dem_votes"] = df["dem_votes"].cumsum()
    df["cumulative_gop_votes"] = df["gop_votes"].cumsum()
    df["cumulative_other_votes"] = df["other_votes"].cumsum()
    df["cumulative_dem_pct"] = 100 * df["cumulative_dem_votes"] / df["cumulative_votes"]
    df["cumulative_gop_pct"] = 100 * df["cumulative_gop_votes"] / df["cumulative_votes"]
    df["racewide_margin"] = raw["New racewide vote margin"].map(parse_margin)
    df["racewide_margin_pct"] = df["cumulative_dem_pct"] - df["cumulative_gop_pct"]
    df["est_pct_in"] = raw["Est. % in"].map(parse_pct_in)

    # Reported racewide margin minus the sum of the batch margins so far (nonzero = a
    # correction or a batch missing from the table).
    df["unexplained_shift"] = df["racewide_margin"] - df["batch_margin"].cumsum()
    return df


if __name__ == "__main__":
    in_path = Path(sys.argv[1])
    batches = load_batches(in_path)
    out_path = in_path.with_name(in_path.stem + "_batches.csv")
    batches.to_csv(out_path, index=False)
    print(batches.to_string(max_rows=20))
    print(f"\nWrote {len(batches)} batches to {out_path}")
