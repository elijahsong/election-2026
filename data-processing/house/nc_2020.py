"""Rebuild 2016-2020 ACS estimates for North Carolina's 2020 congressional map, which Census never tabulated.

NC redrew its map under court order for the 2020 election (the 117th Congress), but Census tabulates ACS districts
only for maps it collects, and the 2016-2020 5-year release uses the 116th-Congress map. This module rebuilds the 13
districts from smaller ACS geographies that nest inside 2020 Census geography:

1. Map: the enacted plan's block file (SL 2019-249, HB 1029 3rd edition) assigns every 2010 census block a district.
2. 2010 -> 2020 blocks: Census's block relationship file gives the land area each 2010 block shares with each 2020
   block, so each 2020 block gets a share in each district (almost always 1.0 in a single district).
3. Block populations: 2020 Census (P.L. 94-171) counts turn those block shares into people per district.
4. Unit weights: each block group (or tract) is split across districts by where its 2020 population lives.
5. Estimates: 2016-2020 ACS counts per block group (tract for mobility and nativity, which aren't published by block
   group) are multiplied by the weights and summed per district.
6. Fields: shares and averages are ratios of those district sums, using the same counts as Census's district tables.
   Medians can't be summed, so they are interpolated from the summed income and age distributions.

Assumption: people within a split block group look alike, so a block group's estimates divide like its population.
Accuracy: applied to NC's 116th map, which Census did tabulate, the rebuild is within 0.14 pp of every published share,
$474 of median income ($160 on average), and 0.21 years of median age. Run this module to repeat that check.
"""

import argparse
import hashlib
import os
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import acs

YEAR = 2020  # House election year; resolves to the 2016-2020 ACS 5-year in acs.YEAR_TO_ACS_VINTAGE
STATE = "37"  # North Carolina FIPS code
RAW_DIR = Path(__file__).resolve().parents[2] / "data/raw/nc_2020"  # download cache (gitignored)

# name -> (url, zip member, sha256). Pinned so a changed upstream file fails loudly instead of silently moving lines.
#   plan_2020: NC General Assembly block file for the enacted 2020 map, on 2010 blocks (columns Block, District)
#   plan_116: Census 116th-Congress block equivalency file, national, on 2010 blocks (BLOCKID, CD116); used to check
#   blocks_2010_2020: Census 2010-to-2020 tabulation block relationship file for NC, with intersection areas
SOURCES = {
    "plan_2020": (
        "https://webservices.ncleg.gov/ViewBillDocument/2019/6950/0/HB%201029,%203rd%20Edition%20-%20Blockfile",
        "C-Goodwin-A-1-TC.csv", "d669b2143e5cb44cef7a37774cc5cd2f1c5ad2ab5a59488d074fa419102dabfa",
    ),
    "plan_116": (
        "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files/2019/116-congressional-district-bef/cd116.zip",
        "National_CD116.txt", "0364b7b4ef5bf80681721ac667e71f3be52110637d2c79c908ed4a6df4ae506a",
    ),
    "blocks_2010_2020": (
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/TAB2010_TAB2020_ST37.zip",
        "tab2010_tab2020_st37_nc.txt", "b8749f2c00fa4339c107433cc8ff875fbfda1d920a32cfb2f5b731cff3abe8a6",
    ),
}
DISTRICTS = set(range(1, 14))

# Census API geography clauses for every NC block group and tract (wildcards over county and tract)
BLOCK_GROUP_GEO = {"for": "block group:*", "in": f"state:{STATE} county:* tract:*"}
TRACT_GEO = {"for": "tract:*", "in": f"state:{STATE} county:*"}

# Field formulas summed from block groups. Race, college, unemployment, and housing burden reuse the district-level
# 2020 formulas (B03002, B15002, B23025, B25070 + B25091). Poverty comes from B17021, which has B17001's universe and "below poverty" count but is
# published by block group.
BLOCK_GROUP_SPECS = {
    **{field: acs.ACS_FIELDS[YEAR][field]
       for field in ("white", "black", "hispanic", "college", "unemployment", "housing_burden")},
    "poverty": acs.Ratio(("B17021_002E",), ("B17021_001E",)),
    # people in households / households, the definition behind B25010 (a published average can't be summed)
    "avg_household_size": acs.Ratio(("B25008_001E",), ("B25003_001E",), scale=1),
}
# Mobility (B07003) and nativity (B05002) are published only by tract. B07201 has mobility by block group, but only
# for people living in metro areas, which understated moves in rural districts by up to 3 pp.
TRACT_SPECS = {field: acs.ACS_FIELDS[YEAR][field] for field in ("moved_within_past_yr", "foreign_born")}
# Stand-in table -> the published table whose universe it must share, checked against the API metadata:
# B17021 for B17001 (poverty), B25003 for B25010 (households), B19001 for B19013 (households), B01001 for B01002 (people)
SUBSTITUTES = {"B17021": "B17001", "B25003": "B25010", "B19001": "B19013", "B01001": "B01002"}

# Bin labels exactly as Census words them. They serve twice: as the expected variable labels (checked against the
# metadata, so a reordered table fails) and as the source of each bin's lower edge for interpolation.
INCOME_BINS = ["Less than $10,000", "$10,000 to $14,999", "$15,000 to $19,999", "$20,000 to $24,999",
               "$25,000 to $29,999", "$30,000 to $34,999", "$35,000 to $39,999", "$40,000 to $44,999",
               "$45,000 to $49,999", "$50,000 to $59,999", "$60,000 to $74,999", "$75,000 to $99,999",
               "$100,000 to $124,999", "$125,000 to $149,999", "$150,000 to $199,999", "$200,000 or more"]
AGE_BINS = ["Under 5 years", "5 to 9 years", "10 to 14 years", "15 to 17 years", "18 and 19 years", "20 years",
            "21 years", "22 to 24 years", "25 to 29 years", "30 to 34 years", "35 to 39 years", "40 to 44 years",
            "45 to 49 years", "50 to 54 years", "55 to 59 years", "60 and 61 years", "62 to 64 years",
            "65 and 66 years", "67 to 69 years", "70 to 74 years", "75 to 79 years", "80 to 84 years",
            "85 years and over"]
INCOME_CELLS = acs._cells("B19001", *range(2, 18))  # households by income bracket (B19001_001 is the total)
# people by age band, separately by sex: male B01001_003-025, female B01001_027-049 (002 and 026 are sex totals)
AGE_CELLS = {sex: acs._cells("B01001", *range(start, start + 23)) for sex, start in (("Male", 3), ("Female", 27))}

# Expected labels and concepts for the tables only the rebuild uses, merged with acs.py's when verifying variables
REBUILD_LABELS = {
    "B17021_001E": "Total", "B17021_002E": "Total!!Income in the past 12 months below poverty level",
    "B25008_001E": "Total", "B25003_001E": "Total",
    **dict(zip(INCOME_CELLS, (f"Total!!{b}" for b in INCOME_BINS))),
    **{v: f"Total!!{sex}!!{b}" for sex, cells in AGE_CELLS.items() for v, b in zip(cells, AGE_BINS)},
}
REBUILD_CONCEPTS = {
    "B17021": "Poverty status of individuals in the past 12 months by living arrangement",
    "B25008": "Total population in occupied housing units by tenure",
    "B25003": "Tenure",
    "B19001": "Household income in the past 12 months",
    "B01001": "Sex by age",
}


def _lower_bound(label: str) -> float:
    """Bottom of a Census bin label: "Under 5 years" -> 0, "$10,000 to $14,999" -> 10000, "85 years and over" -> 85."""
    return 0.0 if label.startswith(("Under", "Less than")) else float(re.search(r"[\d,]+", label)[0].replace(",", ""))


def interpolated_median(counts: np.ndarray, lower: list[float], pareto: bool = False) -> np.ndarray:
    """Median of each row of a binned distribution, interpolated within the bin that holds the midpoint.

    Args:
        counts: one row per distribution (here, per district), one column per bin, in ascending order.
        lower: each bin's lower edge. Bin i covers [lower[i], lower[i+1]); the last bin is open-ended.
        pareto: interpolate assuming a Pareto (power-law) density instead of a flat one.

    With N the row total, the median bin is the first whose cumulative count reaches N/2. For a bin [lo, hi) with
    `below` people under lo and n inside it:

        linear:  lo + (N/2 - below) / n * (hi - lo)
        Pareto:  lo * (p_lo / 0.5) ** (1 / theta),  theta = ln(p_lo / p_hi) / ln(hi / lo)

    where p_lo and p_hi are the shares of the distribution at or above lo and hi. Pareto fits a survival curve
    S(x) ~ x**-theta through both bin edges, so it recovers a true Pareto distribution exactly.
    Incomes thin out within each bracket, so a flat assumption overshoots: on NC's 116th districts, linear ran $206
    high on average and Pareto $119 low, with smaller absolute error ($160 vs $210). Pareto needs lo > 0 and someone
    at or above hi, so the bottom bin and a median bin with nothing above it fall back to linear.

    Raises:
        ValueError: if a row is empty or its median lands in the open-ended top bin.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(lower + [np.inf])
    cum = counts.cumsum(axis=1)
    total = cum[:, -1]
    if (total <= 0).any():
        raise ValueError("Cannot take the median of an empty distribution")
    i = (cum >= total[:, None] / 2).argmax(axis=1)  # first bin whose cumulative count reaches half the total
    if (i == counts.shape[1] - 1).any():
        raise ValueError("Median falls in the open-ended top bin")
    rows = np.arange(len(counts))
    below = cum[rows, i] - counts[rows, i]
    lo, hi = edges[i], edges[i + 1]
    linear = lo + (total / 2 - below) / counts[rows, i] * (hi - lo)
    if not pareto:
        return linear
    at_or_above_lo, at_or_above_hi = (total - below) / total, (total - cum[rows, i]) / total
    usable = (lo > 0) & (at_or_above_hi > 0)
    with np.errstate(divide="ignore", invalid="ignore"):  # unusable rows produce inf/nan here and are discarded
        theta = np.log(at_or_above_lo / at_or_above_hi) / np.log(hi / lo)
        power = lo * (at_or_above_lo / 0.5) ** (1 / theta)
    return np.where(usable, power, linear)


def download(name: str) -> Path:
    """Cached path to a pinned source zip, downloading it on first use; raises if its checksum isn't the pinned one."""
    url, _, sha256 = SOURCES[name]
    path = RAW_DIR / f"{name}.zip"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with urllib.request.urlopen(url, timeout=600) as resp, open(tmp, "wb") as f:
            shutil.copyfileobj(resp, f)
        os.replace(tmp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise ValueError(f"{path} sha256 {digest} does not match the pinned {sha256}; delete it to re-download")
    return path


def read_plan(name: str) -> pd.Series:
    """2010 block GEOID (15 digits) -> district number (1-13) for one NC plan's block file.

    Both block files list the block in the first column and the district in the second, so they're read by position.
    The national 116th file is filtered to NC in chunks to keep memory low.
    """
    _, member, _ = SOURCES[name]
    with zipfile.ZipFile(download(name)) as z, z.open(member) as f:
        chunks = pd.read_csv(f, dtype=str, usecols=[0, 1], chunksize=2_000_000)
        plan = pd.concat(c[c.iloc[:, 0].str.startswith(STATE)] for c in chunks)
    plan.columns = ["block", "district"]
    plan["district"] = plan["district"].astype(int)
    if plan["block"].duplicated().any() or not plan["block"].str.fullmatch(r"\d{15}").all():
        raise ValueError(f"{name}: block IDs must be unique 15-digit GEOIDs")
    if set(plan["district"]) != DISTRICTS:
        raise ValueError(f"{name}: districts {sorted(set(plan['district']))}, expected 1-13")
    return plan.set_index("block")["district"]


def block_shares(plan: pd.Series) -> pd.DataFrame:
    """Carry a 2010-block plan onto 2020 blocks.

    Returns one row per (2020 block, district) pair with columns block, district, area (the 2020 block's area that
    falls in the district, in square meters), and share (that area over the block's total, so shares sum to 1 per
    block). Land area is used because people live on land; all-water blocks fall back to water area so they still
    get a district. Raises if any NC 2010 block is missing from the plan.
    """
    _, member, _ = SOURCES["blocks_2010_2020"]
    with zipfile.ZipFile(download("blocks_2010_2020")) as z, z.open(member) as f:
        rel = pd.read_csv(f, sep="|", dtype=str, encoding="utf-8-sig")
    rel = rel[rel["STATE_2020"] == STATE]
    block_2010 = rel["STATE_2010"] + rel["COUNTY_2010"] + rel["TRACT_2010"] + rel["BLK_2010"]
    unassigned = block_2010.str.startswith(STATE) & ~block_2010.isin(plan.index)
    if unassigned.any():
        raise ValueError(f"{unassigned.sum()} NC 2010 blocks are missing from the plan, e.g. {block_2010[unassigned].iat[0]}")

    # each relationship row is one 2010-block x 2020-block intersection; rows whose 2010 block was across the state
    # line (boundary corrections) have no district and are dropped
    df = pd.DataFrame({
        "block": (rel["STATE_2020"] + rel["COUNTY_2020"] + rel["TRACT_2020"] + rel["BLK_2020"]).to_numpy(),
        "district": block_2010.map(plan).to_numpy(),
        "land": rel["AREALAND_INT"].astype(float).to_numpy(),
        "water": rel["AREAWATER_INT"].astype(float).to_numpy(),
    }).dropna(subset=["district"])
    df = df.groupby(["block", "district"], as_index=False)[["land", "water"]].sum()
    land_total = df.groupby("block")["land"].transform("sum")
    df["area"] = np.where(land_total > 0, df["land"], df["water"])
    df["share"] = df["area"] / df.groupby("block")["area"].transform("sum")
    df["district"] = df["district"].astype(int)
    return df[["block", "district", "area", "share"]]


def block_population(key: str) -> pd.Series:
    """2020 Census (P.L. 94-171) total population of every NC block, indexed by 15-digit block GEOID."""
    params = {"get": "P1_001N", "for": "block:*", "in": f"state:{STATE} county:* tract:*", "key": key}
    rows = acs._get_json(f"{acs.API_ROOT}/2020/dec/pl?{urllib.parse.urlencode(params)}")
    df = pd.DataFrame(rows[1:], columns=rows[0])
    block = df["state"] + df["county"] + df["tract"] + df["block"]
    return pd.Series(pd.to_numeric(df["P1_001N"]).to_numpy(), index=block)


def unit_weights(shares: pd.DataFrame, population: pd.Series, digits: int) -> pd.DataFrame:
    """Split each block group or tract across districts by where its 2020 population lives.

    Args:
        shares: block_shares() output.
        population: 2020 population per block, from block_population().
        digits: GEOID prefix length of the unit: 12 for block groups, 11 for tracts (2020 blocks nest in both).

    Returns one row per (unit, district) with columns unit, district, and weight, where weight is the unit's people in
    the district (block population x block share, summed) over the unit's total, so weights sum to 1 per unit. A unit
    with no 2020 population is split by area instead, so its (near-zero) ACS estimates still land somewhere.
    """
    missing = ~shares["block"].isin(population.index)
    if missing.any():
        raise ValueError(f"{missing.sum()} blocks have no 2020 population, e.g. {shares['block'][missing].iat[0]}")
    df = shares.assign(unit=shares["block"].str[:digits], pop=shares["block"].map(population) * shares["share"])
    w = df.groupby(["unit", "district"], as_index=False)[["pop", "area"]].sum()
    pop_total = w.groupby("unit")["pop"].transform("sum")
    area_total = w.groupby("unit")["area"].transform("sum")
    w["weight"] = np.where(pop_total > 0, w["pop"] / pop_total.where(pop_total > 0), w["area"] / area_total)
    return w[["unit", "district", "weight"]]


def aggregate(values: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """District totals of unit estimates: sum over units u of weight[u, d] * values[u], for each district d.

    Args:
        values: ACS counts, one row per unit (index = unit GEOID), one column per variable.
        weights: unit_weights() output for the same units.

    Because each unit's weights sum to 1, the district totals add up to the statewide sum of the units. Raises if any
    weight is undefined or if the ACS units and the mapped units aren't the same set, so no unit is dropped or invented.
    """
    if weights["weight"].isna().any():
        raise ValueError(f"Undefined weights for units {weights.loc[weights['weight'].isna(), 'unit'].unique()[:5]}")
    if set(values.index) != set(weights["unit"]):
        only_acs, only_map = set(values.index) - set(weights["unit"]), set(weights["unit"]) - set(values.index)
        raise ValueError(f"ACS units and mapped units differ: {len(only_acs)} only in ACS, {len(only_map)} only in map")
    joined = weights.join(values, on="unit")
    return joined[values.columns].mul(joined["weight"], axis=0).groupby(joined["district"]).sum()


def fetch_units(key: str, geo: dict[str, str], variables: list[str]) -> pd.DataFrame:
    """2016-2020 ACS estimates for every NC block group or tract (per geo), as floats indexed by unit GEOID.

    Raises on any missing estimate or Census sentinel, via acs.parse_estimates.
    """
    raw = acs.fetch_acs(YEAR, key, geo, variables)
    geo_cols = [c for c in ("state", "county", "tract", "block group") if c in raw]
    unit = raw[geo_cols].agg("".join, axis=1)
    return acs.parse_estimates(raw, variables, unit).set_axis(unit)


def check_substitutes() -> None:
    """Raise unless each stand-in table has the same universe as the published table it replaces (see SUBSTITUTES)."""
    universe = {}
    for table in {t for pair in SUBSTITUTES.items() for t in pair}:
        meta = acs._get_json(f"{acs.API_ROOT}/{YEAR}/acs/acs5/groups/{table}.json")["variables"]
        universe[table] = meta[f"{table}_001E"].get("universe")
    bad = {s: (universe[s], universe[p]) for s, p in SUBSTITUTES.items() if universe[s] != universe[p]}
    if bad:
        raise ValueError(f"Stand-in tables with a different universe than the published table: {bad}")


def rebuild_districts(key: str, plan_name: str = "plan_2020") -> pd.DataFrame:
    """Unrounded schema fields (index = district 1-13) for an NC plan, rebuilt from 2016-2020 ACS block groups/tracts.

    plan_name picks the map from SOURCES: "plan_2020" for the output, "plan_116" for the accuracy check.
    """
    # 1. confirm every variable means what the formulas assume, in this vintage
    bg_vars = acs.spec_variables(BLOCK_GROUP_SPECS) + list(INCOME_CELLS) + [v for c in AGE_CELLS.values() for v in c]
    tract_vars = acs.spec_variables(TRACT_SPECS)
    acs.verify_variables(YEAR, bg_vars + tract_vars, {**acs.EXPECTED_LABELS, **REBUILD_LABELS},
                         {**acs.TABLE_CONCEPTS, **REBUILD_CONCEPTS})
    check_substitutes()

    # 2. people per (2020 block, district)
    shares = block_shares(read_plan(plan_name))
    population = block_population(key)
    # 2020 blocks that were across the state line in 2010 (boundary corrections) are in no 2010 plan; they drop out
    # of the weights, so their block group divides like its other blocks. In NC that is one Gaston County block of
    # 13 people; anything larger means a broken crosswalk.
    unmapped = population[~population.index.isin(shares["block"])].sum()
    if unmapped > 1e-4 * population.sum():
        raise ValueError(f"{unmapped:,.0f} people live in 2020 blocks outside the {plan_name} map")

    # 3. district totals of every count, then ratios of those totals
    block_groups = aggregate(fetch_units(key, BLOCK_GROUP_GEO, bg_vars), unit_weights(shares, population, 12))
    tracts = aggregate(fetch_units(key, TRACT_GEO, tract_vars), unit_weights(shares, population, 11))
    ids = pd.Series(block_groups.index.astype(str), index=block_groups.index)  # district labels for error messages
    out = pd.concat([acs.compute_fields(block_groups, BLOCK_GROUP_SPECS, ids),
                     acs.compute_fields(tracts, TRACT_SPECS, ids)], axis=1)

    # 4. medians from the district distributions: households by income bracket, people by age band (both sexes)
    out["median_income"] = interpolated_median(
        block_groups[list(INCOME_CELLS)], [_lower_bound(b) for b in INCOME_BINS], pareto=True)
    ages = block_groups[list(AGE_CELLS["Male"])].to_numpy() + block_groups[list(AGE_CELLS["Female"])].to_numpy()
    out["median_age"] = interpolated_median(ages, [_lower_bound(b) for b in AGE_BINS])
    return out


def rebuild(key: str) -> pd.DataFrame:
    """NC's 13 districts on the 2020 election map, as rounded acs.csv rows (called by acs.build_acs_year for 2020)."""
    fields = acs.round_fields(rebuild_districts(key, "plan_2020"))
    keys = pd.DataFrame({"year": YEAR, "state": "NC", "district": fields.index.astype(int),
                         "acs_vintage": acs.acs_vintage_label(YEAR)}, index=fields.index)
    return pd.concat([keys, fields], axis=1)[acs.COLUMNS].reset_index(drop=True)


def published_116th(key: str) -> pd.DataFrame:
    """Census's own 2016-2020 estimates for NC's 116th-Congress districts, unrounded."""
    raw = acs.fetch_acs(YEAR, key, {"for": "congressional district:*", "in": f"state:{STATE}"})
    district = raw["congressional district"].astype(int)
    values = acs.parse_estimates(raw, acs.field_variables(YEAR), district.astype(str)).set_axis(district)
    return acs.compute_fields(values, acs.ACS_FIELDS[YEAR], values.index.to_series().astype(str))


def sanity_check(key: str) -> pd.DataFrame:
    """Rebuilt minus published values for NC's 116th districts, the map Census did tabulate."""
    fields = acs.COLUMNS[4:]
    return (rebuild_districts(key, "plan_116")[fields] - published_116th(key)[fields]).sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    key = acs.census_key()
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 2):
        diff = sanity_check(key)
        print("Rebuilt minus published, NC 116th districts (pp for shares, $ for income, years for age):")
        print(diff)
        print("\nMean absolute error / max absolute error:")
        print(pd.DataFrame({"mae": diff.abs().mean(), "max": diff.abs().max()}).T)
        print("\nRebuilt 2020-map districts:")
        print(rebuild(key).set_index("district").iloc[:, 3:])


if __name__ == "__main__":
    main()
