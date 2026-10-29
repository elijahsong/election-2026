# ACS district demographics (`acs.py`)

Builds `acs.csv`: one row per House election year and congressional district, with demographics from the Census ACS REST API.

```bash
export CENSUS_API=<your key>            # https://api.census.gov/data/key_signup.html
python src/acs.py --all                 # every supported year -> data/interim/acs.csv
python src/acs.py 2022 2024 --output acs.csv
python -m pytest                        # offline tests; a live 2024 test runs when CENSUS_API is set
```

From Python: `build_acs_year(year, district_counts_by_year, output_path="acs.csv")`. Each call replaces that year's rows, keeps the other years, sorts by `year, state, district`, and writes atomically. Per-state seat counts live in `data/reference/district_counts.json` (2000, 2010, and 2020 apportionments).

## Schema

`year,state,district,acs_vintage,white,black,hispanic,college,median_income,median_age,moved_within_past_yr,foreign_born,unemployment,poverty,avg_household_size`

- `district` is 1..n, and **0 for at-large states**. This matches the Census code `00` and the MIT House data.
- `acs_vintage` is the ACS product actually used: `2018_1yr`, or `2016-2020_5yr` for the 2020 election.
- Percentages are on a 0–100 scale and rounded to 2 decimals. `median_income` is in nominal dollars of the ACS year.

| Field | ACS source |
|---|---|
| white / black / hispanic | C03002 (B03002 for 2020): non-Hispanic White alone, non-Hispanic Black alone, Hispanic (any race) ÷ total |
| college | C15002 (B15002 for 2020): bachelor's degree or higher, both sexes ÷ pop 25+ |
| median_income / median_age / avg_household_size | B19013_001 / B01002_001 / B25010_001 |
| moved_within_past_yr | B07003: 1 − same house 1 year ago ÷ pop 1+ |
| foreign_born | C05002_008 ÷ C05002_001 (B05002_013 ÷ B05002_001 for 2020) |
| unemployment | B23025_005 ÷ B23025_003 (2012+); before 2012, the matching cells summed from C23001, since B23025 doesn't exist yet |
| poverty | C17001_002 ÷ C17001_001 (B17001 for 2020); poverty-status universe |

**Why C tables:** in the 1-year data, Census sometimes withholds a whole detailed (B) table for a district when too many of its cells are unreliable. For example, B03002 is null for AR-1, LA-3, ME-1, MS-1 and OH-14 in 2007. The collapsed (C) version is still published, and its cells are sums of the B cells, so the estimates are identical. 1-year vintages therefore use C tables wherever one exists. The 2020 5-year data isn't filtered and has no C tables, so it uses B.

On every run, each variable's label and concept are checked against that vintage's API metadata.

## Relative supplement (`acs_supplement.py`)

```bash
python src/acs_supplement.py            # data/interim/acs.csv -> data/interim/acs_supplement.csv
```

This writes every `acs.csv` column plus six comparison columns. `acs.csv` itself is never modified, so `acs.py` can keep updating it.

| Column | Definition |
|---|---|
| `rel_college`, `rel_state_college` | district − national / state college share (percentage points) |
| `log_rel_income`, `log_rel_state_income` | log(district median income ÷ national / state median income) |
| `rel_unemployment`, `rel_state_unemployment` | district − national / state unemployment rate (percentage points) |

- **Where the comparison values come from:** they're fetched from the API (`us:1`, `state:*`) using the same vintage, variables and formulas as the districts. Averaging districts wouldn't work: an average of district medians isn't a national median, and the rates would need weighting by population.
- **Rounding:** the comparison values are rounded like the district values, so at-large districts come out as exactly 0.
- **Log ratios** are computed as `log1p((x − ref) / ref)`, which stays accurate when the ratio is close to 1.
- **Rebuilds:** every run rebuilds the whole file from `acs.csv`.
- **2026:** it has no ACS vintage of its own. `notebooks/03c-demographics-2026.ipynb` fills it with 2024 1-year values:
  - Unchanged districts copy their 2024 rows (`2024_1yr`).
  - Districts in redrawn states are rebuilt from 2020–2024 block groups and adjusted to the 2024 1-year (`2024_1yr_rebuilt`).

  `REFERENCE_YEAR = {2026: 2024}` makes 2026 compare against 2024 1-year national and state values. Only such borrowed-vintage
  years may carry the `_rebuilt` label.

## Boundaries and caveats

- **Boundary check.** Every row's `NAME` must show the Congress whose map was used that year (`BOUNDARY_CONGRESS`):
  - 2006 → 110th; 2006 uses the 2007 ACS because the 2006 ACS predates that year's Texas and Georgia redraws.
  - 2008 → 110th (the 2008 ACS is on the 110th, whose maps the 111th kept); 2010 → 111th.
  - Each later year → the Congress it elected; 2024 → 119th, which includes the 2024 AL, GA, LA, NC and NY maps.
- **North Carolina 2020 is rebuilt** by `nc_2020.py`. Census never tabulated the ACS on NC's court-ordered 2020 map, and the 2016–2020 5-year data uses the 116th-Congress map ([Census note](https://www2.census.gov/programs-surveys/acs/summary_file/2020/data/5_year_seq_by_state/A_Note_re_2016-2020_ACS_5yr_Congressional_Districts.pdf)). See below.
- **2026 raises an error** until a matching vintage is added to `YEAR_TO_ACS_VINTAGE`.
- **Missing values raise an error.** A missing estimate or a Census sentinel (e.g. `-666666666`) stops the run; nothing is filled in. Top-coded medians (e.g. income `250001`) are kept as published.
- **DC and Puerto Rico** are excluded, along with unassigned `ZZ` areas.

## North Carolina 2020 rebuild (`nc_2020.py`)

`acs.py` calls this automatically for 2020; run `python src/nc_2020.py` on its own to print the accuracy check. The 13 rows still carry `acs_vintage = 2016-2020_5yr`, but they are rebuilt, not published by Census.

- **Map:** the enacted plan's block file (SL 2019-249, [HB 1029 3rd edition](https://www3.ncleg.gov/BillLookUp/2019/H1029)), which is drawn on 2010 blocks.
  - All 2,744 blocks the ratified bill lists individually match the file.
  - Every district has 733,498–733,499 people in the 2010 Census, so it's the enacted equal-population plan.
- **Moving to 2020 geography:** Census's 2010→2020 block relationship file assigns each 2020 block to districts by land area. The source files download on first use to `data/raw/nc_2020/` and are checked against pinned SHA-256 hashes.
  - One 2020 block of 13 people was in South Carolina in 2010 (a boundary correction), so it isn't on the map; it takes its block group's split.
- **Estimates:** 2016–2020 ACS block-group estimates are summed into districts, splitting each block group by 2020 Census block population.
  - Mobility (B07003) and foreign-born (B05002) are published only by tract, so they're summed by tract instead.
  - Poverty uses B17021, which has the same universe and count as B17001 but is published by block group.
  - Household size is people in households ÷ households.
- **Medians:** interpolated from the summed distributions. Income uses Pareto interpolation over B19001's brackets; age uses linear interpolation over B01001's age bands.

**Accuracy check:** the same method applied to NC's 116th map (Census's 2010-block file for that map), compared with Census's published values for those districts:

| Field | Mean abs. error | Max abs. error |
|---|---|---|
| white / black / hispanic / college | 0.01–0.04 pp | 0.11 pp |
| moved / foreign_born / unemployment / poverty | 0.02–0.05 pp | 0.14 pp |
| median_income | $160 | $474 |
| median_age | 0.09 yr | 0.21 yr |
| avg_household_size | 0.003 | 0.005 |

These errors are well below the ACS's own margins of error for districts. The live test `test_live_rebuild_matches_published_116th_districts` re-runs this check.

# House incumbency (`incumbency.py`)

```bash
python src/incumbency.py                # -> data/interim/incumbency.csv (downloads and caches CAGE on first run)
```

Builds one row per House election year and district for 2006–2026, keyed `year, state, district` like `acs.csv`.

## Schema

`year,state,district,inc,inc_signed,incumbent_party,incumbent_names,status,source`

| Column | Meaning |
|---|---|
| `inc` | 1 if an incumbent is running in the general election, 0 for an open seat |
| `inc_signed` | +1 for a Democratic incumbent, −1 for a Republican, 0 for an open seat. Blank when it isn't a single major party; never treat blank as 0 |
| `status` | `incumbent`, `open`, `override: …`, or `review: …` (the blank `inc_signed` rows) |

## Sources

- **2006–2024:** Cha, Kuriwaki & Snyder, *Candidates in American General Elections* (CAGE), Harvard Dataverse,
  [doi:10.7910/DVN/DGDRDT](https://doi.org/10.7910/DVN/DGDRDT), v4.1, CC0.
  - Only House general-election candidates are used (`office = H`, `type = G`). A candidate with CAGE `inc` above 0 is an
    incumbent: 1 regular, 2 won a special election, 3 appointed.
  - The file is checksum-pinned and cached in `data/raw/cage/`.
  - CAGE numbers at-large seats 1. They are renumbered 0, and every year's districts are checked against
    `data/reference/district_counts.json`.
- **2026:** Cook's 2026 PVI list (`data/raw/pvi.sources/cook_pvi_public_2026.csv`), which is on the 2026 lines.
  - `OPEN (…)` rows are open seats; their party column is the departing member's.
  - Every other row's incumbent runs with the listed party.

## Overrides

`OVERRIDES` is applied after both sources are combined. Each override must match exactly one row. They cover:

- **Source corrections:**
  - TX-34 2024: Mayra Flores is coded as an incumbent, but she lost in 2022, so the row becomes +1 (Gonzalez).
  - MI-7 2006 and KS-1 2016: the incumbent lost the primary and ran as a write-in, so the rows become open (0).
- **Incumbents of both parties in one redrawn district, resolved by hand:**
  - IA-3 2012 (Latham) and OH-16 2012 (Renacci) are coded −1, as Republican incumbents.
  - PA-17 2018, FL-2 2022 and TX-34 2022 are coded open (0).
- **CA-6 2026:** Cook lists Kevin Kiley as an independent; he's coded −1, as a Republican incumbent.

With these, every row has an `inc_signed` value. A new unresolved case would show up as `status = review: …` with a blank
`inc_signed`.
