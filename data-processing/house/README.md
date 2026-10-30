# ACS district demographics (`acs.py`)

Builds `acs.csv`: one row per House election year and congressional district, with demographics from the Census ACS REST API.

```bash
export CENSUS_API=<your key>            # https://api.census.gov/data/key_signup.html
python data-processing/house/acs.py --all     # every supported year -> data/interim/house/acs.csv
python data-processing/house/acs.py 2022 2024 --output acs.csv
python -m pytest                        # offline tests; a live 2024 test runs when CENSUS_API is set
```

From Python: `build_acs_year(year, district_counts_by_year, output_path="acs.csv")`. Each call replaces that year's rows, keeps the other years, sorts by `year, state, district`, and writes atomically. Per-state seat counts live in `data/reference/district_counts.json` (2000, 2010, and 2020 apportionments).

## Schema

`year,state,district,acs_vintage,white,black,hispanic,college,median_income,median_age,moved_within_past_yr,foreign_born,unemployment,poverty,avg_household_size,housing_burden`

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
| housing_burden | Households spending 30%+ of income on housing ÷ households whose burden is computable. Renters (B25070, gross rent; cells 007–010 over 002–010) and owners (B25091, owner costs, with a mortgage 008–011 over 003–011 and without 019–022 over 014–022). "Not computed" households are excluded. Both tables are published for every district in every vintage, so 1-year data reads them directly. |

**Why C tables:** in the 1-year data, Census sometimes withholds a whole detailed (B) table for a district when too many of its cells are unreliable. For example, B03002 is null for AR-1, LA-3, ME-1, MS-1 and OH-14 in 2007. The collapsed (C) version is still published, and its cells are sums of the B cells, so the estimates are identical. 1-year vintages therefore use C tables wherever one exists. The 2020 5-year data isn't filtered and has no C tables, so it uses B.

On every run, each variable's label and concept are checked against that vintage's API metadata.

## Relative supplement (`acs_supplement.py`)

```bash
python data-processing/house/acs_supplement.py  # data/interim/house/acs.csv -> data/interim/house/acs_supplement.csv
```

This writes every `acs.csv` column plus eight comparison columns. `acs.csv` itself is never modified, so `acs.py` can keep updating it.

| Column | Definition |
|---|---|
| `rel_college`, `rel_state_college` | district − national / state college share (percentage points) |
| `log_rel_income`, `log_rel_state_income` | log(district median income ÷ national / state median income) |
| `rel_unemployment`, `rel_state_unemployment` | district − national / state unemployment rate (percentage points) |
| `rel_housing_burden`, `rel_state_housing_burden` | district − national / state housing burden (percentage points) |

- **Where the comparison values come from:** they're fetched from the API (`us:1`, `state:*`) using the same vintage, variables and formulas as the districts. Averaging districts wouldn't work: an average of district medians isn't a national median, and the rates would need weighting by population.
- **Rounding:** the comparison values are rounded like the district values, so at-large districts come out as exactly 0.
- **Log ratios** are computed as `log1p((x − ref) / ref)`, which stays accurate when the ratio is close to 1.
- **Rebuilds:** every run rebuilds the whole file from `acs.csv`.
- **2026:** it has no ACS vintage of its own. `data-processing/house/03c-demographics-2026.ipynb` fills it with 2024 1-year values:
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

`acs.py` calls this automatically for 2020; run `python data-processing/house/nc_2020.py` on its own to print the accuracy check. The 13 rows still carry `acs_vintage = 2016-2020_5yr`, but they are rebuilt, not published by Census.

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

# House modeling features (`house_features.py`)

```bash
python data-processing/house/house_features.py   # -> data/processed/house/{basic_f,nat_political_f,acs_f,modeling}.csv
```

| File | Key | Contents |
|---|---|---|
| `basic_f.csv` | `year, state, district` | `gen_margin_raw` (`generic_ballot.csv` `margin_raw`, election = year), `partisan_lean` (`house_partisan_lean.csv`), `inc_signed` (`house_incumbency.csv`) |
| `nat_political_f.csv` | `year` | Every `data/reference/generic_ballot.csv` field prefixed `gb_`; `election` → `year`, `dem/gop/und_estimate` → `gb_dem_pct / gb_gop_pct / gb_und_pct` |
| `acs_f.csv` | `year, state, district` | `acs_vintage` through `rel_state_housing_burden` from `acs_supplement.csv` |
| `modeling.csv` | `year, state, district` | `basic_f` and `acs_f` joined on the key, then `nat_political_f` joined on `year` |

- **Joins:** the district tables must have unique keys and exactly the same keys, and no column may appear in two of
  them; otherwise the script raises.
- **2026** has no generic-ballot row yet, so its `gen_margin_raw` and `gb_*` fields are blank. Nothing is filled in.
- `src/house/regression` loads `modeling.csv` as the predictors, so a feature set in its `config.py` can mix columns
  from every tranche. It loads the targets separately from `targets.csv`, and `modeling.csv` must not contain any.
  `run_experiment` joins the config's target on the key and raises unless the two tables have exactly the same keys
  in every year `targets.csv` covers. Years with predictors but no targets, such as 2026, get a blank target.

# House election targets (`house_targets.py`)

```bash
python data-processing/house/house_targets.py   # data/raw/1976-2024-house.tab -> data/processed/house/targets*.csv
```

Built from MEDSL's candidate-level returns ([codebook](../data/reference/codebook-us-house-1976–2024.md)), using
observed results only: nothing is imputed or filled in. Each run validates all four files and prints a report.

| File | Rows | Key |
|---|---|---|
| `targets_raw.csv` | Every House general election from 1976 to 2024, 435 a year (10,875), exactly as observed | `year, state, district` |
| `targets.csv` | **The Stage 1 targets**: the rows from 2006 (`FIRST_YEAR`, the first year of `modeling.csv`) to 2024 (4,350), with the three changes below | `year, state, district` |
| `targets_supp.csv` | Companion to `targets.csv`, from 2006: every contest of the districts whose source has more than one or whose race is jungle or RCV, plus the actual result of every imputed `*_vs_3p` or treated-as-uncontested race (268 contests in 259 districts); see below | `year, state, district, contest` |
| `targets_candidates.csv` | The candidates of every `targets_supp.csv` contest (879 rows); see below | `year, state, district, contest, candidate` |

`targets.csv` differs from `targets_raw.csv`'s 2006–2024 rows in three ways:
- **Jungle and RCV targets are left blank** (`IMPUTE_RACE_TYPES`: 40 `dr_jungle` and 10 `dr_rcv` rows), with
  `needs_imputation` `True`. The combined D vs R first-round vote is the right target. The next step
  (`house_impute_targets_jungle_rcv.py`, below) sets it as observed. `targets_raw.csv` keeps it as observed. The vote
  columns are unchanged.
- **A cross-endorsed candidate counts for their larger line** (D or R) when they were the only major-party candidate.
  That sets `race_type` (e.g. `dem_uncontested`), the candidate count and `winner_party`, and the row is flagged
  `cross_endorsed_assigned`. D/R votes stay blank. No row from 2006 on is cross-endorsed, so this changes nothing yet.
- **Eight D-R races are treated as uncontested** (`TREATED_UNCONTESTED`), because their D vs R margin isn't a
  contested result. Each becomes the winner's party + `_uncontested`, or `_vs_3p` with a named third-party candidate.
  Its target is blank with `needs_imputation` `True`, for the uncontested imputation, and the row is flagged with the
  reason. The vote columns and candidate counts are kept as observed, and the actual result, with its candidates, is
  in `targets_supp.csv` / `targets_candidates.csv`.

  | Flag | Races | New `race_type` |
  |---|---|---|
  | `writein_treated_uncontested` (the other major party had only write-ins) | AZ-3 2016; AZ-8, AZ-9, WI-8 2022 | `dem_uncontested`; `gop_uncontested` ×2, `gop_vs_3p` |
  | `major_party_third_treated_uncontested` (a major party finished behind an independent) | TN-9 2006, CA-8 2008, MT-2 2022 | `dem_vs_3p` ×2, `gop_vs_3p` |
  | `voided_election_treated_uncontested` (never certified; rerun in 2019) | NC-9 2018 | `gop_vs_3p` |

  TX-22 2006 stays `dr_writein` with its target: Sekula-Gibbs was the Republican write-in nominee in a real race.

Both files have the columns `TARGET_COLUMNS` followed by `TARGET_PROVENANCE_COLUMNS`, both in
`src/house/regression/schema.py`, and `TARGETS_SCHEMA` types every column. Row counts below cover 1976–2024, with
2006–2024 in parentheses.

| Column | Meaning |
|---|---|
| `dem_votes`, `gop_votes`, `3p_votes`, `total_votes` | Each candidate's party lines are summed, as with fusion tickets. A candidate counts as D or R if any of their lines is Democratic or Republican (`DEM_LABELS` / `GOP_LABELS`, which include DFL and Dem-NPL). Party-less write-in, scattering and other rows go to `3p_votes`. `total_votes` is D + R + 3p |
| `dem_pct` … `margin_2p_norm` | Percentages are on a 0–100 scale. `margin_raw` is over `total_votes` and `margin_2p_norm` is over D + R; each is blank when its denominator is 0 |
| `race_type` | What kind of race the row is; see the table below |
| `adj_margin_2p_norm` | **The canonical Stage 1 target.** `margin_2p_norm` for `dr*` rows with reported votes, plus any imputed value (`is_imputed`); blank otherwise. Imputation goes here, not into the vote columns |
| `is_imputed` | `False` where the target is observed (including jungle and RCV), `True` where it's imputed (uncontested races: `state_mean_residual`, or `national_shave` with no vote counts), blank where it's missing. `dem/gop_votes_imputed` and `imputation_method` are set only on imputed rows |
| `needs_imputation` | `True` where there is no observed target: every race type without a target in the table below, plus a `dr*` race whose votes aren't reported (none so far). It stays `True` after a value is imputed; `is_imputed` records that |
| `is_nonstandard_race` | `True` for every `race_type` except `dr` and `dr_multi`, including `dr_jungle`, `dr_rcv` and `dr_writein`, which do have a target |
| `source_totalvotes`, `nonvote_votes`, `is_unofficial`, `quality_flags`, `source` | Provenance: the source's own `totalvotes`; the blank, void, over/under-vote and exhausted-ballot rows that are left out of `total_votes`; MEDSL's unofficial flag; per-row issues; and the source version |

### Jungle and RCV targets (`house_impute_targets_jungle_rcv.py`)

```bash
python data-processing/house/house_impute_targets_jungle_rcv.py   # after house_targets.py; rewrites targets.csv in place
```

For the 40 `dr_jungle` and 10 `dr_rcv` rows, `adj_margin_2p_norm` is set to the row's own `margin_2p_norm`, the
combined D vs R first-round vote. That is an observed result, not an imputation: `is_imputed` is `False`,
`needs_imputation` is `False`, and `dem/gop_votes_imputed` and `imputation_method` stay blank. These rows are trained
on and scored like any other observed target. Every value set must match `targets_raw.csv`'s observed target. ME-2
2018, 2022 and 2024 are set from the source rows as they are, though those aren't first-round counts
(`rcv_first_round_not_in_source`). Rerunning the script gives the same file. The other three target files aren't
changed.

It also writes `data-processing/house/exceptions.csv`: one row per `race_type` with `Rows`, `Blank` (blank
`adj_margin_2p_norm`) and `Applies to`, the list of `(year, state, district)` keys (left blank for `dr`).
`house_exceptions.ipynb` looks a `race_type` up in it and prints its description and first `targets.csv` row.

### Uncontested imputation (`house_impute_targets_uncontested.py`)

```bash
python data-processing/house/house_impute_targets_uncontested.py   # after the jungle / RCV step; rewrites targets.csv in place
```

Imputes the 470 rows of `UNCONTESTED_RACE_TYPES` (`dem_vs_3p` through `no_major_party` in `RACE_TYPES` order), which
have no observed D vs R result. The method is deterministic and fitted on nothing:

- **Lean:** `L = 0.5 + cook_pvi / 100`, the district's D two-party share in a neutral year, from
  `data/interim/house/house_partisan_lean.csv`. It is built only from the two presidential elections before the
  year, so it carries no House result.
- **References:** in the same year and state, the `dr`, `dr_multi`, `dr_jungle` and `dr_rcv` rows with reported D and
  R votes. Jungle and RCV rows use their party totals. Imputed rows and `dr_writein` are never references, and
  neither are the ME-2 2018, 2022 and 2024 rows, whose source counts aren't the first round
  (`rcv_first_round_not_in_source`). Turnout history isn't used, because redistricting changes the lines.
- **Estimate:**
  - `turnout_hat = round(mean(D_j + R_j))`
  - `state_adjustment = mean(D_j / (D_j + R_j) − L_j)`
  - `share = clip(L + state_adjustment, 0, 1)`
  - `D_hat = round(turnout_hat · share)`, `R_hat = turnout_hat − D_hat` (`round` is half-to-even)

  This equals predicting `national + (L − 0.5)` and correcting by the state's mean error; the national term cancels.
- **Output:** `dem/gop_votes_imputed` are `D_hat` and `R_hat`, and `adj_margin_2p_norm = 100 · (D_hat − R_hat) /
  turnout_hat`. `is_imputed` is `True` and `imputation_method` is `state_mean_residual`. `needs_imputation` stays
  `True`. `race_type`, the vote columns and every other column are unchanged; the script checks the vote columns
  against `targets_raw.csv`.
- **Fallback (national SHAVE):** a row with no reference in its state and year uses Split Ticket's national SHAVE
  for the year, from `data/reference/shave.csv`, in place of the state adjustment:
  - `share = clip(L + margin_2p_norm / 200, 0, 1)` and `adj_margin_2p_norm = 100 · (2 · share − 1)`
  - **Note:** this uses SHAVE's `margin_2p_norm`, the national D two-party margin in points. Since
    `SHAVE_2P − 0.5 = margin_2p_norm / 200`, it's already a two-party quantity. `dem_2p_norm` and `gop_2p_norm` aren't
    used; the file rounds each column separately, so `50 + margin_2p_norm / 2` can differ from `dem_2p_norm` by
    about 0.005 points (2008: 54.835 vs 54.84). The raw and vote-estimate columns aren't used either.
  - With no turnout reference, `dem/gop_votes_imputed` stay blank and the row keeps the flag
    `imputation_no_reference`. `is_imputed` is `True` and `imputation_method` is `national_shave`.
  - This covers 9 rows: AR-1 to AR-4 2008 (every seat uncontested), VT 2008 and 2016, SD 2020 and 2022, and ND 2022
    (at-large, so no other race in the state).
- **Left blank:** a row with no lean is flagged `imputation_no_partisan_lean`, and a fallback row whose year isn't in
  `shave.csv` is flagged `imputation_no_shave`. None is blank now.
- **Not ground truth:** `state_mean_residual` and `national_shave` are in `MODELED_IMPUTATION_METHODS` (`schema.py`),
  so `data.join_target` blanks these targets. The regression doesn't train on them, and `score` leaves them out.

Rerunning gives the same file. It also refreshes `exceptions.csv`. A leave-one-out check on the observed races (each
district predicted from its state's other races) gives a margin MAE of 8.5 points with no bias. In 10 imputed rows,
the margin favors the party with no candidate, e.g. Southern Democratic seats in 2006–2008 (AL-5, TN-6, TN-8, VA-9)
whose lean is Republican.

### `race_type`

`race_type` is assigned from the named candidates per party in the November round. Aggregate write-in, scattering and
"other" rows aren't candidates. Every `dr*` type has a target in `targets_raw.csv`. In `targets.csv`, `dr_jungle` and
`dr_rcv` are set by the jungle / RCV step, and the other types are imputed (uncontested imputation, above). A race with a D and an R gets the first matching `dr*` type in the order listed.

| `race_type` | Meaning | Rows (2006–2024) | Target |
|---|---|---|---|
| `dr_writein` | One major party has only write-in candidates, so the race is effectively uncontested on the ballot and its margin is near ±100. Includes TX-22 2006 and VA-9 2002 via `PARTY_OVERRIDES` | 7 (5) | yes in `targets_raw.csv`; in `targets.csv` only TX-22 2006, the other four are uncontested (`TREATED_UNCONTESTED`) |
| `dr_rcv` | Ranked-choice voting: ME from 2018, AK from 2022. The first round is used and treated like a jungle round; the source lacks it for ME-2 2018, 2022 and 2024 (see `is_rcv`) | 10 (10) | yes (set by the jungle / RCV step in `targets.csv`) |
| `dr_jungle` | All-candidate November round, often with several candidates per party; the target combines each party's votes. LA 1998–2006 and 2012–2024, and the Texas specials of 1996 and 2006. The runoff isn't used even where the source has it | 66 (40) | yes (set by the jungle / RCV step in `targets.csv`) |
| `dr` | Ordinary race: exactly one D and one R, possibly with third-party candidates | 9,361 (3,821) | yes |
| `dr_multi` | Ordinary race with more than one D or R, almost always a party-labeled write-in with a handful of votes next to the nominee | 20 (12) | yes |
| `dem_vs_3p`, `gop_vs_3p` | One major-party candidate against third-party or write-in candidates only | 553 (200) | imputed in `targets.csv` |
| `dem_uncontested`, `gop_uncontested` | A single named candidate; scattered write-in votes may exist | 766 (204) | imputed in `targets.csv` |
| `dem_vs_dem`, `gop_vs_gop` | Two or more candidates of one major party and none of the other, e.g. CA/WA top-two or LA jungle | 68 (58) | imputed in `targets.csv` |
| `cross_endorsed` | The only major-party candidate was on both the D and R lines (NY 1976–1990); D/R votes are blank | 24 (0) | imputed in `targets.csv` |
| `no_major_party` | No D or R candidate | 0 | imputed in `targets.csv` |

- **No votes reported:** when every candidate in a contest has at most 1 vote (unopposed candidates are listed as 1, 0 or −1, mostly in FL, LA, AR and OK), the vote and margin fields are blank and the row is flagged `votes_not_reported`.
- **One contest per key, always the November round:** thirteen keys have a second contest. TX 1996 8, 9 and 25, LA 2002 5, TX 2006 23 and the seven `LA_OFFICIAL` races (added from official results, below) also have their December runoff (flag `runoff_in_source`); `winner_party` comes from the runoff. TX 2006 22 also has a same-day special for the unexpired term (flag `special_in_source`); the regular election is used. MEDSL labels TX-23 2006's runoff as the regular election, so `RUNOFF_FIXES` relabels it. The 2006 keys' contests are in `targets_supp.csv`.
- **Party overrides:** `PARTY_OVERRIDES` recodes two Republican nominees whom the source lists as party-less write-ins: Sekula-Gibbs (TX-22 2006) and Katzen (VA-9 2002). Those rows are flagged `party_override`.
- **Cross-endorsed candidates** (NY 1976–1990, nominated on both the D and R lines) are counted as neither party in `targets_raw.csv`. Their D/R votes, margins and `winner_party` are blank. `targets.csv` assigns them to their larger line (see above).
- **`winner_party`** is blank where the source doesn't settle it: a cross-endorsed winner, or a majority-rule round whose leader has 50% or less, whose runoff isn't in the source, and whose top two are from different parties (`no_majority_runoff_not_in_source`). Only the 24 cross-endorsed rows are blank now.
- **Official corrections (`LA_OFFICIAL`):** every 2006–2024 Louisiana race was checked against Louisiana's official results, saved unmodified in [`data/raw/la_sos`](../data/raw/la_sos/README.md). In six races MEDSL gives the top two candidates' December runoff totals in place of their November counts (in LA-5 2020, the runoff totals plus 100,000). The script replaces those counts with the official November results, flags the rows `official_results_correction`, and `source` names both sources. Every MEDSL candidate must match exactly one official candidate (`LA_NAME_ALIASES` covers one spelling difference), or the script stops. The official December runoffs of those six and of LA-3 2012 (whose November counts already match) are added as `runoff` contests in `targets_supp.csv`, and `winner_party` comes from them. Every other race matches the official November results; none of them had a runoff.

  | Race | Runoff | Combined November margin (before → after) | `winner_party` |
  |---|---|---|---|
  | LA-2 2006 | Jefferson (D) vs Carter (D) | 74.0 → 70.2 | D |
  | LA-3 2012 | Boustany (R) vs Landry (R) | −56.4 (unchanged) | R |
  | LA-5 2014 | Abraham (R) vs Mayo (D) | −53.4 → −42.6 | blank → R |
  | LA-6 2014 | Graves (R) vs Edwards (D) | −42.2 → −30.0 | blank → R |
  | LA-3 2016 | Higgins (R) vs Angelle (R) | −59.4 → −64.3 | R |
  | LA-4 2016 | Johnson (R) vs Jones (D) | −64.6 → −42.5 | R |
  | LA-5 2020 | Letlow (R) vs Harris (R) | −54.2 → −35.7 | R |

  All seven are `dr_jungle`, so in `targets.csv` the jungle / RCV step sets their targets.
- **`is_rcv`** is set for ME from 2018 and AK from 2022. The first round is wanted. The source has it for AK and for every ME race decided in round 1, but not for ME-2 2018 (matches neither round), 2022 or 2024 (a later round, with exhausted ballots). Those three are flagged `rcv_first_round_not_in_source` (`RCV_FIRST_ROUND_NOT_IN_SOURCE`).
- **Excluded:** DC.
- **Choices:** [`house_targets_choices.md`](house_targets_choices.md) records each judgment call, the alternatives with how many rows each would change, and the decision made.

### `targets_supp.csv`

The targets use one contest per district, the November round. For `targets.csv`'s years (2006–2024),
`targets_supp.csv` keeps every contest of the districts whose source has more than one, and the November round of
every jungle and RCV race, whose target in `targets.csv` the jungle / RCV step sets. It also keeps the actual November
result of every race whose `targets.csv` target is imputed but had an opponent: every `*_vs_3p` race and every
`TREATED_UNCONTESTED` race (where its `race_type` is the observed one, e.g. `dr`). It is **not joined into `targets.csv` or
`targets_raw.csv`**. It is reference data for future imputation of `adj_margin_2p_norm` (for example, from a
jungle race's combined vote or a runoff), and for post-processing and model evaluation. Its columns are `TARGET_SUPP_COLUMNS`, typed by
`TARGETS_SUPP_SCHEMA` in `schema.py`. Each row's votes, margins, candidate counts, `race_type`, `winner_party` and
provenance are computed the same way as in the targets, for that contest alone. It has no target or imputation columns.

| `contest` | Meaning | Districts |
|---|---|---|
| `first_round` | The November round (an RCV race's first round). It is the contest the targets use, so these rows repeat `targets_raw.csv`'s values | 40 jungle, 10 RCV, TX 2006 22, and 208 imputed races with an opponent (205 `*_vs_3p` in `targets.csv`, incl. 5 treated, plus AZ-3 2016, AZ-8 and AZ-9 2022) |
| `runoff` | The December runoff. Never `dr_jungle` | TX 2006 23; LA 2006 2, 2012 3, 2014 5 and 6, 2016 3 and 4, 2020 5 (official results) |
| `special` | A same-day special for the rest of the unexpired term | TX 2006 22 |

`winner_party` on a `first_round` row with a runoff is the seat's winner, as in the targets, so it matches the runoff
row. `is_special_election` is the source's flag for each contest. `year, state, district, contest` is unique, and every
key is in `targets.csv`. Pre-2006 districts aren't in this file. Their November rounds are in `targets_raw.csv`.

### `targets_candidates.csv`

`targets_raw.csv` and `targets_supp.csv` store each contest's totals per party: for a jungle race, `dem_votes` and
`gop_votes` are already summed over all of that party's candidates. `targets_candidates.csv` keeps the breakdown, with
one row per candidate of every `targets_supp.csv` contest (`TARGET_CANDIDATE_COLUMNS`, typed by
`TARGETS_CANDIDATES_SCHEMA` in `schema.py`). It joins to `targets_supp.csv` on `year, state, district, contest`, and
within each contest its votes add up to that file's `dem_votes`, `gop_votes`, `3p_votes` and `nonvote_votes`, and
its `is_candidate` rows to the candidate counts. Like `targets_supp.csv`, it is reference data and isn't joined into
the targets.

| Column | Meaning |
|---|---|
| `candidate` | As the source spells it |
| `parties` | The source's party labels on the candidate's lines, `; `-separated (e.g. a fusion ticket); blank for party-less rows |
| `party_group` | `D`, `R`, `3P`, `D+R` (cross-endorsed) or `nonvote` (blank, exhausted and similar rows), as used for the party totals |
| `is_candidate` | `False` for aggregate write-in / other rows and non-vote rows |
| `is_writein` | Every one of the candidate's lines is a write-in |
| `votes` | Summed across the candidate's party lines |
| `source` | MEDSL, or Louisiana's official results for the corrected Louisiana rows and the Louisiana runoffs |
