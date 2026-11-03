# Senate targets, PVI and political features (`data/processed/senate/`)

This builds statewide and county U.S. Senate results, state and county presidential PVI, and the feature tables
`basic_f.csv`, `acs_f.csv`, `fec_f.csv`, `prices_f.csv` and `nat_political_f.csv`, joined in `modeling.csv`. It
follows the House pipeline (`../house/`) and reuses its code where it applies.

```bash
python data-processing/senate/download_sources.py        # raw sources -> data/raw/ (checks every sha256)
python data-processing/incumbency.py                     # once: data/interim/senate_incumbency.csv (classes, inc_signed)
python data-processing/senate/presidential.py            # pres_state_votes.csv, pres_county_votes.csv
python data-processing/senate/senate_pvi.py              # state_pvi.csv, county_pvi.csv
python data-processing/senate/senate_targets.py          # targets_raw / targets / targets_supp / targets_candidates
python data-processing/senate/senate_county_results.py   # county_results.csv, county_results_supp.csv
python data-processing/cpi_gas.py                        # data/processed/cpi_gas_region.csv (shared with the House)
python data-processing/unemployment.py                   # data/processed/unemployment_state.csv (shared with the House)
python data-processing/senate/senate_acs.py              # data/interim/senate/acs_state.csv (Census API; CENSUS_API)
python data-processing/senate/senate_features.py         # basic_f, acs_f, fec_f, prices_f, nat_political_f, modeling
python data-processing/senate/senate_polls.py            # polls_clean.csv, polls.csv [--as-of YYYY-MM-DD]
python data-processing/senate/senate_ratings.py          # expert-ratings.csv
```

Each script prints a report and refuses to write on a blocking problem. The column order, dtypes, keys and FIPS rules
of every output are defined in `senate_schema.py` and checked on write. A rerun gives byte-identical files.
`tests/test_senate.py` checks:
- the PVI arithmetic and look-ahead rule;
- the AK/HI At-Large rows;
- the calendar of classes;
- the Election Day rounds;
- that imputation leaves the observed votes unchanged;
- that `nat_political_f.csv` is the House file.

## Sources (`data/raw/`, each folder with a `SOURCE.md`)

| Folder | Source | Used for |
|---|---|---|
| `medsl_senate/` | MEDSL U.S. Senate statewide 1976–2024, doi:10.7910/DVN/PEJ5QU V8, CC0 | statewide targets, all 50 states |
| `heideman/v1.0.0/` | Heideman, county results 1990–2024, doi:10.5281/zenodo.22976553 v1.0.0, CC BY 4.0 | county results, 48 states |
| `fec_federal_elections/` | FEC *Federal Elections* 2006–2022 (official, public domain) | eleven statewide contests, and the official check |
| `la_sos/` (existing), `la_sos_parish/` | Louisiana SOS official results | LA 2014/2016/2020 statewide, parish rounds |
| `medsl_pres_state/` | MEDSL U.S. President 1976–2024, doi:10.7910/DVN/42MVDX V10.1, CC0 | `pres_state_votes.csv` |
| `medsl_pres_county/` | MEDSL County Presidential Returns 2000–2024, doi:10.7910/DVN/VOQCHQ V20, CC0 | `pres_county_votes.csv` |
| `senate_manual/` | Hand-entered county results (from WV SOS, SC Election Commission, OK Election Board, OpenElections and Wikipedia tables) | county results for GA 2008 and GA 2022 (November) and the MS 2008, DE 2010, NY 2010, WV 2010, SC 2014, OK 2014, AZ 2020, OK 2022 and NE 2024 specials |
| `mn-2018-senate-special/` | Minnesota SOS 2018 general election county results files | MN 2018 special (county) |
| `ms-2018-senate-special-r1/` | Mississippi 2018 general election precinct results | MS 2018 special, November round (county) |
| `ga_2020_senate_special_r1/` | Georgia 2020 general election precinct results | GA 2020 special, November round (county) |
| `openelections_senate/` | OpenElections county, town and precinct results | ME 2012, NH 2016, NY 2016, SC 2016, DE 2018, NJ 2018 and SC 2022 county results (replacing Heideman's); RI and ME votes in no county |
| `maine_sos_2024/` | Maine SOS 2024 U.S. Senate results by municipality | ME 2024 UOCAVA votes (in no county) |
| `idaho_sos_2022/` | Idaho SOS 2022 general election canvass | ID 2022 Power County |
| `arkansas_sos_2022/` | Arkansas SOS 2022 general election results file | AR 2022 Phillips County |
| `All-US-Senate-polls-2006-2024/` | Jack Whitcomb's RealClearPolling Senate polls 2006–2024 (GitHub, commit 5aea8c5) | `polls_clean.csv`, `polls.csv` |
| `senate_ratings_historical/` | JHK Forecasts' forecast archive (`output.csv`, supplied; final pre-election ratings) | `expert-ratings.csv`, 2016–2024 |
| `kansas_city_mo/` | Kansas City (MO) Election Board official results 2006–2022; OpenElections MO 2024 precincts | Kansas City's Senate votes, added to Jackson County |

Inputs reused unchanged:
- **National presidential baseline:** `data/interim/presidential.csv`. It is read only and never rebuilt.
- **Incumbency:** `data/interim/senate_incumbency.csv`, built by `../incumbency.py`.
- **Generic ballot:** `data/reference/generic_ballot.csv`.
- **SHAVE:** `data/reference/shave.csv`.
- **House national features:** `data/processed/house/nat_political_f.csv`.

**Why the targets are MEDSL and not Heideman.** Heideman's Senate rows can't carry statewide targets:
- They exclude specials and AK/HI.
- They give the runoff in place of the November round for GA 2008, GA 2022 and LA 2014.
- For 2006–2014 (from Algara & Amlani) they name only the two major-party nominees. Everyone else is one row, which
  hides candidate counts and `race_type`, and some independents are coded DEM (King 2012, Sanders 2006/2012).

So Heideman is the county source, and its county sums are reconciled against the statewide results.

## Keys and geography

| File | Key |
|---|---|
| `targets*.csv`, `basic_f.csv`, `acs_f.csv`, `fec_f.csv`, `prices_f.csv`, `modeling.csv` | `year, state, class` (`targets_supp.csv` and `targets_candidates.csv` add `contest`, and `candidate`) |
| `county_results.csv` | `year, state, class, county_id, county_fips` (`_supp` adds `contest`) |
| `state_pvi.csv` | `year, state` |
| `county_pvi.csv` | `year, state, county_id, county_fips` |
| `nat_political_f.csv` | `year` |
| `polls.csv`, `expert-ratings.csv` | `year, state, class` (every race; 2006–2026 and 2016–2026), with `election_id` |
| `polls_clean.csv` | `source_line` (one row per poll question; race `year, state, class`, plus `stage`) |

- **Classes:** `class` and the race list come from `senate_incumbency.csv`. Its classes come from CAGE's next-up year,
  so a special gets the class of the seat it fills. A test checks that each regular race is its year's calendar class
  and each special is another.
- **Specials:** only specials on the November ballot are included (14 in 2006–2024, plus FL and OH in 2026). For the
  nonpartisan MS 2018 and GA 2020 specials, the November round is the main row and the runoff goes to `_supp`.
  Off-cycle specials (MA Jan 2010 and Jun 2013, NJ Oct 2013, AL Dec 2017) are in neither MEDSL's file nor
  `senate_incumbency.csv`.
- **County FIPS:** `county_fips` is a five-character string with its state's prefix, and `county_id` equals it.
- **AK and HI:** each has one At-Large county-equivalent per contest and per PVI year:
  - `county_id` and `county_name` are `At-Large`, and `is_at_large` is true.
  - `county_fips` is `02000` or `15000`. **These are synthetic state-aggregate codes, not county FIPS.**
  - The rows hold the actual statewide votes and the statewide PVI. No county distribution is invented.
  - Hawaii's real counties are left out of `county_pvi.csv` for the same reason.
- **Oglala Lakota County, SD:** recoded to 46113 in every presidential year, as Heideman codes it (MEDSL uses 46102
  in 2016 and 2020).

## Statewide targets (`senate_targets.py`)

Votes, `race_type`, flags, targets and imputation follow the House rules exactly, reusing `house_targets.race_types`,
the label sets, `impute_jungle_rcv` and `impute_uncontested`.

**Votes and margins**
- `total_votes` excludes non-votes, which go in `nonvote_votes`.
- Margins are Democratic-positive, on a 0–100 scale.
- **Independents who caucus with Democrats count as the Democrat (`CAUCUS_DEMOCRATS`):** King (ME 2012, 2018, 2024)
  and Sanders (VT 2006, 2012, 2018, 2024), unlike the House files' rule. Their votes are `dem_votes`, and the races'
  Democratic nominees (Dill, Ringelstein, Costello; Vermont had none) are third party, in the statewide and county
  files alike. Flag `independent_counted_as_dem`. So these are observed D-R races, not treated as uncontested.
- The other independents (Orman KS 2014, McMullin UT 2022, Osborn NE 2024), who didn't commit to caucusing with
  Democrats, are third party. Incumbents are coded by caucus in `inc_signed` (King +1 in 2018 and 2024).

**Rounds**
- The November (Election Day) round is the one used in `targets*.csv` and `county_results.csv`. Runoffs are in the
  `_supp` files, under the cycle year, with `stage = runoff` and an `election_date`.
- `winner_party` is the eventual winner, and `election_day_leader_lost` flags GA 2020 (Perdue) and LA 2014 (Landrieu).

| Kind | Races (2006–2024) |
|---|---|
| Runoffs (in `targets_supp.csv`) | GA 2008, GA 2020 (both seats, Jan 5 2021), GA 2022, LA 2014, LA 2016, MS 2018 special |
| Jungle (`dr_jungle`) | LA 2014, 2016, 2020, 2022; nonpartisan specials MS 2008, MS 2018, GA 2020 |
| RCV (`dr_rcv`) | ME 2018, 2020, 2024; AK 2022 |
| Same-day specials for a seat's last weeks (contest `special`) | CA 2022, CA 2024 |

**Corrections to MEDSL** (each is an explicit table in the script that must apply):
- **Mislabelled rounds:**
  - GA 2008: MEDSL's `pre` row is the November round and its `gen` row is the runoff.
  - MS 2018 special: MEDSL has only the runoff, labelled `gen`.
  - IL 2010: the full-term race is marked special.
- **LA 2014, 2016 and 2020** come from the LA SOS official results. MEDSL gives 2014's and 2016's top two candidates
  their runoff totals (plus 1,000,000 in 2014).
- **From FEC's official results:**
  - CT 2022 (missing from MEDSL) and the MS 2018 November round.
  - WV 2006 and UT 2010 (MEDSL's counts are partial).
  - MO 2022 and DE 2006 (MEDSL lacks candidates).
  - ME 2006, CO 2010, NY 2012, PA 2018 and WV 2018 (MEDSL isn't the certified count).
  - FEC lists no blank ballots, so `nonvote_votes` is 0 for these contests.
- **Party labels:**
  - Nonpartisan specials: MS 2008 (Musgrove D) and MS 2018 (Hyde-Smith and McDaniel R, Espy and Bartee D).
  - Unlabelled rows: AK 2022 and WY 2020.
  - MS and NV 2024 have the party only in `party_simplified`.
- **AK 2022** keeps MEDSL's RCV round-1 tabulation (Murkowski 114,118), the same kind of count as the House AK rows.
  FEC's first-choice summary is about 600 votes lower per candidate (flag `rcv_round1_tabulation`).

**Treated as uncontested** in `targets.csv` (House `major_party_third_treated_uncontested`): a major party finished
third behind an independent.
- **Races:** CT 2006, AK 2010, FL 2010 and AK 2016.
- **race_type:** `<party>_vs_3p`, for the major party that finished second.

**Imputation:** 12 uncontested targets are imputed, 11 with `national_shave` and 1 with `state_mean_residual` (NE 2024
regular, from the NE special). The lean is `0.5 + state pvi / 100`.
- **Intentionally blank:** three 2006 races (CT, IN, VT) have no target, with flag `imputation_no_shave`, because
  `shave.csv` has no 2006. This is the House rule; nothing is invented.

**Official check:** every 2006–2022 contest is compared with FEC on D, R and total votes. `differs_from_fec` remains
on 10 contests, each explained (in the comparison, FEC's King and Sanders count as D too):
- **FEC codes the candidate differently:** CT 2006 (Lieberman `DEM/CFL`), AK 2010 (Murkowski counted R), VT 2006
  (Ericson `REP/IND`) and 2010 (Freilich), MS 2008 and 2018 (nonpartisan `N`), CT 2010.
- **AK 2022:** the RCV tabulation, above.
- **WY 2020:** FEC's 73,766 for Ben-David against 72,766 in MEDSL and the state canvass.

FEC hasn't published 2024 results.

## County results (`senate_county_results.py`)

- **Sources:** Heideman county rows for the 48 states; for the rounds Heideman lacks, LA SOS parish files (LA 2014
  November, LA 2016 runoff) and hand-entered county tables in `senate_manual/` (from official results pages for WV 2010, SC 2014 and OK 2014, OpenElections
  for GA 2008, Wikipedia for the rest; GA 2008 and GA 2022 November, and
  the MS 2008, DE 2010, NY 2010, WV 2010, SC 2014, OK 2014, AZ 2020, OK 2022 and NE 2024 specials; only vote counts
  are read, and D and R sum to the statewide result exactly, or for NY 2010, AZ 2020 and NE 2024 within 0.03%);
  official files parsed directly for the MN 2018 special (Minnesota SOS county file, exact) and the November rounds
  of the MS 2018 special (precinct results summed by county, within 0.03%) and the GA 2020 special (precinct
  results, exact);
  and At-Large rows for AK/HI.
- **Typos in hand-entered files (`MANUAL_VALUE_FIXES`):** NE 2024 special: Dakota R 5,206 → 4,068 and Cass D 4,506
  → 4,606 (each disagreed with its own share; the corrections make statewide R exact and D within 1 vote), and five
  `total_votes` set to the county's D + R (Dakota, Cass, Saunders 4,321 → 13,321, Banner 372 → 382, Loup 432 → 428).
  Each fix checks the file's value first; the raw file is unchanged, and the rows aren't flagged.
- **GA 2020 special parties:** all Democrats are D and all Republicans R, as in the statewide target (Warnock
  1,617,035 of D's 2,378,312; Loeffler and Collins 2,253,668 of R's 2,426,120).
- **Coverage:** 346 of 347 races have Election Day county results.
- **Unknown third-party votes:** DE 2010 special. The hand-entered file's third-party counts lack a candidate in Kent
  and New Castle, so `3p_votes`, `total_votes` and the shares of the total are blank; the D-vs-R margins are
  complete. Flag `3p_votes_unknown`.
- **Parties:** each Heideman candidate counts as D, R or 3P as the statewide results count that candidate, matched by
  surname, or failing that by statewide vote total (Heideman misspells some names). Flag `party_reassigned`, e.g. on
  King 2018 and 2024 (Heideman: other) and Osborn (Heideman: DEM).
- **Corrections to Heideman (`HEIDEMAN_COUNTY_SWAPS`):** OK 2014 regular (Inhofe): Heideman's Beaver and Beckham rows
  hold each other's votes, so they're swapped back (Beaver 1,512 votes, R+82.3; Beckham 4,606, R+64.4). Beckham's
  vote is 3-4 times Beaver's in every other year, in Heideman's 2014 House rows, and in the official 2014 special.
  The corrected rows aren't flagged; the raw file is unchanged.
- **Kansas City, MO (`KANSAS_CITY_SENATE`):** Missouri tabulates Kansas City apart from its counties, and Heideman
  leaves it out, so its Jackson County row covered only the county outside the city (62–110k D and 17–25k R votes
  missing each year). The Kansas City Election Board serves only the part of the city inside Jackson County, so its
  votes are added to Jackson County. Each named candidate's count is written out in the code, from the board's
  official results (2006–2022) and OpenElections (2024); the board's lumped write-ins aren't included. Missouri's
  county sums now match the statewide D and R exactly in 2006, 2010, 2012, 2016 and 2024, within 15 votes in 2018,
  and within 742 in 2022 (Heideman's other counties). The rows aren't flagged; their `source` names both sources.
- **Heideman races replaced (`OPENELECTIONS_ROUNDS`):** seven races where Heideman's counties don't add up to the
  statewide result use OpenElections results summed by county instead (`data/raw/openelections_senate/SOURCE.md`):
  - ME 2012: no votes for Dill (the Democrat, third party here); King, Summers and the others agree exactly. Now
    exact.
  - NH 2016: every county short (536,712 of 739,140 votes). Now exact.
  - NY 2016: 20,492 of Long's votes under "Wendy Ross", coded OTHER; R 35k short. Now within 4 votes.
  - SC 2016: only Dixon's DEM line, not his Working Families and Green lines (D 52k short), and one of Bledsoe's two
    lines. Now exact.
  - DE 2018: New Castle County has 11 votes. Now exact.
  - NJ 2018: Middlesex County 27k D and 16k R short. Now exact.
  - SC 2022: nearly every county slightly over (D +2,292, R +3,757). Now exact.
- **Typos in Heideman (`HEIDEMAN_VALUE_FIXES`):** VT 2012: Caledonia Macgovern 4,158 → 3,858, Chittenden Sanders
  55,615 → 55,115, Washington Sanders 21,502 → 20,797, as in OpenElections' town results; the statewide result is then
  exact. Each fix checks Heideman's value first; the rows aren't flagged.
- **Counties missing from Heideman (`HEIDEMAN_MISSING_COUNTIES`):** AR 2022 Phillips (from the Arkansas SOS results
  file, `data/raw/arkansas_sos_2022/`) and ID 2022 Power (Idaho SOS canvass, `data/raw/idaho_sos_2022/`). Each is
  exactly the statewide result minus Heideman's other counties.
- **Hidden major-party votes:** where a statewide D or R candidate is inside Heideman's "All other candidates" row,
  that party's county votes are blank (flag `party_votes_in_other`): DE 2006 (O'Donnell's write-in).
- **No county Election Day results (1 race):** the WY 2008 special (Heideman excludes specials), left out by
  choice.
- **Runoffs without county results:** the GA 2020 runoffs (both seats, Jan 5 2021) and the MS 2018 special's runoff
  (Nov 27) aren't in `_supp`. Their November rows in `county_results.csv` are flagged
  `runoff_county_results_missing`. Not being filled for now.
- **Votes in no county (`NON_COUNTY_VOTES`):** RI's statewide results include "Federal Precinct" ballots (federal
  offices only), and Maine's include UOCAVA ballots (overseas and military voters); neither belongs to a county, so
  the county sums are short by exactly these votes (King counted as D): ME 2012 (D 1,519, R 587, 3P 805), ME 2018
  (1,475, 249, 698), ME 2020 (4,068, 1,089, 491), RI 2020 (1,147, 81, 0), RI 2024 (1,356, 112, 5), from
  OpenElections, and ME 2024 (3,080, 796, 2,558), from the Maine SOS. The reconciliation subtracts them from the
  statewide result; they aren't spread over counties.
- **Reconciliation:** each contest's county sums are compared with its statewide totals less the votes in no county.
  Every contest agrees within max(100 votes, 0.1%) on D and R. A contest that didn't would have its rows flagged
  `county_sum_differs_from_statewide`; nothing is forced to match.
- **LA SOS vs Heideman:** the parish files agree exactly, parish by parish, in the three rounds both have. AK/HI
  At-Large rows equal the statewide totals exactly.

## PVI (`senate_pvi.py`)

`pvi = 0.75 * rel(prev) + 0.25 * rel(anteprev)` and `partisan_lean = 2 * pvi`, where:
- `rel(p) = dem_2p_norm(geography, p) − national dem_pct_2p_norm(p)`;
- prev and anteprev are the two presidential years strictly before the Senate year (2006/2008: 2004 and 2000; ...;
  2026: 2024 and 2020).

This is the House formula, and states and counties use the same national baseline. A county needs a result under the
same FIPS in both presidential years. Four county-years are left out: Broomfield, CO in 2006 and 2008 (created 2001),
and Bedford city, VA in 2018 and 2020 (merged 2013).

**Presidential inputs.** The repo had no state or county presidential results, so `presidential.py` builds them from
MEDSL. `pres_*_votes.csv` hold raw votes as `Int64` and `dem_2p_norm`, `gop_2p_norm` and `margin_2p_norm`.
- **Ballot lines:** state nominees' lines are summed.
- **Modes:** where a county reports a `TOTAL` mode, only `TOTAL` is used (UT 2020 and nine 2024 states list both).
- **Left out:**
  - Alaska's district rows (they collide with borough FIPS), and DC.
  - Statewide write-in, UOCAVA and federal-precinct rows.
  - Bedford city's empty 2016 row.
- **Kansas City, MO:** MEDSL reports it apart from its counties, as Missouri does; its rows are added to Jackson
  County, the only county its election board serves (in 2000 its row is 0 and Jackson already includes it). Without
  this, Jackson's 2004–2024 votes left out the city and its PVI was 10–14 points too Republican (2014: −5.8
  instead of +8.3). Missouri's county
  sums now equal the state's D and R exactly.
- **Check:** county sums against the state file are printed. NY and ME differ for the reasons above.

## `basic_f.csv` and `nat_political_f.csv`

- **Rows:** `basic_f.csv` has 382 rows, one per race in `senate_incumbency.csv` for 2006–2026 (35 in 2026). No 2026
  targets are created.
- **gen_margin_raw:** the same year's national generic-ballot raw margin (not the Senate result). It is blank for 2026,
  which has no generic-ballot row yet.
- **partisan_lean:** the statewide `2 * pvi`.
- **inc_signed:** `incumbency.py`'s, with the House convention: +1 D, −1 R, 0 open; appointees open; independents by
  caucus.
- **nat_political_f.csv:** a byte-identical copy of the House file.

## `prices_f.csv`

One row per `basic_f.csv` race: `census_region`, `price_data_month` and the 18 CPI-U and gasoline features of the
state's Census region, then the state's eight unemployment features. These are the same values as the House
`prices_f.csv`, from the shared `data/processed/cpi_gas_region.csv` and `data/processed/unemployment_state.csv`;
definitions, sources and the publication-lag rule are in `../README.md` (`cpi_gas.py`, `unemployment.py`). Every race uses September of its election year.

## `acs_f.csv` (`senate_acs.py`)

One row per race: `acs_vintage`, the twelve `acs.py` fields for the state (`white` … `housing_burden`), and the four
national comparisons `rel_college`, `log_rel_income`, `rel_unemployment` and `rel_housing_burden`.

- **The House's ACS, at state level:** the same vintages (2007 1-year for 2006, 2016–2020 5-year for 2020, and 2024
  1-year for 2026), variables, formulas and rounding. The values are `acs_supplement.fetch_reference`'s state rows,
  which the House compares its districts with, fetched for every field (`state:*`). They're saved in
  `data/interim/senate/acs_state.csv`.
- **Check:** each single-district state's House row (district 0) covers the whole state, and all 74 such rows equal
  the state rows in every column.
- **No `rel_state_*` columns:** they would compare a state with itself.
- None of the House's district complications apply: boundary checks, the NC 2020 rebuild, or the 2026 rebuilt
  districts.
- **Timing, as in the House:** an election year's ACS is released the following September, so its values describe the
  election year but weren't public before Election Day.

## `fec_f.csv`

`fundraising_advantage`, `spending_advantage`, `small_dollar_share_diff`, `dem_fec_missing` and `gop_fec_missing`
from `data/processed/senate_fec.csv` (`../fec_finance.py`). `house_features.fec_features` prepares them with the
Senate key, so the definitions match the House exactly.
- **Cap:** the two log advantages are capped at ±5, as in the House. 37 races exceed it, mostly a funded incumbent
  against an unfunded or missing opponent.
- **No known gaps:** every Senate nominee has FEC data. The one empty record, Kunce in MO 2024, is filled from the
  FEC API (see `../README.md`).
- **Blank `small_dollar_share_diff`:** 57 races, either 2006 (no itemized split) or races where a party has no
  nominee or no individual contributions.
- **Coverage:** the totals cover the two-year cycle. A Senate incumbent's money from earlier in the six-year term
  shows only through cash on hand.

## `modeling.csv`

`basic_f`, `acs_f`, `fec_f` and `prices_f` joined on `year, state, class`, then `nat_political_f` on `year`. This uses
`house_features.join_districts` and `add_national`, so every table must have exactly the same race keys and no
column may appear twice. The columns are the House `modeling.csv`'s, in the same order, with `class` for `district`
and without the four `rel_state_*` columns. The only blanks are 2026's generic-ballot fields (`gen_margin_raw`,
`gb_*`) and `small_dollar_share_diff`. There are no targets in this file; they are in `targets.csv`.

## Polls (`senate_polls.py`)

Two files, both inputs for after Stage 1 (they aren't joined into `modeling.csv`):
- `polls_clean.csv`: one row per poll question from RealClearPolling's general-election tables
  (`All-US-Senate-polls-2006-2024/2006_to_2024_senate_polls.csv`), keyed to its race.
- `polls.csv`: one row per race (all 382 in `senate_incumbency.csv`, 2006–2026), the polling average. Change the
  average with `aggregate_polls`'s arguments (`window_days`, `publication_lag_days`, `as_of`); `polls_clean.csv` stays
  as it is.

The columns are described in the script's docstring.

**Telling apart two races in one state-year.** The source has only `year` and `state`, and in two-race state-years
its `overlapping_special_election` flag marks every row. But its rows keep RealClearPolling's page order, so each
run of consecutive rows with the same year and state is one RCP table. `TABLE_CLASSES` assigns each table in a
two-race state-year to its race. Each assignment was checked against the races' Wikipedia polling tables
(pollster, dates, R and D shares), and the evidence is in a comment.

| State-year | Tables → race | Race with no polls in the source |
|---|---|---|
| NY 2010 | 1 → class 3, Schumer (regular) | Gillibrand special |
| OK 2014 | 2 → class 2, Inhofe (regular); class 3, Lankford (special) | — |
| SC 2014 | 1 → class 2, Graham (regular) | Scott special |
| MN 2018 | 2 → class 1, Klobuchar; class 2, Smith (special) | — |
| MS 2018 | 2 → class 1, Wicker; class 2, Hyde-Smith–Espy head-to-heads (special) | — |
| GA 2020 | 2 → class 2, Perdue–Ossoff, November and runoff | Loeffler–Warnock special |
| OK 2022 | 1 → class 3, Lankford (regular) | Mullin special |
| NE 2024 | 1 → class 1, Fischer–Osborn (regular) | Ricketts special |

MS 2008 and WY 2008 have no polls at all. The script stops if a two-race table has no class, or if a
`TABLE_CLASSES` entry matches no table.

**Rules**
- **Stage:** `runoff` for polls that ended after Election Day: LA 2014 (7), the MS 2018 special (1) and GA 2020
  class 2 (13). Before Election Day in a jungle race (LA 2014, the MS 2018 special), RCP lists hypothetical D-vs-R
  head-to-heads, not first-round questions.
- **Independents counted as the Democrat**, as in `targets.csv`: King (ME 2012, 2018) and Sanders (VT 2018), plus
  Gross (AK 2020, the Democratic nominee). The source has them in `other`, so `dem_pct` is the independent's share
  and `other_pct` is the Democratic nominee's. Flag `independent_counted_as_dem`. Orman, McMullin, Osborn,
  Lieberman 2006, Crist 2010 and Murkowski 2010 stay in `other_pct`, as in the targets.
- **Repeated polls:** `rcp_table > 1` is mostly RCP's "with third-party candidate" version of a race. 134 of its
  137 rows repeat a first-table poll (same pollster and dates, margins within a few points), so they're marked
  duplicates. Counting both would count one poll twice.
- **Duplicates (`is_duplicate`, `duplicate_of`, 135):** a poll is a duplicate of the first row (by RCP table, then
  file order) that has either the same race, stage, pollster and field dates (the 134 above), or the same race,
  stage, dates, sample size and D and R shares under another label: OH 2016 "Suffolk" and "Suffolk University".
  There are no exact duplicate rows. Two other same-date, same-share pairs (FL 2018 Gravis and St. Pete Polls,
  PA 2024 Muhlenberg, USA Today/Suffolk and Atlas Intel) have different samples, so they're separate polls.
- **Pollsters:** `pollster` removes RCP's `*` markers, decodes HTML entities and merges spelling variants of one
  firm (`POLLSTER_ALIASES`: Rasmussen / Rasmussen Reports, Gravis / Gravis Marketing, ...). Sponsor/firm labels
  ("NBC News/Marist", "Marist") stay distinct, so `n_pollsters` counts labels. `pollster_partisan` is the label's
  "(D)" or "(R)". `pollster_raw` is the source label.
- **Candidates:** the source names none; its polls are RCP's tables for the eventual nominees. `dem_candidate`,
  `gop_candidate` and `ind_candidate` are the race's nominees from `senate_fec.csv`. `matchup` says which shares
  the poll reports (`dem-gop` 4,602, `dem-gop-other` 236, `gop-other` 31, ...).
- **Margin:** `poll_margin_dem = 100 * (dem_pct - gop_pct) / (dem_pct + gop_pct)`, Democratic-positive.
- **Blank margins (56 polls):** no D or no R share was polled. These are D-vs-D (CA 2016, 2018), no Democrat (AR
  2020), an independent as the main opponent (KS 2014, UT 2022, NE 2024) and one CT 2006 poll. AK 2022 (flag
  `alaska_four_candidates`) has two Republicans, Murkowski and Tshibaka, so the `rep` column is one of them and its
  margins are blanked.
- **Corrections (each checks the source value first):**
  - `DATE_FIXES`: seven start-after-end typos, fixed from Wikipedia (flag `date_fixed`).
  - One NY 2010 row has its sample type in `sample_size` (flag `sample_fields_shifted`).
  - `sample_type` is upper-cased.
- **Dropped tables (`DROPPED_TABLES`, 12 rows):**
  - MD 2006, line 346: `rep` is blank and `dem` is a minor candidate's 4–16%.
  - ME 2024: King is missing (R 33–35, Costello 7–9, other 1–3).

**Coverage:** 259 of the 347 races of 2006–2024 have polls. 2006 has polls for only 13 states. 2010 stops at
Oklahoma, so its OR, PA, SC, SD, UT, VT, WA, WI and WV races have none (PA, WA and WI were competitive). Most
other gaps are safe seats. The script prints the full list.

**The polling average (`polls.csv`)**
- **Eligible polls:** general stage, not a duplicate, with a `poll_margin_dem`, and fieldwork that ended in the 30
  days before Election Day (`window_start` = Election Day − 30 through the day before; Election Day excluded).
- **Cutoff:** `cutoff_date` is the day before Election Day (`data/reference/election_dates.json`, as the generic
  ballot's `gb_cutoff_date`). With `--as-of` it's that date if earlier, for a cycle in progress, and polls that ended
  later aren't used.
- **Availability:** the source has no release dates, so a poll counts as public `PUBLICATION_LAG_DAYS` (0) after its
  fieldwork ended. A poll that ended the day before Election Day but came out on Election Day is still counted; raise
  the lag to exclude such polls.
- **Average:** the simple mean of the eligible `poll_margin_dem`, with no sample-size, voter-type or house-effect
  weighting. `n_polls`, `n_pollsters` (distinct labels) and `latest_poll_date` describe the eligible polls.
- **No polls:** `poll_margin_dem` blank (never 0), `n_polls = 0`, `poll_available = False`. Nothing is imputed.
- **Exceptions in `quality_flags` (from the nominees, so known before the election):**
  - `independent_vs_gop`: KS 2014, UT 2022, NE 2024 class 1, NE 2026. The polls have no Democratic share, so there's
    no average.
  - `no_dem_nominee`, `no_gop_nominee`, `independent_counted_as_dem`.
- **Runoffs:** the average is for the November round. GA 2020 class 2's runoff polls are in `polls_clean.csv` only.
  The MS 2018 special's only eligible poll is a Hyde-Smith–Espy head-to-head.

| Year | Races | With a 30-day average | Mean polls per averaged race |
|---|---|---|---|
| 2006 | 33 | 11 | 12.5 |
| 2008 | 35 | 22 | 7.0 |
| 2010 | 37 | 25 | 7.4 |
| 2012 | 33 | 27 | 6.7 |
| 2014 | 36 | 33 | 6.0 |
| 2016 | 34 | 22 | 7.3 |
| 2018 | 35 | 27 | 5.1 |
| 2020 | 35 | 20 | 7.2 |
| 2022 | 35 | 23 | 6.4 |
| 2024 | 34 | 17 | 10.5 |
| 2026 | 35 | 11 (FiftyPlusOne) | — |

227 of the 347 races of 2006–2024 have an average, and the script prints the other 120. Most are safe seats, or
races with polls only before the window. The rest come from the coverage gaps above. 2026 has no RCP polls, since that source ends
in 2024.

**2026: FiftyPlusOne averages (`add_published_averages`).** 11 races get the election-eve FiftyPlusOne Senate
polling average, transcribed by hand on 2026-11-02 into `data/reference/polls_2026_fiftyplusone.csv`: AK, FL (class 3
special), GA, IA, KS, ME, MI, NC, NH, OH (class 3 special) and TX. No other race, NE included, had one.
- `poll_margin_dem` is computed from the D and R shares as above, `poll_source` names the file, and `n_polls`,
  `n_pollsters` and `latest_poll_date` are blank (FiftyPlusOne's poll counts aren't known).
- A year never mixes the two sources.
- **Not the same method as 2006–2024:** FiftyPlusOne's average isn't a simple 30-day mean of RCP polls, so its errors
  may differ from the historical averages that any blend is calibrated on.

## Expert ratings (`senate_ratings.py`)

`expert-ratings.csv` has one row per race 2016–2026 (208). Each row has the Sabato's Crystal Ball, Cook Political
Report and Inside Elections ratings, their scores and a consensus. It's a post-Stage 1 input, not joined into
`modeling.csv`.

- **Sources:**
  - 2016–2024: the JHK Forecasts archive (`forecast` sabato, cook, inside), the final pre-election ratings. Only
    `forecast, year, office, s_id, probwin, rating` are read, so the archive's outcome columns (`probwin_outcome`,
    `correct`, `brier`, ...) never reach the file.
  - 2026: entered by hand in `data/reference/ratings_2026.csv`.
- **Race keys:** JHK's `s_id` is the state, or `<state>-special` and `NES` (NE 2024) for specials. These are matched
  to the state-year's regular or special race in `senate_incumbency.csv`, so the two races of MN and MS 2018,
  AZ/GA 2020, OK 2022 and NE 2024 stay separate. The key is `class`, as in every Senate file, with `election_id`
  (`<year>_<state>_<class>`) and `special`.
- **Scores:** `<provider>_prob_dem = 1 − RATING_GOP_SCORE[rating]`, since JHK's `probwin` is the Republican's score.
  The script checks that every archive `probwin` equals the score of its rating.

  | Rating | Solid D | Likely D | Lean D | Tilt D | Tossup | Tilt R | Lean R | Likely R | Solid R |
  |---|---|---|---|---|---|---|---|---|---|
  | GOP score | 0.01 | 0.10 | 0.25 | 0.40 | 0.50 | 0.60 | 0.75 | 0.90 | 0.99 |
  | `prob_dem` | 0.99 | 0.90 | 0.75 | 0.60 | 0.50 | 0.40 | 0.25 | 0.10 | 0.01 |

  Inside Elections' Tilt ratings are kept. **These are heuristic encodings of ordinal ratings, not calibrated win
  probabilities, and not vote margins.**
- **Consensus:** `expert_consensus_dem = 0.40 sabato + 0.30 cook + 0.30 ie`. If a provider were missing, its rating
  and score would stay blank, the remaining weights would be renormalized, `coverage_complete` would be False and
  `missing_<provider>` flagged. Every race 2016–2026 has all three ratings, so none is. The weights only combine the
  three providers; they aren't weights against polls or fundamentals.
- **`as_of_date`** is blank: neither source dates its ratings.
- **Exceptions (`quality_flags`, as in `polls.csv`):** in these races `prob_dem` is the probability that the
  Republican loses, which isn't a Democratic win:
  - `independent_vs_gop`: UT 2022 (McMullin), NE 2024 class 1 and NE 2026 (Osborn).
  - `no_dem_nominee`: AR 2020, and ID and SD 2026, whose opponents are independents.

  Also: `independent_counted_as_dem` (King, Sanders) and `no_gop_nominee` (CA 2016, 2018).
