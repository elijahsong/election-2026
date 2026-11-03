# Interim data layout

Interim data that is House-specific (district-level) lives in `data/interim/house/`: `acs.csv`, `acs_2026.csv`,
`acs_supplement.csv`, `house_incumbency.csv`, `house_partisan_lean.csv`, `partisan_lean*.csv` and the cleaned Cook PVI
files. National and Senate data stays in `data/interim/`: `generic_ballot_all.*`, `presidential.csv` and
`senate_incumbency.csv`.

House-specific scripts and notebooks (generic ballot, ACS demographics, partisan lean, modeling features and targets)
live in `house/`; see `house/README.md`. The Senate pipeline (statewide and county targets, state and county PVI,
`basic_f.csv`) lives in `senate/`; see `senate/README.md`. `incumbency.py`, `fec_finance.py`, `cpi_gas.py` and `unemployment.py` stay here because they build both House and Senate data.

# House and Senate incumbency (`incumbency.py`)

```bash
python data-processing/incumbency.py    # -> data/interim/house/house_incumbency.csv and data/interim/senate_incumbency.csv
```

Downloads and caches CAGE on first run, then builds one row per race for 2006–2026:

- **`house_incumbency.csv`:** keyed `year, state, district`, like `acs.csv`.
- **`senate_incumbency.csv`:** keyed `year, state, class`, with `special = 1` for a special election.

## Schema

`year,state,district,inc,inc_signed,incumbent_party,incumbent_names,status,source` (House)

`year,state,class,special,inc,inc_signed,incumbent_party,incumbent_names,status,source` (Senate)

| Column | Meaning |
|---|---|
| `inc` | 1 if an incumbent is running in the general election, 0 for an open seat |
| `inc_signed` | +1 for a Democratic incumbent, −1 for a Republican, 0 for an open seat. Blank when it isn't a single major party; never treat blank as 0 |
| `status` | `incumbent`, `open`, `open: appointed incumbent …`, `override: …`, or `review: …` (the blank `inc_signed` rows) |

## Sources

- **2006–2024:** Cha, Kuriwaki & Snyder, *Candidates in American General Elections* (CAGE), Harvard Dataverse,
  [doi:10.7910/DVN/DGDRDT](https://doi.org/10.7910/DVN/DGDRDT), v4.1, CC0.
  - The file is checksum-pinned and cached in `data/raw/cage/`.
  - CAGE codes `inc` 1 (regular) and 2 (won a special election) count as incumbents.
  - **Appointed incumbents are coded open.** CAGE `inc = 3` (14 Senate races, e.g. Gillibrand 2010, Loeffler 2020,
    Ricketts 2024) gives `status = open: appointed incumbent …`.
- **House 2026:** Cook's 2026 PVI list (`data/raw/pvi.sources/cook_pvi_public_2026.csv`), which is on the 2026 lines.
  - `OPEN (…)` rows are open seats; their party column is the departing member's.
  - Every other row's incumbent runs with the listed party.
- **Senate 2026:** `data/raw/incumbency_2026/senate_2026.csv`, with one row per race and columns `state, class, special, inc, party`.
  - `inc` is **unsigned** (1 = an incumbent runs, 0 = open). `party` is the running incumbent's party and is blank for open seats.
  - The sign is `party`'s. The reader raises unless `party` is given exactly when `inc` is 1. The file has no names.

## House details

- Only House general-election candidates are used (`office = H`, `type = G`).
- CAGE numbers at-large seats 1. They are renumbered 0, and every year's districts are checked against
  `data/reference/district_counts.json`.

## Senate details

- **Seat class:** comes from CAGE's `nextup`, the year the seat is next on the ballot. `nextup % 6` is 2 for class I, 4 for
  class II and 0 for class III, so a special election gets the class of the seat it fills.
- **Validation:**
  - Each class has the same states in every cycle, 2026 included (33 / 33 / 34).
  - Every state sits in exactly two classes.
  - Each year's regular races are its own class, and its specials are for other classes.
- **Data fixes:**
  - **LA 2014** (`NEXTUP_FIXES`): the Landrieu–Cassidy runoff is coded next up in 2016. It's a class II seat, next up in 2020.
  - **MA 2010** (`NON_NOVEMBER_SPECIALS`): the January 2010 special is dropped, since it wasn't on the November ballot.
  - **CA 2022 and 2024:** the same-day specials for a seat's last weeks (`nextup == year`) are dropped; each repeats that
    seat's full-term race.

## Overrides

`HOUSE_OVERRIDES` and `SENATE_OVERRIDES` are applied after all years are combined. Each override must match exactly one row.

**House:**
- **Source corrections:**
  - TX-34 2024: Mayra Flores is coded as an incumbent, but she lost in 2022, so the row becomes +1 (Gonzalez).
  - MI-7 2006 and KS-1 2016: the incumbent lost the primary and ran as a write-in, so the rows become open (0).
  - LA 2006 (districts 1, 3, 4, 5, 6, 7): CAGE codes every incumbent in Louisiana's November jungle round as a
    non-incumbent. Only LA-2, whose rows are the December runoff, is coded. Jindal, McCrery, Alexander, Baker and
    Boustany become −1 and Melancon +1; each won the seat in 2004. Checked against CAGE's own prior-year winners
    (2008–2024) and MEDSL's 2004 winners (2006): no other incumbent is missed.
- **Incumbents of both parties in one redrawn district, resolved by hand:**
  - IA-3 2012 (Latham) and OH-16 2012 (Renacci) are coded −1, as Republican incumbents.
  - PA-17 2018, FL-2 2022 and TX-34 2022 are coded open (0).
- **CA-6 2026:** Cook lists Kevin Kiley as an independent; he's coded −1, as a Republican incumbent.

**Senate** (independents and non-nominee incumbents are coded by caucus or original party):
- **ME 2012:** CAGE codes Angus King as an incumbent, but Snowe retired and King was first elected that year, so it's open (0).
- **Sanders** (VT 2012, 2018 and 2024) and **King** (ME 2018 and 2024) caucus with Democrats, so they're +1.
- **Lieberman** (CT 2006) is +1: a Democratic incumbent who lost the primary and won as an independent.
- **Murkowski** (AK 2010) is −1: a Republican incumbent who lost the primary and won as a write-in.
- **FL and OH 2026 specials** (class III; Rubio's and Vance's seats) are coded open (0). Their appointed incumbents are running,
  but `senate_2026.csv` doesn't mark appointments, so these are overrides.

With these, every row has an `inc_signed` value. A new unresolved case would show up as `status = review: …` with a blank
`inc_signed`.


# 2026 nominees (`nominees_2026.py`)

```bash
python data-processing/nominees_2026.py    # -> data/raw/nominees_2026/{house,senate}.csv and SOURCE.json
```

The repo's other 2026 sources name only running incumbents, so this builds the 2026 general-election candidate lists
from Wikipedia: "2026 United States House of Representatives elections" and "2026 United States Senate elections".

- **Pinned revisions:** both pages are read at the revisions in `PAGES` (retrieved 2026-10-30). The API response is
  cached unmodified in `data/raw/nominees_2026/`, so a rerun parses the same text offline. `SOURCE.json` records each
  title, revision, timestamp, URL and sha256.
- **Sourcing:** each House state table cites its official source, usually the state's certified candidate list. Its
  URL is kept in `source`.
- **One row per listed candidate:** columns `year, state, district` (House) or `year, state, class, special` (Senate),
  then `party` (D / R / other), `party_label` and `name`.
  - Third parties are kept, and so are a party's several candidates in a top-two or top-four race, or in Louisiana's
    November 2026 round.
  - Senate class and special come from `data/raw/incumbency_2026/senate_2026.csv`.
- **Blocking checks:**
  - The House rows must cover exactly 2026's 435 districts, and the Senate rows exactly `senate_2026.csv`'s 35 races.
  - Every `*{{Party stripe}}` candidate item must parse.
- **Report-only checks:**
  - races with several or no candidates of a major party;
  - Cook-listed running incumbents (`house_incumbency.csv`) missing from their district's list. In the pinned revision
    these are 4 `VACANT` seats that `house_incumbency.csv` codes as running incumbents, 2 Cook misspellings, and
    7 incumbents who aren't on the ballot.

# FEC candidate finance (`fec_finance.py`)

```bash
python data-processing/fec_api_totals.py    # once; cached in data/raw/fec_candidate_summaries/api/
python data-processing/fec_finance.py       # after nominees_2026.py
```

One row per general-election race, with each major party's nominee and that nominee's FEC cycle totals.
`house_targets.py` and `incumbency.py` are reused read-only, and no other file is changed.

| File | Key | Contents |
|---|---|---|
| `data/interim/house_fec_raw.csv` | `year, state, district` | `race_type` (from `targets_raw.csv`; blank for 2026), then per party (`dem_`, `gop_`) the fields below |
| `data/interim/senate_fec_raw.csv` | `year, state, class` | `special`, then per party the fields below |
| `data/processed/house_fec.csv`, `senate_fec.csv` | same | the interim columns plus the five features and `small_dollar_share_flag` |
| `data/interim/fec_match_review.csv` | | every slot that isn't a clean full-name match, with the FEC records considered |

Per-party fields:

- **Identity:** `name`, `candidate_id`, `match_status`, `fec_missing`, `candidate_count` (the party's general-election
  candidates) and `fec_file`.
- **Finance:** `receipts`, `disbursements`, `individual_contributions`, `individual_unitemized_contributions`,
  `individual_itemized_contributions`, `cash_on_hand`, `debt`, `coverage_start_date`, `coverage_end_date`.

Features:
- `fundraising_advantage = log1p(dem_receipts) − log1p(gop_receipts)`.
- `spending_advantage`: the same, with disbursements.
- `small_dollar_share_{dem,gop}`: unitemized ÷ individual contributions. Blank when the denominator is 0.
- `small_dollar_share_diff`: D − R. Blank when either share is blank.
- A share outside [0, 1] means FEC's own unitemized and individual totals are inconsistent. It's set blank and named
  in `small_dollar_share_flag` (`dem_share_outside_0_1`, `gop_share_outside_0_1`), and the FEC amounts are kept.

## FEC sources (`data/raw/fec_candidate_summaries`)

- **2008–2026:** `candidate_summary_YYYY.csv`, FEC's candidate summary file (50 columns, with a header; the same
  columns in every year).
  - 2008–2024 were downloaded from FEC bulk data on 2026-10-30; see that folder's `README.md` / `SOURCE.json`.
- **2006:** `webl06.txt`, FEC's older "current campaigns for House and Senate" layout: pipe-delimited, no header,
  30 columns, with the office read from the first letter of the ID.
  - It has no itemized / unitemized split and no coverage start date, so those fields, and the small-dollar shares,
    are blank for 2006.
- **`webl08–24.txt`:** a fallback for nominees missing from that year's candidate summary (e.g. Spartz, IN-5 2024), and
  a cross-check: receipts agree for all but 15 of ~7,600 matched records.
- **Coverage:** each summary covers the two-year cycle, with each candidate's own coverage dates. A Senate incumbent's
  earlier money shows only through cash on hand.
- **Timing mismatch (accepted):** 2006–2024 totals mostly run through December 31, after the election, while 2026
  runs through mid-October, before it. A model trained on the historical totals learns from end-of-cycle money.
  Matching a pre-general cutoff in every year would need report-level data that the summary files don't have.

## Nominees

- **Sources:**
  - House 2006–2024: MEDSL via `house_targets` (November / first round).
  - Senate 2006–2024: CAGE via `incumbency.senate_candidates`, which gives class and special.
  - 2026: `nominees_2026.py`.
- **A party's slot is its leading general-election candidate by votes:** this covers jungle, top-two, RCV and party
  write-ins. Primary candidates and their money are never used.
- **2026, which has no votes:** the party's only candidate fills the slot. With several, the one FEC flags `INCUMBENT`
  does, provided exactly one is.
  - **Louisiana (`TOP_FUNDRAISER_STATES_2026`):** failing that, the party's top FEC fundraiser does
    (`resolve_2026_by_receipts`; a tie, or nobody with receipts, stays unresolved). November is Louisiana's
    all-candidate round, where 2006–2024 used the leading vote-getter; before votes, the top fundraiser is the stand-in
    known in advance (decision 2026-11-02). This fills six slots: the D slots of LA-3 to LA-6 and the R slots of LA-5
    and LA-6. The review file lists every candidate's receipts.
  - Otherwise the slot is `nominee_unresolved`: CA-11 and CA-14 (two Democrats) and CA-40 (Calvert vs Kim). These
    top-two races have no opposite-party candidate, so `fundraising_advantage` is blank there. How to handle them is
    still to be decided.
- **`INDEPENDENTS_AS_DEM`:** independents who caucus with Democrats take the `dem_` slot (`independent_as_dem`, with
  `dem_is_independent` True; the report lists any D nominee displaced):
  - Sanders (VT 2006, 2012, 2018, 2024)
  - King (ME 2012, 2018, 2024)
- **`INDEPENDENTS_AS_GOP`:** an independent takes the `gop_` slot as the Republican proxy for the regression and
  fundraising features (`independent_as_gop`). This is Kiley, CA-6 2026: the incumbent, running as an independent
  against Pan (D) with no Republican. Only the model slot changes; his ballot party stays independent, and Stage 2
  counts a Kiley win as OTHER (`src/house/simulation/races.py`).
- **`INDEPENDENT_CANDIDATES`:** other leading independents stay independents. They never fill the `dem_` slot, which
  keeps the race's Democrat (or `no_nominee`), and their FEC totals are kept in a third slot, `ind_` (same columns
  as `dem_`, status `independent`; blank in every other race). The features use only `dem_` and `gop_`.
  - Orman (KS 2014), McMullin (UT 2022), Osborn (NE 2024, 2026): no D nominee, so `dem_` is `no_nominee`.
  - Buchanan (MT-2 2022): `dem_` is Ronning (D).
- **Strict otherwise:** party labels are never relaxed. A race without a D or R candidate is `no_nominee`, and no D–R
  matchup is invented.
- **Source fix:** `CAGE_SWAPPED_RACES`. CAGE has Maine's 2012 and 2018 class I races swapped (its 2012 rows are the
  2018 race, Brakey and Ringelstein), so the two trade years.

## Matching

No crosswalk to FEC IDs exists, so nominees are matched by name. The pools are tried in order:

1. The same year, office, state, district (House) and party.
2. The same, but any party label; full names only (`name_party_differs`).
3. House only: another district of the state, full names only, among records no race has matched
   (`name_other_district`; FEC's district is often stale after redistricting).

Candidate summary records are tried before webl-only ones. Nothing is matched on fundraising.

| `match_status` | Meaning |
|---|---|
| `name_full` | surname and first name (or an initial or nickname, or a middle name in use) agree, with no conflicting middle initial |
| `name_variant` | full after a filing quirk: FEC's FIRST, LAST order (`NICK, LALOTA`), or a surname filed as a middle name (`ARENHOLZ, ASHLEY HINSON`) |
| `name_surname` | the surname alone, unique in its pool |
| `name_fuzzy` | first name plus a surname at least 80% similar, unique; same district and party only (MEDSL's `JEFF DENHEM`, CAGE's `HASSEN`) |
| `duplicate_registration` | one person's several IDs: the only one with filings, or one of records with identical totals |
| `fec_api` | `API_CORRECTIONS`: cycle totals from the FEC API for a nominee missing from both bulk files |
| `override` | `MATCH_OVERRIDES`: 15 reviewed cases, each commented (e.g. Gus vs Michael Bilirakis, Ron Estes vs Ron M. Estes) |
| `independent_as_dem` | `INDEPENDENTS_AS_DEM` |
| `independent_as_gop` | `INDEPENDENTS_AS_GOP` |
| `independent` | `INDEPENDENT_CANDIDATES` (the `ind_` slot) |
| `no_fec_record` / `ambiguous` | the nominee is known, but no record (or no single record) matches |
| `no_nominee` | no candidate of that party on the general-election ballot |
| `nominee_unresolved` | several candidates of the party and no way to pick one (2026) |

**Missing data:**
- **No nominee or no usable record** (`no_nominee`, `no_fec_record`, `ambiguous`): amounts are 0, with `fec_missing`
  True.
- **`nominee_unresolved`:** amounts are blank, since the nominee isn't known.
- **Never filled in:** names, IDs and dates. A blank in a matched FEC record stays blank, and so do 2006's missing
  fields. Negative source values are kept and reported; a log feature of an amount below −1 is blank.

## Checks and known gaps

- **Blocking:**
  - Every input file has its expected columns.
  - Race keys are unique and equal `targets_raw.csv`'s (plus 2026's districts) and `senate_incumbency.csv`'s.
  - No FEC record fills two slots.
  - Each matched record's office and state are its race's.
  - Every override and exception applies.
  - The processed files have exactly the interim rows and columns plus the five features.
- **Reported:**
  - match coverage by chamber and year;
  - negative values, blank amounts, and itemized + unitemized ≠ individual;
  - shares outside [0, 1] (14 House rows), set blank and flagged;
  - FEC's own 2006–2012 general-election flags against the slots;
  - **possibly empty records** (`suspect_records`): nominees who won, or finished within 20 points, whose FEC money is
    under $50,000 with no coverage dates or coverage ending before July of the election year. See below.
- **Gaps in FEC's files:** these nominees have no record in either bulk file, or only an empty one (every amount 0,
  no coverage dates):
  - 11 House winners and two challengers (Tarkanian NV-3 2018, Martin AZ-2 2020);
  - Cisneros (CA-31 2024), Tinsley (NM-2 2008) and Sonza (OH-1 2024), whose candidate summary rows are empty;
  - Kunce (MO 2024), in the Senate.

  `API_CORRECTIONS` maps each slot to its candidate ID. A correction applies to a `no_fec_record` slot or to one
  matched to such an empty record.
  - `fec_api_totals.py` fetches the FEC API's two-year totals for each one
    (`/v1/candidate/{id}/totals/?cycle=YEAR&election_full=false`). The fields agree with the bulk file where a
    candidate is in both (checked on Donalds 2022).
  - **Committee totals:** for 10 of the 17 corrections, FEC's candidate totals are empty or leave out the campaign,
    though its principal committee filed. `API_COMMITTEE_TOTALS` maps each of these candidates to that committee, and
    `fec_api_totals.py` takes the committee's two-year totals instead (`/v1/committee/{id}/totals/?cycle=YEAR`). Each
    committee was checked against the candidate's committee list on 2026-11-02:
    - **Empty candidate totals:** Ivey (MD-4 2022, 2024), Self (TX-3 2022, 2024), Stutzman (IN-3 2024) and McGuire
      (VA-5 2024; his $7,000 recount committee is left out). Each committee is designated `P`.
    - **Gillen (NY-4 2024):** her candidate totals ($4.89, January 2023) are only her 2022 committee winding down. Her
      2024 campaign is Gillen for NY (C00840165): $6.83M.
    - **Kunce (MO 2024, Senate):** Heartland Patriots (C00771972) is his principal committee for 2022–2026, but is
      designated `U` for 2024, so his candidate totals are 0. It raised $23.2M.
    - **Cisneros (CA-31 2024):** Cisneros for Congress (C00850420), $6.48M.
    - **Tinsley (NM-2 2008):** Tinsley for Congress (C00441394), $2.40M. It is a principal House campaign committee
      in Capitan, NM, which FEC links to no candidate ID.
    - **Sonza (OH-1 2024), by decision:** FEC lists no campaign committee for him. The only committee linked to his
      ID is Southwest Ohio Leadership PAC (C00845065), a leadership PAC, first filed 2023-07-10, with $1.17M in
      2024. Its totals are counted as his campaign's money, though leadership PAC receipts aren't campaign receipts
      in FEC's terms.
  - Each response with data is cached unmodified in `data/raw/fec_candidate_summaries/api/`, and `SOURCE.json` there
    records its URL (key redacted), time and sha256. Responses with no data aren't kept.
  - **Filled (`fec_api`), all 17:**
    - Trone 2018, Tarkanian 2018; Issa, Jacobs, Donalds and Martin 2020 (candidate totals).
    - Tinsley 2008; Ivey and Self 2022 and 2024; Stutzman, McGuire, Gillen, Cisneros and Sonza 2024 (committee
      totals).
    - Kunce 2024, Senate (committee totals).

    No House winner is left without FEC data.
- **Possibly empty records, reviewed 2026-11-02:** `python data-processing/fec_api_totals.py --review` prints, for
  each slot `suspect_records` flags, the nominee's FEC committees with their cycle totals and any other FEC
  candidates of that surname, state and office. It only prints. The first run flagged 24 slots:
  - **Wrong record matched** (now `MATCH_OVERRIDES`):
    - Jason Smith, MO-8 2016: his record's party is `UNK`, so the surname match took Phillip S. Smith's empty one.
    - Mike Kelly, PA-3 2016: filed as George J. Kelly Jr.
    - Saucedo Mercer, AZ-3 2014, and Ben Clark, NC-9 2022: filed for NC-4 before the 2022 redraw.
  - **No usable bulk record** (now `API_CORRECTIONS`): Tarkanian, Martin, Cisneros, Tinsley and Sonza, above.
  - **Genuinely low-funded** (`REVIEWED_LOW_FUNDING`, no longer reported): 15 slots with no committee totals for
    the cycle, or only $40 (Hewett, WA-10 2024) or $17,105 (Conley, SC 2008 Senate).
  - None is left unresolved.
- **2006 Republicans:** about 11% of 2006 R nominees (44, mostly in safe D seats) have no `webl06` record.

# CPI-U and gasoline prices (`cpi_gas.py`)

```bash
python data-processing/cpi_gas.py           # -> data/processed/cpi_gas_monthly.csv, data/processed/cpi_gas_region.csv
python data-processing/house/house_features.py     # then: House prices_f.csv, joined into modeling.csv
python data-processing/senate/senate_features.py   # and: Senate prices_f.csv
```

Regional inflation and gasoline-price features shared by the House and Senate models. Each district or race gets its
state's Census region row; nothing is computed per chamber.

| File | Key | Contents |
|---|---|---|
| `data/processed/cpi_gas_monthly.csv` | `month, region` | Monthly levels from January 2004: `cpi` (CPI-U all items, NSA, 1982–84 = 100) and `gas` (regular gasoline, $/gallon) for `us` and the four regions |
| `data/processed/cpi_gas_region.csv` | `year, census_region` | `election_date`, `price_data_month` and the 18 features for each election year 2006–2026 |
| `data/processed/house/prices_f.csv` | `year, state, district` | `census_region`, `price_data_month` and the features |
| `data/processed/senate/prices_f.csv` | `year, state, class` | the same |

## Sources

| Series | Region | File |
|---|---|---|
| CPI-U, all items, NSA | U.S. city average `CUUR0000SA0` | `data/raw/fred-cpi/SeriesReport-20261102172633_254299.csv` (BLS Data Finder export, 2005–2026) |
| | Northeast `CUUR0100SA0`, Midwest `CUUR0200SA0`, South `CUUR0300SA0`, West `CUUR0400SA0` | `data/raw/fred-cpi/<id>.csv` (FRED) |
| Average price, unleaded regular gasoline, $/gallon | U.S. `APU000074714`, Northeast `APU010074714`, Midwest `APU020074714`, South `APU030074714`, West `APU040074714` | `data/raw/fred_gas/fred-gas-price.csv` (FRED, all five) |

- All ten series are BLS's. FRED republishes them unchanged: BLS API values for 2015–2024 match the FRED files exactly
  (checked 2026-11-02 on `CUUR0100SA0`, `APU000074714` and `APU040074714`).
- Every CPI series here is NSA (FRED's seasonally adjusted `CPIAUCSL` isn't used), so the national and regional changes are comparable. NSA values are never revised, while seasonal factors
  are re-estimated each year.
- **Census regions** (`STATE_REGION`): Northeast 9 states, Midwest 12, South 16 plus DC, West 13 (including AK and
  HI). These match BLS's regional CPI and average-price areas. A district never crosses a state line, so House
  districts use their state's region.

## Timing (no look-ahead)

BLS publishes month M's CPI and average prices in mid-month M + 1: September 2024 came out on October 10, 2024, and
October 2024 on November 13, 2024. `data_month` treats a month as public from the first day of the month
`PUBLICATION_LAG_MONTHS` (2) after it. Election Day falls on November 2–8, so every election uses **September**
(`price_data_month`), the latest month out before the vote. October's figures are never used. September is enforced:
the script raises for any other data month, so every 12- and 24-month change is September over September and every
6-month change September over March. Both series are final
on release, so today's files show what was public before each election.

## Features

M is `price_data_month`. A percent change is `100 × (x(M) / x(earlier) − 1)`, so 3.5 means 3.5%.

| Column | Definition |
|---|---|
| `cpi_region_yoy` | regional CPI-U, M vs M − 12 (%) |
| `cpi_us_yoy` | national CPI-U, M vs M − 12 (%) |
| `cpi_region_minus_us_yoy` | `cpi_region_yoy − cpi_us_yoy` (percentage points) |
| `cpi_region_24m` | regional CPI-U, M vs M − 24 (%) |
| `cpi_region_6m` | regional CPI-U, M vs M − 6 (%) |
| `gas_region_price_3m` | mean regional gasoline price over M − 2 … M ($/gallon) |
| `gas_region_yoy` | regional gasoline price, M vs M − 12 (%) |
| `gas_region_minus_us` | regional − national gasoline price in M ($/gallon) |
| `gas_region_real_price` | regional gasoline price in M × regional CPI(Sep 2026) / regional CPI(M): **September 2026 dollars** (`REAL_PRICE_REFERENCE_MONTH`), so 2026's real price equals its nominal price |
| `gas_region_6m_change` | regional gasoline price, M vs M − 6 (%) |

### Log features

These use natural logs, in the same form as `log_rel_income` (`log1p((x − ref) / ref)`). A log change is
`100 × log(x(M) / x(earlier))`, close to the percent change for small moves. The percent-change and level features
above are kept for comparison.

**Real prices for the log features are in September 2005 dollars** (`DEFLATOR_BASE_MONTH`). Each CPI series, national
and regional, is rescaled to September 2005 = 100, and `real = gas / (CPI / 100)`. Regional gasoline is deflated by
the regional CPI, national gasoline by the national CPI. This uses a different reference from
`gas_region_real_price` (September 2026 dollars), which is unchanged.

| Column | Definition |
|---|---|
| `cpi_region_log_yoy` | `100 × log(CPI_r(M) / CPI_r(M − 12))` |
| `cpi_region_log_diff_us` | `cpi_region_log_yoy − 100 × log(CPI_us(M) / CPI_us(M − 12))` (percentage points) |
| `log_real_gas_price` | `log(real_r(M))`, September 2005 dollars |
| `log_rel_gas_price` | `log(gas_r(M) / gas_us(M))`, nominal |
| `log_rel_real_gas` | `log(real_r(M) / real_us(M))` |
| `log_gas_yoy` | `100 × log(gas_r(M) / gas_r(M − 12))` |
| `log_real_gas_yoy` | `100 × log(real_r(M) / real_r(M − 12))` |

The relative trend compares the region's 12-month change with the nation's, as `cpi_region_minus_us_yoy` does for CPI:

| Column | Definition |
|---|---|
| `gas_region_minus_us_yoy` | `gas_region_yoy` − national gasoline price, M vs M − 12 (percentage points) |

**What `log_rel_real_gas` measures.** It equals `log_rel_gas_price` minus the regional − national gap in log CPI
growth since September 2005. That is, it's the nominal gasoline-price gap adjusted for the difference in general
inflation accumulated since that date. It is **not** a cost-of-living comparison. A regional CPI index shows how
prices in that region changed over time, not how high they are compared with another region. Rescaling every series
to 100 in September 2005 assumes nothing about relative price levels in that month. It only gives every deflator the
same reference date. A different base month would shift `log_real_gas_price` and `log_rel_real_gas` by a constant per
region and leave `log_real_gas_yoy` unchanged.

Features are rounded to 4 decimals. Rerunning the script produces byte-identical files.

## Missing data and limitations

- **Nothing is filled in.** A feature that needs a missing month is blank, and so is a 3-month average missing any
  month. A data month not yet published would leave its year blank; it's never replaced by an earlier month.
- **Start dates:** the regional CPI files begin in January 2004, so 2006's 24-month change (from September 2004)
  exists. National CPI and gasoline begin in January 2005, so their 2004 months are blank; no feature uses them.
  Every feature is filled in for every election year 2006–2026.
- **October 2025:** BLS published no CPI for it, because of the government shutdown, and the month stays blank in
  `cpi_gas_monthly.csv`. No election feature uses it (2026's windows start in September 2024 and September 2025, or
  later), but monthly analyses spanning it will see the gap. Gasoline prices for October 2025 were published.
- **Regions are broad:** every state in a region gets the same values, e.g. Texas and Delaware share the South's.
  Within-region differences, such as California against the rest of the West, aren't captured.
- Each region's real gasoline price is deflated by that region's own CPI. Real prices are comparable across years
  within a region, and roughly comparable across regions (all four indexes use 1982–84 = 100).

# Unemployment rates (`unemployment.py`)

```bash
python data-processing/unemployment.py      # -> data/processed/unemployment_monthly.csv, unemployment_state.csv
python data-processing/house/house_features.py     # then: added to House prices_f.csv and modeling.csv
python data-processing/senate/senate_features.py   # and: Senate prices_f.csv
```

Each race and district gets its state's unemployment features. They sit in `prices_f.csv` next to the CPI and
gasoline features, and use the same timing rule.

| File | Key | Contents |
|---|---|---|
| `data/processed/unemployment_monthly.csv` | `month, area` | Monthly unemployment rate (%, seasonally adjusted) from January 2004 for the 50 states, DC and `US` |
| `data/processed/unemployment_state.csv` | `year, state` | `unemp_data_month` and the eight features for each election year 2006–2026 |

## Sources (`data/raw/unemployment_rate/`)

- `monthly.csv`: FRED's data list 337, "State level unemployment rate" (`<USPS>UR`, e.g. `CAUR`, for the 50 states
  and DC). These are BLS Local Area Unemployment Statistics, seasonally adjusted, downloaded 2026-11-02 (FRED vintage
  2026-10-20; `README.txt` lists every series).
- `UNRATE.csv`: FRED `UNRATE`, the national rate from the Current Population Survey, seasonally adjusted.

## Timing

The data month is September, the same `cpi_gas.data_month` rule as the CPI features, which raises for any other
month. BLS publishes the national September rate in early October and the state rates about three weeks after the
month ends, so both are out before Election Day. October's national rate is sometimes released before Election Day
(November 1, 2024), but the state rates never are. So September is used for both, which keeps the state − national
gap comparable.

## Features

M is September of the election year. Rates are in %, and gaps and changes in percentage points. Logs are natural
logs, in the `log_rel_income` form.

| Column | Definition |
|---|---|
| `unemp_state_rate` | `u_s(M)` |
| `unemp_us_rate` | `u_US(M)` |
| `unemp_state_minus_us` | `u_s(M) − u_US(M)` |
| `log_unemp_state` | `log(u_s(M))` |
| `log_rel_unemp` | `log(u_s(M) / u_US(M))` |
| `unemp_change_12m` | `u_s(M) − u_s(M − 12)`: September over the previous September |
| `unemp_change_3m` | `u_s(M) − u_s(M − 3)`: September over June |
| `unemp_change_12m_minus_us` | `unemp_change_12m − (u_US(M) − u_US(M − 12))`: the state's 12-month change against the nation's |

The rates are published to one decimal, so the rates, gaps and changes keep 1 decimal and the logs keep 4.

## Missing data and limitations

- **Nothing is filled in.** October 2025 is blank for every area: BLS published no rate for it, because of the 2025
  shutdown. No election feature uses it (2026 uses June and September 2026, and September 2025). Every feature is
  filled in for every election year 2006–2026.
- **Revisions (accepted, not fixed):** unlike NSA CPI, seasonally adjusted unemployment rates are revised after
  release. The state series are re-benchmarked every year, with several past years of seasonally adjusted values
  re-estimated. The values here are the October 2026 vintage, not what was public before each election. Typical
  revisions are a few tenths of a point. A real-time version would need ALFRED's archived vintages.
- **National vs state sources:** `UNRATE` comes from the household survey, and the state rates from LAUS models. They
  aren't one system, so the state rates don't average exactly to `UNRATE`.
