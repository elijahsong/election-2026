# Interim data layout

Interim data that is House-specific (district-level) lives in `data/interim/house/`: `acs.csv`, `acs_2026.csv`,
`acs_supplement.csv`, `house_incumbency.csv`, `house_partisan_lean.csv`, `partisan_lean*.csv` and the cleaned Cook PVI
files. National and Senate data stays in `data/interim/`: `generic_ballot_all.*`, `presidential.csv` and
`senate_incumbency.csv`.

House-specific scripts and notebooks (generic ballot, ACS demographics, partisan lean, modeling features and targets)
live in `house/`; see `house/README.md`. `incumbency.py` stays here because it builds both House and Senate data.

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
- **Sanders** (VT 2012, 2018 and 2024) and **King** (ME 2024) caucus with Democrats, so they're +1.
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

## Nominees

- **Sources:**
  - House 2006–2024: MEDSL via `house_targets` (November / first round).
  - Senate 2006–2024: CAGE via `incumbency.senate_candidates`, which gives class and special.
  - 2026: `nominees_2026.py`.
- **A party's slot is its leading general-election candidate by votes:** this covers jungle, top-two, RCV and party
  write-ins. Primary candidates and their money are never used.
- **2026, which has no votes:** the party's only candidate fills the slot. With several, the one FEC flags `INCUMBENT`
  does, provided exactly one is. Otherwise the slot is `nominee_unresolved`: CA-11, CA-14 and CA-40 (Calvert vs Kim)
  and six Louisiana slots.
- **`INDEPENDENTS_AS_DEM`:** independents who were the Democratic side take the `dem_` slot (`independent_as_dem`;
  the report lists any D nominee displaced):
  - Sanders (VT 2006, 2012, 2018, 2024)
  - King (ME 2012, 2018, 2024)
  - Orman (KS 2014)
  - McMullin (UT 2022)
  - Osborn (NE 2024, 2026)
  - Buchanan (MT-2 2022)
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
| `override` | `MATCH_OVERRIDES`: 11 reviewed cases, each commented (e.g. Gus vs Michael Bilirakis, Ron Estes vs Ron M. Estes) |
| `independent_as_dem` | `INDEPENDENTS_AS_DEM` |
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
  - FEC's own 2006–2012 general-election flags against the slots.
- **Gaps in FEC's files:** 11 House winners have no record in either bulk file for their cycle. `API_CORRECTIONS`
  maps each slot to its candidate ID.
  - `fec_api_totals.py` fetches the FEC API's two-year totals for each one
    (`/v1/candidate/{id}/totals/?cycle=YEAR&election_full=false`). The fields agree with the bulk file where a
    candidate is in both (checked on Donalds 2022).
  - Each response with data is cached unmodified in `data/raw/fec_candidate_summaries/api/`, and `SOURCE.json` there
    records its URL (key redacted), time and sha256. Responses with no data aren't kept.
  - Filled (`fec_api`): Trone 2018; Issa, Jacobs and Donalds 2020; Gillen 2024.
  - Still `no_fec_record` and zero-filled, because the API has no cycle totals for them either: Ivey 2022 and 2024,
    Self 2022 and 2024, Stutzman 2024, McGuire 2024.
- **2006 Republicans:** about 11% of 2006 R nominees (44, mostly in safe D seats) have no `webl06` record.
