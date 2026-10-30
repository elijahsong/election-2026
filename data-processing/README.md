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

