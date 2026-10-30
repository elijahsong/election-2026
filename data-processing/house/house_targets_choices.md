# `targets.csv`: choices to review

`data-processing/house/house_targets.py` had to make some calls the source data doesn't settle. Each one is listed below with
what the script does now, how many rows it touches, and the alternatives. Nothing here is imputed. Every option uses
only observed votes.

Row counts cover all of 1976–2024 (`targets_raw.csv`; `targets.csv` keeps 2006 onward). Counts for **2006–2024**
(the years `config.py` trains and evaluates on) are given separately, because that's where a choice changes the model. The **Decision** line under each item records what you choose.

| # | Choice | 2006–2024 target rows affected | Decision |
|---|---|---|---|
| 1 | Which contest to use when a district has two | 1 (TX-23 2006) | November round, always |
| 2 | What `total_votes` counts | 0 (only `margin_raw` and the `_pct` columns change) | A: exclude non-votes |
| 3 | Targets for jungle, RCV and write-in-only races | 50 | jungle and RCV set as observed first-round party totals; `dr_multi` normal; `dr_writein` → uncontested except TX-22 2006 |
| 4 | TX-22 2006 (Republican write-in with no party in the source) | 1 | add `PARTY_OVERRIDES` |
| 5 | Party-label mapping (DFL, Dem-NPL, …) | ~55 (ND and MN rows) | A: keep the mapping |
| 6 | Blank `winner_party` | 0 | B in `targets.csv` only; LA-5/LA-6 2014 from official results |
| 7 | DC excluded | 0 | A: exclude |
| 8 | Independents who won or got most of the vote (found during this review) | 0 | A: keep; 2006–2024 only |
| 9 | Which races are imputed as uncontested | 470 (incl. the 8 races of #10) | `dem_vs_3p` through `no_major_party`, by state mean residual; national SHAVE fallback for 9 |
| 10 | D-R races treated as uncontested (write-in-only major party, major party third, voided election) | 8 | impute; actual results kept in `targets_supp.csv` |

---

## 1. Which contest to use when a district has two

**Originally:** six districts have two contests in the source. The script uses the regular general election, and that
election's runoff when the source includes one. This is the round that elected the member. The row is flagged
`other_contests_in_source(... used)`.

| District | Originally used (decisive round) | Other contest in source | `margin_2p_norm` originally → with the other contest |
|---|---|---|---|
| TX-8 1996 | Dec runoff: Brady (R) vs Fontenot (R) | Nov jungle: 2 D, 2 R | blank (R vs R) → **−60.8** |
| TX-9 1996 | Dec runoff: Lampson (D) vs Stockman (R) | Nov jungle | 5.7 → 7.1 |
| TX-25 1996 | Dec runoff: Bentsen (D) vs McKenna (R) | Nov jungle: 2 D, 8 R | 14.6 → 2.1 |
| LA-5 2002 | Dec runoff: Alexander (D) vs Fletcher (R) | Nov jungle: 2 D, 4 R | 0.6 → **−37.3** |
| TX-23 2006 | Dec runoff: Rodriguez (D) vs Bonilla (R)\* | Nov special jungle: 6 D, 1 R | 8.6 → 0.1 |
| TX-22 2006 | Regular election for the full term | Same-day special for the rest of DeLay's term | blank (see #4) → −100 (R only) |

\* MEDSL marks the TX-23 runoff as the regular election (`special = FALSE`) and the November round as special.

**The consistency problem:** for every other Louisiana jungle race (1998–2006, 2012–2024), the source has only the
November round, never the December runoff. So right now these six rows use the runoff, while the other ~75 LA rows from
jungle years use the November round.

**Options**
- **A (current).** Use the decisive round: the actual head-to-head contest, and it matches `winner_party`.
- **B.** Use the November round everywhere (for TX-22 2006, still the regular election). This matches how every other
  jungle race in the file is handled, and the November electorate matches the November predictors. The cost is that
  TX-8 1996 and LA-5 2002 get combined-party margins that differ a lot from their runoffs. `winner_party` would still
  come from the runoff.

**Decision (2026-10-29): B, the November round.** Every row now uses the November ballot. That's consistent with the
other jungle races and with the November predictors. Where the source has the December runoff, `winner_party` still
comes from it (flag `runoff_in_source`). TX-22 2006 keeps the regular election (flag `special_in_source`). MEDSL labels
TX-23 2006's runoff as the regular election, so `RUNOFF_FIXES` relabels it as the runoff. Resulting targets: TX-8 1996
−60.8, TX-9 1996 7.1, TX-25 1996 2.1, LA-5 2002 −37.3, TX-23 2006 0.1. `is_special_election` is now true for the four
November special rounds (TX 1996 8, 9, 25 and TX 2006 23).

**Confirmed (2026-10-29):** always use the November round where a district also has a runoff.

---

## 2. What `total_votes` counts

**Now:** `total_votes` = D + R + third-party votes. Third-party includes write-ins, scattering and "other." Blank, void,
over-vote, under-vote and exhausted-ballot rows aren't votes for anyone, so they're left out. Their sum is kept in
`nonvote_votes`, and the source's own `totalvotes` in `source_totalvotes`. Combined labels like
`BLANK VOTE/SCATTERING` (NY, MA, WI, IA) can't be split, so the whole row counts as non-votes. That drops a few
scattered write-ins.

**Effect:** only `margin_raw`, `dem_pct`, `gop_pct` and `3p_pct` change. **`margin_2p_norm` and the targets don't.**
- 632 rows have non-vote rows (379 in 2006–2024), mostly NY (428), MA (104) and WI (34).
- `source_totalvotes` ≠ `total_votes` on 609 rows (355 in 2006–2024).
- Dividing by the source total instead would move `margin_raw` by more than 0.5 pp on 475 rows (269 in 2006–2024).
  The median shift is about 2 pp, and the largest is 40 pp (NY 2002–2008, where blank ballots were up to 43% of the
  source total).
- The source isn't consistent about it either: NY 2024's `totalvotes` excludes blanks, while some other NY and MA
  years include them.

**Options**
- **A (current).** Divide by the votes cast for candidates. The shares add up to 100 and the denominator means the same
  thing in every state.
- **B.** Divide by `source_totalvotes`. This matches MEDSL, but the meaning changes by state and year as described above.

**Decision (2026-10-29): A.** Leaving out blank, invalid and other non-votes is fine.

---

## 3. Targets for jungle, RCV and write-in-only races

**Originally:** a race gets a target whenever at least one D and one R received reported votes (`race_type` `dr` or
`dr_multi`). Each party's votes are summed across all its candidates. These kinds of race have a target but aren't
ordinary two-candidate contests:

| Kind | 2006–2024 rows with a target | Notes |
|---|---|---|
| **Louisiana jungle rounds** (2006, 2012–2024) | 36 of 49 | Every candidate runs on one November ballot, often several from each party. The target is the combined D vs R vote. LA 2008–2010 used closed primaries and plurality generals, so those years are ordinary |
| **TX court-ordered specials** (TX-15, 21, 25 2006) | 3 | Jungle-style November elections for the full term |
| **RCV: Maine 2018–2024, Alaska 2022–2024** | 10 | The source mixes rounds. AK rows are first-round counts (Peltola vs Palin + Begich). ME-2 2022 and 2024 carry the final-round D and R totals, and ME-2 2018's totals match neither round |
| **Write-in-only major party** (`major_party_writein_only`) | 4 | AZ-3 2016 (+99.6), AZ-8 2022 (−93.0), AZ-9 2022 (−95.5), WI-8 2022 (−97.1). One party had no ballot-listed nominee, only write-ins. Effectively uncontested, but they keep a near-±100 target. (Also PA-3 1998, outside the window) |

`dr_multi` also covers 16 harmless rows where a party write-in got a handful of votes next to the real nominees
(AZ and MD 2020, CO-1 2024 and others). Those margins are fine.

**Options** (they can be mixed, one per kind)
- **A (current).** Keep every target. Rows can be filtered with `race_type`, `is_rcv`, `is_special_election` and
  `quality_flags`.
- **B.** Blank the target for write-in-only races, treating them like uncontested races. This changes 4 rows in the
  window, and these outliers are the most likely to distort an OLS fit.
- **C.** Blank the target for LA jungle rounds and/or TX specials (39 rows in the window). They'd then wait for the
  future imputation step.
- **D.** For RCV, keep the target but add a flag saying which round the source reports. Or blank ME-2 2018, whose
  round can't be identified.

**Decision (2026-10-29): A, keep every target, but label each kind in `race_type`.** No target is blanked yet. Races
with a D and an R are now split into `dr_writein`, `dr_rcv`, `dr_jungle`, `dr` and `dr_multi`, in that order of
precedence. The full `race_type` table is in [`README.md`](README.md#race_type). In 2006–2024 that's 5 `dr_writein`,
10 `dr_rcv` and 40 `dr_jungle` rows, so any of them can be filtered out or imputed later without changing the script.
The `major_party_writein_only` flag is gone, replaced by `dr_writein`. The ME-2 2018 round question is still open
(still `dr_rcv`).

**Revised decision (2026-10-29):**
- **Jungle:** the combined D vs R vote is the correct target, but it belongs to the imputation step, which will use
  `targets_raw.csv` (and `targets_supp.csv`) as its source. In `targets.csv` the 40 `dr_jungle` targets are blank
  and `needs_imputation` is `True`. `targets_raw.csv` keeps the combined targets.
- **RCV:** use the first-round result and treat it like a jungle round. So the 10 `dr_rcv` targets are also blank in
  `targets.csv` and left to imputation, while `targets_raw.csv` keeps them. Three rows aren't first-round counts in
  the source: ME-2 2018 (matches neither round), 2022 and 2024 (a later round, with exhausted ballots). They're
  flagged `rcv_first_round_not_in_source`. Their first-round counts would need data from outside the source.
- **`dr_multi`** (party write-ins next to the nominee): no special treatment. It's a normal race
  (`is_nonstandard_race` is `False`) and keeps its target.
- **`dr_writein`:** unchanged; it keeps its target.

**Decision (2026-10-30), implemented: treat `dr_writein` as uncontested, except TX-22 2006.** The four
write-in-only races become `dem_uncontested` or `gop_uncontested`, by the party with the ballot-listed nominee:
AZ-3 2016 (D, +99.6) is `dem_uncontested`, and AZ-8 2022 (−93.0), AZ-9 2022 (−95.5) and WI-8 2022 (−97.1) are
`gop_uncontested`. Like other uncontested races, they'd have no observed target. TX-22 2006 (Lampson D vs
Sekula-Gibbs, a Republican write-in nominee; +10.7) was a real two-party race, so it keeps its target and stays
`dr_writein`. In `house_targets.py` these are in `TREATED_UNCONTESTED` (see section 10), applied to `targets.csv` only (`targets_raw.csv`
keeps them as observed `dr_writein`). WI-8 2022 had two named third-party candidates, so, by the same rule as a
cross-endorsed assignment, it is `gop_vs_3p` rather than `gop_uncontested`; both are imputed the same way. The rows
are flagged `writein_treated_uncontested`, and their vote columns and candidate counts are kept as observed.
- Every jungle and RCV race from 2006 on has its November round in `targets_supp.csv`, alongside the other contests
  of the two-contest districts.

**Revised again (2026-10-29): jungle and RCV targets are observed, not imputed.** The combined first-round party
totals are an observed result. `house_impute_targets_jungle_rcv.py` sets `adj_margin_2p_norm = margin_2p_norm` with
`is_imputed` `False` and `needs_imputation` `False`, and leaves `dem/gop_votes_imputed` and `imputation_method` blank.
These rows are trained on and scored. They're also reference races for the uncontested imputation (section 9).

---

## 4. TX-22 2006 (Republican write-in with no party in the source)

**Originally:** Shelley Sekula-Gibbs was the Republican write-in candidate after DeLay withdrew. MEDSL lists her with no party
(`writein = TRUE`). The script goes by the source label, so the race is `dem_vs_3p` (Lampson D 76,775 vs Sekula-Gibbs
61,938), and the target is blank.

The source has the same problem in VA-9 2002: Jay Katzen, the Republican nominee, is listed without a party
(Boucher D 100,075 vs Katzen 52,076). That race is outside the window. No other party-less named candidate has more
than 15% of a race's vote.

**Options**
- **A (current).** Follow the source label: the target is blank.
- **B.** Add a small, documented `PARTY_OVERRIDES` table (TX-22 2006 Sekula-Gibbs → R; VA-9 2002 Katzen → R). The
  TX-22 target would become **+10.7**. This is the same pattern as `HOUSE_OVERRIDES` in `incumbency.py`.

**Decision (2026-10-29): B, add the overrides.** `PARTY_OVERRIDES` in `house_targets.py` recodes Sekula-Gibbs
(TX-22 2006) and Katzen (VA-9 2002) as Republican. Each entry must match exactly one party-less source row, and the
row is flagged `party_override`. Both candidates are write-ins in the source, so both rows are `dr_writein`. Targets:
TX-22 2006 +10.7, VA-9 2002 +31.5.

---

## 5. Party-label mapping

**Now:** a candidate counts as D or R if any of their ballot lines carries one of these labels:

| Label | Mapped to | Rows (2006–2024) | Where |
|---|---|---|---|
| DEMOCRATIC-FARMER-LABOR, DEMOCRATIC-FARM-LABOR | D | 113 (49) | MN, 1988 onward |
| DEMOCRATIC-NONPARTISAN LEAGUE, DEMOCRATIC-NPL | D | 5 (5) | ND-AL 2012, 2014, 2018, 2020, 2024 |
| DEMOCRATIC | D | 9 (9) | 2022 write-ins (AZ, WI, NM) |
| WRITE-IN (DEMOCRATIC), WRITE-IN (REPUBLICAN) | D, R | 6 (6) | AZ, MD 2020 write-ins |
| REPUBLICAN, LIBERTARIAN | R | 1 (1) | VT-AL 2024 (Coester, fusion) |
| INDEPENDENT-REPUBLICAN | R | 8 (0) | MN 1992, the state GOP's name at the time |

These labels are left as third party: NATIONAL DEMOCRAT, REGULAR DEMOCRACY, ALOHA DEMOCRATIC, NATIONAL DEMOCRATIC
PARTY OF ALABAMA, FOGLIETTA (DEMOCRAT) (PA-1 1980), and D-R PARTY (an NJ minor party, 2014). All are minor-party or
personal lines, and none is in the window except D-R PARTY, whose candidates got under 2%.

Votes on other lines are added to the candidate's major-party total, for example Working Families or Conservative in
NY. A WFP line with no named candidate stays third party.

**Options**
- **A (current).** Use the mapping above. Without the DFL and Dem-NPL entries, every MN and ND Democrat would count
  as third party, and those races would lose their targets.
- **B.** Change any individual entry.

**Decision (2026-10-29): A.** The fusion and DFL / Dem-NPL mapping is fine.

---

## 6. Blank `winner_party`

**Now:** 26 rows have a blank `winner_party`. This doesn't affect any target.
- **24 cross-endorsed rows** (NY 1976–1990): Addabbo, Biaggi, Rangel, Garcia and others were on both the D and R lines.
  Each was the only major-party candidate in the race, so the race has no D-vs-R margin under any rule. Assigning
  each one to their largest line, as the MEDSL codebook suggests, would give 23 D and 1 R (Horton, NY-34 1978). Those
  rows would become uncontested, so the target stays blank either way.
- **LA-5 and LA-6 2014:** the November leader had less than 50%, the top two were from different parties, and the
  December runoff isn't in the source.

**Options**
- **A (current).** Leave `winner_party` blank where the source doesn't settle it.
- **B.** Assign cross-endorsed candidates to their largest line. This fills 24 `winner_party` values and labels those
  races `dem_uncontested` / `gop_uncontested` instead of `cross_endorsed`. The LA 2014 rows would stay blank unless
  the runoff results were added from another source.

**Decision (2026-10-29): B, in `targets.csv` only.** A cross-endorsed candidate who was the only major-party
candidate counts for their larger line in `targets.csv`. That sets `race_type`, the candidate count and `winner_party`,
and the row is flagged `cross_endorsed_assigned`. `targets_raw.csv` keeps `cross_endorsed` and the blank
`winner_party`. All 24 cases are before 2006, so `targets.csv` doesn't change yet.
**LA-5 and LA-6 2014 (decided 2026-10-29): use Louisiana's official results.** The official November and December
2014 results (downloaded with authorization to `data/raw/la_sos`) show that MEDSL's rows mix the rounds: the top
two candidates' counts are their December runoff totals, while every other candidate's count is from November.
`LA_OFFICIAL` in `house_targets.py` replaces those counts with the official November ones and adds the official
runoffs to `targets_supp.csv`. `winner_party` is now R for both, from the runoffs (Abraham, Graves), and the rows are
flagged `official_results_correction`. The combined November margins in `targets_raw.csv` change from −53.4 to −42.6
(LA-5) and from −42.2 to −30.0 (LA-6). Both stay `dr_jungle`, so `targets.csv` still leaves them to imputation.
**All Louisiana runoff years (checked 2026-10-29).** Every 2006–2024 Louisiana race was then compared with the
official November results. Four more have the same problem: LA-2 2006, LA-3 2016, LA-4 2016 and LA-5 2020 (MEDSL's
LA-5 2020 figures are the runoff totals plus 100,000). They are corrected the same way, and their runoffs, plus LA-3
2012's, are added to `targets_supp.csv`. Every other race matches. The README has the before / after table.

---

## 7. DC excluded

**Now:** DC's 20 source rows (2020 and 2024 only) are dropped and counted in the run report. DC elects a non-voting
delegate, and the feature tables (`acs`, partisan lean, incumbency) leave it out too.

**Options**
- **A (current).** Exclude DC.
- **B.** Keep DC rows. The pattern `[A-Z]{2}` accepts "DC", but the coverage check, which expects 435 rows per year,
  would need an exception. The rows also wouldn't join to `modeling.csv`.

**Decision (2026-10-29): A, exclude DC.**

---

## 8. Independents who won or got most of the vote (found during this review)

**Now:** the D-vs-R margin ignores third-party votes. That's misleading when an independent was the real contender.
Every case is before 2006:
- **Bernie Sanders**, VT-AL 1990–2004. The rows from 1990, 1992, 1996, 2000 and 2004 are `dr` with targets between
  −55 and −86 (a minor Democrat vs the Republican). The rest are `gop_vs_3p`.
- **Jo Ann Emerson**, MO-8 1996, elected as an independent: target +55.9.
- **Thomas Foglietta**, PA-1 1980: target +16.6.
- **Virgil Goode**, VA-5 2000: `dem_vs_3p`, no target.

No row from 2006 to 2024 has a third-party winner.

**Options**
- **A (current).** Keep them. Rows with `winner_party = 3P` are easy to filter.
- **B.** Blank the target wherever `winner_party = 3P`. That's 7 rows, none in the window.

**Decision (2026-10-29): A.** The work focuses on 2006–2024, where no row has a third-party winner, so the current
behavior is fine.

---

## 9. Which races are imputed as uncontested

**Originally:** none of these races had a target. In `targets.csv` they were blank, with `needs_imputation` `True`.

**Planned (2026-10-29):** the future `data-processing/imputation.py` treats every `race_type` from
`dem_vs_3p` through `no_major_party` (in `RACE_TYPES` order) as an uncontested race in need of imputation. That's
`dem_vs_3p`, `gop_vs_3p`, `dem_uncontested`, `gop_uncontested`, `dem_vs_dem`, `gop_vs_gop`, `cross_endorsed` and
`no_major_party`. In `targets.csv` (2006–2024) these are 462 rows:

| `race_type` | Rows |
|---|---|
| `dem_uncontested` | 117 |
| `dem_vs_3p` | 112 |
| `gop_vs_3p` | 88 |
| `gop_uncontested` | 87 |
| `dem_vs_dem` | 44 |
| `gop_vs_gop` | 14 |
| `cross_endorsed` | 0 |
| `no_major_party` | 0 |

With the `dr_writein` change (section 3, implemented 2026-10-30), AZ-3 2016, AZ-8 2022, AZ-9 2022 and WI-8 2022 join
them as uncontested, for 466 rows. TX-22 2006 keeps its target.

**Decision (2026-10-29): state mean residual, in `data-processing/house/house_impute_targets_uncontested.py`.** It imputes
these 462 rows (470 with the eight races of section 10), not a separate `imputation.py`, and `race_type` is unchanged. In each (year, state), turnout is the mean D + R
of the reference races, and the D two-party share is the district's lean plus the state's mean residual vs. lean.
Reference races are `dr`, `dr_multi`, `dr_jungle` and `dr_rcv`, with jungle and RCV rows using their party totals.
Imputed rows and `dr_writein` are never references. Details are in [`README.md`](README.md#uncontested-imputation-house_impute_targets_uncontestedpy).

- `imputation_method` is `state_mean_residual`. It's listed in `MODELED_IMPUTATION_METHODS`, so the regression
  neither trains on these rows nor scores them.
- 461 rows are imputed by `state_mean_residual` (453 before section 10's eight races). The 9 rows whose state has no reference race that year (AR 2008 ×4,
  VT 2008 and 2016, SD 2020 and 2022, and ND 2022) use the fallback below.
- The ME-2 2018, 2022 and 2024 rows (`rcv_first_round_not_in_source`) aren't references, because their source counts
  aren't the first round. No imputed value changes, since Maine has no uncontested races.

**Fallback (2026-10-29): national SHAVE.** A row with no reference race uses Split Ticket's national SHAVE for its year
(`data/reference/shave.csv`) in place of the state adjustment:
`share = clip(L + margin_2p_norm / 200, 0, 1)` and `adj_margin_2p_norm = 100 · (2 · share − 1)`, with
`imputation_method` `national_shave`, also in `MODELED_IMPUTATION_METHODS`.

- It uses SHAVE's `margin_2p_norm`, the two-party margin, so `margin_2p_norm / 200` is the national two-party D share
  minus 0.5. `dem_2p_norm` and `gop_2p_norm` aren't used; they're rounded separately and can differ from the margin
  by about 0.005 points.
- National SHAVE, not state SHAVE, and no state-level PVI: the district lean already places the district.
- Without a turnout reference these rows have a target but no vote counts (`dem/gop_votes_imputed` blank), and keep
  `imputation_no_reference`. A year missing from `shave.csv` would leave the row blank with `imputation_no_shave`.
- All 9 margins favor the party that held the seat, from −41.3 (ND 2022) to +30.3 (VT 2016).
- `targets_supp.csv` isn't used. Since 2026-10-30 it holds the actual result of every imputed race with an opponent
  (section 10), as reference data only.
- Considered and not used: historical district turnout (lines change with redistricting), and a regression model
  (kept simple and auditable).

---

## 10. D-R races treated as uncontested (`TREATED_UNCONTESTED`)

**Found during a 2026-10-30 review** of every observed 2006–2024 target, by category. Some races have a D and an R on
the ballot, but their D vs R margin isn't a contested result. Normalizing to two-party doesn't fix this: it drops the
third-party votes but doesn't give them back to the side they came from. In MT-2 2022 the anti-Republican vote was 42%
(D 20.2 + Buchanan 21.9), but the target was −47.4, as if the Democrat alone were the opposition.

**Decision (2026-10-30): treat these 8 races as uncontested in `targets.csv`.** Each becomes the winner's party +
`_uncontested` (or `_vs_3p` with a named third-party candidate), its target is imputed by `state_mean_residual`, and
it's flagged with the reason. They stop being reference races for their state's imputation, which moves other imputed
CA 2008 and NC 2018 targets by 0.1 to 0.6 points. `targets_raw.csv` keeps them as observed.

| Flag | Race | Observed result | Observed target → imputed |
|---|---|---|---|
| `writein_treated_uncontested` | AZ-3 2016, AZ-8 / AZ-9 / WI-8 2022 | see section 3 | +99.6 → +24.9; −93.0 → −15.6; −95.5 → −27.5; −97.1 → −18.0 |
| `major_party_third_treated_uncontested` | TN-9 2006 | Cohen (D) 59.9, Jake Ford (I) 22.2, R 18.0 | +53.8 → +55.4 |
| | CA-8 2008 | Pelosi (D) 71.9, Cindy Sheehan (I) 16.2, R 9.7 | +76.3 → +84.5 |
| | MT-2 2022 | Rosendale (R) 56.6, Gary Buchanan (I) 21.9, D 20.2 | −47.4 → −23.4 |
| `voided_election_treated_uncontested` | NC-9 2018 | Harris (R) 49.3, McCready (D) 48.9; never certified (absentee-ballot fraud), rerun in 2019 | −0.3 → −7.2 |

**Considered and reverted (2026-10-30): MT-2 2022 with Buchanan counted as the D.** Counting Buchanan's votes with
Ronning's (D 90,675 vs R 121,979) gave an observed `dr_multi` target of −14.7. That made it the base model's largest
2022 miss (residual +25.1, prediction −39.8). It is far more Democratic than an ordinary race there: MT-2 2024, an
open D vs R seat, was −32.0, and MT-1 2022's residual was only +8.4. Buchanan likely drew moderate Republicans too, so
counting all his votes as D overstates the D side. MT-2 2022 is imputed like TN-9 and CA-8 instead.

**Kept as observed** (reviewed, no change):
- A strong independent (15% or more) who still finished behind both major parties: MN-5 2006, OH-2 2008, OH-17 2010
  (Traficant), FL-14 2008, TN-3 2010. These were real D vs R races.
- Low-share major-party nominees: no observed race has a losing major party below 2%. The six below 5% are NYC
  Republicans, real ballot-listed nominees.
- Jungle and RCV races: every party with a candidate got at least 10.8%.
- `dr_multi`: all 12 are real D vs R races with a stray same-party write-in.

**Actual results of imputed races:** `targets.csv` keeps every imputed row's observed vote columns (`dem_votes`,
`gop_votes`, `3p_votes`). `targets_supp.csv` adds the November contest of every imputed race with an opponent (each
`*_vs_3p` race and each `TREATED_UNCONTESTED` race, 208 keys), with its observed `race_type`, and
`targets_candidates.csv` names its candidates. `dem_vs_dem` / `gop_vs_gop` races aren't included yet.
