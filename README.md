# Election Prediction Project, 2026

A forecast of the 2026 midterms: all 435 House races and the 35 Senate races. Each chamber has two stages:
- **Stage 1:** a fundamentals regression.
- **Stage 2:** a Monte Carlo simulation of win probabilities and seats. In the Senate, Stage 1 is first blended with
  polling and expert ratings.

**Submission:** `data/processed/full congress track template.csv` (race_id, chamber, predicted_winner_party,
candidate_name, win_probability).

## Data

The raw sources are in `data/raw/` (not in git). Each source folder has a `SOURCE.md` with its citation and license.

| Input | Source |
|---|---|
| Results (targets) | MEDSL House and Senate returns, FEC *Federal Elections*, state election offices |
| Partisan lean | MEDSL presidential returns (state and district PVI from the two prior presidential elections) |
| Generic ballot | FiveThirtyEight averages; 2026 from FiftyPlusOne |
| Incumbency, 2026 nominees | CAGE; Wikipedia (pinned revisions) |
| Fundraising | FEC candidate summaries |
| Candidate history (House) | Split Ticket's Wins Above Replacement database (`data-processing/split_ticket_war.py`) |
| Demographics, economy | Census ACS; BLS CPI and FRED gas prices; BLS state unemployment |
| Senate polls | RealClearPolling (J. Whitcomb's scrape, 2006–2024); FiftyPlusOne averages for 2026 (hand-entered) |
| Senate expert ratings | Sabato, Cook, Inside Elections: JHK Forecasts archive (2016–2024); 2026 hand-entered |

Every transformation is a script in `data-processing/`. The details, corrections and caveats are in
`data-processing/README.md`, `data-processing/house/README.md` and `data-processing/senate/README.md`.

## Method

Margins are always the Democratic two-party margin, in points. Models are fit on expanding windows, so each election
is predicted only from earlier ones, and no setting is tuned on backtest outcomes.

**House**
- **Stage 1** (`src/house/stage1.py`): 0.85 × Model A + 0.15 × Model B, both fit by equal-year WLS.
  - Model A: lean, generic ballot, incumbency fixed at +2.5, a lean × incumbency term, and the nominees' lagged
    Split Ticket WAR (`war_lag`: 0.75 / 0.25 of their two most recent earlier WAR races; 0 where none).
    WAR was added 2026-11-03, after `notebooks/house/stage1/01_reg_war.ipynb`.
  - Model B: lean, fundraising, midterm penalty and inflation.
- **Stage 2** (`src/house/simulation/`, H0): national, state and district Student-t errors, estimated from Stage 1's
  out-of-sample residuals.

**Senate** (races keyed by year, state and class)
- **Stage 1** (`src/senate/stage1.py`, S1): lean, generic ballot, and incumbency fixed at +2.5, fit by equal-year WLS.
  S0 is the same without incumbency.
- **Blend** (`src/senate/stage2.py`): weights for fundamentals / polls / experts are 0.37 / 0.54 / 0.09 when polls
  exist, and 0.65 / 0 / 0.35 when they don't.
  - Experts: Sabato 40%, Cook 30%, Inside Elections 30%. Ratings become margins: Solid ±31, Likely ±15, Lean ±10,
    Tilt ±5, Tossup 0.
- **Monte Carlo** (`src/senate/simulation/`, S-M0): a national and a race Student-t error (ν = 5), estimated from the
  blend's out-of-sample residuals. The national SD is shrunk toward a 3-point prior worth 2 pseudo-elections.
  - Independent-vs-Republican races (NE, ID, SD) are centred on their ratings. Independent winners are counted
    separately from Democrats.
  - S-M1 (separate race scales for polled and unpolled races) and S-M2 (regional error) are experiments.

**Manual adjustments:** candidate quality and special circumstances enter as margin points per race, in
`data/reference/margin_adjustments_2026.csv` (`race_id`, `category`, `party`, `margin_adjustment` D-positive,
`reason`). House adjustments use a standard scale, `data/reference/adjustment_scale.csv` (starting values: Basinger
2013's scandal penalties, e.g. corruption −16 margin points to the scandal-hit candidate's side); manual rows are
flagged. They are added to
a race's margin before the Monte Carlo, so probabilities, seats and generic-ballot scenarios all carry them; the
unadjusted margins and probabilities are kept beside them (`src/congress/adjustments.py`). The earlier probability
overrides (`data/reference/track_overrides_2026.csv`) are retired: the file is kept as a record and is no longer
applied. Since 2026-11-03 the file holds only the Senate adjustment (AK-SEN): the manual House rows were removed
(the same day lagged WAR, which carries candidates' past over- and underperformance, entered House Stage 1).

## How to run

```bash
uv sync                                                # Python environment
python data-processing/senate/download_sources.py      # fetch and verify raw sources (sha256)

# Data: shared, then each chamber (full order and options in the data-processing READMEs)
python data-processing/incumbency.py
python data-processing/nominees_2026.py
python data-processing/fec_finance.py
python data-processing/cpi_gas.py
python data-processing/unemployment.py
python data-processing/house/house_targets.py          # then the two house_impute_targets_*.py steps
python data-processing/split_ticket_war.py             # lagged WAR (after fec_finance.py), before house_features.py
python data-processing/house/house_features.py         # then candidates_2026.py and top_two_same_party.py
python data-processing/senate/senate_targets.py        # after presidential.py and senate_pvi.py; see senate/README.md
python data-processing/senate/senate_features.py
python data-processing/senate/senate_polls.py
python data-processing/senate/senate_ratings.py

# Forecast and submission: one notebook, the production entry point (both chambers, adjustments, ratings,
# generic-ballot shifts; writes the submission through data-processing/congress_track.py)
#   notebooks/full-forecast/full_forecast_2026.ipynb
#       -> data/processed/full congress track template.csv (and full_congress_track_2026.csv)
#       -> data/processed/forecast/races_2026.csv, chambers_2026.csv, gb_scenarios_2026.csv (GB off by ±2, ±4)
#       -> data/processed/kalshi/portfolio_risk_2026.csv, portfolio_stress_bets_2026.csv (the Kelly portfolio
#          priced in every simulation and generic-ballot scenario; cached Kalshi prices)
#       -> predictions/final_preds.csv (section 10: per race, the candidate, predicted margin, 80% margin interval, win probability,
#          win probability with the generic ballot 4 points less / more Democratic, and Polymarket's current price)
# Calibration and backtests, which also write each chamber's model-only track rows:
#   notebooks/house/stage2/02_monte_carlo.ipynb    -> data/processed/house/congress_track_house_2026.csv
#   notebooks/senate/stage2/03_monte_carlo.ipynb   -> data/processed/senate/congress_track_senate_2026.csv
# Generic-ballot scenarios and their Kalshi value: notebooks/scenarios/generic_ballot_shift.ipynb
# Backtest of the full forecast on 2020, 2022, 2024 (ratings, misses, political errors, top two, Kalshi 2024):
#   python data-processing/house/top_two_same_party_history.py   # 2020-2024 same-party top-two finalists
#   uv run python src/kalshi/history.py                          # 2024 Kalshi eve prices (cached after one fetch)
#   notebooks/full-forecast/full_forecast_backtest.ipynb          -> data/processed/forecast/backtest_*.csv

uv run python src/kalshi/run.py                        # Kalshi odds, toss-ups, Kelly bets and portfolio risk (GB ±2, ±4);
                                                       # cached prices (--refresh: live; --no-risk: skip the risk step)
uv run pytest                                          # tests
```

**Kalshi** (`src/kalshi/`, public market-data API only, no orders): `odds.py` matches every submission row to its
Kalshi winner market and price. `optimizer.py` covers the toss-ups, races where the submission or Kalshi puts the pick
at 40–60%:
- It blends the two 50/50 into proposed picks (`proposed_overrides_2026.csv`, in the retired overrides' format).
  They are informational only.
- It sizes quarter-Kelly bets on the submission's probabilities (`kelly_bets_2026.csv`).

Prices are cached: each pull is saved in `data/raw/kalshi/events_<stamp>.json`, and `run.py` reads the latest one without calling the API.
Pass a snapshot path to use another, or `--refresh` to pull and cache live prices.

**Conditional forecasts on election night** (`src/congress/conditional.py`): the submitted forecast's simulations,
conditioned on results as they come in. House races share a national error, and races in one state a state
error, so one result moves the others.
```bash
uv run python src/congress/conditional.py -i        # interactive: call KY-06 D | prob NC-11 D 0.9 | margin VA-02 D+3.5 2
uv run python src/congress/conditional.py --call KY-06=D --prob NC-11=D:0.9 --margin VA-02=D+3.5:2
```
- **Inputs:** a call keeps the simulations where that party wins; a probability reweights the simulations so the race
  hits it; an observed margin with its SD weights each simulation by its likelihood. Several inputs are combined by
  raking.
- **Output:** each chamber's control probability, seats and effective simulations, then the races that moved most.
  Warns below 1,000 effective simulations.
- **Limits:** the House and Senate national errors are independent, so an input in one chamber never moves the other.
  Same-party top-two picks never move.
- **Cache:** the baseline simulations are cached in `data/interim/conditional/`, keyed to the submission and the
  adjustments file, so a rerun of the full forecast rebuilds the cache on next use. `--rebuild` forces it, `--sims`
  draws more.

**Polymarket** (`src/polymarket/odds.py`, the public Gamma API, no key): the race-winner events (party markets, and
person markets for California's same-party races), priced at the bid-ask midpoint (else the last trade) and
normalized within each race. The full forecast pulls **current** prices on every run (`POLYMARKET_REFRESH = True`;
each pull is cached as `data/raw/polymarket/events_<stamp>.json`) and writes Polymarket's probability for each
submitted candidate as `final_preds.csv`'s `polymarket_odd`. The model's own columns use only pre-election
information; on election night the Polymarket prices already reflect returns.
`uv run python src/polymarket/odds.py` pulls and caches a snapshot on its own.

The Stage 1 experiments are in `notebooks/house/stage1/` and `notebooks/senate/stage1/`. The production interfaces
are documented in `src/house/README.md` and `src/senate/README.md`. The 2026 production pipelines are `src/house/forecast.py` and
`src/senate/forecast.py` (Stage 1 through track rows, with a generic-ballot `shift`, margin `adjustments` and, for the
Senate, custom blend `weights`); `src/congress/report.py` has the ratings (Toss-up < 60%, Lean < 75%, Likely < 95%,
else Safe) and the cross-chamber tables.
