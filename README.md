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
  - Model A: lean, generic ballot, and incumbency fixed at +2.5, plus a lean × incumbency term.
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

**Manual overrides:** five races are changed by hand: AK-SEN, TX-35, IA-02, ME-02 and WI-03. Each is listed with its
reason in `data/reference/track_overrides_2026.csv`. The model output files are never edited.

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
python data-processing/house/house_features.py         # then candidates_2026.py and top_two_same_party.py
python data-processing/senate/senate_targets.py        # after presidential.py and senate_pvi.py; see senate/README.md
python data-processing/senate/senate_features.py
python data-processing/senate/senate_polls.py
python data-processing/senate/senate_ratings.py

# Forecasts: run the notebooks, which write each chamber's track file
#   notebooks/house/stage2/02_monte_carlo.ipynb    -> data/processed/house/congress_track_house_2026.csv
#   notebooks/senate/stage2/03_monte_carlo.ipynb   -> data/processed/senate/congress_track_senate_2026.csv

python data-processing/congress_track.py               # merge, apply the overrides -> the submission file
uv run pytest                                          # tests
```

The Stage 1 experiments are in `notebooks/house/stage1/` and `notebooks/senate/stage1/`. The production interfaces
are documented in `src/house/README.md` and `src/senate/README.md`.
