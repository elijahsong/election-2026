# House Stage 1: production configurations

`house/stage1.py` holds the two fixed Stage 1 specifications. The specification is fixed, but the coefficients are refit on each training window:

| name | prediction |
|---|---|
| `house_model_a` | Model A |
| `house_ensemble_ab_85_15` | 0.85 × Model A + 0.15 × Model B |

- **Model A (polling):** `partisan_lean + gen_margin_raw + inc_signed + partisan_lean:inc_signed + war_lag`, with the `inc_signed` coefficient fixed at **+2.5**. `war_lag` is the nominees' lagged Split Ticket WAR (`modeling.csv`'s `war_f` tranche), with the neutral 0 where neither nominee has an earlier WAR race.
- **Model B (fundamentals):** `partisan_lean + fundraising_advantage + pres_party_signed:is_midterm + pres_party_signed:cpi_region_log_yoy`.
- **Fitting:** both models use equal-year WLS. Each valid training race gets weight ∝ 1 / (valid training races in its year).

The module docstring records the methodological choices: the sign conventions, which rows are used, the training windows, and why 2024 is used. The specification comes from `notebooks/house/stage1/01_reg_ensemble_compare.ipynb` and `01_reg_final_2024_holdout.ipynb`. The single-variable ablations (fixed incumbency, WLS weightings, no generic ballot, a lean-free Model B) are in `01_reg_ablations.ipynb`.

On 2026-11-02, Model A dropped `gb_momentum_30d_raw` for parsimony. Momentum is national and nearly collinear with `gen_margin_raw` (r = 0.92 across elections). Its coefficient swung by 2–3 points when a single training election was left out, and it bought only about 0.1 MAE, mostly in 2018. The evidence is in `01_reg_model_a_no_momentum.ipynb` and `01_reg_generic_ballot_smoothing.ipynb`. Those notebooks, and the two above, were run with momentum still in Model A, so their numbers are for the earlier spec.

On 2026-11-03, Model A added `war_lag`. Its coefficient is about 0.6 and stable across windows. On the same scored races it cut the ensemble's 2018–2024 MAE from 5.20 to 4.80, and its ex-ante competitive MAE from 4.70 to 4.28, with every year improving. It also cut the Monte Carlo's competitive Brier from 0.100 to 0.086. The evidence is in `01_reg_war.ipynb`, which keeps the earlier Model A as its baseline. The WAR history is a single vintage that Split Ticket recomputed in 2025, so that backtest isn't cleanly out of sample (`data-processing/README.md`).

Rolling out-of-sample results for the ensemble, pooled over 2018–2024, on the current data:

| subset | n | MAE | winners called | before `war_lag`: MAE | winners called |
|---|---|---|---|---|---|
| ex-ante competitive | 353 | 4.28 | 88.1% | 4.70 | 86.1% |
| ex-post competitive | 303 | 5.12 | 82.5% | 5.51 | 80.5% |
| all districts | 1,597 | 4.80 | 96.6% | 5.20 | 96.1% |

## Rolling out-of-sample predictions (for Stage 2)

Each eval year is fit on 2006 through the election before it (2018: 2006–2016, …, 2024: 2006–2022). An eval year's own results never enter its fit.

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path("../../src").resolve()))
from house import stage1

CONFIG = stage1.HOUSE_ENSEMBLE_AB_85_15  # or stage1.HOUSE_MODEL_A
rolling = stage1.rolling_predictions(CONFIG)  # {2018: Stage1Result, 2020: ..., 2022: ..., 2024: ...}
oos = pd.concat([r.predictions for r in rolling.values()], ignore_index=True)
rolling[2024].metrics          # the blend scored like a single model
rolling[2024].coefficients     # coef per term and component; inc_signed is the fixed 2.5
rolling[2024].components["A_polling"].model.summary()
```

`predictions` has one row per predicted district, with these columns:

- `year`, `state`, `district`
- `train_start`, `train_end`
- `prediction`, `actual`, and `residual` (actual − prediction)
- `prediction_<component>` for each component model

Districts with imputed (uncontested) targets aren't predicted in years that have targets.

## The 2026 forecast

```python
forecast = stage1.forecast(CONFIG)  # fit on 2006-2024, predict 2026; actual and residual are NA
forecast.predictions
```

2026's `gen_margin_raw` is FiftyPlusOne's election-eve margin, transcribed by hand (see `data-processing/house/README.md`). The ensemble predicts 432 of 435 districts: CA-11, CA-14 and CA-40 are same-party top-two races (two Democrats; Calvert vs Kim), so they have no 2026 `fundraising_advantage`, which Model B needs. Louisiana's multi-candidate races (LA-3 to LA-6) use each party's top FEC fundraiser (`data-processing/fec_finance.py`).

## Tests

`tests/test_stage1_configs.py` covers:

- the predictor definitions
- signed incumbency
- the fixed +2.5 coefficient
- equal-year weighting, checked against a hand-computed WLS
- chronological windows, including a leakage check
- exact 85/15 blending

# House Stage 2: simulation

`house/simulation/` holds the Monte Carlo infrastructure shared between experiments. It is used by `notebooks/house/stage2/02_monte_carlo.ipynb`.

`hierarchical.py` defines the three-level error model, M_sim = M_hat + N + S_state + D_district. Each level is a Student-t draw with `nu` degrees of freedom (5 in the notebook) times a level-specific scale.

| function | does |
|---|---|
| `estimate_error_sds(oos)` | pooled standard deviation of each level, from Stage 1's out-of-sample residuals |
| `t_scale(sd, nu)` | converts a standard deviation to a Student-t scale |
| `simulate_margins(point, scales, nu, n_sims, rng)` | the `(n_sims, len(point))` simulated margins, and the raw draws behind them |

The national variance is a pseudo-sample-size-weighted regularized moment estimator, (2 · 3² + Σ yearly mean residual²) / (K + 2): a fixed 3-point prior SD counted as 2 pseudo-elections, never tuned on backtests. The state and district SDs come from a random-effects ANOVA. The module docstring documents both.

**Optional error factors** (`factors.py`). These give four hierarchies, compared in `notebooks/house/stage2/03_hierarchy_comparison.ipynb`, which runs all four. H0 above is the reference, and its results are unchanged.

| hierarchy | error | `backtest(..., regions=, loadings=)` |
|---|---|---|
| H0 | N + S + D | `None, None` |
| H1 | H0 + γ·Z_d: one national coefficient per simulation × each district's college loading | `None, LOADINGS` |
| H2 | H0 + R_g: one draw per political region per simulation, one common scale | `REGIONS, None` |
| H3 | H0 + both | `REGIONS, LOADINGS` |

- **Inputs:** `load_regions()` reads `data/reference/regional_stage2.csv`, the 10 custom regions, and raises unless all 50 states appear exactly once. `demographic_loadings(predictors, column="college", reference_sd=10)` gives Z = (college − the year's 435-district mean) / 10 points. It is centered, so it never moves the national average. The scale only sets γ's units.
- **Estimation:** `estimate_factor_sds(oos, regions=, loadings=, ...)`, on rolling out-of-sample residuals only.
  - **Demographic:** the within-state-year slope of the residual on Z, net of its sampling variance.
  - **Regional:** the same-region minus different-region state-pair covariance, with an exact correction for the bias that centering on the year mean causes. Without it the estimate ran 5–17% low on synthetic data.
  - **Shrinkage:** both use a zero prior of 3 pseudo-elections (`REGIONAL_PRIOR_ELECTIONS`, `DEMOGRAPHIC_PRIOR_ELECTIONS`).
  - **Per-year observations:** `factor_moments`.
- **Variance budget:** `allocate_budget`. The factors take their variance out of H0's state and district levels (σ_S² = σ_S0² − σ_R² − σ_γ²B, σ_D² = σ_D0² − σ_γ²W), so an average district's error variance is exactly H0's. A cap shrinks the factors themselves and is reported. `checks.check_variance_budget` asserts the identity.
- **Approximation:** σ_N is H0's, but the yearly mean residual already contains part of the regional shock. `Calibration.diagnostics()` reports `sim sd of mean error`; it is 0.01–0.04 points above H0's.
- **Joint-distribution scoring:** `group_seats(calibration, column, groups=None)` scores seats within groups (regions, `college_terciles`, `competitive_ex_ante`) against their 80% and 95% intervals.
- **Result (2026-11-02):** the fitted factors are small (regional SD 0.8–1.4, demographic 0–0.34 points per 10 pp). Brier, coverage, seat intervals and group-seat coverage are essentially identical across H0–H3, and 2026's P(218+) is 94.2–94.7%. These three elections give no evidence that a factor improves on H0.

`races.py` handles races around that engine. Each race is classified by its general-election lineup, then given exactly one party outcome per draw (party codes `DEM` 1, `GOP` −1, `OTHER` 0; OTHER, a third party or independent, is never folded into D or R):

| race_type | treatment | per-draw outcome |
|---|---|---|
| `dem_uncontested`, `gop_uncontested` | fixed | the party, every draw |
| `dem_vs_3p`, `gop_vs_3p` | third_party | the party, except OTHER with `THIRD_PARTY_UPSET_PROB` = 0.5% (Bernoulli) |
| `dem_vs_dem`, `gop_vs_gop` | same_party | the party, every draw; resolvers may act per draw |
| `dr`, `dr_multi`, `dr_writein`, `dr_jungle`, `dr_rcv` | simulated | D if the simulated Stage 1 margin is > 0, else the race's `loser_party` (R, or OTHER when the GOP proxy is an independent on the ballot: Kiley, CA-6 2026) |

Only simulated races carry margins; no margin is made up for the others.

| function | does |
|---|---|
| `historical_races(targets, targets_raw)` | one row per past district-year: the pre-election lineup's race_type and treatment, the observed winner (blank for NC-9 2018, never certified), and the observed margin where one is valid. `targets.csv`'s results-based reclassifications (a major party finishing third, a voided election) are undone; write-in-only opponents stay uncontested |
| `lineup_targets(targets)` | blanks those restored races' targets so Stage 1 predicts them as unscored rows; every other prediction and residual is unchanged |
| `races_2026(candidates)` | the same table for 2026, from `data/processed/house/candidates_2026.csv` |
| `attach_predictions(races, primary, primary_source=..., fallback=None)` | Stage 1 margins for the simulated races, from the configuration or, where it lacks inputs, Model A (`prediction_source`) |
| `simulate_races(races, scales, nu, n_sims, rng, resolvers=())` | `RaceDraws`: `winners` (n_sims × races), `margins`, `seats(party, columns=None, alignment=None)` (`alignment` is an optional caucus scenario, never applied by default), `win_probability(party)`. The simulated races' draws are exactly `simulate_margins`' |
| `forecast_table(draws, candidates, resolvers=())` | one row per race: p_dem, p_gop, p_other, predicted party, prediction source, and the predicted candidate where one is resolvable |

`resolvers.py` has the election-rule hooks: `Resolver` (party outcomes unchanged, no candidate picked), `JungleSameParty` (D vs D, R vs R), and `RankedChoice` (`dr_rcv`), which keep the default behavior for now and are where candidate-level models would go. `resolve_candidate` names a winner only when the winning ballot party has exactly one candidate in the race, or a resolver picks one; otherwise it reports `several_<party>_candidates`.

`top_two.py` has the candidate-level model for same-party races. It has no training data; the weights and uncertainty are set by hand, and the module docstring has the full method.
- **Inputs:** `data/processed/house/toptwosameparty.csv`, read with `load_top_two()`.
- **`predict_top_two(finalists, weights_with_poll=..., weights_no_poll=..., points_per_log=..., base_sd=..., split_sd=..., nu=...)`:** blends three predictions of the leader's November two-way margin.
  - The three are the June primary, fundraising (individual contributions, scaled by Model B's `fundraising_advantage` coefficient) and a poll where one exists.
  - The default weights are 60/30/10 with a poll and 60/40 without.
  - The SD widens with the share of June voters who backed neither finalist.
  - It returns p_leader, p_runner_up and an 80% margin interval per race.
- **`TopTwoSameParty(predictions)`:** a `JungleSameParty` resolver that names the finalist with p ≥ 0.5. Party outcomes are unchanged.

`calibration.py` holds the chronological backtest:

| function | does |
|---|---|
| `out_of_sample(rolling, scored_only=True)` | stacks `stage1.rolling_predictions` into one row per scored district-year (or every prediction), labeled with its config |
| `backtest(oos, test_years, *, races, predictions, fallback_predictions=None, ..., train_years=None)` | for each test year, fits the error SDs on earlier years' residuals only (every earlier year in `oos`, or exactly `train_years`), simulates every race of the year under the race rules, and scores the draws; returns a `Calibration` |
| `Calibration.table()`, `.diagnostics()`, `.pooled()` | the compact per-year table (`Brier (all races)`, `contested Brier`, `competitive Brier`, 80/95% coverage, seat median and 80/95% intervals, actual seats and their percentile, unresolved races); races by treatment, interval widths and fitted SDs with `mean_signed_error`; race metrics pooled across years |
| `Calibration.wrong_calls()`, `.competitive_stats()`, `.seats()` | races whose call (the most likely ballot party, `call_party`) missed the winner; the confusion matrix with accuracy, Brier, and coverage for all, contested, `competitive_ex_ante`, and `competitive_ex_post` (hindsight) races; median simulated vs actual seats, and the seats the calls add up to |
| `seat_summary(draws, actual)` | the seat median, 80/95% intervals (inverted-CDF quantiles, so always drawn counts), and mid-rank percentile of the actual count: the numbers behind both the table and the seat plot |

**Two backtests, never pooled.** The main one scores 2020–2024, each year's Stage 2 fit on Stage 1 residuals from 2018 up to the year before it, the same residual period the 2026 fit uses. The **separate 2018 test** scores 2018 with Stage 2 fit on 2012–2016 residuals (`stage1.rolling_predictions(config, eval_years=[2012, 2014, 2016, 2018])`, then `backtest(..., test_years=[2018], train_years=[2012, 2014, 2016])`). Its district residuals have an SD of about 10–11 points, against 6–7 from 2018 on, so it tests Stage 2 when its training elections come from a noisier period, and it never feeds the main backtest or 2026. 2010 is left out: its Stage 1 window would be 2006–2008 only.

**What is scored.** Brier, reliability, calls and seats cover every race with a resolved winner: all 435, except NC-9 2018 (434 of 435, named in `unresolved`). Simulated (`panel_dem_seats`) and observed (`panel_actual_dem_seats`) seats are counted over those same races; `house_seats` has every party's seats over all 435. Margin coverage covers only the simulated races with a valid observed two-party margin. Only the lineup and Stage 1's margin decide how a race is simulated; winners and actual margins only score the draws.

`visualization.py` has `plot_seat_distribution(seat_draws, year=..., actual_seats=..., panel_size=..., house_size=435)`: the histogram of simulated Democratic seats, one bar per seat count, shaded by the central 80% and 95% intervals, with the median, the actual count and its percentile when known (`actual_seats=None` for 2026), and a `Matched panel: N of 435 districts` label. It returns `(fig, ax)` and never saves. Three backtest views take the `{name: Calibration}` dict: `plot_backtest_seat_distributions` (that histogram for every config × test year), `plot_seat_errors` (each election's seat median and 80/95% intervals minus the actual count), and `plot_reliability` (binned district win probabilities against how often Democrats won, from `Calibration.reliability()`).

**What lives where.** The package does the work every Stage 2 notebook repeats. The notebook keeps what is a decision, and the explicit calls that show it.

| module | does |
|---|---|
| `inputs.py` | preprocessing: `load_targets_raw()`, `load_candidates_2026()`, `stage1_inputs(config, ...)` (a rolling Stage 1 run on `lineup_targets`, as `Stage1Inputs`: `.oos` fits Stage 2, `.predictions` is simulated), and `stage1_forecast(config, ...)` |
| `checks.py` | the invariants each run must hold, each raising `AssertionError` or returning a one-line summary: `check_engine`, `check_lineup_run`, `check_backtest` (no future information, every race once, no invented margins), `check_engine_draws`, `check_same_draws`, `check_forecast` |
| `report.py` | display only: `stack(calibrations, method)` (one `Calibration` table for every config, or every backtest × config), `styled(table)` (the shared `FORMATS`), and `seat_table(draws, scenarios)` |

The notebook keeps:
- **Settings:** configurations, fallback, seed, simulation count, `NU`, the national prior, test and training years, resolvers, caucus scenarios.
- **Modeling calls:** the error-model fit (`estimate_error_sds`, `t_scale`), and each `backtest`, `simulate_races` and `forecast_table` call, written out with keyword arguments.
- **Experiment-specific checks:** for example, that the 2018 test is kept apart from the main backtest.
- **Results and their interpretation.**

Tests are in `tests/test_simulation.py`, `tests/test_races.py` and `tests/test_simulation_factors.py` (factor sharing and alignment, H0's draws, the regional estimator's bias on unequal synthetic states and regions, shrinkage, the variance budget and caps, leakage).
