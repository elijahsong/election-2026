# House Forecasting — Proposed CSV Schemas

**Conventions:** Democratic margin is positive; `_votes` means vote counts, `_raw` means unnormalized values, `_2p_norm` means two-party normalized, and `_norm` means normalized demographic shares. Use `dem`, `gop`, `3p`, and `und` consistently.

In the future: add map_vintage to the House raw and adjusted tables.

## 1. Election results

| Table | Primary key | Schema |
|---|---|---|
| `{year}_house_raw.csv` | `state, district, map_vintage` | `state, district, map_vintage, dem_votes, gop_votes, 3p_votes, margin_votes, dem_raw, gop_raw, 3p_raw, margin_raw, margin_2p_norm, pres_margin_raw, pres_margin_2p_norm, split_raw, split_2p_norm, race_type` |
| `{year}_house_adjusted.csv` | `state, district, map_vintage` | Same as raw, plus `is_imputed, imputation_method` |
| `{year}_presidential_district.csv` | `state, district` | `state, district, pres_election, dem_votes, gop_votes, 3p_votes, margin_votes, dem_raw, gop_raw, 3p_raw, margin_raw, margin_2p_norm, map_vintage` |

The adjusted House table uses SHAVE-style estimates for uncontested, missing-major-party, and same-party contests. Ordinary contested results remain unchanged.

## 2. Predictor tables

| Table | Primary key | Schema |
|---|---|---|
| `generic_ballot.csv` | `election, date` | `election, date, dem_raw, gop_raw, 3p_raw, und_raw, margin_raw, margin_2p_norm` |
| `generic_ballot_features.csv` | `election` | `election, cutoff_date, margin_raw, margin_2p_norm, margin_30d_mean_raw, margin_30d_mean_2p_norm, margin_30d_ewma_raw, margin_30d_ewma_2p_norm, momentum_30d_raw, momentum_30d_2p_norm` |
| `incumbency.csv` | `election, state, district` | `election, state, district, incumbency, incumbent_name, incumbent_party, is_open_seat` |
| `cook_pvi.csv` | `election, state, district` | `election, state, district, pvi, pvi_vintage, map_vintage` |
| `demographics.csv` | `acs_vintage, congress, state, district` | `acs_vintage, congress, state, district, college_norm, whitenh_norm, blacknh_norm, asiannh_norm, hispanic_norm, othernh_norm` |

`incumbency` is `+1` for a running Democratic incumbent, `-1` for a running Republican incumbent, and `0` for an open seat.

The generic-ballot features table is **derived from the existing daily `generic_ballot.csv`**, not collected separately. Momentum is defined as a change over time, distinct from the 30-day average or EWMA level.

## 3. SHAVE / imputation reference

| Table | Primary key | Schema |
|---|---|---|
| `shave_retention.csv` | `election, state` | `election, state, pres_election, dem_retention, gop_retention, 3p_norm, n_contested, fallback_method` |
| `house_shave.csv` | `election` | `election, dem_raw, gop_raw, margin_raw, margin_2p_norm, source` |

`shave_retention.csv` contains state-specific vote-retention factors; `house_shave.csv` stores Split Ticket's national counterfactual benchmark.

## 4. Final joined tables

| Table | Primary key | Contents |
|---|---|---|
| `house_modeling.csv` | `election, state, district` | Historical adjusted House margin, all available predictors, and imputation metadata |
| `house_2026.csv` | `election, state, district` | Same predictor fields, without outcomes |
| `house_2026_forecast.csv` | `state, district` | Predicted two-party margin, 80% interval, win probabilities, predicted winner, post-processing method |

In these joined tables, prefix overlapping electoral fields with `gen_`, `house_`, or `pres_`—for example, `gen_margin_2p_norm` and `house_margin_2p_norm`.

## Suggested organization

```text
data/
├── interim/
│   ├── house/
│   │   ├── 2024_house_raw.csv
│   │   ├── 2024_house_adjusted.csv
│   │   └── 2024_presidential_district.csv
│   ├── features/
│   │   ├── generic_ballot.csv
│   │   ├── generic_ballot_features.csv
│   │   ├── incumbency.csv
│   │   ├── cook_pvi.csv
│   │   └── demographics.csv
│   └── national/
│       ├── shave_retention.csv
│       └── house_shave.csv
└── processed/
    ├── house_modeling.csv
    └── house_2026.csv

results/
└── house_2026_forecast.csv
```

## Design details

- Keep the **original ACS and PVI vintages**, even after mapping them to election years. This prevents accidental geographic or temporal mismatches.
- Use `margin_2p_norm` as the House model's historical target. For imputed races, retain `is_imputed` so observed and counterfactual validation errors can be reported separately.

Optional future tables—lagged ticket splitting, cultural regions, and candidate WAR—can be added later without changing this core design.
