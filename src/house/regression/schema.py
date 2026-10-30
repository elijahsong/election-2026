"""Canonical columns of the House modeling data, and a data-quality report for any frame of it.

Edit COLUMNS to register a column's kind, role, and tranche (the data/processed/house file it comes from; see
data-processing/house/house_features.py). Columns not registered are allowed and never checked unless a caller names them
in `required`. Validation only reports: nothing is renamed, coerced, dropped, or filled.

Severity, by column (see validate_schema):
  key (year, state, district)   any problem is a blocking error; so are duplicate keys (a caller may pass another key)
  required                      any problem is a blocking error. Callers pass the selected predictors (and the
                                config's target when fitting or scoring), checked only on the rows they pass in
  every other registered column null or invalid values are non-blocking warnings, and so is an absent column, except
                                a target: targets live in their own table and are often absent from feature data.
                                A nullable column's nulls are expected and not reported
"""

import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

KEY = ["year", "state", "district"]
KINDS = {"int", "str", "numeric", "bool"}
ROLES = {"key", "target", "predictor", "metadata"}
DATE = r"\d{4}-\d{2}-\d{2}"


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # "int", "str", "numeric", or "bool"
    role: str  # "key", "target", "predictor", or "metadata"
    tranche: str = ""  # source table in data/processed/house, e.g. "basic_f"
    pattern: str | None = None  # regex a "str" value must fully match
    min_value: float | None = None
    description: str = ""
    nullable: bool = False  # nulls are expected (e.g. quality_flags, imputation_method): not reported unless required


def _numeric(tranche: str, names: list[str], role: str = "predictor") -> list[Column]:
    return [Column(name, "numeric", role, tranche) for name in names]


COLUMNS = {c.name: c for c in [
    Column("year", "int", "key", description="election year"),
    Column("state", "str", "key", pattern=r"[A-Z]{2}", description="USPS code"),
    Column("district", "int", "key", min_value=0, description="1..n, and 0 for at-large"),
    # targets.csv (data-processing/house/house_targets.py; every column is in TARGETS_SCHEMA); a config picks its "target"
    Column("adj_margin_2p_norm", "numeric", "target", "targets",
           description="canonical Stage 1 target, D positive: margin_2p_norm for a dr* race with reported votes, or an "
                       "imputed value (is_imputed True; never ground truth, see MODELED_IMPUTATION_METHODS); null "
                       "where neither"),
    # basic_f.csv
    Column("gen_margin_raw", "numeric", "predictor", "basic_f", description="generic-ballot D-R margin"),
    Column("partisan_lean", "numeric", "predictor", "basic_f"),
    Column("inc_signed", "numeric", "predictor", "basic_f", description="+1 D incumbent, -1 R, 0 open"),
    # acs_f.csv
    Column("acs_vintage", "str", "metadata", "acs_f", description="ACS product used, e.g. 2018_1yr"),
    *_numeric("acs_f", [
        "white", "black", "hispanic", "college", "median_income", "median_age", "moved_within_past_yr",
        "foreign_born", "unemployment", "poverty", "avg_household_size", "housing_burden", "rel_college",
        "log_rel_income", "rel_state_college", "log_rel_state_income", "rel_unemployment", "rel_state_unemployment",
        "rel_housing_burden", "rel_state_housing_burden",
    ]),
    # fec_f.csv (FEC cycle totals; historical years run through Dec 31, after the election, 2026 through mid-October)
    Column("fundraising_advantage", "numeric", "predictor", "fec_f",
           description="log1p(D receipts) - log1p(R receipts), capped at +-5; 0 money where a party has no nominee"),
    Column("spending_advantage", "numeric", "predictor", "fec_f",
           description="log1p(D disbursements) - log1p(R disbursements), capped at +-5; null for a negative input"),
    Column("small_dollar_share_diff", "numeric", "predictor", "fec_f",
           description="D - R unitemized share of individual contributions; null in 2006, for a zero denominator, "
                       "or a source share outside [0, 1]"),
    Column("dem_fec_missing", "bool", "metadata", "fec_f", description="no FEC record (or no D nominee): amounts are 0"),
    Column("gop_fec_missing", "bool", "metadata", "fec_f", description="no FEC record (or no R nominee): amounts are 0"),
    # nat_political_f.csv
    Column("gb_cutoff_date", "str", "metadata", "nat_political_f", pattern=DATE),
    Column("gb_date", "str", "metadata", "nat_political_f", pattern=DATE),
    *_numeric("nat_political_f", [
        "gb_margin_raw", "gb_margin_2p_norm", "gb_dem_pct", "gb_gop_pct", "gb_und_pct", "gb_dem_hi", "gb_dem_lo",
        "gb_gop_hi", "gb_gop_lo", "gb_margin_30d_mean_raw", "gb_margin_30d_ewma_raw", "gb_momentum_30d_raw",
        "gb_margin_30d_mean_2p_norm", "gb_margin_30d_ewma_2p_norm", "gb_momentum_30d_2p_norm",
    ]),
]}

# tranche -> its registered predictors, for building feature sets in config.py
TRANCHES = {t: [c.name for c in COLUMNS.values() if c.tranche == t and c.role == "predictor"]
            for t in ["basic_f", "acs_f", "fec_f", "nat_political_f"]}

# data/processed/house/targets.csv (and targets_raw.csv, all years), in file order: TARGET_COLUMNS, then provenance.
# Check it with validate_schema(df, schema=TARGETS_SCHEMA).
TARGET_COLUMNS = [
    *KEY,
    "dem_votes", "gop_votes", "3p_votes", "total_votes", "dem_pct", "gop_pct", "3p_pct",
    "margin_votes", "margin_raw", "margin_2p_norm",
    "race_type", "winner_party", "dem_candidate_count", "gop_candidate_count", "3p_candidate_count",
    "is_rcv", "is_special_election",
    "adj_margin_2p_norm",
    "dem_votes_imputed", "gop_votes_imputed", "is_imputed", "imputation_method",
    "needs_imputation", "is_nonstandard_race",
]
TARGET_PROVENANCE_COLUMNS = ["source_totalvotes", "nonvote_votes", "is_unofficial", "quality_flags", "source"]
# imputation_method values estimated from other races rather than observed: never ground truth, so data.join_target
# blanks their target and they are neither trained on nor scored
MODELED_IMPUTATION_METHODS = ["state_mean_residual", "national_shave"]
# documented in data-processing/house/README.md; the dr* types have a target
RACE_TYPES = ["dr", "dr_multi", "dr_jungle", "dr_rcv", "dr_writein", "dem_vs_3p", "gop_vs_3p", "dem_uncontested",
              "gop_uncontested", "dem_vs_dem", "gop_vs_gop", "cross_endorsed", "no_major_party"]
TARGETS_SCHEMA = {c.name: c for c in [
    *(COLUMNS[name] for name in KEY),
    *(Column(name, "numeric", "metadata", "targets", min_value=0) for name in [
        "dem_votes", "gop_votes", "3p_votes", "total_votes",
        "dem_candidate_count", "gop_candidate_count", "3p_candidate_count", "source_totalvotes", "nonvote_votes",
    ]),
    *(Column(name, "numeric", "metadata", "targets", min_value=0) for name in ["dem_pct", "gop_pct", "3p_pct"]),
    *(Column(name, "numeric", "metadata", "targets") for name in ["margin_votes", "margin_raw", "margin_2p_norm"]),
    Column("race_type", "str", "metadata", "targets", pattern="|".join(RACE_TYPES),
           description="from candidate counts by party; see data-processing/house/house_targets.py"),
    Column("winner_party", "str", "metadata", "targets", pattern=r"D|R|3P",
           description="null when the source doesn't settle the winner's party"),
    Column("is_rcv", "bool", "metadata", "targets"),
    Column("is_special_election", "bool", "metadata", "targets"),
    COLUMNS["adj_margin_2p_norm"],
    Column("dem_votes_imputed", "numeric", "metadata", "targets", min_value=0,
           description="the D votes behind an imputed target; null where none is imputed, or where the imputation "
                       "has no turnout (national_shave)", nullable=True),
    Column("gop_votes_imputed", "numeric", "metadata", "targets", min_value=0,
           description="the R votes behind an imputed target; null where none is imputed, or where the imputation "
                       "has no turnout (national_shave)", nullable=True),
    Column("is_imputed", "bool", "metadata", "targets",
           description="False for an observed adj_margin_2p_norm, True for an imputed one, null where it is missing"),
    Column("imputation_method", "str", "metadata", "targets",
           description="how an imputed target was made (house_impute_targets_uncontested.py); null otherwise",
           nullable=True),
    Column("needs_imputation", "bool", "metadata", "targets",
           description="True where there is no observed target; stays True once one is imputed"),
    Column("is_nonstandard_race", "bool", "metadata", "targets",
           description="True for every race_type but dr and dr_multi"),
    Column("is_unofficial", "bool", "metadata", "targets"),
    Column("quality_flags", "str", "metadata", "targets", description="semicolon-separated; null when none",
           nullable=True),
    Column("source", "str", "metadata", "targets"),
]}

# data/processed/house/targets_supp.csv: every contest of the keys whose source has more than one, keyed by SUPP_KEY,
# with TARGETS_SCHEMA's columns computed per contest; kept for future imputation of adj_margin_2p_norm and for
# evaluation, not joined into targets.csv. Check it with
# validate_schema(df, schema=TARGETS_SUPP_SCHEMA, key=SUPP_KEY).
SUPP_KEY = [*KEY, "contest"]
TARGET_SUPP_COLUMNS = [
    *SUPP_KEY,
    "dem_votes", "gop_votes", "3p_votes", "total_votes", "dem_pct", "gop_pct", "3p_pct",
    "margin_votes", "margin_raw", "margin_2p_norm",
    "race_type", "winner_party", "dem_candidate_count", "gop_candidate_count", "3p_candidate_count",
    "is_rcv", "is_special_election", *TARGET_PROVENANCE_COLUMNS,
]
TARGETS_SUPP_SCHEMA = {
    **{name: COLUMNS[name] for name in KEY},
    "contest": Column("contest", "str", "key", "targets_supp", pattern="first_round|runoff|special",
                      description="first_round is the contest the targets use"),
    **{name: TARGETS_SCHEMA[name] for name in TARGET_SUPP_COLUMNS if name not in SUPP_KEY},
}

# data/processed/house/targets_candidates.csv: one row per candidate (and aggregate write-in / non-vote row) of every
# targets_supp.csv contest, keyed by CANDIDATE_KEY; joins to targets_supp.csv on SUPP_KEY. Check it with
# validate_schema(df, schema=TARGETS_CANDIDATES_SCHEMA, key=CANDIDATE_KEY).
CANDIDATE_KEY = [*SUPP_KEY, "candidate"]
TARGET_CANDIDATE_COLUMNS = [*CANDIDATE_KEY, "parties", "party_group", "is_candidate", "is_writein", "votes", "source"]
TARGETS_CANDIDATES_SCHEMA = {
    **{name: TARGETS_SUPP_SCHEMA[name] for name in SUPP_KEY},
    "candidate": Column("candidate", "str", "key", "targets_candidates", description="as the source spells it"),
    "parties": Column("parties", "str", "metadata", "targets_candidates",
                      description="the source's labels on the candidate's lines, '; '-separated; null if party-less"),
    "party_group": Column("party_group", "str", "metadata", "targets_candidates", pattern=r"D|R|3P|D\+R|nonvote",
                          description="D+R for a cross-endorsed candidate; nonvote for blank / exhausted rows"),
    "is_candidate": Column("is_candidate", "bool", "metadata", "targets_candidates",
                           description="False for aggregate write-in / other rows and non-vote rows"),
    "is_writein": Column("is_writein", "bool", "metadata", "targets_candidates", description="every line a write-in"),
    "votes": Column("votes", "numeric", "metadata", "targets_candidates", min_value=0,
                    description="summed across the candidate's party lines"),
    "source": Column("source", "str", "metadata", "targets_candidates"),
}

ISSUE_COLUMNS = ["column", "issue", "severity", "count", "keys"]


class SchemaError(ValueError):
    """Blocking validation errors; `.report` holds the full ValidationReport."""

    def __init__(self, report: "ValidationReport"):
        super().__init__("\n" + report.summary())
        self.report = report


@dataclass
class ValidationReport:
    """One row per (column, issue): severity ("error" / "warning"), affected row count, and every affected key."""

    issues: pd.DataFrame

    @property
    def errors(self) -> pd.DataFrame:
        return self.issues[self.issues["severity"] == "error"]

    @property
    def warnings(self) -> pd.DataFrame:
        return self.issues[self.issues["severity"] == "warning"]

    @property
    def ok(self) -> bool:
        return self.errors.empty

    def summary(self, max_keys: int = 3, max_columns: int = 4) -> str:
        """One line per issue, with columns that share an issue on exactly the same rows listed together."""
        groups = {}
        for issue in self.issues.to_dict("records"):
            groups.setdefault((issue["severity"], issue["issue"], tuple(issue["keys"])), []).append(issue["column"])
        lines = []
        for (severity, issue, keys), columns in groups.items():
            names = ", ".join(columns[:max_columns]) + (f", ... (+{len(columns) - max_columns})"
                                                        if len(columns) > max_columns else "")
            shown = ", ".join(map(str, keys[:max_keys])) + (f", ... (+{len(keys) - max_keys})"
                                                            if len(keys) > max_keys else "")
            lines.append(f"{severity}: {issue} x{len(keys)} in {names}: {shown}")
        return "\n".join(lines) or "no issues"

    def warn(self, stacklevel: int = 2) -> None:
        """Emit the warnings, if any, as one summarized Python warning attributed `stacklevel` frames up."""
        if not self.warnings.empty:
            summary = ValidationReport(self.warnings).summary()
            warnings.warn(f"\n{summary}\n(report.issues lists every affected key)", stacklevel=stacklevel + 1)

    def raise_if_errors(self) -> None:
        if not self.ok:
            raise SchemaError(self)


def _invalid(values: pd.Series, column: Column) -> pd.Series:
    """True where a non-null value doesn't fit the column's kind, pattern, or minimum."""
    present = values.notna()
    if column.kind == "str":
        pattern = re.compile(column.pattern or r".*")
        return present & ~values.map(lambda v: isinstance(v, str) and pattern.fullmatch(v) is not None).astype(bool)
    if column.kind == "bool":
        return present & ~values.map(lambda v: isinstance(v, (bool, np.bool_))).astype(bool)
    is_kind = pd.api.types.is_integer_dtype if column.kind == "int" else pd.api.types.is_numeric_dtype
    if not is_kind(values) or pd.api.types.is_bool_dtype(values):
        return present
    invalid = present & np.isinf(values.astype(float))
    if column.min_value is not None:
        invalid |= present & (values < column.min_value)
    return invalid


def validate_schema(df: pd.DataFrame, required=(), schema: dict[str, Column] = COLUMNS,
                    key: list[str] = KEY) -> ValidationReport:
    """Check the key, the `required` columns (blocking), and every other registered column (warnings) on df's rows."""
    required = list(required)
    keys = list(df.reindex(columns=key).itertuples(index=False, name=None))
    issues = []

    def add(column: str, issue: str, severity: str, mask) -> None:
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            issues.append((column, issue, severity, int(mask.sum()), [k for k, m in zip(keys, mask) if m]))

    checks = [(schema[name], "error") for name in key]
    checks += [(schema.get(name, Column(name, "numeric", "predictor")), "error") for name in required if name not in key]
    checks += [(c, "warning") for c in schema.values() if c.name not in key and c.name not in required]
    for column, severity in checks:
        if column.name not in df.columns:
            if severity == "error" or column.role != "target":
                add(column.name, "missing_column", severity, np.ones(len(df), dtype=bool))
            continue
        if severity == "error" or not column.nullable:
            add(column.name, "null", severity, df[column.name].isna())
        add(column.name, "invalid_value", severity, _invalid(df[column.name], column))
    if set(key) <= set(df.columns):
        add(", ".join(key), "duplicate_key", "error", df.duplicated(key, keep=False))
    return ValidationReport(pd.DataFrame(issues, columns=ISSUE_COLUMNS))
