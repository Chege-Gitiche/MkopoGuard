"""Step 1.7: Pandera schemas that formally validate the generated data.

Each rule below comes from docs/data_card.md or docs/findex_columns.md. The schemas check
VALUES (ranges, allowed codes, ID formats, cross-column rules) rather than exact low-level
types, because those differ between pandas versions.
"""

import pandas as pd
import pandera.pandas as pa

from mkopoguard import config
from mkopoguard.data import clean
from mkopoguard.data.transactions import COLUMNS as TRANSACTION_COLUMNS

APPLICANT_ID = r"^A-\d{5}$"

TRANSACTION_DIRECTIONS = {
    "income_in": "in",
    "p2p_in": "in",
    "overdraft_borrow": "in",
    "p2p_out": "out",
    "paybill": "out",
    "till": "out",
    "airtime": "out",
    "betting": "out",
    "overdraft_repay": "out",
}


def _is_datetime() -> pa.Check:
    return pa.Check(pd.api.types.is_datetime64_any_dtype, element_wise=False, name="is_datetime")


def _code_column(mapping: dict, nullable: bool = True) -> pa.Column:
    """A decoded survey column: only the values in its mapping, or missing."""
    return pa.Column(checks=pa.Check.isin(list(mapping.values())), nullable=nullable)


def _probability(nullable: bool = False) -> pa.Column:
    return pa.Column(float, pa.Check.in_range(0, 1), nullable=nullable)


# Survey columns: allowed values come straight from the cleaning mappings
SURVEY_COLUMNS = {
    col: _code_column(mapping, nullable=col not in {"is_female", "income_quintile"})
    for col, mapping in clean.CODE_MAPS.items()
}

APPLICANT_SCHEMA = pa.DataFrameSchema(
    {
        # Identity
        "applicant_id": pa.Column(checks=pa.Check.str_matches(APPLICANT_ID), unique=True),
        "country_code": pa.Column(checks=pa.Check.isin(config.POOL_COUNTRIES)),
        "respondent_id": pa.Column(int, unique=True),
        "survey_year": pa.Column(int, pa.Check.isin([2021, 2022])),
        "survey_weight": pa.Column(float, pa.Check.gt(0)),
        "age": pa.Column("int16", pa.Check.in_range(config.MIN_AGE, 99)),
        **SURVEY_COLUMNS,
        # Hidden traits (step 1.5)
        "hidden_monthly_income_kes": pa.Column(float, pa.Check.gt(0)),
        "hidden_income_pattern": pa.Column(
            checks=pa.Check.isin(["salaried", "seasonal", "irregular"])
        ),
        "hidden_discipline": pa.Column(float),
        "hidden_stress": pa.Column(float),
        "application_date": pa.Column(
            checks=[
                _is_datetime(),
                pa.Check.in_range(
                    pd.Timestamp(config.APPLICATION_START), pd.Timestamp(config.APPLICATION_END)
                ),
            ]
        ),
        "statement_start": pa.Column(checks=_is_datetime()),
        "has_statement": pa.Column("int8", pa.Check.isin([0, 1])),
        # Loans (step 1.6 part 1)
        "loan_amount_kes": pa.Column(
            int,
            [
                pa.Check.in_range(config.LOAN_MIN, config.LOAN_MAX),
                pa.Check(lambda s: s % config.LOAN_ROUND_TO == 0, name="rounded_to_500"),
            ],
        ),
        "term_months": pa.Column(int, pa.Check.isin(config.TERM_OPTIONS)),
        "hidden_loan_to_income": pa.Column(float, pa.Check.gt(0)),
        # Behaviour (step 1.6 part 2)
        "hidden_overdraft_propensity": _probability(),
        "hidden_uses_overdraft": pa.Column("int8", pa.Check.isin([0, 1])),
        "hidden_overdraft_repay_share": _probability(),
        "hidden_betting_share": pa.Column(
            float,
            pa.Check(lambda s: (s == 0) | s.between(0.02, 0.30), name="zero_or_2_to_30_percent"),
        ),
        "hidden_bill_miss_prob": _probability(),
        # Outcome (step 1.6 part 3)
        "hidden_default_probability": _probability(),
        "defaulted": pa.Column("int8", pa.Check.isin([0, 1])),
    },
    checks=[
        pa.Check(
            lambda df: (
                df["application_date"] - df["statement_start"]
                == pd.Timedelta(days=config.STATEMENT_DAYS)
            ),
            name="statement_window_is_180_days",
        ),
        pa.Check(
            lambda df: df["has_statement"] == (df["has_mobile_money"] == 1).astype(int),
            name="statement_only_with_mobile_money",
        ),
    ],
    strict=False,
    coerce=False,
    name="applicants",
)

TRANSACTION_SCHEMA = pa.DataFrameSchema(
    {
        "applicant_id": pa.Column(checks=pa.Check.str_matches(APPLICANT_ID)),
        "timestamp": pa.Column(checks=_is_datetime()),
        "type": pa.Column(checks=pa.Check.isin(list(TRANSACTION_DIRECTIONS))),
        "direction": pa.Column(checks=pa.Check.isin(["in", "out"])),
        "amount_kes": pa.Column(int, pa.Check.gt(0)),
        "balance_after": pa.Column(int, pa.Check.ge(0)),
    },
    checks=pa.Check(
        lambda df: (
            df["type"].astype(str).map(TRANSACTION_DIRECTIONS) == df["direction"].astype(str)
        ),
        name="direction_matches_type",
    ),
    strict=True,
    ordered=True,
    name="transactions",
)
assert list(TRANSACTION_SCHEMA.columns) == TRANSACTION_COLUMNS


def check_transactions_against_applicants(
    applicants: pd.DataFrame, transactions: pd.DataFrame
) -> None:
    """Rules that need both tables: who has statements, and the no-leakage time window."""
    holders = set(applicants.loc[applicants["has_statement"] == 1, "applicant_id"])
    found = set(transactions["applicant_id"])
    if found != holders:
        raise ValueError(
            f"Statement mismatch: {len(found - holders)} unexpected, {len(holders - found)} missing"
        )
    windows = applicants.set_index("applicant_id")[["statement_start", "application_date"]]
    joined = transactions.join(windows, on="applicant_id")
    outside = ~joined["timestamp"].between(
        joined["statement_start"], joined["application_date"], inclusive="left"
    )
    if outside.any():
        raise ValueError(f"{int(outside.sum())} transactions fall outside their statement window")


def validate_dataset(
    applicants: pd.DataFrame, transactions: pd.DataFrame, full: bool = True
) -> None:
    """Validate both tables. full=True also checks the whole-dataset numbers."""
    APPLICANT_SCHEMA.validate(applicants, lazy=True)
    TRANSACTION_SCHEMA.validate(transactions, lazy=True)
    check_transactions_against_applicants(applicants, transactions)
    if full:
        if len(applicants) != config.EXPECTED_APPLICANTS:
            raise ValueError(
                f"Expected {config.EXPECTED_APPLICANTS:,} applicants, got {len(applicants):,}"
            )
        if set(applicants["country_code"]) != set(config.POOL_COUNTRIES):
            raise ValueError("Not all 20 pool countries are present")
        low, high = config.DEFAULT_RATE_BAND
        if not low <= applicants["defaulted"].mean() <= high:
            raise ValueError(
                f"Default rate {applicants['defaulted'].mean():.1%} outside {low:.0%}-{high:.0%}"
            )
