"""Schema tests. The fixture runs the WHOLE Phase 1 pipeline on a small made-up sample,
so these tests run in CI without the real data."""

import numpy as np
import pandas as pd
import pandera.errors
import pytest

from mkopoguard import config
from mkopoguard.data.clean import COLUMN_MAP, clean_findex
from mkopoguard.data.defaults import add_defaults
from mkopoguard.data.loans import add_loans
from mkopoguard.data.population import build_population
from mkopoguard.data.schemas import (
    APPLICANT_SCHEMA,
    TRANSACTION_SCHEMA,
    check_transactions_against_applicants,
    validate_dataset,
)
from mkopoguard.data.transactions import add_behaviour, generate_transactions

# Possible raw codes per Findex column, including don't know / refused codes
CODES = {
    "female": [1, 2],
    "educ": [1, 2, 3, 4, 5],
    "inc_q": [1, 2, 3, 4, 5],
    "emp_in": [1, 2],
    "mobileowner": [1, 2, 3, 4],
    "internetaccess": [1, 2, 3, 4],
    "fin24": list(range(1, 10)),
    "fin44c": list(range(1, 7)),
    "receive_wages": [1, 2, 3, 4, 5],
    "receive_agriculture": [1, 2, 3, 4, 5],
    "pay_utilities": [1, 2, 3, 4, 5],
}
ZERO_ONE = ["account_fin", "account_mob", "saved", "borrowed", "merchantpay_dig"]
YES_NO = ["fin17b", "fin22a", "fin22b", "fin20", "fin26", "fin28", "fin37"]


def make_raw_sample(n=80, seed=0):
    """n made-up raw Findex respondents with random but valid codes."""
    rng = np.random.default_rng(seed)
    raw = pd.DataFrame(
        {
            "economycode": rng.choice(config.POOL_COUNTRIES, n),
            "economy": "Somewhere",
            "wpid_random": np.arange(1, n + 1),
            "wgt": rng.uniform(0.2, 3, n),
            "year": 2021,
            "age": rng.integers(15, 80, n).astype(float),  # includes minors, to be dropped
        }
    )
    for col, codes in CODES.items():
        raw[col] = rng.choice(codes, n)
    for col in ZERO_ONE:
        raw[col] = rng.choice([0, 1], n)
    for col in YES_NO:
        raw[col] = rng.choice([1, 2, 3, 4], n, p=[0.45, 0.45, 0.05, 0.05])
    assert set(COLUMN_MAP) <= set(raw.columns)
    return raw


@pytest.fixture(scope="module")
def sample():
    applicants = build_population(clean_findex(make_raw_sample()))
    applicants = add_behaviour(add_loans(applicants))
    transactions = generate_transactions(applicants)
    applicants = add_defaults(applicants)
    return applicants, transactions


# --- the generated sample passes every rule ------------------------------------


def test_generated_sample_is_valid(sample):
    applicants, transactions = sample
    validate_dataset(applicants, transactions, full=False)


# --- broken data is caught ------------------------------------------------------


@pytest.mark.parametrize(
    "column, bad_value",
    [
        ("country_code", "FRA"),
        ("age", np.int16(17)),
        ("term_months", 2),
        ("loan_amount_kes", 1_234),
        ("defaulted", np.int8(2)),
        ("hidden_default_probability", 1.2),
        ("education_level", "phd"),
    ],
)
def test_applicant_schema_rejects_bad_values(sample, column, bad_value):
    applicants = sample[0].copy()
    if isinstance(applicants[column].dtype, pd.CategoricalDtype):
        applicants[column] = applicants[column].astype(str)
    applicants.loc[0, column] = bad_value
    with pytest.raises(pandera.errors.SchemaError):
        APPLICANT_SCHEMA.validate(applicants)


def test_applicant_schema_rejects_wrong_statement_window(sample):
    applicants = sample[0].copy()
    applicants.loc[0, "statement_start"] += pd.Timedelta(days=1)
    with pytest.raises(pandera.errors.SchemaError, match="statement_window"):
        APPLICANT_SCHEMA.validate(applicants)


def test_applicant_schema_rejects_duplicate_ids(sample):
    applicants = sample[0].copy()
    applicants.loc[1, "applicant_id"] = applicants.loc[0, "applicant_id"]
    with pytest.raises(pandera.errors.SchemaError):
        APPLICANT_SCHEMA.validate(applicants)


@pytest.mark.parametrize(
    "column, bad_value",
    [("amount_kes", -50), ("balance_after", -1), ("type", "loan_shark")],
)
def test_transaction_schema_rejects_bad_values(sample, column, bad_value):
    transactions = sample[1].copy()
    transactions[column] = transactions[column].astype(object)
    transactions.loc[0, column] = bad_value
    with pytest.raises(pandera.errors.SchemaError):
        TRANSACTION_SCHEMA.validate(transactions)


def test_transaction_schema_rejects_wrong_direction(sample):
    transactions = sample[1].copy()
    transactions["direction"] = transactions["direction"].astype(str)
    first_out = transactions.index[transactions["direction"] == "out"][0]
    transactions.loc[first_out, "direction"] = "in"
    with pytest.raises(pandera.errors.SchemaError, match="direction_matches_type"):
        TRANSACTION_SCHEMA.validate(transactions)


def test_leak_after_application_date_is_caught(sample):
    applicants, transactions = sample
    leaked = transactions.copy()
    first = leaked.loc[0, "applicant_id"]
    app_date = applicants.set_index("applicant_id").loc[first, "application_date"]
    leaked.loc[0, "timestamp"] = app_date + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="outside their statement window"):
        check_transactions_against_applicants(applicants, leaked)


def test_full_check_rejects_a_partial_dataset(sample):
    with pytest.raises(ValueError, match="Expected"):
        validate_dataset(*sample, full=True)
