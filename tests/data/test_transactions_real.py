"""Checks on the generated statements. Skipped until generate_synthetic.py has been run."""

import pandas as pd
import pytest

from mkopoguard import config

APPLICANTS_FILE = config.DATA_PROCESSED / "applicants.parquet"
TRANSACTIONS_FILE = config.DATA_PROCESSED / "transactions.parquet"

pytestmark = pytest.mark.skipif(
    not TRANSACTIONS_FILE.exists(), reason="transactions.parquet not generated"
)


@pytest.fixture(scope="module")
def applicants():
    return pd.read_parquet(APPLICANTS_FILE)


@pytest.fixture(scope="module")
def transactions():
    return pd.read_parquet(TRANSACTIONS_FILE)


def test_size_is_in_the_expected_range(transactions):
    assert 1_000_000 <= len(transactions) <= 3_000_000


def test_only_statement_holders_have_transactions(applicants, transactions):
    holders = set(applicants.loc[applicants["has_statement"] == 1, "applicant_id"])
    assert set(transactions["applicant_id"]) == holders


def test_no_transaction_outside_the_statement_window(applicants, transactions):
    windows = applicants.set_index("applicant_id")[["statement_start", "application_date"]]
    joined = transactions.join(windows, on="applicant_id")
    assert (joined["timestamp"] >= joined["statement_start"]).all()
    assert (joined["timestamp"] < joined["application_date"]).all()


def test_no_negative_balances(transactions):
    assert (transactions["balance_after"] >= 0).all()


def test_default_rate_is_in_the_band_overall_and_in_kenya(applicants):
    low, high = config.DEFAULT_RATE_BAND
    assert low <= applicants["defaulted"].mean() <= high
    kenya = applicants.loc[applicants["country_code"] == "KEN", "defaulted"]
    assert low <= kenya.mean() <= high


def test_every_applicant_has_an_outcome(applicants):
    assert applicants["defaulted"].notna().all()
