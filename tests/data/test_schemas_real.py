"""Validate the real generated dataset. Skipped until generate_synthetic.py has been run."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.schemas import validate_dataset

APPLICANTS_FILE = config.DATA_PROCESSED / "applicants.parquet"
TRANSACTIONS_FILE = config.DATA_PROCESSED / "transactions.parquet"

pytestmark = pytest.mark.skipif(not TRANSACTIONS_FILE.exists(), reason="generated data not found")


def test_generated_dataset_passes_all_schemas():
    applicants = pd.read_parquet(APPLICANTS_FILE)
    transactions = pd.read_parquet(TRANSACTIONS_FILE)
    validate_dataset(applicants, transactions, full=True)
