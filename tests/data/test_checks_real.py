"""Plausibility of the real generated data. Skipped until generate_synthetic.py has been run."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.checks import assert_plausible

APPLICANTS_FILE = config.DATA_PROCESSED / "applicants.parquet"
TRANSACTIONS_FILE = config.DATA_PROCESSED / "transactions.parquet"

pytestmark = pytest.mark.skipif(not TRANSACTIONS_FILE.exists(), reason="generated data not found")


def test_generated_data_is_plausible():
    assert_plausible(pd.read_parquet(APPLICANTS_FILE), pd.read_parquet(TRANSACTIONS_FILE))
