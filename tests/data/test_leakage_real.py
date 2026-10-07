"""Leakage check on the real generated applicants. Skipped until the data exists."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.features.columns import assert_no_leakage, model_columns

APPLICANTS_FILE = config.DATA_PROCESSED / "applicants.parquet"

pytestmark = pytest.mark.skipif(not APPLICANTS_FILE.exists(), reason="generated data not found")


def test_real_model_columns_are_leak_free():
    applicants = pd.read_parquet(APPLICANTS_FILE)
    columns = model_columns(applicants)
    assert_no_leakage(columns)
    assert len(columns) == 27
