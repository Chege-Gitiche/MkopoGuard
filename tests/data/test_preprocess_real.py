"""The pipeline on the real model table. Skipped until build_features.py has been run."""

import numpy as np
import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.features.preprocess import build_preprocessor, input_columns

FEATURES_FILE = config.DATA_PROCESSED / "features.parquet"

pytestmark = pytest.mark.skipif(not FEATURES_FILE.exists(), reason="features.parquet not built")


def test_real_table_becomes_68_clean_numeric_columns():
    table = pd.read_parquet(FEATURES_FILE)
    out = build_preprocessor().fit_transform(table[input_columns()])
    assert out.shape == (config.EXPECTED_APPLICANTS, 68)
    assert np.isfinite(out.to_numpy()).all()
