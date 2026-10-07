"""Population checks on the real cleaned data. Skipped when the file isn't present (e.g. in CI)."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.population import build_population

CLEAN_FILE = config.DATA_INTERIM / "findex_clean.parquet"

pytestmark = pytest.mark.skipif(not CLEAN_FILE.exists(), reason="findex_clean.parquet not built")


@pytest.fixture(scope="module")
def clean():
    return pd.read_parquet(CLEAN_FILE)


def test_population_size(clean):
    assert len(build_population(clean)) == config.EXPECTED_APPLICANTS


def test_real_population_is_reproducible(clean):
    pd.testing.assert_frame_equal(build_population(clean), build_population(clean))


def test_statement_split_matches_data_card(clean):
    population = build_population(clean)
    assert population["has_statement"].sum() == 9_848
