"""Checks on the real model table. Skipped until build_features.py has been run."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.features.transactions import FEATURES

FEATURES_FILE = config.DATA_PROCESSED / "features.parquet"

pytestmark = pytest.mark.skipif(not FEATURES_FILE.exists(), reason="features.parquet not built")


@pytest.fixture(scope="module")
def table():
    return pd.read_parquet(FEATURES_FILE)


def test_one_row_per_applicant(table):
    assert len(table) == config.EXPECTED_APPLICANTS
    assert table["applicant_id"].is_unique


def test_no_hidden_columns_in_the_file(table):
    assert not [c for c in table.columns if c.startswith("hidden_")]


def test_statement_holders_have_every_feature(table):
    holders = table[table["has_statement"] == 1]
    assert holders[FEATURES].notna().all().all()


def test_thin_file_applicants_have_no_statement_features(table):
    thin = table[table["has_statement"] == 0]
    assert thin[FEATURES].isna().all().all()
