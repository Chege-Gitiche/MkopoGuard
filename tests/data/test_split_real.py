"""The committed split against the current model table. Skipped until both exist."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.split import make_splits, read_splits

FEATURES_FILE = config.DATA_PROCESSED / "features.parquet"

pytestmark = pytest.mark.skipif(
    not (FEATURES_FILE.exists() and config.SPLIT_FILE.exists()),
    reason="features.parquet or the split file not built",
)


@pytest.fixture(scope="module")
def table():
    return pd.read_parquet(FEATURES_FILE)


def test_split_covers_exactly_the_current_applicants(table):
    assert set(read_splits()["applicant_id"]) == set(table["applicant_id"])


def test_committed_split_has_not_been_changed_by_hand(table):
    fresh = make_splits(table)
    pd.testing.assert_frame_equal(read_splits().astype(str), fresh.astype(str))


def test_enough_kenyans_in_the_test_set(table):
    t = table.merge(read_splits(), on="applicant_id")
    kenyan_test = ((t["split"] == "test") & (t["country_code"] == config.TARGET_COUNTRY)).sum()
    assert kenyan_test >= 130
