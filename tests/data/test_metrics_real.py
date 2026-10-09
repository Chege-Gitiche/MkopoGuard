"""The leakage alarm against the real data. Skipped until the data and split exist."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.split import SPLITS, read_splits
from mkopoguard.models.metrics import evaluate, leakage_alarms

APPLICANTS_FILE = config.DATA_PROCESSED / "applicants.parquet"

pytestmark = pytest.mark.skipif(
    not (APPLICANTS_FILE.exists() and config.SPLIT_FILE.exists()),
    reason="applicants.parquet or the split file not built",
)


@pytest.mark.parametrize("split", SPLITS)
def test_true_probabilities_stay_below_the_alarm(split):
    """Even perfect knowledge of the hidden default probability must not trip the alarm,
    so only a genuine leak (a model seeing something it shouldn't) can."""
    applicants = pd.read_parquet(APPLICANTS_FILE).merge(read_splits(), on="applicant_id")
    part = applicants[applicants["split"] == split]
    ceiling = evaluate(part["defaulted"], part["hidden_default_probability"])
    assert leakage_alarms(ceiling) == {}
    assert 0.80 < ceiling["roc_auc"] < config.LEAKAGE_ALARM["roc_auc"]
