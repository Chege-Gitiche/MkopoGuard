"""Step 7.2: the release gates for model quality (docs/test_strategy.md, section 5).

The final test report (Phase 3) measured the champion once on the locked test set. These
tests hold the shipped numbers to the floors the strategy agreed, so a retrained model that is
worse can't be released by accident.
"""

import json

import pytest

from mkopoguard import config

MIN_TEST_PR_AUC = 0.35  # a random guess scores the default rate, about 0.20
MAX_TEST_ECE = 0.03

pytestmark = pytest.mark.skipif(
    not config.TEST_REPORT_FILE.exists(), reason="final test report not written yet"
)


@pytest.fixture(scope="module")
def report():
    return json.loads(config.TEST_REPORT_FILE.read_text(encoding="utf-8"))


def test_the_model_clears_the_pr_auc_floor(report):
    pr_auc = report["ranking"]["pr_auc"]["value"]
    assert pr_auc >= MIN_TEST_PR_AUC
    assert pr_auc > 1.5 * report["default_rate"]  # well above guessing, not just above it


def test_the_model_is_well_calibrated(report):
    assert report["calibration"]["ece"] <= MAX_TEST_ECE


def test_the_floor_is_below_the_measured_interval(report):
    """The floor isn't set so close to the score that noise alone would fail a release."""
    assert MIN_TEST_PR_AUC < report["ranking"]["pr_auc"]["low"]
