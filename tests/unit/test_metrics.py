"""Known-answer tests for the evaluation metrics (step 3.2)."""

import numpy as np
import pytest

from mkopoguard.models.metrics import (
    assert_no_leakage_alarm,
    bootstrap_ci,
    calibration_table,
    decision_metrics,
    evaluate,
    evaluate_by_group,
    expected_calibration_error,
    ks_statistic,
    leakage_alarms,
)

Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 20% default, like the real data
PERFECT = np.array([0.1] * 8 + [0.9] * 2)  # every defaulter scored above every non-defaulter
CONSTANT = np.full(10, 0.2)  # "guess the average"


# --- ranking ------------------------------------------------------------------


def test_perfect_ranking():
    m = evaluate(Y, PERFECT)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["ks"] == pytest.approx(1.0)


def test_guessing_the_average_has_no_skill():
    m = evaluate(Y, CONSTANT)
    assert m["roc_auc"] == pytest.approx(0.5)
    assert m["ks"] == pytest.approx(0.0)
    # With no ranking at all, PR-AUC equals the default rate: that is its "no skill" line
    assert m["pr_auc"] == pytest.approx(0.2)


def test_ks_is_the_largest_gap_between_the_two_groups():
    # Defaulters at 0.6 and 0.9; non-defaulters spread out with two of them above 0.6.
    # Threshold 0.6 catches both defaulters (TPR 1.0) and 2 of 8 others (FPR 0.25): KS 0.75
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.7, 0.8, 0.6, 0.9])
    assert ks_statistic(Y, p) == pytest.approx(0.75)


# --- probabilities --------------------------------------------------------------


def test_brier_known_values():
    assert evaluate(Y, np.clip(Y.astype(float), 0, 1))["brier"] == pytest.approx(0.0)
    assert evaluate(Y, np.full(10, 0.5))["brier"] == pytest.approx(0.25)


def test_brier_skill_is_zero_for_guessing_the_average_and_one_for_perfect():
    assert evaluate(Y, CONSTANT)["brier_skill"] == pytest.approx(0.0)
    assert evaluate(Y, Y.astype(float))["brier_skill"] == pytest.approx(1.0)


def test_calibration_table_bins():
    p = np.array([0.05, 0.05, 0.15, 0.15, 0.95])
    y = np.array([0, 1, 0, 0, 1])
    table = calibration_table(y, p)
    assert table["applicants"].tolist() == [2, 2, 1]  # empty bins are left out
    assert table["actual"].tolist() == [0.5, 0.0, 1.0]
    assert table["lower"].tolist() == pytest.approx([0.0, 0.1, 0.9])


def test_well_calibrated_scores_have_small_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50_000)
    y = (rng.random(50_000) < p).astype(int)  # outcomes drawn from the predictions themselves
    assert expected_calibration_error(y, p) < 0.01


def test_overconfident_scores_have_large_ece():
    # Everyone is predicted at 0.9 but only 20% default: off by 0.7
    assert expected_calibration_error(Y, np.full(10, 0.9)) == pytest.approx(0.7)


# --- decisions -------------------------------------------------------------------


def test_decision_metrics_by_hand():
    # Threshold 0.5 declines rows 6, 7 (non-defaulters) and 9 (defaulter); row 8 is missed
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 0.7, 0.4, 0.8])
    d = decision_metrics(Y, p, threshold=0.5)
    assert d["recall"] == pytest.approx(0.5)  # 1 of 2 defaulters caught
    assert d["precision"] == pytest.approx(1 / 3)  # 1 of 3 declines was a defaulter
    assert d["false_positive_rate"] == pytest.approx(2 / 8)
    assert d["accuracy"] == pytest.approx(7 / 10)
    assert d["approval_rate"] == pytest.approx(7 / 10)


def test_accuracy_rewards_approving_everyone():
    """Why accuracy is never the headline: approving everyone scores 80% here."""
    d = decision_metrics(Y, CONSTANT, threshold=0.5)
    assert d["accuracy"] == pytest.approx(0.8)
    assert d["recall"] == 0.0


def test_precision_is_undefined_when_nobody_is_declined():
    assert np.isnan(decision_metrics(Y, CONSTANT, threshold=0.5)["precision"])


# --- input checks -----------------------------------------------------------------


def test_evaluate_needs_both_outcomes():
    with pytest.raises(ValueError, match="both"):
        evaluate(np.zeros(10), CONSTANT)


@pytest.mark.parametrize("bad", [np.full(10, 1.5), np.full(10, -0.1), np.full(10, np.nan)])
def test_evaluate_rejects_impossible_probabilities(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate(Y, bad)


def test_evaluate_reports_every_planned_metric():
    expected = {
        "pr_auc", "roc_auc", "ks", "brier", "brier_skill", "ece", "recall", "precision",
        "false_positive_rate", "accuracy", "approval_rate", "threshold", "default_rate",
        "mean_predicted", "n",
    }  # fmt: skip
    assert set(evaluate(Y, PERFECT)) == expected


# --- uncertainty and groups ---------------------------------------------------------


def test_bootstrap_interval_contains_the_estimate_and_narrows_with_more_data():
    rng = np.random.default_rng(0)

    def sample(n):
        y = (rng.random(n) < 0.2).astype(int)
        p = np.clip(0.2 + 0.3 * (y - 0.2) + rng.normal(0, 0.15, n), 0, 1)
        return y, p

    y_small, p_small = sample(150)
    y_big, p_big = sample(3_000)
    low, high = bootstrap_ci(y_small, p_small, "roc_auc", resamples=300)
    assert low < evaluate(y_small, p_small)["roc_auc"] < high
    big_low, big_high = bootstrap_ci(y_big, p_big, "roc_auc", resamples=300)
    assert (big_high - big_low) < (high - low)


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.2).astype(int)
    p = rng.random(200)
    assert bootstrap_ci(y, p, resamples=100) == bootstrap_ci(y, p, resamples=100)


def test_evaluate_by_group_gives_one_row_per_group():
    groups = {"all": np.ones(10, bool), "first_half": np.arange(10) < 5}
    first_half = np.array([0, 1, 0, 0, 1, 0, 0, 0, 1, 1])
    table = evaluate_by_group(first_half, PERFECT, groups)
    assert list(table.index) == ["all", "first_half"]
    assert table.loc["first_half", "n"] == 5


# --- leakage alarm -------------------------------------------------------------------


def test_realistic_scores_raise_no_alarm():
    assert leakage_alarms({"roc_auc": 0.72, "pr_auc": 0.41}) == {}


def test_impossibly_good_scores_raise_the_alarm():
    with pytest.raises(ValueError, match="leakage"):
        assert_no_leakage_alarm({"roc_auc": 0.93, "pr_auc": 0.41})
