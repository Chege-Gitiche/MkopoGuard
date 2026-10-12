"""Tests for step 3.11: the cost-based decision rule (pure arithmetic, runs in CI)."""

import json

import numpy as np
import pytest

from mkopoguard import config
from mkopoguard.models import decision as dec

# --- the threshold formula ----------------------------------------------------------------


def test_threshold_is_where_expected_profit_is_zero():
    margin, lgd, term = 0.065, 0.80, 3
    t = float(dec.threshold(term, margin, lgd))
    gain_if_repaid, loss_if_default = margin * term, lgd
    assert (1 - t) * gain_if_repaid - t * loss_if_default == pytest.approx(0.0)


def test_known_values():
    assert float(dec.threshold(1, 0.065, 0.80)) == pytest.approx(0.065 / 0.865)
    assert float(dec.threshold(6, 0.10, 0.60)) == pytest.approx(0.5)


def test_longer_loans_can_carry_more_risk():
    t = dec.thresholds_by_term(0.065, 0.80)
    assert t[1] < t[3] < t[6]


def test_higher_losses_or_thinner_margins_mean_stricter_lending():
    assert dec.threshold(3, 0.065, 1.0) < dec.threshold(3, 0.065, 0.6)
    assert dec.threshold(3, 0.03, 0.8) < dec.threshold(3, 0.10, 0.8)


def test_a_loan_that_earns_nothing_is_never_approved():
    assert float(dec.threshold(3, 0.0, 0.8)) == 0.0
    assert not dec.cost_rule([0.001], [3], margin=0.0).any()


def test_default_net_margin_is_fee_minus_funding_cost():
    assert dec.NET_MARGIN == pytest.approx(config.MONTHLY_FEE_RATE - config.MONTHLY_FUNDING_COST)


# --- profit -----------------------------------------------------------------------------------


def test_realised_profit_for_each_outcome():
    profit = dec.realised_profit(
        y=[0, 1, 0, 1],
        approve=[True, True, False, False],
        amount=[10_000, 10_000, 10_000, 10_000],
        term_months=[3, 3, 3, 3],
        margin=0.05,
        lgd=0.8,
    )
    np.testing.assert_allclose(profit, [1_500, -8_000, 0, 0])  # repaid, defaulted, declined x2


def test_expected_profit_matches_realised_on_average_when_p_is_true():
    rng = np.random.default_rng(0)
    n = 200_000
    p = rng.uniform(0.02, 0.5, n)
    y = (rng.random(n) < p).astype(int)
    amount, term = np.full(n, 10_000), rng.choice([1, 3, 6], n)
    approve = dec.cost_rule(p, term)
    realised = dec.realised_profit(y, approve, amount, term).sum()
    expected = dec.expected_profit(p, approve, amount, term).sum()
    assert realised == pytest.approx(expected, rel=0.02)


def test_cost_rule_beats_any_single_threshold_when_p_is_true():
    rng = np.random.default_rng(1)
    n = 100_000
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    amount, term = rng.integers(2, 40, n) * 500, rng.choice([1, 3, 6], n)
    rule = dec.realised_profit(y, dec.cost_rule(p, term), amount, term).sum()
    single = dec.best_single_threshold(y, p, amount, term)
    best_single = dec.realised_profit(y, p < single, amount, term).sum()
    assert rule > best_single


def test_scale_check_peaks_near_one_when_p_is_true():
    rng = np.random.default_rng(2)
    n = 100_000
    p = rng.uniform(0.02, 0.6, n)
    y = (rng.random(n) < p).astype(int)
    amount, term = np.full(n, 10_000), rng.choice([1, 3, 6], n)
    scale = dec.scale_check(y, p, amount, term)
    assert 0.9 <= scale["profit_kes"].idxmax() <= 1.1


def test_policy_summary_counts():
    summary = dec.policy_summary(
        "test",
        y=np.array([1, 1, 0, 0]),
        approve=np.array([False, True, True, False]),
        amount=np.full(4, 1_000),
        term_months=np.full(4, 1),
    )
    assert summary["approval_rate"] == 0.5
    assert summary["defaulters_caught"] == 0.5
    assert summary["good_borrowers_declined"] == 0.5
    assert summary["default_rate_approved"] == 0.5


def test_profit_difference_of_a_policy_with_itself_is_zero():
    rng = np.random.default_rng(0)
    y, approve = rng.integers(0, 2, 500), rng.random(500) < 0.5
    d = dec.profit_difference(y, approve, approve, np.full(500, 5_000), np.full(500, 3), 200)
    assert d == {"difference": 0.0, "low": 0.0, "high": 0.0}


def test_sensitivity_covers_the_whole_grid():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.5, 1_000)
    y = (rng.random(1_000) < p).astype(int)
    table = dec.sensitivity(
        y,
        p,
        np.full(1_000, 5_000),
        rng.choice([1, 3, 6], 1_000),
        margins=(0.03, 0.065),
        lgds=(0.6, 1.0),
    )
    assert len(table) == 4
    assert {"threshold_1m", "threshold_3m", "threshold_6m", "approval_rate"} <= set(table)


# --- saved policy and single decisions ----------------------------------------------------------


@pytest.fixture
def policy(tmp_path):
    path = dec.save_policy(tmp_path / "decision_policy.json")
    return dec.load_policy(path)


def test_saved_policy_round_trips_with_integer_terms(policy, tmp_path):
    raw = json.loads((tmp_path / "decision_policy.json").read_text(encoding="utf-8"))
    assert set(raw["thresholds_by_term_months"]) == {"1", "3", "6"}  # JSON keys are text
    assert policy["thresholds_by_term_months"] == dec.thresholds_by_term()  # loaded as ints


def test_decide_one_applicant(policy):
    t3 = policy["thresholds_by_term_months"][3]
    assert dec.decide(t3 - 0.001, 3, policy) == "approve"
    assert dec.decide(t3, 3, policy) == "decline"  # exactly at the threshold: decline


@pytest.mark.parametrize("p, term", [(0.1, 2), (1.5, 3), (-0.1, 3)])
def test_decide_rejects_bad_input(policy, p, term):
    with pytest.raises(ValueError):
        dec.decide(p, term, policy)


def test_missing_policy_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="decision_policy.py"):
        dec.load_policy(tmp_path / "nope.json")
