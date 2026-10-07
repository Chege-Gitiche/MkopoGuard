import numpy as np
import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.defaults import add_defaults, solve_intercept


def make_applicants(n=2_000, **overrides):
    """n identical, average applicants; override columns to create a riskier group."""
    data = {
        "applicant_id": [f"A-{i:05d}" for i in range(1, n + 1)],
        "country_code": "KEN",
        "is_female": 0,
        "age": 35,
        "in_workforce": 1,
        "saved_past_year": 1,
        "saved_in_savings_club": 0,
        "hidden_loan_to_income": 1.0,
        "hidden_income_pattern": "salaried",
        "hidden_stress": 0.0,
        "hidden_overdraft_propensity": 0.3,
        "hidden_betting_share": 0.0,
        "hidden_discipline": 0.0,
    }
    data.update(overrides)
    return pd.DataFrame(data)


def mean_probability(**overrides):
    """Average true default probability for a group that differs from the baseline."""
    group = make_applicants(**overrides)
    both = pd.concat([make_applicants(), group], ignore_index=True)
    probability = add_defaults(both)["hidden_default_probability"]
    return probability[2_000:].mean(), probability[:2_000].mean()


def test_same_seed_gives_identical_outcomes():
    applicants = make_applicants(500, hidden_discipline=np.linspace(-2, 2, 500))
    pd.testing.assert_frame_equal(
        add_defaults(applicants, seed=1), add_defaults(applicants, seed=1)
    )


def test_intercept_hits_the_target_rate_exactly():
    base = np.random.default_rng(0).normal(0, 1, 10_000)
    b0 = solve_intercept(base, 0.2)
    assert (1 / (1 + np.exp(-(b0 + base)))).mean() == pytest.approx(0.2, abs=1e-6)


def test_overall_default_rate_is_close_to_target():
    applicants = make_applicants(
        10_000, hidden_discipline=np.random.default_rng(0).normal(0, 1, 10_000)
    )
    rate = add_defaults(applicants)["defaulted"].mean()
    assert rate == pytest.approx(config.DEFAULT_TARGET_RATE, abs=0.015)


@pytest.mark.parametrize(
    "riskier",
    [
        {"hidden_loan_to_income": 2.5},
        {"hidden_income_pattern": "irregular"},
        {"hidden_stress": 1.5},
        {"hidden_overdraft_propensity": 0.9},
        {"hidden_betting_share": 0.25},
        {"hidden_discipline": -1.5},
        {"saved_past_year": 0},
        {"in_workforce": 0},
        {"age": 20},
    ],
)
def test_each_driver_raises_risk_in_the_documented_direction(riskier):
    group, baseline = mean_probability(**riskier)
    assert group > baseline


def test_savings_club_lowers_risk():
    group, baseline = mean_probability(saved_in_savings_club=1)
    assert group < baseline


@pytest.mark.parametrize("change", [{"is_female": 1}, {"country_code": "GHA"}])
def test_gender_and_country_never_change_the_outcome(change):
    applicants = make_applicants(500, hidden_discipline=np.linspace(-2, 2, 500))
    changed = applicants.assign(**change)
    a, b = add_defaults(applicants), add_defaults(changed)
    pd.testing.assert_series_equal(a["hidden_default_probability"], b["hidden_default_probability"])
    pd.testing.assert_series_equal(a["defaulted"], b["defaulted"])


def test_label_is_zero_or_one_and_probability_is_hidden():
    out = add_defaults(make_applicants(100))
    assert set(out["defaulted"]) <= {0, 1}
    assert "hidden_default_probability" in out.columns


def test_input_is_not_modified():
    applicants = make_applicants(50)
    before = applicants.copy()
    add_defaults(applicants)
    pd.testing.assert_frame_equal(applicants, before)
