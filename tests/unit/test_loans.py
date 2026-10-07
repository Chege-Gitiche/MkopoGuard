import numpy as np
import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.loans import add_loans, draw_terms


def make_population(incomes):
    """A minimal population: just the columns add_loans needs."""
    return pd.DataFrame(
        {
            "applicant_id": [f"A-{i:05d}" for i in range(1, len(incomes) + 1)],
            "hidden_monthly_income_kes": incomes,
        }
    )


def test_same_seed_gives_identical_loans():
    population = make_population([10_000] * 500)
    pd.testing.assert_frame_equal(add_loans(population, seed=1), add_loans(population, seed=1))


def test_amounts_are_rounded_and_within_limits():
    incomes = np.linspace(300, 200_000, 2_000)  # includes very low and very high earners
    loans = add_loans(make_population(incomes))
    amounts = loans["loan_amount_kes"]
    assert (amounts % config.LOAN_ROUND_TO == 0).all()
    assert amounts.min() >= config.LOAN_MIN
    assert amounts.max() <= config.LOAN_MAX


def test_terms_are_only_allowed_values():
    loans = add_loans(make_population([20_000] * 1_000))
    assert set(loans["term_months"]) <= set(config.TERM_OPTIONS)


def test_typical_loan_is_about_one_month_of_income():
    # At KES 20,000 income, no clipping applies, so the median ratio should be close to 1.0
    loans = add_loans(make_population([20_000] * 5_000))
    assert loans["hidden_loan_to_income"].median() == pytest.approx(1.0, abs=0.05)


def test_higher_earners_get_larger_loans():
    incomes = [5_000] * 1_000 + [15_000] * 1_000 + [40_000] * 1_000
    loans = add_loans(make_population(incomes))
    medians = loans.groupby("hidden_monthly_income_kes")["loan_amount_kes"].median()
    assert medians.is_monotonic_increasing


@pytest.mark.parametrize("amount, band", [(5_000, 0), (30_000, 1), (80_000, 2)])
def test_term_probabilities_match_config(amount, band):
    rng = np.random.default_rng(0)
    terms = draw_terms(pd.Series([amount] * 20_000), rng)
    shares = terms.value_counts(normalize=True).reindex(config.TERM_OPTIONS, fill_value=0)
    expected = config.TERM_PROBS_BY_AMOUNT[band][1]
    assert shares.to_numpy() == pytest.approx(expected, abs=0.02)


def test_loan_to_income_is_amount_over_income():
    loans = add_loans(make_population([8_000, 12_000, 30_000]))
    expected = loans["loan_amount_kes"] / loans["hidden_monthly_income_kes"]
    pd.testing.assert_series_equal(loans["hidden_loan_to_income"], expected, check_names=False)


def test_population_columns_are_not_changed():
    population = make_population([10_000] * 50)
    before = population.copy()
    loans = add_loans(population)
    pd.testing.assert_frame_equal(population, before)
    pd.testing.assert_frame_equal(loans[population.columns], before)
