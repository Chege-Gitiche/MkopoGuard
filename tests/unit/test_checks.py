import pandas as pd
import pytest

from mkopoguard.data.checks import LIMITS, assert_plausible, failed_checks, plausibility_report


def middle_report():
    """A report where every value sits in the middle of its allowed range."""
    return pd.Series({name: (low + high) / 2 for name, (low, high) in LIMITS.items()})


def test_a_report_inside_every_limit_passes():
    assert failed_checks(middle_report()) == {}


@pytest.mark.parametrize("name", list(LIMITS))
def test_each_limit_catches_a_value_above_it(name):
    report = middle_report()
    report[name] = LIMITS[name][1] + 1
    assert list(failed_checks(report)) == [name]


def make_tiny_dataset():
    """Two applicants with hand-checkable statements (income KES 10,000 each)."""
    applicants = pd.DataFrame(
        {
            "applicant_id": ["A-00001", "A-00002"],
            "country_code": ["KEN", "UGA"],
            "defaulted": [0, 1],
            "loan_amount_kes": [10_000, 12_000],
            "hidden_monthly_income_kes": [10_000.0, 10_000.0],
        }
    )
    t0 = pd.Timestamp("2025-01-01 08:00")
    rows = [
        # A-00001: paid, withdraws, ends with 5,000 (0.5 months)
        ("A-00001", t0, "income_in", 10_000, 10_000),
        ("A-00001", t0, "cash_out", 5_000, 5_000),
        # A-00002: borrows 3,000 then 2,000 (owes 5,000 = 0.5 months), repays 1,000
        ("A-00002", t0, "income_in", 1_000, 1_000),
        ("A-00002", t0, "overdraft_borrow", 3_000, 4_000),
        ("A-00002", t0 + pd.Timedelta(days=1), "overdraft_borrow", 2_000, 6_000),
        ("A-00002", t0 + pd.Timedelta(days=2), "overdraft_repay", 1_000, 5_000),
    ]
    transactions = pd.DataFrame(
        rows, columns=["applicant_id", "timestamp", "type", "amount_kes", "balance_after"]
    )
    return applicants, transactions


def test_report_values_on_a_hand_built_dataset():
    report = plausibility_report(*make_tiny_dataset())
    assert report["default_rate"] == pytest.approx(0.5)
    assert report["default_rate_kenya"] == pytest.approx(0.0)
    assert report["max_transactions_in_one_day"] == 2
    assert report["median_final_balance_months"] == pytest.approx(0.5)
    assert report["share_using_overdraft"] == pytest.approx(0.5)
    assert report["max_overdraft_owed_months"] == pytest.approx(0.5)
    assert report["share_without_income"] == 0


def test_assert_plausible_names_the_failing_check():
    applicants, transactions = make_tiny_dataset()
    with pytest.raises(ValueError, match="default_rate"):
        assert_plausible(applicants, transactions)  # 50% default rate is far too high
