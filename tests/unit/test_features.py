"""Known-answer tests for the statement features (steps 2.3 and 2.4).

Each test builds a tiny statement where the right answer can be worked out on paper.
"""

import numpy as np
import pandas as pd
import pytest

from mkopoguard.features.columns import assert_no_leakage
from mkopoguard.features.transactions import FEATURES, statement_features

APPLICATION_DATE = pd.Timestamp("2025-07-01")


def make_applicants(*ids, loan=6_000):
    return pd.DataFrame(
        {
            "applicant_id": list(ids),
            "application_date": APPLICATION_DATE,
            "loan_amount_kes": loan,
        }
    )


def make_statement(rows, applicant_id="A-00001"):
    """rows: (days before the application date, type, amount, balance after)."""
    directions = {"income_in": "in", "p2p_in": "in", "overdraft_borrow": "in"}
    return pd.DataFrame(
        {
            "applicant_id": applicant_id,
            "timestamp": [APPLICATION_DATE - pd.Timedelta(days=d) for d, *_ in rows],
            "type": [r[1] for r in rows],
            "direction": [directions.get(r[1], "out") for r in rows],
            "amount_kes": [r[2] for r in rows],
            "balance_after": [r[3] for r in rows],
        }
    )


def features_for(rows, **applicant_kwargs):
    applicants = make_applicants("A-00001", **applicant_kwargs)
    return statement_features(make_statement(rows), applicants).loc["A-00001"]


STEADY_INCOME = [(5 + 30 * m, "income_in", 3_000, 3_000) for m in range(6)]


# --- income -------------------------------------------------------------------


def test_steady_income():
    f = features_for(STEADY_INCOME)
    assert f["stmt_avg_monthly_inflow"] == pytest.approx(3_000)
    assert f["stmt_median_monthly_inflow"] == pytest.approx(3_000)
    assert f["stmt_inflow_volatility"] == pytest.approx(0)
    assert f["stmt_income_payments_per_month"] == pytest.approx(1)
    assert f["stmt_days_since_last_income"] == pytest.approx(5)


def test_all_income_in_one_month_is_volatile():
    # 18,000 in one month, nothing in the other five: average 3,000, volatility sqrt(5)
    f = features_for([(10, "income_in", 18_000, 18_000)])
    assert f["stmt_avg_monthly_inflow"] == pytest.approx(3_000)
    assert f["stmt_median_monthly_inflow"] == pytest.approx(0)
    assert f["stmt_inflow_volatility"] == pytest.approx(np.sqrt(5))


def test_loan_to_inflow():
    assert features_for(STEADY_INCOME, loan=6_000)["stmt_loan_to_inflow"] == pytest.approx(2)


# --- spending -----------------------------------------------------------------


def test_spending_shares():
    # Outflows: betting 600 + airtime 400 + cash 1,000 = 2,000
    rows = STEADY_INCOME + [
        (10, "betting", 600, 1_000),
        (11, "airtime", 400, 600),
        (12, "cash_out", 1_000, 2_000),
        (13, "p2p_in", 2_000, 4_000),
    ]
    f = features_for(rows)
    assert f["stmt_betting_share"] == pytest.approx(0.3)
    assert f["stmt_airtime_share"] == pytest.approx(0.2)
    assert f["stmt_cash_out_share"] == pytest.approx(0.5)
    # p2p_in 2,000 vs income 18,000
    assert f["stmt_p2p_in_share"] == pytest.approx(0.1)


def test_paybill_months_counts_distinct_months():
    rows = STEADY_INCOME + [
        (2, "paybill", 300, 100),
        (20, "paybill", 300, 100),  # same month as the one above
        (70, "paybill", 300, 100),
    ]
    assert features_for(rows)["stmt_paybill_months"] == 2


# --- overdrafts ---------------------------------------------------------------


def test_overdraft_features():
    rows = STEADY_INCOME + [
        (50, "overdraft_borrow", 1_000, 1_000),
        (40, "overdraft_borrow", 1_000, 1_000),
        (30, "overdraft_repay", 500, 500),
    ]
    f = features_for(rows)
    assert f["stmt_overdraft_borrow_count"] == 2
    assert f["stmt_overdraft_repay_ratio"] == pytest.approx(0.25)
    # 1,500 still owed against 3,000 average monthly income
    assert f["stmt_overdraft_outstanding_months"] == pytest.approx(0.5)


def test_never_borrowing_means_nothing_outstanding():
    f = features_for(STEADY_INCOME)
    assert f["stmt_overdraft_borrow_count"] == 0
    assert f["stmt_overdraft_repay_ratio"] == 1.0
    assert f["stmt_overdraft_outstanding_months"] == 0.0


# --- balances and activity ----------------------------------------------------


def test_balance_features():
    # Last transaction leaves 1,500 = half a month; 2 of 8 rows are under KES 100
    rows = STEADY_INCOME + [(3, "airtime", 50, 50), (1, "cash_out", 10, 1_500)]
    rows[0] = (5, "income_in", 3_000, 20)
    f = features_for(rows)
    assert f["stmt_end_balance_months"] == pytest.approx(0.5)
    assert f["stmt_low_balance_share"] == pytest.approx(2 / 8)
    assert f["stmt_transactions_per_month"] == pytest.approx(8 / 6)


# --- safety -------------------------------------------------------------------


def test_transactions_on_or_after_the_application_date_are_ignored():
    clean = features_for(STEADY_INCOME)
    leaked = features_for(STEADY_INCOME + [(0, "betting", 99_999, 0), (-3, "betting", 99_999, 0)])
    pd.testing.assert_series_equal(clean, leaked)


def test_transactions_older_than_the_window_are_ignored():
    clean = features_for(STEADY_INCOME)
    old = features_for(STEADY_INCOME + [(181, "betting", 99_999, 0)])
    pd.testing.assert_series_equal(clean, old)


def test_thin_file_applicants_get_a_row_of_missing_values():
    applicants = make_applicants("A-00001", "A-00002")
    out = statement_features(make_statement(STEADY_INCOME), applicants)
    assert list(out.index) == ["A-00001", "A-00002"]
    assert out.loc["A-00002"].isna().all()
    assert out.loc["A-00001"].notna().all()


def test_each_applicant_is_computed_independently():
    other = make_statement([(7, "income_in", 50_000, 50_000)], applicant_id="A-00002")
    applicants = make_applicants("A-00001", "A-00002")
    together = statement_features(pd.concat([make_statement(STEADY_INCOME), other]), applicants)
    alone = features_for(STEADY_INCOME)
    pd.testing.assert_series_equal(together.loc["A-00001"], alone)


def test_feature_names_are_documented_and_allowed():
    out = statement_features(make_statement(STEADY_INCOME), make_applicants("A-00001"))
    assert list(out.columns) == FEATURES
    assert all(name.startswith("stmt_") for name in FEATURES)
    assert_no_leakage(FEATURES)


def test_income_but_no_spending_gives_zero_shares_not_missing():
    f = features_for(STEADY_INCOME)
    for name in ["stmt_betting_share", "stmt_airtime_share", "stmt_cash_out_share"]:
        assert f[name] == 0
