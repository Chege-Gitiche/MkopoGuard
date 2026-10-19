"""Step 7.4: property tests on the statement features and the statement reader (Hypothesis).

The known-answer tests (test_features.py) check a few statements worked out on paper. These
state rules that must hold for EVERY valid statement, and Hypothesis generates hundreds of
statements to try to break them, shrinking any failure to the smallest statement that shows it.
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from mkopoguard.api import statement as reader
from mkopoguard.features.transactions import statement_features

APPLIED = pd.Timestamp("2025-07-01")
TYPES = sorted(reader.DIRECTION)
# Features that divide by the average monthly income: missing when there was no income
PER_INCOME = {
    "stmt_inflow_volatility",
    "stmt_days_since_last_income",
    "stmt_end_balance_months",
    "stmt_loan_to_inflow",
    "stmt_overdraft_outstanding_months",
}
SHARES = [
    "stmt_p2p_in_share",
    "stmt_betting_share",
    "stmt_airtime_share",
    "stmt_cash_out_share",
    "stmt_low_balance_share",
]
COUNTS = [
    "stmt_paybill_months",
    "stmt_overdraft_borrow_count",
    "stmt_income_payments_per_month",
    "stmt_transactions_per_month",
]
SCALE_FREE = [  # the same whatever the currency unit: ratios of money to money
    "stmt_inflow_volatility",
    "stmt_p2p_in_share",
    "stmt_betting_share",
    "stmt_airtime_share",
    "stmt_cash_out_share",
    "stmt_overdraft_repay_ratio",
    "stmt_overdraft_outstanding_months",
    "stmt_end_balance_months",
    "stmt_paybill_months",
    "stmt_overdraft_borrow_count",
    "stmt_transactions_per_month",
]
PROPERTY = settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])


@st.composite
def statements(draw, min_rows=1, max_rows=60, oldest=180.0, newest=0.01, income=None):
    """A valid statement: (transactions, loan amount). Balances add up and never go negative."""
    n = draw(st.integers(min_rows, max_rows))
    days = draw(st.lists(st.floats(newest, oldest), min_size=n, max_size=n))
    types = draw(st.lists(st.sampled_from(TYPES), min_size=n, max_size=n))
    if income is True:
        types[0] = "income_in"
    elif income is False:
        types = [t if t != "income_in" else "p2p_in" for t in types]
    amounts = draw(st.lists(st.integers(1, 50_000), min_size=n, max_size=n))
    balance = draw(st.integers(0, 100_000))
    rows = []
    for day, kind, amount in sorted(zip(days, types, amounts, strict=True), reverse=True):
        direction = reader.DIRECTION[kind]
        if direction == "out":
            amount = min(amount, balance) or 1
            if amount > balance:  # an empty account can't pay; receive the money instead
                kind, direction = "p2p_in", "in"
        balance += amount if direction == "in" else -amount
        rows.append((APPLIED - pd.Timedelta(days=day), kind, direction, amount, balance))
    frame = pd.DataFrame(rows, columns=list(reader.COLUMNS))
    frame.insert(0, "applicant_id", "A")
    return frame, draw(st.integers(1_000, 100_000))


def features(transactions: pd.DataFrame, loan: int) -> pd.Series:
    applicant = pd.DataFrame(
        {"applicant_id": ["A"], "application_date": [APPLIED], "loan_amount_kes": [loan]}
    )
    return statement_features(transactions, applicant).loc["A"]


def same(a: pd.Series, b: pd.Series) -> bool:
    return np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-9, equal_nan=True)


# --- the features ------------------------------------------------------------------------


@PROPERTY
@given(statements(income=True))
def test_with_any_income_no_feature_is_ever_missing(example):
    f = features(*example)
    assert f.notna().all(), f[f.isna()].index.tolist()
    assert np.isfinite(f.to_numpy(float)).all()


@PROPERTY
@given(statements(income=False))
def test_without_income_only_features_divided_by_income_can_be_missing(example):
    f = features(*example)
    assert set(f[f.isna()].index) <= PER_INCOME


@PROPERTY
@given(statements())
def test_shares_stay_between_0_and_1_and_counts_are_never_negative(example):
    f = features(*example)
    assert ((f[SHARES] >= 0) & (f[SHARES] <= 1)).all(), f[SHARES]
    assert (f[COUNTS] >= 0).all()
    assert f["stmt_paybill_months"] <= 6
    assert f["stmt_overdraft_repay_ratio"] >= 0
    assert f["stmt_transactions_per_month"] == pytest.approx(len(example[0]) / 6)


@PROPERTY
@given(statements(), st.randoms(use_true_random=False))
def test_the_order_of_the_rows_never_matters(example, random):
    """Features follow the time of each transaction, not its place in the file.

    Hypothesis first broke this with two transactions in the same second: then the closing
    balance depends on which came last. The reader only accepts files whose balances add up in
    file order, so for a tie the file order is the truth (next test).
    """
    transactions, loan = example
    transactions = transactions.drop_duplicates("timestamp")
    shuffled = transactions.sample(frac=1, random_state=random.randint(0, 2**31))
    assert same(features(transactions, loan), features(shuffled.reset_index(drop=True), loan))


def test_two_transactions_in_the_same_second_keep_their_file_order():
    second = APPLIED - pd.Timedelta(days=3)
    rows = [
        (APPLIED - pd.Timedelta(days=20), "income_in", "in", 6_000, 6_000),
        (second, "airtime", "out", 100, 5_900),
        (second, "betting", "out", 5_000, 900),  # the last row: the closing balance is 900
    ]
    transactions = pd.DataFrame(rows, columns=list(reader.COLUMNS)).assign(applicant_id="A")
    f = features(transactions, 5_000)
    assert f["stmt_end_balance_months"] == pytest.approx(900 / f["stmt_avg_monthly_inflow"])


@PROPERTY
@given(statements(), statements(min_rows=1, max_rows=10, oldest=0.0, newest=-30.0))
def test_transactions_on_or_after_the_application_day_change_nothing(example, later):
    transactions, loan = example
    leaked = pd.concat([transactions, later[0]], ignore_index=True)
    assert same(features(transactions, loan), features(leaked, loan))


@PROPERTY
@given(statements(), st.integers(2, 20))
def test_ratios_do_not_depend_on_the_size_of_the_amounts(example, k):
    transactions, loan = example
    scaled = transactions.assign(
        amount_kes=transactions["amount_kes"] * k, balance_after=transactions["balance_after"] * k
    )
    before, after = features(transactions, loan), features(scaled, loan)
    assert same(before[SCALE_FREE], after[SCALE_FREE])
    assert after["stmt_avg_monthly_inflow"] == pytest.approx(k * before["stmt_avg_monthly_inflow"])


@PROPERTY
@given(
    st.integers(1, 50_000),
    st.lists(st.floats(0.01, 29.99), min_size=6, max_size=6),
)
def test_the_same_income_every_month_has_zero_volatility(amount, offsets):
    """The guide's example, for any amount and any day within each month."""
    rows = [
        (APPLIED - pd.Timedelta(days=30 * month + offset), "income_in", "in", amount,
         amount * (6 - month))
        for month, offset in enumerate(offsets)
    ]  # fmt: skip
    transactions = pd.DataFrame(rows, columns=list(reader.COLUMNS)).assign(applicant_id="A")
    f = features(transactions, 5_000)
    assert f["stmt_inflow_volatility"] == pytest.approx(0, abs=1e-12)
    assert f["stmt_avg_monthly_inflow"] == pytest.approx(amount)
    assert f["stmt_income_payments_per_month"] == pytest.approx(1)


# --- the statement reader -------------------------------------------------------------------


def as_csv(transactions: pd.DataFrame) -> bytes:
    rows = transactions[list(reader.COLUMNS)].copy()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return rows.to_csv(index=False, lineterminator="\n").encode("utf-8")


uploadable = statements(  # within the reader's rules: 6 months, recent, at least 10 rows
    min_rows=reader.MIN_ROWS, max_rows=80, oldest=179.0, newest=1.0
).filter(lambda ex: (APPLIED - ex[0]["timestamp"].min()).days >= reader.MIN_COVERAGE_DAYS)


@PROPERTY
@given(uploadable)
def test_a_valid_statement_uploads_and_gives_the_same_features_as_training(example):
    transactions, loan = example
    # The CSV keeps whole seconds; build the expected features from exactly what is uploaded
    transactions = transactions.assign(timestamp=transactions["timestamp"].dt.floor("s"))
    parsed = reader.parse_statement(as_csv(transactions), APPLIED.date())
    assert parsed.warnings == []
    assert same(
        features(parsed.transactions.assign(applicant_id="A"), loan), features(transactions, loan)
    )


@settings(max_examples=300, deadline=None)
@given(st.binary(max_size=3_000))
def test_any_bytes_are_either_read_or_refused_with_reasons(data):
    try:
        reader.parse_statement(data, APPLIED.date())
    except reader.StatementError as error:
        assert error.problems and all(where and what for where, what in error.problems)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.lists(
            st.text(alphabet=st.characters(codec="utf-8"), max_size=12), min_size=5, max_size=5
        ),
        min_size=1,
        max_size=30,
    )
)
def test_any_five_column_text_is_either_read_or_refused_with_reasons(cells):
    lines = [",".join(reader.COLUMNS)] + [",".join(row) for row in cells]
    try:
        reader.parse_statement("\n".join(lines).encode("utf-8"), APPLIED.date())
    except reader.StatementError as error:
        assert error.problems
