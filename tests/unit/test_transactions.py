import numpy as np
import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.transactions import (
    COLUMNS,
    add_behaviour,
    generate_transactions,
    statement_for,
)


def make_applicant(number=1, **overrides):
    """One applicant with everything the statement generator needs."""
    start = pd.Timestamp("2025-01-01")
    a = {
        "applicant_id": f"A-{number:05d}",
        "statement_start": start,
        "application_date": start + pd.Timedelta(days=config.STATEMENT_DAYS),
        "has_statement": 1,
        "hidden_monthly_income_kes": 12_000.0,
        "hidden_income_pattern": "irregular",
        "received_domestic_remittance": 1,
        "sent_domestic_remittance": 1,
        "utility_payment_channel": "account",
        "made_digital_merchant_payment": 1,
        "hidden_betting_share": 0.1,
        "hidden_bill_miss_prob": 0.1,
        "hidden_uses_overdraft": 1,
        "hidden_overdraft_repay_share": 0.8,
    }
    a.update(overrides)
    return a


def make_applicants(n, **overrides):
    return pd.DataFrame([make_applicant(i, **overrides) for i in range(1, n + 1)])


# --- hidden behaviour --------------------------------------------------------


def make_traits(discipline, stress):
    return pd.DataFrame({"hidden_discipline": discipline, "hidden_stress": stress})


def test_behaviour_is_reproducible_and_leaves_input_alone():
    traits = make_traits(np.linspace(-2, 2, 100), 0.0)
    before = traits.copy()
    pd.testing.assert_frame_equal(add_behaviour(traits, seed=1), add_behaviour(traits, seed=1))
    pd.testing.assert_frame_equal(traits, before)


def test_low_discipline_bets_more():
    traits = make_traits([-1.5] * 2_000 + [1.5] * 2_000, 0.0)
    share = add_behaviour(traits)["hidden_betting_share"]
    assert share[:2_000].mean() > share[2_000:].mean()


def test_high_stress_means_more_overdraft_use():
    traits = make_traits(0.0, [-1.5] * 2_000 + [1.5] * 2_000)
    uses = add_behaviour(traits)["hidden_uses_overdraft"]
    assert uses[2_000:].mean() > uses[:2_000].mean()


def test_betting_share_is_zero_or_within_range():
    share = add_behaviour(make_traits(np.linspace(-3, 3, 5_000), 0.0))["hidden_betting_share"]
    bettors = share[share > 0]
    assert len(bettors) > 0
    assert bettors.between(0.02, 0.30).all()


# --- statements --------------------------------------------------------------


def test_statement_is_reproducible():
    a = make_applicant()
    pd.testing.assert_frame_equal(statement_for(a, seed=1), statement_for(a, seed=1))


def test_each_statement_is_independent_of_other_applicants():
    applicants = make_applicants(3)
    together = generate_transactions(applicants)
    alone = statement_for(make_applicant(2))
    from_batch = together[together["applicant_id"] == "A-00002"].reset_index(drop=True)
    pd.testing.assert_frame_equal(from_batch.astype(alone.dtypes), alone)


def test_columns_are_as_documented():
    assert list(statement_for(make_applicant()).columns) == COLUMNS


def test_all_transactions_are_inside_the_statement_window():
    a = make_applicant()
    stmt = statement_for(a)
    assert stmt["timestamp"].min() >= a["statement_start"]
    assert stmt["timestamp"].max() < a["application_date"]


@pytest.mark.parametrize("pattern", ["salaried", "seasonal", "irregular"])
def test_balance_never_goes_negative(pattern):
    stmt = statement_for(make_applicant(hidden_income_pattern=pattern, hidden_betting_share=0.3))
    assert (stmt["balance_after"] >= 0).all()


def test_balances_add_up_row_by_row():
    stmt = statement_for(make_applicant())
    signed = np.where(stmt["direction"] == "in", stmt["amount_kes"], -stmt["amount_kes"])
    change = stmt["balance_after"].diff().iloc[1:]
    np.testing.assert_array_equal(change.to_numpy(), signed[1:])


def test_salaried_applicant_is_paid_once_a_month():
    stmt = statement_for(make_applicant(hidden_income_pattern="salaried"))
    assert (stmt["type"] == "income_in").sum() == config.STATEMENT_DAYS // 30


@pytest.mark.parametrize(
    "override, absent_type",
    [
        ({"received_domestic_remittance": 0}, "p2p_in"),
        ({"sent_domestic_remittance": 0}, "p2p_out"),
        ({"utility_payment_channel": "cash_only"}, "paybill"),
        ({"made_digital_merchant_payment": 0}, "till"),
        ({"hidden_betting_share": 0.0}, "betting"),
    ],
)
def test_transaction_types_follow_survey_answers_and_habits(override, absent_type):
    assert absent_type in set(statement_for(make_applicant()).type)
    assert absent_type not in set(statement_for(make_applicant(**override)).type)


def test_no_overdraft_for_applicants_who_dont_use_it():
    a = make_applicant(hidden_monthly_income_kes=3_000.0, hidden_uses_overdraft=0)
    assert not set(statement_for(a).type) & {"overdraft_borrow", "overdraft_repay"}


def test_never_repays_more_than_was_borrowed():
    a = make_applicant(hidden_monthly_income_kes=4_000.0, hidden_betting_share=0.3)
    stmt = statement_for(a)
    borrowed = stmt.loc[stmt["type"] == "overdraft_borrow", "amount_kes"].sum()
    repaid = stmt.loc[stmt["type"] == "overdraft_repay", "amount_kes"].sum()
    assert borrowed > 0
    assert repaid <= borrowed


def test_only_applicants_with_a_statement_get_transactions():
    applicants = make_applicants(4)
    applicants["has_statement"] = [1, 0, 1, 0]
    ids = set(generate_transactions(applicants)["applicant_id"])
    assert ids == {"A-00001", "A-00003"}


# --- step 2.2: cash withdrawals and the overdraft limit ----------------------


def test_cash_is_withdrawn_only_right_after_money_arrives():
    stmt = statement_for(make_applicant())
    cash_rows = stmt.index[stmt["type"] == "cash_out"]
    assert len(cash_rows) > 0
    row_before = stmt.loc[cash_rows - 1, "type"]
    assert set(row_before) <= {"income_in", "p2p_in", "overdraft_repay"}


def test_balances_stay_realistic_over_six_months():
    """Money must not pile up: the final balance stays below 4 months of income."""
    for number in range(1, 31):
        a = make_applicant(number, hidden_betting_share=0.0)
        final = statement_for(a)["balance_after"].iloc[-1]
        assert final < 4 * a["hidden_monthly_income_kes"]


def test_overdraft_debt_never_exceeds_the_limit():
    for number in range(1, 31):
        a = make_applicant(
            number,
            hidden_monthly_income_kes=4_000.0,
            hidden_betting_share=0.3,
            hidden_overdraft_repay_share=0.1,
        )
        stmt = statement_for(a)
        owed = np.where(
            stmt["type"] == "overdraft_borrow",
            stmt["amount_kes"],
            np.where(stmt["type"] == "overdraft_repay", -stmt["amount_kes"], 0),
        ).cumsum()
        assert owed.max() <= a["hidden_monthly_income_kes"]
