"""Step 4.3: the behaviour-test machinery, checked with toy scorers whose behaviour we choose."""

import numpy as np
import pandas as pd
import pytest

from mkopoguard.explain import behaviour as bh
from mkopoguard.explain.design import DRIVERS
from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC


@pytest.fixture
def applicants() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    has = (np.arange(n) % 3 != 0).astype(int)  # 40 statement holders, 20 thin-file
    users = has.astype(bool) & (np.arange(n) % 2 == 0)
    stmt = {
        "stmt_betting_share": rng.uniform(0, 0.2, n),
        "stmt_overdraft_borrow_count": np.where(users, rng.integers(1, 60, n), 0).astype(float),
        "stmt_overdraft_repay_ratio": np.where(users, rng.uniform(0.3, 1, n), 1.0),
        "stmt_overdraft_outstanding_months": np.where(users, rng.uniform(0, 1, n), 0.0),
        "stmt_low_balance_share": rng.uniform(0, 0.5, n),
        "stmt_end_balance_months": rng.uniform(0, 2, n),
        "stmt_paybill_months": rng.integers(0, 7, n).astype(float),
        "stmt_avg_monthly_inflow": rng.uniform(2_000, 50_000, n),
        "stmt_median_monthly_inflow": rng.uniform(2_000, 50_000, n),
        "stmt_loan_to_inflow": rng.uniform(0.3, 3, n),
        "stmt_inflow_volatility": rng.uniform(0, 1, n),
        "stmt_income_payments_per_month": rng.uniform(0.3, 15, n),
        "stmt_days_since_last_income": rng.uniform(0, 60, n),
        "stmt_airtime_share": rng.uniform(0, 0.3, n),
        "stmt_cash_out_share": rng.uniform(0.3, 0.9, n),
        "stmt_p2p_in_share": rng.uniform(0, 0.2, n),
        "stmt_transactions_per_month": rng.uniform(5, 80, n),
    }
    frame = pd.DataFrame(stmt)
    frame.loc[has == 0, :] = np.nan  # thin-file: no statement values
    return frame.assign(
        has_statement=has,
        loan_amount_kes=rng.uniform(1_000, 50_000, n),
        term_months=rng.choice([1, 3, 6], n),
        income_quintile=rng.integers(1, 6, n),
        age=rng.integers(18, 70, n),
        in_workforce=rng.integers(0, 2, n),
        saved_past_year=rng.integers(0, 2, n),
        saved_in_savings_club=rng.integers(0, 2, n),
        borrowed_for_medical=rng.integers(0, 2, n),
        borrowed_past_year=rng.integers(0, 2, n),
        borrowed_from_fin_institution=rng.integers(0, 2, n),
        borrowed_from_family_friends=rng.integers(0, 2, n),
        emergency_fund_source=rng.choice(["could_not", "savings", "work"], n),
        worried_about_bills=rng.choice(["very", "somewhat", "not_at_all"], n),
        education_level=rng.choice(["primary_or_less", "secondary", "tertiary"], n),
    )


# Each input pushes risk the way the simulator says (non-drivers: no effect)
DESIGN = {
    "stmt_betting_share": 8.0,
    "stmt_overdraft_borrow_count": 0.02,
    "stmt_overdraft_outstanding_months": 1.0,
    "stmt_overdraft_repay_ratio": -1.0,
    "stmt_low_balance_share": 1.0,
    "stmt_end_balance_months": -0.3,
    "stmt_paybill_months": -0.1,
    "stmt_avg_monthly_inflow": -0.3,  # on log money
    "stmt_median_monthly_inflow": -0.3,
    "stmt_loan_to_inflow": 0.4,
    "stmt_inflow_volatility": 0.5,
    "loan_amount_kes": 0.3,
    "income_quintile": -0.2,
    "in_workforce": -0.3,
    "saved_past_year": -0.3,
    "saved_in_savings_club": -0.4,
    "borrowed_for_medical": 0.5,
}
LOGGED = {"stmt_avg_monthly_inflow", "stmt_median_monthly_inflow", "loan_amount_kes"}


def design_scorer(flip=(), cap=None):
    """A model that behaves as designed; `flip` reverses inputs, `cap` clips betting."""

    def predict(frame: pd.DataFrame) -> np.ndarray:
        z = np.full(len(frame), -2.0)
        for col, w in DESIGN.items():
            x = frame[col].astype(float).fillna(0).to_numpy()
            if col in LOGGED:
                x = np.log1p(x)
            if cap and col == "stmt_betting_share":
                x = np.minimum(x, cap)
            z += (-w if col in flip else w) * x
        z += 1.0 * (frame["emergency_fund_source"] == "could_not").to_numpy()
        z += 0.8 * (frame["worried_about_bills"] == "very").to_numpy()
        z += 0.2 * frame["age"].between(18, 24).to_numpy()
        return 1 / (1 + np.exp(-z))

    return predict


def test_a_model_that_behaves_as_designed_passes_every_check(applicants):
    result = bh.run_all(design_scorer(), applicants, bh.CHECKS)
    assert result["passed"].all(), result.loc[~result["passed"].astype(bool)].index.tolist()


def test_the_known_flips_are_the_models_not_the_checks(applicants):
    """The checks the champion fails pass for a model with the designed directions."""
    result = bh.run_all(design_scorer(), applicants, [bh.BY_NAME[n] for n in bh.KNOWN_FLIPS])
    assert result["passed"].all()


def test_a_wrong_direction_is_caught_for_every_affected_applicant(applicants):
    result = bh.run_check(
        design_scorer(flip={"stmt_betting_share"}), applicants, bh.BY_NAME["more_betting"]
    )
    assert result["passed"] is False
    assert result["share_wrong"] == 1.0
    assert result["worst_wrong_pts"] > 0


def test_a_model_that_ignores_the_input_fails_for_not_responding(applicants):
    flat = lambda frame: np.full(len(frame), 0.2)  # noqa: E731
    result = bh.run_check(flat, applicants, bh.BY_NAME["more_betting"])
    assert result["share_wrong"] == 0.0
    assert result["responds"] is False and result["passed"] is False


def test_applicants_already_at_a_clipped_edge_may_stay_unchanged(applicants):
    result = bh.run_check(design_scorer(cap=0.1), applicants, bh.BY_NAME["more_betting"])
    assert result["share_up"] < 1.0  # some were at the cap
    assert result["passed"] is True


def test_checks_apply_only_to_the_applicants_they_are_for(applicants):
    holders = int((applicants["has_statement"] == 1).sum())
    users = int((applicants["stmt_overdraft_borrow_count"] > 0).sum())
    predict = design_scorer()
    assert bh.run_check(predict, applicants, bh.BY_NAME["more_betting"])["n"] == holders
    assert bh.run_check(predict, applicants, bh.BY_NAME["repays_less_overdraft"])["n"] == users
    assert bh.run_check(predict, applicants, bh.BY_NAME["bigger_loan"])["n"] == len(applicants)
    starts = bh.run_check(predict, applicants, bh.BY_NAME["starts_using_overdraft"])["n"]
    assert starts == holders - users


def test_pair_checks_compare_the_same_applicant_at_both_values(applicants):
    check = bh.BY_NAME["saved"]
    before, after = check.before(applicants), check.after(applicants)
    assert (before["saved_past_year"] == 0).all() and (after["saved_past_year"] == 1).all()
    others = applicants.columns.drop("saved_past_year")
    pd.testing.assert_frame_equal(before[others], applicants[others])


def test_reports_measure_without_passing_or_failing(applicants):
    result = bh.run_check(design_scorer(), applicants, bh.BY_NAME["more_airtime"])
    assert result["passed"] is None and result["mean_change_pts"] == 0.0


def test_no_eligible_applicants_is_not_a_pass(applicants):
    thin = applicants[applicants["has_statement"] == 0]
    result = bh.run_check(design_scorer(), thin, bh.BY_NAME["more_betting"])
    assert result["n"] == 0 and result["passed"] is None


def test_wrong_way_decisions_are_counted(applicants):
    check = bh.BY_NAME["saved"]
    flipped = design_scorer(flip={"saved_past_year"})
    cut = np.median(flipped(check.before(applicants)))  # a threshold the change crosses

    def decide(frame, p):
        return np.where(p < cut, "approve", "decline")

    wrong = bh.run_check(flipped, applicants, check, decide)
    right = bh.run_check(design_scorer(), applicants, check, decide)
    assert wrong["share_decisions_wrong"] > 0
    assert right["share_decisions_wrong"] == 0


def test_shares_stay_between_zero_and_one(applicants):
    for name in ("more_betting", "repays_less_overdraft", "more_low_balance_days"):
        changed = bh.BY_NAME[name].after(applicants)
        for col in bh.BY_NAME[name].columns:
            values = changed[col].dropna()
            assert values.between(0, 1).all()


def test_probe_adds_an_overdraft_user_copy_of_every_non_user(applicants):
    probe = bh.probe_applicants(applicants)
    non_users = int(bh.WHO["overdraft_non_users"](applicants).sum())
    assert len(probe) == len(applicants) + non_users
    copies = probe.iloc[len(applicants) :]
    for col, value in bh.TYPICAL_OVERDRAFT_USER.items():
        assert (copies[col] == value).all()
    pd.testing.assert_frame_equal(probe.iloc[: len(applicants)], applicants.reset_index(drop=True))


def test_the_check_table_is_well_formed():
    names = [c.name for c in bh.ALL_CHECKS]
    assert len(names) == len(set(names))
    inputs = set(NUMERIC + BINARY + CATEGORICAL)
    for check in bh.ALL_CHECKS:
        assert check.expect in (bh.HIGHER, bh.LOWER, bh.NONE)
        assert check.who in bh.WHO
        assert check.columns and set(check.columns) <= inputs
        assert (check.kind == "report") == (check.expect == bh.NONE)
    assert set(bh.KNOWN_FLIPS) <= {c.name for c in bh.CHECKS}


def test_every_driver_has_an_expectation_or_a_reason_it_has_none():
    tested = {col for c in bh.CHECKS for col in c.columns}
    reported = {col for c in bh.REPORTS for col in c.columns}
    for driver in DRIVERS:
        assert driver in tested or (driver in bh.NO_DIRECTION and driver in reported), driver
    for non_driver in reported - set(bh.NO_DIRECTION):
        assert non_driver not in DRIVERS  # reports are for inputs without an expectation
