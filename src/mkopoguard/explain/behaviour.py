"""Step 4.3: behaviour tests. Does the model move risk the way the simulator says it should?

Each check takes real applicants, changes ONE thing about them (or a few things that always
move together), keeps everything else equal, and compares the model's probability of default
before and after. The expected direction comes from how the simulator creates default
(data/defaults.py), so it is a known answer, not an opinion.

  higher  - the change must never LOWER anyone's risk (and must raise it on average)
  lower   - the change must never RAISE anyone's risk (and must lower it on average)
  none    - no expectation (a non-driver, or a driver with no single direction): the effect is
            only measured and reported

"single" checks change one input. "linked" checks change inputs that are computed together
(stmt_loan_to_inflow is the loan divided by the statement's average inflow, so a bigger loan
or a bigger income moves both). An applicant whose input is already at the edge the model
clips to sees no change; that is allowed.

KNOWN_FLIPS lists checks the champion (v1.0.0) is known to fail, with the reason found when
the failure was investigated. The tests mark them as expected failures, so they are visible
and a model that fixes them is noticed (the test then "unexpectedly passes").
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HIGHER, LOWER, NONE = "higher", "lower", "none"
TOLERANCE = 1e-9  # changes smaller than this are rounding, not a movement

# A typical overdraft user's statement, rounded from the medians of development applicants who
# used an overdraft (35 borrows, repaid 59%, owes 0.58 months of income, low balance 23% of days)
TYPICAL_OVERDRAFT_USER = {
    "stmt_overdraft_borrow_count": 35.0,
    "stmt_overdraft_repay_ratio": 0.6,
    "stmt_overdraft_outstanding_months": 0.6,
    "stmt_low_balance_share": 0.25,
}

# --- who a check applies to --------------------------------------------------------------


def _everyone(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=frame.index)


def _statement(frame: pd.DataFrame) -> pd.Series:
    return frame["has_statement"] == 1


def _overdraft_users(frame: pd.DataFrame) -> pd.Series:
    return _statement(frame) & (frame["stmt_overdraft_borrow_count"] > 0)


def _overdraft_non_users(frame: pd.DataFrame) -> pd.Series:
    return _statement(frame) & (frame["stmt_overdraft_borrow_count"] == 0)


WHO = {
    "everyone": _everyone,
    "statement": _statement,
    "overdraft_users": _overdraft_users,
    "overdraft_non_users": _overdraft_non_users,
}

# --- changes ------------------------------------------------------------------------------


def unchanged(frame: pd.DataFrame) -> pd.DataFrame:
    return frame


def set_to(**values) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Every applicant gets these values."""

    def change(frame):
        out = frame.copy()
        for col, value in values.items():
            out[col] = value
        return out

    return change


def scale(factors: dict[str, float]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Multiply columns (e.g. income x 1.5)."""

    def change(frame):
        out = frame.copy()
        for col, factor in factors.items():
            out[col] = out[col].astype(float) * factor
        return out

    return change


def shift(
    amounts: dict[str, float], low: float | None = None, high: float | None = None
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Add to columns, kept inside [low, high] (a share can't go above 1 or below 0)."""

    def change(frame):
        out = frame.copy()
        for col, amount in amounts.items():
            out[col] = (out[col].astype(float) + amount).clip(lower=low, upper=high)
        return out

    return change


# --- the checks ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    name: str
    description: str  # plain words: what changes
    expect: str  # HIGHER, LOWER or NONE
    after: Callable[[pd.DataFrame], pd.DataFrame]
    before: Callable[[pd.DataFrame], pd.DataFrame] = unchanged
    who: str = "everyone"
    kind: str = "single"  # single, linked, or report (no expectation)
    columns: tuple[str, ...] = field(default=())


def _pair(name, column, high, low, expect, description, who="everyone"):
    """Risk with column = high compared with the SAME applicant with column = low."""
    return Check(
        name,
        description,
        expect,
        after=set_to(**{column: high}),
        before=set_to(**{column: low}),
        who=who,
        columns=(column,),
    )


EMERGENCY_OTHERS = ("savings", "work", "family_friends", "borrowing", "selling_assets", "other")

CHECKS: list[Check] = [
    # --- statement behaviour (statement holders) ---
    Check(
        "more_betting",
        "Betting share of spending +5 points",
        HIGHER,
        shift({"stmt_betting_share": 0.05}, 0, 1),
        who="statement",
        columns=("stmt_betting_share",),
    ),
    Check(
        "more_overdraft_borrowing",
        "10 more overdraft borrowings, same amount owed",
        HIGHER,
        shift({"stmt_overdraft_borrow_count": 10}, 0),
        who="statement",
        columns=("stmt_overdraft_borrow_count",),
    ),
    Check(
        "owes_more_overdraft",
        "Overdraft owed +0.25 months of income",
        HIGHER,
        shift({"stmt_overdraft_outstanding_months": 0.25}, 0),
        who="statement",
        columns=("stmt_overdraft_outstanding_months",),
    ),
    Check(
        "repays_less_overdraft",
        "Repays 20 points less of what was borrowed on overdraft",
        HIGHER,
        shift({"stmt_overdraft_repay_ratio": -0.2}, 0, 1),
        who="overdraft_users",
        columns=("stmt_overdraft_repay_ratio",),
    ),
    Check(
        "more_low_balance_days",
        "10 points more of days with a near-empty balance",
        HIGHER,
        shift({"stmt_low_balance_share": 0.1}, 0, 1),
        who="statement",
        columns=("stmt_low_balance_share",),
    ),
    Check(
        "higher_end_balance",
        "Month-end balance +0.5 months of income",
        LOWER,
        shift({"stmt_end_balance_months": 0.5}, 0),
        who="statement",
        columns=("stmt_end_balance_months",),
    ),
    Check(
        "misses_a_bill",
        "Pays bills in one month fewer",
        HIGHER,
        shift({"stmt_paybill_months": -1}, 0, 6),
        who="statement",
        columns=("stmt_paybill_months",),
    ),
    Check(
        "higher_average_inflow",
        "Average monthly inflow x 1.5",
        LOWER,
        scale({"stmt_avg_monthly_inflow": 1.5}),
        who="statement",
        columns=("stmt_avg_monthly_inflow",),
    ),
    Check(
        "higher_median_inflow",
        "Median monthly inflow x 1.5",
        LOWER,
        scale({"stmt_median_monthly_inflow": 1.5}),
        who="statement",
        columns=("stmt_median_monthly_inflow",),
    ),
    Check(
        "higher_loan_to_inflow",
        "Loan-to-inflow ratio x 1.5",
        HIGHER,
        scale({"stmt_loan_to_inflow": 1.5}),
        who="statement",
        columns=("stmt_loan_to_inflow",),
    ),
    Check(
        "more_volatile_income",
        "Income volatility +0.2",
        HIGHER,
        shift({"stmt_inflow_volatility": 0.2}, 0),
        who="statement",
        columns=("stmt_inflow_volatility",),
    ),
    # --- loan and profile (everyone) ---
    Check(
        "bigger_loan",
        "Loan amount x 1.5",
        HIGHER,
        scale({"loan_amount_kes": 1.5}),
        columns=("loan_amount_kes",),
    ),
    Check(
        "higher_income_group",
        "One income quintile higher",
        LOWER,
        shift({"income_quintile": 1}, 1, 5),
        columns=("income_quintile",),
    ),
    _pair("in_work", "in_workforce", 1, 0, LOWER, "In the workforce vs not"),
    _pair("saved", "saved_past_year", 1, 0, LOWER, "Saved in the past year vs not"),
    _pair("savings_group", "saved_in_savings_club", 1, 0, LOWER, "Saves in a savings group vs not"),
    _pair(
        "medical_borrowing",
        "borrowed_for_medical",
        1,
        0,
        HIGHER,
        "Borrowed for medical bills vs not",
    ),
    *[
        _pair(
            f"no_emergency_money_vs_{other}",
            "emergency_fund_source",
            "could_not",
            other,
            HIGHER,
            f"Could not find emergency money vs would use {other.replace('_', ' ')}",
        )
        for other in EMERGENCY_OTHERS
    ],
    _pair(
        "very_worried_vs_not",
        "worried_about_bills",
        "very",
        "not_at_all",
        HIGHER,
        "Very worried about bills vs not at all",
    ),
    _pair(
        "very_worried_vs_somewhat",
        "worried_about_bills",
        "very",
        "somewhat",
        HIGHER,
        "Very worried about bills vs somewhat",
    ),
    _pair("young", "age", 22, 35, HIGHER, "Aged 22 vs 35 (18-24 year olds are riskier)"),
    # --- linked: inputs that are computed together ---
    Check(
        "bigger_loan_linked",
        "Loan x 1.5, and so loan-to-inflow x 1.5",
        HIGHER,
        scale({"loan_amount_kes": 1.5, "stmt_loan_to_inflow": 1.5}),
        who="statement",
        kind="linked",
        columns=("loan_amount_kes", "stmt_loan_to_inflow"),
    ),
    Check(
        "higher_income_linked",
        "Average and median inflow x 1.5, and so loan-to-inflow / 1.5",
        LOWER,
        scale(
            {
                "stmt_avg_monthly_inflow": 1.5,
                "stmt_median_monthly_inflow": 1.5,
                "stmt_loan_to_inflow": 1 / 1.5,
            }
        ),
        who="statement",
        kind="linked",
        columns=("stmt_avg_monthly_inflow", "stmt_median_monthly_inflow", "stmt_loan_to_inflow"),
    ),
    Check(
        "starts_using_overdraft",
        "A non-user gets a typical overdraft user's statement",
        HIGHER,
        set_to(**TYPICAL_OVERDRAFT_USER),
        who="overdraft_non_users",
        kind="linked",
        columns=tuple(TYPICAL_OVERDRAFT_USER),
    ),
    Check(
        "borrows_and_owes_more",
        "10 more overdraft borrowings and +0.1 months of income owed",
        HIGHER,
        shift({"stmt_overdraft_borrow_count": 10, "stmt_overdraft_outstanding_months": 0.1}, 0),
        who="overdraft_users",
        kind="linked",
        columns=("stmt_overdraft_borrow_count", "stmt_overdraft_outstanding_months"),
    ),
]


def _report(name, description, after, columns, who="everyone", before=unchanged):
    return Check(name, description, NONE, after, before, who, "report", columns)


# Drivers the simulator uses, but with no single direction: seasonal earners (few payments,
# long gaps) and irregular earners (many payments, short gaps) are BOTH riskier than salaried
# ones (one payment a month), so risk isn't monotone in either input.
NO_DIRECTION = {
    "stmt_income_payments_per_month": "seasonal (fewest) and irregular (most) are both riskier",
    "stmt_days_since_last_income": "seasonal (longest gaps), irregular (shortest): both riskier",
}

REPORTS: list[Check] = [
    _report(
        "more_income_payments",
        "2 more income payments a month (no expected direction)",
        shift({"stmt_income_payments_per_month": 2}, 0),
        ("stmt_income_payments_per_month",),
        "statement",
    ),
    _report(
        "longer_since_income",
        "Last income 10 days earlier (no expected direction)",
        shift({"stmt_days_since_last_income": 10}, 0),
        ("stmt_days_since_last_income",),
        "statement",
    ),
    # non-drivers: the simulator never uses these, so ideally their effect is small
    _report(
        "more_airtime",
        "Airtime share +5 points (non-driver)",
        shift({"stmt_airtime_share": 0.05}, 0, 1),
        ("stmt_airtime_share",),
        "statement",
    ),
    _report(
        "more_cash_out",
        "Cash-out share +5 points (non-driver)",
        shift({"stmt_cash_out_share": 0.05}, 0, 1),
        ("stmt_cash_out_share",),
        "statement",
    ),
    _report(
        "more_p2p_in",
        "Money from family and friends +5 points (non-driver)",
        shift({"stmt_p2p_in_share": 0.05}, 0, 1),
        ("stmt_p2p_in_share",),
        "statement",
    ),
    _report(
        "more_transactions",
        "10 more transactions a month (non-driver)",
        shift({"stmt_transactions_per_month": 10}, 0),
        ("stmt_transactions_per_month",),
        "statement",
    ),
    _report(
        "longer_term",
        "6-month vs 1-month loan (non-driver)",
        set_to(term_months=6),
        ("term_months",),
        before=set_to(term_months=1),
    ),
    _report(
        "tertiary_education",
        "Tertiary vs primary education (non-driver)",
        set_to(education_level="tertiary"),
        ("education_level",),
        before=set_to(education_level="primary_or_less"),
    ),
    *[
        _report(
            col.removeprefix("borrowed_"),
            f"{col.replace('_', ' ').capitalize()}: yes vs no (non-driver)",
            set_to(**{col: 1}),
            (col,),
            before=set_to(**{col: 0}),
        )
        for col in (
            "borrowed_past_year",
            "borrowed_from_fin_institution",
            "borrowed_from_family_friends",
        )
    ],
]

ALL_CHECKS = CHECKS + REPORTS
BY_NAME = {c.name: c for c in ALL_CHECKS}

# Checks the v1.0.0 champion fails, and what investigating them showed (step 4.3)
KNOWN_FLIPS = {
    "more_overdraft_borrowing": (
        "With the amount owed held equal, more borrowings means the debt revolves (borrowed and "
        "repaid again). In the simulator only users with room under their limit can keep "
        "borrowing, so among overdraft users the count is itself slightly linked with LOWER true "
        "risk (Spearman -0.12). A linear fit to the true risk gives it a negative weight too. "
        "Starting to use an overdraft and owing more both raise risk as expected."
    ),
    "borrows_and_owes_more": (
        "The same pattern: 10 more borrowings lower risk by more than 0.1 months of extra debt "
        "raises it, for 95% of overdraft users. Owing more on its own raises risk for everyone."
    ),
    "higher_median_inflow": (
        "Average and median inflow move together (correlated twins). Held against a fixed "
        "average, a higher median gets a small positive weight; a linear fit to the true risk "
        "agrees (+0.02). When income really rises, both rise and the loan-to-inflow ratio falls: "
        "that linked check lowers risk for everyone."
    ),
}

# --- running a check ----------------------------------------------------------------------


def run_check(
    predict: Callable[[pd.DataFrame], np.ndarray],
    frame: pd.DataFrame,
    check: Check,
    decide: Callable[[pd.DataFrame, np.ndarray], np.ndarray] | None = None,
) -> dict:
    """Apply one check to the applicants it is for and summarise how the risk moved.

    predict(frame) gives probabilities of default. decide(frame, p), if given, turns them into
    'approve' / 'decline', to count decisions that moved the wrong way.
    """
    eligible = frame[WHO[check.who](frame)]
    result = {
        "check": check.name,
        "description": check.description,
        "kind": check.kind,
        "expect": check.expect,
        "who": check.who,
        "n": len(eligible),
    }
    if eligible.empty:
        return result | {"passed": None, "note": "no eligible applicants"}

    before, after = check.before(eligible), check.after(eligible)
    p_before, p_after = predict(before), predict(after)
    change = np.asarray(p_after) - np.asarray(p_before)
    result |= {
        "mean_change_pts": float(change.mean() * 100),
        "min_change_pts": float(change.min() * 100),
        "max_change_pts": float(change.max() * 100),
        "share_up": float(np.mean(change > TOLERANCE)),
        "share_down": float(np.mean(change < -TOLERANCE)),
    }
    if check.expect == NONE:
        return result | {"passed": None, "note": "measured only"}

    wrong = change < -TOLERANCE if check.expect == HIGHER else change > TOLERANCE
    responds = change.mean() > TOLERANCE if check.expect == HIGHER else change.mean() < -TOLERANCE
    result |= {
        "share_wrong": float(wrong.mean()),
        "worst_wrong_pts": float(np.abs(change[wrong]).max() * 100) if wrong.any() else 0.0,
        "responds": bool(responds),
    }
    if decide is not None:
        d_before, d_after = np.asarray(decide(before, p_before)), np.asarray(decide(after, p_after))
        eased = (d_before == "decline") & (d_after == "approve")  # riskier, yet approved
        tightened = (d_before == "approve") & (d_after == "decline")  # safer, yet declined
        result["share_decisions_wrong"] = float(
            (eased if check.expect == HIGHER else tightened).mean()
        )
    passed = not wrong.any() and responds
    return result | {
        "passed": bool(passed),
        "note": "known flip" if check.name in KNOWN_FLIPS else "",
    }


def run_all(predict, frame, checks=None, decide=None) -> pd.DataFrame:
    """Every check (default: the expectation checks and the reports) as one table."""
    rows = [run_check(predict, frame, c, decide) for c in (checks or ALL_CHECKS)]
    return pd.DataFrame(rows).set_index("check")


def probe_applicants(frame: pd.DataFrame) -> pd.DataFrame:
    """A small set of applicants that every check applies to.

    The applicants themselves, plus a copy of every statement holder who never used an
    overdraft, given a typical overdraft user's statement (so overdraft-user checks can run on a
    handful of examples, such as the bundle's golden examples in CI).
    """
    non_users = frame[_overdraft_non_users(frame)]
    users = set_to(**TYPICAL_OVERDRAFT_USER)(non_users)
    return pd.concat([frame, users], ignore_index=True)
