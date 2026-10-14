"""Step 4.1: what the simulator actually uses, so the explanations can be checked against it.

The data is synthetic, so we KNOW what causes default (data card, section 2; data/defaults.py).
An input is a "driver" if the default rule uses it directly or through a hidden trait it
reflects. Everything else is a non-driver: it may be correlated with drivers, but changing it
would not change the true risk. If the model leaned heavily on non-drivers, something would be
wrong (leakage, a bug in the features, or a broken simulator).
"""

import pandas as pd

DRIVERS = {
    # loan size relative to income
    "loan_amount_kes": "loan-to-income ratio",
    "income_quintile": "income (sets the hidden monthly income)",
    "in_workforce": "direct effect, and income",
    "stmt_avg_monthly_inflow": "income",
    "stmt_median_monthly_inflow": "income",
    "stmt_loan_to_inflow": "loan-to-income ratio",
    # income pattern (seasonal and irregular earners are riskier)
    "stmt_inflow_volatility": "income pattern",
    "stmt_income_payments_per_month": "income pattern",
    "stmt_days_since_last_income": "income pattern",
    # financial stress
    "saved_past_year": "direct effect, and stress",
    "saved_in_savings_club": "direct effect",
    "emergency_fund_source": "stress (could not find emergency funds)",
    "worried_about_bills": "stress (very worried about bills)",
    "borrowed_for_medical": "stress",
    # behaviour, driven by hidden discipline and overdraft habit
    "stmt_betting_share": "betting share (the strongest single weight)",
    "stmt_paybill_months": "discipline (missed bill payments)",
    "stmt_overdraft_borrow_count": "overdraft habit",
    "stmt_overdraft_repay_ratio": "discipline (overdraft repayment)",
    "stmt_overdraft_outstanding_months": "overdraft habit and discipline",
    "stmt_end_balance_months": "discipline and overdraft (balance behaviour)",
    "stmt_low_balance_share": "discipline and overdraft (balance behaviour)",
    # age: only 18-24 year olds carry extra risk
    "age": "18-24 year olds are riskier",
}

# has_statement is special: for thin-file applicants every statement feature is filled with
# its training median, so has_statement carries the whole thin-file vs statement difference.
# Its explanations are always read together with the statement features (one "block").
STATEMENT_FLAG = "has_statement"


def is_driver(column: str) -> bool:
    return column in DRIVERS


# --- the known-answer check ---------------------------------------------------------------

TOP_DRIVERS = 10  # the 10 most important inputs (statement flag aside) must all be drivers
MAX_NON_DRIVER_SHARE = 0.15  # non-drivers may carry at most 15% of the total importance


def check_against_design(importance: pd.Series) -> dict:
    """Compare a global importance ranking (biggest first) with what the simulator uses."""
    ranked = importance.drop(index=STATEMENT_FLAG, errors="ignore")
    top = ranked.head(TOP_DRIVERS)
    share = ranked / importance.sum()
    non_driver_share = float(share[[not is_driver(c) for c in ranked.index]].sum())
    ranks = pd.Series(range(1, len(importance) + 1), index=importance.index)
    bottom_third = ranks > len(importance) * 2 // 3
    under_used = [c for c in importance.index if is_driver(c) and bottom_third[c]]
    top_non_drivers = [c for c in top.index if not is_driver(c)]
    return {
        "top_inputs_all_drivers": not top_non_drivers,
        "top_non_drivers": top_non_drivers,
        "non_driver_share": non_driver_share,
        "non_driver_share_ok": non_driver_share <= MAX_NON_DRIVER_SHARE,
        "under_used_drivers": under_used,
        "passed": not top_non_drivers and non_driver_share <= MAX_NON_DRIVER_SHARE,
    }
