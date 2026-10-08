"""Step 2.2: plausibility checks on the generated data.

Schemas (step 1.7) check that each value is ALLOWED. These checks ask whether the data
as a whole is BELIEVABLE: do balances pile up, does debt explode, are the default rates
sensible? Each limit is a judgement recorded in docs/data_card.md.
"""

import pandas as pd

from mkopoguard import config

# name -> (lowest allowed, highest allowed)
LIMITS = {
    "default_rate": config.DEFAULT_RATE_BAND,
    "default_rate_kenya": config.DEFAULT_RATE_BAND,
    "country_default_rate_min": (0.10, 0.30),
    "country_default_rate_max": (0.10, 0.30),
    "median_loan_kes": (5_000, 20_000),
    "median_transactions_per_applicant": (100, 400),
    "max_transactions_in_one_day": (1, 30),
    "median_final_balance_months": (0.1, 2.0),
    "p99_final_balance_months": (0.5, 6.0),
    "share_using_overdraft": (0.05, 0.50),
    "max_overdraft_owed_months": (0.0, 1.0),
    "share_without_income": (0.0, 0.0),
}


def plausibility_report(applicants: pd.DataFrame, transactions: pd.DataFrame) -> pd.Series:
    """Headline numbers that a human would sanity-check, one per entry in LIMITS."""
    income = applicants.set_index("applicant_id")["hidden_monthly_income_kes"]
    by_country = applicants.groupby("country_code")["defaulted"].mean()
    tx = transactions
    per_applicant = tx.groupby("applicant_id")
    per_day = tx.groupby([tx["applicant_id"], tx["timestamp"].dt.floor("D")]).size()

    final_balance = per_applicant["balance_after"].last()
    final_months = final_balance / income.loc[final_balance.index]

    signed_debt = tx["amount_kes"].where(tx["type"] == "overdraft_borrow", 0) - tx[
        "amount_kes"
    ].where(tx["type"] == "overdraft_repay", 0)
    owed = signed_debt.groupby(tx["applicant_id"]).cumsum().groupby(tx["applicant_id"]).max()
    owed_months = owed / income.loc[owed.index]

    has_income = set(tx.loc[tx["type"] == "income_in", "applicant_id"])
    holders = set(tx["applicant_id"])

    return pd.Series(
        {
            "default_rate": applicants["defaulted"].mean(),
            "default_rate_kenya": by_country.get(config.TARGET_COUNTRY, float("nan")),
            "country_default_rate_min": by_country.min(),
            "country_default_rate_max": by_country.max(),
            "median_loan_kes": applicants["loan_amount_kes"].median(),
            "median_transactions_per_applicant": per_applicant.size().median(),
            "max_transactions_in_one_day": per_day.max(),
            "median_final_balance_months": final_months.median(),
            "p99_final_balance_months": final_months.quantile(0.99),
            "share_using_overdraft": (owed > 0).mean(),
            "max_overdraft_owed_months": owed_months.max(),
            "share_without_income": len(holders - has_income) / len(holders),
        }
    )


def failed_checks(report: pd.Series) -> dict[str, str]:
    """Every report value outside its limits, with a readable explanation."""
    failures = {}
    for name, (low, high) in LIMITS.items():
        value = report[name]
        if not low <= value <= high:
            failures[name] = f"{value:,.3f} is outside {low:,} to {high:,}"
    return failures


def assert_plausible(applicants: pd.DataFrame, transactions: pd.DataFrame) -> pd.Series:
    """Raise if any check fails; otherwise return the report."""
    report = plausibility_report(applicants, transactions)
    failures = failed_checks(report)
    if failures:
        details = "; ".join(f"{k}: {v}" for k, v in failures.items())
        raise ValueError(f"Implausible data: {details}")
    return report
