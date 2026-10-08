"""Step 2.3: turn each applicant's statement into one row of features.

Every feature starts with "stmt_". The same function runs in training (on all 2.1 million
transactions) and later in the API (on one uploaded statement), so the model always sees
features built exactly the same way.

Leakage rule: only transactions in the STATEMENT_DAYS before the application date are used.
Anything on or after the application date is dropped here as a final safety net.
Applicants without a statement get missing values (NaN) for every feature.
"""

import numpy as np
import pandas as pd

from mkopoguard import config

PREFIX = "stmt_"
MONTHS = config.STATEMENT_DAYS // 30
LOW_BALANCE_KES = 100  # a balance under this counts as "empty"

FEATURES = [
    "stmt_avg_monthly_inflow",
    "stmt_median_monthly_inflow",
    "stmt_inflow_volatility",
    "stmt_income_payments_per_month",
    "stmt_days_since_last_income",
    "stmt_p2p_in_share",
    "stmt_betting_share",
    "stmt_airtime_share",
    "stmt_cash_out_share",
    "stmt_paybill_months",
    "stmt_overdraft_borrow_count",
    "stmt_overdraft_repay_ratio",
    "stmt_overdraft_outstanding_months",
    "stmt_end_balance_months",
    "stmt_low_balance_share",
    "stmt_transactions_per_month",
    "stmt_loan_to_inflow",
]


def _safe_divide(top, bottom):
    """top / bottom, with NaN wherever bottom is zero or missing."""
    top = pd.Series(top, dtype=float)
    bottom = pd.Series(bottom, dtype=float).reindex(top.index)
    return top.where(bottom > 0) / bottom.where(bottom > 0)


def _within_window(transactions: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    """Attach each transaction's month bucket (0 = most recent 30 days) and drop leaks."""
    tx = transactions.join(dates.rename("_application_date"), on="applicant_id", how="inner")
    days_before = (tx["_application_date"] - tx["timestamp"]).dt.total_seconds() / 86_400
    keep = (days_before > 0) & (days_before <= config.STATEMENT_DAYS)
    tx = tx.loc[keep].copy()
    tx["_month"] = np.minimum((days_before[keep] // 30).astype(int), MONTHS - 1)
    tx["_days_before"] = days_before[keep]
    tx["type"] = tx["type"].astype(str)
    return tx


def statement_features(transactions: pd.DataFrame, applicants: pd.DataFrame) -> pd.DataFrame:
    """One row per applicant (indexed by applicant_id) with every feature in FEATURES.

    applicants needs: applicant_id, application_date, loan_amount_kes.
    """
    apps = applicants.set_index("applicant_id")
    tx = _within_window(transactions, apps["application_date"])
    ids = tx["applicant_id"]
    amount = tx["amount_kes"].astype(float)

    def total(kind):
        return amount.where(tx["type"] == kind, 0.0).groupby(ids).sum()

    def count(kind):
        return (tx["type"] == kind).groupby(ids).sum()

    # Income: monthly totals over the 6 months, including months with nothing
    income = tx[tx["type"] == "income_in"]
    monthly = (
        income.groupby(["applicant_id", "_month"])["amount_kes"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(MONTHS), fill_value=0)
        .astype(float)
    )
    avg_inflow = monthly.mean(axis=1)
    outflow = amount.where(tx["direction"].astype(str) == "out", 0.0).groupby(ids).sum()
    borrowed = total("overdraft_borrow")
    repaid = total("overdraft_repay")
    last_balance = tx.sort_values("timestamp").groupby("applicant_id")["balance_after"].last()

    out = pd.DataFrame(index=tx["applicant_id"].unique())
    out["stmt_avg_monthly_inflow"] = avg_inflow
    out["stmt_median_monthly_inflow"] = monthly.median(axis=1)
    out["stmt_inflow_volatility"] = _safe_divide(monthly.std(axis=1, ddof=0), avg_inflow)
    out["stmt_income_payments_per_month"] = count("income_in") / MONTHS
    out["stmt_days_since_last_income"] = income.groupby("applicant_id")["_days_before"].min()
    out["stmt_p2p_in_share"] = _safe_divide(total("p2p_in"), total("p2p_in") + total("income_in"))
    out["stmt_betting_share"] = _safe_divide(total("betting"), outflow)
    out["stmt_airtime_share"] = _safe_divide(total("airtime"), outflow)
    out["stmt_cash_out_share"] = _safe_divide(total("cash_out"), outflow)
    paybill = tx[tx["type"] == "paybill"]
    out["stmt_paybill_months"] = paybill.groupby("applicant_id")["_month"].nunique()
    out["stmt_overdraft_borrow_count"] = count("overdraft_borrow")
    # Never borrowed means nothing left unpaid, so the repay ratio is 1.0
    out["stmt_overdraft_repay_ratio"] = _safe_divide(repaid, borrowed)
    out["stmt_overdraft_outstanding_months"] = _safe_divide(borrowed - repaid, avg_inflow)
    out["stmt_end_balance_months"] = _safe_divide(last_balance, avg_inflow)
    low = (tx["balance_after"] < LOW_BALANCE_KES).groupby(ids).mean()
    out["stmt_low_balance_share"] = low
    out["stmt_transactions_per_month"] = tx.groupby("applicant_id").size() / MONTHS
    out["stmt_loan_to_inflow"] = _safe_divide(apps["loan_amount_kes"], avg_inflow)

    # Counts and shares are 0 (not missing) for statement holders who never did that thing,
    # e.g. a statement with income but no spending has a betting share of 0
    zero_if_none = [
        "stmt_paybill_months",
        "stmt_overdraft_borrow_count",
        "stmt_p2p_in_share",
        "stmt_betting_share",
        "stmt_airtime_share",
        "stmt_cash_out_share",
    ]
    for col in zero_if_none:
        out[col] = out[col].fillna(0)
    out["stmt_avg_monthly_inflow"] = out["stmt_avg_monthly_inflow"].fillna(0)
    out["stmt_median_monthly_inflow"] = out["stmt_median_monthly_inflow"].fillna(0)
    out["stmt_overdraft_repay_ratio"] = out["stmt_overdraft_repay_ratio"].where(
        out["stmt_overdraft_borrow_count"] > 0, 1.0
    )
    out["stmt_overdraft_outstanding_months"] = out["stmt_overdraft_outstanding_months"].where(
        out["stmt_overdraft_borrow_count"] > 0, 0.0
    )

    # Every applicant gets a row; thin-file applicants are all NaN
    out = out.reindex(apps.index)[FEATURES].astype(float)
    out.index.name = "applicant_id"
    return out
