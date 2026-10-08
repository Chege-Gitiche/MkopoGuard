"""Step 2.3: build the model table, one row per applicant.

Run from the project root:  python scripts/build_features.py

Output columns: applicant_id, every column the model may use (survey answers, loan details
and statement features), plus the target and is_female for evaluation and the fairness audit.
Hidden simulator traits are left out of the file entirely.
"""

import time

import pandas as pd

from mkopoguard import config
from mkopoguard.features.columns import TARGET, assert_no_leakage, model_columns
from mkopoguard.features.transactions import FEATURES, statement_features

APPLICANTS_PATH = config.DATA_PROCESSED / "applicants.parquet"
TRANSACTIONS_PATH = config.DATA_PROCESSED / "transactions.parquet"
FEATURES_PATH = config.DATA_PROCESSED / "features.parquet"
AUDIT_COLUMNS = ["is_female"]


def build_model_table(applicants: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Applicant columns the model may use, joined with the statement features."""
    stmt = statement_features(transactions, applicants)
    inputs = model_columns(applicants)
    table = applicants.set_index("applicant_id")[inputs + [TARGET] + AUDIT_COLUMNS].join(stmt)
    assert_no_leakage(inputs + FEATURES)
    return table.reset_index()


def main() -> None:
    applicants = pd.read_parquet(APPLICANTS_PATH)
    transactions = pd.read_parquet(TRANSACTIONS_PATH)

    start = time.time()
    table = build_model_table(applicants, transactions)
    table.to_parquet(FEATURES_PATH, index=False)

    holders = table["has_statement"] == 1
    print(f"Built {len(table):,} rows x {table.shape[1]} columns in {time.time() - start:.0f}s")
    print(f"Statement features: {len(FEATURES)} (missing for {(~holders).sum():,} thin-file)")
    print(f"Saved to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
