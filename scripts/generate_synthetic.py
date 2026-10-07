"""Step 1.6: generate loans (part 1), transactions (part 2) and defaults (part 3).

Run from the project root:  python scripts/generate_synthetic.py
"""

import time

import pandas as pd

from mkopoguard import config
from mkopoguard.data.loans import add_loans
from mkopoguard.data.transactions import add_behaviour, generate_transactions

IN_PATH = config.DATA_INTERIM / "population.parquet"
APPLICANTS_PATH = config.DATA_PROCESSED / "applicants.parquet"
TRANSACTIONS_PATH = config.DATA_PROCESSED / "transactions.parquet"


def fingerprint(df: pd.DataFrame) -> int:
    """One number summarising a whole table: identical runs must print the same value."""
    return int(pd.util.hash_pandas_object(df, index=False).sum())


def main() -> None:
    population = pd.read_parquet(IN_PATH)

    # Part 1: loans
    applicants = add_loans(population)
    amounts = applicants["loan_amount_kes"]
    print(f"Loans: median KES {amounts.median():,.0f}", end="")
    print(f", min {amounts.min():,}, max {amounts.max():,}")

    # Part 2: hidden spending behaviour and statements
    applicants = add_behaviour(applicants)
    start = time.time()
    print("Generating statements (about 30-60 seconds)...")
    transactions = generate_transactions(applicants)
    print(f"Transactions: {len(transactions):,} in {time.time() - start:.0f}s")
    print(transactions["type"].value_counts().to_string())

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    applicants.to_parquet(APPLICANTS_PATH, index=False)
    transactions.to_parquet(TRANSACTIONS_PATH, index=False)

    print(f"Saved {len(applicants):,} applicants and {len(transactions):,} transactions")
    print(f"Applicants fingerprint:   {fingerprint(applicants)}")
    print(f"Transactions fingerprint: {fingerprint(transactions)}")


if __name__ == "__main__":
    main()
