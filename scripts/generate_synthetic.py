"""Step 1.6: generate loans (part 1), transactions (part 2) and defaults (part 3).

Run from the project root:  python scripts/generate_synthetic.py
"""

import pandas as pd

from mkopoguard import config
from mkopoguard.data.loans import add_loans

IN_PATH = config.DATA_INTERIM / "population.parquet"
APPLICANTS_PATH = config.DATA_PROCESSED / "applicants.parquet"


def main() -> None:
    population = pd.read_parquet(IN_PATH)

    # Part 1: loans
    applicants = add_loans(population)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    applicants.to_parquet(APPLICANTS_PATH, index=False)

    amounts = applicants["loan_amount_kes"]
    print(f"{len(applicants):,} applicants saved to {APPLICANTS_PATH}")
    print(f"Loan amount (KES): median {amounts.median():,.0f}", end="")
    print(f", min {amounts.min():,}, max {amounts.max():,}")
    print("Term (months):", applicants["term_months"].value_counts().sort_index().to_dict())
    fingerprint = pd.util.hash_pandas_object(applicants, index=False).sum()
    print(f"Fingerprint (seed {config.SIMULATION_SEED}): {fingerprint}")


if __name__ == "__main__":
    main()
