"""Step 1.5: build the applicant population from the cleaned Findex data.

Run from the project root:  python scripts/build_population.py
"""

import pandas as pd

from mkopoguard import config
from mkopoguard.data.population import build_population

IN_PATH = config.DATA_INTERIM / "findex_clean.parquet"
OUT_PATH = config.DATA_INTERIM / "population.parquet"


def main() -> None:
    clean = pd.read_parquet(IN_PATH)
    population = build_population(clean)
    assert len(population) == config.EXPECTED_APPLICANTS

    population.to_parquet(OUT_PATH, index=False)

    # A fingerprint of the whole table: identical runs must print the same number
    fingerprint = pd.util.hash_pandas_object(population, index=False).sum()
    print(f"{len(population):,} applicants saved to {OUT_PATH}")
    print(f"With a statement: {population['has_statement'].sum():,}")
    print("Income pattern:", population["hidden_income_pattern"].value_counts().to_dict())
    print(f"Median monthly income: KES {population['hidden_monthly_income_kes'].median():,.0f}")
    print(f"Fingerprint (seed {config.SIMULATION_SEED}): {fingerprint}")


if __name__ == "__main__":
    main()
