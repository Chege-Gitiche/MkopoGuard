"""Loan checks on the real population. Skipped when the file isn't present (e.g. in CI)."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.loans import add_loans

POPULATION_FILE = config.DATA_INTERIM / "population.parquet"

pytestmark = pytest.mark.skipif(not POPULATION_FILE.exists(), reason="population.parquet not built")


@pytest.fixture(scope="module")
def applicants():
    return add_loans(pd.read_parquet(POPULATION_FILE))


def test_every_applicant_has_a_loan(applicants):
    assert len(applicants) == config.EXPECTED_APPLICANTS
    assert applicants["loan_amount_kes"].notna().all()
    assert applicants["term_months"].notna().all()


def test_typical_loan_size_is_plausible(applicants):
    # Median income is about KES 11,000 and the median ratio about 1.0
    assert 8_000 <= applicants["loan_amount_kes"].median() <= 15_000
