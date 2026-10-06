"""Checks on the real Findex file. Skipped when the file isn't present (e.g. in CI)."""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.clean import clean_findex

pytestmark = pytest.mark.skipif(
    not config.FINDEX_FILE.exists(), reason="Findex CSV not in data/raw"
)


@pytest.fixture(scope="module")
def applicants():
    raw = pd.read_csv(config.FINDEX_FILE, encoding=config.FINDEX_ENCODING, low_memory=False)
    pool = raw[raw["economycode"].isin(config.POOL_COUNTRIES)]
    return clean_findex(pool)


def test_applicant_count(applicants):
    assert len(applicants) == config.EXPECTED_APPLICANTS


def test_everyone_is_an_adult(applicants):
    assert applicants["age"].min() >= config.MIN_AGE


def test_all_20_countries_present(applicants):
    assert set(applicants["country_code"]) == set(config.POOL_COUNTRIES)


def test_respondent_ids_are_unique(applicants):
    assert applicants["respondent_id"].is_unique
