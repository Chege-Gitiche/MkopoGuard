"""Step 4.2 on the real champion: every validation applicant gets 3 + 2 sound reasons."""

import numpy as np
import pandas as pd
import pytest

from mkopoguard import bundle, config
from mkopoguard.explain import reasons as rc

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "features.parquet").exists()
    or not (config.BUNDLE_DIR / bundle.MANIFEST).exists(),
    reason="features.parquet or the model bundle not built",
)
STATEMENT_CODES = {u.code for u in rc.UNITS if u.needs_statement}


@pytest.fixture(scope="module")
def explained():
    from mkopoguard.models.experiment import load_training_data

    champion = bundle.load_bundle()
    train, validation = load_training_data()
    reasoner = rc.Reasoner(champion.model, pd.concat([train, validation], ignore_index=True))
    sample = validation.sample(1_000, random_state=0).reset_index(drop=True)
    return champion, sample, reasoner, reasoner.explain_rows(sample)


def test_three_plus_two_different_reasons_for_everyone(explained):
    *_, results = explained
    for result in results:
        codes = [r["code"] for r in result["raising"] + result["lowering"]]
        assert len(result["raising"]) == 3 and len(result["lowering"]) == 2
        assert len(set(codes)) == 5


def test_thin_file_applicants_get_no_statement_reasons(explained):
    _, sample, _, results = explained
    for result, has in zip(results, sample["has_statement"], strict=True):
        codes = {r["code"] for r in result["raising"] + result["lowering"]}
        assert not (codes & STATEMENT_CODES) if has == 0 else "no_statement" not in codes


def test_explained_probability_is_the_bundles_probability(explained):
    champion, sample, _, results = explained
    expected = champion.predict_proba(sample)
    got = np.array([r["probability_of_default"] for r in results])
    assert np.abs(got - expected).max() < 1e-9
