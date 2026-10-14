"""Step 4.3 on all validation applicants: the same checks, on 2,808 real (synthetic) people."""

import pandas as pd
import pytest

from mkopoguard import bundle, config
from mkopoguard.explain import behaviour as bh

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "features.parquet").exists()
    or not (config.BUNDLE_DIR / bundle.MANIFEST).exists(),
    reason="features.parquet or the model bundle not built",
)


@pytest.fixture(scope="module")
def results() -> pd.DataFrame:
    from mkopoguard.models.experiment import load_training_data

    champion = bundle.load_bundle()
    _, validation = load_training_data()
    return bh.run_all(champion.predict_proba, validation, bh.CHECKS)


def test_every_check_except_the_known_flips_passes(results):
    expected = results.drop(index=list(bh.KNOWN_FLIPS))
    assert expected["passed"].all(), expected.index[~expected["passed"].astype(bool)].tolist()


def test_the_known_flips_still_fail(results):
    assert not results.loc[list(bh.KNOWN_FLIPS), "passed"].any()
