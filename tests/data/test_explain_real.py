"""Step 4.1 on the real champion: explanations add up and match the simulator's design."""

import numpy as np
import pandas as pd
import pytest

from mkopoguard import bundle, config
from mkopoguard.explain import design
from mkopoguard.explain.linear_shap import LinearShap, combine, importance
from mkopoguard.features.groups import FEATURE_GROUPS

pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "features.parquet").exists()
    or not (config.BUNDLE_DIR / bundle.MANIFEST).exists(),
    reason="features.parquet or the model bundle not built",
)


@pytest.fixture(scope="module")
def explained():
    from mkopoguard.models.experiment import load_training_data

    champion = bundle.load_bundle()
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)
    explainer = LinearShap(champion.model, development)
    return champion, development, explainer, explainer.contributions(development)


def test_contributions_add_up_for_every_development_applicant(explained):
    champion, development, explainer, contribs = explained
    expected = champion.model.decision_function(development[champion.columns])
    assert np.abs(explainer.log_odds(contribs) - expected).max() < 1e-9


def test_most_important_inputs_are_the_simulators_drivers(explained):
    *_, contribs = explained
    result = design.check_against_design(importance(contribs))
    assert result["passed"], result


def test_thin_file_statement_block_is_constant(explained):
    _, development, _, contribs = explained
    thin = (development["has_statement"] == 0).to_numpy()
    block = combine(contribs, {"statement": FEATURE_GROUPS["statement"]})["statement"]
    assert block[thin].std() < 1e-9  # no statement, so nothing in it can differ
