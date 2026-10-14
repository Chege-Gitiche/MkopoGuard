"""Step 4.4 on the real data: the audit's own guarantees, and the recorded findings still hold."""

import json

import numpy as np
import pandas as pd
import pytest

from mkopoguard import bundle, config
from mkopoguard import fairness as fa

SUMMARY = config.RESULTS_DIR / "fairness_summary.json"
pytestmark = pytest.mark.skipif(
    not (config.DATA_PROCESSED / "features.parquet").exists() or not SUMMARY.exists(),
    reason="features.parquet or the fairness summary not built (run scripts/fairness_check.py)",
)


@pytest.fixture(scope="module")
def tables():
    from mkopoguard.models.experiment import load_training_data

    champion = bundle.load_bundle()
    train, validation = load_training_data()
    development = pd.concat([train, validation], ignore_index=True)
    threshold = (
        development["term_months"].astype(int).map(champion.policy["thresholds_by_term_months"])
    )
    decline = champion.predict_proba(development) >= threshold.to_numpy()
    groups = fa.audit_columns(development)
    y = development["defaulted"].to_numpy()
    return {a: fa.audit(groups[a], y, decline, resamples=200) for a in fa.ATTRIBUTES}


def test_gender_and_country_are_not_inputs_of_the_shipped_model():
    champion = bundle.load_bundle()
    assert not set(fa.NEVER_INPUTS) & set(champion.columns)


def test_every_applicant_is_in_exactly_one_group(tables):
    sizes = {a: int(t["n"].sum()) for a, t in tables.items()}
    assert len(set(sizes.values())) == 1


def test_recorded_flags_still_hold_for_the_shipped_model(tables):
    """The model card reports these findings; if a new model changes them, update it on purpose."""
    recorded = json.loads(SUMMARY.read_text(encoding="utf-8"))["flags"]
    assert recorded["gender"]["fpr_flag"] == ["female"]
    assert recorded["age_group"]["adverse_impact"] == ["50+"]
    gender, age = tables["gender"], tables["age_group"]
    assert gender.loc["female", "fpr_gap"] > fa.MAX_FPR_GAP
    assert age.loc["50+", "approval_ratio"] < fa.FOUR_FIFTHS
    assert np.isclose(gender.loc["male", "approval_ratio"], 1.0)
