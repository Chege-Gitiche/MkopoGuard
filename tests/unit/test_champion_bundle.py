"""The shipped bundle in models/champion/ must load and reproduce its golden examples.

The bundle is committed to Git, so this runs in CI too: a model built on one machine (Windows)
is checked on another (Linux). Skipped only if the bundle hasn't been packaged yet (step 3.13).
"""

import json

import pytest

from mkopoguard import bundle, config
from mkopoguard.features.groups import columns_in

pytestmark = pytest.mark.skipif(
    not (config.BUNDLE_DIR / bundle.MANIFEST).exists(), reason="bundle not packaged yet"
)


@pytest.fixture(scope="module")
def champion():
    return bundle.load_bundle(config.BUNDLE_DIR)  # checksums + golden examples


def test_bundle_loads_and_replays_its_golden_examples(champion):
    assert champion.verify_golden() <= config.GOLDEN_TOLERANCE


def test_bundle_is_the_version_the_code_expects(champion):
    assert champion.manifest["name"] == config.MODEL_NAME
    assert champion.manifest["version"] == config.MODEL_VERSION


def test_bundle_uses_exactly_the_kept_feature_groups(champion):
    selection = json.loads(config.FEATURE_SELECTION_FILE.read_text(encoding="utf-8"))
    assert sorted(champion.columns) == sorted(columns_in(selection["kept_groups"]))
    assert not set(champion.columns) & set(columns_in(selection["dropped_groups"]))


def test_bundle_matches_the_final_test_report(champion):
    report = json.loads(config.TEST_REPORT_FILE.read_text(encoding="utf-8"))
    assert champion.manifest["frozen_decisions"] == report["frozen_decisions"]
    assert champion.manifest["test_pr_auc"] == pytest.approx(report["ranking"]["pr_auc"]["value"])


def test_bundle_policy_is_the_decided_policy(champion):
    policy = json.loads(config.DECISION_POLICY_FILE.read_text(encoding="utf-8"))
    decided = {int(k): v for k, v in policy["thresholds_by_term_months"].items()}
    assert champion.policy["thresholds_by_term_months"] == decided


def test_no_forbidden_column_is_a_model_input(champion):
    for column in ("country_code", "is_female", "applicant_id", "defaulted"):
        assert column not in champion.columns
    assert not any(c.startswith("hidden_") for c in champion.columns)


# --- step 4.5: reason codes from the bundle alone, and the published model card -----------------


def test_champion_gives_3_plus_2_reasons_from_the_bundle_alone(champion):
    frame = bundle.golden_frame(champion.golden, champion.schema)
    results = champion.reasoner().explain_rows(frame)
    expected = champion.predict_proba(frame)
    for result, p in zip(results, expected, strict=True):
        assert len(result["raising"]) == 3 and len(result["lowering"]) == 2
        assert abs(result["probability_of_default"] - p) < 1e-9


def test_published_model_card_is_the_bundles_model_card(champion):
    shipped = (champion.path / bundle.FILES["card"]).read_text(encoding="utf-8")
    published = config.MODEL_CARD_FILE.read_text(encoding="utf-8")
    assert published.replace("\r\n", "\n") == shipped.replace("\r\n", "\n")
