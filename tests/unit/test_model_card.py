"""Step 4.5: the model card is complete, honest, and matches the saved results (runs in CI).

The results in docs/results/ are committed, so CI rebuilds the card from them and checks it.
"""

import json

import pytest

from mkopoguard import config
from mkopoguard.explain.behaviour import BY_NAME, KNOWN_FLIPS
from mkopoguard.model_card import (
    EVIDENCE_FILES,
    NOT_FOR_LENDING,
    REQUIRED_SECTIONS,
    build_model_card,
    load_evidence,
)

pytestmark = pytest.mark.skipif(
    not all((config.RESULTS_DIR / f).exists() for f in EVIDENCE_FILES.values())
    or not config.TEST_REPORT_FILE.exists(),
    reason="results of steps 3.7-4.4 not saved yet",
)


@pytest.fixture(scope="module")
def evidence():
    return load_evidence()


@pytest.fixture(scope="module")
def card(evidence):
    report = json.loads(config.TEST_REPORT_FILE.read_text(encoding="utf-8"))
    policy = json.loads(config.DECISION_POLICY_FILE.read_text(encoding="utf-8"))
    policy["thresholds_by_term_months"] = {
        int(k): v for k, v in policy["thresholds_by_term_months"].items()
    }
    selection = json.loads(config.FEATURE_SELECTION_FILE.read_text(encoding="utf-8"))
    return build_model_card(report, policy, selection["kept_groups"], "run-x", evidence)


def test_card_has_every_section_the_guide_asks_for(card):
    headings = [line.removeprefix("## ") for line in card.splitlines() if line.startswith("## ")]
    for section in REQUIRED_SECTIONS:
        assert section in headings, section


def test_card_says_it_is_not_for_real_lending_at_the_top(card):
    assert NOT_FOR_LENDING in card.split("## ")[0]
    assert "synthetic" in card.lower()


def test_card_names_the_version_the_code_expects(card):
    assert f"{config.MODEL_NAME} v{config.MODEL_VERSION}" in card.splitlines()[0]


def test_card_lists_every_feature_group_with_its_ablation_decision(card, evidence):
    selection = json.loads(config.FEATURE_SELECTION_FILE.read_text(encoding="utf-8"))
    for group in evidence["ablation"].index:
        decision = "kept" if group in selection["kept_groups"] else "dropped"
        assert any(
            line.startswith(f"| {group} |") and line.rstrip().endswith(f"| {decision} |")
            for line in card.splitlines()
        ), group


def test_card_reports_every_known_behaviour_flip(card):
    for name in KNOWN_FLIPS:
        assert BY_NAME[name].description in card, name


def test_card_reports_every_fairness_flag(card, evidence):
    flags = evidence["fairness_summary"]["flags"]
    for attribute, by_flag in flags.items():
        for groups in by_flag.values():
            for group in groups:
                assert group in card, (attribute, group)


def test_card_hides_no_gap_it_found(card, evidence):
    """If the audit found a gap the model adds, the card must say what it means."""
    flags = evidence["fairness_summary"]["flags"]
    if flags["gender"]["model_adds_gap"] or flags["age_group"]["model_adds_gap"]:
        assert "declined more often" in card and "Mitigations" in card


def test_missing_results_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing results"):
        load_evidence(tmp_path)
