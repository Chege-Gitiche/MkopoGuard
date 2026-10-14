"""Step 4.4: the fairness audit, checked on small made-up groups whose answers we know."""

import numpy as np
import pandas as pd
import pytest

from mkopoguard import fairness as fa
from mkopoguard.features.preprocess import input_columns


def make_groups(n_per_group=2_000, decline_rates=(0.4, 0.4), default_rate=0.2, seed=0):
    """Two groups, A and B, with the same default rate; each declined at its own rate,
    independently of whether the applicant later defaults."""
    rng = np.random.default_rng(seed)
    groups, y, decline = [], [], []
    for name, rate in zip("AB", decline_rates, strict=True):
        groups += [name] * n_per_group
        y.append(rng.random(n_per_group) < default_rate)
        decline.append(rng.random(n_per_group) < rate)
    return np.array(groups), np.concatenate(y), np.concatenate(decline)


def test_audit_labels_gender_age_and_country():
    frame = pd.DataFrame(
        {
            "is_female": pd.array([1, 0, 1, 0, 1], dtype="Int64"),
            "age": [18, 24, 25, 49, 50],
            "country_code": ["KEN", "UGA", "KEN", "TZA", "KEN"],
        }
    )
    labels = fa.audit_columns(frame)
    assert list(labels["gender"]) == ["female", "male", "female", "male", "female"]
    assert list(labels["age_group"].astype(str)) == ["18-24", "18-24", "25-34", "35-49", "50+"]
    assert list(labels["country"]) == ["KEN", "UGA", "KEN", "TZA", "KEN"]


def test_missing_gender_is_refused():
    frame = pd.DataFrame(
        {
            "is_female": pd.array([1, None], dtype="Int64"),
            "age": [30, 30],
            "country_code": ["KEN"] * 2,
        }
    )
    with pytest.raises(ValueError, match="is_female"):
        fa.audit_columns(frame)


def test_rates_match_a_hand_count():
    #           group  A  A  A  A  B  B  B  B
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    decline = np.array([1, 0, 1, 0, 1, 1, 0, 0])
    table = fa.audit(np.array(list("AAAABBBB")), y, decline, resamples=20)
    a, b = table.loc["A"], table.loc["B"]
    assert a["approval_rate"] == 0.5 and a["recall"] == 0.5 and a["fpr"] == 0.5
    assert b["approval_rate"] == 0.5 and b["recall"] == 1.0 and b["fpr"] == pytest.approx(1 / 3)


def test_equal_treatment_raises_no_flags():
    table = fa.audit(*make_groups(decline_rates=(0.4, 0.4)), resamples=200)
    assert not table[["adverse_impact", "fpr_flag"]].any().any()
    assert table["approval_ratio"].min() > 0.9


def test_a_group_declined_much_more_is_flagged():
    table = fa.audit(*make_groups(decline_rates=(0.3, 0.6)), resamples=200)
    assert table.loc["A", "reference"]  # the most-approved group is the reference
    b = table.loc["B"]
    assert b["approval_ratio"] == pytest.approx(0.4 / 0.7, abs=0.05)
    assert b["adverse_impact"] and b["fpr_flag"]
    assert not table.loc["A", ["adverse_impact", "fpr_flag"]].any()  # never flag the reference


def test_a_gap_in_a_tiny_group_is_not_flagged_on_noise():
    groups, y, decline = make_groups(n_per_group=30, decline_rates=(0.30, 0.40), seed=3)
    table = fa.audit(groups, y, decline, resamples=500)
    b = table.loc["B"]
    assert b["fpr_gap_low"] < 0 < b["fpr_gap_high"] or b["fpr_gap"] <= fa.MAX_FPR_GAP
    assert not b["fpr_flag"]


def test_no_excess_when_the_model_decides_like_the_oracle():
    groups, y, decline = make_groups(decline_rates=(0.3, 0.6))
    table = fa.audit(groups, y, decline, oracle_decline=decline, resamples=200)
    assert table["excess_approval_gap"].abs().max() == 0
    assert not table["model_adds_gap"].any()
    assert table.loc["B", "adverse_impact"]  # a real gap is still reported, even if justified


def test_excess_is_flagged_when_the_model_adds_to_the_oracle_gap():
    groups, y, oracle = make_groups(decline_rates=(0.3, 0.4), seed=1)
    decline = oracle.copy()
    b = np.flatnonzero((groups == "B") & ~oracle)
    decline[b[: len(b) // 3]] = True  # the model declines extra B applicants
    table = fa.audit(groups, y, decline, oracle_decline=oracle, resamples=200)
    assert table.loc["B", "excess_approval_gap"] > 0.15
    assert table.loc["B", "model_adds_gap"]


def test_an_explicit_reference_group_is_used():
    groups, y, decline = make_groups(decline_rates=(0.3, 0.6))
    table = fa.audit(groups, y, decline, reference="B", resamples=50)
    assert table.loc["B", "reference"] and table.loc["B", "approval_ratio"] == 1.0
    assert table.loc["A", "approval_ratio"] > 1


def test_missing_group_labels_are_refused():
    with pytest.raises(ValueError, match="group"):
        fa.audit(pd.Series(["A", None, "B"]), [0, 1, 0], [0, 1, 1], resamples=10)


def test_intervals_widen_with_more_groups():
    assert fa.bonferroni_level(2) == pytest.approx(0.95)
    assert fa.bonferroni_level(20) == pytest.approx(1 - 0.05 / 19)


def test_same_risk_gap_is_zero_for_equal_treatment_and_shows_extra_declines():
    rng = np.random.default_rng(0)
    n = 4_000
    groups = np.where(rng.random(n) < 0.5, "female", "male")
    true_p = rng.random(n)
    fair = rng.random(n) < true_p
    biased = fair | ((groups == "female") & (rng.random(n) < 0.2))
    assert abs(fa.same_risk_gap(groups, fair, true_p, "male").loc["average", "female_gap"]) < 0.03
    gap = fa.same_risk_gap(groups, biased, true_p, "male").loc["average", "female_gap"]
    assert 0.05 < gap < 0.2


def test_audited_attributes_are_never_model_inputs():
    for column in fa.NEVER_INPUTS:
        assert column not in input_columns()
