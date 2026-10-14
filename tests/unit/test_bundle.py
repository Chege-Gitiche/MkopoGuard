"""Tests for step 3.13: building, loading and checking a model bundle (runs in CI)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mkopoguard import bundle, config
from mkopoguard.explain.reasons import Reasoner
from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)
from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES
from mkopoguard.model_card import build_model_card, load_evidence
from mkopoguard.models import decision as dec
from mkopoguard.models import final


def make_table(n=1_200, seed=0):
    """Applicants shaped like the real model table, with thin-file gaps and real loan terms."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    table["age"] = rng.integers(18, 80, n)
    table["term_months"] = rng.choice([1, 3, 6], n)
    table["loan_amount_kes"] = rng.integers(2, 100, n) * 500
    thin = table["has_statement"] == 0
    table.loc[thin, STATEMENT_FEATURES] = np.nan
    logit = -2.0 + 1.5 * (table["saved_in_savings_club"] == 0)
    table["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return table


POLICY = {
    "monthly_fee_rate": 0.075,
    "monthly_funding_cost": 0.01,
    "net_margin_per_month": 0.065,
    "loss_given_default": 0.8,
    "thresholds_by_term_months": dec.thresholds_by_term(),
}


@pytest.fixture(scope="module")
def data():
    table = make_table()
    return table.iloc[:900].reset_index(drop=True), table.iloc[900:].reset_index(drop=True)


@pytest.fixture(scope="module")
def pipeline(data):
    train, _ = data
    return make_pipeline(LogisticRegression(max_iter=500)).fit(
        train[input_columns()], train["defaulted"]
    )


@pytest.fixture
def built(tmp_path, data, pipeline):
    train, holdout = data
    path = bundle.build_bundle(
        pipeline, train, holdout, tmp_path / "champion", POLICY, "# card\n", {"mlflow_run_id": "x"}
    )
    return path


def rehash(path, name):
    """Update the manifest after editing a file (as a careless 'fix' would)."""
    manifest = json.loads((path / bundle.MANIFEST).read_text(encoding="utf-8"))
    manifest["files"][name] = bundle.sha256(path / name)
    (path / bundle.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")


# --- build and load -------------------------------------------------------------------------


def test_bundle_has_every_file_with_a_checksum(built):
    manifest = json.loads((built / bundle.MANIFEST).read_text(encoding="utf-8"))
    assert set(manifest["files"]) == set(bundle.FILES.values())
    for name, digest in manifest["files"].items():
        assert bundle.sha256(built / name) == digest
    assert {"python", "sklearn", "numpy", "pandas", "skops"} <= set(manifest["environment"])
    assert manifest["version"] == config.MODEL_VERSION


def test_reloaded_bundle_predicts_exactly_like_the_original(built, data, pipeline):
    _, holdout = data
    loaded = bundle.load_bundle(built)
    expected = pipeline.predict_proba(holdout[input_columns()])[:, 1]
    np.testing.assert_array_equal(loaded.predict_proba(holdout), expected)


def test_golden_examples_cover_every_term_and_statement_combination(built):
    golden = bundle.load_bundle(built).golden
    combos = {(g["inputs"]["term_months"], g["inputs"]["has_statement"]) for g in golden}
    assert combos == {(t, s) for t in (1, 3, 6) for s in (0, 1)}


def test_thin_file_golden_examples_keep_their_gaps(built):
    golden = bundle.load_bundle(built).golden
    thin = [g for g in golden if g["inputs"]["has_statement"] == 0]
    assert thin and all(g["inputs"]["stmt_betting_share"] is None for g in thin)


def test_decisions_follow_the_policy(built, data):
    _, holdout = data
    result = bundle.load_bundle(built).decide(holdout)
    expected = np.where(
        result["probability_of_default"] < result["threshold"], "approve", "decline"
    )
    assert (result["decision"] == expected).all()
    assert set(result["threshold"].round(6)) <= {
        round(v, 6) for v in POLICY["thresholds_by_term_months"].values()
    }


# --- tampering and drift ----------------------------------------------------------------------


def test_an_edited_file_is_refused(built):
    (built / "model_card.md").write_text("# edited\n", encoding="utf-8")
    with pytest.raises(bundle.BundleError, match="checksum"):
        bundle.load_bundle(built)


def test_a_missing_file_is_refused(built):
    (built / "input_schema.json").unlink()
    with pytest.raises(bundle.BundleError, match="missing"):
        bundle.load_bundle(built)


def test_no_manifest_is_refused(tmp_path):
    with pytest.raises(bundle.BundleError, match="package_model.py"):
        bundle.load_bundle(tmp_path)


def test_changed_golden_answer_is_caught_even_with_a_matching_checksum(built):
    golden = json.loads((built / "golden_examples.json").read_text(encoding="utf-8"))
    golden[0]["probability_of_default"] += 0.01  # the model "drifted"
    (built / "golden_examples.json").write_text(json.dumps(golden), encoding="utf-8")
    rehash(built, "golden_examples.json")
    with pytest.raises(bundle.BundleError, match="Golden"):
        bundle.load_bundle(built)


def test_a_different_scikit_learn_version_warns(built):
    manifest = json.loads((built / bundle.MANIFEST).read_text(encoding="utf-8"))
    manifest["environment"]["sklearn"] = "0.1.0"
    (built / bundle.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.warns(UserWarning, match="scikit-learn"):
        bundle.load_bundle(built)


# --- input checks --------------------------------------------------------------------------------


@pytest.fixture
def loaded(built):
    return bundle.load_bundle(built)


def one(data):
    _, holdout = data
    return holdout[holdout["has_statement"] == 1].head(1).copy()


def test_clean_input_has_no_problems(loaded, data):
    assert bundle.check_input(one(data), loaded.schema) == {"errors": [], "warnings": []}


@pytest.mark.parametrize(
    "column, value, kind, message",
    [
        ("education_level", "z", "warnings", "unseen categories"),
        ("loan_amount_kes", 10_000_000, "warnings", "outside the training range"),
        ("age", np.nan, "errors", "empty values"),
        ("age", "old", "errors", "must be numbers"),
        ("in_workforce", 2, "errors", "0 or 1"),
        ("term_months", 2, "errors", "one of [1, 3, 6]"),
    ],
)
def test_bad_input_is_reported(loaded, data, column, value, kind, message):
    row = one(data)
    row[column] = row[column].astype(object)
    row[column] = value
    problems = bundle.check_input(row, loaded.schema)
    assert any(message in p for p in problems[kind]), problems


def test_missing_column_is_an_error_and_scoring_refuses_it(loaded, data):
    row = one(data).drop(columns=["age"])
    assert any("missing columns" in p for p in bundle.check_input(row, loaded.schema)["errors"])
    with pytest.raises(ValueError, match="Can't score"):
        loaded.predict_proba(row)


def test_extra_columns_are_ignored(loaded, data):
    row = one(data).assign(favourite_colour="blue")
    assert loaded.predict_proba(row).shape == (1,)


# --- model card ----------------------------------------------------------------------------------


def test_model_card_quotes_the_test_report_numbers(data, pipeline):
    _, holdout = data
    p = pipeline.predict_proba(holdout[input_columns()])[:, 1]
    report = final.build_report(holdout, p, resamples=100)
    report |= {"trained_on": {"applicants": 900}, "ceiling_pr_auc": 0.6}
    card = build_model_card(report, POLICY, ["loan", "profile"], "run123", load_evidence())
    assert f"{report['ranking']['pr_auc']['value']:.3f}" in card
    assert "run123" in card and "synthetic" in card.lower()
    assert f"{POLICY['thresholds_by_term_months'][1]:.3f}" in card


# --- explainer background (step 4.5) -----------------------------------------------------------


def test_bundle_carries_the_explainer_background(built):
    saved = json.loads((built / "explainer_background.json").read_text(encoding="utf-8"))
    assert {"transformed_mean", "statement_shift", "base_value"} <= set(saved)


def test_saved_background_gives_exactly_the_same_reasons(built, data, pipeline):
    train, holdout = data
    sample = holdout.head(40)
    from_bundle = bundle.load_bundle(built).reasoner().units(sample)
    from_data = Reasoner(pipeline, train).units(sample)
    pd.testing.assert_frame_equal(from_bundle, from_data)


def test_a_background_from_another_model_is_refused(built, data):
    train, _ = data
    other = make_pipeline(LogisticRegression(C=0.01, max_iter=500)).fit(
        train[input_columns()], train["defaulted"]
    )
    saved = bundle.load_bundle(built).explainer
    with pytest.raises(ValueError, match="base value"):
        Reasoner(other, saved=saved)


def test_saved_means_must_cover_every_model_column(built, pipeline):
    saved = bundle.load_bundle(built).explainer
    saved["transformed_mean"].pop(next(iter(saved["transformed_mean"])))
    with pytest.raises(ValueError, match="don't match"):
        Reasoner(pipeline, saved=saved)


def test_reasoner_needs_exactly_one_background(pipeline, data):
    train, _ = data
    with pytest.raises(ValueError, match="exactly one"):
        Reasoner(pipeline)
    with pytest.raises(ValueError, match="exactly one"):
        Reasoner(pipeline, train, saved={})


def test_a_bundle_without_a_background_cannot_give_reasons(built):
    loaded = bundle.load_bundle(built)
    loaded.explainer = None
    with pytest.raises(bundle.BundleError, match="explainer"):
        loaded.reasoner()
