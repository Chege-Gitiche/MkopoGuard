"""Tests for step 4.1: exact SHAP for the linear champion, and the known-answer check."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from mkopoguard.explain import design
from mkopoguard.explain.linear_shap import LinearShap, combine, importance, to_probability
from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)


def make_table(n=1_000, seed=0):
    """Applicants whose risk depends on betting share and savings club membership only."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["stmt_betting_share"] = rng.uniform(0, 0.3, n)
    logit = -2.0 + 8 * table["stmt_betting_share"] - 1.0 * table["saved_in_savings_club"]
    table["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return table


@pytest.fixture(scope="module")
def table():
    return make_table()


@pytest.fixture(scope="module")
def pipeline(table):
    model = make_pipeline(LogisticRegression(max_iter=1_000))
    return model.fit(table[input_columns()], table["defaulted"])


@pytest.fixture(scope="module")
def explainer(pipeline, table):
    return LinearShap(pipeline, table)


# --- exactness ----------------------------------------------------------------------------


def test_contributions_add_up_to_the_models_log_odds(explainer, pipeline, table):
    contribs = explainer.contributions(table.head(200))
    expected = pipeline.decision_function(table.head(200)[input_columns()])
    np.testing.assert_allclose(explainer.log_odds(contribs), expected, atol=1e-10)
    np.testing.assert_allclose(
        to_probability(explainer.log_odds(contribs)),
        pipeline.predict_proba(table.head(200)[input_columns()])[:, 1],
        atol=1e-10,
    )


def test_matches_the_shap_library(explainer, pipeline, table):
    shap = pytest.importorskip("shap")
    prep, lr = pipeline.named_steps["prep"], pipeline.named_steps["model"]
    background = prep.transform(table[input_columns()]).to_numpy()
    masker = shap.maskers.Independent(background, max_samples=len(background))
    theirs = shap.LinearExplainer(lr, masker).shap_values(background[:100])
    ours = explainer.transformed_contributions(table.head(100)).to_numpy()
    np.testing.assert_allclose(ours, theirs, atol=1e-10)


def test_average_contribution_over_the_background_is_zero(explainer, table):
    contribs = explainer.contributions(table)
    np.testing.assert_allclose(contribs.mean(), 0, atol=1e-10)


# --- mapping to inputs ------------------------------------------------------------------------


def test_every_input_column_gets_one_contribution(explainer, table):
    contribs = explainer.contributions(table.head(5))
    assert list(contribs.columns) == input_columns()


def test_one_hot_columns_are_added_back_into_their_question(explainer, table):
    parts = explainer.transformed_contributions(table.head(5))
    one_hot = [c for c in parts.columns if c.startswith("education_level_")]
    assert len(one_hot) >= 3
    np.testing.assert_allclose(
        explainer.contributions(table.head(5))["education_level"], parts[one_hot].sum(axis=1)
    )


def test_changing_one_answer_only_changes_that_answers_contribution(explainer, table):
    before = table.head(1).copy()
    after = before.copy()
    after["stmt_betting_share"] = before["stmt_betting_share"] + 0.2
    diff = (explainer.contributions(after) - explainer.contributions(before)).iloc[0]
    assert diff["stmt_betting_share"] > 0  # more betting, more risk
    assert (diff.drop("stmt_betting_share").abs() < 1e-12).all()


def test_the_real_signal_is_the_most_important(explainer, table):
    ranking = importance(explainer.contributions(table))
    assert ranking.index[0] == "stmt_betting_share"
    assert ranking.is_monotonic_decreasing


def test_combine_adds_up_groups_and_skips_empty_ones(explainer, table):
    contribs = explainer.contributions(table.head(10))
    groups = {"loan": ["loan_amount_kes", "term_months"], "gone": ["not_a_column"]}
    combined = combine(contribs, groups)
    np.testing.assert_allclose(
        combined["loan"], contribs["loan_amount_kes"] + contribs["term_months"]
    )
    assert "gone" not in combined and "loan_amount_kes" not in combined
    np.testing.assert_allclose(combined.sum(axis=1), contribs.sum(axis=1))


def test_a_non_linear_model_is_refused(table):
    forest = make_pipeline(RandomForestClassifier(n_estimators=5, random_state=0))
    forest.fit(table[input_columns()], table["defaulted"])
    with pytest.raises(TypeError, match="linear"):
        LinearShap(forest, table)


# --- the design check ----------------------------------------------------------------------------


def test_every_driver_is_a_real_input_column():
    assert set(design.DRIVERS) <= set(input_columns())
    assert design.STATEMENT_FLAG in input_columns()


def ranking(**scores):
    return pd.Series(scores).sort_values(ascending=False)


def test_check_passes_when_drivers_lead():
    drivers = {c: 1.0 - i * 0.01 for i, c in enumerate(list(design.DRIVERS)[:12])}
    result = design.check_against_design(ranking(**drivers, education_level=0.01))
    assert result["passed"] and result["top_non_drivers"] == []


def test_check_fails_when_a_non_driver_is_near_the_top():
    drivers = {c: 0.5 for c in list(design.DRIVERS)[:12]}
    result = design.check_against_design(ranking(**drivers, education_level=0.9))
    assert not result["passed"] and result["top_non_drivers"] == ["education_level"]


def test_check_fails_when_non_drivers_carry_too_much_weight():
    drivers = {c: 0.2 for c in list(design.DRIVERS)[:12]}  # still lead the ranking...
    heavy = {"term_months": 0.15, "education_level": 0.15, "borrowed_past_year": 0.15}
    result = design.check_against_design(ranking(**drivers, **heavy))
    assert result["top_inputs_all_drivers"]
    # ...but together the non-drivers carry too much of the importance
    assert result["non_driver_share"] > design.MAX_NON_DRIVER_SHARE
    assert not result["passed"]


def test_statement_flag_is_set_aside_not_counted():
    drivers = {c: 0.5 for c in list(design.DRIVERS)[:12]}
    result = design.check_against_design(ranking(**drivers, has_statement=0.9))
    assert result["passed"]
