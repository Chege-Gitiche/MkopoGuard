"""Tests for step 4.2: reason codes always give exactly 3 + 2 readable, truthful reasons."""

import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mkopoguard.explain import design
from mkopoguard.explain import reasons as rc
from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)
from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES

STATEMENT_CODES = {u.code for u in rc.UNITS if u.needs_statement}
UNREADABLE = re.compile(r"\{|\}|nan|None|<NA>")
CATEGORY_VALUES = {
    "education_level": ["primary_or_less", "secondary", "tertiary"],
    "emergency_fund_source": list(rc.EMERGENCY),
    "worried_about_bills": list(rc.WORRY),
}


def make_table(n=1_500, seed=0):
    """Applicants shaped like the real table: thin-file gaps, real categories and ranges."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(CATEGORY_VALUES.get(col, ["a", "b", "c"]), n)
    table["age"] = rng.integers(18, 80, n)
    table["income_quintile"] = rng.integers(1, 6, n)
    table["term_months"] = rng.choice([1, 3, 6], n)
    table["loan_amount_kes"] = rng.integers(2, 100, n) * 500
    table["stmt_betting_share"] = rng.uniform(0, 0.3, n)
    table["stmt_overdraft_borrow_count"] = rng.integers(0, 30, n)
    table["stmt_overdraft_repay_ratio"] = rng.uniform(0, 1, n)
    table["stmt_paybill_months"] = rng.integers(0, 7, n)
    table["stmt_low_balance_share"] = rng.uniform(0, 0.5, n)
    thin = table["has_statement"] == 0
    table.loc[thin, STATEMENT_FEATURES] = np.nan
    betting = table["stmt_betting_share"].fillna(0.05)
    logit = -2.0 + 8 * betting - 1.0 * table["saved_past_year"] + 0.8 * thin
    table["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return table


@pytest.fixture(scope="module")
def table():
    return make_table()


@pytest.fixture(scope="module")
def pipeline(table):
    model = make_pipeline(LogisticRegression(max_iter=2_000))
    return model.fit(table[input_columns()], table["defaulted"])


@pytest.fixture(scope="module")
def reasoner(pipeline, table):
    return rc.Reasoner(pipeline, table)


@pytest.fixture(scope="module")
def results(reasoner, table):
    return reasoner.explain_rows(table.head(400))


# --- the promise: exactly 3 + 2 readable reasons -------------------------------------------


def test_always_exactly_three_raising_and_two_lowering(results):
    for result in results:
        assert len(result["raising"]) == rc.N_RAISING
        assert len(result["lowering"]) == rc.N_LOWERING


def test_reasons_are_all_different(results):
    for result in results:
        codes = [r["code"] for r in result["raising"] + result["lowering"]]
        assert len(set(codes)) == len(codes)


def test_every_reason_is_a_readable_sentence(results):
    for result in results:
        for reason in result["raising"] + result["lowering"]:
            assert reason["text"] and not UNREADABLE.search(reason["text"]), reason
            assert reason["text"][0].isupper()


def test_reasons_are_in_order_of_strength(results):
    for result in results:
        up = [r["contribution"] for r in result["raising"]]
        down = [r["contribution"] for r in result["lowering"]]
        assert up == sorted(up, reverse=True) and down == sorted(down)
        assert min(up) >= max(down)


def test_direction_and_impact_agree_with_the_contribution(results):
    for result in results:
        for reason in result["raising"] + result["lowering"]:
            expected = "raises" if reason["contribution"] > 0 else "lowers"
            assert reason["direction"] == expected
            assert np.sign(reason["impact_points"]) == np.sign(reason["contribution"])


def test_probability_is_the_models_probability(results, pipeline, table):
    expected = pipeline.predict_proba(table.head(400)[input_columns()])[:, 1]
    got = [r["probability_of_default"] for r in results]
    np.testing.assert_allclose(got, expected, atol=1e-10)


# --- statement rules -------------------------------------------------------------------------


def test_thin_file_applicants_never_get_statement_reasons(results, table):
    thin = (table.head(400)["has_statement"] == 0).to_numpy()
    assert thin.any() and (~thin).any()
    for result, is_thin in zip(results, thin, strict=True):
        codes = {r["code"] for r in result["raising"] + result["lowering"]}
        if is_thin:
            assert not codes & STATEMENT_CODES
        else:
            assert "no_statement" not in codes


def test_units_apply_only_where_they_make_sense(reasoner, table):
    units = reasoner.units(table)
    thin = (table["has_statement"] == 0).to_numpy()
    assert units.loc[thin, list(STATEMENT_CODES)].isna().all().all()
    assert units.loc[thin, "no_statement"].notna().all()
    assert units.loc[~thin, "no_statement"].isna().all()


def test_units_and_other_factors_add_up_to_the_score(reasoner, table):
    contribs = reasoner.explainer.contributions(table)
    units = reasoner.units(table)
    np.testing.assert_allclose(units.sum(axis=1), contribs.sum(axis=1), atol=1e-10)


def test_statement_holders_are_compared_with_the_typical_holder(reasoner, table):
    holders = table[table["has_statement"] == 1]
    units = reasoner.units(holders)
    varying = [u.code for u in rc.UNITS if u.needs_statement and u.code != "statement_provided"]
    np.testing.assert_allclose(units[varying].mean(), 0, atol=1e-10)


def test_a_background_without_statement_holders_is_refused(reasoner, table):
    with pytest.raises(ValueError, match="statement holders"):
        rc.statement_shift(reasoner.explainer, table[table["has_statement"] == 0])


# --- only real drivers can be reasons ----------------------------------------------------------


def test_reasons_only_use_simulator_drivers():
    for unit in rc.UNITS:
        for col in unit.columns:
            assert design.is_driver(col) or col == design.STATEMENT_FLAG, col


def test_non_drivers_are_reported_as_other_factors(results):
    for result in results:
        other = result["other_factors"]
        assert set(other) == {"contribution", "impact_points", "material"}


# --- the reasons say the right thing -------------------------------------------------------------


def holder(table):
    return table[table["has_statement"] == 1].head(1).copy()


def test_heavy_betting_becomes_the_top_reason(reasoner, table):
    row = holder(table)
    row["stmt_betting_share"] = 0.3
    result = reasoner.explain_applicant(row)
    assert result["raising"][0]["code"] == "betting"
    assert "30% of outgoing money on betting" in result["raising"][0]["text"]


@pytest.mark.parametrize(
    "code, column, value, text",
    [
        ("emergency_money", "emergency_fund_source", "could_not", "Could not raise emergency"),
        ("saved_past_year", "saved_past_year", 0, "Did not save in the past year"),
        ("saved_past_year", "saved_past_year", np.nan, "Did not say whether they saved"),
        ("bill_worry", "worried_about_bills", "very", "Very worried about paying"),
        ("income_group", "income_quintile", 1, "lowest income group"),
    ],
)
def test_texts_describe_the_applicants_answer(code, column, value, text):
    values = {column: value}
    assert text in rc.BY_CODE[code].text(values, True)


def test_no_overdraft_is_described_as_such():
    values = {"stmt_overdraft_borrow_count": 0, "stmt_overdraft_repay_ratio": 1.0}
    assert "Did not use overdraft" in rc.BY_CODE["overdraft"].text(values, False)


@pytest.mark.parametrize("points, strength", [(8, "major"), (-2, "moderate"), (0.3, "minor")])
def test_strength_labels(points, strength):
    assert rc._strength(points) == strength


def test_explain_applicant_takes_exactly_one_row(reasoner, table):
    with pytest.raises(ValueError, match="one applicant"):
        reasoner.explain_applicant(table.head(2))
