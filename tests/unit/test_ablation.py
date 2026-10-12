"""Tests for step 3.7: feature groups, dropping columns, and the ablation rule."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mkopoguard.features.groups import FEATURE_GROUPS, SURVEY_GROUPS, columns_in
from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)
from mkopoguard.models import ablation, experiment
from mkopoguard.tracking import setup_tracking


def make_table(n=3_000, seed=0):
    """Applicants whose default depends on savings club membership (and nothing else)."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    risk = np.where(table["saved_in_savings_club"] == 1, 0.08, 0.35)
    table["defaulted"] = (rng.random(n) < risk).astype(int)
    return table


def model():
    # Fairly strong regularisation, so pure-noise columns barely move the fit
    return LogisticRegression(C=0.05, max_iter=500)


@pytest.fixture(scope="module")
def train():
    return make_table()


# --- groups ----------------------------------------------------------------------------


def test_groups_cover_every_input_column_exactly_once():
    grouped = columns_in(FEATURE_GROUPS)
    assert sorted(grouped) == sorted(input_columns())
    assert len(grouped) == len(set(grouped))


def test_survey_groups_are_real_groups_without_loan_or_statement():
    assert set(SURVEY_GROUPS) <= set(FEATURE_GROUPS)
    assert "loan" not in SURVEY_GROUPS and "statement" not in SURVEY_GROUPS


def test_unknown_group_fails_clearly():
    with pytest.raises(KeyError, match="mood"):
        columns_in(["mood"])


# --- dropping columns ------------------------------------------------------------------


def test_input_columns_leaves_out_dropped_columns_in_order():
    drop = columns_in(["transfers"])
    kept = input_columns(drop=drop)
    assert not set(drop) & set(kept)
    assert kept == [c for c in input_columns() if c not in drop]


def test_reduced_pipeline_works_without_the_dropped_columns(train):
    """The API won't collect dropped answers, so the pipeline must not need them."""
    drop = columns_in(["financial_access", "income_channels", "transfers"])
    pipeline = make_pipeline(model(), drop=drop)
    pipeline.fit(train[input_columns(drop=drop)], train["defaulted"])
    p = pipeline.predict_proba(train[input_columns(drop=drop)].head(5))[:, 1]
    assert len(p) == 5
    out = pipeline.named_steps["prep"].get_feature_names_out()
    assert not any(name.startswith(tuple(drop)) for name in out)


def test_dropping_a_whole_column_type_still_works(train):
    drop = CATEGORICAL  # the categorical branch is left with no columns at all
    pipeline = make_pipeline(model(), drop=drop)
    pipeline.fit(train[input_columns(drop=drop)], train["defaulted"])
    assert pipeline.predict_proba(train[input_columns(drop=drop)])[:, 1].shape == (len(train),)


def test_run_experiment_records_the_smaller_input(train, tmp_path):
    setup_tracking("test-ablation", root=tmp_path)
    drop = columns_in(["transfers"])
    result = experiment.run_experiment(
        "unit-reduced", model(), train.iloc[:450], train.iloc[450:], "test", drop=drop
    )
    params = mlflow.get_run(result["run_id"]).data.params
    assert int(params["n_features_in"]) == len(input_columns()) - len(drop)


# --- scores and the rule -----------------------------------------------------------------


def test_fold_scores_has_one_row_per_fold_for_both_groups(train):
    scores = ablation.fold_scores(train, model())
    assert len(scores) == 5
    assert list(scores.columns) == ["all", "thin_file"]
    assert ((scores >= 0) & (scores <= 1)).all().all()


def test_removing_the_only_real_signal_shows_a_needed_group(train):
    full = ablation.fold_scores(train, model())
    without = ablation.fold_scores(train, model(), drop_groups=["savings_resilience"])
    assert ablation.is_needed(ablation.paired_loss(full, without))


def test_removing_columns_that_carry_no_information_is_not_needed(train):
    blank = train.copy()
    blank[columns_in(["transfers"])] = 0  # everyone gives the same answer
    full = ablation.fold_scores(blank, model())
    without = ablation.fold_scores(blank, model(), drop_groups=["transfers"])
    loss = ablation.paired_loss(full, without)
    assert abs(loss["loss_all"]) < 0.001
    assert not ablation.is_needed(loss)


def test_paired_loss_maths():
    base = pd.DataFrame({"all": [0.40, 0.42, 0.44], "thin_file": [0.30, 0.30, 0.30]})
    other = pd.DataFrame({"all": [0.38, 0.41, 0.41], "thin_file": [0.31, 0.30, 0.29]})
    loss = ablation.paired_loss(base, other)
    assert loss["loss_all"] == pytest.approx(0.02)
    assert loss["se_all"] == pytest.approx(np.std([0.02, 0.01, 0.03], ddof=1) / np.sqrt(3))
    assert loss["folds_worse_all"] == 3
    assert loss["loss_thin_file"] == pytest.approx(0.0)


def make_loss(all_loss, all_se, thin_loss=0.0, thin_se=0.001):
    return {
        "loss_all": all_loss,
        "se_all": all_se,
        "loss_thin_file": thin_loss,
        "se_thin_file": thin_se,
    }


@pytest.mark.parametrize(
    "loss, needed",
    [
        (make_loss(0.010, 0.002), True),  # consistent and meaningful
        (make_loss(0.010, 0.020), False),  # meaningful but inconsistent (noise)
        (make_loss(0.001, 0.0001), False),  # consistent but too small to matter
        (make_loss(-0.003, 0.001), False),  # removing it helps
        (make_loss(0.000, 0.001, thin_loss=0.009, thin_se=0.003), True),  # thin-file needs it
    ],
)
def test_rule_seven(loss, needed):
    assert ablation.is_needed(loss, min_loss=0.002) is needed


# --- validation check and saved selection -----------------------------------------------


def test_paired_bootstrap_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 300), rng.random(300)
    result = ablation.paired_bootstrap(y, p, p, resamples=200)
    assert result["loss"] == 0 and result["low"] == 0 and result["high"] == 0


def test_paired_bootstrap_sees_a_clearly_worse_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    good, bad = y * 0.6 + rng.random(500) * 0.4, rng.random(500)
    result = ablation.paired_bootstrap(y, good, bad, resamples=200)
    assert result["low"] > 0


def test_selection_round_trips_through_the_file(tmp_path):
    path = tmp_path / "feature_selection.json"
    ablation.save_selection(["transfers", "income_channels"], path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["dropped_groups"] == ["transfers", "income_channels"]
    assert "transfers" not in saved["kept_groups"] and "statement" in saved["kept_groups"]
    assert ablation.chosen_drop(path) == columns_in(["transfers", "income_channels"])


def test_missing_selection_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="ablate_features.py"):
        ablation.chosen_drop(tmp_path / "nope.json")
