"""Tests for the experiment harness, on a small made-up table and a throwaway MLflow store."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC
from mkopoguard.models import experiment
from mkopoguard.tracking import load_pipeline, setup_tracking


def make_table(n=1_200, seed=0, leak=False):
    """Applicants with a mild real signal in betting share (or a perfect one if leak=True)."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    table["defaulted"] = (rng.random(n) < 0.2).astype(int)
    noise = 0.0 if leak else 2.0
    table["stmt_betting_share"] = table["defaulted"] + rng.normal(0, noise, n)
    return table


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    setup_tracking("test-experiments", root=tmp_path_factory.mktemp("mlruns"))


@pytest.fixture(scope="module")
def data():
    table = make_table()
    return table.iloc[:900].reset_index(drop=True), table.iloc[900:].reset_index(drop=True)


# --- folds ------------------------------------------------------------------------


def test_folds_cover_every_training_row_exactly_once(data):
    train, _ = data
    held_out = np.concatenate([held for _, held in experiment.cv_folds(train)])
    assert sorted(held_out) == list(range(len(train)))


def test_each_fold_has_the_same_default_rate(data):
    train, _ = data
    overall = train["defaulted"].mean()
    for _, held in experiment.cv_folds(train):
        assert train["defaulted"].iloc[held].mean() == pytest.approx(overall, abs=0.03)


# --- cross-validation -----------------------------------------------------------------


def test_cross_validation_is_reproducible_and_complete(data):
    train, _ = data
    columns = experiment.input_columns()
    pipeline = experiment.make_pipeline(LogisticRegression(max_iter=500))
    first = experiment.cross_validate(pipeline, train, columns)
    second = experiment.cross_validate(pipeline, train, columns)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5
    assert {"pr_auc", "roc_auc", "brier"} <= set(first.columns)


# --- groups -------------------------------------------------------------------------


def test_group_without_both_outcomes_gets_missing_metrics():
    frame = pd.DataFrame(
        {
            "defaulted": [0, 1, 0, 1, 0, 0],
            "country_code": ["UGA", "UGA", "UGA", "UGA", "KEN", "KEN"],  # Kenyans never default
            "has_statement": [1, 1, 0, 0, 1, 0],
        }
    )
    table = experiment.evaluate_groups(frame, np.array([0.1, 0.8, 0.2, 0.7, 0.3, 0.4]))
    assert table.loc["all", "roc_auc"] == pytest.approx(1.0)
    assert np.isnan(table.loc["kenya", "roc_auc"])
    assert table.loc["kenya", "n"] == 2


# --- a full run -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result(store, data):
    train, validation = data
    model = LogisticRegression(max_iter=500)
    return experiment.run_experiment("unit-logistic", model, train, validation, "test", {"C": 1})


def test_run_logs_cv_and_validation_metrics(result):
    saved = mlflow.get_run(result["run_id"]).data
    for key in ["cv_pr_auc_mean", "cv_pr_auc_std", "val_pr_auc", "val_kenya_pr_auc", "fit_seconds"]:
        assert key in saved.metrics
    assert saved.params["model"] == "LogisticRegression"
    assert saved.params["C"] == "1"
    assert saved.tags["evaluated_on"] == "validation"


def test_run_saves_tables_and_a_reloadable_model(result, data):
    _, validation = data
    artifacts = {a.path for a in mlflow.MlflowClient().list_artifacts(result["run_id"])}
    assert {"cv_folds.csv", "validation_by_group.csv", "validation_calibration.csv"} <= artifacts
    reloaded = load_pipeline(result["run_id"])
    p = reloaded.predict_proba(validation[experiment.input_columns()])[:, 1]
    np.testing.assert_array_equal(p, result["p_val"])


def test_real_signal_beats_the_dummy(store, data, result):
    train, validation = data
    dummy = DummyClassifier(strategy="prior")
    baseline = experiment.run_experiment("unit-dummy", dummy, train, validation, "test")
    assert baseline["cv_pr_auc"] < result["cv_pr_auc"]
    table = experiment.results_table([baseline, result])
    assert list(table.index) == ["unit-logistic", "unit-dummy"]  # best first
    assert "pipeline" not in table.columns


def test_too_good_to_be_true_stops_the_run(store):
    table = make_table(leak=True)  # betting share IS the answer
    train, validation = table.iloc[:900], table.iloc[900:].reset_index(drop=True)
    with pytest.raises(ValueError, match="leakage"):
        experiment.run_experiment(
            "unit-leak", LogisticRegression(max_iter=500), train, validation, "test"
        )
    runs = mlflow.search_runs(filter_string="attributes.run_name = 'unit-leak'")
    assert runs["status"].tolist() == ["FAILED"]
