"""Tracking tests. Each uses its own throwaway MLflow store, never the project's mlruns/."""

import subprocess
from pathlib import PureWindowsPath

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression

from mkopoguard import tracking
from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlruns")
    experiment_id = tracking.setup_tracking("test-experiment", root=root)
    return root, experiment_id


@pytest.fixture(scope="module")
def fitted():
    """A real (tiny) MkopoGuard pipeline, so saving exercises the custom transformers."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        X[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        X[col] = rng.choice(["a", "b", "c"], n)
    y = (rng.random(n) < 0.3).astype(int)
    pipeline = make_pipeline(LogisticRegression(max_iter=500)).fit(X[input_columns()], y)
    return pipeline, X[input_columns()]


# --- setup ---------------------------------------------------------------------


def test_windows_paths_become_valid_sqlite_uris():
    uri = tracking.tracking_uri(PureWindowsPath(r"D:\MkopoGuard\mlruns"))
    assert uri == "sqlite:///D:/MkopoGuard/mlruns/mlflow.db"


def test_setup_creates_the_experiment_inside_the_store(store):
    root, experiment_id = store
    experiment = mlflow.get_experiment(experiment_id)
    assert experiment.name == "test-experiment"
    assert (root / "mlflow.db").exists()
    assert experiment.artifact_location == (root / "artifacts").resolve().as_uri()


def test_setup_twice_reuses_the_same_experiment(store):
    root, experiment_id = store
    assert tracking.setup_tracking("test-experiment", root=root) == experiment_id


def test_telemetry_is_off():
    import os

    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"


# --- runs ----------------------------------------------------------------------


def test_run_records_provenance_params_and_metrics(store):
    with tracking.start_run("unit", evaluated_on="validation", step="3.1", model="lr") as run:
        mlflow.log_param("C", 1.0)
        mlflow.log_metric("roc_auc", 0.71)
    saved = mlflow.get_run(run.info.run_id).data
    for key in ["git_commit", "split_file_md5", "features_file_md5", "split_seed"]:
        assert key in saved.tags
    assert saved.tags["evaluated_on"] == "validation"
    assert saved.tags["guide_step"] == "3.1"
    assert saved.tags["model"] == "lr"
    assert saved.params["C"] == "1.0"
    assert saved.metrics["roc_auc"] == pytest.approx(0.71)


def test_unknown_evaluation_split_is_rejected(store):
    with pytest.raises(ValueError, match="evaluated_on"):
        with tracking.start_run("bad", evaluated_on="holdout", step="3.1"):
            pass


# --- saving models ----------------------------------------------------------------


def test_saved_pipeline_reloads_and_predicts_identically(store, fitted):
    pipeline, X = fitted
    with tracking.start_run("save", evaluated_on="train", step="3.1") as run:
        tracking.log_pipeline(pipeline)
    reloaded = tracking.load_pipeline(run.info.run_id)
    np.testing.assert_array_equal(reloaded.predict_proba(X), pipeline.predict_proba(X))


def test_untrusted_types_are_refused_without_the_allow_list(store, fitted):
    """The safe format must refuse our custom code unless we explicitly trust it."""
    pipeline, _ = fitted
    with tracking.start_run("untrusted", evaluated_on="train", step="3.1"):
        with pytest.raises(MlflowException, match="untrusted types"):
            mlflow.sklearn.log_model(pipeline, name="model")


# --- provenance helpers -----------------------------------------------------------


def test_file_fingerprint(tmp_path):
    path = tmp_path / "f.txt"
    path.write_bytes(b"mkopoguard")
    assert tracking.file_fingerprint(path) == "d54c951b26f5"
    assert tracking.file_fingerprint(tmp_path / "nope") == "missing"


def test_git_commit_is_unknown_when_git_fails(monkeypatch):
    def broken(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(tracking.subprocess, "run", broken)
    assert tracking.git_commit() == "unknown"


def test_git_commit_marks_uncommitted_changes(monkeypatch):
    outputs = iter(["abc1234", " M src/mkopoguard/config.py"])
    monkeypatch.setattr(tracking, "_git", lambda *args: next(outputs))
    assert tracking.git_commit() == "abc1234-dirty"
