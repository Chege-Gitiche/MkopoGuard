"""Every candidate model must be reproducible and must survive saving to MLflow."""

import mlflow
import numpy as np
import pandas as pd
import pytest

from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)
from mkopoguard.models.candidates import CANDIDATES, logistic
from mkopoguard.tracking import load_pipeline, log_pipeline, setup_tracking

ALL_MODELS = [("logistic", logistic, {})] + CANDIDATES
IDS = [name for name, _, _ in ALL_MODELS]


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    X = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        X[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        X[col] = rng.choice(["a", "b", "c"], n)
    y = (rng.random(n) < 0.2).astype(int)
    X["stmt_betting_share"] += y  # a real signal to learn
    return X[input_columns()], y


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    setup_tracking("test-candidates", root=tmp_path_factory.mktemp("mlruns"))


@pytest.mark.parametrize("name, factory, params", ALL_MODELS, ids=IDS)
def test_factory_builds_a_fresh_model_each_time(name, factory, params):
    assert factory() is not factory()
    model = factory()
    for key, value in params.items():
        assert model.get_params()[key] == value


@pytest.mark.parametrize("name, factory, params", ALL_MODELS, ids=IDS)
def test_same_seed_gives_identical_predictions(name, factory, params, data):
    X, y = data
    first = make_pipeline(factory()).fit(X, y).predict_proba(X)
    second = make_pipeline(factory()).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("name, factory, params", ALL_MODELS, ids=IDS)
def test_model_saves_and_reloads_identically(name, factory, params, data, store):
    """Fails if a model type is missing from SKOPS_TRUSTED_TYPES."""
    X, y = data
    pipeline = make_pipeline(factory()).fit(X, y)
    with mlflow.start_run(run_name=name) as run:
        log_pipeline(pipeline)
    reloaded = load_pipeline(run.info.run_id)
    np.testing.assert_array_equal(reloaded.predict_proba(X), pipeline.predict_proba(X))


@pytest.mark.parametrize("name, factory, params", ALL_MODELS, ids=IDS)
def test_model_learns_a_real_signal(name, factory, params, data):
    X, y = data
    p = make_pipeline(factory()).fit(X, y).predict_proba(X)[:, 1]
    assert p[y == 1].mean() > p[y == 0].mean()
