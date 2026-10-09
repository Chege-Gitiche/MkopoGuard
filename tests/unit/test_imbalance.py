"""Tests for step 3.5: resampling (SMOTE) must only ever touch training data."""

import mlflow
import numpy as np
import pandas as pd
import pytest
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    NUMERIC,
    input_columns,
    make_pipeline,
)
from mkopoguard.models import experiment
from mkopoguard.tracking import load_pipeline, setup_tracking


def make_table(n=1_000, seed=0):
    """Applicants with a mild real signal in betting share, about 20% defaulters."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    table["defaulted"] = (rng.random(n) < 0.2).astype(int)
    table["stmt_betting_share"] = table["defaulted"] + rng.normal(0, 2.0, n)
    return table


class RecordingSMOTE(SMOTE):
    """SMOTE that remembers how many rows it was given each time it resampled."""

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.seen = []

    def _fit_resample(self, X, y):
        self.seen.append(len(X))
        return super()._fit_resample(X, y)


@pytest.fixture(scope="module")
def data():
    table = make_table()
    return table.iloc[:800].reset_index(drop=True), table.iloc[800:].reset_index(drop=True)


def test_no_sampler_gives_the_plain_pipeline():
    pipeline = make_pipeline(LogisticRegression())
    assert type(pipeline) is Pipeline
    assert list(pipeline.named_steps) == ["prep", "model"]


def test_sampler_sits_between_preprocessing_and_model():
    pipeline = make_pipeline(LogisticRegression(), sampler=SMOTE(random_state=0))
    assert isinstance(pipeline, ImbPipeline)
    assert list(pipeline.named_steps) == ["prep", "sampler", "model"]


def test_smote_balances_the_training_rows_it_is_given(data):
    train, _ = data
    columns = input_columns()
    pipeline = make_pipeline(LogisticRegression(max_iter=500))
    X = pipeline.named_steps["prep"].fit_transform(train[columns])
    _, y_res = SMOTE(random_state=0).fit_resample(X, train["defaulted"])
    assert y_res.mean() == pytest.approx(0.5)
    assert len(y_res) > len(train)


def test_prediction_is_never_resampled(data):
    train, validation = data
    columns = input_columns()
    sampler = RecordingSMOTE(random_state=0)
    pipeline = make_pipeline(LogisticRegression(max_iter=500), sampler=sampler)
    pipeline.fit(train[columns], train["defaulted"])
    p = pipeline.predict_proba(validation[columns])[:, 1]
    assert len(p) == len(validation)  # one score per real applicant, no invented ones
    assert sampler.seen == [len(train)]  # resampled once, while fitting, on training rows only


def test_each_cv_fold_resamples_only_its_own_training_rows(data):
    train, _ = data
    columns = input_columns()
    for fit_rows, held_out in experiment.cv_folds(train):
        sampler = RecordingSMOTE(random_state=0)
        fold = make_pipeline(LogisticRegression(max_iter=500), sampler=sampler)
        fold.fit(train[columns].iloc[fit_rows], train["defaulted"].iloc[fit_rows])
        p = fold.predict_proba(train[columns].iloc[held_out])[:, 1]
        assert sampler.seen == [len(fit_rows)]  # the held-out fold is scored as it really is
        assert len(p) == len(held_out)


def test_smote_is_reproducible(data):
    train, validation = data
    columns = input_columns()
    scores = []
    for _ in range(2):
        pipeline = make_pipeline(LogisticRegression(max_iter=500), sampler=SMOTE(random_state=0))
        pipeline.fit(train[columns], train["defaulted"])
        scores.append(pipeline.predict_proba(validation[columns])[:, 1])
    np.testing.assert_array_equal(scores[0], scores[1])


def test_smote_pipeline_is_logged_and_reloads_identically(data, tmp_path):
    setup_tracking("test-imbalance", root=tmp_path)
    train, validation = data
    result = experiment.run_experiment(
        "unit-smote",
        LogisticRegression(max_iter=500),
        train,
        validation,
        "test",
        {"imbalance": "smote"},
        sampler=SMOTE(random_state=0),
    )
    params = mlflow.get_run(result["run_id"]).data.params
    assert params["sampler"] == "SMOTE"
    assert params["imbalance"] == "smote"
    reloaded = load_pipeline(result["run_id"])
    np.testing.assert_array_equal(
        reloaded.predict_proba(validation[input_columns()])[:, 1], result["p_val"]
    )


def test_runs_without_a_sampler_say_so(data, tmp_path):
    setup_tracking("test-imbalance-none", root=tmp_path)
    train, validation = data
    result = experiment.run_experiment(
        "unit-none", LogisticRegression(max_iter=500), train, validation, "test"
    )
    assert mlflow.get_run(result["run_id"]).data.params["sampler"] == "none"
