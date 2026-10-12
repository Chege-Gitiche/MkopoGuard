"""Tests for step 3.6 tuning, on a small made-up table (runs in CI, a few seconds)."""

import json

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC
from mkopoguard.models import tuning


def make_table(n=600, seed=0):
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


@pytest.fixture(scope="module")
def train():
    return make_table()


@pytest.fixture(scope="module")
def study(train):
    return tuning.tune("logistic", train, n_trials=4, seed=1)


# --- search spaces ----------------------------------------------------------------------


@pytest.mark.parametrize("name", ["logistic", "xgboost"])
def test_every_suggestion_builds_a_model_that_trains(name):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(80, 4)), np.tile([0, 0, 0, 1], 20)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(8):
        trial = study.ask()
        model = tuning.build_model(name, tuning.suggest_params(trial, name))
        p = model.fit(X, y).predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        study.tell(trial, 0.0)


def test_xgboost_space_includes_additive_depth_one_trees():
    trial = optuna.trial.FixedTrial(
        {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 1,
            "min_child_weight": 1.0,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
        }
    )
    assert tuning.suggest_params(trial, "xgboost")["max_depth"] == 1


def test_defaults_are_inside_their_search_space():
    for name, defaults in tuning.DEFAULTS.items():
        trial = optuna.trial.FixedTrial(defaults)
        assert tuning.suggest_params(trial, name) == defaults  # FixedTrial rejects out-of-range


def test_unknown_model_fails_clearly():
    with pytest.raises(ValueError, match="random_forest"):
        tuning.build_model("random_forest", {})
    with pytest.raises(ValueError, match="random_forest"):
        tuning.suggest_params(optuna.trial.FixedTrial({}), "random_forest")


def test_build_model_returns_fresh_models_of_the_right_type():
    assert isinstance(
        tuning.build_model("logistic", {"C": 0.1, "l1_ratio": 0.5}), LogisticRegression
    )
    first = tuning.build_model("xgboost", tuning.DEFAULTS["xgboost"])
    second = tuning.build_model("xgboost", tuning.DEFAULTS["xgboost"])
    assert isinstance(first, XGBClassifier) and first is not second


# --- studies ----------------------------------------------------------------------------


def test_trial_zero_is_the_untuned_default(study):
    assert study.trials[0].params == tuning.DEFAULTS["logistic"]


def test_scores_are_valid_pr_aucs_with_fold_std(study):
    for trial in study.trials:
        assert 0.0 <= trial.value <= 1.0
        assert trial.user_attrs["cv_pr_auc_std"] >= 0.0


def test_same_seed_gives_the_same_study(train, study):
    again = tuning.tune("logistic", train, n_trials=4, seed=1)
    assert [t.params for t in again.trials] == [t.params for t in study.trials]
    assert [t.value for t in again.trials] == [t.value for t in study.trials]


def test_trials_table_has_one_row_per_trial_and_a_rising_best(study):
    table = tuning.trials_table("logistic", study)
    assert len(table) == len(study.trials)
    assert table["best_so_far"].is_monotonic_increasing
    assert table["best_so_far"].iloc[-1] == pytest.approx(study.best_value)


# --- tie rule and saved settings ----------------------------------------------------------


def fake_study(default_value, best_value, std):
    """A study with a hand-made default trial and best trial."""
    study = optuna.create_study(direction="maximize")
    for value, params in (
        (default_value, {"C": 1.0, "l1_ratio": 0.0}),
        (best_value, {"C": 0.05, "l1_ratio": 0.5}),
    ):
        trial = optuna.trial.create_trial(
            params=params,
            distributions={
                "C": optuna.distributions.FloatDistribution(1e-3, 3.0, log=True),
                "l1_ratio": optuna.distributions.FloatDistribution(0.0, 1.0),
            },
            value=value,
            user_attrs={"cv_pr_auc_std": std},
        )
        study.add_trial(trial)
    return study


def test_small_gain_is_a_tie_and_keeps_defaults():
    summary = tuning.summarise("logistic", fake_study(0.390, 0.395, std=0.03))
    assert not summary["worth_it"]
    assert summary["chosen_params"] == tuning.DEFAULTS["logistic"]


def test_gain_bigger_than_one_std_adopts_tuned_settings():
    summary = tuning.summarise("logistic", fake_study(0.390, 0.450, std=0.03))
    assert summary["worth_it"]
    assert summary["chosen_params"] == {"C": 0.05, "l1_ratio": 0.5}


def test_chosen_settings_round_trip_through_the_file(tmp_path):
    path = tmp_path / "tuned_params.json"
    summary = tuning.summarise("logistic", fake_study(0.390, 0.450, std=0.03))
    tuning.save_chosen([summary], path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["logistic"] == {"params": {"C": 0.05, "l1_ratio": 0.5}, "tuned": True}
    model = tuning.chosen_model("logistic", path)
    assert model.C == 0.05 and model.l1_ratio == 0.5


def test_chosen_model_for_untuned_name_fails_clearly(tmp_path):
    path = tmp_path / "tuned_params.json"
    tuning.save_chosen([tuning.summarise("logistic", fake_study(0.39, 0.39, 0.03))], path)
    with pytest.raises(KeyError, match="tune_models.py"):
        tuning.chosen_model("xgboost", path)
