"""Tests for step 3.10 calibration, on made-up data with a known true risk (runs in CI)."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC, input_columns
from mkopoguard.models import calibration as cal
from mkopoguard.models import experiment
from mkopoguard.tracking import load_pipeline, setup_tracking


def make_table(n=3_000, seed=0):
    """Applicants whose risk depends on savings club membership and age, with known true risk."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = rng.choice(["KEN", "UGA", "GHA"], n)
    logit = -2.0 + 1.5 * (table["saved_in_savings_club"] == 0) + 0.8 * np.log(table["age"])
    table["true_p"] = 1 / (1 + np.exp(-logit))
    table["defaulted"] = (rng.random(n) < table["true_p"]).astype(int)
    return table


def plain():
    return LogisticRegression(C=0.05, max_iter=500)


def weighted():
    """Deliberately miscalibrated: defaulters count several times over."""
    return LogisticRegression(C=0.05, max_iter=500, class_weight="balanced")


@pytest.fixture(scope="module")
def train():
    return make_table()


@pytest.fixture(scope="module")
def oof(train):
    return {
        "plain": cal.out_of_fold(train, plain()),
        "weighted": cal.out_of_fold(train, weighted()),
        "weighted_sigmoid": cal.out_of_fold(train, cal.calibrated_model(weighted(), "sigmoid")),
    }


# --- building ------------------------------------------------------------------------------


def test_none_returns_the_model_unchanged():
    model = plain()
    assert cal.calibrated_model(model, "none") is model


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_methods_wrap_the_model(method):
    wrapped = cal.calibrated_model(plain(), method)
    assert isinstance(wrapped, CalibratedClassifierCV) and wrapped.method == method


def test_unknown_method_fails_clearly():
    with pytest.raises(ValueError, match="magic"):
        cal.calibrated_model(plain(), "magic")


# --- out-of-fold predictions -------------------------------------------------------------------


def test_every_applicant_gets_exactly_one_valid_prediction(train, oof):
    p = oof["plain"]
    assert len(p) == len(train)
    assert not np.isnan(p).any() and ((p > 0) & (p < 1)).all()


def test_out_of_fold_is_reproducible(train, oof):
    np.testing.assert_array_equal(cal.out_of_fold(train, plain()), oof["plain"])


# --- measuring calibration ---------------------------------------------------------------------


def test_true_probabilities_have_slope_one_and_intercept_zero(train):
    slope, intercept = cal.calibration_slope(train["defaulted"], train["true_p"])
    assert slope == pytest.approx(1.0, abs=0.15)
    assert intercept == pytest.approx(0.0, abs=0.15)


def test_over_confident_probabilities_have_a_slope_below_one(train):
    p = train["true_p"].to_numpy()
    too_extreme = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))  # logit doubled
    slope, _ = cal.calibration_slope(train["defaulted"], too_extreme)
    assert slope == pytest.approx(0.5, abs=0.1)


def test_positive_control_class_weights_are_miscalibrated_and_platt_fixes_them(train, oof):
    y = train["defaulted"]
    raw, fixed = cal.summary(y, oof["weighted"]), cal.summary(y, oof["weighted_sigmoid"])
    assert raw["ece"] > 0.10  # predictions far too high
    assert raw["mean_predicted"] > raw["default_rate"] + 0.1
    assert fixed["ece"] < 0.04
    assert fixed["mean_predicted"] == pytest.approx(fixed["default_rate"], abs=0.02)


def test_calibration_gain_is_positive_when_the_fix_helps(train, oof):
    gain = cal.calibration_gain(
        cal.fold_metrics(train, oof["weighted"]), cal.fold_metrics(train, oof["weighted_sigmoid"])
    )
    assert gain["ece_gain"] > 0.05 and gain["brier_gain"] > 0
    assert cal.worth_calibrating(gain)


def make_gain(ece_gain, ece_se=0.001, brier_gain=0.0, brier_se=0.0005, pr_auc_gain=0.0):
    return {
        "ece_gain": ece_gain,
        "ece_gain_se": ece_se,
        "brier_gain": brier_gain,
        "brier_gain_se": brier_se,
        "pr_auc_gain": pr_auc_gain,
    }


@pytest.mark.parametrize(
    "gain, worth",
    [
        (make_gain(0.020), True),  # clear, meaningful cut
        (make_gain(0.003), False),  # too small to matter
        (make_gain(0.020, ece_se=0.030), False),  # inconsistent across folds
        (make_gain(0.020, brier_gain=-0.01), False),  # Brier gets worse
        (make_gain(0.020, pr_auc_gain=-0.01), False),  # ranking gets worse
    ],
)
def test_rule_eight(gain, worth):
    assert cal.worth_calibrating(gain, min_ece_gain=0.005, max_pr_auc_loss=0.002) is worth


# --- groups --------------------------------------------------------------------------------------


def test_flag_bar_rises_with_the_number_of_groups():
    assert cal.flag_z(1) == pytest.approx(1.96, abs=0.01)
    assert cal.flag_z(20) == pytest.approx(3.02, abs=0.01)


def test_well_calibrated_groups_are_not_flagged(train):
    table = cal.group_calibration(train, train["true_p"].to_numpy(), "country_code")
    assert set(table.index) == {"KEN", "UGA", "GHA"}
    assert table["n"].sum() == len(train)
    assert not table["flagged"].any()


def test_a_group_with_biased_predictions_is_flagged(train):
    p = train["true_p"].to_numpy().copy()
    kenya = (train["country_code"] == "KEN").to_numpy()
    p[kenya] = p[kenya] * 0.5  # predict half the real risk for Kenyans
    table = cal.group_calibration(train, p, "country_code")
    assert table.loc["KEN", "flagged"] and table.loc["KEN", "observed_to_expected"] > 1.5
    assert not table.loc[["UGA", "GHA"], "flagged"].any()


# --- saving -------------------------------------------------------------------------------------


def test_calibrated_pipeline_is_logged_and_reloads_identically(train, tmp_path):
    setup_tracking("test-calibration", root=tmp_path)
    model = cal.calibrated_model(plain(), "sigmoid")
    result = experiment.run_experiment(
        "unit-sigmoid", model, train.iloc[:2_400], train.iloc[2_400:], "test"
    )
    assert mlflow.get_run(result["run_id"]).info.status == "FINISHED"
    reloaded = load_pipeline(result["run_id"])
    np.testing.assert_array_equal(
        reloaded.predict_proba(train.iloc[2_400:][input_columns()])[:, 1], result["p_val"]
    )


def test_choice_round_trips_through_the_file(tmp_path):
    path = tmp_path / "calibration.json"
    cal.save_choice("isotonic", path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"method": "isotonic"}
    assert cal.chosen_calibration(path) == "isotonic"


def test_bad_or_missing_choice_fails_clearly(tmp_path):
    with pytest.raises(ValueError, match="magic"):
        cal.save_choice("magic", tmp_path / "calibration.json")
    with pytest.raises(FileNotFoundError, match="calibrate.py"):
        cal.chosen_calibration(tmp_path / "nope.json")
