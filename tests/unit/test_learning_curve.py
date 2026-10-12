"""Tests for step 3.8 learning curves, on a small made-up table (runs in CI)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC
from mkopoguard.models import learning_curve as lc

FRACTIONS = (0.25, 0.5, 1.0)


def make_table(n=2_000, seed=0):
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


def model():
    return LogisticRegression(C=0.05, max_iter=500)


@pytest.fixture(scope="module")
def train():
    return make_table()


@pytest.fixture(scope="module")
def points(train):
    return lc.curve_points(train, model, fractions=FRACTIONS, repeats=2)


# --- subsampling ---------------------------------------------------------------------------


def test_subsample_is_a_stratified_subset_of_the_rows_given():
    rng = np.random.default_rng(0)
    groups = rng.choice(["a", "b"], 1_000, p=[0.8, 0.2])
    rows = np.arange(0, 1_000, 2)  # only even rows are allowed
    picked = lc.subsample(rows, groups, 0.25, seed=1)
    assert set(picked) <= set(rows)
    assert len(picked) == pytest.approx(len(rows) * 0.25, abs=1)
    assert (groups[picked] == "b").mean() == pytest.approx((groups[rows] == "b").mean(), abs=0.02)


def test_subsample_is_reproducible_and_seed_dependent():
    groups = np.array(["a", "b"] * 200)
    rows = np.arange(400)
    first, again = lc.subsample(rows, groups, 0.5, 3), lc.subsample(rows, groups, 0.5, 3)
    other = lc.subsample(rows, groups, 0.5, 4)
    assert np.array_equal(first, again) and not np.array_equal(first, other)


def test_full_fraction_returns_every_row():
    rows = np.arange(50)
    assert np.array_equal(lc.subsample(rows, np.zeros(50), 1.0, 0), rows)


# --- curve points ------------------------------------------------------------------------


def test_one_point_per_fold_fraction_and_repeat(points):
    # 5 folds x (2 repeats for each smaller fraction + 1 for the full set)
    assert len(points) == 5 * (2 + 2 + 1)
    assert set(points["fraction"]) == set(FRACTIONS)


def test_training_size_doubles_with_each_fraction(points):
    sizes = points.groupby("fraction")["n_train"].mean()
    assert sizes.loc[0.5] == pytest.approx(2 * sizes.loc[0.25], rel=0.01)
    assert sizes.loc[1.0] == pytest.approx(2 * sizes.loc[0.5], rel=0.01)


def test_scores_are_valid_and_reproducible(train, points):
    assert points[["cv_pr_auc", "cv_pr_auc_thin_file", "train_pr_auc"]].stack().between(0, 1).all()
    again = lc.curve_points(train, model, fractions=FRACTIONS, repeats=2)
    pd.testing.assert_frame_equal(points, again)


def test_overfitting_gap_shrinks_with_more_data(points):
    table = lc.summarise(points)
    assert table["overfit_gap"].iloc[0] > table["overfit_gap"].iloc[-1]


# --- projection and verdict ------------------------------------------------------------------


def curve(*scores):
    return pd.DataFrame({"cv_pr_auc": scores})


def test_projection_shrinks_like_the_last_doubling():
    gain = lc.projected_gain(curve(0.30, 0.34, 0.36))  # gains 0.04 then 0.02
    assert gain["last_gain"] == pytest.approx(0.02)
    assert gain["next_gain"] == pytest.approx(0.01)  # halves again


def test_projection_never_predicts_speeding_up():
    gain = lc.projected_gain(curve(0.30, 0.31, 0.34))  # a lucky jump
    assert gain["next_gain"] == pytest.approx(0.03)  # capped at the last gain


def test_no_gain_projects_no_gain():
    assert lc.projected_gain(curve(0.30, 0.35, 0.34))["next_gain"] == 0.0


@pytest.mark.parametrize("next_gain, helps", [(0.010, True), (0.002, False), (0.0, False)])
def test_more_data_helps_only_above_the_minimum(next_gain, helps):
    assert lc.more_data_helps({"next_gain": next_gain}, min_gain=0.002) is helps


# --- ceiling -------------------------------------------------------------------------------


def test_true_probabilities_beat_a_trained_model(train, points):
    top = lc.ceiling(train, train["true_p"].to_numpy())
    assert top["all"] > lc.summarise(points)["cv_pr_auc"].iloc[-1]
    assert 0 < top["thin_file"] <= 1
