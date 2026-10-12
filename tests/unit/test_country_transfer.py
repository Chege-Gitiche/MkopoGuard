"""Tests for step 3.9 leave-one-country-out, on made-up countries with known differences."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from mkopoguard.features.preprocess import BINARY, CATEGORICAL, NUMERIC
from mkopoguard.models import country_transfer as ct
from mkopoguard.models.ablation import paired_bootstrap

COUNTRIES = ["BEN", "GHA", "KEN", "SEN", "TZA", "UGA"]


def make_table(kenya="same", n_per_country=2_000, seed=0):
    """Six countries. Everywhere, savings club members default less. In Kenya:
    same     - the same rule (the model should transfer)
    shifted  - every Kenyan is riskier (a base-rate difference the inputs can't see)
    own_rule - risk follows internet access instead of savings (a pattern only Kenya has)
    """
    rng = np.random.default_rng(seed)
    n = n_per_country * len(COUNTRIES)
    table = pd.DataFrame({col: rng.lognormal(0, 1, n) for col in NUMERIC})
    for col in BINARY:
        table[col] = rng.integers(0, 2, n)
    for col in CATEGORICAL:
        table[col] = rng.choice(["a", "b", "c"], n)
    table["applicant_id"] = [f"A-{i:05d}" for i in range(1, n + 1)]
    table["country_code"] = np.repeat(COUNTRIES, n_per_country)
    kenyan = (table["country_code"] == "KEN").to_numpy()
    logit = -2.0 + 1.8 * (table["saved_in_savings_club"] == 0).to_numpy()
    if kenya == "shifted":
        logit = logit + 1.0 * kenyan
    if kenya == "own_rule":
        own = -2.0 + 3.0 * (table["has_internet_access"] == 1).to_numpy()
        logit = np.where(kenyan, own, logit)
    table["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return table


def model():
    return LogisticRegression(C=0.05, max_iter=500)


@pytest.fixture(scope="module")
def same():
    return ct.seen_and_unseen(make_table("same"), model)


# --- predictions --------------------------------------------------------------------------


def test_every_applicant_is_scored_both_ways(same):
    assert len(same) == 2_000 * len(COUNTRIES)
    for column in ("p_seen", "p_unseen"):
        assert not same[column].isna().any()
        assert same[column].between(0, 1).all()


def test_predictions_are_reproducible(same):
    again = ct.seen_and_unseen(make_table("same"), model)
    pd.testing.assert_frame_equal(same, again)


def test_unseen_model_really_never_saw_the_country():
    """If Kenya's own rule only exists in Kenya, the never-seen model can't have learned it."""
    preds = ct.seen_and_unseen(make_table("own_rule"), model)
    table = ct.country_table(preds, resamples=300)
    assert table.loc["KEN", "loss"] > 0.05
    assert table.loc["KEN", "clear_loss"]


# --- rule 9 on known answers -------------------------------------------------------------------


def test_same_rule_everywhere_transfers(same):
    table, calibration = ct.country_table(same, resamples=300), ct.unseen_calibration(same)
    assert set(table.index) == set(COUNTRIES)
    assert not table["clear_loss"].any()
    assert not calibration["flagged"].any()
    assert ct.transfers(table, calibration)


def test_a_country_with_hidden_extra_risk_is_flagged_for_calibration():
    preds = ct.seen_and_unseen(make_table("shifted"), model)
    calibration = ct.unseen_calibration(preds)
    assert calibration.loc["KEN", "flagged"]
    assert calibration.loc["KEN", "observed_to_expected"] > 1.2  # more defaults than predicted
    assert not ct.transfers(ct.country_table(preds, resamples=300), calibration)


def test_clear_loss_needs_a_big_enough_loss(same):
    strict = ct.country_table(same, min_loss=-1.0, resamples=100)  # any loss beyond chance
    lenient = ct.country_table(same, min_loss=1.0, resamples=100)  # impossible to lose
    assert not lenient["clear_loss"].any()
    assert (strict["clear_loss"] == (strict["loss_low"] > 0)).all()


def test_transfers_needs_both_conditions():
    ok_table = pd.DataFrame({"clear_loss": [False, False]})
    bad_table = pd.DataFrame({"clear_loss": [False, True]})
    ok_calibration = pd.DataFrame({"flagged": [False, False]})
    bad_calibration = pd.DataFrame({"flagged": [True, False]})
    assert ct.transfers(ok_table, ok_calibration)
    assert not ct.transfers(bad_table, ok_calibration)
    assert not ct.transfers(ok_table, bad_calibration)


# --- the widened interval ------------------------------------------------------------------------


def test_a_higher_level_gives_a_wider_interval():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    a, b = y * 0.3 + rng.random(400), rng.random(400)
    usual = paired_bootstrap(y, a, b, resamples=500)
    wide = paired_bootstrap(y, a, b, resamples=500, level=1 - 0.05 / 20)
    assert wide["low"] < usual["low"] and wide["high"] > usual["high"]
    assert wide["loss"] == usual["loss"]
