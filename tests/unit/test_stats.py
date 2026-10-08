import numpy as np
import pandas as pd
import pytest

from mkopoguard.stats import pool_average, weighted_share


def make_survey(values, weights, country="KEN", group=None):
    return pd.DataFrame(
        {
            "country_code": country,
            "group": group if group is not None else "all",
            "answer": values,
            "survey_weight": weights,
        }
    )


def test_equal_weights_give_the_plain_share():
    df = make_survey([1, 1, 0, 0], [1.0, 1.0, 1.0, 1.0])
    assert weighted_share(df, "answer", ["country_code"]).loc["KEN"] == pytest.approx(0.5)


def test_weights_change_the_share():
    # The "yes" respondent represents three times as many people
    df = make_survey([1, 0], [3.0, 1.0])
    assert weighted_share(df, "answer", ["country_code"]).loc["KEN"] == pytest.approx(0.75)


def test_missing_answers_are_left_out():
    df = make_survey([1, 0, np.nan], [1.0, 1.0, 10.0])
    assert weighted_share(df, "answer", ["country_code"]).loc["KEN"] == pytest.approx(0.5)


def test_value_can_be_a_label():
    df = make_survey(["could_not", "savings", "savings", "work"], [1.0, 1.0, 1.0, 1.0])
    share = weighted_share(df, "answer", ["country_code"], value="could_not")
    assert share.loc["KEN"] == pytest.approx(0.25)


def test_shares_are_computed_per_group():
    df = make_survey([1, 0, 1, 1], [1.0, 1.0, 1.0, 1.0], group=["a", "a", "b", "b"])
    share = weighted_share(df, "answer", ["country_code", "group"])
    assert share.loc[("KEN", "a")] == pytest.approx(0.5)
    assert share.loc[("KEN", "b")] == pytest.approx(1.0)


def test_pool_average_gives_each_country_equal_weight():
    # A big country and a small one: the pool average must not favour the bigger sample
    big = make_survey([1] * 900 + [0] * 100, [1.0] * 1_000, country="AAA")
    small = make_survey([0] * 10, [1.0] * 10, country="BBB")
    shares = weighted_share(pd.concat([big, small]), "answer", ["country_code"])
    assert pool_average(shares).loc["pool"] == pytest.approx((0.9 + 0.0) / 2)


def test_pool_average_keeps_other_groups():
    df = pd.concat(
        [
            make_survey([1, 0], [1.0, 1.0], country="AAA", group=["a", "b"]),
            make_survey([1, 1], [1.0, 1.0], country="BBB", group=["a", "b"]),
        ]
    )
    pooled = pool_average(weighted_share(df, "answer", ["country_code", "group"]))
    assert pooled.loc["a"] == pytest.approx(1.0)
    assert pooled.loc["b"] == pytest.approx(0.5)
