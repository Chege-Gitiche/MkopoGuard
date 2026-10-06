import numpy as np
import pandas as pd
import pytest

from mkopoguard.data import clean


def make_raw(**overrides):
    """One valid raw Findex row; override any column to test a case."""
    row = {col: 1 for col in clean.COLUMN_MAP}
    row.update(
        economycode="KEN",
        economy="Kenya",
        wpid_random=123,
        wgt=1.0,
        year=2021,
        age=30,
        account_fin=1,
        account_mob=1,
        saved=1,
        borrowed=1,
        merchantpay_dig=0,
    )
    row.update(overrides)
    return pd.DataFrame([row])


def clean_value(column, raw_column, code):
    return clean.clean_findex(make_raw(**{raw_column: code}))[column].iloc[0]


@pytest.mark.parametrize("code, expected", [(1, 1), (2, 0)])
def test_yes_no_codes(code, expected):
    assert clean_value("borrowed_from_fin_institution", "fin22a", code) == expected


@pytest.mark.parametrize("code", [3, 4])
def test_dont_know_and_refused_become_missing(code):
    assert pd.isna(clean_value("borrowed_from_fin_institution", "fin22a", code))


@pytest.mark.parametrize(
    "code, expected", [(1, "account"), (2, "cash_only"), (3, "other"), (4, "none")]
)
def test_payment_channel_codes(code, expected):
    assert clean_value("wage_payment_channel", "receive_wages", code) == expected


def test_payment_channel_dk_ref_is_missing():
    assert pd.isna(clean_value("utility_payment_channel", "pay_utilities", 5))


def test_emergency_fund_could_not_and_dk():
    assert clean_value("emergency_fund_source", "fin24", 7) == "could_not"
    assert pd.isna(clean_value("emergency_fund_source", "fin24", 8))
    assert pd.isna(clean_value("emergency_fund_source", "fin24", 9))


@pytest.mark.parametrize("code", [4, 5, 6])
def test_worry_non_answers_are_missing(code):
    assert pd.isna(clean_value("worried_about_bills", "fin44c", code))


def test_female_and_education_codes():
    assert clean_value("is_female", "female", 2) == 0
    assert clean_value("education_level", "educ", 3) == "tertiary"
    assert pd.isna(clean_value("education_level", "educ", 4))


def test_minors_and_missing_age_are_dropped():
    raw = pd.concat([make_raw(age=17), make_raw(age=18), make_raw(age=np.nan), make_raw(age=45)])
    out = clean.clean_findex(raw)
    assert out["age"].tolist() == [18, 45]


def test_output_has_exactly_the_readable_columns():
    out = clean.clean_findex(make_raw())
    assert list(out.columns) == list(clean.COLUMN_MAP.values())


def test_input_is_not_modified():
    raw = make_raw()
    before = raw.copy()
    clean.clean_findex(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_missing_input_column_raises_clear_error():
    with pytest.raises(KeyError, match="fin24"):
        clean.clean_findex(make_raw().drop(columns=["fin24"]))
