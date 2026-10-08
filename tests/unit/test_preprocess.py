import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from mkopoguard.features.preprocess import (
    BINARY,
    CATEGORICAL,
    COUNTRY,
    NUMERIC,
    STATEMENT_FEATURES,
    Winsorizer,
    build_preprocessor,
    input_columns,
    make_pipeline,
)

CATEGORY_VALUES = {
    "education_level": ["primary_or_less", "secondary", "tertiary"],
    "emergency_fund_source": ["savings", "family_friends", "work", "could_not"],
    "worried_about_bills": ["very", "somewhat", "not_at_all"],
    "wage_payment_channel": ["account", "cash_only", "none"],
    "agri_payment_channel": ["account", "cash_only", "none"],
    "utility_payment_channel": ["account", "cash_only", "none"],
}


def make_table(n=200, seed=0):
    """A small model table shaped like features.parquet, with gaps like the real one."""
    rng = np.random.default_rng(seed)
    table = pd.DataFrame(
        {
            "country_code": rng.choice(["KEN", "UGA", "GHA"], n),
            "age": rng.integers(18, 80, n).astype("int16"),
            "income_quintile": pd.array(rng.integers(1, 6, n), dtype="Int8"),
            "loan_amount_kes": rng.integers(2, 200, n) * 500,
            "term_months": rng.choice([1, 3, 6], n),
        }
    )
    for col in BINARY:
        values = pd.array(rng.integers(0, 2, n), dtype="Int8")
        values[rng.random(n) < 0.05] = pd.NA  # "don't know" answers
        table[col] = values
    for col, values in CATEGORY_VALUES.items():
        table[col] = pd.Categorical(rng.choice(values + [None], n))
    has_statement = rng.random(n) < 0.5
    table["has_statement"] = has_statement.astype("int8")
    for col in STATEMENT_FEATURES:
        table[col] = np.where(has_statement, rng.lognormal(0, 1, n), np.nan)
    table["defaulted"] = (rng.random(n) < 0.2).astype("int8")
    return table


# --- column groups ------------------------------------------------------------


def test_every_input_column_is_in_exactly_one_group():
    columns = input_columns(include_country=True)
    assert len(columns) == len(set(columns)) == 44
    assert set(NUMERIC) | set(BINARY) | set(CATEGORICAL) | {COUNTRY} == set(columns)


def test_country_is_only_used_when_asked():
    table = make_table()
    without = build_preprocessor().fit_transform(table[input_columns()])
    with_country = build_preprocessor(True).fit_transform(table[input_columns(True)])
    assert not [c for c in without.columns if c.startswith("country_code")]
    assert {"country_code_KEN", "country_code_UGA", "country_code_GHA"} <= set(with_country)


# --- output -------------------------------------------------------------------


def test_output_is_complete_numbers_only():
    out = build_preprocessor().fit_transform(make_table()[input_columns()])
    assert out.notna().all().all()
    assert np.isfinite(out.to_numpy()).all()


def test_thin_file_applicants_get_the_training_median():
    table = make_table()
    out = build_preprocessor().fit_transform(table[input_columns()])
    thin = out.loc[table["has_statement"] == 0, STATEMENT_FEATURES]
    assert (thin.nunique() == 1).all()  # every thin-file row gets the same filled value


def test_missing_answers_become_their_own_category():
    out = build_preprocessor().fit_transform(make_table()[input_columns()])
    assert "emergency_fund_source_missing" in out.columns


def test_unknown_category_at_prediction_time_does_not_crash():
    table = make_table()
    prep = build_preprocessor().fit(table[input_columns()])
    new = table[input_columns()].head(1).copy()
    new["emergency_fund_source"] = "won_the_lottery"
    out = prep.transform(new)
    assert out.filter(like="emergency_fund_source_").sum(axis=1).iloc[0] == 0


def test_one_applicant_alone_matches_the_same_applicant_in_a_batch():
    table = make_table()
    prep = build_preprocessor().fit(table[input_columns()])
    batch = prep.transform(table[input_columns()])
    alone = prep.transform(table[input_columns()].iloc[[7]])
    pd.testing.assert_frame_equal(alone, batch.iloc[[7]])


def test_missing_input_column_fails_clearly():
    table = make_table()
    prep = build_preprocessor().fit(table[input_columns()])
    with pytest.raises(ValueError, match="stmt_betting_share"):
        prep.transform(table[input_columns()].drop(columns="stmt_betting_share"))


# --- no leakage: everything is learned in fit() --------------------------------


def test_learned_values_come_from_training_data_only():
    train, other = make_table(seed=1), make_table(seed=2)
    other["age"] = 99  # very different data
    prep = build_preprocessor().fit(train[input_columns()])
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    before = scaler.mean_.copy()
    prep.transform(other[input_columns()])
    np.testing.assert_array_equal(scaler.mean_, before)


def test_pipeline_runs_inside_cross_validation():
    table = make_table(400)
    pipeline = make_pipeline(LogisticRegression(max_iter=1000))
    scores = cross_val_score(pipeline, table[input_columns()], table["defaulted"], cv=3)
    assert len(scores) == 3


def test_saved_and_reloaded_pipeline_gives_identical_output(tmp_path):
    table = make_table()
    prep = build_preprocessor().fit(table[input_columns()])
    joblib.dump(prep, tmp_path / "prep.joblib")
    reloaded = joblib.load(tmp_path / "prep.joblib")
    pd.testing.assert_frame_equal(
        prep.transform(table[input_columns()]), reloaded.transform(table[input_columns()])
    )


# --- Winsorizer -----------------------------------------------------------------


def test_winsorizer_clips_to_learned_percentiles():
    train = pd.DataFrame({"x": np.arange(1, 101, dtype=float)})  # 1 to 100
    w = Winsorizer(lower=0.01, upper=0.99).fit(train)
    out = w.transform(pd.DataFrame({"x": [-500.0, 50.0, 10_000.0]}))
    assert out["x"].tolist() == pytest.approx([1.99, 50.0, 99.01])


def test_winsorizer_leaves_missing_values_for_the_imputer():
    w = Winsorizer().fit(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    assert w.transform(pd.DataFrame({"x": [np.nan]}))["x"].isna().all()
