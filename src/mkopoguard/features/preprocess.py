"""Step 2.5: the preprocessing pipeline.

Turns the model table into clean numbers: fill gaps, tame extreme values, scale numbers and
one-hot encode categories. Everything a step LEARNS from data (medians, clip bounds, means,
category lists) is learned in fit(), so wrapping it with the model in one scikit-learn
Pipeline means cross-validation and the test set can never leak into it.

Thin-file applicants have every statement feature missing. They are filled with the training
median, and the model tells them apart through `has_statement` (1 = has a statement).
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mkopoguard.features.transactions import FEATURES as STATEMENT_FEATURES

# Money amounts are very skewed (a few large earners), so they are logged first
LOG_COLUMNS = ["loan_amount_kes", "stmt_avg_monthly_inflow", "stmt_median_monthly_inflow"]

NUMERIC = ["age", "income_quintile", "loan_amount_kes", "term_months", *STATEMENT_FEATURES]

BINARY = [
    "has_statement",
    "in_workforce",
    "has_bank_account",
    "has_mobile_money",
    "owns_mobile_phone",
    "has_internet_access",
    "saved_past_year",
    "saved_in_savings_club",
    "borrowed_past_year",
    "borrowed_from_fin_institution",
    "borrowed_from_family_friends",
    "borrowed_for_medical",
    "sent_domestic_remittance",
    "received_domestic_remittance",
    "received_govt_transfer",
    "made_digital_merchant_payment",
]

CATEGORICAL = [
    "education_level",
    "emergency_fund_source",
    "worried_about_bills",
    "wage_payment_channel",
    "agri_payment_channel",
    "utility_payment_channel",
]

COUNTRY = "country_code"
MISSING = "missing"


def input_columns(include_country: bool = False) -> list[str]:
    """Every column the pipeline reads, in a fixed order."""
    return NUMERIC + BINARY + CATEGORICAL + ([COUNTRY] if include_country else [])


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to bounds learned in fit (by default its 1st and 99th percentiles).

    Extreme values are kept but pulled in, instead of deleting the rows.
    """

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X, dtype=float)
        self.lower_bounds_ = X.quantile(self.lower).to_numpy()
        self.upper_bounds_ = X.quantile(self.upper).to_numpy()
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, dtype=float)
        clipped = np.clip(frame.to_numpy(), self.lower_bounds_, self.upper_bounds_)
        return pd.DataFrame(clipped, columns=frame.columns, index=frame.index)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def _to_float(X):
    """Nullable integers (Int8 with <NA>) become plain floats with NaN."""
    return pd.DataFrame(X).astype("float64")


def _log_money(X):
    frame = _to_float(X)
    for col in LOG_COLUMNS:
        if col in frame:
            frame[col] = np.log1p(frame[col].clip(lower=0))
    return frame


def _to_labels(X):
    """Categories become plain text; missing answers become their own 'missing' category."""
    frame = pd.DataFrame(X).astype(object)
    return frame.where(frame.notna(), MISSING).astype(str)


def build_preprocessor(include_country: bool = False) -> ColumnTransformer:
    """The unfitted ColumnTransformer. Fit it (inside a Pipeline) on training data only."""
    numeric = Pipeline(
        [
            ("log_money", FunctionTransformer(_log_money, feature_names_out="one-to-one")),
            ("clip", Winsorizer()),
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = Pipeline(
        [
            ("to_float", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="most_frequent")),
        ]
    )
    categorical = Pipeline(
        [
            ("labels", FunctionTransformer(_to_labels, feature_names_out="one-to-one")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    groups = CATEGORICAL + ([COUNTRY] if include_country else [])
    transformer = ColumnTransformer(
        [("num", numeric, NUMERIC), ("bin", binary, BINARY), ("cat", categorical, groups)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return transformer.set_output(transform="pandas")


def make_pipeline(model, include_country: bool = False) -> Pipeline:
    """Preprocessing + model as one object: fit, cross-validate and save them together."""
    return Pipeline([("prep", build_preprocessor(include_country)), ("model", model)])


# Types MLflow's safe model format (skops) may load from a saved pipeline. Skops refuses any
# type not listed, so a tampered model file can't run arbitrary code when it is loaded.
SKOPS_TRUSTED_TYPES = [
    "mkopoguard.features.preprocess.Winsorizer",
    "mkopoguard.features.preprocess._log_money",
    "mkopoguard.features.preprocess._to_float",
    "mkopoguard.features.preprocess._to_labels",
    "numpy.dtype",
]
