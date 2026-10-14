"""Step 4.1: exact SHAP values for the champion (a logistic regression).

For a linear model, SHAP has an exact closed form. With the features independent and the
background = the applicants the model was trained on:

    contribution of transformed column j  =  coef_j * (x_j - average x_j)

and   base value + sum of contributions  =  the model's log-odds for that applicant, exactly.

This is what shap.LinearExplainer computes (a test checks the two agree), but it needs no
extra library at prediction time, so the API (phase 5) can explain every score instantly.

Contributions are first computed per TRANSFORMED column (e.g. the 9 one-hot columns of
emergency_fund_source) and then added up per INPUT column, so every explanation talks about
something the applicant actually answered or uploaded.

Units: log-odds. Positive = pushes the risk of default up, negative = pushes it down.
"""

import numpy as np
import pandas as pd


class LinearShap:
    def __init__(self, pipeline, background: pd.DataFrame):
        self.prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        if not hasattr(model, "coef_"):
            raise TypeError("LinearShap needs a linear model with coef_ (the champion is one)")
        self.columns = list(self.prep.feature_names_in_)
        self.coef = pd.Series(model.coef_[0], index=self.prep.get_feature_names_out())
        self.intercept = float(model.intercept_[0])
        transformed = self.prep.transform(background[self.columns])
        self.mean = transformed.mean()
        self.base_value = self.intercept + float((self.mean * self.coef).sum())
        self.source = self._source_columns()

    def _source_columns(self) -> pd.Series:
        """Which input column each transformed column came from."""
        source = {}
        for name, transformer, cols in self.prep.transformers_:
            if transformer == "drop" or name == "remainder":
                continue
            if name == "cat":
                encoder = transformer.named_steps["one_hot"]
                for col, cats in zip(cols, encoder.categories_, strict=True):
                    for _ in cats:
                        source[len(source)] = col
            else:
                for col in cols:
                    source[len(source)] = col
        out = pd.Series(source).set_axis(self.coef.index)
        if set(out) != set(self.columns):
            raise ValueError("Couldn't map every transformed column back to an input column")
        return out

    def transformed_contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Contribution of each transformed column, per applicant (log-odds)."""
        x = self.prep.transform(frame[self.columns])
        return (x - self.mean) * self.coef

    def contributions(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Contribution of each INPUT column, per applicant (log-odds)."""
        parts = self.transformed_contributions(frame)
        summed = parts.T.groupby(self.source).sum().T
        return summed[self.columns]

    def log_odds(self, contributions: pd.DataFrame) -> pd.Series:
        """base value + contributions: equals the model's own log-odds (checked in tests)."""
        return self.base_value + contributions.sum(axis=1)


def combine(contributions: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Add up the contributions of columns that belong together (e.g. the whole statement)."""
    out = pd.DataFrame(index=contributions.index)
    used = set()
    for name, cols in groups.items():
        present = [c for c in cols if c in contributions]
        if not present:  # e.g. a feature group dropped in step 3.7
            continue
        out[name] = contributions[present].sum(axis=1)
        used |= set(present)
    for col in contributions:
        if col not in used:
            out[col] = contributions[col]
    return out


def importance(contributions: pd.DataFrame) -> pd.Series:
    """Global importance: average size of a column's contribution, biggest first."""
    return contributions.abs().mean().sort_values(ascending=False)


def to_probability(log_odds) -> np.ndarray:
    return 1 / (1 + np.exp(-np.asarray(log_odds, dtype=float)))
