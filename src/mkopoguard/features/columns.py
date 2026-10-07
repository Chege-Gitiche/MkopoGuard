"""Step 1.8: which applicant columns the model may see.

Everything is allowed EXCEPT the forbidden groups below. Phase 2 and 3 must pick model
inputs through model_columns(), and assert_no_leakage() fails loudly if a forbidden
column ever sneaks in.
"""

import pandas as pd

from mkopoguard.data.population import HIDDEN_PREFIX

TARGET = "defaulted"

FORBIDDEN = {
    # The answer itself
    TARGET: "the outcome the model predicts",
    # Identifiers and survey bookkeeping: no information about the person
    "applicant_id": "identifier",
    "respondent_id": "identifier",
    "country": "duplicate of country_code",
    "survey_weight": "survey bookkeeping",
    "survey_year": "survey bookkeeping",
    # Dates: only used to cut the statement window
    "application_date": "date, not a characteristic",
    "statement_start": "date, not a characteristic",
    # Fairness audit only (Phase 4)
    "is_female": "protected attribute, audit only",
}


def forbidden_reason(column: str) -> str | None:
    """Why a column is forbidden, or None if the model may use it."""
    if column.startswith(HIDDEN_PREFIX):
        return "hidden simulator trait"
    return FORBIDDEN.get(column)


def model_columns(df: pd.DataFrame) -> list[str]:
    """The columns of df that the model is allowed to use, in their original order."""
    return [c for c in df.columns if forbidden_reason(c) is None]


def assert_no_leakage(columns) -> None:
    """Raise if any forbidden column is in the list."""
    leaks = {c: forbidden_reason(c) for c in columns if forbidden_reason(c) is not None}
    if leaks:
        details = ", ".join(f"{c} ({why})" for c, why in leaks.items())
        raise ValueError(f"Leakage: forbidden columns in model input: {details}")
