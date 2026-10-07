"""Step 1.8: leakage tests. Two kinds:
- time leakage: nothing from on or after the application date
- column leakage: hidden traits, the outcome and audit-only columns never reach the model
"""

import pandas as pd
import pytest

from mkopoguard import config
from mkopoguard.data.schemas import APPLICANT_SCHEMA
from mkopoguard.data.transactions import statement_for
from mkopoguard.features.columns import (
    TARGET,
    assert_no_leakage,
    forbidden_reason,
    model_columns,
)

ALL_APPLICANT_COLUMNS = list(APPLICANT_SCHEMA.columns)

EXPECTED_MODEL_COLUMNS = [
    "country_code",
    "age",
    "education_level",
    "income_quintile",
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
    "emergency_fund_source",
    "worried_about_bills",
    "wage_payment_channel",
    "agri_payment_channel",
    "utility_payment_channel",
    "sent_domestic_remittance",
    "received_domestic_remittance",
    "received_govt_transfer",
    "made_digital_merchant_payment",
    "has_statement",
    "loan_amount_kes",
    "term_months",
]


# --- column leakage ----------------------------------------------------------


def test_model_sees_exactly_the_expected_columns():
    applicants = pd.DataFrame(columns=ALL_APPLICANT_COLUMNS)
    assert model_columns(applicants) == EXPECTED_MODEL_COLUMNS


def test_no_hidden_column_is_ever_allowed():
    applicants = pd.DataFrame(columns=ALL_APPLICANT_COLUMNS + ["hidden_added_later"])
    allowed = model_columns(applicants)
    assert not [c for c in allowed if c.startswith("hidden_")]


@pytest.mark.parametrize("column", [TARGET, "is_female", "applicant_id", "application_date"])
def test_outcome_gender_ids_and_dates_are_forbidden(column):
    assert forbidden_reason(column) is not None


def test_assert_no_leakage_names_every_forbidden_column():
    with pytest.raises(ValueError) as error:
        assert_no_leakage(["age", "hidden_discipline", "defaulted", "is_female"])
    message = str(error.value)
    for column in ["hidden_discipline", "defaulted", "is_female"]:
        assert column in message
    assert "age" not in message.split(":", 1)[1]


def test_assert_no_leakage_accepts_allowed_columns():
    assert_no_leakage(EXPECTED_MODEL_COLUMNS)


# --- time leakage ------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["salaried", "seasonal", "irregular"])
def test_no_transaction_on_or_after_the_application_date(pattern):
    """Busy applicants with heavy overdraft use, many seeds: nothing may leak past the date."""
    application_date = pd.Timestamp("2025-01-01")
    for number in range(1, 51):
        a = {
            "applicant_id": f"A-{number:05d}",
            "statement_start": application_date - pd.Timedelta(days=config.STATEMENT_DAYS),
            "hidden_monthly_income_kes": 3_000.0,
            "hidden_income_pattern": pattern,
            "received_domestic_remittance": 1,
            "sent_domestic_remittance": 1,
            "utility_payment_channel": "account",
            "made_digital_merchant_payment": 1,
            "hidden_betting_share": 0.3,
            "hidden_bill_miss_prob": 0.0,
            "hidden_uses_overdraft": 1,
            "hidden_overdraft_repay_share": 0.9,
        }
        assert statement_for(a)["timestamp"].max() < application_date
