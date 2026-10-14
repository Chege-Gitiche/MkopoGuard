"""Step 5.4: the applicant profile accepts only valid, consistent answers."""

import json
import typing
from datetime import date, timedelta

import pytest
from pydantic import ValidationError

from mkopoguard import config
from mkopoguard.api import schemas
from mkopoguard.api.schemas import ApplicantProfile, band_for

VALID = {
    "applicant_ref": "APP-2026-0042",
    "application_date": "2025-04-24",
    "age": 32,
    "income_quintile": 3,
    "in_workforce": True,
    "education_level": "secondary",
    "loan_amount_kes": 15000,
    "term_months": 3,
    "saved_past_year": True,
    "saved_in_savings_club": True,
    "borrowed_past_year": True,
    "borrowed_from_fin_institution": False,
    "borrowed_from_family_friends": True,
    "borrowed_for_medical": False,
    "emergency_fund_source": "savings",
    "worried_about_bills": "somewhat",
}


def parse(**changes) -> ApplicantProfile:
    """Validate the profile as the API receives it: JSON text."""
    return ApplicantProfile.model_validate_json(json.dumps(VALID | changes))


def errors_for(**changes) -> list[dict]:
    with pytest.raises(ValidationError) as caught:
        parse(**changes)
    return caught.value.errors()


def test_a_valid_profile_is_accepted():
    profile = parse()
    assert profile.age == 32 and profile.application_date == date(2025, 4, 24)


@pytest.mark.parametrize(
    "field, value",
    [
        ("age", 17),
        ("age", 101),
        ("income_quintile", 0),
        ("income_quintile", 6),
        ("loan_amount_kes", 999),
        ("loan_amount_kes", 100_001),
        ("term_months", 2),
        ("education_level", "university"),
        ("emergency_fund_source", "lottery"),
        ("worried_about_bills", "a_little"),
    ],
)
def test_out_of_range_or_unknown_values_are_refused(field, value):
    assert errors_for(**{field: value})[0]["loc"] == (field,)


@pytest.mark.parametrize(
    "field, value",
    [("age", "32"), ("age", 32.5), ("in_workforce", 1), ("in_workforce", "yes"),
     ("loan_amount_kes", "15000")],
)  # fmt: skip
def test_no_type_guessing(field, value):
    """Strict: "32" is not an age and 1 is not "yes"; the frontend must send real types."""
    assert errors_for(**{field: value})[0]["loc"] == (field,)


def test_every_field_must_be_present_even_if_unanswered():
    data = dict(VALID)
    del data["borrowed_for_medical"]
    with pytest.raises(ValidationError, match="borrowed_for_medical"):
        ApplicantProfile.model_validate_json(json.dumps(data))
    assert parse(borrowed_for_medical=None).borrowed_for_medical is None  # null = not answered


def test_required_answers_cannot_be_null():
    for field in ("age", "in_workforce", "saved_past_year", "borrowed_past_year"):
        assert errors_for(**{field: None})[0]["loc"] == (field,)


def test_unknown_fields_are_refused():
    assert errors_for(is_female=True)[0]["type"] == "extra_forbidden"  # never a model input


@pytest.mark.parametrize(
    "ref", ["0712345678", "12345678", "Asha Wanjiru", "+254712345678", "ab", ""]
)
def test_the_reference_cannot_be_a_phone_id_number_or_name(ref):
    assert errors_for(applicant_ref=ref)[0]["loc"] == ("applicant_ref",)


def test_the_application_date_cannot_be_in_the_future():
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    assert "future" in errors_for(application_date=tomorrow)[0]["msg"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"saved_past_year": False, "saved_in_savings_club": True}, "saved_in_savings_club"),
        (
            {"borrowed_past_year": False, "borrowed_from_family_friends": False,
             "borrowed_for_medical": True},
            "borrowed_for_medical",
        ),
        ({"borrowed_past_year": False, "borrowed_from_family_friends": True}, "family_friends"),
    ],
)  # fmt: skip
def test_contradicting_answers_are_refused(changes, message):
    assert message in errors_for(**changes)[0]["msg"]


def test_contradictions_need_an_actual_yes():
    profile = parse(saved_past_year=False, saved_in_savings_club=None)
    assert profile.saved_in_savings_club is None


def test_model_inputs_use_the_models_column_names_and_numbers():
    inputs = parse(borrowed_for_medical=None).model_inputs()
    assert inputs["in_workforce"] == 1 and inputs["borrowed_from_fin_institution"] == 0
    assert inputs["borrowed_for_medical"] is None
    assert "applicant_ref" not in inputs and "application_date" not in inputs


def test_allowed_values_match_the_models_training_data():
    """The API's lists come from the bundle's input schema, minus "missing" (sent as null)."""
    fields = json.loads((config.BUNDLE_DIR / "input_schema.json").read_text())["fields"]
    lists = {
        "education_level": schemas.Education,
        "emergency_fund_source": schemas.EmergencySource,
        "worried_about_bills": schemas.BillWorry,
    }
    for column, literal in lists.items():
        assert set(typing.get_args(literal)) == set(fields[column]["allowed"]) - {"missing"}
    assert set(typing.get_args(schemas.Term)) == set(fields["term_months"]["allowed"])


def test_the_profile_asks_for_every_survey_input_the_model_uses():
    columns = json.loads((config.BUNDLE_DIR / "input_schema.json").read_text())["columns"]
    survey = {c for c in columns if not c.startswith("stmt_") and c != "has_statement"}
    assert survey == set(parse().model_inputs())


@pytest.mark.parametrize(
    "p, band",
    [(0.0, "low"), (0.0999, "low"), (0.10, "medium"), (0.199, "medium"), (0.20, "high"),
     (0.349, "high"), (0.35, "very_high"), (1.0, "very_high")],
)  # fmt: skip
def test_bands(p, band):
    assert band_for(p) == band


def test_a_band_needs_a_probability():
    with pytest.raises(ValueError):
        band_for(1.2)
