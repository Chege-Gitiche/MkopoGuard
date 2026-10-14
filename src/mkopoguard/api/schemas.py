"""Step 5.4: what the API accepts and what it returns (Pydantic models).

Strict on the way in:
- every field must be present (answers the applicant didn't give are sent as null, never
  left out), no unknown fields, no type guessing ("31" is not an age, 1 is not true);
- ranges and allowed values match the model's training data (a test checks the category lists
  against the bundle's input schema, so the API and the model can't drift apart);
- answers that contradict each other are refused (no savings in the past year, but saves in a
  savings group).

Bad input is answered with 422 and a message saying which field is wrong and why, never a 500.
"""

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mkopoguard import config

Education = Literal["primary_or_less", "secondary", "tertiary"]
EmergencySource = Literal[
    "savings", "work", "family_friends", "borrowing", "selling_assets", "other", "could_not"
]
BillWorry = Literal["very", "somewhat", "not_at_all"]
Term = Literal[1, 3, 6]

MIN_LOAN_KES, MAX_LOAN_KES = 1_000, 100_000
# The lender's own reference for the application: letters, digits and - _ / only, with at least
# one letter, so a phone or ID number can't be typed in by mistake (names don't fit either)
REF_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_/-]{2,63}$"

# Risk bands shown to the loan officer (the decision itself uses the per-term threshold).
# The average applicant's risk is about 20%.
BANDS = (("low", 0.10), ("medium", 0.20), ("high", 0.35), ("very_high", 1.0))


def band_for(probability: float) -> str:
    """The band a probability of default falls in: low < 10% <= medium < 20% <= high < 35%."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability must be between 0 and 1, got {probability}")
    for name, upper in BANDS:
        if probability < upper:
            return name
    return BANDS[-1][0]


class ApplicantProfile(BaseModel):
    """The applicant's answers and the loan they asked for (everything except the statement)."""

    model_config = ConfigDict(strict=True, extra="forbid", str_strip_whitespace=True)

    applicant_ref: str = Field(
        pattern=REF_PATTERN,
        description="Your reference for this application, e.g. APP-2026-0042. "
        "Never a name, phone number or ID number.",
        examples=["APP-2026-0042"],
    )
    application_date: date = Field(
        description="The day the applicant applied; the statement must cover the 6 months before"
    )
    age: int = Field(ge=18, le=100)
    income_quintile: int = Field(ge=1, le=5, description="1 = poorest fifth, 5 = richest fifth")
    in_workforce: bool
    education_level: Education | None
    loan_amount_kes: int = Field(ge=MIN_LOAN_KES, le=MAX_LOAN_KES)
    term_months: Term
    saved_past_year: bool
    saved_in_savings_club: bool | None = Field(description="Chama / savings group")
    borrowed_past_year: bool
    borrowed_from_fin_institution: bool | None
    borrowed_from_family_friends: bool | None
    borrowed_for_medical: bool | None
    emergency_fund_source: EmergencySource | None
    worried_about_bills: BillWorry | None

    @field_validator("applicant_ref")
    @classmethod
    def _looks_like_a_reference(cls, value: str) -> str:
        if not any(ch.isalpha() for ch in value):
            raise ValueError("applicant_ref needs at least one letter (e.g. APP-2026-0042)")
        return value

    @field_validator("application_date")
    @classmethod
    def _not_in_the_future(cls, value: date) -> date:
        if value > date.today():
            raise ValueError("application_date can't be in the future")
        return value

    @model_validator(mode="after")
    def _answers_agree(self) -> "ApplicantProfile":
        if not self.saved_past_year and self.saved_in_savings_club:
            raise ValueError("saved_in_savings_club is yes, but saved_past_year is no")
        if not self.borrowed_past_year:
            for name in (
                "borrowed_from_fin_institution",
                "borrowed_from_family_friends",
                "borrowed_for_medical",
            ):
                if getattr(self, name):
                    raise ValueError(f"{name} is yes, but borrowed_past_year is no")
        return self

    def model_inputs(self) -> dict:
        """The profile as the model's input columns: yes/no as 1/0, unanswered as None."""
        values = self.model_dump(exclude={"applicant_ref", "application_date"})
        return {k: (int(v) if isinstance(v, bool) else v) for k, v in values.items()}


# --- responses -------------------------------------------------------------------------------


class ReasonOut(BaseModel):
    code: str = Field(description="Stable code, e.g. 'betting'")
    text: str = Field(description="Plain-English reason for the loan officer")
    direction: Literal["raises", "lowers"]
    impact_points: float = Field(description="Percentage points of risk this reason accounts for")
    strength: Literal["major", "moderate", "minor"]


class OtherFactorsOut(BaseModel):
    impact_points: float
    material: bool = Field(description="As large as a stated reason (and at least 1 point)")


class ReasonsOut(BaseModel):
    raising: list[ReasonOut] = Field(min_length=3, max_length=3)
    lowering: list[ReasonOut] = Field(min_length=2, max_length=2)
    other_factors: OtherFactorsOut


class PredictionOut(BaseModel):
    """One scored applicant, as returned by POST /predict and GET /predictions/{id}."""

    id: int | None = Field(description="None for the demo account: nothing is saved")
    applicant_ref: str
    probability_of_default: float = Field(ge=0, le=1)
    band: Literal["low", "medium", "high", "very_high"]
    decision: Literal["approve", "decline"]
    threshold: float = Field(description="Decline at or above this, for this loan term")
    term_months: Term
    has_statement: bool
    reasons: ReasonsOut
    warnings: list[str] = Field(description="Unusual inputs: scored, but treat with care")
    model_version: str = Field(examples=[config.MODEL_VERSION])
    created_at: datetime
