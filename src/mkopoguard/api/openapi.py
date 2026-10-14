"""Step 5.8: what the interactive API docs (/docs) say.

The description, the groups ("tags"), and the error answers every endpoint can give, with
examples, so someone trying the API from /docs knows what each call does and what can go wrong.
A test checks every endpoint is documented this way.
"""

import json

from pydantic import BaseModel

DESCRIPTION = """
Scores microloan applicants for **loan officers**: the probability of default, a risk band,
an approve / decline decision from the lender's costs, and the **3 + 2 reasons** behind it.

> **Synthetic data only: not for real lending decisions.**

### Try it in three steps
1. Click **Authorize**, enter your email (in *username*) and password, then **Authorize**.
   A login lasts 30 minutes.
2. Open **POST /predict**. The *profile* field is already filled in with a sample thin-file
   applicant: click **Execute**. For an applicant with a statement, paste a profile from
   `samples/profiles/` and choose the matching CSV from `samples/statements/`.
3. Open **GET /predictions** to see your history, newest first.

Every error has a status and a message: **401** not logged in, **404** not found (including
another officer's predictions), **422** invalid input (with the field or statement row),
**429** too many login attempts. Each response carries an `X-Request-ID`; quote it when
reporting a problem.
"""

TAGS = [
    {"name": "auth", "description": "Log in and see who you are. There is no public sign-up."},
    {"name": "predictions", "description": "Score an applicant (saved to your history)."},
    {"name": "history", "description": "Your own past predictions; nobody else's."},
    {"name": "model", "description": "Which model is scoring, and its decision rule."},
    {"name": "system", "description": "Health check for the host."},
]

# A real sample applicant (samples/profiles/sample_10_thin_file.json): no statement needed, so
# "Try it out" on /predict works straight away. A test checks it is a valid profile.
EXAMPLE_PROFILE = {
    "applicant_ref": "SAMPLE-10",
    "application_date": "2025-03-15",
    "age": 52,
    "income_quintile": 2,
    "in_workforce": True,
    "education_level": "primary_or_less",
    "loan_amount_kes": 2000,
    "term_months": 1,
    "saved_past_year": True,
    "saved_in_savings_club": True,
    "borrowed_past_year": True,
    "borrowed_from_fin_institution": False,
    "borrowed_from_family_friends": False,
    "borrowed_for_medical": False,
    "emergency_fund_source": "could_not",
    "worried_about_bills": "very",
}
EXAMPLE_PROFILE_JSON = json.dumps(EXAMPLE_PROFILE)


class Message(BaseModel):
    detail: str


class FieldError(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class FieldErrors(BaseModel):
    detail: list[FieldError]


def _example(model, status: str, value) -> dict:
    return {
        "model": model,
        "description": status,
        "content": {"application/json": {"example": value}},
    }


UNAUTHORIZED = {
    401: _example(Message, "Not logged in, or the login has expired",
                  {"detail": "Not logged in, or the session has expired"})
}  # fmt: skip
NOT_FOUND = {404: _example(Message, "No such prediction in your history",
                           {"detail": "Prediction not found"})}  # fmt: skip
INVALID = {
    422: _example(
        FieldErrors,
        "Invalid input: each problem names its field or statement row",
        {"detail": [
            {"loc": ["profile", "age"], "msg": "Input should be greater than or equal to 18",
             "type": "greater_than_equal"},
            {"loc": ["statement", "row 12"], "msg": "type must be one of: income_in, ...",
             "type": "statement_invalid"},
        ]},
    )
}  # fmt: skip
LOGIN_ERRORS = {
    401: _example(Message, "Wrong email or password, or the account is deactivated",
                  {"detail": "Incorrect email or password"}),
    422: INVALID[422],
    429: _example(Message, "Too many attempts: wait (see the Retry-After header)",
                  {"detail": "Too many login attempts. Wait 15 minute(s) and try again."}),
}  # fmt: skip
