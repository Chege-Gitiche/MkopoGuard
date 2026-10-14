"""Step 5.5: POST /predict (score one applicant) and GET /model/info.

POST /predict takes a multipart form, so a browser can send a file:
  profile    the applicant profile as JSON text (see ApplicantProfile)
  statement  the M-Pesa statement CSV (optional: leave it out for a thin-file applicant)

Every problem with either is a 422 that says what to fix: a profile field (loc
["profile", "age"]) or a statement row (loc ["statement", "row 12"]). Officers' and admins'
predictions are saved with the model version that made them; the demo account gets the same
result, but nothing is written to the database.
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ValidationError

from mkopoguard.api.auth import SessionDep, require_roles
from mkopoguard.api.models import Prediction, User
from mkopoguard.api.schemas import BANDS, ApplicantProfile, PredictionOut
from mkopoguard.api.service import score
from mkopoguard.api.statement import MAX_BYTES, StatementError, parse_statement

router = APIRouter(tags=["predictions"])
SCORERS = ("officer", "admin", "demo")
Scorer = Annotated[User, Depends(require_roles(*SCORERS))]


def _unprocessable(detail: list[dict]) -> HTTPException:
    return HTTPException(422, detail=detail)


def _profile_errors(error: ValidationError) -> list[dict]:
    """Pydantic's errors, cleaned to plain JSON and placed under "profile"."""
    return [
        {"loc": ["profile", *e["loc"]], "msg": e["msg"], "type": e["type"]}
        for e in error.errors(include_url=False, include_context=False, include_input=False)
    ]


def read_profile(text: str) -> ApplicantProfile:
    try:
        return ApplicantProfile.model_validate_json(text)
    except ValidationError as error:
        if error.errors()[0]["type"] == "json_invalid":
            raise _unprocessable(
                [{"loc": ["profile"], "msg": "profile must be valid JSON", "type": "json_invalid"}]
            ) from None
        raise _unprocessable(_profile_errors(error)) from None


def read_statement(upload: UploadFile | None) -> bytes | None:
    """The uploaded bytes, or None when no file was chosen. Never reads more than 1 MB + 1."""
    if upload is None:
        return None
    data = upload.file.read(MAX_BYTES + 1)
    if not data and not upload.filename:  # an empty file field: the browser sent no file
        return None
    return data


@router.post("/predict", response_model=PredictionOut, status_code=status.HTTP_200_OK)
def predict(
    request: Request,
    user: Scorer,
    session: SessionDep,
    profile: Annotated[str, Form(description="The applicant profile, as JSON text")],
    statement: Annotated[
        UploadFile | None, File(description="M-Pesa statement CSV (optional)")
    ] = None,
) -> PredictionOut:
    """Score one applicant: probability, band, decision and the 3 + 2 reasons."""
    applicant = read_profile(profile)
    data = read_statement(statement)
    parsed = None
    if data is not None:
        try:
            parsed = parse_statement(data, applicant.application_date)
        except StatementError as error:
            raise _unprocessable(error.detail()) from None

    state = request.app.state
    result = score(state.bundle, state.reasoner, applicant, parsed)
    created = datetime.now(UTC)
    prediction_id = None
    if user.role != "demo":  # the demo account is read-only: nothing is saved
        row = Prediction(
            officer_id=user.id,
            applicant_ref=result["applicant_ref"],
            inputs=result["inputs"],
            probability=result["probability_of_default"],
            band=result["band"],
            decision=result["decision"],
            reasons=result["reasons"],
            model_version=result["model_version"],
            created_at=created,
        )
        session.add(row)
        session.commit()
        prediction_id, created = row.id, row.created_at
    return PredictionOut(id=prediction_id, created_at=created, **result)


# --- GET /model/info ---------------------------------------------------------------------------


class ModelInfo(BaseModel):
    name: str
    version: str
    packaged_at: str
    trained_on_applicants: int
    test_pr_auc: float
    thresholds_by_term_months: dict[int, float]
    bands: dict[str, list[float]]
    costs: dict[str, float]
    notice: str


@router.get("/model/info", response_model=ModelInfo, tags=["model"])
def model_info(request: Request, user: Scorer) -> ModelInfo:
    """Which model is scoring, how good it was on the locked test set, and its decision rule."""
    bundle = request.app.state.bundle
    manifest, policy = bundle.manifest, bundle.policy
    lower = 0.0
    bands = {}
    for name, upper in BANDS:
        bands[name] = [lower, upper]
        lower = upper
    return ModelInfo(
        name=manifest["name"],
        version=manifest["version"],
        packaged_at=manifest["created_at"],
        trained_on_applicants=manifest["trained_on"]["applicants"],
        test_pr_auc=round(manifest["test_pr_auc"], 4),
        thresholds_by_term_months=policy["thresholds_by_term_months"],
        bands=bands,
        costs={
            "monthly_fee_rate": policy["monthly_fee_rate"],
            "monthly_funding_cost": policy["monthly_funding_cost"],
            "loss_given_default": policy["loss_given_default"],
        },
        notice="Trained and tested on synthetic data: not for real lending decisions.",
    )
