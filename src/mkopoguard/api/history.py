"""Step 5.6: an officer's own prediction history.

GET /predictions           newest first, ?page=&size=, filters: band, decision, ref (prefix),
                           date_from / date_to (Kenyan calendar days, inclusive)
GET /predictions/{id}      one past prediction, with its reasons and the answers it was made from

Every query is filtered by the logged-in officer's id INSIDE the database query, never after.
Someone else's prediction is answered exactly like one that doesn't exist (404 "Prediction not
found"), so an officer can't even find out which ids belong to colleagues. This holds for
admins too: nobody reads another person's history through the API.
"""

from datetime import date, datetime, time, timedelta
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from mkopoguard.api import openapi
from mkopoguard.api.auth import CurrentUser, SessionDep
from mkopoguard.api.models import Prediction
from mkopoguard.api.schemas import ReasonsOut

router = APIRouter(prefix="/predictions", tags=["history"])
KENYA = ZoneInfo("Africa/Nairobi")  # officers think in Kenyan days; times are stored in UTC
MAX_PAGE_SIZE = 100
NOT_FOUND = "Prediction not found"

Band = Literal["low", "medium", "high", "very_high"]
Decision = Literal["approve", "decline"]


class PredictionSummary(BaseModel):
    """One row of the history list."""

    id: int
    applicant_ref: str
    probability_of_default: float
    band: Band
    decision: Decision
    threshold: float | None
    term_months: int | None
    has_statement: bool
    model_version: str
    created_at: datetime


class PredictionDetail(PredictionSummary):
    """One past prediction in full: why, and from what."""

    reasons: ReasonsOut
    profile: dict = Field(description="The answers the prediction was made from")
    statement_transactions: int = Field(description="How many transactions the statement had")


class HistoryPage(BaseModel):
    items: list[PredictionSummary]
    page: int
    size: int
    total: int = Field(description="Matching predictions, on all pages")
    pages: int


def _summary_fields(row: Prediction) -> dict:
    inputs = row.inputs or {}
    profile = inputs.get("profile") or {}
    return {
        "id": row.id,
        "applicant_ref": row.applicant_ref,
        "probability_of_default": row.probability,
        "band": row.band,
        "decision": row.decision,
        "threshold": row.threshold,
        "term_months": profile.get("term_months"),
        "has_statement": inputs.get("statement_features") is not None,
        "model_version": row.model_version,
        "created_at": row.created_at,
    }


def kenyan_day_starts(day: date) -> datetime:
    """Midnight at the start of a Kenyan calendar day, as an aware datetime."""
    return datetime.combine(day, time.min, tzinfo=KENYA)


@router.get(
    "",
    response_model=HistoryPage,
    summary="Your prediction history",
    responses=openapi.UNAUTHORIZED | openapi.INVALID,
)
def list_predictions(
    user: CurrentUser,
    session: SessionDep,
    page: Annotated[int, Query(ge=1)] = 1,
    size: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = 20,
    band: Band | None = None,
    decision: Decision | None = None,
    ref: Annotated[str | None, Query(min_length=1, max_length=64)] = None,
    date_from: date | None = None,
    date_to: date | None = None,
) -> HistoryPage:
    """Your own predictions, newest first."""
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            422,
            detail=[
                {
                    "loc": ["query", "date_from"],
                    "msg": "date_from must be on or before date_to",
                    "type": "value_error",
                }
            ],
        )
    where = [Prediction.officer_id == user.id]  # the one rule that matters most
    if band:
        where.append(Prediction.band == band)
    if decision:
        where.append(Prediction.decision == decision)
    if ref:
        where.append(func.lower(Prediction.applicant_ref).startswith(ref.lower(), autoescape=True))
    if date_from:
        where.append(Prediction.created_at >= kenyan_day_starts(date_from))
    if date_to:
        where.append(Prediction.created_at < kenyan_day_starts(date_to + timedelta(days=1)))

    total = session.scalar(select(func.count()).select_from(Prediction).where(*where))
    rows = session.scalars(
        select(Prediction)
        .where(*where)
        .order_by(Prediction.created_at.desc(), Prediction.id.desc())  # id breaks ties
        .offset((page - 1) * size)
        .limit(size)
    ).all()
    return HistoryPage(
        items=[PredictionSummary(**_summary_fields(r)) for r in rows],
        page=page,
        size=size,
        total=total,
        pages=-(-total // size),  # ceiling division; 0 when there is nothing
    )


@router.get(
    "/{prediction_id}",
    response_model=PredictionDetail,
    summary="One past prediction",
    responses=openapi.UNAUTHORIZED | openapi.NOT_FOUND | openapi.INVALID,
)
def get_prediction(prediction_id: int, user: CurrentUser, session: SessionDep) -> PredictionDetail:
    """One of your past predictions. Anyone else's is "not found", exactly like a missing one."""
    row = session.scalar(
        select(Prediction).where(Prediction.id == prediction_id, Prediction.officer_id == user.id)
    )
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=NOT_FOUND)
    inputs = row.inputs or {}
    return PredictionDetail(
        **_summary_fields(row),
        reasons=row.reasons,
        profile=inputs.get("profile") or {},
        statement_transactions=inputs.get("statement_transactions", 0),
    )
