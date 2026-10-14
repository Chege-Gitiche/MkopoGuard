"""Step 5.2: the database tables, as SQLAlchemy models.

users        the loan officers (and admins) who can log in. There is no public sign-up:
             accounts are created by scripts/create_user.py (step 5.3).
predictions  every score an officer asked for: the inputs, the result, the reasons and the
             model version, so a past decision can always be traced back to the exact model.

The schema itself is created and changed only by Alembic migrations (migrations/versions/),
never by create_all(); a test checks these models and the migrations agree.

All times are stored in UTC. applicant_ref is the lender's own reference for the application
(e.g. "APP-2026-0042"); names, phone numbers and ID numbers are never stored (data minimisation,
Kenya Data Protection Act 2019).
"""

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    TypeDecorator,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, validates

ROLES = ("officer", "admin", "demo")  # demo: read-only (migration 0002)
DECISIONS = ("approve", "decline")

# Fixed names for every constraint and index, so migrations can refer to them on any database
NAMING = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


def utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """A timestamp that is always UTC: converted to UTC when saved, labelled UTC when read
    (SQLite stores no timezone, Postgres does; both now behave the same)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(UTC)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value


def _quoted(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{v}'" for v in values)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(f"role IN ({_quoted(ROLES)})", name="role_allowed"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(120))
    role: Mapped[str] = mapped_column(String(20), default="officer")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="officer", order_by="Prediction.created_at.desc()"
    )

    @validates("email")
    def _normalise_email(self, _key: str, value: str) -> str:
        """One account per address, however it is typed: stored trimmed and lowercase."""
        return value.strip().lower()

    def __repr__(self) -> str:  # never shows the password hash
        return f"User(id={self.id}, email={self.email!r}, role={self.role!r})"


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (
        CheckConstraint("probability >= 0 AND probability <= 1", name="probability_range"),
        CheckConstraint(f"decision IN ({_quoted(DECISIONS)})", name="decision_allowed"),
        CheckConstraint(
            "threshold IS NULL OR (threshold >= 0 AND threshold <= 1)", name="threshold_range"
        ),
        # An officer's history, newest first (step 5.6), is one index lookup
        Index("ix_predictions_officer_created", "officer_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # RESTRICT: an officer with history can't be deleted (deactivate them instead), so no
    # past decision ever loses track of who made it
    officer_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="RESTRICT"))
    applicant_ref: Mapped[str] = mapped_column(String(64))
    inputs: Mapped[dict] = mapped_column(JSON)
    probability: Mapped[float] = mapped_column(Float)
    band: Mapped[str] = mapped_column(String(20))
    decision: Mapped[str] = mapped_column(String(10))
    # The cut-off the decision was made against (migration 0003; empty for older predictions)
    threshold: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasons: Mapped[dict] = mapped_column(JSON)
    model_version: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    officer: Mapped[User] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"Prediction(id={self.id}, officer_id={self.officer_id}, "
            f"ref={self.applicant_ref!r}, decision={self.decision!r})"
        )
