"""Create the users and predictions tables (step 5.2).

Revision ID: 0001
Revises:
Create Date: 2026-10-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=120), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("role IN ('officer', 'admin')", name=op.f("ck_users_role_allowed")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
    )
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("officer_id", sa.Integer(), nullable=False),
        sa.Column("applicant_ref", sa.String(length=64), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("band", sa.String(length=20), nullable=False),
        sa.Column("decision", sa.String(length=10), nullable=False),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(length=20), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "probability >= 0 AND probability <= 1", name=op.f("ck_predictions_probability_range")
        ),
        sa.CheckConstraint(
            "decision IN ('approve', 'decline')", name=op.f("ck_predictions_decision_allowed")
        ),
        sa.ForeignKeyConstraint(
            ["officer_id"],
            ["users.id"],
            name=op.f("fk_predictions_officer_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
    )
    op.create_index(
        "ix_predictions_officer_created", "predictions", ["officer_id", "created_at"], unique=False
    )


def downgrade() -> None:
    op.drop_index("ix_predictions_officer_created", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_table("users")
