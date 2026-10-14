"""Keep the threshold each decision was made against (step 5.6).

History shows why a 10.6% risk was declined (a 1-month loan: threshold 7.5%) and a 25% risk
approved (6 months: 32.8%). Predictions made before this migration have no threshold (NULL).

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-14
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.add_column(sa.Column("threshold", sa.Float(), nullable=True))
        batch.create_check_constraint(
            "threshold_range", "threshold IS NULL OR (threshold >= 0 AND threshold <= 1)"
        )


def downgrade() -> None:
    with op.batch_alter_table("predictions") as batch:
        batch.drop_constraint(op.f("ck_predictions_threshold_range"), type_="check")
        batch.drop_column("threshold")
