"""Allow a third role, "demo": the public, read-only portfolio account (step 5.3).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONSTRAINT = "role_allowed"  # becomes ck_users_role_allowed through the naming convention


def _replace_role_check(roles: str) -> None:
    # SQLite can't change a constraint in place: batch mode rebuilds the table with the new one
    with op.batch_alter_table("users") as batch:
        batch.drop_constraint(op.f("ck_users_role_allowed"), type_="check")
        batch.create_check_constraint(CONSTRAINT, f"role IN ({roles})")


def upgrade() -> None:
    _replace_role_check("'officer', 'admin', 'demo'")


def downgrade() -> None:
    # Fails (on purpose) while demo accounts exist: remove or re-role them first
    _replace_role_check("'officer', 'admin'")
