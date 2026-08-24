"""principal registry — org_id + lifecycle columns on `principals`; NEW `principal_credentials`

mu-server-phase3-devices-sync-spec.md §4b.2 (tables A + B), build step 6a; ADR 0047 (pure
vocabulary lives in mu-core). Adds the tenancy root and lifecycle columns `Principal` needs to stop
being a two-column stub (``schema.py``'s ``Principal`` class), and the NEW `principal_credentials`
table (`ApiKeyRecord`'s field set, adopted verbatim per D-43) that maps a presented bearer
credential to a resolved principal — nothing in any repo held a credential row before this.

DEVIATION FROM §4b.2/D, RECORDED PER THE TEAM-LEAD'S EXPLICIT DIRECTION: the spec text instructs
folding these three edits into build step 2c's single revision (``1ab7f7175baa``), "not a second
revision". That instruction is now stale: ``1ab7f7175baa`` is already committed, and Alembic
migrations are append-only once committed — amending a landed revision would break any database
already stamped at it (this dev database included: ``select version_num from alembic_version``
against the real ``mu-dev-postgres`` returned ``1ab7f7175baa`` before this revision was authored).
So this is a NEW additive revision layered on top of it rather than an edit to the landed one.

Adding `NOT NULL` columns with no server default to `principals` is safe ONLY because the table has
zero rows (verified: no ``Principal(`` call site in any repo — ``grep -rn "Principal(" --include=
*.py mu-core mu-client mu-server`` returns only the class definition and the DTO — and `select
count(*) from principals` against the real dev Postgres returned 0 before this revision ran).
§4b.2/D states this precondition; the companion integration test
(``test_principal_registry_migration_int.py``) asserts it in the test rather than trusting it.
`status` is the one column with a stated literal default (`'active'`, §4b.2/A) and gets a matching
server default, so a row written without it (raw SQL, bypassing the ORM's own `default="active"`)
round-trips to the same value a `Principal(...)` construction would produce.

Revision ID: f6fc5f7052cc
Revises: 1ab7f7175baa
Create Date: 2026-08-24 23:40:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "f6fc5f7052cc"
down_revision: str | None = "1ab7f7175baa"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # --- principals: extended in place (§4b.2/A) -----------------------------------------------
    op.add_column("principals", sa.Column("org_id", sa.String(length=128), nullable=False))
    op.add_column("principals", sa.Column("display_name", sa.String(length=200), nullable=True))
    op.add_column(
        "principals",
        sa.Column(
            "status", sa.String(length=128), nullable=False, server_default=sa.text("'active'")
        ),
    )
    op.add_column("principals", sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))
    op.add_column("principals", sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.add_column("principals", sa.Column("disabled_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("principals", sa.Column("created_by", sa.String(length=128), nullable=False))
    op.add_column("principals", sa.Column("source", sa.String(length=128), nullable=False))
    op.create_index("ix_principals_org", "principals", ["org_id"])

    # --- principal_credentials: NEW (§4b.2/B, D-43) ---------------------------------------------
    op.create_table(
        "principal_credentials",
        sa.Column("key_id", sa.String(length=128), primary_key=True),
        sa.Column("kind", sa.String(length=128), nullable=False),
        sa.Column("org_id", sa.String(length=128), nullable=False),
        sa.Column("workspace_id", sa.String(length=128), nullable=False),
        sa.Column("principal_id", sa.String(length=128), nullable=False),
        sa.Column("hashed_secret", sa.String(length=128), nullable=False),
        sa.Column("display_prefix", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False, server_default=sa.text("''")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_by", sa.String(length=128), nullable=True),
        sa.Column("rotated_from_key_id", sa.String(length=128), nullable=True),
        sa.Column("issued_by", sa.String(length=128), nullable=False),
        sa.UniqueConstraint("hashed_secret", name="ux_principal_credentials_secret"),
    )
    op.create_index(
        "ix_principal_credentials_principal",
        "principal_credentials",
        ["org_id", "principal_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_principal_credentials_principal", table_name="principal_credentials")
    op.drop_table("principal_credentials")

    op.drop_index("ix_principals_org", table_name="principals")
    op.drop_column("principals", "source")
    op.drop_column("principals", "created_by")
    op.drop_column("principals", "disabled_at")
    op.drop_column("principals", "updated_at")
    op.drop_column("principals", "created_at")
    op.drop_column("principals", "status")
    op.drop_column("principals", "display_name")
    op.drop_column("principals", "org_id")
