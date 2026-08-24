"""private sync log pinned resolution_origin

CANONICAL-CONTRACTS.md §7.17 item 4a: the total order's two leading (dominant) terms,
``pinned`` and ``resolution_origin == "manual"``, MUST be readable from fields on the delta
itself (CANONICAL:777) — ``total_order_key`` (mu_engine.services.conflict.order) is a pure
function over two ``PrivateDelta`` candidates with no store/registry access. Additive columns
only, both nullable-safe (``pinned`` defaults False, ``resolution_origin`` defaults NULL) so
every existing row round-trips unchanged.

Revision ID: 1ab7f7175baa
Revises: 46ae4bcc2472
Create Date: 2026-08-24 22:15:18.603108
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "1ab7f7175baa"
down_revision: str | None = "46ae4bcc2472"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "private_sync_log",
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "private_sync_log",
        sa.Column("resolution_origin", sa.String(length=128), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("private_sync_log", "resolution_origin")
    op.drop_column("private_sync_log", "pinned")
