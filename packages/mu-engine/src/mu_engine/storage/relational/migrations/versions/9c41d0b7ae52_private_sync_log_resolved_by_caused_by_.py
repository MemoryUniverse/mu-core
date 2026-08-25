"""private sync log resolved_by caused_by_seq appended_at

The remainder of build step 2c's D-28 column list (mu-server-phase3-devices-sync-spec.md §13):
``pinned``/``resolution_origin`` landed in ``1ab7f7175baa``; ``resolved_by``, ``caused_by_seq`` and
``appended_at`` did not, and build step 8 (``PostgresPrivateSyncLog``) is where their absence
becomes visible.

Why each is genuinely missing rather than scope creep:

* ``caused_by_seq`` is **already a field on the shipped wire delta**
  (``PrivateDelta.caused_by_seq`` — *"replica-apply echo of log seq N; projector DROPS it"*). With
  no column it round-trips to ``None``, so §10.2/15's *"delta round-trip is field-identical"*
  cannot pass and, worse, a replica-apply echo becomes indistinguishable from an original write on
  read-back — the one distinction appender B's loop suppression rests on.
* ``resolved_by`` is pinned by ``CANONICAL-CONTRACTS.md:538`` (*"SUPERSEDE/REINSTATE additionally
  carry resolution_origin + resolved_by"*) and is added to ``PrivateDelta`` in the same change, so
  the column has a real source rather than being a column with nothing to put in it.
* ``appended_at`` is the hub's receive time and the ONLY possible basis for the retention window:
  ``SyncSettings.private_sync_log_retention_s`` is a duration, and a pruner over a table with no
  timestamp cannot be written. It has no reader today — that is **O-32**, and this removes one of
  its two blockers.

All three are additive and nullable-safe. ``appended_at`` is ``NOT NULL`` with a
``server_default=now()`` so existing rows backfill deterministically; the ORM sets it explicitly on
every new row, so the server default is a migration device and not a second writer.

Revision ID: 9c41d0b7ae52
Revises: f6fc5f7052cc
Create Date: 2026-08-25
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "9c41d0b7ae52"
down_revision: str | None = "f6fc5f7052cc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "private_sync_log",
        sa.Column("resolved_by", sa.String(length=128), nullable=True),
    )
    op.add_column(
        "private_sync_log",
        sa.Column("caused_by_seq", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "private_sync_log",
        sa.Column(
            "appended_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_column("private_sync_log", "appended_at")
    op.drop_column("private_sync_log", "caused_by_seq")
    op.drop_column("private_sync_log", "resolved_by")
