"""acl_entries.permission — the column the standing ACL grid was missing (AD-67)

``governance-transfer-core-spec.md:189-201`` declares ``AclEntry.permission``; the shipped
contract (``mu_contracts/domain/model/governance.py``) and this schema both dropped it. Without
it the row means *"some unspecified access"*, and ``AccessController`` stage 4 (*"an active
``AclEntry`` grants the permission"*, spec:522) cannot be a direct row lookup — it would have to
fold the grant chain, destroying the *"reads never fold the chain"* property (spec:204) that is
the entire reason this table exists beside ``grants``.

The contract model gains ``org_id`` in the same change; this table has carried that column since
the initial revision (``46ae4bcc2472``), so ONLY ``permission`` is a DDL delta.

**NOT NULL with no ``server_default``, and the reason is a security argument, not a style one.**
The only value a default could take is a member of ``Permission`` — and every member GRANTS
something. A ``server_default`` of ``'read'`` would silently issue READ on every pre-existing row
and on every future INSERT that forgot the column: exactly the silent widening this whole plane
exists to prevent. So the column is added strict, and a populated table is refused LOUDLY instead
(the pattern ``b3d47c9a1e02`` established for an undecidable NOT NULL backfill).

The refusal is not expected to fire anywhere: ``acl_entries`` has **no adapter in any repo**.
``mu_engine.storage.adapters.relational_control`` binds only ``AuditLogRow`` and
``MemoryProvenance``; the shipped governance plane writes its own ``transfer_acl`` table
(``mu-server/src/mu_server/transfer/store_pg.py:203``). Verified 2026-08-28 by
``grep -rn "acl_entries" --include=*.py`` over mu-core, mu-client and mu-server: the only hits are
this schema, this migration and the initial revision. The guard exists because "no writer today"
is a fact about today, and a migration that assumes it without checking would corrupt the one
database that proved it wrong.

⚠ **``provenance_ledger.action`` is deliberately NOT touched by this revision** — and that is the
correction AD-62 needs. That column is a plain ``VARCHAR`` with no ``CHECK`` and no database
``ENUM`` type (``46ae4bcc2472:313``), so extending ``ProvenanceAction`` from four members to the
spec's eight produces **zero schema delta**. The four-member list at ``schema.py`` was a COMMENT,
not a constraint. No migration was owed for it, and inventing one would have been ceremony.

Revision ID: c4a1e07b9d33
Revises: b3d47c9a1e02
Create Date: 2026-08-28 00:00:00.000000
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c4a1e07b9d33"
down_revision: str | None = "b3d47c9a1e02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: The table and column this revision owns. Named once so ``upgrade`` and ``downgrade`` cannot
#: disagree about what it added (the discipline ``b3d47c9a1e02:246`` states for the same reason).
_TABLE = "acl_entries"
_COLUMN = "permission"
#: Matches the identifier/enum width every other ``String`` column in ``schema.py`` uses, which is
#: what keeps this schema buildable on MySQL (``schema.py`` module docstring, item 1).
_WIDTH = 128


def _has_rows() -> bool:
    """Whether ``acl_entries`` currently holds anything. Table name is a module constant, never
    caller input."""
    bind = op.get_bind()
    return bool(bind.execute(sa.text(f"SELECT 1 FROM {_TABLE} LIMIT 1")).first())  # noqa: S608


def upgrade() -> None:
    if _has_rows():
        raise RuntimeError(
            f"cannot add NOT NULL `{_TABLE}.{_COLUMN}`: the table already holds rows and the "
            "permission an existing row conferred is not derivable from anything true. Every "
            "candidate default GRANTS access, so inventing one would widen authorization "
            "silently. Backfill `permission` from the originating grant by hand, then re-run "
            "this revision."
        )
    op.add_column(_TABLE, sa.Column(_COLUMN, sa.String(length=_WIDTH), nullable=False))


def downgrade() -> None:
    op.drop_column(_TABLE, _COLUMN)
