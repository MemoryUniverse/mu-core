"""``acl_entries.permission`` — the schema half of AD-67, and AD-62's DDL correction.

Unit, not integration, and deliberately so: everything asserted here is a property of the SCHEMA
and of the Alembic script directory, both of which are readable without a database. The
integration suites (``test_room_tables_migration_int.py`` and friends) then apply the chain for
real; this file is what makes a broken chain fail in the fast gate rather than only on the VM.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from mu_contracts.domain.model.governance import AclEntry as AclEntryModel
from mu_engine.storage.relational.schema import AclEntry, ProvenanceLedgerRow

pytestmark = pytest.mark.unit

_ALEMBIC_INI = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "mu_engine"
    / "storage"
    / "relational"
    / "alembic.ini"
)
_REVISION = "c4a1e07b9d33"  # THE REVISION UNDER TEST
_PRIOR = "b3d47c9a1e02"


def _script() -> ScriptDirectory:
    cfg = Config(str(_ALEMBIC_INI))
    cfg.set_main_option("script_location", str(_ALEMBIC_INI.parent / "migrations"))
    return ScriptDirectory.from_config(cfg)


def test_the_revision_is_the_single_chain_head_and_chains_from_the_previous_one() -> None:
    """A second head is the failure mode that makes ``alembic upgrade head`` ambiguous and leaves
    a deployment silently missing one branch's tables."""
    script = _script()
    assert list(script.get_heads()) == [_REVISION]
    assert script.get_revision(_REVISION).down_revision == _PRIOR


def test_the_acl_row_carries_permission_not_null() -> None:
    """Without it the standing grid means "some unspecified access" and stage-4 authorization has
    to fold the grant chain — the property ``AclEntry`` exists beside ``grants`` to avoid
    (governance-transfer-core-spec.md:204).

    NOT NULL matters as much as the column: a nullable permission is a row that grants an
    *unknown* amount, which is the same defect wearing a different type.
    """
    column = AclEntry.__table__.columns["permission"]
    assert column.nullable is False
    assert column.server_default is None, (
        "a server_default here would have to be a real Permission member, and every member GRANTS "
        "something — a forgotten INSERT would silently issue access"
    )


def test_the_contract_model_and_the_row_agree_on_the_governance_columns() -> None:
    """The contract is the vocabulary and this table is its storage; a field on one and not the
    other is the silent drift that produced two ACL shapes in the first place."""
    row_columns = set(AclEntry.__table__.columns.keys())
    assert set(AclEntryModel.model_fields) <= row_columns


def test_the_provenance_action_column_is_unconstrained_so_the_enum_can_widen() -> None:
    """AD-62's DDL correction, pinned. ``ProvenanceAction`` went from four members to the spec's
    eight and **no migration was owed**, because this column is a plain ``VARCHAR`` with no
    ``CHECK`` and no database ``ENUM`` type — the value set is enforced by pydantic at the
    boundary. A future ``CHECK`` here would turn every vocabulary addition into a migration on a
    table whose values are already validated before they arrive.
    """
    column = ProvenanceLedgerRow.__table__.columns["action"]
    assert column.type.python_type is str
    for table in (AclEntry.__table__, ProvenanceLedgerRow.__table__):
        assert not [
            c for c in table.constraints if c.__class__.__name__ == "CheckConstraint"
        ], f"{table.name} gained a CHECK constraint; the enum can no longer widen without DDL"
