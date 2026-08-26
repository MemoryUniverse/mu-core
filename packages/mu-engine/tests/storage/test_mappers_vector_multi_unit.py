"""RowMapper + naming contract tests for the three NEW vector backends (pgvector/chroma/faiss),
pure logic, mocks-free (marked unit) — mirrors ``test_mappers_unit.py``'s pattern for the
pre-existing Qdrant mapper.

Covers: round-trip fidelity (spec §8.1, delegated to ``QdrantMapper`` — see each mapper's
docstring for the DRY rationale), id-stability, and the identifier-safety invariant that makes the
per-backend partition-naming schemes injection-safe even though ``Namespace`` only forbids
separator characters, not arbitrary SQL/collection-name metacharacters.
"""

from __future__ import annotations

import pytest

from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, Polarity
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.chroma_mapper import ChromaMapper, chroma_collection_name
from mu_engine.storage.mappers.faiss_mapper import FaissMapper, faiss_collection_name
from mu_engine.storage.mappers.pgvector_mapper import PgVectorMapper, pgvector_table_name
from mu_engine.storage.mappers.qdrant_mapper import point_id

pytestmark = pytest.mark.unit

_NS = Namespace(org="o", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE)

# a workspace value packed with characters that are legal per Namespace._no_separator_injection
# (only "/:|\\ \t\n\x00" are forbidden — notably NOT space-free quotes/semicolons/parens) but
# would be dangerous if ever embedded raw into SQL DDL or a Chroma collection name — the naming
# schemes below must survive this untouched.
_HOSTILE_WORKSPACE = "w';DROPTABLEx;--.,()[]{}$%"


def _item() -> MemoryItem:
    return MemoryItem(
        content="Ada uses Postgres",
        kind=MemoryKind.PROPOSITION,
        namespace=_NS,
        owner_id="u",
        workspace_id="w",
        session_id="s",
        subject="Ada",
        predicate="uses",
        object="Postgres",
        polarity=Polarity.POSITIVE,
        artifact_ref="art_1",
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


# ------------------------------------------------------------------ round-trip + id-stability


def test_pgvector_roundtrip_and_id_stability() -> None:
    m = PgVectorMapper(dim=4)
    item = _item()
    row = m.to_store(item)
    assert row.point_id == point_id(item.id)
    assert "embedding" not in row.payload
    assert m.from_store(row) == item


def test_chroma_roundtrip_and_id_stability() -> None:
    m = ChromaMapper(dim=4)
    item = _item()
    row = m.to_store(item)
    assert row.point_id == point_id(item.id)
    assert "embedding" not in row.payload
    assert m.from_store(row) == item


def test_faiss_roundtrip_and_id_stability() -> None:
    m = FaissMapper(dim=4)
    item = _item()
    row = m.to_store(item)
    assert row.point_id == point_id(item.id)
    assert "embedding" not in row.payload
    assert m.from_store(row) == item


# ------------------------------------------------------------------ naming safety + determinism


@pytest.mark.parametrize(
    "namer",
    [pgvector_table_name, chroma_collection_name, faiss_collection_name],
)
def test_naming_is_deterministic(namer: object) -> None:
    ns = Namespace(org="o", workspace="w1", user="u", session="s", visibility=Visibility.PRIVATE)
    assert namer(ns, 8) == namer(ns, 8)  # type: ignore[operator]


@pytest.mark.parametrize(
    "namer",
    [pgvector_table_name, chroma_collection_name, faiss_collection_name],
)
def test_naming_differs_by_workspace(namer: object) -> None:
    ns_a = Namespace(org="o", workspace="w1", user="u", session="s", visibility=Visibility.PRIVATE)
    ns_b = Namespace(org="o", workspace="w2", user="u", session="s", visibility=Visibility.PRIVATE)
    assert namer(ns_a, 8) != namer(ns_b, 8)  # type: ignore[operator]


@pytest.mark.parametrize(
    "namer",
    [pgvector_table_name, chroma_collection_name, faiss_collection_name],
)
def test_naming_differs_by_org(namer: object) -> None:
    """CANONICAL §1 rule 6 pins the collection/graph grain at ``org``: two namespaces identical
    except for ``org`` must resolve to different physical partitions here too, on every one of
    the three backends that DELEGATE to ``QdrantMapper``'s payload shape but derive their OWN
    partition name (``ARCHITECTURE-CONFORMANCE.md`` §8/§10.4 — the org-missing defect was not
    unique to Qdrant's own ``collection_name``)."""
    ns_a = Namespace(
        org="org-a", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    ns_b = Namespace(
        org="org-b", workspace="w", user="u", session="s", visibility=Visibility.PRIVATE
    )
    assert namer(ns_a, 8) != namer(ns_b, 8)  # type: ignore[operator]


# NOTE: no ``test_naming_survives_underscore_boundary_ambiguity`` parametrization here. It existed
# for one round and was deleted (not fixed) after the verifier proved it could not fail: swapping
# in the PRE-REWORK pgvector/chroma/faiss namers (each of which hashed ``ns.workspace`` ALONE —
# ``org`` was entirely absent, the tracked §8 defect, not a "__"-join ambiguity) still passed it,
# because the adversarial pair's WORKSPACE strings ("ws" vs "eu__ws") already differ on their own —
# the test never needed ``org`` to participate in the digest at all. These three mappers never had
# the qdrant-side D-6 join-ambiguity bug (they had no ``org`` in the name pre-rework, so there was
# no two-segment join to be ambiguous), so there is no genuine regression here to pin; org-presence
# is already covered by ``test_naming_differs_by_org`` above. The Qdrant twin
# (``test_mappers_unit.py::test_qdrant_collection_name_survives_underscore_boundary_ambiguity``)
# stays — it DOES fail against Qdrant's actual pre-rework namer, which joined ``org``+``workspace``
# with a literal ``"__"`` before either was hashed.


def test_pgvector_table_name_is_sql_identifier_safe() -> None:
    ns = Namespace(
        org="o", workspace=_HOSTILE_WORKSPACE, user="u", session="s", visibility=Visibility.PRIVATE
    )
    name = pgvector_table_name(ns, 8)
    # only the hash-based, fixed-alphabet scheme reaches the DDL/DML f-strings in
    # pgvector_mtm.py — none of the hostile workspace's characters may leak through.
    assert all(c.isalnum() or c == "_" for c in name)
    assert "DROP TABLE" not in name
    assert "'" not in name


def test_chroma_collection_name_is_charset_safe() -> None:
    ns = Namespace(
        org="o", workspace=_HOSTILE_WORKSPACE, user="u", session="s", visibility=Visibility.PRIVATE
    )
    name = chroma_collection_name(ns, 8)
    # Chroma collection names: alnum + '.'/'_'/'-' only, 3-63 chars, start/end alnum.
    assert all(c.isalnum() or c in "._-" for c in name)
    assert 3 <= len(name) <= 63
    assert name[0].isalnum()
    assert name[-1].isalnum()
