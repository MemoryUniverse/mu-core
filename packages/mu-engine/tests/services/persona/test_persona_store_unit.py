"""``InMemoryPersonaRepository`` + the PRIVATE-only gate (``persona-design.md`` §3.2/§3.3).

The tenancy tests here are the ones that matter: persona is inferred from a user's most private
data, so a store access that forgets to key on ``Namespace.to_prefix()`` is a cross-user leak, not
a cache miss (CANONICAL §1 rule 5 / CLAUDE.md rule 4).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.persona import PersonaProfile, PersonaSlot, SlotValue
from mu_engine.services.persona.store import (
    InMemoryPersonaRepository,
    PersonaVersionConflictError,
    assert_private,
)

from .conftest import T0

pytestmark = pytest.mark.unit


def _profile(ns: Namespace, *, version: int = 1, brief: str = "b", at: datetime = T0):
    return PersonaProfile(
        namespace=ns,
        slots={
            PersonaSlot.PREFERENCE: SlotValue(
                value="dark mode", confidence=0.9, support_ids=("mem_1",), updated_at=at
            )
        },
        overall_brief=brief,
        brief_etag=f"etag-{brief}",
        version=version,
        rebuilt_at=at,
        source_memory_count=3,
    )


# --------------------------------------------------------------------------- tenancy (§1 r5)
async def test_two_users_personas_do_not_collide(ns: Namespace, other_ns: Namespace):
    """Same org, same workspace, DIFFERENT user. The partition is the key — not a filter over a
    shared bag — so one user's portrait is unreachable from the other's η."""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, brief="mine"))
    await repo.upsert(_profile(other_ns, brief="theirs"))

    assert (await repo.get(ns)).overall_brief == "mine"
    assert (await repo.get(other_ns)).overall_brief == "theirs"
    assert await repo.load_brief(ns) == ("mine", "etag-mine")
    assert await repo.load_brief(other_ns) == ("theirs", "etag-theirs")


async def test_an_unwritten_partition_reads_as_absent(ns: Namespace, other_ns: Namespace):
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns))
    assert await repo.get(other_ns) is None
    assert await repo.load_brief(other_ns) is None


async def test_the_key_spans_sessions(ns: Namespace):
    """A persona must SURVIVE into the user's next session, and this is the assertion that says so.

    The first build of this store keyed on ``ns.to_prefix()``, whose sixth segment is the session
    (``domain/model/memory.py:137-146``). The consequence was invisible to every other test here
    and fatal to the subsystem: ``load_brief`` missed on every new session, a rebuild there minted
    a fresh version 1, and the ``min_support`` create-gate restarted from zero — persona could
    never accumulate, which is the entire point of it. Spec line 10 keys persona on
    ``(workspace, namespace, user)`` with no session, and ADR 0030 (``platform/tenancy.py:39-41``)
    says a PRIVATE session is "a filter/provenance stamp, never an isolation boundary".
    """
    repo = InMemoryPersonaRepository()
    other_session = ns.model_copy(update={"session": "s2"})
    await repo.upsert(_profile(ns, brief="v1"))

    assert (await repo.get(other_session)).overall_brief == "v1"
    assert await repo.load_brief(other_session) == ("v1", "etag-v1")
    # ...and a write from the new session CONTINUES the same record rather than forking one.
    await repo.upsert(_profile(other_session, version=2, brief="v2"))
    assert (await repo.get(ns)).version == 2


async def test_every_other_isolation_dimension_is_still_in_the_key(ns: Namespace):
    """Dropping the session must not drop anything else: org, workspace and user still partition.
    (Visibility does too, structurally — a non-PRIVATE key is refused before it is built.)"""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, brief="mine"))
    for update in ({"org": "org2"}, {"workspace": "ws2"}, {"user": "u2"}):
        assert await repo.get(ns.model_copy(update=update)) is None


# ------------------------------------------------------------------ PRIVATE-only (§3.1/§5.4)
async def test_a_shared_partition_is_refused_on_every_method(shared_ns: Namespace):
    """§0 line 53: a room has no personality. Refused for READS too, not just writes — the
    partition must never even be looked at."""
    repo = InMemoryPersonaRepository()
    with pytest.raises(NamespaceIsolationError):
        await repo.get(shared_ns)
    with pytest.raises(NamespaceIsolationError):
        await repo.load_brief(shared_ns)


def test_the_refusal_is_non_enumerating(shared_ns: Namespace):
    """The fixed ``"not found"`` envelope (platform-layer0 §5): a prober cannot tell "SHARED has no
    persona" from "this user has none", and the namespace is never echoed back."""
    with pytest.raises(NamespaceIsolationError) as exc:
        assert_private(shared_ns, "persona.get")
    assert str(exc.value) == "not found"
    assert shared_ns.to_prefix() not in str(exc.value)


def test_a_shared_profile_cannot_even_be_constructed(shared_ns: Namespace):
    """The §3.1 line 149 validator — the storage-level twin of the §5.4 firewall."""
    with pytest.raises(ValueError, match="PRIVATE-only"):
        _profile(shared_ns)


def test_assert_private_admits_only_private():
    for visibility in Visibility:
        ns = Namespace(
            org="o",
            workspace="w",
            user="*" if visibility is Visibility.SHARED else "u",
            session="s",
            visibility=visibility,
        )
        if visibility is Visibility.PRIVATE:
            assert_private(ns, "op")
        else:
            with pytest.raises(NamespaceIsolationError):
                assert_private(ns, "op")


# ---------------------------------------------------------- optimistic versioning (line 166)
async def test_upsert_is_strictly_monotone_by_one(ns: Namespace):
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, version=1))
    await repo.upsert(_profile(ns, version=2, brief="v2"))
    assert (await repo.get(ns)).version == 2


async def test_a_stale_version_write_is_refused(ns: Namespace):
    """Spec line 166. A second writer landing in between must LOSE, not overwrite."""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, version=1))
    await repo.upsert(_profile(ns, version=2))
    with pytest.raises(PersonaVersionConflictError):
        await repo.upsert(_profile(ns, version=2, brief="stale writer"))
    assert (await repo.get(ns)).overall_brief == "b"


async def test_a_version_gap_is_refused(ns: Namespace):
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, version=1))
    with pytest.raises(PersonaVersionConflictError):
        await repo.upsert(_profile(ns, version=5))


async def test_a_first_write_must_be_version_one(ns: Namespace):
    repo = InMemoryPersonaRepository()
    with pytest.raises(PersonaVersionConflictError):
        await repo.upsert(_profile(ns, version=3))


# --------------------------------------------------- right-to-be-forgotten (line 169)
async def test_delete_erases_the_record_entirely(ns: Namespace):
    """Spec line 169. A persona is an LLM-written portrait of a person inferred from their private
    data; when the memories go, it must go — brief, slots and ``support_ids`` provenance alike."""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns))
    assert await repo.delete(ns) is True
    assert await repo.get(ns) is None
    assert await repo.load_brief(ns) is None


async def test_delete_is_idempotent_and_partition_scoped(ns: Namespace, other_ns: Namespace):
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns))
    await repo.upsert(_profile(other_ns))
    assert await repo.delete(ns) is True
    assert await repo.delete(ns) is False  # nothing left to erase is not an error
    assert (await repo.get(other_ns)) is not None  # the neighbour is untouched


async def test_delete_refuses_a_shared_partition(shared_ns: Namespace):
    repo = InMemoryPersonaRepository()
    with pytest.raises(NamespaceIsolationError):
        await repo.delete(shared_ns)


async def test_a_rebuilt_persona_after_delete_starts_at_version_one(ns: Namespace):
    """The erase is real, not a tombstone: the next write is a FIRST write."""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, version=1))
    await repo.upsert(_profile(ns, version=2))
    await repo.delete(ns)
    with pytest.raises(PersonaVersionConflictError):
        await repo.upsert(_profile(ns, version=3))
    await repo.upsert(_profile(ns, version=1))


async def test_the_stored_record_is_not_aliased(ns: Namespace):
    """Handing out the stored object would let a caller mutate the "persisted" record in place
    and walk straight past the version check."""
    repo = InMemoryPersonaRepository()
    await repo.upsert(_profile(ns, version=1))
    fetched = await repo.get(ns)
    fetched.overall_brief = "tampered"
    assert (await repo.get(ns)).overall_brief == "b"
