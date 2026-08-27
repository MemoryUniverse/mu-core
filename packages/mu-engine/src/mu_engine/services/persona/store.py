"""``InMemoryPersonaRepository`` — the LOCAL-plane default ``PersonaRepository`` (spec §3.2).

Authority: ``persona-design.md`` §3.2 (lines 151-161) + §3.3 (lines 163-169).

A REAL in-process adapter, not a test double — the same sanctioned pattern as
``lifecycle/conflict.py``'s ``InMemoryConflictRecordRepository`` and
``pipelines/ledger.py``'s ``InMemoryStageLedger``: the composition root wires it when no durable
store is configured, which is what lets FULL-LOCAL persona work today with zero API keys and zero
containers.

**The durable adapter spec line 161 names is NOT built here, and that is deliberate — reported,
not silently skipped.** Spec line 161 puts the default on "the Redis/KV plane (``KVStorePort``,
memory-layer §3)". **There is no ``KVStorePort`` in this repo** — ``mu_contracts/ports/`` has no
kv module and a workspace-wide grep for the name returns nothing; the nearest shipped things are
the ``StmTierRepository`` adapters, which are memory-TIER repositories, not a generic keyed-doc
store. Building one would mean inventing a port CANONICAL has not ratified, and
``storage/adapters/**`` is outside this lane's file ownership besides. So the port is honoured by
an in-process adapter with the full §3.3 semantics, and the KV/Postgres binding lands with the
port.

TENANCY (CANONICAL §1 rule 5 / CLAUDE.md rule 4): the map is keyed by :func:`persona_key` and by
nothing else, so a profile physically cannot be reachable from another tenant's, user's or
visibility's key — the partition IS the key, not a filter applied to a shared bag.
"""

from __future__ import annotations

import structlog

from mu_contracts.domain.errors import MemoryUniverseError, NamespaceIsolationError
from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.domain.model.persona import PersonaProfile

__all__ = [
    "InMemoryPersonaRepository",
    "PersonaVersionConflictError",
    "assert_private",
    "persona_key",
]

_log = structlog.get_logger("mu_engine.services.persona.store")


class PersonaVersionConflictError(MemoryUniverseError):
    """A stale-version ``upsert`` was refused (spec line 166: *"``upsert`` rejects a stale-version
    write"*).

    **Placement delta (recorded).** CANONICAL §7.27's "New named errors" rule puts named errors in
    ``mu_contracts.domain.errors``; that file is shared and outside this lane's ownership, so this
    one is declared beside its raiser — the same place ``mu_engine.providers._contracts`` declares
    ``ModelLayerError``/``ModelGroupUnavailableError``. It should move to the contracts error
    hierarchy when persona's contracts edits are granted.
    """


def assert_private(ns: Namespace, operation: str) -> None:
    """Refuse a non-PRIVATE persona namespace BEFORE any read or write touches a store.

    ``PersonaProfile``'s ``_must_be_private`` validator already makes a SHARED profile
    unconstructible (spec §3.1 line 149), but that fires at the END of a rebuild — after the
    evidence read. §0 line 53 and §5.4 rule 3 are stronger than "it cannot be persisted": a room
    has no persona, so a room's partition must never be READ for one either.

    Raises the non-enumerating ``NamespaceIsolationError("not found")`` — the same fixed envelope
    ``DefaultTenancyGuard`` uses, so a probe cannot tell "SHARED partitions have no persona" from
    "this user has none" (platform-layer0 §5 non-enumerating denial).
    """
    if ns.visibility is not Visibility.PRIVATE:
        # Content-free: a namespace prefix + an enum + the operation name, never a slot value.
        _log.warning(
            "persona_non_private_refused",
            ns=ns.to_prefix(),
            visibility=ns.visibility.value,
            operation=operation,
        )
        raise NamespaceIsolationError("not found")


def persona_key(ns: Namespace) -> str:
    """The persona partition key — ``mu/{org}/{workspace}/{visibility}/{user_slot}/``.

    **NOT ``ns.to_prefix()``, and that is the point.** ``to_prefix()`` is six segments and the
    sixth is the SESSION (``domain/model/memory.py:137-146``), so keying persona on it makes a
    user's portrait unreadable from their next session and restarts the ``min_support``
    create-gate (§3.3 line 165) from zero every time — persona could never accumulate, which is
    the entire point of the subsystem. Spec line 10 keys persona on ``(workspace, namespace,
    user)`` with no session, and ADR 0030 (quoted in ``platform/tenancy.py:39-41``) states that a
    PRIVATE session is "a filter/provenance stamp, never an isolation boundary".

    ``UserPrefix`` (``mu_contracts/domain/model/lifecycle.py:56-58``) is the shipped, ratified
    "session-spanning lease-grain prefix" for exactly this grain — adopted rather than re-derived,
    so persona and the lifecycle sweeps cannot drift into two different notions of "one user".
    Every other isolation dimension (org, workspace, visibility, user slot) is still in the key.
    """
    assert_private(ns, "persona.key")
    return str(UserPrefix(ns))


class InMemoryPersonaRepository:
    """A ``PersonaRepository`` (``mu_contracts.ports.persona``) over one process's memory.

    No lock is taken: every method is a single synchronous critical section with no ``await``
    inside it, so the event loop cannot interleave two writers mid-update. A durable adapter needs
    a real compare-and-set; that is the whole reason ``upsert`` is version-checked rather than
    last-write-wins.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, PersonaProfile] = {}

    async def get(self, ns: Namespace) -> PersonaProfile | None:
        assert_private(ns, "persona.get")
        profile = self._profiles.get(persona_key(ns))
        # Deep copy: `PersonaProfile` is not frozen, and handing out the stored object would let
        # a caller mutate the "persisted" record in place and defeat the version check.
        return None if profile is None else profile.model_copy(deep=True)

    async def upsert(self, profile: PersonaProfile) -> None:
        """Optimistic, strictly monotone by one (spec line 166).

        The sleep-time worker has exclusive write access (§3.3 line 166: "reads never mutate
        persona"), so a version that is not exactly ``existing.version + 1`` means a SECOND writer
        landed in between — which is precisely the case that must be refused rather than
        overwritten. A first write must be ``version == 1``; ``version == 0`` is reserved for a
        profile that was never built.
        """
        assert_private(profile.namespace, "persona.upsert")
        key = persona_key(profile.namespace)
        existing = self._profiles.get(key)
        expected = 1 if existing is None else existing.version + 1
        if profile.version != expected:
            raise PersonaVersionConflictError(
                f"persona version {profile.version} is stale (expected {expected})"
            )
        self._profiles[key] = profile.model_copy(deep=True)

    async def load_brief(self, ns: Namespace) -> tuple[str, str] | None:
        """``(overall_brief, etag)`` — the warm/hot read (spec line 158, preload §4.6).

        This is the ONLY persona read the query-time path ever performs, and it is a load BY KEY:
        persona is never vector-searched (spec line 161), and this method takes no query, no
        candidate set and no ``authorized_ids`` — it cannot participate in a recall filter even by
        accident (§5.4 rule 2).
        """
        assert_private(ns, "persona.load_brief")
        profile = self._profiles.get(persona_key(ns))
        return None if profile is None else (profile.overall_brief, profile.brief_etag)

    async def delete(self, ns: Namespace) -> bool:
        """Right-to-be-forgotten (spec line 169) — erase the whole record, not only the brief.

        Idempotent: ``False`` when there was nothing to erase. Nothing is retained: the portrait,
        the slot values and the ``support_ids`` provenance all go, because all three are derived
        from — and are as sensitive as — the memories being forgotten (CLAUDE.md rule 3).
        """
        assert_private(ns, "persona.delete")
        return self._profiles.pop(persona_key(ns), None) is not None
