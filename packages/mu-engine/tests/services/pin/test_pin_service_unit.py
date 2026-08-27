"""``PinService.pin`` / ``.unpin`` — the write side (memory-health §5.2, §6.5, §9).

Pure unit tests over in-memory doubles: no store, no bus adapter, no container.

The five ordered steps of §5.2 are asserted as ORDER, not just as outcomes — a bound check that
runs after the write, or an event that publishes before it, is a real defect that outcome-only
assertions miss.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mu_contracts.domain.errors import (
    NamespaceIsolationError,
    PinAuthorizationError,
    PinLimitExceededError,
    PinTargetNotFoundError,
    PinTargetNotPinnableError,
)
from mu_contracts.domain.events import DomainEvent, MemoryPinned, MemoryUnpinned
from mu_contracts.domain.model.memory import (
    MemoryItem,
    MemoryKind,
    Namespace,
    State,
    Tier,
    Validity,
    Visibility,
)
from mu_contracts.domain.model.pin import PinRequest
from mu_contracts.domain.model.scope import ClientScope
from mu_engine.platform.clock import FrozenClock
from mu_engine.services.pin.service import PINNABLE_STATES, PinService
from mu_engine.services.pin.settings import PinSettings

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


@pytest.fixture
def ns() -> Namespace:
    return Namespace(
        org="org1", workspace="ws1", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


@pytest.fixture
def scope() -> ClientScope:
    return ClientScope(
        principal_id="u1",
        agent_principal_id="u1",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
    )


def _item(
    ns: Namespace,
    *,
    memory_id: str = "mem_1",
    pinned: bool = False,
    state: State = State.ACTIVE,
) -> MemoryItem:
    return MemoryItem(
        id=memory_id,
        namespace=ns,
        kind=MemoryKind.PROPOSITION,
        content="the user prefers dark mode",
        tier=Tier.MTM,
        state=state,
        validity=Validity(valid_at=_T0, recorded_at=_T0),
        last_seen=_T0,
        pinned=pinned,
        provenance_id=f"prov_{memory_id}",
    )


class _Repo:
    """An in-memory ``MemoryRepository`` recording the ORDER of every call it receives."""

    def __init__(self, items: dict[str, MemoryItem], *, pinned_count: int = 0) -> None:
        self.items = items
        self.trace: list[str] = []
        self.set_pinned_args: list[tuple[str, bool, str, str | None]] = []
        self._pinned_count = pinned_count
        self.enumerate_limits: list[int] = []

    async def get(self, ns: Namespace, id: str) -> MemoryItem | None:
        self.trace.append("get")
        return self.items.get(id)

    async def set_pinned(
        self,
        ns: Namespace,
        id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None = None,
    ) -> int:
        self.trace.append("set_pinned")
        self.set_pinned_args.append((id, pinned, by, reason))
        item = self.items[id]
        self.items[id] = (
            item.with_pin(by=by, at=at, reason=reason) if pinned else (item.without_pin(at=at))
        )
        return 7

    async def enumerate(
        self, ns: Namespace, *, limit: int, **kwargs: object
    ) -> tuple[list[MemoryItem], str | None]:
        self.trace.append("enumerate")
        self.enumerate_limits.append(limit)
        page = [_item(ns, memory_id=f"p{i}", pinned=True) for i in range(self._pinned_count)]
        return page[:limit], None


class _DirectWriteRepo(_Repo):
    """Writes the pin group FIELD BY FIELD instead of delegating to ``MemoryItem.with_pin``.

    ``with_pin`` is a ``model_copy(update=...)``, so a double built on it preserves every other
    field by construction and can never detect a service that clobbers one. This one cannot lean
    on that.
    """

    async def set_pinned(
        self,
        ns: Namespace,
        id: str,
        pinned: bool,
        *,
        at: datetime,
        by: str,
        reason: str | None = None,
    ) -> int:
        self.trace.append("set_pinned")
        self.set_pinned_args.append((id, pinned, by, reason))
        item = self.items[id]
        item.pinned = pinned
        item.pinned_at = at if pinned else None
        item.pinned_by = by if pinned else None
        item.pin_reason = reason if pinned else None
        return 7


class _BusSpy:
    def __init__(self) -> None:
        self.published: list[DomainEvent] = []

    async def publish(self, event: DomainEvent) -> None:
        self.published.append(event)


def _service(repo: _Repo, bus: _BusSpy, *, settings: PinSettings | None = None) -> PinService:
    return PinService(
        repo=repo,  # type: ignore[arg-type]
        bus=bus,  # type: ignore[arg-type]
        settings=settings or PinSettings(),
        clock=FrozenClock(_T0),
    )


# ═══════════════════════════════════════════════════════════════════ the happy path (§5.2) ══
async def test_pin_sets_the_whole_group_id_stably_and_emits_a_content_free_event(
    ns: Namespace, scope: ClientScope
) -> None:
    repo, bus = _Repo({"mem_1": _item(ns)}), _BusSpy()
    service = _service(repo, bus)

    result = await service.pin(scope, ns, PinRequest(memory_id="mem_1", reason="policy"))

    assert result.pinned is True
    assert result.pinned_at == _T0
    assert result.version == 7
    assert repo.set_pinned_args == [("mem_1", True, "u1", "policy")]
    stored = repo.items["mem_1"]
    assert (stored.pinned, stored.pinned_at, stored.pinned_by, stored.pin_reason) == (
        True,
        _T0,
        "u1",
        "policy",
    )
    assert [type(e) for e in bus.published] == [MemoryPinned]
    published = bus.published[0]
    assert isinstance(published, MemoryPinned)
    assert (published.id, published.by) == ("mem_1", "u1")
    # content-free: the reason is persisted on the item and NEVER carried on the bus (§2.4)
    assert "reason" not in MemoryPinned.model_fields


async def test_unpin_clears_the_whole_group(ns: Namespace, scope: ClientScope) -> None:
    # Built through `with_pin` on purpose: an item with the FULL audit group populated is the
    # only starting state that can prove `without_pin` clears all four fields rather than just
    # flipping the boolean and leaving a stale `pinned_by`/`pinned_at` behind.
    already_pinned = _item(ns).with_pin(by="u1", at=_T0, reason="policy")
    assert (already_pinned.pinned_by, already_pinned.pin_reason) == ("u1", "policy")
    repo, bus = _Repo({"mem_1": already_pinned}), _BusSpy()
    service = _service(repo, bus)

    result = await service.unpin(scope, ns, "mem_1")

    assert result.pinned is False
    assert result.pinned_at is None
    stored = repo.items["mem_1"]
    assert (stored.pinned, stored.pinned_at, stored.pinned_by, stored.pin_reason) == (
        False,
        None,
        None,
        None,
    )
    assert [type(e) for e in bus.published] == [MemoryUnpinned]


async def test_the_event_publishes_after_the_write(ns: Namespace, scope: ClientScope) -> None:
    """The ``IdempotentWriteScope`` contract: buffer, write, then publish. An event that fires
    before a failed write would announce a pin that does not exist."""

    class _OrderBus(_BusSpy):
        def __init__(self, repo: _Repo) -> None:
            super().__init__()
            self._repo = repo

        async def publish(self, event: DomainEvent) -> None:
            self._repo.trace.append("publish")
            await super().publish(event)

    repo = _Repo({"mem_1": _item(ns)})
    service = _service(repo, _OrderBus(repo))

    await service.pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert repo.trace.index("set_pinned") < repo.trace.index("publish")


# ══════════════════════════════════════════════════════════ step ordering + the bound (§5.2) ══
async def test_the_pin_bound_is_checked_before_the_write_and_reads_one_bounded_page(
    ns: Namespace, scope: ClientScope
) -> None:
    repo = _Repo({"mem_1": _item(ns)}, pinned_count=0)
    service = _service(repo, _BusSpy(), settings=PinSettings(max_pins_per_namespace=3))

    await service.pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert repo.trace == ["get", "enumerate", "set_pinned"]
    assert repo.enumerate_limits == [4]  # max + 1: one round trip, never a scan


async def test_pin_is_refused_once_the_namespace_is_at_its_bound(
    ns: Namespace, scope: ClientScope
) -> None:
    repo = _Repo({"mem_1": _item(ns)}, pinned_count=3)
    service = _service(repo, _BusSpy(), settings=PinSettings(max_pins_per_namespace=3))

    with pytest.raises(PinLimitExceededError):
        await service.pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert "set_pinned" not in repo.trace


async def test_repinning_an_already_pinned_item_skips_the_bound_check(
    ns: Namespace, scope: ClientScope
) -> None:
    """Idempotent re-pin must not be refused by a bound it is already inside."""
    repo = _Repo({"mem_1": _item(ns, pinned=True)}, pinned_count=99)
    service = _service(repo, _BusSpy(), settings=PinSettings(max_pins_per_namespace=1))

    result = await service.pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert result.pinned is True
    assert "enumerate" not in repo.trace


async def test_unpin_never_consults_the_bound(ns: Namespace, scope: ClientScope) -> None:
    repo = _Repo({"mem_1": _item(ns, pinned=True)}, pinned_count=99)
    service = _service(repo, _BusSpy(), settings=PinSettings(max_pins_per_namespace=1))

    await service.unpin(scope, ns, "mem_1")

    assert "enumerate" not in repo.trace


# ═══════════════════════════════════════════════════════ the authorization firewall (§6.5) ══
async def test_a_non_owner_pin_raises_and_never_reaches_the_store(ns: Namespace) -> None:
    """§6.5 obligation (d). Refused by the SAME tenancy substrate that isolates every read."""
    repo, bus = _Repo({"mem_1": _item(ns)}), _BusSpy()
    service = _service(repo, bus)
    intruder = ClientScope(
        principal_id="u2",
        agent_principal_id="u2",
        org_id="org1",
        workspace_id="ws1",
        session_id="s1",
    )

    with pytest.raises(NamespaceIsolationError):
        await service.pin(intruder, ns, PinRequest(memory_id="mem_1"))

    assert repo.trace == []
    assert bus.published == []


async def test_shared_origin_pin_is_off_in_v1(scope: ClientScope) -> None:
    shared = Namespace.shared(org="org1", workspace="ws1", session="s1")
    repo, bus = _Repo({}), _BusSpy()
    service = _service(repo, bus)

    with pytest.raises(PinAuthorizationError):
        await service.pin(scope, shared, PinRequest(memory_id="mem_1"))

    assert repo.trace == []


async def test_a_disabled_deployment_refuses_loud(ns: Namespace, scope: ClientScope) -> None:
    service = _service(_Repo({"mem_1": _item(ns)}), _BusSpy(), settings=PinSettings(enabled=False))

    with pytest.raises(PinAuthorizationError):
        await service.pin(scope, ns, PinRequest(memory_id="mem_1"))


async def test_pinning_does_not_change_the_caller_identity_set_any_reader_gets(
    ns: Namespace, scope: ClientScope
) -> None:
    """§6.5 obligation (a): ``AuthorizedIdsResolver.for_scope`` output is byte-identical with and
    without a pin — asserted against the REAL shipped resolver, not a double.

    The previous version of this test compared the item before/after a ``set_pinned`` double that
    itself called ``MemoryItem.with_pin`` (a ``model_copy``): preserving every other field was a
    tautology of pydantic, and the test passed even when ``PinService`` wrote no pin at all. This
    version exercises the actual authorization component instead.
    """
    from mu_engine.services.recall.authz import PrincipalAuthorizedIdsResolver
    from mu_engine.storage.domain.namespace import Namespace as EngineNamespace
    from mu_engine.storage.domain.namespace import Visibility as EngineVisibility

    resolver = PrincipalAuthorizedIdsResolver()
    engine_ns = EngineNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session=ns.session,
        visibility=EngineVisibility.PRIVATE,
    )
    shared_ns = EngineNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user="*",
        session=ns.session,
        visibility=EngineVisibility.SHARED,
    )
    before = (
        await resolver.for_scope(scope, engine_ns),
        await resolver.for_scope(scope, shared_ns),
    )

    repo = _Repo({"mem_1": _item(ns)})
    await _service(repo, _BusSpy()).pin(scope, ns, PinRequest(memory_id="mem_1"))
    assert repo.items["mem_1"].pinned is True  # the pin really happened

    after = (
        await resolver.for_scope(scope, engine_ns),
        await resolver.for_scope(scope, shared_ns),
    )
    assert after == before


def test_the_authorized_ids_resolver_cannot_see_a_pin_at_all(ns: Namespace) -> None:
    """§6.5 rules 1-2, asserted STRUCTURALLY. ``for_scope`` takes ``(scope, ns)`` and nothing
    else, so no pin state can reach ``authorized_ids`` or any compiled recall ``query_filter``
    even in principle — and the recall authorization module never mentions ``pinned``."""
    import inspect

    from mu_engine.services.recall import authz

    params = inspect.signature(authz.PrincipalAuthorizedIdsResolver.for_scope).parameters
    assert list(params) == ["self", "scope", "ns"]
    assert "pinned" not in inspect.getsource(authz)


async def test_pinning_leaves_every_read_relevant_field_untouched(
    ns: Namespace, scope: ClientScope
) -> None:
    """§6.5 obligation (c) as far as this plane can assert it: the pin write touches ONLY the pin
    group, so no read filter (which keys on id/η/tier/state/validity) can see a difference.

    Unlike the removed version, this one is checked against a repository double that writes the
    pin fields DIRECTLY rather than through ``MemoryItem.with_pin`` — so "everything else is
    unchanged" is a property of the service's call, not of ``model_copy``.
    """
    item = _item(ns)
    repo = _DirectWriteRepo({"mem_1": item})
    read_relevant_before = item.model_dump(
        include={"id", "namespace", "tier", "state", "validity", "provenance_id"}
    )

    await _service(repo, _BusSpy()).pin(scope, ns, PinRequest(memory_id="mem_1", reason="keep"))

    after = repo.items["mem_1"]
    assert after.pinned is True
    assert after.pin_reason == "keep"
    assert (
        after.model_dump(include={"id", "namespace", "tier", "state", "validity", "provenance_id"})
        == read_relevant_before
    )


# ══════════════════════════════════════════════════════════════════════════ not-found (§9) ══
async def test_a_missing_target_raises_non_enumeratingly(ns: Namespace, scope: ClientScope) -> None:
    repo, bus = _Repo({}), _BusSpy()
    service = _service(repo, bus)

    with pytest.raises(PinTargetNotFoundError) as exc:
        await service.pin(scope, ns, PinRequest(memory_id="does_not_exist"))

    assert "does_not_exist" not in str(exc.value)  # a probe learns nothing
    assert bus.published == []


# ═══════════════════════════════════════════════════ pinnable states — ENFORCED, not declared ══
@pytest.mark.parametrize("state", [State.SUPERSEDED, State.EXPIRED, State.DELETED])
async def test_pinning_a_settled_exit_is_refused(
    ns: Namespace, scope: ClientScope, state: State
) -> None:
    """Pin is a RETENTION override, so pinning something that has already LEFT is meaningless —
    and worse, a pinned row is unconditionally GC-ineligible (CANONICAL §7.10), so accepting it
    would strand a dead row in the graph forever.

    ``PINNABLE_STATES`` used to be asserted only as a set-membership fact about a constant, which
    passed while the service checked nothing.
    """
    repo, bus = _Repo({"mem_1": _item(ns, state=state)}), _BusSpy()

    with pytest.raises(PinTargetNotPinnableError):
        await _service(repo, bus).pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert "set_pinned" not in repo.trace
    assert bus.published == []


@pytest.mark.parametrize("state", [State.ACTIVE, State.ARCHIVED, State.QUARANTINED])
async def test_pinning_a_live_state_is_allowed(
    ns: Namespace, scope: ClientScope, state: State
) -> None:
    """Non-vacuity control for the three states that ARE pinnable."""
    repo = _Repo({"mem_1": _item(ns, state=state)})

    result = await _service(repo, _BusSpy()).pin(scope, ns, PinRequest(memory_id="mem_1"))

    assert result.pinned is True
    assert "set_pinned" in repo.trace


@pytest.mark.parametrize("state", [State.SUPERSEDED, State.EXPIRED, State.DELETED])
async def test_unpinning_a_settled_exit_is_always_allowed(
    ns: Namespace, scope: ClientScope, state: State
) -> None:
    """The release valve. An item that reached a settled exit while pinned (an EXPLICIT owner-
    driven supersede is legal, CANONICAL §7.10) would otherwise be permanently un-GC-able with no
    way to let it go — so the state gate applies to PIN only."""
    repo = _Repo({"mem_1": _item(ns, state=state, pinned=True)})

    result = await _service(repo, _BusSpy()).unpin(scope, ns, "mem_1")

    assert result.pinned is False
    assert "set_pinned" in repo.trace


def test_pinnable_states_exclude_the_settled_exits() -> None:
    """The constant itself, kept as documentation of the enforced set above."""
    assert State.DELETED not in PINNABLE_STATES
    assert State.SUPERSEDED not in PINNABLE_STATES
    assert State.EXPIRED not in PINNABLE_STATES


async def test_shared_origin_pin_is_refused_even_with_the_flag_on(scope: ClientScope) -> None:
    """Spec §5.2 step 1 line 265 is a CONJUNCTION: the flag AND "the caller is the item's origin
    principal (provenance ORIGIN, §7.10) or a workspace admin".

    mu-core can evaluate only the first conjunct — there is no provenance-ORIGIN reader and no
    admin role on this plane. An un-evaluable conjunct is FALSE, so the flag alone must NEVER
    grant pin over a shared item; before the fix it did.
    """
    shared = Namespace.shared(org="org1", workspace="ws1", session="s1")
    repo, bus = _Repo({}), _BusSpy()
    service = _service(repo, bus, settings=PinSettings(allow_shared_origin_pin=True))

    with pytest.raises(PinAuthorizationError, match="origin-principal"):
        await service.pin(scope, shared, PinRequest(memory_id="mem_1"))

    assert repo.trace == []
    assert bus.published == []
