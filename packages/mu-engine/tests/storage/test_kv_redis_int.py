"""KV/STM adapter — REAL mu-dev-cache (Redis/Valkey), ZERO mocks."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from redis.asyncio import Redis

from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace
from mu_engine.storage.mappers.redis_mapper import RedisMapper
from mu_engine.storage.user_registry import UserPrefixRegistryPort

pytestmark = pytest.mark.integration


async def test_put_get_roundtrip(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    item = make_item(ns, "the sky is blue")
    await adapter.put(item)
    got = await adapter.get(ns, item.id)
    assert got is not None
    assert got == item  # lossless blob round-trip on the REAL store
    await adapter.evict(ns, item.id)
    assert await adapter.get(ns, item.id) is None


async def test_recency_floor(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    items = [make_item(ns, f"fact {i}") for i in range(3)]
    for it in items:
        await adapter.put(it)
    recent = await adapter.recent(ns, limit=10)
    assert {s.item.id for s in recent} == {it.id for it in items}
    assert all(s.is_floor for s in recent)  # STM floor members (spec §1.1)
    for it in items:
        await adapter.evict(ns, it.id)


async def test_namespace_isolation(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    adapter = RedisStmAdapter(redis_client)
    ns_a = make_ns(session="sa")
    ns_b = make_ns(session="sb")  # differs in one η component
    item_a = make_item(ns_a, "belongs to A")
    await adapter.put(item_a)
    # a read scoped to B never returns A's row (to_prefix() key partition).
    assert await adapter.get(ns_b, item_a.id) is None
    assert (await adapter.recent(ns_b, limit=10)) == []
    await adapter.evict(ns_a, item_a.id)


async def test_write_time_dedup_skips_the_second_row(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D4 (conformance D-8): two `put()`s of byte-identical content (same content_hash, two
    DIFFERENT random ids — exactly what two independent `add()` calls produce) must land as
    ONE STM row, not two (the empirically observed `ada_coffee` written-twice defect)."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")
    assert first.id != second.id
    assert first.content_hash == second.content_hash

    first_resident_id = await adapter.put(first)
    second_resident_id = await adapter.put(second)

    # RETURN-IDEMPOTENCY (add() return contract, DATA-QUALITY-REASSESSMENT §3 "add() idempotency"):
    # put() reports the id the store ACTUALLY kept — the SECOND call's own minted id was never
    # resident, so put() must hand back the FIRST call's id, not `second.id`.
    assert first_resident_id == first.id
    assert second_resident_id == first.id
    assert second_resident_id != second.id

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 1, "duplicate content forked a second STM row"
    assert recent[0].item.id == first.id, "the FIRST (winner) id must be kept, not overwritten"
    assert await adapter.get(ns, second.id) is None, "the duplicate's own id must never be written"

    await adapter.evict(ns, first.id)


async def test_write_time_dedup_bumps_recency_on_the_winner(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A later duplicate still moves the WINNER to the front of the recency ordering (a
    resubmission of the same fact is treated as a fresh mention, not silently dropped)."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    base = datetime.now(UTC)
    winner = make_item(ns, "Ada drinks black coffee")
    winner = winner.model_copy(update={"created_at": base})
    older_other = make_item(ns, "an unrelated fact")
    older_other = older_other.model_copy(update={"created_at": base + timedelta(seconds=1)})
    dup = make_item(ns, "Ada drinks black coffee")
    dup = dup.model_copy(update={"created_at": base + timedelta(seconds=2)})

    await adapter.put(winner)
    await adapter.put(older_other)
    # before the dup: older_other (later created_at) is more recent than winner.
    before = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in before) == older_other.id

    await adapter.put(dup)  # bump: winner should now be most-recent again (dup's created_at).
    after = await adapter.recent(ns, limit=10)
    assert next(s.item.id for s in after) == winner.id
    assert len(after) == 2  # still no forked third row

    await adapter.evict(ns, winner.id)
    await adapter.evict(ns, older_other.id)


async def test_write_time_dedup_bumps_mention_count_on_the_winner(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D3 fix (AD-266b, ``docs/tracking/PROTOTYPE-DEBT-0924.md``): before this fix,
    ``_bump_if_duplicate`` moved the winner's recency score + TTL but never re-read/rewrote its
    payload, so ``mention_count`` stayed pinned at 1 forever and
    ``IngestSettings.mention_promote`` (default 2) could never fire against a real STM row.

    MUTATION CHECK (run, red, restored): drop the ``"mention_count": current.mention_count + 1``
    key from ``_bump_if_duplicate``'s ``update`` dict — this test goes red
    (``mention_count == 1``) while ``test_write_time_dedup_bumps_recency_on_the_winner`` above
    stays green, proving that test alone could not have caught D3."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee every morning")
    second = make_item(ns, "Ada drinks black coffee every morning")
    third = make_item(ns, "Ada drinks black coffee every morning")
    assert first.content_hash == second.content_hash == third.content_hash

    await adapter.put(first)
    row = await adapter.get(ns, first.id)
    assert row is not None and row.mention_count == 1

    await adapter.put(second)
    row = await adapter.get(ns, first.id)
    assert (
        row is not None and row.mention_count == 2
    ), f"first repeat did not bump mention_count (got {row.mention_count if row else None})"

    await adapter.put(third)
    row = await adapter.get(ns, first.id)
    assert row is not None and row.mention_count == 3, "second repeat did not bump mention_count"
    # last_seen must advance with each repeat too (D2), and access_count/created_at (a DIFFERENT
    # axis — genuine RECALLS, not re-assertions) must NOT move just from a repeat write.
    assert row.last_seen == third.created_at
    assert row.access_count == 0
    assert row.created_at == first.created_at

    await adapter.evict(ns, first.id)


async def test_write_time_dedup_toggle_off_allows_duplicates(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """``stm_dedup_enabled=False`` (the ``MU_INGEST__STM_DEDUP=false`` DI path) reverts to the
    pre-fix behavior — proves the toggle genuinely drives the mechanism, not just a default."""
    adapter = RedisStmAdapter(redis_client, stm_dedup_enabled=False)
    ns = make_ns()
    first = make_item(ns, "Ada drinks black coffee")
    second = make_item(ns, "Ada drinks black coffee")

    first_resident_id = await adapter.put(first)
    second_resident_id = await adapter.put(second)

    # the toggle drives put()'s return contract too: dedup off -> always the given id back,
    # never a substituted "existing winner" id (mint-new behavior, unchanged from pre-D4).
    assert first_resident_id == first.id
    assert second_resident_id == second.id

    recent = await adapter.recent(ns, limit=10)
    assert len(recent) == 2, "toggle off must allow the duplicate through (pre-fix parity)"

    await adapter.evict(ns, first.id)
    await adapter.evict(ns, second.id)


# =====================================================================================
# AD-268 (ADR 0075, PROTOTYPE-DEBT-0924.md D5) — the durable, cross-namespace user-prefix
# registry. Proves the adapter half of the fix against REAL Redis/Valkey: a namespace that has
# never fired a bus event is still durably discoverable, which is exactly the gap
# ``MaintenanceLoop``'s own probe measured (``active_user_count 0`` after a restart).
# =====================================================================================


async def _cleanup_registry(client: Redis, *prefixes: UserPrefix) -> None:
    if prefixes:
        await client.zrem(RedisMapper.user_registry_key(), *(str(p) for p in prefixes))


async def test_adapter_satisfies_user_prefix_registry_port(redis_client: Redis) -> None:
    """Structural (PEP 544) contract check: a real ``RedisStmAdapter`` IS a
    ``UserPrefixRegistryPort`` with no inheritance and no import of that module — mirrors the
    same structural-satisfaction proof every other narrow capability port in this package gets."""
    assert isinstance(RedisStmAdapter(redis_client), UserPrefixRegistryPort)


async def test_put_registers_the_namespace_in_the_durable_registry(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A single ``put()`` — no bus event, no second call — is enough for this namespace's
    ``UserPrefix`` to show up in a FRESH adapter instance's ``list_user_prefixes()``. Proves the
    registration survives independently of any in-process state (a new ``RedisStmAdapter`` here
    shares nothing with the one that wrote, exactly like a restarted daemon reconnecting)."""
    adapter = RedisStmAdapter(redis_client)
    ns = make_ns()
    item = make_item(ns, "Ada prefers filter coffee")
    prefix = UserPrefix(ns)
    try:
        await adapter.put(item)
        fresh = RedisStmAdapter(redis_client)  # a second instance — no shared Python state
        prefixes = await fresh.list_user_prefixes(limit=1000)
        assert prefix in prefixes
    finally:
        await adapter.evict(ns, item.id)
        await _cleanup_registry(redis_client, prefix)


async def test_list_user_prefixes_is_most_recently_active_first(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """Two distinct users, written oldest-first: the registry orders by LAST WRITE, not by
    insertion order, and a namespace written to twice appears once (`ZADD` overwrites the score,
    module docstring) — never a duplicate entry piling up per write."""
    adapter = RedisStmAdapter(redis_client)
    ns_older = make_ns(user="older-user")
    ns_newer = make_ns(user="newer-user")
    prefix_older, prefix_newer = UserPrefix(ns_older), UserPrefix(ns_newer)
    try:
        item_older = make_item(ns_older, "older user's memory")
        await adapter.put(item_older)
        item_newer = make_item(ns_newer, "newer user's memory")
        await adapter.put(item_newer)
        # a second write to the OLDER user's namespace should re-promote it to most-recent.
        item_older_again = make_item(ns_older, "older user's second memory")
        await adapter.put(item_older_again)

        prefixes = await adapter.list_user_prefixes(limit=1000)
        assert prefixes.count(prefix_older) == 1  # deduped despite two writes
        assert prefixes.index(prefix_older) < prefixes.index(prefix_newer)

        await adapter.evict(ns_older, item_older.id)
        await adapter.evict(ns_older, item_older_again.id)
        await adapter.evict(ns_newer, item_newer.id)
    finally:
        await _cleanup_registry(redis_client, prefix_older, prefix_newer)


async def test_list_user_prefixes_limit_is_never_exceeded(
    redis_client: Redis,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The "never unbounded" discipline every enumeration verb in this package follows
    (``tier_capabilities.py``): a ``limit`` narrower than the true registered population is
    honored, not silently widened."""
    adapter = RedisStmAdapter(redis_client)
    namespaces = [make_ns(user=f"bounded-user-{i}") for i in range(3)]
    prefixes = [UserPrefix(ns) for ns in namespaces]
    items = [make_item(ns, f"fact {i}") for i, ns in enumerate(namespaces)]
    try:
        for item in items:
            await adapter.put(item)
        page = await adapter.list_user_prefixes(limit=2)
        assert len(page) == 2
        for ns, item in zip(namespaces, items, strict=True):
            await adapter.evict(ns, item.id)
    finally:
        await _cleanup_registry(redis_client, *prefixes)
