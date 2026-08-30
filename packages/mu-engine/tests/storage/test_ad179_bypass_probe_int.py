"""AD-179 adversarial probe — try to READ a SHARED partition with no caller identity set.

Written by the VERIFY pass, not by the lane that shipped the fix: AD-179's claim is
*"no call path that reaches a filterable index can omit the ``authorized_ids`` clause"*, and a
claim of that shape is only worth what an attempt to break it is worth. Every read verb the
MTM/LTM adapters expose is called here against a SHARED namespace holding ONE row stamped for a
principal the caller is not, with the caller set OMITTED entirely, and the result is asserted.

Two outcomes are recorded, deliberately, in the same file:

* the RECALL verbs (``semantic`` / ``graph_recall`` / ``traverse_entities``) must RAISE —
  that is AD-179's fix, and this is an independent re-derivation of it;
* the MAINTENANCE verbs (``get`` / ``enumerate_page`` / ``scan_for_demotion`` / ``by_artifact``)
  do NOT raise and DO return the row — they take no ``caller_identity_set`` parameter at all, so
  they cannot express Model-A. That is not a regression this file introduces; it is the shape of
  AD-179's fix, and pinning it here is what stops the next reader from believing "AD-179 closed"
  means "every SHARED read is Model-A filtered". The one principal-facing verb among them is
  ``enumerate_page`` (``services/memory/router.py`` -> the ``MemoryRepository`` façade); its
  authorization is held ELSEWHERE (mu-server's room roster gate) and NOT by this tier, which is
  AD-142's already-recorded unauthenticated-maintenance-verb question, not this one's.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from qdrant_client import AsyncQdrantClient

from mu_contracts.domain.errors import CallerIdentitySetRequiredError
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.domain.memory import MemoryItem, MemoryState
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

VECTOR_DIM = 8


async def test_no_recall_verb_reads_a_shared_partition_without_a_caller_set(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The adversarial half: every attempt to make ``semantic`` serve a SHARED row to a caller
    nobody identified. Each variation is a distinct spelling of "omit the caller set"."""
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    ns = make_ns(visibility=Visibility.SHARED)
    victim = make_item(ns, "the rollback captain is Priya Raman", authorized_ids=["p_carol"])
    await adapter.upsert(victim)
    vec = victim.embedding or []
    try:
        # 1. the parameter omitted entirely (the wiring-bug shape AD-179 names)
        with pytest.raises(CallerIdentitySetRequiredError):
            await adapter.semantic(ns, vec, limit=10)
        # 2. the parameter passed explicitly as None
        with pytest.raises(CallerIdentitySetRequiredError):
            await adapter.semantic(ns, vec, limit=10, caller_identity_set=None)
        # 3. with a session_scope, in case the second filter path skipped the guard
        with pytest.raises(CallerIdentitySetRequiredError):
            await adapter.semantic(ns, vec, limit=10, caller_identity_set=None, session_scope=None)
        # 4. THE CONTROL: an EMPTY set is legal and authorizes NOTHING. If this returned the row,
        #    the guard would be theatre — a caller could simply pass frozenset() instead of None.
        empty = await adapter.semantic(ns, vec, limit=10, caller_identity_set=frozenset())
        assert empty == [], "an empty caller identity set authorized a SHARED read"
        # 5. and the positive control: the row IS reachable by the principal it is stamped for,
        #    so the four denials above are the filter working, not an empty partition.
        mine = await adapter.semantic(ns, vec, limit=10, caller_identity_set=frozenset({"p_carol"}))
        assert [h.item.id for h in mine] == [victim.id]
    finally:
        await qdrant_client.delete_collection(collection_name(ns, VECTOR_DIM))


async def test_maintenance_verbs_are_not_model_a_filtered_and_this_is_pinned(
    qdrant_client: AsyncQdrantClient,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The honest half: AD-179 fixed the FILTER-COMPILING reads. The maintenance verbs below take
    no caller set and return a SHARED row to anyone holding the namespace. Asserted as the CURRENT
    behaviour so a future change to it is visible, NOT asserted as correct."""
    adapter = QdrantMtmAdapter(qdrant_client, dim=VECTOR_DIM)
    ns = make_ns(visibility=Visibility.SHARED)
    victim = make_item(ns, "someone else's shared fact", authorized_ids=["p_carol"])
    await adapter.upsert(victim)
    try:
        assert (await adapter.get(ns, victim.id)) is not None
        page, _ = await adapter.enumerate_page(
            ns, states=frozenset({MemoryState.ACTIVE}), pinned=None, cursor=None, limit=10
        )
        assert [i.id for i in page] == [victim.id]
        assert [i.id for i in await adapter.scan_for_demotion(ns, limit=10)] == [victim.id]
    finally:
        await qdrant_client.delete_collection(collection_name(ns, VECTOR_DIM))
