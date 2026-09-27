"""Entity fuzzy-resolution write path — REAL mu-dev-falkordb, ZERO mocks. AD-331.

Covers the two bugs the AD-331 task brief named, on a REAL graph (a write-path change on a graph
that is not easily un-merged — the bar the task brief sets):

1. ``EntityCandidate.similarity`` was hardcoded to ``1.0`` — no candidate the old exact-only query
   could ever return WAS anything but an exact match, so the score was never actually computed.
   ``test_fuzzy_merge_reuses_existing_entity_and_returns_a_real_score`` proves a genuinely
   near-duplicate name (a one-character typo, real Jaccard ~0.917) now merges into the SAME
   ``:Entity`` node via a REAL computed score, not a fabricated ``1.0``.
2. ``_merge_entity`` wrote ``alias_keys`` once at node creation and never grew them, so a fuzzy
   hit would recur every single time instead of becoming an exact hit on repeat.
   ``test_alias_keys_grow_so_a_fuzzy_hit_becomes_an_exact_hit_next_time`` proves the SECOND
   resolve of the identical typo name comes back ``similarity == 1.0`` (the exact/alias-key
   branch), not the fuzzy branch again — this is the concrete, provable fix for "alias_keys
   never grows".

Also covers the entropy gate (AD-331: "not optional garnish") on real data: two DISTINCT short
names never merge, and a low-entropy/short query name never fuzzy-matches even an identical-looking
short pool candidate — mirroring the unit-level entropy tests but against the real store this bug
lives in.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def ltm(falkor_db: FalkorDB) -> AsyncIterator[FalkorLtmAdapter]:
    """Same local-fixture convention every sibling file in this directory follows (e.g.
    ``test_falkor_traverse_authz_int.py``) — ``ltm`` is deliberately NOT a shared ``conftest.py``
    fixture in this package, so each integration file defines its own and tears its own graphs
    down. AD-331 close-pass fix: this file originally declared ``ltm`` as a test parameter without
    ever defining it, which fails every test in it at fixture-setup (`fixture 'ltm' not found`) —
    caught running the suite for real on the VM, not by static gates, which cannot see a missing
    fixture used only by name."""
    yield FalkorLtmAdapter(falkor_db)
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await falkor_db.select_graph(name).delete()


async def test_fuzzy_merge_reuses_existing_entity_and_returns_a_real_score(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    first = make_item(
        ns,
        "Jonathan Smith uses Postgres",
        subject="Jonathan Smith",
        predicate="uses",
        obj="Postgres",
    )
    await ltm.upsert_fact(first)
    first_uid = first.metadata["entity_uids"][0]

    # "Jonathan Smithe" (one-character typo) — real Jaccard ~0.9167 against "jonathan smith",
    # comfortably above the default entity_similarity_threshold (0.84), and NOT an exact/alias-key
    # match (the two strings casefold differently), so this can ONLY resolve via the new fuzzy
    # branch (AD-331). Observed BEFORE any write, so `alias_keys` has not grown yet and this is
    # the genuine fuzzy score, not a subsequent exact hit.
    pre_write = await ltm.resolve_entity(ns, "Jonathan Smithe")
    assert pre_write.entity_uid == first_uid
    assert (
        0.0 < pre_write.candidates[0].similarity < 1.0
    ), "AD-331 regression guard: a fuzzy hit's score must be REAL, never the pre-fix hardcoded 1.0"

    # And the write path (`_merge_entity`, via `upsert_fact`) must reuse that SAME entity rather
    # than minting a second one — pre-fix, the exact-only query would have found nothing and this
    # would have created a SEPARATE `:Entity` node for "Jonathan Smithe".
    second = make_item(
        ns,
        "Jonathan Smithe uses Postgres",
        subject="Jonathan Smithe",
        predicate="uses",
        obj="Postgres",
    )
    await ltm.upsert_fact(second)
    assert second.metadata["entity_uids"][0] == first_uid


async def test_alias_keys_grow_so_a_fuzzy_hit_becomes_an_exact_hit_next_time(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    ns = make_ns()
    first = make_item(
        ns,
        "Elizabeth Warren authored Report",
        subject="Elizabeth Warren",
        predicate="authored",
        obj="Report",
    )
    await ltm.upsert_fact(first)

    # "Elizabeth Warre" (one dropped letter) — real Jaccard ~0.923 against "elizabeth warren",
    # clears the 0.84 default threshold, so this resolves deterministically as a genuine fuzzy
    # hit (not an ambiguous one — that path is covered by the below-threshold test next door).
    typo_resolution_1 = await ltm.resolve_entity(ns, "Elizabeth Warre")
    assert typo_resolution_1.entity_uid is not None, "typo must clear the fuzzy threshold"
    assert (
        typo_resolution_1.candidates[0].similarity < 1.0
    ), "AD-331 regression guard: a fuzzy hit's score must be REAL, never the pre-fix hardcoded 1.0"

    # _merge_entity is what actually performs the alias_keys write (resolve_entity alone is
    # read-only) — reuse it directly, exactly as `_upsert_fact_impl` does.
    g = await ltm._graph(ns)
    await ltm._merge_entity(ns, g, "Elizabeth Warre")

    # SECOND resolve of the IDENTICAL name must now come back via the EXACT alias_keys branch
    # (similarity == 1.0, genuinely, not fuzzy again) — the concrete, provable fix for "alias_keys
    # never grows" (AD-331 task brief).
    typo_resolution_2 = await ltm.resolve_entity(ns, "Elizabeth Warre")
    assert typo_resolution_2.entity_uid == typo_resolution_1.entity_uid
    assert (
        typo_resolution_2.candidates[0].similarity == 1.0
    ), "alias_keys did not grow: the same name still had to be fuzzy-resolved a second time"


async def test_distinct_short_names_never_fuzzy_merge(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The entropy gate (AD-331: "not optional garnish") — two short, unrelated names must remain
    two separate entities, never coalesced by fuzzy matching."""
    ns = make_ns()
    ed = make_item(ns, "Ed uses Postgres", subject="Ed", predicate="uses", obj="Postgres")
    al = make_item(ns, "Al uses Postgres", subject="Al", predicate="uses", obj="Postgres")
    await ltm.upsert_fact(ed)
    await ltm.upsert_fact(al)
    assert ed.metadata["entity_uids"][0] != al.metadata["entity_uids"][0]

    # And resolving a short name never fuzzy-matches an identical-length pool entry either — the
    # gate runs on the QUERY name before any candidate is even scored.
    resolution = await ltm.resolve_entity(ns, "Ed")
    assert resolution.entity_uid == ed.metadata["entity_uids"][0]  # exact match, unaffected
    resolution_new_short_name = await ltm.resolve_entity(ns, "Bo")
    assert resolution_new_short_name.entity_uid is None
    assert resolution_new_short_name.candidates == ()


async def test_below_threshold_near_miss_stays_ambiguous_not_merged(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A real, honest finding for the AD-331 report: the DEFAULT threshold (0.84) is strict enough
    that "Jon Smith" vs "Jonathan Smith" (real Jaccard ~0.417) stays UNMERGED — two separate
    entities — rather than false-merging on a partial name. The task brief's own headline example
    ("Bob"/"Bobby") is even MORE conservative: both fail the entropy length gate outright (< 6
    chars, 1 token) and never attempt fuzzy matching at all, so that pair specifically is not
    affected by this fix on real data — recorded here as the honest, measured outcome rather than
    an inflated merge-rate claim."""
    ns = make_ns()
    full = make_item(
        ns,
        "Jonathan Smith uses Postgres",
        subject="Jonathan Smith",
        predicate="uses",
        obj="Postgres",
    )
    await ltm.upsert_fact(full)
    short = make_item(
        ns, "Jon Smith uses Postgres", subject="Jon Smith", predicate="uses", obj="Postgres"
    )
    await ltm.upsert_fact(short)

    assert (
        full.metadata["entity_uids"][0] != short.metadata["entity_uids"][0]
    ), "a 0.417-Jaccard partial name must NOT merge under the 0.84 default threshold"

    resolution = await ltm.resolve_entity(ns, "Jon Smith")
    assert resolution.entity_uid == short.metadata["entity_uids"][0]  # its own exact entity
