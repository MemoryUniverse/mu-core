"""AD-258 (follow-through) — the content-aware traversal seed. REAL FalkorDB, ZERO mocks.

``FAULT-HUNT-0924.md`` / ADR 0060 named the root cause and left it unfixed: ``traverse_entities``
seeds its frontier by casefolding the QUERY TEXT and exact-matching it against ``canonical_name``
— a query that never spells an entity's name verbatim (a paraphrase, a pronoun, "what does she do
for a living?") seeds nothing, no matter how relevant the entity's own facts are. This is the
regression proof for the fix: ``seed_entity_uids`` lets the frontier be seeded STRUCTURALLY, by
``entity_uid`` (a stable graph identity, never fuzzy-matched), from a source outside the query
text entirely — in production, the ranker's own top MTM dense-vector hits.

Every test in this file constructs the SAME shape: upsert a fact (which resolves + returns its
own subject/object ``entity_uid``s via ``item.metadata['entity_uids']``, D-5), then traverse with
a query that shares NO token with the fact's subject, predicate or object — proving the fact is
findable ONLY through ``seed_entity_uids``, never through the pre-existing token match.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Callable

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.integration

# Deliberately shares NO token (even after casefolding) with "Ada", "manages", "Bo", "Denver" or
# "team" — a real paraphrase/follow-up shape, not a synonym-of-the-predicate trick.
_PARAPHRASE_QUERY = "what does she do for a living these days?"


@pytest_asyncio.fixture
async def ltm(falkor_db: FalkorDB) -> AsyncIterator[FalkorLtmAdapter]:
    yield FalkorLtmAdapter(falkor_db)
    for g in await falkor_db.list_graphs():
        name = g.decode() if isinstance(g, bytes) else g
        if name.startswith("mu_g__"):
            with contextlib.suppress(Exception):  # best-effort teardown only
                await falkor_db.select_graph(name).delete()


async def test_entity_uid_seed_finds_a_fact_the_query_text_never_names(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """The core claim: a paraphrase that seeds NOTHING via the token match still finds the fact
    once its subject's resolved ``entity_uid`` is passed as ``seed_entity_uids`` — exactly the
    shape the ranker produces from the MTM channel's own top hits (``_resolve_seed_entity_uids``,
    ``ranker.py``)."""
    ns = make_ns(visibility=Visibility.PRIVATE)
    fact = make_item(
        ns, "Ada manages the Denver team", subject="Ada", predicate="manages", obj="Denver team"
    )
    await ltm.upsert_fact(fact)
    subj_uid, obj_uid = fact.metadata["entity_uids"]

    # Precondition: the paraphrase alone (no seed_entity_uids) finds nothing — this is the
    # defect FAULT-HUNT-0924/ADR 0060 named, reproduced here as the control.
    token_only = await ltm.traverse_entities(ns, query=_PARAPHRASE_QUERY, max_hops=1, limit=10)
    assert token_only == [], "precondition: the paraphrase must NOT seed via the token match"

    seeded = await ltm.traverse_entities(
        ns, query=_PARAPHRASE_QUERY, max_hops=1, limit=10, seed_entity_uids=[subj_uid]
    )

    assert [h.item.id for h in seeded] == [fact.id], (
        "AD-258: an entity_uid seed must find a fact the query text never names — got "
        f"{[h.item.id for h in seeded]!r}"
    )


async def test_entity_uid_seed_unions_with_the_token_match_never_replaces_it(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """D-4's own "ADDS, never REPLACES" precedent, extended: a query that DOES name an entity in
    plain text keeps finding that entity's facts via the pre-existing token match, with
    ``seed_entity_uids`` supplying an UNRELATED entity's facts alongside it — neither mechanism
    starves the other."""
    ns = make_ns(visibility=Visibility.PRIVATE)
    named = make_item(ns, "Bo owns the Q3 report", subject="Bo", predicate="owns", obj="Q3 report")
    await ltm.upsert_fact(named)
    unnamed = make_item(
        ns, "Ada manages the Denver team", subject="Ada", predicate="manages", obj="Denver team"
    )
    await ltm.upsert_fact(unnamed)
    ada_uid, _ = unnamed.metadata["entity_uids"]

    hits = await ltm.traverse_entities(
        ns, query="what does Bo own?", max_hops=1, limit=10, seed_entity_uids=[ada_uid]
    )

    ids = {h.item.id for h in hits}
    assert ids == {
        named.id,
        unnamed.id,
    }, f"expected BOTH the token-matched fact and the uid-seeded fact, got ids={ids!r}"


async def test_entity_uid_seed_walks_a_second_hop_from_the_seeded_entity(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """A uid-seeded entity's OWN neighbours are walked exactly like a token-seeded one's — the
    seed only widens HOP 1's frontier match (falkor_ltm.py's own AD-258 comment); hop 2+ must
    still generalize from whichever entities hop 1 discovered, uid-seeded or not."""
    ns = make_ns(visibility=Visibility.PRIVATE)
    hop1 = make_item(ns, "Ada manages Bo", subject="Ada", predicate="manages", obj="Bo")
    await ltm.upsert_fact(hop1)
    hop2 = make_item(ns, "Bo owns the Q3 report", subject="Bo", predicate="owns", obj="Q3 report")
    await ltm.upsert_fact(hop2)
    ada_uid, _ = hop1.metadata["entity_uids"]

    one_hop = await ltm.traverse_entities(
        ns, query=_PARAPHRASE_QUERY, max_hops=1, limit=10, seed_entity_uids=[ada_uid]
    )
    assert {h.item.id for h in one_hop} == {hop1.id}, "sanity: hop=1 must not already reach hop2"

    two_hops = await ltm.traverse_entities(
        ns, query=_PARAPHRASE_QUERY, max_hops=2, limit=10, seed_entity_uids=[ada_uid]
    )
    assert {h.item.id for h in two_hops} == {hop1.id, hop2.id}, (
        f"expected the 2nd-hop fact reachable via Bo (discovered at hop 1), got "
        f"{[h.item.id for h in two_hops]!r}"
    )


async def test_no_seed_entity_uids_reproduces_pre_fix_behavior_exactly(
    ltm: FalkorLtmAdapter,
    make_ns: Callable[..., Namespace],
    make_item: Callable[..., MemoryItem],
) -> None:
    """``seed_entity_uids=None`` (the shipped default at the ranker, ``ltm_entity_seed_pool=0``)
    must be byte-identical to calling ``traverse_entities`` without the parameter at all —
    DEV-STANDARDS rule 3 A/B parity."""
    ns = make_ns(visibility=Visibility.PRIVATE)
    fact = make_item(ns, "Ada manages Bo", subject="Ada", predicate="manages", obj="Bo")
    await ltm.upsert_fact(fact)

    omitted = await ltm.traverse_entities(ns, query="who is Bo's manager?", max_hops=1, limit=10)
    explicit_none = await ltm.traverse_entities(
        ns, query="who is Bo's manager?", max_hops=1, limit=10, seed_entity_uids=None
    )
    explicit_empty = await ltm.traverse_entities(
        ns, query="who is Bo's manager?", max_hops=1, limit=10, seed_entity_uids=[]
    )

    assert [h.item.id for h in omitted] == [fact.id]
    assert [h.item.id for h in explicit_none] == [h.item.id for h in omitted]
    assert [h.item.id for h in explicit_empty] == [h.item.id for h in omitted]
