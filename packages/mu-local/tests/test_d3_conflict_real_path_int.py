"""D3 real-path verify (REMEDIATION Rank 4, conformance B4/B7; plan D3) — proves the injected
``ConflictAdjudicator`` actually fires on the REAL public ``add()``/``consolidate()`` path against
the REAL mu-dev SLM (``qwen2.5:0.5b``, :11435) + FalkorDB (:16380), not a curated seed fed straight
to ``DistillPipeline.distill()``. ZERO mocks (DEV-STANDARDS non-negotiable).

Reproduces, then closes, the exact defect the Rank-2 verify + this remediation's own pipeline-trace
harness (``demos/pipeline_trace``) flagged: two same-(subject,predicate) contradicting facts
landing in the SAME ``consolidate()`` window (e.g. "The Q3 planning meeting is in Room A" /
"...was moved to Room B", both extracted in ONE session's ONE sweep) never saw each other in
``DistillPipeline._reconcile``'s candidate gather — that gather is a snapshot taken for the WHOLE
window BEFORE any ``_resolve`` call writes (module docstring), so neither fact is in the store yet
when either's own reconcile runs. Both silently ADDed as permanently-active "simultaneous truths"
— 0 ``CONFLICTS_WITH``/``SUPERSEDED_BY`` edges, confirmed live against real mu-dev-falkordb before
this fix (``pipetrace_20260731T155206Z.json``: ``ada/s2`` facts_extracted=8, superseded=0). Fixed by
``DistillPipeline._live_residue`` — a fresh ``find_conflicts`` re-read immediately before the write
decision (mirrors the existing ``_current_node`` DEFECT-2 precedent for the identical/NOOP branch)
— which now sees a same-batch sibling this SAME resolve loop already wrote earlier in the batch.

Three conflict pairs, each seeded via TWO free-text ``mem.add()`` calls in the SAME session
followed by ONE ``mem.consolidate()`` call (the real same-batch shape):
  1. ada_room   — "...is in Room A" / "...was moved to Room B" (the exact pipeline-trace finding).
  2. bo_standup — "...standup is at 9am" / "...standup is now at 9:30am" (phrasing probed
     directly against the live SLM first: "...moved to 9:30am" alone extracts to an EMPTY fact
     list from ``qwen2.5:0.5b`` — a genuine extraction-quality gap, D5's separately-tracked
     territory, NOT this fix's — so the update-style phrasing the model reliably extracts is used
     instead, isolating the D3 wiring proof from that concern).
  3. ada_flight — "Ada's flight to Denver is on Tuesday" / "...moved to Thursday" (``flight``, one
     of ``DistillSettings.functional_predicates``' own D-7 canonical-predicate entries — probed as
     the steadiest third pair the 0.5b model extracts with an IDENTICAL predicate both times).

Verified via a DIRECT ``GRAPH.QUERY`` against the real FalkorDB graph (bypassing
``GraphStorePort``'s own active-only filter) — every pair must resolve to EXACTLY ONE active head
sharing that (subject, predicate) plus a ``SUPERSEDED_BY``/``CONFLICTS_WITH`` edge linking the pair,
never two permanently-active unrelated nodes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
import pytest_asyncio
from falkordb.asyncio import FalkorDB

from mu_contracts.config import Settings
from mu_contracts.contracts.views import ConsolidateView
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_local import LocalMemory
from mu_local.config import ModelProfileSettings, StorageSettings

from .test_local_conflict_adjudicator_int import _teardown
from .test_local_llm_slm_int import _SLM_CFG, _SLM_UP

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not _SLM_UP,
        reason=(
            f"mu-dev-slm unreachable at {_SLM_CFG.probe_url} (env probe) — "
            "start it: docker compose -f docker-compose.slm.yml up -d"
        ),
    ),
]


@pytest_asyncio.fixture
async def slm_mem(settings: Settings, uid: str) -> AsyncIterator[LocalMemory]:
    profile = ModelProfileSettings(
        base_url=_SLM_CFG.base_url,
        model=_SLM_CFG.model,
        max_tokens=_SLM_CFG.max_tokens,
        temperature=_SLM_CFG.temperature,
    )
    memory = LocalMemory(
        StorageSettings(llm=profile),
        workspace=f"wsd3{uid}",
        namespace=f"orgd3{uid}",
        settings=settings,
    )
    try:
        yield memory
    finally:
        await _teardown(settings, f"d3{uid}")
        await memory.aclose()


async def _live_nodes(  # type: ignore[no-any-unimported]  # falkordb ships no stubs
    falkor: FalkorDB, adapter: FalkorLtmAdapter, ns: Namespace, *, subject_contains: str
) -> list[dict[str, object]]:
    """Raw ``MATCH (m:Memory)`` read — bypasses ``GraphStorePort``'s ``state='active'``/
    ``invalid_at`` filter so a superseded-but-retained (invalidate-don't-delete) node is visible
    too, alongside its full edge set (both directions)."""
    graph_name = adapter.graph_name_for(ns)
    g = falkor.select_graph(graph_name)
    res = await g.query(
        "MATCH (m:Memory) WHERE m.namespace = $ns AND toLower(m.subject) CONTAINS toLower($sub) "
        "RETURN m.id, m.subject, m.predicate, m.object, m.state, m.invalid_at",
        params={"ns": ns.to_prefix(), "sub": subject_contains},
    )
    return [
        {
            "id": row[0],
            "subject": row[1],
            "predicate": row[2],
            "object": row[3],
            "state": row[4],
            "invalid_at": row[5],
        }
        for row in res.result_set
    ]


async def _edges_between(  # type: ignore[no-any-unimported]  # falkordb ships no stubs
    falkor: FalkorDB, adapter: FalkorLtmAdapter, ns: Namespace, ids: list[str]
) -> list[tuple[str, str, str]]:
    """Every ``SUPERSEDED_BY``/``CONFLICTS_WITH`` edge whose BOTH endpoints are in ``ids`` —
    direction-agnostic (``(a)-[r]->(b)`` either way), content-free (rel type + endpoint ids
    only)."""
    graph_name = adapter.graph_name_for(ns)
    g = falkor.select_graph(graph_name)
    res = await g.query(
        "MATCH (a:Memory)-[r:SUPERSEDED_BY|CONFLICTS_WITH]->(b:Memory) "
        "WHERE a.namespace = $ns AND a.id IN $ids AND b.id IN $ids "
        "RETURN a.id, type(r), b.id",
        params={"ns": ns.to_prefix(), "ids": ids},
    )
    return [(row[0], row[1], row[2]) for row in res.result_set]


@dataclass
class _PairResult:
    report: ConsolidateView
    nodes: list[dict[str, object]] = field(default_factory=list)
    edges: list[tuple[str, str, str]] = field(default_factory=list)


async def test_same_batch_conflict_pairs_resolve_to_one_active_head(
    slm_mem: LocalMemory, settings: Settings
) -> None:
    container = slm_mem._container
    assert container.conflict_adjudicator is not None, (
        "the SLM profile must build a REAL ConflictAdjudicator — D3 composition-root wiring"
    )

    falkor = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
    adapter = FalkorLtmAdapter(falkor)
    try:
        pairs = [
            ("ada", "s_room", "The Q3 planning meeting is in Room A.",
             "The Q3 planning meeting was moved to Room B.", "q3 planning meeting"),
            ("bo", "s_standup", "Bo's team standup is at 9am.",
             "Bo's team standup is now at 9:30am.", "bo"),
            ("ada", "s_flight", "Ada's flight to Denver is on Tuesday.",
             "Ada's flight to Denver moved to Thursday.", "ada"),
        ]

        results: dict[str, _PairResult] = {}
        for user, session, text_v1, text_v2, subject_needle in pairs:
            await slm_mem.add(text_v1, user=user, session=session)
            await slm_mem.add(text_v2, user=user, session=session)
            report = await slm_mem.consolidate(user=user, session=session)

            ns = Namespace(
                org=slm_mem._org,
                workspace=slm_mem._workspace,
                user=user,
                session=session,
                visibility=Visibility.PRIVATE,
            )
            nodes = await _live_nodes(falkor, adapter, ns, subject_contains=subject_needle)
            ids = [str(n["id"]) for n in nodes]
            edges = await _edges_between(falkor, adapter, ns, ids)

            key = f"{user}/{session}"
            results[key] = _PairResult(report=report, nodes=nodes, edges=edges)
            print(f"\n[D3-VERIFY {key}] consolidate: {report}")  # noqa: T201
            for n in nodes:
                print(f"  node: {n}")  # noqa: T201
            for e in edges:
                print(f"  edge: {e[0]} -[{e[1]}]-> {e[2]}")  # noqa: T201

        # ---- assertions: each pair collapses to exactly ONE active head + a linking edge ----
        for key, data in results.items():
            nodes = data.nodes
            same_spo_groups: dict[tuple[str | None, str | None], list[dict[str, object]]] = {}
            for n in nodes:
                same_spo_groups.setdefault(
                    (str(n["subject"]).lower(), str(n["predicate"])), []
                ).append(n)
            # the colliding (subject, predicate) group is the one with >1 member.
            colliding = [g for g in same_spo_groups.values() if len(g) > 1]
            assert colliding, f"{key}: expected a detected (subject,predicate) collision"
            group = colliding[0]
            active = [n for n in group if n["state"] == "active"]
            assert len(active) == 1, (
                f"{key}: expected exactly ONE active head in the colliding group, got "
                f"{len(active)}: {group}"
            )
            edges = data.edges
            assert edges, f"{key}: expected a SUPERSEDED_BY/CONFLICTS_WITH edge linking the pair"
    finally:
        await falkor.connection.aclose()
