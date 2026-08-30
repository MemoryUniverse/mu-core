"""``PersonaAffinityShaper`` + ``PersonaShapedRankedRead`` — the CONSEQUENCE (§5.2, §5.4).

This is the file that answers "so what?". ``PersonaSettings.affinity_weight`` shipped with no
consumer and said so in its own comment; a persona that is written, versioned, decayed and erased
but never changes a byte a caller reads is a field that round-trips. Everything below is asserted
on the REAL shipped recall DTOs — ``mu_engine.services.recall.dto`` — because a shaper proved only
against a stand-in would be exactly the kind of "passes its unit tests, composed nowhere" the
subsystem is being rescued from.

Note that this TEST file names the recall DTOs and the persona package does NOT. That asymmetry
is the design (``shaping.py``'s module docstring): the shaper depends on three structural fields,
the test pins those fields to the real classes, and the §5.4 rule 1-2 import ban over the persona
package stays intact.
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime

import pytest

from mu_contracts.domain.events import DegradedModeEntered, DegradeReason
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.domain.model.persona import PersonaProfile, PersonaSlot, SlotValue
from mu_engine.services.persona.settings import PersonaSettings
from mu_engine.services.persona.shaping import (
    AFFINITY_SLOTS,
    PersonaAffinityShaper,
    PersonaShapedRankedRead,
    RankedHit,
    RankedRead,
)
from mu_engine.services.persona.store import InMemoryPersonaRepository
from mu_engine.services.recall.dto import (
    RecallChannels,
    RecallItemView,
    RecallQuery,
    RecallResult,
)
from mu_engine.services.recall.service import RecallService
from mu_engine.storage.domain.namespace import Namespace as EngineNamespace

from .conftest import T0, RecordingBus

pytestmark = pytest.mark.unit


def _engine_ns(ns: Namespace) -> EngineNamespace:
    return EngineNamespace(
        org=ns.org,
        workspace=ns.workspace,
        user=ns.user,
        session=ns.session,
        visibility=ns.visibility,
    )


def _hit(memory_id: str, content: str, score: float, *, is_floor: bool = False) -> RecallItemView:
    return RecallItemView(
        memory_id=memory_id,
        content=content,
        namespace=EngineNamespace(
            org="org1", workspace="ws1", user="u1", session="s1", visibility="private"
        ),
        content_hash=f"h_{memory_id}",
        tier="stm" if is_floor else "mtm",
        channel="stm" if is_floor else "mtm",
        fused_score=score,
        is_floor=is_floor,
    )


def _result(ns: Namespace, items: list[RecallItemView]) -> RecallResult:
    return RecallResult(
        namespace=_engine_ns(ns),
        items=items,
        channels_run=RecallChannels(),
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


async def _with_profile(
    ns: Namespace,
    *,
    slot: PersonaSlot = PersonaSlot.FOLLOWED_TOPIC,
    value: str = "kayaking",
    confidence: float = 0.9,
    settings: PersonaSettings | None = None,
    bus: RecordingBus | None = None,
) -> PersonaAffinityShaper:
    repo = InMemoryPersonaRepository()
    await repo.upsert(
        PersonaProfile(
            namespace=ns,
            slots={
                slot: SlotValue(
                    value=value, confidence=confidence, support_ids=("m1",), updated_at=T0
                )
            },
            overall_brief="brief",
            brief_etag="etag",
            version=1,
            rebuilt_at=T0,
            source_memory_count=3,
        )
    )
    return PersonaAffinityShaper(repo=repo, settings=settings or PersonaSettings(), bus=bus)


# ------------------------------------------------------- the structural contract with recall
def test_the_real_recall_dtos_satisfy_the_structural_seams():
    """``shaping.py`` deliberately names no recall type (§5.4 rules 1-2, enforced package-wide by
    ``test_persona_boundary_unit``). The price of that is that a shape change in the recall DTOs
    would be caught nowhere — so it is caught HERE, against the real classes."""
    hit = _hit("m1", "x", 1.0)
    assert isinstance(hit, RankedHit)
    assert isinstance(RecallService, type) and isinstance(
        inspect.getattr_static(RecallService, "recall"), object
    )
    for name in ("recall", "recall_degraded"):
        assert hasattr(RecallService, name)
        assert inspect.iscoroutinefunction(getattr(RecallService, name))


def test_the_decorator_covers_recallservices_entire_public_surface():
    """The composition roots ``cast`` this wrapper to ``RecallService`` because
    ``mu_engine.surface.facade.LocalContainerLike`` declares ``recall: RecallService`` as an
    invariant attribute (a file neither this lane owns nor may name persona from). A cast is only
    safe while the wrapper really can stand in — so that is asserted, not assumed. The day
    ``RecallService`` grows a public method, this fails and the cast stops being safe LOUDLY."""
    public = {n for n in vars(RecallService) if not n.startswith("_")}
    wrapper = {n for n in vars(PersonaShapedRankedRead) if not n.startswith("_")}
    assert public <= wrapper, public - wrapper


def test_recallservice_satisfies_the_ranked_read_protocol():
    assert isinstance(RecallService, type)
    assert set(RankedRead.__protocol_attrs__) <= set(dir(RecallService))  # type: ignore[attr-defined]


# ----------------------------------------------------------------- THE CONSEQUENCE (§5.4 4c)
async def test_a_persona_changes_the_order_of_the_same_candidate_set(ns: Namespace):
    """§5.4 rule 4(c) verbatim: *"two callers with opposite personas over the same PRIVATE
    partition get the SAME candidate set, differing only in order"*.

    This is the whole point of the subsystem being alive. Set the shaper aside and the order is
    the ranker's; wire it and a persona-matching hit rises — with the SAME memories present."""
    hits = [_hit("m1", "notes about compilers", 0.90), _hit("m2", "a kayaking trip", 0.85)]
    shaper = await _with_profile(ns, value="kayaking")

    shaped = await shaper.shape(ns, hits)

    assert [h.memory_id for h in hits] == ["m1", "m2"]  # the ranker's own order
    assert [h.memory_id for h in shaped] == ["m2", "m1"]  # persona's
    assert {h.memory_id for h in shaped} == {h.memory_id for h in hits}  # SAME set


async def test_two_opposite_personas_reorder_the_same_set_two_ways(ns: Namespace):
    hits = [_hit("m1", "notes about compilers", 0.90), _hit("m2", "a kayaking trip", 0.85)]
    kayaker = await _with_profile(ns, value="kayaking")
    compiler_nerd = await _with_profile(ns, value="compilers")

    assert [h.memory_id for h in await kayaker.shape(ns, hits)] == ["m2", "m1"]
    assert [h.memory_id for h in await compiler_nerd.shape(ns, hits)] == ["m1", "m2"]


async def test_the_boost_cannot_exceed_affinity_weight(ns: Namespace):
    """Spec line 194: *"persona re-orders but cannot dominate semantic evidence"*. With the
    default 0.15 a matching hit gains at most 15 %, so a hit that is more than 15 % behind stays
    behind. Raise ``affinity_weight`` past that gap and it overtakes — which is what makes this
    an assertion about the BOUND rather than about one example."""
    hits = [_hit("m1", "notes about compilers", 1.00), _hit("m2", "a kayaking trip", 0.80)]

    bounded = await _with_profile(ns, value="kayaking")
    assert [h.memory_id for h in await bounded.shape(ns, hits)] == ["m1", "m2"]

    loud = await _with_profile(ns, value="kayaking", settings=PersonaSettings(affinity_weight=0.5))
    assert [h.memory_id for h in await loud.shape(ns, hits)] == ["m2", "m1"]


async def test_a_multi_word_slot_value_cannot_out_shout_the_bound(ns: Namespace):
    """``_match`` returns the STRONGEST matching term, never the sum. Summing would let a slot
    value made of three ordinary words multiply the bound by three and turn ``affinity_weight``
    from a ceiling into a suggestion — which is precisely what §5.2 bounds the magnitude to
    prevent. (Written because a sum-instead-of-max mutant survived the first draft of this file.)
    """
    shaper = await _with_profile(ns, value="kayaking sea rowing")
    hits = [_hit("m1", "compilers", 1.00), _hit("m2", "kayaking on the sea while rowing", 0.80)]

    # 0.80 * (1 + 0.15*0.9) = 0.908 < 1.00 -> stays behind. Summing gives 0.80 * 1.405 = 1.12.
    assert [h.memory_id for h in await shaper.shape(ns, hits)] == ["m1", "m2"]


async def test_shaping_never_adds_or_drops_a_candidate(ns: Namespace):
    hits = [
        _hit(f"m{i}", f"body {i} kayaking" if i % 2 else f"body {i}", 1.0 - i / 100)
        for i in range(10)
    ]
    shaper = await _with_profile(ns, value="kayaking")

    shaped = await shaper.shape(ns, hits)

    assert sorted(h.memory_id for h in shaped) == sorted(h.memory_id for h in hits)
    assert len(shaped) == len(hits)


async def test_shaping_does_not_rewrite_fused_score(ns: Namespace):
    """``fused_score`` is the RANKER's testimony about semantic evidence. Persona re-orders; it
    does not get to speak in the ranker's voice."""
    hits = [_hit("m1", "a kayaking trip", 0.5), _hit("m2", "other", 0.9)]
    shaper = await _with_profile(ns, value="kayaking")

    shaped = await shaper.shape(ns, hits)

    assert {h.memory_id: h.fused_score for h in shaped} == {"m1": 0.5, "m2": 0.9}


async def test_a_tie_keeps_the_rankers_own_relative_order(ns: Namespace):
    hits = [_hit("m1", "alpha", 0.5), _hit("m2", "beta", 0.5), _hit("m3", "gamma", 0.5)]
    shaper = await _with_profile(ns, value="nothing-matches-here")
    assert [h.memory_id for h in await shaper.shape(ns, hits)] == ["m1", "m2", "m3"]


async def test_the_protected_stm_floor_still_leads_after_shaping(ns: Namespace):
    """The ranked read this shaper wraps is TWO score scales, not one sorted list.

    A protected floor member carries an STM query-relevance score and a fused hit carries an RRF
    rank score, so a shaper that sorted the whole list would compare incomparable numbers and pull
    a fused hit past a protected floor member. This case's incoming order happens to be
    floor-first — the shape ``_merge_floor`` always produced before D3 — so what it pins is that
    the shaper does not REORDER ACROSS the two scales. Which slots are floor slots is the ranker's
    decision now, and
    :func:`test_shaping_keeps_an_interleaved_floor_member_at_the_rank_fusion_gave_it` pins that.

    **The numbers are chosen to DISCRIMINATE, and the first version of this test was not** — it
    used floor scores of 0.30/0.20 against RRF scores of 0.016, where a single global sort happens
    to produce the same answer, and a mutation that restored the global sort passed it. The floor
    protects a *just-said* fact, and `_score_stm` scores it by RELEVANCE to the query, so a
    protected floor row that has nothing to do with the query scores near ZERO — below an RRF
    score, which is bounded below by `1/(k + rank)`. That is the case where the two
    implementations disagree, so that is the case this test uses.
    """
    hits = [
        _hit("f1", "alpha", 0.011, is_floor=True),  # just said, irrelevant to the query
        _hit("f2", "beta", 0.004, is_floor=True),
        _hit("t1", "gamma kayaking", 0.016),  # RRF rank 1, and the persona matches it
        _hit("t2", "delta", 0.015),
    ]
    shaper = await _with_profile(ns, value="kayaking")

    shaped = await shaper.shape(ns, hits)

    assert [h.memory_id for h in shaped][:2] == ["f1", "f2"], "a fused hit overtook the floor"
    assert [h.memory_id for h in shaped] == ["f1", "f2", "t1", "t2"]


async def test_shaping_reorders_inside_the_floor_block(ns: Namespace):
    """The floor block leads as a BLOCK; persona may still re-order within it, exactly as the
    ranker itself does (``_score_stm`` re-sorts the protected floor by relevance)."""
    hits = [
        _hit("f1", "alpha", 0.30, is_floor=True),
        _hit("f2", "beta kayaking", 0.29, is_floor=True),
        _hit("t1", "gamma", 0.016),
    ]
    shaper = await _with_profile(ns, value="kayaking")

    assert [h.memory_id for h in await shaper.shape(ns, hits)] == ["f2", "f1", "t1"]


async def test_shaping_keeps_an_interleaved_floor_member_at_the_rank_fusion_gave_it(
    ns: Namespace,
):
    """D3 (STATE-AND-DEFECTS-0829.md): the shaper must not re-prepend the floor.

    ``ranker.py:_merge_floor`` and ``recall/service.py:_protect_floor`` were both changed so a
    protected member keeps whatever rank fusion actually earned it — the fix for the measured
    ``recall@1 = 0.0003`` (LoCoMo, 1,531 queries), where the top ``floor_protect_limit`` slots of
    every result were the most-recently-written memories, chosen without reference to the query.
    ``is_floor`` entries therefore arrive INTERLEAVED.

    ``_reorder`` used to rebuild the list as ``[*floor_block, *fused_block]``. Against an
    interleaved input that CONCATENATION silently restores the query-blind prefix one decorator
    downstream of the fix — and persona is wrapped around ``recall`` on every deployment that
    wires a model (``mu_local/composition.py``, ``mu_engine_server/composition.py``), so it would
    have undone D3 in the product while the ranker's own unit tests stayed green.

    Two things are asserted, and the second is the one that fails against the pre-fix code:
    persona still re-orders WITHIN the fused block (``t2`` overtakes ``t1``), and the floor member
    ``f1`` stays in the slot fusion gave it.
    """
    hits = [
        _hit("t1", "gamma", 0.016),  # fusion ranked this first — no persona signal
        _hit("f1", "alpha", 0.011, is_floor=True),  # protected, but fusion ranked it SECOND
        _hit("t2", "delta kayaking", 0.015),  # persona matches; must overtake t1, not f1
    ]
    shaper = await _with_profile(ns, value="kayaking")

    shaped = await shaper.shape(ns, hits)

    assert [h.memory_id for h in shaped] == [
        "t2",
        "f1",
        "t1",
    ], "the floor member was re-prepended (or the fused block was not re-ordered)"
    assert shaped[1].is_floor, "the floor member left the slot rank fusion gave it"


async def test_a_persona_that_matches_nothing_perturbs_a_two_block_result_not_at_all(
    ns: Namespace,
):
    """The regression this pins was live and was found BY CONSEQUENCE, not by reading.

    With a single global sort, a persona that matched no hit at all still re-ordered any result
    whose incoming order was not score-descending — which every real recall's is, because the
    floor block leads on a different scale. A relevance prior that perturbs reads it does not
    match is not a prior. Identity, not equality: the caller's own object comes back.
    """
    hits = [
        _hit("f1", "alpha", 0.011, is_floor=True),
        _hit("f2", "beta", 0.004, is_floor=True),
        _hit("t1", "gamma", 0.016),
        _hit("t2", "delta", 0.015),
    ]
    shaper = await _with_profile(ns, value="nothing-matches-here")

    # NOT score-descending across the block boundary — which is what a REAL ranked read looks
    # like, and what the global sort silently rewrote even with no persona signal at all.
    assert [h.fused_score for h in hits] != sorted((h.fused_score for h in hits), reverse=True)
    assert await shaper.shape(ns, hits) is hits


# ---------------------------------------------------------------------- when NOT to shape
async def test_no_profile_means_the_identical_object_back(ns: Namespace):
    shaper = PersonaAffinityShaper(repo=InMemoryPersonaRepository())
    hits = [_hit("m1", "a", 1.0), _hit("m2", "b", 0.5)]
    assert await shaper.shape(ns, hits) is hits


async def test_a_shared_namespace_is_never_shaped(shared_ns: Namespace, ns: Namespace):
    """§5.4 rule 3 / §0 line 53. Also proves the repo is never even asked: ``assert_private``
    would raise inside ``InMemoryPersonaRepository.get`` if it were."""
    shaper = await _with_profile(ns)
    hits = [_hit("m1", "kayaking", 1.0), _hit("m2", "b", 0.9)]
    assert await shaper.shape(shared_ns, hits) is hits


async def test_disabled_persona_never_shapes(ns: Namespace):
    shaper = await _with_profile(ns, settings=PersonaSettings(enabled=False))
    hits = [_hit("m1", "b", 1.0), _hit("m2", "a kayaking trip", 0.9)]
    assert await shaper.shape(ns, hits) is hits


async def test_a_low_confidence_slot_does_not_move_a_ranking(ns: Namespace):
    """An affinity prior built from a slot the classifier itself was unsure of would re-order a
    user's recall on a guess.

    **The scores are chosen to DISCRIMINATE.** The first version used 1.0 against 0.9, where the
    boost a confidence of 0.2 would buy (x1.03) cannot close the gap anyway — so the test passed
    with the `affinity_min_confidence` gate DELETED, which a mutation proved. The gap here is
    smaller than the boost, so the gate is the only thing holding the order.
    """
    shaper = await _with_profile(ns, value="kayaking", confidence=0.2)
    hits = [_hit("m1", "b", 0.90), _hit("m2", "a kayaking trip", 0.89)]

    assert 0.89 * (1 + PersonaSettings().affinity_weight * 0.2) > 0.90  # the boost WOULD flip it
    assert await shaper.shape(ns, hits) is hits


async def test_a_voice_slot_is_not_an_affinity_slot(ns: Namespace):
    """§5.2 names three slots. ``response_style`` describes HOW to speak (§5.1's job); letting it
    move a ranking would shape relevance with voice."""
    assert PersonaSlot.RESPONSE_STYLE not in AFFINITY_SLOTS
    shaper = await _with_profile(ns, slot=PersonaSlot.RESPONSE_STYLE, value="terse")
    hits = [_hit("m1", "b", 1.0), _hit("m2", "be terse", 0.9)]
    assert await shaper.shape(ns, hits) is hits


async def test_a_tiny_term_cannot_match_inside_a_word(ns: Namespace):
    shaper = await _with_profile(ns, value="go")
    hits = [_hit("m1", "b", 1.0), _hit("m2", "the algorithm is good", 0.9)]
    assert await shaper.shape(ns, hits) is hits


async def test_matching_is_whole_word(ns: Namespace):
    shaper = await _with_profile(ns, value="kayak")
    hits = [_hit("m1", "b", 1.0), _hit("m2", "kayaking is unrelated to kayak-adjacent", 0.9)]
    # "kayaking" must NOT match "kayak"; the hyphenated "kayak-adjacent" tokenises to "kayak".
    assert [h.memory_id for h in await shaper.shape(ns, hits)] == ["m2", "m1"]


async def test_a_failed_persona_read_degrades_and_returns_the_unshaped_result(ns: Namespace):
    """Persona is optional; recall is not. A persona store that is down must cost the user their
    persona, never their recall."""

    class Broken(InMemoryPersonaRepository):
        async def get(self, ns: Namespace) -> PersonaProfile | None:
            raise RuntimeError("store down")

    bus = RecordingBus()
    shaper = PersonaAffinityShaper(repo=Broken(), bus=bus)
    hits = [_hit("m1", "a", 1.0), _hit("m2", "b", 0.5)]

    assert await shaper.shape(ns, hits) is hits
    degrades = [e for e in bus.events if isinstance(e, DegradedModeEntered)]
    assert [(d.component, d.mode, d.reason) for d in degrades] == [
        ("persona", "persona_shaping_unavailable", DegradeReason.LLM_UNAVAILABLE_HEURISTIC)
    ]


# ------------------------------------------------------------------------- the decorator
class SpyRead:
    def __init__(self, result: RecallResult) -> None:
        self.result = result
        self.calls = 0

    async def recall(self, scope: object, q: RecallQuery) -> RecallResult:
        self.calls += 1
        return self.result

    async def recall_degraded(
        self, scope: object, q: RecallQuery, *, without: set[str], reason: DegradeReason
    ) -> RecallResult:
        self.calls += 1
        return self.result


def _query(ns: Namespace) -> RecallQuery:
    return RecallQuery(namespace=_engine_ns(ns), text="anything")


async def test_the_decorator_shapes_the_real_recall_result(ns: Namespace, scope: object):
    inner = SpyRead(_result(ns, [_hit("m1", "compilers", 0.9), _hit("m2", "kayaking", 0.85)]))
    shaped = PersonaShapedRankedRead(inner=inner, shaper=await _with_profile(ns, value="kayaking"))

    out = await shaped.recall(scope, _query(ns))

    assert [i.memory_id for i in out.items] == ["m2", "m1"]
    assert out.namespace == inner.result.namespace
    assert out.generated_at == inner.result.generated_at
    assert out.channels_run == inner.result.channels_run


async def test_the_decorator_returns_the_inner_object_when_nothing_moved(
    ns: Namespace, scope: object
):
    inner = SpyRead(_result(ns, [_hit("m1", "compilers", 0.9), _hit("m2", "rowing", 0.85)]))
    shaped = PersonaShapedRankedRead(inner=inner, shaper=await _with_profile(ns, value="kayaking"))
    assert await shaped.recall(scope, _query(ns)) is inner.result


async def test_the_degraded_read_is_shaped_by_the_same_rule(ns: Namespace, scope: object):
    """``RecallService.recall_degraded`` calls its OWN ``self.recall``, so a decorator forwarding
    only ``recall`` would leave every degraded read unshaped — persona would silently stop
    mattering exactly when a channel is down."""
    inner = SpyRead(_result(ns, [_hit("m1", "compilers", 0.9), _hit("m2", "kayaking", 0.85)]))
    shaped = PersonaShapedRankedRead(inner=inner, shaper=await _with_profile(ns, value="kayaking"))

    out = await shaped.recall_degraded(
        scope, _query(ns), without={"ltm"}, reason=DegradeReason.LTM_UNAVAILABLE
    )

    assert [i.memory_id for i in out.items] == ["m2", "m1"]
