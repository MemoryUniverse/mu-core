"""``LiveSessionContext`` DTO shapes — the §2.3 zone split as a STRUCTURAL invariant.

`live-session-context-design.md` §1/§2.3, CANONICAL-CONTRACTS.md §7.22.

The first test in this file is the only one here that is about a **privacy breach** rather than a
quality defect: this object is the single place in the system where shared and private content sit
side by side in one structure, and every participant of a room reads the shared zone.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.contracts.live_context import (
    SECTION_ORDER,
    ContextSlab,
    LiveSessionContext,
    PrivateSlice,
    Section,
    SharedZone,
    content_hash_of,
)
from mu_contracts.domain.model.memory import Namespace, Visibility

pytestmark = pytest.mark.unit

_NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _ns(*, user: str = "alice", session: str = "s1") -> Namespace:
    return Namespace(
        org="acme", workspace="main", user=user, session=session, visibility=Visibility.PRIVATE
    )


def _slab(text: str, *, visibility: Visibility, section: Section = Section.RECENT) -> ContextSlab:
    return ContextSlab(
        slab_id=f"m-{abs(hash(text)) % 10_000}",
        content_hash=content_hash_of(text),
        text=text,
        section=section,
        visibility=visibility,
    )


# ============================================================ 1. THE PRIVACY INVARIANT (§2.3)
def test_a_private_slab_can_never_enter_the_shared_zone() -> None:
    """`live-session-context-design.md:117`: "A PRIVATE item can never appear here."

    Every participant of a room reads ``SharedZone``. A private fact landing in it is a disclosure
    to every other participant — not a rendering defect that shows up as a wrong-looking block.
    Enforced at CONSTRUCTION, so the breach is unrepresentable rather than reviewed-for."""
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    with pytest.raises(ValidationError, match="shared-visibility only"):
        SharedZone(recent_shared=(private,))


def test_the_shared_zone_guard_survives_assignment_not_only_construction() -> None:
    """``SharedZone`` is MUTABLE per §1 (`:71` declares a bare ``BaseModel``) — so a validator
    that ran only inside ``__init__`` would be walked straight past by ``zone.recent_shared =
    (private_slab,)``, which is exactly how a live zone gets updated. ``validate_assignment`` is
    what makes the guard hold for the object's whole life, not just its first instant."""
    zone = SharedZone(running_summary="the team agreed to ship Friday")
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    with pytest.raises(ValidationError, match="shared-visibility only"):
        zone.recent_shared = (private,)
    assert zone.recent_shared == ()


def test_a_shared_slab_can_never_enter_a_private_slice() -> None:
    """The mirror guard (§2.3 `:131`). Not a disclosure by itself — it is the bookkeeping error
    that makes the other direction possible, because a zone that accepts either kind of slab has
    stopped being evidence of anything."""
    shared = _slab("standup moved to 10am", visibility=Visibility.SHARED)
    with pytest.raises(ValidationError, match="PRIVATE slabs only"):
        PrivateSlice(recalled=(shared,))


def test_one_principals_slice_is_never_reachable_from_anothers_read() -> None:
    """§1 `:107`: "each participant's host receives ``SharedZone ⊕ private[self]``, never another
    member's slice." ``slice_for`` is the ONE sanctioned read, and an unknown principal gets an
    EMPTY slice rather than someone else's or a raised error a caller might paper over."""
    alice = PrivateSlice(recalled=(_slab("alice fact", visibility=Visibility.PRIVATE),))
    bob = PrivateSlice(recalled=(_slab("bob fact", visibility=Visibility.PRIVATE),))
    state = LiveSessionContext(
        namespace=_ns(), session_id="s1", private={"alice": alice, "bob": bob}, updated_at=_NOW
    )
    assert state.slice_for("alice").recalled[0].text == "alice fact"
    assert state.slice_for("bob").recalled[0].text == "bob fact"
    assert state.slice_for("mallory").recalled == ()


# =========================================================== 2. SLAB SHAPE + CONTENT-FREEDOM
def test_a_slab_with_neither_text_nor_handle_is_refused() -> None:
    """§6: ``text=None`` means "pointer slab — hydrate by id". A slab with no text AND no handle
    is not a pointer, it is a hole, and a hole renders as nothing — the silent drop CANONICAL
    §7.10-G5 forbids. Refused at construction so it cannot reach the assembler."""
    with pytest.raises(ValidationError, match="must carry an artifact_ref"):
        ContextSlab(
            slab_id="m1",
            content_hash="h",
            section=Section.REFERENCES,
            visibility=Visibility.PRIVATE,
        )


def test_the_loggable_projection_of_a_slab_carries_no_content() -> None:
    """CLAUDE.md rule 3. ``ContextSlab.text`` IS memory content — that is the point of the object.
    What must never happen is that content reaching a log/trace/bus/metering, so the projection a
    caller is meant to log excludes both the text AND its hash (a hash of a short fact is a
    plausible lookup key for it)."""
    slab = _slab("the db password is hunter2", visibility=Visibility.PRIVATE)
    routing = slab.routing
    flat = repr(routing)
    assert "hunter2" not in flat
    assert slab.content_hash not in flat
    assert routing["section"] == "recent" and routing["visibility"] == "private"


def test_the_render_order_is_read_off_the_enum_so_the_two_cannot_drift() -> None:
    """§4 `:161-173` / CANONICAL §7.22's "ordered-XML render FORMAT invariant". A hand-written
    second list of sections is a list that goes stale the first time a section is added."""
    assert SECTION_ORDER == (
        Section.PERSONA,
        Section.SESSION_STATE,
        Section.RECALLED_MEMORY,
        Section.RECENT,
        Section.REFERENCES,
        Section.REASONING,
    )


# ================================================= 3. THE DIGEST (§5.3 store + §5.5 eviction)
def test_the_injected_digest_evicts_oldest_first_under_its_bound() -> None:
    """G-LSC2 (`:261`): "the digest grows with every distinct injected fact; a multi-day session
    could accumulate thousands of hashes. Needs a bound (LRU by injected-at...)". §1 types the
    field ``frozenset[str]``, which cannot express eviction order at all — so it is stored
    ordered and read as a set."""
    slice_ = PrivateSlice().with_injected(("h1", "h2", "h3"), bound=2)
    assert slice_.injected_digest == ("h2", "h3")
    assert slice_.digest_set == frozenset({"h2", "h3"})


def test_re_emitting_a_hash_refreshes_it_rather_than_duplicating_it() -> None:
    """A fact the assembler keeps choosing is still live in the host's window; evicting it before
    a fact injected once and never mentioned again would be exactly backwards."""
    slice_ = PrivateSlice().with_injected(("h1", "h2"), bound=2).with_injected(("h1",), bound=2)
    assert slice_.injected_digest == ("h2", "h1")


def test_forgetting_a_memory_clears_its_slabs_and_its_digest_entry() -> None:
    """§5.5 row 2 (`:201`): "invalidate the slab(s) deriving from the demoted/GC'd ``memory_id``;
    **clear its ``injected_digest`` entry so a later re-promotion can be re-injected**".

    The digest half is the one that is easy to omit and expensive to: a demoted fact whose hash
    stayed behind would be permanently un-injectable for the rest of the session even after it is
    promoted back — a fact silently missing from every future block."""
    doomed = _slab("the on-call is Ada", visibility=Visibility.PRIVATE)
    kept = _slab("deploys go to staging-eu", visibility=Visibility.PRIVATE)
    slice_ = PrivateSlice(recency_floor=(doomed, kept)).with_injected(
        (doomed.content_hash, kept.content_hash), bound=8
    )
    state = LiveSessionContext(
        namespace=_ns(), session_id="s1", private={"alice": slice_}, updated_at=_NOW
    ).forget_memory(doomed.slab_id, now=_NOW)

    after = state.slice_for("alice")
    assert [s.text for s in after.recency_floor] == ["deploys go to staging-eu"]
    assert doomed.content_hash not in after.digest_set
    assert kept.content_hash in after.digest_set


def test_forgetting_a_memory_also_filters_the_shared_zone() -> None:
    """A GC'd shared turn must stop being injected in the same tick it leaves the tier (§5.5) —
    the shared zone is not exempt from the lifecycle just because it is not the private one."""
    doomed = _slab("the room decided on FalkorDB", visibility=Visibility.SHARED)
    state = LiveSessionContext(
        namespace=_ns(),
        session_id="s1",
        shared=SharedZone(recent_shared=(doomed,)),
        updated_at=_NOW,
    ).forget_memory(doomed.slab_id, now=_NOW)
    assert state.shared.recent_shared == ()


def test_the_dedup_key_is_normalisation_stable() -> None:
    """§5.3's key must collapse the same TEXT to one entry — the host's window holds text, not
    engine rows. Matches ``mu_client.inject.distill._normalise`` so the within-render and
    cross-turn dedups agree instead of disagreeing."""
    assert content_hash_of("Ada  lives in Paris") == content_hash_of("ada lives in paris")
    assert content_hash_of("Ada lives in Paris") != content_hash_of("Ada lives in Berlin")


# ==================== 1b. THE GUARD THE VALIDATORS ALONE DO NOT GIVE (reviewer blocker) ========
def test_the_shared_zone_guard_survives_model_copy_update() -> None:
    """The one that made the "unrepresentable rather than reviewed-for" claim false.

    pydantic runs NO validators on ``model_copy(update=...)`` — and ``model_copy(update=...)`` is
    this module's own dominant mutation idiom (``with_injected``, ``forget``, ``with_slice``,
    ``forget_memory`` are all built from it). A guard the file's own idiom walks past is not an
    invariant, so :class:`_ZoneGuard` re-asserts it on the copy."""
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    with pytest.raises(ValueError, match="shared-visibility only"):
        SharedZone().model_copy(update={"recent_shared": (private,)})


def test_the_shared_zone_guard_survives_model_construct() -> None:
    """``model_construct`` is pydantic's documented validation-free constructor. A composer using
    it for speed on a hot fan-out path would otherwise build a poisoned zone with no error at all
    — the failure mode is silence, which is why this one is closed rather than documented."""
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    with pytest.raises(ValueError, match="shared-visibility only"):
        SharedZone.model_construct(recent_shared=(private,))


def test_the_private_slice_guard_survives_model_copy_update() -> None:
    """The mirror direction, through the same hole — and the one that bites first in practice,
    because ``with_injected`` and ``forget`` are themselves ``model_copy(update=...)`` calls."""
    shared = _slab("standup moved to 10am", visibility=Visibility.SHARED)
    with pytest.raises(ValueError, match="PRIVATE slabs only"):
        PrivateSlice().model_copy(update={"recalled": (shared,)})


def test_a_poisoned_zone_cannot_be_attached_to_the_state_that_renders_it() -> None:
    """``validate_assignment`` on ``LiveSessionContext`` re-checks the field's TYPE only — that
    ``shared`` is a ``SharedZone`` — never the zone's own inner guard. So a zone poisoned by any
    route at all (here the last-resort raw ``__dict__`` write, which no pydantic hook can see) must
    still be refused by the container every render reads through. This is the backstop that makes
    the invariant hold for the object rather than only for the constructor."""
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    poisoned = SharedZone()
    poisoned.__dict__["recent_shared"] = (private,)  # bypasses every pydantic seam

    state = LiveSessionContext(namespace=_ns(), session_id="s1", updated_at=_NOW)
    with pytest.raises(ValidationError, match="shared-visibility only"):
        state.shared = poisoned
    with pytest.raises(ValidationError, match="shared-visibility only"):
        LiveSessionContext(namespace=_ns(), session_id="s1", updated_at=_NOW, shared=poisoned)
    with pytest.raises(ValueError, match="shared-visibility only"):
        state.model_copy(update={"shared": poisoned})


def test_open_threads_are_labels_not_prose() -> None:
    """§2.1 `:116`: ``open_threads`` is "Content-free-safe (labels/ids only)". A ``str`` carries no
    visibility to check, so the enforceable half of that sentence is the SHAPE: one short line. It
    is the field a leak would use if it could not use a slab, and the guard is what makes "a
    summarizer dumped a room transcript in here" fail loudly instead of rendering to everyone."""
    SharedZone(open_threads=("agent dispatch running", "fact corrected"))  # labels: fine
    with pytest.raises(ValidationError, match="labels/ids only"):
        SharedZone(open_threads=("alice's salary is 200k\nand her address is …",))
    with pytest.raises(ValidationError, match="labels/ids only"):
        SharedZone(open_threads=("x" * 500,))


def test_the_running_summary_has_one_guarded_write_path() -> None:
    """The honest limit of §2.3's split, made into a seam rather than left as a comment: a ``str``
    cannot be asked what it was derived from, so :meth:`SharedZone.fold_summary` asks the CALLER
    for the slabs the summary was built over and checks THOSE. A summarizer over a room log has
    them in hand at exactly this call; one that reaches past it is doing something no type can
    see."""
    shared = _slab("standup moved to 10am", visibility=Visibility.SHARED)
    private = _slab("alice's salary is 200k", visibility=Visibility.PRIVATE)
    zone = SharedZone()

    rolled = zone.fold_summary("the team moved standup", derived_from=(shared,), version=1)
    assert rolled.running_summary == "the team moved standup"
    assert rolled.summary_version == 1

    with pytest.raises(ValueError, match="shared-visibility only"):
        zone.fold_summary("…", derived_from=(private,), version=1)
    with pytest.raises(ValueError, match="must not go backwards"):
        rolled.fold_summary("…", derived_from=(shared,), version=0)


def test_forgetting_a_shared_memory_clears_it_from_every_principals_digest() -> None:
    """§5.5 row 2 for the SHARED half — the direction ``PrivateSlice.forget`` cannot see.

    A shared slab's hash enters EVERY participant's private ``injected_digest`` when it is injected
    (the digest records what that host's window holds, whatever zone the text came from). Filtering
    the zone without clearing those entries leaves the fact un-injectable for the rest of the
    session even after a re-promotion or a re-grant — row 2's stated failure, reached by the one
    route the slice cannot close for itself."""
    shared = _slab("standup moved to 10am", visibility=Visibility.SHARED)
    state = LiveSessionContext(
        namespace=_ns(),
        session_id="s1",
        updated_at=_NOW,
        shared=SharedZone(recent_shared=(shared,)),
        private={
            "alice": PrivateSlice(injected_digest=(shared.content_hash,)),
            "bob": PrivateSlice(injected_digest=(shared.content_hash, "other")),
        },
    )

    after = state.forget_memory(shared.slab_id, now=_NOW)

    assert after.shared.recent_shared == ()
    assert shared.content_hash not in after.slice_for("alice").digest_set
    assert shared.content_hash not in after.slice_for("bob").digest_set
    assert after.slice_for("bob").injected_digest == ("other",)  # unrelated entries survive
