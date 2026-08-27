"""§7 — a manual decision is sticky, first-class, and never silently flipped. Offline, pure.

Covers conflict-resolution-async-design.md §7 (lines 257-264) and CANONICAL §7.17 item 4a term 2.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.conflict import ResolutionOrigin
from mu_contracts.domain.model.device_sync import PrivateDelta, SyncOp
from mu_engine.services.conflict.convergence import (
    converge_pair,
    manual_reinstate_delta,
    manual_supersede_delta,
)

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 6, 1, tzinfo=UTC)


def _auto_supersede(
    *,
    seq: int,
    device: str,
    winner_id: str,
    loser_id: str,
    lamport: int = 1,
    valid_at: datetime = _T0,
    content_hash: str = "h",
) -> PrivateDelta:
    return PrivateDelta(
        seq=seq,
        origin_device_id=device,
        op=SyncOp.SUPERSEDE,
        memory_id=loser_id,
        content_hash=content_hash,
        tier="ltm",
        valid_at=valid_at,
        lamport=lamport,
        occurred_at=valid_at,
        provenance_id="prov-1",
        winner_id=winner_id,
        loser_id=loser_id,
    )


def _manual(
    *,
    seq: int,
    device: str,
    winner_id: str,
    loser_id: str,
    lamport: int = 1,
    valid_at: datetime = _T0,
) -> PrivateDelta:
    return manual_supersede_delta(
        seq=seq,
        origin_device_id=device,
        winner_id=winner_id,
        loser_id=loser_id,
        content_hash="h",
        tier="ltm",
        valid_at=valid_at,
        occurred_at=valid_at,
        provenance_id="prov-1",
        lamport=lamport,
        resolved_by="principal-u1",
    )


# ══════════════════════════════════════ 1. THE WRITER EXISTS ════════════════════════════════
def test_a_manual_resolution_emits_a_delta_that_actually_says_manual() -> None:
    """Term 2 of the §7.17 order was permanently dead: ``PrivateDelta.resolution_origin`` was
    declared and nothing ever set it to MANUAL, so stickiness was unreachable."""
    delta = _manual(seq=1, device="laptop", winner_id="a", loser_id="b")

    assert delta.op is SyncOp.SUPERSEDE
    assert delta.resolution_origin is ResolutionOrigin.MANUAL
    assert delta.resolved_by == "principal-u1"
    assert delta.winner_id == "a"
    assert delta.loser_id == "b"
    assert delta.memory_id == "b", "the delta names the item whose state actually changes"


def test_a_reinstate_delta_names_the_item_coming_back_to_active() -> None:
    delta = manual_reinstate_delta(
        seq=2,
        origin_device_id="phone",
        winner_id="a",
        loser_id="b",
        content_hash="h",
        tier="ltm",
        valid_at=_T0,
        occurred_at=_T0,
        provenance_id="prov-1",
        lamport=5,
        resolved_by="principal-u1",
    )
    assert delta.op is SyncOp.REINSTATE
    assert delta.memory_id == "a"
    assert delta.resolution_origin is ResolutionOrigin.MANUAL


# ═══════════════════════════════════════ 2. STICKINESS ══════════════════════════════════════
def test_a_human_decision_beats_a_later_automatic_re_derivation() -> None:
    """§7 line 259, the whole point: *"otherwise a phone that auto-picks the recency winner
    would silently override the human's laptop decision"*."""
    human = _manual(seq=1, device="laptop", winner_id="a", loser_id="b", lamport=1)
    phone_auto = _auto_supersede(
        seq=2,
        device="phone",
        winner_id="b",
        loser_id="a",
        lamport=99,
        valid_at=_T0 + timedelta(days=10),
    )

    outcome = converge_pair([human, phone_auto])

    assert outcome.winner_id == "a", "the human's winner stands"
    assert outcome.origin is ResolutionOrigin.MANUAL


def test_stickiness_does_not_depend_on_delta_order() -> None:
    human = _manual(seq=1, device="laptop", winner_id="a", loser_id="b")
    phone_auto = _auto_supersede(seq=2, device="phone", winner_id="b", loser_id="a", lamport=99)
    assert (
        converge_pair([human, phone_auto]).winner_id
        == converge_pair([phone_auto, human]).winner_id
        == "a"
    )


def test_two_automatic_deltas_converge_by_the_plain_total_order() -> None:
    """Without a human in the set, nothing is sticky — the ordinary §7.5 re-derivation."""
    older = _auto_supersede(seq=1, device="laptop", winner_id="a", loser_id="b", lamport=1)
    newer = _auto_supersede(seq=2, device="laptop", winner_id="b", loser_id="a", lamport=9)
    outcome = converge_pair([older, newer])
    assert outcome.winner_id == "b"
    assert outcome.origin is ResolutionOrigin.AUTO
    assert outcome.reopen is False


# ══════════════════════════════════ 3. REOPEN, NEVER A SILENT FLIP ══════════════════════════
def test_a_contradicting_automatic_delta_reopens_rather_than_flips() -> None:
    """§7 line 262: *"a later purely-automatic contradicting delta does not flip it (it instead
    REOPENEDs the conflict for human review, never a silent auto-override of a human)"*."""
    human = _manual(seq=1, device="laptop", winner_id="a", loser_id="b", lamport=1)
    contradicting = _auto_supersede(
        seq=2,
        device="phone",
        winner_id="b",
        loser_id="a",
        lamport=99,
        valid_at=_T0 + timedelta(days=10),
    )

    outcome = converge_pair([human, contradicting])

    assert outcome.winner_id == "a", "not flipped"
    assert outcome.reopen is True, "but the human is asked again"


def test_an_agreeing_automatic_delta_does_not_reopen() -> None:
    """A REOPEN storm on every corroborating delta would make the signal useless."""
    human = _manual(seq=1, device="laptop", winner_id="a", loser_id="b")
    agreeing = _auto_supersede(seq=2, device="phone", winner_id="a", loser_id="b", lamport=99)
    assert converge_pair([human, agreeing]).reopen is False


def test_a_weaker_contradicting_delta_does_not_reopen() -> None:
    """The counterfactual asks whether the automatic delta WOULD have won with the manual term
    neutralised — one that loses on its own merits changes nothing and must not nag."""
    human = _manual(seq=2, device="laptop", winner_id="a", loser_id="b", lamport=9, valid_at=_T0)
    weaker = _auto_supersede(
        seq=1,
        device="laptop",
        winner_id="b",
        loser_id="a",
        lamport=1,
        valid_at=_T0 - timedelta(days=1),
    )
    assert converge_pair([human, weaker]).reopen is False


def test_an_automatic_winner_never_reports_reopen() -> None:
    """``reopen`` means "an automatic delta contradicts a HUMAN". With no human decision in the
    set there is nothing to protect and nothing to ask about."""
    a = _auto_supersede(seq=1, device="laptop", winner_id="a", loser_id="b", lamport=1)
    b = _auto_supersede(seq=2, device="phone", winner_id="b", loser_id="a", lamport=99)
    assert converge_pair([a, b]).reopen is False


# ════════════════════════════════════════ 4. REINSTATE ══════════════════════════════════════
def test_a_locally_superseded_winner_is_reported_for_reinstate() -> None:
    """§7 line 263 — a device that auto-superseded before the human's delta arrived must get
    ``a`` back to ACTIVE, or the fleet's ``state='active'`` sets are not byte-identical."""
    human = _manual(seq=2, device="laptop", winner_id="a", loser_id="b")
    local_auto = _auto_supersede(seq=1, device="phone", winner_id="b", loser_id="a")

    outcome = converge_pair([local_auto, human], locally_superseded=frozenset({"a"}))

    assert outcome.winner_id == "a"
    assert outcome.reinstate_ids == ("a",)


def test_the_converged_loser_is_never_reinstated() -> None:
    """It being superseded IS the decision."""
    human = _manual(seq=1, device="laptop", winner_id="a", loser_id="b")
    outcome = converge_pair([human], locally_superseded=frozenset({"b"}))
    assert outcome.reinstate_ids == ()


# ═════════════════════════════════════════ 5. GUARDS ════════════════════════════════════════
def test_converging_an_empty_delta_set_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="empty delta set"):
        converge_pair([])


def test_deltas_about_different_pairs_are_refused() -> None:
    """Silently converging a mixed bag would produce a confident winner for a conflict nobody
    asserted."""
    one = _auto_supersede(seq=1, device="laptop", winner_id="a", loser_id="b")
    other = _auto_supersede(seq=2, device="laptop", winner_id="c", loser_id="d")
    with pytest.raises(ValueError, match="exactly one"):
        converge_pair([one, other])
