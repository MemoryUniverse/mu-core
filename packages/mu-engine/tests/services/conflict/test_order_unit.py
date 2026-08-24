"""``total_order_key`` (CANONICAL §7.17 item 4a) — pure logic, zero I/O, mocks-free by
construction (there is nothing to mock). Every test cites the clause it pins.

The mutation-testing report for this suite (each clause's guard was deliberately broken in
``order.py`` and the corresponding test below re-run to confirm it fails) is recorded in the
team-lead handoff report, not duplicated here.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from mu_contracts.domain.model.conflict import ResolutionOrigin
from mu_contracts.domain.model.device_sync import PrivateDelta, SyncOp
from mu_engine.services.conflict.order import total_order_key

pytestmark = pytest.mark.unit

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _delta(
    *,
    device: str = "dev_a",
    lamport: int = 1,
    valid_at: datetime = _T0,
    valid_at_inferred: bool = False,
    pinned: bool = False,
    resolution_origin: ResolutionOrigin | None = None,
    content_hash: str = "hash_a",
    memory_id: str = "mem_1",
    seq: int = 1,
) -> PrivateDelta:
    return PrivateDelta(
        seq=seq,
        origin_device_id=device,
        op=SyncOp.UPSERT,
        memory_id=memory_id,
        content_hash=content_hash,
        tier="ltm",
        valid_at=valid_at,
        valid_at_inferred=valid_at_inferred,
        lamport=lamport,
        occurred_at=valid_at,
        provenance_id="prov_1",
        pinned=pinned,
        resolution_origin=resolution_origin,
    )


# =================================================================================================
# item 4a(b) — the leading pair is a PRECEDENCE, not a lexicographic tuple.
# =================================================================================================


def test_pinned_dominates_a_later_greater_loser() -> None:
    """A pinned candidate wins even against a non-pinned candidate that is strictly GREATER on
    every term below `pinned` (same device — so lamport actually decides if reached — later
    valid_at, lexicographically greater content_hash/device_id). Proves DOMINANCE, not merely
    "compared first": a lexicographic misreading that let later terms override could still let
    the non-pinned candidate win here; this test would catch that."""
    pinned = _delta(device="dev_a", lamport=1, valid_at=_T0, pinned=True, content_hash="a")
    later_unpinned = _delta(
        device="dev_a",
        lamport=99,
        valid_at=_T0 + timedelta(days=365),
        pinned=False,
        content_hash="z",
    )
    assert total_order_key(pinned, later_unpinned) > 0
    assert total_order_key(later_unpinned, pinned) < 0


def test_manual_beats_everything_except_pinned() -> None:
    """A `resolution_origin=MANUAL` candidate beats a non-manual candidate that is strictly
    greater on every term below it (item 4a(b) "sticky... dominated only by pinned") — but a
    PINNED (non-manual) candidate still beats it, because pinned dominates manual too."""
    manual = _delta(
        device="dev_a",
        lamport=1,
        valid_at=_T0,
        resolution_origin=ResolutionOrigin.MANUAL,
        content_hash="a",
    )
    later_automatic = _delta(
        device="dev_a",
        lamport=99,
        valid_at=_T0 + timedelta(days=365),
        resolution_origin=ResolutionOrigin.AUTO,
        content_hash="z",
    )
    assert total_order_key(manual, later_automatic) > 0
    assert total_order_key(later_automatic, manual) < 0

    pinned_non_manual = _delta(device="dev_a", lamport=1, valid_at=_T0, pinned=True)
    assert total_order_key(pinned_non_manual, manual) > 0
    assert total_order_key(manual, pinned_non_manual) < 0


def test_manual_does_not_dominate_pinned_via_lexicographic_shortcut() -> None:
    """A non-pinned MANUAL candidate must not beat a PINNED non-manual candidate even when the
    manual candidate wins every term below `resolution_origin` — proves `pinned` is compared
    strictly BEFORE `resolution_origin`, matching the item 4 term order
    `(pinned, resolution_origin==manual)`, not the reverse."""
    pinned_non_manual = _delta(
        device="dev_a", lamport=1, valid_at=_T0, pinned=True, content_hash="a"
    )
    manual_non_pinned = _delta(
        device="dev_a",
        lamport=99,
        valid_at=_T0 + timedelta(days=365),
        resolution_origin=ResolutionOrigin.MANUAL,
        content_hash="z",
    )
    assert total_order_key(pinned_non_manual, manual_non_pinned) > 0


# =================================================================================================
# item 4a(c) — an INCOMPARABLE (concurrent, cross-device) pair is a TIE that falls through.
# =================================================================================================


def test_incomparable_cross_device_pair_falls_through_to_valid_at() -> None:
    """Two candidates from DIFFERENT devices: a naive "just compare the raw lamport ints"
    reading would make the huge-lamport candidate win outright. The correct §7.17 item 4a(c)
    reading treats a cross-device pair as INCOMPARABLE (a tie on `lamport_vc_order`), so the
    comparison falls through to `valid_at` instead — and the candidate with the LOWER lamport
    but LATER valid_at wins. This is the test that would fail if lamport were (wrongly) compared
    directly across devices. `content_hash`/`device_id` are set AGAINST the `valid_at` winner so
    this cannot pass via an accidental tiebreak alignment either."""
    huge_lamport_older = _delta(device="dev_z", lamport=1000, valid_at=_T0, content_hash="hash_zzz")
    tiny_lamport_newer = _delta(
        device="dev_a", lamport=1, valid_at=_T0 + timedelta(days=1), content_hash="hash_low"
    )
    assert total_order_key(tiny_lamport_newer, huge_lamport_older) > 0
    assert total_order_key(huge_lamport_older, tiny_lamport_newer) < 0


def test_same_device_lamport_totally_orders_regardless_of_valid_at() -> None:
    """Two candidates from the SAME device ARE causally comparable via their own monotonic
    counter (item 4a(c)'s "same device" branch, contrasted with the cross-device test above):
    the higher-lamport candidate wins even when its `valid_at` is EARLIER — `lamport_vc_order`
    decides before `valid_at` is ever reached, for a same-device pair."""
    higher_lamport_earlier_date = _delta(device="dev_a", lamport=5, valid_at=_T0)
    lower_lamport_later_date = _delta(device="dev_a", lamport=2, valid_at=_T0 + timedelta(days=1))
    assert total_order_key(higher_lamport_earlier_date, lower_lamport_later_date) > 0
    assert total_order_key(lower_lamport_later_date, higher_lamport_earlier_date) < 0


# =================================================================================================
# item 4a(a) — valid_at_asserted_only gate + direction (later/greater wins).
# =================================================================================================


def test_valid_at_asserted_beats_inferred_ahead_of_lamport_and_valid_at() -> None:
    """PINNED 2 / item 4a(a): `valid_at_asserted_only` sits BEFORE `lamport_vc_order` and
    `valid_at` in the term order — an asserted candidate wins even against a cross-device
    (incomparable-lamport) candidate whose inferred `valid_at` is later."""
    asserted = _delta(device="dev_x", lamport=1, valid_at=_T0, valid_at_inferred=False)
    inferred_later = _delta(
        device="dev_y",
        lamport=1,
        valid_at=_T0 + timedelta(days=365),
        valid_at_inferred=True,
    )
    assert total_order_key(asserted, inferred_later) > 0


def test_both_asserted_falls_through_to_valid_at_directly() -> None:
    """When both sides assert `valid_at` (both `valid_at_inferred=False`), the
    `valid_at_asserted_only` term ties and, for a cross-device (incomparable) pair, so does
    `lamport_vc_order` — so `valid_at` itself decides, later wins (PINNED 2: "two asserted
    valid_at's compare as before"). `content_hash`/`device_id` are set AGAINST the `valid_at`
    winner (as in the clause 4a(d) test below) so this cannot pass via an accidental tiebreak."""
    earlier = _delta(
        device="dev_z", lamport=1, valid_at=_T0, valid_at_inferred=False, content_hash="hash_zzz"
    )
    later = _delta(
        device="dev_a",
        lamport=1,
        valid_at=_T0 + timedelta(days=1),
        valid_at_inferred=False,
        content_hash="hash_low",
    )
    assert total_order_key(later, earlier) > 0


def test_clause_4a_d_both_inferred_cross_device_pair_still_falls_through_to_valid_at() -> None:
    """Item 4a(d) [added 2026-08-24] — THE locking test for the ambiguity this module's own
    review surfaced and CANONICAL was amended to close. Two CROSS-DEVICE candidates, BOTH
    `valid_at_inferred=True` (neither asserted), tie on all three booleans (`pinned`, `manual`,
    `valid_at_asserted_only` — False==False on the last one) and tie on `lamport_vc_order`
    (concurrent, per clause (c), since the devices differ) — so the chain reaches `valid_at`
    with NEITHER side having asserted it, and `valid_at` still decides, later wins.

    This is NOT a bug and does NOT reintroduce X8's "a phone with a clock 3 hours fast always
    wins": clause (d) grounds this in PINNED 3 (the hub CLAMPS an inferred `valid_at` to
    `min(inferred, hub_receive_time + max_skew_s)` on append) — the clamp is the mitigation that
    makes consulting a bounded, unreliable-but-meaningful signal safe, in preference to falling
    through to the purely arbitrary `content_hash`/`device_id` tiebreakers instead.

    If a future change makes this test fail by skipping `valid_at` on a both-inferred tie
    (reading (B), rejected by clause (d)), that is the exact "fix" item 4a(d)'s own text warns
    someone will eventually attempt — do not silently take it; re-read clause (d) first.

    `content_hash`/`device_id` are deliberately set AGAINST the `valid_at` winner (the later
    candidate gets the lexicographically SMALLER hash/device — a losing tiebreak) so this test
    can only pass because `valid_at` itself decided; if the comparator instead fell through past
    `valid_at` to the tiebreakers (reading (B)), the EARLIER candidate would win and this
    assertion would fail — proving the test is not vacuously true via an accidental tiebreak."""
    later_wins_only_via_valid_at = _delta(
        device="dev_a",  # smaller device_id — LOSES the device_id tiebreak
        lamport=1,
        valid_at=_T0 + timedelta(days=1),  # later — must win via valid_at alone
        valid_at_inferred=True,
        content_hash="hash_low",  # smaller content_hash — LOSES that tiebreak too
    )
    earlier_wins_every_tiebreak = _delta(
        device="dev_z",  # larger device_id — wins the device_id tiebreak
        lamport=1,
        valid_at=_T0,  # earlier — must lose despite winning every tiebreak below valid_at
        valid_at_inferred=True,
        content_hash="hash_zzz",  # larger content_hash — wins that tiebreak too
    )
    assert total_order_key(later_wins_only_via_valid_at, earlier_wins_every_tiebreak) > 0
    assert total_order_key(earlier_wins_every_tiebreak, later_wins_only_via_valid_at) < 0


# =================================================================================================
# totality — device_id is unique per device, so the chain always terminates in one winner.
# =================================================================================================


def test_totality_device_id_breaks_an_otherwise_total_tie() -> None:
    """Two candidates identical on every term above `device_id` (including `content_hash`, so
    the `content_hash` tiebreak ties too) still produce exactly ONE winner via `device_id` —
    proving the chain is TOTAL, per item 4a(c)'s closing claim (device_id unique per device,
    CANONICAL §7.11)."""
    a = _delta(device="dev_aaa", lamport=1, valid_at=_T0, content_hash="same")
    b = _delta(device="dev_bbb", lamport=1, valid_at=_T0, content_hash="same")
    result_ab = total_order_key(a, b)
    result_ba = total_order_key(b, a)
    assert result_ab != 0
    assert result_ba != 0
    assert result_ab == -result_ba


def test_totality_content_hash_breaks_a_tie_before_device_id_is_reached() -> None:
    """Two candidates from DIFFERENT devices (lamport ties as concurrent), identical `valid_at`,
    but a different `content_hash` — the `content_hash` term (ascending, no semantic meaning)
    decides deterministically before `device_id` is even consulted."""
    a = _delta(device="dev_aaa", lamport=1, valid_at=_T0, content_hash="hash_low")
    b = _delta(device="dev_bbb", lamport=1, valid_at=_T0, content_hash="hash_high")
    assert total_order_key(a, b) != 0
    assert total_order_key(a, b) == -total_order_key(b, a)


# =================================================================================================
# determinism / antisymmetry — the same pair, both argument orders, repeatedly.
# =================================================================================================


def test_deterministic_and_antisymmetric_across_repeated_calls_and_argument_order() -> None:
    a = _delta(device="dev_a", lamport=3, valid_at=_T0, pinned=True)
    b = _delta(
        device="dev_b",
        lamport=1,
        valid_at=_T0 + timedelta(days=1),
        resolution_origin=ResolutionOrigin.MANUAL,
    )
    first = total_order_key(a, b)
    for _ in range(5):
        assert total_order_key(a, b) == first  # same inputs -> same output, every time
        assert total_order_key(b, a) == -first  # antisymmetric in both argument orders


def test_identical_candidate_compared_to_itself_is_a_tie() -> None:
    a = _delta()
    assert total_order_key(a, a) == 0
