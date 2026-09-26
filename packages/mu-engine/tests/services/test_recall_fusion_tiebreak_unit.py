"""``reciprocal_rank_fusion`` deterministic tie-break — AD-317/AD-318.

Repro + regression test for AD-317 (``docs/tracking/ARCHITECTURE-DELTAS.md``): a plain
``sorted(scores, key=scores.get, reverse=True)`` breaks ties on Python's stable sort, which
preserves INSERTION order into the ``scores`` dict — i.e. whichever order each channel's own
ranked list happened to present its candidates in a given call. That per-channel presentation
order is not itself guaranteed stable across separate calls (a vector store's internal tie-break
among equal/near-equal distances can vary run to run), so two calls with the SAME channel inputs
but candidates arriving in a DIFFERENT per-channel order used to fuse to a different item
COMPOSITION even though every fused score was byte-identical.

Pure unit test: no containers, no network, exercises ``reciprocal_rank_fusion`` directly.
"""

from __future__ import annotations

from mu_engine.services.recall.fusion import reciprocal_rank_fusion


def test_tied_scores_break_on_id_not_insertion_order() -> None:
    """Two calls whose channels present the SAME tied candidates in OPPOSITE orders must still
    fuse to the SAME item composition and order — the whole point of AD-317's fix. Every element
    below ties exactly (each appears in exactly one single-channel list, at rank 0, weight 1.0,
    so every fused score is identically ``1/(k+1)``): the pre-fix code broke that tie by
    insertion order into ``scores`` (i.e. channel/list order), which is exactly what differs
    between the two calls constructed here.
    """
    # Call 1: channels present tied candidates as [a, b, c, d]
    channels_1 = [["a"], ["b"], ["c"], ["d"]]
    # Call 2: the SAME four tied candidates, channels presenting them in the OPPOSITE order —
    # simulating a vector store returning equal-distance hits in a different internal order.
    channels_2 = [["d"], ["c"], ["b"], ["a"]]
    weights = [1.0, 1.0, 1.0, 1.0]

    result_1 = reciprocal_rank_fusion(channels_1, key=lambda x: x, weights=weights, k=60)
    result_2 = reciprocal_rank_fusion(channels_2, key=lambda x: x, weights=weights, k=60)

    ids_1 = [eid for eid, _score in result_1]
    ids_2 = [eid for eid, _score in result_2]

    # Every score really is tied (proves this is a tie-break test, not a scoring difference).
    assert len({round(s, 12) for _eid, s in result_1}) == 1
    assert len({round(s, 12) for _eid, s in result_2}) == 1

    # The fix: deterministic secondary key on the id itself — independent of channel/call order.
    assert ids_1 == sorted(ids_1)
    assert ids_2 == sorted(ids_2)
    assert ids_1 == ids_2


def test_repeated_calls_are_byte_identical_across_many_ties() -> None:
    """Wider repro of the measured AD-317 symptom (30/150 rows differed on real conv-26 data):
    many tied candidates across several channels, called twice with each channel's internal order
    reversed the second time (the same "vector-store tie-break varies run to run" shape) — the
    fused output must be identical both times, not merely same-scored.
    """
    ids = [f"m{i}" for i in range(20)]
    # Spread the 20 ids across 4 channels, each channel's own list already best-first: item i in
    # channel c sits at rank i, so items at the SAME rank in DIFFERENT channels tie exactly
    # (same weight, same shared k) while items at different ranks WITHIN a channel do not — only
    # the ORDER THE CHANNELS ARE VISITED IN changes between the two calls below, never any item's
    # own rank (and therefore never its score): reversing the channel LIST (not each channel's own
    # internal best-first order) reverses insertion order into `scores` at every tied rank-tier
    # without changing what any single item's fused score is.
    channels_forward = [ids[0:5], ids[5:10], ids[10:15], ids[15:20]]
    channels_reversed = list(reversed(channels_forward))
    weights = [1.0, 1.0, 1.0, 1.0]

    out_a = reciprocal_rank_fusion(channels_forward, key=lambda x: x, weights=weights, k=60)
    out_b = reciprocal_rank_fusion(channels_reversed, key=lambda x: x, weights=weights, k=60)

    assert [eid for eid, _ in out_a] == [eid for eid, _ in out_b]


def test_higher_score_still_wins_the_primary_key() -> None:
    """The secondary key (id) only ever resolves EXACT ties — a genuinely higher-scored element
    (present in two channels vs one) must still rank first regardless of id ordering."""
    # "z" appears in two channels (higher fused score); "a" in only one (lower fused score).
    channels = [["z"], ["z", "a"]]
    weights = [1.0, 1.0]

    result = reciprocal_rank_fusion(channels, key=lambda x: x, weights=weights, k=60)
    ids = [eid for eid, _score in result]

    assert ids[0] == "z"  # higher score wins even though "a" < "z" alphabetically
    assert ids == ["z", "a"]
