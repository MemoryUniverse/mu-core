"""AD-333 / ADR 0106 — the store-liveness guard.

The VM's 20-minute `vm_side_reclaim.sh` cron deletes every `mu_mtm__*` Qdrant collection, and
`ours_arm.py`'s command line matches none of its process guards. On 2026-09-27 a sweep deleted a
live collection 8 minutes into an arm and the arm **completed and reported** `gold_in_context`
6/150 with `width_mean` 20.00 — numbers that are real measurements of a store that no longer held
the corpus. The same pass measured that failure's cost at ~72 pp (16/150 vs an expected ~124/150).

So the guard is not decoration: without it, the harness's headline metric can be off by more than
any effect it is used to detect, with nothing in the output saying so.

These tests run nowhere near a store — they exercise the decision rule and the partition
derivation, which is where the mistakes live.
"""

from __future__ import annotations

import pytest
from mem0_h2h.ours_arm import mtm_liveness_verdict
from mu_eval.corpus import _run_namespaces, _slug


class TestRunNamespaceDerivation:
    """`mtm_point_count` must look at the SAME partitions `local_memory_for` writes to. A second,
    re-typed derivation that drifts would read zero on a clean run and fail every arm — which is
    worse than no guard, so the shared helper is the thing under test.
    """

    def test_derivation_covers_private_and_shared_planes(self) -> None:
        names = {ns.to_prefix() for ns in _run_namespaces("h2habc123conv26")}
        assert len(names) == 3, names

    def test_org_and_workspace_come_from_the_tag(self) -> None:
        for ns in _run_namespaces("tagX"):
            assert ns.org == "orgtagX"
            assert ns.workspace == "wstagX"

    def test_tag_is_run_id_plus_slugged_sample_id(self) -> None:
        # The tag `mtm_point_count` rebuilds must equal the one `local_memory_for` built, or the
        # guard inspects collections nothing ever wrote to.
        assert f"r1{_slug('conv-26')}" == "r1conv26"


class TestCollapseRule:
    """The rule `ours_arm.py` applies: fail on a COLLAPSE (a halving) of a tier that was real to
    begin with — not on any decrease.

    Why not any decrease: `DemotionService` legitimately removes MTM points (MTM->STM tier-down is
    a `remove`) and promotion moves them to LTM, so `after < before` fires on ordinary lifecycle
    activity. A guard that cries wolf on healthy runs gets deleted, and then the real failure comes
    back. `-1` is "could not ask the store" — a different fault, never reported as contamination.
    """

    @staticmethod
    def _fires(before: int, after: int) -> bool:
        """Calls the REAL rule `ours_arm.py` applies — not a re-typed copy of it. A test that
        restates the predicate passes just as happily when the shipped one is mutated, which is the
        failure mode `mtm_liveness_verdict` was extracted to avoid."""
        return mtm_liveness_verdict(before, after) is not None

    @pytest.mark.parametrize(
        ("before", "after", "expected"),
        [
            (419, 419, False),  # healthy: untouched
            (419, 420, False),  # healthy: read-path reinforcement can add
            (419, 400, False),  # healthy: a little lifecycle demotion
            (419, 210, False),  # just inside the threshold — still not called contamination
            (419, 209, True),  # just past it
            (419, 0, True),  # THE INCIDENT: collection deleted mid-run
            (419, 12, True),  # near-total loss is still a destroyed measurement
            (0, 0, False),  # nothing was ever written (importance gate) — not contamination
            (-1, 0, False),  # store unreachable BEFORE: unknown, never claim contamination
            (419, -1, True),  # unreadable AFTER a healthy read: cause unknown (outage or
            # deletion), so `ours_arm.py` refuses with its OWN message rather than asserting the
            # sweep — but it still refuses, because an unverified tier is not a result
        ],
    )
    def test_collapse_rule(self, before: int, after: int, expected: bool) -> None:
        assert self._fires(before, after) is expected

    def test_the_incident_shape_is_caught(self) -> None:
        """419 turns ingested, collection deleted by the sweep, query phase continues."""
        assert self._fires(419, 0) is True

    def test_a_healthy_run_is_not_flagged(self) -> None:
        """The measured shape of this pass's own four clean arms: the pool never emptied."""
        assert self._fires(419, 419) is False

    def test_the_two_causes_get_different_messages(self) -> None:
        """An outage and a deletion must not be reported as the same thing — a run that is told
        "the reclaim sweep deleted your collection" when the store was merely down sends the next
        engineer to the wrong place."""
        deleted = mtm_liveness_verdict(419, 0)
        unreadable = mtm_liveness_verdict(419, -1)
        assert deleted is not None and unreadable is not None
        assert "collapsed" in deleted and "reclaim sweep" in deleted
        assert "could NOT be read" in unreadable
        assert "Cause unknown" in unreadable

    def test_no_baseline_is_never_contamination(self) -> None:
        """`before <= 0` means the corpus never reached the vector tier (the importance gate) or the
        store could not be read at the start either. Neither is evidence of a mid-run deletion."""
        assert mtm_liveness_verdict(0, 0) is None
        assert mtm_liveness_verdict(-1, 0) is None
        assert mtm_liveness_verdict(-1, -1) is None
