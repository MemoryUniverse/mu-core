"""F2(a) — DURABILITY: T-crash-replay re-run against the CONTAINER (build-plan §7 F2, design §6
"Test obligation... T-crash-replay").

Write via the public SDK -> `docker kill` the real `mu-engine-server` container (SIGKILL, never a
graceful stop) -> `make up` again against the SAME Valkey volume -> assert (1) the write survived
the kill, (2) the content-hash-scoped MTM dedup holds (no duplicate MTM point) across a subsequent
re-`add()` of the identical content, and (3) that re-`add()`'s own HTTP receipt is HONEST about what
it actually did this call (reinforcement of already-promoted content, not a fresh MTM write).

**Re-corrected wording (this revision — supersedes the previous "Corrected wording" section).** A
prior revision of this file asserted that a re-`add()` of identical content after the restart MUST
NOT mint the SAME `memory_id` as the pre-kill write, reasoning that `SurfaceFacade.add()`'s FRESH
`uuid.uuid4().hex` `session_offset` per call (`facade.py::_fresh_offset`) makes
`WriteStmStage`'s own activity-id ledger (`activity_id = sha256(host|session|session_offset|kind)`)
incapable of ever collapsing two independent calls onto one id. That reasoning about the ACTIVITY-
ID ledger was correct and still holds — but it conflated "not ledger-gated on activity_id" with
"cannot return the same id at all", missing the SEPARATE, store-level mechanism that also governs
`add()`'s return contract: D4's write-time content-hash dedup index
(`storage/adapters/{redis,valkey,memory}_stm.py`, `content_hash -> memory_id`). Pre-this-fix, D4
already kept exactly ONE physical STM row for identical content in one namespace, but
`WriteStmStage` never learned which id that row lived under — it kept minting+returning a FRESH id
the store had already discarded (DATA-QUALITY-REASSESSMENT §3 "add() idempotency" / the D4 report).
Now `StmTierRepository.put` reports the RESIDENT id back to `WriteStmStage`
(`mu_engine/pipelines/concrete/ingest.py`), which re-stamps its own item onto it — so a re-`add()`
of identical content in the SAME namespace DOES now return the SAME `memory_id` as the earlier
call, whether or not a kill/restart happened in between (return-idempotency, gated on the SAME
`MU_INGEST__STM_DEDUP` toggle D4 introduced).

What this file now asserts: (a) the pre-kill write survives the kill+restart untouched (GET still
200 with the original content), (b) a re-`add()` of the identical content after the restart
RETURNS THE SAME `memory_id` as the pre-kill write (the return-idempotency fix above — this is now
the file's headline assertion, not "two distinct ids by design"), and (c)
`DeterministicPromoteStage`'s content-hash-scoped ledger (`{namespace}:{content_hash}`,
`mu_engine/pipelines/concrete/ingest.py`) still dedupes the MTM WRITE across the two calls: exactly
one MTM point for this content_hash, at that SAME shared `memory_id`, never a duplicate.

**The receipt-honesty fix this revision also pins (unchanged by the return-idempotency fix above,
still verified here).** Pre-that-fix, the SECOND call's HTTP receipt (`MemoryWriteResult`) falsely
reported `promoted=true, tiers_written=["stm", "mtm"]` for a memory_id that was NEVER actually
written to MTM this call (a data-quality/honesty defect, separate from — and unaffected by — the
return-idempotency fix above: the two receipts now share ONE `memory_id`, but only the FIRST call's
receipt honestly claims the MTM write that id actually received).
`IngestService._promoted_this_call` (`mu_engine/services/ingest.py`) now derives the receipt from
each stage's OWN `StageOutcome.status` (`OK` vs a content-hash ledger-hit `SKIPPED`) rather than
merely checking whether a `MemoryPromoted` event is anywhere in the aggregated emitted-events list
(a SKIPPED ledger-hit republishes that SAME event shape from an EARLIER, DIFFERENT call —
"SKIPPED never means no events", `pipelines/base.py`). The second call's receipt is now honest:
`promoted=false`, `tiers_written=["stm"]` — reinforcement, not a new write.

**Same-invocation crash-resume is a SEPARATE, already-proven guarantee, not this file's job.**
A crash occurring MID-FLIGHT inside one `add()` invocation, then resuming that SAME invocation
(same `session_offset`/activity) with the SAME `memory_id`, IS real and already proven by
`mu-engine/tests/pipelines/test_crash_replay_resume_int.py` (Stage C, commit `8970373`,
`test_crash_replay_resumes_and_promotes_with_same_id` +
`test_remember_is_idempotent_on_replay` in `mu-engine/tests/services/test_ingest_int.py`) — both
manipulate the SAME `IngestActivity`/`session_offset` directly, which is not reachable from a
black-box HTTP retry (that necessarily starts a brand-new activity/offset every call). Both are
re-run as part of this stage's "full suite green" gate (build-plan §7).

**Scope vs. the existing lower-level CG-2 test.** `mu-engine/tests/pipelines/
test_crash_replay_resume_int.py` (Stage C, commit `8970373`) already proves the PRECISE mid-pipeline
timing case — a real SIGKILL fired between two specific in-process pipeline stages, verified via
direct `EngineContainer`/pipeline access. This black-box, SDK-level acceptance test cannot reach
inside the container to time a kill between two specific stages of one `add()` call (the whole
point of testing over the wire, not in-process) — so it proves the OUTER, container-restart shape
of the SAME guarantee instead: a write fully durably recorded before the kill survives a real
`docker kill` + restart, and the idempotency key (`{namespace}:{content_hash}`,
`mu_engine/pipelines/concrete/ingest.py:254`) recorded in the now-durable `RedisStageLedger`
(composition.py item 6c) is honored across the restart — a replay of the identical `add()` resumes
rather than re-running/duplicating. Both tests are re-run as part of this stage's "full suite
green" gate (build-plan §7: "Tests authored in earlier stages... are re-run here as part of the
whole") — this file does not duplicate CG-2's own mid-pipeline-timing assertion, it extends the
guarantee one layer out, to the actual deployable artifact.

**Why this test owns its own `docker kill`/`make up` (unlike every other Stage F file, which only
VERIFIES the `make up` precondition, `conftest.py`'s own docstring)**: the kill+restart IS the
scenario under test, not a precondition to it — F2(a) genuinely cannot be expressed any other way.
It restores the SAME container name (`docker-compose.yml`'s `container_name: mu-engine-server`,
never a `mu-dev-*`/`gcmem-*` name) via THIS package's own `Makefile`/compose project, and never
passes `-v` (the Valkey/Qdrant/FalkorDB volumes are the durability under test — wiping them would
defeat the entire point)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from subprocess import CompletedProcess

import httpx
import pytest

pytestmark = pytest.mark.integration

# Duplicated (not imported from ./conftest.py — see conftest.py's `engine_cli` fixture docstring
# for why a direct cross-file dotted import is fragile under this workspace's sibling `tests/`
# package layout): the same literal `conftest.py` itself uses.
ENGINE_BASE_URL = "http://127.0.0.1:8300"

_HEALTH_POLL_TIMEOUT_S = 90.0
_HEALTH_POLL_INTERVAL_S = 2.0


def _wait_for_health(timeout_s: float = _HEALTH_POLL_TIMEOUT_S) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{ENGINE_BASE_URL}/health", timeout=5.0)
            if response.status_code == 200:
                return
            last_error = AssertionError(f"health returned {response.status_code}: {response.text}")
        except httpx.TransportError as exc:  # polling loop, exception IS the signal
            last_error = exc
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    raise AssertionError(
        f"mu-engine-server did not become healthy within {timeout_s}s after restart: {last_error!r}"
    )


def test_write_survives_container_kill_and_replay_is_honest_about_reinforcement(
    engine_up: None,
    engine_token: str,
    engine_cli: Callable[..., CompletedProcess[str]],
    engine_container_name: str,
) -> None:
    del engine_up  # verifies the initial make-up precondition before this test takes over

    marker = uuid.uuid4().hex[:12]
    content = f"F2a-crash-replay-{marker}: Grace worked at IBM"
    user = f"f2a-{marker}"
    session = "s1"

    headers = {"Authorization": f"Bearer {engine_token}", "Content-Type": "application/json"}

    # ---- 1. write via the public HTTP surface (the same wire shape MemoryClient.add() sends) ----
    # STALE-PREMISE FIX. F2a is about CRASH DURABILITY of a genuinely two-tier (STM+MTM) write, not
    # about the promote gate. It used to post a DEFAULT-importance body and assert
    # `promoted is True`, which only held while `SurfaceFacade.add` hardcoded `promote=True` — the
    # REMEDIATION Rank 2 / conformance A6 fix removed that, so a default-importance add correctly
    # stays STM-only and this criterion started failing for a reason that has nothing to do with
    # durability. Sending an explicit high `importance_score` restores the test's ACTUAL premise
    # (a real STM+MTM write to crash-test) instead of weakening its assertions.
    add_response = httpx.post(
        f"{ENGINE_BASE_URL}/memories",
        headers=headers,
        json={
            "content": content,
            "user": user,
            "session": session,
            "importance_score": 0.95,
        },
        timeout=15.0,
    )
    assert add_response.status_code == 201, add_response.text
    first_write = add_response.json()
    memory_id_1 = first_write["memory_id"]
    # The FIRST write is a genuine, fresh promotion: an HONEST receipt claims BOTH tiers.
    assert first_write["promoted"] is True
    assert first_write["tiers_written"] == ["stm", "mtm"]

    # ---- 2. docker kill (ungraceful — SIGKILL, not `docker stop`/`compose down`) ----
    kill = engine_cli("docker", "kill", engine_container_name)
    assert kill.returncode == 0, (
        f"docker kill {engine_container_name} failed: rc={kill.returncode} "
        f"stdout={kill.stdout!r} stderr={kill.stderr!r}"
    )
    assert kill.stdout.strip() == engine_container_name

    # ---- 3. `make up` again — SAME compose project, SAME named volumes (no `-v`, no `reset`) ----
    up = engine_cli("make", "up", timeout=180.0)
    assert up.returncode == 0, (
        f"make up (post-kill) failed: rc={up.returncode}\n{up.stdout}\n{up.stderr}"
    )
    _wait_for_health()

    # ---- 4. assert the write survived the kill (durability) ----
    get_response = httpx.get(
        f"{ENGINE_BASE_URL}/memories/{memory_id_1}",
        headers=headers,
        params={"user": user, "session": session},
        timeout=15.0,
    )
    assert get_response.status_code == 200, (
        f"the pre-kill write (memory_id={memory_id_1!r}) did not survive the container "
        f"kill+restart: GET returned {get_response.status_code} {get_response.text!r}"
    )
    survived = get_response.json()
    assert survived["content"] == content
    assert survived["id"] == memory_id_1

    # ---- 5. re-add of IDENTICAL content after the restart: a legitimate SECOND, INDEPENDENT ----
    # occurrence (reinforcement) — NOT the same-invocation resume CG-2/test_ingest_int.py already
    # prove (this is a brand-new HTTP call, brand-new session_offset, by construction). The content-
    # hash-scoped MTM dedup must still hold, and the receipt must be HONEST about what this call
    # actually did (no new MTM write).
    replay_response = httpx.post(
        f"{ENGINE_BASE_URL}/memories",
        headers=headers,
        json={
            "content": content,
            "user": user,
            "session": session,
            "importance_score": 0.95,  # same premise as the first write (see step 1)
        },
        timeout=15.0,
    )
    assert replay_response.status_code == 201, replay_response.text
    replay_write = replay_response.json()

    # RETURN-IDEMPOTENCY (module docstring's "Re-corrected wording" section): identical content in
    # the SAME namespace resolves to the SAME resident memory_id, even though the two HTTP calls
    # are independent (distinct session_offsets — WriteStmStage's OWN activity-id ledger never
    # collapses them; the SAME id comes back because D4's write-time STM content-hash dedup kept
    # only ONE physical row, and WriteStmStage now surfaces THAT row's id instead of minting a
    # fresh one the store would have discarded).
    assert replay_write["memory_id"] == memory_id_1, (
        "a re-add() of identical content after the kill+restart minted a DIFFERENT memory_id "
        f"than the pre-kill write ({memory_id_1!r} vs {replay_write['memory_id']!r}) — the D4 "
        "write-time STM content-hash dedup index kept only ONE physical row for this content, "
        "so add()'s receipt must return THAT row's id (return-idempotency fix, "
        "DATA-QUALITY-REASSESSMENT §3 'add() idempotency'), not a fresh id the store never "
        "actually kept."
    )
    assert replay_write["content_hash"] == first_write["content_hash"]

    # THE RECEIPT-HONESTY FIX: this call performed NO new MTM write (the content was already
    # promoted by the FIRST call) — the receipt must say so, not claim a fresh two-tier write.
    assert replay_write["promoted"] is False, (
        "the re-add()'s receipt claims `promoted=true` for a call that only reinforced "
        "already-promoted content — no NEW MTM upsert happened for this memory_id "
        f"({replay_write['memory_id']!r}). See mu_engine.services.ingest.IngestService."
        "_promoted_this_call — DeterministicPromoteStage SKIPPED (content-hash ledger-hit) while "
        "WriteStmStage ran fresh (a new activity) means reinforcement, not a new write."
    )
    assert replay_write["tiers_written"] == ["stm"], (
        "the re-add()'s receipt claims MTM was written this call "
        f"(tiers_written={replay_write['tiers_written']!r}), but the content-hash dedup means no "
        "new MTM upsert happened — tiers_written must reflect ONLY what this call actually wrote."
    )

    # No DUPLICATE PROMOTED RECORD: recall for this exact marker must surface exactly ONE matching
    # item — now the STRONGER guarantee that both calls share ONE memory_id makes this close to
    # tautological at the STM layer, but it is still the right black-box check: it also rules out
    # a stray SECOND Qdrant point (DeterministicPromoteStage's content-hash ledger dedupes the
    # actual MTM write). The precise MTM-level guarantee (exactly one Qdrant point, at the shared
    # memory_id, never re-pointed) is proven directly against the real store in
    # `mu-engine/tests/services/test_ingest_int.py::
    # test_second_call_with_different_activity_and_same_content_returns_existing_id`; this
    # black-box recall check only needs to rule out a visible DUPLICATE.
    recall_response = httpx.post(
        f"{ENGINE_BASE_URL}/v1/memories/recall",
        headers=headers,
        json={"text": content, "user": user, "session": session, "limit": 25},
        timeout=15.0,
    )
    assert recall_response.status_code == 200, recall_response.text
    items = recall_response.json()["items"]
    matching = [item for item in items if item.get("content") == content]
    assert len(matching) == 1, (
        f"expected exactly ONE memory matching the marker content after the idempotent replay, "
        f"found {len(matching)}: {matching!r} — the crash+replay produced a duplicate."
    )
    assert matching[0]["memory_id"] in {memory_id_1, replay_write["memory_id"]}
