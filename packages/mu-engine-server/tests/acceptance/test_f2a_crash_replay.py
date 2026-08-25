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
⚠ Since 2026-08-25 this file ALSO issues one `make up` as a genuine precondition (step 1, the
freshness build), so the old absolute phrasing "this file's `make up` is never a precondition" no
longer holds: it now owns BOTH a precondition build and the scenario restart, and step 1's own
comment explains why the first is what makes the second a cache hit.
It restores the SAME container name (`docker-compose.yml`'s `container_name: mu-engine-server`,
never a `mu-dev-*`/`gcmem-*` name) via THIS package's own `Makefile`/compose project, and never
passes `-v` (the Valkey/Qdrant/FalkorDB volumes are the durability under test — wiping them would
defeat the entire point).

**Why this test ALSO runs `make up` once BEFORE the kill (the "freshness build" step below), not
just after.** `make up` (`docker compose up -d --build`) BUILDS, not merely starts — a real VM run
timed out mid-build under the old fixed 180s budget, SIGKILLed `make up` itself, and left the
container `Exited(137)` for the NEXT test in the session to find. Two independent fixes address
that, at two different layers: `MAKE_UP_TIMEOUT_S` (conftest.py) is a raised, env-overridable
safety margin for a genuine cold build; this pre-kill build step is what tries to make a cold build
UNNECESSARY at the point that matters (the post-kill restart) in the first place. `engine_up`
(the precondition fixture) only proves the container is reachable NOW — it says nothing about
whether the image currently running matches this checkout's current source, so the post-kill
rebuild's cache-hit-ness was previously an unfalsifiable assumption about how recently the operator
happened to run their OWN `make up`. Building once, immediately before the kill, replaces that
assumption with a construction: the image behind the container `docker kill` destroys (`docker
kill` sends SIGKILL to the process; it removes NEITHER the container nor the image) is, by the
time of the kill,
provably built from this exact checkout, with zero time and zero source drift between that build
and the post-kill rebuild — so the post-kill `make up` has nothing new to build. The post-kill
duration assertion further down makes that claim observable rather than trusting it in prose (a
prior revision's central hypothesis was exactly this, asserted but measured nowhere).

**Disk cost of this design, and how we know.** Per F2a run this adds at most ONE extra `docker
compose up -d --build` invocation (the pre-kill freshness build) beyond the single post-kill
restart the test always needed. Docker's build cache is content-addressed per layer (verified
against this package's own `Dockerfile`/`.dockerignore` design, not measured — see below): if the
operator's own precondition `make up` already built this exact checkout (the common case — the
release-gate run instruction is "bring `make up` up ONCE, [...] run all criteria"), the pre-kill
build is ALSO a cache hit and this design costs nothing extra. The one case it costs a real build is
when source drifted between the operator's `make up` and this test running — exactly the case
where a real rebuild is correct and unavoidable, and it costs ONE cold build's cache growth, never
two (the post-kill rebuild that immediately follows is a hit by construction, per the paragraph
above). This reasoning was NOT verified by actually running `make up` twice and diffing build-cache
size — that requires the dev VM, which this task is expressly forbidden from touching; it is
reasoned from `Dockerfile`'s `COPY` layers and `.dockerignore`'s exclusion of `.venv`/caches/
`tests/`/logs (mu-core/.dockerignore, the ACTUALLY-active ignore file per that file's own header —
the per-Dockerfile `Dockerfile.dockerignore` convention is a BuildKit
feature and is NOT known to be inactive on the machine that matters: that was checked on a dev
sandbox, not on the VM this suite runs against, so treat it as unverified rather than as a
premise)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from subprocess import CompletedProcess

import httpx
import pytest

pytestmark = pytest.mark.integration

# NOTE: the engine base URL is NOT re-declared here any more. It used to be a duplicated literal,
# which silently diverged from conftest.py's env-aware `ENGINE_BASE_URL`: setting
# MU_ENGINE_SERVER_BASE_URL moved the health poll but not this file's HTTP calls, pointing the two
# halves of one test at two different servers. It arrives as the `engine_base_url` fixture instead.

# NOTE: no local `_wait_for_health`/timeout constants here anymore — `wait_for_health`,
# `make_up_timeout_s`, and `make_up_cache_hit_budget_s` are injected fixtures from ./conftest.py
# (single source of truth; see that module's own docstrings for why — a prior revision hardcoded
# the same budgets independently in both files).


def test_write_survives_container_kill_and_replay_is_honest_about_reinforcement(
    engine_up: None,
    engine_token: str,
    engine_cli: Callable[..., CompletedProcess[str]],
    engine_container_name: str,
    engine_base_url: str,
    engine_restore_guard: None,
    make_up_timeout_s: float,
    make_up_cache_hit_budget_s: float,
    wait_for_health: Callable[..., None],
) -> None:
    del engine_up  # verifies the initial make-up precondition before this test takes over
    del engine_restore_guard  # teardown-only backstop — see conftest.py's own fixture docstring

    marker = uuid.uuid4().hex[:12]
    content = f"F2a-crash-replay-{marker}: Grace worked at IBM"
    user = f"f2a-{marker}"
    session = "s1"

    headers = {"Authorization": f"Bearer {engine_token}", "Content-Type": "application/json"}

    # ---- 1. pre-kill freshness build: guarantees the image behind the container `docker kill` is
    # about to destroy was built from THIS checkout's current source (module docstring's "Why this
    # test ALSO runs `make up` once BEFORE the kill" section) — makes step 4's post-kill rebuild a
    # cache hit BY CONSTRUCTION rather than by trusting how recently the operator's own precondition
    # `make up` happened to run. Uses the full safety-margin timeout: this IS the one call in this
    # test that might be a genuine cold build, if source drifted since that precondition run.
    prekill_build = engine_cli("make", "up", timeout=make_up_timeout_s)
    assert prekill_build.returncode == 0, (
        f"pre-kill `make up` (freshness build) failed: rc={prekill_build.returncode}\n"
        f"{prekill_build.stdout}\n{prekill_build.stderr}"
    )
    wait_for_health()

    # ---- 2. write via the public HTTP surface (the same wire shape MemoryClient.add() sends) ----
    # STALE-PREMISE FIX. F2a is about CRASH DURABILITY of a genuinely two-tier (STM+MTM) write, not
    # about the promote gate. It used to post a DEFAULT-importance body and assert
    # `promoted is True`, which only held while `SurfaceFacade.add` hardcoded `promote=True` — the
    # REMEDIATION Rank 2 / conformance A6 fix removed that, so a default-importance add correctly
    # stays STM-only and this criterion started failing for a reason that has nothing to do with
    # durability. Sending an explicit high `importance_score` restores the test's ACTUAL premise
    # (a real STM+MTM write to crash-test) instead of weakening its assertions.
    add_response = httpx.post(
        f"{engine_base_url}/memories",
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

    # ---- 3. docker kill (ungraceful — SIGKILL, not `docker stop`/`compose down`) ----
    kill = engine_cli("docker", "kill", engine_container_name)
    assert kill.returncode == 0, (
        f"docker kill {engine_container_name} failed: rc={kill.returncode} "
        f"stdout={kill.stdout!r} stderr={kill.stderr!r}"
    )
    assert kill.stdout.strip() == engine_container_name

    # ---- 4. `make up` again — SAME compose project, SAME named volumes (no `-v`, no `reset`).
    # By construction (step 1 above), this rebuilds from the exact same source it just built, so it
    # is EXPECTED to be a Docker layer-cache hit. Measured, not assumed: a cache MISS here means
    # step 1's "nothing changed in between" premise broke, and that needs to be visible rather than
    # silently absorbed into the full safety-margin timeout (module docstring; redo-spec finding 6).
    restart_started = time.monotonic()
    up = engine_cli("make", "up", timeout=make_up_timeout_s)
    restart_duration_s = time.monotonic() - restart_started
    assert (
        up.returncode == 0
    ), f"make up (post-kill) failed: rc={up.returncode}\n{up.stdout}\n{up.stderr}"
    wait_for_health()
    # NOTE: the cache-hit budget is CHECKED AT THE END of this test, not here. Asserting it at this
    # point would let a merely-SLOW restart fail the test BEFORE steps 5-7 ever run — and those
    # steps are the test's actual contract (the write survived SIGKILL; the replay receipt is
    # honest). A cache miss is a performance/disk regression worth failing on, but it must never
    # pre-empt the durability verdict, or a red F2a stops telling us the one thing we built it for.

    # ---- 5. assert the write survived the kill (durability) ----
    get_response = httpx.get(
        f"{engine_base_url}/memories/{memory_id_1}",
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

    # ---- 6. re-add of IDENTICAL content after the restart: a legitimate SECOND, INDEPENDENT ----
    # occurrence (reinforcement) — NOT the same-invocation resume CG-2/test_ingest_int.py already
    # prove (this is a brand-new HTTP call, brand-new session_offset, by construction). The content-
    # hash-scoped MTM dedup must still hold, and the receipt must be HONEST about what this call
    # actually did (no new MTM write).
    replay_response = httpx.post(
        f"{engine_base_url}/memories",
        headers=headers,
        json={
            "content": content,
            "user": user,
            "session": session,
            "importance_score": 0.95,  # same premise as the first write (see step 2)
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
        f"{engine_base_url}/v1/memories/recall",
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

    # ---- 8. LAST: was the post-kill restart the cache hit step 1 makes it by construction? ----
    # Deliberately the final assertion in this test. Every durability and replay-honesty claim
    # above has already been checked by the time we get here, so a cache miss reports a real
    # performance/disk regression WITHOUT ever masking the verdict on what F2a actually exists to
    # prove. Failing here means step 1's "nothing changed in between" premise broke — which is
    # worth investigating rather than silently absorbing into the full safety-margin timeout, since
    # a cold rebuild on the dev VM costs ~17 GB of Docker build cache (measured 2026-08-25).
    assert restart_duration_s <= make_up_cache_hit_budget_s, (
        f"post-kill `make up` took {restart_duration_s:.1f}s, expected a Docker layer-cache hit "
        f"(budget {make_up_cache_hit_budget_s}s — override with "
        "MU_ENGINE_SERVER_MAKE_UP_CACHE_HIT_BUDGET_S if this host is just slower). This test "
        "rebuilt from the SAME source immediately before the kill (step 1), so the post-kill "
        "rebuild should have had nothing new to build. NOTE: every durability assertion above "
        "PASSED — the crash-replay contract itself is intact; this is a build-cache regression."
    )
