"""F2(a) — DURABILITY: T-crash-replay re-run against the CONTAINER (build-plan §7 F2, design §6
"Test obligation... T-crash-replay").

Write via the public SDK -> `docker kill` the real `mu-engine-server` container (SIGKILL, never a
graceful stop) -> `make up` again against the SAME Valkey volume -> assert (1) the write survived
the kill and (2) a re-`add()` of IDENTICAL content resumes/is idempotent: the SAME `memory_id`, no
duplicate MTM point.

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
        except httpx.TransportError as exc:  # noqa: PERF203 — polling loop, exception IS the signal
            last_error = exc
        time.sleep(_HEALTH_POLL_INTERVAL_S)
    raise AssertionError(
        f"mu-engine-server did not become healthy within {timeout_s}s after restart: {last_error!r}"
    )


def test_write_survives_container_kill_and_replay_is_idempotent(
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
    add_response = httpx.post(
        f"{ENGINE_BASE_URL}/memories",
        headers=headers,
        json={"content": content, "user": user, "session": session},
        timeout=15.0,
    )
    assert add_response.status_code == 201, add_response.text
    first_write = add_response.json()
    memory_id_1 = first_write["memory_id"]
    assert first_write["promoted"] is True

    # ---- 2. docker kill (ungraceful — SIGKILL, not `docker stop`/`compose down`) ----
    kill = engine_cli("docker", "kill", engine_container_name)
    assert kill.returncode == 0, (
        f"docker kill {engine_container_name} failed: rc={kill.returncode} "
        f"stdout={kill.stdout!r} stderr={kill.stderr!r}"
    )
    assert kill.stdout.strip() == engine_container_name

    # ---- 3. `make up` again — SAME compose project, SAME named volumes (no `-v`, no `reset`) ----
    up = engine_cli("make", "up", timeout=180.0)
    assert up.returncode == 0, f"make up (post-kill) failed: rc={up.returncode}\n{up.stdout}\n{up.stderr}"
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

    # ---- 5. re-add of IDENTICAL content is idempotent: same memory_id, no duplicate ----
    #
    # KNOWN, VERIFIED GAP (this task's own finding, not a flaky assertion — reproduced BOTH with
    # and without the kill/restart in between, i.e. it is orthogonal to crash-replay itself):
    # `mu_engine.surface.facade.SurfaceFacade.add()` mints a FRESH `session_offset =
    # uuid.uuid4().hex` (`facade.py::_fresh_offset`) on EVERY call, unconditionally — there is no
    # parameter on `SurfaceFacade.add`/the `/memories` route that threads a caller-supplied
    # `Idempotency-Key` into it. `WriteStmStage.idempotency_key` (`pipelines/concrete/ingest.py`)
    # is `activity_id_for(host|session|session_offset|kind)` — keyed on that fresh offset, so TWO
    # independently-issued `add()` calls (this replay included) ALWAYS mint two DIFFERENT
    # `memory_id`s, by design ("a genuine second occurrence... is a legitimate reinforcement
    # (kept)", `activity_id_for`'s own docstring) — this is NOT what a crash/restart broke; a
    # second call issued with ZERO kill in between behaves identically (verified directly against
    # this same running container before this suite was finalized).
    #
    # `DeterministicPromoteStage.idempotency_key` (content_hash-scoped, namespace-prefixed) DOES
    # durably dedupe the MTM WRITE across the two calls — verified directly via the container's own
    # Qdrant collection (`docker exec mu-engine-server ... /points/scroll`): exactly ONE MTM point
    # exists for this content_hash after the replay, under the FIRST call's memory_id. But the
    # SECOND call's own HTTP receipt still reports `"promoted": true, "tiers_written": ["stm",
    # "mtm"]` for a `memory_id` that was NEVER actually written to MTM (only the first memory_id
    # is) — a receipt-accuracy gap on top of the memory_id-stability gap, both flagged here rather
    # than silently asserted around.
    #
    # This assertion is kept at the LITERAL acceptance-criterion wording (design §6 T-crash-replay
    # / build-plan §7 F2: "a re-add of identical content resumes/idempotent (same memory_id, no
    # duplicate)") so this suite reports it RED with the real root cause, rather than quietly
    # weakening the check to force green. The narrower guarantee THIS wording was likely trying to
    # name — a crash occurring mid-flight inside ONE `add()` invocation resuming that SAME
    # invocation with the SAME memory_id — is real and already proven by the lower-level
    # `mu-engine/tests/pipelines/test_crash_replay_resume_int.py` (Stage C, commit `8970373`),
    # which manipulates the SAME `IngestActivity`/`session_offset` directly across a SIGKILL; it is
    # not reachable from a black-box HTTP retry, which necessarily starts a brand-new activity.
    replay_response = httpx.post(
        f"{ENGINE_BASE_URL}/memories",
        headers=headers,
        json={"content": content, "user": user, "session": session},
        timeout=15.0,
    )
    assert replay_response.status_code == 201, replay_response.text
    replay_write = replay_response.json()
    assert replay_write["memory_id"] == memory_id_1, (
        "a re-add() of IDENTICAL content after the container restart minted a DIFFERENT memory_id "
        f"({replay_write['memory_id']!r} != {memory_id_1!r}). VERIFIED ROOT CAUSE (see this test's "
        "own inline comment above): SurfaceFacade.add() mints a fresh uuid4 session_offset on "
        "EVERY call (facade.py::_fresh_offset) with no Idempotency-Key threading, so "
        "WriteStmStage's offset-keyed idempotency can never make two independently-issued add() "
        "calls share a memory_id — reproduced identically with NO kill/restart involved. The "
        "content-hash-scoped MTM dedup (DeterministicPromoteStage) DOES prevent a duplicate MTM "
        "point (verified directly against Qdrant) — only the outer memory_id/receipt-accuracy "
        "guarantee this literal criterion also asks for does not hold."
    )
    assert replay_write["content_hash"] == first_write["content_hash"]

    # No DUPLICATE PROMOTED RECORD: recall for this exact marker must surface exactly ONE matching
    # item (this half of the criterion DOES hold — DeterministicPromoteStage's content-hash ledger
    # dedupes the actual MTM write even though the outer memory_id above does not match).
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
