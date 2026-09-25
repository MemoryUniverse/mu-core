"""Repo-root pytest conftest — one job: no test run ever writes ``__pycache__``.

**Why this is here and not an ``env`` entry under ``[tool.pytest.ini_options]``.** The obvious
fix — ``env = ["PYTHONDONTWRITEBYTECODE=1"]`` in ``pyproject.toml`` — does not work, in two
independent ways, and both were checked rather than assumed:

1. ``env`` is not a pytest ini key; it belongs to the ``pytest-env`` plugin, which this repo does
   not depend on. ``addopts`` carries ``--strict-config``, so an unknown key is a hard
   ``ERROR: Unknown config option: env`` and **zero tests run** — it would break every gate in
   the repo rather than harden one.
2. Even with the plugin installed it would be too late to do anything. CPython reads
   ``PYTHONDONTWRITEBYTECODE`` exactly once, at interpreter start, into
   ``sys.dont_write_bytecode``; ``importlib`` consults that flag, never the environment. Setting
   the variable from inside a running process leaves ``sys.dont_write_bytecode`` ``False``.

So the flag itself is set, at the earliest point this repo controls. pytest imports the rootdir
``conftest.py`` before it imports any test module — and therefore before any test module imports
``mu_engine``/``mu_contracts``/``mu_local`` — so every source file the suite touches is compiled
in memory and nothing is written next to it.

**What this prevents.** A stale ``.pyc`` left behind by an earlier tree (a since-renamed or
deleted module still shadowed by its cached bytecode) reported five tests RED against a source
tree that was correct. That failure mode costs a full investigation and reproduces on no other
machine, which is the worst combination a test signal can have; a run that writes no bytecode
cannot have it. Nothing about test SEMANTICS changes — the only cost is recompiling on each run.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio

sys.dont_write_bytecode = True


# --- BEGIN VM-ONLY TEST GUARD (CLAUDE.md rule 13) ---
def pytest_cmdline_main(config: object) -> int | None:
    """Refuse to run a SUITE on the developer laptop. See root CLAUDE.md rule 13.

    Suites run on `mu-dev-vm` via `infra/mu-vm/vm_test.sh <repo>`. This is enforced here rather
    than merely written down, because the written rule was ignored repeatedly and a full suite
    costs ~2.5 GB of pytest RSS on a 15 GB laptop that is also running the agents — twice it
    exhausted swap and killed the session mid-run.

    It is not only about footprint. A local run reads the developer's working tree, its caches and
    its ``sys.path``: mu-core's own CI line passes here with 1568 tests and, under
    ``python -m pytest``, collects ZERO with an error that reads like a healthy deselect count.
    A local green does not tell you what a clean checkout or a CI runner sees.

    ESCAPE HATCHES, in order of preference:
      * run it on the VM: ``infra/mu-vm/vm_test.sh <repo> [args]``  (sets MU_ON_VM=1 there)
      * one file runs on the VM too: ``vm_test.sh <repo> tests/x/test_y.py``
      * a deliberate local full run: ``MU_ALLOW_LOCAL_TESTS=1 pytest ...`` — say why in your report.
    """
    import os
    import sys

    if os.environ.get("MU_ON_VM") or os.environ.get("MU_ALLOW_LOCAL_TESTS"):
        return None
    if os.environ.get("CI"):
        return None

    sys.stderr.write(
        "\n"
        "REFUSED: suites run on the VM, not on this laptop (root CLAUDE.md rule 13).\n"
        "  ->  infra/mu-vm/vm_test.sh <repo> [pytest args]\n"
        "      (~15x faster; the stores are localhost there)\n"
        "  ->  ONE file on the VM too: vm_test.sh <repo> tests/x/test_y.py\n"
        "  ->  deliberate local full run: MU_ALLOW_LOCAL_TESTS=1 (and say why)\n"
        "\n"
    )
    return 4


# --- END VM-ONLY TEST GUARD ---


# --- BEGIN SHARED INTEGRATION STORE TEARDOWN (AD-295) ---
#
# AD-295 (2026-09-25): 15 integration files across mu-engine and mu-local hand-rolled a
# `_teardown(settings, uid)` that swept Qdrant collections (and, it turns out, FalkorDB graphs —
# see below) with `if uid in coll.name`. Both physical names are keyed on
# `tenant_partition_digest(org, workspace)`, a SHA-256 digest
# (`mu_engine.storage.mappers.tenancy`) — `mu_mtm__{digest}__{visibility}__{dim}` for Qdrant
# (`qdrant_mapper.collection_name`, AD-1/AD-2) and `mu_g__{digest}__shared` /
# `mu_g__{digest}__u_{user}` for FalkorDB (`falkor_ltm.py::graph_name_for`, AD-8). A digest never
# contains the random `uid` substring the test built its org/workspace from, so `uid in name`
# NEVER matched, in ANY of the 15 files — confirmed live: 293 accumulated Qdrant collections
# (924 MB) eventually took the store down with a `RocksDB IO error`, reported as 50 unrelated-
# looking product-test failures.
#
# Two files (`test_ad266_stat_fields_on_real_stores_int.py`,
# `test_ad267_credential_guard_live_path_int.py`) were hand-fixed for the Qdrant half only, by
# computing `collection_name(ns, 0).removesuffix("0")` as a delete-prefix. That is the CORRECT
# formula but was re-typed per file — exactly the copy-paste that produced the original bug. This
# fixture is the single shared home for it, for BOTH stores, so a new integration file calls
# `register(org, workspace)` once and cannot forget the rest, and an existing file can delete its
# own copy instead of re-typing it a 16th time.
#
# NOTE — a second, previously unreported instance of the SAME bug: every one of the 15 files'
# FalkorDB teardown ALSO reads `if uid in name`, and `graph_name_for` has been digest-keyed since
# AD-8 (2026-08-27) — so FalkorDB graphs from these tests have never been deleted either. It has
# not (yet) produced a visible failure the way Qdrant did, but it is the identical defect and is
# fixed here too rather than left for a future AD-295.
@pytest.fixture
def uid() -> str:
    """Per-test random id used to build an isolated (org, workspace) tenant. Kept as ONE canonical
    definition — most of the 15 files below redefined this identically."""
    import uuid as _uuid

    return _uuid.uuid4().hex[:12]


class TenantRegistry:
    """Handed to a test by the `tenant_store_cleanup` fixture; `register` each tenant it creates."""

    def __init__(self) -> None:
        self._pairs: list[tuple[str, str]] = []

    def register(self, *, org: str, workspace: str) -> None:
        self._pairs.append((org, workspace))

    def pairs(self) -> list[tuple[str, str]]:
        return list(self._pairs)


@pytest_asyncio.fixture
async def tenant_store_cleanup(settings: Any) -> AsyncIterator[TenantRegistry]:
    """Register every (org, workspace) tenant an integration test creates; on teardown, delete the
    REAL Qdrant collections and FalkorDB graphs those tenants own, by their DIGEST-based physical
    name (see module note above) rather than by a `uid` substring, which can never match.

    Usage — replaces a file's own `_teardown`/`_teardown_stores` helper and its `finally: await
    _teardown(...)` call:

        async def mem(settings, uid, tenant_store_cleanup):
            tenant_store_cleanup.register(org=f"org{uid}", workspace=f"ws{uid}")
            memory = LocalMemory(workspace=f"ws{uid}", namespace=f"org{uid}", settings=settings)
            try:
                yield memory
            finally:
                await memory.aclose()
            # no manual store teardown — this fixture's own finalizer runs after `mem`'s and
            # sweeps every (org, workspace) pair registered above.

    A test with more than one tenant (e.g. an `env_uid` variant) calls `register` once per pair.
    """
    import contextlib

    from mu_contracts.domain.model.memory import Namespace, Visibility
    from mu_engine.storage.mappers.tenancy import tenant_partition_digest

    registry = TenantRegistry()
    yield registry

    pairs = registry.pairs()
    if not pairs:
        return

    mtm_prefixes: set[str] = set()
    graph_digests: set[str] = set()
    for org, workspace in pairs:
        probe_ns = Namespace(
            org=org, workspace=workspace, user="_", session="_", visibility=Visibility.PRIVATE
        )
        digest = tenant_partition_digest(probe_ns)
        # visibility/dim vary per write; the digest is the whole tenant partition, so a prefix on
        # `mu_mtm__{digest}__` (no visibility/dim suffix) catches every collection this tenant owns.
        mtm_prefixes.add(f"mu_mtm__{digest}__")
        graph_digests.add(digest)

    with contextlib.suppress(Exception):
        from qdrant_client import AsyncQdrantClient

        qdrant = AsyncQdrantClient(url=settings.storage.vector.url)
        try:
            for coll in (await qdrant.get_collections()).collections:
                if any(coll.name.startswith(p) for p in mtm_prefixes):
                    with contextlib.suppress(Exception):
                        await qdrant.delete_collection(coll.name)
        finally:
            await qdrant.close()

    with contextlib.suppress(Exception):
        from falkordb.asyncio import FalkorDB

        db = FalkorDB(host=settings.storage.graph.host, port=settings.storage.graph.port)
        try:
            for g in await db.list_graphs():
                name = g.decode() if isinstance(g, bytes) else g
                if any(d in name for d in graph_digests):
                    with contextlib.suppress(Exception):
                        await db.select_graph(name).delete()
        finally:
            with contextlib.suppress(Exception):
                await db.connection.aclose()


# --- END SHARED INTEGRATION STORE TEARDOWN (AD-295) ---

# ---------------------------------------------------------------------- real local model gate
_MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"


def minilm_is_cached() -> bool:
    """True when the real MiniLM weights are already in this machine's HF cache.

    The provider tests force `HF_HUB_OFFLINE` on the assumption the weights are present, which
    holds on a developer box and on the VM and is FALSE on a fresh CI runner. That is why
    `pytest (everything that does not need a real store)` has been red with
    `OSError: We couldn't connect to 'https://huggingface.co'` on every run — first 12 tests,
    now 14. The workflow's own comment records the symptom and not the cause.

    A test whose dependency is absent is UNCONFIGURED, not failing, and must never be
    indistinguishable from a real regression — the principle AD-297 applied to the Stage-F tier.
    """
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(_MINILM_REPO, local_files_only=True)
    except Exception:
        return False
    return True


def pytest_collection_modifyitems(config: object, items: list) -> None:  # type: ignore[type-arg]
    """Skip `needs_local_model` tests when the weights are not cached, with the fix named."""
    import pytest as _pytest

    if minilm_is_cached():
        return
    skip = _pytest.mark.skip(
        reason=(
            f"the real {_MINILM_REPO} weights are not in this machine's HF cache; "
            "warm it once with `huggingface-cli download " + _MINILM_REPO + "` "
            "(unconfigured, not a regression)"
        )
    )
    for item in items:
        if "needs_local_model" in item.keywords:
            item.add_marker(skip)
