"""``SurfaceFacade.delete`` -> artifact GC, against the REAL adapters (FAULT-HUNT-0924 F4b).

`test_facade_artifact_delete_unit.py` proves the fan-out ORDER with in-memory tier doubles whose
`by_artifact` answer is a settable canned list. That is the right shape for the ordering question
and the wrong shape for the question a user asks, which is "after I delete, is my text still on
disk?" — because the canned answer is exactly the thing the real adapters decide differently.

This file therefore uses the REAL `QdrantMtmAdapter` / `FalkorLtmAdapter` / `ValkeyStmAdapter` /
`FsContextRepositoryAdapter` (ZERO mocks — the `_RealContainer` below is a wiring holder, not a
double: every attribute on it is a shipped adapter against a live mu-dev-* container) and then
goes looking for the verbatim bytes on the filesystem.

Fixtures `settings`/`uid`/`valkey_client`/`make_stm`/`falkor_db`/`ltm`/`qdrant_client`/`mtm` come
from `tests/lifecycle/conftest.py`'s siblings — see this directory's own `conftest.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from mu_engine.platform.clock import FrozenClock
from mu_engine.storage.adapters.content_fs import FsContextRepositoryAdapter
from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.valkey_stm import ValkeyStmAdapter
from mu_engine.storage.domain.artifact import ArtifactKind, ContextArtifact
from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemorySource, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.surface.facade import SurfaceFacade

_T0 = datetime(2026, 6, 1, tzinfo=UTC)
_BODY_TEXT = "Ada lives in Paris and works at Acme"


class _RealContainer:
    """A `LocalContainerLike` wiring holder carrying the REAL adapters. Not a double: nothing
    here answers a question — every attribute forwards to a live store."""

    def __init__(
        self,
        *,
        stm: ValkeyStmAdapter,
        mtm: QdrantMtmAdapter,
        ltm: FalkorLtmAdapter,
        artifacts: FsContextRepositoryAdapter,
    ) -> None:
        self.stm = stm
        self.mtm = mtm
        self.ltm = ltm
        self._artifacts = artifacts
        self.ingest = None
        self.distill = None
        self.recall = None
        self.mode_gate = None
        self.llm = None

    @property
    def artifacts(self) -> FsContextRepositoryAdapter:
        return self._artifacts

    @property
    def bus(self) -> None:
        return None


def _facade(container: _RealContainer, *, org: str, workspace: str) -> SurfaceFacade:
    return SurfaceFacade(
        container,  # type: ignore[arg-type]
        workspace=workspace,
        namespace=org,
        clock=FrozenClock(_T0),
    )


def _blob_paths(root: Path) -> list[Path]:
    return [p for p in root.rglob("*.bin") if p.is_file()]


def _grep_tree_for(root: Path, needle: str) -> list[Path]:
    """Go looking for the verbatim text on disk — the F4 acceptance question, asked literally."""
    hits: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            if needle.encode("utf-8") in p.read_bytes():
                hits.append(p)
        except OSError:  # pragma: no cover - a tree we just wrote is readable
            continue
    return hits


async def _write_reference_memory(
    *,
    mtm: QdrantMtmAdapter,
    artifacts: FsContextRepositoryAdapter,
    ns: Namespace,
    content: str,
) -> tuple[str, str]:
    """One captured activity: its raw body in the artifact store, one REFERENCE memory in MTM
    pointing at it — the shape `PersistRawArtifactStage` + `WriteStmStage`/promotion produce."""
    content_hash = f"h{abs(hash(content)):x}"
    art = await artifacts.put(
        ContextArtifact(
            namespace=ns,
            kind=ArtifactKind.TRANSCRIPT,
            version=content_hash,
            uri=f"local://{content_hash}",
            content_hash=content_hash,
            provenance_id="prov_verify",
        ),
        content.encode("utf-8"),
    )
    item = MemoryItem(
        content=content,
        kind=MemoryKind.REFERENCE,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        source=MemorySource.USER,
        artifact_ref=art.id,
        created_at=_T0,
        updated_at=_T0,
        embedding=[0.1] * mtm._dim,
        embedding_model="verify-fixture",
    )
    await mtm.upsert(item)
    return item.id, art.id


@pytest.mark.integration
async def test_delete_really_removes_the_artifact_body_from_disk(
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
    uid: str,
    tmp_path: Path,
) -> None:
    """FAULT-HUNT-0924 F4: "write, delete, then go looking for the text in the artifact store
    directly. Absence, or it is not fixed."

    The memory has a REAL MTM copy, which is what makes this different from the unit suite: every
    case there put the item in LTM only, so the MTM fan-out arm returned `[]` by construction and
    never had to decide anything.

    MUTATION CHECK (run, red): delete the `if located.state is MemoryState.ACTIVE` guard from
    `SurfaceFacade._maybe_gc_artifact`'s fan-out filter — the just-expired memory answers its own
    reference-count question, the GC short-circuits, and the blob is still on disk.
    """
    ns = Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user="u1",
        session="s1",
        visibility=Visibility.PRIVATE,
    )
    artifacts = FsContextRepositoryAdapter(content_root=str(tmp_path))
    container = _RealContainer(stm=make_stm(), mtm=mtm, ltm=ltm, artifacts=artifacts)
    facade = _facade(container, org=ns.org, workspace=ns.workspace)

    memory_id, artifact_id = await _write_reference_memory(
        mtm=mtm, artifacts=artifacts, ns=ns, content=_BODY_TEXT
    )
    # CONTROL: the verbatim text really is on disk before the delete (otherwise "absent
    # afterwards" proves nothing at all).
    assert _grep_tree_for(tmp_path, _BODY_TEXT), "precondition: the body was never written"
    assert _blob_paths(tmp_path)

    result = await facade.delete(memory_id, user="u1", session="s1")
    assert result.invalidated is True

    assert await artifacts.get(ns, artifact_id) is None, "the meta handle survived the delete"
    assert await artifacts.get_blob(ns, artifact_id) is None
    leftovers = _grep_tree_for(tmp_path, _BODY_TEXT)
    assert leftovers == [], f"verbatim memory content still on disk after delete: {leftovers}"


@pytest.mark.integration
async def test_delete_keeps_a_body_a_second_live_memory_still_references(
    mtm: QdrantMtmAdapter,
    ltm: FalkorLtmAdapter,
    make_stm: Callable[..., ValkeyStmAdapter],
    uid: str,
    tmp_path: Path,
) -> None:
    """The other side of the same guard, against real Qdrant: content-addressed sharing must NOT
    be broken by the fix above. A second, still-ACTIVE MTM memory pointing at the same artifact
    keeps the body alive — proved by a real `by_artifact` scroll, not a canned list."""
    ns = Namespace(
        org=f"org{uid}",
        workspace=f"ws{uid}",
        user="u1",
        session="s2",
        visibility=Visibility.PRIVATE,
    )
    artifacts = FsContextRepositoryAdapter(content_root=str(tmp_path))
    container = _RealContainer(stm=make_stm(), mtm=mtm, ltm=ltm, artifacts=artifacts)
    facade = _facade(container, org=ns.org, workspace=ns.workspace)

    memory_id, artifact_id = await _write_reference_memory(
        mtm=mtm, artifacts=artifacts, ns=ns, content=_BODY_TEXT
    )
    sibling = MemoryItem(
        content="Ada's employer, as a proposition",
        kind=MemoryKind.PROPOSITION,
        namespace=ns,
        owner_id=ns.user,
        workspace_id=ns.workspace,
        session_id=ns.session,
        tier=MemoryTier.MTM,
        source=MemorySource.USER,
        artifact_ref=artifact_id,
        created_at=_T0,
        updated_at=_T0,
        embedding=[0.2] * mtm._dim,
        embedding_model="verify-fixture",
    )
    await mtm.upsert(sibling)

    await facade.delete(memory_id, user="u1", session="s2")

    still: Any = await artifacts.get(ns, artifact_id)
    assert still is not None, "a body another live memory still references was deleted"
    assert await artifacts.get_blob(ns, artifact_id) == _BODY_TEXT.encode("utf-8")
