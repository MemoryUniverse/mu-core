"""``FsContextRepositoryAdapter.delete`` — the real delete path FAULT-HUNT-0924.md F4b found
missing entirely ("no ``unlink``, no ``remove``, no ``delete`` anywhere in the file"), plus F4c's
deterministic-root fix. A REAL on-disk directory tree (``tmp_path``), zero mocks — the same
discipline ``test_context_provenance_int.py`` already uses for this adapter's ``put``/``get``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mu_engine.storage.adapters.content_fs import (
    _DEFAULT_CONTENT_ROOT,
    FsContextRepositoryAdapter,
)
from mu_engine.storage.domain.artifact import ArtifactKind, ContextArtifact
from mu_engine.storage.domain.namespace import Namespace, Visibility

pytestmark = pytest.mark.unit


def _ns() -> Namespace:
    return Namespace(
        org="default", workspace="local", user="u1", session="s1", visibility=Visibility.PRIVATE
    )


def _artifact(ns: Namespace, *, artifact_id: str, content_hash: str) -> ContextArtifact:
    return ContextArtifact(
        id=artifact_id,
        namespace=ns,
        kind=ArtifactKind.TRANSCRIPT,
        version=content_hash,
        uri=f"local://{artifact_id}",
        content_hash=content_hash,
        provenance_id="prov_x",
    )


async def test_delete_removes_the_meta_handle_and_the_blob(tmp_path: Path) -> None:
    repo = FsContextRepositoryAdapter(content_root=str(tmp_path))
    ns = _ns()
    art = _artifact(ns, artifact_id="art_1", content_hash="hash_a")
    await repo.put(art, b"Ada lives in Paris")

    removed = await repo.delete(ns, "art_1")

    assert removed is True
    assert await repo.get(ns, "art_1") is None
    assert await repo.get_blob(ns, "art_1") is None
    # the blob itself is gone off disk, not merely unreachable through the port
    blob_path = tmp_path / ns.to_prefix() / "blobs" / "ha" / "hash_a.bin"
    assert not blob_path.exists()


async def test_delete_of_an_absent_artifact_is_an_idempotent_false(tmp_path: Path) -> None:
    repo = FsContextRepositoryAdapter(content_root=str(tmp_path))
    removed = await repo.delete(_ns(), "never_put")
    assert removed is False


async def test_delete_keeps_a_blob_still_named_by_another_artifact_id(tmp_path: Path) -> None:
    """Content-addressed sharing at the FILESYSTEM layer: two DISTINCT artifact ids (two
    captures of byte-identical content) land on the SAME blob path. Deleting one must not corrupt
    the other's body."""
    repo = FsContextRepositoryAdapter(content_root=str(tmp_path))
    ns = _ns()
    art_a = _artifact(ns, artifact_id="art_a", content_hash="shared_hash")
    art_b = _artifact(ns, artifact_id="art_b", content_hash="shared_hash")
    await repo.put(art_a, b"identical text")
    await repo.put(art_b, b"identical text")

    removed = await repo.delete(ns, "art_a")

    assert removed is True
    assert await repo.get(ns, "art_a") is None  # art_a's own handle IS gone
    # art_b's handle AND body both survive
    assert await repo.get(ns, "art_b") is not None
    assert await repo.get_blob(ns, "art_b") == b"identical text"


async def test_delete_is_idempotent_on_a_retried_call(tmp_path: Path) -> None:
    repo = FsContextRepositoryAdapter(content_root=str(tmp_path))
    ns = _ns()
    art = _artifact(ns, artifact_id="art_1", content_hash="hash_a")
    await repo.put(art, b"body")

    first = await repo.delete(ns, "art_1")
    second = await repo.delete(ns, "art_1")  # retried delete — no error, honest False

    assert first is True
    assert second is False


def test_default_content_root_is_deterministic_not_cwd_relative() -> None:
    """FAULT-HUNT-0924.md F4c: the default used to be ``"./.mu_data/artifacts"`` — CWD-relative,
    so it scattered a fresh tree under every directory a process was ever launched from. The
    fixed default is `~`-rooted and resolves the SAME regardless of the process's working
    directory."""
    assert not _DEFAULT_CONTENT_ROOT.startswith(".")
    assert _DEFAULT_CONTENT_ROOT.startswith("~")

    repo_from_cwd_a = FsContextRepositoryAdapter()
    # `.expanduser()`'d, never a literal `~` directory under wherever the test happens to run.
    assert "~" not in str(repo_from_cwd_a._root)
    assert repo_from_cwd_a._root == Path(_DEFAULT_CONTENT_ROOT).expanduser()
