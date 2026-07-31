"""``FsContextRepositoryAdapter`` — the LOCAL-plane, filesystem-backed ``ContextRepository``.

Implements ``mu_engine.storage.ports.ContextRepository`` (software-arch spec §5, l.260-263):
the store step 1 of ``IngestService.ingest`` (spec §6, l.340) writes THROUGH, minting the
:class:`~mu_engine.storage.domain.artifact.ContextArtifact` provenance root a
``kind=REFERENCE`` ``MemoryItem`` then targets via ``artifact_ref``.

FLAGGED SIMPLIFICATION (deliberate, minimal-correct floor — task instruction "the full
git-backed content_git store can be a later refinement; flag it"): the spec's tree listing
(l.78) and §Ported-adoptions note (l.437) name ``adapters/stores/content_git.py`` — a
VERSIONED, worktree-merge store "ported from Letta Context Repositories" — as the eventual
``ContextRepository`` adapter. That does not exist yet (confirmed: no ``content_git``/
``ContextRepository``/``ArtifactRepository`` symbol anywhere in this tree before this module).
This adapter is the floor BENEATH it, not a replacement: a plain, real, persistent,
content-addressed filesystem store —

    {content_root}/{namespace.to_prefix()}/meta/{artifact.id}.json    — the metadata handle
    {content_root}/{namespace.to_prefix()}/blobs/{hash[:2]}/{hash}.bin — the body, content-addressed

— genuinely persisted (survives process restart; on-disk, not in-memory) and genuinely
hydratable by id (``get``) or by content (``get_blob``), satisfying CANONICAL §3.1's
"content-free handle, hydrated by id" contract without git plumbing. Upgrading to
``content_git.py`` (worktree merge, version history beyond "one hash overwrite") is future
work; this adapter's on-disk LAYOUT is deliberately git-repo-COMPATIBLE (content-addressed
blobs under a 2-char fan-out directory, exactly git's own object-store shape) so that upgrade
is a drop-in swap of the adapter class, never a data-migration.

Fully async at the port boundary (DEV-STANDARDS rule 1): every blocking ``pathlib``/file call
runs via ``asyncio.to_thread`` so the event loop is never blocked, matching every other
adapter's async discipline — though, unlike the network stores, this one has no ``retry_io``
wrapper: local filesystem I/O is not a "transient 5xx/429/network" failure class
(``platform.exceptions.classify_error``'s predicate), it either works or the process has a
hard local-disk fault a retry cannot paper over (fail loud, DEV-STANDARDS rule 8).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from mu_engine.storage.domain.artifact import ContextArtifact
from mu_engine.storage.domain.namespace import Namespace

__all__ = ["FsContextRepositoryAdapter"]

# Constructor DEFAULT only (DEV-STANDARDS rule 3: no hardcoded constant lives in adapter logic).
# The live value is DI-threaded from the central Settings tree
# (``mu_contracts.config.ArtifactFsSettings.content_root``) by the ``STORE_REGISTRY`` factory
# (``mu_engine.storage.factories._build_artifact_fs``); a bare ``FsContextRepositoryAdapter()``
# (e.g. in a unit test) still gets a sane, named default rather than a silent unconfigured path.
_DEFAULT_CONTENT_ROOT = "./.mu_data/artifacts"


class FsContextRepositoryAdapter:
    """Implements ``ContextRepository`` over a real local directory tree."""

    def __init__(self, *, content_root: str = _DEFAULT_CONTENT_ROOT) -> None:
        self._root = Path(content_root)

    def _meta_path(self, ns: Namespace, artifact_id: str) -> Path:
        return self._root / ns.to_prefix() / "meta" / f"{artifact_id}.json"

    def _blob_path(self, ns: Namespace, content_hash: str) -> Path:
        # 2-char fan-out (git's own object-store shape — module docstring "drop-in swap" note).
        return self._root / ns.to_prefix() / "blobs" / content_hash[:2] / f"{content_hash}.bin"

    async def put(self, art: ContextArtifact, blob: bytes) -> ContextArtifact:
        return await asyncio.to_thread(self._put_sync, art, blob)

    def _put_sync(self, art: ContextArtifact, blob: bytes) -> ContextArtifact:
        blob_path = self._blob_path(art.namespace, art.content_hash)
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        # content-addressed: identical (namespace, content_hash) => identical bytes, so a
        # re-`put()` (crash-replay retry) is an idempotent overwrite, never a duplicate blob.
        blob_path.write_bytes(blob)
        meta_path = self._meta_path(art.namespace, art.id)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(art.model_dump_json(), encoding="utf-8")
        return art

    async def get(self, ns: Namespace, artifact_id: str) -> ContextArtifact | None:
        return await asyncio.to_thread(self._get_sync, ns, artifact_id)

    def _get_sync(self, ns: Namespace, artifact_id: str) -> ContextArtifact | None:
        meta_path = self._meta_path(ns, artifact_id)
        if not meta_path.exists():
            return None
        return ContextArtifact.model_validate_json(meta_path.read_text(encoding="utf-8"))

    async def get_blob(self, ns: Namespace, artifact_id: str) -> bytes | None:
        art = await self.get(ns, artifact_id)
        if art is None:
            return None
        return await asyncio.to_thread(self._get_blob_sync, ns, art.content_hash)

    def _get_blob_sync(self, ns: Namespace, content_hash: str) -> bytes | None:
        blob_path = self._blob_path(ns, content_hash)
        if not blob_path.exists():
            return None
        return blob_path.read_bytes()
