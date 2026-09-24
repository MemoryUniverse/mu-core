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

**FAULT-HUNT-0924.md F4b/F4c, closed here.** Two defects, both in this one adapter:

1. *No delete path anywhere.* This class had ``put``/``get``/``get_blob`` and nothing that ever
   removes a byte from disk. :meth:`delete` is the real fix, not a stub: it removes the meta
   handle unconditionally (idempotent — an already-absent handle is a no-op ``False``, never an
   error) and removes the underlying blob ONLY when no other meta handle in the SAME namespace
   still names the same ``content_hash`` (a bounded scan of this namespace's own ``meta/``
   directory — cheap, and it is the one ref-count question this adapter can answer on its own
   without reaching into the tier stores; the OTHER ref-count question, "does any live
   ``MemoryItem`` still point at this artifact id", is the caller's — see the port docstring and
   ``mu_engine.surface.facade.SurfaceFacade.delete``, which asks it before calling this at all).
2. *Rooted at a CWD-relative path* (``"./.mu_data/artifacts"``) — F4c measured TEN such trees
   scattered across this one project by shell history alone, an operator asked to purge a user's
   content had no way to enumerate them, and every blob sat in plaintext wherever a process
   happened to be launched from. The default below is now the SAME deterministic,
   user-scoped, ``~``-rooted directory ``SqliteOutbox`` already uses for the daemon's outbox
   (``~/.memory-universe/...``) — one root per machine, independent of CWD, and
   ``.expanduser()``-resolved here the same way ``SqliteOutbox.__init__`` resolves its own path.
   This is a DEFAULT only (DEV-STANDARDS rule 3): the live value is still DI-threaded from
   ``ArtifactFsSettings.content_root`` by the ``STORE_REGISTRY`` factory, which moved to the same
   deterministic default in lockstep (see that settings field's own docstring). Existing
   deployments with blobs already scattered under a project-relative ``.mu_data/artifacts`` are
   NOT migrated by this change — that is a data-location migration (ARCHITECTURE-DELTAS.md), not
   a code fix, and is out of this pass's scope.
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
# Deterministic + user-scoped (FAULT-HUNT-0924.md F4c) — NOT CWD-relative, mirroring
# ``SqliteOutbox``'s own ``~/.memory-universe/...`` default (``mu_client.outbox.sqlite_outbox``);
# ``.expanduser()``-resolved in ``__init__`` below regardless of the process's working directory.
_DEFAULT_CONTENT_ROOT = "~/.memory-universe/artifacts"


class FsContextRepositoryAdapter:
    """Implements ``ContextRepository`` over a real local directory tree."""

    def __init__(self, *, content_root: str = _DEFAULT_CONTENT_ROOT) -> None:
        # ``.expanduser()`` — F4c: a bare ``Path("~/...")`` is NOT resolved automatically and
        # would literally create a directory named ``~`` in the CWD, silently recreating the
        # exact CWD-relative-scatter bug this default exists to close.
        self._root = Path(content_root).expanduser()

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

    async def delete(self, ns: Namespace, artifact_id: str) -> bool:
        return await asyncio.to_thread(self._delete_sync, ns, artifact_id)

    def _delete_sync(self, ns: Namespace, artifact_id: str) -> bool:
        meta_path = self._meta_path(ns, artifact_id)
        if not meta_path.exists():
            return False
        art = ContextArtifact.model_validate_json(meta_path.read_text(encoding="utf-8"))
        meta_path.unlink()
        # Blob GC: only when no OTHER meta handle in this namespace still names the same
        # content_hash (two distinct artifact ids can land on the identical blob — e.g. two
        # captures of identical text; the store is content-addressed, not id-addressed, at the
        # blob layer). A bounded scan of this namespace's own meta/ directory — never a
        # cross-namespace or whole-store walk.
        if not self._content_hash_still_referenced(ns, art.content_hash, excluding=artifact_id):
            blob_path = self._blob_path(ns, art.content_hash)
            blob_path.unlink(missing_ok=True)
        return True

    def _content_hash_still_referenced(
        self, ns: Namespace, content_hash: str, *, excluding: str
    ) -> bool:
        meta_dir = self._root / ns.to_prefix() / "meta"
        if not meta_dir.is_dir():
            return False
        for meta_path in meta_dir.glob("*.json"):
            if meta_path.stem == excluding:
                continue
            other = ContextArtifact.model_validate_json(meta_path.read_text(encoding="utf-8"))
            if other.content_hash == content_hash:
                return True
        return False

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
