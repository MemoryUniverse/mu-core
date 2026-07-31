"""``ContextArtifact`` — the provenance-root value every ``kind=REFERENCE`` memory targets.

Authority: ``docs/superpowers/specs/2026-07-22-memory-universe-software-architecture.md``
§4 ``domain/model/context.py`` (l.204-213 — ``ArtifactKind``/``ContextArtifact`` fields) and
§5 ``ContextRepository`` port (l.260-263: ``put``/``get``/``open``), driven from §6
``IngestService.ingest`` step 1 (l.340: "persist raw as ContextArtifact(s) (provenance)").

RE-HOME NOTE (same precedent as ``storage/domain/memory.py:16-21``): CANONICAL pins a
``ContextArtifact`` into ``mu-contracts`` too (``mu_contracts.domain.model.artifact``,
CANONICAL §7.10) — but that scaffold shape carries flat ``org_id``/``workspace_id``/
``namespace_id`` fields, an un-reconciled generation that predates the un-collapsed 5-field
``Namespace`` this package's shipped ``MemoryItem`` (``storage/domain/memory.py:161``) already
uses end-to-end. This module defines the SHIPPED, storage-layer-consistent shape
(``namespace: Namespace``) so a minted artifact and the ``MemoryItem.artifact_ref`` that points
at it share ONE ``Namespace`` type, with no adapter-boundary translation. Reconciling the two
``ContextArtifact`` definitions is OUT OF SCOPE here — the same explicit, flagged debt item
``storage/domain/memory.py`` already carries for ``MemoryItem``/``MemoryNode``.

Immutable by construction (``frozen=True``, mirrors ``mu_contracts.domain.model.artifact.
ContextArtifact``'s own docstring: "a change is a NEW version, never an in-place edit") — a
content change is a fresh ``put()`` that mints a new ``id``/``content_hash``, never a mutation.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from mu_engine.storage.domain.namespace import Namespace

__all__ = ["ArtifactKind", "ContextArtifact"]


def _artifact_id() -> str:
    # mirrors storage/domain/memory.py::_memory_id's "mem_"-prefixed uuid4 convention.
    return f"art_{uuid4().hex}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


class ArtifactKind(StrEnum):
    """The provenance-root source kinds (software-arch spec l.205-206)."""

    TRANSCRIPT = "transcript"
    FILE = "file"
    CODE = "code"
    URL = "url"
    IMAGE = "image"


class ContextArtifact(BaseModel):
    """The provenance root a ``MemoryItem(kind=REFERENCE)`` targets via ``artifact_ref``
    (spec l.190/l.213). Content-free handle: the body is hydrated by id through
    ``ContextRepository.get_blob`` (storage/ports.py) — this DTO never carries the bytes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=_artifact_id, min_length=1)
    namespace: Namespace
    kind: ArtifactKind
    version: str = Field(min_length=1)  # content hash — the immutable version handle (spec l.209)
    uri: str = Field(min_length=1)  # by-id locator into the artifact store, never the body
    content_hash: str = Field(min_length=1)
    provenance_id: str = Field(min_length=1)  # origin lineage stream, required non-empty
    created_at: datetime = Field(default_factory=_utcnow)
