"""ContextArtifact — the immutable, content-addressed, versioned resource object.

Authority: CANONICAL §7.10 (the 3-object model MemoryItem / ContextArtifact /
ComposedContext), software-arch §3.2/§4 (l.144-151 via LINKAGE-VERIFY.md:20), and the
reference-aware retention decision (CLOSEOUT-2-RESOLUTIONS.md:147, CANONICAL §7.10 G6).

A ``ContextArtifact`` is the provenance root a ``MemoryItem(kind=REFERENCE)`` points at via
``artifact_ref``. Its BODY lives in the artifact store (server object-store on SHARED,
``content_root`` on LOCAL) and is hydrated by id at render time (CANONICAL §3.1); this DTO is
a content-free handle — it carries the ``uri``/``content_hash`` locator, never the bytes.
Immutable by construction: a change is a NEW version (content_hash), never an in-place edit.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ArtifactKind", "ContextArtifact", "Retention", "RetentionPolicy"]


class ArtifactKind(StrEnum):
    """The provenance-root source kinds (governance ShareableType ARTIFACT comment:
    "transcript/file/code/url/image")."""

    TRANSCRIPT = "transcript"
    FILE = "file"
    CODE = "code"
    URL = "url"
    IMAGE = "image"


class RetentionPolicy(StrEnum):
    """How the artifact's retention is decided (software-arch §3.2)."""

    REFERENCE_COUNTED = (
        "reference_counted"  # kept while ≥1 live MemoryItem.artifact_ref / composed ref
    )
    PINNED = "pinned"  # never GC-eligible (explicit user pin)
    RECENCY_BOUNDED = "recency_bounded"  # aged out past a recency window


class Retention(BaseModel):
    """Reference-aware retention marker (CANONICAL §7.10 G6): an artifact with ≥1 live
    ``MemoryItem.artifact_ref`` or ``ComposedContext.source_artifact_ids`` reference is NOT
    GC-eligible. The ref-count is maintained by the bounded cross-store integrity check (the
    same obligation class as supersession, §7.5), not stored authoritatively here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: RetentionPolicy = RetentionPolicy.REFERENCE_COUNTED
    reference_count: int = Field(default=0, ge=0)  # advisory snapshot; authority = by_artifact()
    pinned: bool = False
    expires_at: datetime | None = None  # set under RECENCY_BOUNDED

    def is_gc_eligible(self, *, at: datetime) -> bool:
        """An artifact is GC-eligible only when unpinned, unreferenced, and past any recency
        window (CANONICAL §7.10 G6)."""
        if self.pinned or self.policy is RetentionPolicy.PINNED:
            return False
        if self.reference_count > 0:
            return False
        if self.policy is RetentionPolicy.RECENCY_BOUNDED and self.expires_at is not None:
            return self.expires_at <= at
        return self.policy is RetentionPolicy.REFERENCE_COUNTED


class ContextArtifact(BaseModel):
    """A content-addressed, versioned, immutable resource — the provenance root of a
    ``kind=REFERENCE`` memory. Content-free handle: body is hydrated by id (CANONICAL §3.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    org_id: str = Field(min_length=1)  # tenant/billing/residency root (un-collapse)
    workspace_id: str = Field(min_length=1)
    namespace_id: str = Field(min_length=1)  # the logical origin partition (η .workspace-grouping)
    kind: ArtifactKind
    version: str = Field(min_length=1)  # content hash / git rev — the immutable version handle
    uri: str = Field(min_length=1)  # by-id locator into the artifact store (NOT the body)
    content_hash: str = Field(min_length=1)  # ties a grant to an EXACT version (ShareableRef)
    provenance_id: str = Field(min_length=1)  # origin lineage stream (required non-empty, §7.10 G4)
    retention: Retention = Retention()
    created_at: datetime
