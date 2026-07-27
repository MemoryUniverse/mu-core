"""``RelationalMapper`` — MemoryItem -> RelationalRow, CONTENT-FREE by construction (spec §5).

Projects ``MemoryItem`` into the content-free ``memory_provenance`` row (spec §2.4). Its
``to_store`` DROPS content/subject/object text by construction — never a leak path
(spec §5 line 662, Contract-change 2 test 3). ``from_store`` reconstructs a metadata-only
``MemoryItem`` SHELL for correlation (documented as such): the relational mirror is NOT the
content authority (content lives in Redis/Qdrant/FalkorDB), so a full round-trip is
impossible-by-design; identity + lifecycle fields round-trip, content does not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from mu_engine.storage.domain.memory import MemoryItem, MemoryKind, MemoryState, MemoryTier
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.ports import RelationalRow

__all__ = ["CONTENT_FREE_SHELL", "RelationalMapper"]

CONTENT_FREE_SHELL = ""  # the sentinel content of a reconstructed correlation-only shell


class RelationalMapper:
    """Implements ``RowMapper[RelationalRow]`` (spec §5) — content-free by construction."""

    def to_store(self, item: MemoryItem) -> RelationalRow:
        # ONLY ids / hashes / enums / counts / timestamps — NEVER content/subject/object text.
        cols: dict[str, Any] = {
            "workspace_id": item.workspace_id,
            "namespace_prefix": item.namespace.to_prefix(),
            "owner_id": item.owner_id,
            "visibility": item.namespace.visibility.value,
            "kind": item.kind.value,
            "content_hash": item.content_hash,
            "source": item.source.value,
            "tier": item.tier.value,
            "state": item.state.value,
            "artifact_ref": item.artifact_ref,
            "provenance_id": item.provenance_id,
            "superseded_by": item.metadata.get("superseded_by"),
            "synced_at": datetime.now(UTC),
            "meta": {"access_count": item.access_count},  # content-free counts only
        }
        return RelationalRow(table="memory_provenance", pk={"memory_id": item.id}, cols=cols)

    def from_store(self, row: RelationalRow) -> MemoryItem:
        """Reconstruct a CORRELATION-ONLY shell (content is NOT stored relationally).

        The returned item carries the identity/lifecycle subset with an empty content
        sentinel — it is explicitly not a content round-trip (spec §5 contract 1).
        """
        c = row.cols
        prefix = str(c["namespace_prefix"])  # mu/{org}/{workspace}/{visibility}/{user}/{session}
        _, org, workspace, visibility, user_slot, session = prefix.split("/")
        vis = Visibility(visibility)
        ns = (
            Namespace.shared(org=org, workspace=workspace, session=session)
            if vis is Visibility.SHARED
            else Namespace(
                org=org,
                workspace=workspace,
                user=user_slot,
                session=session,
                visibility=vis,
            )
        )
        return MemoryItem(
            id=str(row.pk["memory_id"]),
            content=CONTENT_FREE_SHELL,
            kind=MemoryKind(c["kind"]),
            tier=MemoryTier(c["tier"]),
            state=MemoryState(c["state"]),
            namespace=ns,
            owner_id=str(c["owner_id"]),
            workspace_id=str(c["workspace_id"]),
            session_id=session,
            content_hash=str(c["content_hash"]),
            artifact_ref=c.get("artifact_ref"),
            provenance_id=str(c["provenance_id"]),
        )
