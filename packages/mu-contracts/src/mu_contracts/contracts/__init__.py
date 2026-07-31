"""Wire schemas (CANONICAL-CONTRACTS.md §1.3 contracts/; PACKAGING-v2 §1.3).

Home of the canonical per-verb return DTO family (sdk-engine-server-design.md §2.5 "Return DTOs";
build-plan Stage B item 3; ``SDK-BUILD-DECISIONS.md`` Decision B) — ``MemoryResponse``
(:mod:`mu_contracts.contracts.memory`), ``RecallResult``/``RecallItemView``/``RecallChannels``/
``RecallMode`` (:mod:`mu_contracts.contracts.recall`), and ``MemoryWriteResult``/``ContextView``/
``ConsolidateView`` (:mod:`mu_contracts.contracts.views`) — the ONE home both ``mu-local``
(embedded ``LocalMemory``/``SurfaceFacade``) and ``mu-sdk-python`` (``MemoryClient``) import these
from, per each submodule's docstring for the exact re-home provenance + migration status.

Remaining SCAFFOLD: rest_schema, mcp_schema, the event-catalog re-export, ModelSettings
(re-exported from config, CANONICAL §7.2), error envelopes, and the machine-readable
OpenAPI + JSON-Schema the TS SDK and community/conformant servers generate from.
"""

from __future__ import annotations

from mu_contracts.contracts.defaults import (
    DEFAULT_CONSOLIDATE_LIMIT,
    DEFAULT_RECALL_LIMIT,
    RecallDefaults,
)
from mu_contracts.contracts.memory import (
    ContentType,
    MemoryResponse,
    MemoryTierLiteral,
    PolarityLiteral,
)
from mu_contracts.contracts.recall import RecallChannels, RecallItemView, RecallMode, RecallResult
from mu_contracts.contracts.requests import (
    AddRequest,
    ConsolidateRequest,
    ContextWindowRequest,
    GetRequest,
    RecallRequest,
)
from mu_contracts.contracts.views import ConsolidateView, ContextView, MemoryWriteResult

__all__ = [
    "DEFAULT_CONSOLIDATE_LIMIT",
    "DEFAULT_RECALL_LIMIT",
    "AddRequest",
    "ConsolidateRequest",
    "ConsolidateView",
    "ContentType",
    "ContextView",
    "ContextWindowRequest",
    "GetRequest",
    "MemoryResponse",
    "MemoryTierLiteral",
    "MemoryWriteResult",
    "PolarityLiteral",
    "RecallChannels",
    "RecallDefaults",
    "RecallItemView",
    "RecallMode",
    "RecallRequest",
    "RecallResult",
]
