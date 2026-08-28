"""Wire schemas (CANONICAL-CONTRACTS.md §1.3 contracts/; PACKAGING-v2 §1.3).

Home of the canonical per-verb return DTO family (sdk-engine-server-design.md §2.5 "Return DTOs";
build-plan Stage B item 3; ``SDK-BUILD-DECISIONS.md`` Decision B) — ``MemoryResponse``
(:mod:`mu_contracts.contracts.memory`), ``RecallResult``/``RecallItemView``/``RecallChannels``/
``RecallMode`` (:mod:`mu_contracts.contracts.recall`), and ``MemoryWriteResult``/``ContextView``/
``ConsolidateView`` (:mod:`mu_contracts.contracts.views`) — the ONE home both ``mu-local``
(embedded ``LocalMemory``/``SurfaceFacade``) and ``mu-sdk-python`` (``MemoryClient``) import these
from, per each submodule's docstring for the exact re-home provenance + migration status.

The machine-readable **OpenAPI + JSON-Schema** (PACKAGING-v2 §1.3) is NO LONGER SCAFFOLD: it is
generated from these same pydantic models by :mod:`mu_contracts.contracts.wire` and committed as
``contracts/schema/openapi.json`` + ``contracts/schema/wire-models.schema.json`` (shipped in the
wheel, so an SDK generator or a conformant-server author can read it off an installed
``mu-contracts``). Regenerate with ``uv run python scripts/wire_contract.py emit``; three gates
keep it honest — ``packages/mu-contracts/tests/test_wire_contract.py`` (the artifacts match the
models), ``packages/mu-contracts/tests/test_sdk_wire_parity.py`` (neither SDK has drifted from
them), and ``packages/mu-engine-server/tests/test_wire_contract_routes.py`` (the real app matches
the declared operations, and no route escapes the contract unlisted).

Remaining SCAFFOLD: rest_schema, mcp_schema, the event-catalog re-export, ModelSettings
(re-exported from config, CANONICAL §7.2), and error envelopes.
"""

from __future__ import annotations

from mu_contracts.contracts.defaults import (
    DEFAULT_CONSOLIDATE_LIMIT,
    DEFAULT_RECALL_LIMIT,
    RecallDefaults,
)
from mu_contracts.contracts.live_context import (
    SECTION_ORDER,
    ContextSlab,
    LiveSessionContext,
    PrivateSlice,
    Section,
    SharedZone,
    ToolTurnState,
    content_hash_of,
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
    "SECTION_ORDER",
    "AddRequest",
    "ConsolidateRequest",
    "ConsolidateView",
    "ContentType",
    "ContextSlab",
    "ContextView",
    "ContextWindowRequest",
    "GetRequest",
    "LiveSessionContext",
    "MemoryResponse",
    "MemoryTierLiteral",
    "MemoryWriteResult",
    "PolarityLiteral",
    "PrivateSlice",
    "RecallChannels",
    "RecallDefaults",
    "RecallItemView",
    "RecallMode",
    "RecallRequest",
    "RecallResult",
    "Section",
    "SharedZone",
    "ToolTurnState",
    "content_hash_of",
]
