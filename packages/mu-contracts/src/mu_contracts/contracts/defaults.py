"""Wire/ergonomic default limits — the ``limit=10 / limit=50`` bug-class fix (CONFIG-AND-DATA-FIX-
PLAN.md PART 1 §1.1 Group D, C0/C4).

**The bug this closes (`02fbed9`).** Two independent hardcoded ``10``s collided:
``RecallSettings.recency_floor_limit=10`` (an intelligence knob, ``mu_engine.services.recall.dto``)
happened to equal ``RecallQuery.limit=10`` (a wire/ergonomic default). Nothing *linked* those two
literals — they were free-floating and only equal by coincidence, so the STM recency floor silently
consumed the entire result budget on any session with >= ``limit`` STM items. The fix-plan's Group D
inventory found this SAME "bare literal, no shared source" pattern recurring at 8 sites (contracts
wire defaults, ``mu-local`` facade signatures, both SDKs' fallback defaults) — none deriving from a
single named constant.

**Why these two constants, and why they live HERE, not in `EngineSettings`.** These are wire/
ergonomic defaults (the ``limit`` a caller gets if it sends none) — NOT intelligence knobs (an
intelligence knob tunes HOW the engine reasons, e.g. `RecallSettings.recency_floor_limit`, which
stays in `mu_engine.config.EngineSettings`; see that module's docstring for the boundary). A wire
default belongs beside the wire contracts themselves (`mu_contracts.contracts.requests`), the one
package both `mu-engine-server` and every SDK are permitted to import
(`contracts-imports-nothing-in-project`, `.importlinter`) — so ONE named source can be referenced
from `mu-contracts` (`RecallRequest.limit`/`ConsolidateRequest.limit` defaults), `mu-local`
(`LocalMemory` facade signatures), and both SDKs (`mu-sdk-python` transport fallbacks, `mu-sdk-js`
`defaultConsolidateLimit`), closing every one of the 8 Group-D sites onto ONE literal.

**Staging note (plan §1.2 C0 vs C4).** This module lands the named constants now (C0); wiring the
8 call sites onto them (replacing each bare `10`/`50` with a reference to ``RecallDefaults``) is
Group D / stage **C4** — deliberately out of this slice's scope (C0 is additive-only, no existing
call site is touched).
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "DEFAULT_CONSOLIDATE_LIMIT",
    "DEFAULT_RECALL_LIMIT",
    "RecallDefaults",
]

# The one `RecallQuery`/`RecallRequest`/`ContextWindowRequest` result-count default (dto.py:68,
# requests.py:223,279 pre-fix; `mu-local/local_memory.py` facade signatures; SDK transport
# fallbacks). Deliberately the SAME value `RecallSettings.recency_floor_limit` used to collide
# with — kept equal is fine now that `floor_protect_limit` (not this constant) bounds the STM
# floor; this constant governs ONLY the wire-level "how many results" default.
DEFAULT_RECALL_LIMIT: Final[int] = 10

# The `ConsolidateRequest`/`mu-local` `consolidate()` sweep-size default (requests.py:315;
# `local_memory.py:202`; SDK `defaultConsolidateLimit`).
DEFAULT_CONSOLIDATE_LIMIT: Final[int] = 50


class RecallDefaults:
    """Namespace grouping the wire/ergonomic default limits (never instantiated).

    A plain class, not a pydantic model: these are compile-time constants referenced by both
    contracts-side pydantic ``Field(default=...)`` declarations and plain call-site defaults
    (``mu-local`` facade method signatures, SDK fallback dict lookups) — a pydantic value object
    would force every one of those callers to construct/import an instance just to read a scalar.
    """

    RECALL_LIMIT: Final[int] = DEFAULT_RECALL_LIMIT
    CONSOLIDATE_LIMIT: Final[int] = DEFAULT_CONSOLIDATE_LIMIT
