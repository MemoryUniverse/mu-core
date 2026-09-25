"""Ingest / promotion knobs — a central, injectable Settings model (DEV-STANDARDS rule 3).

NO hardcode at a call-site: the deterministic-promotion thresholds and the STM TTL live here as
pydantic fields (defaults mirror engine-core-spec §12 ``SalienceSettings``), injected into
``IngestService``/stages and overridable from the environment via the owning ``Settings`` tree.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from mu_engine.platform.observability import CredentialPolicy

__all__ = ["IngestSettings"]


class IngestSettings(BaseModel, frozen=True):
    """Deterministic STM->MTM promotion gates (engine-core-spec §12; ``service.py:335`` rule).

    The mem0 diff-loop / salience-weighted promotion (§3/§4) is a later slice; this slice promotes
    on the two deterministic, LLM-free signals available at capture: an explicit ``promote`` flag
    and the importance threshold. (The ``mention_count`` arm needs the content-hash-indexed STM
    lookup and is wired with that index.)
    """

    importance_promote: float = Field(default=0.6, ge=0.0, le=1.0)
    mention_promote: int = Field(default=2, ge=1)
    stm_ttl_s: int = Field(default=3600, ge=1)

    # D4 write-time STM dedup (CONFIG-AND-DATA-FIX-PLAN.md PART 2 D4; conformance D-8): the
    # content-hash-indexed lookup this class's own docstring above flagged as needed for the
    # ``mention_count`` arm is now built (``StmTierRepository`` adapters, ``storage/adapters/
    # {redis,valkey,memory}_stm.py``) — a namespace-prefixed ``content_hash -> memory_id`` index
    # maintained on every ``put()`` so identical content lands ONCE, not twice (the empirically
    # observed ``ada_coffee`` written-twice defect, DATA-QUALITY-ASSESSMENT.md §3.1/#5). Env
    # override: ``MU_INGEST__STM_DEDUP=false`` reverts to the old always-fork-a-new-entry
    # behavior (DEV-STANDARDS rule 3 — a toggle, never a silent unconditional change).
    stm_dedup: bool = Field(default=True)

    # AD-294 — the owner's policy for a credential a user dictates into captured text, applied
    # once, centrally, in `IngestService.remember` (the one method every ingest path in this tree
    # already funnels through — `LocalMemory.add`, `SurfaceFacade.add`, mu-server's
    # `SharedMemoryService.add` alike, same precedent as this class's own `_ensure_turn_seq`
    # centralisation). Reuses the AD-267 credential-shape catalog (`platform/observability.py`),
    # never a second matcher. Default REDACT — the sane default per AD-294: a user saying "my key
    # is X, remember I use Anthropic" keeps the useful half. Env: `MU_INGEST__CREDENTIAL_POLICY`
    # (`redact` | `refuse` | `mark`).
    credential_policy: CredentialPolicy = Field(default=CredentialPolicy.REDACT)
