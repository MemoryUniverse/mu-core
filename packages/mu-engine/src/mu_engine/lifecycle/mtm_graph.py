"""RESERVED seam — MTM working-graph tier-tagged partition (ADR 0036; spec §11 DECISION 11).

**STATUS: RESERVED, default OFF.** ``MtmWorkingGraphSettings.enabled`` defaults to ``False`` and
this build ships exactly one state: disabled. This module wires the settings surface and a
no-op ``MtmWorkingGraphService`` stub only — it does **not** implement GraphRAG fusion (vector
hits unioned with graph-neighbor expansion in the MTM recall channel), the MTM-tier
tagging/write path, or entity resolution. Those land in a future slice once the RAM sizing
estimate below is validated with a real measurement (spec §18 R2: "a sizing estimate to
validate at ``enabled=true`` time ... not a substitute for one").

Authority (read in this order):

- ``docs/decisions/0036-mtm-hybrid-org-graph-partition-graphrag.md`` — Amendment section
  (2026-07-28, X2): the corrected per-visibility grain. PRIVATE already gets a distinct
  physical FalkorDB graph per ``(org, workspace, user)`` — this is an EXISTING tenancy
  primitive, not a new one this seam would mint. SHARED stays one per-org graph
  (CANONICAL §1 rule 6 / SCALE-GEO §2 — collection/graph grain pinned at ``org``, for the
  SHARED plane only).
- ``docs/superpowers/design/memory-lifecycle-manager-spec.md`` §11 (design rationale) and
  §16 (the exact field defaults reproduced on ``MtmWorkingGraphSettings`` below).
- This plan's server-seams section / DECISION 10's RESERVED-state convention (declared, not
  built, so the FSM/seam is complete and a future driver has one obvious place to land —
  the same pattern already used for ``ltm_demotion_enabled`` and the Φ(C,T) quarantine hook).

``FalkorLtmAdapter.graph_name_for`` (``mu_engine/storage/adapters/falkor_ltm.py:101-113``,
EXISTS) already resolves every namespace to its physical graph — ``mu_g__{org}__{workspace}__
{user}`` for PRIVATE, ``mu_g__{org}__{workspace}__shared`` for SHARED. A future real
implementation of this seam would tag MTM-tier rows inside whichever graph that call already
selects; this build does not call it, tag anything, or open any graph connection — there is no
tier-tagging write path here to review.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MtmWorkingGraphService", "MtmWorkingGraphSettings"]

#: Allowed values for :attr:`MtmWorkingGraphSettings.expansion` (spec §16): ``"none"`` — no
#: MTM-tier graph expansion at all (the only value this build's no-op service honors);
#: ``"mtm_seed"`` — seed-node expansion from ``entity_uids`` (mtm-retrieval §2.1 join key),
#: RESERVED for the future real implementation.
ExpansionMode = Literal["none", "mtm_seed"]


class MtmWorkingGraphSettings(BaseModel):
    """RESERVED — MTM-tier working-graph knobs (spec §11/§16; ADR 0036).

    Every field here only takes effect once a future task flips ``enabled=True`` and lands the
    real tier-tagging write path + GraphRAG fusion read path this settings block gates. Ships
    ``enabled=False`` so this slice's behavior is identical to mtm-retrieval's Stage-0/1 (no MTM
    graph structure of its own) — additive-by-default-off, per spec §11.

    This is the ONE definition of this class (integrate-phase fix, S0-07/S0-10 both independently
    drafted a ``MtmWorkingGraphSettings`` — this module's copy wins: it pairs with
    :class:`MtmWorkingGraphService` and types ``expansion`` as the closed :data:`ExpansionMode`
    Literal rather than a bare ``str``. ``mu_engine.lifecycle.settings.LifecycleSettings`` imports
    this class rather than redefining it (DRY, DEV-STANDARDS rule 6) — see that module's
    ``mtm_working_graph`` field.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    #: Bound on MTM-tier nodes (spec §11 RAM guard, §18 R2). PRIVATE: rows inside the user's
    #: ALREADY-open per-user graph (``falkor_ltm.py:113``). SHARED: user-tagged rows inside the
    #: per-org graph (``falkor_ltm.py:112``). Never enforced by this build — no writer exists yet.
    max_nodes_per_user: int = Field(default=5_000, ge=1)
    expansion: ExpansionMode = "none"
    #: Graph-neighbor hop count a future GraphRAG-fusion read would traverse from a seed node.
    hops: int = Field(default=1, ge=0)
    #: Top-N MTM vector hits a future implementation would seed graph expansion from.
    seed_top_n: int = Field(default=5, ge=0)


class MtmWorkingGraphService:
    """RESERVED seam — MTM working-graph GraphRAG fusion (ADR 0036, spec §11).

    The **only state this build ships is disabled** (``settings.enabled is False``, the
    :class:`MtmWorkingGraphSettings` default). In that state every public method is a
    documented no-op: no graph is opened, no GraphRAG fusion runs, no MTM-tier tag is ever
    written. This class exists purely as the stable injection point a future
    ``RecallService``/``LifecycleManager`` slice will call into once the real driver lands —
    the "seam, not built" the owner's RESERVED directive calls for.

    Deliberately not implemented here, and out of scope for this task:

    - GraphRAG fusion (union of MTM vector hits with MTM-tier graph neighbors) in the recall
      MTM channel (spec §11 "Resolution" paragraph).
    - The MTM-tier tagging write path (tagging rows in whichever graph
      ``FalkorLtmAdapter.graph_name_for`` already selects for a namespace).
    - Distillation (MTM-tier -> LTM-tier re-tag on promotion).
    """

    def __init__(self, settings: MtmWorkingGraphSettings | None = None) -> None:
        self._settings = settings if settings is not None else MtmWorkingGraphSettings()

    @property
    def settings(self) -> MtmWorkingGraphSettings:
        """The settings this instance was constructed with (read-only, ``frozen=True``)."""
        return self._settings

    @property
    def is_active(self) -> bool:
        """Whether the working-graph seam is live. Always ``False`` this build (default-off,
        spec §11) — no code path in this repository flips it, since the driver does not exist.
        """
        return self._settings.enabled

    async def expand(self, seed_entity_uids: Sequence[str]) -> list[str]:
        """GraphRAG-fusion seam (spec §11 "Resolution" paragraph).

        A future real implementation would union ``seed_entity_uids`` (the MTM vector hits'
        resolved entities) with their MTM-tier graph neighbors, up to
        :attr:`MtmWorkingGraphSettings.hops` hops, capped by
        :attr:`MtmWorkingGraphSettings.seed_top_n`. This build always returns an empty list —
        no expansion, no graph read — because ``enabled=False`` is the only state shipped;
        callers get exactly mtm-retrieval's Stage-0/1 behavior (LTM-seed-only, no MTM graph
        structure), unaffected by this seam's presence.

        Raises:
            NotImplementedError: if ever constructed with ``enabled=True`` — flipping the
                default requires landing the real tier-tagging write path first (out of scope
                for this RESERVED seam); this build has no code path that sets ``enabled=True``.
        """
        if not self._settings.enabled:
            return []
        raise NotImplementedError(
            "MtmWorkingGraphService.enabled=True has no driver in this build "
            "(ADR 0036 RESERVED seam, spec §11) — the tier-tagging write path and GraphRAG "
            "fusion read path are a future slice, not this one."
        )
