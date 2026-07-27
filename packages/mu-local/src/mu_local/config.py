"""``StorageSettings`` / ``BackendChoice`` — the pluggable-backend selector for ``mu-local``.

Adopted from ``research-pluggable-infrastructure.md §3.2`` and the mem0 ``{provider, config}``
config shape (PORT of ``other_repos/mem0/mem0/configs/base.py:30-58`` — a provider key + a
per-backend config dict). One :class:`BackendChoice` per storage ROLE; the composition root
(:mod:`mu_local.composition`) resolves each to a concrete adapter, filling connection knobs from
the central ``Settings`` tree when the choice's ``config`` is empty (DEV-STANDARDS rule 3: no
hardcoded host/port — everything flows from the single env boundary).

This VO lives in ``mu-local`` (not ``mu-contracts``) for now: the spec files it under
``§Contract-changes 2`` as a PROPOSED additive contract (``StorageSettings``/``BackendChoice``/
``STORE_REGISTRY`` become canonical config surface), not yet pinned into CANONICAL. mu-local owns
the ONE default set until the owner pins it (spec §2.2, APPLY-PLAN B-4).

GRAPH IS MANDATORY (CANONICAL storage invariant, spec §3.1): the ``graph`` role must bind a real
graph engine; ``none``/``sqlfold`` are refused at build (``storage.registry`` mandatory-roles).

Phase-0 reality (honest, DEV-STANDARDS "no silent stubs"): the ``STORE_REGISTRY`` shipped so far
binds the mu-dev-container backends (``redis`` KV, ``qdrant`` vector, ``falkordb`` graph,
``sqlite`` or ``postgres`` relational) + the offline ``minilm_local`` embedder. The zero-infra
embedded floor named in the spec (in-proc KV / FAISS / embedded Kùzu) is NOT built yet — selecting
one is a NAMED fail-loud ``BackendUnavailableError`` at the composition root, never a silent
fallback. The default below therefore binds the backends that EXIST and resolves their host ports
from ``Settings`` (``.env.test`` -> the live mu-dev-* stack).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["BackendChoice", "ObservabilitySettings", "StorageSettings"]


class ObservabilitySettings(BaseModel):
    """Which of the three content-free sinks the embedded LOCAL engine wires (DEV-STANDARDS rule 4).

    Config-sourced (never hardcoded in the container): the composition root reads this and builds
    the real ``Tracer``/``MetricSink``/``AuditLog`` via ``mu_engine.platform.observability`` —
    tracer + metrics + a structured-log audit are ON by default so a real embedded run emits spans,
    latency/error metrics and content-free audit rows on every meaningful op. Flip any off (e.g. in
    a bare unit context) without touching wiring. Mirrors the SHARED-plane ``PlatformSelectors``
    observability flags; folds into ``settings.observability`` when that subtree lands.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    otel_enabled: bool = True
    metrics_enabled: bool = True
    audit_enabled: bool = True


class BackendChoice(BaseModel):
    """A ``{backend, config}`` selection for one storage role (mem0 ``configs/base.py:30-58``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str = Field(min_length=1)  # STORE_REGISTRY key: "redis"|"qdrant"|"falkordb"|"sqlite"…
    config: dict[str, Any] = Field(default_factory=dict)  # per-backend knobs (url/dsn/host/port…)


class StorageSettings(BaseModel):
    """One :class:`BackendChoice` per storage ROLE (research-pluggable-infrastructure §3.2).

    The default binds the backends the phase-0 registry actually ships; empty ``config`` dicts are
    filled from the central ``Settings`` tree at the composition root. ``llm=None`` ⇒ heuristic
    mode (no LLM key, Azure PARKED) — every LLM-dependent verb then refuses loudly (spec §7, T7).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    relational: BackendChoice = BackendChoice(backend="sqlite")  # control-plane + history
    kv: BackendChoice = BackendChoice(backend="redis")  # STM (shared with the pipeline ledger)
    vector: BackendChoice = BackendChoice(backend="qdrant")  # MTM dense
    graph: BackendChoice = BackendChoice(backend="falkordb")  # LTM — MANDATORY graph engine
    embedding: BackendChoice = BackendChoice(backend="minilm_local")  # REAL offline MiniLM
    llm: BackendChoice | None = None  # None ⇒ heuristic mode (no synthesis)
