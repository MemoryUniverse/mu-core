"""Legacy-partition detection + honest tenancy recovery for the MTM/graph naming migration.

**Why this module exists.** Two landed fixes renamed physical partitions and left the OLD
partitions behind, orphaned (`docs/tracking/ARCHITECTURE-DELTAS.md` AD-2/AD-6/AD-8):

- **MTM (Qdrant), two prior generations:**
  1. ``mu_mtm__{workspace}__{visibility}__{dim}`` — pre-ADR-0026 vocabulary, no ``org`` segment
     at all (AD-2).
  2. ``mu_mtm__{org}__{workspace}__{visibility}__{dim}`` — the AD-2 fix attempt, briefly shipped;
     a raw ``"__"`` join of two caller-controlled segments, provably ambiguous (AD-6: ``org="acme
     __eu", workspace="ws"`` and ``org="acme", workspace="eu__ws"`` collide).
  3. **Current:** ``mu_mtm__{tenant_partition_digest(ns)}__{visibility}__{dim}``
     (:func:`mu_engine.storage.mappers.qdrant_mapper.collection_name`).
- **Graph (FalkorDB), one prior generation:** ``mu_g__{org}__{workspace}__shared`` /
  ``mu_g__{org}__{workspace}__{user}`` — the SAME raw-join ambiguity (AD-8). **Current:**
  ``mu_g__{digest}__shared`` / ``mu_g__{digest}__u_{user}``
  (:meth:`mu_engine.storage.adapters.falkor_ltm.FalkorLtmAdapter.graph_name_for`).

**The hard part.** ``tenant_partition_digest`` needs BOTH ``org`` and ``workspace``. Generation 1
of the MTM name never carried ``org`` at all, and generation 2's raw join makes it unsafe to
*parse* ``org``/``workspace`` back out of either old name — the same ambiguity that made the join
unsafe to build makes it unsafe to reverse. **Never guess it from the name.** The honest source is
data the point/node ALREADY carries, written by the mapper at ingest time and untouched by the
naming bug:

- MTM point payload: ``namespace_parts`` — the exact ``Namespace.parts()`` 5-tuple
  (:func:`mu_engine.storage.mappers.qdrant_mapper.QdrantMapper.to_store`) — or, if that flattened
  key is somehow absent, the ``namespace`` payload key (``Namespace.to_prefix()``).
- Graph node props: ``memory_json`` (the full ``MemoryItem``, ``:Memory`` nodes only —
  :func:`mu_engine.storage.mappers.graph_mapper.GraphMapper.to_store`) — or, for every node shape
  (``:Memory``/``:Artifact``/``:Entity``), the ``namespace`` prop, which is EITHER the full
  ``to_prefix()`` (6 segments) or the session-less user-scope prefix
  (:func:`mu_engine.storage.adapters.falkor_ltm._user_scope_prefix`, 5 segments) — both begin
  ``mu/{org}/{workspace}/{visibility}/{user_slot}[/{session}]`` and both are unambiguous to split
  on ``/`` (forbidden in every ``Namespace`` component,
  ``mu_contracts.domain.model.memory._FORBIDDEN_NS_CHARS``).

When NEITHER source is present or parses, :func:`resolve_tenancy_from_mtm_payload` /
:func:`resolve_tenancy_from_graph_props` return ``None`` — the caller's contract (spec'd by the
owning brief) is to SKIP that point/node and say so, never to fabricate a default ``org``. A wrong
guess here silently merges two tenants into one physical partition, which is precisely the class
of bug AD-2/AD-6/AD-8 exist to close.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from re import Pattern
from re import compile as re_compile
from typing import Any

from pydantic import BaseModel, ValidationError

from mu_engine.storage.domain.memory import MemoryItem
from mu_engine.storage.domain.namespace import Namespace, Visibility
from mu_engine.storage.mappers.qdrant_mapper import collection_name as _mtm_collection_name
from mu_engine.storage.mappers.tenancy import tenant_partition_digest

__all__ = [
    "TenancyKey",
    "discover_legacy_graph_names",
    "discover_legacy_mtm_collection_names",
    "is_legacy_graph_name",
    "is_legacy_mtm_collection_name",
    "resolve_tenancy_from_graph_props",
    "resolve_tenancy_from_mtm_payload",
]

_MTM_PREFIX = "mu_mtm__"
_GRAPH_PREFIX = "mu_g__"

# The CURRENT naming schemes (post AD-2/AD-6/AD-8) — anything with the legacy PREFIX that does
# NOT match one of these is a migration candidate. A digest is always exactly 16 lowercase hex
# characters (`tenant_partition_digest`'s truncated-SHA-256), so this cannot false-positive-match
# an old name whose middle segment is an arbitrary org/workspace slug UNLESS that slug happens to
# itself be 16 hex characters — an intentionally accepted, vanishingly unlikely residual (see the
# module docstring: the whole point of this file is to never rely on parsing that middle segment).
_CURRENT_MTM_RE: Pattern[str] = re_compile(r"^mu_mtm__[0-9a-f]{16}__(private|shared)__[1-9][0-9]*$")
_CURRENT_GRAPH_RE: Pattern[str] = re_compile(r"^mu_g__[0-9a-f]{16}__(shared|u_.+)$")

# A ``Namespace`` requires a non-empty ``session`` (``_no_separator_injection``) but neither
# target-name formula this module reproduces (``collection_name`` / ``graph_name_for``) ever
# reads ``.session`` — both are pure functions of ``(org, workspace, visibility, user)``. This
# placeholder exists ONLY to satisfy the pydantic model's arity; its value can never influence a
# computed target name. Recovered sources that lack a session (graph ``:Entity`` nodes, whose
# `namespace` prop is the session-less user-scope prprefix) are therefore not disadvantaged.
_UNUSED_SESSION_PLACEHOLDER = "_migration_probe"


class TenancyKey(BaseModel, frozen=True):
    """The four ``Namespace`` fields BOTH current naming formulas actually key on —
    ``org``/``workspace`` (via :func:`tenant_partition_digest`) plus ``visibility``/``user`` (the
    graph tier's PRIVATE suffix). Deliberately narrower than a full ``Namespace``: some legacy
    recovery sources (graph ``:Entity`` nodes) never carry a ``session``, and neither target-name
    formula needs one — modeling the narrower, always-recoverable shape keeps this type honest
    about what it actually guarantees.
    """

    org: str
    workspace: str
    visibility: Visibility
    user: str  # "*" when visibility is SHARED (CANONICAL §1 rule 4)

    def _probe_namespace(self) -> Namespace:
        """A ``Namespace`` good ONLY for feeding the two target-name formulas below — see
        ``_UNUSED_SESSION_PLACEHOLDER``'s docstring for why the fabricated session is safe."""
        return Namespace(
            org=self.org,
            workspace=self.workspace,
            user=self.user,
            session=_UNUSED_SESSION_PLACEHOLDER,
            visibility=self.visibility,
        )

    def target_mtm_collection_name(self, dim: int) -> str:
        """The CURRENT Qdrant collection name this tenancy resolves to — byte-identical to
        :func:`mu_engine.storage.mappers.qdrant_mapper.collection_name`, reused (not
        re-derived) so the two can never drift apart."""
        return _mtm_collection_name(self._probe_namespace(), dim)

    def target_graph_name(self) -> str:
        """The CURRENT FalkorDB graph name this tenancy resolves to — mirrors
        :meth:`mu_engine.storage.adapters.falkor_ltm.FalkorLtmAdapter.graph_name_for` exactly
        (that method takes no other input; kept here rather than called through an adapter
        instance so this module stays a pure, dependency-free planning helper). The shared,
        collision-resistant piece (the digest) is IMPORTED, never re-derived — only the two-line
        SHARED/PRIVATE branch is duplicated, and it is pinned by the unit tests in this package
        against the same fixtures the adapter itself is tested with.
        """
        digest = tenant_partition_digest(self._probe_namespace())
        if self.visibility is Visibility.SHARED:
            return f"mu_g__{digest}__shared"
        return f"mu_g__{digest}__u_{self.user}"


def is_legacy_mtm_collection_name(name: str) -> bool:
    """True for a Qdrant collection that carries the ``mu_mtm__`` partition prefix but does NOT
    match the current digest-keyed scheme — i.e. a migration candidate (either pre-AD-2 or the
    briefly-shipped AD-6 raw-join generation). False for anything else, INCLUDING collections with
    no ``mu_mtm__`` prefix at all (out of this migration's scope by construction)."""
    return name.startswith(_MTM_PREFIX) and _CURRENT_MTM_RE.match(name) is None


def is_legacy_graph_name(name: str) -> bool:
    """True for a FalkorDB graph that carries the ``mu_g__`` partition prefix but does NOT match
    the current digest-keyed scheme — the AD-8 raw-join generation. False for anything else,
    INCLUDING graphs with no ``mu_g__`` prefix (e.g. an unrelated probe/health-check graph)."""
    return name.startswith(_GRAPH_PREFIX) and _CURRENT_GRAPH_RE.match(name) is None


def discover_legacy_mtm_collection_names(names: Iterable[str]) -> list[str]:
    """Filter a live ``GET /collections`` listing (or any iterable of names) down to migration
    candidates, order-preserving. Pure — no I/O; the caller supplies the listing."""
    return [n for n in names if is_legacy_mtm_collection_name(n)]


def discover_legacy_graph_names(names: Iterable[str]) -> list[str]:
    """Filter a live ``GRAPH.LIST`` listing (or any iterable of names) down to migration
    candidates, order-preserving. Pure — no I/O; the caller supplies the listing."""
    return [n for n in names if is_legacy_graph_name(n)]


def _tenancy_from_namespace(ns: Namespace) -> TenancyKey:
    return TenancyKey(org=ns.org, workspace=ns.workspace, visibility=ns.visibility, user=ns.user)


def _tenancy_from_parts(parts: object) -> TenancyKey | None:
    """``payload["namespace_parts"]`` is ``list(Namespace.parts())`` — the exact 5-tuple
    ``(org, workspace, user, session, visibility)``. Reconstructing through
    ``Namespace.from_parts`` (rather than hand-building a ``TenancyKey``) re-runs the model's own
    validators (separator-injection guard, SHARED-requires-``user="*"``), so a payload that was
    never a legal ``Namespace`` to begin with is refused here rather than silently accepted."""
    if not (isinstance(parts, list | tuple) and len(parts) == 5):
        return None
    org, workspace, user, _session, visibility = parts
    if not all(isinstance(x, str) for x in (org, workspace, user, visibility)):
        return None
    try:
        ns = Namespace.from_parts((org, workspace, user, _session or "_", visibility))
    except (ValueError, TypeError):
        return None
    return _tenancy_from_namespace(ns)


def _tenancy_from_prefix_string(value: object) -> TenancyKey | None:
    """Parse EITHER shape a ``namespace``-keyed field can hold: the full ``to_prefix()``
    (``mu/{org}/{workspace}/{visibility}/{user_slot}/{session}``, 6 segments) or the session-less
    user-scope prefix graph ``:Entity`` nodes carry (5 segments, no trailing session). Splitting on
    ``/`` is unambiguous because ``/`` is in ``Namespace._FORBIDDEN_NS_CHARS`` — no component can
    contain it. Returns ``None`` (never a guess) on anything that doesn't fit that shape."""
    if not isinstance(value, str):
        return None
    segments = value.split("/")
    if len(segments) not in (5, 6) or segments[0] != "mu":
        return None
    _mu, org, workspace, visibility_raw, user = segments[:5]
    try:
        return TenancyKey(
            org=org, workspace=workspace, visibility=Visibility(visibility_raw), user=user
        )
    except (ValueError, TypeError):
        return None


def resolve_tenancy_from_mtm_payload(payload: Mapping[str, Any]) -> TenancyKey | None:
    """Recover the true tenancy for one legacy MTM point. Tries the most direct source first
    (``namespace_parts``, the flattened ``Namespace.parts()`` 5-tuple) and falls back to parsing
    the ``namespace`` (``to_prefix()``) payload key. ``None`` when neither is present or parses —
    the caller MUST skip this point, never fabricate a target."""
    resolved = _tenancy_from_parts(payload.get("namespace_parts"))
    if resolved is not None:
        return resolved
    return _tenancy_from_prefix_string(payload.get("namespace"))


def resolve_tenancy_from_graph_props(props: Mapping[str, Any]) -> TenancyKey | None:
    """Recover the true tenancy for one legacy graph node. Tries the most direct source first
    (``memory_json``, present on ``:Memory`` nodes only — the full structured ``Namespace``, no
    string-splitting at all) and falls back to parsing the ``namespace`` prop every node shape
    (``:Memory``/``:Artifact``/``:Entity``) carries. ``None`` when neither is present or parses —
    the caller MUST skip this node, never fabricate a target."""
    blob = props.get("memory_json")
    if isinstance(blob, str):
        try:
            item = MemoryItem.model_validate_json(blob)
        except ValidationError:
            item = None
        if item is not None:
            return _tenancy_from_namespace(item.namespace)
    return _tenancy_from_prefix_string(props.get("namespace"))
