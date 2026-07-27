"""Store adapters — thin connection+query wrappers behind the repository ports (rule 5).

Each adapter delegates ALL field-projection to its ``RowMapper`` (spec Contract-change 5)
and prefixes every physical key with ``Namespace.to_prefix()`` (CANONICAL §1 rule 5).
Fully async; ported per CODE-ADOPTION-METHODOLOGY (cite file:line in each module).
"""

from __future__ import annotations

from mu_engine.storage.adapters.falkor_ltm import FalkorLtmAdapter
from mu_engine.storage.adapters.qdrant_mtm import QdrantMtmAdapter
from mu_engine.storage.adapters.redis_stm import RedisStmAdapter
from mu_engine.storage.adapters.relational_control import RelationalControlPlaneAdapter

__all__ = [
    "FalkorLtmAdapter",
    "QdrantMtmAdapter",
    "RedisStmAdapter",
    "RelationalControlPlaneAdapter",
]
