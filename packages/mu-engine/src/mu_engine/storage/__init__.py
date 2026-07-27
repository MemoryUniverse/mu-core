"""storage/ — the pluggable store layer behind the repository ports (DEV-STANDARDS rule 5).

Built from ``storage-schema-rowmapper-spec.md`` + ``storage-pluggable-spec.md`` (authority:
``CANONICAL-CONTRACTS.md``). Contents:

- ``domain/``   the value objects the layer binds to (Namespace/MemoryItem + the 5 DTOs);
                re-home to ``mu-contracts`` when its domain phase lands.
- ``ports``     the async repository Protocols + StoreModels + RowMapper + ConflictEdgeReader.
- ``errors``    the typed error hierarchy + ``DegradeReason`` (fail-loud, no silent fallback).
- ``relational/schema`` the full SQLAlchemy 2.x schema (all tables) + index catalog.
- ``relational/migrations`` the Alembic forward+reversible migration.
- ``mappers/``  one ``RowMapper`` per substrate (the ONLY field-projection seam).
- ``adapters/`` the real async store adapters (Redis/Qdrant/FalkorDB/SQLAlchemy).
- ``registry`` + ``factories`` the ``StoreRegistry`` selection mechanism (fail-loud).

Every adapter is fully async and prefixes physical keys with ``Namespace.to_prefix()``
(CANONICAL §1 rule 5, the tenancy guarantee). Ported per CODE-ADOPTION-METHODOLOGY.
"""

from __future__ import annotations

from mu_engine.storage.errors import (
    DegradeReason,
    InvalidBackendError,
    MandatoryBackendMissingError,
    StorageError,
    TierRepositoryUnavailableError,
    UnknownBackendError,
    VectorNotFilterableError,
)
from mu_engine.storage.ports import (
    ConflictEdgeReader,
    ControlPlaneRepository,
    GraphStorePort,
    LtmTierRepository,
    MtmTierRepository,
    RowMapper,
    StmTierRepository,
)
from mu_engine.storage.registry import StoreRegistry, assert_mandatory_roles

__all__ = [
    "ConflictEdgeReader",
    "ControlPlaneRepository",
    "DegradeReason",
    "GraphStorePort",
    "InvalidBackendError",
    "LtmTierRepository",
    "MandatoryBackendMissingError",
    "MtmTierRepository",
    "RowMapper",
    "StmTierRepository",
    "StorageError",
    "StoreRegistry",
    "TierRepositoryUnavailableError",
    "UnknownBackendError",
    "VectorNotFilterableError",
    "assert_mandatory_roles",
]
