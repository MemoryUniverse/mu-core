"""The AD-2/AD-6 (MTM/Qdrant) + AD-8 (graph/FalkorDB) legacy-partition-naming migration.

See ``naming.py`` for the tenancy-recovery problem this package solves and why it must never
guess; ``planning.py`` for the pure per-item routing logic (unit-tested with fakes);
``mtm_migration.py`` / ``graph_migration.py`` for the blue-green I/O over the real stores; and
``runner.py`` for the dry-run-by-default CLI (``python -m mu_engine.storage.migrations``).
"""

from __future__ import annotations

from mu_engine.storage.migrations.graph_migration import (
    GraphMigrationResult,
    migrate_all_legacy_graphs,
    migrate_graph,
)
from mu_engine.storage.migrations.mtm_migration import (
    MtmCollectionMigrationResult,
    migrate_all_legacy_mtm_collections,
    migrate_mtm_collection,
)
from mu_engine.storage.migrations.naming import TenancyKey
from mu_engine.storage.migrations.planning import (
    GraphMigrationPlan,
    MtmMigrationPlan,
    plan_graph_migration,
    plan_mtm_migration,
)

__all__ = [
    "GraphMigrationPlan",
    "GraphMigrationResult",
    "MtmCollectionMigrationResult",
    "MtmMigrationPlan",
    "TenancyKey",
    "migrate_all_legacy_graphs",
    "migrate_all_legacy_mtm_collections",
    "migrate_graph",
    "migrate_mtm_collection",
    "plan_graph_migration",
    "plan_mtm_migration",
]
