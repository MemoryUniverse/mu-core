"""Dry-run-by-default CLI for the AD-2/AD-6/AD-8 legacy-partition-naming migration.

    python -m mu_engine.storage.migrations --qdrant-url http://127.0.0.1:16333 \\
        --falkor-host 127.0.0.1 --falkor-port 16380

Reports what would be migrated; writes and deletes NOTHING unless ``--apply`` is passed. Never
touches the dense embedder (dense vectors are read back and re-written verbatim — §1.7 "no
re-embed") and never guesses a tenancy it cannot recover (`naming.py`).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import structlog
from falkordb.asyncio import FalkorDB
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from mu_engine.storage.migrations.graph_migration import migrate_all_legacy_graphs
from mu_engine.storage.migrations.mtm_migration import migrate_all_legacy_mtm_collections

__all__ = ["main", "run"]

_log = structlog.get_logger("mu_engine.storage.migrations.runner")


class RunnerArgs(BaseModel, frozen=True):
    qdrant_url: str
    falkor_host: str
    falkor_port: int
    apply: bool


def _parse_args(argv: list[str] | None) -> RunnerArgs:
    parser = argparse.ArgumentParser(
        prog="mu_engine.storage.migrations",
        description=(
            "Blue-green migration off legacy-named MTM/graph partitions (AD-2/AD-6/AD-8). "
            "Dry-run by default."
        ),
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:16333")
    parser.add_argument("--falkor-host", default="127.0.0.1")
    parser.add_argument("--falkor-port", type=int, default=16380)
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually write/upsert/drop. Omit for a dry-run report only.",
    )
    ns = parser.parse_args(argv)
    return RunnerArgs(
        qdrant_url=ns.qdrant_url,
        falkor_host=ns.falkor_host,
        falkor_port=ns.falkor_port,
        apply=ns.apply,
    )


async def run(args: RunnerArgs) -> dict[str, object]:
    qdrant = AsyncQdrantClient(url=args.qdrant_url)
    falkor = FalkorDB(host=args.falkor_host, port=args.falkor_port)
    try:
        mtm_results = await migrate_all_legacy_mtm_collections(qdrant, apply=args.apply)
        graph_results = await migrate_all_legacy_graphs(falkor, apply=args.apply)
    finally:
        await qdrant.close()

    report = {
        "apply": args.apply,
        "mtm": [r.model_dump(mode="json") for r in mtm_results],
        "graph": [r.model_dump(mode="json") for r in graph_results],
        "summary": {
            "legacy_mtm_collections_found": len(mtm_results),
            "legacy_graphs_found": len(graph_results),
            "mtm_points_unresolved": sum(r.unresolved for r in mtm_results),
            "graph_nodes_unresolved": sum(r.nodes_unresolved for r in graph_results),
        },
    }
    _log.info("migration_runner.done", **report["summary"])  # type: ignore[arg-type]
    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = asyncio.run(run(args))
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
