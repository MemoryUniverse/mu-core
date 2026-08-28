"""The golden-artifact gate for the machine-readable wire contract.

``api-mcp-surface-spec.md`` §10 / ``api-sdk-mcp-surface-design.md`` §921 both require a checked-in
OpenAPI snapshot whose test "fails the build if the generated schema drifts". This is that test.
It is deliberately NOT a smoke test: it re-generates from the live pydantic models and compares
byte-for-byte against the committed artifacts, so ANY change to a wire model — a renamed field, a
new optional, a narrowed type — turns this RED until the author regenerates deliberately with
``uv run python scripts/wire_contract.py emit``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mu_contracts.contracts.wire import (
    OPENAPI_VERSION,
    UNCONTRACTED_ROUTES,
    WIRE_MODELS,
    WIRE_OPERATIONS,
    build_json_schema,
    build_openapi,
)

pytestmark = pytest.mark.unit

_SCHEMA_DIR = Path(__file__).resolve().parents[1] / "src" / "mu_contracts" / "contracts" / "schema"
OPENAPI_PATH = _SCHEMA_DIR / "openapi.json"
JSON_SCHEMA_PATH = _SCHEMA_DIR / "wire-models.schema.json"


# Byte-identical to scripts/wire_contract.py::_render. Duplicated (three lines) rather than
# imported, because importing a top-level script from a package test would make the package's test
# suite depend on the repo layout above it.
def _render(doc: object) -> str:
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_openapi_artifact_matches_the_models() -> None:
    assert OPENAPI_PATH.exists(), (
        f"{OPENAPI_PATH} is missing — the machine-readable contract PACKAGING-v2 §1.3 requires "
        "is not committed. Run: uv run python scripts/wire_contract.py emit"
    )
    assert OPENAPI_PATH.read_text(encoding="utf-8") == _render(build_openapi()), (
        "The committed OpenAPI is STALE: a wire model changed without regenerating. Run: "
        "uv run python scripts/wire_contract.py emit"
    )


def test_json_schema_artifact_matches_the_models() -> None:
    assert (
        JSON_SCHEMA_PATH.exists()
    ), f"{JSON_SCHEMA_PATH} is missing. Run: uv run python scripts/wire_contract.py emit"
    assert JSON_SCHEMA_PATH.read_text(encoding="utf-8") == _render(
        build_json_schema()
    ), "The committed JSON-Schema is STALE. Run: uv run python scripts/wire_contract.py emit"


def test_every_operation_names_a_declared_model() -> None:
    """A contract that $refs a schema it does not define is worse than no contract."""
    for op in WIRE_OPERATIONS:
        assert op.response_model in WIRE_MODELS, f"{op.operation_id}: unknown response model"
        if op.request_model is not None:
            assert op.request_model in WIRE_MODELS, f"{op.operation_id}: unknown request model"


def test_every_ref_in_the_openapi_resolves() -> None:
    doc = build_openapi()
    defined = set(doc["components"]["schemas"])
    dangling: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                key = ref.rsplit("/", 1)[-1]
                if key not in defined:
                    dangling.append(ref)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    assert not dangling, f"dangling $ref(s) in the emitted OpenAPI: {sorted(set(dangling))}"


def test_openapi_is_3_1_and_declares_the_2020_12_dialect() -> None:
    """3.1 IS JSON-Schema 2020-12 — the property that lets ONE definition family serve both
    artifacts. If this ever regresses to 3.0 the two artifacts stop being the same schemas."""
    doc = build_openapi()
    assert doc["openapi"] == OPENAPI_VERSION == "3.1.0"
    assert doc["jsonSchemaDialect"] == build_json_schema()["$schema"]


def test_uncontracted_routes_are_disjoint_from_contracted_ones() -> None:
    contracted = {(op.method, op.path) for op in WIRE_OPERATIONS}
    for route in UNCONTRACTED_ROUTES:
        assert (
            (route.method, route.path) not in contracted
        ), f"{route.method} {route.path} is listed as BOTH contracted and uncontracted."


def test_uncontracted_routes_are_visible_in_the_emitted_document() -> None:
    """The gap must be readable by a consumer of the artifact, not only by a reader of our source:
    an OpenAPI showing five paths and nothing else would misrepresent the surface as complete."""
    listed = build_openapi()["x-uncontracted-routes"]
    assert len(listed) == len(UNCONTRACTED_ROUTES)
    for entry in listed:
        assert entry["reason"].strip(), f"{entry['method']} {entry['path']} has an empty reason"
