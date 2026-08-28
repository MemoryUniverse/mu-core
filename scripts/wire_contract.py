#!/usr/bin/env python3
"""Emit and CHECK the machine-readable wire contract (``mu_contracts.contracts.wire``).

Closes the Phase-0 checklist line ``contracts/__init__.py:11-14`` has carried as SCAFFOLD since the
package was written: "the machine-readable OpenAPI + JSON-Schema the TS SDK and community/
conformant servers generate from" (``PACKAGING-v2.md`` §1.3).

Subcommands
-----------
``emit``        regenerate the two committed artifacts under
                ``packages/mu-contracts/src/mu_contracts/contracts/schema/``.
``check``       fail (exit 1) if the committed artifacts differ from a fresh generation — the
                golden-snapshot gate ``api-mcp-surface-spec.md`` §10 and
                ``api-sdk-mcp-surface-design.md`` §921 both require.
``check-sdks``  fail (exit 1) if EITHER SDK has drifted from the contract:
                ``mu-sdk-python`` (pydantic, sibling repo) and ``mu-sdk-js`` (hand-written zod,
                sibling repo). This is the check that did not exist, and whose absence is why
                ``mu-sdk-js`` is 14 hand-maintained ``.ts`` modules mirroring 15 Python modules
                with nothing verifying the mirror.
``check-all``   ``check`` then ``check-sdks``.

The drift rule (and why it is this rule)
----------------------------------------
Per model, comparing the SDK's field set against the canonical JSON-Schema:

* **RESPONSE models — exact set equality.** Both SDKs parse responses with a CLOSED schema
  (pydantic ``extra="forbid"``, zod ``.strict()``), so a field added canonically makes the SDK
  RAISE on a valid server response, and a field the SDK declares but the server never sends is a
  field the SDK invented. Either direction is a real break.
* **REQUEST models — the SDK may be a SUBSET, never a SUPERSET, and must carry every REQUIRED
  field.** A client that omits an optional field still produces a message the server accepts; a
  client that sends a field the canonical model does not declare gets a hard ``422``
  (``extra_forbidden``) — the exact bug ``contracts/requests.py``'s docstring records as the
  reason that module exists.
* **Type KIND must agree** for every shared field (string / number / integer / boolean / array /
  object / enum), which is what catches a retype or a rename-by-shape.
* **An UNCLASSIFIABLE type is a FAILURE, never a pass.** If either side's declaration cannot be
  reduced to a kind — a zod expression outside ``_TS_KIND_PATTERNS`` (``z.unknown()``, ``z.any()``,
  ``z.custom()``, ``z.lazy()``), an annotation outside ``_PY_KIND_OF_ANNOTATION``, an unresolvable
  or too-deep ``$ref`` — the field is REPORTED, not skipped. The empty kind set used to
  short-circuit the ``want & have`` test, so ``z.boolean() -> z.string()`` failed (RC=1) while
  ``z.boolean() -> z.unknown()`` passed silently (RC=0): the checker went blind exactly where the
  SDK was vaguest, which is the same fail-open shape as a ``frozenset(x) or None`` guard. A checker
  that cannot read a declaration has not verified it, and must say so.

Defaults are deliberately NOT compared — see ``mu_contracts.contracts.wire``'s module docstring
for why (``contracts/defaults.py`` stages the SDKs' default wiring as C4, still open).

Usage::

    uv run python scripts/wire_contract.py emit
    uv run python scripts/wire_contract.py check
    uv run python scripts/wire_contract.py check-sdks
    uv run python scripts/wire_contract.py check-all --verbose
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_CONTRACTS_SRC = _REPO_ROOT / "packages" / "mu-contracts" / "src"
# Bootstrap so `python scripts/wire_contract.py` works without an editable install of mu-contracts.
if str(_CONTRACTS_SRC) not in sys.path:
    sys.path.insert(0, str(_CONTRACTS_SRC))

from mu_contracts.contracts.wire import (  # noqa: E402  (path bootstrap must precede the import)
    WIRE_MODELS,
    build_json_schema,
    build_openapi,
)


def _out(line: str) -> None:
    """stdout, not ``print``: ruff's ``T20`` bans ``print`` repo-wide (DEV-STANDARDS "no print")
    and ``ruff.toml`` is not this task's file to amend, but a CLI that says nothing is useless."""
    sys.stdout.write(line + "\n")


def _err(line: str) -> None:
    sys.stderr.write(line + "\n")


SCHEMA_DIR = _CONTRACTS_SRC / "mu_contracts" / "contracts" / "schema"
OPENAPI_PATH = SCHEMA_DIR / "openapi.json"
JSON_SCHEMA_PATH = SCHEMA_DIR / "wire-models.schema.json"

# The two sibling SDK repos (the ratified 5-repo split). Absent in a standalone `mu-core` clone.
_PROJECT_ROOT = _REPO_ROOT.parent
SDK_PY_MODELS = _PROJECT_ROOT / "mu-sdk-python" / "src" / "mu_sdk" / "models"
SDK_TS_MODELS = _PROJECT_ROOT / "mu-sdk-js" / "src" / "models"


# ======================================================================================
# Artifact emission / golden check
# ======================================================================================
def _render(doc: Mapping[str, Any]) -> str:
    """Byte-stable rendering: sorted keys, 2-space indent, trailing newline.

    ``sort_keys`` is what makes the golden check meaningful — without it a dict-ordering change
    inside pydantic would show up as spurious drift and the gate would be turned off.
    """
    return json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def emit(*, verbose: bool = False) -> int:
    SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
    OPENAPI_PATH.write_text(_render(build_openapi()), encoding="utf-8")
    JSON_SCHEMA_PATH.write_text(_render(build_json_schema()), encoding="utf-8")
    if verbose:
        _out(f"wrote {OPENAPI_PATH.relative_to(_REPO_ROOT)}")
        _out(f"wrote {JSON_SCHEMA_PATH.relative_to(_REPO_ROOT)}")
    return 0


def check_artifacts(*, verbose: bool = False) -> int:
    failures: list[str] = []
    for path, doc in ((OPENAPI_PATH, build_openapi()), (JSON_SCHEMA_PATH, build_json_schema())):
        expected = _render(doc)
        if not path.exists():
            failures.append(f"MISSING artifact {path.relative_to(_REPO_ROOT)}")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            failures.append(
                f"STALE artifact {path.relative_to(_REPO_ROOT)}: the committed file does not "
                f"match a fresh generation from the pydantic models."
            )
    if failures:
        for f in failures:
            _err(f"FAIL: {f}")
        _err("\nRegenerate with: uv run python scripts/wire_contract.py emit")
        return 1
    if verbose:
        _out("OK: committed OpenAPI + JSON-Schema match the pydantic models.")
    return 0


# ======================================================================================
# The canonical field view the SDK checks compare against
# ======================================================================================
_KIND_OF_JSON_TYPE = {
    "string": "string",
    "integer": "integer",
    "number": "number",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
}


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kinds: frozenset[str]
    required: bool
    # The declaration the kinds were read from, verbatim (zod expression / python annotation /
    # schema node). Only ever surfaced when `kinds` is EMPTY: an "unclassifiable type" report that
    # does not quote the text it failed on sends the reader hunting for it.
    source: str = ""


_MAX_SOURCE_ECHO = 160


def _echo(source: str) -> str:
    """One-line, length-capped rendering of a declaration for an error message."""
    flat = " ".join(source.split())
    return flat if len(flat) <= _MAX_SOURCE_ECHO else flat[: _MAX_SOURCE_ECHO - 1] + "\u2026"


def unclassified_problem(side: str, where: str, spec: FieldSpec) -> str:
    """The one place the unclassifiable-kind failure is worded — `compare` is the only caller, so
    the CLI and the pytest gate (which calls `compare` too) cannot drift apart on this rule."""
    return (
        f"{side}: {where} declares a type this checker cannot classify: `{_echo(spec.source)}`. "
        "An unreadable declaration is NOT a match — it is an unchecked field. Teach the checker "
        "the construct (_TS_KIND_PATTERNS / _TS_CONST_KIND / _PY_KIND_OF_ANNOTATION) or declare "
        "the field with a concrete wire type."
    )


@dataclass
class ModelSpec:
    name: str
    role: str
    fields: dict[str, FieldSpec] = field(default_factory=dict)

    @property
    def required(self) -> frozenset[str]:
        return frozenset(n for n, f in self.fields.items() if f.required)


def _kinds_from_schema(
    node: Mapping[str, Any], defs: Mapping[str, Any], depth: int = 0
) -> frozenset[str]:
    """The JSON type KINDS a schema node admits, with ``$ref``/``anyOf``/``allOf`` resolved.

    ``null`` is dropped: optionality is carried by ``required``, not by the kind set, so
    ``str | None`` and ``str`` compare equal in kind — which is correct, because a zod
    ``.nullable().optional()`` and a pydantic ``str | None`` are the same wire shape.
    """
    if depth > 8:  # cycle guard; the wire family is shallow, this can only be a pathological ref
        return frozenset()
    if "$ref" in node:
        key = str(node["$ref"]).rsplit("/", 1)[-1]
        target = defs.get(key)
        return _kinds_from_schema(target, defs, depth + 1) if target else frozenset()
    kinds: set[str] = set()
    if "enum" in node or "const" in node:
        kinds.add("enum")
    raw_type = node.get("type")
    for t in [raw_type] if isinstance(raw_type, str) else (raw_type or []):
        if t == "null":
            continue
        kinds.add(_KIND_OF_JSON_TYPE.get(str(t), str(t)))
    for combinator in ("anyOf", "oneOf", "allOf"):
        for sub in node.get(combinator, []):
            kinds |= _kinds_from_schema(sub, defs, depth + 1)
    return frozenset(kinds)


def canonical_specs() -> dict[str, ModelSpec]:
    """The canonical field view, read from the GENERATED JSON-Schema (not from the classes).

    Reading the artifact — the thing an SDK author actually generates from — is the point: if the
    artifact were wrong, an SDK faithful to the models would be reported as drifted, which is
    exactly the signal we want.
    """
    doc = build_json_schema()
    defs: Mapping[str, Any] = doc["$defs"]
    key_of: Mapping[str, str] = doc["x-schema-key"]
    specs: dict[str, ModelSpec] = {}
    for name, (_model, role) in WIRE_MODELS.items():
        lookup = name if role != "shared" else f"{name}.request"
        node = defs[key_of[lookup]]
        spec = ModelSpec(name=name, role=role)
        required = set(node.get("required", []))
        for fname, fnode in node.get("properties", {}).items():
            spec.fields[fname] = FieldSpec(
                name=fname,
                kinds=_kinds_from_schema(fnode, defs),
                required=fname in required,
                source=json.dumps(fnode, sort_keys=True),
            )
        specs[name] = spec
    return specs


# ======================================================================================
# mu-sdk-python — pydantic models, read with `ast` (no import, so no install needed)
# ======================================================================================
# canonical model name -> (module stem, class name). Only models the SDK RE-DECLARES are listed;
# the ones it imports verbatim from mu-contracts are covered by _PY_REEXPORTS below, which is a
# stronger check (identity, not similarity).
_PY_LOCAL_MODELS: Mapping[str, tuple[str, str]] = {
    "MemoryCreateRequest": ("memory", "MemoryCreateRequest"),
    "RecallRequest": ("recall", "RecallRequest"),
    "ConsolidateRequest": ("consolidate", "ConsolidateRequest"),
}

# canonical model name -> the module that must import it from mu_contracts rather than re-declare.
_PY_REEXPORTS: Mapping[str, str] = {
    "MemoryResponse": "memory",
    "MemoryWriteResult": "memory",
    "MemoryVerbResult": "memory",
    "RecallChannels": "recall",
    "RecallItemView": "recall",
    "RecallResult": "recall",
    "ContextView": "context",
}

_PY_KIND_OF_ANNOTATION = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "tuple": "array",
    "dict": "object",
    "datetime": "string",
    "ContentType": "enum",
    "MemoryTierLiteral": "enum",
    "PolarityLiteral": "enum",
    "Visibility": "enum",
    "RecallMode": "enum",
    "RecallChannels": "object",
    "Namespace": "object",
    "DegradeReason": "enum",
    "Tier": "enum",
    "MemoryResponse": "object",
    "RecallItemView": "object",
}


def _py_kind(annotation: ast.expr) -> frozenset[str]:
    kinds: set[str] = set()
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name):
            mapped = _PY_KIND_OF_ANNOTATION.get(node.id)
            if mapped:
                kinds.add(mapped)
        elif isinstance(node, ast.Attribute):
            mapped = _PY_KIND_OF_ANNOTATION.get(node.attr)
            if mapped:
                kinds.add(mapped)
        elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            if node.value.id == "Literal":
                kinds.add("enum")
    return frozenset(kinds)


def _py_required(node: ast.AnnAssign) -> bool:
    """REQUIRED iff there is no value and no ``Field(default=...)`` / ``default_factory``."""
    if node.value is None:
        return True
    call = node.value
    if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "Field":
        if call.args:  # Field(<positional default>, ...)
            return False
        kwargs = {kw.arg for kw in call.keywords}
        return not ({"default", "default_factory"} & kwargs)
    return False


def parse_python_sdk() -> tuple[dict[str, ModelSpec], list[str]]:
    """Field views for the models ``mu-sdk-python`` declares itself, plus its re-export evidence."""
    specs: dict[str, ModelSpec] = {}
    reexport_notes: list[str] = []
    trees: dict[str, ast.Module] = {}
    for stem in {*(m for m, _c in _PY_LOCAL_MODELS.values()), *_PY_REEXPORTS.values()}:
        path = SDK_PY_MODELS / f"{stem}.py"
        trees[stem] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for canonical, (stem, cls_name) in _PY_LOCAL_MODELS.items():
        for node in ast.walk(trees[stem]):
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                spec = ModelSpec(name=canonical, role=WIRE_MODELS[canonical][1])
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        fname = stmt.target.id
                        if fname.startswith("_") or fname == "model_config":
                            continue
                        spec.fields[fname] = FieldSpec(
                            name=fname,
                            kinds=_py_kind(stmt.annotation),
                            required=_py_required(stmt),
                            source=ast.unparse(stmt.annotation),
                        )
                specs[canonical] = spec
                break

    for canonical, stem in _PY_REEXPORTS.items():
        imported = any(
            isinstance(n, ast.ImportFrom)
            and (n.module or "").startswith("mu_contracts")
            and any(a.name == canonical for a in n.names)
            for n in ast.walk(trees[stem])
        )
        redeclared = any(
            isinstance(n, ast.ClassDef) and n.name == canonical for n in ast.walk(trees[stem])
        )
        if redeclared or not imported:
            reexport_notes.append(
                f"mu-sdk-python models/{stem}.py: {canonical} must be IMPORTED from mu_contracts "
                f"(single-sourced), not re-declared. imported={imported} redeclared={redeclared}"
            )
    return specs, reexport_notes


# ======================================================================================
# mu-sdk-js — hand-written zod, read with a targeted parser
# ======================================================================================
# canonical model name -> the exported zod const in mu-sdk-js/src/models/*.ts.
_TS_MODELS: Mapping[str, str] = {
    "MemoryCreateRequest": "memoryCreateRequestSchema",
    "MemoryResponse": "memoryResponseSchema",
    "MemoryWriteResult": "memoryWriteResultSchema",
    "MemoryVerbResult": "memoryVerbResultSchema",
    "RecallChannels": "recallChannelsSchema",
    "RecallItemView": "recallItemViewSchema",
    "RecallResult": "recallResultSchema",
    "RecallRequest": "recallRequestSchema",
    "ConsolidateRequest": "consolidateRequestSchema",
    "ConsolidateView": "consolidateViewSchema",
    "ContextView": "contextViewSchema",
    "Namespace": "namespaceSchema",
}

_TS_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"z\.enum\(", "enum"),
    (r"z\.array\(", "array"),
    (r"z\.record\(", "object"),
    (r"z\.object\(", "object"),
    (r"z\.boolean\(", "boolean"),
    (r"z\.coerce\.date\(", "string"),
    (r"z\.string\(", "string"),
    (r"z\.number\(\)\.int\(", "integer"),
    (r"z\.number\(", "number"),
)

# zod const -> the kind it resolves to when referenced as another schema's field type.
_TS_CONST_KIND: Mapping[str, str] = {
    "contentTypeSchema": "enum",
    "memoryTierSchema": "enum",
    "polaritySchema": "enum",
    "visibilitySchema": "enum",
    "recallModeSchema": "enum",
    "degradeReasonSchema": "enum",
    "recallChannelsSchema": "object",
    "namespaceSchema": "object",
    "memoryResponseSchema": "object",
    "recallItemViewSchema": "object",
}

_TS_STRIP_COMMENT = re.compile(r"//[^\n]*")


def _ts_blocks(source: str) -> dict[str, str]:
    """Map ``export const <name>Schema = z.object({ ... })`` to the text between its braces.

    Brace-counted, not regex-matched, so a nested ``z.object({...})`` inside a field cannot end the
    block early.
    """
    blocks: dict[str, str] = {}
    for match in re.finditer(r"export const (\w+)\s*=\s*z\s*\n?\s*\.object\(\{", source):
        name = match.group(1)
        depth = 1
        i = match.end()
        while i < len(source) and depth:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        blocks[name] = source[match.end() : i - 1]
    return blocks


def _ts_fields(block: str) -> dict[str, FieldSpec]:
    """Top-level ``name: <zod expr>,`` entries of one zod object block."""
    fields: dict[str, FieldSpec] = {}
    depth = 0
    current: list[str] = []
    entries: list[str] = []
    for ch in block:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            entries.append("".join(current))
            current = []
        else:
            current.append(ch)
    entries.append("".join(current))

    for raw in entries:
        entry = _TS_STRIP_COMMENT.sub("", raw).strip()
        if not entry or ":" not in entry:
            continue
        name, _, expr = entry.partition(":")
        name = name.strip().strip('"')
        if not re.fullmatch(r"[A-Za-z_]\w*", name):
            continue
        kinds: set[str] = set()
        for pattern, kind in _TS_KIND_PATTERNS:
            if re.search(pattern, expr):
                kinds.add(kind)
        for const, kind in _TS_CONST_KIND.items():
            if re.search(rf"\b{const}\b", expr):
                kinds.add(kind)
        # z.number().int() also matches z.number( — keep only the narrower integer kind.
        if "integer" in kinds:
            kinds.discard("number")
        # A z.array(...)/z.record(...) wrapper subsumes the element kinds it wraps.
        if "array" in kinds:
            kinds = {"array"}
        elif "object" in kinds and re.search(r"z\.record\(|z\.object\(", expr):
            kinds = {"object"}
        required = not re.search(r"\.optional\(\)|\.default\(", expr)
        fields[name] = FieldSpec(
            name=name, kinds=frozenset(kinds), required=required, source=expr.strip()
        )
    return fields


def parse_ts_sdk() -> dict[str, ModelSpec]:
    source = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SDK_TS_MODELS.glob("*.ts")))
    blocks = _ts_blocks(source)
    specs: dict[str, ModelSpec] = {}
    for canonical, const in _TS_MODELS.items():
        if const not in blocks:
            continue
        spec = ModelSpec(name=canonical, role=WIRE_MODELS[canonical][1])
        spec.fields = _ts_fields(blocks[const])
        specs[canonical] = spec
    return specs


# ======================================================================================
# The drift comparison
# ======================================================================================
def compare(
    sdk: str,
    canonical: Mapping[str, ModelSpec],
    actual: Mapping[str, ModelSpec],
    expected_models: Iterable[str],
) -> list[str]:
    problems: list[str] = []
    for name in expected_models:
        spec = canonical[name]
        got = actual.get(name)
        if got is None:
            problems.append(f"{sdk}: model {name} is MISSING — the SDK does not declare it at all.")
            continue

        extra = sorted(set(got.fields) - set(spec.fields))
        missing = sorted(set(spec.fields) - set(got.fields))
        if extra:
            problems.append(
                f"{sdk}: {name} declares field(s) the contract does not: {extra}. "
                "Every canonical wire model is extra=forbid/.strict() — an invented field is a 422."
            )
        if spec.role == "response":
            if missing:
                problems.append(
                    f"{sdk}: {name} (RESPONSE) is missing field(s) the contract emits: {missing}. "
                    "The SDK parses responses with a closed schema, so it will RAISE on a valid "
                    "server response."
                )
        else:
            missing_required = sorted(spec.required - set(got.fields))
            if missing_required:
                problems.append(
                    f"{sdk}: {name} (REQUEST) is missing REQUIRED field(s): {missing_required}. "
                    "The SDK cannot construct a message the server will accept."
                )
        # Every canonical field is audited, not only the shared ones: a REQUEST field the SDK is
        # allowed to omit still has to be READABLE on the contract side, or the artifact the SDKs
        # are generated from is itself unverified.
        for fname in sorted(spec.fields):
            want = spec.fields[fname]
            if not want.kinds:
                problems.append(unclassified_problem("contract", f"{name}.{fname}", want))
                continue
            have = got.fields.get(fname)
            if have is None:
                continue  # absence is reported above, under the rule for this model's role.
            if not have.kinds:
                problems.append(unclassified_problem(sdk, f"{name}.{fname}", have))
                continue
            if not (want.kinds & have.kinds):
                problems.append(
                    f"{sdk}: {name}.{fname} type KIND disagrees — contract {sorted(want.kinds)}, "
                    f"SDK {sorted(have.kinds)}."
                )
    return problems


def check_sdks(*, verbose: bool = False) -> int:
    canonical = canonical_specs()
    problems: list[str] = []
    checked: list[str] = []

    if SDK_PY_MODELS.is_dir():
        py_specs, reexport_notes = parse_python_sdk()
        problems += reexport_notes
        problems += compare("mu-sdk-python", canonical, py_specs, _PY_LOCAL_MODELS)
        checked.append(
            f"mu-sdk-python ({len(_PY_LOCAL_MODELS)} re-declared "
            f"+ {len(_PY_REEXPORTS)} re-exported)"
        )
    else:
        problems.append(
            f"mu-sdk-python not found at {SDK_PY_MODELS} — the sibling repo must be checked out "
            "for this gate to mean anything."
        )

    if SDK_TS_MODELS.is_dir():
        ts_specs = parse_ts_sdk()
        problems += compare("mu-sdk-js", canonical, ts_specs, _TS_MODELS)
        checked.append(f"mu-sdk-js ({len(_TS_MODELS)} hand-written zod schemas)")
    else:
        problems.append(
            f"mu-sdk-js not found at {SDK_TS_MODELS} — the sibling repo must be checked out for "
            "this gate to mean anything."
        )

    # A contract-side problem is worded identically for every SDK; collapse the repeats while
    # keeping first-seen order, so the count is a count of DEFECTS, not of comparisons.
    problems = list(dict.fromkeys(problems))

    if problems:
        _err(f"SDK DRIFT ({len(problems)} problem(s)):")
        for problem in problems:
            _err(f"  FAIL: {problem}")
        return 1
    if verbose:
        _out(f"OK: no SDK drift. Checked {', '.join(checked)}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "command", choices=("emit", "check", "check-sdks", "check-all"), help="what to do"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "emit":
        return emit(verbose=args.verbose)
    if args.command == "check":
        return check_artifacts(verbose=args.verbose)
    if args.command == "check-sdks":
        return check_sdks(verbose=args.verbose)
    return check_artifacts(verbose=args.verbose) or check_sdks(verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
