"""The SDK-drift gate — the check whose ABSENCE is why ``mu-sdk-js`` is hand-written.

``PACKAGING-v2.md`` §74/§103 and ``mu-local-and-sdk-spec.md`` §C3 describe ``mu-sdk-js`` as "the TS
twin **generated from** ``mu-contracts`` OpenAPI". It is not generated — it is 14 hand-maintained
``.ts`` modules re-declaring the same shapes in zod, with nothing anywhere verifying the mirror.
This test is the verification: it compares BOTH SDKs' declared field sets against the generated
JSON-Schema and fails on any drift. The rule, and why it is asymmetric between requests and
responses, is documented in ``scripts/wire_contract.py``'s module docstring.

**Why this test may skip, and why that is not a silent skip.** ``mu-sdk-python`` and ``mu-sdk-js``
are SIBLING REPOSITORIES (the ratified 5-repo split, ``PACKAGING-v2.md``). ``mu-core`` is the OPEN
repo whose whole promise is that it depends on nothing, so it cannot require either sibling to be
on disk — exactly the reasoning the workspace ``pyproject.toml`` records for keeping the
``acceptance`` group non-default. The skip therefore names the missing path, and the orchestrated
repo-set run has both siblings present so the gate really runs (proved in this task's report). The
same comparison is also available standalone as
``uv run python scripts/wire_contract.py check-sdks``.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "wire_contract.py"


def _load_checker() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mu_wire_contract_script", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: `dataclasses` resolves `from __future__ import annotations` string
    # annotations through `sys.modules[cls.__module__]`, and a module loaded purely from a spec is
    # not there yet -- the @dataclass decorators in the script raise without this line.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker() -> ModuleType:
    if not _SCRIPT.exists():
        pytest.fail(f"the generator/checker is missing at {_SCRIPT}")
    return _load_checker()


def test_python_sdk_has_not_drifted(checker: ModuleType) -> None:
    if not checker.SDK_PY_MODELS.is_dir():
        pytest.skip(f"sibling repo mu-sdk-python not checked out at {checker.SDK_PY_MODELS}")
    specs, reexport_problems = checker.parse_python_sdk()
    assert specs, "parsed ZERO models from mu-sdk-python — the parser, not the SDK, is broken"
    problems = reexport_problems + checker.compare(
        "mu-sdk-python", checker.canonical_specs(), specs, checker._PY_LOCAL_MODELS
    )
    assert not problems, "\n".join(problems)


def test_ts_sdk_has_not_drifted(checker: ModuleType) -> None:
    if not checker.SDK_TS_MODELS.is_dir():
        pytest.skip(f"sibling repo mu-sdk-js not checked out at {checker.SDK_TS_MODELS}")
    specs = checker.parse_ts_sdk()
    missing = sorted(set(checker._TS_MODELS) - set(specs))
    assert not missing, (
        f"mu-sdk-js declares no zod schema for {missing} — the TS twin is missing part of the "
        "wire vocabulary entirely."
    )
    problems = checker.compare("mu-sdk-js", checker.canonical_specs(), specs, checker._TS_MODELS)
    assert not problems, "\n".join(problems)


def test_the_parsers_actually_extract_fields(checker: ModuleType) -> None:
    """Guards the gate itself: a parser that silently returns empty field sets would make every
    drift check pass vacuously, which is the classic way a conformance gate rots into decoration.
    """
    if not (checker.SDK_TS_MODELS.is_dir() and checker.SDK_PY_MODELS.is_dir()):
        pytest.skip("both sibling SDK repos must be present to audit the parsers")

    canonical = checker.canonical_specs()
    ts_specs = checker.parse_ts_sdk()
    py_specs, _ = checker.parse_python_sdk()

    for label, specs in (("mu-sdk-js", ts_specs), ("mu-sdk-python", py_specs)):
        for name, spec in specs.items():
            assert spec.fields, f"{label}: parsed {name} with ZERO fields"
            untyped = sorted(f for f, v in spec.fields.items() if not v.kinds)
            assert not untyped, f"{label}: {name} field(s) parsed with no type kind: {untyped}"

    # And the canonical side is non-trivial too: MemoryResponse is the widest wire model
    # (37 fields), so a canonical view that collapsed would be obvious here.
    assert len(canonical["MemoryResponse"].fields) == len(ts_specs["MemoryResponse"].fields) == 37


# ======================================================================================
# The unclassifiable-kind rule
# ======================================================================================
# Regression cover for the fail-open `compare` shipped with: `if want and have and not (want &
# have)`. `_ts_fields` yields an EMPTY kind set for any zod expression outside `_TS_KIND_PATTERNS`,
# and an empty set made that condition False — so the kind check was skipped, not satisfied.
# Measured before the fix: `z.boolean() -> z.string()` failed (RC=1); `z.boolean() -> z.unknown()`
# passed silently (RC=0). The checker was blind exactly where the SDK was vaguest.
_UNREADABLE_ZOD = (
    "z.unknown()",
    "z.any()",
    "z.custom<boolean>()",
    "z.lazy(() => someSchemaTheTableDoesNotKnow)",
)


def _spec(
    checker: ModuleType,
    name: str,
    role: str,
    *,
    optional: tuple[str, ...] = (),
    **fields: object,
) -> Any:
    """A synthetic ModelSpec: `kinds` given as a string kind, or None for "unclassifiable"."""
    spec = checker.ModelSpec(name=name, role=role)
    for fname, kind in fields.items():
        spec.fields[fname] = checker.FieldSpec(
            name=fname,
            kinds=frozenset() if kind is None else frozenset({str(kind)}),
            required=fname not in optional,
            source="<unreadable declaration>" if kind is None else str(kind),
        )
    return spec


@pytest.mark.parametrize("expr", _UNREADABLE_ZOD)
def test_zod_outside_the_pattern_table_parses_as_unclassified_not_as_anything(
    checker: ModuleType, expr: str
) -> None:
    """Step 1 of the bug: these expressions genuinely produce an empty kind set.

    That is legitimate — the parser cannot know every zod construct. What is NOT legitimate is
    `compare` reading that emptiness as agreement, which is what the next test pins.
    """
    fields = checker._ts_fields(f"  flag: {expr},\n  other: z.string(),\n")
    assert fields["flag"].kinds == frozenset(), f"expected {expr} to be unclassifiable"
    assert fields["flag"].source == expr, "the failure message must quote the real declaration"
    assert fields["other"].kinds == frozenset({"string"}), "the parser itself must still work"


def test_an_unclassifiable_sdk_kind_is_a_failure_not_a_pass(checker: ModuleType) -> None:
    """Step 2: the actual fix. An empty SDK kind set must be reported, never short-circuited."""
    canonical = {"M": _spec(checker, "M", "response", flag="boolean")}
    readable_and_wrong = {"M": _spec(checker, "M", "response", flag="string")}
    unreadable = {"M": _spec(checker, "M", "response", flag=None)}

    assert checker.compare(
        "sdk", canonical, readable_and_wrong, ["M"]
    ), "the kind check that always worked must keep working"
    problems = checker.compare("sdk", canonical, unreadable, ["M"])
    assert problems, "an unclassifiable SDK type silently passed — the fail-open is back"
    assert "cannot classify" in problems[0]
    assert "<unreadable declaration>" in problems[0], "the report must quote what it failed on"


def test_an_unclassifiable_contract_kind_is_also_a_failure(checker: ModuleType) -> None:
    """The same blindness on the CONTRACT side (an unresolvable/too-deep `$ref` returns an empty
    kind set too) — and it is audited even for a REQUEST field the SDK is allowed to omit, since
    an unreadable canonical field is an unverified one whichever side declares it."""
    canonical = {"M": _spec(checker, "M", "request", optional=("flag",), flag=None, other="string")}
    sdk_omits_it = {"M": _spec(checker, "M", "request", other="string")}

    problems = checker.compare("sdk", canonical, sdk_omits_it, ["M"])
    assert problems == [
        checker.unclassified_problem("contract", "M.flag", canonical["M"].fields["flag"])
    ], "an unreadable canonical field passed unaudited (the SDK omitting it is legal here)"


def test_the_cli_and_the_pytest_gate_apply_the_same_rule(
    checker: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reconciliation. Before the fix these two DISAGREED on a real mutation: with
    `promoted: z.unknown()` in mu-sdk-js, `wire_contract.py check-sdks` exited 0 while
    `test_the_parsers_actually_extract_fields` went red — so the CLI, which is what a human or a
    hook actually runs, reported clean on a defect the suite could see. Both now route through
    `compare`, and this test fails if they are ever split again.
    """
    if not (checker.SDK_TS_MODELS.is_dir() and checker.SDK_PY_MODELS.is_dir()):
        pytest.skip("both sibling SDK repos must be present to compare the two gates")

    def pytest_gate_problems() -> list[str]:
        canonical = checker.canonical_specs()
        py_specs, reexport_problems = checker.parse_python_sdk()
        problems: list[str] = [
            *reexport_problems,
            *checker.compare("mu-sdk-python", canonical, py_specs, checker._PY_LOCAL_MODELS),
            *checker.compare("mu-sdk-js", canonical, checker.parse_ts_sdk(), checker._TS_MODELS),
        ]
        return problems

    assert pytest_gate_problems() == []
    assert checker.check_sdks() == 0

    # Blind exactly one field the way `z.unknown()` blinded it, and require BOTH gates to go red.
    real_parse = checker.parse_ts_sdk

    def blinded() -> dict[str, Any]:
        specs: dict[str, Any] = real_parse()
        fields = specs["MemoryWriteResult"].fields
        fields["promoted"] = dataclasses.replace(
            fields["promoted"], kinds=frozenset(), source="z.unknown()"
        )
        return specs

    monkeypatch.setattr(checker, "parse_ts_sdk", blinded)
    assert pytest_gate_problems(), "the pytest gate went blind"
    assert checker.check_sdks() == 1, "the CLI went blind while the pytest gate did not"
