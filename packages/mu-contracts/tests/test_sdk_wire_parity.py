"""The SDK-drift gate — the check whose ABSENCE is why ``mu-sdk-js`` is hand-written.

``PACKAGING-v2.md`` §74/§103 and ``mu-local-and-sdk-spec.md`` §C3 describe ``mu-sdk-js`` as "the TS
twin **generated from** ``mu-contracts`` OpenAPI". It is not generated — it is 14 hand-maintained
``.ts`` modules re-declaring the same shapes in zod, with nothing anywhere verifying the mirror.
This test is the verification: it compares BOTH SDKs' declared field sets against the generated
JSON-Schema and fails on any drift. The rule, and why it is asymmetric between requests and
responses, is documented in ``scripts/wire_contract.py``'s module docstring.

**Why this test may skip, and why that skip is no longer silent.** ``mu-sdk-python`` and
``mu-sdk-js`` are SIBLING REPOSITORIES (the ratified 5-repo split, ``PACKAGING-v2.md``).
``mu-core`` is the OPEN repo whose whole promise is that it depends on nothing, so it cannot
require either sibling to be on disk — exactly the reasoning the workspace ``pyproject.toml``
records for keeping the ``acceptance`` group non-default. A developer with one clone must be able
to run the suite.

That is the right answer for a laptop and the WRONG answer for any run that was DECLARED possible.
MEASURED before this was fixed, on the dev VM at mu-core dev/mlm-build@eaf6c00, via
``SYNC=head infra/mu-vm/vm_test.sh mu-core packages/mu-contracts/tests/test_sdk_wire_parity.py``::

    packages/mu-contracts/tests/test_sdk_wire_parity.py .ss......s   [100%]
    SKIPPED [1] ...:69: sibling repo mu-sdk-js not checked out at ~/mu_project/mu-sdk-js/src/models
    SKIPPED [1] ...:85: both sibling SDK repos must be present to audit the parsers
    SKIPPED [1] ...:191: both sibling SDK repos must be present to compare the two gates
    7 passed, 3 skipped in 0.37s          # exit 0

``vm_test.sh`` synced ``mu-sdk-python`` (mu-core's ``pyproject`` names it) but nothing anywhere
synced ``mu-sdk-js``, which no ``pyproject`` mentions because it is READ AS TEXT, not imported. So
the only check of the TS twin took a green skip on the machine the suite is supposed to run on.

``MU_REQUIRE_SIBLING_REPOS=1`` turns every one of those skips into a FAILURE naming the path it
looked in. ``vm_test.sh`` exports it once it has actually put both siblings on the far side, and
mu-core's CI sets it on the parity step. The same comparison is also available standalone as
``uv run python scripts/wire_contract.py check-sdks``, which treats an absent sibling as a failure
unconditionally.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "wire_contract.py"

#: The environment variable, spelled ONCE here and referenced by name in two other repositories:
#: ``infra/mu-vm/vm_test.sh`` exports it after it has put both siblings on the far side, and
#: ``mu-core/.github/workflows/ci.yml`` sets it as job env after the two sibling checkouts. A typo
#: in any of the three restores the silent green skip this module exists to end, which is why
#: ``test_a_missing_sibling_fails_loudly_when_the_run_was_declared_possible`` below pins the
#: semantics of the switch itself.
_REQUIRE_ENV = "MU_REQUIRE_SIBLING_REPOS"


def _run_was_declared_possible() -> bool:
    """Read at CALL time, not at import time.

    Set wherever both sibling SDK checkouts were actually provisioned, which is precisely the
    condition under which "this run was supposed to be possible" is a TRUE statement — and
    therefore the only condition under which a missing sibling is a defect rather than an honest
    report of a machine that cannot host the run. Same escape hatch, same reasoning, as
    ``MU_REQUIRE_PRIVACY_IT`` in mu-client's cross-plane privacy test.

    It was a module-level constant, which made the switch itself untestable without reloading the
    module out from under pytest — so the one line that decides between a green skip and a red
    failure had no test at all.
    """
    return os.environ.get(_REQUIRE_ENV, "").strip().lower() in {"1", "true", "yes"}


def _sibling_missing(reason: str) -> None:
    """Skip, or FAIL when this run was declared possible. ONE place the decision is made, so the
    four call sites cannot drift into four different policies."""
    if _run_was_declared_possible():
        pytest.fail(
            f"{_REQUIRE_ENV} is set, so this run was declared able to compare the SDKs "
            f"— but {reason}. The drift comparison did NOT happen; fix the checkout, never the "
            f"assertion."
        )
    pytest.skip(f"{reason} (set {_REQUIRE_ENV}=1 to make this a failure)")


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
        _sibling_missing(
            f"sibling repo mu-sdk-python is not checked out at {checker.SDK_PY_MODELS}"
        )
    specs, reexport_problems = checker.parse_python_sdk()
    assert specs, "parsed ZERO models from mu-sdk-python — the parser, not the SDK, is broken"
    problems = reexport_problems + checker.compare(
        "mu-sdk-python", checker.canonical_specs(), specs, checker._PY_LOCAL_MODELS
    )
    assert not problems, "\n".join(problems)


def test_ts_sdk_has_not_drifted(checker: ModuleType) -> None:
    if not checker.SDK_TS_MODELS.is_dir():
        _sibling_missing(f"sibling repo mu-sdk-js is not checked out at {checker.SDK_TS_MODELS}")
    specs = checker.parse_ts_sdk()
    missing = sorted(set(checker._TS_MODELS) - set(specs))
    assert not missing, (
        f"mu-sdk-js declares no zod schema for {missing} — the TS twin is missing part of the "
        "wire vocabulary entirely."
    )
    problems = checker.compare("mu-sdk-js", checker.canonical_specs(), specs, checker._TS_MODELS)
    assert not problems, "\n".join(problems)


def test_a_prose_comma_in_a_comment_does_not_eat_the_field_below_it(checker: ModuleType) -> None:
    """AD-328. ``_ts_fields`` splits a zod block into entries on depth-0 commas. It used to strip
    ``//`` comments per-entry AFTERWARDS, so a comma in ordinary comment prose split the entry: the
    fragment carrying ``name: z...`` began mid-comment-line with no ``//`` left to strip, its
    ``partition(":")`` yielded prose, and the ``fullmatch`` identifier guard dropped the field.

    MEASURED, not hypothetical: adding the AD-308/AD-316 temporal trio to ``mu-sdk-js``'s
    ``recallItemViewSchema`` under a normal prose comment left ``valid_at`` — only ``valid_at``, the
    one field whose comment line above it ended in a depth-0 comma — still reported missing by
    ``test_ts_sdk_has_not_drifted`` while plainly present in the file. Commas inside parentheses
    never triggered it (``(`` raises the depth), which is why the pre-existing comments in that file
    happened to survive and this went unnoticed.

    Direction of the bug matters and is asserted by the sibling ``test_the_parsers_actually_
    extract_fields``: it fails CLOSED (a swallowed field reads as SDK drift), so no past green was
    wrong — but it accuses the SDK of a break that is really in the parser.
    """
    block = """
    kept_before: z.string(),
    // A comment whose prose contains a comma, exactly like this one, followed by a field.
    eaten: z.coerce.date().nullable().optional(),
    // Commas inside parentheses (like this, and this) never split, so this one always parsed.
    never_eaten: z.boolean().default(false),
"""
    fields = checker._ts_fields(block)
    assert set(fields) == {
        "kept_before",
        "eaten",
        "never_eaten",
    }, f"a prose comma swallowed a field: parsed {sorted(fields)}"
    # And the survivor is parsed CORRECTLY, not merely present: a fragment-mangled entry would
    # classify from truncated text.
    assert fields["eaten"].kinds == frozenset({"string"})  # z.coerce.date() is a wire string
    assert fields["eaten"].required is False


def test_the_parsers_actually_extract_fields(checker: ModuleType) -> None:
    """Guards the gate itself: a parser that silently returns empty field sets would make every
    drift check pass vacuously, which is the classic way a conformance gate rots into decoration.
    """
    if not (checker.SDK_TS_MODELS.is_dir() and checker.SDK_PY_MODELS.is_dir()):
        _sibling_missing(
            f"both sibling SDK repos must be present to audit the parsers "
            f"({checker.SDK_PY_MODELS}, {checker.SDK_TS_MODELS})"
        )

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
        _sibling_missing(
            f"both sibling SDK repos must be present to compare the two gates "
            f"({checker.SDK_PY_MODELS}, {checker.SDK_TS_MODELS})"
        )

    def pytest_gate_problems() -> list[str]:
        return _pytest_gate_problems(checker)

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


# ======================================================================================
# The request/response ASYMMETRY, and the "shared" role that is BOTH
# ======================================================================================
def _pytest_gate_problems(checker: ModuleType) -> list[str]:
    """Exactly what `test_python_sdk_has_not_drifted` + `test_ts_sdk_has_not_drifted` assert on,
    as one list — the pytest gate's verdict, expressed once so it cannot be described two ways."""
    canonical = checker.canonical_specs()
    py_specs, reexport_problems = checker.parse_python_sdk()
    return [
        *reexport_problems,
        *checker.compare("mu-sdk-python", canonical, py_specs, checker._PY_LOCAL_MODELS),
        *checker.compare("mu-sdk-js", canonical, checker.parse_ts_sdk(), checker._TS_MODELS),
    ]


@pytest.mark.parametrize("role", ["request", "response", "shared"])
def test_a_field_the_contract_does_not_declare_is_a_break_in_every_role(
    checker: ModuleType, role: str
) -> None:
    """The SYMMETRIC half: every canonical model is extra=forbid/.strict(), so an invented field
    is a hard 422 (request) or an SDK inventing data (response) whatever the role."""
    canonical = {"M": _spec(checker, "M", role, kept="string")}
    superset = {"M": _spec(checker, "M", role, kept="string", invented="string")}
    problems = checker.compare("sdk", canonical, superset, ["M"])
    assert problems, f"a superset passed for role {role!r} — extra=forbid/.strict() is not gated"
    assert "does not: ['invented']" in problems[0]


@pytest.mark.parametrize(
    ("role", "omission_is_legal"),
    [("request", True), ("response", False), ("shared", False)],
)
def test_omitting_an_optional_field_is_legal_only_for_a_request(
    checker: ModuleType, role: str, omission_is_legal: bool
) -> None:
    """The ASYMMETRIC half, and the bug that was in it.

    A client that omits an optional REQUEST field still sends a message the server accepts. A
    RESPONSE model is parsed closed, so an omitted field is a key the SDK will reject. `shared`
    took the request rule and therefore the laxer answer — but `Namespace` and `RecallChannels`
    are nested inside `RecallResult` (contracts/recall.py:109,111), so they are parsed closed too.
    MEASURED before the fix, deleting `ltm` from mu-sdk-js's `recallChannelsSchema`:
    `check-sdks` -> "OK: no SDK drift", RC=0.
    """
    canonical = {
        "M": _spec(checker, "M", role, optional=("droppable",), kept="string", droppable="string")
    }
    subset = {"M": _spec(checker, "M", role, kept="string")}
    problems = checker.compare("sdk", canonical, subset, ["M"])
    assert (problems == []) is omission_is_legal, (
        f"role {role!r}: expected omission to be "
        f"{'legal' if omission_is_legal else 'a FAILURE'}; got {problems}"
    )


def test_omitting_a_required_request_field_is_a_break(checker: ModuleType) -> None:
    canonical = {"M": _spec(checker, "M", "request", kept="string", mandatory="string")}
    subset = {"M": _spec(checker, "M", "request", kept="string")}
    problems = checker.compare("sdk", canonical, subset, ["M"])
    assert problems and "REQUIRED field(s): ['mandatory']" in problems[0]


# (model, action, field) -> whether BOTH gates must report a problem.
# Every entry names a REAL canonical model and a REAL field, so a rename on either side turns this
# table red rather than letting it drift into fiction.
_ASYMMETRY_MUTATIONS = (
    # The measured fail-open: a shared model is carried inside a response, so dropping one of its
    # (optional!) fields is a ZodError on the first real recall.
    ("RecallChannels", "drop", "ltm", True),
    # Response, optional on the contract side — still exact-set.
    ("MemoryResponse", "drop", "asserted_state", True),
    # Request, optional — the ONE legal omission. Both gates must stay GREEN, or the asymmetry has
    # collapsed into "everything is exact" and the gate will be turned off within a week.
    ("RecallRequest", "drop", "persona", False),
    # Request, required.
    ("RecallRequest", "drop", "text", True),
    # Superset — illegal everywhere.
    ("RecallRequest", "add", "invented_by_the_ts_twin", True),
)


@pytest.mark.parametrize(("model", "action", "field_name", "must_be_red"), _ASYMMETRY_MUTATIONS)
def test_the_two_gates_agree_on_the_asymmetric_rule(
    checker: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    action: str,
    field_name: str,
    must_be_red: bool,
) -> None:
    """Plant one real mutation in the parsed TS SDK and require the CLI (`check_sdks`) and the
    pytest gate to reach the SAME verdict — red together, or green together.

    `test_the_cli_and_the_pytest_gate_apply_the_same_rule` above pins them together on the
    UNCLASSIFIABLE-type rule. This pins them on the request/response ASYMMETRY, which is the other
    rule either side could be taught separately: `compare` is the only implementation of it, and
    the moment someone re-implements the role test inside `check_sdks` (or inside a test) this
    goes red.
    """
    if not (checker.SDK_TS_MODELS.is_dir() and checker.SDK_PY_MODELS.is_dir()):
        _sibling_missing(
            f"both sibling SDK repos must be present to compare the two gates "
            f"({checker.SDK_PY_MODELS}, {checker.SDK_TS_MODELS})"
        )

    assert _pytest_gate_problems(checker) == [], "the SDKs are dirty before the mutation"
    real_parse = checker.parse_ts_sdk

    def mutated() -> dict[str, Any]:
        specs: dict[str, Any] = real_parse()
        fields = specs[model].fields
        if action == "drop":
            assert field_name in fields, f"{model}.{field_name} is not in the TS SDK any more"
            del fields[field_name]
        else:
            assert field_name not in fields
            fields[field_name] = checker.FieldSpec(
                name=field_name,
                kinds=frozenset({"string"}),
                required=False,
                source="z.string().optional()",
            )
        return specs

    monkeypatch.setattr(checker, "parse_ts_sdk", mutated)
    pytest_red = bool(_pytest_gate_problems(checker))
    cli_red = checker.check_sdks() == 1
    assert pytest_red == cli_red, (
        f"{model}.{field_name} ({action}): the two gates DISAGREE — "
        f"pytest {'red' if pytest_red else 'green'}, CLI {'red' if cli_red else 'green'}."
    )
    assert pytest_red == must_be_red, (
        f"{model}.{field_name} ({action}): expected both gates "
        f"{'RED' if must_be_red else 'GREEN'}, both were {'RED' if pytest_red else 'GREEN'}."
    )


# ======================================================================================
# A shared model is emitted in BOTH schema modes, and both are part of the wire
# ======================================================================================
def _doc_with_a_response_only_field(build: Any, model: str, field_name: str) -> Any:
    """The real generated document, with ``<model>.response`` split off into its own ``$defs``
    entry carrying one extra property.

    This is the shape pydantic produces the moment a shared model grows anything
    serialization-only (a ``@computed_field``, a ``Field(exclude=True)`` on the way in). Today
    both modes collapse to a single key — MEASURED at dev/mlm-build@eaf6c00, ``x-schema-key``
    maps ``RecallChannels.request`` and ``RecallChannels.response`` to the same ``$defs`` entry —
    so nothing in the real contract exercises the second lookup yet, and a regression there would
    be invisible without this.
    """
    doc = build()
    request_key = doc["x-schema-key"][f"{model}.request"]
    response_key = f"{model}__ResponseMode"
    node = json.loads(json.dumps(doc["$defs"][request_key]))  # deep copy, no aliasing
    node["properties"][field_name] = {"type": "string"}
    doc["$defs"][response_key] = node
    doc["x-schema-key"][f"{model}.response"] = response_key
    return doc


def test_a_shared_models_response_only_field_is_part_of_the_contract(
    checker: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``canonical_specs`` read ``<name>.request`` and NOTHING ELSE for a shared model.

    Two consequences, both fail-open, and this pins them shut:
      * a response-only field was never required of the SDK — the exact hole
        ``_CLOSED_ON_THE_WIRE`` was added to close, reopened one level down; and
      * an SDK that DID declare it was reported as INVENTING it, which is the failure mode that
        gets a gate switched off.
    """
    real = checker.build_json_schema
    monkeypatch.setattr(
        checker,
        "build_json_schema",
        lambda: _doc_with_a_response_only_field(real, "RecallChannels", "serialized_only"),
    )
    spec = checker.canonical_specs()["RecallChannels"]
    assert "serialized_only" in spec.fields, (
        "a shared model's response-mode schema is not being read: a field the server really "
        "serializes is invisible to the gate."
    )
    # ...and the union is a UNION, not a replacement: the request-mode fields survive.
    assert {"stm", "mtm", "ltm"} <= set(spec.fields)

    # An SDK that declares only the request-mode fields is now MISSING one. Under the shared
    # (== response) rule that is a FAILURE, and it must be reported as an ABSENCE — never as the
    # SDK inventing a field, which is what reading only `.request` would have produced for an SDK
    # that got it right. Asserted against a synthetic SDK spec rather than the parsed TS twin, so
    # this half is unconditional: it runs on a one-clone laptop with no sibling on disk.
    sdk = _spec(checker, "RecallChannels", "shared", stm="boolean", mtm="boolean", ltm="boolean")
    problems = checker.compare(
        "sdk", {"RecallChannels": spec}, {"RecallChannels": sdk}, ["RecallChannels"]
    )
    assert problems, "a response-only field the SDK does not declare passed the closed-schema rule"
    assert "missing field(s)" in problems[0] and "serialized_only" in problems[0], problems
    assert "declares field(s) the contract does not" not in " ".join(
        problems
    ), "reported as the SDK INVENTING a field — that is the reading-only-.request failure mode."


def test_the_two_schema_modes_of_every_shared_model_are_both_looked_up(
    checker: ModuleType,
) -> None:
    """The lookup table itself, so the fix cannot be undone by editing one line back."""
    for name, (_model, role) in checker.WIRE_MODELS.items():
        lookups = checker.canonical_lookups(name, role)
        if role == "shared":
            assert lookups == (f"{name}.request", f"{name}.response"), (
                f"{name} is shared: it is nested inside a response body AND accepted on a request "
                "body, so both emitted schemas are part of the wire."
            )
        else:
            assert lookups == (name,)


# ======================================================================================
# The skip-to-failure switch itself
# ======================================================================================
# Nothing tested `_sibling_missing`. It is the single point that decides whether a missing sibling
# is an honest report or a defect, it is driven by an environment variable NAME that is spelled
# again in two other repositories (`infra/mu-vm/vm_test.sh` exports it; `mu-core/.github/
# workflows/ci.yml` sets it as job env), and a typo in any of the three silently restores the
# green skip this whole module exists to end. These two tests are cheap and they run everywhere.
def test_a_missing_sibling_skips_when_the_run_was_never_declared_possible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-clone laptop must still be able to run mu-core's suite — mu-core depends on nothing."""
    monkeypatch.delenv(_REQUIRE_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception) as caught:
        _sibling_missing("mu-sdk-js is not checked out at /nowhere")
    assert "/nowhere" in str(caught.value)
    assert _REQUIRE_ENV in str(caught.value), "the skip must say how to make itself a failure"


@pytest.mark.parametrize("value", ["1", "true", "YES", " 1 "])
def test_a_missing_sibling_fails_loudly_when_the_run_was_declared_possible(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """`vm_test.sh` and mu-core CI both set this once they have actually put the siblings on disk.
    From that moment a "not checked out" report is a broken checkout, not a limitation — and the
    failure must NAME the path, so the next person fixes the checkout instead of the assertion.
    """
    monkeypatch.setenv(_REQUIRE_ENV, value)
    with pytest.raises(pytest.fail.Exception) as caught:
        _sibling_missing("mu-sdk-js is not checked out at /nowhere")
    assert "/nowhere" in str(caught.value), "the failure must name the path it looked in"
    assert "never the assertion" in str(caught.value)


@pytest.mark.parametrize("value", ["", "0", "no", "false", "MU_REQUIRE_SIBLING_REPOS"])
def test_only_a_real_truthy_value_arms_the_switch(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """A gate armed by any non-empty string would arm itself on `MU_REQUIRE_SIBLING_REPOS=0`, and a
    developer who set it that way to turn it OFF would get failures they cannot explain."""
    monkeypatch.setenv(_REQUIRE_ENV, value)
    with pytest.raises(pytest.skip.Exception):
        _sibling_missing("mu-sdk-js is not checked out at /nowhere")


def test_the_switch_name_matches_the_one_this_repos_own_ci_sets() -> None:
    """The switch is a STRING shared across three repositories, and only one of them is here.

    ``infra/mu-vm/vm_test.sh`` (the mu_project root repo) exports it once it has put both siblings
    on the VM; ``.github/workflows/ci.yml`` (this repo) sets it as job env once it has checked both
    siblings out; this module reads it. Rename it in one place and the other two go on setting a
    variable nobody reads — which does not fail, it silently restores the green skip. The tests
    above cannot catch that on their own: they reference ``_REQUIRE_ENV``, so a rename is
    self-consistent and they stay green. MEASURED: renaming the constant to
    ``MU_REQUIRE_SIBLING_REPOZ`` left the module at 34 passed.

    So the literal is pinned against the one other spelling that lives in this repo. vm_test.sh
    cannot be reached from a clean mu-core clone (different repository) and is named here instead.
    """
    workflow = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
    assert workflow.is_file(), (
        f"{workflow} is gone. The pytest half of the SDK-drift gate is only loud in CI because "
        "that workflow sets the switch and asserts zero skips; without it this module can go "
        "green having compared nothing."
    )
    text = workflow.read_text(encoding="utf-8")
    assert f"{_REQUIRE_ENV}:" in text, (
        f"{workflow.name} does not set {_REQUIRE_ENV}. CI checks both siblings out, so a "
        "'not checked out' skip there is a broken checkout — it must be a FAILURE, not a skip."
    )
