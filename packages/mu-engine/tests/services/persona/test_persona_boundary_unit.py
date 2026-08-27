"""The §0 / §2.4 / §5.4 BOUNDARY — where persona must NOT be (``persona-design.md``).

Spec §0 is titled *"Where persona sits (and where it must NOT)"*, and the negative half is the
half that fails silently: a persona subsystem that leaks onto the recall hot path still passes
every functional test in this directory. Every assertion below is structural, so it keeps holding
as the code changes rather than describing one moment.

Four rules, four shapes of test:

* **§2.4 line 122 — never on the hot path / never at warm time.** The scan is INVERTED: every
  module in ``mu_engine`` and ``mu_local`` is banned from importing ``mu_engine.services.persona``
  except the persona package itself and the enumerated composition roots. The first version of
  this file hand-picked nine roots and consequently scanned 91 of ``mu_engine``'s 137 modules —
  missing ``services/__init__.py`` (which eagerly imports ``services.ingest``, so a persona import
  there would have been transitively on every capture), ``platform/composition.py`` (where
  hot-path wiring is actually decided), ``providers/warm_local.py`` (the warm path spec line 122
  explicitly bans persona from) and all of ``mu_local`` (the FULL-LOCAL plane's own hot path).
  An allowlist of what may import persona cannot rot the way a denylist of where it may not does.
* **A static scan cannot see a bus subscription** — the leak that actually happened. So the
  runtime shape is asserted too: the incremental entry point is SYNCHRONOUS, which is what makes
  it structurally incapable of awaiting a store, a bus or a model on the capture stack.
* **§5.4 rules 1-2 — never access.** Persona never resolves into ``authorized_ids`` and never
  appears in a recall ``query_filter``: the persona package imports no recall/authz/vector-store
  module, ``PersonaService`` exposes no read of ANY kind (function, property, classmethod or
  staticmethod), and the ONLY persona read (``PersonaRepository.load_brief``) takes a namespace
  and nothing else.
* **§5.4 rule 3 — PRIVATE-only.** A persona cannot become a shared artifact.

**Out of reach from here, reported not faked:** ``mu-client`` (the on-device capture daemon) is a
separate repository and cannot be imported into this test session, so its own hot path is not
covered by this scan. The ``.importlinter`` ``persona-off-the-hot-path`` contract in this repo's
CI gates the same rule for ``mu_engine``/``mu_local``; mu-client needs the equivalent grep gate in
its own CI (PACKAGING-v2 §5.3, the same split the cross-plane contracts already live under).
"""

from __future__ import annotations

import ast
import inspect
import typing
from collections.abc import Iterable
from pathlib import Path

import pytest

import mu_engine
import mu_local
from mu_contracts.domain.model.memory import Namespace
from mu_contracts.ports.persona import PersonaRepository
from mu_engine.services import persona as persona_pkg
from mu_engine.services.persona.service import PersonaService

pytestmark = pytest.mark.unit

PERSONA_MODULE = "mu_engine.services.persona"

_ENGINE_ROOT = Path(mu_engine.__file__).parent
_LOCAL_ROOT = Path(mu_local.__file__).parent
_PERSONA_ROOT = Path(persona_pkg.__file__).parent

#: EVERY package whose modules may run while a user is waiting. Both root packages in full — the
#: denylist that used to live here is what let 46 modules through.
SCANNED_ROOTS: tuple[Path, ...] = (_ENGINE_ROOT, _LOCAL_ROOT)

#: The ONLY modules permitted to import persona: the package itself, and the composition roots
#: whose job is to name every subsystem. A composition root importing persona is how persona gets
#: BUILT; a hot-path module importing it is the defect. Adding a name here is a deliberate,
#: reviewable act — which is exactly the property the old denylist did not have.
IMPORT_ALLOWED: frozenset[Path] = frozenset(
    {
        _ENGINE_ROOT / "platform" / "composition.py",
        _ENGINE_ROOT / "platform" / "registries.py",
        _LOCAL_ROOT / "composition.py",
    }
)


def _python_files(roots: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts))
    return files


def _banned_files() -> list[Path]:
    return [
        path
        for path in _python_files(SCANNED_ROOTS)
        if _PERSONA_ROOT not in path.parents and path not in IMPORT_ALLOWED
    ]


def _module_dotted_path(path: Path) -> str:
    """``.../mu_engine/services/ingest.py`` -> ``mu_engine.services.ingest`` — needed to resolve a
    RELATIVE import, which is the spelling the first version of this detector could not see."""
    root = _ENGINE_ROOT if _ENGINE_ROOT in path.parents or path == _ENGINE_ROOT else _LOCAL_ROOT
    rel = path.relative_to(root.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve_relative(module: str, level: int, importer: str, is_package: bool) -> str:
    """Absolute form of a ``from ..x import y`` inside ``importer`` (PEP 328 resolution)."""
    base = importer.split(".") if is_package else importer.split(".")[:-1]
    if level > 1:
        base = base[: -(level - 1)] if level - 1 <= len(base) else []
    package = ".".join(base)
    return f"{package}.{module}" if module else package


def _imports_persona(source: str, *, importer: str = "", is_package: bool = False) -> bool:
    """Does this module reach the persona package, by ANY spelling?

    Three families, all of which a real leak could use and two of which the first version missed:
    absolute imports, RELATIVE imports (``from ..persona import ...`` — resolved through
    ``importer``), and NAME-BASED imports (``importlib.import_module(...)``, ``__import__``),
    which are caught by looking for the dotted path in any string literal.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_persona(alias.name) for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                if not importer:
                    # No package context: fall back to the tail-matching spelling so a bare
                    # snippet is still judged rather than silently passed.
                    if module.split(".")[-1:] == ["persona"] or any(
                        alias.name == "persona" for alias in node.names
                    ):
                        return True
                    continue
                module = _resolve_relative(module, node.level, importer, is_package)
            if _is_persona(module):
                return True
            # ``from <pkg> import persona`` — the module arrives as a NAME, not in ``node.module``.
            if any(_is_persona(f"{module}.{alias.name}") for alias in node.names):
                return True
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_persona(node.value):
                return True
    return False


def _is_persona(dotted: str) -> bool:
    return dotted == PERSONA_MODULE or dotted.startswith(f"{PERSONA_MODULE}.")


def _persona_importers(paths: Iterable[Path]) -> list[str]:
    offenders: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        if _imports_persona(
            source,
            importer=_module_dotted_path(path),
            is_package=path.name == "__init__.py",
        ):
            offenders.append(str(path))
    return offenders


# --------------------------------------------------- the detector itself (positive control)
_RECALL = "mu_engine.services.recall.service"
_INGEST = "mu_engine.services.ingest"
_HEALTH = "mu_engine.services.health.service"


@pytest.mark.parametrize(
    ("line", "importer"),
    [
        # ABSOLUTE.
        ("from mu_engine.services.persona.service import PersonaService", _RECALL),
        ("from mu_engine.services.persona import PersonaService", _RECALL),
        ("import mu_engine.services.persona", _RECALL),
        ("import mu_engine.services.persona.store as s", _RECALL),
        ("from mu_engine.services import persona", _RECALL),
        # RELATIVE — resolved against the importing module (PEP 328), the family the first
        # version of this detector could not see at all.
        ("from ..persona import PersonaService", _HEALTH),
        ("from . import persona", _INGEST),
        ("from ...services.persona.service import PersonaService", _HEALTH),
        ("from .persona import PersonaService", _INGEST),
        # NAME-BASED — no import statement at all.
        ("importlib.import_module('mu_engine.services.persona')", _RECALL),
        ("__import__('mu_engine.services.persona.service')", _RECALL),
        ("MOD = 'mu_engine.services.persona'", _RECALL),
    ],
)
def test_the_detector_catches_every_import_spelling(line: str, importer: str):
    """The scanner below is only worth its assertion if it actually detects a leak. Each spelling
    a real leak could use is checked against it explicitly — including the two families (relative
    and name-based) that the first version of this detector silently passed."""
    assert _imports_persona(f"{line}\n", importer=importer, is_package=False)


def test_a_relative_import_of_a_different_persona_is_not_a_false_positive():
    """``from . import persona`` inside ``mu_engine.services.recall.service`` names
    ``mu_engine.services.recall.persona`` — a different module. Resolution is real, not a
    tail-match, so the scan cannot be defeated by pointing at the wrong thing OR fire on it."""
    assert not _imports_persona("from . import persona\n", importer=_RECALL)


def test_the_detector_does_not_fire_on_unrelated_imports():
    assert not _imports_persona(
        "from mu_engine.services.health import HealthSettings\nimport json\n",
        importer="mu_engine.services.recall.service",
    )
    # A same-named module in another package must not be mistaken for ours.
    assert not _imports_persona(
        "from mu_contracts.ports.persona import PersonaRepository\n",
        importer="mu_engine.services.recall.service",
    )
    assert not _imports_persona(
        "from ..health import HealthSettings\n", importer="mu_engine.services.recall.service"
    )


def test_the_scanner_actually_found_source_files():
    """Rule-11 guard: a gitignore/path mistake that made the walk see zero files would turn the
    boundary assertion below into a vacuous pass. The floor is high enough to fail if a whole
    package silently dropped out of the walk (mu_engine alone is ~137 modules)."""
    banned = _banned_files()
    assert len(banned) > 120
    # Both planes are really in the walk, not just the engine.
    assert any(_LOCAL_ROOT in path.parents for path in banned)
    # ...and the modules the first version of this file missed are in it now.
    for missed in (
        _ENGINE_ROOT / "services" / "__init__.py",
        _ENGINE_ROOT / "providers" / "warm_local.py",
        _ENGINE_ROOT / "config" / "engine_settings.py",
        _LOCAL_ROOT / "local_memory.py",
    ):
        assert missed in banned, missed


def test_the_allowlist_holds_composition_roots_and_nothing_else():
    """The allowlist is the one way a module can be excused from the ban, so its CONTENT is part
    of the rule, not a convenience. Pinned exactly: widening it to let a hot-path module import
    persona has to fail here first, in a diff a reviewer reads, rather than passing silently.

    Every entry must also still exist — an allowlist entry naming a deleted module is an excuse
    nobody is checking.
    """
    assert IMPORT_ALLOWED == frozenset(
        {
            _ENGINE_ROOT / "platform" / "composition.py",
            _ENGINE_ROOT / "platform" / "registries.py",
            _LOCAL_ROOT / "composition.py",
        }
    )
    for path in IMPORT_ALLOWED:
        assert path.is_file(), path
        assert path.name in {"composition.py", "registries.py"}, path


# --------------------------------------------------------- §2.4 line 122 — off the hot path
def test_no_hot_path_module_imports_persona():
    """The rule that matters: if persona is reachable from recall/ingest/warm/local code, the
    wrong thing was built (spec §2.4 lines 117-122, preload §4.6). If you are here because you
    wired persona into a composition root, add that root to ``IMPORT_ALLOWED`` — deliberately."""
    assert _persona_importers(_banned_files()) == []


def test_the_incremental_entry_point_cannot_do_io():
    """The leak a static scan CANNOT see: ``MemoryPromoted`` is published inline on the ingest
    capture path (``pipelines/concrete/ingest.py:414`` -> ``services/ingest.py:275-276`` ->
    ``platform/adapters/bus_inproc.py:59-60``, which awaits every handler and propagates its
    exceptions to the publisher). An ``async def`` entry point there can await a store, a bus and
    the classifier behind ``PersonaEvidenceReader``, all inside the user's ``remember()``.

    A plain ``def`` cannot. This assertion is the gate on that: the day someone makes
    ``note_promoted`` async to "just fetch the item", this fails."""
    assert not inspect.iscoroutinefunction(PersonaService.note_promoted)
    # ...and the methods that DO touch the evidence reader stay on the sleep-time side.
    for sleep_time in (PersonaService.rebuild, PersonaService.refresh, PersonaService.forget):
        assert inspect.iscoroutinefunction(sleep_time)


def test_note_promoted_calls_no_collaborator():
    """Totality, structurally: the sync entry point must not reach ``self._repo`` / ``self._bus``
    / ``self._evidence`` / ``self._clock`` / ``self._metrics`` / ``self._tracer`` / ``self._audit``
    at all, so there is nothing in it that can raise into a user's capture. Read off the AST of
    the shipped method rather than trusted from a docstring."""
    source = inspect.getsource(PersonaService.note_promoted)
    tree = ast.parse(_dedent(source))
    touched = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    forbidden = {"_repo", "_bus", "_evidence", "_clock", "_metrics", "_tracer", "_audit"}
    assert touched & forbidden == set()


def _dedent(source: str) -> str:
    import textwrap

    return textwrap.dedent(source)


# ------------------------------------------------------------------ §5.4 rules 1-2 — not authz
def test_the_persona_package_imports_no_recall_authz_or_vector_module():
    """Persona cannot resolve into ``authorized_ids`` or reach a ``query_filter`` if it cannot
    name them (§5.4 rules 1-2)."""
    forbidden = ("recall", "authz", "qdrant", "authorized", "surface")
    offenders: dict[str, list[str]] = {}
    for path in sorted(_PERSONA_ROOT.glob("*.py")):
        modules = _imported_modules(path.read_text(encoding="utf-8"))
        hits = [m for m in modules if any(bad in m for bad in forbidden)]
        if hits:
            offenders[path.name] = hits
    assert offenders == {}


def _public_surface(klass: type) -> set[str]:
    """Every public name a caller can reach on ``klass``, by ANY descriptor kind.

    NOT ``inspect.getmembers(..., inspect.isfunction)``: that predicate sees neither ``@property``
    nor ``@classmethod`` nor ``@staticmethod``, and ``@property`` is this repo's own read-seam
    idiom (``surface/facade.py:178`` ``def bus(self)``) — so a public property named ``brief``
    would have been a read doorway with the assertion below still green. Shared by the rule and
    its positive control on purpose: one implementation, so weakening it fails both.
    """
    names: set[str] = set()
    for base in klass.__mro__:
        if base is object:
            continue
        names |= {name for name in vars(base) if not name.startswith("_")}
    return names


def test_persona_service_exposes_no_read():
    """Persona's query-time seam is ``PersonaRepository.load_brief``, on the contracts port — NOT
    a method on this service. A ``get``/``brief``/``load_for`` here would be the doorway a recall
    path walks through, so the public surface is asserted exactly.

    Enumerated by :func:`_public_surface`, which sees properties and classmethods too.
    """
    assert _public_surface(PersonaService) == {"rebuild", "refresh", "note_promoted", "forget"}


def test_a_public_property_or_classmethod_would_be_caught():
    """Positive control for the predicate above — the exact blind spot that made the first version
    of this assertion weaker than it read."""

    class WithDoorways(PersonaService):
        @property
        def brief(self) -> str:  # pragma: no cover - never called
            return ""

        @classmethod
        def load_for(cls) -> None:  # pragma: no cover - never called
            return None

        @staticmethod
        def peek() -> None:  # pragma: no cover - never called
            return None

    assert {"brief", "load_for", "peek"} <= _public_surface(WithDoorways)


def test_load_brief_takes_a_namespace_and_nothing_else():
    """§5.4 rule 2: the one persona read is a load BY KEY. It receives no query, no candidate set
    and no ``CallerIdentitySet``, so it structurally cannot become a filter condition."""
    hints = typing.get_type_hints(PersonaRepository.load_brief)
    sig = inspect.signature(PersonaRepository.load_brief)
    assert [p.name for p in sig.parameters.values()] == ["self", "ns"]
    assert hints["ns"] is Namespace


def test_persona_profile_is_private_only(shared_ns: Namespace):
    """§5.4 rule 3 / §3.1 line 149 — restated here because it is a firewall rule, not only a DTO
    detail: a persona cannot become a shared artifact, so it cannot leak one user's model into
    another's read."""
    from mu_contracts.domain.model.persona import PersonaProfile

    from .conftest import T0

    with pytest.raises(ValueError, match="PRIVATE-only"):
        PersonaProfile(
            namespace=shared_ns,
            slots={},
            overall_brief="",
            brief_etag="",
            version=1,
            rebuilt_at=T0,
            source_memory_count=0,
        )


# --------------------------------------------------------------------------------------- util
def _imported_modules(source: str) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append(node.module)
    return modules
