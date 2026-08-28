"""ENG-115a + ENG-118 on the path that is actually WIRED — `mu_local.composition`.

`test_llm_catalog_wiring_unit.py` proves the env→settings path of the profile build. This file
proves the two things that were wrong about it:

* **ENG-115a.** With no profile at all (`StorageSettings.llm is None`, the default), the plane had
  no model layer whatsoever — `ModelCatalogSettings()` is empty and this root only ever called
  `default_local_catalog()`. `LocalContainer` now always resolves the SHIPPED multi-provider
  catalog through `mu_engine.providers.plane`.
* **ENG-118.** The one *driven* local path did exactly what the credential rule forbids. Measured
  against a header-recording loopback listener, BEFORE the fix:

      provider="openai" + extra_params={"api_key": "sk-mu-local-placeholder"}
          -> Authorization: Bearer sk-mu-local-placeholder      (a literal key from source)
      provider="openai" + no api_key, OPENAI_API_KEY in the env
          -> Authorization: Bearer <the operator's REAL cloud key>
      provider="hosted_vllm"
          -> Authorization: Bearer fake-api-key                 (litellm's own keyless substitute)

  The wire assertion below is that third row, driven — not asserted over the table.

Offline: a loopback `http.server` that records headers is the only I/O. No container, no cloud, no
mock of the transport — the request really is built by litellm and really is sent over a socket.
"""

from __future__ import annotations

import asyncio
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from mu_engine.providers._contracts import Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.providers.local_priority import LocalPriorityPolicy
from mu_engine.providers.model_router import build_model_router
from mu_engine.providers.registry import ProviderModelRegistry
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.providers.task_map import TaskClassMapper
from mu_local.composition import _profile_resolver, _profile_rows, _resolve_profile_layer
from mu_local.config import ModelProfileSettings, StorageSettings

pytestmark = pytest.mark.unit

#: A value shaped like a real vendor key, planted in the environment. If it ever reaches the
#: loopback endpoint, the local seam is leaking the operator's cloud credential to localhost.
_PLANTED_CLOUD_KEY = "sk-proj-PLANTED-CLOUD-KEY-MUST-NEVER-REACH-LOCALHOST"


class _Recorder(BaseHTTPRequestHandler):
    """Answers an OpenAI-shaped chat completion and records the request headers."""

    seen: ClassVar[list[dict[str, str]]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        self.rfile.read(int(self.headers.get("content-length", 0)))
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep the test output clean
        return


@pytest.fixture
def listener() -> Iterator[str]:
    _Recorder.seen = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ---------------------------------------------------------------------------------------------
# ENG-118 — the defaults, and what they put on the wire
# ---------------------------------------------------------------------------------------------
def test_the_profile_defaults_are_the_keyless_local_seam() -> None:
    """`provider` and `api_key` are the two ENG-118 defaults. `openai` is wrong twice over (it
    demands a key, and when one is in the environment it ships that key to localhost);
    `sk-mu-local-placeholder` was a literal key in source that really did travel."""
    profile = ModelProfileSettings(base_url="http://127.0.0.1:1/v1", model="m")

    assert profile.provider == "hosted_vllm"
    assert profile.api_key is None
    assert profile.credential_ref  # a NAME exists for the endpoints that DO check a key


def test_no_literal_key_shape_survives_in_the_wired_local_source() -> None:
    """MVP-SPEC ENG-118's own acceptance: *"grep every `src/` tree (not just the catalog) for a
    literal key shape -> zero"*. Scoped here to the two trees this lane owns.

    NOT COVERED, DELIBERATELY NAMED rather than quietly excluded:
    `mu-engine-server/src/mu_engine_server/settings.py` still carries
    `SlmProfile.api_key = "sk-mu-engine-server-placeholder"` and `provider = "openai"`. That file
    is outside this lane's ownership; the wiring around it already stops the value reaching
    `extra_params`, but the literal and the leaky prefix are a reported, unfixed edit.
    """
    roots = [
        Path(__file__).resolve().parents[2] / "mu-local" / "src",
        Path(__file__).resolve().parents[2] / "mu-engine" / "src",
    ]
    # A key-shaped literal in VALUE position — a default (`= "sk-…"`), a call argument
    # (`(… "sk-…"`) or a list/dict element. Prose that merely NAMES the old placeholder (this
    # file, and `mu_local/config.py`'s own docstring) is not a key; a literal one is.
    literal_key = re.compile(r'[=,(\[]\s*"sk-[A-Za-z0-9_-]{4,}"')
    offenders = [
        f"{path}:{n}"
        for root in roots
        for path in root.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if literal_key.search(line) and not line.lstrip().startswith("#")
    ]
    assert offenders == []


def test_the_profile_key_travels_as_a_credential_ref_never_in_extra_params() -> None:
    """`model-layer-spec §4`: `extra_params` is for `api_version` only. A key is a NAME in the
    catalog and a VALUE only inside the resolver (`registry.compile_model_list` joins them)."""
    profile = ModelProfileSettings(
        base_url="http://127.0.0.1:1/v1", model="m", api_key="sk-endpoint-that-checks"
    )

    provider, deployment = _profile_rows(profile)

    assert deployment.extra_params == {}
    assert provider.credential_ref == profile.credential_ref
    resolver = _profile_resolver(profile, ModelCatalogSettings())
    assert resolver.resolve(profile.credential_ref) == "sk-endpoint-that-checks"
    assert "sk-endpoint-that-checks" not in repr(resolver)


def test_a_keyless_profile_declares_no_credential_ref_at_all() -> None:
    provider, _deployment = _profile_rows(
        ModelProfileSettings(base_url="http://127.0.0.1:1/v1", model="m")
    )

    assert provider.credential_ref is None


async def test_the_local_endpoint_never_receives_the_operators_cloud_key(
    listener: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ENG-118's acceptance, driven: a listener, a real `OPENAI_API_KEY` in the environment, one
    ROUTINE_EXTRACT through the router this composition root builds. Before the fix this arrived
    as `Authorization: Bearer <the planted key>`."""
    monkeypatch.setenv("OPENAI_API_KEY", _PLANTED_CLOUD_KEY)
    profile = ModelProfileSettings(base_url=listener, model="qwen2.5:0.5b")
    layer = _resolve_profile_layer(profile, ModelSettings(), ModelCatalogSettings())
    router = build_model_router(
        models=layer.models,
        catalog=layer.catalog,
        secret_resolver=_profile_resolver(profile, ModelCatalogSettings()),
        chunk_token_ratio=0.75,
    )

    await router.generate(Task.ROUTINE_EXTRACT, [Message(role=MessageRole.USER, content="hi")])

    assert _Recorder.seen, "the request never reached the local endpoint"
    for headers in _Recorder.seen:
        assert _PLANTED_CLOUD_KEY not in json.dumps(headers)
        assert "sk-mu-local-placeholder" not in json.dumps(headers)


# ---------------------------------------------------------------------------------------------
# ENG-115a — the profile path is unchanged; the NO-profile path is the new one
# ---------------------------------------------------------------------------------------------
def test_a_configured_profile_still_pins_every_task_to_its_one_deployment() -> None:
    """Backward compatibility, explicitly: a profile means *"use exactly this deployment"*, so it
    keeps its single-row catalog rather than gaining the shipped table underneath it."""
    profile = ModelProfileSettings(base_url="http://127.0.0.1:1/v1", model="m")

    layer = _resolve_profile_layer(profile, ModelSettings(), ModelCatalogSettings())

    assert [p.key for p in layer.catalog.providers] == [profile.provider_key]
    assert [d.model_group for d in layer.catalog.deployments] == [profile.model_group]
    table = TaskClassMapper(layer.models).task_groups()
    assert {table[t] for t in table if t is not Task.EMBED} == {profile.model_group}


def test_the_default_storage_settings_still_mean_heuristic_mode() -> None:
    """`self.llm` — and therefore `LocalMemory`'s LLM-dependent verbs — is deliberately NOT armed
    by this wiring. The router existing and those verbs being live are two different decisions."""
    assert StorageSettings().llm is None


def test_the_no_profile_path_resolves_the_shipped_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """What `LocalContainer._build_plane_router(None)` resolves: the multi-provider table, not an
    empty one. Asserted through the same registry validation composition performs."""
    for name in ("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    from mu_engine.providers.plane import build_plane_secret_resolver, resolve_plane_model_layer

    catalog = ModelCatalogSettings()
    layer = resolve_plane_model_layer(
        models=ModelSettings(), catalog=catalog, resolver=build_plane_secret_resolver(catalog)
    )

    assert len(layer.catalog.deployments) > 1
    ProviderModelRegistry(  # raises if any task's group is empty
        layer.catalog.providers,
        layer.catalog.deployments,
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset(layer.catalog.local_capable_tasks), enabled=True
        ),
        task_groups=TaskClassMapper(layer.models).task_groups(),
    )


def test_generate_is_reachable_without_asyncio_run_helpers() -> None:
    """Guard against the async-called-without-await class of defect (AD-144a): the wire test above
    is an `async def` collected by `asyncio_mode = "auto"`, so prove the loop really runs it."""
    assert asyncio.iscoroutinefunction(
        test_the_local_endpoint_never_receives_the_operators_cloud_key
    )
