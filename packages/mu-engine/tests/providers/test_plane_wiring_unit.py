"""ENG-115a — the shipped catalog is WIRED: `mu_engine.providers.plane`.

The gap these tests close is a wiring gap, not a table gap. `test_shipped_catalog_unit.py` already
proves the ~915-line table is correct and callable; what nothing proved is that anything *builds*
it. Measured at HEAD before `plane.py` existed::

    EngineSettings().model_catalog.deployments == []
    build_model_router(models=EngineSettings().model,
                       catalog=default_local_catalog(EngineSettings().model_catalog))
    -> RegistryError: task 'answer' maps to model-group 'gpt-5-chat' which has no deployment

So every assertion below is deliberately made over the layer a COMPOSITION ROOT resolves
(`resolve_plane_model_layer` / `build_plane_router`), never over `shipped_catalog()` — MVP-SPEC
§5.1 ENG-115a is explicit that the criterion which can fail is *"the container's actual router"*.

Offline by construction: pydantic objects, the L1 registry (the exact validation
`build_model_router` performs), and — for the wire assertions — a loopback HTTP listener that
records headers. No cloud, no store, no weights.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from structlog.testing import capture_logs

from mu_engine.providers.catalog import ProviderKind, Task
from mu_engine.providers.local_priority import LocalPriorityPolicy
from mu_engine.providers.plane import (
    build_plane_router,
    build_plane_secret_resolver,
    resolve_plane_model_layer,
    resolve_task_models,
)
from mu_engine.providers.registry import ProviderModelRegistry, RegistryError
from mu_engine.providers.secrets import SecretSeamResolver
from mu_engine.providers.settings import (
    CatalogSource,
    LocalFallbackPosture,
    ModelCatalogSettings,
    ModelSettings,
    TaskDefaults,
)
from mu_engine.providers.shipped_catalog import ModelGroup, ProviderKey
from mu_engine.providers.task_map import TaskClassMapper

pytestmark = pytest.mark.unit

#: Every vendor env var the shipped table's `credential_ref`s upper-case to. A developer box very
#: often carries one of these, and a test whose result depends on the developer's shell is not a
#: test — so the keyless cases scrub them explicitly.
_VENDOR_ENV = (
    "AZURE_OPENAI_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
)


@pytest.fixture
def keyless(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A box with NO cloud credentials anywhere — the FULL-LOCAL posture (gate G6)."""
    for name in _VENDOR_ENV:
        monkeypatch.delenv(name, raising=False)
    yield


def _layer(catalog: ModelCatalogSettings, models: ModelSettings | None = None):
    return resolve_plane_model_layer(
        models=models if models is not None else ModelSettings(),
        catalog=catalog,
        resolver=build_plane_secret_resolver(catalog),
    )


def _registry(layer) -> ProviderModelRegistry:
    """The EXACT validation `build_model_router` performs (registry.py:65) — it raises if any
    task's model-group has no deployment, which is the whole question here."""
    return ProviderModelRegistry(
        layer.catalog.providers,
        layer.catalog.deployments,
        local_policy=LocalPriorityPolicy(
            local_capable_tasks=frozenset(layer.catalog.local_capable_tasks),
            enabled=layer.catalog.local_priority_enabled,
        ),
        task_groups=TaskClassMapper(layer.models).task_groups(),
        secret_resolver=build_plane_secret_resolver(layer.catalog),
    )


# ---------------------------------------------------------------------------------------------
# A. gate G6 — complete and good with zero credentials
# ---------------------------------------------------------------------------------------------
def test_a_keyless_plane_resolves_every_task_to_a_reachable_deployment(keyless: None) -> None:
    """Gate G6's falsifiable half, asserted over the layer a composition root resolves.

    Before the wiring this was `RegistryError: task 'answer' maps to model-group 'gpt-5-chat'
    which has no deployment`; the identical claim in `test_shipped_catalog_unit.py:214-218` was
    made over `shipped_catalog()`, a table nothing called.
    """
    layer = _layer(ModelCatalogSettings())

    served: dict[str, list[str]] = {}
    for dep in layer.catalog.deployments:
        served.setdefault(dep.model_group, []).append(dep.model_id)

    table = TaskClassMapper(layer.models).task_groups()
    for task, group in table.items():
        assert served.get(group), f"task {task.value} -> group {group} has no deployment"

    _registry(layer)  # raises if any task group is empty — the composition-time check itself

    # ...and every surviving provider is LOCAL: nothing here needs a key or a non-loopback socket.
    assert {p.key for p in layer.catalog.providers} == {
        ProviderKey.LOCAL_OPENAI_HTTP.value,
        ProviderKey.LOCAL_EMBED_RERANK_HTTP.value,
    }
    assert all(p.credential_ref is None for p in layer.catalog.providers)
    assert all(p.kind is ProviderKind.LOCAL_HTTP for p in layer.catalog.providers)
    assert layer.active_credential_refs == ()


def test_the_keyless_posture_is_a_named_event_never_a_silent_flip(keyless: None) -> None:
    """The shipped table deliberately leaves the adjudicating groups EMPTY on a keyless box
    (ADR 0037: degrade to the deterministic heuristic, never to a weaker model), and
    `test_shipped_catalog_unit.py::test_without_the_flag_a_keyless_box_fails_loud_on_the_hard_tier`
    pins that. A plane that must BOOT with zero keys has to make a choice; this asserts the choice
    is announced with its ADR consequence named, not taken quietly."""
    with capture_logs() as events:
        layer = _layer(ModelCatalogSettings())

    named = [e for e in events if e["event"] == "model_layer_no_remote_credentials_posture"]
    assert len(named) == 1
    assert named[0]["posture"] == LocalFallbackPosture.AUTO.value
    assert named[0]["resolved_credential_refs"] == 0
    assert named[0]["adjudication"] == "local_serves_adjudicating_groups"
    assert layer.local_fallback_adopted is True
    # content-free: names and counts only, never a credential VALUE
    assert not any("sk-" in str(value) for event in events for value in event.values())


def test_never_posture_keeps_the_designed_fail_loud(keyless: None) -> None:
    """The other half of the same decision, still reachable by one env var: `NEVER` restores the
    shipped default, so a keyless box refuses to start rather than adjudicate on a local model."""
    layer = _layer(ModelCatalogSettings(local_fallback=LocalFallbackPosture.NEVER))

    assert layer.local_fallback_adopted is False
    with pytest.raises(RegistryError) as err:
        _registry(layer)
    assert ModelGroup.REASON_HARD.value in str(err.value)


def test_a_resolved_credential_suppresses_the_local_primary_posture(
    monkeypatch: pytest.MonkeyPatch, keyless: None
) -> None:
    """AUTO is evidence-driven, not a constant: one real remote credential and the hard tier is
    frontier again, exactly as ADR 0037 wants."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-only")

    layer = _layer(ModelCatalogSettings())

    assert layer.local_fallback_adopted is False
    assert layer.active_credential_refs == ("anthropic_api_key",)
    hard = {
        d.model_id for d in layer.catalog.deployments if d.model_group == ModelGroup.REASON_HARD
    }
    assert hard == {"anthropic/claude-opus-4-1"}  # no local row adjudicates
    _registry(layer)


# ---------------------------------------------------------------------------------------------
# B. per-task selection — principled defaults, and the operator always wins
# ---------------------------------------------------------------------------------------------
def test_unset_task_fields_adopt_the_logical_groups(keyless: None) -> None:
    """Per-task selection on capability / cost / latency / context / local-vs-remote (Phase 4)."""
    layer = _layer(ModelCatalogSettings())

    table = TaskClassMapper(layer.models).task_groups()
    assert table[Task.ANSWER] == ModelGroup.CHAT.value
    assert table[Task.ADJUDICATE] == ModelGroup.REASON_HARD.value
    assert table[Task.ROUTINE_EXTRACT] == ModelGroup.EXTRACT_FAST.value
    assert table[Task.SUMMARIZE] == ModelGroup.SUMMARIZE.value
    assert table[Task.CLASSIFY] == ModelGroup.CLASSIFY.value
    # the pre-existing `rerank_model="gpt-4.1-mini"` placeholder — a CHAT group no provider can
    # serve `arerank` from — is replaced by the real cross-encoder group.
    assert table[Task.RERANK] == ModelGroup.RERANK.value
    # the EmbeddingPort seam (§6-P5) is a different mechanism and is deliberately untouched
    assert layer.models.embed_backend == ModelSettings().embed_backend


def test_an_operator_set_task_field_is_never_overwritten(keyless: None) -> None:
    """`ModelSettings` stays the ONE user-configurable seam (CANONICAL §7.2). `model_fields_set`
    is the discriminator, and pydantic-settings populates it from the env keys that were present,
    so `MU_MODEL__CLASSIFY_MODEL=...` survives while its unset siblings adopt a `ModelGroup`."""
    chosen = ModelSettings(classify_model=ModelGroup.CHAT.value)
    assert chosen.model_fields_set == {"classify_model"}

    layer = _layer(ModelCatalogSettings(), chosen)

    table = TaskClassMapper(layer.models).task_groups()
    assert table[Task.CLASSIFY] == ModelGroup.CHAT.value  # the operator's, untouched
    assert table[Task.SUMMARIZE] == ModelGroup.SUMMARIZE.value  # the unset one, adopted


def test_legacy_task_defaults_leave_model_settings_exactly_alone() -> None:
    """`TaskDefaults.LEGACY` is a pure no-op on the task map — the escape hatch is real."""
    models = ModelSettings()
    assert resolve_task_models(models, task_defaults=TaskDefaults.LEGACY) is models


def test_source_empty_reproduces_the_pre_wiring_posture(keyless: None) -> None:
    """One env var (`MU_MODEL_CATALOG__SOURCE=empty`) puts the plane back exactly where it was —
    and does NOT then apply a task map whose groups only exist in the table it just turned off."""
    layer = _layer(ModelCatalogSettings(source=CatalogSource.EMPTY))

    assert layer.catalog.deployments == []
    assert layer.catalog.providers == []
    assert layer.models == ModelSettings()  # legacy gpt-* names, untouched
    assert set(layer.catalog.embedders) == {"minilm_local"}  # the embedder seam still wired


def test_operator_rows_are_overlaid_on_the_shipped_table_not_replaced_by_it(
    keyless: None,
) -> None:
    """`ModelCatalogSettings.providers`/`.deployments` are the operator's own explicit rows; the
    shipped table is a floor under them, never a replacement for them."""
    from mu_engine.providers.catalog import ModelDeployment, ProviderRecord

    own = ProviderRecord(
        key="my_endpoint",
        kind=ProviderKind.LOCAL_HTTP,
        litellm_provider="hosted_vllm",
        api_base="http://127.0.0.1:9/v1",
        is_local=True,
    )
    row = ModelDeployment(
        model_group="my-group", provider_key="my_endpoint", model_id="hosted_vllm/mine"
    )
    layer = _layer(ModelCatalogSettings(providers=[own], deployments=[row]))

    assert "my_endpoint" in {p.key for p in layer.catalog.providers}
    assert "my-group" in {d.model_group for d in layer.catalog.deployments}
    assert ProviderKey.LOCAL_OPENAI_HTTP.value in {p.key for p in layer.catalog.providers}


# ---------------------------------------------------------------------------------------------
# C. ENG-118 — a key is a NAME in the catalog and a VALUE only inside the resolver
# ---------------------------------------------------------------------------------------------
def test_no_deployment_carries_a_credential_in_extra_params(keyless: None) -> None:
    """`model-layer-spec §4`: `extra_params` is for `api_version` ONLY. This is the assertion the
    earlier draft scoped to `shipped_catalog.py` and so could not fail on the live path — it runs
    over the RESOLVED plane catalog, which is the live path."""
    layer = _layer(ModelCatalogSettings())

    for dep in layer.catalog.deployments:
        assert "api_key" not in dep.extra_params, dep.model_group
        assert not any("key" in name.lower() for name in dep.extra_params), dep.model_group


def test_the_secret_seam_reads_a_file_then_the_environment(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "anthropic_api_key").write_text("from-the-file\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "from-the-env")
    resolver = SecretSeamResolver(secrets_dir=tmp_path, use_env=True)

    assert resolver.resolve("anthropic_api_key") == "from-the-file"  # trailing newline stripped
    assert resolver.resolve("openai_api_key") == "from-the-env"
    with pytest.raises(RegistryError):
        resolver.resolve("moonshot_api_key")


def test_the_secret_seam_never_reveals_a_value_and_cannot_escape_its_directory(
    tmp_path,
) -> None:
    """Two separate claims, and the escape half has to be able to FAIL.

    The first draft of this test asserted only ``pytest.raises(RegistryError)`` on
    ``"../../etc/passwd"`` — which a resolver with NO traversal guard also satisfies, because the
    traversed path does not exist and the miss raises the same error. Mutation-tested: deleting
    the guard in ``secrets.py`` left this file at 24 passed. So the escape target is now a file
    that REALLY EXISTS one directory above the seam, and the error is matched on the guard's own
    message — an unguarded resolver returns its contents and never raises.
    """
    seam = tmp_path / "secrets"
    seam.mkdir()
    (tmp_path / "outside_secret").write_text("sk-escaped-the-seam\n", encoding="utf-8")
    resolver = SecretSeamResolver(
        overrides={"k": "sk-super-secret"}, secrets_dir=seam, use_env=False
    )

    assert "sk-super-secret" not in repr(resolver)
    assert "sk-super-secret" not in str(resolver)
    with pytest.raises(RegistryError, match="not a valid secret name"):
        resolver.resolve("../outside_secret")
    with pytest.raises(RegistryError, match="not a valid secret name"):
        resolver.resolve("../../etc/passwd")


def test_an_override_value_reaches_litellm_through_the_credential_ref_seam(
    keyless: None,
) -> None:
    """The whole point of the seam: the catalog holds a NAME, the compiled `litellm_params` holds
    the VALUE, and the value came from the resolver — never from a row in the table."""
    monkey = SecretSeamResolver(overrides={"anthropic_api_key": "sk-ant-from-the-seam"})
    layer = resolve_plane_model_layer(
        models=ModelSettings(), catalog=ModelCatalogSettings(), resolver=monkey
    )

    assert layer.active_credential_refs == ("anthropic_api_key",)
    registry = ProviderModelRegistry(
        layer.catalog.providers,
        layer.catalog.deployments,
        local_policy=LocalPriorityPolicy(local_capable_tasks=frozenset(), enabled=False),
        task_groups=TaskClassMapper(layer.models).task_groups(),
        secret_resolver=monkey,
    )
    rows = [r for r in registry.compile_model_list() if r["model_name"] == ModelGroup.REASON_HARD]
    assert rows and all(r["litellm_params"]["api_key"] == "sk-ant-from-the-seam" for r in rows)


def test_a_keyless_local_row_compiles_with_no_api_key_at_all(keyless: None) -> None:
    """`hosted_vllm` substitutes `"fake-api-key"` when none is configured
    (`litellm/llms/hosted_vllm/chat/transformation.py:125`), which is what lets the local seam be
    credential-free on the WIRE. Declaring it `openai` instead would make litellm reach for the
    operator's real `OPENAI_API_KEY` — measured, see `mu_local/config.py`'s `provider` comment."""
    layer = _layer(ModelCatalogSettings())

    assert all(p.litellm_provider == "hosted_vllm" for p in layer.catalog.providers)
    for row in _registry(layer).compile_model_list():
        assert "api_key" not in row["litellm_params"], row["model_name"]


# ---------------------------------------------------------------------------------------------
# D. the router itself
# ---------------------------------------------------------------------------------------------
@pytest.mark.needs_local_model
def test_build_plane_router_constructs_a_real_router_with_zero_credentials(
    keyless: None,
) -> None:
    """The end of the wiring: what a composition root calls, on a box with nothing configured.
    The MiniLM embedder is REAL (the warm singleton, offline from the local HF cache)."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    catalog = ModelCatalogSettings()

    router = build_plane_router(
        models=ModelSettings(),
        catalog=catalog,
        chunk_token_ratio=0.75,
        resolver=build_plane_secret_resolver(catalog),
    )

    assert router.dimension > 0
    assert router.model_name
