"""The ONE composition entry point for a plane's model layer (ENG-115a).

**What was missing.** Every piece of the multi-provider model layer existed and none of it was
connected. `shipped_catalog.py` declares the whole Phase-4 table (anthropic · openai · azure ·
deepseek · moonshot · a keyless local chat endpoint · a local embed+rerank endpoint · the warm
in-process singleton) and its only non-test callers were `__init__.py` re-exports;
`ModelCatalogSettings()` defaults to EMPTY providers/deployments; both composition roots called
`default_local_catalog(...)`, which layers one offline MiniLM embedder and nothing else. Measured
at HEAD, before this module existed::

    EngineSettings().model_catalog.deployments == []
    build_model_router(models=EngineSettings().model, catalog=default_local_catalog(...))
    -> RegistryError: task 'answer' maps to model-group 'gpt-5-chat' which has no deployment

So the plane shipped **one embedder and zero LLM deployments** — against `CLAUDE.md:18-27`
(*"FULL-LOCAL is a complete, good, on-device system — never a crippled baseline"*) and gate **G6**.

**What this module is.** The four steps between "settings" and "a `ModelRouter`", in one place, so
each composition root is one call rather than four opinions:

  1. **base** — `default_local_catalog(settings.model_catalog)`, so the embedder seam and every
     `MU_MODEL_CATALOG__*` subtree flow through unchanged (CONFIG-AND-DATA-FIX-PLAN §1.2 C2);
  2. **declare** — the shipped table (`CatalogSource.SHIPPED`, the default) with the operator's own
     `providers`/`deployments` overlaid on top, never replaced by it;
  3. **probe + activate** — ask the secret seam which `credential_ref`s actually resolve
     (`resolvable_credential_refs`) and narrow the table to them (`active_catalog`). A provider
     with no `credential_ref` — every local one — always survives, which is what makes a
     zero-credential box a complete system rather than an empty one. **A resolving credential is
     not a live deployment (AD-205):** the same call also narrows to the rows whose deployment
     the operator says EXISTS (`ShippedCatalogSettings.known_deployments` →
     `DeclaredDeploymentProbe`), because a key admits an account, not a model — and a row naming
     an undeployed model kept its group nominally populated, which suppressed the very fallback
     chain declared to cover that group being unavailable;
  4. **task map** — point the per-task fields the operator did NOT set at the logical
     `ModelGroup`s (`recommended_model_settings`), chosen on capability / cost / latency / context
     / local-vs-remote. A field the operator set anywhere — in code, or via `MU_MODEL__*` — is
     never overwritten: `ModelSettings` stays the ONE user-configurable seam (CANONICAL §7.2).

**The keyless-box decision, made explicitly.** With zero cloud credentials the shipped table's
`mu-reason-hard` / `gpt-5-chat` groups are empty by design (ADR 0037 keeps the adjudicator
frontier-only) and `ProviderModelRegistry` then refuses to start — pinned by
`test_shipped_catalog_unit.py::test_without_the_flag_a_keyless_box_fails_loud_on_the_hard_tier`.
The catalog ships its own answer for that box, `local_serves_remote_preferred_groups`, whose
docstring calls it *"the posture of a box with NO remote credentials at all"*. This module makes
adopting it **conditional on the probe and NAMED**: `LocalFallbackPosture.AUTO` (the default)
adopts it only when no remote deployment can actually serve — measured as *a remote row whose
credential resolved AND whose deployment is not known-absent*, not as "a credential resolved"
(AD-205) — and emits `model_layer_no_remote_credentials_posture`. For the adjudicating groups
that is the deliberate,
logged ADR 0037 deviation the flag itself describes; `NEVER` keeps the fail-loud original and
`ALWAYS` forces local primacy. It is a config field with three named values, not an accident.

Nothing here re-architects: the many-to-many catalog, the config-driven task map and the
credential-aware activation were all already built (MVP-BUILD-PLAN Phase 4 — *"a catalog/config
expansion … no rearchitecture"*). This module is the caller they never had.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import structlog
from pydantic import BaseModel, ConfigDict, SecretStr

from mu_engine.providers._contracts import DegradeEmitterPort, EmbeddingPort
from mu_engine.providers.catalog import ModelDeployment, ProviderRecord
from mu_engine.providers.model_router import ModelRouter, build_model_router
from mu_engine.providers.secrets import SecretSeamResolver
from mu_engine.providers.settings import (
    CatalogSource,
    LocalFallbackPosture,
    ModelCatalogSettings,
    ModelSettings,
    TaskDefaults,
    default_local_catalog,
)
from mu_engine.providers.shipped_catalog import (
    CredentialProbe,
    DeclaredDeploymentProbe,
    DeploymentProbe,
    active_catalog,
    recommended_model_settings,
    resolvable_credential_refs,
    shipped_catalog,
)

__all__ = [
    "PlaneModelLayer",
    "build_plane_router",
    "build_plane_secret_resolver",
    "resolve_plane_model_layer",
    "resolve_task_models",
]

log = structlog.get_logger("mu_engine.providers")


class PlaneModelLayer(BaseModel):
    """What a plane's settings resolve to, before a `Router` is constructed from it.

    Returned rather than swallowed so a composition root — or a test asserting gate G6 — can look
    at the ACTIVE table (*"the container's real router"*, MVP-SPEC §5.1 ENG-115a) instead of at
    `shipped_catalog()`, the table nothing wired.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: ModelSettings
    catalog: ModelCatalogSettings
    #: `credential_ref` NAMES that resolved — never values (house rule 1). Empty on a keyless box.
    active_credential_refs: tuple[str, ...] = ()
    #: True when the no-remote-credentials posture was adopted (see the module docstring).
    local_fallback_adopted: bool = False


def build_plane_secret_resolver(
    catalog: ModelCatalogSettings,
    *,
    overrides: Mapping[str, str | SecretStr] | None = None,
) -> SecretSeamResolver:
    """The credential probe a composition root passes in — built FROM the settings tree.

    `secrets_dir` and `credentials_from_env` are `ModelCatalogSettings` fields, so
    `MU_MODEL_CATALOG__SECRETS_DIR=/run/secrets` is all a deployment needs to bind the file seam;
    `overrides` is for a root that was handed a value by its own outer seam.
    """
    return SecretSeamResolver(
        overrides=overrides,
        secrets_dir=catalog.secrets_dir,
        use_env=catalog.credentials_from_env,
    )


def resolve_task_models(
    models: ModelSettings, *, task_defaults: TaskDefaults = TaskDefaults.RECOMMENDED
) -> ModelSettings:
    """Apply the recommended per-task defaults WITHOUT overwriting anything the operator set.

    `ModelSettings.model_fields_set` is the discriminator, and it is exactly right for this:
    pydantic-settings populates it from the env keys that were present, so
    `MU_MODEL__ANSWER_MODEL=gpt-4o` lands in `model_fields_set` and survives, while a field nobody
    mentioned does not and adopts its `ModelGroup`. Returns a FRESH object; `models` is never
    mutated.
    """
    if task_defaults is TaskDefaults.LEGACY:
        return models
    operator_set = {field: getattr(models, field) for field in models.model_fields_set}
    return recommended_model_settings(models).model_copy(update=operator_set)


def _merge(
    shipped: Sequence[ProviderRecord], overlay: Sequence[ProviderRecord]
) -> list[ProviderRecord]:
    """Overlay provider rows onto the shipped ones, keyed by `ProviderRecord.key` — an operator's
    own row for a key the table also ships WINS (it is the more specific statement of intent)."""
    by_key = {p.key: p for p in shipped}
    by_key.update({p.key: p for p in overlay})
    return list(by_key.values())


def resolve_plane_model_layer(
    *,
    models: ModelSettings,
    catalog: ModelCatalogSettings,
    resolver: CredentialProbe | None = None,
    overlay_providers: Sequence[ProviderRecord] = (),
    overlay_deployments: Sequence[ModelDeployment] = (),
) -> PlaneModelLayer:
    """Settings -> the ACTIVE catalog + the resolved task map (the four steps above).

    `overlay_providers`/`overlay_deployments` are rows the composition root itself constructs —
    e.g. `mu-local`'s configured `ModelProfileSettings` SLM. They are appended AFTER the shipped
    table and after `catalog.providers`/`catalog.deployments`, and they are activated by the same
    credential probe as everything else (a local overlay row carries no `credential_ref`, so it
    always survives).

    Never mutates `models` or `catalog`.
    """
    base = default_local_catalog(catalog)
    operator_providers = _merge(catalog.providers, overlay_providers)
    operator_deployments = [*catalog.deployments, *overlay_deployments]

    if catalog.source is CatalogSource.EMPTY:
        # The pre-2026-08-28 posture, kept reachable by one env var. The recommended task map is
        # NOT applied here: its `ModelGroup` names exist only in the shipped table, so applying it
        # to an empty catalog would fail composition for someone who only wanted the table off.
        declared = base.model_copy(
            update={"providers": operator_providers, "deployments": operator_deployments}
        )
        task_models = resolve_task_models(models, task_defaults=TaskDefaults.LEGACY)
        return _activate(
            declared,
            task_models,
            available=_probe(declared, resolver),
            adopted=False,
            deployment_probe=DeclaredDeploymentProbe(catalog.shipped.known_deployments),
        )

    cfg = catalog.shipped
    declared = shipped_catalog(base, cfg=cfg)
    available = _probe(declared, resolver)
    adopted = False

    dep_probe = DeclaredDeploymentProbe(cfg.known_deployments)
    remote_refs = {p.credential_ref for p in declared.providers if p.credential_ref is not None}
    # AD-205: the AUTO posture used to key off "did a remote CREDENTIAL resolve". That is the
    # wrong question, and answering it made a valid key strictly worse than none: the key
    # resolved, so the box was declared remote-capable and the local posture was NOT adopted --
    # while every remote row the key admitted named a deployment that does not exist. The
    # question AUTO actually means is "can any remote deployment serve this box", so ask that.
    serving_remote = _surviving_remote_rows(declared, available=available, probe=dep_probe)
    wants_local_primary = catalog.local_fallback is LocalFallbackPosture.ALWAYS or (
        catalog.local_fallback is LocalFallbackPosture.AUTO and not serving_remote
    )
    if wants_local_primary and not cfg.local_serves_remote_preferred_groups:
        # NAMED, content-free, and stating the ADR 0037 consequence explicitly — never silent.
        log.info(
            "model_layer_no_remote_credentials_posture",
            posture=catalog.local_fallback.value,
            remote_credential_refs=len(remote_refs),
            resolved_credential_refs=len(available & remote_refs),
            adjudication="local_serves_adjudicating_groups",  # the ADR 0037 deviation, opted into
            remote_rows_serving=serving_remote,  # AD-205: 0 here with a key means dead rows
        )
        cfg = cfg.model_copy(update={"local_serves_remote_preferred_groups": True})
        declared = shipped_catalog(base, cfg=cfg)
        available = _probe(declared, resolver)
        adopted = True

    declared = declared.model_copy(
        update={
            "providers": _merge(declared.providers, operator_providers),
            "deployments": [*declared.deployments, *operator_deployments],
        }
    )
    task_models = resolve_task_models(models, task_defaults=catalog.task_defaults)
    # The overlay may add its own credentialed rows, so probe once more over the FINAL table —
    # `_probe` is the only thing that touches the seam and it is called once per distinct table.
    return _activate(
        declared,
        task_models,
        available=_probe(declared, resolver),
        adopted=adopted,
        deployment_probe=dep_probe,
    )


def _surviving_remote_rows(
    catalog: ModelCatalogSettings, *, available: frozenset[str], probe: DeploymentProbe
) -> int:
    """How many REMOTE deployments would actually survive activation (AD-205).

    "Remote" = its provider declares a `credential_ref`. A row survives when that ref resolved
    AND the deployment it names is not known-absent. This is the predicate the AUTO local-fallback
    posture needs; credential resolution alone is not it.
    """
    remote_keys = {p.key for p in catalog.providers if p.credential_ref is not None}
    credentialed = {
        p.key
        for p in catalog.providers
        if p.credential_ref is not None and p.credential_ref in available
    }
    return sum(
        1
        for d in catalog.deployments
        if d.provider_key in remote_keys
        and d.provider_key in credentialed
        and probe.exists(d) is not False
    )


def _probe(catalog: ModelCatalogSettings, resolver: CredentialProbe | None) -> frozenset[str]:
    """Which `credential_ref`s resolve. No resolver ⇒ none do — the honest answer, and the one
    that leaves every keyless local provider standing."""
    if resolver is None:
        return frozenset()
    return resolvable_credential_refs(catalog, resolver)


def _activate(
    declared: ModelCatalogSettings,
    models: ModelSettings,
    *,
    available: frozenset[str],
    adopted: bool,
    deployment_probe: DeploymentProbe | None = None,
) -> PlaneModelLayer:
    active = active_catalog(
        declared, available_credentials=available, deployment_probe=deployment_probe
    )
    log.info(
        "model_layer_resolved",  # content-free: names and counts only
        providers=len(active.providers),
        deployments=len(active.deployments),
        groups=len({d.model_group for d in active.deployments}),
        credentialed_providers=len(available),
        local_fallback_adopted=adopted,
    )
    return PlaneModelLayer(
        models=models,
        catalog=active,
        active_credential_refs=tuple(sorted(available)),
        local_fallback_adopted=adopted,
    )


def build_plane_router(
    *,
    models: ModelSettings,
    catalog: ModelCatalogSettings,
    chunk_token_ratio: float,
    resolver: CredentialProbe | None = None,
    overlay_providers: Sequence[ProviderRecord] = (),
    overlay_deployments: Sequence[ModelDeployment] = (),
    degrade_emitter: DegradeEmitterPort | None = None,
    embedder: EmbeddingPort | None = None,
) -> ModelRouter:
    """`resolve_plane_model_layer` + `build_model_router` — what a composition root calls.

    The resolver is threaded into the registry as well as the probe, so the SAME seam that decided
    a provider is active supplies its key at compile time (`registry.compile_model_list`) — a
    provider can never be activated by one path and credential-less on the other.

    `embedder` is a straight pass-through to `build_model_router` (see ITS docstring for why the
    injection exists): a root that already owns the plane's one `EmbeddingPort` hands it over
    rather than letting the factory resolve a SECOND one from `models.embed_backend`. None — every
    existing caller — keeps the build-my-own behaviour unchanged.
    """
    layer = resolve_plane_model_layer(
        models=models,
        catalog=catalog,
        resolver=resolver,
        overlay_providers=overlay_providers,
        overlay_deployments=overlay_deployments,
    )
    return build_model_router(
        models=layer.models,
        catalog=layer.catalog,
        degrade_emitter=degrade_emitter,
        secret_resolver=resolver,
        chunk_token_ratio=chunk_token_ratio,
        embedder=embedder,
    )
