"""L1 — ProviderRegistry + ModelRegistry (model-layer-spec §2.1).

Compiles the many-to-many model↔provider catalog into LiteLLM's flat `model_list`. A
`model_group` may have N deployments across N providers; a provider may serve M groups. LiteLLM
assigns each compiled row a unique id and tracks health/cooldown PER deployment — that IS the
many-to-many failover, native, no glue (research §1.4). L4's local-priority predicate is applied
HERE by stamping each row's `order:`; the ordered failover itself is the Router's (§2.4).

Fail-loud (model-layer-spec §5, memory-layer §10): an unknown `provider_key`, or a `ModelSettings`
task whose group has zero deployments, raises `RegistryError` at COMPOSITION — the process does
not start. Mirrors mem0 `LlmFactory` unknown-provider raise (`other_repos/mem0/mem0/utils/
factory.py`) and MemOS `LLMFactory.from_config` `raise ValueError(f"Invalid backend: {backend}")`
(`other_repos/MemOS/src/memos/llms/factory.py:33-34`), our typed error + our many-to-many shape.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mu_engine.providers._contracts import ModelLayerError
from mu_engine.providers.catalog import (
    ModelDeployment,
    ProviderKind,
    ProviderRecord,
    Task,
)
from mu_engine.providers.local_priority import REMOTE_ORDER, LocalPriorityPolicy

__all__ = ["ProviderModelRegistry", "RegistryError"]


class RegistryError(ModelLayerError):
    """Catalog invalid at composition — unknown provider_key or a task group with no deployment."""


class ProviderModelRegistry:
    """Holds the plane's providers + deployments and compiles the LiteLLM `model_list` (§2.1)."""

    def __init__(
        self,
        providers: list[ProviderRecord],
        deployments: list[ModelDeployment],
        *,
        local_policy: LocalPriorityPolicy,
        task_groups: dict[Task, str],
        secret_resolver: SecretResolver | None = None,
    ) -> None:
        """`task_groups` is `TaskClassMapper.task_groups()` — the task→group table this catalog
        MUST satisfy. `secret_resolver` turns a `credential_ref` into an api-key value at compile
        (never inline; §4). A `None` resolver means no credentialed providers are registered."""
        self._providers = {p.key: p for p in providers}
        self._deployments = deployments
        self._local_policy = local_policy
        self._task_groups = task_groups
        self._resolver = secret_resolver or _NullSecretResolver()
        # group -> set of tasks that reference it (for the local-priority stamp, §2.4)
        self._group_tasks: dict[str, set[Task]] = defaultdict(set)
        for task, group in task_groups.items():
            self._group_tasks[group].add(task)
        # group -> deployments
        self._group_deployments: dict[str, list[ModelDeployment]] = defaultdict(list)
        for dep in deployments:
            self._group_deployments[dep.model_group].append(dep)
        self._validate()

    # ---- validation (fail-loud, §2.2 / §5) ------------------------------------------------
    def _validate(self) -> None:
        # duplicate provider keys
        if len(self._providers) != len({p.key for p in self._providers.values()}):
            raise RegistryError("duplicate provider keys in catalog")
        # every deployment resolves to a known provider
        for dep in self._deployments:
            if dep.provider_key not in self._providers:
                raise RegistryError(
                    f"deployment {dep.model_group!r} references unknown provider_key "
                    f"{dep.provider_key!r}"
                )
        # every task's model-group has >= 1 deployment
        # (EMBED is served by the dedicated EmbeddingPort seam, not a litellm deployment — it is
        #  validated by the embedder registry, not here; §2.3/§6-P5.)
        for task, group in self._task_groups.items():
            if task is Task.EMBED:
                continue
            if not self._group_deployments.get(group):
                raise RegistryError(
                    f"task {task.value!r} maps to model-group {group!r} which has no deployment"
                )

    # ---- L4 helper ------------------------------------------------------------------------
    def is_local_group(self, model_group: str) -> bool:
        """True iff any deployment in the group is served by a local provider (§2.1 resp. 4)."""
        return any(
            self._providers[d.provider_key].is_local
            for d in self._group_deployments.get(model_group, [])
        )

    def _order_for(self, dep: ModelDeployment) -> int:
        """Stamp the `order:` tier for a deployment from the L4 predicate (§2.4). A local
        deployment in a group referenced by a local-capable task → tier 1; else tier 2. If no
        task references the group (e.g. an operator-only group) the group is treated as
        remote-tier unless a task claims it."""
        provider = self._providers[dep.provider_key]
        tasks = self._group_tasks.get(dep.model_group)
        if not tasks:
            return REMOTE_ORDER
        # a group may serve several tasks; local if ANY referencing task is local-capable
        return min(self._local_policy.order_for(is_local=provider.is_local, task=t) for t in tasks)

    # ---- L1 compile -----------------------------------------------------------------------
    def compile_model_list(self) -> list[dict[str, Any]]:
        """Emit the flat LiteLLM `model_list` — one entry per deployment, `order:` stamped from
        L4, credentials resolved from the secret seam (never inline). §2.1 responsibility 3."""
        out: list[dict[str, Any]] = []
        for dep in self._deployments:
            provider = self._providers[dep.provider_key]
            params: dict[str, Any] = {
                "model": dep.model_id,
                "order": self._order_for(dep),
            }
            if provider.api_base is not None:
                params["api_base"] = provider.api_base
            if provider.credential_ref is not None:
                params["api_key"] = self._resolver.resolve(provider.credential_ref)
            if dep.weight is not None:
                params["weight"] = dep.weight
            params.update(dep.extra_params)
            out.append({"model_name": dep.model_group, "litellm_params": params})
        return out

    def max_input_tokens(self, model_group: str) -> int | None:
        """The largest declared input context across the group's deployments (for L6 chunking),
        or `None` if no deployment declares one (caller falls back to `litellm.get_max_tokens`)."""
        declared = [
            d.max_input_tokens
            for d in self._group_deployments.get(model_group, [])
            if d.max_input_tokens is not None
        ]
        return max(declared) if declared else None

    def custom_provider_keys(self) -> set[str]:
        """The litellm provider prefixes for in-process (LOCAL_INPROC) providers — the set that
        must be registered in `custom_provider_map` before the Router is built (§2.5b)."""
        return {
            p.litellm_provider
            for p in self._providers.values()
            if p.kind is ProviderKind.LOCAL_INPROC
        }


class SecretResolver:
    """Resolves a `credential_ref` to a secret value (model-layer-spec §4 — `/run/secrets`).

    A Protocol-shaped seam; the concrete resolver (a `Settings(secrets_dir=...)` reader) is
    injected by the composition root so no key is ever inlined or logged.
    """

    def resolve(self, credential_ref: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class _NullSecretResolver(SecretResolver):
    """No credentialed providers registered — resolving a ref is a fail-loud error."""

    def resolve(self, credential_ref: str) -> str:
        raise RegistryError(
            f"credential_ref {credential_ref!r} requested but no SecretResolver was injected"
        )
