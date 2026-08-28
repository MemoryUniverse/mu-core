"""The concrete `SecretResolver` — turns a `credential_ref` NAME into a key VALUE (§4).

`registry.SecretResolver` is a seam with no implementation in the tree: every composition root
built its router with `secret_resolver=None`, so `_NullSecretResolver` raised on any credentialed
provider and the ONLY catalog that could ever compile was one with no cloud provider in it. That
is half of why `shipped_catalog.py` had no caller — the table is 5/8 credentialed rows.

The seam this implements is the one `catalog.py:77` names: *"name under `Settings(secrets_dir=...)`;
NEVER inline a key"*. Lookup order, first hit wins:

  1. an explicit in-process mapping (`overrides`) — how a composition root hands through a key it
     was itself given as a value (e.g. `mu-client` passing a `SecretStr` it read from its own
     seam). Held as `SecretStr`, so it cannot be printed by accident;
  2. `<secrets_dir>/<credential_ref>` — one file per secret, the docker/k8s convention
     (model-layer-spec §4). Read at resolve time, stripped of the trailing newline every secret
     manager writes;
  3. `os.environ[<CREDENTIAL_REF upper-cased>]` — the ordinary way `ANTHROPIC_API_KEY`,
     `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`… are already set on a developer box.

A miss raises `RegistryError` (fail-loud, never an empty string that would reach a provider as a
credential and fail 300 ms later as a 401). `resolvable_credential_refs`
(`shipped_catalog.py`) calls `resolve` exactly to turn that raise into "this provider is not
active" BEFORE the registry compiles, which is what keeps a keyless box bootable.

CONTENT-FREE (DEV-STANDARDS rule 4 / house rule 1): no value is logged, and `__repr__` names only
the seam's SHAPE — the directory, whether env is consulted, and the COUNT of overrides. `__str__`
inherits it. A resolved value is returned and nothing else.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import SecretStr

from mu_engine.providers.registry import RegistryError, SecretResolver

__all__ = ["SecretSeamResolver"]


class SecretSeamResolver(SecretResolver):
    """Resolve a `credential_ref` from an in-process mapping, a secrets dir, then the environment.

    Args:
        overrides: `credential_ref` -> value, for a root that already holds the secret. Plain
            `str` values are wrapped in `SecretStr` on the way in so this object never stores a
            bare key.
        secrets_dir: root of the one-file-per-secret seam (`ModelCatalogSettings.secrets_dir`).
        use_env: consult `os.environ[ref.upper()]` as the last resort.
    """

    def __init__(
        self,
        *,
        overrides: Mapping[str, str | SecretStr] | None = None,
        secrets_dir: str | os.PathLike[str] | None = None,
        use_env: bool = True,
    ) -> None:
        self._overrides: dict[str, SecretStr] = {
            ref: value if isinstance(value, SecretStr) else SecretStr(value)
            for ref, value in (overrides or {}).items()
        }
        self._secrets_dir = Path(secrets_dir) if secrets_dir is not None else None
        self._use_env = use_env

    def resolve(self, credential_ref: str) -> str:
        """The seam's one method. Raises `RegistryError` when the ref resolves nowhere."""
        override = self._overrides.get(credential_ref)
        if override is not None:
            return override.get_secret_value()

        from_file = self._from_file(credential_ref)
        if from_file is not None:
            return from_file

        if self._use_env:
            from_env = os.environ.get(credential_ref.upper())
            if from_env:
                return from_env

        # The MESSAGE names the ref and the places looked — never a value, and never the content
        # of a file that failed to parse.
        raise RegistryError(
            f"credential_ref {credential_ref!r} did not resolve "
            f"(secrets_dir={'set' if self._secrets_dir is not None else 'unset'}, "
            f"env={'consulted' if self._use_env else 'not consulted'})"
        )

    def _from_file(self, credential_ref: str) -> str | None:
        """`<secrets_dir>/<ref>`, or `None` when there is no dir / no such file / it is empty.

        A ref is used as a FILE NAME, so it is refused if it could escape the directory — a
        catalog is data and a hostile catalog must not be able to read `../../etc/passwd`.
        `OSError` (a directory, a permission error, a dangling symlink) is a MISS, not a crash:
        the caller's next step is the environment, and a genuinely absent credential is reported
        once, by `resolve`, with the ref name only.
        """
        if self._secrets_dir is None:
            return None
        if credential_ref != Path(credential_ref).name or credential_ref in {"", ".", ".."}:
            raise RegistryError(f"credential_ref {credential_ref!r} is not a valid secret name")
        try:
            value = (self._secrets_dir / credential_ref).read_text(encoding="utf-8")
        except OSError:
            return None
        stripped = value.strip()
        return stripped or None

    def __repr__(self) -> str:
        """Content-free by construction — a count and two shapes, never a ref's VALUE."""
        seam = str(self._secrets_dir) if self._secrets_dir is not None else None
        return (
            f"{type(self).__name__}(overrides={len(self._overrides)}, "
            f"secrets_dir={seam!r}, use_env={self._use_env})"
        )
