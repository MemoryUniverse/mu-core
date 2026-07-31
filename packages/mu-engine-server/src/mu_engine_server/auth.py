"""Server-side bearer-token verifier (build-plan §4 C3, item 7; design §2.4).

`mu-engine-server` is `mu-local`'s zero-auth justification's exact negation: it is a real,
network-reachable HTTP server (even loopback-bound by default), so unlike the in-process,
daemonless `mu-local` composition root, it needs a real gate on `Authorization`. The **mode** is
LOCKED (design §2.4, candidate (a)): bearer + a per-process token minted locally on first
``make up`` and written to disk (Stage E's job, not this module's), read here and compared
against the presented ``Authorization: Bearer <token>`` header.

This is the **first** concrete server-side ``AuthPort``-shaped bearer verifier in the codebase —
design §2.4 grepped the repo for a server-side verifier (``AuthPort`` impl, ``*AuthAdapter``,
``TokenVerifier``, ``verify_token``) and found zero. The design corpus also refers to a shared
``AuthenticationError`` ("existing envelope") in several places (`mu-local-and-sdk-spec.md:752`,
`auth-key-issuance-spec.md:396`) but a repo-wide grep of ``mu-contracts``/``mu-engine`` turns up no
such class either — only ``mu_contracts.domain.errors.AuthorizationError``, the actual "Base for
every authorization denial" already in the shared typed-error hierarchy. Rather than inventing an
unrelated second root (banned — DEV-STANDARDS rule 8's "one typed error hierarchy"), this module
defines :class:`AuthenticationError` as a subclass of that existing base: it reuses the pinned
root, and gives the name the design corpus already uses everywhere a first concrete definition.

Two layers, deliberately split:

* :func:`load_token` / :func:`verify_bearer_token` — pure, framework-agnostic. Unit-testable
  without a FastAPI app; this is what raises :class:`AuthenticationError`.
* :func:`make_bearer_verifier` / :data:`require_bearer_token` — the FastAPI ``Depends``-shaped
  adapter C2's routes wire in. It catches :class:`AuthenticationError` at the transport boundary
  and re-raises as ``HTTPException(401)`` — C2 does not need a central exception-handler registered
  for auth denials to come back as 401 (`build-plan §4 CG-3: "no/blank/wrong Authorization -> 401;
  correct token -> 200"`).
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import Header, HTTPException

from mu_contracts.domain.errors import AuthorizationError

__all__ = [
    "DEFAULT_TOKEN_PATH",
    "TOKEN_PATH_ENV_VAR",
    "AuthenticationError",
    "get_token_path",
    "load_token",
    "make_bearer_verifier",
    "require_bearer_token",
    "verify_bearer_token",
]


class AuthenticationError(AuthorizationError):
    """A request's ``Authorization`` header failed bearer verification — missing, blank,
    malformed, or not matching the per-process token (design §2.4). Subclasses the shared
    ``AuthorizationError`` root (`mu_contracts.domain.errors`) rather than a new hierarchy."""


# Contract (design §2.4/§11.2, build-plan §13 item 7): the per-process token lives at
# ~/.memory-universe/engine-server.token, mode 0600, written by Stage E's `make up`. This module
# only ever READS it. Override for compose-mounts (a different absolute path inside the container)
# or tests via TOKEN_PATH_ENV_VAR, or by passing an explicit path to the functions below.
DEFAULT_TOKEN_PATH: Path = Path.home() / ".memory-universe" / "engine-server.token"
TOKEN_PATH_ENV_VAR = "MU_ENGINE_SERVER_TOKEN_PATH"  # noqa: S105 - env var NAME, not a secret


def get_token_path(token_path: str | Path | None = None) -> Path:
    """Resolve the token-file path: explicit arg > ``MU_ENGINE_SERVER_TOKEN_PATH`` env > default.

    The explicit-arg case is what lets :func:`make_bearer_verifier` bind a fixed path (settings-
    driven in C4's composition, or a ``tmp_path`` fixture in a test) without mutating process-wide
    environment state.
    """
    if token_path is not None:
        return Path(token_path)
    env_override = os.environ.get(TOKEN_PATH_ENV_VAR)
    if env_override:
        return Path(env_override)
    return DEFAULT_TOKEN_PATH


def load_token(token_path: str | Path | None = None) -> str:
    """Read and return the per-process bearer token from disk.

    Raises :class:`AuthenticationError` if the file is missing or its contents are blank — a
    verifier with nothing to compare against can never succeed, so this fails loud at load time
    rather than deferring to (and disguising itself inside) the first request's 401
    (DEV-STANDARDS rule 8: fail-loud, no silent fallback).
    """
    path = get_token_path(token_path)
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise AuthenticationError(f"engine-server token file not found: {path}") from exc
    except OSError as exc:
        raise AuthenticationError(f"engine-server token file unreadable: {path}") from exc
    token = raw.strip()
    if not token:
        raise AuthenticationError(f"engine-server token file is empty: {path}")
    return token


def verify_bearer_token(authorization: str | None, *, expected_token: str) -> None:
    """Verify an ``Authorization`` header value against ``expected_token``.

    Raises :class:`AuthenticationError` on a missing header, a header not shaped
    ``Bearer <token>``, a blank presented token, or a mismatched token. The comparison itself is
    constant-time (``hmac.compare_digest``) so response timing cannot be used to learn how much of
    the token a guess got right; the error message never echoes back which of the failure modes
    fired (content-free discipline, platform-layer0-spec §Content-free) or any part of either
    token.
    """
    if not authorization:
        raise AuthenticationError("missing Authorization header")
    scheme, _, presented = authorization.partition(" ")
    if scheme != "Bearer" or not presented:
        raise AuthenticationError("malformed Authorization header")
    if not hmac.compare_digest(presented, expected_token):
        raise AuthenticationError("bearer token mismatch")


def make_bearer_verifier(
    token_path: str | Path | None = None,
) -> Callable[[str | None], None]:
    """Build a FastAPI dependency bound to a specific token-file path.

    The returned callable re-reads the token file on every call (so a rotated token takes
    effect without a server restart) rather than caching it at bind time. Use this directly
    when the path needs to come from settings (C4's composition) or a test fixture;
    :data:`require_bearer_token` below is the same thing pre-bound to the default resolution
    (arg > env > ``~/.memory-universe/``) for C2's routes to depend on out of the box.
    """

    def _verify(authorization: str | None = Header(default=None)) -> None:
        try:
            expected = load_token(token_path)
            verify_bearer_token(authorization, expected_token=expected)
        except AuthenticationError as exc:
            raise HTTPException(status_code=401, detail="unauthorized") from exc

    return _verify


# The dependency C2's routes actually import: `Depends(require_bearer_token)`.
require_bearer_token: Callable[[str | None], None] = make_bearer_verifier()
