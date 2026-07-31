"""Unit tests for the C3 bearer verifier (build-plan §4 C3; CG-3 gate).

Pure-logic tests against :func:`load_token`/:func:`verify_bearer_token` (no HTTP), plus an
integration-shaped-but-still-``unit``-marked pass through a real FastAPI app + ``TestClient`` to
prove the ``Depends(require_bearer_token)`` wiring returns 401/200 exactly as C2's routes will see
it. No mocks of the verifier itself — real files on disk, real constant-time compare, real FastAPI
dependency resolution.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from mu_engine_server.auth import (
    AuthenticationError,
    get_token_path,
    load_token,
    make_bearer_verifier,
    verify_bearer_token,
)

pytestmark = pytest.mark.unit

TOKEN = "s3cr3t-engine-token-abc123"  # noqa: S105 - test fixture value, not a real credential


def _write_token(path: Path, token: str = TOKEN) -> Path:
    path.write_text(token, encoding="utf-8")
    path.chmod(0o600)
    return path


# ── get_token_path resolution order ─────────────────────────────────────────────────────────


def test_get_token_path_explicit_arg_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MU_ENGINE_SERVER_TOKEN_PATH", str(tmp_path / "env-token"))
    explicit = tmp_path / "explicit-token"
    assert get_token_path(explicit) == explicit


def test_get_token_path_env_var_used_when_no_explicit_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_path = tmp_path / "env-token"
    monkeypatch.setenv("MU_ENGINE_SERVER_TOKEN_PATH", str(env_path))
    assert get_token_path() == env_path


def test_get_token_path_default_when_nothing_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MU_ENGINE_SERVER_TOKEN_PATH", raising=False)
    assert get_token_path() == Path.home() / ".memory-universe" / "engine-server.token"


# ── load_token ───────────────────────────────────────────────────────────────────────────────


def test_load_token_reads_and_strips(tmp_path: Path) -> None:
    path = _write_token(tmp_path / "token", "  padded-token  \n")
    assert load_token(path) == "padded-token"


def test_load_token_missing_file_raises_authentication_error(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(AuthenticationError):
        load_token(missing)


def test_load_token_blank_file_raises_authentication_error(tmp_path: Path) -> None:
    path = _write_token(tmp_path / "token", "   \n")
    with pytest.raises(AuthenticationError):
        load_token(path)


# ── verify_bearer_token (pure header check) ─────────────────────────────────────────────────


def test_verify_bearer_token_correct_token_passes() -> None:
    verify_bearer_token(f"Bearer {TOKEN}", expected_token=TOKEN)  # must not raise


def test_verify_bearer_token_missing_header_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token(None, expected_token=TOKEN)


def test_verify_bearer_token_blank_header_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token("", expected_token=TOKEN)


def test_verify_bearer_token_wrong_token_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token("Bearer wrong-token", expected_token=TOKEN)


def test_verify_bearer_token_malformed_no_scheme_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token(TOKEN, expected_token=TOKEN)


def test_verify_bearer_token_wrong_scheme_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token(f"Basic {TOKEN}", expected_token=TOKEN)


def test_verify_bearer_token_bearer_with_no_token_raises() -> None:
    with pytest.raises(AuthenticationError):
        verify_bearer_token("Bearer ", expected_token=TOKEN)


def test_verify_bearer_token_is_constant_time_compare(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert the implementation routes through hmac.compare_digest rather than `==`/`in`."""
    import hmac

    calls: list[tuple[str, str]] = []
    real_compare = hmac.compare_digest

    def _spy(a: str, b: str) -> bool:
        calls.append((a, b))
        return bool(real_compare(a, b))

    monkeypatch.setattr(hmac, "compare_digest", _spy)
    verify_bearer_token(f"Bearer {TOKEN}", expected_token=TOKEN)
    assert calls == [(TOKEN, TOKEN)]


# ── AuthenticationError is a real AuthorizationError, not a parallel hierarchy ──────────────


def test_authentication_error_subclasses_shared_authorization_error() -> None:
    from mu_contracts.domain.errors import AuthorizationError, MemoryUniverseError

    assert issubclass(AuthenticationError, AuthorizationError)
    assert issubclass(AuthenticationError, MemoryUniverseError)


# ── FastAPI dependency wiring (what C2's routes actually depend on) ─────────────────────────


def _make_test_app(token_path: Path) -> FastAPI:
    app = FastAPI()
    verifier = make_bearer_verifier(token_path)

    @app.get("/protected", dependencies=[Depends(verifier)])
    def _protected() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_dependency_correct_token_returns_200(tmp_path: Path) -> None:
    token_path = _write_token(tmp_path / "token")
    client = TestClient(_make_test_app(token_path))

    response = client.get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_dependency_missing_header_returns_401(tmp_path: Path) -> None:
    token_path = _write_token(tmp_path / "token")
    client = TestClient(_make_test_app(token_path))

    response = client.get("/protected")

    assert response.status_code == 401


def test_dependency_blank_header_returns_401(tmp_path: Path) -> None:
    token_path = _write_token(tmp_path / "token")
    client = TestClient(_make_test_app(token_path))

    response = client.get("/protected", headers={"Authorization": ""})

    assert response.status_code == 401


def test_dependency_wrong_token_returns_401(tmp_path: Path) -> None:
    token_path = _write_token(tmp_path / "token")
    client = TestClient(_make_test_app(token_path))

    response = client.get("/protected", headers={"Authorization": "Bearer nope"})

    assert response.status_code == 401


def test_dependency_missing_token_file_returns_401(tmp_path: Path) -> None:
    token_path = tmp_path / "never-written"
    client = TestClient(_make_test_app(token_path))

    response = client.get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})

    assert response.status_code == 401


def test_require_bearer_token_default_respects_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The module-level `require_bearer_token` (no explicit path bound) is what C2 imports
    directly; prove it honors TOKEN_PATH_ENV_VAR end-to-end through a real FastAPI app."""
    from mu_engine_server.auth import require_bearer_token

    token_path = _write_token(tmp_path / "token")
    monkeypatch.setenv("MU_ENGINE_SERVER_TOKEN_PATH", str(token_path))

    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_bearer_token)])
    def _protected() -> dict[str, bool]:
        return {"ok": True}

    client = TestClient(app)

    good = client.get("/protected", headers={"Authorization": f"Bearer {TOKEN}"})
    bad = client.get("/protected", headers={"Authorization": "Bearer wrong"})

    assert good.status_code == 200
    assert bad.status_code == 401
