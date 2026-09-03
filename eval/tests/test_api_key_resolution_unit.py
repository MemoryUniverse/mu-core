"""``_resolve_api_key`` — the fix for the live Azure-key leak (verified via `pgrep`, see the
module-level comment in ``mu_eval/__main__.py``): ``--api-key`` used to be the ONLY way to hand
this CLI a key, and ``eval/vm_eval.sh`` embedded whatever was passed straight into the remote SSH
command string — argv on both ends, plus its own stdout log. These tests pin the safe env-var
path as the one that wins, that the flag still works but is flagged, and that neither being set
fails LOUD rather than silently falling back to a fake key.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_EVAL_ROOT = str(Path(__file__).resolve().parents[1])
if _EVAL_ROOT not in sys.path:  # pragma: no cover - import shim, mirrors eval/conftest.py
    sys.path.insert(0, _EVAL_ROOT)

from mu_eval.__main__ import _API_KEY_ENV_VAR, _resolve_api_key  # noqa: E402


def _args(api_key: str | None) -> argparse.Namespace:
    return argparse.Namespace(api_key=api_key)


def test_env_var_wins_when_both_are_set(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_API_KEY_ENV_VAR, "env-secret")
    resolved = _resolve_api_key(_args("flag-secret"), subcommand="answer-quality")
    assert resolved == "env-secret"
    # the safe path never prints a warning — only the discouraged flag path does.
    assert "!" not in capsys.readouterr().out


def test_env_var_alone_resolves_with_no_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(_API_KEY_ENV_VAR, "env-secret")
    resolved = _resolve_api_key(_args(None), subcommand="judge-probe")
    assert resolved == "env-secret"
    assert capsys.readouterr().out == ""


def test_explicit_flag_alone_still_works_but_is_flagged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv(_API_KEY_ENV_VAR, raising=False)
    resolved = _resolve_api_key(_args("flag-secret"), subcommand="answer-quality")
    assert resolved == "flag-secret"
    out = capsys.readouterr().out
    assert "answer-quality" in out
    assert _API_KEY_ENV_VAR in out
    # THE KEY ITSELF is never printed — only the fact that the flag path was used.
    assert "flag-secret" not in out


def test_neither_set_fails_loud_not_a_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_API_KEY_ENV_VAR, raising=False)
    with pytest.raises(SystemExit) as exc_info:
        _resolve_api_key(_args(None), subcommand="answer-quality")
    assert _API_KEY_ENV_VAR in str(exc_info.value)


def test_empty_string_env_var_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    # an exported-but-empty var (e.g. a broken secret-manager pipe) must not silently "resolve"
    # to an empty-string key that would fail opaquely three layers down inside httpx instead.
    monkeypatch.setenv(_API_KEY_ENV_VAR, "")
    with pytest.raises(SystemExit):
        _resolve_api_key(_args(None), subcommand="judge-probe")


def test_judge_control_default_still_works_with_no_key_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # judge-control's own call site passes default="unused" — its default target is the keyless
    # local SLM sidecar, so it must keep working with NEITHER --api-key NOR MU_EVAL_API_KEY set
    # (unlike answer-quality/judge-probe, which fail loud in that same situation, tested above).
    monkeypatch.delenv(_API_KEY_ENV_VAR, raising=False)
    resolved = _resolve_api_key(_args(None), subcommand="judge-control", default="unused")
    assert resolved == "unused"


def test_judge_control_still_prefers_a_real_key_when_one_is_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # a user who points judge-control at a real, key-requiring endpoint gets the SAME safe
    # resolution as answer-quality/judge-probe — the default is a fallback, not an override.
    monkeypatch.setenv(_API_KEY_ENV_VAR, "env-secret")
    resolved = _resolve_api_key(_args(None), subcommand="judge-control", default="unused")
    assert resolved == "env-secret"
