"""Registry[T] acceptance — fail-loud open registration (platform-layer0-spec §7, §15.4)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from mu_contracts.config.settings import Settings
from mu_contracts.domain.errors import DuplicateComponentError, UnknownComponentError
from mu_engine.platform.registry import Registry

pytestmark = pytest.mark.unit


def _settings() -> Settings:
    return Settings()


def _const(value: str) -> Callable[[Settings], str]:
    """A properly-bound constant factory (avoids the loop-variable closure pitfall)."""

    def factory(settings: Settings) -> str:
        del settings
        return value

    return factory


def test_register_and_create() -> None:
    reg: Registry[str] = Registry("thing")

    @reg.register("a")
    def _a(settings: Settings) -> str:
        del settings
        return "A"

    assert reg.create("a", _settings()) == "A"
    assert reg.is_registered("a")


def test_unknown_key_raises_and_lists_known() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register_factory("known", lambda s: "K")
    with pytest.raises(UnknownComponentError) as ei:
        reg.create("missing", _settings())
    assert "known" in str(ei.value)  # lists known keys, never a silent default


def test_duplicate_key_raises() -> None:
    reg: Registry[str] = Registry("thing")
    reg.register_factory("dup", lambda s: "1")
    with pytest.raises(DuplicateComponentError):
        reg.register_factory("dup", lambda s: "2")


def test_names_stable_sorted() -> None:
    reg: Registry[str] = Registry("thing")
    for key in ("c", "a", "b"):
        reg.register_factory(key, _const(key))
    assert reg.names() == ("a", "b", "c")


def test_no_import_time_socket() -> None:
    # Registration must NOT invoke the factory (spec §7: the socket opens only at create()).
    calls: list[int] = []
    reg: Registry[str] = Registry("thing")

    @reg.register("lazy")
    def _f(settings: Settings) -> str:
        del settings
        calls.append(1)
        return "x"

    assert calls == []  # not built yet
    reg.create("lazy", _settings())
    assert calls == [1]
