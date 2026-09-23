"""The cache must not cost correctness: one scan per namespace, sequence still continues."""

import pytest

from mu_local.local_memory import LocalMemory


class _Item:
    def __init__(self, seq):
        self.turn_seq = seq


class _Scored:
    def __init__(self, seq):
        self.item = _Item(seq)


@pytest.mark.asyncio
async def test_store_is_scanned_once_per_namespace_not_once_per_add(monkeypatch):
    lm = LocalMemory.__new__(LocalMemory)
    lm._turn_seq_next = {}
    calls = {"n": 0}

    class _NS:
        def to_prefix(self):
            return "org/ws/u/s"

    async def fake_base(ns):
        calls["n"] += 1
        return 7

    lm._next_turn_seq_base = fake_base  # type: ignore[method-assign]
    ns = _NS()
    assert await lm._next_turn_seq_cached(ns) == 7
    lm._remember_turn_seq(ns, 9)
    assert await lm._next_turn_seq_cached(ns) == 9  # continues, no rescan
    lm._remember_turn_seq(ns, 11)
    assert await lm._next_turn_seq_cached(ns) == 11
    assert calls["n"] == 1, "the store must be read once per namespace, not once per add()"


@pytest.mark.asyncio
async def test_a_second_namespace_reads_the_store_on_its_own_first_touch(monkeypatch):
    lm = LocalMemory.__new__(LocalMemory)
    lm._turn_seq_next = {}
    calls = {"n": 0}

    class _NS:
        def __init__(self, p):
            self._p = p

        def to_prefix(self):
            return self._p

    async def fake_base(ns):
        calls["n"] += 1
        return 3

    lm._next_turn_seq_base = fake_base  # type: ignore[method-assign]
    await lm._next_turn_seq_cached(_NS("a"))
    await lm._next_turn_seq_cached(_NS("b"))
    assert calls["n"] == 2, "each namespace keeps its own sequence"
