import pytest

from exo.worker.engines.mlx import cache as cache_module
from exo.worker.engines.mlx.generator.generate import (
    PrefillCancelled,
    PrefillOutOfMemory,
    abort_prefill_if_memory_critical,
)


def test_prefill_out_of_memory_is_prefill_cancelled() -> None:
    # Cleanup paths catch PrefillCancelled; the OOM abort must flow through them.
    assert issubclass(PrefillOutOfMemory, PrefillCancelled)


def test_memory_pressure_critical_on_system_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "get_memory_used_percentage", lambda: 0.99)
    assert cache_module.memory_pressure_critical() is True


def test_memory_pressure_not_critical_when_memory_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "get_memory_used_percentage", lambda: 0.10)
    monkeypatch.setattr(cache_module.mx, "get_active_memory", lambda: 0, raising=False)
    assert cache_module.memory_pressure_critical() is False


def test_abort_prefill_raises_when_critical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "get_memory_used_percentage", lambda: 0.99)
    with pytest.raises(PrefillOutOfMemory):
        abort_prefill_if_memory_critical(group=None)


def test_abort_prefill_noop_when_memory_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "get_memory_used_percentage", lambda: 0.10)
    monkeypatch.setattr(cache_module.mx, "get_active_memory", lambda: 0, raising=False)
    abort_prefill_if_memory_critical(group=None)
