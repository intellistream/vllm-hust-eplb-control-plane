from __future__ import annotations

import importlib
import types

import pytest

from vllm_hust_eplb_control_plane import probe


def test_probe_is_noop_without_flag(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_EPLB_WAIT_PROBE", raising=False)

    def forbidden(_name):
        raise AssertionError("disabled probe imported a runtime module")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    probe.register()


class Updator:
    def __init__(self, flag):
        self.flag = flag
        self.calls = 0

    def get_update_info_flag(self):
        return self.flag

    def forward_before(self):
        if self.get_update_info_flag():
            self.eplb_process.block_update_q.get()
        self.calls += 1
        return "ret"


def test_probe_times_only_the_fetch_iteration(monkeypatch, capsys):
    mod = types.SimpleNamespace(EplbUpdator=type("U", (Updator,), {}))
    probe._install(mod)
    cls = mod.EplbUpdator

    quiet = cls(False)
    assert quiet.forward_before() == "ret"
    assert "EPLB_WAIT_PROBE" not in capsys.readouterr().err

    fetch = cls(True)
    fetch.eplb_process = types.SimpleNamespace(block_update_q=types.SimpleNamespace(get=lambda: None))
    assert fetch.forward_before() == "ret"
    assert fetch.calls == 1  # the original ran exactly once
    assert "EPLB_WAIT_PROBE wait_ms=" in capsys.readouterr().err


def test_probe_install_is_idempotent_and_guarded():
    mod = types.SimpleNamespace(EplbUpdator=type("U", (Updator,), {}))
    probe._install(mod)
    first = mod.EplbUpdator.forward_before
    probe._install(mod)
    assert mod.EplbUpdator.forward_before is first

    class Other:
        def forward_before(self):
            return 1

    with pytest.raises(RuntimeError, match="unsupported host"):
        probe._install(types.SimpleNamespace(EplbUpdator=Other))
