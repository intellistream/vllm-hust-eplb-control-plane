from __future__ import annotations

import importlib

from vllm_hust_eplb_control_plane import plugin


def _forbid_imports(monkeypatch):
    def forbidden(_name):
        raise AssertionError("registration imported a runtime module")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    # `from x import y` inside register() goes through __import__, not import_module.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("vllm_ascend"):
            raise AssertionError(f"registration imported host module {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded)


def test_discovery_is_noop_without_enable(monkeypatch):
    monkeypatch.delenv("VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE", raising=False)
    monkeypatch.delenv("VLLM_HUST_EPLB_CONTROL_PLANE_KILL_SWITCH", raising=False)
    _forbid_imports(monkeypatch)
    plugin.register()


def test_kill_switch_wins_over_enable(monkeypatch):
    monkeypatch.setenv("VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE", "1")
    monkeypatch.setenv("VLLM_HUST_EPLB_CONTROL_PLANE_KILL_SWITCH", "1")
    _forbid_imports(monkeypatch)
    plugin.register()
