"""The spawned EPLB process must install the worker patches before running the host loop."""

from __future__ import annotations

import pickle
import sys
import types

from vllm_hust_eplb_control_plane import plugin


class FakeProcess:
    instances: list = []

    def __init__(self, target, args, daemon):
        self.target, self.args, self.daemon, self.started = target, args, daemon, False
        FakeProcess.instances.append(self)

    def start(self):
        self.started = True


def test_launcher_targets_module_level_entry_and_passes_self():
    launch = plugin._make_launch_process(types.SimpleNamespace(Process=FakeProcess))
    owner = types.SimpleNamespace(planner_q="pq", block_update_q="bq")
    proc = launch(owner)
    assert proc.started and proc.daemon is True
    assert proc.target is plugin._child_entry
    assert proc.args == (owner, "pq", "bq")


def test_entry_is_picklable_by_reference_for_spawn():
    # spawn pickles the Process target; a closure or bound method of a patched class would not survive.
    assert pickle.loads(pickle.dumps(plugin._child_entry)) is plugin._child_entry


def _run_entry(monkeypatch, env):
    for key in ("ENABLE", "KILL_SWITCH"):
        monkeypatch.delenv(f"VLLM_HUST_EPLB_CONTROL_PLANE_{key}", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(f"VLLM_HUST_EPLB_CONTROL_PLANE_{key}", value)
    calls = []
    fake_worker_mod = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    monkeypatch.setitem(sys.modules, "vllm_ascend.eplb", types.ModuleType("vllm_ascend.eplb"))
    core = types.ModuleType("vllm_ascend.eplb.core")
    core.eplb_worker = fake_worker_mod
    monkeypatch.setitem(sys.modules, "vllm_ascend.eplb.core", core)
    monkeypatch.setitem(sys.modules, "vllm_ascend.eplb.core.eplb_worker", fake_worker_mod)
    monkeypatch.setattr(
        plugin,
        "_install",
        lambda mod, install_launcher=True: calls.append(("install", install_launcher)),
    )
    host = types.SimpleNamespace(worker_process=lambda pq, bq: calls.append(("run", pq, bq)))
    plugin._child_entry(host, "pq", "bq")
    return calls


def test_entry_installs_then_runs_host_loop(monkeypatch):
    calls = _run_entry(monkeypatch, {"ENABLE": "1"})
    assert [c[0] for c in calls] == ["install", "run"]
    assert calls[0][1] is False  # the child never re-launches an EPLB process
    assert calls[1] == ("run", "pq", "bq")


def test_entry_respects_kill_switch_and_disabled_state(monkeypatch):
    assert [c[0] for c in _run_entry(monkeypatch, {"ENABLE": "1", "KILL_SWITCH": "1"})] == ["run"]
    assert [c[0] for c in _run_entry(monkeypatch, {})] == ["run"]
