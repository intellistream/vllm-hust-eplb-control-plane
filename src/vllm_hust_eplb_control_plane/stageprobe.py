"""Measurement-only probe: where does the EPLB planner spend its time?

Not part of the optimization. ``EplbWorker.do_update`` runs in the spawned EPLB process; this
probe wraps its stages with timers and prints one line per plan:

    EPLB_STAGE_PROBE total_ms=... fetch=... g2l=... policy=... hotness=... check=... l2g=...
        update_map=... pack_total=... compose=... log2phy=... pack_other=... other=... pid=...

Enabled with ``VLLM_HUST_EPLB_STAGE_PROBE=1`` (independent of the optimization switch, so the
original planner can be measured). Installed from the child entry in ``plugin.py``.

``compose_expert_update_info_greedy`` is a lazy generator: ``do_update`` only creates it, and the
work runs when ``pack_update_info`` consumes it. Its time is therefore summed over each ``next``.
``generate_log2phy_map`` is wrapped at module level. ``compose`` and ``log2phy`` run inside
``pack_total``; ``pack_other`` is what remains of it. ``other`` is ``total`` minus every timed
stage. Nothing here changes behaviour: wrappers only read the clock.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

MARKER = "__vllm_hust_eplb_stage_probe__"

# host method name -> label
_STAGES = {
    "fetch_and_sum_load_info": "fetch",
    "global2local": "g2l",
    "calculate_rebalance_experts": "policy",
    "_calculate_hotness": "hotness",
    "_compute_imbalance": "hotness",
    "check_expert_placement": "check",
    "local2global": "l2g",
    "update_expert_map": "update_map",
    "pack_update_info": "pack_total",
}

_acc: dict[str, float] = {}


def _add(label: str, started: float) -> None:
    _acc[label] = _acc.get(label, 0.0) + (time.perf_counter() - started) * 1e3


def _timed(label: str, fn: Any) -> Any:
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _add(label, started)

    wrapper.__wrapped__ = fn
    return wrapper


def _timed_generator(label: str, fn: Any) -> Any:
    def wrapper(*args, **kwargs):
        gen = fn(*args, **kwargs)
        while True:
            started = time.perf_counter()
            try:
                item = next(gen)
            except StopIteration:
                _add(label, started)
                return
            _add(label, started)
            yield item

    wrapper.__wrapped__ = fn
    return wrapper


def _wrap_in_class(cls: Any, name: str, label: str) -> None:
    if name not in cls.__dict__:
        return
    raw = cls.__dict__[name]
    if isinstance(raw, staticmethod):
        setattr(cls, name, staticmethod(_timed(label, raw.__func__)))
    else:
        setattr(cls, name, _timed(label, raw))


def _emit(total_ms: float) -> None:
    g = lambda k: _acc.get(k, 0.0)  # noqa: E731
    timed = sum(g(k) for k in ("fetch", "g2l", "policy", "hotness", "check", "l2g", "update_map", "pack_total"))
    pack_other = g("pack_total") - g("compose") - g("log2phy")
    fields = {
        "total_ms": total_ms,
        "fetch": g("fetch"),
        "g2l": g("g2l"),
        "policy": g("policy"),
        "hotness": g("hotness"),
        "check": g("check"),
        "l2g": g("l2g"),
        "update_map": g("update_map"),
        "pack_total": g("pack_total"),
        "compose": g("compose"),
        "log2phy": g("log2phy"),
        "pack_other": pack_other,
        "other": total_ms - timed,
    }
    body = " ".join(f"{k}={v:.3f}" for k, v in fields.items())
    print(f"EPLB_STAGE_PROBE {body} pid={os.getpid()}", file=sys.stderr, flush=True)


def install(worker_mod: Any) -> None:
    cls = worker_mod.EplbWorker
    if getattr(cls, MARKER, False):
        return
    for name, label in _STAGES.items():
        _wrap_in_class(cls, name, label)
    if "compose_expert_update_info_greedy" in cls.__dict__:
        cls.compose_expert_update_info_greedy = _timed_generator(
            "compose", cls.compose_expert_update_info_greedy
        )
    if hasattr(worker_mod, "generate_log2phy_map"):
        worker_mod.generate_log2phy_map = _timed("log2phy", worker_mod.generate_log2phy_map)

    original = cls.do_update

    def do_update(self):
        _acc.clear()
        started = time.perf_counter()
        result = original(self)
        total_ms = (time.perf_counter() - started) * 1e3
        if result is not None:  # a cycle with no load data yet returns early and is not a plan
            _emit(total_ms)
        return result

    cls.do_update = do_update
    setattr(cls, MARKER, True)
