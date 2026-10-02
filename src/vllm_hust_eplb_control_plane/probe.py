"""Measurement-only probe: how long does the main process wait for the EPLB plan?

Not part of the optimization. It exists so that an OFF arm (patch not installed) and an ON
arm can be compared on the one quantity through which the planner can reach serving:
the time the model-runner process blocks on ``block_update_q.get()``.

Default-off; enabled with ``VLLM_HUST_EPLB_WAIT_PROBE=1`` and independent of
``VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE``. It only wraps ``EplbUpdator.forward_before`` and times
the call on the single iteration where the host fetches the plan
(``get_update_info_flag()``), where that call does nothing but the blocking ``get``.
Each wait is printed as ``EPLB_WAIT_PROBE wait_ms=... pid=...``.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from typing import Any

PROBE_MARKER = "__vllm_hust_eplb_wait_probe__"


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _install(updator_mod: Any) -> None:
    cls = updator_mod.EplbUpdator
    if getattr(cls, PROBE_MARKER, False):
        return
    source = inspect.getsource(cls.forward_before)
    if "block_update_q.get()" not in source or "get_update_info_flag()" not in source:
        raise RuntimeError(
            "eplb wait probe cannot find the expected `block_update_q.get()` fetch in "
            "EplbUpdator.forward_before; unsupported host"
        )
    original = cls.forward_before

    def forward_before(self):
        if not self.get_update_info_flag():
            return original(self)
        started = time.perf_counter()
        result = original(self)
        waited_ms = (time.perf_counter() - started) * 1e3
        print(f"EPLB_WAIT_PROBE wait_ms={waited_ms:.3f} pid={os.getpid()}", file=sys.stderr, flush=True)
        return result

    cls.forward_before = forward_before
    setattr(cls, PROBE_MARKER, True)


def register() -> None:
    if not _enabled("VLLM_HUST_EPLB_WAIT_PROBE"):
        return
    from vllm_ascend.eplb import eplb_updator

    _install(eplb_updator)
