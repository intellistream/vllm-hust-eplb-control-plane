"""vLLM general-plugin entry point for the EPLB greedy-plan candidate.

Import-safe and default-off. The enable switch must be explicit; the kill
switch always wins.

One mechanism: ``EplbWorker.compose_expert_update_info_greedy`` builds per-expert
source tables once per layer instead of calling ``torch.isin`` / ``torch.where``
per expert. Output is identical to the host function (see tests).

``EplbWorker`` runs in the EPLB side process, which the host starts with
``multiprocessing`` *spawn*: a fresh interpreter that never runs vLLM plugin
loading. Patching the class in the parent is therefore not enough (measured: 0
``runtime_effective`` events from the child). This plugin also replaces
``EplbProcess._launch_process`` so the child starts at a module-level entry that
installs the same patch and then runs the host's own ``worker_process``. The
``runtime_effective`` event carries the pid so execution in the child is checked
rather than assumed.

This package never imports torch or any host runtime module at import time.
See README.md for the (candidate, not performance-admitted) evidence status.
"""

from __future__ import annotations

import inspect
import logging
import os
import sys
import time
from typing import Any

LOGGER = logging.getLogger(__name__)
PATCH_MARKER = "__vllm_hust_eplb_control_plane__"


def _emit_evidence(message: str) -> None:
    """Emit an audit event before vLLM configures worker-process logging."""
    print(f"LEGACY017_EVIDENCE {message} pid={os.getpid()}", file=sys.stderr, flush=True)


def _enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _evidence_on() -> bool:
    return _enabled("VLLM_HUST_EPLB_CONTROL_PLANE_EVIDENCE")


_runtime_effective_emitted = False


def _note_runtime_effective() -> None:
    """Emit one `runtime_effective` event per process the first time the patched
    planner actually produces a plan for a changed layer."""
    global _runtime_effective_emitted
    if _runtime_effective_emitted or not _evidence_on():
        return
    _runtime_effective_emitted = True
    _emit_evidence("runtime_effective mechanism=eplb_greedy_plan_tables")


# --------------------------------------------------------------------------- guards


def _supported_worker(worker_cls: Any) -> None:
    """Refuse to patch an implementation that doesn't look like the one this
    package was ported from -- never patch speculatively."""
    if not hasattr(worker_cls, "compose_expert_update_info_greedy"):
        raise RuntimeError("eplb-control-plane cannot find EplbWorker.compose_expert_update_info_greedy")
    source = inspect.getsource(worker_cls.compose_expert_update_info_greedy)
    if "torch.isin(torch.tensor(expert_id), experts_to_send)" not in source:
        raise RuntimeError(
            "eplb-control-plane cannot find the expected per-expert `torch.isin` lookup in "
            "EplbWorker.compose_expert_update_info_greedy; this release is unsupported or "
            "already contains an equivalent optimization"
        )


def _supported_launcher(process_cls: Any) -> None:
    expected = "Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)"
    if not hasattr(process_cls, "_launch_process"):
        raise RuntimeError("eplb-control-plane cannot find EplbProcess._launch_process")
    if expected not in inspect.getsource(process_cls._launch_process):
        raise RuntimeError(
            "eplb-control-plane cannot find the expected EPLB subprocess launch in "
            "EplbProcess._launch_process; the worker patch would not reach the child "
            "process on this release"
        )


# ----------------------------------------------------------------- replacements


def _make_compose(worker_mod: Any) -> Any:
    """Build the replacement, closing over the host module's own already-imported
    torch so this plugin module never imports it itself."""
    torch = worker_mod.torch

    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer: dict[Any, Any] = {}
            expert_recv_info_this_layer: dict[Any, Any] = {}

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (
                    expert_send_info_this_layer,
                    expert_recv_info_this_layer,
                    updated_expert_maps_this_layer,
                    layer_id,
                )
                continue

            started = time.perf_counter()
            dst_rank_indices, experts_to_recv = torch.where(
                (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
            )
            src_rank_indices, experts_to_send = torch.where(
                (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
            )

            # First sender (lowest rank, in torch.where order) per expert.
            send_src_by_expert: dict[int, int] = {}
            for src_rank_id, expert_id in zip(src_rank_indices.tolist(), experts_to_send.tolist()):
                send_src_by_expert.setdefault(expert_id, src_rank_id)

            # First holder (lowest rank) per expert, for experts nobody sends out.
            holder_src_by_expert: dict[int, int] = {}
            holder_rank_indices, holder_expert_ids = torch.where(current_expert_maps_this_layer != -1)
            for src_rank_id, expert_id in zip(holder_rank_indices.tolist(), holder_expert_ids.tolist()):
                holder_src_by_expert.setdefault(expert_id, src_rank_id)

            for dst_rank_id, expert_id in zip(dst_rank_indices.tolist(), experts_to_recv.tolist()):
                expert_recv_info_this_layer.setdefault(dst_rank_id, [])

                src_rank_id = send_src_by_expert.get(expert_id)
                if src_rank_id is None:
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    src_rank_id = holder_src_by_expert[expert_id]

                # TODO (upstream): improve selection criterion of NPU sending expert_id,
                # considering intra-node or inter-node...
                expert_send_info_this_layer.setdefault(src_rank_id, [])
                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            _note_runtime_effective()
            if _evidence_on():
                _emit_evidence(f"timing part=greedy_plan_layer ms={(time.perf_counter() - started) * 1e3:.4f}")
            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )

    return compose_expert_update_info_greedy


def _child_entry(eplb_process: Any, planner_q: Any, block_update_q: Any) -> None:
    """Target of the spawned EPLB process (module level so it pickles by reference).

    Installs the planner patch in the fresh interpreter, then runs the host's own
    ``worker_process`` unchanged.
    """
    if not _enabled("VLLM_HUST_EPLB_CONTROL_PLANE_KILL_SWITCH") and _enabled(
        "VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE"
    ):
        from vllm_ascend.eplb.core import eplb_worker

        # The child never spawns another EPLB process, so no launcher is needed here.
        _install(eplb_worker, install_launcher=False)
    eplb_process.worker_process(planner_q, block_update_q)


def _make_launch_process(worker_mod: Any) -> Any:
    Process = worker_mod.Process

    def _launch_process(self):
        """Launch the EPLB subprocess through the patch-installing entry point."""
        proc = Process(
            target=_child_entry,
            args=(self, self.planner_q, self.block_update_q),
            daemon=True,
        )
        proc.start()
        return proc

    return _launch_process


# ------------------------------------------------------------------ installation


def _install(worker_mod: Any, install_launcher: bool = True) -> None:
    worker_cls = worker_mod.EplbWorker
    process_cls = worker_mod.EplbProcess
    if not getattr(worker_cls, PATCH_MARKER, False):
        _supported_worker(worker_cls)
        worker_cls.compose_expert_update_info_greedy = _make_compose(worker_mod)
        setattr(worker_cls, PATCH_MARKER, True)
        LOGGER.info("eplb-control-plane installed greedy_plan on %s", worker_cls.__name__)
        if _evidence_on():
            _emit_evidence(f"installed part=greedy_plan class={worker_cls.__name__}")
    if install_launcher and not getattr(process_cls, PATCH_MARKER, False):
        _supported_launcher(process_cls)
        process_cls._launch_process = _make_launch_process(worker_mod)
        setattr(process_cls, PATCH_MARKER, True)
        LOGGER.info("eplb-control-plane installed child_bootstrap on %s", process_cls.__name__)
        if _evidence_on():
            _emit_evidence(f"installed part=child_bootstrap class={process_cls.__name__}")


def register() -> None:
    """Entry point invoked by vLLM in API, engine-core, and worker processes."""
    if _enabled("VLLM_HUST_EPLB_CONTROL_PLANE_KILL_SWITCH"):
        LOGGER.warning("eplb-control-plane kill switch is active; not installed")
        return
    if not _enabled("VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE"):
        LOGGER.info("eplb-control-plane is discovered but disabled")
        return

    from vllm_ascend.eplb.core import eplb_worker

    _install(eplb_worker)
