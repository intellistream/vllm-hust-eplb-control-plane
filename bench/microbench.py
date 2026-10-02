"""Microbenchmark the replaced host planner against the verbatim original.

Run on the NPU container (needs the real vllm_ascend for the host module). Prints a
table; makes no claim about end-to-end throughput -- the planner runs in the EPLB side
process, off the serving critical path.

    source /usr/local/Ascend/nnal/atb/set_env.sh
    python bench/microbench.py
"""

from __future__ import annotations

import os
import random
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402
from _host_reference import ref_compose_expert_update_info_greedy  # noqa: E402
from test_equivalence import random_placement  # noqa: E402
from vllm_hust_eplb_control_plane import plugin  # noqa: E402
from vllm_ascend.eplb.core import eplb_worker as worker_mod  # noqa: E402


def timeit(fn, repeat=7):
    samples = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t) * 1e3)
    return statistics.median(samples)


def plan_case(world, experts, layers, churn, seed=0):
    rng = random.Random(seed)
    cur = torch.stack([random_placement(rng, world, experts, 0, experts // 8) for _ in range(layers)])
    upd = cur.clone()
    for layer in range(layers):
        if rng.random() < churn:
            upd[layer] = random_placement(rng, world, experts, 0, experts // 8)
    return cur, upd


def main() -> None:
    compose_new = plugin._make_compose(worker_mod)
    w = types.SimpleNamespace()
    print("## compose_expert_update_info_greedy (CPU, EPLB process); ms, median of 7")
    print(f"{'world':>5} {'experts':>7} {'layers':>6} {'churn':>5} | {'original':>9} {'patched':>9} {'speedup':>8}")
    for world, experts, layers in [(2, 256, 40), (8, 256, 40), (16, 256, 58), (32, 256, 58)]:
        for churn in (0.1, 0.9):
            cur, upd = plan_case(world, experts, layers, churn)
            ref = timeit(lambda: list(ref_compose_expert_update_info_greedy(w, upd, cur)))
            new = timeit(lambda: list(compose_new(w, upd, cur)))
            print(f"{world:>5} {experts:>7} {layers:>6} {churn:>5} | {ref:>9.2f} {new:>9.2f} {ref / new:>7.1f}x")


if __name__ == "__main__":
    main()
