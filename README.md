# vllm-hust-eplb-control-plane

Standalone vLLM `general_plugin` that speeds up the expert-rebalancing **planner** of dynamic
EPLB (expert-parallel load balancing) in vLLM-Ascend-HUST, by replacing one host function,
`EplbWorker.compose_expert_update_info_greedy`, at runtime.

It is one mechanism extracted from the ledger candidate `#36` (`vllm-hust-legacy017-perf`,
patchset `ascend-eplb-control-plane`, Ascend commit `46e5edf0`). That patch bundled three
changes; the other two were measured and dropped (see "What was dropped"). This package does
**not** import `vllm-hust-legacy017-perf`.

## Status: candidate, not performance-admitted

Packaging this as an installable, default-off plugin is **not** a claim that it speeds up
serving. What has actually been established (host: vLLM-HUST `0.28.1.post1.dev143`,
vLLM-Ascend-HUST `0.25.1rc2.dev125`, 2x Ascend 910B2, Qwen3.5-35B-A3B, TP2 + EP):

- **Equivalence (tested).** 76 tests pass. The differential test runs the replacement against a
  verbatim copy of the host function on random expert placements (world 2-16, up to 256
  experts, 1-10 layers, unchanged layers, redundant experts) and requires identical output,
  including dict key order.
- **Reachability (measured on real serving).** With dynamic EPLB on, `runtime_effective` was
  emitted in both EPLB side processes (the pids that the host logs as "Launched EPLB
  subprocess"), with no errors, and nothing from the two dropped parts.
- **Component speed (microbenchmark, `bench/microbench.py`, CPU, median of 7):**

  | world | layers | expert churn | original | patched | speedup |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2 | 40 | 10% / 90% | 108 / 721 ms | 2.0 / 12.3 ms | 53x / 59x |
  | 8 | 40 | 10% / 90% | 207 / 1442 ms | 2.9 / 18.2 ms | 72x / 79x |
  | 16 | 58 | 10% / 90% | 410 / 2470 ms | 6.4 / 32.2 ms | 64x / 77x |
  | 32 | 58 | 10% / 90% | 424 / 2596 ms | 7.9 / 37.7 ms | 54x / 69x |

  In serving, the patched planner takes ~0.2 ms per changed layer.
- **End-to-end (swe-prefix-reuse C8, 900 s): no advantage.** Six valid runs, default planner
  window (50 iterations, OFF/ON) and a narrow window (10 iterations, OFF/ON/ON/OFF). Output
  throughput differs by -0.02% (default) and -0.03% (narrow); decode P90, TPOT and E2E are flat
  within noise. The time the main process waits for the plan did **not** shrink (it is +4.5% to
  +7.3% in ON, smaller than the 5.9% spread between the two OFF runs). The one favourable signal,
  narrow-window TTFT P95 -7.6%, has two repeats, no matching movement in other metrics and no
  mechanism, so it is **unconfirmed**. The likely reason is that `compose` is not the dominant
  part of planning time; that has **not** been verified by timing the planner's stages.
  Full tables, caveats and the ledger-gate status are in [docs/REPORT.md](docs/REPORT.md).

## The EPLB process is spawned

`EplbWorker` runs in a side process started by `EplbProcess._launch_process` with
`multiprocessing` **spawn**: a fresh interpreter that does not run vLLM plugin loading.
Patching the class in the parent is therefore not enough. In the first real run the planner
patch emitted **0** `runtime_effective` events while the planner was demonstrably running the
original code. The ledger's source patch never had this problem because it edits the file the
child imports.

The plugin therefore also replaces `EplbProcess._launch_process` so the child starts at a
module-level entry (`_child_entry`, pickled by reference) that installs the planner patch and
then runs the host's own `worker_process`. It is guarded like the planner patch, and the child
honours the same enable and kill-switch variables. The `pid` on every evidence event is what
shows the child really ran the patch.

## What was dropped

- **Batched `pack_update_info`:** 1.0x. Its time is dominated by `generate_log2phy_map`
  (per-element `.item()` loops), not by the `tolist` calls it batched. That function is the
  real cost there and would be a different candidate.
- **Reusable CPU buffer in `compute_and_set_moe_load`:** ~20 us saved on a call that takes
  ~18 ms in serving (all-gather and manager-dict pickling dominate), once per EPLB cycle.
  Negligible.

## Differences from the ledger patch

- The host has drifted since the ledger diff was cut, so the replacement is written against
  the current host source, and the guard refuses hosts that look different.
- A missing holder for an expert now raises `KeyError` instead of `IndexError`. Valid
  placements always have a holder.

## Activation

| Variable | Effect |
| --- | --- |
| `VLLM_HUST_EPLB_CONTROL_PLANE_KILL_SWITCH=1` | Always wins. |
| `VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE=1` | Required to install. Default off: discovered, does nothing, imports no host module. `vllm-hust-ext extension enable` alone does **not** set it. |
| `VLLM_HUST_EPLB_CONTROL_PLANE_EVIDENCE=1` | Emits `LEGACY017_EVIDENCE installed ...`, one `runtime_effective ...` per process, and `timing part=greedy_plan_layer ms=...` lines, each with `pid`. |

EPLB itself is off by default. To exercise this package: `DYNAMIC_EPLB=true` plus
`--additional-config '{"eplb_config":{"dynamic_eplb":true,...}}'`.

Each patch refuses (`RuntimeError`) unless the exact expected source marker is present, so an
unrecognised or already-optimised host is never patched speculatively.

## Measurement probe

`probe.py` is a separate, default-off entry point (`VLLM_HUST_EPLB_WAIT_PROBE=1`, independent of
`ENABLE`) that prints `EPLB_WAIT_PROBE wait_ms=... pid=...` each time the main process fetches the
plan, so the OFF arm can be measured on the same quantity. It is instrumentation, not part of the
optimization.

## Testing

```
python -m pytest tests/        # 14 tests run without torch; the rest need torch
python bench/microbench.py     # on the NPU container, after sourcing the ATB env
```

`tests/_host_reference.py` holds the verbatim host function used as the baseline; refresh it
when the pinned host changes.
