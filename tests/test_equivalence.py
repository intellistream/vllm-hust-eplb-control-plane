"""Differential tests: replacements must match the verbatim host functions exactly.

Needs a real torch; skipped where it is missing (run on the NPU container).
"""

from __future__ import annotations

import random
import types

import pytest

torch = pytest.importorskip("torch")

from _host_reference import ref_compose_expert_update_info_greedy  # noqa: E402
from vllm_hust_eplb_control_plane import plugin  # noqa: E402


WORKER_MOD = types.SimpleNamespace(torch=torch)
COMPOSE = plugin._make_compose(WORKER_MOD)


def random_placement(rng: random.Random, world: int, num_experts: int, per_rank: int, redundant: int):
    """A valid expert map: [world, num_experts] of local slot ids or -1; every expert held >=1."""
    holders = [[] for _ in range(num_experts)]
    for e in range(num_experts):
        holders[e].append(rng.randrange(world))
    for _ in range(redundant):
        holders[rng.randrange(num_experts)].append(rng.randrange(world))
    m = torch.full((world, num_experts), -1, dtype=torch.int64)
    counts = [0] * world
    for e, rs in enumerate(holders):
        for r in set(rs):
            m[r, e] = counts[r]
            counts[r] += 1
    return m


def random_case(seed: int):
    rng = random.Random(seed)
    world = rng.choice([2, 4, 8, 16])
    num_experts = rng.choice([8, 16, 32, 64, 256])
    layers = rng.choice([1, 3, 10])
    redundant = rng.choice([0, 2, num_experts // 4])
    cur = torch.stack([random_placement(rng, world, num_experts, 0, redundant) for _ in range(layers)])
    upd = cur.clone()
    for layer in range(layers):
        if rng.random() < 0.3:
            continue  # unchanged layer exercises the guard clause
        upd[layer] = random_placement(rng, world, num_experts, 0, redundant)
    return cur, upd, world


@pytest.mark.parametrize("seed", range(60))
def test_compose_matches_reference(seed):
    cur, upd, _ = random_case(seed)
    worker = types.SimpleNamespace()
    ref = list(ref_compose_expert_update_info_greedy(worker, upd, cur))
    new = list(COMPOSE(worker, upd, cur))
    assert len(ref) == len(new)
    for (rs, rr, rm, rl), (ns, nr, nm, nl) in zip(ref, new):
        assert rs == ns
        assert rr == nr
        assert rl == nl
        assert torch.equal(rm, nm)
        # dict insertion order is observable downstream (iteration order); keep it identical.
        assert list(rs.keys()) == list(ns.keys())
        assert list(rr.keys()) == list(nr.keys())


def test_unchanged_layers_yield_empty_plans():
    cur, _, _ = random_case(3)
    worker = types.SimpleNamespace()
    out = list(COMPOSE(worker, cur, cur))
    assert len(out) == cur.shape[0]
    assert all(send == {} and recv == {} for send, recv, _m, _l in out)


def test_every_received_expert_has_a_matching_sender_entry():
    cur, upd, _ = random_case(11)
    worker = types.SimpleNamespace()
    for send, recv, _m, _l in COMPOSE(worker, upd, cur):
        sends = {(src, dst, e) for src, pairs in send.items() for dst, e in pairs}
        recvs = {(src, dst, e) for dst, pairs in recv.items() for src, e in pairs}
        assert sends == recvs
