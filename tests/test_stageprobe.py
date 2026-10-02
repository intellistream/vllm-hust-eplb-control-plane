from __future__ import annotations

import time
import types

from vllm_hust_eplb_control_plane import stageprobe


def make_worker_mod():
    def generate_log2phy_map(*args, **kwargs):
        time.sleep(0.004)
        return "map"

    class EplbWorker:
        def fetch_and_sum_load_info(self):
            time.sleep(0.001)
            return "load"

        def global2local(self, m, n):
            return m

        def calculate_rebalance_experts(self, load, old):
            time.sleep(0.003)
            return True, None, "new"

        def _calculate_hotness(self, a, b):
            return 1

        @staticmethod
        def _compute_imbalance(a, b, return_list=False):
            time.sleep(0.001)
            return 1, 1, []

        def check_expert_placement(self, a, b):
            return None

        def local2global(self, p):
            return p

        def update_expert_map(self, m):
            return None

        def compose_expert_update_info_greedy(self, new, old):
            for layer in range(3):
                time.sleep(0.002)
                yield layer

        def pack_update_info(self, gen):
            out = []
            for layer in gen:
                out.append((layer, generate_log2phy_map_ref[0]()))
            return out

        def do_update(self):
            if getattr(self, "skip", False):
                return None
            self.fetch_and_sum_load_info()
            old = self.global2local("o", 1)
            self.calculate_rebalance_experts("l", old)
            self._calculate_hotness(old, "l")
            self._compute_imbalance(old, "h", return_list=True)
            self.check_expert_placement(old, "n")
            new = self.local2global("n")
            self.update_expert_map(new)
            info = self.compose_expert_update_info_greedy(new, old)
            return self.pack_update_info(info)

    mod = types.SimpleNamespace(EplbWorker=EplbWorker, generate_log2phy_map=generate_log2phy_map)
    # pack_update_info resolves the module-level function at call time, like the host does
    generate_log2phy_map_ref = [lambda: mod.generate_log2phy_map()]
    return mod


def parse(err):
    line = [ln for ln in err.splitlines() if ln.startswith("EPLB_STAGE_PROBE")][-1]
    return {k: float(v) for k, v in (kv.split("=") for kv in line.split()[1:])}


def test_each_stage_is_attributed_and_results_are_unchanged(capsys):
    mod = make_worker_mod()
    expected = mod.EplbWorker().do_update()
    capsys.readouterr()
    stageprobe.install(mod)
    worker = mod.EplbWorker()
    assert worker.do_update() == expected  # measurement must not change behaviour
    f = parse(capsys.readouterr().err)
    assert f["compose"] >= 5.5  # three lazy `next` calls of ~2 ms each, summed
    assert f["log2phy"] >= 11.0  # three calls of ~4 ms
    assert f["policy"] >= 2.5 and f["fetch"] >= 0.8 and f["hotness"] >= 0.8
    assert f["pack_total"] >= f["compose"] + f["log2phy"]
    assert abs(f["pack_other"] - (f["pack_total"] - f["compose"] - f["log2phy"])) < 1e-2
    assert f["total_ms"] >= f["pack_total"]


def test_compose_time_is_counted_inside_the_consumer_not_at_call_site(capsys):
    mod = make_worker_mod()
    stageprobe.install(mod)
    worker = mod.EplbWorker()
    gen = worker.compose_expert_update_info_greedy("n", "o")
    stageprobe._acc.clear()
    assert stageprobe._acc.get("compose", 0.0) == 0.0  # creating the generator does no work
    list(gen)
    assert stageprobe._acc["compose"] >= 5.5
    capsys.readouterr()


def test_staticmethod_is_preserved(capsys):
    mod = make_worker_mod()
    stageprobe.install(mod)
    assert isinstance(mod.EplbWorker.__dict__["_compute_imbalance"], staticmethod)
    assert mod.EplbWorker._compute_imbalance(1, 2, return_list=True) == (1, 1, [])
    capsys.readouterr()


def test_cycle_without_data_is_not_reported_as_a_plan(capsys):
    mod = make_worker_mod()
    stageprobe.install(mod)
    worker = mod.EplbWorker()
    worker.skip = True
    assert worker.do_update() is None
    assert "EPLB_STAGE_PROBE" not in capsys.readouterr().err


def test_install_is_idempotent(capsys):
    mod = make_worker_mod()
    stageprobe.install(mod)
    first = mod.EplbWorker.do_update
    stageprobe.install(mod)
    assert mod.EplbWorker.do_update is first
    capsys.readouterr()
