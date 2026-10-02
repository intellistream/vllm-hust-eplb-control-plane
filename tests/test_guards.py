"""The plugin must refuse hosts whose code does not look like what it replaces."""

from __future__ import annotations

import pytest

from vllm_hust_eplb_control_plane import plugin


class GoodWorker:
    def compose_expert_update_info_greedy(self, updated, current):
        if not torch.isin(torch.tensor(expert_id), experts_to_send).any():  # noqa: F821
            pass


class AlreadyFastComposeWorker:
    def compose_expert_update_info_greedy(self, updated, current):
        return send_src_by_expert  # noqa: F821


class NoMethodWorker:
    pass


class GoodProcess:
    def _launch_process(self):
        proc = Process(target=self.worker_process, args=(self.planner_q, self.block_update_q), daemon=True)  # noqa: F821
        proc.start()
        return proc


class AlreadyBootstrappedProcess:
    def _launch_process(self):
        return Process(target=other_entry, args=(self,), daemon=True)  # noqa: F821


class NoLauncherProcess:
    pass


def test_good_host_passes():
    plugin._supported_worker(GoodWorker)
    plugin._supported_launcher(GoodProcess)


def test_compose_refused_when_already_fast():
    with pytest.raises(RuntimeError, match="isin"):
        plugin._supported_worker(AlreadyFastComposeWorker)


def test_compose_refused_when_method_missing():
    with pytest.raises(RuntimeError, match="cannot find"):
        plugin._supported_worker(NoMethodWorker)


def test_launcher_refused_when_launch_differs():
    with pytest.raises(RuntimeError, match="would not reach"):
        plugin._supported_launcher(AlreadyBootstrappedProcess)


def test_launcher_refused_when_missing():
    with pytest.raises(RuntimeError, match="cannot find"):
        plugin._supported_launcher(NoLauncherProcess)
