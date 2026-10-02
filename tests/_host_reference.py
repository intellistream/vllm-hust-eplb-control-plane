"""Verbatim copies of the host functions this package replaces.

Source: /vllm-workspace/vllm-ascend (vllm-ascend 0.25.1rc2.dev125+hust.20260903),
``vllm_ascend/eplb/core/eplb_worker.py`` ``EplbWorker.compose_expert_update_info_greedy``.
It is the baseline for the differential test: the plugin's replacement must produce
identical output. Refresh it when the pinned host changes.
"""

from __future__ import annotations

from typing import Any

import torch


def ref_compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
    num_layers = current_expert_maps.shape[0]
    for layer_id in range(num_layers):
        updated_expert_maps_this_layer = updated_expert_maps[layer_id]
        current_expert_maps_this_layer = current_expert_maps[layer_id]

        expert_send_info_this_layer: dict[Any, Any] = {}
        expert_recv_info_this_layer: dict[Any, Any] = {}

        if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
            yield (
                expert_send_info_this_layer,
                expert_recv_info_this_layer,
                updated_expert_maps_this_layer,
                layer_id,
            )
            continue

        dst_rank_indices, experts_to_recv = torch.where(
            (current_expert_maps_this_layer == -1) & (updated_expert_maps_this_layer != -1)
        )

        src_rank_indices, experts_to_send = torch.where(
            (current_expert_maps_this_layer != -1) & (updated_expert_maps_this_layer == -1)
        )

        for idx in range(len(dst_rank_indices)):
            dst_rank_id = dst_rank_indices[idx].item()
            expert_id = experts_to_recv[idx].item()
            if dst_rank_id not in expert_recv_info_this_layer:
                expert_recv_info_this_layer[dst_rank_id] = []

            if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
            else:
                candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

            src_rank_id = candidate_src_rank_indices[0].item()
            if src_rank_id not in expert_send_info_this_layer:
                expert_send_info_this_layer[src_rank_id] = []

            expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
            expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

        yield (
            expert_send_info_this_layer,
            expert_recv_info_this_layer,
            updated_expert_maps_this_layer,
            layer_id,
        )
