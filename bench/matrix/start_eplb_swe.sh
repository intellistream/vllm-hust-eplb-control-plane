#!/bin/bash
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export DYNAMIC_EPLB=true
export VLLM_HUST_EPLB_CONTROL_PLANE_ENABLE=${ECP_ENABLE:-0}
export VLLM_HUST_EPLB_CONTROL_PLANE_EVIDENCE=1
export VLLM_HUST_EPLB_WAIT_PROBE=1
exec vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b \
  --host 127.0.0.1 --port 18180 \
  --dtype bfloat16 --block-size 128 \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --enable-chunked-prefill \
  --distributed-executor-backend mp --disable-custom-all-reduce \
  --additional-config "{\"eplb_config\":{\"dynamic_eplb\":true,\"expert_heat_collection_interval\":600,\"algorithm_execution_interval\":${EPLB_ALGO:-50},\"num_redundant_experts\":0}}" \
  --compilation-config "{\"mode\":3,\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}" \
  --cudagraph-capture-sizes 1 2 4 8 16
