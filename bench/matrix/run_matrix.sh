#!/bin/bash
# name algo ecp
MATRIX="D-off 50 0
D-on 50 1
S-off-a 10 0
S-on-a 10 1
S-on-b 10 1
S-off-b 10 0"
cd /root/workspace/swe-prefix-reuse
M=/root/workspace/eplb_matrix
echo "$MATRIX" | while read name algo ecp; do
  echo "[$(date -u +%T)] START $name algo=$algo ecp=$ecp" >> $M/progress.log
  pkill -TERM -f "[V]LLM::"; pkill -TERM -f "[v]llm serve"; sleep 15
  if ps -eo cmd | grep -qE "[V]LLM::|[v]llm serve"; then echo "[$(date -u +%T)] ABORT leftover processes" >> $M/progress.log; exit 1; fi
  arm=off; [ "$ecp" = 1 ] && arm=on
  cat > $M/$name.meta.json <<EOM
{"engine":"vLLM + vLLM-Ascend","engine_version":"0.28.1.post1.dev143 / 0.25.1rc2.dev125+hust.20260903","mod":"org.vllm-hust.eplb-control-plane","mod_version":"0.1.0.dev0","mod_enabled":"$arm","model":"Qwen3.5-35B-A3B","precision":"bfloat16","tokenizer":"/models/Qwen3.5-35B-A3B","chip":"Ascend 910B2","chip_count":2,"launch_command":"/root/start_eplb_swe.sh with DYNAMIC_EPLB=true eplb collect=600 algo=$algo ECP_ENABLE=$ecp"}
EOM
  ECP_ENABLE=$ecp EPLB_ALGO=$algo nohup setsid /root/start_eplb_swe.sh > $M/$name.server.log 2>&1 < /dev/null &
  ok=0
  for i in $(seq 1 90); do
    if grep -q "startup complete" $M/$name.server.log; then ok=1; break; fi
    if grep -q "ERROR" $M/$name.server.log; then break; fi
    sleep 10
  done
  if [ $ok != 1 ]; then echo "[$(date -u +%T)] FAIL-START $name" >> $M/progress.log; continue; fi
  swe-prefix-reuse run --workload prepared/qwen35.json --endpoint http://127.0.0.1:18180/v1/completions --model qwen3.5-35b-a3b --server-max-context 262144 --concurrency 8 --duration 20 --chips 2 --server-metadata $M/$name.meta.json --output results/eplb-$name-probe > $M/$name.probe.log 2>&1
  swe-prefix-reuse run --workload prepared/qwen35.json --endpoint http://127.0.0.1:18180/v1/completions --model qwen3.5-35b-a3b --server-max-context 262144 --concurrency 8 --duration 900 --chips 2 --server-metadata $M/$name.meta.json --output results/eplb-$name > $M/$name.bench.log 2>&1
  pkill -TERM -f "[V]LLM::"; pkill -TERM -f "[v]llm serve"; sleep 15
  python $M/collect.py eplb-$name $M/$name.server.log $algo $arm >> $M/progress.log 2>&1 || echo "[$(date -u +%T)] COLLECT-FAIL $name" >> $M/progress.log
  echo "[$(date -u +%T)] DONE $name" >> $M/progress.log
done
echo "[$(date -u +%T)] ALL-DONE" >> $M/progress.log
