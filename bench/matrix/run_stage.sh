#!/bin/bash
# Stage-timing runs: original planner (OFF) vs patched planner (ON), default planner window.
# The swe-prefix-reuse run is only a load generator here; its scores are not reported.
cd /root/workspace/swe-prefix-reuse
M=/root/workspace/eplb_matrix
echo "T-off 0
T-on 1" | while read name ecp; do
  echo "[$(date -u +%T)] START $name ecp=$ecp" >> $M/stage_progress.log
  pkill -TERM -f "[V]LLM::"; pkill -TERM -f "[v]llm serve"; sleep 15
  if ps -eo cmd | grep -qE "[V]LLM::|[v]llm serve"; then echo "[$(date -u +%T)] ABORT leftover processes" >> $M/stage_progress.log; exit 1; fi
  ECP_ENABLE=$ecp EPLB_ALGO=50 STAGE=1 nohup setsid /root/start_eplb_swe.sh > $M/$name.server.log 2>&1 < /dev/null &
  ok=0
  for i in $(seq 1 90); do
    if grep -q "startup complete" $M/$name.server.log; then ok=1; break; fi
    if grep -q "ERROR" $M/$name.server.log; then break; fi
    sleep 10
  done
  if [ $ok != 1 ]; then echo "[$(date -u +%T)] FAIL-START $name" >> $M/stage_progress.log; continue; fi
  swe-prefix-reuse run --workload prepared/qwen35.json --endpoint http://127.0.0.1:18180/v1/completions --model qwen3.5-35b-a3b --server-max-context 262144 --concurrency 8 --duration 400 --chips 2 --server-metadata $M/D-off.meta.json --output results/eplb-stage-$name > $M/$name.bench.log 2>&1
  pkill -TERM -f "[V]LLM::"; pkill -TERM -f "[v]llm serve"; sleep 15
  echo "[$(date -u +%T)] DONE $name" >> $M/stage_progress.log
done
echo "[$(date -u +%T)] ALL-DONE" >> $M/stage_progress.log
