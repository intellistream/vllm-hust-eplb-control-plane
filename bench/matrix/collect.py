import json, re, statistics, sys
name, log, algo, arm = sys.argv[1:5]
root = "/root/workspace/swe-prefix-reuse/results/" + name
s = json.load(open(root + "/summary.json")); c = json.load(open(root + "/config.json"))
R = [json.loads(l) for l in open(root + "/requests.jsonl")]
inw = [r for r in R if r["success"] and r["end"] <= 900] if max(r["end"] for r in R) < 1e9 else [r for r in R if r["success"] and r["end"] - c["started_at_unix"] <= 900]
def p(a, q):
    a = sorted(a); return a[min(len(a) - 1, int(round(q * (len(a) - 1))))]
tp = [(r["last_token"] - r["first_token"]) / (len(r["token_ids"]) - 1) for r in inw if len(r["token_ids"]) > 1]
L = open(log).read()
waits = [float(m) for m in re.findall(r"EPLB_WAIT_PROBE wait_ms=([0-9.]+)", L)]
# first fetch per process is the cycle after warmup; keep all, report count
eff = sorted(set(re.findall(r"runtime_effective mechanism=eplb_greedy_plan_tables pid=(\d+)", L)))
kids = re.findall(r"Launched EPLB subprocess, pid=(\d+)", L)
out = dict(name=name, arm=arm, algo_interval=int(algo), valid=s["valid"], failed=s["failed_requests"],
  completed=s["requests_completed_in_window"], out_tps=s["output_tokens_per_second"], tps_chip=s["output_tokens_per_second_per_chip"],
  dec_p90=s["decode_tokens_per_second_p90"], ttft_p50_ms=p([r["ttft_seconds"] for r in inw], .5) * 1e3,
  ttft_p95_ms=s["ttft_seconds_p95"] * 1e3, tpot_ms=sum(tp) / len(tp) * 1e3, tpot_p95_ms=p(tp, .95) * 1e3,
  e2e_p95_ms=p([r["e2e_seconds"] for r in inw], .95) * 1e3, max_prompt=s["max_prompt_tokens_observed"],
  sessions_completed=s["sessions_completed"], inflight=s["mean_client_inflight"], run_id=c["run_id"],
  plans=len(re.findall("Expert hotness imbalance", L)),
  wait_n=len(waits), wait_mean_ms=(statistics.mean(waits) if waits else None), wait_median_ms=(statistics.median(waits) if waits else None),
  wait_max_ms=(max(waits) if waits else None), wait_sum_ms=sum(waits) if waits else 0,
  wait_values_ms=waits, eplb_child_pids=kids, runtime_effective_pids=eff,
  patch_installed_lines=len(re.findall("installed part=greedy_plan", L)))
json.dump(out, open("/root/workspace/eplb_matrix/" + name + ".json", "w"), indent=1)
print(name, "out_tps=%.2f dec_p90=%.2f wait_n=%d wait_med=%s wait_max=%s eff=%d" % (out["out_tps"], out["dec_p90"], out["wait_n"], out["wait_median_ms"], out["wait_max_ms"], len(eff)))
