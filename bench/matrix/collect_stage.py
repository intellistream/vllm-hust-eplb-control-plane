import json, re, statistics as st, sys
out = {}
for name in ("T-off", "T-on"):
    L = open(f"/root/workspace/eplb_matrix/{name}.server.log").read()
    rows = []
    for m in re.finditer(r"EPLB_STAGE_PROBE (.*?) pid=(\d+)", L):
        d = {k: float(v) for k, v in (kv.split("=") for kv in m.group(1).split())}
        d["pid"] = int(m.group(2)); rows.append(d)
    out[name] = rows
json.dump(out, open("/root/workspace/eplb_matrix/stage_raw.json", "w"), indent=1)
keys = ["total_ms", "policy", "hotness", "check", "fetch", "g2l", "l2g", "update_map", "pack_total", "compose", "log2phy", "pack_other", "other"]
for name, rows in out.items():
    print(name, "plans(per-process lines):", len(rows), "pids:", sorted({r["pid"] for r in rows}))
    print("  %-11s %9s %9s %9s %9s" % ("stage", "mean", "median", "max", "share%"))
    tot = st.mean(r["total_ms"] for r in rows)
    for k in keys:
        v = [r[k] for r in rows]
        print("  %-11s %9.1f %9.1f %9.1f %8.1f%%" % (k, st.mean(v), st.median(v), max(v), st.mean(v) / tot * 100))
