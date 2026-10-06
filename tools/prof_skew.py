"""Rank skew inside one replayed graph: is a collective's cost its own latency or the wait for the slowest rank?

    python tools/prof_skew.py <prof dir> <tag> [--workers 8] [--layers KKKDKKKDKKKD]

<prof dir> holds `<tag>_rank_<r>_exec_2.ntff` for every rank (tools/util_report.py replay --profile-all --keep-ntff) and
`<tag>.summary.json` (its `_neff`). Each rank's profile is viewed as JSON (neuron-explorer view, in parallel) and cut at
its collectives as tools/prof_step.py does: segment i is the compute from the end of collective i - 1 to the trigger of
collective i on that rank. Per segment the table gives the compute time on the median rank and on the slowest, which rank
was slowest, and rank 0's collective wait (trigger to start) and transfer. If the slowest rank's compute exceeds the
median's by about the collective's wait plus transfer, the collective's cost is imbalance; if every rank computes the
same and the collective is still slow, it is the collective itself.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor


def segments(path: str) -> dict:
    d = json.load(open(path))
    t_end = d["metadata"][0]["last_hw_timestamp"]
    cc = sorted(d["cc_ops"], key=lambda c: c["timestamp"])
    comp, start = [], 0
    for c in cc:
        # a collective the tensor engine triggers back to back with the previous one carries no cc_trigger (and no
        # trigger delay): its trigger is its start
        trig = c.get("cc_trigger", c["timestamp"] - c.get("cc_trigger_start_delay", 0))
        comp.append(max(trig - start, 0))
        start = c["timestamp"] + c["duration"]
    comp.append(t_end - start)
    return {"total": t_end, "compute": comp, "wait": [c.get("cc_trigger_start_delay", 0) for c in cc],
            "transfer": [c["duration"] for c in cc], "ops": [f"{c['operation']} {c['input_size'] // 1024}KB" for c in cc]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prof")
    ap.add_argument("tag")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--layers", default="")
    ap.add_argument("--keep-json", action="store_true")
    ap.add_argument("--cap", default="", help="the capture dir (tools/dc_cap3.sh cap-<cfg>): each rank's own NEFF")
    ap.add_argument("--call", default="decode:40")
    ap.add_argument("--group-keys", default="", help="lo=key,... : the NEFF (compile-cache key) of the group whose "
                    "first rank is lo, when the capture dir is gone (the config's keys.json names them)")
    ap.add_argument("--group-size", type=int, default=8)
    a = ap.parse_args()
    neff = json.load(open(os.path.join(a.prof, f"{a.tag}.summary.json")))["_neff"]
    # DP-attention groups run their own NEFF of a piece (their collectives' replica groups differ): viewing rank 8's
    # profile with group 0's NEFF gives an empty JSON (profile_info only), so each rank is viewed with its own
    neffs = {}
    if a.group_keys:
        cache = "/root/.cache/neuron_libtorch/neuron/compile_cache"
        for kv in a.group_keys.split(","):
            lo, k = kv.split("=")
            for r in range(int(lo), int(lo) + a.group_size):
                neffs[r] = os.path.join(cache, k, f"graph_{k}.neff")
    if a.cap:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from util_report import manifests

        seq = int(a.tag.split("-")[0])
        for r, es in manifests(a.cap).items():
            e = next((x for x in es if x["call"] == a.call and x["seq"] == seq), None)
            if e is not None:
                neffs[r] = glob.glob(os.path.join(e["artifact_dir"], "*.neff"))[0]
    ntffs = {}
    for f in glob.glob(os.path.join(a.prof, f"{a.tag}_rank_*_exec_*.ntff")):
        m = re.search(r"_rank_(\d+)_exec_", f)
        ntffs[int(m.group(1))] = f
    env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
    env["PATH"] = "/opt/aws/neuron/bin:" + env.get("PATH", "")

    def one(r: int):
        js = os.path.join(a.prof, f"{a.tag}.rank{r}.json")
        if not os.path.exists(js):
            subprocess.run(["neuron-explorer", "view", "-n", neffs.get(r, neff), "-s", ntffs[r], "--output-format", "json",
                            "--output-file", js, "--ignore-dma-trace", "--ignore-nc-buf-usage"], env=env,
                           capture_output=True, check=True)
        s = segments(js)  # KeyError 'metadata': the JSON is empty, the profile viewed with another group's NEFF
        if not a.keep_json:
            os.remove(js)
        return r, s

    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(ex.map(one, sorted(ntffs)))
    ranks = sorted(res)
    n = len(res[ranks[0]]["compute"])
    labels = [x for c in a.layers for x in (f"attn-{c}", "ffn")] + ["tail"] if a.layers else [""] * n
    print(f"{a.tag}: {len(ranks)} ranks; graph total on rank 0 {res[0]['total'] / 1e3:.1f} us, slowest rank "
          f"{max(res[r]['total'] for r in ranks) / 1e3:.1f} us")
    print(f"{'seg':>3} {'label':8s} {'median us':>9} {'max us':>8} {'slowest':>7} | rank 0: {'wait us':>8} {'xfer us':>8}  op")
    tot = collections.Counter()
    for i in range(n):
        v = sorted(res[r]["compute"][i] for r in ranks)
        med, mx = v[len(v) // 2], v[-1]
        slow = max(ranks, key=lambda r: res[r]["compute"][i])
        w = res[0]["wait"][i] if i < len(res[0]["wait"]) else 0
        x = res[0]["transfer"][i] if i < len(res[0]["transfer"]) else 0
        op = res[0]["ops"][i] if i < len(res[0]["ops"]) else "end"
        tot["median compute"] += med
        tot["max compute"] += mx
        tot["rank0 compute"] += res[0]["compute"][i]
        tot["rank0 wait"] += w
        tot["rank0 transfer"] += x
        lab = labels[i] if i < len(labels) else ""
        print(f"{i:3d} {lab:8s} {med / 1e3:9.1f} {mx / 1e3:8.1f} {slow:7d} | {w / 1e3:8.1f} {x / 1e3:8.1f}  {op}")
    print("totals (ms): " + ", ".join(f"{k} {v / 1e6:.2f}" for k, v in tot.items()))


if __name__ == "__main__":
    main()
