"""What a compiled graph will ask of HBM besides its tensors: DMA queues and spill, read from the NEFF.

    python tools/neff_info.py <cache dir, NEFF, or s3:// entry prefix> [...]
    # a whole capture: every graph it loads, summed (and the farm's compile times, with --queue)
    python tools/neff_info.py --keys-file capture/keys.json --cache s3://.../lnl/ [--queue s3://.../q/<job>/]

A NEFF is a 1024-byte header plus a tar.gz. Its sg00/<engine>.json files list every DMA the
engine queues ("dma": entries with a "queue" and "desc" copies of from/to sizes and steps); the
Neuron runtime turns each queue's entries into a DMA descriptor ring in HBM when it loads the
graph, and reports the spill queues' rings as "dma rings spill" in its memory table. For each
queue this prints the entries and the contiguous runs they move (product of every size but the
innermost, summed over copies, on the side with more runs), which is what a descriptor ring grows
with. It also prints the compiler's own DRAM spill-space line and peak-usage line from
log-neuron-cc.txt when the entry has one ("[DRAM_Allocator]: spill space = N bytes",
"[OOMChecker]: Peak internal HBM memory usage of module"). Runs-to-bytes is NOT a documented
formula: docs/neuron-notes.md "Compile farm" compares the counts with what the runtime allocated.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import defaultdict


def _runs(c: dict) -> int:
    def side(sizes):
        n = 1
        for s in (sizes or [1])[1:]:
            n *= int(s)
        return n

    return max(side(c.get("from_sizes")), side(c.get("to_sizes")))


def neff_queues(neff_path: str) -> dict:
    with open(neff_path, "rb") as f:
        f.seek(1024)
        data = f.read()
    q: dict = defaultdict(lambda: [0, 0])
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        for m in t.getmembers():
            if not re.search(r"sg\d+/[A-Za-z]+\d+\.json$", m.name):
                continue
            d = json.load(t.extractfile(m))
            for e in d.get("dma") or []:
                # newer NEFFs name a queue instance (qSPSpillReload0_defId_1) instead of the queue
                k = q[e.get("queue") or e.get("instance_name") or "?"]
                k[0] += 1
                k[1] += sum(_runs(c) for c in e.get("desc") or [])
    return dict(q)


def log_lines(log_path: str) -> list[str]:
    if not os.path.exists(log_path):
        return []
    out = []
    with open(log_path, errors="replace") as f:
        for line in f:
            if "spill space =" in line or "Peak internal HBM memory usage of module" in line:
                out.append(line.strip().split("]: ", 1)[-1])
    return out


def resolve(arg: str) -> tuple[str, str]:
    """(neff, log) local paths for a cache dir, a NEFF, or an s3:// entry prefix."""
    if arg.startswith("s3://"):
        key = arg.rstrip("/").rsplit("/", 1)[-1]
        d = tempfile.mkdtemp(prefix=f"neffinfo-{key[:8]}-")
        for fn in (f"graph_{key}.neff", "log-neuron-cc.txt"):
            subprocess.run(["aws", "s3", "cp", "--quiet", f"{arg.rstrip('/')}/{fn}", os.path.join(d, fn)], check=False)
        arg = d
        return os.path.join(d, f"graph_{key}.neff"), os.path.join(d, "log-neuron-cc.txt")
    if os.path.isdir(arg):
        key = os.path.basename(arg.rstrip("/"))
        return os.path.join(arg, f"graph_{key}.neff"), os.path.join(arg, "log-neuron-cc.txt")
    return arg, os.path.join(os.path.dirname(arg), "log-neuron-cc.txt")


RING_BYTES_PER_RUN = 32  # calibration, docs/neuron-notes.md "Compile farm" (one queue, within 1%)


def one(arg: str) -> dict:
    neff, log = resolve(arg)
    q = neff_queues(neff)
    spill = {k: v for k, v in q.items() if "Spill" in k}
    lines = log_lines(log)
    space = [int(m.group(1)) for x in lines for m in [re.search(r"^\s*spill space = (\d+)", x)] if m]
    return {"neff": os.path.basename(neff), "bytes": os.path.getsize(neff),
            "queues": {k: {"entries": v[0], "runs": v[1]} for k, v in sorted(q.items())},
            "spill_entries": sum(v[0] for v in spill.values()),
            "spill_runs": sum(v[1] for v in spill.values()),
            "spill_space_bytes": space[0] if space else None, "compiler": lines}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--queue", default=None)
    a = ap.parse_args()
    if not a.keys_file:
        for arg in a.paths:
            print(json.dumps(one(arg)), flush=True)
        return
    with open(a.keys_file) as f:
        keys = list(json.load(f))
    rows, pending = [], []
    for k in keys:
        try:
            d = one(f"{a.cache.rstrip('/')}/{k}")
        except (OSError, tarfile.TarError):  # not compiled (yet)
            pending.append(k)
            continue
        if a.queue:
            p = subprocess.run(["aws", "s3", "cp", "--quiet", f"{a.queue.rstrip('/')}/done/{k}.json", "-"],
                               capture_output=True, text=True)
            if p.returncode == 0 and p.stdout:
                r = json.loads(p.stdout)
                d["compile_seconds"], d["peak_gb"] = round(r["seconds"], 1), round(r["peak_tree_rss_gb"], 1)
        d["key"] = k
        rows.append(d)
        print(json.dumps({x: d.get(x) for x in ("key", "bytes", "spill_runs", "spill_space_bytes", "compile_seconds",
                                                 "peak_gb")}), flush=True)
    rings = sum(r["spill_runs"] for r in rows) * RING_BYTES_PER_RUN
    print(json.dumps({"summary": True, "graphs": len(rows), "pending": pending, "neff_bytes": sum(r["bytes"] for r in rows),
                      "est_spill_rings_gb": round(rings / 1e9, 3),
                      "max_spill_space_gb": round(max((r["spill_space_bytes"] or 0) for r in rows) / 1e9, 3),
                      "sum_spill_space_gb": round(sum((r["spill_space_bytes"] or 0) for r in rows) / 1e9, 3),
                      "max_compile_seconds": max((r.get("compile_seconds") or 0) for r in rows),
                      "sum_compile_seconds": round(sum((r.get("compile_seconds") or 0) for r in rows), 1)}), flush=True)


if __name__ == "__main__":
    main()
