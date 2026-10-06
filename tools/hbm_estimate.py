"""Device memory per rank of a configuration, before a device loads it: the Neuron runtime's own
accounting (kiln/compile_farm.py "device memory a set of NEFFs takes") applied to the compiled graphs
of a capture, plus the rank's tensors.

    python tools/hbm_estimate.py --keys-file <capture>/keys.json --cache s3://.../compile-cache/trn1-sdk2.32/lnl/ \\
        --tensors-gb <tools/tensor_bytes.py total_gb>

Per graph: the instruction streams (Model Code), the compiler's scratchpad reservation, and the spill
rings (32 B per run of the Spill DMA queues, times the queue count for queue-instance entries:
spill_ring_bytes). Then the total: tensors + code +
the largest scratchpad (page-rounded, shared by every loaded NEFF) + rings + 0.13 GiB of fixed runtime
items. Calibrated on trn1 (16 GiB per NeuronCore): configurations at 12.6-14.1 GiB loaded, and the one
the runtime refused came to 15.91 GiB (docs/neuron-notes.md "Compile farm"). trn2 has 24 GiB per logical
core.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kiln import compile_farm  # noqa: E402

GiB = 2**30
_STREAM = re.compile(r"sg\d+/[A-Za-z]+\d+\.bin$")


def code_bytes(neff_path: str) -> int:
    """Bytes of a NEFF's instruction streams, what the runtime loads as Model Code."""
    with open(neff_path, "rb") as f:
        f.seek(1024)  # the NEFF header; a tar.gz follows
        data = f.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        return sum(m.size for m in t.getmembers() if _STREAM.search(m.name))


_INSTANCE = re.compile(r"_defId_\d+$")


def spill_ring_bytes(neff_path: str) -> tuple[int, int, int]:
    """(ring bytes, runs in queue-named spill entries, runs in queue-instance spill entries).

    A DMA entry names its queue either as "queue" (qSPSpillReload0) or as "instance_name"
    (qSPSpillReload0_defId_1, an instance listed under the queue's "queue_instances" in
    sg<N>/def.json). The runtime's DMA Rings Spill is 32 B per run for the first kind (tools/
    neff_info.py; within 1-4% on the decode NEFFs of the kiln-mimo-trn1 table of 2026-10-04 11:3x,
    e.g. 3.233 MiB of runs for 3.111 printed) and that times the queue's num_queues (16) for the
    second: a GLM-5.3-Flash 8K prefill group at 2048 rows per group has only instance entries,
    6.15M runs = 187 MiB at 32 B, and brought 2.67 GiB of spill rings to its failed load (x16 =
    2.92 GiB)."""
    with open(neff_path, "rb") as f:
        f.seek(1024)
        data = f.read()
    named = inst = 0
    ring = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
        members = {m.name: m for m in t.getmembers()}
        nq: dict[str, dict] = {}
        for name, m in members.items():
            if re.search(r"sg\d+/def\.json$", name):
                nq[name.split("/")[0]] = {k: int(v.get("num_queues", 1)) for k, v in
                                          (json.load(t.extractfile(m)).get("dma_queue") or {}).items()}
        for name, m in members.items():
            if not re.search(r"sg\d+/[A-Za-z]+\d+\.json$", name):
                continue
            sg = name.split("/")[0]
            for e in json.load(t.extractfile(m)).get("dma") or []:
                q = e.get("queue") or e.get("instance_name") or ""
                if "Spill" not in q:
                    continue
                runs = 0
                for c in e.get("desc") or []:
                    runs += max(_inner_runs(c.get("from_sizes")), _inner_runs(c.get("to_sizes")))
                if "queue" in e:
                    named += runs
                    ring += 32 * runs
                else:
                    inst += runs
                    ring += 32 * runs * nq.get(sg, {}).get(_INSTANCE.sub("", q), 16)
    return ring, named, inst


def _inner_runs(sizes) -> int:
    n = 1
    for s in (sizes or [1])[1:]:
        n *= int(s)
    return n


def graph(entry: str) -> dict:
    """One cache entry (a local directory or an s3:// prefix): code, scratchpad, spill-ring bytes."""
    import neff_info

    neff, log = neff_info.resolve(entry)
    text = open(log, errors="replace").read() if os.path.exists(log) else ""
    ring, named, inst = spill_ring_bytes(neff)
    return {"key": os.path.basename(entry.rstrip("/")), "code": code_bytes(neff),
            "scratchpad_gib": compile_farm.scratchpad_gib(text), "rings": ring, "spill_runs_named": named,
            "spill_runs_instance": inst}


def estimate(graphs: list[dict], tensors_bytes: float) -> dict:
    code = sum(g["code"] for g in graphs)
    scratch = compile_farm.shared_scratchpad_bytes(g["scratchpad_gib"] for g in graphs)
    rings = sum(g["rings"] for g in graphs)
    total = tensors_bytes + code + scratch + rings + compile_farm.RUNTIME_FIXED_GIB * GiB
    return {"graphs": len(graphs), "tensors_gib": round(tensors_bytes / GiB, 3), "code_gib": round(code / GiB, 3),
            "scratchpad_gib": round(scratch / GiB, 3), "spill_rings_gib": round(rings / GiB, 3),
            "fixed_gib": compile_farm.RUNTIME_FIXED_GIB, "total_gib": round(total / GiB, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-file", required=True, help="a capture's keys.json (or a queue's configs/<name>.keys.json)")
    ap.add_argument("--cache", required=True, help="the compile cache: s3://.../lnl/ or a local compile_cache dir")
    ap.add_argument("--tensors-gb", type=float, required=True, help="tools/tensor_bytes.py total_gb (1e9 bytes)")
    a = ap.parse_args()
    with open(a.keys_file) as f:
        kmap = json.load(f)
    keys = list(kmap)
    graphs = []
    for k in keys:
        g = graph(f"{a.cache.rstrip('/')}/{k}")
        graphs.append(g)
        print(json.dumps({**g, "code_mib": round(g["code"] / 2**20, 2), "rings_mib": round(g["rings"] / 2**20, 2)}),
              flush=True)
    # A capture's keys.json maps each key to the captured ranks that run it. Under DP attention every group's ranks
    # have graphs of their own (their expert placement differs), so the sum over the file is several ranks' graphs:
    # G64-EPLB-P8K came to 20.9 GiB summed and 15.8 per rank, and it loads. Estimate each captured rank on its own
    # graphs and report the worst.
    ranks = sorted({r for v in kmap.values() for r in v}) if isinstance(kmap, dict) and all(
        isinstance(v, list) for v in kmap.values()) else []
    if not ranks:
        print(json.dumps({"summary": True, **estimate(graphs, a.tensors_gb * 1e9)}), flush=True)
        return
    per = {r: estimate([g for g in graphs if r in kmap[g["key"]]], a.tensors_gb * 1e9) for r in ranks}
    worst = max(ranks, key=lambda r: per[r]["total_gib"])
    print(json.dumps({"summary": True, "rank": worst, **per[worst],
                      "per_rank_total_gib": {str(r): per[r]["total_gib"] for r in ranks}}), flush=True)


if __name__ == "__main__":
    main()
