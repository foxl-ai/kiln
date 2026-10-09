"""Accelerator utilization of the real serving graphs: per engine, HBM, tensor engine, collectives, host gaps,
and model-level MFU (prefill) / MBU (decode). trn1 (NeuronCore-v2), SDK 2.32, neuron-explorer 2.32.

    # 1. a serving run with the host timeline and rank 0's runtime trace (kiln/profiling.py, engine/tp.py)
    KILN_TIMELINE=tl.jsonl KILN_RT_INSPECT=insp python bench/serve_sweep.py ...
    python tools/util_report.py timeline tl.jsonl [--systrace insp/r0]
    # 2. the inputs of one prefill and one decode call on every rank, then each NEFF replayed on all 32
    #    cores with every rank's real inputs (rank 0 profiled), then the table
    KILN_CAPTURE_INPUTS=cap KILN_CAPTURE_AT=prefill:20,decode:60 python bench/serve_sweep.py ... --requests 64
    python tools/util_report.py replay cap --call prefill:20 --out prof/prefill [--bins]
    python tools/util_report.py report prof/prefill --kind prefill --rows 4096 --shape-dir <dir> ...
    # model FLOPs per token and the decode step's HBM floor, from config.json + safetensors headers
    python tools/util_report.py model --shape-dir <dir> [--rows-per-group 16 --groups 4 --kv fp8]

Replay: `neuron-explorer capture -n <neff> -s <ntff> -r <n> -i 0 --multi-input <file> --num-exec 2 --profile-nth-exec 2`
runs a NEFF on n workers, worker r with rank r's captured inputs (one line per worker, "input0 <npy> input1 <npy> ...", the
format of the ifmap argument; neuron-explorer capture --help), and profiles the second execution of worker 0 (the first
starts cold: a decode prep graph replayed in 6.55 ms once, 3.06 ms as the second execution, 0.36 ms in the serving run).
Ranks that ran a different NEFF for the same execution (each DP-attention group compiles its own prefill pieces; a group's
NEFF only loads on its group's ranks) run as one neuron-explorer process per NEFF over their own cores, joined into one
collectives world with --collectives-worker-start-id / --collectives-worker-count and one NEURON_RT_ROOT_COMM_ID; each
process profiles its first worker (--profile-all: every worker). `view --output-format summary-json` gives neuron-explorer's
own per-execution counters (engine active times, HBM bytes, hardware FLOPs, collective op time); `bins` cuts the instruction
trace (`view --output-format json --ignore-dma-trace`) into 0.1 ms bins and into segments between collectives. Without
captured inputs every input is zero and MoE routing sends every row to experts 0-7 (docs/neuron-notes.md), which under
expert parallelism puts the whole MoE on rank 0.

Peaks (per NeuronCore-v2, trn1; two cores per Trainium device):
- Tensor engine: AWS, "Trainium architecture" (awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/
  arch/neuron-hardware/trainium.html): "Two NeuronCore-v2 delivering 380 INT8 TOPS, 190 FP16/BF16/cFP8/TF32
  TFLOPS, and 47.5 FP32 TFLOP" -> 95 TFLOPS bf16 / cFP8 per core (cFP8 is not faster than bf16 on trn1),
  23.75 FP32. neuron-explorer 2.32 divides by 91.75 TFLOPS per core (its mfu / hfu fields: 128 x 128 PE x 2
  x 2.8 GHz), which is used for the engine-level percentages it reports.
- HBM: the same page, "32 GiB of device memory ... with 820 GiB/sec of bandwidth" -> 410 GiB/s = 440.2 GB/s
  per core. neuron-explorer 2.32's mbu fields divide by 410e9 B/s (its peak_flops_bandwidth_ratio 223.78 =
  91.75e12 / 410e9). The EC2 trn1 page gives no per-instance bandwidth (read 2026-10-05).
"""

from __future__ import annotations

import argparse
import bisect
import collections
import glob
import json
import math
import os
import re
import subprocess
import sys

TE_PEAK = 95e12  # bf16 / cFP8 FLOP/s per NeuronCore-v2 (AWS Trainium architecture page, 190 per 2 cores)
TE_PEAK_NX = 91.75e12  # what neuron-explorer 2.32 divides by
HBM_PEAK = 410 * 2**30  # B/s per core (820 GiB/s per 2-core Trainium device)
HBM_PEAK_NX = 410e9
ENGINES = ("tensor", "vector", "scalar", "gpsimd")
SUMMARY_KEYS = ("total_time", "tensor_engine_active_time", "vector_engine_active_time", "scalar_engine_active_time",
                "gpsimd_engine_active_time", "dma_active_time", "cc_op_time", "cc_op_count", "hbm_read_bytes",
                "hbm_write_bytes", "hardware_flops", "transpose_flops", "model_flops", "spill_save_bytes",
                "spill_reload_bytes", "software_dynamic_dma_size", "static_dma_size", "dma_transfer_total_bytes")


# ---------------------------------------------------------------- model FLOPs and bytes (config + headers)

def load_shapes(shape_dir: str) -> tuple[dict, dict]:
    """(text config, {tensor name: (dtype, shape)}) from a compile-farm shape dir (config.json +
    kiln-safetensors-headers.json, tools/compile_farm.py shapes) or a checkpoint dir (its safetensors)."""
    cfg = json.load(open(os.path.join(shape_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    hp = os.path.join(shape_dir, "kiln-safetensors-headers.json")
    if os.path.exists(hp):
        heads = {k: (v[0], v[1]) for k, v in json.load(open(hp)).items()}
    else:
        heads = {}
        for f in glob.glob(os.path.join(shape_dir, "*.safetensors")):
            with open(f, "rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                h = json.loads(fh.read(n))
            heads.update({k: (v["dtype"], v["shape"]) for k, v in h.items() if k != "__metadata__"})
    return tc, heads


DT_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "F8_E4M3": 1, "I64": 8, "I32": 4, "U8": 1}


def layer_of(name: str) -> int | None:
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def model_counts(tc: dict, heads: dict) -> dict:
    """Matmul parameters a token touches, per family (GLM-5.3-Flash / glm5_next names; the MTP layer
    num_hidden_layers is left out), and the weight bytes per family."""
    L = tc["num_hidden_layers"]
    E, k = tc["n_routed_experts"], tc["num_experts_per_tok"]
    fam = collections.defaultdict(lambda: [0, 0])  # family -> [params a token touches, bytes stored]
    for name, (dt, shp) in heads.items():
        if "visual" in name or not name.endswith(("weight", "_fn", "weight_scale_inv")):
            continue
        li = layer_of(name)
        if li is not None and li >= L:
            continue  # the MTP layer
        n = math.prod(shp)
        b = n * DT_BYTES.get(dt, 2)
        if name.endswith("weight_scale_inv"):
            f = "experts" if ".experts." in name else "dense"
            fam[f + " scales"][1] += b
            continue
        if len(shp) < 2:
            fam["norms"][1] += b
            continue
        if ".mlp.experts." in name:
            fam["routed experts"][0] += n * k / E  # top-k of E experts per token
            fam["routed experts"][1] += b
        elif "shared_experts" in name:
            fam["shared expert"][0] += n
            fam["shared expert"][1] += b
        elif ".mlp.gate." in name:
            fam["router"][0] += n
            fam["router"][1] += b
        elif ".mlp." in name:
            fam["dense MLP"][0] += n
            fam["dense MLP"][1] += b
        elif ".indexer." in name:
            fam["DSA indexer"][0] += n
            fam["DSA indexer"][1] += b
        elif ".self_attn." in name and li is not None:
            kind = tc["layer_types"][li]
            f = "KDA projections" if kind == "linear_attention" else "DSA projections"
            fam[f][0] += n if "conv1d" not in name else 0
            fam[f][1] += b
        elif "hc_" in name:
            fam["mHC"][0] += n
            fam["mHC"][1] += b
        elif name.startswith("lm_head"):
            fam["lm_head"][1] += b
        elif "embed_tokens" in name:
            fam["embedding"][1] += b
        else:
            fam["other"][1] += b
    return {f: (p, b) for f, (p, b) in fam.items()}


def attention_flops(tc: dict, ctx: int) -> dict:
    """Attention-core FLOPs per token, averaged over the positions 0..ctx-1 of a prompt (prefill), as the
    model defines them: DSA attends to its top index_topk tokens (pooled: index_topk / index_kpool pools of
    index_kpool) plus everything before them when the context is shorter; the indexer scores every pool; KDA's
    recurrent update is ~7 dk dv FLOPs per head per token (S^T q, S^T k, the rank-1 update, the decay).
    'dsa as run' is the dense masked form over a context bucket of ctx keys (what the XLA attention core
    computes in prefill)."""
    H, dq, dv = tc["num_attention_heads"], tc["qk_head_dim"], tc["v_head_dim"]
    topk = tc.get("index_topk", 2048)
    n_dsa = sum(t != "linear_attention" for t in tc["layer_types"])
    n_kda = len(tc["layer_types"]) - n_dsa
    mean_keys = sum(min(p + 1, topk) for p in range(ctx)) / ctx
    pools = sum((p + 1) / tc.get("index_kpool", 1) for p in range(ctx)) / ctx
    la = tc.get("linear_attn_config", {})
    kh, kd = la.get("num_heads", H), la.get("head_dim", 128)
    return {
        "dsa core": n_dsa * 2 * H * mean_keys * (dq + dv),
        "dsa indexer scores": n_dsa * 2 * tc.get("index_n_heads", 0) * tc.get("index_head_dim", 0) * pools,
        "kda recurrence": n_kda * 7 * kh * kd * kd,
        "dsa as run (dense over the bucket)": n_dsa * 2 * H * ctx * (dq + dv),
        "mean dsa keys": mean_keys,
    }


def prefill_flops_per_token(tc: dict, heads: dict, ctx: int) -> tuple[float, dict]:
    c = model_counts(tc, heads)
    parts = {f: 2 * p for f, (p, _) in c.items() if p}
    a = attention_flops(tc, ctx)
    parts.update({k: v for k, v in a.items() if k in ("dsa core", "dsa indexer scores", "kda recurrence")})
    return sum(parts.values()), parts


def decode_bytes_per_rank(tc: dict, heads: dict, tp: int, attn_tp: int, rows_step: int, rows_group: int,
                          ctx: int, kv: str) -> dict:
    """HBM bytes one rank must read (and write) in one decode step: dense weights by their split at tp ranks with
    attention TP attn_tp (attention projections split attn_tp ways, the DSA indexer, the router and mHC
    replicated, shared expert / dense MLP / lm_head split tp ways), the routed experts the step's rows touch
    (1 - (1 - k/E)^rows of them: every group's rows go through every MoE layer), each of the rank's group's
    rows' DSA cache over ctx keys (the fp8 or bf16 latent, the bf16 indexer key, the bf16 pool key per
    index_kpool tokens; read whole, as the decode mask form does) and its KDA state rows read and written."""
    c = model_counts(tc, heads)
    split = {"KDA projections": attn_tp, "DSA projections": attn_tp, "DSA indexer": 1, "router": 1, "mHC": 1,
             "shared expert": tp, "dense MLP": tp, "lm_head": tp, "dense scales": tp, "norms": 1}
    dense = {f: c[f][1] / s for f, s in split.items() if f in c}
    E, k = tc["n_routed_experts"], tc["num_experts_per_tok"]
    cover = 1 - (1 - k / E) ** rows_step
    experts = (c["routed experts"][1] + c.get("experts scales", (0, 0))[1]) / tp
    n_dsa = sum(t != "linear_attention" for t in tc["layer_types"])
    n_kda = len(tc["layer_types"]) - n_dsa
    lat = tc["kv_lora_rank"] * (1 if kv == "fp8" else 2)
    per_tok = lat + 2 * tc.get("index_head_dim", 0) + 2 * tc.get("index_head_dim", 0) / tc.get("index_kpool", 1)
    kv_row = n_dsa * ctx * per_tok
    la = tc.get("linear_attn_config", {})
    st_row = 2 * n_kda * (la.get("num_heads", 64) // attn_tp) * la.get("head_dim", 128) ** 2 * 4
    out = {"dense weights": sum(dense.values()), "routed experts touched": experts * cover,
           "expert coverage": cover, "kv rows": rows_group * kv_row, "kda state r+w": rows_group * st_row}
    out["total"] = out["dense weights"] + out["routed experts touched"] + out["kv rows"] + out["kda state r+w"]
    out["dense by family"] = dense
    return out


def cmd_model(a) -> None:
    tc, heads = load_shapes(a.shape_dir)
    tot, parts = prefill_flops_per_token(tc, heads, a.ctx)
    print(f"prefill model FLOPs per token (prompt of {a.ctx}): {tot / 1e9:.2f} GFLOP")
    for f, v in sorted(parts.items(), key=lambda x: -x[1]):
        print(f"  {f:28s} {v / 1e9:8.3f} GFLOP  {v / tot:6.1%}")
    at = attention_flops(tc, a.ctx)
    print(f"  (mean DSA keys per query {at['mean dsa keys']:.0f}; the dense masked core over {a.ctx} keys would be "
          f"{at['dsa as run (dense over the bucket)'] / 1e9:.2f} GFLOP per token instead of {at['dsa core'] / 1e9:.2f})")
    for rg in a.rows_per_group:
        d = decode_bytes_per_rank(tc, heads, a.tp, a.attn_tp, rg * a.groups, rg, a.decode_ctx, a.kv)
        fl = d["total"] / HBM_PEAK
        print(f"decode step, {rg} rows per group x {a.groups} groups, ctx {a.decode_ctx}, {a.kv} KV: per rank "
              f"{d['total'] / 1e9:.2f} GB = dense {d['dense weights'] / 1e9:.2f} + experts "
              f"{d['routed experts touched'] / 1e9:.2f} ({d['expert coverage']:.0%} touched) + KV "
              f"{d['kv rows'] / 1e9:.2f} + KDA state {d['kda state r+w'] / 1e9:.2f}; floor {fl * 1e3:.1f} ms at "
              f"{HBM_PEAK / 1e9:.0f} GB/s")
    d = decode_bytes_per_rank(tc, heads, a.tp, a.attn_tp, 64, 16, a.decode_ctx, a.kv)
    print("  dense weights per rank by family: " + ", ".join(f"{f} {v / 1e9:.3f} GB" for f, v in
                                                            sorted(d["dense by family"].items(), key=lambda x: -x[1])))


# ---------------------------------------------------------------- replay captured inputs

def manifests(cap: str) -> dict[int, list[dict]]:
    out = {}
    for d in sorted(glob.glob(os.path.join(cap, "r*"))):
        m = os.path.join(d, "manifest.jsonl")
        if os.path.exists(m):
            out[int(os.path.basename(d)[1:])] = [json.loads(x) for x in open(m)]
    return out


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
    env["PATH"] = "/opt/aws/neuron/bin:" + env.get("PATH", "")
    return subprocess.run(cmd, env=env, capture_output=True, text=True, **kw)


def _summary(neff: str, prof: str) -> dict:
    v = run(["neuron-explorer", "view", "-n", neff, "-s", prof, "--output-format", "summary-json"])
    if v.returncode:
        raise SystemExit(f"view failed for {prof}: {v.stderr[-800:]}")
    return next(iter(json.loads(v.stdout).values()))


def cmd_replay(a) -> None:
    """Each execution of the call on rank 0, replayed on all world workers with every rank's captured inputs. Ranks
    that ran a different NEFF for it (DP-attention groups compile their own prefill pieces, which differ only in
    their group collectives' replica groups: a NEFF of group 0 cannot load on rank 8, "replica groups (0/2) does not
    have myself", measured 2026-10-05) run as separate neuron-explorer processes, one per NEFF, over their own cores,
    joined into one collectives world by --collectives-worker-start-id / --collectives-worker-count (its multi-node
    form) and one NEURON_RT_ROOT_COMM_ID. Each process profiles its first worker, so a grouped replay also profiles
    ranks 8, 16 and 24: the spread across them is the call's imbalance."""
    import socket

    ms = manifests(a.cap)
    if 0 not in ms:
        raise SystemExit(f"no r0/manifest.jsonl under {a.cap}")
    os.makedirs(a.out, exist_ok=True)
    mine = [e for e in ms[0] if e["call"] == a.call and (a.seq is None or e["seq"] in a.seq)]
    print(f"{a.call}: {len(mine)} executions on rank 0; ranks captured {len(ms)}", flush=True)
    for e in mine:
        tag = f"{e['seq']:03d}-{e['neff_id']}"
        per = []  # (rank, neff_id, inputs)
        for r in range(a.world):
            er = next((x for x in ms.get(r, []) if x["call"] == a.call and x["seq"] == e["seq"]), None)
            if er is None or len(er["inputs"]) != len(e["inputs"]):
                raise SystemExit(f"{tag}: rank {r} has no matching capture")
            per.append((r, er["neff_id"], er["artifact_dir"], er["inputs"]))
        blocks = []  # contiguous runs of ranks on one NEFF
        for r, k, d, ins in per:
            if blocks and blocks[-1]["neff_id"] == k:
                blocks[-1]["ranks"].append(r)
                blocks[-1]["inputs"].append(ins)
            else:
                blocks.append({"neff_id": k, "dir": d, "ranks": [r], "inputs": [ins]})
        summ = os.path.join(a.out, f"{tag}.summary.json")
        if not os.path.exists(summ) or a.force:
            with socket.socket() as so:
                so.bind(("127.0.0.1", 0))
                port = so.getsockname()[1]
            procs = []
            for b in blocks:
                lo, n = b["ranks"][0], len(b["ranks"])
                neff = glob.glob(os.path.join(b["dir"], "*.neff"))[0]
                mi = os.path.join(a.out, f"{tag}.r{lo}.inputs")
                open(mi, "w").write("\n".join(" ".join(f"input{i} {f}" for i, f in enumerate(x)) for x in b["inputs"]) + "\n")
                cmd = ["neuron-explorer", "capture", "-n", neff, "-s", os.path.join(a.out, f"{tag}.ntff"), "-r", str(n),
                       "-i", "all" if a.profile_all else "0", "--multi-input", mi, "--ignore-exec-errors", "--num-exec", str(a.exec),
                       "--profile-nth-exec", str(a.exec)]
                env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
                env["PATH"] = "/opt/aws/neuron/bin:" + env.get("PATH", "")
                if len(blocks) > 1:
                    cmd += ["--collectives-worker-start-id", str(lo), "--collectives-worker-count", str(a.world)]
                    env.update(NEURON_RT_VISIBLE_CORES=f"{lo}-{lo + n - 1}", NEURON_RT_ROOT_COMM_ID=f"localhost:{port}")
                log = open(os.path.join(a.out, f"{tag}.r{lo}.log"), "w")
                procs.append((b, neff, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)))
            bad = [b["ranks"][0] for b, _, p in procs if p.wait()]
            if bad:
                raise SystemExit(f"capture failed for {tag} (processes from ranks {bad}; logs {a.out}/{tag}.r*.log)")
            profs = {}
            for b, neff, _ in procs:
                lo = b["ranks"][0]
                got = sorted(glob.glob(os.path.join(a.out, f"{tag}_rank_{lo}*.ntff")), key=os.path.getmtime)
                if got:
                    profs[lo] = (neff, got[-1])
            s = _summary(*profs[0])
            s["_neff"], s["_ntff"], s["_blocks"] = profs[0][0], profs[0][1], [b["ranks"][0] for b in blocks]
            s["_others"] = {str(lo): {k: v for k, v in _summary(*nf).items() if k in SUMMARY_KEYS}
                            for lo, nf in profs.items() if lo != 0}
            json.dump(s, open(summ, "w"))
            if not a.keep_ntff:
                for f in glob.glob(os.path.join(a.out, f"{tag}*.ntff")):
                    os.remove(f)
        s = json.load(open(summ))
        oth = " ".join(f"r{lo} {o['total_time'] * 1e3:.2f}" for lo, o in s.get("_others", {}).items())
        print(f"  {tag}: {s['total_time'] * 1e3:8.2f} ms, tensor {s['tensor_engine_active_time'] / s['total_time']:.0%} "
              f"vector {s['vector_engine_active_time'] / s['total_time']:.0%}, HBM "
              f"{(s['hbm_read_bytes'] + s['hbm_write_bytes']) / s['total_time'] / 1e9:.0f} GB/s"
              + (f"; other groups' first ranks (ms): {oth}" if oth else ""), flush=True)


ENGINE_OF = (("pe", "tensor"), ("act", "scalar"), ("dve", "vector"), ("pool", "gpsimd"), ("sp", "sync"))


def engine_name(sub: str) -> str:
    s = sub.lower()
    for k, v in ENGINE_OF:
        if s.startswith(k) or k in s.split("_")[0]:
            return v
    return s


def bins(js: str, bin_us: float) -> dict:
    """Occupancy of each compute engine per bin from a profile's instruction trace, every collective op's
    interval, and the time (in bins) every compute engine was idle, split by whether a collective was in
    flight (cc_ops: its trigger to its end)."""
    d = json.load(open(js))
    B = bin_us * 1e3
    ins = d.get("instruction", [])
    t0 = min(i["timestamp"] for i in ins)
    t1 = max(i["timestamp"] + (i.get("duration") or 0) for i in ins)
    nb = int((t1 - t0) // B) + 1
    occ = {e: [0.0] * nb for e in ENGINES}
    names = collections.Counter()
    iv = collections.defaultdict(list)
    for i in ins:
        e = engine_name(i.get("subgroup", "?"))
        names[i.get("subgroup", "?")] += 1
        if e not in occ or str(i.get("opcode", "")).startswith("EVENT_SEMAPHORE"):
            continue
        iv[e].append((i["timestamp"] - t0, i["timestamp"] - t0 + (i.get("duration") or 0)))
    unions = {}
    for e, xs in iv.items():  # an engine's instructions overlap in the trace (queued): bin their union
        xs.sort()
        merged = []
        for a0, a1 in xs:
            if merged and a0 <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], a1)
            else:
                merged.append([a0, a1])
        unions[e] = merged
        for s, end in merged:
            while s < end:
                b = int(s // B)
                take = min(end, (b + 1) * B) - s
                occ[e][b] += take / B
                s += take
    cc = sorted((c.get("cc_trigger", c["timestamp"]) - t0, c["timestamp"] + c["duration"] - t0)
                for c in d.get("cc_ops", []))
    incc = [False] * nb
    for lo, hi in cc:
        for b in range(max(0, int(lo // B)), min(nb, int(hi // B) + 1)):
            incc[b] = True
    idle = [all(occ[e][b] < 0.05 for e in ENGINES) for b in range(nb)]
    # Segments between collectives (prof_step.py's cut: from the end of collective i - 1 to the trigger of
    # collective i), each with its wall and every engine's busy time (union of its instructions in it).
    ccs = sorted(d.get("cc_ops", []), key=lambda c: c["timestamp"])
    cuts, start = [], 0
    for c in ccs:
        cuts.append((start, c.get("cc_trigger", c["timestamp"]) - t0))
        start = c["timestamp"] + c["duration"] - t0
    cuts.append((start, t1 - t0))
    segs = []
    starts = {e: [m[0] for m in merged] for e, merged in unions.items()}
    for lo, hi in cuts:
        row = {"lo_us": lo / 1e3, "ms": max(0, hi - lo) / 1e6}
        for e, merged in unions.items():  # disjoint sorted intervals, clipped to [lo, hi)
            busy, j = 0, max(0, bisect.bisect_right(starts[e], lo) - 1)
            while j < len(merged) and merged[j][0] < hi:
                busy += max(0, min(merged[j][1], hi) - max(merged[j][0], lo))
                j += 1
            row[e] = busy / 1e6
        segs.append(row)
    cc_list = [{"op": c.get("operation", "?"), "bytes": c.get("input_size", 0),
                "wait_ms": c.get("cc_trigger_start_delay", 0) / 1e6, "ms": c["duration"] / 1e6} for c in ccs]
    return {"bin_us": bin_us, "bins": nb, "span_ms": (t1 - t0) / 1e6, "subgroups": dict(names.most_common(30)),
            "segments": segs, "collectives": cc_list,
            "busy_ms": {e: sum(occ[e]) * bin_us / 1e3 for e in ENGINES},
            "idle_cc_ms": sum(i and c for i, c in zip(idle, incc)) * bin_us / 1e3,
            "idle_other_ms": sum(i and not c for i, c in zip(idle, incc)) * bin_us / 1e3,
            "cc_span_ms": sum(incc) * bin_us / 1e3, "cc_ops": len(cc),
            "occ": {e: [round(x, 3) for x in v] for e, v in occ.items()}}


# ---------------------------------------------------------------- report

def cmd_bins(a) -> None:
    """The instruction-trace bins of every replayed graph under prof whose bins are missing (kept .ntff files,
    replay --keep-ntff): CPU only, so it can run while the NeuronCores do something else."""
    for summ in sorted(glob.glob(os.path.join(a.prof, "*.summary.json"))):
        tag = os.path.basename(summ)[: -len(".summary.json")]
        out = os.path.join(a.prof, f"{tag}.bins.json")
        sj = json.load(open(summ))
        got = [sj["_ntff"]] if os.path.exists(sj.get("_ntff", "")) else sorted(
            glob.glob(os.path.join(a.prof, f"{tag}_rank_0*.ntff")), key=os.path.getmtime)
        if os.path.exists(out) or not got:
            continue
        neff = sj["_neff"]
        js = os.path.join(a.prof, f"{tag}.json")
        v = run(["neuron-explorer", "view", "-n", neff, "-s", got[-1], "--output-format", "json", "--output-file", js,
                 "--ignore-dma-trace", "--ignore-nc-buf-usage"])
        if v.returncode:
            print(f"json view failed for {tag}: {v.stderr[-500:]}", flush=True)
            continue
        b = bins(js, a.bin_us)
        json.dump(b, open(out, "w"))
        os.remove(js)
        print(f"  {tag}: span {b['span_ms']:.2f} ms, every engine idle with a collective in flight {b['idle_cc_ms']:.2f} "
              f"ms, idle otherwise {b['idle_other_ms']:.2f} ms ({b['cc_ops']} collectives)", flush=True)


def segment_table(rows: list) -> None:
    """Segments of every graph (from bins), grouped by the collective that ends them (operation and input bytes;
    the graph's tail as 'end'): what runs between collectives, with each engine's busy share, and each kind of
    collective's trigger-to-start wait and transfer. In a sequence-parallel prefill layer the five collectives are
    the attention group's gather and reduce-scatter (8 MB inputs at 1024 rows per group), the FFN block's world
    gather (32 MB), the routing gather (256 KB) and the world reduce-scatter (32 MB), so the segment ending in the
    group reduce-scatter is the token mixer, the one ending in the world reduce-scatter the MLP / MoE, and so on."""
    seg = collections.defaultdict(lambda: collections.Counter())
    cc = collections.defaultdict(lambda: collections.Counter())
    for tag, s, b in rows:
        if not b or "segments" not in b:
            continue
        colls = b["collectives"]
        for i, sg in enumerate(b["segments"]):
            k = f"{colls[i]['op']} {colls[i]['bytes'] / 2**20:.2f} MiB" if i < len(colls) else "end of graph"
            c = seg[k]
            c["n"] += 1
            c["ms"] += sg["ms"]
            for e in ENGINES:
                c[e] += sg.get(e, 0.0)
        for c in colls:
            k = f"{c['op']} {c['bytes'] / 2**20:.2f} MiB"
            cc[k]["n"] += 1
            cc[k]["wait"] += c["wait_ms"]
            cc[k]["ms"] += c["ms"]
    if not seg:
        return
    print("  segments, by the collective that ends them (n, mean ms, each engine's busy share of the segment):")
    for k, c in sorted(seg.items(), key=lambda kv: -kv[1]["ms"]):
        print(f"    {k:42s} n={c['n']:4d} mean {c['ms'] / c['n']:7.3f} ms total {c['ms']:8.2f} ms | " +
              " ".join(f"{e} {c[e] / c['ms']:5.1%}" for e in ENGINES if c["ms"]))
    print("  collectives (n, mean trigger-to-start wait, mean transfer):")
    for k, c in sorted(cc.items(), key=lambda kv: -(kv[1]["ms"] + kv[1]["wait"])):
        print(f"    {k:42s} n={c['n']:4d} wait {c['wait'] / c['n']:6.3f} ms + {c['ms'] / c['n']:6.3f} ms "
              f"(total {c['wait'] + c['ms']:7.2f} ms)")


def cmd_report(a) -> None:
    rows = []
    for f in sorted(glob.glob(os.path.join(a.prof, "*.summary.json"))):
        s = json.load(open(f))
        tag = os.path.basename(f)[: -len(".summary.json")]
        bf = os.path.join(a.prof, f"{tag}.bins.json")
        b = json.load(open(bf)) if os.path.exists(bf) else None
        rows.append((tag, s, b))
    if not rows:
        raise SystemExit(f"no summaries under {a.prof}")
    tot = collections.Counter()
    print(f"{a.kind} call, rank 0, {len(rows)} graphs (replayed with captured inputs; neuron-explorer 2.32 counters)")
    hdr = (f"{'graph':40s} {'ms':>8s} {'tensor':>7s} {'vector':>7s} {'scalar':>7s} {'gpsimd':>7s} {'HBM GB/s':>9s} "
           f"{'%peak':>6s} {'TE TF/s':>8s} {'%peak':>6s} {'cc ms':>6s} {'idle+cc':>7s} {'idle':>6s}")
    print(hdr)
    for tag, s, b in rows:
        T = s["total_time"]
        hbm = s["hbm_read_bytes"] + s["hbm_write_bytes"]
        for k in SUMMARY_KEYS:
            if isinstance(s.get(k), (int, float)):
                tot[k] += s[k]
        if b:
            tot["idle_cc_ms"] += b["idle_cc_ms"]
            tot["idle_other_ms"] += b["idle_other_ms"]
        print(f"{tag[:40]:40s} {T * 1e3:8.2f} " + " ".join(
            f"{s[e + '_engine_active_time'] / T:7.1%}" for e in ENGINES) +
            f" {hbm / T / 1e9:9.0f} {hbm / T / HBM_PEAK:6.1%} {s.get('hardware_flops', 0) / T / 1e12:8.2f} "
            f"{s.get('hardware_flops', 0) / T / TE_PEAK:6.1%} {s.get('cc_op_time', 0) * 1e3:6.2f} "
            + (f"{b['idle_cc_ms']:7.2f} {b['idle_other_ms']:6.2f}" if b else f"{'-':>7s} {'-':>6s}"))
    T = tot["total_time"]
    hbm = tot["hbm_read_bytes"] + tot["hbm_write_bytes"]
    print(f"{'sum':40s} {T * 1e3:8.2f} " + " ".join(f"{tot[e + '_engine_active_time'] / T:7.1%}" for e in ENGINES) +
          f" {hbm / T / 1e9:9.0f} {hbm / T / HBM_PEAK:6.1%} {tot['hardware_flops'] / T / 1e12:8.2f} "
          f"{tot['hardware_flops'] / T / TE_PEAK:6.1%} {tot['cc_op_time'] * 1e3:6.2f} "
          + (f"{tot['idle_cc_ms']:7.2f} {tot['idle_other_ms']:6.2f}" if tot["idle_cc_ms"] or tot["idle_other_ms"] else ""))
    print(f"  HBM read {tot['hbm_read_bytes'] / 1e9:.2f} GB, written {tot['hbm_write_bytes'] / 1e9:.2f} GB (spill "
          f"save / reload {tot['spill_save_bytes'] / 1e9:.2f} / {tot['spill_reload_bytes'] / 1e9:.2f} GB); tensor engine "
          f"{tot['hardware_flops'] / 1e12:.2f} TFLOP done (+ {tot['transpose_flops'] / 1e12:.2f} TFLOP of transposes), "
          f"HLO matmul FLOPs {tot['model_flops'] / 1e12:.2f} TFLOP (NKI kernels' matmuls are not in this count)")
    segment_table(rows)
    if a.shape_dir:
        tc, heads = load_shapes(a.shape_dir)
        wall = a.call_s or T
        if a.kind == "prefill":
            per, _ = prefill_flops_per_token(tc, heads, a.ctx)
            mf = per * a.rows
            print(f"  model FLOPs: {a.rows} tokens x {per / 1e9:.2f} GFLOP = {mf / 1e12:.1f} TFLOP over {a.world} cores; "
                  f"at {wall * 1e3:.0f} ms per call MFU = {mf / (wall * a.world * TE_PEAK):.1%} of {a.world} x 95 "
                  f"TFLOPS (graphs alone, {T * 1e3:.0f} ms: {mf / (T * a.world * TE_PEAK):.1%})")
        else:
            d = decode_bytes_per_rank(tc, heads, a.world, a.attn_tp, a.rows, a.rows // a.groups, a.decode_ctx, a.kv)
            print(f"  decode bytes that must move per rank: {d['total'] / 1e9:.2f} GB; at {wall * 1e3:.1f} ms per call MBU = "
                  f"{d['total'] / wall / HBM_PEAK:.1%} of {HBM_PEAK / 1e9:.0f} GB/s (graphs alone, {T * 1e3:.1f} ms: "
                  f"{d['total'] / T / HBM_PEAK:.1%}); the graphs moved {hbm / 1e9:.2f} GB, "
                  f"{hbm / max(d['total'], 1):.1f}x the floor")


# ---------------------------------------------------------------- host timeline

def cmd_timeline(a) -> None:
    """Where rank 0's wall time goes in a serving level (kiln/profiling.py records), and how long the host made
    the device wait. Under overlap the device has step N+1 queued while the host reads step N back, so a
    read that blocks means the device was busy; a step() call whose reads did not block left the device
    without queued work for (the host time after its previous read returned) minus (the queued call's device
    time), which is bounded here by the host time itself."""
    recs = [json.loads(x) for x in open(a.timeline)]
    lv = [r for r in recs if r[0] == "level"]
    if lv:
        lo, hi = lv[a.level][1], lv[a.level][2]
        recs = [r for r in recs if lo <= r[1] <= hi]
        print(f"level conc {lv[a.level][3]}: {hi - lo:.1f} s wall, {lv[a.level][4]} requests")
    else:
        lo, hi = min(r[1] for r in recs), max(r[2] for r in recs)
    wall = hi - lo
    by = collections.defaultdict(list)
    for r in recs:
        by[r[0]].append(r)
    reads = by["read"]
    wait = sum(r[2] - r[1] for r in reads)
    ex = by["exec"]
    send = sum(r[5] - r[1] for r in ex)
    upload = sum(r[6] - r[5] for r in ex)
    launch = sum(r[2] - r[6] for r in ex)
    sched = sum(r[2] - r[1] for r in by["sched"])
    steps = by["step"]
    print(f"rank 0 host: {len(steps)} step() calls, {len(ex)} graph calls; blocked on the device {wait:.1f} s "
          f"({wait / wall:.1%} of wall), schedule {sched:.1f} s, broadcast to the other ranks {send:.1f} s, "
          f"argument upload {upload:.1f} s, graph launches {launch:.1f} s, other host work "
          f"{wall - wait - sched - send - upload - launch:.1f} s")
    kinds = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    for r in ex:
        k = kinds[r[3]]
        k[0] += 1
        k[1] += r[5] - r[1]
        k[2] += r[6] - r[5]
        k[3] += r[2] - r[6]
    for n, (c, s1, s2, s3, _) in kinds.items():
        print(f"  {n:8s} {c:5d} calls: broadcast {s1 / c * 1e3:.2f} ms, upload {s2 / c * 1e3:.2f} ms, launch "
              f"{s3 / c * 1e3:.2f} ms per call")
    # per step(): host time outside the blocked reads
    rs = sorted(reads, key=lambda r: r[1])
    rstarts = [r[1] for r in rs]
    free, unblocked = [], 0
    for s in steps:
        i, j = bisect.bisect_left(rstarts, s[1]), bisect.bisect_right(rstarts, s[2])
        w = sum(r[2] - r[1] for r in rs[i:j])
        free.append((s[2] - s[1]) - w)
        unblocked += j > i and all(r[2] - r[1] < a.block_ms / 1e3 for r in rs[i:j])
    free.sort()
    if free:
        print(f"  host time per step() outside blocked reads: p50 {free[len(free) // 2] * 1e3:.1f} ms, p90 "
              f"{free[int(len(free) * 0.9)] * 1e3:.1f} ms, total {sum(free):.1f} s; steps whose reads did not block "
              f"(< {a.block_ms} ms: the device had finished before the host asked) {unblocked} of {len(steps)}")
    if a.systrace:
        cmd_systrace_report(a.systrace, lo, hi)


def cmd_systrace_report(path: str, lo: float, hi: float) -> None:
    """The runtime trace of one rank (KILN_RT_INSPECT): every NEFF execution's device start and stop
    (`nc_exec_running`, nc_start_timestamp_ns / nc_stop_timestamp_ns), so the share of the traced window the device
    was executing and the gaps between executions (host scheduling, launches, waits on other ranks outside a graph).
    The ring buffer keeps the last ~0.5M events per NeuronCore by default; the warnings say how many were dropped."""
    js = path if path.endswith(".json") else None
    if js is None:
        inst = glob.glob(os.path.join(path, "*_pid_*")) or [path]
        js = os.path.join(path, "system_profile.json")
        if not os.path.exists(js):
            v = run(["neuron-explorer", "view", "-d", inst[0], "--output-format", "json", "--output-file", js,
                     "--ignore-device-profile"])
            if v.returncode:
                print(f"systrace view failed: {v.stderr[-800:]}")
                return
    d = json.load(open(js))
    for w in d.get("warnings", []):
        print(f"  trace warning: {w.get('message', w)[:300]}")
    ex = sorted((e for e in d["trace_event"] if e["name"] == "nc_exec_running"), key=lambda e: e["nc_start_timestamp_ns"])
    if not ex:
        print("  no nc_exec_running events")
        return
    by = collections.defaultdict(list)
    for e in ex:
        m = re.search(r"compile_cache/([0-9a-f]{32})/", e.get("model_name", ""))
        by[m.group(1) if m else "?"].append((e["nc_stop_timestamp_ns"] - e["nc_start_timestamp_ns"]) / 1e6)
    t0, t1 = ex[0]["nc_start_timestamp_ns"], ex[-1]["nc_stop_timestamp_ns"]
    busy = sum(e["nc_stop_timestamp_ns"] - e["nc_start_timestamp_ns"] for e in ex)
    gaps = sorted(ex[i + 1]["nc_start_timestamp_ns"] - ex[i]["nc_stop_timestamp_ns"] for i in range(len(ex) - 1))
    print(f"  runtime trace: {len(ex)} executions over {(t1 - t0) / 1e9:.2f} s, device executing {busy / (t1 - t0):.2%}; "
          f"gaps p50 {gaps[len(gaps) // 2] / 1e3:.1f} us, max {gaps[-1] / 1e6:.2f} ms, total {sum(gaps) / 1e9:.3f} s")
    for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        v.sort()
        print(f"    {k}: n={len(v)} p50 {v[len(v) // 2]:.2f} ms, total {sum(v) / 1e3:.2f} s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("model")
    m.add_argument("--shape-dir", required=True)
    m.add_argument("--ctx", type=int, default=8192, help="prompt length the prefill FLOPs are averaged over")
    m.add_argument("--decode-ctx", type=int, default=8448, help="keys a decode row reads (the page bucket)")
    m.add_argument("--tp", type=int, default=32)
    m.add_argument("--attn-tp", type=int, default=8)
    m.add_argument("--groups", type=int, default=4)
    m.add_argument("--rows-per-group", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    m.add_argument("--kv", default="fp8", choices=("fp8", "bf16"))
    r = sp.add_parser("replay")
    r.add_argument("cap")
    r.add_argument("--call", required=True, help="e.g. prefill:20 (KILN_CAPTURE_AT's name:n)")
    r.add_argument("--out", required=True)
    r.add_argument("--world", type=int, default=32)
    r.add_argument("--exec", type=int, default=2, help="executions per worker; the last is profiled, so the "
                   "workers start it from the previous one's end instead of from a cold start (default 2)")
    r.add_argument("--force", action="store_true")
    r.add_argument("--keep-ntff", action="store_true", help="keep the profiles for the bins subcommand")
    r.add_argument("--profile-all", action="store_true", help="profile every worker, not each process's first "
                   "(per-rank imbalance; one profile per rank is kept)")
    r.add_argument("--seq", type=int, nargs="*", default=None, help="only these executions (manifest seq)")
    p = sp.add_parser("report")
    p.add_argument("prof")
    p.add_argument("--kind", choices=("prefill", "decode"), required=True)
    p.add_argument("--rows", type=int, default=4096, help="prefill: tokens in the call; decode: rows in the step")
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--world", type=int, default=32)
    p.add_argument("--attn-tp", type=int, default=8)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--decode-ctx", type=int, default=8448)
    p.add_argument("--kv", default="fp8", choices=("fp8", "bf16"))
    p.add_argument("--shape-dir", default=None)
    p.add_argument("--call-s", type=float, default=None, help="the call's cost in the serving run (device_split)")
    bn = sp.add_parser("bins")
    bn.add_argument("prof")
    bn.add_argument("--bin-us", type=float, default=100.0)
    t = sp.add_parser("timeline")
    t.add_argument("timeline")
    t.add_argument("--level", type=int, default=0)
    t.add_argument("--block-ms", type=float, default=1.0)
    t.add_argument("--systrace", default=None)
    a = ap.parse_args()
    {"model": cmd_model, "replay": cmd_replay, "bins": cmd_bins, "report": cmd_report, "timeline": cmd_timeline}[a.cmd](a)


if __name__ == "__main__":
    main()
