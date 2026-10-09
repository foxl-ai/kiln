"""Time and device-profile the attention kernels alone at GLM-5.3-Flash's serving shapes (tp=32, DP attention 4,
attention TP 8: 8 attention heads / 8 KDA heads per rank, a 32 x 128 indexer, 8448 keys = page bucket 264), one
NeuronCore, with random inputs of the right kind, and say where each engine's time goes.

    python tools/prof_attn_kernels.py <case> [<case> ...] [--rows N ...] [--no-profile]

Cases (each at every --rows given, else its serving rows):
  kda_prefill   kernels/delta_rule.py chunk, KDA, 8 heads, C rows (serving: 1024 per DP group)
  kda_decode    kernels/kda_decode.py, B rows of 8 heads (serving: 4 / 8 / 16 rows per group)
  dsa_decode    kernels/dsa_decode.py, B rows, fp8 latent cache, 640 slots of 4 tokens
  dsa_score     kernels/dsa_topk.py score_select: C queries x 2112 pools, 32 heads, keep 512, kp 4 + tail
  dsa_select    kernels/dsa_topk.py select: B rows x 2112 pool scores, keep 512, kp 1 (the decode-slot selection)
  dsa_slots     kernels/dsa_slots.py: n rows x KILN_PROF_HEADS (8) heads over 640 slots, fp8 latent cache
  dsa_long      kernels/dsa_long_select.py: C queries x KILN_PROF_POOLS (32768) pools, 32 heads, keep 512
  dsa_long_pipe kernels/dsa_long_pipe.py: the same over C queries (whole tiles of 128, at least 2) in one pipelined call
  dsa_index     kernels/dsa_index.py: B decode rows, each its own 1M-token context (32768 pages, 262,144 pools), 32
                indexer heads of 128 (the long-context decode scores)
  dsa_prefill   kernels/dsa_prefill.py at C queries over 8448 keys (tools/probe_dsa_prefill.py's case, the chunk at the
                bucket's end)
  dsa_fused     kernels/dsa_fused.py: C queries x 8448 keys, 32 x 128 indexer, keep 512, 8 heads, the chunk at
                KILN_PROF_OFFSET (default the bucket's end; tools/probe_dsa_fused.py's case)
  dsa_fused_c   kernels/dsa_fused_c.py (the causal form, KILN_DSA_FUSED_CAUSAL_LADDER as set) on the same case
  dsa_split_att kernels/dsa_split.py's attention-only kernel on that case's pool selection (emulated on the host)
  dsa_core      the XLA attention core of a pooled DSA prefill chunk (models/mla.py _core, expand): C queries x
                8448 keys with an additive mask, 8 heads, latent 512, dn = dv = 256

Per case: p50 of synchronous calls (profile_layer.timed, each graph's output reduced to a scalar so the read-back is
nothing), then (unless --no-profile) `neuron-explorer capture` of the graph's NEFF on its real inputs and, from the
profile's JSON, the kernel's span, each engine's busy time (union of its instruction intervals), its instruction
count and top opcodes, and the DMA bytes by queue. Outputs under /opt/kiln/prof/ak/<case>-<rows>.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

OUT = "/opt/kiln/prof/ak"
NEG_INF = -1e30
SERVING = {"kda_prefill": [1024], "kda_decode": [4, 16, 64], "dsa_decode": [4, 16, 64], "dsa_score": [1024],
           "dsa_select": [16, 64], "dsa_core": [1024], "dsa_prefill": [1024], "dsa_long": [128], "dsa_long_pipe": [1024], "dsa_slots": [128],
           "dsa_index": [1, 4], "gemv": [1, 16], "cp_classes": [1024], "cp_classes_x": [1024],
           "dsa_long_pipe_x": [1024], "dsa_fused": [1024], "dsa_fused_c": [1024], "dsa_split_att": [1024]}


def _raw(t: torch.Tensor) -> np.ndarray:
    t = t.contiguous()
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.view(view).numpy() if t.dtype.is_floating_point or t.dtype == torch.bool else t.numpy()


def build(case: str, n: int):
    """(graph function, {argument name: host tensor}) of one case at n rows."""
    g = torch.Generator().manual_seed(n)
    if case == "kda_prefill":
        from tests.test_delta_rule import inputs

        from kiln.kernels import delta_rule as dr

        q, k, v, gl, b, S = inputs(n, 8, 8, True, seed=n)

        def f(q, k, v, g, b, S):
            o, S2 = dr.chunk(q, k, v, g, b, S)
            return o.sum() + S2.sum()

        return f, dict(q=q, k=k, v=v, g=gl, b=b, S=S)
    if case == "kda_decode":
        from probe_kda_decode import inputs as kin

        from kiln.kernels.kda_decode import decode_step

        pool, slot, keep, q, k, v, gl, beta = kin(n, max(65, n + 2), 8, 128, n)

        def f(pool, slot, keep, q, k, v, g, beta):
            return decode_step(pool, slot, keep, q, k, v, g, beta).sum()

        return f, dict(pool=pool, slot=slot, keep=keep, q=q, k=k, v=v, g=gl, beta=beta)
    if case == "dsa_decode":
        from probe_dsa_decode import case as dcase

        from kiln.kernels.dsa_decode import attend

        kc, table, q_lat, rows, bias, mask = dcase(n, 264, 264 * 32 - 1, torch.float8_e4m3fn, n)

        def f(kc, rows, q_lat, bias):
            return attend(q_lat, kc, rows, bias, 256 ** -0.5).sum()

        return f, dict(kc=kc, rows=rows, q_lat=q_lat, bias=bias)
    if case == "dsa_score":
        from kiln.kernels import dsa_topk as dk

        P = 2112
        q = torch.randn(n, 32, 128, generator=g).to(torch.bfloat16)
        pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
        w = torch.randn(n, 32, generator=g) * 32 ** -0.5
        cand = torch.zeros(n, P)

        def f(q, w, pk, cand):
            return dk.score_select(q, w, pk, cand, 512, 128 ** -0.5, 4, True).sum()

        return f, dict(q=q, w=w, pk=pk, cand=cand)
    if case == "dsa_long_pipe":  # kernels/dsa_long_pipe.py: n queries (>= 256) in one pipelined call
        from kiln.kernels import dsa_long_pipe as dp

        P = int(os.environ.get("KILN_PROF_POOLS", 32768))
        q = torch.randn(n, 32, 128, generator=g).to(torch.bfloat16)
        pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
        w = torch.randn(n, 32, generator=g) * 32 ** -0.5
        npool = torch.full((n,), P)

        def f(q, w, pk, npool):
            return dp.select(q, w, pk, npool, 512, 128 ** -0.5)[0].sum()

        return f, dict(q=q, w=w, pk=pk, npool=npool)
    if case == "dsa_long_pipe_x":  # kernels/dsa_long_pipe_x.py: the chunk skip (KILN_PROF_TOP: the chunk's end as a
        # fraction of the bucket, every query within n pools below it) and / or KILN_PROF_PE=1 the tensor-engine head sum
        from kiln.kernels import dsa_long_pipe_x as dpx

        P = int(os.environ.get("KILN_PROF_POOLS", 32768))
        top = int(round(float(os.environ.get("KILN_PROF_TOP", 1.0)) * P))
        skip = os.environ.get("KILN_PROF_SKIP", "1") == "1"
        pe = os.environ.get("KILN_PROF_PE", "0") == "1"
        q = torch.randn(n, 32, 128, generator=g).to(torch.bfloat16)
        pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
        w = torch.randn(n, 32, generator=g) * 32 ** -0.5
        npool = (top - n + torch.arange(n)).clamp(0, P)

        def f(q, w, pk, npool):
            return dpx.select(q, w, pk, npool, 512, 128 ** -0.5, skip=skip, pe=pe)[0].sum()

        return f, dict(q=q, w=w, pk=pk, npool=npool)
    if case in ("cp_classes", "cp_classes_x"):  # models/mla.py _cp_attend_classes(_x): tools/probe_cp_slots_x.py's inputs
        import probe_cp_slots_x as pcx

        from kiln.models import mla

        fits = float(os.environ.get("KILN_PROF_FITS", 0.7))
        names = ("q_all", "kc", "rows_sel", "rows_tail", "mine", "tail_own", "npool", "positions")
        ins = dict(zip(names, pcx.make_inputs(n, int(os.environ.get("KILN_PROF_HEADS", 64)), fits, g)))
        fn = mla._cp_attend_classes_x if case == "cp_classes_x" else mla._cp_attend_classes

        def f(q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions):
            o, ls = fn(q_all, kc, rows_sel, rows_tail, mine, tail_own, npool, positions, 192 ** -0.5, 4, 128, 640)
            return o.clamp(min=-1.0).sum() + ls.clamp(min=-1.0).sum()

        return f, ins
    if case == "dsa_long":  # kernels/dsa_long_select.py: n queries x KILN_PROF_POOLS pools (default 32768), keep 512
        from kiln.kernels import dsa_long_select as dl

        P = int(os.environ.get("KILN_PROF_POOLS", 32768))
        q = torch.randn(n, 32, 128, generator=g).to(torch.bfloat16)
        pk = torch.randn(P, 128, generator=g).to(torch.bfloat16)
        w = torch.randn(n, 32, generator=g) * 32 ** -0.5
        npool = torch.full((n,), P)

        def f(q, w, pk, npool):
            return dl.select(q, w, pk, npool, 512, 128 ** -0.5)[0].sum()

        return f, dict(q=q, w=w, pk=pk, npool=npool)
    if case == "dsa_slots":  # kernels/dsa_slots.py: n rows x KILN_PROF_HEADS (8) heads, 640 slots, fp8 latent
        from kiln.kernels import dsa_slots

        H = int(os.environ.get("KILN_PROF_HEADS", 8))
        # tools/probe_dsa_long.py slots' inputs: 640 random slots per row over a 4096-page fp8 latent cache
        R, NS = 512, 640
        kc = (torch.randn(4096 * 32, 1, R, generator=g) * 2).clamp(-200, 200).to(torch.float8_e4m3fn)
        q_lat = (torch.randn(n, H, R, generator=g) * 0.05).to(torch.bfloat16)
        rows = torch.randint(0, 4096 * 32 // 4, (n, NS), generator=g)
        bias = torch.zeros(n, NS, 4)
        bias[:, 513:] = NEG_INF

        def f(kc, rows, q_lat, bias):
            return dsa_slots.attend(q_lat, kc, rows, bias, 256 ** -0.5).sum()

        return f, dict(kc=kc, rows=rows, q_lat=q_lat, bias=bias)
    if case == "dsa_select":
        from kiln.kernels import dsa_topk as dk

        sc = torch.randn(n, 1, 2112, generator=g)

        def f(sc):
            return dk.select(sc, 512, vis_only=True, kp=1, tail=False).sum()

        return f, dict(sc=sc)
    if case == "gemv":
        from kiln.kernels import gemv as gk

        K = N = 4096
        x = torch.randn(n, K, generator=g).to(torch.bfloat16)
        wT = (torch.randn(K, N, generator=g) * K ** -0.5).to(torch.bfloat16)

        def f(x, wT):
            return gk.gemv(x, wT).float().sum()

        return f, dict(x=x, wT=wT)
    if case == "dsa_index":
        from probe_dsa_index import case as icase

        from kiln.kernels import dsa_index as di

        pkc, table, q, w, cand = icase(n, int(os.environ.get("KILN_PROF_PAGES", 32768)), n)
        ppg = di.page_groups(table)

        def f(q, w, pkc, ppg, cand):
            return di.scores(q, w, pkc, ppg, cand).sum()

        return f, dict(q=q, w=w, pkc=pkc, ppg=ppg, cand=cand)
    if case == "dsa_core":
        L, H, r, dn = 8448, 8, 512, 256
        kc = (torch.randn(1, L, r, generator=g)).to(torch.bfloat16)
        qn = (torch.randn(n, H, dn, generator=g) * 0.1).to(torch.bfloat16)
        w_uk = (torch.randn(H, dn, r, generator=g) * r ** -0.5).to(torch.bfloat16)
        w_uv = (torch.randn(H, dn, r, generator=g) * r ** -0.5).to(torch.bfloat16)
        sel = torch.rand(n, L // 4, generator=g) < 0.25
        mask = torch.where(sel, 0.0, NEG_INF).repeat_interleave(4, dim=1)
        pos = torch.arange(n) + (L - n)
        vis = torch.where(torch.arange(L).view(1, L) <= pos.view(n, 1), 0.0, NEG_INF)
        bias = (mask + vis).view(1, n, L).float()

        def f(qn, kc, w_uk, w_uv, bias):
            k_nope = torch.einsum("blr,hdr->blhd", kc, w_uk)
            v = torch.einsum("blr,hvr->blhv", kc, w_uv)
            s = torch.einsum("bqhd,blhd->bhql", qn.unsqueeze(0), k_nope)
            p = torch.softmax(s.float() * dn ** -0.5 + bias.unsqueeze(1), dim=-1).to(torch.bfloat16)
            return torch.einsum("bhql,blhv->bqhv", p, v).float().sum()

        return f, dict(qn=qn, kc=kc, w_uk=w_uk, w_uv=w_uv, bias=bias)
    if case == "dsa_split_att":
        from probe_dsa_fused import D, KEEP
        from probe_dsa_fused import case as fcase

        from kiln.kernels import dsa_split as dsp

        off = int(os.environ.get("KILN_PROF_OFFSET", 8448 - n))
        qI, w, pk, pos, q_lat, kc = fcase(n, 8448, off, n + off)
        sel = dsp.emulate_select(qI.float(), w, pk.float(), pos, KEEP, D ** -0.5)

        def f(sel, pos, q_lat, kc):
            return dsp.attend(sel, pos, q_lat, kc, 256 ** -0.5).sum()

        return f, dict(sel=sel, pos=pos, q_lat=q_lat, kc=kc)
    if case in ("dsa_fused", "dsa_fused_c"):
        from probe_dsa_fused import D, KEEP
        from probe_dsa_fused import case as fcase

        from kiln.kernels import dsa_fused as df
        from kiln.kernels import dsa_fused_c as dfc

        off = int(os.environ.get("KILN_PROF_OFFSET", 8448 - n))
        qI, w, pk, pos, q_lat, kc = fcase(n, 8448, off, n + off)
        fn = df.attend if case == "dsa_fused" else dfc.attend

        def f(qI, w, pk, pos, q_lat, kc):
            return fn(qI, w, pk, pos, q_lat, kc, KEEP, D ** -0.5, 256 ** -0.5).sum()

        return f, dict(qI=qI, w=w, pk=pk, pos=pos, q_lat=q_lat, kc=kc)
    if case == "dsa_prefill":
        from probe_dsa_prefill import case as pcase

        from kiln.kernels import dsa_prefill as dp

        q_lat, kc, mask, *_ = pcase(n, 8448, 8448 - n, n)

        def f(q_lat, kc, mask):
            return dp.attend(q_lat, kc, mask, 256 ** -0.5).sum()

        return f, dict(q_lat=q_lat, kc=kc, mask=mask)
    raise SystemExit(f"unknown case {case}")


def union(iv) -> int:
    tot, end = 0, -1
    for a, b in sorted(iv):
        if b <= end:
            continue
        tot += b - max(a, end)
        end = b
    return tot


def save_inputs(entry: str, vals: dict, out: str) -> None:
    """The graph's inputs as input<i>.npy in its placeholder order, and its cache entry, for capture()."""
    order = re.findall(r"placeholder\[target=L_(\w+?)_\]", open(os.path.join(entry, "fxgraph.txt")).read())
    for i, nm in enumerate(order):
        np.save(os.path.join(out, f"input{i}.npy"), _raw(vals[nm]))
    with open(os.path.join(out, "entry.txt"), "w") as f:
        f.write(f"{entry}\n{len(order)}\n")


def capture(out: str) -> None:
    """Profile one saved case (a process that holds no NeuronCore: neuron-explorer's nrt_init needs it free)."""
    entry, n = open(os.path.join(out, "entry.txt")).read().split()
    neff = glob.glob(os.path.join(entry, "*.neff"))[0]
    ifm = []
    for i in range(int(n)):
        ifm += [f"input{i}", os.path.join(out, f"input{i}.npy")]
    print(f"{os.path.basename(out)}:", flush=True)
    ntff = os.path.join(out, "profile.ntff")
    js = os.path.join(out, "profile.json")
    env = dict(os.environ, HOME=os.environ.get("HOME", "/root"))
    cap = subprocess.run(["neuron-explorer", "capture", "-n", neff, "-s", ntff, *ifm], env=env, capture_output=True,
                         text=True)
    if cap.returncode:
        print(cap.stdout[-1500:], cap.stderr[-1500:], flush=True)
        return
    for f in glob.glob(os.path.join(out, "*.json")):
        os.remove(f)
    subprocess.run(["neuron-explorer", "view", "-n", neff, "-s", ntff, "--output-format", "json"], check=True, env=env,
                   cwd=out, capture_output=True)
    os.replace(sorted(glob.glob(os.path.join(out, "*.json")), key=os.path.getmtime)[-1], js)
    d = json.load(open(js))
    ins = d["instruction"]
    kern = [i for i in ins if not i.get("hlo_name")]
    t0 = min(i["timestamp"] for i in kern) if kern else 0
    t1 = max(i["timestamp"] + (i.get("duration") or 0) for i in kern) if kern else 0
    allt0 = min(i["timestamp"] for i in ins)
    allt1 = max(i["timestamp"] + (i.get("duration") or 0) for i in ins)
    print(f"    profile: {len(ins)} instructions over {(allt1 - allt0) / 1e3:.1f} us; kernel (no HLO name) "
          f"{len(kern)} instructions over {(t1 - t0) / 1e3:.1f} us", flush=True)
    for label, sel in (("kernel", kern), ("xla", [i for i in ins if i.get("hlo_name")])):
        if not sel:
            continue
        iv, n_, wait, op = collections.defaultdict(list), collections.Counter(), collections.Counter(), \
            collections.defaultdict(collections.Counter)
        for i in sel:
            e = i.get("subgroup", "?")
            du = i.get("duration") or 0
            iv[e].append((i["timestamp"], i["timestamp"] + du))
            n_[e] += 1
            wait[e] += i.get("evt_wait_time", 0) or 0
            op[e][i.get("opcode", "?")] += du
        for e in sorted(iv, key=lambda k: -union(iv[k])):
            top = ", ".join(f"{k} {v / 1e3:.0f}" for k, v in op[e].most_common(5))
            print(f"    {label:6s} {e:<12} busy {union(iv[e]) / 1e3:8.1f} us  {n_[e]:6d} instr  waited "
                  f"{wait[e] / 1e3:8.1f} us | {top}", flush=True)
    q = collections.Counter()
    for p in d.get("dma", []):
        q[p.get("subgroup", "?")] += p.get("transfer_size", 0) or 0
    print("    dma by queue: " + ", ".join(f"{k} {v / 1e6:.1f}MB" for k, v in q.most_common(8)), flush=True)
    s = d["summary"][0]
    keys = ("total_time", "hbm_read_bytes", "hbm_write_bytes", "spill_save_bytes", "spill_reload_bytes",
            "software_dynamic_dma_size", "static_dma_size", "tensor_engine_active_time", "vector_engine_active_time",
            "scalar_engine_active_time", "gpsimd_engine_active_time", "dma_active_time")
    print("    summary: " + ", ".join(f"{k}={s[k]}" for k in keys if k in s), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--rows", type=int, nargs="*", default=None)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--capture", action="store_true", help="the cases are saved output directories: profile them")
    a = ap.parse_args()
    if a.capture:
        for out in a.cases:
            capture(out)
        return
    outs = []
    import logging

    import neff_instructions as ni
    import profile_layer as pl

    keys = []

    class _Keys(logging.Handler):  # LNL logs every graph's cache key (backend.py "Compilation cache key: <hash>")
        def emit(self, rec):
            m = re.search(r"Compilation cache key: (\w+)", rec.getMessage())
            if m:
                keys.append(m.group(1))

    pl.setup_device()
    logging.getLogger("libtorch_neuronx_lite.compile.backend").addHandler(_Keys())
    for case in a.cases:
        for n in a.rows or SERVING[case]:
            f, vals = build(case, n)
            dev = tuple(v.to(pl.DEV) for v in vals.values())
            keys.clear()
            t = pl.timed(f"{case} rows {n}", f, dev, a.iters)
            ents = [os.path.join(ni.cache_root(), keys[0])] if keys else []
            if t != t or not ents:
                continue
            print(f"  {case} rows {n}: {t * 1e3:.3f} ms; graph {os.path.basename(ents[0])}: {ni.line(ents[0])}",
                  flush=True)
            if a.no_profile:
                continue
            out = os.path.join(OUT, f"{case}-{n}")
            os.makedirs(out, exist_ok=True)
            save_inputs(ents[0], vals, out)
            outs.append(out)
    if outs:  # this process holds the NeuronCores (an exec keeps the device open): a fresh one captures after it exits
        sys.stdout.flush()
        # NEURON_RT_VISIBLE_CORES stays: on a shared box the capture must use this process's core, not core 0
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEURON_RT_") or k == "NEURON_RT_VISIBLE_CORES"}
        cmd = " ".join([sys.executable, os.path.abspath(__file__), "--capture", *outs])
        subprocess.Popen(["bash", "-c", f"while kill -0 {os.getpid()} 2>/dev/null; do sleep 1; done; sleep 3; {cmd}"],
                         env=env, start_new_session=True)


if __name__ == "__main__":
    main()
