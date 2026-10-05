"""Expert-parallel load balance on REAL routing: GLM-5.3-Flash's routers on the real hidden states of real
sequences, every MoE layer, on the host CPU (no device), and what each rank of an expert-parallel layout
would get.

    python tools/ep_routing.py run [--model zai-org/GLM-5.3-Flash] [--text-file <wikitext>] [--random 2]
        [--tokens 4096] [--layers 45] --save <routing.pt>
    python tools/ep_routing.py stats --load <routing.pt> [--ranks 32] [--prefill 4096] [--group 1024]
        [--decode 64] [--block 128]

`run` loads the checkpoint (models/loader.py, real weights, FP8 experts kept FP8 and dequantized per
expert on the host) and runs the whole decoder in the sequence form (models/hybrid.py layer, no cache) over
each sequence: --text-file's first --tokens tokens (and further --tokens-long pieces of it, see --texts)
and --random sequences of random token ids drawn as bench/serve_sweep.py draws them (randrange(1000,
100000), seeded 0, 1, ...). Each MoE layer's routing (DecoderForCausalLM._route on that layer's FFN
input) is recorded, and the experts are computed for the routed pairs only, one expert at a time in fp32
(the same function as hybrid._moe_clamped's CPU form, written per expert so a 4096-token chunk does not
build the [E, T, 2 Im] intermediate). Saved: {"topi": {layer: int16 [T, k] per sequence}, "names": [...]}.

`stats` forms serving batches as the sweep does (prefill: --prefill rows = --prefill / --group sequences'
chunks of --group tokens each, the DP-attention groups'; decode: --decode rows of single tokens drawn from
all sequences), and for each MoE layer counts the pairs every rank holds under an expert-to-rank map:
contiguous (rank r: experts r E / ranks .. (r + 1) E / ranks - 1, models/decoder.py ep_experts),
round-robin (e % ranks), and a static balanced map (greedy longest-processing-time over the expert loads
of the OTHER half of the sequences, so it is judged on routing it was not built from). The step is paced by
the busiest rank of every layer (each layer's all-reduce waits for it), so it prints the mean over layers of
max-over-ranks / mean, of pairs and of 128-lane blocks (the EP kernel's weight passes, sum_e ceil(n_e / B)).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def run(args) -> None:
    from transformers import AutoTokenizer

    from kiln.config import ModelConfig
    from kiln.models import hybrid
    from kiln.models.loader import load_model, resolve_model_path

    path = resolve_model_path(args.model)
    cfg = ModelConfig.from_pretrained(path)
    if args.layers < cfg.num_layers:
        cfg = cfg.truncated(args.layers)
    t0 = time.time()
    model = load_model(path, cfg, torch.bfloat16, torch.device("cpu"), args.tokens + 64, 0, 1, None, keep_fp8=True)
    print(f"loaded {cfg.num_layers} layers in {time.time() - t0:.0f} s", flush=True)
    seqs, names = [], []
    if args.text_file:
        tok = AutoTokenizer.from_pretrained(path)
        ids = tok(open(args.text_file).read())["input_ids"]
        for i in range(args.texts):
            piece = ids[i * args.tokens:(i + 1) * args.tokens]
            if len(piece) < args.tokens:
                break
            seqs.append(torch.tensor(piece))
            names.append(f"text{i}")
    for s in range(args.random):
        rng = random.Random(s)
        seqs.append(torch.tensor([rng.randrange(1000, 100_000) for _ in range(args.tokens)]))
        names.append(f"random{s}")
    rec: dict[int, list] = {}
    cur = {}

    def moe_fast(model_, layer, x, limit):
        topv, topi = model_._route(layer, x)
        rec.setdefault(cur["layer"], []).append(topi.to(torch.int16).clone())
        T, H = x.shape
        xf = x.float()
        y = torch.zeros(T, H, dtype=torch.float32)
        for e in torch.unique(topi).tolist():
            t, j = (topi == e).nonzero(as_tuple=True)
            idx = torch.tensor([e])
            w_gu = model_._experts(layer, "w_gu", idx)[0].float()  # [2 Im, H]
            w_dn = model_._experts(layer, "w_down", idx)[0].float()  # [Im, H]
            gu = xf[t] @ w_gu.T
            n = gu.shape[1] // 2
            a = F.silu(gu[:, :n].clamp(max=limit)) * gu[:, n:].clamp(min=-limit, max=limit)
            y.index_add_(0, t, (a @ w_dn) * topv[t, j].float().unsqueeze(1))
        return y.to(x.dtype)

    real = hybrid._moe_clamped
    hybrid._moe_clamped = moe_fast
    try:
        with torch.no_grad():
            for si, ids in enumerate(seqs):
                T = ids.shape[0]
                positions = torch.arange(T)
                h = hybrid.hidden_in(model, ids, None)
                seq: dict = {}
                t1 = time.time()
                for i, l in enumerate(model.layers):
                    cur["layer"] = i
                    h = hybrid.layer(model, l, h, positions, None, None, None, None, seq)
                print(f"{names[si]}: {T} tokens through {len(model.layers)} layers in {time.time() - t1:.0f} s", flush=True)
                torch.save({"topi": rec, "names": names[: si + 1], "experts": cfg.num_experts}, args.save)
    finally:
        hybrid._moe_clamped = real


def owner_maps(E: int, R: int, calib: torch.Tensor) -> dict:
    """Expert -> rank maps: contiguous, round-robin, and per layer a greedy balanced one from calib
    loads [L, E] (each rank E / R experts)."""
    per = E // R
    maps = {"contiguous": torch.arange(E) // per, "round-robin": torch.arange(E) % R}
    bal = []
    for load in calib:
        order = torch.argsort(load, descending=True).tolist()
        tot = [0.0] * R
        cnt = [0] * R
        own = torch.empty(E, dtype=torch.long)
        for e in order:
            r = min((q for q in range(R) if cnt[q] < per), key=lambda q: tot[q])
            own[e] = r
            tot[r] += float(load[e])
            cnt[r] += 1
        bal.append(own)
    maps["balanced (other half)"] = torch.stack(bal)  # [L, E]
    return maps


def stats(args) -> None:
    d = torch.load(args.load)
    topi, names, E = d["topi"], d["names"], d["experts"]
    layers = sorted(topi)
    S = len(names)
    R, B = args.ranks, args.block
    k = topi[layers[0]][0].shape[1]
    print(f"{S} sequences ({', '.join(names)}), {len(layers)} MoE layers, {E} experts top-{k}, {R} ranks, "
          f"{E // R} experts per rank")
    g = torch.Generator().manual_seed(0)
    for kind in ("all", "text", "random"):
        sel = [i for i, n in enumerate(names) if kind == "all" or n.startswith(kind)]
        if len(sel) < 2:
            continue
        half_a, half_b = sel[: len(sel) // 2], sel[len(sel) // 2:]
        for evalset, calset in ((half_a, half_b), (half_b, half_a)):
            calib = torch.stack([torch.bincount(torch.cat([topi[l][i].long().flatten() for i in calset]), minlength=E)
                                 .float() for l in layers])
            maps = owner_maps(E, R, calib)
            print(f"\n[{kind}] evaluated on {[names[i] for i in evalset]}, balanced map from {[names[i] for i in calset]}")
            for shape, rows in (("prefill", args.prefill), ("decode", args.decode)):
                res = {m: [[], [], []] for m in maps}
                T = topi[layers[0]][evalset[0]].shape[0]
                for trial in range(args.trials):
                    # A batch: prefill = rows / group chunks of `group` consecutive tokens, each from a random
                    # sequence of the eval set at a random group-aligned offset; decode = rows random tokens.
                    if shape == "prefill":
                        G = min(args.group, rows)
                        picks = [(evalset[int(torch.randint(len(evalset), (1,), generator=g))],
                                  int(torch.randint(T // G, (1,), generator=g)) * G) for _ in range(rows // G)]
                        take = lambda t: torch.cat([t[i][o:o + G] for i, o in picks])  # noqa: E731
                    else:
                        picks = [(evalset[int(torch.randint(len(evalset), (1,), generator=g))],
                                  int(torch.randint(T, (1,), generator=g))) for _ in range(rows)]
                        take = lambda t: torch.stack([t[i][o] for i, o in picks])  # noqa: E731
                    for li, l in enumerate(layers):
                        n = torch.bincount(take(topi[l]).long().flatten(), minlength=E)  # pairs per expert
                        for m, own in maps.items():
                            o = own[li] if own.dim() == 2 else own
                            pairs = torch.zeros(R, dtype=torch.long).index_add_(0, o, n)
                            blocks = torch.zeros(R, dtype=torch.long).index_add_(0, o, -(-n // B))
                            res[m][0].append(pairs.max().item() / pairs.float().mean().item())
                            res[m][1].append(blocks.max().item())
                            res[m][2].append(pairs.max().item())
                mean_pairs = rows * k / R
                for m, (ratio, blk, mx) in res.items():
                    r_ = torch.tensor(ratio)
                    print(f"  {shape:7s} {rows:5d} rows ({mean_pairs:.0f} pairs per rank on average)  {m:22s} "
                          f"max/mean pairs {r_.mean():.3f} (p90 {r_.quantile(0.9):.3f}, worst {r_.max():.3f}), "
                          f"busiest rank {sum(mx) / len(mx):.0f} pairs, {sum(blk) / len(blk):.1f} blocks of {B}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    r.add_argument("--text-file", default=None)
    r.add_argument("--texts", type=int, default=2, help="pieces of --tokens from the text file")
    r.add_argument("--random", type=int, default=2, help="random-token sequences (bench/serve_sweep.py's draw)")
    r.add_argument("--tokens", type=int, default=4096)
    r.add_argument("--layers", type=int, default=45)
    r.add_argument("--save", required=True)
    s = sub.add_parser("stats")
    s.add_argument("--load", required=True)
    s.add_argument("--ranks", type=int, default=32)
    s.add_argument("--prefill", type=int, default=4096)
    s.add_argument("--group", type=int, default=1024)
    s.add_argument("--decode", type=int, default=64)
    s.add_argument("--block", type=int, default=128)
    s.add_argument("--trials", type=int, default=20)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else stats(a)


if __name__ == "__main__":
    main()
