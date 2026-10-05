"""How neuronx-cc lowers the dynamic gathers of a run of decoder blocks, by graph context, at
one tensor-parallel rank's shape (MiMo-V2.6-Flash at tp=32 by default), on one NeuronCore.

    python tools/probe_lowering.py [--blocks layer attn mlp] [--chain 1 2 6] [--first 6]

For each block kind (`layer` = DecoderForCausalLM._layer, `attn` = _attention, `mlp` = _mlp)
it compiles a graph running that block over n consecutive layers, as group_fn does, times it
(p50 of synchronous calls) and prints the compiler's count of unrolled dynamic-access-pattern
DMA instructions (neuronx-cc log, "Unrolled DGE count with Dynamic AP") for that graph: a
graph that lowers a gather element-wise shows up there as tens of thousands of DMAs.

Experiments that did NOT make multi-layer graphs lower well (trn1.2xlarge, SDK 2.32,
2026-10-03; see docs/neuron-notes.md): `layer-nostore` (no KV write: 244 DMAs but still
17.8 ms for 2 layers), `layer-noload` (KV read by a static slice), `layer-noroute` (experts
from a fixed index input), `--expert-dequant bf16` (no f32 intermediate), `--gu-layout t`
(gate_up stored input dim first: 2 layers 7.0 ms, 6 layers still 82.7 ms), `--gather-view flat`
(each expert gathered as one row), `--pad-pairs 32` (B=1 9.2 -> 23.5 ms), `--scale-select
onehot` (no scale gather: shape-dependent, and no effect once the graph holds real
all-reduces), and KILN_CC_ARGS=-O1 or --model-type=transformer. One graph per MoE layer is what
lowers well at decode B=4 and B=8.
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

import tools.profile_layer as pl

CACHE = "/root/.cache/neuron_libtorch/neuron/compile_cache"


def newest_cache_dir() -> str:
    ds = glob.glob(os.path.join(CACHE, "*"))
    return max(ds, key=os.path.getmtime) if ds else ""


def dge_count(d: str) -> str:
    try:
        with open(os.path.join(d, "log-neuron-cc.txt")) as f:
            counts = [int(m) for m in re.findall(r"Unrolled DGE count with Dynamic AP: (\d+)", f.read())]
        return str(sum(counts))
    except OSError:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="XiaomiMiMo/MiMo-V2.6-Flash-RL")
    ap.add_argument("--tp", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--pages", type=int, default=16)
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--chain", type=int, nargs="+", default=[1, 2, 6])
    ap.add_argument("--blocks", nargs="+", default=["layer", "attn", "mlp"])
    ap.add_argument("--first", type=int, default=6, help="first layer of the chain")
    ap.add_argument("--gu-layout", default="model", choices=["model", "t"],
                    help="t: gate_up experts stored [E, H, 2Im] (input dim first), as w_down is")
    ap.add_argument("--expert-dequant", default="model", choices=["model", "bf16"],
                    help="bf16: dequantize gathered experts with a bf16 multiply (experiment)")
    ap.add_argument("--pad-pairs", type=int, default=0,
                    help="gather path: pad the (token, expert) pairs to at least this many, padded rows discarded")
    ap.add_argument("--scale-select", default="gather", choices=["gather", "onehot"],
                    help="onehot: select gathered experts' bf16 scales by an exact one-hot matmul")
    ap.add_argument("--gather-view", default="model", choices=["model", "flat"],
                    help="flat: gather each expert as one row of a 2-D view, then reshape")
    args = ap.parse_args()
    if args.gather_view == "flat":
        from kiln.models import decoder as dec

        orig_experts2 = dec.DecoderForCausalLM._experts

        def experts(self, layer, name, idx=None):
            if idx is None:
                return orig_experts2(self, layer, name, idx)
            w, sc = getattr(layer, name), getattr(layer, name + "_scale")
            E = w.shape[0]
            w = w.reshape(E, -1)[idx].view(-1, *w.shape[1:])
            sc = sc.reshape(E, -1)[idx].view(-1, *sc.shape[1:])
            if name == "w_down" and layer.down_t:
                return dec.dequant_t(w, sc, self.dtype)
            return dec.dequant(w, sc, self.dtype)

        dec.DecoderForCausalLM._experts = experts
    if args.scale_select == "onehot":
        from kiln.models import decoder as dec

        orig_experts = dec.DecoderForCausalLM._experts

        def experts(self, layer, name, idx=None):
            w, sc = getattr(layer, name), getattr(layer, name + "_scale")
            if idx is None or sc is None or sc.dtype != torch.bfloat16:
                return orig_experts(self, layer, name, idx)
            E = w.shape[0]
            oh = (idx.unsqueeze(1) == torch.arange(E, device=idx.device)).to(sc.dtype)  # [P, E]
            s = (oh @ sc.reshape(E, -1)).view(-1, *sc.shape[1:])
            w = w[idx]
            if name == "w_down" and layer.down_t:
                return dec.dequant_t(w, s, self.dtype)
            return dec.dequant(w, s, self.dtype)

        dec.DecoderForCausalLM._experts = experts
    if args.pad_pairs:
        import torch.nn.functional as F

        from kiln.models import decoder as dec

        def moe(self, layer, x, _n=args.pad_pairs):
            T, H = x.shape
            k, Im = self.cfg.num_experts_per_tok, self.moe_inter
            topv, topi = self._route(layer, x)
            P = T * k
            flat = topi.reshape(P)
            xs = x.unsqueeze(1).expand(T, k, H).reshape(P, H, 1)
            if P < _n:  # padded pairs repeat pair 0 and are dropped below
                flat = torch.cat([flat, flat[:1].expand(_n - P)])
                xs = torch.cat([xs, xs[:1].expand(_n - P, H, 1)])
            gu = torch.bmm(self._experts(layer, "w_gu", flat), xs).squeeze(-1)
            a = F.silu(gu[:, :Im]) * gu[:, Im:]
            y = torch.bmm(a.unsqueeze(1), self._experts(layer, "w_down", flat)).squeeze(1)[:P]
            return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)

        dec.DecoderForCausalLM._moe = moe
    if args.expert_dequant == "bf16":
        from kiln.models import decoder as dec

        def experts(self, layer, name, idx=None):
            w, sc = getattr(layer, name), getattr(layer, name + "_scale")
            if idx is not None:
                w, sc = w[idx], sc[idx]
            if name == "w_down" and layer.down_t:
                K, nb = w.shape[-2], sc.shape[-2]
                s = sc.repeat_interleave(-(-K // nb), dim=-2)[..., :K, :] if nb > 1 else sc
            else:
                K = w.shape[-1]
                s = sc.repeat_interleave(-(-K // sc.shape[-1]), dim=-1)[..., :K]
            return w.to(self.dtype) * s.to(self.dtype)

        dec.DecoderForCausalLM._experts = experts
    pl.setup_device()
    B, P, ps = args.batch, args.pages, args.page_size
    cfg, model = pl.build(args.model, args.tp, args.first + max(args.chain), 1 + B * P, ps, P * ps)
    inp = pl.decode_inputs(model, B, P, ps)
    if args.gu_layout == "t":
        import torch.nn.functional as F

        from kiln.models import decoder as dec

        for layer in model.layers:
            if layer.moe:  # same values, input dim first; scales [E, H / 32, 2Im]
                layer.w_gu = torch.nn.Parameter(layer.w_gu.cpu().transpose(1, 2).contiguous().to(pl.DEV), requires_grad=False)
                layer.w_gu_scale = torch.nn.Parameter(layer.w_gu_scale.cpu().transpose(1, 2).contiguous().to(pl.DEV),
                                                      requires_grad=False)

        def moe(self, layer, x):
            T, H = x.shape
            k, Im = self.cfg.num_experts_per_tok, self.moe_inter
            topv, topi = self._route(layer, x)
            flat = topi.reshape(T * k)
            xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, 1, H)
            w = dec.dequant_t(layer.w_gu[flat], layer.w_gu_scale[flat], self.dtype)  # [T*k, H, 2Im]
            gu = torch.bmm(xs, w).squeeze(1)
            a = F.silu(gu[:, :Im]) * gu[:, Im:]
            y = torch.bmm(a.unsqueeze(1), self._experts(layer, "w_down", flat)).squeeze(1)
            return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)

        dec.DecoderForCausalLM._moe = moe
    from kiln.models.decoder import _LayerView

    g = torch.Generator().manual_seed(3)
    h = torch.randn(B, cfg.hidden_size, generator=g).to(torch.bfloat16).to(pl.DEV)
    fixed_idx = torch.randint(0, cfg.num_experts, (B, cfg.num_experts_per_tok), generator=g).to(pl.DEV)
    a = (inp["positions"], inp["slots"], inp["table"], inp["bias"], inp["table_w"], inp["bias_w"])
    for block in args.blocks:
        for n in args.chain:
            idxs = list(range(args.first, args.first + n))
            parts = []
            for i in idxs:
                static, tensors = model._layer_split(i)
                parts.append((static, tuple(tensors), static["spec"].window is not None))

            def f(h, pos, slots, table, bias, table_w, bias_w, *ts, _parts=parts, _block=block):
                o = 0
                for static, names, windowed in _parts:
                    v = _LayerView(static, dict(zip(names, ts[o : o + len(names)])))
                    o += len(names)
                    tb, bs = (table_w, bias_w) if windowed else (table, bias)
                    if _block == "layer":
                        h = model._layer(v, h, pos, slots, tb, bs)
                    elif _block == "layer-nostore":  # no KV write
                        st = model._store
                        model._store = lambda *a: None
                        h = model._layer(v, h, pos, slots, tb, bs)
                        model._store = st
                    elif _block == "layer-noload":  # KV read by a static slice, no gather
                        ld = model._load
                        model._load = lambda c, t: c[: t.numel() * model.page_size].view(
                            *t.shape[:-1], t.shape[-1] * model.page_size, *c.shape[1:])
                        h = model._layer(v, h, pos, slots, tb, bs)
                        model._load = ld
                    elif _block == "layer-noroute":  # experts chosen by a fixed index input
                        rt = model._route
                        model._route = lambda L, x: (torch.full((x.shape[0], cfg.num_experts_per_tok), 0.125,
                                                                dtype=x.dtype, device=x.device), fixed_idx)
                        h = model._layer(v, h, pos, slots, tb, bs)
                        model._route = rt
                    elif _block == "attn":
                        h = model._attention(v, h, pos, slots, tb, bs)
                    else:
                        h = model._mlp(v, h)
                return h

            ts = tuple(t for i in idxs for t in model.layer_tensors(i))
            kinds = "".join("w" if model.layers[i].spec.window else "F" for i in idxs)
            pl.timed(f"{block:<5} x {n} [{kinds}]", f, (h, *a, *ts))
            d = newest_cache_dir()
            print(f"    dynamic-AP DMAs {dge_count(d):>7}   {os.path.basename(d)}", flush=True)


if __name__ == "__main__":
    main()
