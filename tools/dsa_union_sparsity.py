"""How much of a pooled DSA prefill chunk's key range its query tiles actually attend: the design input for a
sparse (per-tile compacted) kernel. CPU only.

For GLM-5.3-Flash's DSA layers on real text: every query's exact pool selection (kernels/dsa_topk.py emulate, keep =
index_topk / kpool, the tail included, the visibility applied), then per tile of 128 consecutive queries the union of
the keys its queries attend, counted in blocks of --block keys. Reported per layer and per chunk of --chunk rows: the
fraction of the tile's visible blocks (those holding a key at or before its last query) that the union needs, the
same at 1024-key blocks (the causal kernels' granularity), and the fraction of the dense work a tile-compacted kernel
would do (needed / visible blocks summed over tiles).

The indexer's input is each layer's input_layernorm of the token EMBEDDINGS (as tools/make_real_indexer_inputs.py: the
weights and the text are real, the residual stream is not that layer's), so the selections are a proxy.

    python tools/dsa_union_sparsity.py --model /opt/kiln/hfx --prompt-ids long-glm.npy --rows 0 1 --layers 3 7 11
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from kiln.kernels import dsa_topk  # noqa: E402

NEG_INF = -1e30


def layer_inputs(a, idx, ids: torch.Tensor, layer: int):
    """q [T, Hi, Di] bf16, w [T, Hi] fp32, pk [P, Di] bf16 of one layer's indexer over the token ids (embedding input)."""
    from safetensors import safe_open

    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    Hi, Di, kp, eps = tc["index_n_heads"], tc["index_head_dim"], tc["index_kpool"], tc["rms_norm_eps"]
    pre = "model.language_model."
    L = f"{pre}layers.{layer}."

    def get(name):
        with safe_open(os.path.join(a.model, idx[name]), framework="pt") as f:
            return f.get_tensor(name)

    emb = get(pre + "embed_tokens.weight")
    ln = get(L + "input_layernorm.weight").float()
    wk = get(L + "self_attn.indexer.wk.weight").float()
    knw, knb = get(L + "self_attn.indexer.k_norm.weight").float(), get(L + "self_attn.indexer.k_norm.bias").float()
    gate = get(L + "self_attn.indexer.index_kpool_compress_gate").float()
    ape = get(L + "self_attn.indexer.index_kpool_compress_ape").float()
    wq_b = get(L + "self_attn.indexer.wq_b.weight").float()
    wproj = get(L + "self_attn.indexer.weights_proj.weight").float()
    qa = get(L + "self_attn.q_a_proj.weight").float()
    qs = get(L + "self_attn.q_a_proj.weight_scale_inv").float()
    qa = qa * qs.repeat_interleave(128, 0)[:qa.shape[0]].repeat_interleave(128, 1)[:, :qa.shape[1]]
    qaln = get(L + "self_attn.q_a_layernorm.weight").float()
    h = emb[ids].float()
    h = (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * ln).to(torch.bfloat16).float()
    T = ids.shape[0]
    k = torch.nn.functional.layer_norm(h @ wk.T, (Di,), knw, knb, 1e-6).to(torch.bfloat16)
    g = (h @ gate.T).to(torch.bfloat16)
    P = T // kp
    logits = g.view(P, kp, Di).float() + ape
    pk = (torch.softmax(logits, dim=-2).to(torch.bfloat16) * k.view(P, kp, Di)).sum(dim=-2)  # glm5_next.pool_keys
    qr = h @ qa.T
    qr = (qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + eps) * qaln).to(torch.bfloat16).float()
    q = (qr @ wq_b.T).to(torch.bfloat16).view(T, Hi, Di)
    w = (h @ wproj.T) * Hi ** -0.5
    return q, w.float(), pk, Di


def attended(q, w, pk, pos: torch.Tensor, keep: int, Di: int, kp: int = 4) -> torch.Tensor:
    """bool [Q, L]: the keys each query attends (its selected pools' tokens and its tail, at or before its position)."""
    Q = q.shape[0]
    P = pk.shape[0]
    last = torch.arange(P) * kp + kp - 1
    cand = torch.where(last.view(1, P) <= pos.view(Q, 1), 0.0, NEG_INF)
    sc = dsa_topk.emulate_scores(q.float(), w, pk.float(), cand, Di ** -0.5)
    sel = dsa_topk.emulate(sc, keep, True, kp, True) == 0  # [Q, kp P]: selected or tail
    L = P * kp
    return sel & (torch.arange(L).view(1, L) <= pos.view(Q, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a local checkpoint dir with config, index, and the shards needed")
    ap.add_argument("--prompt-ids", required=True, help=".npy of token ids, one prompt per row (bench/serve_sweep.py)")
    ap.add_argument("--rows", type=int, nargs="+", default=[0])
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--layers", type=int, nargs="+", default=[3])
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--out", default=None, help="write the per (row, layer, chunk) numbers as JSON")
    a = ap.parse_args()
    import numpy as np

    torch.set_num_threads(os.cpu_count() or 8)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    keep = tc["index_topk"] // tc["index_kpool"]
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    ids_all = np.load(a.prompt_ids)
    if ids_all.ndim == 1:
        ids_all = ids_all[: len(ids_all) // a.tokens * a.tokens].reshape(-1, a.tokens)
    rec = []
    T, C, B = a.tokens, a.chunk, a.block
    for layer in a.layers:
        tot = {}
        for row in a.rows:
            ids = torch.as_tensor(ids_all[row][:T].astype("int64"))
            q, w, pk, Di = layer_inputs(a, idx, ids, layer)
            for c0 in range(0, T, C):
                pos = torch.arange(c0, c0 + C)
                att = attended(q[c0:c0 + C], w[c0:c0 + C], pk, pos, keep, Di)  # [C, L]
                need = vis = need_k = vis_k = 0
                nb_call = (c0 + C - 1) // 1024 + 1  # the causal kernels' blocks for this call (its last position)
                causal = (C // 128) * nb_call * (1024 // B)  # their work, in blocks of B keys
                for t0 in range(0, C, 128):
                    u = att[t0:t0 + 128].any(0)  # the tile's union over keys
                    last = c0 + t0 + 127
                    nb = last // B + 1  # blocks holding a key at or before the tile's last query
                    ub = u[: nb * B].view(nb, B).any(-1)
                    need, vis = need + int(ub.sum()), vis + nb
                    nk = last // 1024 + 1
                    uk = torch.nn.functional.pad(u, (0, (-u.shape[0]) % 1024))[: nk * 1024].view(nk, 1024).any(-1)
                    need_k, vis_k = need_k + int(uk.sum()), vis_k + nk
                sel_frac = float(att[:, : c0 + C].float().sum(-1).mean() / (pos.float() + 1).mean())
                r = dict(row=row, layer=layer, chunk_end=c0 + C, need=need, vis=vis, frac=need / vis, need_k=need_k,
                         vis_k=vis_k, frac_k=need_k / vis_k, sel_frac=sel_frac, causal=causal)
                rec.append(r)
                t = tot.setdefault(c0 + C, [0, 0, 0, 0, 0])
                t[0] += need
                t[1] += vis
                t[2] += need_k
                t[3] += vis_k
                t[4] += causal
                print(f"layer {layer} row {row} chunk to {c0 + C:5d}: {B}-key blocks needed {need}/{vis} = {need / vis:.3f} "
                      f"({need / causal:.3f} of the causal kernel's {causal}); "
                      f"1024-key blocks {need_k}/{vis_k} = {need_k / vis_k:.3f}; keys a query attends / visible "
                      f"{sel_frac:.3f}", flush=True)
        n_all = sum(t[0] for t in tot.values())
        v_all = sum(t[1] for t in tot.values())
        c_all = sum(t[4] for t in tot.values())
        print(f"LAYER {layer}: over the {T}-token prompts' chunks a tile-compacted kernel at {B}-key blocks does "
              f"{n_all / c_all:.3f} of the causal kernel's work ({n_all / v_all:.3f} of the tiles' visible blocks); "
              "needed / visible by chunk end: " + " ".join(f"{e}:{t[0] / t[1]:.2f}" for e, t in sorted(tot.items())),
              flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rec, f)


if __name__ == "__main__":
    main()
