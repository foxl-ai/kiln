"""Real-text inputs for the long-context selection kernels (kernels/dsa_long_select.py, dsa_long_pipe*.py): GLM-5.3-Flash's
first DSA layer's indexer (layer 3) on real text, as one context-parallel rank's local pools, saved for
tools/probe_lc_pipe.py --kinds real:<file>.

    hf download zai-org/GLM-5.3-Flash config.json tokenizer.json tokenizer_config.json \\
        model-00001-of-00062.safetensors model-00031-of-00062.safetensors model-00032-of-00062.safetensors \\
        --local-dir /opt/kiln/hfx
    python tools/make_real_indexer_inputs.py --model /opt/kiln/hfx --text lc-long.txt --tokens 1048576 --cp 8 \\
        --queries 1024 --out /opt/kiln/real-1m-r0.pt

The indexer's input is layer 3's input_layernorm of the token EMBEDDINGS (layers 0-2, three Kimi Delta Attention
layers and the hyper-connection streams, are skipped: the weights and the text are real, the residual stream is not
yet the one layer 3 sees). From it, as models/mla.py _indexer and models/glm5_next.py pool_keys: every token's key
LayerNorm(wk x) and pool-gate logits, each pool of 4 tokens' key the per-channel softmax(gate + ape)-weighted mean;
for the last --queries tokens the query wq_b(q_a_layernorm(q_a_proj x)) (q_a_proj FP8 with 128 x 128 block scales)
and the head weights weights_proj(x) / sqrt(32). Rank --rank of --cp holds pools p = rank mod cp (local pool p // cp);
a query at position i has the pools whose last token is at most i (models/glm5_next.py), of which its local ones are
the npool saved. Saved: q [Q, 32, 128] bf16, w [Q, 32] fp32, pk [P / cp, 128] bf16, npool [Q] int64.
"""

from __future__ import annotations

import argparse
import json
import os

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/opt/kiln/hfx")
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokens", type=int, default=1048576)
    ap.add_argument("--cp", type=int, default=8)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--queries", type=int, default=1024)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from safetensors import safe_open
    from transformers import AutoTokenizer

    torch.set_num_threads(os.cpu_count() or 8)
    cfg = json.load(open(os.path.join(a.model, "config.json")))
    tc = cfg.get("text_config", cfg)
    Hi, Di, kp, eps = tc["index_n_heads"], tc["index_head_dim"], tc["index_kpool"], tc["rms_norm_eps"]
    idx = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    pre = "model.language_model."
    L = f"{pre}layers.{a.layer}."

    def get(name):
        with safe_open(os.path.join(a.model, idx[name]), framework="pt") as f:
            return f.get_tensor(name)

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = tok(open(a.text).read())["input_ids"]
    if len(ids) < a.tokens:
        raise SystemExit(f"{a.text} has {len(ids)} tokens, {a.tokens} needed")
    ids = torch.tensor(ids[:a.tokens])
    T = a.tokens
    emb = get(pre + "embed_tokens.weight")
    ln = get(L + "input_layernorm.weight").float()
    wk = get(L + "self_attn.indexer.wk.weight").float()
    knw, knb = get(L + "self_attn.indexer.k_norm.weight").float(), get(L + "self_attn.indexer.k_norm.bias").float()
    gate = get(L + "self_attn.indexer.index_kpool_compress_gate").float()
    ape = get(L + "self_attn.indexer.index_kpool_compress_ape").float()  # [kp, Di]
    wq_b = get(L + "self_attn.indexer.wq_b.weight").float()
    wproj = get(L + "self_attn.indexer.weights_proj.weight").float()
    qa = get(L + "self_attn.q_a_proj.weight").float()
    qs = get(L + "self_attn.q_a_proj.weight_scale_inv").float()
    qa = qa * qs.repeat_interleave(128, 0)[:qa.shape[0]].repeat_interleave(128, 1)[:, :qa.shape[1]]
    qaln = get(L + "self_attn.q_a_layernorm.weight").float()

    def norm(x):  # the layer's input RMSNorm, in fp32, rounded to bf16 as the model's activations
        x = x.float()
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * ln).to(torch.bfloat16)

    keys = torch.empty(T, 2 * Di, dtype=torch.bfloat16)
    B = 32768
    for s in range(0, T, B):
        h = norm(emb[ids[s:s + B]]).float()
        k = torch.nn.functional.layer_norm(h @ wk.T, (Di,), knw, knb, 1e-6)
        keys[s:s + B, :Di] = k.to(torch.bfloat16)
        keys[s:s + B, Di:] = (h @ gate.T).to(torch.bfloat16)
    P = T // kp
    rows = keys.view(P, kp, 2 * Di)
    logits = rows[..., Di:].float() + ape
    pk = (torch.softmax(logits, dim=-2).to(torch.bfloat16) * rows[..., :Di]).sum(dim=-2)  # glm5_next.pool_keys
    Q = a.queries
    h = norm(emb[ids[T - Q:]]).float()
    qr = h @ qa.T
    qr = (qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + eps) * qaln).to(torch.bfloat16).float()
    q = (qr @ wq_b.T).to(torch.bfloat16).view(Q, Hi, Di)
    w = (h @ wproj.T) * Hi ** -0.5
    pos = torch.arange(T - Q, T)
    npool = (pos + 1) // kp  # pools whose last token is visible
    nloc = ((npool - a.rank + a.cp - 1) // a.cp).clamp(min=0)  # of them, rank's local ones (p = rank mod cp)
    pkl = pk[a.rank::a.cp].contiguous()
    torch.save(dict(q=q, w=w.float(), pk=pkl, npool=nloc, layer=a.layer, tokens=T, cp=a.cp, rank=a.rank,
                    text=os.path.basename(a.text)), a.out)
    print(f"saved {a.out}: q {tuple(q.shape)}, pk {tuple(pkl.shape)} local of {P} pools, npool {int(nloc.min())} .. "
          f"{int(nloc.max())}; |w| mean {w.abs().mean().item():.4f}, w > 0 {float((w > 0).float().mean()):.3f}",
          flush=True)


if __name__ == "__main__":
    main()
