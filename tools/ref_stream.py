"""Reference prompt logprobs of a real MiMo-V2 checkpoint from Xiaomi's own modeling code
(tests/reference/mimo_v2), run layer by layer on the host in fp32: each layer's weights are
dequantized, used once for every prompt, and freed, so a 309B model fits in host memory.

    python tools/ref_stream.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL [--layers N]

Weights come from Kiln's checkpoint readers (FP8 128x128 blocks, MXFP4 experts) and the fused
qkv_proj is re-ordered from the checkpoint's TP interleave into the reference code's
contiguous [Q | K | V] split (kiln/models/loader.py, _grouped_qkv_rows), so this checks Kiln's
MODEL CODE against the reference with the same weight interpretation. The printed numbers are
comparable with tools/check_ppl.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.check_ppl import TEXTS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(os.cpu_count())
    from transformers import AutoTokenizer

    from kiln.models.loader import _Checkpoint, _grouped_qkv_rows, resolve_model_path
    from kiln.models.quant import dequant
    from tests.reference.mimo_v2.configuration_mimo_v2 import MiMoV2Config
    from tests.reference.mimo_v2.modeling_mimo_v2 import MiMoV2DecoderLayer, MiMoV2RMSNorm, MiMoV2RotaryEmbedding

    path = resolve_model_path(args.model)
    cfg = MiMoV2Config.from_pretrained(path)
    cfg._attn_implementation = "eager"
    n_layers = args.layers or cfg.num_hidden_layers
    ck = _Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    ids = [torch.tensor(tok(t)["input_ids"]) for t in TEXTS]

    def lin(name):
        w, s = ck.linear(name, block=128)
        return (dequant(w, s, torch.float32) if s is not None else w.float())

    hs = [ck.get("model.embed_tokens.weight")[x].float().unsqueeze(0) for x in ids]
    rot = MiMoV2RotaryEmbedding(config=cfg, is_swa=False)
    rot_swa = MiMoV2RotaryEmbedding(config=cfg, is_swa=True)
    t0 = time.time()
    for i in range(n_layers):
        layer = MiMoV2DecoderLayer(cfg, i, attention_projection_layout="fused_qkv").float().eval()
        swa = cfg.hybrid_layer_pattern[i] == 1
        p = f"model.layers.{i}."
        sd = {"input_layernorm.weight": ck.get(p + "input_layernorm.weight").float(),
              "post_attention_layernorm.weight": ck.get(p + "post_attention_layernorm.weight").float(),
              "self_attn.o_proj.weight": lin(p + "self_attn.o_proj")}
        nh = cfg.swa_num_attention_heads if swa else cfg.num_attention_heads
        nkv = cfg.swa_num_key_value_heads if swa else cfg.num_key_value_heads
        dk = cfg.swa_head_dim if swa else cfg.head_dim
        dv = cfg.swa_v_head_dim if swa else cfg.v_head_dim
        spec = types.SimpleNamespace(num_heads=nh, num_kv_heads=nkv, head_dim=dk, v_head_dim=dv)
        whole = types.SimpleNamespace(kv_offset=0, nkv=nkv)
        qkv = lin(p + "self_attn.qkv_proj")
        rows = _grouped_qkv_rows(spec, slice(0, nh * dk), whole, cfg.num_key_value_heads)
        sd["self_attn.qkv_proj.weight"] = torch.cat([qkv[r] for r in rows])
        if layer.self_attn.attention_sink_bias is not None:
            sd["self_attn.attention_sink_bias"] = ck.get(p + "self_attn.attention_sink_bias").float()
        if hasattr(layer.mlp, "experts"):
            sd["mlp.gate.weight"] = ck.get(p + "mlp.gate.weight").float()
            sd["mlp.gate.e_score_correction_bias"] = ck.get(p + "mlp.gate.e_score_correction_bias").float()
            for e in range(cfg.n_routed_experts):
                for n in ("gate_proj", "up_proj", "down_proj"):
                    sd[f"mlp.experts.{e}.{n}.weight"] = lin(f"{p}mlp.experts.{e}.{n}")
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                sd[f"mlp.{n}.weight"] = lin(f"{p}mlp.{n}")
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        if [m for m in missing if "rotary" not in m] or unexpected:
            raise RuntimeError(f"layer {i}: missing {missing[:5]}, unexpected {unexpected[:5]}")
        with torch.no_grad():
            for k, h in enumerate(hs):
                T = h.shape[1]
                pos = torch.arange(T).unsqueeze(0)
                q, kk = torch.arange(T).view(-1, 1), torch.arange(T).view(1, -1)
                vis = kk <= q
                if swa:
                    vis = vis & (kk > q - cfg.sliding_window)
                mask = torch.where(vis, 0.0, float("-inf")).view(1, 1, T, T)
                pe = (rot_swa if swa else rot)(h, pos)
                hs[k] = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=pe)
        del layer, sd
        print(f"layer {i} done ({time.time() - t0:.0f}s), |h| {hs[0].norm(dim=-1).mean():.1f}", flush=True)
    norm = MiMoV2RMSNorm(cfg.hidden_size, eps=cfg.layernorm_epsilon)
    norm.weight.data = ck.get("model.norm.weight").float()
    head = ck.get("lm_head.weight").float()
    total = n = 0
    with torch.no_grad():
        for t, x, h in zip(TEXTS, ids, hs):
            lp = torch.log_softmax(norm(h)[0] @ head.T, dim=-1)
            tl = lp[:-1].gather(1, x[1:].unsqueeze(1)).squeeze(1)
            total, n = total + tl.sum().item(), n + len(tl)
            nxt = tok.decode([int(lp[-1].argmax())])
            print(f"  {tl.mean().item():7.3f}  next {nxt!r}  {t[:50]!r}")
    print(f"RESULT reference_mean_prompt_logprob={total / n:.3f} tokens={n} layers={n_layers}")


if __name__ == "__main__":
    main()
