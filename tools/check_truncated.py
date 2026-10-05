"""Compare the first L layers of a real checkpoint on the device (tensor-parallel, as served)
against the same L layers on the host CPU, by the next-token distribution of one prompt.

    python tools/check_truncated.py --model XiaomiMiMo/MiMo-V2.6-Flash-RL --layers 2 --tp 32 --piecewise

A truncated model's text is meaningless; its logits are not: device and host must agree on
the top tokens and their logprobs. Bisecting L localizes a wrong layer.
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--piecewise", action="store_true")
    ap.add_argument("--piecewise-group", type=int, default=None)
    ap.add_argument("--prompt", default="The capital of France is Paris. The capital of Italy is")
    # fp32: a bf16 host run disagreed with an fp32 host run by 2.4 nats after 2 MiMo layers
    # while the device agreed with fp32 to 0.06 (2026-10-02), so bf16 is no reference.
    ap.add_argument("--host-dtype", default="float32", choices=["bfloat16", "float32"])
    ap.add_argument("--hf-reference", action="store_true",
                    help="also compare against transformers' own model truncated to the same layers (fp32)")
    ap.add_argument("--weight-dtype", default="auto", choices=["auto", "bf16", "fp8", "fp8-experts"])
    ap.add_argument("--mxfp4-packed", action="store_true")
    ap.add_argument("--prefill-tokens", type=int, default=32, help="the prefill bucket (the prompt is cut to it)")
    ap.add_argument("--device", default="neuron", help="cpu: the same tensor-parallel engine on host processes")
    args = ap.parse_args()

    from kiln.config import EngineConfig, ModelConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams
    from kiln.models.loader import load_model, resolve_model_path

    path = resolve_model_path(args.model)
    eng = LLMEngine(EngineConfig(
        model_path=path, device=args.device, dtype=torch.bfloat16, page_size=32, max_num_seqs=1,
        max_model_len=256, max_prefill_tokens=args.prefill_tokens, kv_cache_gb=0.25, decode_batch_buckets=(1,),
        prefill_token_buckets=(args.prefill_tokens,), page_buckets=(8,), tp=args.tp, num_layers=args.layers,
        piecewise=args.piecewise, piecewise_group=args.piecewise_group, weight_dtype=args.weight_dtype,
        mxfp4_packed=args.mxfp4_packed))
    ids = eng.tokenizer(args.prompt)["input_ids"][: args.prefill_tokens]
    (r,) = eng.generate([ids], SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=20))
    eng.close()
    _, dev_ids, dev_lps = r.logprobs[0]
    import os

    torch.set_num_threads(os.cpu_count())  # rank 0 ran with cpu_count / tp threads (kiln/engine/tp.py)

    from kiln.engine.engine import weight_config

    cfg = weight_config(eng.cfg, ModelConfig.from_pretrained(path).truncated(args.layers))
    m = load_model(path, cfg, getattr(torch, args.host_dtype), torch.device("cpu"), 256, keep_fp8=True)
    with torch.no_grad():
        lp = torch.log_softmax(m.forward_logits(torch.tensor(ids)).float()[-1], -1)
    del m
    refs = {"host": lp}
    for name in ("host", "transformers"):
        if name == "transformers":
            if not args.hf_reference:
                break
            refs[name] = hf_logprobs(path, args.layers, ids)
        lp = refs[name]
        host_lps, host_ids = lp.topk(20)
        host = dict(zip(host_ids.tolist(), host_lps.tolist()))
        shared = [t for t in dev_ids if t in host]
        diff = max((abs(dict(zip(dev_ids, dev_lps))[t] - host[t]) for t in shared), default=float("nan"))
        print(f"layers {args.layers}: device top5 {dev_ids[:5]} {name} top5 {host_ids[:5].tolist()}")
        print(f"  top-20 overlap {len(shared)}/20, max |dlogprob| on shared {diff:.3f}, "
              f"top1 equal {dev_ids[0] == host_ids[0].item()}")
        print(f"RESULT ref={name} layers={args.layers} overlap={len(shared)} maxdiff={diff:.4f} "
              f"top1={dev_ids[0] == host_ids[0].item()}")
    if args.hf_reference:
        d = (refs["host"] - refs["transformers"]).abs()
        top = refs["transformers"].topk(20).indices
        print(f"RESULT host-vs-transformers max |dlogprob| over its top 20 {d[top].max().item():.5f}")


def hf_logprobs(path: str, layers: int, ids: list[int]) -> torch.Tensor:
    """Next-token logprobs of transformers' implementation with only the first `layers` layers,
    fp32 on the host. Weights are read here and handed over dequantized (transformers' own MXFP4
    dequantizer for gpt-oss), since transformers' quantized loaders need accelerate, which the SDK
    2.32 vLLM venv does not ship."""
    from transformers import AutoConfig, AutoModelForCausalLM

    import json
    import os

    if json.load(open(os.path.join(path, "config.json")))["architectures"][0] == "InklingForConditionalGeneration":
        return inkling_logprobs(path, layers, ids)
    hc = AutoConfig.from_pretrained(path, trust_remote_code=True)
    hc.num_hidden_layers = layers
    for attr in ("layer_types", "mlp_layer_types"):
        if isinstance(getattr(hc, attr, None), list):
            setattr(hc, attr, getattr(hc, attr)[:layers])
    if hasattr(hc, "quantization_config"):
        del hc.quantization_config
    hc._attn_implementation = "eager"
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(hc, dtype=torch.float32, trust_remote_code=True)
    model.load_state_dict(hf_state_dict(path, hc.architectures[0], model.state_dict()), strict=True, assign=True)
    rot = model.model.rotary_emb  # non-persistent buffers were built on meta
    model.model.rotary_emb = type(rot)(config=hc)
    with torch.no_grad():
        return torch.log_softmax(model.eval()(torch.tensor([ids])).logits[0, -1].float(), -1)


def inkling_logprobs(path: str, layers: int, ids: list[int]) -> torch.Tensor:
    """transformers 5.15 InklingForCausalLM (the text model) from the first `layers` layers of a
    thinkingmachines/Inkling checkpoint, fp32. Its config is given moe_intermediate_size = the
    text config's intermediate_size explicitly (kiln/config.py _inkling: InklingTextConfig keeps its
    default 3072 when dense_intermediate_size is set, where Inkling-Small's experts are 2048), and
    the hub tensors are renamed and de-interleaved as conversion_mapping.py "inkling_mm_model" does."""
    import json
    import os

    from transformers.models.inkling.configuration_inkling import InklingTextConfig
    from transformers.models.inkling.modeling_inkling import InklingForCausalLM

    from kiln.models.loader import _Checkpoint

    t = dict(json.load(open(os.path.join(path, "config.json")))["text_config"])
    t.pop("torch_dtype", None)
    local = t.get("local_layer_ids")
    t["num_hidden_layers"] = layers
    if local is not None:
        t["local_layer_ids"] = [i for i in local if i < layers]
    hc = InklingTextConfig(**t, moe_intermediate_size=t["intermediate_size"])
    hc._attn_implementation = "eager"
    with torch.device("meta"):
        model = InklingForCausalLM(hc)
    ck = _Checkpoint(path)

    def de(w, dim):  # rows 2u gate, 2u + 1 up -> [gate; up] (core_model_loading.Interleave)
        return torch.cat([w.index_select(dim, torch.arange(0, w.shape[dim], 2)),
                          w.index_select(dim, torch.arange(1, w.shape[dim], 2))], dim=dim)

    renames = [("self_attn.q_proj", "attn.wq_du"), ("self_attn.k_proj", "attn.wk_dv"),
               ("self_attn.v_proj", "attn.wv_dv"), ("self_attn.r_proj", "attn.wr_du"), ("self_attn.o_proj", "attn.wo_ud"),
               ("self_attn.k_sconv.conv1d", "attn.k_sconv"), ("self_attn.v_sconv.conv1d", "attn.v_sconv"),
               ("attn_sconv.conv1d", "attn_sconv"), ("mlp_sconv.conv1d", "mlp_sconv"), ("self_attn.", "attn."),
               ("input_layernorm", "attn_norm"), ("post_attention_layernorm", "mlp_norm"),
               ("gate.e_score_correction_bias", "gate.bias"), ("model.layers", "model.llm.layers"),
               ("model.embed_tokens", "model.llm.embed"), ("model.embed_norm", "model.llm.embed_norm"),
               ("model.norm", "model.llm.norm"), ("lm_head", "model.llm.unembed")]
    sd = {}
    for k, ref in model.state_dict().items():
        hub = k
        for a, b in renames:
            hub = hub.replace(a, b)
        if k.endswith("mlp.gate_proj.weight") or k.endswith("mlp.up_proj.weight"):
            w = de(ck.get(hub.rsplit(".", 2)[0] + ".w13_dn.weight").float(), 0)
            v = w[: w.shape[0] // 2] if "gate_proj" in k else w[w.shape[0] // 2 :]
        elif k.endswith("mlp.down_proj.weight"):
            v = ck.get(hub.replace("down_proj.weight", "w2_md.weight")).float()
        elif k.endswith("experts.gate_up_proj"):
            v = de(ck.get(hub.replace("gate_up_proj", "w13_weight")).float(), 1)
        elif k.endswith("experts.down_proj") and "shared" not in k:
            v = ck.get(hub.replace("down_proj", "w2_weight")).float()
        elif k.endswith(("shared_experts.gate_proj", "shared_experts.up_proj")):
            w = de(ck.get(hub.rsplit(".", 1)[0] + ".shared_w13_weight").float(), 1)
            v = w[:, : w.shape[1] // 2] if k.endswith("gate_proj") else w[:, w.shape[1] // 2 :]
        elif k.endswith("shared_experts.down_proj"):
            v = ck.get(hub.replace("down_proj", "shared_w2_weight")).float()
        else:
            v = ck.get(hub).float()
        if v.shape != ref.shape:
            raise ValueError(f"{k}: checkpoint {tuple(v.shape)} vs model {tuple(ref.shape)}")
        sd[k] = v
    model.load_state_dict(sd, strict=True, assign=True)
    with torch.no_grad():
        return torch.log_softmax(model.eval()(torch.tensor([ids])).logits[0, -1].float(), -1)


def hf_state_dict(path: str, arch: str, want: dict) -> dict:
    """transformers' parameter names -> fp32 tensors from the hub checkpoint: gpt-oss's fused MXFP4
    experts (`*_blocks` / `*_scales`, integrations/mxfp4.py convert_moe_packed_tensors); Hy3's
    per-expert weights stacked and its renames (conversion_mapping.py "hy_v3"); per-tensor FP8
    weights times their weight_scale."""
    from kiln.models.loader import _Checkpoint

    ck = _Checkpoint(path)

    def w(name):
        t = ck.get(name).float()
        sc = name[: -len("weight")] + "weight_scale"
        return t * ck.get(sc).float() if name.endswith(".weight") and sc in ck else t

    out = {}
    for k, ref in want.items():
        if k.endswith(("experts.gate_up_proj", "experts.down_proj")) and k + "_blocks" in ck:
            from transformers.integrations.mxfp4 import convert_moe_packed_tensors

            out[k] = convert_moe_packed_tensors(ck.get(k + "_blocks"), ck.get(k + "_scales"), dtype=torch.float32)
        elif arch == "HYV3ForCausalLM" and k.endswith(("experts.gate_up_proj", "experts.down_proj")):
            pre, E = k[: k.rindex(".") + 1], ref.shape[0]
            if k.endswith("gate_up_proj"):
                out[k] = torch.stack([torch.cat([w(f"{pre}{e}.gate_proj.weight"), w(f"{pre}{e}.up_proj.weight")])
                                      for e in range(E)])
            else:
                out[k] = torch.stack([w(f"{pre}{e}.down_proj.weight") for e in range(E)])
        else:
            name = k
            if arch == "HYV3ForCausalLM":
                name = (k.replace("mlp.gate.weight", "mlp.router.gate.weight")
                        .replace("mlp.e_score_correction_bias", "mlp.expert_bias")
                        .replace("mlp.shared_experts.", "mlp.shared_mlp."))
            out[k] = w(name)
        if out[k].shape != ref.shape:
            raise ValueError(f"{k}: checkpoint {tuple(out[k].shape)} vs model {tuple(ref.shape)}")
    return out


if __name__ == "__main__":
    main()
