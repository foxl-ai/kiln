"""MiMo-V2 multi-token prediction drafts (spec_method "mtp").

The MTP head follows vLLM's mimo_v2_mtp.py (v0.30.0): x = eh_proj(cat(enorm(embed(token after
p)), hnorm(target hidden at p))), then a sliding-window decoder layer with a dense MLP, then
final_layernorm and the shared lm_head; drafts recurse on the MTP hidden. The reference here
runs that block as Xiaomi's own MiMoV2 decoder layer (tests/reference/mimo_v2), with the MTP
weights loaded into a one-layer sliding-window, dense-MLP model whose final norm is the MTP's
final_layernorm.
"""

import copy
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.test_mimo_v2 import build_reference


def add_mtp(path, layout, seed=1):
    """Random MTP weights in the checkpoint's own names (model.mtp.layers.0.*)."""
    cfg = json.load(open(os.path.join(path, "config.json")))
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    nh, nkv, dk, dv = (cfg["swa_num_attention_heads"], cfg["swa_num_key_value_heads"], cfg["swa_head_dim"],
                       cfg["swa_v_head_dim"])
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.08  # noqa: E731
    m = "model.mtp.layers.0."
    w = {m + "enorm.weight": 1 + r(H), m + "hnorm.weight": 1 + r(H), m + "final_layernorm.weight": 1 + r(H),
         m + "eh_proj.weight": r(H, 2 * H), m + "input_layernorm.weight": 1 + r(H),
         m + "pre_mlp_layernorm.weight": 1 + r(H), m + "self_attn.o_proj.weight": r(H, nh * dv),
         m + "self_attn.attention_sink_bias": r(nh) * 6, m + "mlp.gate_proj.weight": r(I, H),
         m + "mlp.up_proj.weight": r(I, H), m + "mlp.down_proj.weight": r(H, I)}
    q, k, v = r(nh * dk, H), r(nkv * dk, H), r(nkv * dv, H)
    if layout == "fused_qkv":  # stored grouped by KV head, like the real checkpoints
        from tests.test_mimo_v2 import to_grouped

        w[m + "self_attn.qkv_proj.weight"] = to_grouped(torch.cat([q, k, v]), nh, nkv, dk, dv, cfg["num_key_value_heads"])
    else:
        w[m + "self_attn.q_proj.weight"], w[m + "self_attn.k_proj.weight"], w[m + "self_attn.v_proj.weight"] = q, k, v
    tensors = load_file(os.path.join(path, "model.safetensors"))
    tensors.update(w)
    save_file(tensors, os.path.join(path, "model.safetensors"))
    cfg["num_nextn_predict_layers"] = 1
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    return w


def reference_block(hf, mtp_w, layout):
    from tests.reference.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM

    c = copy.deepcopy(hf.config)
    c.num_hidden_layers, c.hybrid_layer_pattern, c.moe_layer_freq = 1, [1], [0]
    c.attention_projection_layout = layout
    blk = MiMoV2ForCausalLM(c).eval()
    sd = {}
    for name, t in mtp_w.items():
        rest = name[len("model.mtp.layers.0."):]
        if rest.startswith(("enorm", "hnorm", "eh_proj")):
            continue
        if rest == "self_attn.qkv_proj.weight":  # back to the reference code's contiguous split
            from tests.test_mimo_v2 import from_grouped

            t = from_grouped(t, c.swa_num_attention_heads, c.swa_num_key_value_heads, c.swa_head_dim, c.swa_v_head_dim,
                             c.num_key_value_heads)
        if rest == "final_layernorm.weight":
            sd["model.norm.weight"] = t
        else:
            sd["model.layers.0." + rest.replace("pre_mlp_layernorm", "post_attention_layernorm")] = t
    missing = blk.load_state_dict(sd, strict=False)
    assert not [k for k in missing.missing_keys if k.startswith("model.layers.0") or k == "model.norm.weight"]
    return blk


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def reference_drafts(hf, blk, mtp_w, tokens, k):
    """Drafts for the position after tokens[-1]: MTP positions 0..N-1 take tokens[1..N] and
    the target's normalised hidden states, then k - 1 recursive positions."""
    m = "model.mtp.layers.0."
    eps = hf.config.layernorm_epsilon
    with torch.no_grad():
        T = torch.tensor([tokens])
        H = hf.model(T[:, :-1]).last_hidden_state[0]  # positions 0 .. N-1
        ids, hid, drafts = list(tokens[1:]), [h for h in H], []
        for _ in range(k):
            e = hf.model.embed_tokens(torch.tensor([ids]))[0]
            x = torch.cat([rms(e, mtp_w[m + "enorm.weight"], eps), rms(torch.stack(hid), mtp_w[m + "hnorm.weight"], eps)], -1)
            x = x @ mtp_w[m + "eh_proj.weight"].T
            hm = blk.model(inputs_embeds=x.unsqueeze(0)).last_hidden_state[0]
            d = int(hf.lm_head(hm[-1]).argmax())
            drafts.append(d)
            ids.append(d)
            hid.append(hm[-1])
    return drafts


@pytest.fixture(scope="module", params=["split", "fused_qkv"])
def mtp_model(request, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("mimo_mtp"))
    hf = build_reference(path, request.param)
    w = add_mtp(path, request.param)
    return path, hf, w, reference_block(hf, w, request.param)


def test_drafts_match_the_reference_after_every_step(mtp_model):
    """Chunked prefill (MTP KV filled chunk by chunk), then verify steps of every acceptance
    length: after each step, every running request's drafts equal the reference's."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf, w, blk = mtp_model
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                                 max_num_seqs=2, max_model_len=256, max_prefill_tokens=8,
                                 spec_method="mtp", spec_k=2))
    g = torch.Generator().manual_seed(3)
    reqs = [eng.add_request(torch.randint(0, 384, (n,), generator=g).tolist(),
                            SamplingParams(max_new_tokens=14, ignore_eos=True)) for n in (11, 23)]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft:
                assert r.mtp_draft == reference_drafts(hf, blk, w, r.token_ids, 2), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8


@pytest.mark.parametrize("piecewise", [False, True])
def test_mtp_speculation_leaves_greedy_output_unchanged(mtp_model, piecewise):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, *_ = mtp_model
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=2,
              max_model_len=256, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12], list(range(40, 75))]
    sp = SamplingParams(max_new_tokens=20, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    spec = LLMEngine(EngineConfig(spec_method="mtp", spec_k=3, piecewise=piecewise, **kw))
    assert [r.output_ids for r in spec.generate(prompts, sp)] == want
    assert spec.spec_proposed > 0


def test_mtp_under_tensor_parallelism_matches_tp1(mtp_model):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, *_ = mtp_model
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=128, max_num_seqs=2,
              max_model_len=128, max_prefill_tokens=8, spec_method="mtp", spec_k=2)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=12, ignore_eos=True)
    one = LLMEngine(EngineConfig(**kw))
    want = [r.output_ids for r in one.generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
        assert got == want and (two.spec_proposed, two.spec_accepted) == (one.spec_proposed, one.spec_accepted)
    finally:
        two.close()


def test_warmup_compiles_every_mtp_graph_drafting_uses(mtp_model):
    """After warmup, a whole MTP generation adds no graph key the warmup did not create."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, *_ = mtp_model
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                                 max_num_seqs=2, max_model_len=64, max_prefill_tokens=8, spec_method="mtp", spec_k=2,
                                 decode_batch_buckets=(1, 2), prefill_token_buckets=(8,), page_buckets=(4, 16)))
    eng.warmup()
    warm = {k[0] for k in eng.runner.calls}
    before = set(eng.runner.compile_seconds)
    eng.generate([[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 53))],
                 SamplingParams(max_new_tokens=16, ignore_eos=True))
    assert "mtp" in warm and set(eng.runner.compile_seconds) == before
