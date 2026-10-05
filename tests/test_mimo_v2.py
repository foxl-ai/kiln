"""MiMo-V2 (hybrid full / sliding-window attention, attention sinks, partial RoPE, Dk != Dv,
value scale, sigmoid + correction-bias MoE) against Xiaomi's own modeling code.

The reference is tests/reference/mimo_v2 (Apache-2.0, from the XiaomiMiMo/MiMo-V2.6-Flash-RL
model repository). A tiny text-only config exercises every feature; both sides run on the
same random weights.
"""

import pytest
import torch


def build_reference(path, layout="split", hidden_size=64, moe_intermediate_size=32):
    from tests.reference.mimo_v2.configuration_mimo_v2 import MiMoV2Config
    from tests.reference.mimo_v2.modeling_mimo_v2 import MiMoV2ForCausalLM

    cfg = MiMoV2Config(
        vocab_size=384, hidden_size=hidden_size, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, layernorm_epsilon=1e-6,
        rope_theta=1.0e7, attention_value_scale=0.707, head_dim=24, v_head_dim=16,
        swa_num_attention_heads=4, swa_num_key_value_heads=4, swa_head_dim=24, swa_v_head_dim=16,
        swa_rope_theta=1.0e4, sliding_window=8, sliding_window_size=8, add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False, hybrid_layer_pattern=[0, 1, 1, 0], partial_rotary_factor=0.334,
        n_routed_experts=8, moe_intermediate_size=moe_intermediate_size, num_experts_per_tok=2, scoring_func="sigmoid",
        topk_method="noaux_tc", n_group=1, topk_group=1, norm_topk_prob=True, moe_layer_freq=[0, 1, 1, 1],
        attention_projection_layout=layout, tie_word_embeddings=False,
    )
    cfg._attn_implementation = "eager"
    model = MiMoV2ForCausalLM(cfg)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.normal_(0, 0.5 if ("sink" in name or "correction" in name) else 0.08)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    if layout == "fused_qkv":
        regroup_fused_qkv(path, cfg)
    return model


def to_grouped(w, nh, nkv, dk, dv, T):
    """Contiguous [Q | K | V] rows (the reference code's split) -> the checkpoints' T-chunk
    interleave, chunk c = [Q | K | V] of heads c * nh / T ... (see kiln/models/loader.py,
    _grouped_qkv_rows); T is the config's top-level num_key_value_heads."""
    qc, kc = nh // T, nkv // T
    q, k, v = w[: nh * dk], w[nh * dk : (nh + nkv) * dk], w[(nh + nkv) * dk :]
    parts = []
    for c in range(T):
        parts += [q[c * qc * dk : (c + 1) * qc * dk], k[c * kc * dk : (c + 1) * kc * dk], v[c * kc * dv : (c + 1) * kc * dv]]
    return torch.cat(parts)


def from_grouped(w, nh, nkv, dk, dv, T):
    qc, kc = nh // T, nkv // T
    stride = qc * dk + kc * dk + kc * dv
    g = [w[c * stride : (c + 1) * stride] for c in range(T)]
    return torch.cat([x[: qc * dk] for x in g] + [x[qc * dk : qc * dk + kc * dk] for x in g]
                     + [x[qc * dk + kc * dk :] for x in g])


def qkv_dims(cfg, i):
    swa = cfg.hybrid_layer_pattern[i] == 1
    T = cfg.num_key_value_heads
    if swa:
        return cfg.swa_num_attention_heads, cfg.swa_num_key_value_heads, cfg.swa_head_dim, cfg.swa_v_head_dim, T
    return cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.v_head_dim, T


def regroup_fused_qkv(path, cfg):
    """Store every fused qkv_proj the way the real MiMo-V2 checkpoints do."""
    import os

    from safetensors.torch import load_file, save_file

    f = os.path.join(path, "model.safetensors")
    t = load_file(f)
    for name in list(t):
        if name.endswith("self_attn.qkv_proj.weight") and name.startswith("model.layers."):
            t[name] = to_grouped(t[name], *qkv_dims(cfg, int(name.split(".")[2]))).contiguous()
    save_file(t, f, metadata={"format": "pt"})


@pytest.fixture(scope="module", params=["split", "fused_qkv"])
def ref(request, tmp_path_factory):
    path = tmp_path_factory.mktemp("mimo")
    return str(path), build_reference(str(path), request.param)


def test_config_is_parsed_as_hybrid(ref):
    from kiln.config import ModelConfig

    path, _ = ref
    cfg = ModelConfig.from_pretrained(path)
    kinds = [(s.window, s.num_kv_heads, s.sink, s.rope_theta) for s in cfg.attn_layers]
    assert kinds == [(None, 2, False, 1e7), (8, 4, True, 1e4), (8, 4, True, 1e4), (None, 2, False, 1e7)]
    assert cfg.attn_layers[0].rope_dim == 8 and cfg.attn_layers[0].v_head_dim == 16
    assert cfg.moe_layers == (1, 2, 3) and cfg.router_scoring == "sigmoid" and cfg.router_bias


def test_logits_match_reference(ref):
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = ref
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (41,))  # longer than the window, so the window bites
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = ours.forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-4


def test_paged_chunked_greedy_matches_reference(ref):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=3, max_model_len=256, max_prefill_tokens=8))
    g = torch.Generator().manual_seed(2)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (5, 19, 30)]
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=12, ignore_eos=True))
    for ids, r in zip(prompts, reqs):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=12, do_sample=False,
                               eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == want


def test_tp2_matches_tp1(tmp_path):
    """Hybrid layers shard differently: 2 full-attention KV heads (1 per rank) and 4
    sliding-window KV heads (2 per rank); sinks follow their query heads."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build_reference(str(tmp_path))
    kw = dict(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4, num_pages=128,
              max_num_seqs=2, max_model_len=128, max_prefill_tokens=8)
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**kw)).generate(prompts, sp)]
    two = LLMEngine(EngineConfig(tp=2, **kw))
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
    finally:
        two.close()
    assert got == want


def test_every_tensor_lands_on_the_target_device(ref):
    """The loader builds on meta and moves module by module; a module it forgets to move
    stays on the host and only fails at trace time on a real device (MiMo-V2.6-Flash tp=32,
    2026-10-02). Loading onto the meta device exposes it on CPU."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, _ = ref
    m = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("meta"), 512)
    assert all(t.device.type == "meta" for t in [*m.parameters(), *m.buffers()])


@pytest.mark.parametrize("spec", [None, "ngram"])
def test_window_sized_kv_reads_match_reference(ref, spec):
    """Sliding-window layers gather only the pages under their window (swa_table) once the
    context outgrows it: decode, chunked prefill and speculative verify all take that path
    here (window 8, pages of 4, contexts up to 100), and greedy output must still be the
    reference's token for token."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    kw = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
              max_num_seqs=2, max_model_len=256, max_prefill_tokens=16)
    if spec:
        kw.update(spec_method=spec, spec_k=3)
    eng = LLMEngine(EngineConfig(**kw))
    run = eng.runner
    assert run.window == 8
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (37, 70)]
    prompts[1][40:70] = prompts[1][10:40]  # repeats give the n-gram drafter something to propose
    reqs = eng.generate(prompts, SamplingParams(max_new_tokens=30, ignore_eos=True))

    def windowed(key):  # ("decode", B, P, ...), ("prefill", C, P, ...), ("verify", B, Q, P)
        queries, P = (key[2], key[3]) if key[0] == "verify" else (1 if key[0] == "decode" else key[1], key[2])
        return run._swa_pages(queries, P) > 0

    used = {k[0] for k in run.calls if windowed(k)}
    assert used >= ({"prefill", "verify"} if spec else {"prefill", "decode"}), used
    for ids, r in zip(prompts, reqs):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=30, do_sample=False,
                               eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        assert r.output_ids == want


def test_prompt_logprobs_match_reference(ref):
    """vLLM `prompt_logprobs` / SGLang `logprob_start_len`: every prompt position's logprob
    and rank under the reference model, across chunked prefill, with the prompt already in
    the prefix cache (which must not skip the rows that score it), and from a start offset."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, hf = ref
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=256, max_num_seqs=2, max_model_len=256, max_prefill_tokens=8))
    ids = torch.randint(0, 384, (30,), generator=torch.Generator().manual_seed(9)).tolist()
    with torch.no_grad():
        logp = torch.log_softmax(hf(torch.tensor([ids])).logits[0].double(), -1)
    want = {q: logp[q - 1, ids[q]].item() for q in range(1, len(ids))}
    rank = {q: int((logp[q - 1] > logp[q - 1, ids[q]]).sum()) + 1 for q in want}
    for start in (0, 0, 13):  # the second run finds the whole prompt cached
        sp = SamplingParams(max_new_tokens=2, ignore_eos=True, prompt_logprobs=3, prompt_logprobs_start=start)
        (r,) = eng.generate([ids], sp)
        got = r.prompt_logprobs
        assert sorted(got) == [q for q in want if q >= max(1, start)]
        for q, (lp, rk, top_ids, top_lps) in got.items():
            assert abs(lp - want[q]) < 1e-4 and rk == rank[q]
            assert top_ids == logp[q - 1].topk(3).indices.tolist() and len(top_lps) == 3


def test_piecewise_matches_whole_model(ref):
    """Piecewise execution (prep graph, one graph per layer kind, post graph) is the same
    computation as the whole-model forwards: greedy tokens, sampled tokens and logprobs agree
    through chunked prefill, window-sized reads, n-gram speculation and prompt logprobs."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, _ = ref
    g = torch.Generator().manual_seed(11)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (9, 45)]
    prompts[1][30:45] = prompts[1][5:20]
    runs = []
    for piecewise, group in ((False, None), (True, 1), (True, 3)):
        eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256,
                                     max_num_seqs=2, max_model_len=256, max_prefill_tokens=16, piecewise=piecewise,
                                     piecewise_group=group, spec_method="ngram", spec_k=3))
        a = eng.generate(prompts, SamplingParams(max_new_tokens=20, ignore_eos=True, logprobs=2, prompt_logprobs=1))
        b = eng.generate(prompts, SamplingParams(max_new_tokens=20, ignore_eos=True, temperature=0.8, seed=3))
        runs.append([(r.output_ids, r.logprobs, r.prompt_logprobs) for r in a] + [r.output_ids for r in b])
    assert runs[0] == runs[1] == runs[2]


def test_piecewise_traces_one_graph_per_layer_kind(ref):
    """Layers of one kind share a graph: 4 layers of 3 kinds trace 3 layer graphs, plus the
    decode prep and post graphs."""
    import numpy as np
    from torch._dynamo.testing import CompileCounter

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.model_runner import _piecewise
    from kiln.engine.scheduler import ScheduledSeq

    path, _ = ref
    eng = LLMEngine(EngineConfig(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=64,
                                 max_num_seqs=2, max_model_len=64, max_prefill_tokens=16))
    model, run = eng.model, eng.runner
    kinds = {model.layer_kind(i) for i in range(len(model.layers))}
    assert len(model.layers) == 4 and len(kinds) == 3
    torch._dynamo.reset()
    cnt = CompileCounter()
    run._decode, *_ = _piecewise(model, lambda f: torch.compile(f, backend=cnt, fullgraph=True, dynamic=False), 1)
    (r,) = eng.generate([[5, 6, 7, 8, 9, 10]], __import__("kiln.engine.request", fromlist=["SamplingParams"])
                        .SamplingParams(max_new_tokens=3, ignore_eos=True))
    assert len(r.output_ids) == 3
    assert cnt.frame_count == len(kinds) + 2, cnt.frame_count


def test_piecewise_gives_each_moe_layer_its_own_graph():
    """Dense layers are grouped up to the group size; a MoE layer is always a graph of its own
    (two attention + MoE layers in one graph ran 13x slower than two graphs, six 25x; see
    model_runner.piecewise_groups)."""
    from kiln.engine.model_runner import piecewise_groups

    def runs(*a):
        return [list(r) for r in piecewise_groups(*a)]

    assert runs(5, 2) == [[0, 1], [2, 3], [4]]  # dense: unchanged
    assert runs(6, 3, {2, 3}) == [[0, 1], [2], [3], [4, 5]]
    assert runs(4, 6, {1, 2, 3}) == [[0], [1], [2], [3]]  # MiMo-V2: dense layer 0, then MoE
    assert runs(3, None, {0, 1, 2}) == [[0], [1], [2]]
