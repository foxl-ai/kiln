"""EAGLE-3 drafts (spec_method "mtp" with spec_draft_model, models/eagle3.py).

The reference is vLLM v0.24.0's EAGLE-3 math written out in torch over whole sequences
(vllm/model_executor/models/llama_eagle3.py; the target side from models/llama.py and interfaces.py
EagleModelMixin): the target's residual streams after aux_layers layers, fc over their concatenation,
one Llama layer over cat(input_layernorm(embed(token after p)), hidden_norm(h)), the draft's final norm
and lm_head over its own vocabulary, d2t back to the target's ids; a later draft step feeds the previous
step's prenorm output back as h. The target is transformers' Qwen3 with output_hidden_states (entry n:
the stream after n layers, entry 0 the embedding).
"""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from tests.test_architectures import build

AUX = [0, 1, 2]


def make_draft(path, hf, draft_vocab=48, seed=7, norm_before_residual=True, aux=AUX):
    """A random EAGLE-3 draft for `hf` in the speculators format (RedHatAI/*-speculator.eagle3)."""
    c = hf.config
    H, I, nh, nkv, D, V = c.hidden_size, c.intermediate_size, c.num_attention_heads, c.num_key_value_heads, \
        c.head_dim, c.vocab_size
    theta = getattr(c, "rope_theta", None) or (getattr(c, "rope_parameters", None) or {}).get("rope_theta")
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g) * 0.08  # noqa: E731
    p = "layers.0."
    w = {"fc.weight": r(H, len(aux) * H), "norm.weight": 1 + r(H), "lm_head.weight": r(draft_vocab, H),
         "embed_tokens.weight": r(V, H), p + "input_layernorm.weight": 1 + r(H), p + "hidden_norm.weight": 1 + r(H),
         p + "post_attention_layernorm.weight": 1 + r(H), p + "self_attn.q_proj.weight": r(nh * D, 2 * H),
         p + "self_attn.k_proj.weight": r(nkv * D, 2 * H), p + "self_attn.v_proj.weight": r(nkv * D, 2 * H),
         p + "self_attn.o_proj.weight": r(H, nh * D), p + "mlp.gate_proj.weight": r(I, H),
         p + "mlp.up_proj.weight": r(I, H), p + "mlp.down_proj.weight": r(H, I)}
    targets = torch.randperm(V, generator=g)[:draft_vocab].sort().values
    w["d2t"] = targets - torch.arange(draft_vocab)
    w["t2d"] = torch.zeros(V, dtype=torch.bool).index_fill_(0, targets, True)
    os.makedirs(path, exist_ok=True)
    save_file({k: v.contiguous() for k, v in w.items()}, os.path.join(path, "model.safetensors"))
    cfg = {"architectures": ["Eagle3Speculator"], "draft_vocab_size": draft_vocab,
           "norm_before_residual": norm_before_residual, "eagle_aux_hidden_state_layer_ids": list(aux),
           "transformer_layer_config": {"hidden_size": H, "intermediate_size": I, "num_attention_heads": nh,
                                        "num_key_value_heads": nkv, "head_dim": D, "num_hidden_layers": 1,
                                        "rms_norm_eps": c.rms_norm_eps, "rope_theta": theta, "vocab_size": V,
                                        "max_position_embeddings": c.max_position_embeddings,
                                        "hidden_act": "silu", "model_type": "llama"}}
    json.dump(cfg, open(os.path.join(path, "config.json"), "w"))
    return w, theta


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, pos, theta):
    D = x.shape[-1]
    inv = 1.0 / theta ** (torch.arange(0, D, 2, dtype=torch.float32) / D)
    f = pos[:, None].float() * inv[None]
    emb = torch.cat([f, f], -1)
    cos, sin = emb.cos()[:, None], emb.sin()[:, None]
    rot = torch.cat([-x[..., D // 2:], x[..., : D // 2]], -1)
    return x * cos + rot * sin


def reference_drafts(hf, w, theta, tokens, k, norm_before_residual=True, aux=AUX):
    c = hf.config
    eps, nh, nkv, D = c.rms_norm_eps, c.num_attention_heads, c.num_key_value_heads, c.head_dim
    p = "layers.0."
    with torch.no_grad():
        hs = hf.model(torch.tensor([tokens[:-1]]), output_hidden_states=True).hidden_states
        fused = torch.cat([hs[a][0] for a in aux], -1) @ w["fc.weight"].T  # positions 0 .. N-1
        ids, hid, drafts = list(tokens[1:]), [h for h in fused], []
        for _ in range(k):
            e = w["embed_tokens.weight"][torch.tensor(ids)]
            h = torch.stack(hid)
            hn = rms(h, w[p + "hidden_norm.weight"], eps)
            residual = hn if norm_before_residual else h
            x = torch.cat([rms(e, w[p + "input_layernorm.weight"], eps), hn], -1)
            T = x.shape[0]
            q = (x @ w[p + "self_attn.q_proj.weight"].T).view(T, nh, D)
            kk = (x @ w[p + "self_attn.k_proj.weight"].T).view(T, nkv, D)
            v = (x @ w[p + "self_attn.v_proj.weight"].T).view(T, nkv, D)
            pos = torch.arange(T)
            q, kk = rope(q, pos, theta), rope(kk, pos, theta)
            kk, v = kk.repeat_interleave(nh // nkv, 1), v.repeat_interleave(nh // nkv, 1)
            s = torch.einsum("thd,shd->hts", q, kk) / D ** 0.5
            s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
            o = torch.einsum("hts,shd->thd", s.softmax(-1), v).reshape(T, nh * D) @ w[p + "self_attn.o_proj.weight"].T
            h1 = residual + o
            m = rms(h1, w[p + "post_attention_layernorm.weight"], eps)
            y = h1 + (torch.nn.functional.silu(m @ w[p + "mlp.gate_proj.weight"].T) * (m @ w[p + "mlp.up_proj.weight"].T)) \
                @ w[p + "mlp.down_proj.weight"].T
            dlog = rms(y[-1], w["norm.weight"], eps) @ w["lm_head.weight"].T
            di = int(dlog.argmax())
            t = di + int(w["d2t"][di])
            drafts.append(t)
            ids.append(t)
            hid.append(y[-1])
    return drafts


@pytest.fixture(scope="module")
def eagle_model(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("qwen3_target"))
    hf = build("qwen3", path)
    dpath = str(tmp_path_factory.mktemp("eagle3_draft"))
    w, theta = make_draft(dpath, hf)
    return path, dpath, hf, w, theta


@pytest.fixture(scope="module")
def eagle_llama3(tmp_path_factory):
    """A llama3-scaled target with a draft whose config has no rope_scaling, as Llama-3.1-8B-Instruct and
    RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3 (transformer_layer_config rope_scaling null)."""
    path = str(tmp_path_factory.mktemp("llama3_target"))
    hf = build("llama3_rope", path)
    dpath = str(tmp_path_factory.mktemp("eagle3_llama3_draft"))
    w, theta = make_draft(dpath, hf)
    return path, dpath, hf, w, theta


def _kw(path, dpath, **extra):
    return dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=2,
                max_model_len=256, max_prefill_tokens=8, spec_method="mtp", spec_draft_model=dpath, **extra)


def test_config_reads_the_speculators_format(eagle_model):
    from kiln.config import ModelConfig
    from kiln.models import eagle3

    path, dpath, hf, *_ = eagle_model
    d = eagle3.load_config(dpath, ModelConfig.from_pretrained(path))
    assert d.aux_layers == tuple(AUX) and d.draft_vocab_size == 48 and d.prefix == "layers.0."
    assert d.has_embed and d.norm_before_residual and not d.layer.qk_norm


@pytest.mark.parametrize("piecewise", [False, True])
def test_drafts_match_the_reference_after_every_step(eagle_model, piecewise):
    """Chunked prefill, then verify steps: after each step every running request's drafts equal vLLM's EAGLE-3
    math over the whole sequence."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, dpath, hf, w, theta = eagle_model
    eng = LLMEngine(EngineConfig(**_kw(path, dpath, spec_k=2, piecewise=piecewise)))
    g = torch.Generator().manual_seed(3)
    reqs = [eng.add_request(torch.randint(0, 384, (n,), generator=g).tolist(),
                            SamplingParams(max_new_tokens=14, ignore_eos=True)) for n in (11, 23)]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft:
                assert r.mtp_draft == reference_drafts(hf, w, theta, r.token_ids, 2), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8


def test_draft_keeps_its_own_rope_under_a_llama3_target(eagle_llama3):
    """The draft layer rotates with its own config's RoPE (plain theta here), the target layers with llama3 scaling:
    vLLM llama_eagle3.py builds the draft's rotary from the draft config."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, dpath, hf, w, theta = eagle_llama3
    eng = LLMEngine(EngineConfig(**_kw(path, dpath, spec_k=2)))
    m = eng.runner.model
    assert m.mtp.rope != m.layers[0].rope
    g = torch.Generator().manual_seed(5)
    reqs = [eng.add_request(torch.randint(0, 384, (n,), generator=g).tolist(),
                            SamplingParams(max_new_tokens=14, ignore_eos=True)) for n in (13, 21)]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft:
                assert r.mtp_draft == reference_drafts(hf, w, theta, r.token_ids, 2), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8


def test_eagle3_speculation_leaves_greedy_output_unchanged(eagle_model):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, dpath, *_ = eagle_model
    kw = _kw(path, dpath)
    base = {k: v for k, v in kw.items() if k not in ("spec_method", "spec_draft_model")}
    prompts = [[5, 9, 11, 200, 3, 77, 12], list(range(40, 75))]
    sp = SamplingParams(max_new_tokens=20, ignore_eos=True)
    want = [r.output_ids for r in LLMEngine(EngineConfig(**base)).generate(prompts, sp)]
    spec = LLMEngine(EngineConfig(spec_k=3, **kw))
    assert [r.output_ids for r in spec.generate(prompts, sp)] == want
    assert spec.spec_proposed > 0


def test_eagle3_under_tensor_parallelism_matches_tp1(eagle_model):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, dpath, *_ = eagle_model
    kw = _kw(path, dpath, spec_k=2)
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


@pytest.mark.parametrize("k", [1, 3])
def test_eagle3_async_equals_sync(eagle_model, k):
    """EAGLE-3 under overlap scheduling (spec_async, engine/spec_async.py; k 3: the later draft passes' positions from
    mtp_prep): the blind engine's tokens and logprobs equal the synchronous engine's, with at least its accepted
    drafts (a blind step always carries k)."""
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path, dpath, *_ = eagle_model
    prompts = [[5, 9, 11, 200, 3, 77, 12], list(range(40, 75)), [7] * 13]
    sp = SamplingParams(max_new_tokens=18, ignore_eos=True, logprobs=1)
    runs = []
    for extra in (dict(), dict(overlap=True, spec_async=True)):
        eng = LLMEngine(EngineConfig(**_kw(path, dpath, spec_k=k, **extra)))
        assert eng.runner.spec_async == bool(extra)
        reqs = eng.generate(prompts, sp)
        runs.append(([r.output_ids for r in reqs], [[x[0] for x in r.logprobs] for r in reqs],
                     (eng.spec_proposed, eng.spec_accepted)))
    (ids0, lp0, acc0), (ids, lp, acc) = runs
    assert ids == ids0
    assert max(abs(x - y) for p, q in zip(lp, lp0) for x, y in zip(p, q)) < 1e-4
    assert acc0[0] > 0 and acc[1] >= acc0[1]


def test_aux_cuts_and_prefill_cuts_both_bound_the_prefill_runs(tmp_path, monkeypatch):
    """An 8-layer target in runs of 4 with EAGLE-3 auxiliary streams after 2 and 5 layers and KILN_PIECEWISE_PREFILL_CUTS
    3,6: the prefill plan is cut at both sets of bounds (2, 3, 5, 6), the decode plan at the auxiliary ones only, and
    the drafts still equal the reference."""
    import transformers as tf

    from kiln.config import EngineConfig
    from kiln.engine import model_runner
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    path = str(tmp_path / "target")
    cfg = tf.Qwen3Config(vocab_size=384, hidden_size=64, intermediate_size=96, num_hidden_layers=8,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512,
                         rms_norm_eps=1e-6, tie_word_embeddings=False)
    torch.manual_seed(0)
    hf = tf.Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in hf.parameters():
            p.normal_(0, 0.08)
    hf.save_pretrained(path)
    aux = (0, 2, 5)
    w, theta = make_draft(str(tmp_path / "draft"), hf, aux=aux)
    monkeypatch.setattr(model_runner, "PREFILL_CUTS", (3, 6))
    eng = LLMEngine(EngineConfig(**_kw(path, str(tmp_path / "draft"), spec_k=2, piecewise=True, piecewise_group=4)))
    assert eng.runner._prefill.plan_len == [2, 1, 1, 1, 1, 2]
    assert eng.runner._decode.plan_len == [2, 2, 1, 3]
    g = torch.Generator().manual_seed(3)
    reqs = [eng.add_request(torch.randint(0, 384, (n,), generator=g).tolist(),
                            SamplingParams(max_new_tokens=10, ignore_eos=True)) for n in (11, 23)]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft:
                assert r.mtp_draft == reference_drafts(hf, w, theta, r.token_ids, 2, aux=aux), (r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 6
