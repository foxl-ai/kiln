"""GLM-5.3-Flash (`glm5_next`: KDA + NoPE DSA with a pooled indexer, mHC hyper-connections,
clamped SwiGLU, shared-expert MoE) against transformers' Glm5NextForConditionalGeneration, on CPU
in fp32 with identical weights. Needs transformers >= 5.18 (skipped below it).

Every model is built by transformers itself from the REAL config.json
(tests/reference/hybrid_configs/GLM-5.3-Flash.json, fetched 2026-10-03 from
https://huggingface.co/zai-org/GLM-5.3-Flash/raw/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json),
truncated to its first layers (3 KDA layers with dense MLPs, a DSA layer with MoE, KDA layers with
MoE, ...), with fewer heads and experts, a small hidden size and vocabulary, and everything else as
published: MLA geometry (q_lora 1536, kv_lora 512, 256 nope + 256 value dims, no RoPE), the
indexer (128-dim heads, kpool 4, tail selection), KDA (128-dim heads, lower bound -5), mHC (4
streams, 20 Sinkhorn iterations), swiglu_limit 10, routing (sigmoid, bias, scale 2.5, one shared
expert). The reference for generation is a full forward per step without a cache.
"""

import importlib.util
import json
import os

import pytest
import torch

mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")

CONFIG = os.path.join(os.path.dirname(__file__), "reference", "hybrid_configs", "GLM-5.3-Flash.json")
# The indexer keeps its 32 heads: with few heads a pool's score is exactly 0 (every head's ReLU
# zero) often enough to tie at the top-k boundary, where any tie-break is a valid top-k but
# transformers' and Kiln's differ (measured with 4 heads: 1 query in 70 at layer 3).
SMALL = dict(hidden_size=128, intermediate_size=256, moe_intermediate_size=128, num_attention_heads=4,
             num_key_value_heads=4, vocab_size=384, n_routed_experts=8, num_experts_per_tok=2,
             max_position_embeddings=1024, pad_token_id=None, eos_token_id=1, num_nextn_predict_layers=0)
VISION = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=128,
              projection_intermediate_size=64)


def text_config(layers=8, linear_heads=4, **extra):
    with open(CONFIG) as f:
        c = json.load(f)
    t = c["text_config"]
    t.pop("dtype", None)
    t.update(SMALL)
    t["linear_attn_config"] = dict(t["linear_attn_config"], num_heads=linear_heads)
    for k in ("layer_types", "mlp_layer_types", "indexer_types"):
        t[k] = t[k][:layers]
    t["num_hidden_layers"] = layers
    t.update(extra)
    return c, t


def randomize(model, seed=0):
    """Weights at scales that exercise every piece: unit-variance matmul outputs, SwiGLU inputs
    large enough that swiglu_limit clamps some of them, KDA decays from fast to slow."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".visual." in n or n.startswith("model.visual"):
                continue
            r = torch.randn(p.shape, generator=g)
            if n.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(-4.0, 1.0, generator=g))
            elif n.endswith("dt_bias"):
                p.copy_(r)
            elif "norm" in n and n.endswith("weight") or n.endswith("hc_attn.scale") or n.endswith("hc_ffn.scale") \
                    or n.endswith("_hc.scale"):
                p.copy_(1 + 0.1 * r)
            elif n.endswith("bias") or n.endswith(".base"):
                p.copy_((0.5 if "correction" in n else 0.1) * r)
            elif "embed_tokens" in n:
                p.copy_(r)
            elif "kpool_compress_ape" in n:
                p.copy_(0.5 * r)
            elif "mlp" in n and ("gate_up" in n or "gate_proj" in n or "up_proj" in n):
                p.copy_(r * 3 * p.shape[-1] ** -0.5)  # SwiGLU inputs ~N(0, 9): some cross +-10
            elif "mlp" in n and "down" in n:
                p.copy_(r * p.shape[-1] ** -0.5 / 3)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(0.5 * torch.randn(b.shape, generator=g))


def build(path, seed=0, layers=8, linear_heads=4, **extra):
    import transformers as tf

    c, t = text_config(layers, linear_heads, **extra)
    cfg = tf.Glm5NextConfig(text_config=t, vision_config=VISION)
    cfg._attn_implementation = "eager"
    cfg.text_config._attn_implementation = "eager"
    model = tf.Glm5NextForConditionalGeneration(cfg)
    randomize(model, seed)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def hf_greedy(model, ids, n):
    out = list(ids)
    with torch.no_grad():
        for _ in range(n):
            out.append(int(model(torch.tensor([out]), use_cache=False).logits[0, -1].argmax()))
    return out[len(ids):]


def engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=3,
                max_model_len=256, max_prefill_tokens=8)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def prompts(seed, lengths, vocab=384):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    path = tmp_path_factory.mktemp("glm5_next")
    return str(path), build(str(path))


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """index_topk 16: four pools of four keys, so prompts past 16 tokens are in the sparse regime."""
    path = tmp_path_factory.mktemp("glm5_next_sparse")
    return str(path), build(str(path), seed=1, index_topk=16)


def test_config_is_parsed(dense):
    from kiln.config import LinearSpec, ModelConfig
    from kiln.models.decoder import DecoderForCausalLM

    path, hf = dense
    cfg = ModelConfig.from_pretrained(path)
    hc = hf.config.text_config
    kinds = ["kda" if isinstance(s, LinearSpec) else "dsa" for s in cfg.attn_layers]
    assert kinds == ["kda", "kda", "kda", "dsa", "kda", "kda", "kda", "dsa"]
    assert list(cfg.moe_layers) == [i for i, l in enumerate(hf.model.language_model.layers) if hasattr(l.mlp, "experts")]
    kda = cfg.attn_layers[0]
    assert (kda.num_k_heads, kda.head_k_dim, kda.lower_bound, kda.gate_rank) == (4, 128, -5.0, 128)
    m = cfg.attn_layers[3].mla
    assert (m.q_lora_rank, m.kv_lora_rank, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim) == (1536, 512, 256, 0, 256)
    assert abs(m.softmax_scale - hf.model.language_model.layers[3].self_attn.scaling) < 1e-12
    assert m.norm_eps == hc.rms_norm_eps and (m.dsa.kpool, m.dsa.kpool_tail, m.dsa.head_dim) == (4, True, 128)
    hy = cfg.hybrid
    assert (hy.hc, hy.sinkhorn_iters, hy.swiglu_limit) == (4, 20, 10.0)
    assert cfg.routed_scaling_factor == 2.5 and cfg.n_shared_experts == 1 and cfg.router_scoring == "sigmoid"
    # The MLA cache: the latent, then the indexer key and its pool gate logits (no rope key).
    model = DecoderForCausalLM(cfg, torch.float32, 64)
    assert model.kv_shapes() == [((1, 512), (1, 256))] * 2


def test_logits_match(dense):
    """Dense regime (index_topk 2048): 70 tokens through the sequence form."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = dense
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 256)
    ids = torch.randint(0, 384, (70,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = ours.forward_logits(ids)
    err = (got - want).abs().max().item()
    print(f"glm5_next dense: logits max abs diff {err:.2e} (|logits| max {want.abs().max().item():.2f})")
    assert err < 1e-4


@pytest.mark.parametrize("mode", ["graph", "piecewise"])
def test_paged_chunked_greedy_matches(dense, mode):
    """Prompts of 1 to 61 tokens prefilled in chunks of 8, four requests for three running slots,
    batched decode; the KDA state and the MLA latent cache carried across chunks."""
    from kiln.engine.request import SamplingParams

    path, hf = dense
    eng = engine(path, piecewise=mode == "piecewise", piecewise_group=2)
    ps = prompts(1, (5, 19, 61, 1))
    want = [hf_greedy(hf, p, 10) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
    assert eng.pool.num_free + eng.radix.total_pages() == eng.pool.num_usable  # nothing leaked (the cache holds the rest)


def test_sparse_logits_and_greedy_match(sparse):
    """index_topk 16 (4 pools of 4): the pooled indexer selects per query, plus its tail."""
    from kiln.engine.request import SamplingParams

    path, hf = sparse
    eng = engine(path, max_prefill_tokens=16)
    ids = torch.randint(0, 384, (70,), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = eng.model.forward_logits(ids)
    err = (got - want).abs().max().item()
    print(f"glm5_next sparse: logits max abs diff {err:.2e}")
    assert err < 1e-4
    ps = prompts(2, (37, 70, 9))
    want = [hf_greedy(hf, p, 12) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=12, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


def test_sparse_nki_selection_matches(sparse, monkeypatch):
    """KILN_DSA_SELECT=nki (kernels/dsa_topk.py; on the host its emulation) through the whole model in
    the sparse regime: the same logits and greedy tokens as the reference."""
    from kiln.engine.request import SamplingParams
    from kiln.models import dsa_select

    monkeypatch.setattr(dsa_select, "SELECT", "nki")
    path, hf = sparse
    eng = engine(path, max_prefill_tokens=16)
    ids = torch.randint(0, 384, (70,), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = eng.model.forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-4
    ps = prompts(2, (37, 70, 9))
    want = [hf_greedy(hf, p, 12) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=12, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


def _pool_key_check(eng) -> int:
    """For every running request and DSA layer: each complete pool of the request's written tokens
    holds, in its kpool slots' pool-key pieces, the key glm5_next.pool_keys makes from those tokens'
    cached indexer rows (to the last ulp). Returns the pools checked."""
    from kiln.engine.request import Status
    from kiln.models.glm5_next import pool_keys

    n = 0
    ps = eng.runner.ps
    for layer in eng.model.kv_layers():
        if getattr(layer, "pool_key", None) is None:
            continue
        d = layer.spec.mla.dsa
        kp, Di = d.kpool, d.head_dim
        for r in eng.scheduler.running:
            if r.status != Status.RUNNING:
                continue
            for q in range(r.num_computed // kp):
                pos = [q * kp + i for i in range(kp)]
                slots = torch.tensor([r.pages[p // ps] * ps + p % ps for p in pos])
                want = pool_keys(layer, d, layer.v_cache[slots, 0, : 2 * Di])
                # The same arithmetic on a [kpool, 2 Di] block here and on [T, kpool, 2 Di] in the
                # engine: CPU kernels may round the softmax's exp differently per shape (one ulp).
                torch.testing.assert_close(layer.pool_key[slots].reshape(Di), want, rtol=1e-6, atol=1e-6,
                                           msg=lambda m, r=r, q=q: f"{r.rid} pool {q}: {m}")
                n += 1
    return n


def _inplace_check(inp, sep) -> int:
    """Two engines stepped in lockstep on the same requests (identical pages): every complete pool's
    last cached row in the inplace engine holds, in its indexer-key part, the separate engine's pool
    key for that pool, and every other row equals the separate engine's row (its own key and gates)."""
    from kiln.engine.request import Status

    n = 0
    ps = sep.runner.ps
    for li, ls in zip(inp.model.kv_layers(), sep.model.kv_layers()):
        if ls.pool_key is None:
            continue
        d = ls.spec.mla.dsa
        kp, Di = d.kpool, d.head_dim
        for ri, rs in zip(inp.scheduler.running, sep.scheduler.running):
            assert ri.pages == rs.pages and ri.num_computed == rs.num_computed
            if rs.status != Status.RUNNING:
                continue
            for q in range(rs.num_computed // kp):
                slots = torch.tensor([rs.pages[p // ps] * ps + p % ps for p in range(q * kp, q * kp + kp)])
                # The same function of the same rows (to the last ulp: in the separate form every token
                # of a pool in the call writes its pool's key, each from its own place in the batch).
                torch.testing.assert_close(li.v_cache[slots[-1], 0, :Di], ls.pool_key[slots].reshape(Di),
                                           rtol=1e-6, atol=1e-6)
                assert torch.equal(li.v_cache[slots[-1], 0, Di:], ls.v_cache[slots[-1], 0, Di:])
                assert torch.equal(li.v_cache[slots[:-1]], ls.v_cache[slots[:-1]])
                n += 1
    return n


@pytest.mark.parametrize("page_size,chunk", [(4, 16), (8, 6)])
def test_pool_key_cache(sparse, page_size, chunk, monkeypatch):
    """The pool-key cache (models/mla.py pool_cache_mode, KILN_DSA_POOL_CACHE): through chunked
    prefill (chunks of 6 cut pools of 4 in two) and batched decode every complete pool holds the key
    of its tokens' rows, in the separate form's pieces and in the inplace form's last row; and the
    three forms give the same tokens and logprobs as transformers' greedy path (off: rebuilt per
    call); pages of 4 (one pool) and of 8 (two). The default (auto) is inplace for this fp32 cache."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, hf = sparse
    ps = prompts(6, (37, 70, 9))
    p = SamplingParams(max_new_tokens=12, ignore_eos=True, logprobs=0)
    assert mla.POOL_CACHE == "auto" and mla.pool_cache_mode(False) == "inplace" and mla.pool_cache_mode(True) == "separate"
    inp = engine(path, page_size=page_size, max_prefill_tokens=chunk)
    assert all(l.pool_inplace and l.pool_key is None for l in inp.model.kv_layers())
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    sep = engine(path, page_size=page_size, max_prefill_tokens=chunk)
    assert all(l.pool_key is not None and not l.pool_inplace for l in sep.model.kv_layers())
    from kiln.models.decoder import state_bytes_per_token_rank

    assert state_bytes_per_token_rank(sep.mcfg, 1, torch.bfloat16) == 2 * 32 * 2  # 2 DSA layers x 32 x bf16
    monkeypatch.setattr(mla, "POOL_CACHE", "auto")
    assert state_bytes_per_token_rank(sep.mcfg, 1, torch.bfloat16) == 0  # inplace: no bytes
    monkeypatch.setattr(mla, "POOL_CACHE", "separate")
    reqs = [eng.add_request(x, p) for eng in (inp, sep) for x in ps]
    checked = inplace_checked = 0
    while sep.has_work():
        inp.step()
        sep.step()
        checked += _pool_key_check(sep)
        inplace_checked += _inplace_check(inp, sep)
    assert not inp.has_work()
    monkeypatch.setattr(mla, "POOL_CACHE", "off")
    off = engine(path, page_size=page_size, max_prefill_tokens=chunk)
    assert all(l.pool_key is None and not l.pool_inplace for l in off.model.kv_layers())
    want = off.generate(ps, p)
    n = len(ps)
    got_i, got_s = reqs[:n], reqs[n:]
    assert [r.output_ids for r in got_i] == [r.output_ids for r in got_s] == [r.output_ids for r in want] \
        == [hf_greedy(hf, x, 12) for x in ps]
    for a, b in [*zip(got_i, got_s), *zip(got_i, want)]:  # the same keys, to the last ulp (see the checks)
        assert torch.allclose(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                              rtol=0, atol=1e-5)
    print(f"pool-key cache: {checked} separate and {inplace_checked} inplace pool checks")
    assert checked > 0 and inplace_checked == checked


def test_pool_cache_separate_under_fp8_kv(sparse, monkeypatch):
    """auto keeps the pool-key cache as a separate bf16 state under an FP8 KV cache (an FP8 row would
    round the key, so not inplace), and serves the same tokens as rebuilding the keys (off)."""
    from kiln.engine.request import SamplingParams
    from kiln.models import mla

    path, _ = sparse
    ps, sp = prompts(7, (41,)), SamplingParams(max_new_tokens=6, ignore_eos=True)
    eng = engine(path, kv_cache_dtype="fp8")
    layers = eng.model.kv_layers()
    assert not any(l.pool_inplace for l in layers)
    assert any(l.pool_key is not None for l in layers)
    (got,) = eng.generate(ps, sp)
    monkeypatch.setattr(mla, "POOL_CACHE", "off")
    off = engine(path, kv_cache_dtype="fp8")
    assert all(l.pool_key is None and not l.pool_inplace for l in off.model.kv_layers())
    (want,) = off.generate(ps, sp)
    assert got.output_ids == want.output_ids


def test_tp2_matches_tp1(sparse):
    """KDA heads, MLA heads, the MLP and experts' intermediate split over two ranks; the latent,
    the indexer and the hyper-connections replicated."""
    from kiln.engine.request import SamplingParams

    path, _ = sparse
    ps = prompts(3, (11, 41))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in engine(path, max_num_seqs=2).generate(ps, sp)]
    two = engine(path, max_num_seqs=2, tp=2)
    try:
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want


def test_real_width_layers(tmp_path):
    """The first 4 layers (3 KDA + 1 DSA) at GLM-5.3-Flash's real widths: hidden 4096, 64 KDA heads,
    64 MLA heads (and the indexer's 32, as everywhere here). The MLP and expert widths, expert count and vocabulary are cut
    (about 2.4 GB per fp32 copy); they are not what this test is about."""
    from kiln.engine.request import SamplingParams

    hf = build(str(tmp_path), seed=11, layers=4, linear_heads=64, hidden_size=4096, num_attention_heads=64,
               num_key_value_heads=64, intermediate_size=256, moe_intermediate_size=128,
               vocab_size=512)
    ps = prompts(8, (70, 9), vocab=512)
    with torch.no_grad():
        want_logits = hf(torch.tensor([ps[0]])).logits[0]
    want = [hf_greedy(hf, p, 6) for p in ps]
    del hf
    eng = engine(str(tmp_path), max_prefill_tokens=32, max_num_seqs=2)
    with torch.no_grad():
        got = eng.model.forward_logits(torch.tensor(ps[0]))
    err = (got - want_logits).abs().max().item()
    print(f"GLM-5.3-Flash widths: logits max abs diff {err:.2e} (|logits| max {want_logits.abs().max().item():.2f})")
    assert err < 1e-3
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=6, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


def test_dsa_decode_kernel_path_matches(sparse, monkeypatch):
    """KILN_DSA_DECODE_KERNEL=nki: a decode step attends only the selected pools and its own tail pool, gathered as
    cache rows (glm5_next.decode_slots, kernels/dsa_decode.py; its emulation on the host), instead of the whole
    bucket under a mask: the reference's tokens, and logprobs equal to the mask path's."""
    from kiln.engine.request import SamplingParams
    from kiln.kernels import dsa_decode

    path, hf = sparse
    ps = prompts(2, (37, 70, 9))
    p = SamplingParams(max_new_tokens=12, ignore_eos=True, logprobs=0)
    monkeypatch.setattr(dsa_decode, "KERNEL", "xla")  # the mask path, whatever the host's default
    base = engine(path, max_prefill_tokens=16).generate(ps, p)
    calls = []
    real = dsa_decode.attend
    monkeypatch.setattr(dsa_decode, "KERNEL", "nki")
    monkeypatch.setattr(dsa_decode, "attend", lambda *a, **k: calls.append(tuple(a[2].shape)) or real(*a, **k))
    got = engine(path, max_prefill_tokens=16).generate(ps, p)
    assert calls and all(c[1] == dsa_decode.NCH * 128 for c in calls)  # the kernel path ran, NCH chunks of slots
    assert [r.output_ids for r in got] == [r.output_ids for r in base] == [hf_greedy(hf, x, 12) for x in ps]
    for a, b in zip(got, base):
        assert torch.allclose(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                              rtol=0, atol=1e-5)


def test_dsa_prefill_kernel_path_matches(sparse, monkeypatch):
    """KILN_DSA_PREFILL_KERNEL=nki: a sparse prefill chunk of the pooled DSA layers runs kernels/dsa_prefill.py (its
    emulation on the host) over the latent with the selection and visibility as its mask, instead of _core's expand
    form: the same tokens and logprobs (fp32, so only the order of the sums differs) for prompts cut into chunks of
    128 over page buckets of 128-key multiples."""
    from kiln.engine.request import SamplingParams
    from kiln.kernels import dsa_prefill

    from kiln.kernels import dsa_fused

    path, hf = sparse
    ps = prompts(5, (300, 140, 129))
    p = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=0)
    kw = dict(max_prefill_tokens=128, prefill_token_buckets=(128,), page_size=32, num_pages=64, max_model_len=512,
              page_buckets=(4, 8, 12, 16))
    monkeypatch.setattr(dsa_fused, "FUSED", False)  # on by default on a trn1 host; it would take these chunks first
    monkeypatch.setattr(dsa_prefill, "KERNEL", "xla")
    base = engine(path, **kw).generate(ps, p)
    calls = []
    real = dsa_prefill.attend
    monkeypatch.setattr(dsa_prefill, "KERNEL", "nki")
    monkeypatch.setattr(dsa_prefill, "attend", lambda *a, **k: calls.append((tuple(a[0].shape), tuple(a[1].shape)))
                        or real(*a, **k))
    got = engine(path, **kw).generate(ps, p)
    assert calls and all(q[0] == 128 and kc[0] % 128 == 0 for q, kc in calls)  # the kernel path ran (128-row chunks)
    assert [r.output_ids for r in got] == [r.output_ids for r in base]
    for a, b in zip(got, base):
        assert torch.allclose(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                              rtol=0, atol=1e-5)


def test_dsa_prefill_kernel_path_after_a_prefix_hit(sparse, monkeypatch):
    """The prefill kernel path on a chunk that resumes from the prefix cache off the 128-token grid (a shared system
    prompt of 170 tokens: the second request leaves a checkpoint at the junction, page 5 = token 160, and the third
    resumes there, so its first chunk starts at position 160): the same tokens and logprobs as the mask path."""
    from kiln.engine.request import SamplingParams
    from kiln.kernels import dsa_prefill

    path, _ = sparse
    (system,) = prompts(6, (170,))
    ps = [system + u for u in prompts(7, (40, 70, 55))]
    p = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=0)
    kw = dict(max_prefill_tokens=128, prefill_token_buckets=(128,), page_size=32, num_pages=64, max_model_len=512,
              page_buckets=(4, 8, 12, 16), state_track_interval=8)

    def run(e):
        return [e.generate([x], p)[0] for x in ps]  # one after the other: the second resumes from the first's prefix

    from kiln.kernels import dsa_fused

    monkeypatch.setattr(dsa_fused, "FUSED", False)
    monkeypatch.setattr(dsa_prefill, "KERNEL", "xla")
    base = run(engine(path, **kw))
    monkeypatch.setattr(dsa_prefill, "KERNEL", "nki")
    got = run(engine(path, **kw))
    hit = got[2].num_cached_tokens
    assert hit == base[2].num_cached_tokens == 160
    assert [r.output_ids for r in got] == [r.output_ids for r in base]
    for a, b in zip(got, base):
        assert torch.allclose(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                              rtol=0, atol=1e-5)


def test_dsa_prefill_emulation_is_the_masked_softmax():
    """kernels/dsa_prefill.emulate is softmax attention over the latent under the additive mask, absorbed: the expand
    form's values (q_nope . W_UK c) to fp32 rounding, every attended-key pattern including a row whose keys sit in one
    block."""
    from kiln.kernels import dsa_prefill as dp

    g = torch.Generator().manual_seed(3)
    C, H, R, DN, L = 256, 2, 512, 64, 640
    kc = torch.randn(L, R, generator=g)
    qn = torch.randn(C, H, DN, generator=g) * 0.2
    wk = torch.randn(H, DN, R, generator=g) * DN ** -0.5
    mask = torch.where(torch.rand(C, L, generator=g) < 0.3, 0.0, dp.NEG_INF)
    mask[0] = dp.NEG_INF
    mask[0, 600:604] = 0.0  # one row attends four keys of the last block only
    q_lat = torch.einsum("chd,hdr->chr", qn, wk)
    got = dp.emulate(q_lat, kc, mask, 0.125)
    k_nope = torch.einsum("lr,hdr->lhd", kc, wk)
    p = torch.softmax(torch.einsum("chd,lhd->hcl", qn, k_nope) * 0.125 + mask.unsqueeze(0), dim=-1)
    want = torch.einsum("hcl,lr->chr", p, kc)
    assert torch.allclose(got, want, rtol=0, atol=1e-4 * want.abs().max())


@pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
def test_dsa_decode_simulator_matches_emulation(monkeypatch):
    """kernels/dsa_decode.py in nki.simulate against emulate(): 2 rows, bf16 cache, 512 selected pools and a tail
    pool of 3 visible tokens per row, padding slots masked."""
    import nki

    from kiln.kernels import dsa_decode as dd

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    g = torch.Generator().manual_seed(4)
    B, H, R, PS, pages = 2, 8, 512, 32, 70
    kc = torch.randn(B * pages * PS, R, generator=g).to(torch.bfloat16)
    q_lat = (torch.randn(B, H, R, generator=g) * 0.05).to(torch.bfloat16)
    rows = torch.zeros(B, dd.NCH * 128, dtype=torch.long)
    bias = torch.full((B, dd.NCH * 128, dd.KP), dd.NEG_INF)
    for b in range(B):
        pools = torch.randperm(pages * PS // dd.KP - 1, generator=g)[:513]
        rows[b, :513] = b * pages * PS // dd.KP + pools  # pool rows
        bias[b, :512] = 0.0
        bias[b, 512, :3] = 0.0
    want = dd.emulate(q_lat, kc, rows, bias, 256 ** -0.5)
    kc4 = kc.reshape(-1, dd.KP * R)
    rows_t = rows.to(torch.int32).reshape(B, dd.NCH, 128).permute(2, 0, 1).contiguous()
    bias_t = bias.reshape(B, dd.NCH, 128, dd.KP).permute(0, 1, 3, 2).contiguous()
    got = torch.as_tensor(nki.simulate(dd.kiln_dsa_decode_kernel)(
        q_lat=q_lat, kc=kc4, rows_t=rows_t, bias_t=bias_t, identb=torch.eye(128).to(torch.bfloat16),
        scale=256 ** -0.5, fp8=0, rev=dd.REV)).float()
    assert ((got - want).abs().max() / want.abs().max()).item() < 5e-3  # p rounded before / after normalising


def test_dsa_prefill_loop_args_count_visible_pairs():
    """kernels/dsa_prefill.loop_args: per pass of QPASS query tiles, the number of 1024-key pairs holding a key at or
    before the pass's last position (by comparisons, no integer division), capped at the bucket's full pairs; tab
    holds each pair's first key and latent element."""
    from kiln.kernels import dsa_prefill as dp

    C, L, R = 1024, 8448, 512
    for off in (0, 1000, 1023, 1024, 3072, 7424):
        pos = torch.arange(C) + off
        npair, tab = dp.loop_args(pos, C, L, R)
        npass = C // 128 // dp.QPASS
        last = pos.view(npass, -1).amax(-1)
        assert npair.tolist() == [[min(int(x) // 1024 + 1, L // 1024) for x in last]]
    assert tab.tolist() == [[1024 * i, 1024 * i * R] for i in range(L // 1024)]


def test_dsa_fused_kernel_path_matches(sparse, monkeypatch):
    """KILN_DSA_FUSED=1: a sparse prefill chunk's indexer scores, selection and attention in one kernel
    (kernels/dsa_fused.py, its emulation on the host: dsa_topk's score and selection emulations, the visibility, then
    dsa_prefill's attention) instead of pooled_selection + the scratch + _core: the same tokens and logprobs, on the
    128-token chunks and a prefix hit at 160."""
    from kiln.engine.request import SamplingParams
    from kiln.kernels import dsa_fused

    path, _ = sparse
    (system,) = prompts(6, (170,))
    ps = [system + u for u in prompts(7, (40, 70, 55))] + prompts(5, (300,))
    p = SamplingParams(max_new_tokens=6, ignore_eos=True, logprobs=0)
    kw = dict(max_prefill_tokens=128, prefill_token_buckets=(128,), page_size=32, num_pages=64, max_model_len=512,
              page_buckets=(4, 8, 12, 16), state_track_interval=8)

    def run(e):
        return [e.generate([x], p)[0] for x in ps]

    monkeypatch.setattr(dsa_fused, "FUSED", False)
    base = run(engine(path, **kw))
    calls = []
    real = dsa_fused.attend
    monkeypatch.setattr(dsa_fused, "FUSED", True)
    monkeypatch.setattr(dsa_fused, "attend", lambda *a, **k: calls.append(tuple(a[0].shape)) or real(*a, **k))
    got = run(engine(path, **kw))
    assert calls and got[2].num_cached_tokens == 160
    assert [r.output_ids for r in got] == [r.output_ids for r in base]
    for a, b in zip(got, base):
        assert torch.allclose(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]),
                              rtol=0, atol=1e-5)


def test_decode_kernel_defaults_are_trn1_and_trn2(monkeypatch):
    """KILN_KDA_DECODE_KERNEL / KILN_DSA_DECODE_KERNEL / KILN_DECODE_SP unset: the decode kernels and the
    sequence-parallel decode streams on trn1 and trn2, where they were measured, off on inf2 and on a host without a
    Neuron device; each variable wins either way."""
    from kiln.kernels import dsa_decode, kda_decode
    from kiln.models.decoder import decode_sp_enabled

    monkeypatch.delenv("KILN_DECODE_SP", raising=False)
    for target, want in (("trn1", "nki"), ("trn2", "nki"), ("inf2", "xla")):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert kda_decode._default_kernel() == want, target
        assert dsa_decode._default_kernel() == want, target
        assert decode_sp_enabled() == (want == "nki"), target
    monkeypatch.setenv("KILN_DECODE_SP", "0")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    assert not decode_sp_enabled()


def test_dsa_fused_default_is_trn1_and_trn2(monkeypatch):
    """KILN_DSA_FUSED unset: the fused kernel on trn1 and trn2, where it was measured, off on inf2 and on a host without
    a Neuron device; the variable wins either way."""
    from kiln.kernels import dsa_fused as df

    monkeypatch.delenv("KILN_DSA_FUSED", raising=False)
    for target, want in (("trn1", "1"), ("trn2", "1"), ("inf2", "0")):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert df._default_fused() == want, target


def test_sp_gather_default_is_trn1_only(monkeypatch):
    """KILN_SP_GATHER unset: the world row gather as an NKI kernel's all_gather on trn1, where it was measured (G64
    156.1 -> 164.7 out tok/s), the zero-padded all-reduce on trn2 / inf2 and on a host without a Neuron device."""
    from kiln.kernels import sp_gather

    monkeypatch.delenv("KILN_SP_GATHER", raising=False)
    for target, want in (("trn1", "nki"), ("trn2", "xla"), ("inf2", "xla")):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert sp_gather._default_mode() == want, target


def test_sp_gather_kernel_needs_expert_parallel_experts():
    """The NKI world row gather (kernels/sp_gather.py) is for models whose routed experts are expert-parallel or that have
    none: with tensor-parallel experts neuronx-cc 2.27 fails the prefill pieces (NCC_ISCH719), so they keep the
    zero-padded all-reduce (DecoderForCausalLM._sp_gather_kernel_ok)."""
    from types import SimpleNamespace

    from kiln.models.decoder import DecoderForCausalLM

    def ok(moe_ep, experts):
        return DecoderForCausalLM._sp_gather_kernel_ok(SimpleNamespace(moe_ep=moe_ep, cfg=SimpleNamespace(
            num_experts=experts)))

    assert ok(True, 288) and ok(False, 0) and ok(False, None)
    assert not ok(False, 288)


def test_dsa_prefill_kernel_default_is_trn1_and_trn2(monkeypatch):
    """KILN_DSA_PREFILL_KERNEL unset: the kernel on trn1 and trn2 (where the fused kernel that needs it was measured), XLA
    on inf2 and on a host without a Neuron device; the variable wins either way."""
    from kiln.kernels import dsa_prefill as dp

    monkeypatch.delenv("KILN_DSA_PREFILL_KERNEL", raising=False)
    for target, want in (("trn1", "nki"), ("trn2", "nki"), ("inf2", "xla")):
        monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", target)
        assert dp._default_kernel() == want, target


def test_prefix_counts_matmul_form_is_exact(monkeypatch):
    """KILN_DSA_PREFIX=mm: decode_slots' inclusive count of selected pools as two triangular matmuls equals torch.cumsum
    bit for bit on 0 / 1 rows (2112 pools as at the G1 bucket, a P not divisible by 8, and one past MM_GROUPS groups,
    which falls back to cumsum)."""
    from kiln.models import glm5_next

    g = torch.Generator().manual_seed(0)
    for P in (2112, 2110, 8 * (glm5_next.MM_GROUPS + 1)):
        s01 = (torch.rand(3, P, generator=g) < 0.25).float()
        s01[1] = 0.0
        s01[2] = 1.0
        want = s01.cumsum(-1)
        monkeypatch.setattr(glm5_next, "PREFIX", "mm")
        assert torch.equal(glm5_next.prefix_counts(s01), want), P
        monkeypatch.setattr(glm5_next, "PREFIX", "xla")
        assert torch.equal(glm5_next.prefix_counts(s01), want), P


def test_dsa_decode_kernel_path_prefix_mm(sparse, monkeypatch):
    """The decode-kernel path through decode_slots with KILN_DSA_PREFIX=mm gives the cumsum form's tokens and logprobs."""
    from kiln.engine.request import SamplingParams
    from kiln.kernels import dsa_decode
    from kiln.models import glm5_next

    path, hf = sparse
    ps = prompts(2, (37, 70, 9))
    p = SamplingParams(max_new_tokens=12, ignore_eos=True, logprobs=0)
    monkeypatch.setattr(dsa_decode, "KERNEL", "nki")
    base = engine(path, max_prefill_tokens=16).generate(ps, p)
    monkeypatch.setattr(glm5_next, "PREFIX", "mm")
    got = engine(path, max_prefill_tokens=16).generate(ps, p)
    assert [r.output_ids for r in got] == [r.output_ids for r in base]
    for a, b in zip(got, base):
        assert torch.equal(torch.tensor([x[0] for x in a.logprobs]), torch.tensor([x[0] for x in b.logprobs]))
