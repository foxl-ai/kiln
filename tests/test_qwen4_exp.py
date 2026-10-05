"""Qwen3.8-Flash-Next (`qwen4_exp`: Gated DeltaNet + Qwen Sparse Attention, gated-residual
hyper-connections, a Per-Layer Embedding of hashed n-grams, MoE with a sigmoid-gated shared
expert) against transformers' Qwen4ExpForCausalLM, on CPU in fp32 with identical weights. Needs
transformers >= 5.18 (skipped below it).

Every model is built by transformers itself from the REAL config.json's text_config
(tests/reference/hybrid_configs/Qwen3.8-Flash-Next.json, fetched 2026-10-03 from
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json),
truncated to its first layers (GDN, the PLE layer, GDN, QSA, ...), with fewer heads and experts, a
small hidden size, vocabulary and n-gram table (ngram_vocab_size_base 1000), and everything else
as published: attention geometry (head 256, partial RoPE 64, theta 1e7, gated output), the QSA
indexer (4 heads x 128, blocks of 4, one key head), GDN head dims (128), the GDN gate (sigmoid),
the hyper-connections (4 streams), the PLE (bigrams and trigrams, 8 heads each, dilated conv,
seed), the routing (softmax top-10, normalised). The reference for generation is a full forward
per step without a cache. EOS tokens inside the prompts exercise the n-gram reset.
"""

import json
import os

import numpy as np
import pytest
import torch

mod = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")

CONFIG = os.path.join(os.path.dirname(__file__), "reference", "hybrid_configs", "Qwen3.8-Flash-Next.json")
EOS = 1
SMALL = dict(hidden_size=128, num_attention_heads=4, num_key_value_heads=2, linear_num_key_heads=2,
             linear_num_value_heads=6, vocab_size=384, num_experts=16, moe_intermediate_size=64,
             shared_expert_intermediate_size=64, hc_lowrank=32, ple_embed_dim=128, ngram_vocab_size_base=1000,
             split_ngram_parts=2, eos_token_id=EOS, bos_token_id=EOS, mtp_num_hidden_layers=0,
             max_position_embeddings=1024)


def text_config(layers=8, **extra):
    with open(CONFIG) as f:
        t = json.load(f)["text_config"]
    for k in ("dtype", "mtp"):
        t.pop(k, None)
    t.update(SMALL)
    t["layer_types"] = t["layer_types"][:layers]
    t["num_hidden_layers"] = layers
    t.update(extra)
    return t


def randomize(model, seed=0):
    """Unit-variance matmul outputs; zero-centred norms near 1; the PLE conv (zero-initialised
    by transformers) and every gate non-trivial; GDN decays from fast to slow."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if n.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(-4.0, 1.0, generator=g))
            elif n.endswith("dt_bias"):
                p.copy_(r)
            elif n.endswith("linear_attn.norm.weight"):  # Qwen4ExpTextRMSNormGated: plain weight
                p.copy_(1 + 0.1 * r)
            elif "norm" in n:  # Qwen4ExpTextRMSNorm: (1 + w)
                p.copy_(0.1 * r)
            elif "embed" in n:
                p.copy_(r)
            elif "ple.conv1d" in n:
                p.copy_(0.3 * r)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)


def build(path, seed=0, layers=8, **extra):
    import transformers as tf

    cfg = tf.Qwen4ExpTextConfig(**text_config(layers, **extra))
    cfg._attn_implementation = "eager"
    model = tf.Qwen4ExpForCausalLM(cfg)
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


def prompts(seed, lengths, vocab=384, eos_every=13):
    """Random ids with an EOS now and then (the n-gram hash resets after it)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in lengths:
        p = torch.randint(2, vocab, (n,), generator=g)
        p[eos_every - 1 :: eos_every] = EOS
        out.append(p.tolist())
    return out


@pytest.fixture(scope="module")
def dense(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen4_exp")
    return str(path), build(str(path))


@pytest.fixture(scope="module")
def sparse(tmp_path_factory):
    """indexer_budget 16: four blocks of four tokens, so prompts past 16 tokens are sparse. The
    indexer gets 16 heads instead of 4: with 4, a block's score (a sum of 4 ReLUs) is exactly 0 for
    about 1 block in 16, and a query whose 4th-best block ties at 0 keeps whichever tied block its
    top-k picks: measured on this model, query 20 of a 70-token prompt scored its 5 blocks
    [1.44, 1.14, 0, 0, 0.49], transformers kept block 3 and Kiln block 2. Both are valid top-ks;
    the test needs scores without exact ties."""
    path = tmp_path_factory.mktemp("qwen4_exp_sparse")
    return str(path), build(str(path), seed=1, indexer_budget=16, indexer_n_heads=16)


def test_ngram_ids_match_the_reference(dense):
    """The host hash (qwen4_exp.ngram_ids) against Qwen4ExpTextNGramEmbedding's own ids, for a whole
    sequence and for every split of it into two calls (the cached path's history window)."""
    from kiln.config import ModelConfig
    from kiln.models.qwen4_exp import ngram_ids

    path, hf = dense
    spec = ModelConfig.from_pretrained(path).hybrid.ple
    emb = hf.model.layers[1].ple.ple_embedding
    assert list(spec.head_vocab[0]) == emb.head_vocab_sizes and list(spec.head_offset[0]) == emb.head_offsets
    assert list(spec.multipliers[0]) == emb.layer_multipliers.tolist() and spec.rows[0] == emb.ngram_embedding.num_embeddings
    seen = {}
    emb.ngram_embedding.register_forward_hook(lambda m, a, o: seen.update(ids=a[0]))
    for (toks,) in (prompts(3, (40,)), [[EOS, EOS, 5, EOS, 7, 8, 9, EOS] + list(range(10, 30))]):
        with torch.no_grad():
            emb(torch.tensor([toks]), None)
        want = seen["ids"][0].numpy()
        assert np.array_equal(ngram_ids(spec, toks, 0, len(toks)), want)
        for cut in range(1, len(toks)):
            assert np.array_equal(ngram_ids(spec, toks, cut, len(toks)), want[cut:])


def test_config_is_parsed(dense):
    from kiln.config import LinearSpec, ModelConfig
    from kiln.models.qwen4_exp import QSASpec

    path, hf = dense
    cfg = ModelConfig.from_pretrained(path)
    kinds = ["gdn" if isinstance(s, LinearSpec) else "qsa" for s in cfg.attn_layers]
    assert kinds == ["gdn", "gdn", "gdn", "qsa"] * 2
    q = cfg.attn_layers[3]
    assert isinstance(q, QSASpec) and (q.head_dim, q.rope_dim, q.index_heads, q.index_head_dim, q.budget, q.compress) == (
        256, 64, 4, 128, 2048, 4)
    assert cfg.attn_layers[0].gate_act == "sigmoid" and cfg.moe_layers == tuple(range(8))
    hy = cfg.hybrid
    assert (hy.hc, hy.lowrank, hy.shared_expert, hy.block_norms) == (4, 32, 64, False)
    assert hy.ple.layers == (1,) and hy.ple.heads == 16 and hy.ple.head_dim == 8 and hy.ple.conv_state_len == 9


def test_logits_match(dense):
    """Dense regime (indexer_budget 2048): 70 tokens through the sequence form."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = dense
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 256)
    (ids,) = prompts(4, (70,))
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0]
        got = ours.forward_logits(torch.tensor(ids))
    err = (got - want).abs().max().item()
    print(f"qwen4_exp dense: logits max abs diff {err:.2e} (|logits| max {want.abs().max().item():.2f})")
    assert err < 1e-4


@pytest.mark.parametrize("mode", ["graph", "piecewise"])
def test_paged_chunked_greedy_matches(dense, mode):
    """Prompts of 1 to 61 tokens prefilled in chunks of 8 (GDN state, PLE conv history and n-gram
    window carried across chunks), four requests for three running slots, batched decode."""
    from kiln.engine.request import SamplingParams

    path, hf = dense
    eng = engine(path, piecewise=mode == "piecewise", piecewise_group=2)
    ps = prompts(1, (5, 19, 61, 1))
    want = [hf_greedy(hf, p, 10) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
    assert eng.pool.num_free + eng.radix.total_pages() == eng.pool.num_usable  # nothing leaked (the cache holds the rest)


def test_overlap_falls_back_to_sync(dense):
    """overlap=True runs these models synchronously (the n-gram ids need every token on the host)."""
    from kiln.engine.request import SamplingParams

    path, hf = dense
    ps = prompts(6, (12, 7))
    reqs = engine(path, overlap=True).generate(ps, SamplingParams(max_new_tokens=8, ignore_eos=True))
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 8) for p in ps]


def test_sparse_logits_and_greedy_match(sparse):
    """indexer_budget 16 (4 blocks of 4): every query selects its blocks, plus its tail."""
    from kiln.engine.request import SamplingParams

    path, hf = sparse
    eng = engine(path, max_prefill_tokens=16)
    (ids,) = prompts(5, (70,))
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0]
        got = eng.model.forward_logits(torch.tensor(ids))
    err = (got - want).abs().max().item()
    print(f"qwen4_exp sparse: logits max abs diff {err:.2e}")
    assert err < 1e-4
    ps = prompts(2, (37, 70, 9))
    want = [hf_greedy(hf, p, 12) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=12, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


def test_tp2_matches_tp1(sparse):
    """GDN and attention heads, the expert and shared-expert intermediate, and the n-gram tables'
    columns (with key_proj / value_proj's input columns) split over two ranks."""
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
    """The first 4 layers (GDN, GDN + PLE, GDN, QSA) at Qwen3.8-Flash-Next's real widths: hidden
    2560, GDN 16 k / 48 v heads, attention 24 / 2 heads, hc_lowrank 320, PLE 16 heads x 160. Expert
    count and widths, vocabulary and the n-gram table are cut (about 1.4 GB per fp32 copy). Seed 11
    put one token's 10th and 11th of 16 experts at layer 1 within 1e-6 of each other, a routing tie
    that fp32 summation order decides (transformers and Kiln took different experts, logits 0.03
    apart, while every block matched to 4e-6 on the same input)."""
    from kiln.engine.request import SamplingParams

    hf = build(str(tmp_path), seed=12, layers=4, hidden_size=2560, num_attention_heads=24, num_key_value_heads=2,
               linear_num_key_heads=16, linear_num_value_heads=48, hc_lowrank=320, ple_embed_dim=2560,
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
    print(f"Qwen3.8-Flash-Next widths: logits max abs diff {err:.2e} (|logits| max {want_logits.abs().max().item():.2f})")
    assert err < 1e-3
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=6, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


@pytest.mark.parametrize("seed", range(4))
def test_block_mask_bisect_equals_topk(seed):
    """glm5_next.block_mask's two selections (QSA uses the bisection) agree on distinct scores,
    for queries that see fewer blocks than they keep, more, and none complete yet."""
    from kiln.models.decoder import NEG_INF
    from kiln.models.glm5_next import block_mask

    g = torch.Generator().manual_seed(seed)
    B, Q, L, kp, keep = 3, 40, 160, 4, 6
    index = torch.randn(B, Q, L // kp, generator=g) * (10.0 ** seed)
    nvis = torch.randint(1, L + 1, (B, Q), generator=g)
    vis = torch.where(torch.arange(L) < nvis.unsqueeze(-1), 0.0, NEG_INF)
    a = block_mask(index, vis, kp, keep, True, select="topk")
    b = block_mask(index, vis, kp, keep, True, select="range")
    assert torch.equal(a + vis > -1, b + vis > -1)
    # Exact ties (QSA's relu sums are often exactly 0): still `keep` complete blocks, the first ones.
    index = torch.relu(index) * (torch.rand(index.shape, generator=g) < 0.3)
    b = block_mask(index, vis, kp, keep, False, select="range") + vis > -1
    complete = (nvis // kp).unsqueeze(-1)
    assert torch.equal(b.view(B, Q, -1, kp).all(-1).sum(-1), torch.minimum(complete, torch.tensor(keep)).squeeze(-1))
