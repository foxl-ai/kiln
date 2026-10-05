"""Linear-attention hybrids (models/linear_attn.py, engine/state_pool.py) against their Hugging
Face transformers references, on CPU in fp32 with identical weights.

- Gated DeltaNet + gated attention: transformers qwen3_5 (v5.15+), tiny random models built by
  transformers itself; Qwen3.8-27B-shaped layers (whole model, truncated).
- Qwen3.8-Flash-Next's GDN layer (qwen4_exp), Kimi Delta Attention (kimi_linear, glm5_next) and a
  KDA-only Kimi Linear model need transformers >= 5.18 and are skipped below it.
- The real Qwen3.5-0.8B checkpoint runs when KILN_TEST_LINEAR_MODEL is set (downloads 1.7 GB).

Through the engine: paged, chunked prefill (several chunks per prompt), batched decode,
piecewise graphs, overlap scheduling, preemption, state-row reuse with poisoned rows, tp=2,
prompt logprobs. The serving features (prefix cache from state checkpoints, speculation, host
tier) are tests/test_linear_serving.py.
"""

import os

import pytest
import torch

# Real model configs (config.json at the cited revision), the fields the layer shapes use.
# https://huggingface.co/Qwen/Qwen3.8-27B/raw/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json
QWEN3_8_27B = dict(hidden_size=5120, intermediate_size=17408, num_attention_heads=24, num_key_value_heads=4,
                   head_dim=256, linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
                   linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, vocab_size=248320,
                   rope_parameters={"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                                    "mrope_section": [11, 11, 10], "mrope_interleaved": True})
# https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json
QWEN3_8_FLASH_NEXT = dict(hidden_size=2560, linear_num_key_heads=16, linear_num_value_heads=48, linear_key_head_dim=128,
                          linear_value_head_dim=128, linear_conv_kernel_dim=4, rms_norm_eps=1e-6, output_gate_type="sigmoid",
                          hidden_act="silu")
# https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct/raw/e1df551a447157d4658b573f9a695d57658590e9/config.json
KIMI_LINEAR_48B = dict(hidden_size=2304, rms_norm_eps=1e-5,
                       linear_attn_config={"num_heads": 32, "head_dim": 128, "short_conv_kernel_size": 4})
# https://huggingface.co/zai-org/GLM-5.3-Flash/raw/eb9eb208eb0d988989d07a6a12d0fdeb5f52574a/config.json
GLM_5_3_FLASH = dict(hidden_size=4096, rms_norm_eps=1e-5,
                     linear_attn_config={"num_heads": 64, "gate_lower_bound": -5.0, "head_dim": 128,
                                         "short_conv_kernel_size": 4})
# https://huggingface.co/moonshotai/Kimi-K3/raw/f831ab66814297da540d832a5235f8e904f29d06/config.json
KIMI_K3 = dict(hidden_size=7168, rms_norm_eps=1e-5,
               linear_attn_config={"num_heads": 96, "head_dim": 128, "short_conv_kernel_size": 4,
                                   "gate_lower_bound": -5.0, "use_full_rank_gate": True})
K3_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"


def randomize(model, seed=0):
    """Weights at scales that exercise the recurrence: some heads keep memory for many
    tokens (small decay), others forget within a few."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.empty(p.shape).uniform_(-4.0, 1.0, generator=g))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g))
            elif "norm" in name:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3 + (1.0 if p.mean().item() > 0.5 else 0.0))
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.08)


def build_qwen3_5(path, layer_types=("linear_attention", "linear_attention", "full_attention", "linear_attention"),
                  tie=False, seed=0, **shape):
    import transformers as tf

    kw = dict(vocab_size=384, hidden_size=64, intermediate_size=96, num_attention_heads=4, num_key_value_heads=2,
              head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
              linear_value_head_dim=24, linear_conv_kernel_dim=4, rms_norm_eps=1e-6,
              rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25})
    kw.update(shape)
    cfg = tf.Qwen3_5TextConfig(num_hidden_layers=len(layer_types), layer_types=list(layer_types),
                               max_position_embeddings=512, tie_word_embeddings=tie, **kw)
    cfg._attn_implementation = "eager"
    model = tf.Qwen3_5ForCausalLM(cfg)
    randomize(model, seed)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def hf_greedy(model, ids, n, use_cache=True):
    """An explicit all-ones mask: without one, generate() masks every prompt token equal to
    pad_token_id as padding (a random prompt holding token 0 then diverges from forward())."""
    x = torch.tensor([ids])
    with torch.no_grad():
        out = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=n, do_sample=False,
                             eos_token_id=None, pad_token_id=0, use_cache=use_cache)
    return out[0, len(ids):].tolist()


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
def gdn(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_5")
    return str(path), build_qwen3_5(str(path))


# -- the chunked form against the recurrence ------------------------------------------------


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("chunk", [4, 16, 64])
def test_chunk_scan_matches_the_recurrence(per_channel, chunk):
    """chunk_scan (sub-chunks, (I - N)^-1 by repeated squaring, padding) equals the gated delta
    rule token by token, from a non-zero state, for per-head (GDN) and per-channel (KDA) decay."""
    from kiln.models.linear_attn import chunk_scan, recurrent_step

    g = torch.Generator().manual_seed(3)
    T, H, Dk, Dv = 37, 3, 8, 5
    q = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1) * Dk ** -0.5
    k = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1)
    v = torch.randn(T, H, Dv, generator=g)
    gate = -torch.rand((T, H, Dk) if per_channel else (T, H), generator=g) * 3
    beta = torch.rand(T, H, generator=g)
    S0 = torch.randn(H, Dk, Dv, generator=g)
    S, want = S0.unsqueeze(0), []
    for t in range(T):
        o, S = recurrent_step(q[t : t + 1], k[t : t + 1], v[t : t + 1], gate[t : t + 1], beta[t : t + 1], S)
        want.append(o)
    got, S_got = chunk_scan(q, k, v, gate, beta, S0, chunk)
    assert (got - torch.cat(want)).abs().max().item() < 1e-4
    assert (S_got - S[0]).abs().max().item() < 1e-4


@pytest.mark.parametrize("per_channel", [False, True])
@pytest.mark.parametrize("chunk", [16, 32, 64])
def test_chunk_scan_is_stable_on_correlated_keys(per_channel, chunk):
    """Nearly parallel keys, beta near 1 and slow decay, the regime of trained weights: the
    sub-chunk's triangular system (I + A) has an inverse of modest size whose Neumann terms are
    huge. Inverting it by repeated squaring over the whole sub-chunk was off by 1.2e3 (32 tokens)
    and 7.6e18 (64) here; the blocked inverse (linear_attn._unit_lower_inverse) is exact to
    rounding. With Qwen3.5-0.8B's real weights the old form broke every prefill sub-chunk past 16
    tokens (layer 6 of a 63-token prompt: 9.5e7 at 64)."""
    from kiln.models.linear_attn import chunk_scan, recurrent_step

    g = torch.Generator().manual_seed(0)
    T, H, Dk, Dv = 128, 2, 32, 16
    base = torch.randn(H, Dk, generator=g)
    k = torch.nn.functional.normalize(base + 0.05 * torch.randn(T, H, Dk, generator=g), dim=-1)
    q = torch.nn.functional.normalize(torch.randn(T, H, Dk, generator=g), dim=-1) * Dk ** -0.5
    v = torch.randn(T, H, Dv, generator=g)
    gate = -0.01 * torch.rand((T, H, Dk) if per_channel else (T, H), generator=g)
    beta = 0.9 + 0.1 * torch.rand(T, H, generator=g)
    S, want = torch.zeros(1, H, Dk, Dv, dtype=torch.float64), []
    for t in range(T):
        o, S = recurrent_step(*(x[t : t + 1].double() for x in (q, k, v, gate, beta)), S)
        want.append(o)
    got, S_got = chunk_scan(q, k, v, gate, beta, torch.zeros(H, Dk, Dv), chunk)
    assert (got - torch.cat(want).float()).abs().max().item() < 1e-5
    assert (S_got - S[0].float()).abs().max().item() < 1e-4


# -- Gated DeltaNet hybrid (Qwen3.5 architecture), tiny ---------------------------------------


def test_config_is_parsed_as_hybrid(gdn):
    from kiln.config import AttnSpec, LinearSpec, ModelConfig

    path, _ = gdn
    cfg = ModelConfig.from_pretrained(path)
    kinds = [type(s) for s in cfg.attn_layers]
    assert kinds == [LinearSpec, LinearSpec, AttnSpec, LinearSpec]
    lin, full = cfg.attn_layers[0], cfg.attn_layers[2]
    assert (lin.kind, lin.num_k_heads, lin.num_v_heads, lin.head_k_dim, lin.head_v_dim) == ("gdn", 2, 4, 16, 24)
    assert full.rope_dim == 8 and cfg.attn_output_gate and cfg.norm_offset and cfg.qk_norm


def test_logits_match(gdn):
    """Three sub-chunks of 64 plus padding through the sequence form."""
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    path, hf = gdn
    ours = load_model(path, ModelConfig.from_pretrained(path), torch.float32, torch.device("cpu"), 512)
    ids = torch.randint(0, 384, (150,), generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        want = hf(ids.unsqueeze(0)).logits[0]
        got = ours.forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-4


@pytest.mark.parametrize("mode", ["graph", "piecewise", "overlap"])
def test_paged_chunked_greedy_matches(gdn, mode):
    """Prompts of 1 to 61 tokens prefilled in chunks of 8 (state carried across chunks),
    four requests for three running slots, then batched decode; the second batch reuses the
    state rows of the first. No prefix reuse, and every page is freed at the end."""
    from kiln.engine.request import SamplingParams

    path, hf = gdn
    kw = dict(piecewise=mode == "piecewise", piecewise_group=2, overlap=mode == "overlap")
    eng = engine(path, prefix_caching=False, **kw)
    ps = prompts(1, (5, 19, 61, 1))
    want = [hf_greedy(hf, p, 12) for p in ps]
    for _ in range(2):
        reqs = eng.generate(ps, SamplingParams(max_new_tokens=12, ignore_eos=True))
        assert [r.output_ids for r in reqs] == want
        assert [r.num_cached_tokens for r in reqs] == [0, 0, 0, 0]
    assert eng.pool.num_free == eng.pool.num_usable
    assert eng.runner.state._held == {}


def test_stale_state_rows_are_never_read(gdn):
    """A request starting at position 0 must not see what its row held before, even NaN."""
    from kiln.engine.request import SamplingParams

    path, hf = gdn
    eng = engine(path)
    for t in eng.runner.state.conv + eng.runner.state.rec:
        t.fill_(float("nan"))
    ps = prompts(2, (9, 23))
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 10) for p in ps]


def test_preemption_recomputes_from_zero_state(gdn):
    """A pool too small for every request preempts the youngest; its recompute (from position
    0, no prefix cache) restarts its recurrent state and gives the reference's tokens."""
    from kiln.engine.request import SamplingParams

    path, hf = gdn
    eng = engine(path, num_pages=20, max_prefill_tokens=16, admission="eager")  # the preemption path
    ps = prompts(3, (25, 30, 22))
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=16, ignore_eos=True))
    assert eng.scheduler.num_preemptions > 0
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 16) for p in ps]


def test_prompt_logprobs_match(gdn):
    from kiln.engine.request import SamplingParams

    path, hf = gdn
    eng = engine(path)
    (ids,) = prompts(9, (30,))
    with torch.no_grad():
        logp = torch.log_softmax(hf(torch.tensor([ids])).logits[0].double(), -1)
    (r,) = eng.generate([ids], SamplingParams(max_new_tokens=2, ignore_eos=True, prompt_logprobs=0))
    assert sorted(r.prompt_logprobs) == list(range(1, len(ids)))
    for q, (lp, *_rest) in r.prompt_logprobs.items():
        assert abs(lp - logp[q - 1, ids[q]].item()) < 1e-4


def test_tp2_matches_tp1(gdn):
    """GDN heads split across ranks (1 k head and its 2 v heads per rank), each rank with its
    own state pool, rows chosen by rank 0."""
    from kiln.engine.request import SamplingParams

    path, _ = gdn
    ps = prompts(5, (11, 30))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    want = [r.output_ids for r in engine(path, max_num_seqs=2).generate(ps, sp)]
    two = engine(path, max_num_seqs=2, tp=2)
    try:
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want


def test_serving_features_are_accepted(gdn):
    """Speculation, jump-forward and the host tier run for these models now
    (tests/test_linear_serving.py checks them); MTP still needs MTP layers in the checkpoint."""
    path, _ = gdn
    for kw in (dict(spec_method="ngram"), dict(spec_method="suffix"), dict(jump_forward=True),
               dict(hicache_host_gb=0.01)):
        engine(path, **kw)
    with pytest.raises(ValueError, match="MTP"):
        engine(path, spec_method="mtp")
    with pytest.raises(ValueError, match="prefix cache"):
        engine(path, hicache_host_gb=0.01, prefix_caching=False)


def test_tied_embeddings_and_odd_layouts(tmp_path):
    """Tied embeddings, a linear layer last and first, one k head serving every v head."""
    from kiln.engine.request import SamplingParams

    hf = build_qwen3_5(str(tmp_path), layer_types=("linear_attention", "full_attention", "linear_attention"),
                       tie=True, seed=7, linear_num_key_heads=1, linear_num_value_heads=4)
    ps = prompts(6, (13, 40))
    reqs = engine(str(tmp_path)).generate(ps, SamplingParams(max_new_tokens=8, ignore_eos=True))
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 8) for p in ps]


# -- Qwen3.8-27B layer shapes (whole model, truncated) -----------------------------------------


def test_qwen3_8_27b_shapes_truncated(tmp_path):
    """Qwen3.8-27B's attention and GDN layers at their real shapes (hidden 5120, GDN 16 k / 48 v
    heads x 128, gated attention 24 / 4 heads x 256, partial RoPE 64 of 256), 3 GDN + 1 attention
    layer as in the checkpoint's pattern. Vocabulary and MLP width are cut to keep the test at
    about 2 GB per copy; they are not what the test is about."""
    from kiln.engine.request import SamplingParams

    shape = dict(QWEN3_8_27B, vocab_size=512, intermediate_size=256)
    hf = build_qwen3_5(str(tmp_path), seed=11, **shape)
    ps = prompts(8, (70, 9), vocab=512)
    with torch.no_grad():
        want_logits = hf(torch.tensor([ps[0]])).logits[0]
    want = [hf_greedy(hf, p, 6) for p in ps]
    del hf
    eng = engine(str(tmp_path), max_prefill_tokens=32, max_num_seqs=2)
    with torch.no_grad():
        got = eng.model.forward_logits(torch.tensor(ps[0]))
    err = (got - want_logits).abs().max().item()
    print(f"Qwen3.8-27B shapes: logits max abs diff {err:.2e} (|logits| max {want_logits.abs().max().item():.2f})")
    assert err < 1e-3
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=6, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want


# -- single layers at real shapes against the newest references -------------------------------


def kiln_layer(tmp_path, spec, hidden, eps, tensors: dict, prefix: str):
    """A one-layer Kiln model holding `tensors` (checkpoint names under model.layers.0.`prefix`)
    and its mixer, for layer-level comparisons."""
    from safetensors.torch import save_file

    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    cfg = ModelConfig(architecture="KimiLinearForCausalLM", vocab_size=8, hidden_size=hidden, intermediate_size=8, num_layers=1,
                      num_heads=1, num_kv_heads=1, head_dim=8, rms_norm_eps=eps, rope_theta=1e4,
                      max_position_embeddings=64, tie_word_embeddings=True, eos_token_ids=(), attn_layers=(spec,))
    t = {f"model.layers.0.{prefix}{k}": v.detach().contiguous() for k, v in tensors.items()}
    for n in ("model.layers.0.input_layernorm.weight", "model.layers.0.post_attention_layernorm.weight",
              "model.norm.weight"):
        t[n] = torch.ones(hidden)
    for n, shp in (("gate_proj", (8, hidden)), ("up_proj", (8, hidden)), ("down_proj", (hidden, 8))):
        t[f"model.layers.0.mlp.{n}.weight"] = torch.zeros(shp)
    t["model.embed_tokens.weight"] = torch.zeros(8, hidden)
    save_file(t, str(tmp_path / "model.safetensors"))
    return load_model(str(tmp_path), cfg, torch.float32, torch.device("cpu"), 64)


def compare_layer(model, ref_fn, T=70, seed=0):
    """Kiln's mixer (sequence form, and chunk + decode forms through a state pool) against
    ref_fn(normalised hidden [1, T, H]) -> [1, T, H]."""
    from kiln.engine.state_pool import StatePool
    from kiln.models import linear_attn
    from kiln.models.decoder import rms_norm

    layer = model.layers[0]
    H = model.cfg.hidden_size
    h = torch.randn(T, H, generator=torch.Generator().manual_seed(seed))
    x = rms_norm(h, layer.in_norm, model.cfg.rms_norm_eps)
    with torch.no_grad():
        want = ref_fn(x.unsqueeze(0))[0]
        got = linear_attn.mixer(model, layer, h, torch.arange(T), None, None) - h
        scale = want.abs().max().item()
        e_seq = (got - want).abs().max().item()
        assert e_seq < 1e-4 * max(1.0, scale), e_seq
        # The serving forms: a chunk padded by 8 rows, a second chunk that continues its state,
        # then decode steps.
        StatePool(model, 2, torch.device("cpu"), torch.float32)
        model.page_size = 4
        outs, start = [], 0
        for n, C in ((T // 2, T // 2 + 8), (T // 3, T // 3)):
            hc = torch.zeros(C, H)
            hc[:n] = h[start : start + n]
            pos = torch.zeros(C, dtype=torch.long)
            pos[:n] = torch.arange(start, start + n)
            slots = torch.zeros(C, dtype=torch.long)
            slots[:n] = pos[:n] + 4
            outs.append(linear_attn.mixer(model, layer, hc, pos, slots, torch.tensor([1]))[:n] - h[start : start + n])
            start += n
        for t in range(start, T):
            outs.append(linear_attn.mixer(model, layer, h[t : t + 1], torch.tensor([t]), torch.tensor([t + 4]),
                                          torch.tensor([1])) - h[t : t + 1])
        got = torch.cat(outs)
        e_serve = (got - want).abs().max().item()
        assert e_serve < 1e-4 * max(1.0, scale), e_serve
        print(f"layer parity (pytest -s): sequence {e_seq:.2e}, chunked + decode {e_serve:.2e}, |output| max {scale:.3f}")


def test_qwen3_8_flash_next_gdn_layer(tmp_path):
    """Qwen3.8-Flash-Next's GDN layer: 16 k / 48 v heads x 128, hidden 2560, sigmoid output gate
    (output_gate_type), against transformers' Qwen4ExpTextGatedDeltaNet. The rest of that model
    (hyper-connections, n-gram embeddings, QSA, MoE) is not implemented."""
    mod = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    from kiln.models.linear_attn import gdn_spec

    cfg = Qwen4ExpTextConfig(**QWEN3_8_FLASH_NEXT, num_hidden_layers=1, layer_types=["linear_attention"],
                             num_experts=4, num_experts_per_tok=2, moe_intermediate_size=8,
                             shared_expert_intermediate_size=8, vocab_size=64)
    ref = mod.Qwen4ExpTextGatedDeltaNet(cfg, 0)
    randomize(ref, 1)
    spec = gdn_spec(QWEN3_8_FLASH_NEXT, "sigmoid")
    model = kiln_layer(tmp_path, spec, cfg.hidden_size, cfg.rms_norm_eps, ref.state_dict(), "linear_attn.")
    compare_layer(model, lambda x: ref(x))


def test_kimi_linear_kda_layer(tmp_path):
    """Kimi Linear (48B-A3B) KDA layer: 32 heads x 128, hidden 2304, low-rank gates, no lower
    bound, against transformers' KimiLinearDeltaAttention (its own weight layout: stacked conv,
    forget_gate.*, A_log [1, 1, H, 1])."""
    mod = pytest.importorskip("transformers.models.kimi_linear.modeling_kimi_linear")
    from transformers import KimiLinearConfig

    from kiln.models.linear_attn import kda_spec

    cfg = KimiLinearConfig(hidden_size=KIMI_LINEAR_48B["hidden_size"], rms_norm_eps=KIMI_LINEAR_48B["rms_norm_eps"],
                           linear_attn_config=KIMI_LINEAR_48B["linear_attn_config"], num_hidden_layers=1,
                           layer_types=["linear_attention"], vocab_size=64, pad_token_id=0, bos_token_id=1,
                           eos_token_id=2)
    ref = mod.KimiLinearDeltaAttention(cfg, 0)
    randomize(ref, 2)
    model = kiln_layer(tmp_path, kda_spec(KIMI_LINEAR_48B), cfg.hidden_size, cfg.rms_norm_eps, ref.state_dict(),
                       "self_attn.")
    compare_layer(model, lambda x: ref(x))


def test_glm_5_3_flash_kda_layer(tmp_path):
    """GLM-5.3-Flash's KDA layer: 64 heads x 128, hidden 4096, safe gate (lower bound -5), against
    transformers' Glm5NextTextLinearAttention, weights saved under the checkpoint's names."""
    mod = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    from kiln.models.linear_attn import kda_spec

    cfg = Glm5NextTextConfig(hidden_size=GLM_5_3_FLASH["hidden_size"], rms_norm_eps=GLM_5_3_FLASH["rms_norm_eps"],
                             linear_attn_config=GLM_5_3_FLASH["linear_attn_config"], num_hidden_layers=1,
                             layer_types=["linear_attention"], vocab_size=64)
    ref = mod.Glm5NextTextLinearAttention(cfg, 0)
    randomize(ref, 3)
    sd = ref.state_dict()
    # zai-org/GLM-5.3-Flash model.safetensors.index.json names: q/k/v_conv1d, f_* and A_log [H]
    # directly under self_attn (transformers' conversion_mapping.py "glm5_next" maps them back).
    names = {}
    D = cfg.linear_num_heads * cfg.linear_head_dim
    for k, v in sd.items():
        if k == "conv1d.weight":
            for i, x in enumerate("qkv"):
                names[f"{x}_conv1d.weight"] = v[i * D : (i + 1) * D]
        else:
            names[k.replace("forget_gate.", "")] = v
    spec = kda_spec(GLM_5_3_FLASH)
    assert spec.lower_bound == -5.0 and spec.gate_rank == 128
    model = kiln_layer(tmp_path, spec, cfg.hidden_size, cfg.rms_norm_eps, names, "self_attn.")
    compare_layer(model, lambda x: ref(x))


def test_kimi_k3_kda_layer(tmp_path, monkeypatch):
    """Kimi K3's KDA layer (96 heads x 128, hidden 7168, full-rank output gate g_proj, lower
    bound -5) through K3's OWN module (moonshotai/Kimi-K3 modeling_kimi_linear.py, pinned
    revision), whose fla kernels need CUDA: they are replaced by the pure-torch functions of
    transformers' glm5_next port (the same KDA, safe gate included), so this checks K3's wiring
    and weight names against Kiln, not fla itself. Downloads the 60 KB modeling files."""
    glm = pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    pytest.importorskip("einops")
    import importlib.util
    import sys
    import types

    from huggingface_hub import hf_hub_download

    try:
        files = {f: hf_hub_download("moonshotai/Kimi-K3", f, revision=K3_REVISION)
                 for f in ("modeling_kimi_linear.py", "configuration_kimi_k3.py")}
    except Exception as e:  # offline
        pytest.skip(f"cannot fetch the Kimi K3 modeling code: {e}")
    fla = _fla_shim(glm)
    for name, m in fla.items():
        monkeypatch.setitem(sys.modules, name, m)
    # K3's file targets transformers 4.56; OutputRecorder moved to utils.output_capturing in 5.x.
    import transformers.utils.generic as generic

    if not hasattr(generic, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder

        monkeypatch.setattr(generic, "OutputRecorder", OutputRecorder, raising=False)
    pkg = types.ModuleType("kimi_k3")
    pkg.__path__ = [os.path.dirname(files["modeling_kimi_linear.py"])]
    monkeypatch.setitem(sys.modules, "kimi_k3", pkg)
    mods = {}
    for short in ("configuration_kimi_k3", "modeling_kimi_linear"):
        spec = importlib.util.spec_from_file_location(f"kimi_k3.{short}", files[short + ".py"])
        mods[short] = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, f"kimi_k3.{short}", mods[short])
        try:
            spec.loader.exec_module(mods[short])
        except Exception as e:
            pytest.skip(f"Kimi K3 modeling code does not import here: {e!r}")
    cfg = mods["configuration_kimi_k3"].KimiLinearConfig(
        hidden_size=KIMI_K3["hidden_size"], rms_norm_eps=KIMI_K3["rms_norm_eps"], num_hidden_layers=1,
        linear_attn_config=dict(KIMI_K3["linear_attn_config"], kda_layers=[1], full_attn_layers=[]), vocab_size=64)
    ref = mods["modeling_kimi_linear"].KimiDeltaAttention(cfg, 0)
    randomize(ref, 4)
    from kiln.models.linear_attn import kda_spec

    spec = kda_spec(KIMI_K3)
    assert spec.gate_rank == 0 and spec.lower_bound == -5.0
    model = kiln_layer(tmp_path, spec, KIMI_K3["hidden_size"], KIMI_K3["rms_norm_eps"], ref.state_dict(), "self_attn.")
    compare_layer(model, lambda x: ref(x), T=40)


def _fla_shim(glm):
    """The fla entry points K3's module imports, written over transformers' glm5_next functions.
    fla's signatures (fla/ops/kda): chunk_kda(q, k, v, g, beta, A_log, dt_bias, initial_state,
    output_final_state, use_qk_l2norm_in_kernel, use_gate_in_kernel, use_beta_sigmoid_in_kernel,
    safe_gate, lower_bound, transpose_state_layout, cu_seqlens)."""
    import types

    from torch import nn

    def gate(g, A_log, dt_bias, lower_bound):
        H, D = g.shape[-2:]
        g = g.float() + dt_bias.float().view(H, D)
        rate = A_log.float().view(H, 1).exp()
        if lower_bound is not None:  # Glm5NextTextForgetGate.forward
            return lower_bound * torch.sigmoid(rate * g)
        return -rate * torch.where(g > 20.0, g, torch.log(1.0 + torch.exp(g)))

    def chunk_kda(q, k, v, g, beta, A_log, dt_bias, initial_state=None, output_final_state=True,
                  use_qk_l2norm_in_kernel=True, lower_bound=None, use_beta_sigmoid_in_kernel=True, **_):
        b = torch.sigmoid(beta) if use_beta_sigmoid_in_kernel else beta
        return glm.chunk_kimi_delta_attention(q, k, v, gate(g, A_log, dt_bias, lower_bound), b,
                                              initial_state=initial_state, output_final_state=output_final_state,
                                              use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    class ShortConvolution(nn.Conv1d):
        def __init__(self, hidden_size, kernel_size, activation=None, **_):
            super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False,
                             padding=kernel_size - 1)

        def forward(self, x, cache=None, output_final_state=False, **_):
            y = glm.causal_conv1d_fn(x.transpose(1, 2), self.weight.squeeze(1), None, activation="silu")
            return y.transpose(1, 2), None

    class FusedRMSNormGated(glm.Glm5NextTextRMSNormGated):  # sigmoid-gated, fp32 norm
        def __init__(self, hidden_size, eps=1e-6, activation="sigmoid", **_):
            super().__init__(hidden_size, eps)

    ops = types.ModuleType("fla.ops.kda")
    ops.chunk_kda, ops.fused_recurrent_kda = chunk_kda, chunk_kda
    modules = types.ModuleType("fla.modules")
    modules.ShortConvolution, modules.FusedRMSNormGated = ShortConvolution, FusedRMSNormGated
    index = types.ModuleType("fla.ops.utils.index")
    index.prepare_cu_seqlens_from_mask = index.prepare_lens_from_mask = lambda *a, **k: None
    utils = types.ModuleType("fla.utils")
    utils.tensor_cache = lambda f: f
    out = {"fla": types.ModuleType("fla"), "fla.modules": modules, "fla.ops": types.ModuleType("fla.ops"),
           "fla.ops.kda": ops, "fla.ops.utils": types.ModuleType("fla.ops.utils"), "fla.ops.utils.index": index,
           "fla.utils": utils}
    return out


# -- a KDA-only Kimi Linear model through the engine ---------------------------------------------


def test_kimi_linear_kda_model_matches(tmp_path):
    """Every layer KDA with a dense MLP (the MLA and MoE layers of the real Kimi Linear are
    elsewhere), against transformers' KimiLinearForCausalLM: logits and paged chunked greedy."""
    pytest.importorskip("transformers.models.kimi_linear")
    import transformers as tf
    from safetensors.torch import load_file, save_file

    from kiln.engine.request import SamplingParams

    cfg = tf.KimiLinearConfig(vocab_size=384, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                              num_attention_heads=4, first_k_dense_replace=3, rms_norm_eps=1e-5,
                              linear_attn_config={"num_heads": 4, "head_dim": 16, "short_conv_kernel_size": 4,
                                                  "kda_layers": [1, 2, 3], "full_attn_layers": []},
                              pad_token_id=0, bos_token_id=1, eos_token_id=2)
    hf = tf.KimiLinearForCausalLM(cfg)
    randomize(hf, 5)
    hf.eval()
    hf.save_pretrained(str(tmp_path))
    # transformers saves the dense MLP under block_sparse_moe. (its conversion_mapping.py renames
    # every .mlp. back); the hub checkpoint names it mlp. (Kimi-Linear-48B layer 0).
    f = str(tmp_path / "model.safetensors")
    save_file({k.replace(".block_sparse_moe.", ".mlp."): v for k, v in load_file(f).items()}, f, metadata={"format": "pt"})
    eng = engine(str(tmp_path))
    ids = torch.randint(0, 384, (90,), generator=torch.Generator().manual_seed(6))
    # use_cache=False: transformers' DynamicCache cannot report a length without an attention
    # layer ("get_seq_length can only be called on Attention layers"), so the reference
    # recomputes the whole sequence every step.
    with torch.no_grad():
        want = hf(ids.unsqueeze(0), use_cache=False).logits[0]
        got = eng.model.forward_logits(ids)
    assert (got - want).abs().max().item() < 1e-4
    ps = prompts(7, (6, 33))
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == [hf_greedy(hf, p, 10, use_cache=False) for p in ps]


def build_kda(path, full_rank=False, lower_bound=-5.0, heads=4, dim=16, hidden=64, layers=3, seed=0):
    """A KDA-only checkpoint in the hub layout (Kimi K3 / GLM-5.3-Flash names: q/k/v_conv1d,
    A_log [H], g_proj or g_a_proj / g_b_proj), written without transformers."""
    import json

    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(seed)
    r = lambda *s, sc=0.08: torch.randn(*s, generator=g) * sc  # noqa: E731
    D = heads * dim
    t = {"model.embed_tokens.weight": r(384, hidden), "model.norm.weight": 1 + r(hidden, sc=0.3),
         "lm_head.weight": r(384, hidden)}
    for i in range(layers):
        p = f"model.layers.{i}."
        t.update({p + "input_layernorm.weight": 1 + r(hidden, sc=0.3), p + "post_attention_layernorm.weight": 1 + r(hidden, sc=0.3),
                  p + "mlp.gate_proj.weight": r(96, hidden), p + "mlp.up_proj.weight": r(96, hidden),
                  p + "mlp.down_proj.weight": r(hidden, 96)})
        a = p + "self_attn."
        t.update({a + f"{x}_proj.weight": r(D, hidden) for x in "qkv"})
        t.update({a + f"{x}_conv1d.weight": r(D, 1, 4, sc=0.3) for x in "qkv"})
        t.update({a + "f_a_proj.weight": r(dim, hidden), a + "f_b_proj.weight": r(D, dim, sc=0.3),
                  a + "dt_bias": r(D, sc=1.0), a + "A_log": torch.empty(heads).uniform_(-4, 1, generator=g),
                  a + "b_proj.weight": r(heads, hidden), a + "o_norm.weight": 1 + r(dim, sc=0.3), a + "o_proj.weight": r(hidden, D)})
        if full_rank:
            t[a + "g_proj.weight"] = r(D, hidden)
        else:
            t.update({a + "g_a_proj.weight": r(dim, hidden), a + "g_b_proj.weight": r(D, dim, sc=0.3)})
    save_file(t, os.path.join(path, "model.safetensors"))
    la = {"num_heads": heads, "head_dim": dim, "short_conv_kernel_size": 4, "kda_layers": list(range(1, layers + 1)),
          "full_attn_layers": [], "use_full_rank_gate": full_rank}
    if lower_bound is not None:
        la["gate_lower_bound"] = lower_bound
    cfg = {"architectures": ["KimiLinearForCausalLM"], "vocab_size": 384, "hidden_size": hidden, "intermediate_size": 96,
           "num_hidden_layers": layers, "rms_norm_eps": 1e-5, "linear_attn_config": la, "first_k_dense_replace": layers,
           "tie_word_embeddings": False, "eos_token_id": 2, "hidden_act": "silu", "max_position_embeddings": 512}
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(cfg, f)


@pytest.mark.parametrize("full_rank,lower_bound", [(False, None), (True, -5.0)])
def test_kda_engine_forms_and_tp2(tmp_path, full_rank, lower_bound):
    """KDA through the engine (chunked prefill, decode, piecewise) agrees with the sequence form,
    and tp=2 (4 heads, 2 per rank; f_b / g_b / dt_bias sharded, f_a / g_a replicated) with tp=1."""
    from kiln.engine.request import SamplingParams

    build_kda(str(tmp_path), full_rank=full_rank, lower_bound=lower_bound)
    ps = prompts(12, (7, 45))
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    one = engine(str(tmp_path), max_num_seqs=2)
    want = []
    for p in ps:  # greedy by the sequence form, one token at a time
        ids = list(p)
        for _ in range(10):
            with torch.no_grad():
                ids.append(int(one.model.forward_logits(torch.tensor(ids))[-1].argmax()))
        want.append(ids[len(p):])
    assert [r.output_ids for r in one.generate(ps, sp)] == want
    assert [r.output_ids for r in engine(str(tmp_path), max_num_seqs=2, piecewise=True).generate(ps, sp)] == want
    two = engine(str(tmp_path), max_num_seqs=2, tp=2)
    try:
        assert [r.output_ids for r in two.generate(ps, sp)] == want
    finally:
        two.close()


# -- the real checkpoint ---------------------------------------------------------------------------

REAL = os.environ.get("KILN_TEST_LINEAR_MODEL")  # e.g. Qwen/Qwen3.5-0.8B


@pytest.mark.skipif(not REAL, reason="set KILN_TEST_LINEAR_MODEL (e.g. Qwen/Qwen3.5-0.8B) to run")
def test_real_checkpoint_matches_transformers():
    """Reference outputs first, then the reference is freed before Kiln loads (host memory)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kiln.engine.request import SamplingParams
    from kiln.models.loader import resolve_model_path

    path = resolve_model_path(REAL)
    tok = AutoTokenizer.from_pretrained(path)
    # The last prompt is past a 64-token sub-chunk of the chunked prefill: short prompts alone missed
    # the unstable sub-chunk inverse (test_chunk_scan_is_stable_on_correlated_keys).
    texts = ["The capital of France is", "def fibonacci(n):\n    ",
             "Trainium is a machine learning accelerator built by", "1, 2, 3, 5, 8, 13,",
             "You are a careful assistant. Answer in one short sentence, and repeat the key word of the question "
             "at the end of your answer.\nWhat is the capital of France? The capital of France is Paris, a city "
             "on the Seine known for its museums, its cafes and the tower built for the 1889 World's Fair. And "
             "what is the capital of Germany? The capital of Germany is"]
    ps = [tok(t)["input_ids"] for t in texts]
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    with torch.no_grad():
        logits = [hf(torch.tensor([p])).logits[0] for p in ps]
    want = [hf_greedy(hf, p, 24) for p in ps]
    del hf
    eng = engine(path, max_prefill_tokens=64, max_num_seqs=5, num_pages=512)
    for p, w in zip(ps, logits):
        with torch.no_grad():
            got = eng.model.forward_logits(torch.tensor(p))
        err = (got - w).abs().max().item()
        print(f"{REAL}: logits max abs diff {err:.2e} over {len(p)} tokens")
        assert err < 2e-3
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=24, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
