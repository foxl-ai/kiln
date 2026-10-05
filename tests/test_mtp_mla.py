"""Multi-token prediction for DeepSeek-V3, DeepSeek-V3.2 and GLM-5.3 (models/mtp.py), against
transformers' own decoder layers.

Checkpoints: tests/test_mla.py's random models from the REAL config.json (truncated), built by
transformers with one layer more than the target keeps, so that layer is a real
DeepseekV3 / DeepseekV32 / GlmMoeDsa decoder layer (MLA, the DSA indexer, MoE + shared expert); it
is saved under the checkpoints' MTP names (model.layers.<n>.*, plus enorm, hnorm, eh_proj and
shared_head.norm; zai-org/GLM-5.3 and deepseek-ai/DeepSeek-V3 model.safetensors.index.json) and the
config gets num_nextn_predict_layers 1. transformers has no MTP module for these models (it drops
the layer when loading), so the reference recursion is written out here (vLLM v0.30.0
deepseek_mtp.py / SGLang v0.5.21 deepseek_nextn.py): x = eh_proj(cat(enorm(embed(next token)),
hnorm(target hidden after its final norm))), the layer as a one-layer transformers model whose
final norm is shared_head.norm, the target's lm_head, and the next pass fed the normed MTP hidden.
With index_share_for_mtp_iteration the later passes reuse the first pass's DSA selection, done in
the reference by replacing the one-layer model's indexer output with it.
"""

from __future__ import annotations

import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.test_mla import hf_config

# The DSA models in the sparse regime (index_topk 12 against contexts up to ~60 tokens).
MODELS = {"deepseek_v3": {}, "deepseek_v32": dict(index_topk=12), "glm_moe_dsa": dict(index_topk=12)}


def build_with_mtp(name: str, path: str, seed: int = 0, copy_main: bool = False, share: bool | None = None,
                   **extra):
    """A checkpoint with one MTP layer at `path`; returns (target config dict, MTP tensors).
    copy_main: the target has ONE layer and the MTP layer is its copy, eh_proj passes only the
    embedding and enorm / shared_head.norm are the identity / the target's norm, so a draft is the
    target's own prediction one token later on a sequence shifted by one (a meaningful acceptance
    rate from random weights)."""
    from tests.test_mla import build

    base = hf_config(name, **extra)
    n = base.num_hidden_layers
    kw = dict(extra, num_hidden_layers=n + 1)
    if name == "glm_moe_dsa":  # the MTP layer runs its own indexer and the MoE
        kw.update(indexer_types=list(base.indexer_types) + ["full"],
                  mlp_layer_types=list(base.mlp_layer_types) + ["sparse"])
    if copy_main:
        kw.update(first_k_dense_replace=0)
        if name == "glm_moe_dsa":
            kw.update(indexer_types=["full", "full"], mlp_layer_types=["sparse", "sparse"])
    build(name, path, seed=seed, **kw)
    f = os.path.join(path, "model.safetensors")
    t = load_file(f)
    H = base.hidden_size
    p = f"model.layers.{n}."
    g = torch.Generator().manual_seed(seed + 100)
    if copy_main:
        for k in [k for k in t if k.startswith("model.layers.0.")]:
            t[p + k[len("model.layers.0."):]] = t[k].clone()
        w = {p + "enorm.weight": torch.ones(H), p + "hnorm.weight": torch.ones(H),
             p + "eh_proj.weight": torch.cat([torch.eye(H), torch.zeros(H, H)], dim=1),
             p + "shared_head.norm.weight": t["model.norm.weight"].clone()}
    else:
        w = {p + "enorm.weight": 1 + 0.1 * torch.randn(H, generator=g),
             p + "hnorm.weight": 1 + 0.1 * torch.randn(H, generator=g),
             p + "eh_proj.weight": torch.randn(H, 2 * H, generator=g) * (2 * H) ** -0.5,
             p + "shared_head.norm.weight": 1 + 0.1 * torch.randn(H, generator=g)}
    if name != "glm_moe_dsa":  # DeepSeek stores copies of both (unused: the target's are shared)
        w[p + "embed_tokens.weight"] = t["model.embed_tokens.weight"].clone()
        w[p + "shared_head.head.weight"] = t["lm_head.weight"].clone()
    t.update(w)
    save_file(t, f, metadata={"format": "pt"})
    with open(os.path.join(path, "config.json")) as fh:
        c = json.load(fh)
    c["num_hidden_layers"] = n
    c["num_nextn_predict_layers"] = 1
    for key, v in list(c.items()):  # per-layer lists (indexer_types, mlp_layer_types, layer_types)
        if key.endswith("types") and isinstance(v, list) and len(v) == n + 1:
            c[key] = v[:n]
    if share is not None:
        c["index_share_for_mtp_iteration"] = share
    with open(os.path.join(path, "config.json"), "w") as fh:
        json.dump(c, fh)
    return c, {k: v for k, v in t.items() if k.startswith(p)}


def reference_block(name: str, path: str, mtp_w: dict, **extra):
    """The MTP layer as a one-layer transformers model: its decoder layer, shared_head.norm as the
    final norm, the target's embedding and lm_head."""
    import transformers as tf

    kw = dict(extra, num_hidden_layers=1, first_k_dense_replace=0)
    if name == "glm_moe_dsa":
        kw.update(indexer_types=["full"], mlp_layer_types=["sparse"])
    cfg = hf_config(name, **kw)
    cfg._attn_implementation = "eager"
    cls = {"DeepseekV3Config": tf.DeepseekV3ForCausalLM, "DeepseekV32Config": tf.DeepseekV32ForCausalLM,
           "GlmMoeDsaConfig": tf.GlmMoeDsaForCausalLM}[type(cfg).__name__]
    main = load_file(os.path.join(path, "model.safetensors"))
    pre = [k for k in mtp_w if k.endswith("enorm.weight")][0][: -len("enorm.weight")]
    sd = {"model.embed_tokens.weight": main["model.embed_tokens.weight"], "lm_head.weight": main["lm_head.weight"],
          "model.norm.weight": mtp_w[pre + "shared_head.norm.weight"]}
    for k, v in mtp_w.items():
        rest = k[len(pre):]
        if rest.startswith(("enorm", "hnorm", "eh_proj", "shared_head", "embed_tokens")):
            continue
        sd["model.layers.0." + rest] = v
    # Through from_pretrained, which fuses the per-expert tensors the checkpoint names into
    # transformers 5's mlp.experts.gate_up_proj / down_proj.
    out = path.rstrip("/") + "-mtp-block"
    cfg.save_pretrained(out)
    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    blk, info = cls.from_pretrained(out, dtype=torch.float32, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"], info
    return blk.eval()


def rms(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def reference_drafts(hf, blk, mtp_w, tokens, k, share=False):
    """Drafts for the positions after tokens[-1]: MTP positions 0..N-2 take tokens[1..N-1] and
    the target's normed hidden states, then k - 1 recursive positions; with share, every pass after
    the first attends to the first pass's DSA selection of its last position."""
    pre = [n for n in mtp_w if n.endswith("enorm.weight")][0][: -len("enorm.weight")]
    eps = hf.config.rms_norm_eps
    indexer = getattr(blk.model.layers[0].self_attn, "indexer", None)
    first: dict = {}

    def hook(mod, args, out):
        if "top" not in first:
            first["top"] = out[0, -1].clone()
            return out
        return first["top"].view(1, 1, -1).expand(out.shape[0], out.shape[1], -1).to(out.dtype)

    handle = indexer.register_forward_hook(hook) if share and indexer is not None else None
    try:
        with torch.no_grad():
            T = torch.tensor([tokens])
            H = hf.model(T[:, :-1]).last_hidden_state[0]  # after the target's final norm
            ids, hid, drafts = list(tokens[1:]), [h for h in H], []
            for _ in range(k):
                e = hf.model.embed_tokens(torch.tensor([ids]))[0]
                x = torch.cat([rms(e, mtp_w[pre + "enorm.weight"], eps),
                               rms(torch.stack(hid), mtp_w[pre + "hnorm.weight"], eps)], -1)
                x = x @ mtp_w[pre + "eh_proj.weight"].T
                hm = blk.model(inputs_embeds=x.unsqueeze(0)).last_hidden_state[0]  # after shared_head.norm
                d = int(hf.lm_head(hm[-1]).argmax())
                drafts.append(d)
                ids.append(d)
                hid.append(hm[-1])
    finally:
        if handle is not None:
            handle.remove()
    return drafts


def engine(path, **kw):
    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine

    base = dict(model_path=path, device="cpu", dtype=torch.float32, page_size=4, num_pages=256, max_num_seqs=2,
                max_model_len=256, max_prefill_tokens=8, spec_method="mtp", spec_k=2)
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


@pytest.fixture(scope="module", params=list(MODELS))
def mtp_model(request, tmp_path_factory):
    from transformers import AutoModelForCausalLM

    name = request.param
    path = str(tmp_path_factory.mktemp(name + "_mtp"))
    _, w = build_with_mtp(name, path, seed=1, **MODELS[name])
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    return name, path, hf, w, reference_block(name, path, w, **MODELS[name])


def test_config_and_weights(mtp_model):
    """The MTP layer is an MLA (+ DSA indexer) + MoE layer under the checkpoint's own names, and
    the target is unchanged."""
    from kiln.config import ModelConfig

    name, path, hf, w, _ = mtp_model
    cfg = ModelConfig.from_pretrained(path)
    n = cfg.num_layers
    assert cfg.mtp_layers == 1 and cfg.mtp_prefix == f"model.layers.{n}" and cfg.mtp_moe
    assert cfg.mtp_spec.mla is not None
    assert (cfg.mtp_spec.mla.dsa is not None) == (name != "deepseek_v3")
    if cfg.mtp_spec.mla.dsa is not None:
        assert cfg.mtp_spec.mla.dsa.indexer
    assert cfg.mtp_index_share == (name == "glm_moe_dsa")  # GLM-5.3's config.json sets it
    eng = engine(path)
    m = eng.model
    assert torch.equal(m.mtp_eh, w[f"model.layers.{n}.eh_proj.weight"])
    assert torch.equal(m.mtp_norm, w[f"model.layers.{n}.shared_head.norm.weight"])
    assert m.mtp.moe and m.mtp.shared_gate_up is not None
    assert len(eng.runner.k_caches) == n + 1  # the MTP layer's latent cache is one more pool layer


@pytest.mark.parametrize("share", [False, True])
def test_drafts_match_the_reference_after_every_step(mtp_model, share, monkeypatch):
    """Chunked prefill (MTP KV filled chunk by chunk), then verify steps of every acceptance
    length: after each step, every running request's drafts equal the reference recursion's."""
    from kiln.engine.request import SamplingParams

    name, path, hf, w, blk = mtp_model
    if share and name == "deepseek_v3":
        pytest.skip("no DSA indexer to share")
    monkeypatch.setenv("KILN_MTP_INDEX_SHARE", "1" if share else "0")
    eng = engine(path)
    assert eng.model.mtp_index_share == share
    g = torch.Generator().manual_seed(3)
    reqs = [eng.add_request(torch.randint(0, 384, (n,), generator=g).tolist(),
                            SamplingParams(max_new_tokens=14, ignore_eos=True)) for n in (11, 37)]
    checked = 0
    while eng.has_work():
        eng.step()
        for r in reqs:
            if r.mtp_draft:
                assert r.mtp_draft == reference_drafts(hf, blk, w, r.token_ids, 2, share), (name, r.rid, len(r.token_ids))
                checked += 1
    assert checked >= 8


@pytest.mark.parametrize("piecewise", [False, True])
def test_speculation_leaves_greedy_output_unchanged(mtp_model, piecewise):
    from kiln.engine.request import SamplingParams
    from tests.test_mla import hf_greedy

    name, path, hf, *_ = mtp_model
    prompts = [[5, 9, 11, 200, 3, 77, 12], list(range(40, 85))]
    sp = SamplingParams(max_new_tokens=16, ignore_eos=True)
    spec = engine(path, spec_k=3, piecewise=piecewise)
    got = [r.output_ids for r in spec.generate(prompts, sp)]
    assert got == [hf_greedy(hf, p, 16) for p in prompts]
    assert spec.spec_proposed > 0


def test_mtp_under_tensor_parallelism_matches_tp1(mtp_model):
    from kiln.engine.request import SamplingParams

    name, path, *_ = mtp_model
    if name != "glm_moe_dsa":
        pytest.skip("one model is enough: the MTP layer shards like any MLA layer")
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14], list(range(40, 70))]
    sp = SamplingParams(max_new_tokens=12, ignore_eos=True)
    one = engine(path, num_pages=128, max_model_len=128)
    want = [r.output_ids for r in one.generate(prompts, sp)]
    two = engine(path, num_pages=128, max_model_len=128, tp=2)
    try:
        got = [r.output_ids for r in two.generate(prompts, sp)]
        assert got == want and (two.spec_proposed, two.spec_accepted) == (one.spec_proposed, one.spec_accepted)
    finally:
        two.close()


def test_warmup_compiles_every_mtp_graph_drafting_uses(mtp_model):
    from kiln.engine.request import SamplingParams

    name, path, *_ = mtp_model
    if name != "glm_moe_dsa":
        pytest.skip("graph keys do not depend on the MLA family")
    eng = engine(path, max_model_len=64, decode_batch_buckets=(1, 2), prefill_token_buckets=(8,),
                 page_buckets=(4, 16))
    eng.warmup()
    before = set(eng.runner.compile_seconds)
    assert any(k[0] == "mtp" for k in before)
    eng.generate([[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 53))],
                 SamplingParams(max_new_tokens=16, ignore_eos=True))
    assert set(eng.runner.compile_seconds) == before


def test_warmup_covers_prefill_drafting_with_pinned_decode_buckets(mtp_model):
    """A sweep pins one decode bucket (bench/serve_sweep.py --decode-buckets 16). A prefill chunk's MTP call
    holds one sequence (per DP-attention group), so its graph takes one row, the (1, q, P) graph warmup builds:
    it used to be padded to the decode bucket (2 x 8 rows here, 16 x 1024 per group in the GLM-5.3-Flash sweep),
    a graph no warmup built (a runtime compile, 10-16 min at tp=32) over 16 times the rows."""
    from kiln.engine.request import SamplingParams

    name, path, *_ = mtp_model
    if name != "glm_moe_dsa":
        pytest.skip("graph keys do not depend on the MLA family")
    eng = engine(path, max_model_len=64, decode_batch_buckets=(2,), prefill_token_buckets=(8,), page_buckets=(4, 16))
    eng.warmup()
    before = set(eng.runner.compile_seconds)
    eng.generate([[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 53))],
                 SamplingParams(max_new_tokens=16, ignore_eos=True))
    new = set(eng.runner.compile_seconds) - before
    assert not new, new
    assert {k[1] for k in eng.runner.calls if k[0] == "mtp" and k[2] == 8} == {1}  # prefill chunks: one row


def test_draftless_rows_run_in_the_verify_graph(mtp_model):
    """Under MTP a sequence without a draft (its last token: room for no draft) runs in the step's verify
    graph (EngineConfig.spec_verify_plain, default for mtp), so a step makes one launch instead of a verify
    plus a whole decode-bucket step for that one row, and no decode graph is ever built; the output is the
    target's greedy one either way. spec_verify_plain=False keeps the separate decode launch."""
    from kiln.engine.request import SamplingParams

    name, path, *_ = mtp_model
    if name != "glm_moe_dsa":
        pytest.skip("launch structure does not depend on the MLA family")
    prompts = [[5, 9, 11, 200, 3, 77, 12, 13, 14, 15, 16], list(range(40, 53))]
    sp = SamplingParams(max_new_tokens=9, ignore_eos=True)
    want = [r.output_ids for r in engine(path, spec_method=None).generate(prompts, sp)]
    eng = engine(path, decode_batch_buckets=(2,), prefill_token_buckets=(8,), page_buckets=(4, 16), max_model_len=64)
    eng.warmup()
    assert [r.output_ids for r in eng.generate(prompts, sp)] == want
    assert not any(k[0] == "decode" for k in list(eng.runner.compile_seconds) + list(eng.runner.calls))
    assert any(k[0] == "verify" for k in eng.runner.calls)
    old = engine(path, spec_verify_plain=False)
    assert [r.output_ids for r in old.generate(prompts, sp)] == want
    assert any(k[0] == "decode" for k in old.runner.calls)


@pytest.mark.parametrize("share", ["0", "1"])
@pytest.mark.parametrize("name", list(MODELS))
def test_acceptance_with_the_layer_copied_from_the_target(tmp_path, name, share, monkeypatch):
    """A one-layer target whose MTP layer is its own copy (build_with_mtp copy_main): the drafts
    are the target's predictions on a sequence shifted by one, so most are accepted, and the output
    is still the target's greedy one. With index_share_for_mtp_iteration a later pass cannot see the
    keys of the drafts before it (it reuses the first pass's selection, as vLLM does), which this
    construction needs, so fewer are accepted there (measured: 87% -> 36% for GLM-5.3, 89% -> 45%
    for DeepSeek-V3.2, CPU fp32, 2026-10-03)."""
    from kiln.engine.request import SamplingParams

    if share == "1" and name == "deepseek_v3":
        pytest.skip("no DSA indexer to share")
    monkeypatch.setenv("KILN_MTP_INDEX_SHARE", share)
    build_with_mtp(name, str(tmp_path), seed=2, copy_main=True, num_hidden_layers=1, **MODELS[name])
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(0, 384, (n,), generator=g).tolist() for n in (20, 45)]
    sp = SamplingParams(max_new_tokens=24, ignore_eos=True)
    want = [r.output_ids for r in engine(str(tmp_path), spec_method=None).generate(prompts, sp)]
    eng = engine(str(tmp_path), spec_k=3)
    assert [r.output_ids for r in eng.generate(prompts, sp)] == want
    rate = eng.spec_accepted / eng.spec_proposed
    print(f"{name} index share {share}: {eng.spec_accepted}/{eng.spec_proposed} drafts accepted ({rate:.0%})")
    assert rate > (0.5 if share == "0" else 0.2), rate


def test_mtp_graph_rows_beyond_the_dsa_scratch(mtp_model):
    """One decode bucket of 2: the MTP layer selects in its graph instead of through the DSA selection
    scratch (the larger of the prefill bucket and 2 x (1 + k) rows), and the output is still the target's
    greedy one. (A prefill chunk's MTP call took 2 x 8 rows here, more than the scratch, until it was made
    one row per chunk: test_warmup_covers_prefill_drafting_with_pinned_decode_buckets.)"""
    from kiln.engine.request import SamplingParams
    from tests.test_mla import hf_greedy

    name, path, hf, *_ = mtp_model
    eng = engine(path, decode_batch_buckets=(2,))
    assert eng.runner.k_caches[0].shape[0] > 0 and eng.model.layers[0].spec.mla is not None
    prompts = [list(range(40, 75)), [5, 9, 11, 200, 3, 77, 12]]
    sp = SamplingParams(max_new_tokens=10, ignore_eos=True)
    assert [r.output_ids for r in eng.generate(prompts, sp)] == [hf_greedy(hf, p, 10) for p in prompts]
