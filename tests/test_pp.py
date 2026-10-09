"""One long prefill over several engines (engine/pp.py, EngineConfig.pp_stages): 2 and 3 stage engines, each its own
process with its own layer range, against one engine on the tiny GLM-5.3 config (tests/test_glm5_next.py's builder,
transformers >= 5.18) at prompts past the long path's threshold: the last stage's first tokens and prompt logprobs
equal one engine's, every stage's own layers' caches and states untouched by the others."""

import multiprocessing as mp
import os
import socket

import pytest
import torch


class _Lazy:
    """tests/test_glm5_next.py (transformers >= 5.18) on first use, so a 5.15 run collects this file and skips."""

    def __getattr__(self, name):
        if name.startswith("__") or name == "pytestmark":  # pytest's collection probes module objects: not a use
            raise AttributeError(name)
        pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
        from tests import test_glm5_next

        return getattr(test_glm5_next, name)


glm = _Lazy()


def _free_port_block(n: int) -> int:
    """A base port b with b .. b + n - 1 free a moment ago."""
    for _ in range(100):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            b = s.getsockname()[1]
        ok = True
        for p in range(b, b + n):
            with socket.socket() as t:
                try:
                    t.bind(("127.0.0.1", p))
                except OSError:
                    ok = False
                    break
        if ok:
            return b
    raise RuntimeError("no free port block")


def _stage(path, prompts, kw, stage, stages, split, ports, q, env):
    os.environ.update(env)
    from kiln.engine.request import SamplingParams

    sp = SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0)
    pkw = dict(pp_stage=stage, pp_stages=stages, pp_split=split,
               pp_listen=f"127.0.0.1:{ports[stage]}" if stage > 0 else None,
               pp_next=f"127.0.0.1:{ports[stage + 1]}" if stage < stages - 1 else None)
    eng = glm.engine(path, **kw, **pkw)
    try:
        out = eng.generate(prompts, sp)
        q.put((stage, [(r.output_ids, [x[0] for x in r.logprobs],
                        [v[0] for _, v in sorted(r.prompt_logprobs.items())]) for r in out]))
    finally:
        eng.close()


@pytest.mark.parametrize("stages,split,mode", [(2, None, "sync"), (3, (2, 5), "sync"), (3, (2, 5), "async"),
                                              (3, (2, 5), "overlap")])
def test_pipeline_stages_equal_one_engine(tmp_path, stages, split, mode, monkeypatch):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams

    path = str(tmp_path)
    glm.build(path, seed=3, index_topk=16, max_position_embeddings=4096)
    # the long path past 16 keys: every prompt here takes it; KILN_PP_ASYNC: the hops on the link threads, and with
    # KILN_PP_OVERLAP the stages' engines overlap-schedule (kw overlap=True on the stages only)
    env = {"KILN_DSA_LONG_KEYS": "16", "KILN_PP_ASYNC": "1" if mode != "sync" else "0",
           "KILN_PP_OVERLAP": "1" if mode == "overlap" else "0"}
    # dsa_long reads KILN_DSA_LONG_KEYS once, at import: import it BEFORE the env is set, or a first import here makes
    # 16 its "original" value and every later test in the process (one pytest process, release/suite.sh) runs the long
    # path (test_disagg's tp engine aborted in mla._long_prefill_select).
    from kiln.models import dsa_long

    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
    monkeypatch.setenv("KILN_DSA_LONG_KEYS", "16")
    ps = glm.prompts(7, (70, 41, 97))
    kw = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4, piecewise=True, piecewise_group=1)
    eng = glm.engine(path, **kw)
    try:
        ref = eng.generate(ps, SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0))
        want = [(r.output_ids, [x[0] for x in r.logprobs], [v[0] for _, v in sorted(r.prompt_logprobs.items())])
                for r in ref]
    finally:
        eng.close()
    base = _free_port_block(stages)
    ports = [base + s for s in range(stages)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    skw = dict(kw, overlap=True) if mode == "overlap" else kw
    procs = [ctx.Process(target=_stage, args=(path, ps, skw, s, stages, split, ports, q, env)) for s in range(stages)]
    for p in procs:
        p.start()
    got = {}
    for _ in range(stages):
        s, out = q.get(timeout=600)
        got[s] = out
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    last = got[stages - 1]
    for (t_w, lp_w, plp_w), (t_g, lp_g, plp_g) in zip(want, last):
        assert t_g == t_w
        torch.testing.assert_close(torch.tensor(plp_g), torch.tensor(plp_w), rtol=0, atol=2e-5)
        torch.testing.assert_close(torch.tensor(lp_g), torch.tensor(lp_w), rtol=0, atol=2e-5)


def test_default_split_by_weight():
    from kiln.engine import pp

    kinds = [False, False, False, True] * 11 + [False]  # GLM-5.3-Flash: a DSA layer every 4th (45 layers)
    s2 = pp.default_split(kinds, 2)
    s4 = pp.default_split(kinds, 4)
    assert len(s2) == 1 and len(s4) == 3 and all(a < b for a, b in zip(s4, s4[1:]))
    w = [pp.DSA_WEIGHT if k else 1.0 for k in kinds]
    for split, n in ((s2, 2), (s4, 4)):
        bounds = (0, *split, len(kinds))
        loads = [sum(w[a:b]) for a, b in zip(bounds, bounds[1:])]
        assert max(loads) - min(loads) <= pp.DSA_WEIGHT + 3, (split, loads)
    assert pp.stage_range(45, s4, 0)[0] == 0 and pp.stage_range(45, s4, 3)[1] == 45


def _stage_mla(path, prompts, kw, stage, stages, split, ports, q, env):
    """One stage of a DeepSeek-V3 pipeline (tests/test_mla.py's builder, transformers 5.15): its tokens, logprobs and
    prompt logprobs, the layer range it loaded, and which decoder layers' parameters are on the meta device."""
    os.environ.update(env)
    from kiln.engine.request import SamplingParams
    from tests import test_mla

    sp = SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0)
    pkw = dict(pp_stage=stage, pp_stages=stages, pp_split=split,
               pp_listen=f"127.0.0.1:{ports[stage]}" if stage > 0 else None,
               pp_next=f"127.0.0.1:{ports[stage + 1]}" if stage < stages - 1 else None)
    eng = test_mla.engine(path, **kw, **pkw)
    try:
        model = eng.runner.model
        meta = [i for i, l in enumerate(model.layers) if any(p.device.type == "meta" for p in l.parameters())]
        loaded = sum(p.numel() for p in model.parameters() if p.device.type != "meta")
        out = eng.generate(prompts, sp)
        q.put((stage, model.load_range, meta, loaded,
               [(r.output_ids, [x[0] for x in r.logprobs], [v[0] for _, v in sorted(r.prompt_logprobs.items())])
                for r in out]))
    finally:
        eng.close()


def test_stage_loads_only_its_layers_deepseek_v3(tmp_path):
    """Stage-only weight loading is the loader's (model-generic) layer filter: a 2-stage DeepSeek-V3 pipeline (MLA, a
    dense first layer, routed experts) where each stage reads and places only its own layers' weights, the others left
    on the meta device, and the last stage still gives one engine's first tokens, logprobs and prompt logprobs. The
    path a 671B FP8 DeepSeek-V3 / V3.2 needs on trn1 (it does not fit one trn1.32xlarge: two stages of ~335 GB do)."""
    from kiln.engine.request import SamplingParams
    from tests import test_mla

    path = str(tmp_path)
    test_mla.build("deepseek_v3", path, seed=5)  # 4 layers: one dense, three MoE
    g = torch.Generator().manual_seed(9)
    ps = [torch.randint(2, 384, (n,), generator=g).tolist() for n in (37, 22, 51)]
    kw = dict(max_prefill_tokens=8, piecewise=True, piecewise_group=1)
    eng = test_mla.engine(path, **kw)
    try:
        full = sum(p.numel() for p in eng.runner.model.parameters())
        ref = eng.generate(ps, SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0))
        want = [(r.output_ids, [x[0] for x in r.logprobs], [v[0] for _, v in sorted(r.prompt_logprobs.items())])
                for r in ref]
    finally:
        eng.close()
    stages, split = 2, (2,)
    base = _free_port_block(stages)
    ports = [base + s for s in range(stages)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_stage_mla, args=(path, ps, kw, s, stages, split, ports, q, {}))
             for s in range(stages)]
    for p in procs:
        p.start()
    got = {}
    for _ in range(stages):
        s, rng, meta, loaded, out = q.get(timeout=600)
        got[s] = (rng, meta, loaded, out)
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    assert got[0][0] == (0, 2) and got[0][1] == [2, 3]  # stage 0 loaded layers 0-1, layers 2-3 never read
    assert got[1][0] == (2, 4) and got[1][1] == [0, 1]
    assert got[0][2] < full and got[1][2] < full  # each stage holds its own layers (plus embeddings and head)
    for (t_w, lp_w, plp_w), (t_g, lp_g, plp_g) in zip(want, got[1][3]):
        assert t_g == t_w
        torch.testing.assert_close(torch.tensor(plp_g), torch.tensor(plp_w), rtol=0, atol=2e-5)
        torch.testing.assert_close(torch.tensor(lp_g), torch.tensor(lp_w), rtol=0, atol=2e-5)



def _stage_follow(path, prompts, kw, stage, stages, split, ports, q, env):
    """One stage of a FOLLOWING pipeline (pp_follow): stage 0 takes the requests (generate), the others run
    LLMEngine.follow until stage 0 closes the run; every stage reports what it finished, by request id."""
    os.environ.update(env)
    from kiln.engine.request import SamplingParams

    sp = SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0)
    pkw = dict(pp_stage=stage, pp_stages=stages, pp_split=split, pp_follow=True,
               pp_listen=f"127.0.0.1:{ports[stage]}" if stage > 0 else None,
               pp_next=f"127.0.0.1:{ports[stage + 1]}" if stage < stages - 1 else None)
    eng = glm.engine(path, **kw, **pkw)
    out = {}
    try:
        if stage == 0:
            for r in eng.generate(prompts, sp):
                out[r.rid] = (r.output_ids, [x[0] for x in r.logprobs], [v[0] for _, v in sorted(r.prompt_logprobs.items())])
        else:
            go = True
            while go:
                go, done = eng.follow(0.2)
                for r in done:
                    out[r.rid] = (r.output_ids, [x[0] for x in r.logprobs],
                                  [v[0] for _, v in sorted(r.prompt_logprobs.items())])
    finally:
        eng.close()
    q.put((stage, out))


@pytest.mark.parametrize("stages,split,mode", [(3, (2, 5), "async"), (3, (2, 5), "overlap")])
def test_following_stages_equal_one_engine(tmp_path, stages, split, mode, monkeypatch):
    """Only stage 0 is given the requests (all at once, so steps hold several prefill calls); the later stages learn
    them from stage 0's plan frames and must run the same plan: the last stage's first tokens, logprobs and prompt
    logprobs equal one engine's, and every stage finished every request."""
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams

    path = str(tmp_path)
    glm.build(path, seed=3, index_topk=16, max_position_embeddings=4096)
    env = {"KILN_DSA_LONG_KEYS": "16", "KILN_PP_ASYNC": "1", "KILN_PP_OVERLAP": "1" if mode == "overlap" else "0"}
    # dsa_long reads KILN_DSA_LONG_KEYS once, at import: import it BEFORE the env is set, or a first import here makes
    # 16 its "original" value and every later test in the process (one pytest process, release/suite.sh) runs the long
    # path (test_disagg's tp engine aborted in mla._long_prefill_select).
    from kiln.models import dsa_long

    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
    monkeypatch.setenv("KILN_DSA_LONG_KEYS", "16")
    ps = glm.prompts(7, (70, 41, 97, 13))
    kw = dict(page_size=8, max_prefill_tokens=12, max_num_seqs=4, piecewise=True, piecewise_group=1)
    eng = glm.engine(path, **kw)
    try:
        ref = eng.generate(ps, SamplingParams(max_new_tokens=1, ignore_eos=True, logprobs=0, prompt_logprobs=0))
        want = [(r.output_ids, [x[0] for x in r.logprobs], [v[0] for _, v in sorted(r.prompt_logprobs.items())])
                for r in ref]
    finally:
        eng.close()
    base = _free_port_block(stages)
    ports = [base + s for s in range(stages)]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    skw = dict(kw, overlap=True) if mode == "overlap" else kw
    procs = [ctx.Process(target=_stage_follow, args=(path, ps, skw, s, stages, split, ports, q, env))
             for s in range(stages)]
    for p in procs:
        p.start()
    got = {}
    for _ in range(stages):
        s, out = q.get(timeout=600)
        got[s] = out
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0
    rids = sorted(got[0], key=lambda r: int(r.split("-")[1]))
    assert all(sorted(got[s]) == sorted(rids) for s in range(stages))  # every stage finished every request
    last = got[stages - 1]
    for rid, (t_w, lp_w, plp_w) in zip(rids, want):
        t_g, lp_g, plp_g = last[rid]
        assert t_g == t_w
        torch.testing.assert_close(torch.tensor(plp_g), torch.tensor(plp_w), rtol=0, atol=2e-5)
        torch.testing.assert_close(torch.tensor(lp_g), torch.tensor(lp_w), rtol=0, atol=2e-5)
