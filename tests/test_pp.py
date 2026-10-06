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


@pytest.mark.parametrize("stages,split", [(2, None), (3, (2, 5))])
def test_pipeline_stages_equal_one_engine(tmp_path, stages, split, monkeypatch):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    from kiln.engine.request import SamplingParams

    path = str(tmp_path)
    glm.build(path, seed=3, index_topk=16, max_position_embeddings=4096)
    env = {"KILN_DSA_LONG_KEYS": "16"}  # the long path past 16 keys: every prompt here takes it
    monkeypatch.setenv("KILN_DSA_LONG_KEYS", "16")
    from kiln.models import dsa_long

    monkeypatch.setattr(dsa_long, "LONG_KEYS", 16)
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
    procs = [ctx.Process(target=_stage, args=(path, ps, kw, s, stages, split, ports, q, env)) for s in range(stages)]
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
