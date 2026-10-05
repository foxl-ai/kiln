"""bench/serve_sweep.py with MTP speculation: the flags reach the engine, the state pool keeps the baseline's
row count (a graph input shape: with it the target's decode and prefill graphs are the ones the baseline
configuration compiled), and each level reports its acceptance."""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))


def sweep_args(path, *extra):
    import serve_sweep

    return serve_sweep.build_parser().parse_args(
        ["--model", path, "--device", "cpu", "--input-len", "24", "--output-len", "8", "--concurrency", "4",
         "--requests", "4", "--kv-cache-gb", "0.01", "--prefill-tokens", "16", "--max-num-seqs", "4",
         "--decode-buckets", "4", *extra])


def test_sweep_with_mtp_keeps_the_baseline_state_rows_and_reports_acceptance(tmp_path, capsys):
    pytest.importorskip("transformers.models.glm5_next.modeling_glm5_next")
    import serve_sweep

    from tests.test_linear_serving import build_glm5_next_mtp

    path = str(tmp_path)
    build_glm5_next_mtp(path)
    base = serve_sweep.make_engine(sweep_args(path), 0)
    try:
        rows = base.runner.state.rows  # 1 + 4 running + 2 x 4 checkpoint rows
        assert base.cfg.spec_method is None and rows == 13
    finally:
        base.close()
    for k, ckpt in ((1, 4), (2, 0)):
        args = sweep_args(path, "--spec-method", "mtp", "--spec-k", str(k), "--state-checkpoints", str(ckpt))
        eng = serve_sweep.make_engine(args, 0)
        try:
            assert (eng.cfg.spec_method, eng.cfg.spec_k, eng.cfg.state_checkpoints) == ("mtp", k, ckpt)
            assert eng.runner.state.rows == rows and eng.runner.state.per_req == 1 + k
            g = random.Random(k)  # serve_sweep.prompter draws ids from 1000 up; this vocabulary is 384
            prompt, params = (lambda: [g.randrange(eng.mcfg.vocab_size) for _ in range(24)]), serve_sweep.sampling(args)
            recs = serve_sweep.run_level(eng, args, 4, 4, prompt, params)[0]
            assert len(recs) == 4 and all(r[2] == 8 for r in recs)  # (TTFT, ITL, output tokens, ...)
            assert eng.spec_verifies > 0 and eng.spec_pos_reached[0] == eng.spec_verifies
        finally:
            eng.close()
        out = capsys.readouterr().out
        assert f"spec: mtp k={k} conc 4: " in out and "tokens per verify" in out, out
