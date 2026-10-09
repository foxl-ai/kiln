"""bench/serve_sweep.py device_split: the least-squares split of a level's step times into a cost per decode call,
per prefill call and per step. A lone request makes one call per step, so the per-step cost is not identifiable;
the split must then give each call its whole cost instead of the rank-deficient minimum-norm answer (measured on
trn2: prefill 0.786 s + per step 0.411 s + decode -0.375 s for 1.238 s prefill calls and ~33 ms decode calls)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))


def _steps(launches: list[tuple[int, int]], cost, overlap: bool) -> list[tuple]:
    """step() records (wall, decode calls, prefill calls, chunks, tokens, deferred). Under overlap a step's wall is
    the device time of the step launched by the call before it."""
    out = []
    for i, (d, p) in enumerate(launches):
        src = launches[i - 1] if overlap and i else (d, p)
        out.append((cost(*src) if (not overlap or i) else 0.001, d, p, p, 4096 * p, 0))
    return out


@pytest.mark.parametrize("overlap", [True, False])
def test_lone_request_gives_each_call_its_whole_cost(overlap):
    import serve_sweep

    launches = [(0, 1)] * 8 + [(1, 0)] * 128  # a lone 32k prompt in 4096-token chunks, then 128 decode steps
    s = serve_sweep.device_split(_steps(launches, lambda d, p: 0.033 * d + 1.238 * p, overlap), overlap)
    assert s["prefill_call_s"] == pytest.approx(1.238, abs=1e-6)
    assert s["decode_call_s"] == pytest.approx(0.033, abs=1e-6)
    assert s["step_s"] == 0.0 and s["per_step_separable"] is False


def test_mixed_steps_still_separate_the_per_step_cost():
    import serve_sweep

    launches = [(0, 1), (2, 0), (1, 1), (3, 0), (2, 1), (1, 0), (0, 2), (4, 1), (2, 2), (3, 1)] * 4
    s = serve_sweep.device_split(_steps(launches, lambda d, p: 0.01 + 0.03 * d + 0.5 * p, True), True)
    assert s["per_step_separable"] is True
    assert (s["decode_call_s"], s["prefill_call_s"], s["step_s"]) == pytest.approx((0.03, 0.5, 0.01), abs=1e-6)
