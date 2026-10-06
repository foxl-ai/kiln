"""model_runner.prefill_cc_args: the extra neuronx-cc arguments only the piecewise prefill pieces take."""

from types import SimpleNamespace

from kiln.engine import model_runner as mr


def cfg(group, piecewise=True):
    return SimpleNamespace(piecewise=piecewise, piecewise_prefill_moe_group=group)


def test_big_prefill_pieces_get_the_instruction_limit(monkeypatch):
    monkeypatch.delenv("KILN_PREFILL_CC_ARGS", raising=False)
    assert mr.prefill_cc_args(cfg(None)) == []
    assert mr.prefill_cc_args(cfg(12)) == []
    assert mr.prefill_cc_args(cfg(45)) == [mr.PREFILL_BIG_LIMIT]
    assert mr.prefill_cc_args(cfg(45, piecewise=False)) == []


def test_prefill_cc_args_environment_wins(monkeypatch):
    monkeypatch.setenv("KILN_PREFILL_CC_ARGS", "none")
    assert mr.prefill_cc_args(cfg(45)) == []
    monkeypatch.setenv("KILN_PREFILL_CC_ARGS", "-O1 --foo=2")
    assert mr.prefill_cc_args(cfg(12)) == ["-O1", "--foo=2"]
