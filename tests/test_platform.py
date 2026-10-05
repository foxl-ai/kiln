"""kiln/platform.py: target families, LNC and the compiler arguments per chip (no device)."""

import pytest

from kiln import platform


def test_family_of():
    assert platform.family_of("trn1") == "trn1"
    assert platform.family_of("trn1n") == "trn1n"
    assert platform.family_of("trn2") == "trn2"
    assert platform.family_of("trn3pre") == "trn3"
    assert platform.family_of("trn3-rev2") == "trn3"
    assert platform.family_of("inf2") == "inf2"
    assert platform.family_of("trn2.3xlarge") == "trn2"
    assert platform.family_of("trn2u.48xlarge") == "trn2"
    assert platform.family_of("trn1n.32xlarge") == "trn1n"
    with pytest.raises(ValueError):
        platform.family_of("p5.48xlarge")


def test_trn1_args_unchanged(monkeypatch):
    # Exactly what model_runner.neuronx_cc_args produced before kiln/platform.py.
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    assert platform.neuronx_cc_args("trn1", fp8=False) == []
    assert platform.neuronx_cc_args("trn1", fp8=True) == [
        "--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"]
    assert platform.neuronx_cc_args("inf2", fp8=False) == []
    assert platform.lnc("trn1") == 1


def test_trn2_args(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    monkeypatch.delenv("KILN_CC_PRESET", raising=False)
    args = platform.neuronx_cc_args("trn2", fp8=True)
    assert args[0] == "--logical-nc-config=2"
    assert "--auto-cast=none" in args and "-O2" in args
    h2t = [a for a in args if a.startswith("--internal-hlo2tensorizer-options=")]
    # One argument: a second --internal-hlo2tensorizer-options would replace the first.
    assert h2t == ["--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
                   "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"]
    monkeypatch.setenv("NEURON_LOGICAL_NC_CONFIG", "1")
    assert platform.neuronx_cc_args("trn2", fp8=False)[0] == "--logical-nc-config=1"
    monkeypatch.setenv("KILN_CC_PRESET", "lnl")
    assert platform.neuronx_cc_args("trn2", fp8=False) == ["--logical-nc-config=1"]


def test_trn3_never_unsafe_fp8(monkeypatch):
    monkeypatch.delenv("KILN_CC_PRESET", raising=False)
    for t in ("trn3", "trn3pre", "trn3-rev2"):
        args = platform.neuronx_cc_args(t, fp8=True)
        assert not any("unsafe-fp8" in a for a in args)
        assert platform.fp8_max(t) == 448.0
        assert platform.has_mx(t)
    assert platform.fp8_max("trn2") == platform.fp8_max("trn1") == 240.0
    assert not platform.has_mx("trn2")


def test_configure_runtime_env(monkeypatch):
    monkeypatch.delenv("NEURON_LOGICAL_NC_CONFIG", raising=False)
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    platform.configure_runtime_env()
    import os

    assert os.environ["NEURON_LOGICAL_NC_CONFIG"] == "2"
    p = platform.detect()
    assert (p.family, p.lnc, p.nki_gen, p.hbm_gib_per_core, p.logical_cores_per_chip) == ("trn2", 2, 3, 24, 4)
    # Not monkeypatch.delenv: it would record the "2" just set and put it back at teardown (the line
    # at the top recorded the value from before the test, if it had one).
    os.environ.pop("NEURON_LOGICAL_NC_CONFIG")
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    platform.configure_runtime_env()
    assert "NEURON_LOGICAL_NC_CONFIG" not in os.environ  # trn1: nothing set
