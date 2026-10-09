"""kiln/precompile.py: capture every warmup graph from the engine's own EngineConfig, then compile them in parallel.

The capture runs where libtorch_neuronx_lite is installed (the Neuron DLAMI venv, CPU hosts included); the compile
where neuronx-cc is on PATH. The device-side check (every warmup graph a cache hit, and the start time) is a
measurement on an instance, recorded in docs/neuron-notes.md.
"""

import importlib.util
import os
import shutil
import sys

import pytest

from kiln.config import precompile_workers_env

HAS_LNL = importlib.util.find_spec("libtorch_neuronx_lite") is not None


def test_workers_env_parses_off_auto_and_counts(monkeypatch):
    monkeypatch.delenv("KILN_PRECOMPILE_WORKERS", raising=False)
    assert precompile_workers_env() == 0
    for off in ("0", "off", "false", ""):
        assert precompile_workers_env(off) == 0
    assert precompile_workers_env("6") == 6
    assert precompile_workers_env("auto") == max(1, (os.cpu_count() or 2) // 2)
    with pytest.raises(ValueError):
        precompile_workers_env("-1")
    monkeypatch.setenv("KILN_PRECOMPILE_WORKERS", "3")
    from kiln.config import EngineConfig

    assert EngineConfig(model_path="x").precompile_workers == 3


def _sweep_cfg(ckpt: str):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "bench"))
    import serve_sweep

    args = serve_sweep.build_parser().parse_args(
        ["--model", ckpt, "--tp", "2", "--piecewise", "--concurrency", "2", "--input-len", "8", "--output-len", "4",
         "--decode-buckets", "2", "--page-buckets", "2", "--prefill-buckets", "8", "--kv-cache-gb", "0.01"])
    return serve_sweep.engine_config(args)


@pytest.mark.skipif(not HAS_LNL, reason="needs libtorch_neuronx_lite (the Neuron DLAMI venv)")
def test_capture_all_writes_the_compile_farm_keys(tmp_path, monkeypatch):
    """The engine-config capture finds the same 6 graphs as tools/compile_farm.py capture on the same sweep
    (tests/test_compile_farm.py test_capture_tp2_writes_one_entry_per_graph), on both ranks, none from the build;
    then, where neuronx-cc runs, compile_missing completes every one of them in parallel."""
    from tests.test_architectures import build

    from kiln import compile_cache, compile_farm, precompile
    from kiln.config import ModelConfig
    from kiln.engine.engine import pool_pages

    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(tmp_path / "cache"))
    ckpt = tmp_path / "ckpt"
    build("qwen3", str(ckpt))
    cfg = _sweep_cfg(str(ckpt))
    num_pages = pool_pages(cfg, ModelConfig.from_pretrained(str(ckpt)))
    work = tmp_path / "work"
    work.mkdir()
    cap = precompile.capture_all(cfg, str(ckpt), "trn1", num_pages, str(work), procs=1)
    assert len(cap["keys"]) == 6  # decode and prefill: prep, one layer group, post
    assert all(ranks == [0, 1] for ranks in cap["keys"].values())
    assert cap["build_keys"] == []
    root = compile_cache.local_root()
    for k in cap["keys"]:
        files = set(os.listdir(os.path.join(root, k)))
        assert {"graph.hlo", ".artifact_metadata_v0.json", "command.txt"} <= files
    if shutil.which("neuronx-cc") is None:
        pytest.skip("neuronx-cc is not on PATH: capture checked, compile not")
    comp = precompile.compile_missing(cap["keys"], workers=3)
    assert comp["compiled"] == 6 and comp["already"] == 0
    for k in cap["keys"]:
        assert os.path.exists(os.path.join(root, k, compile_farm.MARKER))
    again = precompile.compile_missing(cap["keys"], workers=3)  # every entry complete: nothing to do
    assert again["compiled"] == 0 and again["already"] == 6


@pytest.mark.skipif(not HAS_LNL or shutil.which("neuronx-cc") is None,
                    reason="needs libtorch_neuronx_lite and neuronx-cc (the Neuron DLAMI venv)")
def test_run_writes_a_manifest_and_a_warm_start_skips_the_capture(tmp_path, monkeypatch):
    """run(): capture + compile, then the manifest of the configuration's keys. A Precompile of the same
    configuration finds every key complete and spawns nothing; a different configuration (another kv_cache_gb, so
    other shapes) has no manifest."""
    import dataclasses
    import json

    from tests.test_architectures import build

    from kiln import precompile
    from kiln.config import ModelConfig
    from kiln.engine.engine import pool_pages

    monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn1")
    ckpt = tmp_path / "ckpt"
    build("qwen3", str(ckpt))
    cfg = dataclasses.replace(_sweep_cfg(str(ckpt)), precompile_workers=3)
    num_pages = pool_pages(cfg, ModelConfig.from_pretrained(str(ckpt)))
    res = tmp_path / "rec.json"
    precompile.run(cfg, str(ckpt), "trn1", num_pages, str(res))
    rec = json.loads(res.read_text())
    assert rec["ok"], rec.get("error")
    assert rec["graphs"] == 6 and rec["compile"]["compiled"] == 6
    keys = precompile.manifest_complete(precompile.fingerprint(cfg, str(ckpt), "trn1"))
    assert keys is not None and len(keys) == 6
    warm = precompile.Precompile(cfg, str(ckpt), num_pages)
    assert warm.proc is None and warm.wait()["skipped"] == "manifest"
    other = dataclasses.replace(cfg, kv_cache_gb=0.02)
    assert precompile.manifest_complete(precompile.fingerprint(other, str(ckpt), "trn1")) is None
