"""Compile farm (kiln/compile_farm.py) and device-free capture (kiln/capture.py), on CPU.

The compile itself needs neuronx-cc and is measured on instances (docs/neuron-notes.md "Compile
farm"); here: the command and kernel-reference parsing the farm relies on, that a shape directory
(safetensors headers only) loads into the same parameter shapes and dtypes as the checkpoint,
and, where libtorch_neuronx_lite is installed, a capture of a tiny model at tp=2 in a subprocess.
"""

import base64
import importlib.util
import json
import os
import subprocess
import sys

import pytest
import torch

# An LNL command.txt from kiln-g1-trn1 (GLM-5.3-Flash, tp=32, 2026-10-03), verbatim.
COMMAND = ("/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0/bin/neuronx-cc compile "
           "/root/.cache/neuron_libtorch/neuron/compile_cache/66437b3432fb818ccebad2e5a8628757/graph.hlo "
           "--framework XLA --target trn1 --output /root/.cache/neuron_libtorch/neuron/compile_cache/"
           "66437b3432fb818ccebad2e5a8628757/graph_66437b3432fb818ccebad2e5a8628757.neff --logfile "
           "/root/.cache/neuron_libtorch/neuron/compile_cache/66437b3432fb818ccebad2e5a8628757/log-neuron-cc.txt "
           "--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3")


def test_parse_command_keeps_only_the_compiler_args():
    from kiln.compile_farm import compiler_command, parse_command

    target, args = parse_command(COMMAND)
    assert target == "trn1"
    assert args == ["--internal-hlo2tensorizer-options=--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"]
    cmd = compiler_command("neuronx-cc", "/c/k/graph.hlo", target, "/c/k/graph_k.neff", "/c/k/log", args)
    assert parse_command(" ".join(cmd)) == (target, args)
    with pytest.raises(ValueError):
        parse_command("neuronx-cc list-operators --framework XLA")


def test_kernel_refs_reads_backend_configs_inside_binary_hlo(tmp_path):
    from kiln.compile_farm import kernel_refs, missing_kernels

    present = tmp_path / "kernel_a.colz"
    present.write_bytes(b"x")
    cfgs = [{"func_name": "kiln.kernels.moe_dedupe.kiln_moe_dedupe_v8", "kernel_format": "colz",
             "klir_binary": {"binary": str(present)}},
            {"func_name": "k2", "klir_binary": {"binary": "/var/tmp/nki-intermediate-cache/none/k2.colz"}},
            {"func_name": "no binary here"}]
    blob = b"\x08\x01\x12\x07HloModule\x1a"
    for c in cfgs:
        blob += b"\x22\x9a" + base64.b64encode(json.dumps(c).encode()) + b"\x00\x18AwsNeuronCustomNativeKernel"
    hlo = tmp_path / "graph.hlo"
    hlo.write_bytes(blob)
    assert kernel_refs(str(hlo)) == {str(present), "/var/tmp/nki-intermediate-cache/none/k2.colz"}
    assert missing_kernels(str(hlo)) == ["/var/tmp/nki-intermediate-cache/none/k2.colz"]


def test_entry_args_reads_command_txt(tmp_path):
    from kiln.compile_farm import entry_args

    (tmp_path / "command.txt").write_text(COMMAND)
    assert entry_args(str(tmp_path))[0] == "trn1"


def _params(model):
    return {n: (tuple(t.shape), t.dtype) for n, t in [*model.named_parameters(), *model.named_buffers()]}


@pytest.mark.parametrize("arch", ["qwen3", "qwen3_moe"])
def test_shape_dir_loads_the_same_parameters(tmp_path, arch):
    """The capture loads a model from safetensors headers alone; every parameter and buffer must
    get the shape and dtype the real checkpoint gives it, at tp=1 and on a tp=2 rank."""
    from tests.test_architectures import build

    from kiln import capture
    from kiln.config import ModelConfig
    from kiln.models.loader import load_model

    ckpt, shapes = tmp_path / "ckpt", tmp_path / "shapes"
    build(arch, str(ckpt))
    capture.write_shape_dir(str(ckpt), str(shapes))
    assert not list(shapes.glob("*.safetensors"))
    capture.install_header_checkpoint()
    mcfg = ModelConfig.from_pretrained(str(ckpt))
    for rank, tp in ((0, 1), (1, 2)):
        kw = dict(tp_rank=rank, tp_size=tp)
        real = load_model(str(ckpt), mcfg, torch.float32, torch.device("cpu"), 64, **kw)
        fake = load_model(str(shapes), ModelConfig.from_pretrained(str(shapes)), torch.float32,
                          torch.device("meta"), 64, **kw)
        assert _params(fake) == _params(real)
        assert all(t.device.type == "meta" for t in fake.parameters())


def test_header_slices_have_the_sliced_shape():
    from kiln.capture import _HeaderHandle

    h = _HeaderHandle({"w": ["BF16", [6, 4]], "s": ["F32", []]})
    assert h.get_slice("w")[slice(2, 5)].shape == (3, 4)
    assert h.get_slice("w")[slice(2, 5)].dtype == torch.bfloat16
    assert h.get_slice("w").get_dtype() == "BF16"
    assert h.get_tensor("s").shape == ()


@pytest.mark.skipif(importlib.util.find_spec("libtorch_neuronx_lite") is None,
                    reason="needs libtorch_neuronx_lite (the Neuron DLAMI venv)")
def test_capture_tp2_writes_one_entry_per_graph(tmp_path):
    """tools/compile_farm.py capture of a tiny Qwen3 at tp=2 (fake process group, meta device): both
    ranks trace the same graphs, and every entry carries what the farm compiles from."""
    from tests.test_architectures import build

    from kiln import capture

    ckpt, shapes, out = tmp_path / "ckpt", tmp_path / "shapes", tmp_path / "out"
    build("qwen3", str(ckpt))
    capture.write_shape_dir(str(ckpt), str(shapes))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "NEURON_LIBTORCH_CACHE_ROOT": str(tmp_path / "cache"), "PYTHONPATH": root}
    subprocess.run([sys.executable, os.path.join(root, "tools", "compile_farm.py"), "capture", "--shape-dir",
                    str(shapes), "--out-dir", str(out), "--", "--model", str(ckpt), "--tp", "2", "--piecewise",
                    "--concurrency", "2", "--input-len", "8", "--output-len", "4", "--decode-buckets", "2",
                    "--page-buckets", "2", "--prefill-buckets", "8", "--kv-cache-gb", "0.01"],
                   check=True, env=env, timeout=900)
    keys = json.loads((out / "keys.json").read_text())
    assert len(keys) == 6  # decode and prefill: prep, one layer group, post
    assert all(ranks == [0, 1] for ranks in keys.values())
    cache = tmp_path / "cache" / "neuron" / "compile_cache"
    for k in keys:
        files = set(os.listdir(cache / k))
        assert {"graph.hlo", ".artifact_metadata_v0.json", "command.txt"} <= files
        assert ".compilation_complete" not in files


@pytest.mark.skipif(importlib.util.find_spec("libtorch_neuronx_lite") is None,
                    reason="needs libtorch_neuronx_lite (the Neuron DLAMI venv)")
def test_capture_traces_the_linear_attention_kernel_on_meta(tmp_path):
    """A capture runs on the meta device, which is not cpu, so it traces the device's prefill path:
    KDA layers the NKI delta-rule kernel takes (head dims 128, a gate lower bound) call it with
    KILN_LINEAR_ATTN_KERNEL unset (the default since engine-v0 77862e6) and not with
    KILN_LINEAR_ATTN_KERNEL=torch, the keys differ, and each config.json names the value it used."""
    from tests.test_linear_attn import build_kda

    from kiln import capture
    from kiln.compile_farm import kernel_configs

    ckpt, shapes = tmp_path / "ckpt", tmp_path / "shapes"
    ckpt.mkdir()
    build_kda(str(ckpt), heads=2, dim=128, hidden=128, layers=2)
    capture.write_shape_dir(str(ckpt), str(shapes))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    got = {}
    for mode in ("default", "torch"):
        env = {k: v for k, v in os.environ.items() if k != "KILN_LINEAR_ATTN_KERNEL"}
        env.update(NEURON_LIBTORCH_CACHE_ROOT=str(tmp_path / f"cache-{mode}"), PYTHONPATH=root)
        if mode == "torch":
            env["KILN_LINEAR_ATTN_KERNEL"] = "torch"
        out = tmp_path / f"out-{mode}"
        subprocess.run([sys.executable, os.path.join(root, "tools", "compile_farm.py"), "capture", "--shape-dir",
                        str(shapes), "--out-dir", str(out), "--", "--model", str(ckpt), "--tp", "2", "--piecewise",
                        "--concurrency", "2", "--input-len", "8", "--output-len", "4", "--decode-buckets", "2",
                        "--page-buckets", "2", "--prefill-buckets", "8", "--kv-cache-gb", "0.01"],
                       check=True, env=env, timeout=1200)
        keys = json.loads((out / "keys.json").read_text())
        cache = tmp_path / f"cache-{mode}" / "neuron" / "compile_cache"
        funcs = {k: {c.get("func_name", "") for c in kernel_configs(str(cache / k / "graph.hlo"))} for k in keys}
        got[mode] = (set(keys), funcs, json.loads((out / "config.json").read_text())["env"])
    (kd, fd, ed), (kt, ft, et) = got["default"], got["torch"]
    assert any("delta_rule" in f for fs in fd.values() for f in fs)
    assert not any("delta_rule" in f for fs in ft.values() for f in fs)
    assert kd != kt
    assert ed["KILN_LINEAR_ATTN_KERNEL"] == "nki" and et["KILN_LINEAR_ATTN_KERNEL"] == "torch"


def test_warmup_plp_builds_the_prompt_logprobs_prefill(tmp_path):
    """warmup(plp=True) runs every prefill bucket's prompt-logprobs variant with the argument list
    ModelRunner.prefill builds for scored rows (on CPU the graphs run eagerly, so a wrong list
    fails here), under the key a check_ppl run uses."""
    from tests.test_architectures import build

    from kiln.config import EngineConfig
    from kiln.engine.engine import LLMEngine
    from kiln.engine.request import SamplingParams

    build("qwen3", str(tmp_path))
    eng = LLMEngine(EngineConfig(model_path=str(tmp_path), device="cpu", dtype=torch.float32, page_size=4,
                                 num_pages=64, max_num_seqs=2, max_model_len=64, max_prefill_tokens=8,
                                 decode_batch_buckets=(2,), prefill_token_buckets=(8,), page_buckets=(4,)))
    eng.runner.warmup(plp=True)
    assert ("prefill", 8, 4, "plp") in eng.runner.compile_seconds
    r = eng.generate([[5, 6, 7, 8, 9]], SamplingParams(max_new_tokens=1, prompt_logprobs=0))[0]
    assert ("prefill", 8, 4, "plp") in eng.runner.calls and len(r.prompt_logprobs) == 4


def test_parse_key_records_reads_the_queue_listing():
    """Per-key queue records (keys/<key>__<bytes>), as `aws s3 ls` prints them; the device side
    (KILN_COMPILE_FARM) and the workers both read the queue through this."""
    from kiln.compile_cache import parse_key_records

    ls = ("2026-10-04 03:01:02          0 062c22f800b3ac3b6a0907765ce285c6__1838540\n"
          "2026-10-04 03:01:03          0 862fd0e0942c3e2b9f361dd5ca9d66c7__1150683\n"
          "                           PRE stray/\n")
    assert parse_key_records(ls) == {"062c22f800b3ac3b6a0907765ce285c6": 1838540,
                                     "862fd0e0942c3e2b9f361dd5ca9d66c7": 1150683}
    assert parse_key_records("") == {}


@pytest.mark.parametrize("fmt", ["glm", "glm-rows", "glm-down", "mimo", "loaded"])
def test_decide_one_expert_at_a_time_matches_the_whole_blob(fmt):
    """kiln/capture.decide reads experts until both answers are known; they must be check_blob and
    `down_factors is not None` over every expert at once, as DecoderLayer.pack_experts asks them."""
    from tests.test_moe_prefill import checkpoint_experts, experts

    from kiln import capture
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_prefill as mp

    if fmt == "loaded":
        ws = list(checkpoint_experts()[0])
    else:
        ws = list(experts("mimo" if fmt == "mimo" else "glm"))
    if fmt == "glm-rows":  # one gate row of the last expert with a scale of its own
        ws[1] = ws[1].clone()
        ws[1][5, 3, 0] *= 2
    if fmt == "glm-down":  # one down column of the last expert at twice its chunk's scale
        ws[3] = ws[3].clone()
        ws[3][5, 0, 7] *= 2
    whole = mdd.pack(*ws)
    want = (mp.check_blob(whole, 1024), mp.down_factors(whole, 1024) is not None)
    got = capture.decide((mdd.pack(*(w[e:e + 1] for w in ws)) for e in range(6)), 1024)
    assert got[:2] == want
    assert got[2] == {"glm": 6, "glm-rows": 6, "glm-down": 6, "mimo": 1, "loaded": 1}[fmt]


def _glm_expert_checkpoint(path, refit: bool, E=4, H=256, I=128):
    """config.json of a GLM-5.3-shaped model (tests/test_mla.py hf_config) and one MoE layer's
    routed experts stored as zai-org/GLM-5.3-Flash stores them: one tensor per expert and
    projection, FP8 e4m3fn with 128 x 128 block scales (weight_scale_inv). refit: codes up to 448
    (the checkpoint's range), so models/quant.fit_e4m3_max halves some rows for trn1's 240."""
    from safetensors.torch import save_file
    from tests.test_mla import hf_config

    c = hf_config("glm_moe_dsa", hidden_size=H, moe_intermediate_size=I, n_routed_experts=E).to_dict()
    c["architectures"] = ["GlmMoeDsaForCausalLM"]
    c["quantization_config"] = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
                                "weight_block_size": [128, 128]}
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(c, f)
    g = torch.Generator().manual_seed(7)
    t, pre = {}, "model.layers.1.mlp."
    for e in range(E):
        for proj, (n, k) in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
            w = torch.randn(n, k, generator=g) * 0.02
            s = w.abs().view(n // 128, 128, k // 128, 128).amax((1, 3)) / (448.0 if refit else 200.0)
            t[f"{pre}experts.{e}.{proj}.weight"] = (w / s.repeat_interleave(128, 0).repeat_interleave(128, 1)).to(
                torch.float8_e4m3fn)
            t[f"{pre}experts.{e}.{proj}.weight_scale_inv"] = s.contiguous()
    t[pre + "gate.weight"] = torch.randn(E, H, generator=g).bfloat16()
    t[pre + "gate.e_score_correction_bias"] = torch.randn(E, generator=g)
    save_file(t, os.path.join(path, "model.safetensors"))


@pytest.mark.parametrize("refit", [True, False])
def test_layer_decisions_match_the_loaded_layer(tmp_path, refit):
    """The decisions kiln/capture computes for a capture (one expert of the rank through the loader,
    every other expert zeros) are the ones the device's load reaches with every expert loaded:
    loader._load_experts on the real checkpoint, moe_dedupe.pack, check_blob and down_factors."""
    import types

    from kiln import capture
    from kiln.config import ModelConfig
    from kiln.kernels import moe_dedupe as mdd
    from kiln.kernels import moe_prefill as mp
    from kiln.models import loader
    from kiln.models.quant import FP8

    _glm_expert_checkpoint(str(tmp_path), refit)
    cfg = ModelConfig.from_pretrained(str(tmp_path))
    assert cfg.num_experts == 4 and cfg.quant_expert_block == 128 and cfg.router_bias
    got = capture._layer_decisions(str(tmp_path), None, "model.layers.1.", 2, [0, 1], 240.0)
    for r in (0, 1):
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)  # noqa: E731
        layer = types.SimpleNamespace(router=z(4, 256, dt=torch.bfloat16), down_t=True, router_bias=z(4),
                                      router_logit_bias=None, w_gu=z(4, 128, 256, dt=FP8), w_gu_scale=z(4, 128, 2),
                                      w_down=z(4, 64, 256, dt=FP8), w_down_scale=z(4, 1, 256))
        loader._load_experts(layer, loader._Checkpoint(str(tmp_path)), "model.layers.1.", cfg, r, 2, torch.bfloat16,
                             240.0)
        blob = mdd.pack(layer.w_gu, layer.w_gu_scale, layer.w_down, layer.w_down_scale)
        d = got[capture.decision_key(240.0, 2, r, "model.layers.1.")]
        assert (d["moe_prefill_dq"], d["moe_prefill_down"]) == (mp.check_blob(blob, 256),
                                                                 mp.down_factors(blob, 256) is not None)
        assert (d["moe_prefill_dq"], d["moe_prefill_down"]) == ((False, True) if refit else (True, False))
        assert d["experts_read"] == (1 if refit else 4)


def test_capture_replays_the_decisions_and_refuses_without_them(tmp_path, monkeypatch):
    """A layer packed from a shape directory (zeros) gets the recorded answers, and the tensors the
    real values give it (moe_prefill_dsc / moe_prefill_dfr: their presence, shapes and dtypes enter
    the graphs); an answer the record lacks raises."""
    import types

    from tests.test_moe_prefill import checkpoint_experts

    from kiln import capture
    from kiln.models import decoder

    monkeypatch.setattr(decoder, "MOE_PREFILL_KERNEL", "nki")
    capture.install_header_checkpoint()
    real = types.SimpleNamespace(pack_tiles=True, **dict(zip(("w_gu", "w_gu_scale", "w_down", "w_down_scale"),
                                                             checkpoint_experts()[0])))
    decoder.DecoderLayer.pack_experts(real)  # no expert load from a shape directory: the real answers
    key = capture.decision_key(240.0, 32, 0, "model.layers.3.")
    rec = {"decisions": {key: {"moe_prefill_dq": False, "moe_prefill_down": True}}, "path": "x"}

    def zeros_layer():
        return types.SimpleNamespace(pack_tiles=True, **{n: torch.zeros_like(t) for n, t in zip(
            ("w_gu", "w_gu_scale", "w_down", "w_down_scale"), checkpoint_experts()[0])})

    fake = zeros_layer()
    capture._PENDING[0] = (fake, rec, key)
    decoder.DecoderLayer.pack_experts(fake)
    assert fake.moe_prefill_dq is real.moe_prefill_dq is False
    for n in ("moe_prefill_dsc", "moe_prefill_dfr"):
        a, b = getattr(fake, n), getattr(real, n)
        assert (tuple(a.shape), a.dtype) == (tuple(b.shape), b.dtype)
    other = zeros_layer()
    capture._PENDING[0] = (other, {"decisions": {}, "path": "x"}, key)
    with pytest.raises(RuntimeError, match="no moe_prefill_dq"):
        decoder.DecoderLayer.pack_experts(other)
    assert capture._ACTIVE[0] is None and capture._PENDING[0] is None


def test_peak_estimate_covers_the_measured_peaks():
    """The size table stays above the largest host peak measured in each HLO size band (2026-10-04
    queues, kiln/compile_farm.peak_gb_estimate), and a peak measured on the queue wins."""
    from kiln.compile_farm import peak_gb_estimate

    for hlo_bytes, peak_gb in [(980_000, 10.6), (1_900_000, 14.4), (2_970_000, 22.9), (4_804_186, 17.0),
                               (6_311_178, 30.0)]:
        assert peak_gb < peak_gb_estimate(hlo_bytes) < 3 * peak_gb + 10
    assert peak_gb_estimate(6_300_000, {6_311_178: 30.0}) == pytest.approx(34.5)
    assert peak_gb_estimate(1_000_000, {6_311_178: 30.0}) == 19.0  # nothing within 10%


def test_oom_killed_reads_neuronx_cc_own_report():
    """neuronx-cc reports a subprocess the OOM killer took as exit 70 with [F137] (the farm retries
    those with a bigger reservation); any other exit 70 is a compile error."""
    from kiln.compile_farm import Result, oom_killed

    f137 = ("2026-10-04T08:26:13Z [F137] neuronx-cc was forcibly killed - This most commonly occurs due to "
            "insufficient system memory.")
    assert oom_killed(Result("k", False, 1.0, 70, error=f137))
    assert oom_killed(Result("k", False, 1.0, -9))
    assert not oom_killed(Result("k", False, 1.0, 70, error="[NCC_EBVF030] Instructions generated exceeds"))


def test_workers_on_one_host_share_one_budget(tmp_path):
    """Two farm workers on one host count each other's reservations (kiln-cf-1, 2026-10-04: three
    workers reserved 330 + 294 + 133 GB on 371 GiB and the OOM killer took 11 compiles); a worker
    that died without closing stops counting."""
    from kiln.compile_farm import HostLedger, admit

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    a = HostLedger(str(tmp_path), pid=os.getpid())
    b = HostLedger(str(tmp_path), pid=os.getppid())
    gone = HostLedger(str(tmp_path), pid=dead.pid)
    with a.locked():
        a.set(294.0, queue="q/t2max-U1", keys=["k-shared"])
    b.set(133.0, queue="q/t2max-U0")
    gone.set(500.0, keys=["k-dead"])
    assert b.others() == (294.0, {"k-shared"})  # a, and not the dead worker, whose file is removed
    assert not os.path.exists(gone.mine)
    host_budget = 0.9 * 371
    assert not admit(17.0, used=133.0, others=294.0, budget=host_budget, host_budget=host_budget, free=211.0)
    assert admit(17.0, used=0.0, others=294.0, budget=host_budget, host_budget=host_budget, free=211.0)
    assert not admit(17.0, used=0.0, others=0.0, budget=host_budget, host_budget=host_budget, free=10.0)
    a.close()
    assert b.others_gb() == 0.0


def test_oom_retry_reserves_from_the_peak_at_kill():
    """A graph the OOM killer took is retried with max(2x its estimate, 1.5x the tree RSS when it was
    killed): kiln-cf-1, 2026-10-04, an F0 mixed group estimated at 26.9 GB died at 107.3 GB, where
    twice the estimate would have been killed again."""
    from kiln.compile_farm import oom_reservation

    assert oom_reservation(26.9, 107.3) == 1.5 * 107.3
    assert oom_reservation(17.0, 20.0) == 34.0  # killed early: twice the estimate is the larger


@pytest.mark.skipif(not os.environ.get("KILN_TEST_S3_SCRATCH"),
                    reason="needs a real S3 scratch prefix (KILN_TEST_S3_SCRATCH=s3://.../scratch)")
def test_farm_wait_fetches_a_compiled_key_the_queue_does_not_list(tmp_path, monkeypatch):
    """A key complete in the queue's cache but listed nowhere in the queue is fetched, not compiled
    (kiln-trn2-b, 2026-10-04: q/t2max-EU was enqueued with every key already compiled, listed none, and
    the box compiled five decode groups it could have fetched). Against real S3."""
    import uuid

    from kiln import compile_cache

    base = f"{os.environ['KILN_TEST_S3_SCRATCH'].rstrip('/')}/farmwait-{uuid.uuid4().hex[:12]}"
    key = "0" * 31 + "1"
    entry = tmp_path / "src" / key
    entry.mkdir(parents=True)
    (entry / "graph_x.neff").write_bytes(b"neff")
    (entry / compile_cache.MARKER).write_text("")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"cache": f"{base}/cache", "keys": {}}))
    aws = ["aws", "s3", "cp", "--quiet"]
    try:
        subprocess.run([*aws, "--recursive", str(entry), f"{base}/cache/{key}/"], check=True)
        subprocess.run([*aws, str(man), f"{base}/q/manifest.json"], check=True)
        monkeypatch.setenv("NEURON_LIBTORCH_CACHE_ROOT", str(tmp_path / "local"))
        fw = compile_cache.FarmWait(f"{base}/q", poll=0.1, timeout=5)
        assert key not in fw.keys
        assert fw.before_compile(key) == "fetched"
        assert (tmp_path / "local" / "neuron" / "compile_cache" / key / compile_cache.MARKER).exists()
        assert fw.before_compile("f" * 32) == "not-farmed"  # in neither the queue nor the cache
    finally:
        subprocess.run(["aws", "s3", "rm", "--quiet", "--recursive", f"{base}/"])


def test_runtime_scratchpad_from_the_compiler_log():
    """The runtime's per-NEFF scratchpad reservation is the compiler's "Peak scratchpad usage: local"
    (GiB), and the shared scratchpad is the largest, rounded up to 64 MiB: the values of the runtime's
    memory table on kiln-g1-trn1 (2026-10-04 02:47 UTC) for NEFFs 1002 / 1004 / 1008 / 1010."""
    from kiln.compile_farm import scratchpad_gib, shared_scratchpad_bytes

    line = ("2026-10-04T01:55:41Z INFO 291524 [BackendDriver]: HBM scratchpad usage summary (post-allocation):\n"
            "\u2502 nc00  \u2502 module    \u2502 Peak scratchpad usage: local      \u2502 0.276024 GB \u2502\n")
    assert scratchpad_gib(line) == 0.276024
    assert abs(0.276024 * 1024 - 282.648) < 1e-3  # MiB, as the runtime printed for NEFF 1002
    assert scratchpad_gib("no summary") == 0.0
    peaks = [0.276024, 0.251030, 0.361675, 370.479 / 1024]
    assert shared_scratchpad_bytes(peaks) == 384 * 2**20
    assert shared_scratchpad_bytes([]) == 0
