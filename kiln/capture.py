"""Capture every graph an engine configuration will run, on a host with no NeuronCore.

A compiled graph depends on shapes, dtypes and the collectives' replica groups, never on weight
values, and libtorch_neuronx_lite (LNL) supports lowering on meta tensors for device-free compiles
(NEURON_LIBTORCH_CPU_COMPILE=1: inputs on meta, meta devices hashed as neuron:0, compile/
cache.py _normalize_device_reference; NEURON_PLATFORM_TARGET_OVERRIDE names the target instead of
the runtime, compile/platform.py). So one process per tensor-parallel rank builds the model shard
on the meta device and runs the ordinary bucket warmup (ModelRunner.warmup) through a capture
backend that, for each graph:

- canonicalises it exactly as the device backend does (model_runner.canonical_neuron_backend's
  default-kwarg stripping), so the cache key is the one the device run will compute;
- computes LNL's cache key (create_cache_hash: graph text, input shapes / dtypes / strides,
  replica groups, package versions, compiler arguments) and, unless the entry exists, lowers the
  graph to HLO with LNL's own pipeline (capture_backend.run_fx_to_hlo_pipeline) and writes
  graph.hlo, the artifact metadata and command.txt (the neuronx-cc command neuroncc_compile would
  run) into <cache>/<key>/ WITHOUT compiling;
- returns a stand-in that produces meta outputs of the HLO's result shapes (aliased outputs are
  the input itself, as LNL's Executable does), so the forward continues to the next graph.

What it needs from the outside, and how it gets it without the hardware:

- Weights: only their shapes and dtypes. HeaderCheckpoint serves zeros shaped from the
  safetensors headers (a JSON index, `header_index`), so the real loader runs unchanged,
  including the load-time repacking that changes parameter shapes (DecoderLayer.pack_experts).
- Collectives: a fake process group of the real world size (torch.testing._internal.distributed.
  fake_pg), created in the same order as the engine creates its groups (build_shard), so group
  names and replica groups, which enter the graph text and the key, match the device run.
- NKI kernels: compiled at trace time by LNL's NKI pass on the CPU (nki compile_nki, target from
  the override); their binaries land in /var/tmp/nki-intermediate-cache and are shipped with the
  cache by kiln/compile_cache.py.

tools/compile_farm.py `capture` drives this and fans the HLOs out to compile hosts.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import struct
import time

import torch

# safetensors header dtype strings (safetensors/src/tensor.rs Dtype) -> torch.
ST_DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16, "U16": torch.uint16,
    "I32": torch.int32, "U32": torch.uint32, "I64": torch.int64, "U64": torch.uint64,
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
    **({"F8_E8M0": torch.float8_e8m0fnu} if hasattr(torch, "float8_e8m0fnu") else {}),
}
HEADERS = "kiln-safetensors-headers.json"  # name -> [dtype, shape], beside config.json


def read_header(path: str) -> dict:
    """A safetensors file's header: 8-byte little-endian length, then that many bytes of JSON
    (the safetensors format, https://github.com/huggingface/safetensors#format)."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n))


def header_index(model_dir: str) -> dict[str, list]:
    """{tensor name: [dtype, shape]} over every *.safetensors file of a local checkpoint."""
    import glob

    out = {}
    for fn in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        for name, meta in read_header(fn).items():
            if name != "__metadata__":
                out[name] = [meta["dtype"], meta["shape"]]
    if not out:
        raise FileNotFoundError(f"no .safetensors files in {model_dir}")
    return out


def write_shape_dir(model_dir: str, dest: str) -> str:
    """config.json (and the other small JSON files) plus the header index: everything the capture
    needs from a checkpoint, a few MB instead of hundreds of GB."""
    os.makedirs(dest, exist_ok=True)
    for fn in os.listdir(model_dir):
        if fn.endswith(".json") and not fn.endswith(".safetensors.index.json"):
            shutil.copy(os.path.join(model_dir, fn), os.path.join(dest, fn))
    with open(os.path.join(dest, HEADERS), "w") as f:
        json.dump(header_index(model_dir), f)
    return dest


def write_shape_dir_hub(repo_id: str, dest: str) -> str:
    """write_shape_dir for a Hugging Face repo, without downloading the weights: the headers come
    from huggingface_hub.get_safetensors_metadata (it range-reads each file's header)."""
    from huggingface_hub import get_safetensors_metadata, hf_hub_download

    os.makedirs(dest, exist_ok=True)
    for fn in ("config.json", "generation_config.json"):
        try:
            shutil.copy(hf_hub_download(repo_id, fn), os.path.join(dest, fn))
        except Exception:
            if fn == "config.json":
                raise
    meta = get_safetensors_metadata(repo_id)
    index = {name: [t.dtype, list(t.shape)] for fm in meta.files_metadata.values()
             for name, t in fm.tensors.items()}
    with open(os.path.join(dest, HEADERS), "w") as f:
        json.dump(index, f)
    return dest


class _HeaderSlice:
    def __init__(self, dtype: str, shape: list[int]):
        self._dtype, self._shape = dtype, list(shape)

    def get_shape(self) -> list[int]:
        return list(self._shape)

    def get_dtype(self) -> str:
        return self._dtype

    def __getitem__(self, idx) -> torch.Tensor:
        shape = torch.empty(self._shape, device="meta")[idx].shape
        return torch.zeros(shape, dtype=ST_DTYPES[self._dtype])


class _HeaderHandle:
    """The part of a safetensors safe_open handle that models/loader.py _Checkpoint uses."""

    def __init__(self, index: dict[str, list]):
        self._index = index

    def keys(self):
        return list(self._index)

    def get_slice(self, name: str) -> _HeaderSlice:
        dtype, shape = self._index[name]
        return _HeaderSlice(dtype, shape)

    def get_tensor(self, name: str) -> torch.Tensor:
        dtype, shape = self._index[name]
        return torch.zeros(shape, dtype=ST_DTYPES[dtype])


def _set_handle(ck, h) -> None:
    """A models/loader.py _Checkpoint over one handle, named as _Checkpoint.__init__ names it."""
    from .models import loader

    ck._handles = [h]
    ck._where = {name: h for name in h.keys()}
    vl = "model.language_model."
    for name in [n for n in ck._where if n.startswith(vl)]:
        ck._where.setdefault("model." + name[len(vl):], loader._Alias(h, name))


def install_header_checkpoint() -> None:
    """Make models/loader.py read a shape directory (write_shape_dir): every tensor it asks for
    is zeros of the checkpoint's shape and dtype, and the load-time decisions that read weight
    values come from the directory's DECISIONS file (see "Load-time decisions" below). A
    directory that holds real safetensors files is read as usual."""
    from .models import loader

    _install_decisions()
    if getattr(loader._Checkpoint, "_kiln_capture", False):
        return
    real_init = loader._Checkpoint.__init__

    def __init__(self, path: str):
        p = os.path.join(path, HEADERS)
        if not os.path.exists(p):
            return real_init(self, path)
        with open(p) as f:
            _set_handle(self, _HeaderHandle(json.load(f)))
        self._kiln_decisions = read_decisions(path)

    loader._Checkpoint.__init__ = __init__
    loader._Checkpoint._kiln_capture = True


# -- Load-time decisions that read weight VALUES ---------------------------------------------------
#
# A graph depends on weight values wherever the loader looks at them to choose a layout, and zeros
# answer such a question differently from the checkpoint. One place does it so far:
# DecoderLayer.pack_experts with KILN_MOE_PREFILL_KERNEL=nki asks kernels/moe_prefill.check_blob
# (a static flag, the kernel's dequantize-first path) and down_factors (two more parameters per MoE
# layer, or None). On zeros check_blob says True and down_factors None; on GLM-5.3-Flash at tp=32
# they say False and two tensors (check_blob's docstring). Measured (d25e855, G64p1024,
# 2026-10-04): every MoE layer of the device run carried the two tensors moe_prefill_dsc /
# moe_prefill_dfr that the capture's did not, which renamed the placeholders of the 12-layer decode
# group (its 9 MoE layers: L_ts_108_ became L_ts_110_ and so on; the two are unused there, so
# dynamo pruned them) and the device's key ddfec8b1fbe8c499da25196100cd8c40 was one the capture
# never wrote, though both graph.hlo files were 2,264,167 bytes and the inputs the same 354
# shapes. The prefill groups differ in what they compute too.
#
# So the answers are computed from the real values once per checkpoint (compute_decisions, the
# `decisions` subcommand of tools/compile_farm.py), kept in the shape directory, and replayed by
# the capture; a capture that needs an answer the file does not hold raises instead of guessing.

DECISIONS = "kiln-load-decisions.json"  # beside HEADERS: {"decisions": {decision_key: {...}}, ...}
# The last expert load from a shape directory, (layer, record, key): the loader packs a layer right
# after loading its experts (models/loader.py _load_decoder), before the next layer's.
_PENDING: list = [None]
_ACTIVE: list = [None]  # the (record, key) of the pack_experts call in progress


def decision_key(fp8_max: float, tp: int, rank: int, pre: str) -> str:
    """What a decision depends on besides the checkpoint: the e4m3 limit models/quant.fit_e4m3_max
    refits to (kiln/platform.py fp8_max), how it groups the experts' rows (models/loader.py
    MOE_E4M3_FIT; the default per-row fit keeps the keys written before it existed), the expert shard
    (tp, rank) and the layer (its prefix)."""
    from .models import loader

    fit = "" if loader.MOE_E4M3_FIT == "row" else f" fit={loader.MOE_E4M3_FIT}"
    return f"fp8_max={fp8_max:g}{fit} tp={tp} rank={rank} {pre}"


def read_decisions(shape_dir: str) -> dict:
    p = os.path.join(shape_dir, DECISIONS)
    if not os.path.exists(p):
        return {"decisions": {}, "path": p}
    with open(p) as f:
        return {**json.load(f), "path": p}


def _install_decisions() -> None:
    from .kernels import moe_prefill as mp
    from .models import decoder, loader

    if getattr(mp, "_kiln_capture", False):
        return
    real_load, real_pack = loader._load_experts, decoder.DecoderLayer.pack_experts
    real_check, real_down = mp.check_blob, mp.down_factors

    def _load_experts(layer, ck, pre, cfg, r, n, dtype, fp8_max=240.0):
        rec = getattr(ck, "_kiln_decisions", None)
        _PENDING[0] = (layer, rec, decision_key(fp8_max, n, r, pre)) if rec is not None else None
        return real_load(layer, ck, pre, cfg, r, n, dtype, fp8_max)

    def pack_experts(self):
        p, _PENDING[0] = _PENDING[0], None
        _ACTIVE[0] = p[1:] if p is not None and p[0] is self else None
        try:
            return real_pack(self)
        finally:
            _ACTIVE[0] = None

    def answer(what: str) -> bool:
        rec, key = _ACTIVE[0]
        try:
            return bool(rec["decisions"][key][what])
        except KeyError:
            raise RuntimeError(
                f"capture: {key}: no {what} in {rec['path']}. Zeros cannot answer it (kiln/capture.py "
                "\"Load-time decisions\"); write it from the real checkpoint first: python tools/compile_farm.py "
                "decisions --model <hub id or checkpoint dir> --shape-dir <this directory> --tp <tp>") from None

    def check_blob(blob, H):
        if _ACTIVE[0] is None or mp.blob_scale_bytes(blob, H) == 2:  # bf16 tile scales: False by layout
            return real_check(blob, H)
        return answer("moe_prefill_dq")

    def down_factors(blob, H):
        if _ACTIVE[0] is None:
            return real_down(blob, H)
        # bf16 tile scales always take the per-column path (down_factors' docstring); fp32 ones
        # when the record says so. The values are never read by a graph trace, only the shapes.
        if mp.blob_scale_bytes(blob, H) == 4 and not answer("moe_prefill_down"):
            return None
        E = blob.shape[0]
        return torch.ones(E, H // mp.P), torch.ones(E, H, dtype=torch.bfloat16)

    loader._load_experts = _load_experts
    decoder.DecoderLayer.pack_experts = pack_experts
    mp.check_blob, mp.down_factors = check_blob, down_factors
    mp._kiln_capture = True


def decide(blobs, H: int) -> tuple[bool, bool, int]:
    """(check_blob, down_factors is not None, experts read) over one expert's tiles blob at a time,
    stopping once both are known. check_blob is a conjunction over experts (each expert's gate and
    up rows must share their tile scales) and down_factors is not None unless every expert's down
    scales are constant per chunk (or they are bf16), so the first expert that breaks each decides
    it. Not checked past that expert: down_factors' power-of-two condition, which raises on the
    device, where every expert is packed."""
    from .kernels import moe_prefill as mp

    dq, down, n = None, None, 0
    for blob in blobs:
        n += 1
        if dq is None and not mp.check_blob(blob, H):
            dq = False
        if down is None and mp.down_factors(blob, H) is not None:
            down = True
        if dq is not None and down is not None:
            break
    return (True if dq is None else dq), bool(down), n


class _ValueSlice(_HeaderSlice):
    def __init__(self, dtype: str, shape: list[int], tensor):
        super().__init__(dtype, shape)
        self._tensor = tensor

    def __getitem__(self, idx) -> torch.Tensor:
        return self._tensor()[idx]


class _ValueHandle(_HeaderHandle):
    """A header handle that serves the real values of the tensors `want(name)` selects, read by
    `fetch(name)`, and zeros for the rest."""

    def __init__(self, index: dict[str, list], fetch, want=lambda name: False):
        super().__init__(index)
        self._fetch, self.want = fetch, want

    def get_slice(self, name: str) -> _HeaderSlice:
        if not self.want(name):
            return super().get_slice(name)
        dtype, shape = self._index[name]
        return _ValueSlice(dtype, shape, lambda: self._fetch(name))

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._fetch(name) if self.want(name) else super().get_tensor(name)


class _HubTensors:
    """Single tensors of a Hugging Face repo by HTTP range requests: absolute offsets from
    get_safetensors_metadata's data_offsets after each file's 8-byte header length and header
    (the safetensors format, https://github.com/huggingface/safetensors#format). Measured on
    kiln-cf-1 (2026-10-04): 8 MB in 0.65 s. The table is built once (`table`) and handed to the
    worker processes: the Hub limits an IP without a token to 3000 "resolver" requests per 300 s,
    and 44 workers each reading the metadata of the 62 files of GLM-5.3-Flash hit it (429)."""

    def __init__(self, repo: str, revision: str | None = None, table: dict | None = None):
        self.repo = repo
        if table is None:
            from huggingface_hub import get_safetensors_metadata, model_info

            self.revision = revision or model_info(repo).sha
            meta = get_safetensors_metadata(repo, revision=self.revision)
            bases = {fn: 8 + struct.unpack("<Q", self._get(fn, 0, 8))[0] for fn in meta.files_metadata}
            table = {n: [t.dtype, list(t.shape), fn, bases[fn] + t.data_offsets[0], bases[fn] + t.data_offsets[1]]
                     for fn, fm in meta.files_metadata.items() for n, t in fm.tensors.items()}
        else:
            self.revision = revision
        self.table = table
        self.cache: dict[str, torch.Tensor] = {}

    def index(self) -> dict[str, list]:
        return {n: v[:2] for n, v in self.table.items()}

    def _get(self, fn: str, a: int, b: int) -> bytes:
        """Bytes [a, b) of one file at the revision, one request (and its redirect to the storage)."""
        from huggingface_hub import get_session, hf_hub_url
        from huggingface_hub.utils import build_hf_headers

        url = hf_hub_url(self.repo, fn, revision=self.revision)
        s = get_session()
        for attempt in range(8):
            h = {**build_hf_headers(), "Range": f"bytes={a}-{b - 1}"}
            r = s.get(url, headers=h, follow_redirects=True) if hasattr(s, "follow_redirects") else \
                s.get(url, headers=h, allow_redirects=True)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("retry-after", 5)) + attempt)
                continue
            if r.status_code != 206 or len(r.content) != b - a:
                raise RuntimeError(f"{url} bytes {a}-{b - 1}: HTTP {r.status_code}, {len(r.content)} bytes")
            return r.content
        raise RuntimeError(f"{url}: still rate limited (429) after 8 attempts")

    def __call__(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            dtype, shape, fn, a, b = self.table[name]
            out = torch.frombuffer(bytearray(self._get(fn, a, b)), dtype=torch.uint8) if b > a else \
                torch.zeros(0, dtype=torch.uint8)
            self.cache[name] = out.view(ST_DTYPES[dtype]).reshape(shape)
        return self.cache[name]


class _LocalTensors:
    def __init__(self, path: str):
        import glob

        from safetensors import safe_open

        self._h = {n: h for fn in sorted(glob.glob(os.path.join(path, "*.safetensors")))
                   for h in [safe_open(fn, framework="pt")] for n in h.keys()}
        self.cache: dict[str, torch.Tensor] = {}
        self.revision = None

    def __call__(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            self.cache[name] = self._h[name].get_tensor(name)
        return self.cache[name]


def _layer_decisions(source: str, revision: str | None, pre: str, tp: int, ranks: list[int], fp8_max: float,
                     dtype: torch.dtype = torch.bfloat16, table: dict | None = None,
                     cfg_dir: str | None = None) -> dict:
    """The decisions of one MoE layer for each rank: the rank's shard of expert e through the loader's
    own path (loader._load_experts: the shard, models/quant.fit_e4m3_max), packed by
    moe_dedupe.pack as DecoderLayer.pack_experts packs it, for e = 0, 1, ... until `decide` has both
    answers (tools/check_moe_prefill_layout.py does the same for whole layers). Every other expert
    is zeros and is not looked at."""
    import types

    from .config import ModelConfig
    from .kernels import moe_dedupe as mdd
    from .models import loader
    from .models.quant import FP8

    if os.path.isdir(source):
        src, index, cfg_dir = _LocalTensors(source), header_index(source), source
    else:
        src = _HubTensors(source, revision, table)
        index, cfg_dir = src.index(), cfg_dir or os.path.dirname(_hub_config(source, src.revision))
    cfg = ModelConfig.from_pretrained(cfg_dir)
    handle = _ValueHandle(index, src)
    ck = loader._Checkpoint.__new__(loader._Checkpoint)
    _set_handle(ck, handle)
    E, I = cfg.num_experts, cfg.moe_intermediate_size
    Im, m = I // tp, pre + "mlp.experts."
    H = ck.shape(f"{m}0.down_proj.weight")[0]
    bk = cfg.quant_expert_block or 128
    router = ck.shape(pre + loader.names(cfg)["router"])
    z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt)  # noqa: E731
    pat = re.compile(rf"(?:^|\.){re.escape(m[len('model.'):])}(\d+)\.(?:gate|up|down)_proj\.weight(?:_scale_inv)?$")

    def want(experts):
        return lambda name: bool(mm := pat.search(name)) and (experts is None or int(mm.group(1)) in experts)

    out = {}
    for r in ranks:
        def blobs():
            # Expert 0 first; the whole layer (one more load) only if expert 0 does not decide both.
            for experts in ({0}, None):
                handle.want = want(experts)
                layer = types.SimpleNamespace(
                    router=z(*router, dt=torch.bfloat16), down_t=True,
                    router_bias=z(E) if cfg.router_bias else None,
                    router_logit_bias=z(E, dt=torch.bfloat16) if getattr(cfg, "router_logit_bias", False) else None,
                    w_gu=z(E, 2 * Im, H, dt=FP8), w_gu_scale=z(E, 2 * Im, H // bk),
                    w_down=z(E, Im, H, dt=FP8), w_down_scale=z(E, 1, H))
                loader._load_experts(layer, ck, pre, cfg, r, tp, dtype, fp8_max)
                for e in sorted(experts) if experts else range(1, E):
                    yield mdd.pack(layer.w_gu[e:e + 1], layer.w_gu_scale[e:e + 1], layer.w_down[e:e + 1],
                                   layer.w_down_scale[e:e + 1])

        dq, down, n = decide(blobs(), H)
        out[decision_key(fp8_max, tp, r, pre)] = {"moe_prefill_dq": dq, "moe_prefill_down": down, "experts_read": n}
    return out


def _hub_config(repo: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, "config.json", revision=revision)


def moe_prefixes(index: dict[str, list]) -> list[str]:
    """Every layer prefix with routed experts named per expert (model.layers.N.), the MTP layer's included."""
    import re

    out = set()
    for name in index:
        mm = re.match(r"model\.(?:language_model\.)?(layers\.\d+\.)mlp\.experts\.0\.down_proj\.weight$", name)
        if mm:
            out.add("model." + mm.group(1))
    return sorted(out, key=lambda p: int(p.split(".")[2]))


def compute_decisions(source: str, shape_dir: str, tp: int, fp8_max: float = 240.0, ranks=None, prefixes=None,
                      procs: int = 16, revision: str | None = None) -> dict:
    """Write DECISIONS into shape_dir for every MoE layer and rank (merged with what it holds), from
    the real checkpoint: a local directory or a Hugging Face repo id (range reads, no download)."""
    from concurrent.futures import ProcessPoolExecutor

    table = cfg_dir = None
    if not os.path.isdir(source):
        hub = _HubTensors(source, revision)
        revision, table = hub.revision, hub.table
        cfg_dir = os.path.dirname(_hub_config(source, revision))
    with open(os.path.join(shape_dir, HEADERS)) as f:
        prefixes = prefixes or moe_prefixes(json.load(f))
    ranks = list(range(tp)) if ranks is None else list(ranks)
    rec = read_decisions(shape_dir)
    rec.pop("path")
    t0 = time.time()
    import multiprocessing

    with ProcessPoolExecutor(max_workers=procs, mp_context=multiprocessing.get_context("spawn"),
                             initializer=torch.set_num_threads, initargs=(4,)) as ex:
        jobs = [ex.submit(_layer_decisions, source, revision, p, tp, ranks, fp8_max, table=table, cfg_dir=cfg_dir)
                for p in prefixes]
        for got in (j.result() for j in jobs):
            rec["decisions"].update(got)
    rec.update(model=source, revision=revision, written=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               seconds=round(time.time() - t0, 1))
    with open(os.path.join(shape_dir, DECISIONS) + ".tmp", "w") as f:
        json.dump(rec, f, indent=1, sort_keys=True)
    os.replace(os.path.join(shape_dir, DECISIONS) + ".tmp", os.path.join(shape_dir, DECISIONS))
    return rec


def configure_env(target: str) -> None:
    """Before libtorch_neuronx_lite is imported."""
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = target
    os.environ["NEURON_LIBTORCH_CPU_COMPILE"] = "1"


def init_fake_world(rank: int, world: int) -> None:
    """A process group of `world` ranks in which this process is `rank`, with no peers and no
    communication. Call it BEFORE importing libtorch_neuronx_lite, as a device rank does (engine/
    tp.py init_rank precedes build_shard's import). Measured on SDK 2.32 (c8i.48xlarge, probe in
    docs/neuron-notes.md "Compile farm"): with LNL imported first and the process group initialised
    after it, dynamo records torch.topk(x, k, dim=-1) as a call of a Python function
    (_VariableFunctionsClass.topk) that has no schema, so canonicalize cannot strip its defaults,
    and torch.argmax(x, dim=-1, keepdim=True) keeps its keywords; with the process group first,
    both are LNL's override spelling (topk with largest / sorted / out spelled out, argmax with
    positional arguments), the spelling the device run hashes. Either order of a gloo group does
    the same."""
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", rank=rank, world_size=world, store=FakeStore())


# Keys this process captured (written) or found already captured, in first-seen order.
CAPTURED: dict[str, str] = {}


def _hlo_outputs(hlo_path: str):
    """(shape, dtype) of each HLO result, with compile/backend.py build_executable's dtype table."""
    from libtorch_neuronx_lite.compile.hlo import load_hlo_module
    from libtorch_neuronx_lite.pyhlo import xla_data_pb2 as x

    dt = {x.F32: torch.float32, x.F64: torch.float64, x.BF16: torch.bfloat16, x.F16: torch.float16,
          x.U8: torch.uint8, x.S8: torch.int8, x.U16: torch.uint16, x.S16: torch.int16, x.U32: torch.uint32,
          x.S32: torch.int32, x.U64: torch.uint64, x.S64: torch.int64, x.PRED: torch.bool,
          x.F8E4M3FN: torch.float8_e4m3fn, x.F8E5M2: torch.float8_e5m2}
    hlo = load_hlo_module(hlo_path)
    entry = next(c for c in hlo.computations if c.id == hlo.entry_computation_id)
    return [(tuple(s.dimensions), dt[s.element_type]) for s in entry.program_shape.result.tuple_shapes]


def _write_command(workdir: str, key: str, options: dict) -> None:
    """command.txt as compile/backend.py neuroncc_compile writes it (shlex.join of the command)."""
    from libtorch_neuronx_lite.compile.backend import _parse_compiler_args
    from libtorch_neuronx_lite.compile.platform import resolve_target

    from .compile_farm import compiler_command, neff_name

    args = _parse_compiler_args(options.get("compiler_args", ""))
    target = resolve_target(args)
    if "--target" in args:
        i = args.index("--target")
        args = args[:i] + args[i + 2:]
    cmd = compiler_command(shutil.which("neuronx-cc") or "neuronx-cc", os.path.join(workdir, "graph.hlo"), target,
                           os.path.join(workdir, neff_name(key)), os.path.join(workdir, "log-neuron-cc.txt"), args)
    with open(os.path.join(workdir, "command.txt"), "w") as f:
        f.write(shlex.join(cmd))


def capture_graph(gm, example_inputs, options: dict):
    """The capture half of LNL's compile(): key, HLO, metadata, command; no neuronx-cc."""
    from libtorch_neuronx_lite.compile import cache
    from libtorch_neuronx_lite.compile.backend import (
        _apply_platform_compiler_args,
        _detect_duplicate_inputs,
        preprocess_and_validate_inputs,
    )
    from libtorch_neuronx_lite.compile.capture_backend import run_fx_to_hlo_pipeline, setup_workdir_common
    from libtorch_neuronx_lite.compile.schema import create_metadata

    # compile() registers LNL's collective lowerings before it lowers anything (backend.py: "Ensure
    # collective kernels are registered"); without them torch_xla lowers _c10d_functional.all_reduce
    # itself, as an all-reduce with no replica_groups and constrain_layout set. Measured: the
    # capture then wrote that HLO under the very key the device computed (the key hashes the FX
    # graph, not the HLO), so the farm would have built a different NEFF for it.
    try:
        from libtorch_neuronx_lite.overrides import neuron_collectives, xla_collectives  # noqa: F401
    except RuntimeError:  # already registered
        pass
    keep, _ = _detect_duplicate_inputs(example_inputs)
    gm, inputs = preprocess_and_validate_inputs(gm, example_inputs, options)
    options = _apply_platform_compiler_args(options)
    workdir, key, base = setup_workdir_common(gm, inputs, options, per_rank=False)
    meta_path = os.path.join(workdir, ".artifact_metadata_v0.json")
    if not os.path.exists(os.path.join(workdir, "graph.hlo")) or not os.path.exists(meta_path):
        # Lower in a private directory and rename it into place: the other ranks' capture
        # processes trace the same graphs at the same time.
        tmp = f"{workdir}.capture.{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        t0 = time.perf_counter()
        hlo, unused, rng, io_map, n_out, _ = run_fx_to_hlo_pipeline(gm, inputs, options, tmp)
        with open(os.path.join(tmp, "graph.hlo"), "wb") as f:
            f.write(hlo.SerializeToString())
        cache.save_artifact_metadata(tmp, create_metadata(cache_key=key, output_count=n_out,
                                                          unused_input_indices=unused,
                                                          has_rng_seed_parameter=rng, io_map=io_map))
        try:
            os.rename(tmp, workdir)
            _write_command(workdir, key, options)
            CAPTURED[key] = f"captured {time.perf_counter() - t0:.1f}s"
        except OSError:  # another rank renamed its copy first
            shutil.rmtree(tmp, ignore_errors=True)
            CAPTURED.setdefault(key, "captured by another process")
    else:
        CAPTURED.setdefault(key, "exists")
    with open(meta_path) as f:
        meta = json.load(f)
    io = {int(k): int(v) for k, v in (meta.get("io_map") or {}).items()}
    n_out = meta.get("output_count")
    outs = _hlo_outputs(os.path.join(workdir, "graph.hlo"))

    def stand_in(*args):
        args = [a for a, k in zip(args, keep) if k]
        res = [args[io[i]] if i in io else torch.empty(shape, dtype=dtype, device="meta")
               for i, (shape, dtype) in enumerate(outs)]
        return res[:n_out] if n_out is not None else res

    return stand_in


def backend():
    """torch.compile backend for ModelRunner on the meta device."""
    from .engine.model_runner import canonicalize

    def capture_backend(gm, example_inputs, **kwargs):
        canonicalize(gm)
        return capture_graph(gm, example_inputs, kwargs.get("options", {}))

    capture_backend.__name__ = "kiln_capture"
    return capture_backend


def dump_hash_inputs(directory: str) -> None:
    """Write the exact string LNL hashes into each cache key to <directory>/<key>.txt
    (compile/cache.py create_cache_hash: md5 over "|".join(components)), to diff a captured key
    against the key a device run computed. KILN_HASH_DUMP=<dir> turns it on in ModelRunner."""
    import hashlib

    from libtorch_neuronx_lite.compile import cache

    if getattr(cache.hashlib, "_kiln_dump", False):
        return
    os.makedirs(directory, exist_ok=True)

    class _Hashlib:
        _kiln_dump = True

        def __getattr__(self, name):
            return getattr(hashlib, name)

        @staticmethod
        def md5(data=b"", **kw):
            h = hashlib.md5(data, **kw)
            with open(os.path.join(directory, h.hexdigest()[:32] + ".txt"), "wb") as f:
                f.write(data)
            return h

    cache.hashlib = _Hashlib()
