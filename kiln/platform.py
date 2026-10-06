"""What Kiln needs to know about the Neuron chip it runs on, in one place.

Everything that differs between trn1 / inf2 (NeuronCore-v2), trn2 (v3) and trn3 (v4) is
answered here: the target string, its family, the NKI generation, the logical NeuronCore
config (LNC), HBM per logical core, the largest finite FP8 E4M3 value, and the neuronx-cc
arguments that follow from them. The rest of the engine asks this module instead of
branching on target strings itself.

The chip has to be known BEFORE the Neuron runtime starts, because the runtime's LNC is an
environment variable read at start-up and must match the compiler's: "AWS Neuron currently
doesn't support setting the compiler flag to a different LNC configuration than the Neuron
Runtime environment variable" (aws-neuron-sdk, about-neuron/arch/neuron-features/
logical-neuroncore-config.rst). libtorch_neuronx_lite (LNL) can name the target only through
the runtime (compile/platform.py, _get_target_from_nrt), so `target()` reads what the
Neuron driver itself reads instead: /sys/class/dmi/id/product_name, the EC2 instance type
(aws-neuron-driver neuron_arch.c, narch_get_instance_type_name). After the runtime is up,
`check_runtime()` compares that answer with LNL's.

trn1 behaviour is unchanged by this module: no LNC variable is set and no compiler argument
is added there.
"""

from __future__ import annotations

import glob
import os
import shlex
from dataclasses import dataclass

# LNL compile/platform.py: _TARGET_FAMILIES, matched as prefixes in this order (trn1n
# before trn1); a target may carry a revision suffix ("trn3-rev2") or an A0 marker
# ("trn3pre", vllm_neuron/vllm/worker/neuron_model_runner.py) and keeps its family.
FAMILIES = ("trn1n", "trn1", "trn2", "trn3", "inf2")
# LNL compile/platform.py HBM_MEMORY_GB: HBM per logical NeuronCore in GiB at the default LNC.
HBM_GIB_PER_CORE = {"trn1": 16, "trn1n": 16, "trn2": 24, "trn3": 36, "inf2": 16}
# NKI hardware generation (nki/__init__.py: trn1|inf2 = gen2, trn2 = gen3, trn3 = gen4).
NKI_GEN = {"trn1": 2, "trn1n": 2, "inf2": 2, "trn2": 3, "trn3": 4}
# Physical NeuronCores per chip (docs/research/trn2-trn3.md section 2).
PHYSICAL_CORES_PER_CHIP = {"trn1": 2, "trn1n": 2, "inf2": 2, "trn2": 8, "trn3": 8}
# Families whose compiler and runtime take an LNC: neuronx-cc --logical-nc-config "(Only
# available on trn2; Default: 2)" (compiler/neuronx-cc/api-reference-guide/index.rst) and
# NCC_EARG001 "On trn1, only lnc=1 is supported"; vllm-neuron runs trn3 at 2 as well
# (utils/hardware_config.py, _DEFAULT_LNC_CONFIG = 2).
LNC_FAMILIES = ("trn2", "trn3")
DEFAULT_LNC = 2
# Device names the driver publishes under neuron<N>/info/architecture/device_name
# (aws-neuron-driver v1..v4/neuron_dhal_v*.c, arch_device_name_suffix).
DEVICE_NAMES = {"Inferentia": "inf1", "Trainium1": "trn1", "Inferentia2": "inf2", "Trainium2": "trn2",
                "Trainium3": "trn3"}
DMI_PRODUCT = "/sys/class/dmi/id/product_name"
SYSFS_DEVICE_NAME = "/sys/devices/virtual/neuron_device/neuron*/info/architecture/device_name"
# vllm-neuron's neuronx-cc arguments for every graph (neuron_model_runner.py, compile_options),
# except the optimization level: vllm-neuron 0.24 defaults to -O1 ("Defaulting optimization level
# to O1 on Neuron", vllm_neuron platform.py:204, measured in its log on trn2.48xlarge 2026-10-03);
# Kiln keeps neuronx-cc's own default -O2 (the level LNL compiles at on trn1, where it passes no
# -O). KILN_CC_ARGS=-O1 appends vllm-neuron's level for an A/B. KILN_CC_PRESET=lnl drops the set.
VLLM_NEURON_CC_ARGS = ("--auto-cast=none", "--verbose=35", "-O2")
VLLM_NEURON_HLO2TENSORIZER = ("--modular-flow-mac-threshold=10",)
VLLM_NEURON_BACKEND = "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop"
UNSAFE_FP8 = "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3"
HLO2TENSORIZER = "--internal-hlo2tensorizer-options="


def family_of(target: str) -> str:
    """'trn2' -> 'trn2', 'trn1n' -> 'trn1n', 'trn3pre' / 'trn3-rev2' -> 'trn3',
    'trn2.3xlarge' / 'trn2u.48xlarge' -> 'trn2'."""
    for fam in FAMILIES:
        if target.startswith(fam):
            return fam
    raise ValueError(f"unknown Neuron target {target!r}; families are {FAMILIES}")


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def instance_type() -> str | None:
    """The EC2 instance type, as the Neuron driver reads it (neuron_arch.c)."""
    return _read(DMI_PRODUCT)


def target() -> str | None:
    """The Neuron target string without starting the runtime, or None off Neuron.

    Order: LNL's own override NEURON_PLATFORM_TARGET_OVERRIDE (it also wins inside LNL, so
    both agree); the EC2 instance type; the driver's device name. The instance type keeps the
    trn1n distinction; trn3 A0 parts ("trn3pre") are told apart only by the runtime, which
    `check_runtime` reports."""
    if os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"):
        return os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"]
    it = instance_type()
    if it:
        try:
            return family_of(it)
        except ValueError:
            pass
    for path in sorted(glob.glob(SYSFS_DEVICE_NAME)):
        name = _read(path)
        if name in DEVICE_NAMES:
            return DEVICE_NAMES[name]
    return None


def lnc(fam: str) -> int:
    """The logical NeuronCore config Kiln runs this family at: NEURON_LOGICAL_NC_CONFIG if
    set, else 2 on trn2 / trn3 (the runtime's and vllm-neuron's default), else 1."""
    if fam not in LNC_FAMILIES:
        return 1
    v = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", DEFAULT_LNC))
    if v not in (1, 2):
        raise ValueError(f"NEURON_LOGICAL_NC_CONFIG must be 1 or 2, not {v}")
    return v


_LNC: int | None = None  # resolved by configure_runtime_env, before anything is traced


# KILN_LNC_SPLIT: which NKI kernels split their work over the two physical cores of an LNC=2 logical core
# (kernels/moe_prefill.py, moe_dedupe.py, delta_rule.py, dsa_topk.py; feat/trn2-max): a comma list of those names,
# "all", or "0" / "none". A kernel left out runs as before the split: the same work on both physical cores
# (delta_rule at grid 1). Read when a graph is traced; part of the graph key. The default is the kernels whose
# split real-weight checks proved exact on trn2: delta_rule and dsa_topk (GLM-5.3-Flash wikitext-2 on
# trn2.48xlarge, tp=32, every 256-token chunk identical to the unsplit engine, 2026-10-04). moe_prefill's split
# fails that check (a vector-DGE out-of-bound indirect copy at 256-row chunks, under investigation) and
# moe_dedupe's decode path is not covered by it yet.
# moe_ep (kernels/moe_ep.py, feat/trn2-fast): the expert-parallel prefill and decode-v2 kernels, I-chunks of gate_up and
# output columns of down per program with a^T swapped (exact in the NKI simulator; hangs in trn2 serving, an open bug);
# kda_decode / dsa_decode: halves of the decode rows per program (bit-identical in the simulator, not on the trn2 device,
# where the decode-path gate passed: docs/neuron-notes.md "trn2 on engine-v0 70ddc1b"). dsa_fused (kernels/dsa_fused.py,
# KILN_DSA_FUSED=1): halves of the query tiles per program. Off by default.
LNC_SPLIT_KERNELS = ("moe_prefill", "moe_dedupe", "delta_rule", "dsa_topk", "moe_ep", "kda_decode", "dsa_decode",
                     "dsa_fused")
LNC_SPLIT_DEFAULT = "delta_rule,dsa_topk"
# At LNC=2 (trn2) the default adds the decode splits measured there (feat/trn2-fast, 2026-10-05, trn2.48xlarge, decode-only step at
# 16 / 32 / 64 rows per DP group 110.7 / 154.8 / 264.2 -> 87.7 / 110.7 / 180.7 ms with moe_dedupe + kda_decode + dsa_decode on
# the decode kernels; decode-path gate 28 / 32 equal, signed dlogprob +0.00020). Keyed on the runtime's LNC so that trn1, where
# moe_dedupe's split argument is part of its graph key, traces exactly as before.
# dsa_fused since 2026-10-06 (kernels/dsa_fused.py FUSED_FAMILIES: +3.2-3.7% per engine, the wikitext and check_mixed gates there).
LNC_SPLIT_DEFAULT_LNC2 = "delta_rule,dsa_topk,moe_dedupe,kda_decode,dsa_decode,dsa_fused"


def lnc_split(kernel: str) -> bool:
    v = os.environ.get("KILN_LNC_SPLIT", LNC_SPLIT_DEFAULT_LNC2 if _LNC == 2 else LNC_SPLIT_DEFAULT)
    if v in ("all", "1"):
        return True
    if v in ("0", "none", ""):
        return False
    names = {x.strip() for x in v.split(",")}
    bad = names - set(LNC_SPLIT_KERNELS)
    if bad:
        raise ValueError(f"KILN_LNC_SPLIT: unknown kernels {sorted(bad)} (have {LNC_SPLIT_KERNELS})")
    return kernel in names


def nki_grid() -> int:
    """Grid for an NKI kernel called inside an LNL graph: the runtime's LNC. "The LNC value must
    match the NEURON_LOGICAL_NC_CONFIG environment variable ... Mismatching the two will cause a
    runtime error" (nki/__init__.py). 1 on trn1; on trn2 at LNC=2 both physical cores run a kernel
    that does not split its work by program_id, writing the same output.

    Called while dynamo traces a graph, so it must not touch the runtime (torch.classes.neuron.
    Runtime() is not traceable, measured) or the filesystem: it returns the value
    configure_runtime_env cached."""
    if _LNC is None:
        raise RuntimeError("kiln.platform.configure_runtime_env() must run before an NKI kernel is traced")
    return _LNC


# Families on which the runtime's HARDWARE per-execution barrier replaces its default one
# (NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=1, a string in libnrt.so.1 of aws-neuronx-runtime-lib 2.34.10, SDK
# 2.32; runtime 2.27 release notes: NEFF start overhead "up to 50%" lower "with an on-device hardware barrier
# between ranks"). Measured on trn1.32xlarge at 32 ranks (docs/neuron-notes.md "Upstream harvest (2026-10)"): a
# graph holding one cross-chip all-reduce 5.07 -> 2.80 ms per chained launch, GLM-5.3-Flash G64 156.1 -> 159.2 and
# F0 133.8 -> 137.7 out tok/s, and it keeps the barrier's ordering (the race stress that deadlocks with the barrier
# off passes: 0 mismatches in 3000 launches). Removing the barrier instead (NEURON_RT_DISABLE_EXECUTION_BARRIER=1)
# gives the same throughput and is unsafe (silently wrong collectives on a mismatch, deadlocks). Other families
# keep the runtime default until measured there. NEURON_RT_ENABLE_HW_EXECUTION_BARRIER=0 restores it.
HW_BARRIER_FAMILIES = ("trn1",)


def configure_runtime_env() -> None:
    """Call before libtorch_neuronx_lite is imported. On trn2 / trn3 sets
    NEURON_LOGICAL_NC_CONFIG explicitly (the same value neuronx_cc_args passes to the
    compiler); on HW_BARRIER_FAMILIES the runtime's hardware execution barrier, unless the
    environment already chose; elsewhere sets nothing."""
    global _LNC
    t = target()
    if t is not None and family_of(t) in LNC_FAMILIES:
        os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc(family_of(t)))
    if t is not None and family_of(t) in HW_BARRIER_FAMILIES:
        os.environ.setdefault("NEURON_RT_ENABLE_HW_EXECUTION_BARRIER", "1")
    _LNC = lnc(family_of(t)) if t is not None else 1


def fp8_max(target_str: str) -> float:
    """Largest finite FP8 E4M3 value: vllm-neuron 0.24 (utils/dtype_utils.py): "TRN2 uses
    e4m3 (with inf), max finite = 240. TRN3 uses e4m3fn (no inf), max finite = 448", and the
    clamp is 448 when the target startswith("trn3"), else 240."""
    return 448.0 if target_str.startswith("trn3") else 240.0


def has_mx(target_str: str) -> bool:
    """MXFP8 / MXFP4 matmul (nki.isa.nc_matmul_mx) is "Available only on NeuronCore-v4 and newer"."""
    return NKI_GEN[family_of(target_str)] >= 4


def neuronx_cc_args(target_str: str, fp8: bool) -> list[str]:
    """Platform arguments for every neuronx-cc call (before KILN_CC_ARGS).

    trn1 / inf2: only the unsafe-FP8 option, and only for a graph with FP8 tensors, exactly as
    before this module existed (LNL injects it for trn2 only, compile/backend.py
    _apply_platform_compiler_args). trn2 / trn3: --logical-nc-config matching the runtime, plus
    vllm-neuron's argument set; the FP8 option joins the one --internal-hlo2tensorizer-options
    argument (a second one would replace the first), and never on trn3, where it "triggers
    NCC_EOCP001 against NKI kernels that emit OCP" (LNL backend.py)."""
    fam = family_of(target_str)
    unsafe = fp8 and not target_str.startswith("trn3")
    if fam not in LNC_FAMILIES:
        return [HLO2TENSORIZER + UNSAFE_FP8] if unsafe else []
    args = [f"--logical-nc-config={lnc(fam)}"]
    h2t = list(VLLM_NEURON_HLO2TENSORIZER)
    if os.environ.get("KILN_CC_PRESET", "vllm-neuron") == "lnl":
        h2t = []
    else:
        args += [*VLLM_NEURON_CC_ARGS, VLLM_NEURON_BACKEND]
    if unsafe:
        h2t.append(UNSAFE_FP8)
    if h2t:
        args.append(HLO2TENSORIZER + " ".join(h2t))
    return args


@dataclass(frozen=True)
class Platform:
    target: str
    family: str
    lnc: int
    instance_type: str | None

    @property
    def nki_gen(self) -> int:
        return NKI_GEN[self.family]

    @property
    def hbm_gib_per_core(self) -> float:
        """HBM per logical core at the default LNC (LNL's table). At LNC=1 on trn2 the docs say
        both physical cores "have access to the entire 24GB HBM bank"; the usable share per
        core there is not measured."""
        return HBM_GIB_PER_CORE[self.family]

    @property
    def fp8_max(self) -> float:
        return fp8_max(self.target)

    @property
    def has_mx(self) -> bool:
        return has_mx(self.target)

    @property
    def logical_cores_per_chip(self) -> int:
        return PHYSICAL_CORES_PER_CHIP[self.family] // self.lnc


def detect() -> Platform | None:
    t = target()
    if t is None:
        return None
    fam = family_of(t)
    return Platform(target=t, family=fam, lnc=lnc(fam), instance_type=instance_type())


def runtime_target() -> str:
    """The target LNL compiles for (env override, else the runtime). Needs a live device."""
    from libtorch_neuronx_lite.compile.platform import get_platform_target

    return get_platform_target()


def check_runtime() -> dict:
    """After libtorch_neuronx_lite is imported: the runtime's target must be the family this
    module chose the LNC for, and the runtime's LNC the one Kiln compiles with."""
    p = detect()
    rt = runtime_target()
    if p is not None and family_of(rt) != p.family:
        raise RuntimeError(f"runtime target {rt!r} is not the {p.family!r} Kiln configured for "
                           f"({DMI_PRODUCT}: {p.instance_type!r})")
    return describe(rt)


def versions() -> dict:
    from importlib import metadata

    out = {}
    for dist in ("neuronx-cc", "libtorch-neuronx-lite", "torch", "nki", "vllm-neuron", "vllm"):
        try:
            out[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            pass
    return out


def describe(runtime: str | None = None) -> dict:
    """One record of the platform for logs and bench results."""
    p = detect()
    if p is None:
        return {"target": None}
    rec = {"instance_type": p.instance_type, "target": runtime or p.target, "family": p.family,
           "lnc": p.lnc, "nki_gen": p.nki_gen, "hbm_gib_per_core": p.hbm_gib_per_core,
           "fp8_max": p.fp8_max, "logical_cores_per_chip": p.logical_cores_per_chip,
           "env_lnc": os.environ.get("NEURON_LOGICAL_NC_CONFIG"),
           "cc_args": shlex.join(neuronx_cc_args(runtime or p.target, fp8=False))}
    rec.update(versions())
    return rec
