"""Pair the LNL cache entries of two trees' runs of the same graphs and compare their NEFFs' instruction
streams byte for byte: does a kernel edit leave the compiled code of a platform unchanged?

    python tools/neff_stream_cmp.py [--root ~/.cache/neuron_libtorch/neuron/compile_cache] [--since <epoch s>]

Two entries pair when their FX graph text and example inputs are equal once the kernels' source revision is
taken out: the static argument `rev` (the CRC every Kiln kernel passes so that a source edit changes the
key: kernels/moe_prefill.py _kernel_rev) is replaced by a placeholder, and a trailing `rev` that one tree
passes and the other does not (moe_dedupe before 2026-10-04) is dropped. Per pair the instruction streams
(sg<N>/<Engine><N>.bin inside the NEFF's tar.gz, after its 1024-byte header) are hashed; IDENTICAL means
the same instructions on every engine. Used 2026-10-04 on trn1.2xlarge for feat/trn2-max's LNC split:
24 of 24 kernel graphs (moe_prefill, moe_dedupe, delta_rule, dsa_topk) identical to engine-v0's.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import io
import os
import re
import tarfile

STREAM = re.compile(r"sg\d+/[A-Za-z]+\d+\.bin$")


def normalised(d: str) -> str:
    t = open(os.path.join(d, "fxgraph.txt")).read()
    ex = os.path.join(d, "example_inputs.txt")
    t += open(ex).read() if os.path.exists(ex) else ""
    # KILN_LNC_SPLIT's static `spl` (feat/trn2-max) after rev: dropped with its value (grid 1 never splits)
    t = re.sub(r", -?\d+\), arg_names: \[([^\]]*), spl\]", r"), arg_names: [\1]", t)
    t = re.sub(r", -?\d+\), arg_names: \[([^\]]*), debug, rev\]", r"), arg_names: [\1, debug]", t)
    return re.sub(r"-?\d+\), arg_names: \[([^\]]*), rev\]", r"R), arg_names: [\1, rev]", t)


def streams(neff: str) -> dict[str, str]:
    data = open(neff, "rb").read()[1024:]
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        return {m.name: hashlib.sha256(tf.extractfile(m).read()).hexdigest() for m in tf.getmembers()
                if m.isfile() and STREAM.search(m.name)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.expanduser("~/.cache/neuron_libtorch/neuron/compile_cache"))
    ap.add_argument("--since", type=float, default=0.0)
    a = ap.parse_args()
    groups: dict[str, list[tuple[str, str]]] = {}
    for d in glob.glob(os.path.join(a.root, "*")):
        neffs = glob.glob(os.path.join(d, "*.neff"))
        if not os.path.exists(os.path.join(d, "fxgraph.txt")) or not neffs or os.path.getmtime(d) < a.since:
            continue
        groups.setdefault(normalised(d), []).append((d, neffs[0]))
    n = same = 0
    for ents in groups.values():
        if len(ents) < 2:
            continue
        (da, na), (db, nb) = ents[:2]
        sa, sb = streams(na), streams(nb)
        n += 1
        same += sa == sb
        diff = sorted(k for k in set(sa) | set(sb) if sa.get(k) != sb.get(k))
        print(f"{os.path.basename(da)} vs {os.path.basename(db)}: {len(sa)} / {len(sb)} streams, "
              f"{'IDENTICAL' if not diff else 'DIFFER ' + ' '.join(diff[:6])}", flush=True)
    print(f"{same} of {n} pairs identical")


if __name__ == "__main__":
    main()
