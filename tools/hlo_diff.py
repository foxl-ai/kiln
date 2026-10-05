"""Are two graph.hlo files the same program? Compares HloModuleProtos with the op metadata
(op_name, source_file, source_line: where in Python each op was traced) and the module id
cleared, and prints the first differences otherwise.

    python tools/hlo_diff.py <cache dir or graph.hlo> <cache dir or graph.hlo> [--keep-metadata]

Used to check that a graph captured on a CPU host (kiln/capture.py) lowers to the same HLO as the
device run's own lowering of the same cache key.
"""

from __future__ import annotations

import argparse
import difflib
import os

from google.protobuf import text_format
from libtorch_neuronx_lite.pyhlo.service import hlo_pb2


def load(p: str, keep_metadata: bool = False):
    if os.path.isdir(p):
        p = os.path.join(p, "graph.hlo")
    m = hlo_pb2.HloModuleProto()
    with open(p, "rb") as f:
        m.ParseFromString(f.read())
    if not keep_metadata:
        m.id = 0
        for c in m.computations:
            for i in c.instructions:
                i.ClearField("metadata")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--keep-metadata", action="store_true")
    ap.add_argument("--lines", type=int, default=40)
    args = ap.parse_args()
    a, b = load(args.a, args.keep_metadata), load(args.b, args.keep_metadata)
    if a.SerializeToString(deterministic=True) == b.SerializeToString(deterministic=True):
        print("SAME", sum(len(c.instructions) for c in a.computations), "instructions")
        return
    ta, tb = text_format.MessageToString(a).splitlines(), text_format.MessageToString(b).splitlines()
    diff = list(difflib.unified_diff(ta, tb, "a", "b", n=2, lineterm=""))
    print("DIFFERENT", len([d for d in diff if d[:1] in "+-"]), "changed lines")
    print("\n".join(diff[: args.lines]))


if __name__ == "__main__":
    main()
