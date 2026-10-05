"""Print the HLO that libtorch_neuronx_lite handed neuronx-cc for a compile-cache entry, one
instruction per line (name, opcode, shape, operands), optionally only around some names.

    python tools/hlo_dump.py <cache dir or graph.hlo> [--grep gather convert.77] [--users]
"""

from __future__ import annotations

import argparse
import os

from libtorch_neuronx_lite.pyhlo.service import hlo_pb2

TYPES = {1: "pred", 2: "s8", 3: "s16", 4: "s32", 5: "s64", 6: "u8", 7: "u16", 8: "u32", 9: "u64", 10: "f16",
         11: "f32", 12: "f64", 16: "bf16", 19: "f8e5m2", 20: "f8e4m3fn"}


def shape_str(s) -> str:
    if s.tuple_shapes:
        return "(" + ", ".join(shape_str(t) for t in s.tuple_shapes) + ")"
    return f"{TYPES.get(s.element_type, s.element_type)}{list(s.dimensions)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", nargs="*", default=None, help="print only instructions whose name or opcode contains one")
    ap.add_argument("--users", action="store_true", help="also print the users of every match")
    args = ap.parse_args()
    p = args.path
    if os.path.isdir(p):
        p = os.path.join(p, "graph.hlo")
    m = hlo_pb2.HloModuleProto()
    with open(p, "rb") as f:
        m.ParseFromString(f.read())
    for comp in m.computations:
        ids = {i.id: i for i in comp.instructions}
        users: dict[int, list] = {}
        for i in comp.instructions:
            for o in i.operand_ids:
                users.setdefault(o, []).append(i)
        rows = comp.instructions
        if args.grep is not None:
            rows = [i for i in rows if any(g in i.name or g == i.opcode for g in args.grep)]
        if not rows:
            continue
        print(f"computation {comp.name} ({len(comp.instructions)} instructions)")
        for i in rows:
            ops = ", ".join(ids[o].name if o in ids else str(o) for o in i.operand_ids)
            extra = ""
            if i.opcode == "gather":
                d = i.gather_dimension_numbers
                extra = (f" offset_dims={list(d.offset_dims)} collapsed={list(d.collapsed_slice_dims)} "
                         f"start_index_map={list(d.start_index_map)} index_vector_dim={d.index_vector_dim} "
                         f"slice_sizes={list(i.gather_slice_sizes)}")
            print(f"  {i.name:<28} {i.opcode:<18} {shape_str(i.shape):<28} ({ops}){extra}")
            if args.users:
                for u in users.get(i.id, []):
                    print(f"      -> {u.name} {u.opcode} {shape_str(u.shape)}")


if __name__ == "__main__":
    main()
