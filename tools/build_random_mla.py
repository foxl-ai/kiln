"""A random-weight MLA / DSA checkpoint built by transformers from a real config.json
(tests/test_mla.py MODELS: cut sizes, the real attention geometry), with a real tokenizer
copied beside it, so tools/check_device.py can run it against the transformers reference.

    python tools/build_random_mla.py deepseek_v32 /opt/kiln/work/rand-v32 --index-topk 16
    python tools/build_random_mla.py glm_moe_dsa /opt/kiln/work/rand-glm-mtp --index-topk 16 --mtp
    python tools/check_device.py --model /opt/kiln/work/rand-v32 --dtype float32 ...

Random weights avoid what a lightly trained checkpoint has: index scores tied exactly at 0
(every indexer head's ReLU zero) at the top-k boundary, where any tie-break is a valid top-k
but transformers' and Kiln's differ.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="a tests/test_mla.py MODELS key, e.g. deepseek_v32, glm_moe_dsa, deepseek_v3")
    ap.add_argument("out")
    ap.add_argument("--index-topk", type=int, default=None)
    ap.add_argument("--tokenizer", default="inference-optimization/GLM-5.3-0.6B-A0.4B")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--mtp", action="store_true",
                    help="add one MTP layer (tests/test_mtp_mla.py build_with_mtp: model.layers.<n>.*, a real "
                         "transformers decoder layer of the family, plus enorm / hnorm / eh_proj / shared_head.norm)")
    ap.add_argument("--copy-main", action="store_true",
                    help="with --mtp and --layers 1: the MTP layer is a copy of the target's, so drafts are the "
                         "target's own predictions one token later (a meaningful acceptance rate from random weights)")
    ap.add_argument("--fp8", action="store_true",
                    help="store it as DeepSeek-V3 / GLM-5.3 do (FP8 e4m3fn, 128x128 block scales) in OUT, and the "
                         "same weights dequantized in OUT-deq for the reference (check_device --reference)")
    args = ap.parse_args()
    import json

    from huggingface_hub import snapshot_download

    from tests.test_mla import build

    tok = snapshot_download(args.tokenizer, allow_patterns=["tokenizer*", "*.jinja", "config.json"])
    with open(os.path.join(tok, "config.json")) as f:
        extra = dict(vocab_size=json.load(f)["vocab_size"])  # the tokenizer's model's vocabulary
    if args.index_topk is not None:
        extra["index_topk"] = args.index_topk
    if args.layers is not None:
        extra["num_hidden_layers"] = args.layers
    outs = [args.out]
    if args.fp8:
        from tests.test_mla import _fp8_checkpoint

        src = args.out + "-bf"
        build(args.name, src, **extra)
        _fp8_checkpoint(src, args.out, args.out + "-deq")
        shutil.rmtree(src)
        outs.append(args.out + "-deq")
    elif args.mtp:
        from tests.test_mtp_mla import build_with_mtp

        build_with_mtp(args.name, args.out, copy_main=args.copy_main, **extra)
    else:
        build(args.name, args.out, **extra)
    for out in outs:
        for f in os.listdir(tok):  # the snapshot directory may hold the repo's weights too
            if f.startswith("tokenizer") or f.endswith(".jinja"):
                shutil.copy(os.path.join(tok, f), out)
    print(f"built {args.name} -> {args.out} ({extra})")


if __name__ == "__main__":
    main()
