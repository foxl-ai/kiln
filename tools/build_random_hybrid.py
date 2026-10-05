"""A random-weight GLM-5.3-Flash or Qwen3.8-Flash-Next checkpoint built by transformers (>= 5.18)
from the real config.json (tests/test_glm5_next.py / tests/test_qwen4_exp.py builders: cut sizes,
the real block geometry), with the model's real tokenizer beside it and its vocabulary (and, for
Qwen, its EOS, which the n-gram hash reads), so tools/check_device.py can run it.

    python tools/build_random_hybrid.py glm5_next /opt/kiln/work/rand-glm5n --sparse
    python tools/build_random_hybrid.py qwen4_exp /opt/kiln/work/rand-q4 --sparse
    python tools/hf_reference.py --model /opt/kiln/work/rand-glm5n --tokens 32 --out ref.json
    python tools/check_device.py --model /opt/kiln/work/rand-glm5n --dtype fp32 --reference-json ref.json

--sparse sets index_topk / indexer_budget to 16 (pools / blocks of 4), so prompts past 16 tokens
take the sparse selection (and gives QSA's indexer 16 heads, as the CPU tests do).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOKENIZERS = {"glm5_next": "zai-org/GLM-5.3-Flash", "qwen4_exp": "Qwen/Qwen3.8-Flash-Next"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", choices=sorted(TOKENIZERS))
    ap.add_argument("out")
    ap.add_argument("--sparse", action="store_true")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mtp", action="store_true", help="glm5_next: add an MTP layer (tests/test_linear_serving.py)")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    tok = snapshot_download(TOKENIZERS[args.name], allow_patterns=["tokenizer*", "*.jinja", "vocab.json", "merges.txt",
                                                                   "config.json"])
    with open(os.path.join(tok, "config.json")) as f:
        real = json.load(f)["text_config"]
    extra = dict(vocab_size=real["vocab_size"])
    if args.name == "glm5_next":
        from tests.test_glm5_next import build

        if args.sparse:
            extra["index_topk"] = 16
    else:
        from tests.test_qwen4_exp import build

        extra.update(eos_token_id=real["eos_token_id"], bos_token_id=real["bos_token_id"])
        if args.sparse:  # 16 indexer heads: see tests/test_qwen4_exp.py's sparse fixture (exact-zero ties)
            extra.update(indexer_budget=16, indexer_n_heads=16)
    build(args.out, seed=args.seed, layers=args.layers, **extra)
    if args.mtp:
        from tests.test_linear_serving import add_glm5_next_mtp

        add_glm5_next_mtp(args.out, args.seed)
    for f in os.listdir(tok):
        if f != "config.json":
            shutil.copy(os.path.join(tok, f), args.out)
    print(f"built {args.name} -> {args.out} ({extra}, {args.layers} layers)")


if __name__ == "__main__":
    main()
