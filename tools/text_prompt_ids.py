"""Token ids of a real text for bench/serve_sweep.py --prompt-ids: wikitext-103 (Hugging Face dataset
Salesforce/wikitext, config wikitext-103-raw-v1, train split) tokenized with the model's tokenizer, in document order.

    python tools/text_prompt_ids.py --out /opt/kiln/wt103-glm.npy [--tokens 3000000] [--tokenizer zai-org/GLM-5.3-Flash]

The parquet shards are fetched with huggingface_hub (hf_hub_download, repo_type="dataset"); lines are joined as the
dataset stores them (paragraphs and " = Heading = " lines), empty lines dropped, until --tokens ids are written.
"""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=3_000_000)
    ap.add_argument("--tokenizer", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--repo", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-103-raw-v1")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    files = sorted(f for f in HfApi().list_repo_files(a.repo, repo_type="dataset")
                   if f.startswith(f"{a.config}/train") and f.endswith(".parquet"))
    out: list[int] = []
    for f in files:
        path = hf_hub_download(a.repo, f, repo_type="dataset")
        lines = [t for t in pq.read_table(path, columns=["text"]).column("text").to_pylist() if t.strip()]
        for i in range(0, len(lines), 2000):
            out += tok("".join(lines[i : i + 2000]), add_special_tokens=False)["input_ids"]
            if len(out) >= a.tokens:
                break
        if len(out) >= a.tokens:
            break
    arr = np.asarray(out[: a.tokens], dtype=np.int32)
    np.save(a.out, arr)
    print(f"{len(arr)} tokens of {a.repo}/{a.config} from {len(files)} shards -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
