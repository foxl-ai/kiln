"""Download checkpoints into the Hugging Face cache, one after another, and report size and time.

    python tools/fetch_models.py Qwen/Qwen3-0.6B Qwen/Qwen3-8B zai-org/GLM-5.3-Flash

Same file patterns as kiln/models/loader.py (resolve_model_path). Run it on a box while device
work goes on: it is host and network only. HF_HOME decides where the cache lives
(`infra/fleet.sh nvme` points it at the instance-store RAID0).
"""

from __future__ import annotations

import os
import sys
import time

PATTERNS = ["*.json", "*.safetensors", "tokenizer*", "*.txt", "*.model", "*.jinja", "*.py"]


def main() -> None:
    from huggingface_hub import snapshot_download

    for repo in sys.argv[1:]:
        t = time.perf_counter()
        path = snapshot_download(repo, allow_patterns=PATTERNS, max_workers=int(os.environ.get("KILN_FETCH_WORKERS", "32")))
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs)
        dt = time.perf_counter() - t
        print(f"FETCHED {repo} {size / 1e9:.1f} GB in {dt:.0f} s ({size / 1e6 / max(dt, 1e-9):.0f} MB/s) at {path}",
              flush=True)


if __name__ == "__main__":
    main()
