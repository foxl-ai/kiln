"""Push / pull the NEFF compile cache to the fleet's S3 prefix, and prove the round trip.

    python tools/cache_sync.py push|pull|roundtrip [s3-uri]
    python tools/cache_sync.py pull-keys <s3-uri> <keys.json>...   # only these entries (+ the NKI binaries)

pull-keys takes a capture's keys.json or a farm queue's configs/<name>.keys.json (local paths or s3:// URIs):
a device run with KILN_COMPILE_FARM=<queue> fetches only the graphs that queue compiled, not the ones that were
already in the cache when it was enqueued.
"""

import os
import shutil
import sys

from kiln import compile_cache

URI = os.environ.get("KILN_CACHE_URI", "")  # s3://<bucket>/compile-cache/sdk-2.32/

if __name__ == "__main__":
    cmd = sys.argv[1]
    uri = sys.argv[2] if len(sys.argv) > 2 else URI
    root = compile_cache.local_root()
    if cmd == "push":
        print("pushed", compile_cache.push(uri))
    elif cmd == "pull":
        print("pulled", compile_cache.pull(uri))
    elif cmd == "pull-keys":
        import json
        import subprocess

        keys = set()
        for f in sys.argv[3:]:
            text = (subprocess.run(["aws", "s3", "cp", "--quiet", f, "-"], capture_output=True, text=True, check=True).stdout
                    if f.startswith("s3://") else open(f).read())
            keys |= set(json.loads(text))
        missing = [k for k in sorted(keys) if not compile_cache.fetch_key(uri, k)]
        compile_cache.pull_nki(uri)
        print(f"pull-keys: {len(keys) - len(missing)} of {len(keys)} entries local; not in {uri}: {missing}")
        if missing:
            sys.exit(1)
    elif cmd == "roundtrip":
        print("pushed", compile_cache.push(uri))
        keys = sorted(k for k in os.listdir(root) if os.path.exists(os.path.join(root, k, compile_cache.MARKER)))
        victim = os.path.join(root, keys[0])
        files = sorted(os.listdir(victim))
        neff = [f for f in files if f.endswith(".neff")]
        size = os.path.getsize(os.path.join(victim, neff[0]))
        shutil.rmtree(victim)
        print("removed", keys[0], "pulled", compile_cache.pull(uri))
        back = sorted(os.listdir(victim))
        assert back == [f for f in files if not f.endswith(".lock")], (files, back)
        assert os.path.getsize(os.path.join(victim, neff[0])) == size
        print(f"round trip ok: {keys[0]} restored with {len(back)} files, NEFF {size} bytes")
