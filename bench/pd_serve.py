"""One engine of a disaggregated deployment as an OpenAI-compatible server, configured EXACTLY as
bench/serve_sweep.py configures its engine for the same arguments, so it runs the graphs a compile-farm
capture of that sweep produced (the KV pool, max_num_seqs and the buckets are graph input shapes).

    python bench/pd_serve.py --pd-role prefill --port 8100 -- <serve_sweep arguments>
    python bench/pd_serve.py --pd-role decode --pd-listen 0.0.0.0:7400 --port 8100 -- <serve_sweep arguments>

The role decides which graphs warm up (--warmup in the sweep arguments): a prefill engine only prefill graphs,
a decode engine only decode graphs (plus prefill ones with --pd-bypass-prefill). Then the server answers on
--port; kiln.server.pd_router fronts the engines.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: pd_serve.py [pd options] -- <serve_sweep arguments>")
    i = argv.index("--")
    ap = argparse.ArgumentParser(prog="pd_serve")
    ap.add_argument("--pd-role", required=True, choices=["prefill", "decode", "none"],
                    help="none: one engine doing both phases (the colocated comparison, same graphs as the sweep)")
    ap.add_argument("--pd-listen", default=None, help="decode: host:port for handoffs")
    ap.add_argument("--pd-advertise", default=None, help="decode: this box's address as prefill boxes see it")
    ap.add_argument("--pd-buffer-gb", type=float, default=16.0, help="decode: receive buffer on the host")
    ap.add_argument("--pd-bypass-prefill", action="store_true", help="decode: also serve whole (short) requests")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--served-model-name", default=None)
    a = ap.parse_args(argv[:i])
    import serve_sweep

    args = serve_sweep.build_parser().parse_args(argv[i + 1:])
    cfg = dataclasses.replace(serve_sweep.engine_config(args, args.core_base),
                              pd_role=None if a.pd_role == "none" else a.pd_role,
                              pd_listen=a.pd_listen, pd_advertise=a.pd_advertise, pd_buffer_gb=a.pd_buffer_gb,
                              pd_bypass_prefill=a.pd_bypass_prefill)
    from kiln.engine.engine import LLMEngine
    from kiln.server.api import build_app

    t = time.perf_counter()
    eng = LLMEngine(cfg)
    print(f"engine up {time.perf_counter() - t:.1f}s, role {a.pd_role}", flush=True)
    if args.warmup:
        w = eng.warmup()
        print(f"bucket warmup {w['seconds']:.1f}s over {w['graphs']} graphs", flush=True)
    import uvicorn

    app = build_app(eng, a.served_model_name or args.model)
    print(f"kiln pd serve: {a.pd_role} engine on port {a.port}", flush=True)
    # The router keeps connections to every engine alive and drops idle ones after 30 s (kiln.server.pd_router); uvicorn
    # closing them first (its default keep-alive is 5 s) raced a request onto a closing connection: one prefill POST in
    # 1,320 of a G1 4:1 level failed with ReadError (2026-10-06).
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning", timeout_keep_alive=600)


if __name__ == "__main__":
    main()
