"""A following stage of a layer pipeline (engine/pp.py "Following stages", EngineConfig.pp_follow): stages 1 .. S - 1
of a pipeline whose stage 0 alone takes requests (tools/lc_ttft.py --pp-follow, or a server). Builds the engine from the
same serve_sweep arguments as stage 0's, runs the bucket warmup in step with it, then follows stage 0's plan frames until
stage 0 closes the run. Every request this stage finishes is recorded with its TTFT from stage 0's arrival (the wall
clocks of the boxes; on the last stage that is the pipeline's TTFT) and, with logprobs, its first token.

    python tools/pp_follow.py --out-json s1.json -- <serve_sweep args> --pp-follow --pp-stages 4 --pp-stage 1 ...

In a disaggregated deployment (--pd-role prefill) every stage hands its own layers' share of each request off to the
decode engine stage 0's request names (engine/pp.py with engine/disagg.py), and --health-port serves the /health the PD
router checks (kiln.server.pd_router --prefill-units): the stage's role, handoff layout, stage and layers.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--pp-barrier", default=None,
                    help="host:port of the run's barrier (tools/lc_ttft.py --pp-barrier; the last stage listens): every "
                         "stage meets once after its warmup, so stage 0 takes no request before the others are ready")
    ap.add_argument("--pd-role", default=None, choices=["prefill"],
                    help="a disaggregated pipeline's stage: hand each request's share off (EngineConfig.pd_role)")
    ap.add_argument("--health-port", type=int, default=0, help="serve GET /health on this port (0: none)")
    own, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    import serve_sweep as ss

    args = ss.build_parser().parse_args(rest + ["--concurrency", "1", "--requests", "1"])
    if not (args.pp_follow and args.pp_stages > 1 and args.pp_stage > 0):
        raise SystemExit("pp_follow.py runs stages 1 .. S - 1 of a --pp-follow pipeline")
    args.skip_warm_request = True
    from lc_ttft import _barrier

    if own.pd_role:
        import dataclasses

        from kiln.engine.engine import LLMEngine

        t = time.perf_counter()
        eng = LLMEngine(dataclasses.replace(ss.engine_config(args, args.core_base), pd_role=own.pd_role))
        print(f"engine up {time.perf_counter() - t:.1f}s, role {own.pd_role}", flush=True)
    else:
        eng = ss.make_engine(args, args.core_base)
    meet = _barrier(own.pp_barrier, args.pp_stage, args.pp_stages) if own.pp_barrier else (lambda: None)
    ss.warm(eng, args, lambda: [1], ss.sampling(args))  # the bucket warmup, in step with stage 0's
    if own.health_port:
        _serve_health(eng, own.health_port)
    meet()
    rows = []
    go = True
    # A SIGTERM (tools/pp_box.sh stop) ends the loop between steps, so the stage still writes its RESULT and closes
    # (its workers, KILN_TIMELINE, the local-K line) rather than dying under SIG_DFL.
    term: list[int] = []
    signal.signal(signal.SIGTERM, lambda sig, _f: term.append(sig))
    t0 = time.perf_counter()
    while go and not term:
        go, done = eng.follow(0.2)
        if eng.pd_role == "prefill":
            eng.has_work()  # also takes the decode engine's releases of this stage's pinned pages (nixl)
        for r in done:
            row = {"rid": r.rid, "input_len": r.num_prompt, "ttft_s": round(r.first_token_time - r.arrival_time, 3)
                   if r.first_token_time else None, "cached_tokens": max(r.num_cached_tokens, 0)}
            if r.output_ids:
                row["first_token"] = r.output_ids[0]
            if r.logprobs:
                row["first_logprob"] = r.logprobs[0][0]
            rows.append(row)
            print(f"stage {args.pp_stage}: {row}", flush=True)
    res = {"model": args.model, "stage": args.pp_stage, "stages": args.pp_stages, "seconds": round(time.perf_counter() - t0, 1),
           "rows": rows}
    print("RESULT", json.dumps(res), flush=True)
    if own.out_json:
        with open(own.out_json, "w") as f:
            json.dump(res, f)
    if eng.pd_role == "prefill":
        eng.pd_flush()
    eng.close()


def _serve_health(eng, port: int) -> None:
    """GET /health for the PD router, on a thread: role, handoff layout, stage and layer range (the router checks every
    stage of a prefill unit before it routes to the unit's stage 0)."""
    import http.server
    import threading

    lo, hi = eng.runner.model.load_range or (None, None)
    body = json.dumps({"status": "ok", "role": eng.pd_role or "both", "max_num_seqs": eng.cfg.max_num_seqs,
                       **({"layout": eng.runner.pd_signature()} if eng.pd_role else {}),
                       "pp": {"stage": eng.cfg.pp_stage, "stages": eng.cfg.pp_stages, "layers": [lo, hi],
                              "follow": True}}).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, name="pp-health", daemon=True).start()
    print(f"stage {eng.cfg.pp_stage}: /health on port {port}", flush=True)


if __name__ == "__main__":
    main()
