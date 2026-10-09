"""Device memory per NeuronCore while a command runs, from the Neuron runtime's own accounting (neuron-monitor).

    python tools/hbm_monitor.py record mon.jsonl -- python bench/serve_sweep.py ...   # samples once a second
    python tools/hbm_monitor.py summary mon.jsonl [other.jsonl]                       # peak per core, side by side

`record` starts /opt/aws/neuron/bin/neuron-monitor (not on the SSM shell's PATH) with the memory_used metric of every
runtime at a 1 s period, writes its JSON lines to the file, runs the command, and stops the monitor when the command
exits (the command's exit code is this tool's). `summary` prints, per NeuronCore, the peak of the runtime's
device-memory total over the samples and the breakdown (tensors, model code, shared scratchpad, constants, runtime
memory) at that peak, plus the fullest core; with two files, the two side by side and their difference.

The breakdown is neuron-monitor's report.memory_used.neuron_runtime_used_bytes.usage_breakdown.neuroncore_memory_usage
(neuron-monitor user guide, "memory_used"); it omits the DMA rings' reservations (docs/neuron-notes.md "HBM per core"),
so a configuration can fail to load while this reads below the core's 16 GiB.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile

MONITOR = "/opt/aws/neuron/bin/neuron-monitor"
CONFIG = {"period": "1s", "neuron_runtimes": [{"tag_filter": ".*", "metrics": [{"type": "memory_used"}]}],
          "system_metrics": []}


def record(out: str, cmd: list[str]) -> int:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(CONFIG, f)
        cfg = f.name
    with open(out, "w") as sink:
        mon = subprocess.Popen([MONITOR, "-c", cfg], stdout=sink, stderr=subprocess.DEVNULL, start_new_session=True)
        try:
            rc = subprocess.call(cmd)
        finally:
            os.killpg(mon.pid, signal.SIGTERM)
            mon.wait(timeout=30)
            os.unlink(cfg)
    return rc


def _cores(line: dict) -> dict[str, dict[str, int]]:
    """{core: {kind: bytes}} of one neuron-monitor sample, over every runtime it reports."""
    out: dict[str, dict[str, int]] = {}
    for rt in line.get("neuron_runtime_data") or []:
        mu = ((rt.get("report") or {}).get("memory_used") or {}).get("neuron_runtime_used_bytes") or {}
        per = (mu.get("usage_breakdown") or {}).get("neuroncore_memory_usage") or {}
        for core, kinds in per.items():
            d = out.setdefault(str(core), {})
            for k, v in (kinds or {}).items():
                if isinstance(v, (int, float)):
                    d[k] = d.get(k, 0) + int(v)
    return out


def peaks(path: str) -> dict[str, tuple[int, dict[str, int]]]:
    best: dict[str, tuple[int, dict[str, int]]] = {}
    with open(path) as f:
        for raw in f:
            try:
                line = json.loads(raw)
            except json.JSONDecodeError:
                continue
            for core, kinds in _cores(line).items():
                tot = sum(kinds.values())
                if tot > best.get(core, (-1, {}))[0]:
                    best[core] = (tot, kinds)
    return best


def summary(paths: list[str]) -> None:
    gib = 2**30
    ps = [peaks(p) for p in paths]
    cores = sorted(set().union(*ps), key=lambda c: int(c) if c.isdigit() else c)
    print("core  " + "  ".join(f"{os.path.basename(p)[:28]:>28}" for p in paths) + ("  diff GiB" if len(ps) == 2 else ""))
    for c in cores:
        vals = [p.get(c, (0, {}))[0] for p in ps]
        row = f"{c:>4}  " + "  ".join(f"{v / gib:>28.3f}" for v in vals)
        if len(ps) == 2:
            row += f"  {(vals[1] - vals[0]) / gib:+.3f}"
        print(row)
    for p, pk in zip(paths, ps):
        if not pk:
            print(f"{p}: no runtime samples")
            continue
        c, (tot, kinds) = max(pk.items(), key=lambda kv: kv[1][0])
        parts = ", ".join(f"{k} {v / gib:.3f}" for k, v in sorted(kinds.items()))
        print(f"{p}: fullest core {c} at {tot / gib:.3f} GiB ({parts})")


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "record":
        if "--" not in sys.argv:
            raise SystemExit("usage: hbm_monitor.py record <out.jsonl> -- <command...>")
        i = sys.argv.index("--")
        raise SystemExit(record(sys.argv[2], sys.argv[i + 1:]))
    if len(sys.argv) >= 3 and sys.argv[1] == "summary":
        summary(sys.argv[2:])
        return
    raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
