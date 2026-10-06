"""Run a tool with environment variables set first, for `infra/fleet.sh bg` (which passes arguments, not an
environment):

    infra/fleet.sh bg <box> tools/with_env.py KILN_A=1 KILN_B=x -- tools/profile_layer.py --model ...

Everything before `--` is KEY=VALUE; the script after it replaces this process (exec) with the rest as its argv, so
it and every process it spawns see the variables.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    argv = sys.argv[1:]
    cut = argv.index("--")
    env = dict(os.environ)
    for kv in argv[:cut]:
        k, v = kv.split("=", 1)
        env[k] = v
    script = argv[cut + 1]
    if not os.path.isabs(script):
        script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), script)
    os.execve(sys.executable, [sys.executable, script] + argv[cut + 2:], env)
