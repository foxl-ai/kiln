"""Run named test functions of a tests/ module without its module-level skips (a box without the test-only imports).

    python tools/run_test_fns.py tests/test_dsa_long.py test_a test_b ...

Each function's source is executed alone with torch and pytest in scope; @pytest.mark.parametrize cases each run.
"""

from __future__ import annotations

import ast
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import pytest
    import torch

    path, names = sys.argv[1], sys.argv[2:]
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if not (isinstance(node, ast.FunctionDef) and node.name in names):
            continue
        cases = [{}]
        for d in node.decorator_list:
            if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("parametrize"):
                keys = [k.strip() for k in ast.literal_eval(d.args[0]).split(",")]
                vals = ast.literal_eval(d.args[1])
                vals = [v if isinstance(v, tuple) else (v,) for v in vals]
                cases = [dict(c, **dict(zip(keys, v))) for c, v in itertools.product(cases, vals)]
        node.decorator_list = []
        ns = {"torch": torch, "pytest": pytest}
        exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        for c in cases:
            ns[node.name](**c)
            print(f"PASS {node.name} {c}", flush=True)


if __name__ == "__main__":
    main()
