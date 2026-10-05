"""Dump every option `neuronx-cc compile` accepts, from the installed compiler's own argparse.

`neuronx-cc compile --help` crashes on neuronx-cc 2.27.5334 (SDK 2.32) with "unsupported format
character" (a help string carries a bare `%`), and the driver is Cython-compiled, so the option
list cannot be read from source. This intercepts argparse when the driver parses a real compile
command line and prints each action's option strings, default, choices, nargs and raw help text
(including options registered with help=SUPPRESS). The driver parses in a forked subcommand
process and again per pipeline job (buildPipeline), so every parser it reaches is appended to a
file as it is parsed; the compile itself then fails on the missing input HLO.

    python tools/ncc_options.py [--out options.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys

def _dump(parser: argparse.ArgumentParser, argv, out: str) -> None:
    rows = []
    for a in parser._actions:
        rows.append({
            "parser": parser.prog,
            "options": list(a.option_strings) or [a.dest],
            "dest": a.dest,
            "default": repr(a.default),
            "choices": [str(c) for c in a.choices] if a.choices else None,
            "nargs": a.nargs,
            "type": getattr(a.type, "__name__", repr(a.type)) if a.type else None,
            "action": type(a).__name__,
            "help": a.help if isinstance(a.help, str) else repr(a.help),
            "argv": list(argv),
        })
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ncc_options.jsonl")
    args = ap.parse_args()
    open(args.out, "w").close()

    # The private hook every public parse_* entry point goes through (CPython 3.12 argparse),
    # including subclasses that override parse_known_args (the driver's InterceptingArgumentParser).
    orig = argparse.ArgumentParser._parse_known_args

    def intercept(self, arg_strings, *rest, **kw):
        _dump(self, arg_strings, args.out)
        return orig(self, arg_strings, *rest, **kw)

    argparse.ArgumentParser._parse_known_args = intercept
    from neuronxcc.driver import CommandDriver

    # --no-fork-subcommand keeps the compile command's parser in this process.
    sys.argv = ["neuronx-cc", "--no-fork-subcommand", "compile", "/tmp/none.hlo", "--framework", "XLA",
                "--target", "trn1"]
    try:
        CommandDriver.main()
    except SystemExit:
        pass
    seen = set()
    for line in open(args.out):
        d = json.loads(line)
        key = (d["parser"], tuple(d["options"]))
        if key in seen:
            continue
        seen.add(key)
        print(f"[{d['parser']}] {' '.join(d['options'])}  default={d['default']}  choices={d['choices']}  "
              f"nargs={d['nargs']}  action={d['action']}  help={d['help']!r}")


if __name__ == "__main__":
    main()
