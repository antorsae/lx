"""Export a generator module's gen_step() to a STEP file with build123d's
native export_step -- a self-contained replacement for the CAD skill's
`step` tool, so the Makefile has no external CAD-skill dependency. (Unlike
the skill it writes no hidden .glb/topology viewer companions.)

A fixed header timestamp keeps rebuilds from churning the STEP header.

Run:  python export_steps.py <module.py|module> --output <path.step>
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from build123d import export_step

FIXED_TIMESTAMP = "2020-01-01T00:00:00"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="generator module, e.g. foo.py or foo")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    module = importlib.import_module(Path(args.source).stem)
    shape = module.gen_step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_step(shape, str(args.output), timestamp=FIXED_TIMESTAMP)
    print(f"[export_steps] wrote {args.output}")


if __name__ == "__main__":
    main()
