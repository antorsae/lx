"""Export a generator module's gen_step() to a STEP file with build123d's
native export_step -- a self-contained replacement for the CAD skill's
`step` tool, so the Makefile has no external CAD-skill dependency. (Unlike
the skill it writes no hidden .glb/topology viewer companions.)

A fixed header timestamp keeps rebuilds from churning the STEP header.

Run:  python scripts/export_steps.py <module.py|dotted.module> --output <path.step>
"""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

# Public direct use receives the same process-tree/free-memory guard as Make.
if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

FIXED_TIMESTAMP = "2020-01-01T00:00:00"


def validate_step_transaction(path: Path) -> None:
    if path.stat().st_size < 1024:
        raise RuntimeError(f"temporary STEP is truncated: {path}")
    with path.open("rb") as stream:
        header = stream.read(32)
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read()
    if (not header.startswith(b"ISO-10303-21;")
            or not tail.rstrip().endswith(b"END-ISO-10303-21;")):
        raise RuntimeError(f"temporary STEP transaction is incomplete: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="generator module, e.g. foo.py or foo")
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    source_stem = (
        Path(args.source).stem if args.source.endswith(".py")
        else args.source)
    if (source_stem.startswith("top_baffle_nd25fw4_obiwan")
            or source_stem.startswith("lx521_baffle.obiwan.")):
        raise SystemExit(
            "direct Obi-Wan STEP generation is disabled because it retains "
            "the complete LM OCC build tree; use export_obiwan_staged.py")

    # Delay the CAD-kernel import until after the fail-closed Obi-Wan gate.
    from build123d import export_step

    module = importlib.import_module(source_stem)
    shape = module.gen_step()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.stem}.{os.getpid()}.tmp.step")
    try:
        export_step(shape, str(temporary), timestamp=FIXED_TIMESTAMP)
        validate_step_transaction(temporary)
        temporary.replace(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[export_steps] wrote {args.output}")


if __name__ == "__main__":
    main()
