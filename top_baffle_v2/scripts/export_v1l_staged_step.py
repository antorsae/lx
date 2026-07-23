"""Export the physical four-piece V1L STEP under the selected host profile.

The attested ``osado-512g`` worker builds the complete split once and exports
its four children directly.  That avoids repeating the common baffle and duct
booleans merely to satisfy the workstation's memory floor.  Explicit local
execution retains the original one-piece-per-child staging path: the
lightweight parent imports those four finished STEP transactions and writes
the same labeled review assembly.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import subprocess
import sys
import tempfile

from export_steps import validate_step_transaction


# Keep direct CLI use inside the selected authenticated process-group guard
# used by Make. Local staged children inherit its marker; the remote direct
# build remains bounded by its per-worker limit and enclosing 512 GiB cgroup.
if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))


FIXED_TIMESTAMP = "2020-01-01T00:00:00"
PIECE_ORDER = (
    "piece_bottom",
    "piece_mid_left",
    "piece_mid_right",
    "piece_top_b2",
)


def _large_host_execution() -> bool:
    """True only inside the attested high-memory remote worker profile."""
    return (
        os.environ.get("LX_CAD_EXECUTION") != "local"
        and os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def _export_one(piece_name: str, output: Path) -> None:
    from build123d import export_step
    from lx521_baffle.proud.top_baffle_nd25fw4_v1l_split import pieces_v1l

    solid = pieces_v1l(only=piece_name)[piece_name]
    solid.label = piece_name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(
        f".{output.stem}.{os.getpid()}.tmp.step")
    try:
        export_step(solid, str(temporary), timestamp=FIXED_TIMESTAMP)
        validate_step_transaction(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"[v1l-stage] wrote {piece_name}", flush=True)


def _export_assembly(output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LX_ROUTING_PROFILE", "proud")

    if _large_host_execution():
        # On osado, construct the common baffle/duct tree once.  The returned
        # children are still the exact four physical print pieces; only the
        # macOS-driven computational fragmentation is removed.
        from build123d import Compound, export_step
        from lx521_baffle.proud.top_baffle_nd25fw4_v1l_split import pieces_v1l

        pieces = pieces_v1l()
        if set(pieces) != set(PIECE_ORDER):
            raise RuntimeError(
                f"V1L split returned {sorted(pieces)}, expected "
                f"{list(PIECE_ORDER)}")
        children = []
        for piece_name in PIECE_ORDER:
            solid = pieces[piece_name]
            solid.label = piece_name
            children.append(solid)
        assembly = Compound(children=children)
        assembly.label = "lx521_4_top_baffle_nd25fw4_v1l_split"
        temporary = output.with_name(
            f".{output.stem}.{os.getpid()}.tmp.step")
        try:
            export_step(assembly, str(temporary), timestamp=FIXED_TIMESTAMP)
            validate_step_transaction(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[v1l-direct] wrote four-piece assembly {output}", flush=True)
        return

    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="lx521-v1l-step-") as tmp:
        tmp_dir = Path(tmp)
        staged = []
        for piece_name in PIECE_ORDER:
            path = tmp_dir / f"{piece_name}.step"
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--part", piece_name, "--output", str(path)],
                check=True,
                env=env,
            )
            staged.append((piece_name, path))

        # Delay the heavy CAD-kernel import until every generation child
        # has exited.  Otherwise the orchestration parent retains roughly
        # 0.5 GiB while a child is doing the memory-intensive booleans.
        from build123d import Compound, export_step, import_step

        children = []
        for piece_name, path in staged:
            solid = import_step(str(path))
            solid.label = piece_name
            children.append(solid)
        assembly = Compound(children=children)
        assembly.label = "lx521_4_top_baffle_nd25fw4_v1l_split"
        temporary = output.with_name(
            f".{output.stem}.{os.getpid()}.tmp.step")
        try:
            export_step(assembly, str(temporary), timestamp=FIXED_TIMESTAMP)
            validate_step_transaction(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"[v1l-stage] wrote assembly {output}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--part", choices=PIECE_ORDER,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.part:
        _export_one(args.part, args.output)
    else:
        _export_assembly(args.output)


if __name__ == "__main__":
    main()
