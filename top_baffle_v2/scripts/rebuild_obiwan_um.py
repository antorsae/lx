"""Focused, guarded STEP/STL export for the standalone Obi-Wan UM carrier.

This does not certify or regenerate the full LM/UM/tweeter release. An
optional native BREP from export_obiwan_staged's UM worker can be meshed
without rebuilding the same part; its checksum is retained as authority.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(directory))
if __name__ == "__main__":
    import run_memory_guarded
    run_memory_guarded.reexec_under_guard(Path(__file__))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", choices=("no_floor_stand", "floor_stand"), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--source-brep", type=Path)
    args = parser.parse_args()
    os.environ["LX_ROUTING_PROFILE"] = "obiwan"
    os.environ["LX_STAND_FOOT"] = "1" if args.state == "floor_stand" else "0"
    from build123d import Pos, Rot, import_brep, export_brep, export_step, export_stl
    from lx521_baffle.obiwan.carriers import um_carrier
    from export_piece_stls import (
        _strict_mesh_facts, _canonicalize_transform_zeros,
        _remove_collapsed_apex_facets, _write_print_transform_sidecar,
        OBIWAN_MESH_TOLERANCE_MM, OBIWAN_MESH_ANGULAR_TOLERANCE,
    )
    part = import_brep(str(args.source_brep)) if args.source_brep else um_carrier()
    assert part.is_valid and len(part.solids()) == 1 and part.volume > 0
    output = args.outdir
    output.mkdir(parents=True, exist_ok=True)
    name = "obiwan_core_2_of_2_um_carrier"
    part.label = name
    native = output / (name + ".brep")
    if args.source_brep is None or args.source_brep.resolve() != native.resolve():
        export_brep(part, str(native))
    step = output / (name + ".step")
    export_step(part, str(step))
    oriented = Rot(X=180) * part
    bounds = oriented.bounding_box()
    assert max(bounds.size.X, bounds.size.Y, bounds.size.Z) < 256
    printable = Pos(-bounds.min.X, -bounds.min.Y, -bounds.min.Z) * oriented
    path = output / (name + ".stl")
    temporary = path.with_name("." + path.name)
    export_stl(printable, str(temporary), tolerance=OBIWAN_MESH_TOLERANCE_MM,
               angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE)
    _canonicalize_transform_zeros(temporary)
    removed = _remove_collapsed_apex_facets(temporary)
    facts = _strict_mesh_facts(temporary)
    facts["collapsed_apex_facets_removed"] = removed
    temporary.replace(path)
    _write_print_transform_sidecar(
        path, name=name, variant="obiwan", z_rotation_deg=0,
        oriented_bbox=bounds, mesh_facts=facts,
        mesh_tolerance_mm=OBIWAN_MESH_TOLERANCE_MM,
        mesh_angular_tolerance=OBIWAN_MESH_ANGULAR_TOLERANCE)
    sources = [Path(__file__), ROOT / "src/lx521_baffle/obiwan/carriers.py",
               ROOT / "src/lx521_baffle/obiwan/bumps.py",
               ROOT / "src/lx521_baffle/obiwan/route.py"]
    manifest = {"state": args.state, "scope": "standalone_UM_only",
                "qualification": "geometry_exported_unsliced",
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
                "native_source_brep": str(args.source_brep) if args.source_brep else None,
                "native_source_sha256": sha(args.source_brep) if args.source_brep else None,
                "files": {p.name: sha(p) for p in (native, step, path, path.with_suffix('.print.json'))},
                "volume_mm3": part.volume, "mesh": facts}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
