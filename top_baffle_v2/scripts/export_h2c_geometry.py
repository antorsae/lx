#!/usr/bin/env python3
"""Guarded, explicit H2C CAD targets and print-transform authorities."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    from run_memory_guarded import reexec_under_guard
    reexec_under_guard(Path(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=("stock", "slim", "obiwan", "wing"))
    ap.add_argument("owner")
    ap.add_argument("--state", choices=("no_floor_stand", "floor_stand"), default="no_floor_stand")
    ap.add_argument("--support-only", action="store_true")
    args = ap.parse_args()
    os.environ["LX_STAND_FOOT"] = "1" if args.state == "floor_stand" else "0"
    os.environ["LX_ROUTING_PROFILE"] = "proud" if args.family in {"stock", "slim"} else "obiwan"
    from build123d import Part, Pos, Rot, export_step, export_stl, import_brep, import_step
    from lx521_baffle.io import sha256_file
    from export_piece_stls import (
        _strict_mesh_facts, _canonicalize_transform_zeros,
        _remove_collapsed_apex_facets, _write_print_transform_sidecar,
    )
    dependencies = {}
    if args.support_only:
        if args.family not in {"stock", "slim"}: raise ValueError("support-only is a proud core target")
        from lx521_baffle.h2c.proud import support_tools
        from build123d import Location
        name = f"h2c_{args.family}_{args.owner}" + ("_" + args.state if args.owner.startswith("lm") else "")
        stl = ROOT / "build/h2c/STL" / (name + ".stl")
        auth = json.loads(stl.with_suffix(".print.json").read_text())
        angle = auth["rotation_deg"]["z"]
        move = auth["stl_origin_translation_mm"]
        shape = Pos(*move) * Rot(Z=angle) * Rot(X=180) * support_tools(args.family, "lm" if args.owner.startswith("lm") else args.owner)
        path = ROOT / "build/h2c/support_blockers" / (name + ".stl")
        path.parent.mkdir(parents=True, exist_ok=True)
        export_stl(shape, str(path), tolerance=.03, angular_tolerance=.12)
        print("support blocker", name, flush=True)
        return
    if args.family in {"stock", "slim"}:
        from lx521_baffle.h2c.proud import gen_part
        if args.owner in {"lm_lower", "lm_upper"}:
            from lx521_baffle.proud import b2_split as split
            from build123d import Plane, Polyline, Wire, extrude, make_face
            source=ROOT/f"build/h2c/STEP/h2c_{args.family}_lm_floor_stand.step"
            part=import_step(str(source))
            if args.owner == "lm_lower":
                region=split._above_region(split.SEAM_A_Y,split.DOVETAILS_A)
            else:
                region=split._grown(split._below_region(split.SEAM_A_Y,split.DOVETAILS_A))
            # Include the complete rearward stand, not the legacy flat-only
            # -1..19.3 mm prism; otherwise a detached foot survives the cut.
            cutter=extrude(Plane.XY.offset(-200)*make_face(Wire(Polyline(*list(region.exterior.coords)).edges())),amount=250)
            part-=cutter
            part=part.clean()
            dependencies[str(source.relative_to(ROOT))]=sha256_file(source)
        elif args.owner == "upper_bmr":
            from lx521_baffle.proud.vase_tebm35c10_4 import build_model
            part = build_model(args.family, joint="h2c").solid
        else:
            part = gen_part(args.family, args.owner)
        name = f"h2c_{args.family}_{args.owner}"
        if args.owner.startswith("lm"): name += "_" + args.state
        angle = 90.0 if args.owner == "lm" else 0.0
        for p in (ROOT/"src/lx521_baffle").rglob("*.py"):
            dependencies[str(p.relative_to(ROOT))] = sha256_file(p)
    elif args.family == "obiwan":
        from export_obiwan_staged import load_stage_manifest, staged_part_paths
        manifest = ROOT / "build" / args.state / ".obiwan_stage/manifest.json"
        # Reuse the unchanged finalized native solid, checking transaction,
        # state and every BREP hash. H2C files do not alter its CAD source.
        payload = load_stage_manifest(manifest, stand_foot=args.state == "floor_stand",
            require_active_environment=False, require_current_sources=False)
        staged = staged_part_paths(manifest, payload)
        source = staged[args.owner]
        dependencies = {str(manifest.relative_to(ROOT)): sha256_file(manifest),
                        str(source.relative_to(ROOT)): sha256_file(source)}
        part = import_brep(str(source))
        name = f"h2c_obiwan_{args.owner}_{args.state}" if args.owner == "core_lm_carrier" else f"h2c_obiwan_{args.owner}"
        angle = 26.0 if args.owner == "core_lm_carrier" else 0.0
    else:
        slug, side = args.owner.split(":")
        if slug not in {"flat", "graded"} or side not in {"left", "right"}:
            raise ValueError(args.owner)
        source = ROOT / f"build/wings/{slug}/obiwan_wing_{slug}.step"
        parts = import_step(str(source))
        child = next(child for child in parts.children if side in child.label)
        part = Part([child.solids()[0]])  # Detach the native assembly parent.
        dependencies[str(source.relative_to(ROOT))] = sha256_file(source)
        name = f"h2c_obiwan_wing_{slug}_{side}"
        angle = 43.0 if side == "left" else -43.0
    if not part.is_valid or len(part.solids()) != 1 or part.volume <= 0:
        raise RuntimeError(f"{name}: invalid native solid")
    part.label = name
    out = ROOT / "build/h2c"
    (out / "STEP").mkdir(parents=True, exist_ok=True)
    (out / "STL").mkdir(parents=True, exist_ok=True)
    step = out / "STEP" / f"{name}.step"
    export_step(part, str(step), timestamp="2020-01-01T00:00:00")
    bb = part.bounding_box()
    facts = dict(name=name, source_files=dependencies, step_sha256=sha256_file(step),
        native_valid=True, native_solids=1, native_volume_mm3=part.volume,
        installed_bounds_mm=[[bb.min.X, bb.min.Y, bb.min.Z], [bb.max.X, bb.max.Y, bb.max.Z]])
    posed = Rot(Z=angle) * Rot(X=180) * part
    pb = posed.bounding_box()
    posed = Pos(-pb.min.X, -pb.min.Y, -pb.min.Z) * posed
    stl = out / "STL" / f"{name}.stl"
    export_stl(posed, str(stl), tolerance=0.01, angular_tolerance=0.08)
    _canonicalize_transform_zeros(stl)
    removed = _remove_collapsed_apex_facets(stl)
    mesh = _strict_mesh_facts(stl)
    mesh["collapsed_apex_facets_removed"] = removed
    _write_print_transform_sidecar(stl, name=name, variant="h2c", z_rotation_deg=angle,
        oriented_bbox=pb, mesh_facts=mesh, mesh_tolerance_mm=0.01, mesh_angular_tolerance=0.08)
    facts.update(stl_sha256=sha256_file(stl), mesh=mesh,
        print_size_mm=[pb.size.X, pb.size.Y, pb.size.Z])
    (out / "STEP" / f"{name}.facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({"part":name, "size":facts["print_size_mm"], "mesh":mesh}, indent=2), flush=True)


if __name__ == "__main__":
    main()
