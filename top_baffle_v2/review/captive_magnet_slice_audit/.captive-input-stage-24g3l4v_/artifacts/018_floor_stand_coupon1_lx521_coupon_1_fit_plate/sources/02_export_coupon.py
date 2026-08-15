"""Export print-calibration and physical-fit coupons as separate STLs.
STL files (lx521_coupon_*.stl) -- one body per file, each transformed by the
same source X=180-degree front-face-down process orientation as the released
baffle pieces. The slicer may translate/arrange them but must not reorient.

  1 fit_plate    dovetail female pocket (grown by the working
                 CLEARANCE_MM) + O6.5 x 2.0 entry over the unchanged
                 O6.4 x 6.8 total-depth W22 bore and O4.6 x 4.0
                 (10F) heat-set bores opening UP + the V1 upper captive
                 magnet wall section (D5.20 x 2.10 cavity, 0.45-mm
                 interface/inner skins and a 45-degree closing roof)
  2 fit_key      loose male dovetail key (no clearance) -- tune X-Y
                 hole compensation until it slides snug into the plate
  3 fish_entry   the shared no-foot three-port cluster: LM above,
                 one O6 T trunk lower-left, UM lower-right
  4 um_outlet_proud  real B2 outline + the complete R6P R14 outlet
  5 fish_ts_dive the proud R6P TS notch/oval at full 18.3 depth
  6 fish_foot    a stand-foot R14 elbow pair
  7 recess_seat  Obi-Wan core: a U22REX/P-SL recess-seat sector with
                 ~25 mm of through-void inboard of the
                 D190 cutout edge, the rotated 240-deg insert bore over its
                 rear pad. METHOD -- you never lower
                 the driver into the block: sit the driver CONE-UP on
                 its magnet on the table and FLIP THE BLOCK front-
                 face-down onto it, so the seat drops over the flange
                 edge (the through-void clears the cone/surround; the
                 motor never matters). Straightedge across block face
                 vs flange top = flushness; rotate the driver so a
                 flange hole meets the block's pilot for a REAL
                 M5 x 12 clamp test into the insert-on-pad stack.
                 Verify the owner's 6.0 mm seat depth before printing.
  8 fish_ts_oval_proud  proud-family tweeter oval/morph rehearsal
 9 um_faston_clocking  D104 clocking gauge: screw marks 238/328 and
                 terminal witness at their exact 283 degree midpoint
 12 obiwan_closed_bore_bump  state-specific R6F LM-collar sector at the
                 300-deg axis: continuous D8.2 tunnel cover, preserved
                 pad/clearance, and the complete smooth rear Z bump

Fit pieces: same profile as the real parts. Fishing blocks: 2 walls /
~8 % infill (practice holes, not structure). Dry-fish 3-8 before
committing kilograms.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib
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

from lx521_baffle.print_contract import (
    front_down_transform_record,
    sidecar_path_for_stl,
    write_print_sidecar,
)
from lx521_baffle.stl_export import (
    BinaryStlLayoutError,
    canonicalize_near_zero_stl_coordinates,
    stl_topology_defects,
    validate_binary_stl_length,
)

NECK, HEAD, DEPTH = 10.0, 14.0, 6.0   # seam-B key proportions
BED_MM = 256.0

# Rot(X=180) can leave a mathematically-zero coordinate as tiny, face-local
# float noise in OCC's binary STL tessellation.  Exact edge-parity then sees
# an artificial seam even though the source BREP is a valid solid.  Keep this
# identical to the production piece/wing exporters: 0.2 nm is 250,000 times
# below the 0.05-mm mesh deflection, so this is rigid-transform
# canonicalization, never vertex welding or generic mesh repair.
STL_TRANSFORM_ZERO_EPSILON_MM = 2.0e-7

# Coupon 12 is a Boolean crop of the finalized Obi-Wan carrier rather than a
# purpose-built low-complexity calibration solid.  Its near-surface UM route,
# cover and native R113.8 fairing therefore inherit the same closely spaced
# faces as the production carrier.  Use the production Obi-Wan tessellation
# for that coupon; the generic coupon mesh is intentionally coarser and can
# leave nonconformal edge sampling across those faces even though the source
# BREP and the complete released carrier are valid solids.
DEFAULT_STL_TOLERANCE_MM = 0.05
DEFAULT_STL_ANGULAR_TOLERANCE_RAD = 0.2
OBIWAN_CROP_STL_TOLERANCE_MM = 0.01
OBIWAN_CROP_STL_ANGULAR_TOLERANCE_RAD = 0.08


def _validate_binary_stl(path: Path) -> None:
    try:
        validate_binary_stl_length(path)
    except BinaryStlLayoutError as exc:
        if exc.truncated_header:
            raise RuntimeError(
                f"temporary coupon STL is truncated: {path}") from None
        raise RuntimeError(
            "temporary coupon STL invalid: "
            f"triangles={exc.triangle_count} "
            f"bytes={exc.actual_bytes} expected={exc.expected_bytes}"
        ) from None


def _canonicalize_transform_zeros(
        path: Path, epsilon_mm: float = STL_TRANSFORM_ZERO_EPSILON_MM) -> int:
    """Replace only sub-nanometre transform roundoff with exact +0.0."""
    if epsilon_mm <= 0.0:
        raise ValueError("STL transform-zero epsilon must be positive")
    try:
        return canonicalize_near_zero_stl_coordinates(path, epsilon_mm)
    except BinaryStlLayoutError as exc:
        if exc.truncated_header:
            raise RuntimeError(
                f"temporary coupon STL is truncated: {path}") from None
        raise RuntimeError(
            "temporary coupon STL invalid: "
            f"triangles={exc.triangle_count} "
            f"bytes={exc.actual_bytes} expected={exc.expected_bytes}"
        ) from None


def _strict_mesh_facts(path: Path) -> dict[str, int | float]:
    """Apply the unchanged release topology gate before publication."""
    # Lazy import keeps ordinary source discovery free of checker work while
    # retaining one authoritative edge/component implementation.
    facts, defects = stl_topology_defects(path)
    if defects:
        raise RuntimeError(
            f"temporary coupon STL fails strict manifold contract: "
            f"{path.name}: {defects}; triangles={facts['triangles']} "
            f"components={facts['components']}")
    return facts


COUPON_GROUPS = (
    "fit_1_2",
    "fish_3",
    "outlet_4",
    "fish_5",
    "fish_6",
    "seat_7",
    "fish_8",
    "clock_9",
    "bump_12",
)


def _large_host_execution() -> bool:
    """Use bounded group fan-out only inside the Osado release cgroup."""
    return (
        os.environ.get("LX_CAD_EXECUTION") != "local"
        and os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )

_STATEFUL_MODULES = (
    "lx521_baffle.obiwan.attachments",
    "lx521_baffle.obiwan.assembled",
    "lx521_baffle.obiwan.split",
    "lx521_baffle.obiwan.carriers",
    "lx521_baffle.obiwan.route",
    "lx521_baffle.obiwan.bridge",
    "lx521_baffle.um_fit",
    "lx521_baffle.flush",
    "lx521_baffle.cables",
    "lx521_baffle.base",
)


def _set_state(mode: str, profile: str):
    """Purge the complete flag-dependent graph before importing geometry."""
    os.environ["LX_STAND_FOOT"] = mode
    os.environ["LX_ROUTING_PROFILE"] = profile
    for module in _STATEFUL_MODULES:
        sys.modules.pop(module, None)
    importlib.invalidate_caches()


def _poly_prism(poly, h):
    from build123d import Plane, Polyline, Wire, extrude, make_face

    pts = list(poly.exterior.coords)
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XY.offset(-0.5) * face, amount=h + 1.0)


def _fit_pieces() -> dict:
    from build123d import (
        Box, Cylinder, Plane, Polyline, Pos, Wire, extrude, make_face,
    )
    from shapely import box as sbox
    from shapely.ops import unary_union
    from lx521_baffle.base import (
        L22_PILOT_DEPTH_MM,
        THICKNESS_MM,
        m5_insert_bore_cutter,
    )
    from lx521_baffle.proud.b2_split import (
        _grown,
        _trapezoid_up,
    )
    from lx521_baffle.magnets import apply_wall_cavity

    t = THICKNESS_MM
    plate = Pos(0.0, 20.0, t / 2.0) * Box(92.0, 40.0, t)
    # female dovetail pocket in the y=40 edge (grown like the real seams)
    pocket = _grown(_trapezoid_up(-28.0, 40.0 - DEPTH, NECK, HEAD, DEPTH))
    plate -= _poly_prism(pocket, t)
    # heat-set bores from the top face (open UP for easy insert setting)
    plate -= m5_insert_bore_cutter(
        (0.0, 20.0),
        opening_z=t,
        total_depth=L22_PILOT_DEPTH_MM,
        opening_side="+z",
    )
    plate -= Pos(14.0, 20.0, t - 2.0) * Cylinder(2.3, 4.2)
    # V1 upper wall station, kept as a release regression coupon.  It now
    # uses the exact production pause-and-bury cradle/roof rather than an
    # externally accessible glue pocket.
    # Do not retain the old exposed-pocket inspection notch here.  The
    # qualified loading chimney already keeps the cavity open through the
    # insertion layer; removing the rear slab through z=10.1 would delete
    # the gable apex at z=9.1 and its complete 0.45-mm post-apex seal.
    plate, _coupon_tools = apply_wall_cavity(
        plate,
        name="coupon1_v1_upper_base",
        face=(28.0, 0.0, 14.4),
        outward=(0.0, -1.0, 0.0),
        owner="base",
        print_up=(0.0, 0.0, -1.0),
        bed_datum=(0.0, 0.0, t),
    )
    # one clean extrusion, z 0..t: trapezoid + a grip handle on the
    # HEAD (wide) side, so when the key drops into the plate's edge
    # slot the handle sticks OUT past the edge (a neck-side handle would
    # ram into the plate body -- it cannot then be inserted).
    key2d = unary_union([_trapezoid_up(0.0, 0.0, NECK, HEAD, DEPTH),
                         sbox(-10.0, DEPTH - 0.01, 10.0, DEPTH + 12.0)])
    kface = make_face(Wire(Polyline(*key2d.exterior.coords).edges()))
    male = extrude(Plane.XY * kface, amount=t)
    return {"1_fit_plate": plate, "2_fit_key": male}


def _find_obiwan_stage_manifest(output_dir: Path) -> Path | None:
    """Find the state-local native-BREP transaction above a staging dir."""
    output_dir = Path(output_dir).resolve()
    for root in (output_dir, *output_dir.parents):
        candidate = root / ".obiwan_stage" / "manifest.json"
        if candidate.is_file():
            return candidate
        if root == PROJECT_ROOT.resolve():
            break
    return None


def _staged_lm_carrier(output_dir: Path, target_mode: str):
    """Import the already-qualified LM carrier when its stage is available."""
    manifest = _find_obiwan_stage_manifest(output_dir)
    if manifest is None:
        return None
    from build123d import import_brep
    from export_obiwan_staged import load_stage_manifest, staged_part_paths

    payload = load_stage_manifest(
        manifest, stand_foot=target_mode == "1")
    paths = staged_part_paths(manifest, payload)
    carrier = import_brep(str(paths["core_lm_carrier"]))
    solids = tuple(carrier.solids())
    if (not carrier.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "staged LM carrier for coupon crop is not one valid solid; "
            f"valid={carrier.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    print(
        f"[coupon-stage-reuse] imported {paths['core_lm_carrier']} "
        f"for state={payload['state']}",
        flush=True,
    )
    return carrier


def _fishing_pieces(
        target_mode: str, only: str | None = None,
        *, output_dir: Path | None = None) -> dict:
    """Real duct geometry carved from a region box, one body per hard
    fishing spot."""
    # name -> (LX_STAND_FOOT, region box, rear z0, special geometry)
    specs = {
        "3_fish_entry": ("0", (-24.0, 42.0, 14.0, 70.0), 0.0, None),
        "4_um_outlet_proud": ("1", (-6.0, 292.0, 38.0, 323.0),
                               0.0, "real_outline"),
        "5_fish_ts_dive": ("0", (-44.0, 390.0, 0.0, 434.0), 0.0, None),
        "6_fish_foot": ("1", (-20.0, 2.0, 20.0, 32.0), -22.0, None),
        # Seat/pad geometry is state-invariant; force bridge-free floor
        # ownership so this fit block never inherits the no-floor tail.
        "7_recess_seat": ("1", (-82.0, 84.0, -22.0, 141.0),
                           0.0, "lm_core"),
        "8_fish_ts_oval_proud": ("1", (-58.0, 318.0, 2.0, 430.0),
                                   0.0, None),
        "12_obiwan_closed_bore_bump": (
            target_mode, (62.0, 118.0, 122.0, 180.0),
            -14.0 if target_mode == "1" else -7.0, "obiwan_bump"),
    }
    out = {}
    for name, (mode, (x0, y0, x1, y1), z0, special) in specs.items():
        if only is not None and name != only:
            continue
        profile = ("obiwan" if special in
                   {"lm_core", "obiwan_bump", "um_core"} else "proud")
        _set_state(mode, profile)
        from build123d import Box, Pos
        from lx521_baffle.base import THICKNESS_MM

        cab = importlib.import_module("lx521_baffle.cables")
        crop = Pos((x0 + x1) / 2.0, (y0 + y1) / 2.0,
                   (z0 + THICKNESS_MM) / 2.0) * Box(
            x1 - x0, y1 - y0, THICKNESS_MM - z0)
        if special == "real_outline":
            from lx521_baffle.base import baffle_solid
            from lx521_baffle.proud.b import TWEETER_DROP_MM
            from lx521_baffle.proud.b2 import OUTLINE_B2
            blk = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM) & crop
        elif special in {"lm_core", "obiwan_bump", "um_core"}:
            staged_lm = (
                _staged_lm_carrier(output_dir, target_mode)
                if (output_dir is not None
                    and special in {"lm_core", "obiwan_bump"})
                else None
            )
            if staged_lm is not None:
                carrier = staged_lm
            else:
                from lx521_baffle.obiwan.carriers import lm_carrier, um_carrier
                carrier = (um_carrier() if special == "um_core"
                           else lm_carrier())
            blk = carrier & crop
            # Coupon 7's current x=-82..-22/y=84..141 seat crop contains no
            # released captive site (the lower LM pair is at y=18).  Keep the
            # diagnostic crop unchanged and do not add disconnected fill
            # lands outside it.
        else:
            blk = crop
        if special not in {"lm_core", "obiwan_bump", "um_core"}:
            for c in cab.cable_cutters():
                blk -= c
        out[name] = blk
    return out


def _front_face_down(solid):
    """Return a coupon and its exact production front-down transform."""
    from build123d import Pos, Rot

    oriented = Rot(X=180.0) * solid
    bb = oriented.bounding_box()
    translation = (
        -float(bb.min.X), -float(bb.min.Y), -float(bb.min.Z))
    transform = front_down_transform_record(
        [float(bb.min.X), float(bb.min.Y), float(bb.min.Z)],
        z_rotation_deg=0.0,
    )
    return Pos(*translation) * oriented, transform


def _clocking_piece(target_mode: str):
    """Terminal clocking gauge at the exact 238/283/328-degree axes."""
    import math
    from build123d import Box, Cylinder, Pos, Rot

    _set_state(target_mode, "obiwan")

    from lx521_baffle.base import (UM_PILOT_D_MM, UM_PILOT_PCD_MM,
                                    UM_TERMINAL_CLOCK_DEG)

    gauge = Pos(0.0, 0.0, 1.0) * Cylinder(52.0, 2.0)
    gauge -= Pos(0.0, 0.0, 1.0) * Cylinder(40.0, 3.0)
    for a in (238.0, 328.0):
        x = UM_PILOT_PCD_MM / 2.0 * math.cos(math.radians(a))
        y = UM_PILOT_PCD_MM / 2.0 * math.sin(math.radians(a))
        gauge -= Pos(x, y, 1.0) * Cylinder(UM_PILOT_D_MM / 2.0, 3.0)
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    mx, my = 49.0 * math.cos(a), 49.0 * math.sin(a)
    gauge -= Pos(mx, my, 1.0) * Rot(Z=UM_TERMINAL_CLOCK_DEG) * Box(
        8.0, 1.2, 3.0)

    return {"9_um_faston_clocking": gauge}


_FISHING_GROUPS = {
    "fish_3": "3_fish_entry",
    "outlet_4": "4_um_outlet_proud",
    "fish_5": "5_fish_ts_dive",
    "fish_6": "6_fish_foot",
    "seat_7": "7_recess_seat",
    "fish_8": "8_fish_ts_oval_proud",
    "bump_12": "12_obiwan_closed_bore_bump",
}

_EXPECTED_COUPON_FILES = {
    "lx521_coupon_1_fit_plate.stl",
    "lx521_coupon_2_fit_key.stl",
    "lx521_coupon_3_fish_entry.stl",
    "lx521_coupon_4_um_outlet_proud.stl",
    "lx521_coupon_5_fish_ts_dive.stl",
    "lx521_coupon_6_fish_foot.stl",
    "lx521_coupon_7_recess_seat.stl",
    "lx521_coupon_8_fish_ts_oval_proud.stl",
    "lx521_coupon_9_um_faston_clocking.stl",
    "lx521_coupon_12_obiwan_closed_bore_bump.stl",
}

_EXPECTED_COUPON_SIDECARS = {
    sidecar_path_for_stl(name).name for name in _EXPECTED_COUPON_FILES
}


def _pieces_for_group(group: str, target_mode: str, output_dir: Path):
    if group == "fit_1_2":
        _set_state(target_mode, "proud")
        return _fit_pieces()
    if group in _FISHING_GROUPS:
        return _fishing_pieces(
            target_mode, _FISHING_GROUPS[group], output_dir=output_dir)
    if group == "clock_9":
        return _clocking_piece(target_mode)
    raise ValueError(group)


def _prune_legacy_coupon_outputs(stl_dir: Path) -> None:
    stale = {
        "lx521_coupon.stl",
        "lx521_coupon_4_fish_um_bend.stl",
        "lx521_coupon_4_fish_um_exit.stl",
        "lx521_coupon_8_fish_um_oval.stl",
        "lx521_coupon_12_obiwan_open_bore_jump.stl",
        "lx521_coupon_13_obiwan_crown_crossover.stl",
        "lx521_coupon_10_um_split_grommet_half_a.stl",
        "lx521_coupon_11_um_split_grommet_half_b.stl",
        "lx521_coupon_14_obiwan_grommet_receiver.stl",
    }
    for name in stale:
        stl_path = stl_dir / name
        for path in (stl_path, sidecar_path_for_stl(stl_path)):
            path.unlink(missing_ok=True)


def _run_group_guarded(group: str, stl_dir: Path, target_mode: str) -> None:
    env = os.environ.copy()
    env["LX_STAND_FOOT"] = target_mode
    # The group builder selects Obi-Wan only for R6F coupons. Start from the
    # proud profile so inherited shell state can never trip proud imports.
    env["LX_ROUTING_PROFILE"] = "proud"
    guard = Path(__file__).with_name("run_memory_guarded.py")
    subprocess.run(
        [sys.executable, str(guard), "--", sys.executable,
         str(Path(__file__).resolve()), "--outdir", str(stl_dir),
         "--group", group],
        env=env, check=True)


def _export_group(group: str, stl_dir: Path, target_mode: str) -> None:
    from build123d import export_stl

    pieces = _pieces_for_group(group, target_mode, stl_dir)
    for name, solid in pieces.items():
        raw_solids = list(solid.solids())
        if (not solid.is_valid or len(raw_solids) != 1
                or raw_solids[0].volume <= 0.01):
            raise RuntimeError(
                f"coupon {name}: expected one valid source solid; "
                f"valid={solid.is_valid} volumes="
                f"{[item.volume for item in raw_solids]}")
        laid, print_transform = _front_face_down(solid)
        laid_solids = list(laid.solids())
        size = laid.bounding_box().size
        if (not laid.is_valid or len(laid_solids) != 1
                or laid_solids[0].volume <= 0.01):
            raise RuntimeError(
                f"coupon {name}: front-down transform damaged topology")
        if max(size.X, size.Y, size.Z) > BED_MM:
            raise RuntimeError(
                f"coupon {name}: {size.X:.2f} x {size.Y:.2f} x "
                f"{size.Z:.2f} exceeds {BED_MM:.0f} mm bed envelope")
        out = stl_dir / f"lx521_coupon_{name}.stl"
        temporary = out.with_name(
            f".{out.stem}.{os.getpid()}.tmp.stl")
        canonicalized_zeros = 0
        if name == "12_obiwan_closed_bore_bump":
            mesh_tolerance = OBIWAN_CROP_STL_TOLERANCE_MM
            angular_tolerance = OBIWAN_CROP_STL_ANGULAR_TOLERANCE_RAD
        else:
            mesh_tolerance = DEFAULT_STL_TOLERANCE_MM
            angular_tolerance = DEFAULT_STL_ANGULAR_TOLERANCE_RAD
        try:
            export_stl(
                laid, str(temporary), tolerance=mesh_tolerance,
                angular_tolerance=angular_tolerance)
            _validate_binary_stl(temporary)
            canonicalized_zeros = _canonicalize_transform_zeros(temporary)
            mesh_facts = _strict_mesh_facts(temporary)
            temporary.replace(out)
        finally:
            temporary.unlink(missing_ok=True)
        write_print_sidecar(
            out,
            part=out.stem,
            transform=print_transform,
            extra={"variant_export": "coupon"},
        )
        print(
            f"wrote {out.name}: {size.X:.2f} x {size.Y:.2f} x "
            f"{size.Z:.2f} mm; "
            f"mesh {mesh_facts['triangles']} tris/strict; "
            f"transform-zero fixes {canonicalized_zeros}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path,
                    help="write one state directory (default: both state trees)")
    ap.add_argument("--group", choices=COUPON_GROUPS,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.outdir is None:
        # Keep the convenience dual export, but isolate OCC and module
        # state in one fresh child per stand mode.
        for mode, state in (("1", "floor_stand"),
                            ("0", "no_floor_stand")):
            env = os.environ.copy()
            env["LX_STAND_FOOT"] = mode
            env["LX_ROUTING_PROFILE"] = "proud"
            guard = Path(__file__).with_name("run_memory_guarded.py")
            subprocess.run(
                [sys.executable, str(guard), "--", sys.executable,
                 str(Path(__file__).resolve()), "--outdir",
                 str(PROJECT_ROOT / "build" / state / "stl")],
                env=env, check=True)
        return

    target_mode = os.environ.get("LX_STAND_FOOT", "1")
    if target_mode not in {"0", "1"}:
        raise SystemExit("LX_STAND_FOOT must be 0 or 1")
    stl_dir = args.outdir
    stl_dir.mkdir(parents=True, exist_ok=True)
    if args.group is None:
        # Build the complete coupon family off to the side. Only after every
        # fresh guarded OCC child succeeds do same-filesystem atomic replaces
        # publish the new set. Generation failures preserve the prior set; an
        # interruption during the short per-file publication loop remains
        # fail-closed because Make removed the stamp and the hash manifest can
        # no longer certify a mixed generation.
        with tempfile.TemporaryDirectory(
                prefix=".coupon-stage-", dir=stl_dir) as stage_name:
            stage_dir = Path(stage_name)
            if _large_host_execution():
                # Groups have disjoint outputs and run under the enclosing
                # recipe's one 28-GiB process-tree guard.  Osado has ample
                # CPU/RAM for all nine small crops; local macOS deliberately
                # keeps the established one-at-a-time memory behavior.
                with ThreadPoolExecutor(
                        max_workers=len(COUPON_GROUPS),
                        thread_name_prefix="coupon") as executor:
                    tuple(executor.map(
                        lambda group: _run_group_guarded(
                            group, stage_dir, target_mode),
                        COUPON_GROUPS,
                    ))
            else:
                for group in COUPON_GROUPS:
                    _run_group_guarded(group, stage_dir, target_mode)
            actual = {path.name for path in stage_dir.glob("*.stl")}
            if actual != _EXPECTED_COUPON_FILES:
                raise RuntimeError(
                    "staged coupon set mismatch: "
                    f"missing={sorted(_EXPECTED_COUPON_FILES - actual)} "
                    f"extra={sorted(actual - _EXPECTED_COUPON_FILES)}")
            actual_sidecars = {
                path.name for path in stage_dir.glob("*.print.json")}
            if actual_sidecars != _EXPECTED_COUPON_SIDECARS:
                raise RuntimeError(
                    "staged coupon print-sidecar set mismatch: "
                    f"missing={sorted(_EXPECTED_COUPON_SIDECARS - actual_sidecars)} "
                    f"extra={sorted(actual_sidecars - _EXPECTED_COUPON_SIDECARS)}")
            for name in sorted(
                    _EXPECTED_COUPON_FILES | _EXPECTED_COUPON_SIDECARS):
                (stage_dir / name).replace(stl_dir / name)
        _prune_legacy_coupon_outputs(stl_dir)
        return
    import run_memory_guarded as memory_guard
    if not memory_guard.is_guarded_process():
        # Even the hidden one-group entry point is safe when invoked by hand;
        # no spoofable environment sentinel can bypass the outer guard.
        _run_group_guarded(args.group, stl_dir, target_mode)
        return
    _export_group(args.group, stl_dir, target_mode)


if __name__ == "__main__":
    main()
