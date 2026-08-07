"""Focused source/BREP contract for the candidate BMR crescent.

Pure gates (constants, vase-authority equality, candidate flags) always run.
Geometry gates read the exported BREP under
``build/bmr_crescent_TEBM35C10-4/`` and are skipped with an explicit message
when it is absent; they refuse to pass against a stale export.  The staged
interference gate additionally needs the hash-verified Obi-Wan stage BREPs.

Run with::

    LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan python tests/test_bmr_crescent.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from build123d import Box, Cylinder, Part, Pos, import_brep, import_step

from lx521_baffle.io import sha256_file
from lx521_baffle.obiwan import bmr_crescent as bmr
from lx521_baffle.obiwan.carriers import (
    CORE_REAR_Z,
    TWEETER_ADDON_JOINT_Z,
    TWEETER_CORE_BORE_TOP_Z,
    TWEETER_CORE_JOINT_Z,
    TWEETER_JOINT_CLEAR,
    TWEETER_JOINT_FUNCTIONAL_BOSS_D,
    TWEETER_JOINT_HOLE_D,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_INSERT_BORE_Z,
    TWEETER_JOINT_INSERT_DEPTH_MM,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
)
from lx521_baffle.base import THICKNESS_MM

BUILD_ROOT = PROJECT_ROOT / "build" / "bmr_crescent_TEBM35C10-4"
EXPORTED_BREP = BUILD_ROOT / f"{bmr.PART_NAME}.brep"
EXPORTED_FACTS = BUILD_ROOT / f"{bmr.PART_NAME}.facts.json"
STAGE_STATES = ("floor_stand", "no_floor_stand")
BED_LIMIT_MM = 256.0

SKIPPED: list[str] = []


def _skip(reason: str) -> None:
    SKIPPED.append(reason)
    print(f"  SKIP {reason}")


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


# --------------------------------------------------------------------------
# Pure gates
# --------------------------------------------------------------------------

def test_vase_authority_is_mirrored_exactly() -> None:
    """Every mirrored driver constant must equal the real vase's value.

    ``proud.vase_tebm35c10_4`` cannot be imported beside this obiwan-profile
    part, so the vase is evaluated in a proud-profile subprocess and compared
    value by value.  A drift in the vase fails here instead of silently
    leaving two disagreeing driver definitions in the tree.
    """
    program = (
        "import json, sys;"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r});"
        "from lx521_baffle.proud import vase_tebm35c10_4 as v;"
        "names = json.loads(sys.argv[1]);"
        "out = {n: getattr(v, n) for n in names};"
        "out['_OUTLET_INSET'] = ("
        "v.LOWER_T_OUTLET_Z_MM - v.LOWER_T_POCKET_REAR_Z_MM);"
        "print(json.dumps(out))"
    )
    environment = dict(os.environ)
    environment["LX_ROUTING_PROFILE"] = "proud"
    environment["LX_STAND_FOOT"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", program,
         json.dumps(sorted(bmr.VASE_AUTHORITY))],
        capture_output=True, text=True, env=environment, cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "could not evaluate the vase authority under LX_ROUTING_PROFILE="
            f"proud:\n{completed.stderr.strip()}")
    vase = json.loads(completed.stdout)
    for name, mirrored in sorted(bmr.VASE_AUTHORITY.items()):
        assert _close(vase[name], mirrored), (
            f"{name} drifted from the vase: vase={vase[name]} "
            f"bmr_crescent={mirrored}")
    assert _close(vase["_OUTLET_INSET"], bmr.POCKET_OUTLET_INSET_MM), (
        "pocket outlet inset drifted from the vase's own outlet placement: "
        f"vase={vase['_OUTLET_INSET']} bmr_crescent={bmr.POCKET_OUTLET_INSET_MM}")


def test_mate_constants_match_the_released_crescent() -> None:
    """Numeric identity with the released ND25FW-4 crescent's interface."""
    facts = bmr.design_facts()["mate"]
    assert tuple(facts["joint_x_mm"]) == tuple(TWEETER_JOINT_X) == (-24.0, 24.0)
    assert facts["joint_y_mm"] == TWEETER_JOINT_Y == 421.5
    assert facts["ear_boss_d_mm"] == TWEETER_JOINT_FUNCTIONAL_BOSS_D == 9.8
    assert tuple(facts["ear_z_span_mm"]) == tuple(TWEETER_ADDON_JOINT_Z)
    assert tuple(facts["core_ear_z_span_mm"]) == tuple(TWEETER_CORE_JOINT_Z)
    assert _close(facts["axial_gap_mm"], 0.20)
    assert facts["insert_receiver_d_mm"] == TWEETER_JOINT_INSERT_BORE_D == 4.6
    assert facts["insert_receiver_depth_mm"] == TWEETER_JOINT_INSERT_DEPTH_MM
    assert _close(facts["insert_receiver_depth_mm"], 4.0)
    assert tuple(facts["insert_receiver_z_span_mm"]) == tuple(
        TWEETER_JOINT_INSERT_BORE_Z)
    assert _close(facts["acoustic_front_floor_mm"], 1.9)
    assert facts["clearance_bore_d_mm"] == TWEETER_JOINT_HOLE_D == 3.4
    assert facts["clearance_bore_owner"] == "um_carrier"
    # The rear-driven passage is the UM's; the crescent must never own it.
    assert TWEETER_CORE_BORE_TOP_Z > TWEETER_ADDON_JOINT_Z[0]


def test_depth_stack_is_two_full_driver_envelopes() -> None:
    """Back-to-back stack, partition, and pocket datums."""
    assert _close(bmr.FRONT_MOUNT_Z_MM, THICKNESS_MM)
    assert _close(bmr.FRONT_MOUNT_Z_MM, 18.3)
    assert _close(bmr.REAR_MOUNT_Z_MM, -31.9)
    assert _close(bmr.STACK_DEPTH_MM, 2.0 * bmr.TEBM_DEPTH_MM)
    assert _close(bmr.STACK_DEPTH_MM, 50.2)
    # Each driver keeps a full 1.20 mm blind wall of its own.
    assert _close(bmr.PARTITION_THICKNESS_MM,
                  2.0 * bmr.T_BLIND_BACK_WALL_THICKNESS_MM)
    assert _close(bmr.PARTITION_THICKNESS_MM, 2.4)
    assert _close(bmr.FRONT_POCKET_FLOOR_Z_MM, -5.6)
    assert _close(bmr.REAR_POCKET_ROOF_Z_MM, -8.0)
    assert _close(
        bmr.FRONT_MOUNT_Z_MM - bmr.FRONT_POCKET_FLOOR_Z_MM,
        bmr.T_CLEAR_POCKET_DEPTH_MM)
    assert _close(
        bmr.REAR_POCKET_ROOF_Z_MM - bmr.REAR_MOUNT_Z_MM,
        bmr.T_CLEAR_POCKET_DEPTH_MM)
    assert _close(bmr.REAR_PROTRUSION_MM, CORE_REAR_Z - bmr.REAR_MOUNT_Z_MM)
    # Both lead outlets clear their blind wall by a real ligament.
    for outlet_z, wall_z in (
        (bmr.FRONT_OUTLET_Z_MM, bmr.FRONT_POCKET_FLOOR_Z_MM),
        (bmr.REAR_OUTLET_Z_MM, bmr.REAR_POCKET_ROOF_Z_MM),
    ):
        ligament = abs(outlet_z - wall_z) - bmr.POCKET_OUTLET_D_MM / 2.0
        assert ligament >= bmr.T_BLIND_BACK_WALL_THICKNESS_MM, (
            f"lead outlet at z={outlet_z} leaves only {ligament:.3f} mm to "
            "its blind wall")


def test_boss_stays_clear_of_the_released_mate() -> None:
    """Nothing added may reach the UM ear receiver footprint."""
    ear_radius = math.hypot(
        TWEETER_JOINT_X[1], bmr.BMR_AXIS_XY[1] - TWEETER_JOINT_Y)
    footprint = ear_radius - (
        TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR)
    assert _close(footprint, bmr.UM_EAR_FOOTPRINT_R_MM, 1.0e-6)
    assert bmr.ROOT_FAIRING_R_OUT_MM <= footprint - 0.5 + 1.0e-9, (
        "the root fairing must stop short of the UM ear footprint")
    assert bmr.BOSS_PLAN_R_MM < bmr.ROOT_FAIRING_R_OUT_MM
    # The boss plan is the released scallop plus a fusion overlap only.
    assert _close(bmr.BOSS_PLAN_R_MM,
                  bmr.SCALLOP_R_MM + bmr.BOSS_SCALLOP_OVERLAP_MM)
    assert _close(bmr.SCALLOP_R_MM, 39.25)
    assert _close(bmr.BOSS_REAR_R_MM, bmr.TEBM_LAND_R_MM)
    # The flare is monotonic and lands tangentially on the core rear plane.
    previous = bmr.boss_radius_at(bmr.REAR_MOUNT_Z_MM)
    assert _close(previous, bmr.BOSS_REAR_R_MM)
    for step in range(1, 201):
        z = bmr.REAR_MOUNT_Z_MM + (
            CORE_REAR_Z - bmr.REAR_MOUNT_Z_MM) * step / 200.0
        radius = bmr.boss_radius_at(z)
        assert radius >= previous - 1.0e-9, "boss flare is not monotonic"
        previous = radius
    assert _close(previous, bmr.BOSS_PLAN_R_MM, 1.0e-6)


def test_candidate_flags_are_set() -> None:
    facts = bmr.design_facts()
    assert bmr.RELEASE_AUTHORIZED is False
    assert bmr.PHYSICAL_MEASURE_REQUIRED is True
    assert facts["release_authorized"] is False
    assert facts["physical_measure_required"] is True
    assert facts["status"] == "candidate_not_release_authorized"
    assert facts["counts_against_release_inventory"] is False
    assert bmr.MAGNET_COUNT == 0
    assert facts["magnet_count"] == 0


def test_part_is_not_wired_into_the_release() -> None:
    """The candidate must stay out of the stage, the counts and to_print."""
    staged = (PROJECT_ROOT / "scripts" / "export_obiwan_staged.py").read_text(
        encoding="utf-8")
    assert "bmr_crescent" not in staged, (
        "the BMR crescent must not join the Obi-Wan stage manifest")
    for state in STAGE_STATES:
        manifest = (PROJECT_ROOT / "build" / state / ".obiwan_stage"
                    / "manifest.json")
        if not manifest.is_file():
            continue
        assert "bmr_crescent" not in manifest.read_text(encoding="utf-8")
    to_print = PROJECT_ROOT / "to_print"
    if to_print.is_dir():
        hits = [str(path) for path in to_print.rglob("*bmr_crescent*")]
        assert not hits, f"candidate leaked into to_print: {hits}"


# --------------------------------------------------------------------------
# Geometry gates (exported BREP)
# --------------------------------------------------------------------------

def _exported_solid():
    if not EXPORTED_BREP.is_file() or not EXPORTED_FACTS.is_file():
        _skip(
            f"{EXPORTED_BREP.relative_to(PROJECT_ROOT)} absent; run "
            "LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan python "
            "scripts/export_bmr_crescent.py")
        return None, None
    facts = json.loads(EXPORTED_FACTS.read_text(encoding="utf-8"))
    # An export that predates the current source is not evidence about it.
    recorded = facts["source_file_sha256"]
    for relative, digest in recorded.items():
        path = (EXPORTED_FACTS.parent / relative).resolve()
        assert path.is_file(), f"recorded source vanished: {relative}"
        assert sha256_file(path) == digest, (
            f"exported artifacts are stale against {relative}; re-run "
            "scripts/export_bmr_crescent.py")
    assert sha256_file(EXPORTED_BREP) == facts["files"]["brep"]["sha256"]
    return Part(import_brep(str(EXPORTED_BREP)).solids()), facts


def _material(solid, x: float, y: float, z: float, size: float = 0.6) -> float:
    probe = Pos(x, y, z) * Box(size, size, size)
    intersection = solid & probe
    return 0.0 if intersection is None else float(intersection.volume)


def test_exported_solid_is_one_valid_body_that_fits_the_bed() -> None:
    solid, facts = _exported_solid()
    if solid is None:
        return
    assert solid.is_valid, "exported BREP is not a valid solid"
    assert len(solid.solids()) == 1, "the crescent must be one body"
    # One shell means no sealed internal void: every cavity reaches outside
    # through a declared opening.
    assert len(solid.shells()) == 1, (
        f"exported solid has {len(solid.shells())} shells; a sealed internal "
        "void would be an undeclared cavity")
    size = solid.bounding_box().size
    for axis, value in (("X", size.X), ("Y", size.Y), ("Z", size.Z)):
        assert value <= BED_LIMIT_MM, (
            f"{axis} extent {value:.3f} mm exceeds the {BED_LIMIT_MM} mm bed")
    printed = facts["print_geometry"]["bounds_size_mm"]
    assert max(printed) <= BED_LIMIT_MM
    assert facts["print_geometry"]["p2s_256mm_fit"] is True
    assert facts["print_geometry"]["support_enabled"] is False
    print(f"    envelope {size.X:.3f} x {size.Y:.3f} x {size.Z:.3f} mm, "
          f"{solid.volume / 1000.0:.2f} cm3")


def test_declared_openings_are_the_only_openings() -> None:
    """Each declared feature exists; nothing undeclared breaks the skin."""
    solid, _facts = _exported_solid()
    if solid is None:
        return
    axis_x, axis_y = bmr.BMR_AXIS_XY

    # Both pockets are clear over their full declared depth.
    for name, z_low, z_high in (
        ("front", bmr.FRONT_POCKET_FLOOR_Z_MM + 0.4, THICKNESS_MM - 0.4),
        ("rear", bmr.REAR_MOUNT_Z_MM + 0.4, bmr.REAR_POCKET_ROOF_Z_MM - 0.4),
    ):
        for fraction in (0.05, 0.5, 0.95):
            z = z_low + (z_high - z_low) * fraction
            assert _material(solid, axis_x, axis_y, z) == 0.0, (
                f"{name} driver pocket is obstructed at z={z:.3f}")

    # The partition is solid between them: two chambers, never one.
    partition_mid = (bmr.REAR_POCKET_ROOF_Z_MM
                     + bmr.FRONT_POCKET_FLOOR_Z_MM) / 2.0
    assert _material(solid, axis_x, axis_y, partition_mid, size=0.4) > 0.0, (
        "the back-to-back partition is missing; the two rear volumes would "
        "be one chamber")

    # Both lead outlets really break out of the boss on the -Y meridian.
    for name, outlet_z in (
        ("front", bmr.FRONT_OUTLET_Z_MM),
        ("rear", bmr.REAR_OUTLET_Z_MM),
    ):
        breakout = axis_y - bmr.boss_radius_at(outlet_z)
        inside = axis_y - (bmr.boss_radius_at(outlet_z) - 1.5)
        assert _material(solid, axis_x, inside, outlet_z, size=0.4) == 0.0, (
            f"{name} lead outlet does not reach the boss wall")
        assert _material(solid, axis_x, breakout - 1.5, outlet_z,
                         size=0.4) == 0.0, (
            f"{name} lead outlet does not break out of the boss")
        # A bore one radius off the meridian would be a second, undeclared
        # opening; check the wall beside it is intact.
        assert _material(
            solid, axis_x + bmr.POCKET_OUTLET_D_MM,
            axis_y - (bmr.boss_radius_at(outlet_z) - 1.5),
            outlet_z, size=0.4) > 0.0, (
            f"{name} outlet is wider than declared")

    # Eight blind M2 bores, four per land, on the declared PCD and clocks.
    radius = bmr.TEBM_MOUNT_PCD_MM / 2.0
    for clock, mouth_z, blind_z in (
        (bmr.FRONT_MOUNT_CLOCK_DEG, THICKNESS_MM - 0.5,
         THICKNESS_MM - bmr.M2_INSERT_DEPTH_MM - 0.8),
        (bmr.REAR_MOUNT_CLOCK_DEG, bmr.REAR_MOUNT_Z_MM + 0.5,
         bmr.REAR_MOUNT_Z_MM + bmr.M2_INSERT_DEPTH_MM + 0.8),
    ):
        for index in range(bmr.TEBM_MOUNT_HOLE_COUNT):
            angle = math.radians(clock + 90.0 * index)
            x = axis_x + radius * math.cos(angle)
            y = axis_y + radius * math.sin(angle)
            assert _material(solid, x, y, mouth_z, size=0.4) == 0.0, (
                f"M2 bore at clock {clock} index {index} is missing")
            assert _material(solid, x, y, blind_z, size=0.4) > 0.0, (
                f"M2 bore at clock {clock} index {index} is not blind")

    # The crescent's own insert receivers stay blind behind 1.9 mm of front.
    for x in TWEETER_JOINT_X:
        assert _material(
            solid, x, TWEETER_JOINT_Y, THICKNESS_MM - 0.6, size=0.4) > 0.0, (
            "the acoustic-front floor over an insert receiver is breached")
        assert _material(
            solid, x, TWEETER_JOINT_Y,
            TWEETER_JOINT_INSERT_BORE_Z[1] - 0.6, size=0.4) == 0.0, (
            "an insert receiver is missing")


def test_mate_is_identical_to_the_released_crescent() -> None:
    """Outside the declared growth envelope this part is the ND crescent."""
    solid, _facts = _exported_solid()
    if solid is None:
        return
    axis_x, axis_y = bmr.BMR_AXIS_XY
    checked = 0
    for state in STAGE_STATES:
        released_path = (PROJECT_ROOT / "build" / state / ".obiwan_stage"
                         / "addon_tweeter_crescent.brep")
        if not released_path.is_file():
            _skip(f"{state}: staged ND crescent BREP absent")
            continue
        released = Part(import_brep(str(released_path)).solids())

        # Everything this part adds lies inside r = ROOT_FAIRING_R_OUT_MM of
        # the BMR axis, so outside that cylinder the two crescents must be
        # the same solid.  The envelope has to be a real cylinder: a square
        # of the same half-width would swallow both half-lap ears and quietly
        # exempt the very interface this gate exists to check.
        outside = Pos(0.0, axis_y, -10.0) * Box(400.0, 400.0, 200.0)
        growth = Pos(axis_x, axis_y, -10.0) * Cylinder(
            bmr.ROOT_FAIRING_R_OUT_MM, 200.0)
        for shape in (solid, released):
            assert (shape & outside).volume > 0.0
        for x in TWEETER_JOINT_X:
            assert math.hypot(x - axis_x, TWEETER_JOINT_Y - axis_y) - (
                TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR
            ) > bmr.ROOT_FAIRING_R_OUT_MM, (
                "the half-lap ears must lie wholly outside the growth "
                "envelope, otherwise this gate proves nothing about them")
        mine = (solid & outside) - growth
        theirs = (released & outside) - growth
        difference = (mine - theirs).volume + (theirs - mine).volume
        assert difference < 1.0e-6, (
            f"{state}: geometry outside the declared growth envelope differs "
            f"from the released crescent by {difference:.6f} mm3")

        # And the part really is different inside it, otherwise the gate above
        # would pass on an accidental copy of the released crescent.
        assert abs(solid.volume - released.volume) > 1000.0
        checked += 1
    if checked:
        print(f"    mate identity verified against {checked} staged state(s)")


def test_no_interference_with_the_um_collar() -> None:
    """The rear-protruding boss must clear the UM collar in both states."""
    solid, _facts = _exported_solid()
    if solid is None:
        return
    checked = 0
    for state in STAGE_STATES:
        um_path = (PROJECT_ROOT / "build" / state / ".obiwan_stage"
                   / "core_um_carrier.brep")
        if not um_path.is_file():
            _skip(f"{state}: staged UM carrier BREP absent")
            continue
        um = Part(import_brep(str(um_path)).solids())
        overlap = (solid & um).volume
        assert overlap == 0.0, (
            f"{state}: BMR crescent intersects the UM collar by "
            f"{overlap:.6f} mm3")
        # Report the plan gap the rear protrusion actually keeps.
        gap = (bmr.BMR_AXIS_XY[1] - bmr.BOSS_PLAN_R_MM
               - um.bounding_box().max.Y)
        print(f"    {state}: no interference; boss plan clears the UM's "
              f"furthest feature by {gap:.3f} mm in Y")
        checked += 1
    if not checked:
        _skip("no staged UM carrier available for the interference gate")


def test_wing_clearance_is_unchanged() -> None:
    """Both wing families must clear this part exactly as they clear the ND.

    The growth is inward (the released scallop) and rearward, plus two small
    blends above the released top edge, so no wing envelope should move.  This
    checks that rather than asserting it.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    wings = sorted(
        (PROJECT_ROOT / "build" / "wings").glob(
            "*/obiwan_wing_*_assembled.step"))
    if not wings:
        _skip("no built wing STEP available for the clearance gate")
        return
    released_path = (PROJECT_ROOT / "build" / STAGE_STATES[0]
                     / ".obiwan_stage" / "addon_tweeter_crescent.brep")
    released = (
        Part(import_brep(str(released_path)).solids())
        if released_path.is_file() else None)
    for wing_path in wings:
        wing = Part(import_step(str(wing_path)).solids())
        overlap = (solid & wing).volume
        assert overlap == 0.0, (
            f"{wing_path.parent.name} wing intersects the BMR crescent by "
            f"{overlap:.6f} mm3")
        if released is not None:
            assert (released & wing).volume == 0.0, (
                f"{wing_path.parent.name} wing already intersects the "
                "released crescent; the comparison is meaningless")
        print(f"    {wing_path.parent.name} wing: clear")


def main() -> None:
    tests = (
        test_vase_authority_is_mirrored_exactly,
        test_mate_constants_match_the_released_crescent,
        test_depth_stack_is_two_full_driver_envelopes,
        test_boss_stays_clear_of_the_released_mate,
        test_candidate_flags_are_set,
        test_part_is_not_wired_into_the_release,
        test_exported_solid_is_one_valid_body_that_fits_the_bed,
        test_declared_openings_are_the_only_openings,
        test_mate_is_identical_to_the_released_crescent,
        test_no_interference_with_the_um_collar,
        test_wing_clearance_is_unchanged,
    )
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    suffix = f"; {len(SKIPPED)} skipped gate(s)" if SKIPPED else ""
    print(f"{bmr.PART_NAME}: {len(tests)} focused gates pass{suffix}")


if __name__ == "__main__":
    main()
