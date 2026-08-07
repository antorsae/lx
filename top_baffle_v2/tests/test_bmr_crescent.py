"""Focused source/BREP contract for the candidate BMR pod.

Pure gates (constants, vase-authority equality, candidate flags) always run.
Geometry gates read the exported BREP under
``build/bmr_crescent_TEBM35C10-4/`` and are skipped with an explicit message
when it is absent; they refuse to pass against a stale export.  The staged
gates additionally need the hash-verified Obi-Wan stage BREPs.

This part is no longer a superset of the released ND25FW-4 crescent -- it
keeps that crescent's *mount* and nothing else -- so the mate is proven by
asserting the two ear footprints are geometrically identical and by
assembling the part against the staged UM collar, not by differencing whole
silhouettes.

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

from build123d import (
    Box,
    Cylinder,
    Face,
    Part,
    Plane,
    Pos,
    Rot,
    extrude,
    import_brep,
    import_step,
)

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
    UM_CORE_R,
    _complete_tweeter_joint_ear_plan,
    _plan_prism,
)
from lx521_baffle.base import THICKNESS_MM, UM_CUTOUT

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
        f"vase={vase['_OUTLET_INSET']} "
        f"bmr_crescent={bmr.POCKET_OUTLET_INSET_MM}")


def test_mount_constants_equal_the_released_joint_authority() -> None:
    """Numeric identity with the released UM half-lap interface.

    This is the non-negotiable part of the design: the part must swap onto an
    unmodified UM collar, so every one of these comes from the released joint
    authority in ``obiwan.carriers`` and none may be restated locally.
    """
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
    # The rear-driven passage is the UM's; the pod must never own it.
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


def test_pod_outer_wall_is_the_driver_land() -> None:
    """The pod is the D66 land itself, and that is the printable minimum.

    Both mounting faces have to carry the qualified D66 land and the part
    prints front-face-down, so the plan may never grow rearward.  Anything
    under R33 loses land; anything over R33 at the front cannot come back
    down to R33 at the rear without an overhang.  A straight D66 cylinder is
    the only radius that satisfies both, which is what this checks.
    """
    assert _close(bmr.POD_OUTER_R_MM, bmr.TEBM_LAND_R_MM)
    assert _close(bmr.POD_OUTER_D_MM, 66.0)
    for z in (bmr.REAR_MOUNT_Z_MM, -10.0, 0.0, CORE_REAR_Z, THICKNESS_MM):
        assert _close(bmr.pod_radius_at(z), bmr.TEBM_LAND_R_MM), (
            "the pod must be a straight cylinder; a varying radius either "
            "loses land or leans the wrong way for this print orientation")
    # The wall the land leaves is far above any minimum-wall rule.
    assert _close(bmr.POD_WALL_OVER_POCKET_MM, 11.537)
    assert _close(bmr.POD_WALL_OVER_INSERT_MM, 7.27)
    assert bmr.POD_WALL_OVER_POCKET_MM >= 2.4
    assert bmr.POD_WALL_OVER_INSERT_MM >= 2.4
    assert _close(bmr.POD_LAND_MARGIN_OVER_FLANGE_MM, 6.0)
    # And the whole pod now sits inside the released open scallop instead of
    # filling it, so nothing in plan reaches where the release had material.
    assert _close(bmr.SCALLOP_R_MM, 39.25)
    assert bmr.POD_CLEARANCE_INSIDE_SCALLOP_MM > 0.0
    assert _close(bmr.POD_CLEARANCE_INSIDE_SCALLOP_MM, 6.25)
    # The pod also stops well short of the UM ear footprints.
    ear_radius = math.hypot(
        TWEETER_JOINT_X[1], bmr.BMR_AXIS_XY[1] - TWEETER_JOINT_Y)
    footprint = ear_radius - (
        TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR)
    assert bmr.POD_OUTER_R_MM < footprint, (
        "the pod wall must stop short of the UM ear receiver footprint")
    print(f"    pod D{bmr.POD_OUTER_D_MM:.0f} = the driver land; "
          f"{bmr.POD_WALL_OVER_POCKET_MM:.3f} mm wall outside the pocket, "
          f"{bmr.POD_WALL_OVER_INSERT_MM:.3f} mm outside each M2 bore")


def test_struts_are_sized_from_the_half_lap_they_feed() -> None:
    """The connecting structure must not become a new weakest link."""
    # The strut's smallest section is where it crosses the UM ear's receiver
    # notch and is only the ear's own thickness deep.
    assert _close(bmr.EAR_THICKNESS_MM, 5.9)
    assert _close(bmr.EAR_NET_LIGAMENT_MM, 5.2)
    assert _close(bmr.EAR_NET_SECTION_MM2, 30.68)
    assert bmr.ARM_MIN_SECTION_MM2 > bmr.EAR_NET_SECTION_MM2, (
        f"strut section {bmr.ARM_MIN_SECTION_MM2:.3f} mm2 is below the "
        f"half-lap's own {bmr.EAR_NET_SECTION_MM2:.3f} mm2, so the strut and "
        "not the qualified joint would govern")
    # The rearward draft is bounded by the same ligament.
    assert bmr.ARM_REAR_WIDTH_MM > bmr.EAR_NET_LIGAMENT_MM, (
        f"a {bmr.ARM_DRAFT_DEG} degree draft leaves only "
        f"{bmr.ARM_REAR_WIDTH_MM:.3f} mm of strut at z={bmr.ARM_REAR_Z_MM}")
    assert bmr.ARM_DRAFT_DEG > 0.0
    # Root fillet equal to the strut width.
    assert _close(bmr.ARM_ROOT_FILLET_R_MM, bmr.ARM_WIDTH_MM)
    # The struts take the released crescent's own clearance around the UM ring.
    clearance = bmr.arm_collar_clearance_mm()
    assert clearance >= bmr.UM_COLLAR_CLEAR_MM, (
        f"the struts leave only {clearance:.3f} mm around the UM core ring")
    # One simple region, and the central cable mouth is still open: the plan
    # must not reach the -Y meridian between the pod wall and the ears.
    plan = bmr.arm_plan()
    assert plan.geom_type == "Polygon" and not plan.interiors
    from shapely.geometry import Point as _Point
    mouth_y = bmr.BMR_AXIS_XY[1] - bmr.POD_OUTER_R_MM
    for x in (-6.0, -3.0, 0.0, 3.0, 6.0):
        for y in (TWEETER_JOINT_Y + 4.0, (TWEETER_JOINT_Y + mouth_y) / 2.0,
                  mouth_y - 0.5):
            assert not plan.contains(_Point(x, y)), (
                f"the strut plan closes the central cable mouth at ({x}, {y})")
    print(f"    struts {bmr.ARM_WIDTH_MM:.1f} -> {bmr.ARM_REAR_WIDTH_MM:.3f} mm "
          f"wide, min section {bmr.ARM_MIN_SECTION_MM2:.3f} mm2 = "
          f"{bmr.ARM_MIN_SECTION_MM2 / bmr.EAR_NET_SECTION_MM2:.2f}x the "
          f"half-lap; {clearance:.3f} mm off the UM ring")


def test_inherited_tweeter_clamp_holes_are_gone() -> None:
    """The four M4 ND25FW-4 faceplate passages must not be declared at all.

    They were inherited only to keep the released silhouette exact.  This
    variant clamps no tweeter and no longer has that silhouette, so they
    carry no fastener and no meaning.
    """
    names = [opening["name"] for opening in bmr.declared_openings()]
    assert "inherited_m4_tweeter_clamp_holes" not in names
    assert not [
        opening for opening in bmr.declared_openings()
        if opening["kind"] == "released_inherited"
    ]
    assert names == [
        "front_driver_pocket_mouth",
        "rear_driver_pocket_mouth",
        "front_driver_lead_outlet",
        "rear_driver_lead_outlet",
        "um_half_lap_clearance_passages",
        "um_half_lap_insert_receivers",
        "m2_driver_insert_bores",
    ]
    silhouette = bmr.design_facts()["silhouette"]
    assert silhouette["inherits_released_crescent_outline"] is False
    assert any("M4" in entry
               for entry in silhouette["removed_from_the_first_candidate"])


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
        "the BMR pod must not join the Obi-Wan stage manifest")
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


def _staged(state: str, name: str):
    path = PROJECT_ROOT / "build" / state / ".obiwan_stage" / f"{name}.brep"
    if not path.is_file():
        return None
    return Part(import_brep(str(path)).solids())


def _material(solid, x: float, y: float, z: float,
              size: float = 0.6, height: float | None = None) -> float:
    probe = Pos(x, y, z) * Box(size, size, size if height is None else height)
    intersection = solid & probe
    return 0.0 if intersection is None else float(intersection.volume)


def _ear_prism(x: float, z0: float, z1: float, owner: str = "tweeter"):
    """The joint authority's own ear footprint, extruded over one Z span."""
    return _plan_prism(_complete_tweeter_joint_ear_plan(owner, x), z0, z1)


def test_exported_solid_is_one_valid_body_that_fits_the_bed() -> None:
    solid, facts = _exported_solid()
    if solid is None:
        return
    assert solid.is_valid, "exported BREP is not a valid solid"
    assert len(solid.solids()) == 1, "the pod must be one body"
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
    # X is now set by the pod alone: nothing reaches past the D66 land.
    assert _close(size.X, bmr.POD_OUTER_D_MM, 0.01), (
        f"the X extent is {size.X:.3f} mm, not the D66 pod; something still "
        "reaches outboard of the land")
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

    # Both lead outlets really break out of the pod on the -Y meridian.
    for name, outlet_z in (
        ("front", bmr.FRONT_OUTLET_Z_MM),
        ("rear", bmr.REAR_OUTLET_Z_MM),
    ):
        radius = bmr.pod_radius_at(outlet_z)
        assert _material(solid, axis_x, axis_y - (radius - 1.5), outlet_z,
                         size=0.4) == 0.0, (
            f"{name} lead outlet does not reach the pod wall")
        assert _material(solid, axis_x, axis_y - (radius + 1.5), outlet_z,
                         size=0.4) == 0.0, (
            f"{name} lead outlet does not break out of the pod")
        # A bore one diameter off the meridian would be a second, undeclared
        # opening; check the wall beside it is intact.
        assert _material(
            solid, axis_x + bmr.POCKET_OUTLET_D_MM,
            axis_y - (radius - 1.5), outlet_z, size=0.4) > 0.0, (
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

    # The pod's own insert receivers stay blind behind 1.9 mm of front.
    for x in TWEETER_JOINT_X:
        assert _material(
            solid, x, TWEETER_JOINT_Y, THICKNESS_MM - 0.6, size=0.4) > 0.0, (
            "the acoustic-front floor over an insert receiver is breached")
        assert _material(
            solid, x, TWEETER_JOINT_Y,
            TWEETER_JOINT_INSERT_BORE_Z[1] - 0.6, size=0.4) == 0.0, (
            "an insert receiver is missing")


def test_mount_interface_is_geometrically_identical_to_the_released_ear() -> None:
    """Ear for ear, this part's mount is the released crescent's mount.

    The old symmetric-difference-over-the-whole-silhouette gate cannot apply
    any more: this part is not a superset of the released crescent.  What has
    to be identical is the mount, so the comparison is confined to the joint
    authority's own ear footprints over the add-on's own Z span, where both
    parts must be exactly the complete D9.8 ear less the D3.4 passage and the
    blind D4.6 receiver.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    checked = 0
    for state in STAGE_STATES:
        released = _staged(state, "addon_tweeter_crescent")
        if released is None:
            _skip(f"{state}: staged ND crescent BREP absent")
            continue
        for x in TWEETER_JOINT_X:
            prism = _ear_prism(x, *TWEETER_ADDON_JOINT_Z)
            mine = solid & prism
            theirs = released & prism
            assert mine is not None and float(mine.volume) > 0.0
            difference = (mine - theirs).volume + (theirs - mine).volume
            assert difference < 1.0e-6, (
                f"{state}: the ear at x={x:+.1f} differs from the released "
                f"crescent's by {difference:.6f} mm3")
        # And the part really is a different part, so nobody can read the gate
        # above as the old whole-silhouette identity claim.  Most of the
        # released crescent is simply not here any more.
        dropped = (released - solid).volume
        fraction = dropped / released.volume
        assert fraction > 0.5, (
            "this variant is supposed to have dropped the released crescent's "
            f"arm silhouette; only {100.0 * fraction:.1f}% of it is absent, "
            "so the mount-only comparison above may be hiding an inherited "
            "outline")
        print(f"    {state}: {100.0 * fraction:.1f}% of the released "
              f"crescent's material ({dropped:.0f} mm3) is absent here")
        checked += 1
    if checked:
        print(f"    mount identity verified ear for ear against {checked} "
              "staged state(s)")


def test_mate_simulation_against_the_staged_um_collar() -> None:
    """Assemble on the real UM BREP: no interference, gap and screws intact."""
    solid, _facts = _exported_solid()
    if solid is None:
        return
    gap_low, gap_high = TWEETER_CORE_JOINT_Z[1], TWEETER_ADDON_JOINT_Z[0]
    gap_mid = (gap_low + gap_high) / 2.0
    checked = 0
    for state in STAGE_STATES:
        um = _staged(state, "core_um_carrier")
        if um is None:
            _skip(f"{state}: staged UM carrier BREP absent")
            continue
        # 1. Assembled fit: the two parts occupy disjoint space.
        overlap = (solid & um).volume
        assert overlap == 0.0, (
            f"{state}: assembled BMR pod intersects the UM collar by "
            f"{overlap:.6f} mm3")

        for x in TWEETER_JOINT_X:
            # 2. The 0.20 mm axial gap is real and empty from both sides, and
            # both half-laps actually bear on it over the boss annulus.
            bearing_x = x + 3.3
            assert _material(um, bearing_x, TWEETER_JOINT_Y,
                             gap_low - 0.3, size=0.4, height=0.2) > 0.0, (
                f"{state}: the UM half-lap has no bearing face at x={x:+.1f}")
            assert _material(solid, bearing_x, TWEETER_JOINT_Y,
                             gap_high + 0.3, size=0.4, height=0.2) > 0.0, (
                f"{state}: the pod half-lap has no bearing face at x={x:+.1f}")
            for name, shape in (("UM", um), ("pod", solid)):
                assert _material(shape, bearing_x, TWEETER_JOINT_Y, gap_mid,
                                 size=0.4, height=0.1) == 0.0, (
                    f"{state}: the {name} closes the 0.20 mm axial gap")

            # 3. Nothing of this part reaches below its own ear plane inside
            # the opposing UM ear's footprint, clearance included.
            notch = _plan_prism(
                _complete_tweeter_joint_ear_plan(
                    "um", x, TWEETER_JOINT_CLEAR),
                CORE_REAR_Z - 1.0, gap_high - 1.0e-6)
            intrusion = (solid & notch)
            assert float(0.0 if intrusion is None
                         else intrusion.volume) == 0.0, (
                f"{state}: the pod intrudes into the UM ear receiver notch "
                f"at x={x:+.1f}")

            # 4. One continuous rear-driven M3 path: up the UM's D3.4 bore,
            # across the gap, into the blind D4.6 receiver, stopping under the
            # 1.9 mm acoustic front floor.
            assert _material(um, x, TWEETER_JOINT_Y, CORE_REAR_Z + 1.0,
                             size=0.4) == 0.0, (
                f"{state}: the UM clearance bore is blocked at x={x:+.1f}")
            assert _material(solid, x, TWEETER_JOINT_Y, gap_mid,
                             size=0.4, height=0.1) == 0.0
            assert _material(solid, x, TWEETER_JOINT_Y,
                             TWEETER_JOINT_INSERT_BORE_Z[1] - 0.6,
                             size=0.4) == 0.0, (
                f"{state}: the blind receiver is missing at x={x:+.1f}")
            assert _material(solid, x, TWEETER_JOINT_Y, THICKNESS_MM - 0.6,
                             size=0.4) > 0.0, (
                f"{state}: the 1.9 mm front floor is breached at x={x:+.1f}")
        print(f"    {state}: assembled with zero interference, both 0.20 mm "
              "gaps empty, both M3 paths continuous")
        checked += 1
    if not checked:
        _skip("no staged UM carrier available for the mate simulation")


def test_no_interference_with_the_um_collar() -> None:
    """The rear-protruding pod must clear the UM collar in both states."""
    solid, _facts = _exported_solid()
    if solid is None:
        return
    checked = 0
    for state in STAGE_STATES:
        um = _staged(state, "core_um_carrier")
        if um is None:
            _skip(f"{state}: staged UM carrier BREP absent")
            continue
        overlap = (solid & um).volume
        assert overlap == 0.0, (
            f"{state}: BMR pod intersects the UM collar by "
            f"{overlap:.6f} mm3")
        gap = (bmr.BMR_AXIS_XY[1] - bmr.POD_OUTER_R_MM
               - um.bounding_box().max.Y)
        print(f"    {state}: no interference; the pod wall clears the UM's "
              f"furthest feature by {gap:.3f} mm in Y")
        checked += 1
    if not checked:
        _skip("no staged UM carrier available for the interference gate")


def test_free_t_cable_corridor_stays_open() -> None:
    """No printed structure may trap the free T cable behind the part.

    The struts stop at the core rear plane and no duct is cut outside the pod
    wall, so the corridor the modelled T cable runs through -- including its
    free suffix at TS_FREE_CABLE_Z -- must be untouched.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    checked = 0
    for state in STAGE_STATES:
        cable = _staged(state, "review_reference_ts_cable")
        if cable is None:
            _skip(f"{state}: staged T cable reference BREP absent")
            continue
        overlap = (solid & cable)
        volume = 0.0 if overlap is None else float(overlap.volume)
        assert volume == 0.0, (
            f"{state}: the BMR pod pinches the modelled T cable by "
            f"{volume:.6f} mm3")
        checked += 1
    if checked:
        print(f"    free T cable at z={bmr.TS_FREE_CABLE_Z:.2f} clear in "
              f"{checked} staged state(s); no printed duct")


def _filled_silhouette(solid, z: float):
    """Unit prism of the part's exterior plan at one Z, holes filled.

    Only the outer wire is kept, so pockets, bores and receivers -- which are
    interior and open to a declared face -- never register as plan growth.
    """
    slab = solid & (Pos(0.0, bmr.BMR_AXIS_XY[1], z + 0.05)
                    * Box(400.0, 400.0, 0.1))
    if slab is None or not slab.solids():
        return None
    prism = None
    for face in slab.faces().filter_by(Plane.XY):
        if abs(face.center().Z - z) > 1.0e-6:
            continue
        # Section faces come back with whichever normal OCC chose, so extrude
        # both ways: a one-sided extrusion silently sends some sections to
        # z=-1..0 and others to z=0..1, and disjoint prisms would read as a
        # total leak instead of a containment failure.
        column = extrude(
            Pos(0.0, 0.0, -z) * Face(face.outer_wire()),
            amount=0.5, both=True)
        prism = column if prism is None else prism.fuse(column)
    return None if prism is None else prism.clean()


def test_exterior_never_grows_rearward() -> None:
    """Front-face-down, every layer must sit on the one before it.

    The part is printed with z=18.3 on the bed, so print height runs with
    decreasing Z.  Requiring the exterior plan at each Z to lie inside the
    plan just in front of it is exactly the no-overhang condition for the
    whole outside of the part -- the drafted struts, their roots, the ear
    step and the pod.  The two declared lead outlets are filled back in
    first: a 4.6 mm bore through a wall is a bridge, not plan growth.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    pod = Pos(bmr.BMR_AXIS_XY[0], bmr.BMR_AXIS_XY[1],
              (bmr.REAR_MOUNT_Z_MM + THICKNESS_MM) / 2.0) * Cylinder(
        bmr.POD_OUTER_R_MM, bmr.STACK_DEPTH_MM)
    probe = solid
    for outlet_z in (bmr.FRONT_OUTLET_Z_MM, bmr.REAR_OUTLET_Z_MM):
        length = bmr.POCKET_OUTLET_OUTER_R_MM - bmr.POCKET_OUTLET_INNER_R_MM
        centre_y = bmr.BMR_AXIS_XY[1] - (
            bmr.POCKET_OUTLET_INNER_R_MM + bmr.POCKET_OUTLET_OUTER_R_MM) / 2.0
        filler = (Pos(bmr.BMR_AXIS_XY[0], centre_y, outlet_z) * Rot(X=90.0)
                  * Cylinder(bmr.POCKET_OUTLET_D_MM / 2.0, length)) & pod
        probe = probe.fuse(filler)
    probe = probe.clean()

    ladder = (
        18.29, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.45, 12.35, 12.0,
        11.0, 10.0, 9.0, 8.0, 7.0, 6.85, 6.75, 6.5, 5.0, 0.0,
        -10.0, -20.0, -31.85,
    )
    front = None
    for z in ladder:
        here = _filled_silhouette(probe, z)
        assert here is not None, f"no cross section at z={z}"
        if front is not None:
            leak = (here - front).volume
            assert leak < 1.0e-6, (
                f"the exterior plan at z={z} reaches {leak:.6f} mm3 outside "
                "the plan in front of it; that is an overhang in the "
                "front-face-down print")
        front = here
    print(f"    exterior silhouette never grows rearward over "
          f"{len(ladder)} sections; no support needed outside the two "
          "declared blind pockets")


def test_wing_clearance_is_unchanged() -> None:
    """Both wing families must clear this part exactly as they clear the ND.

    Every part of this silhouette is inside the released crescent's plan, so
    no wing envelope should move.  This checks that rather than asserting it.
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
    released = _staged(STAGE_STATES[0], "addon_tweeter_crescent")
    for wing_path in wings:
        wing = Part(import_step(str(wing_path)).solids())
        overlap = (solid & wing).volume
        assert overlap == 0.0, (
            f"{wing_path.parent.name} wing intersects the BMR pod by "
            f"{overlap:.6f} mm3")
        if released is not None:
            assert (released & wing).volume == 0.0, (
                f"{wing_path.parent.name} wing already intersects the "
                "released crescent; the comparison is meaningless")
        print(f"    {wing_path.parent.name} wing: clear")


def main() -> None:
    tests = (
        test_vase_authority_is_mirrored_exactly,
        test_mount_constants_equal_the_released_joint_authority,
        test_depth_stack_is_two_full_driver_envelopes,
        test_pod_outer_wall_is_the_driver_land,
        test_struts_are_sized_from_the_half_lap_they_feed,
        test_inherited_tweeter_clamp_holes_are_gone,
        test_candidate_flags_are_set,
        test_part_is_not_wired_into_the_release,
        test_exported_solid_is_one_valid_body_that_fits_the_bed,
        test_declared_openings_are_the_only_openings,
        test_mount_interface_is_geometrically_identical_to_the_released_ear,
        test_mate_simulation_against_the_staged_um_collar,
        test_no_interference_with_the_um_collar,
        test_free_t_cable_corridor_stays_open,
        test_exterior_never_grows_rearward,
        test_wing_clearance_is_unchanged,
    )
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    suffix = f"; {len(SKIPPED)} skipped gate(s)" if SKIPPED else ""
    print(f"{bmr.PART_NAME}: {len(tests)} focused gates pass{suffix}")


if __name__ == "__main__":
    main()
