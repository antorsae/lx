"""Focused source/BREP contract for the candidate BMR pod.

Pure gates (constants, vase-authority equality, candidate flags) always run.
Geometry gates read the exported BREP under
``build/bmr_crescent_TEBM35C10-4/`` and are skipped with an explicit message
when it is absent; they refuse to pass against a stale export.  The staged
gates additionally need the hash-verified Obi-Wan stage BREPs.

This part is not a superset of the released ND25FW-4 crescent -- it keeps that
crescent's *mount* and its junction *seam*, and nothing else -- so the mate is
proven by asserting the two ear footprints are geometrically identical and by
assembling the part against the staged UM collar, not by differencing whole
silhouettes.

Four gates carry the flush-junction rework specifically: the axis is
recomputed from the two released constraints and has to be the tighter of
them; the assembly is projected head on against the staged collar and the plan
is walked column by column for windows; every declared opening has to name the
side it faces, with no exterior ones; and the free T cable has to reach the
declared mate-face entry without being touched or exposed on the way.

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
    extrude,
    import_brep,
    import_step,
)

from lx521_baffle.io import sha256_file
from lx521_baffle.obiwan import bmr_crescent as bmr
from lx521_baffle.obiwan.carriers import (
    CORE_REAR_Z,
    T_UM_CABLE_MOUTH_HALF_WIDTH,
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
    _enforce_junction_plan_ownership,
    _plan_prism,
    _subtract_plan_prisms,
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
        "print(json.dumps({n: getattr(v, n) for n in names}))"
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
    # The cable duct runs through the front chamber's own depth, and clears
    # both the acoustic front and the blind wall by a real ligament.
    for wall_z in (THICKNESS_MM, bmr.FRONT_POCKET_FLOOR_Z_MM):
        ligament = abs(bmr.CABLE_DUCT_Z_MM - wall_z) - bmr.CABLE_DUCT_R_MM
        assert ligament >= bmr.T_BLIND_BACK_WALL_THICKNESS_MM, (
            f"the cable duct at z={bmr.CABLE_DUCT_Z_MM} leaves only "
            f"{ligament:.3f} mm to the wall at z={wall_z}")


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
    print(f"    pod D{bmr.POD_OUTER_D_MM:.0f} = the driver land; "
          f"{bmr.POD_WALL_OVER_POCKET_MM:.3f} mm wall outside the pocket, "
          f"{bmr.POD_WALL_OVER_INSERT_MM:.3f} mm outside each M2 bore")


def test_pod_is_dropped_as_far_as_the_mate_allows() -> None:
    """The axis is set by the tighter of the two UM constraints, not chosen.

    Two things stop the drop: the released 0.20 mm clearance on the UM's
    native R51.7 core ring, and the UM half-lap's own receiver notch, which
    the D66 land may not be nicked by.  Both are computed here from the same
    released datums the part uses, and the axis has to be the larger.
    """
    ring = UM_CUTOUT[1] + UM_CORE_R + bmr.UM_MATE_GAP_MM + bmr.POD_OUTER_R_MM
    notch = TWEETER_JOINT_Y + math.sqrt(
        (bmr.POD_OUTER_R_MM + bmr.EAR_NOTCH_R_MM
         + bmr.EAR_NOTCH_LIGAMENT_MM) ** 2 - TWEETER_JOINT_X[1] ** 2)
    assert _close(bmr.AXIS_Y_LIMIT_FROM_UM_RING_MM, ring, 1.0e-6)
    assert _close(bmr.AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM, notch, 1.0e-6)
    assert _close(bmr.BMR_AXIS_XY[1], max(ring, notch), 1.0e-6)
    assert bmr.AXIS_GOVERNING_CONSTRAINT == "um_half_lap_receiver_notch"
    assert _close(bmr.BMR_AXIS_XY[0], 0.0)
    assert _close(bmr.BMR_AXIS_XY[1], 452.494193004, 1.0e-6)

    # The notch's ligament is the vase's own qualified minimum wall, and the
    # D66 land really does clear it: a nick there would show up at z=6.7 as
    # rearward plan growth, which this print orientation cannot take.
    assert _close(bmr.EAR_NOTCH_R_MM,
                  TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR)
    assert _close(bmr.EAR_NOTCH_R_MM, 5.0)
    assert _close(bmr.EAR_NOTCH_LIGAMENT_MM,
                  bmr.T_BLIND_BACK_WALL_THICKNESS_MM)
    assert _close(bmr.POD_WALL_OFF_EAR_NOTCH_MM, 1.20, 1.0e-6)
    assert bmr.POD_WALL_OFF_UM_RING_MM >= bmr.UM_MATE_GAP_MM

    # The move is real and recorded against the released axis it left.
    assert _close(bmr.RELEASED_AXIS_XY[1], 468.193)
    assert bmr.POD_DROP_MM > 15.0
    assert _close(bmr.UM_AXIS_SPACING_MM,
                  bmr.RELEASED_UM_AXIS_SPACING_MM - bmr.POD_DROP_MM, 1.0e-6)
    assert _close(bmr.RELEASED_UM_AXIS_SPACING_MM, 102.112, 1.0e-6)
    assert _close(bmr.SCALLOP_R_MM, 39.25)
    print(f"    axis y {bmr.RELEASED_AXIS_XY[1]:.3f} -> "
          f"{bmr.BMR_AXIS_XY[1]:.6f} ({bmr.POD_DROP_MM:.3f} mm closer); "
          f"MU10-to-BMR spacing {bmr.RELEASED_UM_AXIS_SPACING_MM:.3f} -> "
          f"{bmr.UM_AXIS_SPACING_MM:.3f} mm; "
          f"{bmr.POD_WALL_OFF_EAR_NOTCH_MM:.3f} mm off the notch, "
          f"{bmr.POD_WALL_OFF_UM_RING_MM:.3f} mm off the UM ring")


def test_skirt_fills_the_junction_and_outsections_the_struts() -> None:
    """The junction is solid, on the released seam, and stronger than before.

    The two struts and the window between them are gone.  What replaces them
    has to (a) sit on the released crescent's own seam rather than a new
    boundary, (b) stay inside the plate band so nothing but the driver stack
    reaches behind the core rear plane, and (c) beat the section the struts
    reached, since the point of the qualified half-lap is that it governs.
    """
    assert tuple(bmr.SKIRT_Z) == (CORE_REAR_Z, THICKNESS_MM)
    assert _close(bmr.SKIRT_DEPTH_MM, 11.5)
    assert _close(bmr.UM_MATE_R_MM, UM_CORE_R + 0.20)
    assert _close(bmr.UM_MATE_R_MM, 51.9)

    plan = bmr.skirt_plan()
    assert plan.geom_type == "Polygon" and not plan.interiors, (
        "the plate-band plan must close to one simple region")
    # The fill's own edge is the released recut; the closure web's seam runs
    # closer, exactly as it does on the released crescent.
    assert bmr.base_um_ring_clearance_mm() >= bmr.UM_MATE_GAP_MM, (
        f"the flush fill leaves only {bmr.base_um_ring_clearance_mm():.4f} mm "
        "on the UM core ring")
    assert 0.0 < bmr.skirt_um_ring_clearance_mm() < bmr.UM_MATE_GAP_MM

    # The window the user rejected is gone: on the -Y meridian the plan is
    # continuous from the pod wall down to the mate face.
    from shapely.geometry import LineString as _Line
    mate_y = UM_CUTOUT[1] + bmr.UM_MATE_R_MM
    pod_y = bmr.BMR_AXIS_XY[1] - bmr.POD_OUTER_R_MM
    assert pod_y > mate_y, "the pod wall must stand off the mate face"
    # The one thing allowed to interrupt the plan is a released wing: just
    # outboard of each boss the wings run a tongue into the slot the released
    # crescent leaves there, and this part yields to it exactly as the release
    # does.  Every other break would be a window.
    # Grown by the plan's own decimation budget at both ends: the skirt's
    # boundary is simplified after the subtraction, so it lands a couple of
    # microns either side of the wing's edge.  A real window is orders of
    # magnitude bigger than that.
    keepout = bmr._wing_keepout_plan().buffer(2.0 * bmr.SKIRT_PLAN_SIMPLIFY_MM)
    for x in [value / 2.0 for value in range(-66, 67)]:
        column = _Line([(x, 380.0), (x, bmr.BMR_AXIS_XY[1])])
        run = column.intersection(plan)
        # A grazing column at |x|=33 meets the pod in a single point, and a
        # column lying exactly on the closure web's own vertical edge at
        # |x|=6 comes back as two touching pieces; only a real gap between
        # consecutive pieces is a break.
        spans = sorted(
            (piece.bounds[1], piece.bounds[3])
            for piece in getattr(run, "geoms", [run])
            if piece.geom_type == "LineString" and piece.length > 1.0e-6)
        for lower, upper in zip(spans, spans[1:]):
            gap = _Line([(x, lower[1]), (x, upper[0])])
            if gap.length <= 1.0e-6:
                continue
            assert gap.difference(keepout).length <= 1.0e-6, (
                f"the plan is broken at x={x:+.1f} between y={lower[1]:.4f} "
                f"and y={upper[0]:.4f}, where no released wing sits; a gap "
                "there is a window straight through the assembly")
    run = _Line([(0.0, 380.0), (0.0, bmr.BMR_AXIS_XY[1])]).intersection(plan)
    assert run.geom_type == "LineString", (
        f"the meridian column is {run.geom_type}, not one unbroken run")
    # Faceting puts the decimated arc a few microns outside the nominal
    # R51.90; what matters is that the plan really does start at the mate
    # face and not somewhere short of it.
    assert 0.0 <= run.bounds[1] - mate_y <= 0.010, (
        f"on the meridian the plan starts at {run.bounds[1]:.4f}, not on the "
        f"R{bmr.UM_MATE_R_MM} mate face at {mate_y:.4f}")

    # And it is strictly a fill: nothing reaches outboard of the D66 land.
    minx, miny, maxx, maxy = plan.bounds
    assert _close(maxx, bmr.POD_OUTER_R_MM, 1.0e-6)
    assert _close(minx, -bmr.POD_OUTER_R_MM, 1.0e-6)
    assert _close(maxy, bmr.BMR_AXIS_XY[1] + bmr.POD_OUTER_R_MM, 1.0e-6)

    # Section at the ears: the superseded struts reached 1.44x the half-lap's
    # own net ligament, so the fill has to be at least that.
    assert _close(bmr.EAR_THICKNESS_MM, 5.9)
    assert _close(bmr.EAR_NET_LIGAMENT_MM, 5.2)
    assert _close(bmr.EAR_NET_SECTION_MM2, 30.68)
    section = bmr.ear_load_path_section_mm2()
    ratio = section / bmr.EAR_NET_SECTION_MM2
    assert ratio >= bmr.SUPERSEDED_STRUT_SECTION_RATIO, (
        f"the ear-to-pod load path is only {section:.3f} mm2 = {ratio:.3f}x "
        "the half-lap's net ligament; the two struts it replaced reached "
        f"{bmr.SUPERSEDED_STRUT_SECTION_RATIO}x")
    print(f"    skirt z={bmr.SKIRT_Z[0]}..{bmr.SKIRT_Z[1]}, plan area "
          f"{plan.area:.1f} mm2, fill {bmr.base_um_ring_clearance_mm():.4f} mm "
          f"off the UM ring and the web seam {bmr.skirt_um_ring_clearance_mm():.4f} mm; "
          f"ear load path {section:.2f} mm2 = {ratio:.2f}x the half-lap")


def test_cable_path_is_one_hidden_entry_and_one_partition_pass() -> None:
    """No external outlets; one mate-face entry aligned with the UM mouth."""
    facts = bmr.design_facts()["cable"]
    assert facts["external_outlets"] == 0
    assert facts["entries"] == 1
    # The entry sits inside the UM's own declared central cable mouth.
    assert abs(bmr.CABLE_ENTRY_XY[0]) <= T_UM_CABLE_MOUTH_HALF_WIDTH
    assert _close(bmr.CABLE_DUCT_Z_MM, bmr.TS_FREE_CABLE_Z)
    assert _close(bmr.CABLE_DUCT_Z_MM, 3.8, 1.0e-9)
    # The mouth is on the mate face itself.
    entry_r = math.hypot(bmr.CABLE_ENTRY_XY[0] - UM_CUTOUT[0],
                         bmr.CABLE_ENTRY_XY[1] - UM_CUTOUT[1])
    assert _close(entry_r, bmr.UM_MATE_R_MM, 1.0e-6), (
        f"the cable entry is at r={entry_r:.4f}, not on the R"
        f"{bmr.UM_MATE_R_MM} mate face")
    # Ø6.00 is the UM's own T lumen, and a cable arriving off-axis only fits
    # through the bore's projected aperture.
    assert _close(bmr.CABLE_DUCT_D_MM, bmr.TS_DUCT_D)
    assert _close(bmr.CABLE_DUCT_D_MM, 6.0)
    assert bmr.CABLE_MOUTH_APERTURE_MM >= bmr.TS_CABLE_D_EST, (
        f"a {bmr.TS_CABLE_D_EST} mm cable arriving "
        f"{bmr.CABLE_MOUTH_MISALIGNMENT_DEG:.2f} degrees off the duct sees "
        f"only {bmr.CABLE_MOUTH_APERTURE_MM:.3f} mm of aperture")
    # The partition pass is the vase's own single-driver lead branch, capped
    # at Ø4.6, and it keeps that same 1.20 mm wall to the pocket bore.
    assert _close(bmr.PARTITION_PASS_D_MM, bmr.UPPER_T_BRANCH_D_MM)
    assert bmr.PARTITION_PASS_D_MM <= 4.6
    assert _close(
        bmr.PARTITION_PASS_OFFSET_MM + bmr.PARTITION_PASS_D_MM / 2.0
        + bmr.T_BLIND_BACK_WALL_THICKNESS_MM,
        bmr.TEBM_CUTOUT_D_MM / 2.0, 1.0e-6)
    assert bmr.PARTITION_PASS_XY[1] < bmr.BMR_AXIS_XY[1], (
        "the partition pass must be on the -Y side the cable arrives from")
    # The collar carries the duct and its wall, and is the shape of the duct
    # rather than a slab around it.
    assert bmr.ENTRY_COLLAR_Z[1] == CORE_REAR_Z
    assert _close(bmr.ENTRY_COLLAR_WALL_MM,
                  bmr.T_BLIND_BACK_WALL_THICKNESS_MM)
    assert _close(bmr.ENTRY_COLLAR_R_MM,
                  bmr.CABLE_DUCT_R_MM + bmr.ENTRY_COLLAR_WALL_MM)
    assert _close(bmr.ENTRY_COLLAR_Z[0],
                  bmr.CABLE_DUCT_Z_MM - bmr.ENTRY_COLLAR_R_MM, 1.0e-9)
    collar = bmr.entry_collar_plan()
    relief = bmr._um_owned_relief_plan()
    skirt = bmr.skirt_plan().difference(relief)
    assert collar.within(skirt.buffer(1.0e-9)), (
        "the entry collar plan must stay inside what the skirt above it "
        "actually is, or the exterior grows rearward at the core rear plane")

    # That relief is mirrored from the released ownership helper, which only
    # reaches the closure web's Z band.  Apply both to the same prism: if the
    # mirror ever drifts, they stop removing the same volume.
    probe = _plan_prism(bmr.skirt_plan(), *bmr.SKIRT_Z)
    theirs = _enforce_junction_plan_ownership(probe, "t_um", "tweeter")
    mine = _subtract_plan_prisms(probe, relief, *bmr.SKIRT_Z)
    assert abs(float(theirs.volume) - float(mine.volume)) < 1.0e-6, (
        "the locally mirrored ownership relief has drifted from "
        f"_enforce_junction_plan_ownership: {theirs.volume} vs {mine.volume}")

    # It really is a stadium hugging the bore: every stretch of its boundary
    # that is not the skirt's own edge stands exactly one collar radius off
    # the duct's plan sweep.  A slab, or any face or corner of one, would sit
    # further out than that somewhere.
    from shapely.geometry import LineString as _Line2, Point as _Point
    mouth = bmr.CABLE_ENTRY_XY
    direction = bmr.CABLE_DUCT_DIR
    sweep = _Line2([
        (mouth[0] - bmr.ENTRY_COLLAR_BACK_MM * direction[0],
         mouth[1] - bmr.ENTRY_COLLAR_BACK_MM * direction[1]),
        (mouth[0] + bmr.ENTRY_COLLAR_REACH_MM * direction[0],
         mouth[1] + bmr.ENTRY_COLLAR_REACH_MM * direction[1]),
    ])
    inherited = bmr.skirt_plan().exterior.union(relief.boundary).buffer(
        4.0 * bmr.SKIRT_PLAN_SIMPLIFY_MM)
    own_edge = collar.exterior.difference(inherited)
    assert own_edge.length > 0.5 * collar.exterior.length, (
        "most of the collar's boundary should be its own, not the skirt's")
    strayed = max(
        abs(sweep.distance(_Point(*coordinate)) - bmr.ENTRY_COLLAR_R_MM)
        for piece in getattr(own_edge, "geoms", [own_edge])
        for coordinate in piece.coords)
    assert strayed <= 0.01, (
        f"the collar's own boundary strays {strayed:.4f} mm from a constant "
        f"{bmr.ENTRY_COLLAR_R_MM} mm offset of the bore; it is not a stadium")
    # And it is small: a slab spanning the mouth would be several times this.
    assert collar.area < 150.0, (
        f"the entry collar plan is {collar.area:.1f} mm2; that is a box, not "
        "a collar")
    print(f"    entry Ø{bmr.CABLE_DUCT_D_MM:.2f} at "
          f"({bmr.CABLE_ENTRY_XY[0]:.3f}, {bmr.CABLE_ENTRY_XY[1]:.3f}, "
          f"{bmr.CABLE_DUCT_Z_MM}) bearing {bmr.CABLE_DUCT['bearing_deg']:.2f} "
          f"deg, {bmr.CABLE_MOUTH_MISALIGNMENT_DEG:.2f} deg off the cable "
          f"({bmr.CABLE_MOUTH_APERTURE_MM:.3f} mm aperture for a "
          f"{bmr.TS_CABLE_D_EST} mm cable); partition pass "
          f"Ø{bmr.PARTITION_PASS_D_MM} at y="
          f"{bmr.PARTITION_PASS_XY[1]:.3f}")


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
        "um_mate_face_cable_entry",
        "chamber_partition_cable_pass",
        "um_half_lap_clearance_passages",
        "um_half_lap_insert_receivers",
        "m2_driver_insert_bores",
    ]
    # The two external lead outlets went with the struts.
    assert not [name for name in names if "lead_outlet" in name]
    silhouette = bmr.design_facts()["silhouette"]
    assert silhouette["inherits_released_crescent_outline"] is False
    assert any("M4" in entry
               for entry in silhouette["removed_from_the_first_candidate"])
    assert any("outlet" in entry
               for entry in silhouette["removed_from_the_first_candidate"])


def test_no_declared_opening_reaches_the_assembled_exterior() -> None:
    """Every opening faces the UM mate, a driver, or nothing at all.

    This is the whole point of the cable rework: with the pod assembled on the
    collar and both drivers fitted, there must be no hole anyone can see.
    """
    exposures = {"um_mate", "driver_face", "internal"}
    for opening in bmr.declared_openings():
        assert "exposure" in opening, (
            f"{opening['name']} does not declare which side it faces")
        assert opening["exposure"] in exposures, (
            f"{opening['name']} is exposed to the {opening['exposure']}")
    assert bmr.design_facts()["exterior_openings"] == []
    by_exposure = {}
    for opening in bmr.declared_openings():
        by_exposure.setdefault(opening["exposure"], []).append(opening["name"])
    # One cable entry, on the mate, and one internal pass.  Not two of either.
    assert [name for name in by_exposure["um_mate"]
            if "cable" in name] == ["um_mate_face_cable_entry"]
    assert by_exposure["internal"] == ["chamber_partition_cable_pass"]
    print("    openings: "
          + "; ".join(f"{side}={sorted(names)}"
                      for side, names in sorted(by_exposure.items())))


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

    # The partition separates the two chambers everywhere except at the one
    # declared pass, which really is open over its whole declared span.
    partition_mid = (bmr.REAR_POCKET_ROOF_Z_MM
                     + bmr.FRONT_POCKET_FLOOR_Z_MM) / 2.0
    assert _material(solid, axis_x, axis_y, partition_mid, size=0.4) > 0.0, (
        "the back-to-back partition is missing; the two rear volumes would "
        "be one chamber")
    pass_x, pass_y = bmr.PARTITION_PASS_XY
    for z in (bmr.REAR_POCKET_ROOF_Z_MM + 0.2, partition_mid,
              bmr.FRONT_POCKET_FLOOR_Z_MM - 0.2):
        assert _material(solid, pass_x, pass_y, z, size=0.4) == 0.0, (
            f"the declared partition pass is obstructed at z={z:.3f}")
    # It is exactly one pass of exactly the declared size: the partition is
    # intact a diameter to either side of it and on the opposite meridian.
    for offset in (-bmr.PARTITION_PASS_D_MM, bmr.PARTITION_PASS_D_MM):
        assert _material(solid, pass_x + offset, pass_y, partition_mid,
                         size=0.4) > 0.0, (
            "the partition pass is wider than declared")
    assert _material(
        solid, axis_x,
        axis_y + bmr.PARTITION_PASS_OFFSET_MM, partition_mid,
        size=0.4) > 0.0, "an undeclared second partition pass exists"

    # The one cable entry is open from the mate face into the front chamber,
    # and nothing else breaks the mate face at that height.
    entry_x, entry_y = bmr.CABLE_ENTRY_XY
    direction = bmr.CABLE_DUCT_DIR
    for reach in (-0.8, 1.0, bmr.CABLE_DUCT_LENGTH_MM / 2.0,
                  bmr.CABLE_DUCT_LENGTH_MM - 1.0):
        assert _material(
            solid, entry_x + direction[0] * reach,
            entry_y + direction[1] * reach, bmr.CABLE_DUCT_Z_MM,
            size=0.4) == 0.0, (
            f"the cable duct is obstructed {reach:.2f} mm along its axis")
    # Beside the duct, half way along where the pod wall surrounds it, the
    # material is intact; below it in the entry collar the declared 1.20 mm
    # floor is there.  Probing beside the mouth itself would only sample the
    # free space outside the curved mate face.
    normal = (-direction[1], direction[0])
    middle = bmr.CABLE_DUCT_LENGTH_MM / 2.0
    for sign in (-1.0, 1.0):
        offset = sign * (bmr.CABLE_DUCT_R_MM + 0.6)
        assert _material(
            solid, entry_x + normal[0] * offset + direction[0] * middle,
            entry_y + normal[1] * offset + direction[1] * middle,
            bmr.CABLE_DUCT_Z_MM, size=0.4) > 0.0, (
            "the cable duct is wider than declared")
    assert _material(
        solid, entry_x + direction[0] * 1.0, entry_y + direction[1] * 1.0,
        bmr.ENTRY_COLLAR_Z[0] + 0.4, size=0.3) > 0.0, (
        "the cable duct has no floor under it in the entry collar")

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
        # above as the old whole-silhouette identity claim.  The comparison is
        # confined to the plate band the two share -- this part's driver stack
        # would otherwise swamp it -- and has to show a large difference in
        # *both* directions.  Material of the release that is absent here is
        # what rules out an inherited outline; the strutted candidate dropped
        # half the release, the flush skirt drops less because the pod came
        # down onto the collar, but it is still not a superset.
        band = Pos(0.0, bmr.BMR_AXIS_XY[1], (CORE_REAR_Z + THICKNESS_MM) / 2.0
                   ) * Box(400.0, 400.0, THICKNESS_MM - CORE_REAR_Z)
        mine = solid & band
        absent = (released - solid).volume
        added = (mine - released).volume
        assert absent / released.volume > 0.05, (
            f"{state}: only {100.0 * absent / released.volume:.1f}% of the "
            "released crescent's material is absent here, so the mount-only "
            "comparison above may be hiding an inherited outline")
        assert (absent + added) / released.volume > 1.0, (
            f"{state}: the plate-band symmetric difference against the "
            f"released crescent is only "
            f"{100.0 * (absent + added) / released.volume:.1f}% of it")
        print(f"    {state}: in the shared plate band, "
              f"{100.0 * absent / released.volume:.1f}% of the released "
              f"crescent is absent and {100.0 * added / released.volume:.1f}% "
              "as much again is new")
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
        print(f"    {state}: no interference anywhere, with the skirt now "
              "closed onto the collar")
        checked += 1
    if not checked:
        _skip("no staged UM carrier available for the interference gate")


def test_the_assembled_junction_has_no_window() -> None:
    """Looking at the assembly head on, the junction must be solid.

    This is the user-visible claim.  The pod and the collar are projected
    along -Z onto the plan; between the two parts the projection must be
    continuous, with nothing but the released mate seam between them.

    The sweep runs out to the outer edge of the half-lap bosses and stops.
    Past them the two parts simply end and open space between them is the
    outboard edge of the assembly, not a window -- and the released crescent
    leaves the same space there, which the tail of this gate checks rather
    than assumes.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    boss_edge = abs(TWEETER_JOINT_X[1]) + TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0
    checked = 0
    for state in STAGE_STATES:
        um = _staged(state, "core_um_carrier")
        if um is None:
            _skip(f"{state}: staged UM carrier BREP absent")
            continue
        # A sight line straight through both parts, on the meridian and out
        # to the ears.  Anywhere the pod stands off the collar, something has
        # to be in the way at some Z.
        widest = 0.0
        for x in [value / 2.0 for value in range(-56, 57)]:
            column = Pos(x, 0.0, 0.0) * Box(0.30, 400.0, 60.0)
            here = (solid & column)
            there = (um & column)
            if here is None or here.volume == 0.0:
                continue
            gaps = []
            near = here.bounding_box().min.Y
            far = there.bounding_box().max.Y if (
                there is not None and there.volume > 0.0) else None
            if far is None:
                continue
            gaps.append(near - far)
            widest = max(widest, max(gaps))
            assert near - far <= 0.30, (
                f"{state}: at x={x:+.1f} the pod's nearest material is "
                f"{near:.3f} but the collar stops at {far:.3f}; that "
                f"{near - far:.3f} mm sight line is the window this design "
                "exists to close")
        # Outboard of the bosses both parts really do end.  The released
        # crescent leaves the same open space there, so the sweep stopping at
        # the boss edge is the shape of the assembly, not a chosen bound.
        released = _staged(state, "addon_tweeter_crescent")
        if released is not None:
            for x in (boss_edge + 1.5, boss_edge + 4.0):
                column = Pos(x, 0.0, 0.0) * Box(0.30, 400.0, 60.0)
                theirs = released & column
                collar = um & column
                if (theirs is None or theirs.volume == 0.0
                        or collar is None or collar.volume == 0.0):
                    continue
                assert (theirs.bounding_box().min.Y
                        - collar.bounding_box().max.Y) > 1.0, (
                    f"{state}: the released crescent meets the collar at "
                    f"x={x:+.1f}, so this gate cannot stop at the boss edge")
        print(f"    {state}: no window out to the boss edge at "
              f"x=±{boss_edge:.1f}; the widest sight line across the junction "
              f"is {widest:.3f} mm, the released mate seam")
        checked += 1
    if not checked:
        _skip("no staged UM carrier available for the window gate")


def test_free_t_cable_reaches_the_declared_entry_without_exposure() -> None:
    """The cable now terminates in this part; it must arrive hidden.

    The old gate kept a corridor open behind the crescent because the cable
    floated there.  It does not any more: it goes straight from the UM's own
    declared mouth into the one duct.  What has to hold instead is that (a)
    nothing of this part touches the cable on the UM's side of the mate, and
    (b) where the cable crosses the mate it is inside the declared entry, with
    the whole cable section inside the bore rather than on its rim.  Nothing
    else used that corridor: the modelled T cable is its only occupant, and it
    is the same body checked here.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    mate = Pos(UM_CUTOUT[0], UM_CUTOUT[1], (bmr.REAR_MOUNT_Z_MM
                                            + THICKNESS_MM) / 2.0) * Cylinder(
        bmr.UM_MATE_R_MM, bmr.STACK_DEPTH_MM + 2.0)
    checked = 0
    for state in STAGE_STATES:
        cable = _staged(state, "review_reference_ts_cable")
        if cable is None:
            _skip(f"{state}: staged T cable reference BREP absent")
            continue
        # (a) On the UM's side of the mate face the cable is untouched.
        before = cable & mate
        overlap = None if before is None else (solid & before)
        volume = 0.0 if overlap is None else float(overlap.volume)
        assert volume == 0.0, (
            f"{state}: the BMR pod pinches the modelled T cable by "
            f"{volume:.6f} mm3 before it reaches the mate face")
        # (b) At the mate face the cable's own section is inside the duct.
        duct = bmr._duct_cutter()
        crossing = cable - mate
        assert crossing is not None and crossing.volume > 0.0, (
            f"{state}: the modelled T cable never crosses the mate face")
        # Only the first 6 mm past the face: beyond that the modelled cable
        # keeps its released free-flight path instead of following the duct.
        throat = Pos(bmr.CABLE_ENTRY_XY[0], bmr.CABLE_ENTRY_XY[1],
                     bmr.CABLE_DUCT_Z_MM) * Box(14.0, 14.0, 14.0)
        entering = crossing & throat
        assert entering is not None and entering.volume > 0.0
        escaped = entering - duct
        leak = 0.0 if escaped is None else float(escaped.volume)
        fraction = leak / float(entering.volume)
        assert fraction < 0.02, (
            f"{state}: {100.0 * fraction:.2f}% of the cable entering the mate "
            "face is outside the declared duct, so it would land on the rim")
        checked += 1
    if checked:
        print(f"    free T cable at z={bmr.TS_FREE_CABLE_Z:.2f} runs clear to "
              f"the mate face and into the Ø{bmr.CABLE_DUCT_D_MM:.2f} entry in "
              f"{checked} staged state(s)")


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
    whole outside of the part -- the skirt, the ear step, the entry collar
    and the pod.  The one declared mate-face entry is filled back in first: a
    Ø6 bore through a wall is a bridge, not plan growth.
    """
    solid, _facts = _exported_solid()
    if solid is None:
        return
    envelope = (Pos(bmr.BMR_AXIS_XY[0], bmr.BMR_AXIS_XY[1],
                    (bmr.REAR_MOUNT_Z_MM + THICKNESS_MM) / 2.0) * Cylinder(
        bmr.POD_OUTER_R_MM, bmr.STACK_DEPTH_MM)).fuse(
        _plan_prism(bmr.entry_collar_plan(), *bmr.ENTRY_COLLAR_Z))
    probe = solid.fuse(bmr._duct_cutter() & envelope).clean()

    ladder = (
        18.29, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.45, 12.35, 12.0,
        11.0, 10.0, 9.0, 8.0, 7.0, 6.85, 6.75, 6.5, 5.0, 2.0, 0.0,
        -0.35, -0.45, -1.0, -10.0, -20.0, -31.85,
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

    The dropped pod and its skirt now reach down into plan territory the
    struts never touched, and the wings top out at y=449, so this is a real
    check rather than a formality.
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
        test_pod_is_dropped_as_far_as_the_mate_allows,
        test_skirt_fills_the_junction_and_outsections_the_struts,
        test_cable_path_is_one_hidden_entry_and_one_partition_pass,
        test_inherited_tweeter_clamp_holes_are_gone,
        test_no_declared_opening_reaches_the_assembled_exterior,
        test_candidate_flags_are_set,
        test_part_is_not_wired_into_the_release,
        test_exported_solid_is_one_valid_body_that_fits_the_bed,
        test_declared_openings_are_the_only_openings,
        test_mount_interface_is_geometrically_identical_to_the_released_ear,
        test_mate_simulation_against_the_staged_um_collar,
        test_no_interference_with_the_um_collar,
        test_the_assembled_junction_has_no_window,
        test_free_t_cable_reaches_the_declared_entry_without_exposure,
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
