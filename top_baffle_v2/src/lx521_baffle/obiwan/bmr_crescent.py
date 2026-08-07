"""Candidate Obi-Wan crescent for two coaxial back-to-back TEBM35C10-4 BMRs.

This is an alternative to the released ND25FW-4 tweeter crescent, not a
replacement for it.  It presents the *identical* half-lap interface to an
unmodified Obi-Wan UM collar (x=+/-24, y=421.5, complete front local-D9.8
ears, standalone blind D4.6 x 4.0 heat-set receivers, 360-degree walls,
1.9 mm acoustic-front floors, 0.20 mm axial gap), so the two crescents are
mutually swappable without touching the UM print.

Where the released crescent leaves an open R39.25 scallop for the
face-to-face Dayton pair, this part fills that scallop with one circular
boss carrying two Tectonic TEBM35C10-4 BMRs on the *same* acoustic axis:
the front driver mounts on the shared z=18.3 acoustic plane and the rear
driver mounts on the boss's rear plane facing -z.  The two 25.1 mm driver
envelopes therefore stack back to back about that one axis, and each keeps
the vase's qualified 1.20 mm blind wall, so the shared partition is
2.40 mm and the local stack is exactly 2 x 25.1 = 50.2 mm.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=18.3 is the acoustic
front and z=6.8 is the Obi-Wan core rear plane.  The BMR boss grows only
rearward, to z=-31.9.  The plan silhouette grows only where the released
scallop was already open (the boss disc and two tangent corner blends);
nothing moves outward past the released crescent's flanks, so the wing
clearance envelope is unchanged.

Candidate status
----------------
Nothing here is release-authorized.  ``RELEASE_AUTHORIZED`` is false and
``PHYSICAL_MEASURE_REQUIRED`` is true: the driver envelope, the back-to-back
partition, the boss root and the two-screw joint demand under roughly twice
the released crescent's hanging mass all need physical qualification before
this part is printed for use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from build123d import (
    Axis,
    Box,
    Cylinder,
    Line,
    Part,
    Plane,
    Polyline,
    Pos,
    Rot,
    Spline,
    Wire,
    extrude,
    loft,
    make_face,
    mirror,
    revolve,
)

from ..assembly import ordered_labeled_compound
from ..base import (
    CRESCENT_SCALLOP_CY,
    CRESCENT_TAPER_R_MM,
    THICKNESS_MM,
    TWEETER_HOLE_D_MM,
    TWEETER_HOLE_XY,
    _crescent_taper_cutters,
    _crescent_taper_depth,
)
from ..cables import ROUTING_PROFILE
from ..proud.b import TWEETER_DROP_MM
from .attachments import _cylinder_at, _fuse_required, tweeter_crescent
from .carriers import (
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
    _require_guarded_build,
)
from .route import TS_FREE_CABLE_Z


if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "the Obi-Wan BMR crescent requires LX_ROUTING_PROFILE=obiwan (R6F)")

PRINT_ORIENTATION = "front-face-down"
PART_NAME = "obiwan_bmr_crescent_TEBM35C10-4"
RELEASE_VARIANT = "Obiwan-TEBM35C10-4-BMR-crescent"

# This part has never been printed, fitted or loaded.  Both flags stay set
# until the qualification table in the report is closed out.
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True

# No captive magnets in v1, exactly like the released ND25FW-4 crescent.
MAGNET_COUNT = 0


# --- TEBM35C10-4 driver authority -------------------------------------------
# ``proud.vase_tebm35c10_4`` is the driver-pocket authority for this family,
# but it cannot be imported here: it reaches ``proud.b2_split``, which refuses
# to load unless LX_ROUTING_PROFILE=proud, while this part only builds under
# the obiwan profile.  The two profiles are mutually exclusive in one process,
# so the vase's primitives are mirrored below and ``VASE_AUTHORITY`` binds each
# one to its vase name.  ``tests/test_bmr_crescent.py`` evaluates the real vase
# module in a proud-profile subprocess and asserts exact equality for every
# entry, so a drift in the vase fails this part rather than silently diverging.
TEBM_DEPTH_MM = 25.1
TEBM_MAX_D_MM = 54.0
TEBM_BASKET_D_MM = 43.6
TEBM_MASS_G = 51.3
TEBM_CUTOUT_D_MM = 1.69 * 25.4
TEBM_MOUNT_PCD_MM = 1.90 * 25.4
TEBM_MOUNT_HOLE_COUNT = 4
TEBM_LAND_D_MM = 66.0
TEBM_LAND_R_MM = TEBM_LAND_D_MM / 2.0
M2_INSERT_BORE_D_MM = 3.2
M2_INSERT_DEPTH_MM = 4.0
LOWER_T_MOUNT_CLOCK_DEG = 45.0
UPPER_T_MOUNT_CLOCK_DEG = -45.0
T_BLIND_BACK_WALL_THICKNESS_MM = 1.20
UPPER_T_BRANCH_D_MM = 4.60
# The vase's own outlets sit this far inside a pocket from its blind wall; it
# resolves there through the proud TS section, which is why it is mirrored as a
# plain number instead of recomputed.
VASE_POCKET_OUTLET_INSET_MM = 5.60

# Everything below is derived from the mirrored primitives, never restated.
# Rounding to 9 places keeps binary dust out of the datums the exporter
# publishes and the test compares; every value is exact at that precision.
T_CLEAR_POCKET_DEPTH_MM = round(
    TEBM_DEPTH_MM - T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_T_MOUNT_Z_MM = round(THICKNESS_MM - TEBM_DEPTH_MM, 9)
LOWER_T_POCKET_REAR_Z_MM = round(
    REAR_T_MOUNT_Z_MM + T_BLIND_BACK_WALL_THICKNESS_MM, 9)

VASE_AUTHORITY = {
    "TEBM_DEPTH_MM": TEBM_DEPTH_MM,
    "TEBM_MAX_D_MM": TEBM_MAX_D_MM,
    "TEBM_BASKET_D_MM": TEBM_BASKET_D_MM,
    "TEBM_MASS_G": TEBM_MASS_G,
    "TEBM_CUTOUT_D_MM": TEBM_CUTOUT_D_MM,
    "TEBM_MOUNT_PCD_MM": TEBM_MOUNT_PCD_MM,
    "TEBM_MOUNT_HOLE_COUNT": TEBM_MOUNT_HOLE_COUNT,
    "TEBM_LAND_D_MM": TEBM_LAND_D_MM,
    "TEBM_LAND_R_MM": TEBM_LAND_R_MM,
    "M2_INSERT_BORE_D_MM": M2_INSERT_BORE_D_MM,
    "M2_INSERT_DEPTH_MM": M2_INSERT_DEPTH_MM,
    "LOWER_T_MOUNT_CLOCK_DEG": LOWER_T_MOUNT_CLOCK_DEG,
    "UPPER_T_MOUNT_CLOCK_DEG": UPPER_T_MOUNT_CLOCK_DEG,
    "T_BLIND_BACK_WALL_THICKNESS_MM": T_BLIND_BACK_WALL_THICKNESS_MM,
    "UPPER_T_BRANCH_D_MM": UPPER_T_BRANCH_D_MM,
    "T_CLEAR_POCKET_DEPTH_MM": T_CLEAR_POCKET_DEPTH_MM,
    "REAR_T_MOUNT_Z_MM": REAR_T_MOUNT_Z_MM,
    "LOWER_T_POCKET_REAR_Z_MM": LOWER_T_POCKET_REAR_Z_MM,
}


# --- released datums this part must not move --------------------------------
# The scallop that carries the released face-to-face Dayton pair is the
# D78.50 circle about the dropped scallop centre; that centre is the released
# tweeter acoustic axis and is where the coaxial BMR pair goes.
BMR_AXIS_XY = (0.0, CRESCENT_SCALLOP_CY - TWEETER_DROP_MM)
SCALLOP_R_MM = 78.50 / 2.0
# Drawing vertex at the bottom of the scallop, in the un-dropped frame.  The
# released outline reaches it with a Bezier, so this is the authority the
# radius is checked against rather than sampled geometry.
SCALLOP_BOTTOM_DRAWING_Y_MM = 443.804
CRESCENT_TOP_EDGE_Y_MM = 468.314 - TWEETER_DROP_MM

if abs((CRESCENT_SCALLOP_CY - SCALLOP_BOTTOM_DRAWING_Y_MM)
       - SCALLOP_R_MM) > 0.01:
    raise RuntimeError(
        "released scallop radius drifted from the D78.50 drawing circle")


# --- coaxial back-to-back depth stack ---------------------------------------
# Front driver: acoustic face on the shared z=18.3 plane, pocket cut rearward,
# blind wall over the released 25.1 mm envelope.  These are the vase's own
# numbers, imported rather than restated.
FRONT_MOUNT_Z_MM = THICKNESS_MM
FRONT_POCKET_FLOOR_Z_MM = LOWER_T_POCKET_REAR_Z_MM          # -5.6
FRONT_ENVELOPE_END_Z_MM = REAR_T_MOUNT_Z_MM                 # -6.8

# Rear driver: mirror of the front one about the partition.  Each driver keeps
# a full 1.20 mm blind wall of its own, so the coaxial partition is 2.40 mm
# and the two rear volumes stay separate chambers.  A single 1.20 mm partition
# would have been a shared skin taking differential pressure from both sides;
# doubling it costs 1.2 mm of stack and keeps every driver's qualified wall.
PARTITION_THICKNESS_MM = round(2.0 * T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_POCKET_ROOF_Z_MM = round(
    FRONT_ENVELOPE_END_Z_MM - T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_MOUNT_Z_MM = round(THICKNESS_MM - 2.0 * TEBM_DEPTH_MM, 9)   # -31.9
STACK_DEPTH_MM = round(FRONT_MOUNT_Z_MM - REAR_MOUNT_Z_MM, 9)    # 50.2
REAR_PROTRUSION_MM = round(CORE_REAR_Z - REAR_MOUNT_Z_MM, 9)     # 38.7


# --- boss envelope ----------------------------------------------------------
# The boss plan is the released scallop circle plus a 0.50 mm overlap.  The
# released outline reaches the scallop with a Bezier that runs up to ~0.1 mm
# outside the true circle, so an exact-radius disc could leave a hairline
# sliver at the top of the arc; the overlap makes the fusion unconditional and
# never moves the silhouette anywhere the release has material.
BOSS_SCALLOP_OVERLAP_MM = 0.50
BOSS_PLAN_R_MM = SCALLOP_R_MM + BOSS_SCALLOP_OVERLAP_MM
# The rear face of the boss *is* the rear driver's D66 land.
BOSS_REAR_R_MM = TEBM_LAND_R_MM
BOSS_FLARE_SAMPLES = 48

# Root fairing: the released rear taper is filled back to the core rear plane
# next to the boss and faded out again with the same quintic the vase uses for
# its rear growth, so the boss stands on flat full-depth material instead of a
# feathered arm.  It stops short of the nearest UM-ear receiver footprint, so
# the released mate is untouched by construction rather than by repair.
UM_EAR_FOOTPRINT_R_MM = math.hypot(
    TWEETER_JOINT_X[1], BMR_AXIS_XY[1] - TWEETER_JOINT_Y
) - (TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR)
ROOT_FAIRING_MATE_MARGIN_MM = 0.50
# The fairing starts inside the released scallop arc and ends 0.20 mm above the
# released rear surface, so it overlaps both the boss and the arm instead of
# ending on a surface it shares.  OCC will not fuse two bodies that merely
# touch along a loft it did not build twice the same way; the released taper
# and this ramp are separate lofts, so the contact has to be a real overlap.
ROOT_FAIRING_FUSION_OVERLAP_MM = 0.20
ROOT_FAIRING_R_IN_MM = SCALLOP_R_MM - 0.50
ROOT_FAIRING_R_OUT_MM = min(
    CRESCENT_TAPER_R_MM[1],
    UM_EAR_FOOTPRINT_R_MM - ROOT_FAIRING_MATE_MARGIN_MM,
)
# Same 20-section, 4-degree fan the released taper itself is lofted on.
FAIRING_SECTION_ANGLES_DEG = tuple(-87.0 + 4.0 * index for index in range(20))
FAIRING_RAMP_SAMPLES = 24

# Two tangent plan blends close the 68-degree notch where the boss wall meets
# the released crescent's straight top edge.
PLAN_FILLET_R_MM = 5.0


# --- driver interfaces ------------------------------------------------------
FRONT_MOUNT_CLOCK_DEG = LOWER_T_MOUNT_CLOCK_DEG
REAR_MOUNT_CLOCK_DEG = UPPER_T_MOUNT_CLOCK_DEG

# Both pockets are blind, so each driver needs one declared lead outlet.  The
# vase places its outlets 5.6 mm inside the pocket from the blind wall; that
# inset is re-used here rather than invented.
POCKET_OUTLET_D_MM = UPPER_T_BRANCH_D_MM
POCKET_OUTLET_INSET_MM = VASE_POCKET_OUTLET_INSET_MM
FRONT_OUTLET_Z_MM = round(
    FRONT_POCKET_FLOOR_Z_MM + POCKET_OUTLET_INSET_MM, 9)
REAR_OUTLET_Z_MM = round(REAR_POCKET_ROOF_Z_MM - POCKET_OUTLET_INSET_MM, 9)
# Both outlets leave on the -Y meridian, the side the free T cable arrives
# from, and both sit behind the core rear plane where that cable already runs.
POCKET_OUTLET_INNER_R_MM = 18.0
POCKET_OUTLET_OUTER_R_MM = 45.0

_MATE_CROSS_CHECK = {
    "joint_x_mm": (-24.0, 24.0),
    "joint_y_mm": 421.5,
    "addon_joint_z_mm": (12.40, THICKNESS_MM),
    "core_joint_z_mm": (CORE_REAR_Z, 12.20),
    "insert_bore_d_mm": 4.6,
    "insert_depth_mm": 4.0,
    "clearance_bore_d_mm": 3.4,
}


def _check_released_mate() -> None:
    """Fail loudly if the released half-lap datums ever move upstream."""
    if tuple(float(value) for value in TWEETER_JOINT_X) != _MATE_CROSS_CHECK[
            "joint_x_mm"]:
        raise RuntimeError("released tweeter half-lap X moved")
    if float(TWEETER_JOINT_Y) != _MATE_CROSS_CHECK["joint_y_mm"]:
        raise RuntimeError("released tweeter half-lap Y moved")
    if tuple(float(value) for value in TWEETER_ADDON_JOINT_Z) != (
            _MATE_CROSS_CHECK["addon_joint_z_mm"]):
        raise RuntimeError("released crescent ear Z span moved")
    if tuple(float(value) for value in TWEETER_CORE_JOINT_Z) != (
            _MATE_CROSS_CHECK["core_joint_z_mm"]):
        raise RuntimeError("released UM ear Z span moved")
    if float(TWEETER_JOINT_INSERT_BORE_D) != _MATE_CROSS_CHECK[
            "insert_bore_d_mm"]:
        raise RuntimeError("released crescent insert receiver diameter moved")
    if float(TWEETER_JOINT_INSERT_DEPTH_MM) != _MATE_CROSS_CHECK[
            "insert_depth_mm"]:
        raise RuntimeError("released crescent insert receiver depth moved")
    if float(TWEETER_JOINT_HOLE_D) != _MATE_CROSS_CHECK["clearance_bore_d_mm"]:
        raise RuntimeError("released UM clearance bore diameter moved")


def _smootherstep(value: float) -> float:
    """Quintic smootherstep: zero first and second derivative at both ends."""
    clamped = max(0.0, min(1.0, float(value)))
    return clamped * clamped * clamped * (
        clamped * (clamped * 6.0 - 15.0) + 10.0)


def axial_gap_mm() -> float:
    return TWEETER_ADDON_JOINT_Z[0] - TWEETER_CORE_JOINT_Z[1]


def insert_front_floor_mm() -> float:
    return THICKNESS_MM - TWEETER_JOINT_INSERT_BORE_Z[1]


def boss_radius_at(z: float) -> float:
    """Boss outer radius at one Z, on the released-scallop plan above 6.8."""
    if z >= CORE_REAR_Z:
        return BOSS_PLAN_R_MM
    fraction = (z - REAR_MOUNT_Z_MM) / (CORE_REAR_Z - REAR_MOUNT_Z_MM)
    return BOSS_REAR_R_MM + (
        BOSS_PLAN_R_MM - BOSS_REAR_R_MM) * _smootherstep(fraction)


def _boss_solid():
    """Revolved boss: D66 rear land flaring G1 into the released scallop."""
    flare = []
    for index in range(BOSS_FLARE_SAMPLES + 1):
        fraction = index / BOSS_FLARE_SAMPLES
        z = REAR_MOUNT_Z_MM + (CORE_REAR_Z - REAR_MOUNT_Z_MM) * fraction
        radius = BOSS_REAR_R_MM + (
            BOSS_PLAN_R_MM - BOSS_REAR_R_MM) * _smootherstep(fraction)
        flare.append((radius, 0.0, z))
    profile = Wire([
        Line((0.0, 0.0, REAR_MOUNT_Z_MM),
             (BOSS_REAR_R_MM, 0.0, REAR_MOUNT_Z_MM)).edge(),
        Spline(*flare, tangents=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))).edge(),
        Line((BOSS_PLAN_R_MM, 0.0, CORE_REAR_Z),
             (BOSS_PLAN_R_MM, 0.0, THICKNESS_MM)).edge(),
        Line((BOSS_PLAN_R_MM, 0.0, THICKNESS_MM),
             (0.0, 0.0, THICKNESS_MM)).edge(),
        Line((0.0, 0.0, THICKNESS_MM),
             (0.0, 0.0, REAR_MOUNT_Z_MM)).edge(),
    ])
    boss = revolve(make_face(profile), axis=Axis.Z, revolution_arc=360.0)
    return Pos(BMR_AXIS_XY[0], BMR_AXIS_XY[1], 0.0) * boss


def _fairing_section_face(angle_deg: float, sign: float):
    """One radial (r, z) section of the material the released taper removed.

    The face is bounded above by the released rear surface and below by the
    same surface faded back to the core rear plane, so it is exactly the
    volume that turns the feathered arm into a flat full-depth boss root.
    """
    depth = _crescent_taper_depth(angle_deg, THICKNESS_MM - CORE_REAR_Z)
    top = CORE_REAR_Z + depth + ROOT_FAIRING_FUSION_OVERLAP_MM
    span = ROOT_FAIRING_R_OUT_MM - ROOT_FAIRING_R_IN_MM
    points = [(ROOT_FAIRING_R_IN_MM, CORE_REAR_Z)]
    for index in range(1, FAIRING_RAMP_SAMPLES + 1):
        fraction = index / FAIRING_RAMP_SAMPLES
        points.append((
            ROOT_FAIRING_R_IN_MM + span * fraction,
            CORE_REAR_Z + depth * _smootherstep(fraction),
        ))
    points.append((ROOT_FAIRING_R_OUT_MM, top))
    points.append((ROOT_FAIRING_R_IN_MM, top))
    points.append((ROOT_FAIRING_R_IN_MM, CORE_REAR_Z))
    angle = math.radians(angle_deg)
    plane = Plane(
        origin=(0.0, BMR_AXIS_XY[1], 0.0),
        x_dir=(sign * math.cos(angle), math.sin(angle), 0.0),
        z_dir=(math.sin(angle), -sign * math.cos(angle), 0.0),
    )
    return plane * make_face(Wire(Polyline(*points).edges()))


def _root_fairing_solid():
    """Rear-taper fill that gives the boss a G1 full-depth foundation."""
    fairing = None
    for sign in (1.0, -1.0):
        side = loft([
            _fairing_section_face(angle, sign)
            for angle in FAIRING_SECTION_ANGLES_DEG
        ])
        fairing = side if fairing is None else fairing + side
    return fairing


def plan_fillet_geometry() -> dict[str, tuple[float, float]]:
    """Tangent points and centre of the +X boss/top-edge plan blend."""
    drop = BMR_AXIS_XY[1] - CRESCENT_TOP_EDGE_Y_MM
    centre_x = math.sqrt(
        (BOSS_PLAN_R_MM + PLAN_FILLET_R_MM) ** 2
        - (PLAN_FILLET_R_MM - drop) ** 2)
    centre = (centre_x, CRESCENT_TOP_EDGE_Y_MM + PLAN_FILLET_R_MM)
    span = math.hypot(centre[0], centre[1] - BMR_AXIS_XY[1])
    scale = BOSS_PLAN_R_MM / span
    return {
        "centre": centre,
        "tangent_on_edge": (centre[0], CRESCENT_TOP_EDGE_Y_MM),
        "tangent_on_boss": (
            BMR_AXIS_XY[0] + scale * (centre[0] - BMR_AXIS_XY[0]),
            BMR_AXIS_XY[1] + scale * (centre[1] - BMR_AXIS_XY[1]),
        ),
        "corner": (
            math.sqrt(BOSS_PLAN_R_MM ** 2 - drop ** 2),
            CRESCENT_TOP_EDGE_Y_MM,
        ),
    }


def _plan_fillet_prism():
    """Full-depth plan of both blends, as pure CSG.

    This is kept separate from the tapered solid on purpose.  It is also the
    root fairing's plan mask over the blends, and a mask has to survive OCC
    booleans: a prism recovered from the tapered solid's front face carries
    the loft's edge structure and silently intersects to nothing.
    """
    geometry = plan_fillet_geometry()
    centre = geometry["centre"]
    height = THICKNESS_MM - CORE_REAR_Z
    mid_z = (CORE_REAR_Z + THICKNESS_MM) / 2.0

    # The blend lives above the released top edge, between the notch corner and
    # the tangent point on that edge.  Those four half-spaces plus the two
    # circles bound it exactly: the arc from the boss tangency down to the
    # corner is the only free boundary.
    corner_x = geometry["corner"][0]
    patch = Pos(0.0, CRESCENT_TOP_EDGE_Y_MM + 200.0, mid_z) * Box(
        400.0, 400.0, height)
    patch &= Pos((corner_x + centre[0]) / 2.0, BMR_AXIS_XY[1], mid_z) * Box(
        centre[0] - corner_x, 400.0, height)
    # Keep only the corner side of the boss-centre-to-fillet-centre ray, so the
    # patch closes exactly at the boss tangency instead of running on.
    ray_deg = math.degrees(math.atan2(
        centre[1] - BMR_AXIS_XY[1], centre[0] - BMR_AXIS_XY[0]))
    patch &= (Pos(BMR_AXIS_XY[0], BMR_AXIS_XY[1], mid_z)
              * Rot(Z=ray_deg)
              * Pos(0.0, -200.0, 0.0)
              * Box(400.0, 400.0, height))
    patch -= Pos(BMR_AXIS_XY[0], BMR_AXIS_XY[1], mid_z) * Cylinder(
        BOSS_PLAN_R_MM, height + 2.0)
    patch -= Pos(centre[0], centre[1], mid_z) * Cylinder(
        PLAN_FILLET_R_MM, height + 2.0)
    patch = patch.clean()
    return patch + mirror(patch, about=Plane.YZ)


def _plan_fillet_solid():
    """Both tangent blends, carrying the released rear taper of the arm."""
    patch = _plan_fillet_prism()
    for cutter in _crescent_taper_cutters(
            TWEETER_DROP_MM, THICKNESS_MM, CORE_REAR_Z):
        patch -= cutter
    return patch.clean()


def _plan_prism(part):
    """Prism of every acoustic-front face, used to keep additions in plan."""
    prism = None
    for face in part.faces().filter_by(Plane.XY):
        if abs(face.center().Z - THICKNESS_MM) > 1.0e-6:
            continue
        column = extrude(face, amount=STACK_DEPTH_MM + 20.0,
                         dir=(0.0, 0.0, -1.0))
        prism = column if prism is None else prism + column
    if prism is None:
        raise RuntimeError("crescent has no acoustic-front face to bound with")
    return prism


def _apply_driver_interfaces(part):
    """Two coaxial blind pockets, eight M2 bores, two declared lead outlets."""
    over = 1.0
    part -= _cylinder_at(
        BMR_AXIS_XY[0], BMR_AXIS_XY[1], TEBM_CUTOUT_D_MM / 2.0,
        FRONT_POCKET_FLOOR_Z_MM, THICKNESS_MM + over)
    part -= _cylinder_at(
        BMR_AXIS_XY[0], BMR_AXIS_XY[1], TEBM_CUTOUT_D_MM / 2.0,
        REAR_MOUNT_Z_MM - over, REAR_POCKET_ROOF_Z_MM)

    patterns = (
        (FRONT_MOUNT_CLOCK_DEG,
         THICKNESS_MM - M2_INSERT_DEPTH_MM, THICKNESS_MM),
        (REAR_MOUNT_CLOCK_DEG,
         REAR_MOUNT_Z_MM, REAR_MOUNT_Z_MM + M2_INSERT_DEPTH_MM),
    )
    radius = TEBM_MOUNT_PCD_MM / 2.0
    for clock, bore_z_min, bore_z_max in patterns:
        for index in range(TEBM_MOUNT_HOLE_COUNT):
            angle = math.radians(clock + 90.0 * index)
            part -= _cylinder_at(
                BMR_AXIS_XY[0] + radius * math.cos(angle),
                BMR_AXIS_XY[1] + radius * math.sin(angle),
                M2_INSERT_BORE_D_MM / 2.0, bore_z_min, bore_z_max)

    for outlet_z in (FRONT_OUTLET_Z_MM, REAR_OUTLET_Z_MM):
        length = POCKET_OUTLET_OUTER_R_MM - POCKET_OUTLET_INNER_R_MM
        centre_y = BMR_AXIS_XY[1] - (
            POCKET_OUTLET_INNER_R_MM + POCKET_OUTLET_OUTER_R_MM) / 2.0
        part -= (Pos(BMR_AXIS_XY[0], centre_y, outlet_z)
                 * Rot(X=90.0)
                 * Cylinder(POCKET_OUTLET_D_MM / 2.0, length))
    return part


def _reassert_mate_voids(part):
    """Re-cut the released half-lap voids after every positive addition."""
    for x in TWEETER_JOINT_X:
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2, TWEETER_CORE_BORE_TOP_Z)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_INSERT_BORE_D / 2.0,
            *TWEETER_JOINT_INSERT_BORE_Z)
    return part


def bmr_crescent():
    """Released crescent mate, coaxial BMR boss, two blind driver pockets."""
    _require_guarded_build()
    _check_released_mate()

    released = tweeter_crescent()
    # Both masks are taken before anything is fused: each one then bounds the
    # fairing over exactly one clean region, and no mask is ever recovered
    # from a face of an already-boolean'd body.
    released_plan = _plan_prism(released)
    blend_plan = _plan_fillet_prism()
    fairing = _root_fairing_solid()

    part = _fuse_required(
        released, _plan_fillet_solid(), "boss-to-top-edge tangent plan blends")
    part = _fuse_required(
        part, fairing & released_plan, "BMR boss root fairing over the arms")
    part = _fuse_required(
        part, fairing & blend_plan, "BMR boss root fairing under the blends")
    part = _fuse_required(part, _boss_solid(), "coaxial BMR boss")

    part = _apply_driver_interfaces(part)
    part = _reassert_mate_voids(part)

    part = part.clean()
    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1 or solids[0].volume <= 0.01:
        raise RuntimeError(
            "BMR crescent finalization must retain every required feature; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in part.solids()]}")
    return Part([solids[0]])


@dataclass(frozen=True)
class BmrCrescentModel:
    """Authoritative solid plus the declared-opening bookkeeping."""

    solid: object


def build_model() -> BmrCrescentModel:
    return BmrCrescentModel(solid=bmr_crescent())


def declared_openings() -> list[dict]:
    """Every intentional break through the exterior, with its authority."""
    return [
        {
            "name": "front_driver_pocket_mouth",
            "kind": "driver_pocket",
            "face": "acoustic_front_z_18p3",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(BMR_AXIS_XY),
            "z_span_mm": [FRONT_POCKET_FLOOR_Z_MM, THICKNESS_MM],
        },
        {
            "name": "rear_driver_pocket_mouth",
            "kind": "driver_pocket",
            "face": "bmr_rear_land_z_minus_31p9",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(BMR_AXIS_XY),
            "z_span_mm": [REAR_MOUNT_Z_MM, REAR_POCKET_ROOF_Z_MM],
        },
        {
            "name": "front_driver_lead_outlet",
            "kind": "cable_outlet",
            "diameter_mm": POCKET_OUTLET_D_MM,
            "axis": "-Y",
            "z_mm": FRONT_OUTLET_Z_MM,
            "breakout_y_mm": (
                BMR_AXIS_XY[1] - boss_radius_at(FRONT_OUTLET_Z_MM)),
        },
        {
            "name": "rear_driver_lead_outlet",
            "kind": "cable_outlet",
            "diameter_mm": POCKET_OUTLET_D_MM,
            "axis": "-Y",
            "z_mm": REAR_OUTLET_Z_MM,
            "breakout_y_mm": (
                BMR_AXIS_XY[1] - boss_radius_at(REAR_OUTLET_Z_MM)),
        },
        {
            "name": "um_half_lap_clearance_passages",
            "kind": "released_mate",
            "diameter_mm": TWEETER_JOINT_HOLE_D,
            "count": 2,
            "centres_xy_mm": [[x, TWEETER_JOINT_Y] for x in TWEETER_JOINT_X],
        },
        {
            "name": "um_half_lap_insert_receivers",
            "kind": "released_mate",
            "diameter_mm": TWEETER_JOINT_INSERT_BORE_D,
            "count": 2,
            "blind": True,
            "front_floor_mm": insert_front_floor_mm(),
        },
        {
            "name": "m2_driver_insert_bores",
            "kind": "driver_mount",
            "diameter_mm": M2_INSERT_BORE_D_MM,
            "depth_mm": M2_INSERT_DEPTH_MM,
            "count": 2 * TEBM_MOUNT_HOLE_COUNT,
            "blind": True,
            "pcd_mm": TEBM_MOUNT_PCD_MM,
        },
        {
            "name": "inherited_m4_tweeter_clamp_holes",
            "kind": "released_inherited",
            "diameter_mm": TWEETER_HOLE_D_MM,
            "count": len(TWEETER_HOLE_XY),
            "centres_xy_mm": [
                [x, y - TWEETER_DROP_MM] for x, y in TWEETER_HOLE_XY],
            "note": (
                "released ND25FW-4 faceplate clamp passages; this variant has "
                "no clamped tweeter, so they carry no fastener and are "
                "retained only to keep the released silhouette exact"
            ),
        },
    ]


def design_facts() -> dict:
    """Envelope, mate coordinates, pocket outlets and candidate flags."""
    fillet = plan_fillet_geometry()
    return {
        "part": PART_NAME,
        "release_variant": RELEASE_VARIANT,
        "print_orientation": PRINT_ORIENTATION,
        "release_authorized": RELEASE_AUTHORIZED,
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
        "status": "candidate_not_release_authorized",
        "counts_against_release_inventory": False,
        "magnet_count": MAGNET_COUNT,
        "magnet_note": (
            "none in v1, exactly like the released ND25FW-4 crescent"),
        "driver": {
            "model": "Tectonic TEBM35C10-4",
            "count": 2,
            "arrangement": "coaxial_back_to_back_one_axis",
            "axis_xy_mm": list(BMR_AXIS_XY),
            "axis_authority": (
                "released dropped scallop centre, i.e. the ND25FW-4 "
                "face-to-face acoustic axis"),
            "front_driver_faces": "+z",
            "rear_driver_faces": "-z",
            "depth_mm": TEBM_DEPTH_MM,
            "max_flange_d_mm": TEBM_MAX_D_MM,
            "basket_d_mm": TEBM_BASKET_D_MM,
            "cutout_d_mm": TEBM_CUTOUT_D_MM,
            "land_d_mm": TEBM_LAND_D_MM,
            "mount_pcd_mm": TEBM_MOUNT_PCD_MM,
            "mount_hole_count": TEBM_MOUNT_HOLE_COUNT,
            "mount_clock_deg": {
                "front": FRONT_MOUNT_CLOCK_DEG,
                "rear": REAR_MOUNT_CLOCK_DEG,
            },
            "pair_mass_g": 2.0 * TEBM_MASS_G,
        },
        "depth_stack": {
            "acoustic_front_z_mm": FRONT_MOUNT_Z_MM,
            "front_pocket_floor_z_mm": FRONT_POCKET_FLOOR_Z_MM,
            "partition_z_span_mm": [
                REAR_POCKET_ROOF_Z_MM, FRONT_POCKET_FLOOR_Z_MM],
            "partition_thickness_mm": PARTITION_THICKNESS_MM,
            "partition_rule": (
                "two independent 1.20 mm blind walls back to back, so each "
                "driver keeps the vase's qualified wall and the two rear "
                "volumes remain separate chambers"),
            "rear_pocket_roof_z_mm": REAR_POCKET_ROOF_Z_MM,
            "rear_mount_z_mm": REAR_MOUNT_Z_MM,
            "clear_pocket_depth_mm": T_CLEAR_POCKET_DEPTH_MM,
            "stack_depth_mm": STACK_DEPTH_MM,
            "rear_protrusion_behind_core_rear_mm": REAR_PROTRUSION_MM,
        },
        "boss": {
            "plan_radius_mm": BOSS_PLAN_R_MM,
            "released_scallop_radius_mm": SCALLOP_R_MM,
            "scallop_overlap_mm": BOSS_SCALLOP_OVERLAP_MM,
            "rear_land_radius_mm": BOSS_REAR_R_MM,
            "flare_law": "quintic_smootherstep_zero_slope_both_ends",
            "root_fairing_r_in_mm": ROOT_FAIRING_R_IN_MM,
            "root_fairing_r_out_mm": ROOT_FAIRING_R_OUT_MM,
            "um_ear_footprint_r_mm": UM_EAR_FOOTPRINT_R_MM,
            "root_fairing_mate_margin_mm": ROOT_FAIRING_MATE_MARGIN_MM,
            "plan_fillet_radius_mm": PLAN_FILLET_R_MM,
            "plan_fillet_tangent_on_edge_xy_mm": list(
                fillet["tangent_on_edge"]),
            "plan_fillet_tangent_on_boss_xy_mm": list(
                fillet["tangent_on_boss"]),
            "plan_growth_note": (
                "growth is confined to the released open scallop plus the two "
                "tangent corner blends; no flank moves outward, so the wing "
                "clearance envelope is unchanged"),
        },
        "mate": {
            "interface": "obiwan_um_collar_half_laps",
            "identical_to": "addon_tweeter_crescent",
            "joint_x_mm": list(TWEETER_JOINT_X),
            "joint_y_mm": TWEETER_JOINT_Y,
            "ear_boss_d_mm": TWEETER_JOINT_FUNCTIONAL_BOSS_D,
            "ear_z_span_mm": list(TWEETER_ADDON_JOINT_Z),
            "core_ear_z_span_mm": list(TWEETER_CORE_JOINT_Z),
            "axial_gap_mm": axial_gap_mm(),
            "insert_receiver_d_mm": TWEETER_JOINT_INSERT_BORE_D,
            "insert_receiver_depth_mm": TWEETER_JOINT_INSERT_DEPTH_MM,
            "insert_receiver_z_span_mm": list(TWEETER_JOINT_INSERT_BORE_Z),
            "acoustic_front_floor_mm": insert_front_floor_mm(),
            "clearance_bore_d_mm": TWEETER_JOINT_HOLE_D,
            "clearance_bore_owner": "um_carrier",
        },
        "cable": {
            "free_t_cable_centreline_z_mm": TS_FREE_CABLE_Z,
            "printed_duct": False,
            "note": (
                "the T route stays free behind the part exactly as on the "
                "released crescent; both lead outlets open on the -Y meridian "
                "behind the core rear plane, on the side the free cable "
                "arrives from"),
            "outlet_inset_from_blind_wall_mm": POCKET_OUTLET_INSET_MM,
            "outlet_d_mm": POCKET_OUTLET_D_MM,
            "front_outlet_z_mm": FRONT_OUTLET_Z_MM,
            "rear_outlet_z_mm": REAR_OUTLET_Z_MM,
        },
        "declared_openings": declared_openings(),
    }


def obiwan_bmr_attachments():
    _require_guarded_build()
    return {"addon_bmr_crescent": bmr_crescent()}


def gen_step():
    _require_guarded_build()
    return ordered_labeled_compound(
        obiwan_bmr_attachments(),
        label="lx521_obiwan_r6f_bmr_crescent_candidate")
