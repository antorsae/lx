"""Candidate Obi-Wan BMR pod for two coaxial back-to-back TEBM35C10-4 BMRs.

This is an alternative to the released ND25FW-4 tweeter crescent, not a
replacement for it.  It presents the *identical* half-lap interface to an
unmodified Obi-Wan UM collar (x=+/-24, y=421.5, complete front local-D9.8
ears, standalone blind D4.6 x 4.0 heat-set receivers, 1.9 mm acoustic-front
floors, 0.20 mm axial gap), so the two parts are mutually swappable without
touching the UM print.

Where the released crescent is a full acoustic silhouette carrying a
face-to-face Dayton pair, this part keeps only what the two BMRs and that
one mate actually need:

* a D66 pod on the released tweeter acoustic axis, carrying the front driver
  on the shared z=18.3 plane and the rear driver on z=-31.9 facing -z, with
  each driver keeping the vase's qualified 1.20 mm blind wall (a 2.40 mm
  back-to-back partition and two separate rear chambers);
* the two UM half-lap ears; and
* two drafted struts, nothing else.

The released crescent's arm silhouette, its rear taper, the root fairing over
that taper, the boss/top-edge plan blends and the inherited M4 ND25FW-4
faceplate clamp passages are all gone: this variant clamps no tweeter, so
those four holes carried no fastener and only existed to keep a silhouette
this part no longer has.

Why D66 and not more
--------------------
The pod's outer wall is exactly the driver land.  Both mounting faces must
carry the vase's qualified D66 land, and the part prints front-face-down, so
the plan may never grow rearward.  A radius below 33 would lose land at one
face and a radius above 33 at the front would have to come back down to 33 at
the rear, which is the one direction this print orientation cannot take.  A
straight D66 cylinder is therefore the unique minimum, and it is also how the
qualified ``proud.vase_tebm35c10_4`` treats its own drivers, where the D66
land *is* the exterior surface around each one.  It still leaves 11.537 mm of
wall outside each pocket and 7.270 mm outside each M2 insert bore.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=18.3 is the acoustic front
and z=6.8 is the Obi-Wan core rear plane.  The pod grows only rearward, to
z=-31.9.  In plan the whole part fits inside the released crescent's own
footprint -- the pod sits 6.25 mm clear inside the released open scallop --
so neither wing family nor the UM collar sees anything new.

Candidate status
----------------
Nothing here is release-authorized.  ``RELEASE_AUTHORIZED`` is false and
``PHYSICAL_MEASURE_REQUIRED`` is true: the driver envelope, the back-to-back
partition, the two struts and the two-screw joint demand under roughly twice
the released crescent's hanging mass all need physical qualification before
this part is printed for use.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from build123d import (
    Face,
    Part,
    Polyline,
    Pos,
    Rot,
    Cylinder,
    Wire,
    extrude,
)
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ..assembly import ordered_labeled_compound
from ..base import (
    CRESCENT_SCALLOP_CY,
    THICKNESS_MM,
    UM_CUTOUT,
)
from ..cables import ROUTING_PROFILE
from ..proud.b import TWEETER_DROP_MM
from .attachments import _cylinder_at, _fuse_required
from .carriers import (
    CORE_REAR_Z,
    TWEETER_ADDON_JOINT_Z,
    TWEETER_CORE_JOINT_Z,
    TWEETER_JOINT_FUNCTIONAL_BOSS_D,
    TWEETER_JOINT_HOLE_D,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_INSERT_BORE_Z,
    TWEETER_JOINT_INSERT_DEPTH_MM,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
    UM_CORE_R,
    _apply_complete_um_tweeter_joint,
    _require_guarded_build,
)
from .route import TS_FREE_CABLE_Z


if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "the Obi-Wan BMR pod requires LX_ROUTING_PROFILE=obiwan (R6F)")

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
# The scallop that carried the released face-to-face Dayton pair is the
# D78.50 circle about the dropped scallop centre; that centre is the released
# tweeter acoustic axis and is where the coaxial BMR pair goes.  The pod is
# much smaller than the scallop now, but the scallop circle remains the
# authority that fixes the axis, so its drawing check stays.
BMR_AXIS_XY = (0.0, CRESCENT_SCALLOP_CY - TWEETER_DROP_MM)
SCALLOP_R_MM = 78.50 / 2.0
# Drawing vertex at the bottom of the scallop, in the un-dropped frame.  The
# released outline reaches it with a Bezier, so this is the authority the
# radius is checked against rather than sampled geometry.
SCALLOP_BOTTOM_DRAWING_Y_MM = 443.804

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


# --- pod envelope: the D66 land is the exterior ------------------------------
# See the module docstring.  Two D66 lands plus a front-face-down print with a
# never-growing plan admit exactly one profile, the straight D66 cylinder.
POD_OUTER_R_MM = TEBM_LAND_R_MM
POD_OUTER_D_MM = TEBM_LAND_D_MM
POD_WALL_OVER_POCKET_MM = round(POD_OUTER_R_MM - TEBM_CUTOUT_D_MM / 2.0, 9)
POD_WALL_OVER_INSERT_MM = round(
    POD_OUTER_R_MM - (TEBM_MOUNT_PCD_MM + M2_INSERT_BORE_D_MM) / 2.0, 9)
POD_LAND_MARGIN_OVER_FLANGE_MM = round(
    POD_OUTER_R_MM - TEBM_MAX_D_MM / 2.0, 9)
# How far inside the released open scallop the whole pod now sits.
POD_CLEARANCE_INSIDE_SCALLOP_MM = round(SCALLOP_R_MM - POD_OUTER_R_MM, 9)


# --- half-lap struts --------------------------------------------------------
# The two struts are the only structure between the pod and the mate.  They
# occupy the released crescent's own plate band, z=6.8..18.3, so nothing but
# the pod itself ever reaches behind the core rear plane, and the free T cable
# keeps the same open corridor it has on the released crescent.
ARM_REAR_Z_MM = CORE_REAR_Z
ARM_DEPTH_MM = round(THICKNESS_MM - ARM_REAR_Z_MM, 9)            # 11.5
EAR_THICKNESS_MM = round(
    TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0], 9)      # 5.9
# The half-lap's own governing printed section: the complete D9.8 functional
# boss less the D4.6 heat-set receiver bored through it, over the ear's own
# axial thickness.  That joint is already qualified, so the struts are sized
# to stay clear of becoming a new weakest link rather than to some new rule.
EAR_NET_LIGAMENT_MM = round(
    TWEETER_JOINT_FUNCTIONAL_BOSS_D - TWEETER_JOINT_INSERT_BORE_D, 9)
EAR_NET_SECTION_MM2 = round(EAR_NET_LIGAMENT_MM * EAR_THICKNESS_MM, 9)

ARM_WIDTH_MM = 8.0
# A root fillet equal to the strut width keeps the concave strut-to-pod corner
# from being the stress riser the strut section was chosen to avoid.
ARM_ROOT_FILLET_R_MM = ARM_WIDTH_MM
# Rearward draft.  Printed front-face-down every plan grows toward the bed, so
# this is pure positive draft, and it is bounded by keeping the strut's rear
# width above the ear ligament it feeds.
ARM_DRAFT_DEG = 5.0
ARM_DRAFT_SLOPE = math.tan(math.radians(ARM_DRAFT_DEG))
ARM_REAR_WIDTH_MM = round(
    ARM_WIDTH_MM - 2.0 * ARM_DEPTH_MM * ARM_DRAFT_SLOPE, 9)
# The strut crosses the UM ear's receiver notch, so over that footprint it is
# only the ear's own 5.9 mm thick.  This is the strut's smallest section and
# the number that has to beat ``EAR_NET_SECTION_MM2``.
ARM_MIN_SECTION_MM2 = round(
    (ARM_WIDTH_MM - EAR_THICKNESS_MM * ARM_DRAFT_SLOPE) * EAR_THICKNESS_MM, 9)
# Buffer resolution and the tolerance the closed plan is decimated to.  A
# morphological closing leaves sub-micron segments at every arc handover.
# Undecimated they become the zero-area facets the release mesh contract
# rejects outright, and OCC's draft offset refuses to run on several of the
# denser plans altogether.  96 quadrant segments hold a 33 mm arc to under
# 1.5 um and decimating at 2 um drops the slivers without moving any surface
# a printer could resolve; that pair meshes with no degenerate or collinear
# facet at all.  Changing either number is a geometry change and has to be
# re-checked against the mesh contract, not assumed.
ARM_PLAN_RESOLUTION = 96
ARM_PLAN_SIMPLIFY_MM = 0.002
# The released crescent takes exactly this much clearance around the UM core
# ring; the struts are held to the same figure, and the ears -- which are
# allowed inside it -- are restored afterwards, exactly as on the release.
UM_COLLAR_CLEAR_MM = 0.20


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
# from and the side the mate is on, and both sit behind the core rear plane
# where that cable already runs.  The cutter starts inside the pocket and ends
# clear of the pod wall; nothing is added outside the wall, because in this
# silhouette a printed duct would be the only body able to pinch the free
# cable in the corridor the struts deliberately leave open.
POCKET_OUTLET_INNER_R_MM = 18.0
POCKET_OUTLET_OUTER_R_MM = round(POD_OUTER_R_MM + 2.0, 9)

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


def axial_gap_mm() -> float:
    return TWEETER_ADDON_JOINT_Z[0] - TWEETER_CORE_JOINT_Z[1]


def insert_front_floor_mm() -> float:
    return THICKNESS_MM - TWEETER_JOINT_INSERT_BORE_Z[1]


def pod_radius_at(z: float) -> float:
    """Pod outer radius at one Z.  It is the D66 land at every Z."""
    if not (REAR_MOUNT_Z_MM - 1.0e-9 <= z <= THICKNESS_MM + 1.0e-9):
        raise ValueError(f"z={z} is outside the BMR depth stack")
    return POD_OUTER_R_MM


def arm_plan():
    """Plan of both struts at the acoustic front, blended into the pod.

    The union of the pod disc, one D8 beam per ear and the two complete D9.8
    ear bosses is closed morphologically by ``ARM_ROOT_FILLET_R_MM``.  A
    closing puts an exact fillet of that radius in every concave corner --
    the strut-to-pod roots and the strut-to-boss shoulders -- and moves
    nothing convex, so the struts stay tangent to the pod wall without any
    hand-placed blend geometry.
    """
    pieces = [Point(*BMR_AXIS_XY).buffer(
        POD_OUTER_R_MM, resolution=ARM_PLAN_RESOLUTION)]
    for x in TWEETER_JOINT_X:
        ear = (x, TWEETER_JOINT_Y)
        pieces.append(LineString([ear, BMR_AXIS_XY]).buffer(
            ARM_WIDTH_MM / 2.0, resolution=ARM_PLAN_RESOLUTION))
        pieces.append(Point(*ear).buffer(
            TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0,
            resolution=ARM_PLAN_RESOLUTION))
    blended = unary_union(pieces)
    plan = (blended
            .buffer(ARM_ROOT_FILLET_R_MM,
                    resolution=ARM_PLAN_RESOLUTION, join_style=1)
            .buffer(-ARM_ROOT_FILLET_R_MM,
                    resolution=ARM_PLAN_RESOLUTION, join_style=1))
    plan = plan.simplify(ARM_PLAN_SIMPLIFY_MM, preserve_topology=True)
    if plan.geom_type != "Polygon" or plan.interiors:
        raise RuntimeError(
            "the strut plan must close to one simple region; the fillet "
            "radius has bridged the central cable mouth")
    return plan


def arm_collar_clearance_mm() -> float:
    """Closest approach of the strut plan to the UM core ring."""
    return arm_plan().distance(Point(*UM_CUTOUT[:2])) - UM_CORE_R


def _plan_face(polygon):
    """One build123d Face from a simple Shapely polygon, +Z normal."""
    polygon = orient(polygon, sign=1.0)
    return Face(Wire(Polyline(*[
        (float(x), float(y)) for x, y in polygon.exterior.coords
    ]).edges()))


def _pod_solid():
    """The D66 land swept over the whole coaxial stack."""
    return _cylinder_at(
        BMR_AXIS_XY[0], BMR_AXIS_XY[1], POD_OUTER_R_MM,
        REAR_MOUNT_Z_MM, THICKNESS_MM)


def _arm_solid():
    """Both drafted struts, from the acoustic front back to z=6.8.

    The strut plan deliberately still contains the pod disc: drafted rearward
    that part of the extrusion is a cone strictly inside the pod cylinder, so
    the two bodies fuse on a real volume, and the only place the strut's outer
    surface touches the pod's is the single circle on the front face.
    """
    clearance = arm_collar_clearance_mm()
    if clearance < UM_COLLAR_CLEAR_MM - 1.0e-9:
        raise RuntimeError(
            "the struts must keep the released crescent's own "
            f"{UM_COLLAR_CLEAR_MM} mm clearance around the UM core ring; "
            f"this plan leaves {clearance:.3f} mm")
    face = Pos(0.0, 0.0, THICKNESS_MM) * _plan_face(arm_plan())
    return extrude(face, amount=-ARM_DEPTH_MM, taper=ARM_DRAFT_DEG)


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


def bmr_crescent():
    """Minimal BMR pod on the released UM half-lap mate."""
    _require_guarded_build()
    _check_released_mate()

    part = _fuse_required(
        _pod_solid(), _arm_solid(), "half-lap struts onto the BMR pod")
    # The complete standalone ears, their receiver notch for the opposing UM
    # halves, the rear-driven D3.4 passages and the blind D4.6 receivers, all
    # from the released joint authority rather than restated here.
    part = _apply_complete_um_tweeter_joint(part, "tweeter")
    part = _apply_driver_interfaces(part)

    part = part.clean()
    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1 or solids[0].volume <= 0.01:
        raise RuntimeError(
            "BMR pod finalization must retain every required feature; "
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
            "breakout_y_mm": BMR_AXIS_XY[1] - pod_radius_at(FRONT_OUTLET_Z_MM),
        },
        {
            "name": "rear_driver_lead_outlet",
            "kind": "cable_outlet",
            "diameter_mm": POCKET_OUTLET_D_MM,
            "axis": "-Y",
            "z_mm": REAR_OUTLET_Z_MM,
            "breakout_y_mm": BMR_AXIS_XY[1] - pod_radius_at(REAR_OUTLET_Z_MM),
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
    ]


def design_facts() -> dict:
    """Envelope, mate coordinates, pocket outlets and candidate flags."""
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
        "silhouette": {
            "shape": "minimal_pod_two_struts_two_ears",
            "inherits_released_crescent_outline": False,
            "removed_from_the_first_candidate": [
                "released crescent arm silhouette",
                "released rear taper and its root fairing",
                "boss-to-top-edge tangent plan blends",
                "inherited M4 ND25FW-4 faceplate clamp passages",
            ],
            "removal_note": (
                "this variant clamps no tweeter, so the four inherited M4 "
                "passages carried no fastener; they existed only to keep a "
                "released silhouette this part no longer has"),
        },
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
        "pod": {
            "profile": "straight_cylinder",
            "outer_d_mm": POD_OUTER_D_MM,
            "outer_r_mm": POD_OUTER_R_MM,
            "outer_wall_authority": "tebm35c10_4_driver_land_d66",
            "outer_wall_basis": (
                "both mounting faces must carry the vase's qualified D66 "
                "land and the part prints front-face-down, so the plan may "
                "never grow rearward; a straight D66 cylinder is the unique "
                "minimum, and it is also how the qualified vase treats its "
                "own drivers, where the D66 land is the exterior surface"),
            "wall_outside_pocket_mm": POD_WALL_OVER_POCKET_MM,
            "wall_outside_m2_insert_bore_mm": POD_WALL_OVER_INSERT_MM,
            "land_margin_over_max_flange_mm": POD_LAND_MARGIN_OVER_FLANGE_MM,
            "released_scallop_radius_mm": SCALLOP_R_MM,
            "clearance_inside_released_scallop_mm": (
                POD_CLEARANCE_INSIDE_SCALLOP_MM),
        },
        "struts": {
            "count": 2,
            "plan": "beam_from_pod_wall_to_each_half_lap_boss",
            "front_width_mm": ARM_WIDTH_MM,
            "rear_width_mm": ARM_REAR_WIDTH_MM,
            "root_fillet_r_mm": ARM_ROOT_FILLET_R_MM,
            "root_fillet_rule": "equal to the strut width",
            "draft_deg": ARM_DRAFT_DEG,
            "draft_rule": (
                "printed front-face-down every plan grows toward the bed, so "
                "the rearward draft is pure positive draft; it is bounded by "
                "keeping the rear width above the ear ligament it feeds"),
            "z_span_mm": [ARM_REAR_Z_MM, THICKNESS_MM],
            "depth_mm": ARM_DEPTH_MM,
            "min_section_mm2": ARM_MIN_SECTION_MM2,
            "half_lap_net_section_mm2": EAR_NET_SECTION_MM2,
            "section_ratio": round(
                ARM_MIN_SECTION_MM2 / EAR_NET_SECTION_MM2, 6),
            "section_rule": (
                "the strut's smallest section, where it crosses the UM ear "
                "receiver notch, stays above the half-lap's own net ligament "
                "section, so the already-qualified joint remains governing"),
            "um_core_ring_clearance_mm": round(arm_collar_clearance_mm(), 6),
            "um_core_ring_clearance_rule": (
                "the released crescent's own 0.20 mm clearance around the UM "
                "core ring; only the ears are allowed inside it"),
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
            "scope_note": (
                "the mount is identical to the released crescent's; the rest "
                "of the part is not, so identity is asserted over the two ear "
                "footprints and by assembling against the staged UM collar, "
                "not by differencing whole silhouettes"),
        },
        "cable": {
            "free_t_cable_centreline_z_mm": TS_FREE_CABLE_Z,
            "printed_duct": False,
            "note": (
                "the T route stays free behind the part exactly as on the "
                "released crescent; both lead outlets open on the -Y meridian "
                "behind the core rear plane, on the side the free cable "
                "arrives from and the side the mate is on.  Dropping the arms "
                "opens that whole quadrant, so a printed duct there would be "
                "the only body able to pinch the free cable and none is cut"),
            "outlet_inset_from_blind_wall_mm": POCKET_OUTLET_INSET_MM,
            "outlet_d_mm": POCKET_OUTLET_D_MM,
            "front_outlet_z_mm": FRONT_OUTLET_Z_MM,
            "rear_outlet_z_mm": REAR_OUTLET_Z_MM,
            "outlet_breakout_y_mm": BMR_AXIS_XY[1] - POD_OUTER_R_MM,
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
