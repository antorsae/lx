"""Shared machinery for the candidate Obi-Wan TEBM35C10-4 BMR pods.

Two candidate parts hang two Tectonic TEBM35C10-4 BMRs off the released
Obi-Wan UM half-lap: ``bmr_crescent`` stacks them coaxially back to back, and
``bmr_crescent_opposed`` stands them side by side on the qualified vase's own
opposed layout.  Everything the two share lives here, so there is exactly one
definition of each:

* the mirrored ``proud.vase_tebm35c10_4`` driver authority, which cannot be
  imported beside an obiwan-profile part;
* the D66 land, its two magnet flats and the pocket/insert datums cut into it;
* the drop limit -- how close the mount land may come to the UM collar, and
  which released constraint stops it;
* the flush junction skirt between that land and the collar, its wing keep-out,
  its ownership relief and the ear-to-pod section it has to hold;
* the one hidden Ø6.00 cable entry on the mate face, aligned with the free T
  cable's own tangent, and the stadium collar that carries it; and
* the vase's captive D5x2 side magnets, applied through ``magnets.py`` at the
  vase's own land-local stations.

What the two variants do *not* share is the pod body itself: how many lands
there are, where they sit in Z, which way each driver faces, and how the leads
get from the entry to the second driver.  Each variant owns that and nothing
else.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=18.3 is the acoustic front
and z=6.8 is the Obi-Wan core rear plane.  Both variants print front-face-down,
so the exterior plan may never grow rearward.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from build123d import (
    Box,
    Cylinder,
    Pos,
    Rot,
)
from shapely.geometry import LineString, Point
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

from ..base import (
    CRESCENT_SCALLOP_CY,
    THICKNESS_MM,
    UM_CUTOUT,
)
from ..cables import ROUTING_PROFILE
from ..magnets import CaptiveMagnetTools, apply_wall_cavity
from ..proud.b import STANDARD_MAGNET_Z_MM, TWEETER_DROP_MM
from .attachments import _cylinder_at
from .carriers import (
    CORE_REAR_Z,
    JUNCTION_WEB_Z,
    T_UM_CABLE_MOUTH_HALF_WIDTH,
    TWEETER_ADDON_JOINT_Z,
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
    junction_closure_polygons,
)
from . import route
from .route import TS_CABLE_D_EST, TS_DUCT_D, TS_FREE_CABLE_Z


if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "the Obi-Wan BMR pods require LX_ROUTING_PROFILE=obiwan (R6F)")

PRINT_ORIENTATION = "front-face-down"

# Neither variant has ever been printed, fitted or loaded.  Both flags stay
# set on both parts until their qualification tables are closed out.
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True


# --- TEBM35C10-4 driver authority -------------------------------------------
# ``proud.vase_tebm35c10_4`` is the driver-pocket and captive-magnet authority
# for this family, but it cannot be imported here: it reaches
# ``proud.b2_split``, which refuses to load unless LX_ROUTING_PROFILE=proud,
# while these parts only build under the obiwan profile.  The two profiles are
# mutually exclusive in one process, so the vase's primitives are mirrored
# below and ``VASE_AUTHORITY`` binds each one to its vase name.
# ``tests/test_bmr_crescent.py`` evaluates the real vase module in a
# proud-profile subprocess and asserts exact equality for every entry, so a
# drift in the vase fails these parts rather than silently diverging.
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

# The vase's opposed pitch: half a flange plus half a basket, because each
# driver's basket crosses the other's mounting face, plus 0.50 mm.
BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM = 0.50
PAIR_AXIS_PITCH_MM = (
    TEBM_MAX_D_MM / 2.0
    + TEBM_BASKET_D_MM / 2.0
    + BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM
)

# The vase's two captive side-magnet flats on each D66 land.  A 0.10 mm
# straight-face margin beyond the captive helper's exact 6.40 mm qualified
# land gives each magnet a real planar interface instead of a face that only
# touches the circular silhouette at the land corners.
T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM = 3.20
T_MAGNET_FLAT_EDGE_MARGIN_MM = 0.10
T_MAGNET_FLAT_HALF_HEIGHT_MM = (
    T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM + T_MAGNET_FLAT_EDGE_MARGIN_MM
)
T_MAGNET_FACE_X_MM = math.sqrt(
    TEBM_LAND_R_MM ** 2 - T_MAGNET_FLAT_HALF_HEIGHT_MM ** 2
)
T_MAGNET_TOTAL = 4

# Everything below is derived from the mirrored primitives, never restated.
# Rounding to 9 places keeps binary dust out of the datums the exporter
# publishes and the test compares; every value is exact at that precision.
T_CLEAR_POCKET_DEPTH_MM = round(
    TEBM_DEPTH_MM - T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_T_MOUNT_Z_MM = round(THICKNESS_MM - TEBM_DEPTH_MM, 9)
LOWER_T_POCKET_REAR_Z_MM = round(
    REAR_T_MOUNT_Z_MM + T_BLIND_BACK_WALL_THICKNESS_MM, 9)
UPPER_T_POCKET_FRONT_Z_MM = round(
    THICKNESS_MM - T_BLIND_BACK_WALL_THICKNESS_MM, 9)

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
    "UPPER_T_POCKET_FRONT_Z_MM": UPPER_T_POCKET_FRONT_Z_MM,
    "BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM": (
        BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM),
    "PAIR_AXIS_PITCH_MM": PAIR_AXIS_PITCH_MM,
    "T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM": (
        T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM),
    "T_MAGNET_FLAT_EDGE_MARGIN_MM": T_MAGNET_FLAT_EDGE_MARGIN_MM,
    "T_MAGNET_FLAT_HALF_HEIGHT_MM": T_MAGNET_FLAT_HALF_HEIGHT_MM,
    "T_MAGNET_FACE_X_MM": T_MAGNET_FACE_X_MM,
    "T_MAGNET_TOTAL": T_MAGNET_TOTAL,
}


# --- pod envelope: the D66 land is the exterior ------------------------------
# Both mounting faces must carry the vase's qualified D66 land and the parts
# print front-face-down with a plan that may never grow rearward, so a straight
# D66 cylinder is the unique minimum -- which is also how the qualified vase
# treats its own drivers, where the D66 land *is* the exterior surface around
# each one.  The vase's two magnet flats come with the land: they are what
# makes each captive station a plane rather than a tangent to a circle.
POD_OUTER_R_MM = TEBM_LAND_R_MM
POD_OUTER_D_MM = TEBM_LAND_D_MM
POD_FLAT_HALF_WIDTH_MM = T_MAGNET_FACE_X_MM
POD_PLAN_WIDTH_MM = round(2.0 * POD_FLAT_HALF_WIDTH_MM, 9)
POD_FLAT_DEPTH_MM = round(POD_OUTER_R_MM - POD_FLAT_HALF_WIDTH_MM, 9)
POD_WALL_OVER_POCKET_MM = round(POD_OUTER_R_MM - TEBM_CUTOUT_D_MM / 2.0, 9)
POD_WALL_OVER_INSERT_MM = round(
    POD_OUTER_R_MM - (TEBM_MOUNT_PCD_MM + M2_INSERT_BORE_D_MM) / 2.0, 9)
POD_LAND_MARGIN_OVER_FLANGE_MM = round(
    POD_OUTER_R_MM - TEBM_MAX_D_MM / 2.0, 9)
POD_FLAT_MARGIN_OVER_FLANGE_MM = round(
    POD_FLAT_HALF_WIDTH_MM - TEBM_MAX_D_MM / 2.0, 9)


# --- how far the mount land may drop toward the UM ---------------------------
# The released crescent clears the UM's native R51.7 core ring by 0.20 mm and
# recuts itself on that circle; that is the mate gap for every face these parts
# present to the UM, including the skirt's own.
UM_MATE_GAP_MM = 0.20
UM_MATE_R_MM = round(UM_CORE_R + UM_MATE_GAP_MM, 9)

# The opposing UM half-lap's receiver notch is the complete D9.8 functional
# ear grown by the released 0.10 mm joint clearance.  Its widest point is that
# boss circle, and it is cut out of these parts over z=6.7..12.4 while the D66
# land runs the full depth -- so the land wall must stay outside it or the plan
# grows rearward at z=6.7.  The wall left between the two is held to the
# vase's own qualified 1.20 mm minimum.
EAR_NOTCH_R_MM = round(
    TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR, 9)
EAR_NOTCH_LIGAMENT_MM = T_BLIND_BACK_WALL_THICKNESS_MM

AXIS_Y_LIMIT_FROM_UM_RING_MM = round(
    UM_CUTOUT[1] + UM_MATE_R_MM + POD_OUTER_R_MM, 9)
AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM = round(
    TWEETER_JOINT_Y + math.sqrt(
        (POD_OUTER_R_MM + EAR_NOTCH_R_MM + EAR_NOTCH_LIGAMENT_MM) ** 2
        - TWEETER_JOINT_X[1] ** 2), 9)

# The one land both variants share: the one that mounts on the collar.
MOUNT_AXIS_XY = (0.0, max(AXIS_Y_LIMIT_FROM_UM_RING_MM,
                          AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM))
AXIS_GOVERNING_CONSTRAINT = (
    "um_half_lap_receiver_notch"
    if AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM >= AXIS_Y_LIMIT_FROM_UM_RING_MM
    else "um_core_ring")

# The released ND25FW-4 acoustic axis, kept only as the datum these parts moved
# away from.  The scallop that carried the released face-to-face Dayton pair
# is the D78.50 circle about the dropped scallop centre; its drawing check
# stays because it is still the authority for that released axis.
RELEASED_AXIS_XY = (0.0, round(CRESCENT_SCALLOP_CY - TWEETER_DROP_MM, 9))
SCALLOP_R_MM = 78.50 / 2.0
# Drawing vertex at the bottom of the scallop, in the un-dropped frame.  The
# released outline reaches it with a Bezier, so this is the authority the
# radius is checked against rather than sampled geometry.
SCALLOP_BOTTOM_DRAWING_Y_MM = 443.804

if abs((CRESCENT_SCALLOP_CY - SCALLOP_BOTTOM_DRAWING_Y_MM)
       - SCALLOP_R_MM) > 0.01:
    raise RuntimeError(
        "released scallop radius drifted from the D78.50 drawing circle")

POD_DROP_MM = round(RELEASED_AXIS_XY[1] - MOUNT_AXIS_XY[1], 9)
UM_AXIS_SPACING_MM = round(MOUNT_AXIS_XY[1] - UM_CUTOUT[1], 9)
RELEASED_UM_AXIS_SPACING_MM = round(RELEASED_AXIS_XY[1] - UM_CUTOUT[1], 9)
POD_WALL_OFF_UM_RING_MM = round(
    MOUNT_AXIS_XY[1] - POD_OUTER_R_MM - (UM_CUTOUT[1] + UM_CORE_R), 9)
POD_WALL_OFF_EAR_NOTCH_MM = round(
    math.hypot(TWEETER_JOINT_X[1], MOUNT_AXIS_XY[1] - TWEETER_JOINT_Y)
    - POD_OUTER_R_MM - EAR_NOTCH_R_MM, 9)


# --- junction skirt ----------------------------------------------------------
# The skirt lives in the released crescent's own plate band and in the
# released closure web's own Z span; they are the same span, and the identity
# is asserted rather than assumed.
SKIRT_Z = JUNCTION_WEB_Z
if tuple(SKIRT_Z) != (CORE_REAR_Z, THICKNESS_MM):
    raise RuntimeError("the released T--UM closure web Z span moved")
SKIRT_DEPTH_MM = round(SKIRT_Z[1] - SKIRT_Z[0], 9)

# Buffer resolution and the tolerance the closed plan is decimated to.  A
# union of discs leaves sub-micron segments at every arc handover.
# Undecimated they become the zero-area facets the release mesh contract
# rejects outright.  96 quadrant segments hold the R51.90 UM clearance arc to
# under 2 um and decimating at 2 um drops the slivers without moving any
# surface a printer could resolve; that pair meshes with no degenerate or
# collinear facet at all.  Changing either number is a geometry change and has
# to be re-checked against the mesh contract, not assumed.
SKIRT_PLAN_RESOLUTION = 96
SKIRT_PLAN_SIMPLIFY_MM = 0.002

EAR_THICKNESS_MM = round(
    TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0], 9)
# The half-lap's own governing printed section: the complete D9.8 functional
# boss less the D4.6 heat-set receiver bored through it, over the ear's own
# axial thickness.  That joint is already qualified, so everything feeding it
# is sized to stay clear of becoming a new weakest link.
EAR_NET_LIGAMENT_MM = round(
    TWEETER_JOINT_FUNCTIONAL_BOSS_D - TWEETER_JOINT_INSERT_BORE_D, 9)
EAR_NET_SECTION_MM2 = round(EAR_NET_LIGAMENT_MM * EAR_THICKNESS_MM, 9)
# The two struts this skirt replaces reached 1.44x that section at their
# narrowest.  A solid fill is strictly more material than two beams through
# the same corridor, but the claim is measured on the plan rather than
# assumed, and this is the figure it has to beat.
SUPERSEDED_STRUT_SECTION_RATIO = 1.44


# --- hidden cable path -------------------------------------------------------
# The free T cable emerges from the UM's declared central mouth at
# TS_FREE_CABLE_Z and, on the released crescent, floats behind the part.  Here
# it terminates inside the pod, so the pod has to present the cable a mouth
# exactly where the cable already is, pointing exactly where it already
# points.  The duct is the UM's own T lumen diameter: the same D5.2 cable runs
# through both.
CABLE_DUCT_D_MM = TS_DUCT_D
CABLE_DUCT_R_MM = round(CABLE_DUCT_D_MM / 2.0, 9)
CABLE_DUCT_Z_MM = TS_FREE_CABLE_Z
# Chord half-length used to read the cable's tangent off its own plan.  Short
# enough to be local, long enough not to sample plan quantisation.
CABLE_TANGENT_WINDOW_MM = 0.60
CABLE_TAIL_MIN_Y_MM = 405.0
CABLE_DUCT_MOUTH_OVERSHOOT_MM = 2.0
CABLE_DUCT_POCKET_OVERSHOOT_MM = 1.0


# --- captive magnets ---------------------------------------------------------
# Straight from the qualified vase: two D5x2 pause-and-bury cavities per D66
# land, on the land's own two flats, at the project-wide source Z=15.10, with
# the same front-face-down print frame.  ``magnets.py`` owns every dimension;
# nothing about the cavity is restated here.
MAGNET_AXIS_Z_MM = STANDARD_MAGNET_Z_MM
MAGNET_PRINT_UP = (0.0, 0.0, -1.0)
MAGNET_BED_DATUM = (0.0, 0.0, THICKNESS_MM)
MAGNETS_PER_LAND = 2


_MATE_CROSS_CHECK = {
    "joint_x_mm": (-24.0, 24.0),
    "joint_y_mm": 421.5,
    "addon_joint_z_mm": (12.40, THICKNESS_MM),
    "core_joint_z_mm": (CORE_REAR_Z, 12.20),
    "insert_bore_d_mm": 4.6,
    "insert_depth_mm": 4.0,
    "clearance_bore_d_mm": 3.4,
}


def check_released_mate() -> None:
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


def mate_facts() -> dict:
    """The released half-lap interface, as both variants publish it."""
    return {
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
    }


def axis_placement_facts() -> dict:
    """Why the mount land sits where it does, from the released datums."""
    return {
        "governing_constraint": AXIS_GOVERNING_CONSTRAINT,
        "limit_from_um_core_ring_y_mm": AXIS_Y_LIMIT_FROM_UM_RING_MM,
        "limit_from_ear_notch_y_mm": AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM,
        "um_mate_gap_mm": UM_MATE_GAP_MM,
        "ear_notch_r_mm": EAR_NOTCH_R_MM,
        "ear_notch_ligament_mm": EAR_NOTCH_LIGAMENT_MM,
        "ear_notch_ligament_authority": (
            "the vase's qualified 1.20 mm blind wall, the smallest wall "
            "these parts print anywhere"),
        "pod_wall_off_um_core_ring_mm": POD_WALL_OFF_UM_RING_MM,
        "pod_wall_off_ear_notch_mm": POD_WALL_OFF_EAR_NOTCH_MM,
        "why_the_notch_governs": (
            "the notch is cut over z=6.7..12.4 while the D66 land runs "
            "the full depth, so a land nicked by it would either lose land "
            "or grow its plan rearward at z=6.7"),
    }


def land_facts() -> dict:
    """The D66 land and its two magnet flats, shared by both variants."""
    return {
        "profile": "straight_cylinder_with_two_magnet_flats",
        "outer_d_mm": POD_OUTER_D_MM,
        "outer_r_mm": POD_OUTER_R_MM,
        "outer_wall_authority": "tebm35c10_4_driver_land_d66",
        "outer_wall_basis": (
            "both mounting faces must carry the vase's qualified D66 "
            "land and the parts print front-face-down, so the plan may "
            "never grow rearward; a straight D66 cylinder is the unique "
            "minimum, and it is also how the qualified vase treats its "
            "own drivers, where the D66 land is the exterior surface"),
        "flat_half_width_mm": POD_FLAT_HALF_WIDTH_MM,
        "flat_depth_mm": POD_FLAT_DEPTH_MM,
        "flat_half_height_mm": T_MAGNET_FLAT_HALF_HEIGHT_MM,
        "flat_authority": (
            "the vase's own captive side-magnet flat: the qualified 6.40 mm "
            "land plus a 0.10 mm straight-face margin, so each magnet has a "
            "real planar interface instead of a tangent to the D66 circle"),
        "plan_width_mm": POD_PLAN_WIDTH_MM,
        "wall_outside_pocket_mm": POD_WALL_OVER_POCKET_MM,
        "wall_outside_m2_insert_bore_mm": POD_WALL_OVER_INSERT_MM,
        "land_margin_over_max_flange_mm": POD_LAND_MARGIN_OVER_FLANGE_MM,
        "flat_margin_over_max_flange_mm": POD_FLAT_MARGIN_OVER_FLANGE_MM,
        "released_scallop_radius_mm": SCALLOP_R_MM,
    }


def land_radius_at(z: float, z_span: tuple[float, float]) -> float:
    """Land radius at one Z.  It is the D66 circle at every Z."""
    if not (z_span[0] - 1.0e-9 <= z <= z_span[1] + 1.0e-9):
        raise ValueError(f"z={z} is outside the BMR depth stack")
    return POD_OUTER_R_MM


# --- plan helpers ------------------------------------------------------------
# The plan helpers below are pure and are read many times over -- by each
# other, by the builders and by the facts payloads.  Shapely geometries are
# immutable, so caching them is only a speed change.

@lru_cache(maxsize=1)
def _flat_clip_polygon():
    """The band both magnet flats cut every D66 land back to, in plan."""
    return shapely_box(
        -POD_FLAT_HALF_WIDTH_MM, -4000.0, POD_FLAT_HALF_WIDTH_MM, 4000.0)


def _flat_clip_solid():
    """Exact BREP form of the same band, so the flats stay planar faces."""
    return Box(POD_PLAN_WIDTH_MM, 4000.0, 4000.0)


@lru_cache(maxsize=None)
def land_plan(axis_y: float):
    """One D66 driver land in plan, cut back to its two magnet flats."""
    disc = Point(0.0, axis_y).buffer(
        POD_OUTER_R_MM, resolution=SKIRT_PLAN_RESOLUTION)
    return disc.intersection(_flat_clip_polygon())


def land_solid(axis_y: float, z0: float, z1: float):
    """One D66 land swept over a Z span, exact cylinder and exact flats."""
    return _cylinder_at(0.0, axis_y, POD_OUTER_R_MM, z0, z1) & _flat_clip_solid()


@lru_cache(maxsize=1)
def _um_clearance_disc():
    """The released R51.90 recut, as a plan disc.

    Shapely inscribes its buffer polygons, which would leave the fill a
    facet's sagitta *inside* the released clearance instead of outside it.
    The disc is therefore circumscribed about R51.90 grown by the plan's own
    decimation tolerance, so the mate gap survives both the faceting and the
    simplify pass with the released 0.20 mm intact.
    """
    radius = ((UM_MATE_R_MM + SKIRT_PLAN_SIMPLIFY_MM)
              / math.cos(math.pi / (4.0 * SKIRT_PLAN_RESOLUTION)))
    return Point(*UM_CUTOUT[:2]).buffer(
        radius, resolution=SKIRT_PLAN_RESOLUTION)


@lru_cache(maxsize=1)
def _boss_discs():
    return tuple(
        Point(x, TWEETER_JOINT_Y).buffer(
            TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0,
            resolution=SKIRT_PLAN_RESOLUTION)
        for x in TWEETER_JOINT_X
    )


@lru_cache(maxsize=1)
def _wing_keepout_plan():
    """Both installed wing plans, which these candidates must not disturb.

    The wing family is released and its envelope may not move, so the skirt
    yields to it rather than the other way round.  This matters at exactly one
    place: just outboard of each half-lap boss the wings run a tongue into the
    slot the released crescent leaves above the boss, and a convex hull of the
    land and the bosses fills that slot.  The wing plan already carries the
    family's own clearance notch around the released ear, so subtracting it
    puts this boundary where the released crescent's clearance envelope
    already is.  Only the plan's own decimation tolerance is added, so the two
    cannot end up tangent through faceting alone.
    """
    from .wings import wing_plan

    plans = [wing_plan("flat", side) for side in ("left", "right")]
    graded = wing_plan("graded", "right")
    if not graded.equals(plans[1]):
        raise RuntimeError(
            "the flat and graded wing plans have diverged; the skirt must be "
            "cut against both, not one")
    return unary_union([
        plan.buffer(SKIRT_PLAN_SIMPLIFY_MM, resolution=SKIRT_PLAN_RESOLUTION)
        for plan in plans
    ]).buffer(0)


@lru_cache(maxsize=1)
def _um_owned_relief_plan():
    """The UM's half of the T--UM closure envelope, plus the released seam.

    This is the same plan ``_enforce_junction_plan_ownership`` subtracts, but
    it is needed here as a *plan*: that helper only reaches the closure web's
    own Z band, and the entry collar hangs below it, so a collar cut from the
    unrelieved skirt would poke out past the skirt's real edge at z=6.8 --
    which front-face-down is an overhang.  ``tests/test_bmr_crescent.py``
    applies this and the helper to the same prism and asserts they remove the
    same volume, so the mirror cannot drift from what it mirrors.
    """
    record = junction_closure_polygons()["t_um"]
    relief = record["target"].difference(record["tweeter"]).buffer(0)
    if "terminal_drain" in record:
        relief = unary_union((relief, record["terminal_drain"])).buffer(0)
    return relief


def _simplified(polygon, label: str):
    polygon = polygon.simplify(SKIRT_PLAN_SIMPLIFY_MM, preserve_topology=True)
    if polygon.geom_type != "Polygon" or polygon.interiors:
        raise RuntimeError(
            f"the {label} must close to one simple region; got "
            f"{polygon.geom_type} with "
            f"{len(getattr(polygon, 'interiors', ()))} hole(s)")
    return polygon


@lru_cache(maxsize=1)
def base_plan():
    """Mount land, bosses and the flush fill between them, bounded by the recut.

    The fill is the convex hull of the mount land and the two complete D9.8
    bosses, so it has no concave corner of its own, less the released R51.90
    UM clearance disc.  Where that disc bites into the hull, the boundary
    becomes the released 0.20 mm mate arc; everywhere else the hull's own
    tangents carry the fill out to the bosses and back onto the land.

    Only the *mount* land is hulled, on both variants.  The junction this
    fills is between the collar and the land that mounts on it; a second land
    stacked above would drag the hull over the waist between the two and put
    material where neither the mate nor either driver needs any.
    """
    land = land_plan(MOUNT_AXIS_XY[1])
    bosses = list(_boss_discs())
    hull = unary_union([land] + bosses).convex_hull
    plan = unary_union(
        [land, hull.difference(_um_clearance_disc())] + bosses).buffer(0)
    return _simplified(plan, "junction fill plan")


@lru_cache(maxsize=1)
def skirt_plan():
    """The plate-band plan: the fill, the released web half, less the wings.

    The T--UM closure web is the released authority for this seam.  Its
    crescent half is unioned in rather than fused as a separate body, because
    the fill already covers almost all of it; the complementary ownership
    relief is subtracted afterwards and, by construction, never reaches inside
    the half these parts own.  The wing keep-out then takes back the one place
    the convex fill overreaches, just outboard of each boss.
    """
    web = junction_closure_polygons()["t_um"]["tweeter"]
    plan = unary_union([base_plan(), web]).difference(
        _wing_keepout_plan()).buffer(0)
    return _simplified(plan, "junction skirt plan")


def base_um_ring_clearance_mm() -> float:
    """Closest approach of the flush fill itself to the UM core ring.

    This is the fill's own boundary, so it is the released R51.90 recut and
    has to read back as the 0.20 mm mate gap.
    """
    return base_plan().distance(Point(*UM_CUTOUT[:2])) - UM_CORE_R


def skirt_um_ring_clearance_mm() -> float:
    """Closest approach of the whole plate-band plan to the UM core ring.

    Below the recut this is the released closure web's own seam, which runs
    closer to the ring than 0.20 mm by design; the released crescent reads
    back the same figure.
    """
    return skirt_plan().distance(Point(*UM_CUTOUT[:2])) - UM_CORE_R


@lru_cache(maxsize=1)
def ear_load_path_section_mm2() -> float:
    """Narrowest printed section on either ear-to-land load path.

    Measured, not asserted: at stations along the line from each boss centre
    to the mount land's axis, the plan's own chord across that line, less
    whatever the D4.6 heat-set receiver takes out of that same chord, over the
    ear's 5.9 mm thickness.  The receiver is removed from the chord rather
    than from the plan so that a station running through the bore still
    reports the real ligament either side of it instead of nothing at all.
    """
    plan = skirt_plan()
    receivers = unary_union([
        Point(x, TWEETER_JOINT_Y).buffer(
            TWEETER_JOINT_INSERT_BORE_D / 2.0,
            resolution=SKIRT_PLAN_RESOLUTION)
        for x in TWEETER_JOINT_X
    ])
    axis = np.asarray(MOUNT_AXIS_XY, dtype=float)
    narrowest = float("inf")
    for ear_x in TWEETER_JOINT_X:
        start = np.array([ear_x, TWEETER_JOINT_Y], dtype=float)
        along = axis - start
        reach = float(np.linalg.norm(along)) - POD_OUTER_R_MM
        along = along / np.linalg.norm(along)
        across = np.array([-along[1], along[0]])
        for step in np.linspace(0.0, reach, 121):
            here = start + step * along
            probe = LineString([here - 60.0 * across, here + 60.0 * across])
            cut = probe.intersection(plan)
            if cut.is_empty:
                return 0.0
            pieces = [cut] if cut.geom_type == "LineString" else list(cut.geoms)
            local = [piece for piece in pieces
                     if piece.distance(Point(*here)) < 1.0e-6]
            if not local:
                return 0.0
            chord = max(local, key=lambda piece: piece.length)
            narrowest = min(
                narrowest,
                chord.length - chord.intersection(receivers).length)
    return round(narrowest * EAR_THICKNESS_MM, 6)


def skirt_facts() -> dict:
    """The flush junction, as both variants publish it."""
    return {
        "role": "flush_solid_junction_between_the_mount_land_and_the_um",
        "plan": (
            "convex hull of the mount land and both complete D9.8 bosses, "
            "less the released R51.90 UM clearance disc, plus the "
            "released crescent's own half of the T--UM closure web, less "
            "the released wing plan"),
        "hulled_land": "mount_land_only",
        "z_span_mm": list(SKIRT_Z),
        "depth_mm": SKIRT_DEPTH_MM,
        "z_span_authority": "released JUNCTION_WEB_Z",
        "seam_authority": (
            "the released crescent's own seam: the R51.90 recut at the "
            "cable mouth and the 0.05 mm fit seam across the closure web"),
        "um_mate_r_mm": UM_MATE_R_MM,
        "fill_um_core_ring_clearance_mm": round(
            base_um_ring_clearance_mm(), 6),
        "web_seam_um_core_ring_clearance_mm": round(
            skirt_um_ring_clearance_mm(), 6),
        "web_seam_clearance_note": (
            "below the 0.20 mm recut because the released closure web's "
            "own seam runs closer to the ring; the released crescent "
            "reads back the same figure"),
        "ear_load_path_section_mm2": ear_load_path_section_mm2(),
        "half_lap_net_section_mm2": EAR_NET_SECTION_MM2,
        "ear_load_path_section_ratio": round(
            ear_load_path_section_mm2() / EAR_NET_SECTION_MM2, 6),
        "superseded_strut_section_ratio": SUPERSEDED_STRUT_SECTION_RATIO,
        "section_rule": (
            "the narrowest printed section on either ear-to-land load "
            "path, less the D4.6 receiver, must stay above the section "
            "the two superseded struts reached, so the already-qualified "
            "half-lap remains the governing member"),
        "rear_growth_rule": (
            "the skirt stops at the core rear plane and the cable entry "
            "collar below it is cut from the skirt's own plan, so the "
            "front-face-down exterior never grows rearward"),
        "wing_keepout_rule": (
            "the released wing family's envelope may not move, so the "
            "fill is cut back to it just outboard of each boss, where "
            "the wings run a tongue into the slot the released crescent "
            "leaves above the ear"),
    }


# --- the one hidden cable entry ----------------------------------------------

def _free_t_plan_tail():
    """The free T cable's own analytic plan over the crescent zone.

    ``route._TS_PLAN`` is the analytic T centreline every Obi-Wan owner is
    generated from.  Only its handoff tail matters here, and that tail is
    identical in both stand states -- the states differ upstream, in the LM
    entry.  ``tests/test_bmr_crescent.py`` checks the duct this produces
    against the staged reference cable BREP in both states rather than taking
    that on trust.
    """
    plan = np.asarray(route._TS_PLAN, dtype=float)
    tail = plan[plan[:, 1] > CABLE_TAIL_MIN_Y_MM]
    if len(tail) < 16:
        raise RuntimeError("the free T plan tail is too short to read")
    return tail


def _cable_duct_axis():
    """Mouth, direction and length of the one hidden cable duct.

    The duct is a straight bore along the cable's own tangent where the cable
    crosses the mount land's wall.  Run backwards it reaches the R51.90 mate
    face within a fraction of a millimetre of where the cable itself crosses
    that face, and run forwards it opens into the mount land's chamber; taking
    the tangent at the wall rather than at the mate face is what keeps the
    bore short instead of turning it into a grazing chord through the land.
    """
    tail = _free_t_plan_tail()
    axis = np.asarray(MOUNT_AXIS_XY, dtype=float)
    radius_pod = np.hypot(tail[:, 0] - axis[0], tail[:, 1] - axis[1])
    inside = np.flatnonzero(radius_pod <= POD_OUTER_R_MM)
    if not len(inside) or inside[0] == 0:
        raise RuntimeError("the free T cable never crosses the pod wall")
    index = int(inside[0])
    before, after = radius_pod[index - 1], radius_pod[index]
    blend = (POD_OUTER_R_MM - before) / (after - before)
    wall = tail[index - 1] + blend * (tail[index] - tail[index - 1])

    stations = np.concatenate(([0.0], np.cumsum(
        np.hypot(*np.diff(tail, axis=0).T))))
    station = stations[index - 1] + blend * (
        stations[index] - stations[index - 1])
    span = np.array([
        [np.interp(station + sign * CABLE_TANGENT_WINDOW_MM,
                   stations, tail[:, column])
         for column in (0, 1)]
        for sign in (-1.0, 1.0)
    ])
    direction = span[1] - span[0]
    direction = direction / np.linalg.norm(direction)

    # Backwards to the mate face: the near root of the R51.90 circle.
    offset = wall - np.asarray(UM_CUTOUT[:2], dtype=float)
    half = float(offset @ -direction)
    root = half * half - (float(offset @ offset) - UM_MATE_R_MM ** 2)
    if root <= 0.0:
        raise RuntimeError("the cable duct never reaches the UM mate face")
    to_mouth = -half - math.sqrt(root)
    mouth = wall - to_mouth * direction

    # Forwards to the mount land's chamber: the near root of the pocket circle.
    offset = wall - axis
    half = float(offset @ direction)
    root = half * half - (
        float(offset @ offset) - (TEBM_CUTOUT_D_MM / 2.0) ** 2)
    if root <= 0.0:
        raise RuntimeError("the cable duct never reaches the front chamber")
    to_chamber = -half - math.sqrt(root)

    return {
        "mouth_xy": (round(float(mouth[0]), 9), round(float(mouth[1]), 9)),
        "wall_xy": (round(float(wall[0]), 9), round(float(wall[1]), 9)),
        "direction": (round(float(direction[0]), 9),
                      round(float(direction[1]), 9)),
        "bearing_deg": round(math.degrees(
            math.atan2(direction[1], direction[0])), 6),
        "mouth_to_wall_mm": round(float(to_mouth), 9),
        "mouth_to_chamber_mm": round(float(to_mouth + to_chamber), 9),
    }


CABLE_DUCT = _cable_duct_axis()
CABLE_ENTRY_XY = CABLE_DUCT["mouth_xy"]
CABLE_DUCT_DIR = CABLE_DUCT["direction"]
CABLE_DUCT_LENGTH_MM = CABLE_DUCT["mouth_to_chamber_mm"]

# The mouth has to sit inside the UM's own declared cable mouth, or the cable
# would be entering somewhere the UM never opened.
if abs(CABLE_ENTRY_XY[0]) > T_UM_CABLE_MOUTH_HALF_WIDTH:
    raise RuntimeError(
        f"the cable entry at x={CABLE_ENTRY_XY[0]:.3f} is outside the UM's "
        f"declared +/-{T_UM_CABLE_MOUTH_HALF_WIDTH} mm cable mouth")


def _cable_mouth_alignment():
    """How far off the duct the cable arrives, and what aperture that leaves.

    A cable entering a bore off-axis only fits through the bore's projected
    aperture.  This is the geometric test the mouth alignment has to pass, and
    it is what rules out both a bore normal to the mate face and a bore aimed
    at the land's axis: at this emergence both are more than 28 degrees off the
    cable and a Ø6 bore would not pass a Ø5.2 cable at all.
    """
    tail = _free_t_plan_tail()
    index = int(np.argmin(np.hypot(
        tail[:, 0] - CABLE_ENTRY_XY[0], tail[:, 1] - CABLE_ENTRY_XY[1])))
    tangent = (tail[min(index + 4, len(tail) - 1)]
               - tail[max(index - 4, 0)])
    tangent = tangent / np.linalg.norm(tangent)
    misalignment = round(math.degrees(math.acos(min(1.0, abs(
        float(tangent @ np.asarray(CABLE_DUCT_DIR)))))), 6)
    return misalignment, round(
        CABLE_DUCT_D_MM * math.cos(math.radians(misalignment)), 9)


CABLE_MOUTH_MISALIGNMENT_DEG, CABLE_MOUTH_APERTURE_MM = (
    _cable_mouth_alignment())
if CABLE_MOUTH_APERTURE_MM < TS_CABLE_D_EST:
    raise RuntimeError(
        f"the cable duct presents only {CABLE_MOUTH_APERTURE_MM:.3f} mm of "
        f"aperture to a {TS_CABLE_D_EST} mm cable arriving "
        f"{CABLE_MOUTH_MISALIGNMENT_DEG:.2f} degrees off its axis")

# The skirt stops at the core rear plane, but the cable runs 3.0 mm behind it,
# so something has to carry the duct over that last stretch.  That something is
# a collar the shape of the duct and nothing else: the duct's own plan sweep
# grown by one wall.  Being a constant offset of a straight bore it is a
# stadium -- two parallel sides closed by half-round ends -- so the entry has
# no flat slab face and no corner on it anywhere.
#
# One wall thickness, and it is the vase's qualified 1.20 mm blind wall, which
# is already the thinnest wall these parts print anywhere.  The project's
# 0.85 mm buried-span skin would also have applied here -- this is a buried T
# span -- but taking it would have bought 0.35 mm of radius at the cost of the
# family's one simple wall invariant, so the thicker qualified figure stands.
ENTRY_COLLAR_WALL_MM = T_BLIND_BACK_WALL_THICKNESS_MM
ENTRY_COLLAR_R_MM = round(CABLE_DUCT_R_MM + ENTRY_COLLAR_WALL_MM, 9)
ENTRY_COLLAR_Z = (round(CABLE_DUCT_Z_MM - ENTRY_COLLAR_R_MM, 9), CORE_REAR_Z)
# The collar runs from one radius behind the mouth -- so the mate face carries
# its full section and the R51.90 arc, not a straight cut, terminates it -- to
# one radius past the point where the cable crosses the land wall, beyond which
# the land's own body is the duct's wall.
ENTRY_COLLAR_BACK_MM = ENTRY_COLLAR_R_MM
ENTRY_COLLAR_REACH_MM = round(
    CABLE_DUCT["mouth_to_wall_mm"] + ENTRY_COLLAR_R_MM, 9)


@lru_cache(maxsize=1)
def entry_collar_plan():
    """The stadium of material that carries the duct behind the skirt.

    The duct's own plan sweep offset by one wall, clipped to what the skirt
    above it actually is -- its plan less the UM's ownership relief.  Two
    things fall out of that clip rather than needing constants of their own:
    the collar's mate face becomes the same R51.90 arc and released seam the
    skirt presents, and the collar can never reach outside the skirt, which is
    what keeps the front-face-down silhouette from growing rearward at z=6.8.

    Cut from ``skirt_plan`` rather than rebuilt, and deliberately not decimated
    again: an independently decimated copy of the same arc lands a micron or
    two either side of it, and a couple of microns of collar outside the skirt
    is still an overhang the silhouette gate would report.
    """
    mouth = np.asarray(CABLE_ENTRY_XY, dtype=float)
    direction = np.asarray(CABLE_DUCT_DIR, dtype=float)
    sweep = LineString([
        tuple(mouth - ENTRY_COLLAR_BACK_MM * direction),
        tuple(mouth + ENTRY_COLLAR_REACH_MM * direction),
    ])
    plan = skirt_plan().difference(_um_owned_relief_plan()).intersection(
        sweep.buffer(ENTRY_COLLAR_R_MM, resolution=SKIRT_PLAN_RESOLUTION))
    if plan.geom_type != "Polygon" or plan.interiors:
        raise RuntimeError(
            "the cable entry collar plan must close to one simple region; got "
            f"{plan.geom_type} with "
            f"{len(getattr(plan, 'interiors', ()))} hole(s)")
    return plan


def duct_cutter():
    """The one hidden cable duct, mate face to the mount land's chamber."""
    start = (np.asarray(CABLE_ENTRY_XY, dtype=float)
             - CABLE_DUCT_MOUTH_OVERSHOOT_MM
             * np.asarray(CABLE_DUCT_DIR, dtype=float))
    length = (CABLE_DUCT_LENGTH_MM + CABLE_DUCT_MOUTH_OVERSHOOT_MM
              + CABLE_DUCT_POCKET_OVERSHOOT_MM)
    middle = start + (length / 2.0) * np.asarray(CABLE_DUCT_DIR, dtype=float)
    # Cylinder() lies on +Z; Rot(X=90) lays it on -Y, i.e. bearing -90, and
    # the Z rotation then carries it round to the duct's own bearing.
    return (Pos(float(middle[0]), float(middle[1]), CABLE_DUCT_Z_MM)
            * Rot(Z=CABLE_DUCT["bearing_deg"] + 90.0)
            * Rot(X=90.0)
            * Cylinder(CABLE_DUCT_R_MM, length))


def cable_entry_opening() -> dict:
    """The one declared mate-face entry, as both variants publish it."""
    return {
        "name": "um_mate_face_cable_entry",
        "kind": "cable_entry",
        "exposure": "um_mate",
        "face": f"um_clearance_cylinder_r{UM_MATE_R_MM}",
        "diameter_mm": CABLE_DUCT_D_MM,
        "diameter_authority": "route.TS_DUCT_D, the UM's own T lumen",
        "mouth_xy_mm": list(CABLE_ENTRY_XY),
        "z_mm": CABLE_DUCT_Z_MM,
        "direction_xy": list(CABLE_DUCT_DIR),
        "bearing_deg": CABLE_DUCT["bearing_deg"],
        "length_to_front_chamber_mm": CABLE_DUCT_LENGTH_MM,
        "count": 1,
    }


def cable_entry_facts() -> dict:
    """The shared half of each variant's ``cable`` fact block."""
    return {
        "external_outlets": 0,
        "entries": 1,
        "free_t_cable_centreline_z_mm": TS_FREE_CABLE_Z,
        "cable_d_mm": TS_CABLE_D_EST,
        "entry_mouth_xy_mm": list(CABLE_ENTRY_XY),
        "entry_face": f"um_clearance_cylinder_r{UM_MATE_R_MM}",
        "entry_d_mm": CABLE_DUCT_D_MM,
        "entry_bearing_deg": CABLE_DUCT["bearing_deg"],
        "entry_misalignment_to_cable_deg": CABLE_MOUTH_MISALIGNMENT_DEG,
        "entry_projected_aperture_mm": CABLE_MOUTH_APERTURE_MM,
        "duct_length_to_front_chamber_mm": CABLE_DUCT_LENGTH_MM,
        "um_declared_mouth_half_width_mm": T_UM_CABLE_MOUTH_HALF_WIDTH,
        "entry_collar_plan": (
            "stadium: the duct's own plan sweep offset by one wall, "
            "clipped to the skirt; no flat face and no corner on it"),
        "entry_collar_z_span_mm": list(ENTRY_COLLAR_Z),
        "entry_collar_wall_mm": ENTRY_COLLAR_WALL_MM,
        "entry_collar_wall_authority": (
            "the vase's qualified 1.20 mm blind wall, already the "
            "thinnest wall these parts print anywhere; the project's "
            "0.85 mm buried-span skin would also have applied but was "
            "not taken, for 0.35 mm of radius against that invariant"),
        "entry_collar_r_mm": ENTRY_COLLAR_R_MM,
        "entry_collar_sweep_mm": [
            -ENTRY_COLLAR_BACK_MM, ENTRY_COLLAR_REACH_MM],
        "entry_collar_area_mm2": round(entry_collar_plan().area, 6),
    }


# --- captive side magnets ----------------------------------------------------

def land_magnet_faces(axis_y: float) -> tuple[dict, ...]:
    """The vase's own two side-magnet interface datums on one D66 land."""
    return tuple({
        "side": side,
        "face_xyz_mm": (sign * T_MAGNET_FACE_X_MM, float(axis_y),
                        MAGNET_AXIS_Z_MM),
        "outward_xyz": (sign, 0.0, 0.0),
    } for side, sign in (("left", -1.0), ("right", 1.0)))


def apply_land_magnets(
    part, lands: tuple[tuple[str, float], ...],
) -> tuple[object, tuple[CaptiveMagnetTools, ...]]:
    """Bury the vase's captive D5x2 side cavities on each named D66 land.

    Every dimension, the loading chimney, the 45-degree gable and the two
    0.45 mm skins come from ``magnets.py``; the only thing said here is where
    the stations are, and that is the vase's own land-local answer.  The host
    must already be final: ``apply_wall_cavity`` refuses a host that does not
    already contain the complete 3.00 mm captive land, and it only ever
    subtracts, so no cavity can change the exterior.
    """
    records: list[CaptiveMagnetTools] = []
    for land, axis_y in lands:
        for station in land_magnet_faces(axis_y):
            part, tools = apply_wall_cavity(
                part,
                name=f"tebm_{land}_{station['side']}_base",
                face=station["face_xyz_mm"],
                outward=station["outward_xyz"],
                owner="base",
                print_up=MAGNET_PRINT_UP,
                bed_datum=MAGNET_BED_DATUM,
            )
            records.append(tools)
    return part, tuple(records)


def magnet_facts(
    magnet_tools: tuple[CaptiveMagnetTools, ...],
    lands: tuple[tuple[str, float], ...],
) -> dict:
    """Serializable record of every captive station this variant carries.

    ``count`` is what the design declares, so the block is meaningful with or
    without a built solid to hand; ``stations`` is what the build actually
    buried, and the two are cross-checked whenever both exist.
    """
    declared = MAGNETS_PER_LAND * len(lands)
    if magnet_tools and len(magnet_tools) != declared:
        raise RuntimeError(
            f"{len(magnet_tools)} captive stations were buried but "
            f"{declared} are declared over {len(lands)} land(s)")
    return {
        "count": declared,
        "stations_recorded": len(magnet_tools),
        "per_land": MAGNETS_PER_LAND,
        "lands": [
            {"land": land, "axis_y_mm": round(float(axis_y), 9)}
            for land, axis_y in lands
        ],
        "authority": (
            "proud.vase_tebm35c10_4._apply_t_magnets, applied through the "
            "same lx521_baffle.magnets.apply_wall_cavity helper at the same "
            "land-local station"),
        "interface_face_x_mm": [-T_MAGNET_FACE_X_MM, T_MAGNET_FACE_X_MM],
        "axis_z_mm": MAGNET_AXIS_Z_MM,
        "flat_height_mm": round(2.0 * T_MAGNET_FLAT_HALF_HEIGHT_MM, 9),
        "flat_edge_margin_mm": T_MAGNET_FLAT_EDGE_MARGIN_MM,
        "print_up_source_xyz": list(MAGNET_PRINT_UP),
        "bed_datum_source_xyz": list(MAGNET_BED_DATUM),
        "closure": "pause_and_bury_transverse_gable_45deg",
        "exterior_opening": False,
        "exterior_note": (
            "the loading cradle, its chimney and the gable are one sealed "
            "void behind the qualified 0.45 mm face skin; nothing about a "
            "station reaches the exterior, so the zero-exterior-openings "
            "claim is unaffected and the sealed voids are what the shell "
            "count gate now expects"),
        "release_wiring": (
            "candidate only: these stations are recorded here and are "
            "deliberately absent from the released captive-magnet catalog, "
            "the release_validation counts, the slicing profiles and "
            "to_print.  Wiring them up -- catalog entry plus the sliced "
            "pause events -- is part of this part's future qualification"),
        "stations": [tools.facts() for tools in magnet_tools],
    }
