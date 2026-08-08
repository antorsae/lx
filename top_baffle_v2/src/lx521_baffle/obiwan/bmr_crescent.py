"""Candidate Obi-Wan BMR pod for two coaxial back-to-back TEBM35C10-4 BMRs.

This is an alternative to the released ND25FW-4 tweeter crescent, not a
replacement for it.  It presents the *identical* half-lap interface to an
unmodified Obi-Wan UM collar (x=+/-24, y=421.5, complete front local-D9.8
ears, standalone blind D4.6 x 4.0 heat-set receivers, 1.9 mm acoustic-front
floors, 0.20 mm axial gap), so the two parts are mutually swappable without
touching the UM print.

Where the released crescent is a full acoustic silhouette carrying a
face-to-face Dayton pair, this part keeps only what the two BMRs, that one
mate and one hidden cable actually need:

* a D66 pod carrying the front driver on the shared z=18.3 plane and the rear
  driver on z=-31.9 facing -z, with each driver keeping the vase's qualified
  1.20 mm blind wall (a 2.40 mm back-to-back partition and two rear chambers);
* a solid junction skirt that closes the whole plan between the pod and the UM
  collar, ending on the released crescent's own flush seam; and
* one hidden cable path: a single entry on the UM mate face, in line with the
  UM's free-T emergence, feeding the front chamber, and one declared
  pass-through in the 2.40 mm partition feeding the rear driver.

The released crescent's arm silhouette, its rear taper, the root fairing over
that taper, the boss/top-edge plan blends and the inherited M4 ND25FW-4
faceplate clamp passages are all gone: this variant clamps no tweeter, so
those four holes carried no fastener and only existed to keep a silhouette
this part no longer has.

The pod is as close to the UM as it can get
-------------------------------------------
The first candidate parked the pod on the released ND25FW-4 acoustic axis and
hung it off two struts, which left an open window between the pod and the UM
collar.  The pod is now dropped until its own D66 wall runs out of room, and
the space it used to float above is solid material.

Two things stop the drop, both measured against the released mate rather than
invented here:

* the UM's native R51.7 core ring, which the released crescent clears by
  0.20 mm -- that would put the axis at y=450.981; and
* the UM half-lap's own receiver notch, the complete D9.8 ear plus its
  0.10 mm joint clearance, which the pod wall must not nick.  The notch is
  cut over z=6.7..12.4 while the D66 land runs the full depth, so a nick there
  would either lose land or make the plan grow rearward at z=6.7, which this
  print orientation cannot take.

The notch governs.  Holding the vase's qualified 1.20 mm wall between the
notch and the pod puts the axis at y=452.494193, 15.699 mm below the released
tweeter axis, and leaves the pod wall 1.713 mm off the UM ring at the cable
mouth -- material the skirt fills.

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

The junction skirt
------------------
Between the pod, the two half-lap bosses and the UM collar the plan is solid.
The skirt is the convex hull of the pod disc and the two complete D9.8 bosses,
less the released R51.90 UM clearance disc, plus the released crescent's own
half of the T--UM closure web, less the released wing plan.  Its lower edge is
therefore the released crescent's seam, not a new boundary: 0.20 mm off the UM
at the cable mouth and the released 0.05 mm fit seam across the web.  The one
place the hull overreaches is just outboard of each boss, where both wing
families run a tongue into the slot the released crescent leaves there; the
wing envelope is released and wins, so the fill is cut back to it.  The skirt
occupies the plate band z=6.8..18.3 only, so the rear driver stack alone
reaches behind the core rear plane and the front-face-down silhouette never
grows rearward.

Minimalism here means "no material beyond the flush fill", not "struts".

The cable is invisible
----------------------
There are no external outlets.  The free T cable leaves the UM at its declared
mouth and immediately enters one Ø6.00 duct -- the UM's own T lumen diameter --
whose mouth sits on the pod's R51.90 mate face, on the cable's own centreline
and along the cable's own tangent.  The duct runs into the front chamber.  The
rear driver is fed from that chamber through one Ø4.60 pass in the 2.40 mm
partition, the same branch diameter the qualified vase uses for a single
driver's leads.  Behind the skirt the duct is carried by a collar that is the
bore's own sweep offset by one wall -- a stadium, not a slab, with no flat
face or corner on it.  Every remaining opening either faces the UM mate or is
a driver pocket or a blind bore, so the assembled exterior has none.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=18.3 is the acoustic front
and z=6.8 is the Obi-Wan core rear plane.  The pod grows only rearward, to
z=-31.9.

Candidate status
----------------
Nothing here is release-authorized.  ``RELEASE_AUTHORIZED`` is false and
``PHYSICAL_MEASURE_REQUIRED`` is true: the driver envelope, the back-to-back
partition, the dropped acoustic axis, the hidden cable path and the two-screw
joint demand under roughly twice the released crescent's hanging mass all need
physical qualification before this part is printed for use.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from build123d import (
    Part,
    Pos,
    Rot,
    Cylinder,
)
from shapely.geometry import LineString, Point
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
    _apply_complete_um_tweeter_joint,
    _enforce_junction_plan_ownership,
    _plan_prism,
    _require_guarded_build,
    junction_closure_polygons,
)
from . import route
from .route import TS_CABLE_D_EST, TS_DUCT_D, TS_FREE_CABLE_Z


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


# --- how far the pod may drop toward the UM ----------------------------------
# The released crescent clears the UM's native R51.7 core ring by 0.20 mm and
# recuts itself on that circle; that is the mate gap for every face this part
# presents to the UM, including the skirt's own.
UM_MATE_GAP_MM = 0.20
UM_MATE_R_MM = round(UM_CORE_R + UM_MATE_GAP_MM, 9)

# The opposing UM half-lap's receiver notch is the complete D9.8 functional
# ear grown by the released 0.10 mm joint clearance.  Its widest point is that
# boss circle, and it is cut out of this part over z=6.7..12.4 while the D66
# land runs the full depth -- so the pod wall must stay outside it or the plan
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

BMR_AXIS_XY = (0.0, max(AXIS_Y_LIMIT_FROM_UM_RING_MM,
                        AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM))
AXIS_GOVERNING_CONSTRAINT = (
    "um_half_lap_receiver_notch"
    if AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM >= AXIS_Y_LIMIT_FROM_UM_RING_MM
    else "um_core_ring")

# The released ND25FW-4 acoustic axis, kept only as the datum this part moved
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

POD_DROP_MM = round(RELEASED_AXIS_XY[1] - BMR_AXIS_XY[1], 9)
UM_AXIS_SPACING_MM = round(BMR_AXIS_XY[1] - UM_CUTOUT[1], 9)
RELEASED_UM_AXIS_SPACING_MM = round(RELEASED_AXIS_XY[1] - UM_CUTOUT[1], 9)
POD_WALL_OFF_UM_RING_MM = round(
    BMR_AXIS_XY[1] - POD_OUTER_R_MM - (UM_CUTOUT[1] + UM_CORE_R), 9)
POD_WALL_OFF_EAR_NOTCH_MM = round(
    math.hypot(TWEETER_JOINT_X[1], BMR_AXIS_XY[1] - TWEETER_JOINT_Y)
    - POD_OUTER_R_MM - EAR_NOTCH_R_MM, 9)


# --- coaxial back-to-back depth stack ---------------------------------------
# Front driver: acoustic face on the shared z=18.3 plane, pocket cut rearward,
# blind wall over the released 25.1 mm envelope.  These are the vase's own
# numbers, imported rather than restated.
FRONT_MOUNT_Z_MM = THICKNESS_MM
FRONT_POCKET_FLOOR_Z_MM = LOWER_T_POCKET_REAR_Z_MM          # -5.6
FRONT_ENVELOPE_END_Z_MM = REAR_T_MOUNT_Z_MM                 # -6.8

# Rear driver: mirror of the front one about the partition.  Each driver keeps
# a full 1.20 mm blind wall of its own, so the coaxial partition is 2.40 mm
# and the two rear volumes stay separate chambers except at the one declared
# pass-through that feeds the rear driver.  A single 1.20 mm partition would
# have been a shared skin taking differential pressure from both sides;
# doubling it costs 1.2 mm of stack and keeps every driver's qualified wall.
PARTITION_THICKNESS_MM = round(2.0 * T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_POCKET_ROOF_Z_MM = round(
    FRONT_ENVELOPE_END_Z_MM - T_BLIND_BACK_WALL_THICKNESS_MM, 9)
REAR_MOUNT_Z_MM = round(THICKNESS_MM - 2.0 * TEBM_DEPTH_MM, 9)   # -31.9
STACK_DEPTH_MM = round(FRONT_MOUNT_Z_MM - REAR_MOUNT_Z_MM, 9)    # 50.2
REAR_PROTRUSION_MM = round(CORE_REAR_Z - REAR_MOUNT_Z_MM, 9)     # 38.7


# --- junction skirt ----------------------------------------------------------
# The skirt lives in the released crescent's own plate band and in the
# released closure web's own Z span; they are the same span, and the identity
# is asserted rather than assumed.
SKIRT_Z = JUNCTION_WEB_Z
if tuple(SKIRT_Z) != (CORE_REAR_Z, THICKNESS_MM):
    raise RuntimeError("the released T--UM closure web Z span moved")
SKIRT_DEPTH_MM = round(SKIRT_Z[1] - SKIRT_Z[0], 9)               # 11.5

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
    TWEETER_ADDON_JOINT_Z[1] - TWEETER_ADDON_JOINT_Z[0], 9)      # 5.9
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
    crosses the pod wall.  Run backwards it reaches the R51.90 mate face
    within a fraction of a millimetre of where the cable itself crosses that
    face, and run forwards it opens into the front chamber; taking the tangent
    at the wall rather than at the mate face is what keeps the bore short
    instead of turning it into a grazing chord through the pod.
    """
    tail = _free_t_plan_tail()
    axis = np.asarray(BMR_AXIS_XY, dtype=float)
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

    # Forwards to the front chamber: the near root of the pocket circle.
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
    at the pod axis: at this emergence both are more than 28 degrees off the
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
# is already the thinnest wall this part prints anywhere.  The project's
# 0.85 mm buried-span skin would also have applied here -- this is a buried T
# span -- but taking it would have bought 0.35 mm of radius at the cost of the
# part's one simple wall invariant, so the thicker qualified figure stands.
ENTRY_COLLAR_WALL_MM = T_BLIND_BACK_WALL_THICKNESS_MM
ENTRY_COLLAR_R_MM = round(CABLE_DUCT_R_MM + ENTRY_COLLAR_WALL_MM, 9)
ENTRY_COLLAR_Z = (round(CABLE_DUCT_Z_MM - ENTRY_COLLAR_R_MM, 9), CORE_REAR_Z)
# The collar runs from one radius behind the mouth -- so the mate face carries
# its full section and the R51.90 arc, not a straight cut, terminates it -- to
# one radius past the point where the cable crosses the pod wall, beyond which
# the pod's own body is the duct's wall.
ENTRY_COLLAR_BACK_MM = ENTRY_COLLAR_R_MM
ENTRY_COLLAR_REACH_MM = round(
    CABLE_DUCT["mouth_to_wall_mm"] + ENTRY_COLLAR_R_MM, 9)

# One declared pass feeds the rear driver from the front chamber.  Ø4.60 is
# the qualified vase's own single-driver lead branch -- the smallest lead
# passage this family has ever proven -- and it is pushed as far outboard in
# the partition as its own 1.20 mm wall to the pocket bore allows, which is
# where the driver motor is not.
PARTITION_PASS_D_MM = UPPER_T_BRANCH_D_MM
PARTITION_PASS_OFFSET_MM = round(
    TEBM_CUTOUT_D_MM / 2.0 - PARTITION_PASS_D_MM / 2.0
    - T_BLIND_BACK_WALL_THICKNESS_MM, 9)
PARTITION_PASS_XY = (
    BMR_AXIS_XY[0],
    round(BMR_AXIS_XY[1] - PARTITION_PASS_OFFSET_MM, 9),
)


# --- driver interfaces ------------------------------------------------------
FRONT_MOUNT_CLOCK_DEG = LOWER_T_MOUNT_CLOCK_DEG
REAR_MOUNT_CLOCK_DEG = UPPER_T_MOUNT_CLOCK_DEG

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


# The plan helpers below are pure and are read many times over -- by
# each other, by the builder and by the facts payload.  Shapely
# geometries are immutable, so caching them is only a speed change.
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
    """Both installed wing plans, which this candidate must not disturb.

    The wing family is released and its envelope may not move, so the skirt
    yields to it rather than the other way round.  This matters at exactly one
    place: just outboard of each half-lap boss the wings run a tongue into the
    slot the released crescent leaves above the boss, and a convex hull of the
    pod and the bosses fills that slot.  The wing plan already carries the
    family's own clearance notch around the released ear, so subtracting it
    puts this part's boundary where the released crescent's clearance envelope
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
    """Pod, bosses and the flush fill between them, bounded by the UM recut.

    The fill is the convex hull of the pod disc and the two complete D9.8
    bosses, so it has no concave corner of its own, less the released R51.90
    UM clearance disc.  Where that disc bites into the hull, the boundary
    becomes the released 0.20 mm mate arc; everywhere else the hull's own
    tangents carry the fill out to the bosses and back onto the pod.
    """
    pod = Point(*BMR_AXIS_XY).buffer(
        POD_OUTER_R_MM, resolution=SKIRT_PLAN_RESOLUTION)
    bosses = list(_boss_discs())
    hull = unary_union([pod] + bosses).convex_hull
    plan = unary_union(
        [pod, hull.difference(_um_clearance_disc())] + bosses).buffer(0)
    return _simplified(plan, "junction fill plan")


@lru_cache(maxsize=1)
def skirt_plan():
    """The plate-band plan: the fill, the released web half, less the wings.

    The T--UM closure web is the released authority for this seam.  Its
    crescent half is unioned in rather than fused as a separate body, because
    the fill already covers almost all of it; the complementary ownership
    relief is subtracted afterwards and, by construction, never reaches inside
    the half this part owns.  The wing keep-out then takes back the one place
    the convex fill overreaches, just outboard of each boss.
    """
    web = junction_closure_polygons()["t_um"]["tweeter"]
    plan = unary_union([base_plan(), web]).difference(
        _wing_keepout_plan()).buffer(0)
    return _simplified(plan, "junction skirt plan")


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
    """Narrowest printed section on either ear-to-pod load path.

    Measured, not asserted: at stations along the line from each boss centre
    to the pod axis, the plan's own chord across that line, less whatever the
    D4.6 heat-set receiver takes out of that same chord, over the ear's 5.9 mm
    thickness.  The receiver is removed from the chord rather than from the
    plan so that a station running through the bore still reports the real
    ligament either side of it instead of nothing at all.
    """
    plan = skirt_plan()
    receivers = unary_union([
        Point(x, TWEETER_JOINT_Y).buffer(
            TWEETER_JOINT_INSERT_BORE_D / 2.0,
            resolution=SKIRT_PLAN_RESOLUTION)
        for x in TWEETER_JOINT_X
    ])
    axis = np.asarray(BMR_AXIS_XY, dtype=float)
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


def _pod_solid():
    """The D66 land swept over the whole coaxial stack."""
    return _cylinder_at(
        BMR_AXIS_XY[0], BMR_AXIS_XY[1], POD_OUTER_R_MM,
        REAR_MOUNT_Z_MM, THICKNESS_MM)


def _duct_cutter():
    """The one hidden cable duct, mate face to front chamber."""
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


def _apply_driver_interfaces(part):
    """Two coaxial blind pockets, eight M2 bores, the two cable passages."""
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

    part -= _duct_cutter()
    part -= _cylinder_at(
        PARTITION_PASS_XY[0], PARTITION_PASS_XY[1],
        PARTITION_PASS_D_MM / 2.0,
        REAR_POCKET_ROOF_Z_MM - over, FRONT_POCKET_FLOOR_Z_MM + over)
    return part


def bmr_crescent():
    """Dropped BMR pod flush-skirted onto the released UM half-lap mate."""
    _require_guarded_build()
    _check_released_mate()

    part = _fuse_required(
        _pod_solid(), _plan_prism(skirt_plan(), *SKIRT_Z),
        "flush junction skirt onto the BMR pod")
    part = _fuse_required(
        part, _plan_prism(entry_collar_plan(), *ENTRY_COLLAR_Z),
        "mate-face cable entry collar onto the BMR pod")
    # Hand the UM back its own half of the T--UM closure envelope and the
    # released 0.05 mm fit seam.  The crescent half this part owns is already
    # in the plan above, and the relief is its exact complement, so this only
    # ever removes material that belongs to the UM print.
    part = _enforce_junction_plan_ownership(part, "t_um", "tweeter")
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
    """Every intentional break in the skin, with its authority and side.

    ``exposure`` is the gate: ``um_mate`` faces the collar across the released
    mate gap, ``driver_face`` is under a fitted driver, ``internal`` never
    reaches the skin at all.  Nothing is allowed to be ``exterior``.
    """
    return [
        {
            "name": "front_driver_pocket_mouth",
            "kind": "driver_pocket",
            "exposure": "driver_face",
            "face": "acoustic_front_z_18p3",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(BMR_AXIS_XY),
            "z_span_mm": [FRONT_POCKET_FLOOR_Z_MM, THICKNESS_MM],
        },
        {
            "name": "rear_driver_pocket_mouth",
            "kind": "driver_pocket",
            "exposure": "driver_face",
            "face": "bmr_rear_land_z_minus_31p9",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(BMR_AXIS_XY),
            "z_span_mm": [REAR_MOUNT_Z_MM, REAR_POCKET_ROOF_Z_MM],
        },
        {
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
        },
        {
            "name": "chamber_partition_cable_pass",
            "kind": "cable_pass",
            "exposure": "internal",
            "diameter_mm": PARTITION_PASS_D_MM,
            "diameter_authority": (
                "vase UPPER_T_BRANCH_D_MM, one driver's lead branch"),
            "axis_xy_mm": list(PARTITION_PASS_XY),
            "offset_from_driver_axis_mm": PARTITION_PASS_OFFSET_MM,
            "z_span_mm": [REAR_POCKET_ROOF_Z_MM, FRONT_POCKET_FLOOR_Z_MM],
            "wall_to_pocket_bore_mm": T_BLIND_BACK_WALL_THICKNESS_MM,
            "count": 1,
        },
        {
            "name": "um_half_lap_clearance_passages",
            "kind": "released_mate",
            "exposure": "um_mate",
            "diameter_mm": TWEETER_JOINT_HOLE_D,
            "count": 2,
            "centres_xy_mm": [[x, TWEETER_JOINT_Y] for x in TWEETER_JOINT_X],
        },
        {
            "name": "um_half_lap_insert_receivers",
            "kind": "released_mate",
            "exposure": "um_mate",
            "diameter_mm": TWEETER_JOINT_INSERT_BORE_D,
            "count": 2,
            "blind": True,
            "front_floor_mm": insert_front_floor_mm(),
        },
        {
            "name": "m2_driver_insert_bores",
            "kind": "driver_mount",
            "exposure": "driver_face",
            "diameter_mm": M2_INSERT_BORE_D_MM,
            "depth_mm": M2_INSERT_DEPTH_MM,
            "count": 2 * TEBM_MOUNT_HOLE_COUNT,
            "blind": True,
            "pcd_mm": TEBM_MOUNT_PCD_MM,
        },
    ]


def design_facts() -> dict:
    """Envelope, mate coordinates, hidden cable path and candidate flags."""
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
            "shape": "dropped_pod_flush_junction_skirt_two_ears",
            "inherits_released_crescent_outline": False,
            "removed_from_the_first_candidate": [
                "released crescent arm silhouette",
                "released rear taper and its root fairing",
                "boss-to-top-edge tangent plan blends",
                "inherited M4 ND25FW-4 faceplate clamp passages",
                "the two drafted struts and the open window between them",
                "both external Ø4.6 driver lead outlets",
            ],
            "removal_note": (
                "this variant clamps no tweeter, so the four inherited M4 "
                "passages carried no fastener; they existed only to keep a "
                "released silhouette this part no longer has.  The struts and "
                "the two external outlets are superseded by the flush "
                "junction skirt and the one hidden cable duct"),
            "minimalism_rule": (
                "no material beyond the flush fill: the plan is the pod, the "
                "two bosses, the convex fill between them and the released "
                "closure web, and nothing else"),
        },
        "driver": {
            "model": "Tectonic TEBM35C10-4",
            "count": 2,
            "arrangement": "coaxial_back_to_back_one_axis",
            "axis_xy_mm": list(BMR_AXIS_XY),
            "axis_authority": (
                "dropped until the pod wall keeps the vase's 1.20 mm wall "
                "outside the UM half-lap receiver notch"),
            "released_axis_xy_mm": list(RELEASED_AXIS_XY),
            "drop_below_released_axis_mm": POD_DROP_MM,
            "um_to_bmr_axis_spacing_mm": UM_AXIS_SPACING_MM,
            "released_um_to_tweeter_axis_spacing_mm": (
                RELEASED_UM_AXIS_SPACING_MM),
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
        "axis_placement": {
            "governing_constraint": AXIS_GOVERNING_CONSTRAINT,
            "limit_from_um_core_ring_y_mm": AXIS_Y_LIMIT_FROM_UM_RING_MM,
            "limit_from_ear_notch_y_mm": AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM,
            "um_mate_gap_mm": UM_MATE_GAP_MM,
            "ear_notch_r_mm": EAR_NOTCH_R_MM,
            "ear_notch_ligament_mm": EAR_NOTCH_LIGAMENT_MM,
            "ear_notch_ligament_authority": (
                "the vase's qualified 1.20 mm blind wall, the smallest wall "
                "this part prints anywhere"),
            "pod_wall_off_um_core_ring_mm": POD_WALL_OFF_UM_RING_MM,
            "pod_wall_off_ear_notch_mm": POD_WALL_OFF_EAR_NOTCH_MM,
            "why_the_notch_governs": (
                "the notch is cut over z=6.7..12.4 while the D66 land runs "
                "the full depth, so a pod nicked by it would either lose land "
                "or grow its plan rearward at z=6.7"),
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
                "volumes remain separate chambers apart from the one declared "
                "lead pass-through"),
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
        },
        "skirt": {
            "role": "flush_solid_junction_between_the_pod_and_the_um",
            "plan": (
                "convex hull of the pod disc and both complete D9.8 bosses, "
                "less the released R51.90 UM clearance disc, plus the "
                "released crescent's own half of the T--UM closure web, less "
                "the released wing plan"),
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
                "the narrowest printed section on either ear-to-pod load "
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
                "thinnest wall this part prints anywhere; the project's "
                "0.85 mm buried-span skin would also have applied but was "
                "not taken, for 0.35 mm of radius against that invariant"),
            "entry_collar_r_mm": ENTRY_COLLAR_R_MM,
            "entry_collar_sweep_mm": [
                -ENTRY_COLLAR_BACK_MM, ENTRY_COLLAR_REACH_MM],
            "entry_collar_area_mm2": round(entry_collar_plan().area, 6),
            "partition_pass_d_mm": PARTITION_PASS_D_MM,
            "partition_pass_xy_mm": list(PARTITION_PASS_XY),
            "note": (
                "the free T cable leaves the UM's declared central mouth and "
                "goes straight into the one duct, whose mouth sits on this "
                "part's R51.90 mate face along the cable's own tangent; the "
                "rear driver is fed from the front chamber through the one "
                "declared partition pass.  Nothing about the cable is visible "
                "on the assembled exterior"),
        },
        "declared_openings": declared_openings(),
        "exterior_openings": [
            opening["name"] for opening in declared_openings()
            if opening["exposure"] == "exterior"
        ],
    }


def obiwan_bmr_attachments():
    _require_guarded_build()
    return {"addon_bmr_crescent": bmr_crescent()}


def gen_step():
    _require_guarded_build()
    return ordered_labeled_compound(
        obiwan_bmr_attachments(),
        label="lx521_obiwan_r6f_bmr_crescent_candidate")
