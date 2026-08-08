"""Candidate Obi-Wan BMR pod for two opposed TEBM35C10-4 BMRs.

The second of the two candidate alternatives to the released ND25FW-4 tweeter
crescent.  Where ``bmr_crescent`` stacks the two BMRs coaxially back to back
and grows 50.2 mm rearward, this one adopts the *shape* of the qualified
``proud.vase_tebm35c10_4``: two D66 lands side by side on the vase's own
49.3 mm axis pitch, the lower driver facing front off the shared z=18.3
acoustic plane and the upper driver facing rear off z=-6.8, both inside one
25.1 mm envelope.

It is the vase's arrangement on the crescent's mount.  The vase cannot fit
Obi-Wan -- it is a seam-B vase piece and Obi-Wan has no seam B -- so what
carries over is the layout, not the part: the axis pitch, the two lands and
their flats, the pocket depths, the 1.20 mm blind walls, the M2 clocks, the
four captive side magnets, and the shared-route-to-the-lower-pocket plus
branch-to-the-upper cable topology.  What carries over from the coaxial pod is
everything that touches the collar: the same half-lap interface, the same drop
limit, the same flush junction skirt, the same hidden Ø6.00 mate-face entry
and its stadium collar.  Both sets live in ``bmr_pod``.

The two lands
-------------
The lower land is the mount land, dropped as close to the collar as the
released mate allows -- the same computation, and the same answer, as the
coaxial pod.  The upper land sits one vase pitch above it.  The two D66
circles overlap by 16.7 mm, so the body is one continuous plan with a 43.88 mm
waist between the driver axes, and both lands carry the vase's two magnet
flats.

Every layer sits on the one before it: the body is prismatic over the whole
25.1 mm, the skirt occupies the plate band z=6.8..18.3 above it and the entry
collar is cut from the skirt's own plan, so printed front-face-down the
exterior plan only ever shrinks rearward.

Depth
-----
One 25.1 mm envelope carries both drivers, exactly as on the vase: the lower
pocket is bored from the front to z=-5.6 and closed by a 1.20 mm blind wall on
z=-6.8; the upper pocket is bored from the rear to z=17.1 and closed by its
own 1.20 mm wall under the acoustic front.  No wall is shared, and because the
pockets sit side by side there is no partition between them -- only the
6.374 mm ligament the two pocket bores leave on the axis line, which is what
the lead branch runs through.

Cable
-----
The same one hidden entry as the coaxial pod: the free T cable leaves the UM's
declared mouth and goes straight into a Ø6.00 duct whose mouth lies on this
part's R51.90 mate face along the cable's own tangent, and that duct opens
into the lower/front chamber.  From there one Ø4.60 branch -- the vase's own
single-driver lead branch -- crosses to the upper/rear chamber, straight along
the line joining the two driver axes and at the duct's own height, so the lead
never changes level inside the part.  It is buried under 12.2 mm of front and
8.3 mm of rear cover and there are no exterior openings at all.

Four captive magnets
--------------------
All four of the vase's captive D5 x 2 side stations, two per D66 land, at the
vase's own land-local coordinates and applied through the same ``magnets.py``
helper.  They are sealed voids behind the qualified 0.45 mm face skin.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=18.3 is the acoustic front,
z=6.8 is the Obi-Wan core rear plane and z=-6.8 is this part's rear plane.

Candidate status
----------------
Nothing here is release-authorized.  ``RELEASE_AUTHORIZED`` is false and
``PHYSICAL_MEASURE_REQUIRED`` is true.  This variant is 25.1 mm shallow but
much taller than the coaxial pod, so its hanging load reaches the two-screw
half-lap on a far longer arm; that, the driver envelope, the two magnet pairs
and the lead branch all need physical qualification first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from build123d import Part, Pos, Rot, Cylinder

from ..assembly import ordered_labeled_compound
from ..base import THICKNESS_MM
from .attachments import _cylinder_at, _fuse_required
from .carriers import (
    CORE_REAR_Z,
    TWEETER_JOINT_HOLE_D,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
    _apply_complete_um_tweeter_joint,
    _enforce_junction_plan_ownership,
    _plan_prism,
    _require_guarded_build,
)
# The family names this variant is described in.  They are re-exported
# rather than reached through ``bmr_pod`` so each part keeps one flat
# vocabulary for its own facts, gates and consumers;
# ``test_the_two_variants_share_one_family_module`` asserts they are the
# same objects in both variants, so this can never become a second copy.
from .bmr_pod import (  # noqa: F401
    AXIS_GOVERNING_CONSTRAINT,
    AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM,
    AXIS_Y_LIMIT_FROM_UM_RING_MM,
    CABLE_DUCT,
    CABLE_DUCT_D_MM,
    CABLE_DUCT_DIR,
    CABLE_DUCT_LENGTH_MM,
    CABLE_DUCT_R_MM,
    CABLE_DUCT_Z_MM,
    CABLE_ENTRY_XY,
    CABLE_MOUTH_APERTURE_MM,
    CABLE_MOUTH_MISALIGNMENT_DEG,
    EAR_NET_LIGAMENT_MM,
    EAR_NET_SECTION_MM2,
    EAR_NOTCH_LIGAMENT_MM,
    EAR_NOTCH_R_MM,
    EAR_THICKNESS_MM,
    ENTRY_COLLAR_BACK_MM,
    ENTRY_COLLAR_R_MM,
    ENTRY_COLLAR_REACH_MM,
    ENTRY_COLLAR_WALL_MM,
    ENTRY_COLLAR_Z,
    LOWER_T_MOUNT_CLOCK_DEG,
    LOWER_T_POCKET_REAR_Z_MM,
    M2_INSERT_BORE_D_MM,
    M2_INSERT_DEPTH_MM,
    MAGNETS_PER_LAND,
    MOUNT_AXIS_XY,
    PAIR_AXIS_PITCH_MM,
    PHYSICAL_MEASURE_REQUIRED,
    POD_DROP_MM,
    POD_FLAT_HALF_WIDTH_MM,
    POD_LAND_MARGIN_OVER_FLANGE_MM,
    POD_OUTER_D_MM,
    POD_OUTER_R_MM,
    POD_PLAN_WIDTH_MM,
    POD_WALL_OFF_EAR_NOTCH_MM,
    POD_WALL_OFF_UM_RING_MM,
    POD_WALL_OVER_INSERT_MM,
    POD_WALL_OVER_POCKET_MM,
    PRINT_ORIENTATION,
    REAR_T_MOUNT_Z_MM,
    RELEASE_AUTHORIZED,
    RELEASED_AXIS_XY,
    RELEASED_UM_AXIS_SPACING_MM,
    SCALLOP_R_MM,
    SKIRT_DEPTH_MM,
    SKIRT_PLAN_SIMPLIFY_MM,
    SKIRT_Z,
    SUPERSEDED_STRUT_SECTION_RATIO,
    T_BLIND_BACK_WALL_THICKNESS_MM,
    T_CLEAR_POCKET_DEPTH_MM,
    T_MAGNET_FACE_X_MM,
    TEBM_BASKET_D_MM,
    TEBM_CUTOUT_D_MM,
    TEBM_DEPTH_MM,
    TEBM_LAND_D_MM,
    TEBM_LAND_R_MM,
    TEBM_MASS_G,
    TEBM_MAX_D_MM,
    TEBM_MOUNT_HOLE_COUNT,
    TEBM_MOUNT_PCD_MM,
    UM_AXIS_SPACING_MM,
    UM_MATE_GAP_MM,
    UM_MATE_R_MM,
    UPPER_T_BRANCH_D_MM,
    UPPER_T_MOUNT_CLOCK_DEG,
    UPPER_T_POCKET_FRONT_Z_MM,
    VASE_AUTHORITY,
    apply_land_magnets,
    axial_gap_mm,
    axis_placement_facts,
    base_plan,
    base_um_ring_clearance_mm,
    cable_entry_facts,
    cable_entry_opening,
    check_released_mate,
    duct_cutter as _duct_cutter,
    ear_load_path_section_mm2,
    entry_collar_plan,
    insert_front_floor_mm,
    land_facts,
    land_magnet_faces,
    land_radius_at,
    land_solid,
    magnet_facts,
    mate_facts,
    skirt_facts,
    skirt_plan,
    skirt_um_ring_clearance_mm,
)
from .route import TS_CABLE_D_EST, TS_DUCT_D, TS_FREE_CABLE_Z


PART_NAME = "obiwan_bmr_crescent_opposed_TEBM35C10-4"
RELEASE_VARIANT = "Obiwan-TEBM35C10-4-BMR-crescent-opposed"
VARIANT = "opposed"


# --- the two lands -----------------------------------------------------------
# The lower land is the mount land: the drop limit places it, exactly as it
# places the coaxial pod's single land.  The upper land is one vase pitch
# above it, which is the only spacing the two published driver envelopes admit
# when each basket crosses the other's mounting face.
LOWER_AXIS_XY = MOUNT_AXIS_XY
UPPER_AXIS_XY = (0.0, round(MOUNT_AXIS_XY[1] + PAIR_AXIS_PITCH_MM, 9))
AXIS_PITCH_MM = PAIR_AXIS_PITCH_MM
LAND_OVERLAP_MM = round(2.0 * POD_OUTER_R_MM - AXIS_PITCH_MM, 9)
WAIST_HALF_WIDTH_MM = round(
    math.sqrt(POD_OUTER_R_MM ** 2 - (AXIS_PITCH_MM / 2.0) ** 2), 9)
WAIST_Y_MM = round((LOWER_AXIS_XY[1] + UPPER_AXIS_XY[1]) / 2.0, 9)

MAGNET_LANDS = (("lower", LOWER_AXIS_XY[1]), ("upper", UPPER_AXIS_XY[1]))
MAGNET_COUNT = MAGNETS_PER_LAND * len(MAGNET_LANDS)


# --- one 25.1 mm envelope, two opposed pockets -------------------------------
# Straight from the vase: the lower driver mounts on the acoustic front and is
# blind at the rear plane; the upper driver mounts on the rear plane and is
# blind under the acoustic front.  Each keeps its own qualified 1.20 mm wall
# and neither wall is shared with the other.
FRONT_PLANE_Z_MM = THICKNESS_MM
REAR_PLANE_Z_MM = REAR_T_MOUNT_Z_MM                              # -6.8
SECTION_DEPTH_MM = round(FRONT_PLANE_Z_MM - REAR_PLANE_Z_MM, 9)  # 25.1
POD_Z_SPAN = (REAR_PLANE_Z_MM, FRONT_PLANE_Z_MM)
REAR_PROTRUSION_MM = round(CORE_REAR_Z - REAR_PLANE_Z_MM, 9)     # 13.6

LOWER_MOUNT_Z_MM = FRONT_PLANE_Z_MM
LOWER_POCKET_FLOOR_Z_MM = LOWER_T_POCKET_REAR_Z_MM               # -5.6
UPPER_MOUNT_Z_MM = REAR_PLANE_Z_MM
UPPER_POCKET_ROOF_Z_MM = UPPER_T_POCKET_FRONT_Z_MM               # 17.1

LOWER_MOUNT_CLOCK_DEG = LOWER_T_MOUNT_CLOCK_DEG
UPPER_MOUNT_CLOCK_DEG = UPPER_T_MOUNT_CLOCK_DEG


# --- the lead branch between the two chambers --------------------------------
# The vase routes a Ø4.60 branch off its shared main to feed the upper driver.
# Here the shared main is the one mate-face entry duct, which terminates in the
# lower chamber, so the branch runs on from that chamber to the upper one.  It
# is straight, on the line joining the two driver axes, because that is where
# the two pocket bores leave their shortest ligament and where the plan is at
# its thickest either side; and it is at the entry duct's own height, so the
# lead arrives and leaves at one level and never has to climb inside the part.
INTER_POCKET_BRANCH_D_MM = UPPER_T_BRANCH_D_MM
INTER_POCKET_BRANCH_R_MM = round(INTER_POCKET_BRANCH_D_MM / 2.0, 9)
INTER_POCKET_BRANCH_Z_MM = CABLE_DUCT_Z_MM
INTER_POCKET_BRANCH_XY = (0.0, WAIST_Y_MM)
INTER_POCKET_BRANCH_START_Y_MM = round(
    LOWER_AXIS_XY[1] + TEBM_CUTOUT_D_MM / 2.0, 9)
INTER_POCKET_BRANCH_END_Y_MM = round(
    UPPER_AXIS_XY[1] - TEBM_CUTOUT_D_MM / 2.0, 9)
INTER_POCKET_LIGAMENT_MM = round(
    INTER_POCKET_BRANCH_END_Y_MM - INTER_POCKET_BRANCH_START_Y_MM, 9)
BRANCH_FRONT_COVER_MM = round(
    FRONT_PLANE_Z_MM - INTER_POCKET_BRANCH_Z_MM
    - INTER_POCKET_BRANCH_R_MM, 9)
BRANCH_REAR_COVER_MM = round(
    INTER_POCKET_BRANCH_Z_MM - INTER_POCKET_BRANCH_R_MM - REAR_PLANE_Z_MM, 9)
BRANCH_SIDE_COVER_MM = round(
    WAIST_HALF_WIDTH_MM - INTER_POCKET_BRANCH_R_MM, 9)
BRANCH_OVERSHOOT_MM = 1.0

if min(BRANCH_FRONT_COVER_MM, BRANCH_REAR_COVER_MM) < (
        T_BLIND_BACK_WALL_THICKNESS_MM):
    raise RuntimeError(
        "the inter-pocket lead branch leaves less than the vase's qualified "
        f"1.20 mm wall: front={BRANCH_FRONT_COVER_MM}, "
        f"rear={BRANCH_REAR_COVER_MM}")
if INTER_POCKET_LIGAMENT_MM <= INTER_POCKET_BRANCH_D_MM:
    raise RuntimeError(
        "the two pocket bores leave no ligament for the lead branch to run "
        f"through: {INTER_POCKET_LIGAMENT_MM} mm")


def land_radius_at_z(z: float) -> float:
    """Land outer radius at one Z.  It is the D66 land at every Z."""
    return land_radius_at(z, POD_Z_SPAN)


def _pod_solid():
    """Both D66 lands, with their magnet flats, over the 25.1 mm envelope."""
    return _fuse_required(
        land_solid(LOWER_AXIS_XY[1], *POD_Z_SPAN),
        land_solid(UPPER_AXIS_XY[1], *POD_Z_SPAN),
        "upper D66 land onto the lower")


def _inter_pocket_branch_cutter():
    """The one Ø4.60 lead branch, lower chamber to upper chamber."""
    length = (INTER_POCKET_LIGAMENT_MM + 2.0 * BRANCH_OVERSHOOT_MM)
    middle_y = (INTER_POCKET_BRANCH_START_Y_MM
                + INTER_POCKET_BRANCH_END_Y_MM) / 2.0
    # Cylinder() lies on +Z; Rot(X=90) lays it on -Y, so a further 180 degrees
    # about Z points it up the +Y axis the two drivers share.
    return (Pos(INTER_POCKET_BRANCH_XY[0], middle_y, INTER_POCKET_BRANCH_Z_MM)
            * Rot(Z=180.0)
            * Rot(X=90.0)
            * Cylinder(INTER_POCKET_BRANCH_R_MM, length))


def _apply_driver_interfaces(part):
    """Two opposed blind pockets, eight M2 bores, the two cable passages."""
    over = 1.0
    part -= _cylinder_at(
        LOWER_AXIS_XY[0], LOWER_AXIS_XY[1], TEBM_CUTOUT_D_MM / 2.0,
        LOWER_POCKET_FLOOR_Z_MM, FRONT_PLANE_Z_MM + over)
    part -= _cylinder_at(
        UPPER_AXIS_XY[0], UPPER_AXIS_XY[1], TEBM_CUTOUT_D_MM / 2.0,
        REAR_PLANE_Z_MM - over, UPPER_POCKET_ROOF_Z_MM)

    patterns = (
        (LOWER_AXIS_XY[1], LOWER_MOUNT_CLOCK_DEG,
         FRONT_PLANE_Z_MM - M2_INSERT_DEPTH_MM, FRONT_PLANE_Z_MM),
        (UPPER_AXIS_XY[1], UPPER_MOUNT_CLOCK_DEG,
         REAR_PLANE_Z_MM, REAR_PLANE_Z_MM + M2_INSERT_DEPTH_MM),
    )
    radius = TEBM_MOUNT_PCD_MM / 2.0
    for axis_y, clock, bore_z_min, bore_z_max in patterns:
        for index in range(TEBM_MOUNT_HOLE_COUNT):
            angle = math.radians(clock + 90.0 * index)
            part -= _cylinder_at(
                radius * math.cos(angle),
                axis_y + radius * math.sin(angle),
                M2_INSERT_BORE_D_MM / 2.0, bore_z_min, bore_z_max)

    part -= _duct_cutter()
    part -= _inter_pocket_branch_cutter()
    return part


def bmr_crescent_opposed_body():
    """The finished exterior, before any captive cavity is buried in it."""
    _require_guarded_build()
    check_released_mate()

    part = _fuse_required(
        _pod_solid(), _plan_prism(skirt_plan(), *SKIRT_Z),
        "flush junction skirt onto the opposed BMR pod")
    part = _fuse_required(
        part, _plan_prism(entry_collar_plan(), *ENTRY_COLLAR_Z),
        "mate-face cable entry collar onto the opposed BMR pod")
    part = _enforce_junction_plan_ownership(part, "t_um", "tweeter")
    part = _apply_complete_um_tweeter_joint(part, "tweeter")
    part = _apply_driver_interfaces(part)

    part = part.clean()
    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1 or solids[0].volume <= 0.01:
        raise RuntimeError(
            "opposed BMR pod finalization must retain every required "
            f"feature; valid={part.is_valid} volumes="
            f"{[solid.volume for solid in part.solids()]}")
    return Part([solids[0]])


def bmr_crescent_opposed():
    """Two opposed BMR lands flush-skirted onto the released UM half-lap."""
    return build_model().solid


@dataclass(frozen=True)
class BmrCrescentOpposedModel:
    """Authoritative solid plus the exact four captive-station records."""

    solid: object
    magnet_tools: tuple = field(default=())


def build_model() -> BmrCrescentOpposedModel:
    part, magnet_tools = apply_land_magnets(
        bmr_crescent_opposed_body(), MAGNET_LANDS)
    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1:
        raise RuntimeError(
            "burying the captive magnets must leave one valid solid; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return BmrCrescentOpposedModel(
        solid=Part([solids[0]]), magnet_tools=magnet_tools)


def declared_openings() -> list[dict]:
    """Every intentional break in the skin, with its authority and side.

    ``exposure`` is the gate: ``um_mate`` faces the collar across the released
    mate gap, ``driver_face`` is under a fitted driver, ``internal`` never
    reaches the skin at all.  Nothing is allowed to be ``exterior``.  The four
    captive magnet cavities are not openings at all -- they are sealed voids
    behind the qualified 0.45 mm skin -- so they are declared in the magnet
    block instead.
    """
    return [
        {
            "name": "lower_driver_pocket_mouth",
            "kind": "driver_pocket",
            "exposure": "driver_face",
            "face": "acoustic_front_z_18p3",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(LOWER_AXIS_XY),
            "z_span_mm": [LOWER_POCKET_FLOOR_Z_MM, FRONT_PLANE_Z_MM],
        },
        {
            "name": "upper_driver_pocket_mouth",
            "kind": "driver_pocket",
            "exposure": "driver_face",
            "face": "bmr_rear_plane_z_minus_6p8",
            "diameter_mm": TEBM_CUTOUT_D_MM,
            "axis_xy_mm": list(UPPER_AXIS_XY),
            "z_span_mm": [REAR_PLANE_Z_MM, UPPER_POCKET_ROOF_Z_MM],
        },
        cable_entry_opening(),
        {
            "name": "inter_pocket_lead_branch",
            "kind": "cable_pass",
            "exposure": "internal",
            "diameter_mm": INTER_POCKET_BRANCH_D_MM,
            "diameter_authority": (
                "vase UPPER_T_BRANCH_D_MM, one driver's lead branch"),
            "axis_xy_mm": list(INTER_POCKET_BRANCH_XY),
            "z_mm": INTER_POCKET_BRANCH_Z_MM,
            "y_span_mm": [INTER_POCKET_BRANCH_START_Y_MM,
                          INTER_POCKET_BRANCH_END_Y_MM],
            "length_mm": INTER_POCKET_LIGAMENT_MM,
            "front_cover_mm": BRANCH_FRONT_COVER_MM,
            "rear_cover_mm": BRANCH_REAR_COVER_MM,
            "side_cover_mm": BRANCH_SIDE_COVER_MM,
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


def design_facts(magnet_tools: tuple = ()) -> dict:
    """Envelope, mate coordinates, hidden cable path and candidate flags."""
    return {
        "part": PART_NAME,
        "release_variant": RELEASE_VARIANT,
        "variant": VARIANT,
        "print_orientation": PRINT_ORIENTATION,
        "release_authorized": RELEASE_AUTHORIZED,
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
        "status": "candidate_not_release_authorized",
        "counts_against_release_inventory": False,
        "magnet_count": MAGNET_COUNT,
        "magnets": magnet_facts(magnet_tools, MAGNET_LANDS),
        "silhouette": {
            "shape": "two_opposed_lands_flush_junction_skirt_two_ears",
            "inherits_released_crescent_outline": False,
            "adopted_from_the_qualified_vase": [
                "the 49.3 mm opposed axis pitch",
                "lower driver on the acoustic front, upper on the rear",
                "one 25.1 mm envelope with a 1.20 mm blind wall per driver",
                "the D66 lands and their two captive magnet flats",
                "the +45/-45 degree M2 clocks",
                "shared route to the lower pocket, branch to the upper",
            ],
            "adopted_from_the_coaxial_pod": [
                "the released UM half-lap mate and its drop limit",
                "the flush junction skirt and its wing keep-out",
                "the hidden Ø6.00 mate-face entry and its stadium collar",
            ],
            "minimalism_rule": (
                "no material beyond the two lands and the flush fill: the "
                "plan is the two D66 lands, the two bosses, the convex fill "
                "between the lower land and them, the released closure web, "
                "and nothing else"),
        },
        "driver": {
            "model": "Tectonic TEBM35C10-4",
            "count": 2,
            "arrangement": "opposed_side_by_side_two_axes",
            "lower_axis_xy_mm": list(LOWER_AXIS_XY),
            "upper_axis_xy_mm": list(UPPER_AXIS_XY),
            "axis_pitch_mm": AXIS_PITCH_MM,
            "axis_pitch_authority": (
                "vase PAIR_AXIS_PITCH_MM: half a D54 flange plus half a "
                "D43.6 basket plus 0.50 mm, because each basket crosses the "
                "other driver's mounting face"),
            "mount_axis_authority": (
                "the lower land is dropped until its wall keeps the vase's "
                "1.20 mm wall outside the UM half-lap receiver notch"),
            "released_axis_xy_mm": list(RELEASED_AXIS_XY),
            "drop_below_released_axis_mm": POD_DROP_MM,
            "um_to_lower_axis_spacing_mm": UM_AXIS_SPACING_MM,
            "um_to_upper_axis_spacing_mm": round(
                UPPER_AXIS_XY[1] - LOWER_AXIS_XY[1] + UM_AXIS_SPACING_MM, 9),
            "released_um_to_tweeter_axis_spacing_mm": (
                RELEASED_UM_AXIS_SPACING_MM),
            "lower_driver_faces": "+z",
            "upper_driver_faces": "-z",
            "depth_mm": TEBM_DEPTH_MM,
            "max_flange_d_mm": TEBM_MAX_D_MM,
            "basket_d_mm": TEBM_BASKET_D_MM,
            "cutout_d_mm": TEBM_CUTOUT_D_MM,
            "land_d_mm": TEBM_LAND_D_MM,
            "mount_pcd_mm": TEBM_MOUNT_PCD_MM,
            "mount_hole_count": TEBM_MOUNT_HOLE_COUNT,
            "mount_clock_deg": {
                "lower": LOWER_MOUNT_CLOCK_DEG,
                "upper": UPPER_MOUNT_CLOCK_DEG,
            },
            "pair_mass_g": 2.0 * TEBM_MASS_G,
        },
        "axis_placement": axis_placement_facts(),
        "depth_stack": {
            "acoustic_front_z_mm": FRONT_PLANE_Z_MM,
            "rear_plane_z_mm": REAR_PLANE_Z_MM,
            "section_depth_mm": SECTION_DEPTH_MM,
            "section_rule": (
                "one 25.1 mm envelope carries both drivers, exactly as on "
                "the qualified vase; no wall is shared between them"),
            "lower_mount_z_mm": LOWER_MOUNT_Z_MM,
            "lower_pocket_floor_z_mm": LOWER_POCKET_FLOOR_Z_MM,
            "upper_mount_z_mm": UPPER_MOUNT_Z_MM,
            "upper_pocket_roof_z_mm": UPPER_POCKET_ROOF_Z_MM,
            "blind_wall_mm": T_BLIND_BACK_WALL_THICKNESS_MM,
            "clear_pocket_depth_mm": T_CLEAR_POCKET_DEPTH_MM,
            "rear_protrusion_behind_core_rear_mm": REAR_PROTRUSION_MM,
        },
        "pod": {
            **land_facts(),
            "lands": 2,
            "land_overlap_mm": LAND_OVERLAP_MM,
            "waist_half_width_mm": WAIST_HALF_WIDTH_MM,
            "waist_y_mm": WAIST_Y_MM,
            "waist_note": (
                "the two D66 circles overlap by 16.7 mm, so the body is one "
                "continuous plan and its narrowest section between the "
                "driver axes is the 43.88 mm waist"),
        },
        "skirt": skirt_facts(),
        "mate": mate_facts(),
        "cable": {
            **cable_entry_facts(),
            "duct_feeds": "lower_front_chamber",
            "branch_d_mm": INTER_POCKET_BRANCH_D_MM,
            "branch_axis_xy_mm": list(INTER_POCKET_BRANCH_XY),
            "branch_z_mm": INTER_POCKET_BRANCH_Z_MM,
            "branch_y_span_mm": [INTER_POCKET_BRANCH_START_Y_MM,
                                 INTER_POCKET_BRANCH_END_Y_MM],
            "branch_length_mm": INTER_POCKET_LIGAMENT_MM,
            "branch_front_cover_mm": BRANCH_FRONT_COVER_MM,
            "branch_rear_cover_mm": BRANCH_REAR_COVER_MM,
            "branch_side_cover_mm": BRANCH_SIDE_COVER_MM,
            "branch_rule": (
                "straight, on the line joining the two driver axes and at "
                "the entry duct's own height, so it crosses the thickest "
                "part of the plan through the shortest ligament the two "
                "pocket bores leave and the lead never changes level"),
            "note": (
                "the free T cable leaves the UM's declared central mouth and "
                "goes straight into the one duct, whose mouth sits on this "
                "part's R51.90 mate face along the cable's own tangent; the "
                "upper driver is fed from the lower chamber through the one "
                "declared branch.  Nothing about the cable is visible on the "
                "assembled exterior"),
        },
        "declared_openings": declared_openings(),
        "exterior_openings": [
            opening["name"] for opening in declared_openings()
            if opening["exposure"] == "exterior"
        ],
    }


def obiwan_bmr_opposed_attachments():
    _require_guarded_build()
    return {"addon_bmr_crescent_opposed": bmr_crescent_opposed()}


def gen_step():
    _require_guarded_build()
    return ordered_labeled_compound(
        obiwan_bmr_opposed_attachments(),
        label="lx521_obiwan_r6f_bmr_crescent_opposed_candidate")
