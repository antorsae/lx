"""Candidate Obi-Wan BMR pod for two coaxial back-to-back TEBM35C10-4 BMRs.

This is one of two candidate alternatives to the released ND25FW-4 tweeter
crescent, not a replacement for either.  Its sibling,
``bmr_crescent_opposed``, carries the same two drivers on the qualified vase's
side-by-side layout; everything the two share -- the mount, preserved axis,
the flush skirt, hidden cable entry, selected land and captive magnets
-- lives in ``bmr_pod`` and is imported, not restated.

This variant presents the *identical* half-lap interface to an unmodified
Obi-Wan UM collar (x=+/-24, y=421.5, complete front local-D9.8 ears,
standalone blind D4.6 x 4.0 heat-set receivers, 1.9 mm acoustic-front floors,
0.20 mm axial gap), so it is swappable with the released crescent without
touching the UM print.

Where the released crescent is a full acoustic silhouette carrying a
face-to-face Dayton pair, this part keeps only what the two BMRs, that one
mate and one hidden cable actually need:

* a D63 pod carrying the front driver on the shared z=18.3 plane and the rear
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

The acoustic axis stays fixed
-----------------------------
The first candidate parked the pod on the released ND25FW-4 acoustic axis and
hung it off two struts, which left an open window between the pod and the UM
collar.  Its later BMR acoustic axis is now an explicit datum and does not
move when the land is reduced.  ``bmr_pod`` recomputes clearance to the UM
ring and receiver ears for each topology and rejects one that violates them.

The reduced land
----------------
The default pod wall is the conservative D63 prototype land.  Both mounting
faces use one constant plan through the stack, so the front-face-down print
never grows rearward.  The optional BMR-slim plan instead follows a D56 core,
with four local M2 pads and discrete side lobes at the same magnet datums.

The land carries the vase's two magnet flats with it, because a magnet needs a
plane and a cylinder only offers a tangent line.  The D63 faces sit at
x=+/-31.3267 mm, 4.3267 mm outside the conservative D54 envelope.

Two captive magnets, on the front land
--------------------------------------
The vase buries four captive D5 x 2 discs, two per driver land, on those flats at
the project-wide source Z=15.10.  This variant has one land facing the world,
so it takes that land's pair -- the vase's lower/front stations, at the same
land-local coordinates, through the same ``magnets.py`` helper.  They are
sealed voids behind the qualified 0.45 mm face skin: nothing about them
reaches the exterior, and the parts remain candidates, so the released magnet
catalog, the release counts and the slicing profiles are untouched.

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
partition, the dropped acoustic axis, the hidden cable path, the two captive
magnet stations and the two-screw joint demand under roughly twice the
released crescent's hanging mass all need physical qualification before this
part is printed for use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from build123d import Part

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
    BMR_SLIM_LAND,
    BmrLandTopology,
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
    FULL_CIRCULAR_LAND,
    LOWER_T_MOUNT_CLOCK_DEG,
    LOWER_T_POCKET_REAR_Z_MM,
    M2_INSERT_BORE_D_MM,
    M2_INSERT_DEPTH_MM,
    MAGNETS_PER_LAND,
    MOUNT_AXIS_XY,
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
    resolve_land_topology,
    skirt_facts,
    skirt_plan,
    skirt_um_ring_clearance_mm,
)
from .route import TS_CABLE_D_EST, TS_DUCT_D, TS_FREE_CABLE_Z


PART_NAME = "obiwan_bmr_crescent_TEBM35C10-4"
RELEASE_VARIANT = "Obiwan-TEBM35C10-4-BMR-crescent"
VARIANT = "coaxial"

# One driver land faces the world, so this variant takes that land's captive
# pair: the vase's lower/front stations at the same land-local coordinates.
MAGNET_LANDS = (("front", MOUNT_AXIS_XY[1]),)
MAGNET_COUNT = MAGNETS_PER_LAND * len(MAGNET_LANDS)

# The single driver axis, which is also the mount land's axis.
BMR_AXIS_XY = MOUNT_AXIS_XY


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
POD_Z_SPAN = (REAR_MOUNT_Z_MM, THICKNESS_MM)

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


def pod_radius_at(
    z: float,
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
) -> float:
    """Pod core radius at one Z; its plan is constant through the stack."""
    return land_radius_at(z, POD_Z_SPAN, topology)


def _pod_solid(
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
):
    """The selected land, including its two side-magnet faces."""
    return land_solid(BMR_AXIS_XY[1], *POD_Z_SPAN, topology)


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


def bmr_crescent_body(
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
):
    """The finished exterior, before any captive cavity is buried in it."""
    _require_guarded_build()
    check_released_mate()

    part = _fuse_required(
        _pod_solid(topology), _plan_prism(skirt_plan(topology), *SKIRT_Z),
        "flush junction skirt onto the BMR pod")
    part = _fuse_required(
        part, _plan_prism(entry_collar_plan(topology), *ENTRY_COLLAR_Z),
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


def bmr_crescent(
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
):
    """Dropped BMR pod flush-skirted onto the released UM half-lap mate."""
    return build_model(topology).solid


@dataclass(frozen=True)
class BmrCrescentModel:
    """Authoritative solid plus the exact captive-station records."""

    solid: object
    magnet_tools: tuple = field(default=())


def build_model(
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
) -> BmrCrescentModel:
    # Magnets go last, into a finished exterior: ``apply_wall_cavity`` refuses
    # a host that does not already carry the complete captive land, and only
    # ever subtracts, so no station can move a surface.
    part, magnet_tools = apply_land_magnets(
        bmr_crescent_body(topology), MAGNET_LANDS, topology)
    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1:
        raise RuntimeError(
            "burying the captive magnets must leave one valid solid; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in solids]}")
    return BmrCrescentModel(
        solid=Part([solids[0]]), magnet_tools=magnet_tools)


def declared_openings() -> list[dict]:
    """Every intentional break in the skin, with its authority and side.

    ``exposure`` is the gate: ``um_mate`` faces the collar across the released
    mate gap, ``driver_face`` is under a fitted driver, ``internal`` never
    reaches the skin at all.  Nothing is allowed to be ``exterior``.  The two
    captive magnet cavities are not openings at all -- they are sealed voids
    behind the qualified 0.45 mm skin -- so they are declared in the magnet
    block instead.
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
        cable_entry_opening(),
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


def design_facts(
    magnet_tools: tuple = (),
    topology: str | BmrLandTopology = FULL_CIRCULAR_LAND,
) -> dict:
    """Envelope, mate coordinates, hidden cable path and candidate flags."""
    land_spec = resolve_land_topology(topology)
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
        "magnets": magnet_facts(magnet_tools, MAGNET_LANDS, topology),
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
                "preserved BMR acoustic datum; actual selected-land "
                "clearance to the UM ring and receiver notch is gated"),
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
            "land_d_mm": land_spec.parent_d_mm,
            "land_core_d_mm": land_spec.core_d_mm,
            "mount_pcd_mm": TEBM_MOUNT_PCD_MM,
            "mount_hole_count": TEBM_MOUNT_HOLE_COUNT,
            "mount_clock_deg": {
                "front": FRONT_MOUNT_CLOCK_DEG,
                "rear": REAR_MOUNT_CLOCK_DEG,
            },
            "pair_mass_g": 2.0 * TEBM_MASS_G,
        },
        "axis_placement": axis_placement_facts(topology),
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
        "pod": land_facts(topology),
        "skirt": skirt_facts(topology),
        "mate": mate_facts(),
        "cable": {
            **cable_entry_facts(topology),
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
