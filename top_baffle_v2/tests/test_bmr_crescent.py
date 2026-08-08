"""Focused source/BREP contract for the two candidate BMR pods.

Two variants share one mount, one skirt, one hidden cable entry and one
captive-magnet system, and differ only in how the two TEBM35C10-4 BMRs are
arranged:

* ``coaxial`` -- ``obiwan_bmr_crescent_TEBM35C10-4``, both drivers back to
  back on one axis, 50.2 mm deep, two captive magnets on its one outward land;
* ``opposed`` -- ``obiwan_bmr_crescent_opposed_TEBM35C10-4``, the qualified
  vase's side-by-side layout on the same crescent mount, 25.1 mm deep and much
  taller, with all four of the vase's captive magnets.

Almost every gate below runs against both.  Pure gates (constants,
vase-authority equality, candidate flags) always run.  Geometry gates read the
exported BREPs under ``build/bmr_crescent_TEBM35C10-4/`` and are skipped with
an explicit message when one is absent; they refuse to pass against a stale
export.  The staged gates additionally need the hash-verified Obi-Wan stage
BREPs.

Neither part is a superset of the released ND25FW-4 crescent -- they keep that
crescent's *mount* and its junction *seam*, and nothing else -- so the mate is
proven by asserting the two ear footprints are geometrically identical and by
assembling each part against the staged UM collar, not by differencing whole
silhouettes.

Gates carrying the flush junction specifically: the mount axis is recomputed
from the two released constraints and has to be the tighter of them; each
assembly is projected head on against the staged collar and the plan is walked
column by column for windows; every declared opening has to name the side it
faces, with no exterior ones; and the free T cable has to reach the declared
mate-face entry without being touched or exposed on the way.

Gates carrying the captive magnets: every station has to sit at the vase's own
land-local coordinate, read back from the real vase in a proud-profile
subprocess; each cavity has to be a sealed void behind the qualified 0.45 mm
skin, which is checked both as a shell count and by probing the skin itself;
and neither candidate may appear in the released catalog or move its counts.

Run with::

    LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan python tests/test_bmr_crescent.py
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


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
from lx521_baffle.obiwan import bmr_pod
from lx521_baffle.obiwan import bmr_crescent as coaxial
from lx521_baffle.obiwan import bmr_crescent_opposed as opposed
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
from lx521_baffle.magnet_contract import (
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    MAGNET_DEPTH_MM,
    MAGNET_DIAMETER_MM,
)

BUILD_ROOT = PROJECT_ROOT / "build" / "bmr_crescent_TEBM35C10-4"
STAGE_STATES = ("floor_stand", "no_floor_stand")
BED_LIMIT_MM = 256.0
BASE_SLICING_PROFILE = PROJECT_ROOT / "captive_magnet_slicing_profile.json"
PETG_GF_SLICING_PROFILE = (
    PROJECT_ROOT / "captive_magnet_slicing_profile_petg_gf.json")

# The released totals both candidates' deliveries must leave alone.  A
# candidate that slices its own pauses is exactly the situation in which
# somebody might be tempted to move one of these.
RELEASED_ARTIFACT_TOTAL = 58
RELEASED_SHELF_PAIRS = 51

# The released captive-magnet totals.  Both candidates bury real stations, and
# neither may move these; the number is restated here so that wiring a
# candidate into the release has to break this test on the way past.
RELEASED_MAGNET_TOTAL = 94

SKIPPED: list[str] = []


@dataclass(frozen=True)
class Variant:
    """One candidate pod and the Z ladder its own silhouette needs."""

    key: str
    module: Any
    magnet_count: int
    z_span: tuple[float, float]
    silhouette_ladder: tuple[float, ...]


# The ladders straddle every plane where the exterior changes: the acoustic
# front, the ear step at z=12.40, the core rear plane at z=6.80 where the
# skirt ends, the entry collar's own floor at z=-0.40, and then the body down
# to its rear land.  A pair either side of each is what turns "never grows
# rearward" into a statement about the steps rather than about the flats.
VARIANTS = (
    Variant(
        key="coaxial", module=coaxial, magnet_count=2,
        z_span=(coaxial.REAR_MOUNT_Z_MM, THICKNESS_MM),
        silhouette_ladder=(
            18.29, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.45, 12.35, 12.0,
            11.0, 10.0, 9.0, 8.0, 7.0, 6.85, 6.75, 6.5, 5.0, 2.0, 0.0,
            -0.35, -0.45, -1.0, -10.0, -20.0, -31.85,
        )),
    Variant(
        key="opposed", module=opposed, magnet_count=4,
        z_span=(opposed.REAR_PLANE_Z_MM, THICKNESS_MM),
        silhouette_ladder=(
            18.29, 18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.45, 12.35, 12.0,
            11.0, 10.0, 9.0, 8.0, 7.0, 6.85, 6.75, 6.5, 5.0, 2.0, 0.0,
            -0.35, -0.45, -1.0, -3.0, -5.0, -6.79,
        )),
)


def _skip(reason: str) -> None:
    SKIPPED.append(reason)
    print(f"  SKIP {reason}")


def _close(left: float, right: float, tolerance: float = 1.0e-9) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _brep(variant: Variant) -> Path:
    return BUILD_ROOT / f"{variant.module.PART_NAME}.brep"


def _facts_path(variant: Variant) -> Path:
    return BUILD_ROOT / f"{variant.module.PART_NAME}.facts.json"


def _delivery_paths(variant: Variant) -> dict[str, Path]:
    """Every file the local-only Bambu delivery reads or promotes."""
    stem = variant.module.PART_NAME
    slug = f"shared_{variant.module.RELEASE_VARIANT}_{stem}"
    slices = BUILD_ROOT / f"slice_audit_{variant.key}" / "slices" / slug
    return {
        "catalog": BUILD_ROOT / f"{stem}.catalog.json",
        "profile": BUILD_ROOT / f"{stem}.slicing_profile.json",
        "facts": _facts_path(variant),
        "project": BUILD_ROOT / f"{stem}.gcode.3mf",
        "audit": slices / "captive_magnet_slice_audit.json",
        "gcode": slices / "ready" / "plate_1.gcode",
    }


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Pure gates
# --------------------------------------------------------------------------

def _vase_in_a_proud_subprocess(program: str, argument: str) -> Any:
    """Evaluate one expression against the real vase under its own profile."""
    environment = dict(os.environ)
    environment["LX_ROUTING_PROFILE"] = "proud"
    environment["LX_STAND_FOOT"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", program, argument],
        capture_output=True, text=True, env=environment, cwd=PROJECT_ROOT,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "could not evaluate the vase authority under LX_ROUTING_PROFILE="
            f"proud:\n{completed.stderr.strip()}")
    return json.loads(completed.stdout)


def test_vase_authority_is_mirrored_exactly() -> None:
    """Every mirrored driver and magnet constant equals the real vase's value.

    ``proud.vase_tebm35c10_4`` cannot be imported beside an obiwan-profile
    part, so the vase is evaluated in a proud-profile subprocess and compared
    value by value.  The mirror now carries the captive-magnet flat as well as
    the driver envelope, because both variants take their magnet stations from
    the vase; a drift in either fails here instead of silently leaving two
    disagreeing definitions in the tree.
    """
    program = (
        "import json, sys;"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r});"
        "from lx521_baffle.proud import vase_tebm35c10_4 as v;"
        "names = json.loads(sys.argv[1]);"
        "print(json.dumps({n: getattr(v, n) for n in names}))"
    )
    vase = _vase_in_a_proud_subprocess(
        program, json.dumps(sorted(bmr_pod.VASE_AUTHORITY)))
    for name, mirrored in sorted(bmr_pod.VASE_AUTHORITY.items()):
        assert _close(vase[name], mirrored), (
            f"{name} drifted from the vase: vase={vase[name]} "
            f"bmr_pod={mirrored}")
    # The magnet block has to be in there, or "mirrored exactly" would be a
    # claim about the driver only.
    for name in ("T_MAGNET_FACE_X_MM", "T_MAGNET_FLAT_HALF_HEIGHT_MM",
                 "T_MAGNET_TOTAL", "PAIR_AXIS_PITCH_MM"):
        assert name in bmr_pod.VASE_AUTHORITY
    print(f"    {len(bmr_pod.VASE_AUTHORITY)} vase constants mirrored exactly, "
          f"driver and captive-magnet flat alike")


def test_the_two_variants_share_one_family_module() -> None:
    """The mount, skirt, entry and drop machinery must not have been forked.

    Both variants publish the same family names.  Asserting object identity,
    not equality, is what makes that structural: a copy of ``skirt_plan`` in
    one variant would read back the same numbers today and drift tomorrow.
    """
    shared = (
        "MOUNT_AXIS_XY", "UM_MATE_R_MM", "EAR_NOTCH_R_MM",
        "AXIS_Y_LIMIT_FROM_UM_RING_MM", "AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM",
        "POD_OUTER_R_MM", "POD_FLAT_HALF_WIDTH_MM", "SKIRT_Z",
        "CABLE_ENTRY_XY", "CABLE_DUCT_DIR", "ENTRY_COLLAR_R_MM",
        "T_MAGNET_FACE_X_MM", "VASE_AUTHORITY",
        "skirt_plan", "base_plan", "entry_collar_plan",
        "ear_load_path_section_mm2", "apply_land_magnets",
        "land_magnet_faces", "land_solid", "check_released_mate",
    )
    for name in shared:
        family = getattr(bmr_pod, name)
        for variant in VARIANTS:
            here = getattr(variant.module, name)
            assert here is family, (
                f"{variant.key} has its own {name} instead of the family's")
    # And the two really are different parts, not one module twice.
    assert coaxial.PART_NAME != opposed.PART_NAME
    assert coaxial.RELEASE_VARIANT != opposed.RELEASE_VARIANT
    print(f"    {len(shared)} family names are one object shared by both "
          "variants")


def test_mount_constants_equal_the_released_joint_authority(
        variant: Variant) -> None:
    """Numeric identity with the released UM half-lap interface.

    This is the non-negotiable part of both designs: each part must swap onto
    an unmodified UM collar, so every one of these comes from the released
    joint authority in ``obiwan.carriers`` and none may be restated locally.
    """
    facts = variant.module.design_facts()["mate"]
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
    # The rear-driven passage is the UM's; neither pod may ever own it.
    assert TWEETER_CORE_BORE_TOP_Z > TWEETER_ADDON_JOINT_Z[0]


def test_depth_stack_is_the_variant_it_claims(variant: Variant) -> None:
    """Pocket datums, blind walls and how deep the part actually goes."""
    module = variant.module
    if variant.key == "coaxial":
        # Two full driver envelopes back to back.
        assert _close(module.FRONT_MOUNT_Z_MM, THICKNESS_MM)
        assert _close(module.FRONT_MOUNT_Z_MM, 18.3)
        assert _close(module.REAR_MOUNT_Z_MM, -31.9)
        assert _close(module.STACK_DEPTH_MM, 2.0 * bmr_pod.TEBM_DEPTH_MM)
        assert _close(module.STACK_DEPTH_MM, 50.2)
        # Each driver keeps a full 1.20 mm blind wall of its own.
        assert _close(module.PARTITION_THICKNESS_MM,
                      2.0 * bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM)
        assert _close(module.PARTITION_THICKNESS_MM, 2.4)
        assert _close(module.FRONT_POCKET_FLOOR_Z_MM, -5.6)
        assert _close(module.REAR_POCKET_ROOF_Z_MM, -8.0)
        assert _close(
            module.FRONT_MOUNT_Z_MM - module.FRONT_POCKET_FLOOR_Z_MM,
            bmr_pod.T_CLEAR_POCKET_DEPTH_MM)
        assert _close(
            module.REAR_POCKET_ROOF_Z_MM - module.REAR_MOUNT_Z_MM,
            bmr_pod.T_CLEAR_POCKET_DEPTH_MM)
        assert _close(module.REAR_PROTRUSION_MM,
                      CORE_REAR_Z - module.REAR_MOUNT_Z_MM)
        walls = (THICKNESS_MM, module.FRONT_POCKET_FLOOR_Z_MM)
    else:
        # One 25.1 mm envelope, the vase's own, carrying both drivers.
        assert _close(module.FRONT_PLANE_Z_MM, THICKNESS_MM)
        assert _close(module.REAR_PLANE_Z_MM, -6.8)
        assert _close(module.SECTION_DEPTH_MM, bmr_pod.TEBM_DEPTH_MM)
        assert _close(module.SECTION_DEPTH_MM, 25.1)
        assert _close(module.LOWER_POCKET_FLOOR_Z_MM, -5.6)
        assert _close(module.UPPER_POCKET_ROOF_Z_MM, 17.1)
        # Each pocket is blind at the face opposite its own mount, and both
        # blind walls are the vase's qualified 1.20 mm.  Neither is shared.
        assert _close(
            module.LOWER_POCKET_FLOOR_Z_MM - module.REAR_PLANE_Z_MM,
            bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM)
        assert _close(
            module.FRONT_PLANE_Z_MM - module.UPPER_POCKET_ROOF_Z_MM,
            bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM)
        assert _close(
            module.FRONT_PLANE_Z_MM - module.LOWER_POCKET_FLOOR_Z_MM,
            bmr_pod.T_CLEAR_POCKET_DEPTH_MM)
        assert _close(
            module.UPPER_POCKET_ROOF_Z_MM - module.REAR_PLANE_Z_MM,
            bmr_pod.T_CLEAR_POCKET_DEPTH_MM)
        assert _close(module.REAR_PROTRUSION_MM, 13.6)
        walls = (THICKNESS_MM, module.REAR_PLANE_Z_MM)
    # The cable duct runs through the mount land's own chamber, and clears
    # both bounding walls of that chamber by a real ligament.
    for wall_z in walls:
        ligament = abs(bmr_pod.CABLE_DUCT_Z_MM - wall_z) - bmr_pod.CABLE_DUCT_R_MM
        assert ligament >= bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM, (
            f"the cable duct at z={bmr_pod.CABLE_DUCT_Z_MM} leaves only "
            f"{ligament:.3f} mm to the wall at z={wall_z}")


def test_pod_outer_wall_is_the_driver_land(variant: Variant) -> None:
    """The wall is the D66 land itself, and that is the printable minimum.

    Every mounting face has to carry the qualified D66 land and both parts
    print front-face-down, so the plan may never grow rearward.  Anything
    under R33 loses land; anything over R33 at the front cannot come back
    down to R33 at the rear without an overhang.  A straight D66 cylinder is
    the only radius that satisfies both, which is what this checks -- together
    with the two flats the vase cuts into it so each captive magnet has a
    plane to seat behind rather than a tangent to a circle.
    """
    module = variant.module
    assert _close(bmr_pod.POD_OUTER_R_MM, bmr_pod.TEBM_LAND_R_MM)
    assert _close(bmr_pod.POD_OUTER_D_MM, 66.0)
    radius_at = (module.pod_radius_at if variant.key == "coaxial"
                 else module.land_radius_at_z)
    low, high = variant.z_span
    for z in (low, (low + high) / 2.0, CORE_REAR_Z, high):
        assert _close(radius_at(z), bmr_pod.TEBM_LAND_R_MM), (
            "the land must be a straight cylinder; a varying radius either "
            "loses land or leans the wrong way for this print orientation")
    # The wall the land leaves is far above any minimum-wall rule.
    assert _close(bmr_pod.POD_WALL_OVER_POCKET_MM, 11.537)
    assert _close(bmr_pod.POD_WALL_OVER_INSERT_MM, 7.27)
    assert bmr_pod.POD_WALL_OVER_POCKET_MM >= 2.4
    assert bmr_pod.POD_WALL_OVER_INSERT_MM >= 2.4
    assert _close(bmr_pod.POD_LAND_MARGIN_OVER_FLANGE_MM, 6.0)
    # The flat is exactly the vase's, and it takes so little off the land
    # that the D54 flange still lands on 5.8 mm of it at its narrowest.
    assert _close(bmr_pod.POD_FLAT_HALF_WIDTH_MM, bmr_pod.T_MAGNET_FACE_X_MM)
    assert _close(bmr_pod.POD_FLAT_DEPTH_MM, 0.165414575, 1.0e-6)
    assert bmr_pod.POD_FLAT_MARGIN_OVER_FLANGE_MM > 5.8
    # And the flat is wide enough for the whole captive land, which is what
    # makes the magnet interface planar rather than nearly planar.
    land_half_width = CAVITY_DIAMETER_MM / 2.0 + 0.60
    assert bmr_pod.T_MAGNET_FLAT_HALF_HEIGHT_MM >= land_half_width, (
        f"the flat is {bmr_pod.T_MAGNET_FLAT_HALF_HEIGHT_MM} mm half-high but "
        f"the captive land needs {land_half_width} mm")
    print(f"    land D{bmr_pod.POD_OUTER_D_MM:.0f} with two "
          f"{bmr_pod.POD_FLAT_DEPTH_MM:.3f} mm magnet flats; "
          f"{bmr_pod.POD_WALL_OVER_POCKET_MM:.3f} mm wall outside the pocket, "
          f"{bmr_pod.POD_WALL_OVER_INSERT_MM:.3f} mm outside each M2 bore")


def test_pod_is_dropped_as_far_as_the_mate_allows(variant: Variant) -> None:
    """The mount axis is set by the tighter of two UM constraints, not chosen.

    Two things stop the drop: the released 0.20 mm clearance on the UM's
    native R51.7 core ring, and the UM half-lap's own receiver notch, which
    the D66 land may not be nicked by.  Both are computed here from the same
    released datums the parts use, and the axis has to be the larger.  Both
    variants mount on the same land, so both get the same answer.
    """
    ring = (UM_CUTOUT[1] + UM_CORE_R + bmr_pod.UM_MATE_GAP_MM
            + bmr_pod.POD_OUTER_R_MM)
    notch = TWEETER_JOINT_Y + math.sqrt(
        (bmr_pod.POD_OUTER_R_MM + bmr_pod.EAR_NOTCH_R_MM
         + bmr_pod.EAR_NOTCH_LIGAMENT_MM) ** 2 - TWEETER_JOINT_X[1] ** 2)
    assert _close(bmr_pod.AXIS_Y_LIMIT_FROM_UM_RING_MM, ring, 1.0e-6)
    assert _close(bmr_pod.AXIS_Y_LIMIT_FROM_EAR_NOTCH_MM, notch, 1.0e-6)
    assert _close(bmr_pod.MOUNT_AXIS_XY[1], max(ring, notch), 1.0e-6)
    assert bmr_pod.AXIS_GOVERNING_CONSTRAINT == "um_half_lap_receiver_notch"
    assert _close(bmr_pod.MOUNT_AXIS_XY[0], 0.0)
    assert _close(bmr_pod.MOUNT_AXIS_XY[1], 452.494193004, 1.0e-6)

    # The notch's ligament is the vase's own qualified minimum wall, and the
    # D66 land really does clear it: a nick there would show up at z=6.7 as
    # rearward plan growth, which this print orientation cannot take.
    assert _close(bmr_pod.EAR_NOTCH_R_MM,
                  TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 + TWEETER_JOINT_CLEAR)
    assert _close(bmr_pod.EAR_NOTCH_R_MM, 5.0)
    assert _close(bmr_pod.EAR_NOTCH_LIGAMENT_MM,
                  bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM)
    assert _close(bmr_pod.POD_WALL_OFF_EAR_NOTCH_MM, 1.20, 1.0e-6)
    assert bmr_pod.POD_WALL_OFF_UM_RING_MM >= bmr_pod.UM_MATE_GAP_MM

    # The move is real and recorded against the released axis it left.
    assert _close(bmr_pod.RELEASED_AXIS_XY[1], 468.193)
    assert bmr_pod.POD_DROP_MM > 15.0
    assert _close(bmr_pod.UM_AXIS_SPACING_MM,
                  bmr_pod.RELEASED_UM_AXIS_SPACING_MM - bmr_pod.POD_DROP_MM,
                  1.0e-6)
    assert _close(bmr_pod.RELEASED_UM_AXIS_SPACING_MM, 102.112, 1.0e-6)
    assert _close(bmr_pod.SCALLOP_R_MM, 39.25)

    if variant.key == "opposed":
        # The second land is the vase's pitch above the first, and nothing
        # else: no local choice was made about where it goes.
        module = variant.module
        assert _close(module.AXIS_PITCH_MM, bmr_pod.PAIR_AXIS_PITCH_MM)
        assert _close(module.AXIS_PITCH_MM, 49.3, 1.0e-9)
        assert _close(module.UPPER_AXIS_XY[1],
                      module.LOWER_AXIS_XY[1] + module.AXIS_PITCH_MM, 1.0e-6)
        assert _close(module.UPPER_AXIS_XY[1], 501.794193004, 1.0e-6)
        # The two D66 circles overlap, so the body is one plan and not two
        # discs joined by a neck someone had to invent.
        assert module.LAND_OVERLAP_MM > 0.0
        assert _close(module.LAND_OVERLAP_MM, 16.7, 1.0e-6)
        assert _close(module.WAIST_HALF_WIDTH_MM, 21.940316771, 1.0e-6)
        print(f"    lower axis y {bmr_pod.MOUNT_AXIS_XY[1]:.6f}, upper "
              f"{module.UPPER_AXIS_XY[1]:.6f} at the vase's "
              f"{module.AXIS_PITCH_MM} mm pitch; lands overlap "
              f"{module.LAND_OVERLAP_MM:.1f} mm with a "
              f"{2.0 * module.WAIST_HALF_WIDTH_MM:.2f} mm waist")
    else:
        print(f"    axis y {bmr_pod.RELEASED_AXIS_XY[1]:.3f} -> "
              f"{bmr_pod.MOUNT_AXIS_XY[1]:.6f} "
              f"({bmr_pod.POD_DROP_MM:.3f} mm closer); "
              f"MU10-to-BMR spacing {bmr_pod.RELEASED_UM_AXIS_SPACING_MM:.3f}"
              f" -> {bmr_pod.UM_AXIS_SPACING_MM:.3f} mm; "
              f"{bmr_pod.POD_WALL_OFF_EAR_NOTCH_MM:.3f} mm off the notch, "
              f"{bmr_pod.POD_WALL_OFF_UM_RING_MM:.3f} mm off the UM ring")


def test_skirt_fills_the_junction_and_outsections_the_struts(
        variant: Variant) -> None:
    """The junction is solid, on the released seam, and stronger than before.

    The two struts and the window between them are gone.  What replaces them
    has to (a) sit on the released crescent's own seam rather than a new
    boundary, (b) stay inside the plate band so nothing but the driver body
    reaches behind the core rear plane, and (c) beat the section the struts
    reached, since the point of the qualified half-lap is that it governs.

    The skirt is hulled over the mount land alone on both variants, so the
    plan is the same one for both and the opposed variant's second land never
    drags the fill over the waist between the two.
    """
    assert tuple(bmr_pod.SKIRT_Z) == (CORE_REAR_Z, THICKNESS_MM)
    assert _close(bmr_pod.SKIRT_DEPTH_MM, 11.5)
    assert _close(bmr_pod.UM_MATE_R_MM, UM_CORE_R + 0.20)
    assert _close(bmr_pod.UM_MATE_R_MM, 51.9)

    plan = bmr_pod.skirt_plan()
    assert plan.geom_type == "Polygon" and not plan.interiors, (
        "the plate-band plan must close to one simple region")
    # The fill's own edge is the released recut; the closure web's seam runs
    # closer, exactly as it does on the released crescent.
    assert bmr_pod.base_um_ring_clearance_mm() >= bmr_pod.UM_MATE_GAP_MM, (
        "the flush fill leaves only "
        f"{bmr_pod.base_um_ring_clearance_mm():.4f} mm on the UM core ring")
    assert 0.0 < bmr_pod.skirt_um_ring_clearance_mm() < bmr_pod.UM_MATE_GAP_MM

    # The window the user rejected is gone: on the -Y meridian the plan is
    # continuous from the land wall down to the mate face.
    from shapely.geometry import LineString as _Line
    mate_y = UM_CUTOUT[1] + bmr_pod.UM_MATE_R_MM
    pod_y = bmr_pod.MOUNT_AXIS_XY[1] - bmr_pod.POD_OUTER_R_MM
    assert pod_y > mate_y, "the land wall must stand off the mate face"
    # The one thing allowed to interrupt the plan is a released wing: just
    # outboard of each boss the wings run a tongue into the slot the released
    # crescent leaves there, and both parts yield to it exactly as the release
    # does.  Every other break would be a window.
    # Grown by the plan's own decimation budget at both ends: the skirt's
    # boundary is simplified after the subtraction, so it lands a couple of
    # microns either side of the wing's edge.  A real window is orders of
    # magnitude bigger than that.
    keepout = bmr_pod._wing_keepout_plan().buffer(
        2.0 * bmr_pod.SKIRT_PLAN_SIMPLIFY_MM)
    for x in [value / 2.0 for value in range(-66, 67)]:
        column = _Line([(x, 380.0), (x, bmr_pod.MOUNT_AXIS_XY[1])])
        run = column.intersection(plan)
        # A grazing column at the land's own edge meets it in a single point,
        # and a column lying exactly on the closure web's own vertical edge at
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
    run = _Line([(0.0, 380.0), (0.0, bmr_pod.MOUNT_AXIS_XY[1])]).intersection(
        plan)
    assert run.geom_type == "LineString", (
        f"the meridian column is {run.geom_type}, not one unbroken run")
    # Faceting puts the decimated arc a few microns outside the nominal
    # R51.90; what matters is that the plan really does start at the mate
    # face and not somewhere short of it.
    assert 0.0 <= run.bounds[1] - mate_y <= 0.010, (
        f"on the meridian the plan starts at {run.bounds[1]:.4f}, not on the "
        f"R{bmr_pod.UM_MATE_R_MM} mate face at {mate_y:.4f}")

    # And it is strictly a fill: nothing reaches outboard of the flat-clipped
    # D66 land, and the fill stops at the mount land's own top.
    minx, miny, maxx, maxy = plan.bounds
    assert _close(maxx, bmr_pod.POD_FLAT_HALF_WIDTH_MM, 1.0e-6)
    assert _close(minx, -bmr_pod.POD_FLAT_HALF_WIDTH_MM, 1.0e-6)
    assert _close(maxy, bmr_pod.MOUNT_AXIS_XY[1] + bmr_pod.POD_OUTER_R_MM,
                  1.0e-6)

    # Section at the ears: the superseded struts reached 1.44x the half-lap's
    # own net ligament, so the fill has to be at least that.
    assert _close(bmr_pod.EAR_THICKNESS_MM, 5.9)
    assert _close(bmr_pod.EAR_NET_LIGAMENT_MM, 5.2)
    assert _close(bmr_pod.EAR_NET_SECTION_MM2, 30.68)
    section = bmr_pod.ear_load_path_section_mm2()
    ratio = section / bmr_pod.EAR_NET_SECTION_MM2
    assert ratio >= bmr_pod.SUPERSEDED_STRUT_SECTION_RATIO, (
        f"the ear-to-land load path is only {section:.3f} mm2 = {ratio:.3f}x "
        "the half-lap's net ligament; the two struts it replaced reached "
        f"{bmr_pod.SUPERSEDED_STRUT_SECTION_RATIO}x")
    if variant.key == "coaxial":
        print(f"    skirt z={bmr_pod.SKIRT_Z[0]}..{bmr_pod.SKIRT_Z[1]}, plan "
              f"area {plan.area:.1f} mm2, fill "
              f"{bmr_pod.base_um_ring_clearance_mm():.4f} mm off the UM ring "
              f"and the web seam "
              f"{bmr_pod.skirt_um_ring_clearance_mm():.4f} mm; ear load path "
              f"{section:.2f} mm2 = {ratio:.2f}x the half-lap")


def test_cable_path_is_one_hidden_entry_and_one_declared_pass(
        variant: Variant) -> None:
    """No external outlets; one mate-face entry aligned with the UM mouth."""
    module = variant.module
    facts = module.design_facts()["cable"]
    assert facts["external_outlets"] == 0
    assert facts["entries"] == 1
    # The entry sits inside the UM's own declared central cable mouth.
    assert abs(bmr_pod.CABLE_ENTRY_XY[0]) <= T_UM_CABLE_MOUTH_HALF_WIDTH
    assert _close(bmr_pod.CABLE_DUCT_Z_MM, bmr_pod.TS_FREE_CABLE_Z)
    assert _close(bmr_pod.CABLE_DUCT_Z_MM, 3.8, 1.0e-9)
    # The mouth is on the mate face itself.
    entry_r = math.hypot(bmr_pod.CABLE_ENTRY_XY[0] - UM_CUTOUT[0],
                         bmr_pod.CABLE_ENTRY_XY[1] - UM_CUTOUT[1])
    assert _close(entry_r, bmr_pod.UM_MATE_R_MM, 1.0e-6), (
        f"the cable entry is at r={entry_r:.4f}, not on the R"
        f"{bmr_pod.UM_MATE_R_MM} mate face")
    # Ø6.00 is the UM's own T lumen, and a cable arriving off-axis only fits
    # through the bore's projected aperture.
    assert _close(bmr_pod.CABLE_DUCT_D_MM, bmr_pod.TS_DUCT_D)
    assert _close(bmr_pod.CABLE_DUCT_D_MM, 6.0)
    assert bmr_pod.CABLE_MOUTH_APERTURE_MM >= bmr_pod.TS_CABLE_D_EST, (
        f"a {bmr_pod.TS_CABLE_D_EST} mm cable arriving "
        f"{bmr_pod.CABLE_MOUTH_MISALIGNMENT_DEG:.2f} degrees off the duct "
        f"sees only {bmr_pod.CABLE_MOUTH_APERTURE_MM:.3f} mm of aperture")

    if variant.key == "coaxial":
        # The partition pass is the vase's own single-driver lead branch,
        # capped at Ø4.6, and it keeps that same 1.20 mm wall to the pocket.
        assert _close(module.PARTITION_PASS_D_MM, bmr_pod.UPPER_T_BRANCH_D_MM)
        assert module.PARTITION_PASS_D_MM <= 4.6
        assert _close(
            module.PARTITION_PASS_OFFSET_MM + module.PARTITION_PASS_D_MM / 2.0
            + bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM,
            bmr_pod.TEBM_CUTOUT_D_MM / 2.0, 1.0e-6)
        assert module.PARTITION_PASS_XY[1] < bmr_pod.MOUNT_AXIS_XY[1], (
            "the partition pass must be on the -Y side the cable arrives from")
    else:
        # The branch is the same Ø4.60 vase lead branch, running from the
        # lower chamber to the upper one along the driver-axis line at the
        # entry duct's own height.
        assert _close(module.INTER_POCKET_BRANCH_D_MM,
                      bmr_pod.UPPER_T_BRANCH_D_MM)
        assert _close(module.INTER_POCKET_BRANCH_Z_MM, bmr_pod.CABLE_DUCT_Z_MM)
        assert _close(module.INTER_POCKET_BRANCH_XY[0], 0.0)
        # It starts inside the lower pocket and ends inside the upper one, so
        # it really connects the two chambers and is not a blind hole.
        assert _close(module.INTER_POCKET_BRANCH_START_Y_MM,
                      module.LOWER_AXIS_XY[1]
                      + bmr_pod.TEBM_CUTOUT_D_MM / 2.0, 1.0e-6)
        assert _close(module.INTER_POCKET_BRANCH_END_Y_MM,
                      module.UPPER_AXIS_XY[1]
                      - bmr_pod.TEBM_CUTOUT_D_MM / 2.0, 1.0e-6)
        assert module.INTER_POCKET_LIGAMENT_MM > module.INTER_POCKET_BRANCH_D_MM
        # And it is buried: every cover round it beats the vase's own 0.80 mm
        # guarded duct skin, and the two axial ones beat the 1.20 mm wall.
        for name, cover in (
            ("front", module.BRANCH_FRONT_COVER_MM),
            ("rear", module.BRANCH_REAR_COVER_MM),
            ("side", module.BRANCH_SIDE_COVER_MM),
        ):
            assert cover >= bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM, (
                f"the lead branch leaves only {cover:.3f} mm of {name} cover")
        assert _close(module.BRANCH_FRONT_COVER_MM, 12.2, 1.0e-6)
        assert _close(module.BRANCH_REAR_COVER_MM, 8.3, 1.0e-6)

    # The collar carries the duct and its wall, and is the shape of the duct
    # rather than a slab around it.
    assert bmr_pod.ENTRY_COLLAR_Z[1] == CORE_REAR_Z
    assert _close(bmr_pod.ENTRY_COLLAR_WALL_MM,
                  bmr_pod.T_BLIND_BACK_WALL_THICKNESS_MM)
    assert _close(bmr_pod.ENTRY_COLLAR_R_MM,
                  bmr_pod.CABLE_DUCT_R_MM + bmr_pod.ENTRY_COLLAR_WALL_MM)
    assert _close(bmr_pod.ENTRY_COLLAR_Z[0],
                  bmr_pod.CABLE_DUCT_Z_MM - bmr_pod.ENTRY_COLLAR_R_MM, 1.0e-9)
    collar = bmr_pod.entry_collar_plan()
    relief = bmr_pod._um_owned_relief_plan()
    skirt = bmr_pod.skirt_plan().difference(relief)
    assert collar.within(skirt.buffer(1.0e-9)), (
        "the entry collar plan must stay inside what the skirt above it "
        "actually is, or the exterior grows rearward at the core rear plane")

    # That relief is mirrored from the released ownership helper, which only
    # reaches the closure web's Z band.  Apply both to the same prism: if the
    # mirror ever drifts, they stop removing the same volume.
    probe = _plan_prism(bmr_pod.skirt_plan(), *bmr_pod.SKIRT_Z)
    theirs = _enforce_junction_plan_ownership(probe, "t_um", "tweeter")
    mine = _subtract_plan_prisms(probe, relief, *bmr_pod.SKIRT_Z)
    assert abs(float(theirs.volume) - float(mine.volume)) < 1.0e-6, (
        "the locally mirrored ownership relief has drifted from "
        f"_enforce_junction_plan_ownership: {theirs.volume} vs {mine.volume}")

    # It really is a stadium hugging the bore: every stretch of its boundary
    # that is not the skirt's own edge stands exactly one collar radius off
    # the duct's plan sweep.  A slab, or any face or corner of one, would sit
    # further out than that somewhere.
    from shapely.geometry import LineString as _Line2, Point as _Point
    mouth = bmr_pod.CABLE_ENTRY_XY
    direction = bmr_pod.CABLE_DUCT_DIR
    sweep = _Line2([
        (mouth[0] - bmr_pod.ENTRY_COLLAR_BACK_MM * direction[0],
         mouth[1] - bmr_pod.ENTRY_COLLAR_BACK_MM * direction[1]),
        (mouth[0] + bmr_pod.ENTRY_COLLAR_REACH_MM * direction[0],
         mouth[1] + bmr_pod.ENTRY_COLLAR_REACH_MM * direction[1]),
    ])
    inherited = bmr_pod.skirt_plan().exterior.union(relief.boundary).buffer(
        4.0 * bmr_pod.SKIRT_PLAN_SIMPLIFY_MM)
    own_edge = collar.exterior.difference(inherited)
    assert own_edge.length > 0.5 * collar.exterior.length, (
        "most of the collar's boundary should be its own, not the skirt's")
    strayed = max(
        abs(sweep.distance(_Point(*coordinate)) - bmr_pod.ENTRY_COLLAR_R_MM)
        for piece in getattr(own_edge, "geoms", [own_edge])
        for coordinate in piece.coords)
    assert strayed <= 0.01, (
        f"the collar's own boundary strays {strayed:.4f} mm from a constant "
        f"{bmr_pod.ENTRY_COLLAR_R_MM} mm offset of the bore; it is not a "
        "stadium")
    # And it is small: a slab spanning the mouth would be several times this.
    assert collar.area < 150.0, (
        f"the entry collar plan is {collar.area:.1f} mm2; that is a box, not "
        "a collar")
    if variant.key == "coaxial":
        print(f"    entry Ø{bmr_pod.CABLE_DUCT_D_MM:.2f} at "
              f"({bmr_pod.CABLE_ENTRY_XY[0]:.3f}, "
              f"{bmr_pod.CABLE_ENTRY_XY[1]:.3f}, {bmr_pod.CABLE_DUCT_Z_MM}) "
              f"bearing {bmr_pod.CABLE_DUCT['bearing_deg']:.2f} deg, "
              f"{bmr_pod.CABLE_MOUTH_MISALIGNMENT_DEG:.2f} deg off the cable "
              f"({bmr_pod.CABLE_MOUTH_APERTURE_MM:.3f} mm aperture for a "
              f"{bmr_pod.TS_CABLE_D_EST} mm cable); partition pass "
              f"Ø{module.PARTITION_PASS_D_MM} at "
              f"y={module.PARTITION_PASS_XY[1]:.3f}")
    else:
        print(f"    same Ø{bmr_pod.CABLE_DUCT_D_MM:.2f} entry into the lower "
              f"chamber; Ø{module.INTER_POCKET_BRANCH_D_MM} branch at x=0, "
              f"z={module.INTER_POCKET_BRANCH_Z_MM}, y="
              f"{module.INTER_POCKET_BRANCH_START_Y_MM:.3f}.."
              f"{module.INTER_POCKET_BRANCH_END_Y_MM:.3f} "
              f"({module.INTER_POCKET_LIGAMENT_MM:.3f} mm) under "
              f"{module.BRANCH_FRONT_COVER_MM:.1f}/"
              f"{module.BRANCH_REAR_COVER_MM:.1f}/"
              f"{module.BRANCH_SIDE_COVER_MM:.1f} mm of cover")


def test_captive_stations_are_the_vase_s_own(variant: Variant) -> None:
    """Every magnet sits where the vase puts it on that land, exactly.

    The vase is read in a proud-profile subprocess and its four stations are
    reduced to land-local coordinates -- the offset from the land's own axis --
    which is the only form in which they can be compared, since these parts
    put their lands somewhere else entirely.  The coaxial pod has one outward
    land and takes the vase's lower/front pair; the opposed pod has both.
    """
    module = variant.module
    assert module.MAGNET_COUNT == variant.magnet_count
    assert len(module.MAGNET_LANDS) * bmr_pod.MAGNETS_PER_LAND == (
        module.MAGNET_COUNT)

    program = (
        "import json, sys;"
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r});"
        "from lx521_baffle.proud import vase_tebm35c10_4 as v;"
        # Exactly the loop _apply_t_magnets runs, reduced to land-local form.
        "print(json.dumps([[sign * v.T_MAGNET_FACE_X_MM, 0.0,"
        " v.STANDARD_MAGNET_Z_MM]"
        " for axis_y in (v.LOWER_T_AXIS_Y_MM, v.UPPER_T_AXIS_Y_MM)"
        " for sign in (-1.0, 1.0)]))"
    )
    vase_local = _vase_in_a_proud_subprocess(program, "")
    assert len(vase_local) == bmr_pod.T_MAGNET_TOTAL == 4
    # Both of the vase's lands carry the same land-local pair, which is why
    # the coaxial pod can take "the vase's lower/front pair" and mean it.
    assert vase_local[:2] == vase_local[2:], (
        "the vase's two lands no longer carry the same land-local pair; the "
        "coaxial variant's claim to take the lower/front pair is now empty")

    mine = []
    for land, axis_y in module.MAGNET_LANDS:
        for station in bmr_pod.land_magnet_faces(axis_y):
            face = station["face_xyz_mm"]
            mine.append([face[0], face[1] - axis_y, face[2]])
    for index, (theirs, here) in enumerate(zip(vase_local, mine)):
        for axis, (want, got) in enumerate(zip(theirs, here)):
            assert _close(want, got), (
                f"station {index} axis {axis} is {got}, not the vase's {want}")
    assert _close(bmr_pod.MAGNET_AXIS_Z_MM, 15.10)
    assert tuple(bmr_pod.MAGNET_PRINT_UP) == (0.0, 0.0, -1.0)
    assert tuple(bmr_pod.MAGNET_BED_DATUM) == (0.0, 0.0, THICKNESS_MM)

    # The declared-opening list must not have grown a magnet: a captive
    # cavity is a sealed void, not an opening, and calling it one would let a
    # future station quietly declare itself ``exterior``.
    names = [opening["name"] for opening in module.declared_openings()]
    assert not [name for name in names if "magnet" in name]
    magnets = module.design_facts()["magnets"]
    assert magnets["count"] == module.MAGNET_COUNT
    assert magnets["exterior_opening"] is False
    assert "candidate only" in magnets["release_wiring"]
    print(f"    {module.MAGNET_COUNT} captive stations on "
          f"{len(module.MAGNET_LANDS)} land(s) at the vase's own "
          f"x=±{bmr_pod.T_MAGNET_FACE_X_MM:.6f}, z={bmr_pod.MAGNET_AXIS_Z_MM}")


def test_inherited_tweeter_clamp_holes_are_gone(variant: Variant) -> None:
    """The four M4 ND25FW-4 faceplate passages must not be declared at all.

    They were inherited only to keep the released silhouette exact.  Neither
    variant clamps a tweeter and neither has that silhouette, so they carry no
    fastener and no meaning.
    """
    module = variant.module
    names = [opening["name"] for opening in module.declared_openings()]
    assert "inherited_m4_tweeter_clamp_holes" not in names
    assert not [
        opening for opening in module.declared_openings()
        if opening["kind"] == "released_inherited"
    ]
    expected_pass = ("chamber_partition_cable_pass" if variant.key == "coaxial"
                     else "inter_pocket_lead_branch")
    pockets = (["front_driver_pocket_mouth", "rear_driver_pocket_mouth"]
               if variant.key == "coaxial"
               else ["lower_driver_pocket_mouth", "upper_driver_pocket_mouth"])
    assert names == pockets + [
        "um_mate_face_cable_entry",
        expected_pass,
        "um_half_lap_clearance_passages",
        "um_half_lap_insert_receivers",
        "m2_driver_insert_bores",
    ]
    # The two external lead outlets went with the struts.
    assert not [name for name in names if "lead_outlet" in name]
    silhouette = module.design_facts()["silhouette"]
    assert silhouette["inherits_released_crescent_outline"] is False


def test_no_declared_opening_reaches_the_assembled_exterior(
        variant: Variant) -> None:
    """Every opening faces the UM mate, a driver, or nothing at all.

    This is the whole point of the cable rework: with the pod assembled on the
    collar and both drivers fitted, there must be no hole anyone can see.
    """
    module = variant.module
    exposures = {"um_mate", "driver_face", "internal"}
    for opening in module.declared_openings():
        assert "exposure" in opening, (
            f"{opening['name']} does not declare which side it faces")
        assert opening["exposure"] in exposures, (
            f"{opening['name']} is exposed to the {opening['exposure']}")
    assert module.design_facts()["exterior_openings"] == []
    by_exposure = {}
    for opening in module.declared_openings():
        by_exposure.setdefault(opening["exposure"], []).append(opening["name"])
    # One cable entry, on the mate, and one internal pass.  Not two of either.
    assert [name for name in by_exposure["um_mate"]
            if "cable" in name] == ["um_mate_face_cable_entry"]
    assert len(by_exposure["internal"]) == 1
    print("    openings: "
          + "; ".join(f"{side}={sorted(names)}"
                      for side, names in sorted(by_exposure.items())))


def test_candidate_flags_are_set(variant: Variant) -> None:
    module = variant.module
    facts = module.design_facts()
    assert module.RELEASE_AUTHORIZED is False
    assert module.PHYSICAL_MEASURE_REQUIRED is True
    assert facts["release_authorized"] is False
    assert facts["physical_measure_required"] is True
    assert facts["status"] == "candidate_not_release_authorized"
    assert facts["counts_against_release_inventory"] is False
    assert facts["variant"] == variant.key
    assert facts["magnet_count"] == variant.magnet_count


def test_part_is_not_wired_into_the_release(variant: Variant) -> None:
    """The candidate stays out of the stage, the counts, to_print and catalog.

    Both variants now bury real captive magnets, so the released
    captive-magnet catalog is part of what they must stay out of: its expected
    totals are restated here so that wiring a candidate in has to come past
    this gate.
    """
    module = variant.module
    stem = module.PART_NAME
    staged = (PROJECT_ROOT / "scripts" / "export_obiwan_staged.py").read_text(
        encoding="utf-8")
    assert "bmr_crescent" not in staged, (
        "the BMR pods must not join the Obi-Wan stage manifest")
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

    catalog_source = (
        PROJECT_ROOT / "scripts" / "generate_captive_magnet_catalog.py"
    ).read_text(encoding="utf-8")
    assert "bmr" not in catalog_source.lower(), (
        f"{stem} has reached the released captive-magnet catalog generator")
    assert f"EXPECTED_MAGNET_COUNT = {RELEASED_MAGNET_TOTAL}" in (
        catalog_source), (
        "the released captive-magnet total moved; a candidate's stations must "
        "never be counted against it")
    for profile in ("captive_magnet_slicing_profile.json",
                    "captive_magnet_slicing_profile_petg_gf.json"):
        text = (PROJECT_ROOT / profile).read_text(encoding="utf-8")
        assert "bmr" not in text.lower(), (
            f"{stem} has reached the released slicing profile {profile}")

    # The candidate now has a real sliced delivery, which is exactly when
    # somebody might route it through a release structure to get it printed.
    # Every delivered file has to live in the part's own build child, and the
    # released totals have to be where they were.
    assert f"EXPECTED_ARTIFACT_COUNT = {RELEASED_ARTIFACT_TOTAL}" in (
        catalog_source), (
        "the released artifact total moved; a candidate delivery must never "
        "be counted against it")
    validation = (
        PROJECT_ROOT / "scripts" / "release_validation.py"
    ).read_text(encoding="utf-8")
    assert (f"EXPECTED_RELEASE_ARTIFACT_COUNT = {RELEASED_ARTIFACT_TOTAL}"
            in validation)
    assert f"EXPECTED_RELEASE_MAGNET_COUNT = {RELEASED_MAGNET_TOTAL}" in (
        validation)
    shelf_test = (PROJECT_ROOT / "tests" / "test_to_print_shelf.py").read_text(
        encoding="utf-8")
    assert f"exactly {RELEASED_SHELF_PAIRS} entries" in shelf_test, (
        "the P2S shelf's pair count moved; the candidate delivery is a "
        "parallel path and must not touch it")
    released_catalog = PROJECT_ROOT / "review" / (
        "captive_magnet_release_catalog.json")
    if released_catalog.is_file():
        entries = _json(released_catalog)["artifacts"]
        assert len(entries) == RELEASED_ARTIFACT_TOTAL
        assert not [entry for entry in entries if stem == entry["part"]], (
            f"{stem} has entered the released catalog")
    for name, path in _delivery_paths(variant).items():
        assert BUILD_ROOT in path.parents, (
            f"delivered {name} is outside the candidate's own build child: "
            f"{path}")
    # And nothing of the candidate's may appear in the structures the release
    # slicer, the shelf and the product facade own.
    for root in (PROJECT_ROOT / "review" / "captive_magnet_slice_audit",
                 PROJECT_ROOT / "to_print", PROJECT_ROOT / "artifacts"):
        if root.is_dir():
            hits = [str(path) for path in root.rglob("*bmr_crescent*")]
            assert not hits, f"candidate delivery leaked into {root}: {hits}"


def test_slicing_profile_is_the_base_profile_derived(variant: Variant) -> None:
    """The delivery takes the base profile, material and walls included.

    ``captive_magnet_slicing_profile_petg_gf.json`` exists for the structural
    core -- the two LM keyed halves and the UM carrier -- and says so in its
    own ``artifact_scope``.  A pod hangs off the UM carrier's qualified M3
    half-lap rather than being that joint, so it takes the base profile like
    the qualified vase does.  This gate pins that reading in both directions:
    the derived profile has to match the base field for field, and the
    PETG-GF profile has to keep refusing to name either pod.
    """
    from gen_bmr_crescent_slicing_profile import VARIANTS, generate

    module = variant.module
    base = _json(BASE_SLICING_PROFILE)
    petg = _json(PETG_GF_SLICING_PROFILE)
    assert not [match for match in petg["artifact_scope"]
                if match["part"] == module.PART_NAME], (
        f"{module.PART_NAME} has entered the structural PETG-GF scope")

    spec = VARIANTS[variant.key]
    assert spec.part == module.PART_NAME
    assert spec.release_variant == module.RELEASE_VARIANT
    with tempfile.TemporaryDirectory() as directory:
        derived_path = Path(directory) / f"{module.PART_NAME}.profile.json"
        derived = generate(BASE_SLICING_PROFILE, derived_path, variant.key)
    assert derived["catalog_mode"] == "auxiliary"
    assert derived["artifact_overrides"] == []
    assert derived["artifact_scope"] == [{
        "state": "shared",
        "variant": module.RELEASE_VARIANT,
        "part": module.PART_NAME,
    }]
    assert derived["filament"] == base["filament"]
    assert derived["requirements"]["support_enabled"] is False
    process = derived["repo_overrides"]["process"]
    base_process = base["repo_overrides"]["process"]
    for key in ("wall_loops", "sparse_infill_density", "sparse_infill_pattern",
                "top_shell_layers", "bottom_shell_layers"):
        assert process[key] == base_process[key], (
            f"{key} drifted from the base profile")
    assert {process[name] for name in (
        "enable_support", "support_on_build_plate_only",
        "support_critical_regions_only", "support_remove_small_overhang",
    )} == {"0"}

    built = _delivery_paths(variant)["profile"]
    if not built.is_file():
        _skip(f"{built.relative_to(PROJECT_ROOT)} absent; run "
              "'make obiwan_bmr_crescent_cad'")
        return
    on_disk = _json(built)
    # Only the recorded base path differs: the temporary derivation above
    # computed it relative to its own directory.
    assert on_disk["generated_from"] == os.path.relpath(
        BASE_SLICING_PROFILE, built.parent)
    assert {key: value for key, value in on_disk.items()
            if key != "generated_from"} == {
        key: value for key, value in derived.items()
        if key != "generated_from"}
    print(f"    profile: {derived['filament']}, "
          f"{process['wall_loops']} walls, "
          f"{process['sparse_infill_density']} "
          f"{process['sparse_infill_pattern']}")


def test_pause_plan_follows_the_station_geometry(variant: Variant) -> None:
    """Cavities that close on one plane get one pause, at a predicted Z.

    Both pods put every station on the same source Z, so both must collapse
    to a single pause covering all of their magnets -- the coaxial pod's two
    and the opposed pod's four, whose two lands differ only in Y.  The Z
    itself is the first layer of the profile's own ladder strictly above the
    closing plane, which is what makes the sliced pause a regression rather
    than a number read back from the slicer.
    """
    from gen_bmr_crescent_slicing_profile import pause_layer_z

    paths = _delivery_paths(variant)
    if not paths["facts"].is_file() or not paths["profile"].is_file():
        _skip("no exported facts/profile; run 'make obiwan_bmr_crescent_cad'")
        return
    facts = _json(paths["facts"])
    profile = _json(paths["profile"])
    manifest = facts["delivery"]["pause_manifest"]
    stations = facts["design"]["magnets"]["stations"]
    assert len(stations) == variant.magnet_count

    planes: dict[float, list[str]] = {}
    for station in stations:
        plane = round(
            float(station["cavity_bury_roof_start_print_z_mm"]), 9)
        planes.setdefault(plane, []).append(str(station["name"]))
    assert len(planes) == 1, (
        "the stations no longer share one closing plane; the delivery needs "
        f"one pause per plane, not one: {sorted(planes)}")
    assert manifest["pause_group_count"] == len(planes)
    assert manifest["magnet_count"] == variant.magnet_count
    assert manifest["park_z_mm"] == profile["magnet_insertion_pause"][
        "park_z_mm"]

    (plane, names), = planes.items()
    group, = manifest["groups"]
    assert _close(group["cavity_bury_roof_start_plane_z_mm"], plane)
    assert group["sites"] == sorted(names)
    assert group["magnet_count"] == variant.magnet_count
    expected_z = pause_layer_z(profile, plane)
    assert _close(group["expected_pause_marker_z_mm"], expected_z)
    # The ladder itself, restated rather than re-derived: 0.20 mm first layer
    # then 0.16 mm, so the 5.80 mm closing plane is a layer top and the pause
    # belongs on the next one.
    first = profile["requirements"]["first_layer_height_mm"]
    layer = profile["requirements"]["layer_height_mm"]
    assert (first, layer) == (0.2, 0.16)
    assert _close(plane, 5.80, 1.0e-6) and _close(expected_z, 5.96, 1.0e-9)
    print(f"    1 pause at Z={expected_z} burying {variant.magnet_count} "
          f"magnet(s) over the {plane} mm closing plane")


def test_delivery_validator_accepts_the_sliced_project(
    variant: Variant,
) -> None:
    """The promoted 3MF passes the same fail-closed gate the target runs."""
    from validate_bmr_crescent_delivery import validate

    paths = _delivery_paths(variant)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        _skip(f"no sliced delivery ({', '.join(sorted(missing))}); run "
              "'make obiwan_bmr_crescent_3mf'")
        return
    result = validate(
        catalog=paths["catalog"], facts=paths["facts"],
        profile=paths["profile"], audit=paths["audit"],
        project=paths["project"], gcode=paths["gcode"], variant=variant.key)
    assert result["artifact"] == (
        f"shared:{variant.module.RELEASE_VARIANT}:{variant.module.PART_NAME}")
    assert result["release_status"] == "candidate_not_release_authorized"
    assert result["catalog_sites"] == variant.magnet_count
    assert result["magnet_pause_groups"] == 1
    assert result["slicing_profile_rederived"] == "identical"
    assert result["stl_3mf_equivalence"] == "pass"
    assert result["support_toolpaths"] == "none"
    event, = result["pause_events"]
    assert event["magnet_count"] == variant.magnet_count
    assert _close(event["pause_marker_z_mm"], 5.96, 1.0e-9)
    assert _close(event["park_z_mm"], 250.0, 1.0e-9)
    print(f"    delivery validated: pause Z={event['pause_marker_z_mm']}, "
          f"park Z={event['park_z_mm']}, "
          f"{event['magnet_count']} magnet(s)")


# --------------------------------------------------------------------------
# Geometry gates (exported BREP)
# --------------------------------------------------------------------------

def _exported_solid(variant: Variant):
    brep, facts_path = _brep(variant), _facts_path(variant)
    if not brep.is_file() or not facts_path.is_file():
        _skip(
            f"{brep.relative_to(PROJECT_ROOT)} absent; run "
            "LX_STAND_FOOT=0 LX_ROUTING_PROFILE=obiwan python "
            "scripts/export_bmr_crescent.py")
        return None, None
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    # An export that predates the current source is not evidence about it.
    recorded = facts["source_file_sha256"]
    for relative, digest in recorded.items():
        path = (facts_path.parent / relative).resolve()
        assert path.is_file(), f"recorded source vanished: {relative}"
        assert sha256_file(path) == digest, (
            f"exported artifacts are stale against {relative}; re-run "
            "scripts/export_bmr_crescent.py")
    assert sha256_file(brep) == facts["files"]["brep"]["sha256"]
    return Part(import_brep(str(brep)).solids()), facts


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


def test_exported_solid_is_one_body_with_only_its_magnet_voids(
        variant: Variant) -> None:
    """One material body, one shell per part plus one per buried magnet."""
    solid, facts = _exported_solid(variant)
    if solid is None:
        return
    assert solid.is_valid, "exported BREP is not a valid solid"
    assert len(solid.solids()) == 1, "the pod must be one body"
    # A captive station is a sealed void by design, and there must be exactly
    # as many of those as there are declared stations: one more would be an
    # undeclared cavity and one fewer would be a station that broke out.
    assert len(solid.shells()) == 1 + variant.magnet_count, (
        f"exported solid has {len(solid.shells())} shells; expected one "
        f"outer shell plus {variant.magnet_count} buried captive cavities")
    size = solid.bounding_box().size
    for axis, value in (("X", size.X), ("Y", size.Y), ("Z", size.Z)):
        assert value <= BED_LIMIT_MM, (
            f"{axis} extent {value:.3f} mm exceeds the {BED_LIMIT_MM} mm bed")
    printed = facts["print_geometry"]["bounds_size_mm"]
    assert max(printed) <= BED_LIMIT_MM
    assert facts["print_geometry"]["p2s_256mm_fit"] is True
    assert facts["print_geometry"]["support_enabled"] is False
    # X is set by the land alone -- now the flat-clipped land, because the
    # magnet flats are the widest thing on the part.
    assert _close(size.X, bmr_pod.POD_PLAN_WIDTH_MM, 0.01), (
        f"the X extent is {size.X:.3f} mm, not the flat-clipped D66 land; "
        "something still reaches outboard of it")
    print(f"    envelope {size.X:.3f} x {size.Y:.3f} x {size.Z:.3f} mm, "
          f"{solid.volume / 1000.0:.2f} cm3, "
          f"{len(solid.shells()) - 1} buried magnet cavit(ies)")


def test_declared_openings_are_the_only_openings(variant: Variant) -> None:
    """Each declared feature exists; nothing undeclared breaks the skin."""
    solid, _facts = _exported_solid(variant)
    if solid is None:
        return
    module = variant.module
    if variant.key == "coaxial":
        _probe_coaxial_openings(solid, module)
    else:
        _probe_opposed_openings(solid, module)

    # The mount's own insert receivers stay blind behind 1.9 mm of front, on
    # both variants.
    for x in TWEETER_JOINT_X:
        assert _material(
            solid, x, TWEETER_JOINT_Y, THICKNESS_MM - 0.6, size=0.4) > 0.0, (
            "the acoustic-front floor over an insert receiver is breached")
        assert _material(
            solid, x, TWEETER_JOINT_Y,
            TWEETER_JOINT_INSERT_BORE_Z[1] - 0.6, size=0.4) == 0.0, (
            "an insert receiver is missing")

    # The one cable entry is open from the mate face into the mount chamber.
    entry_x, entry_y = bmr_pod.CABLE_ENTRY_XY
    direction = bmr_pod.CABLE_DUCT_DIR
    for reach in (-0.8, 1.0, bmr_pod.CABLE_DUCT_LENGTH_MM / 2.0,
                  bmr_pod.CABLE_DUCT_LENGTH_MM - 1.0):
        assert _material(
            solid, entry_x + direction[0] * reach,
            entry_y + direction[1] * reach, bmr_pod.CABLE_DUCT_Z_MM,
            size=0.4) == 0.0, (
            f"the cable duct is obstructed {reach:.2f} mm along its axis")
    # Beside the duct, half way along where the land wall surrounds it, the
    # material is intact; below it in the entry collar the declared 1.20 mm
    # floor is there.  Probing beside the mouth itself would only sample the
    # free space outside the curved mate face.
    normal = (-direction[1], direction[0])
    middle = bmr_pod.CABLE_DUCT_LENGTH_MM / 2.0
    for sign in (-1.0, 1.0):
        offset = sign * (bmr_pod.CABLE_DUCT_R_MM + 0.6)
        assert _material(
            solid, entry_x + normal[0] * offset + direction[0] * middle,
            entry_y + normal[1] * offset + direction[1] * middle,
            bmr_pod.CABLE_DUCT_Z_MM, size=0.4) > 0.0, (
            "the cable duct is wider than declared")
    assert _material(
        solid, entry_x + direction[0] * 1.0, entry_y + direction[1] * 1.0,
        bmr_pod.ENTRY_COLLAR_Z[0] + 0.4, size=0.3) > 0.0, (
        "the cable duct has no floor under it in the entry collar")


def _probe_coaxial_openings(solid, module) -> None:
    axis_x, axis_y = bmr_pod.MOUNT_AXIS_XY

    # Both pockets are clear over their full declared depth.
    for name, z_low, z_high in (
        ("front", module.FRONT_POCKET_FLOOR_Z_MM + 0.4, THICKNESS_MM - 0.4),
        ("rear", module.REAR_MOUNT_Z_MM + 0.4,
         module.REAR_POCKET_ROOF_Z_MM - 0.4),
    ):
        for fraction in (0.05, 0.5, 0.95):
            z = z_low + (z_high - z_low) * fraction
            assert _material(solid, axis_x, axis_y, z) == 0.0, (
                f"{name} driver pocket is obstructed at z={z:.3f}")

    # The partition separates the two chambers everywhere except at the one
    # declared pass, which really is open over its whole declared span.
    partition_mid = (module.REAR_POCKET_ROOF_Z_MM
                     + module.FRONT_POCKET_FLOOR_Z_MM) / 2.0
    assert _material(solid, axis_x, axis_y, partition_mid, size=0.4) > 0.0, (
        "the back-to-back partition is missing; the two rear volumes would "
        "be one chamber")
    pass_x, pass_y = module.PARTITION_PASS_XY
    for z in (module.REAR_POCKET_ROOF_Z_MM + 0.2, partition_mid,
              module.FRONT_POCKET_FLOOR_Z_MM - 0.2):
        assert _material(solid, pass_x, pass_y, z, size=0.4) == 0.0, (
            f"the declared partition pass is obstructed at z={z:.3f}")
    # It is exactly one pass of exactly the declared size: the partition is
    # intact a diameter to either side of it and on the opposite meridian.
    for offset in (-module.PARTITION_PASS_D_MM, module.PARTITION_PASS_D_MM):
        assert _material(solid, pass_x + offset, pass_y, partition_mid,
                         size=0.4) > 0.0, (
            "the partition pass is wider than declared")
    assert _material(
        solid, axis_x, axis_y + module.PARTITION_PASS_OFFSET_MM,
        partition_mid, size=0.4) > 0.0, (
        "an undeclared second partition pass exists")

    _probe_m2_bores(solid, (
        (axis_y, module.FRONT_MOUNT_CLOCK_DEG, THICKNESS_MM - 0.5,
         THICKNESS_MM - bmr_pod.M2_INSERT_DEPTH_MM - 0.8),
        (axis_y, module.REAR_MOUNT_CLOCK_DEG, module.REAR_MOUNT_Z_MM + 0.5,
         module.REAR_MOUNT_Z_MM + bmr_pod.M2_INSERT_DEPTH_MM + 0.8),
    ))


def _probe_opposed_openings(solid, module) -> None:
    # Both pockets are clear over their full declared depth, each opening on
    # the face its own driver mounts to.
    for name, axis_y, z_low, z_high in (
        ("lower", module.LOWER_AXIS_XY[1],
         module.LOWER_POCKET_FLOOR_Z_MM + 0.4, module.FRONT_PLANE_Z_MM - 0.4),
        ("upper", module.UPPER_AXIS_XY[1],
         module.REAR_PLANE_Z_MM + 0.4, module.UPPER_POCKET_ROOF_Z_MM - 0.4),
    ):
        for fraction in (0.05, 0.5, 0.95):
            z = z_low + (z_high - z_low) * fraction
            assert _material(solid, 0.0, axis_y, z) == 0.0, (
                f"{name} driver pocket is obstructed at z={z:.3f}")
    # Each pocket really is blind at the opposite face: the 1.20 mm wall is
    # there, and it is the only thing between the pocket and that face.
    assert _material(solid, 0.0, module.LOWER_AXIS_XY[1],
                     module.REAR_PLANE_Z_MM + 0.4, size=0.4) > 0.0, (
        "the lower pocket's 1.20 mm blind wall is missing")
    assert _material(solid, 0.0, module.UPPER_AXIS_XY[1],
                     module.FRONT_PLANE_Z_MM - 0.4, size=0.4) > 0.0, (
        "the upper pocket's 1.20 mm blind wall is missing")

    # The lead branch is open the whole way between the two chambers.
    branch_z = module.INTER_POCKET_BRANCH_Z_MM
    start = module.INTER_POCKET_BRANCH_START_Y_MM
    end = module.INTER_POCKET_BRANCH_END_Y_MM
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = start + (end - start) * fraction
        assert _material(solid, 0.0, y, branch_z, size=0.4) == 0.0, (
            f"the declared lead branch is obstructed at y={y:.3f}")
    # And it is exactly one bore of exactly the declared size: solid a
    # diameter to either side of it, above it and below it, so nothing else
    # crosses the waist.
    waist_y = module.WAIST_Y_MM
    for offset in (-module.INTER_POCKET_BRANCH_D_MM,
                   module.INTER_POCKET_BRANCH_D_MM):
        assert _material(solid, offset, waist_y, branch_z, size=0.4) > 0.0, (
            "the lead branch is wider than declared across the waist")
        assert _material(solid, 0.0, waist_y, branch_z + offset,
                         size=0.4) > 0.0, (
            "the lead branch is taller than declared")
    # Its cover to both exterior faces is real material, not a claim.
    assert _material(solid, 0.0, waist_y, module.FRONT_PLANE_Z_MM - 0.4,
                     size=0.4) > 0.0, "the branch has no front cover"
    assert _material(solid, 0.0, waist_y, module.REAR_PLANE_Z_MM + 0.4,
                     size=0.4) > 0.0, "the branch has no rear cover"

    _probe_m2_bores(solid, (
        (module.LOWER_AXIS_XY[1], module.LOWER_MOUNT_CLOCK_DEG,
         module.FRONT_PLANE_Z_MM - 0.5,
         module.FRONT_PLANE_Z_MM - bmr_pod.M2_INSERT_DEPTH_MM - 0.8),
        (module.UPPER_AXIS_XY[1], module.UPPER_MOUNT_CLOCK_DEG,
         module.REAR_PLANE_Z_MM + 0.5,
         module.REAR_PLANE_Z_MM + bmr_pod.M2_INSERT_DEPTH_MM + 0.8),
    ))


def _probe_m2_bores(solid, patterns) -> None:
    """Four blind M2 bores per land, on the declared PCD and clock."""
    radius = bmr_pod.TEBM_MOUNT_PCD_MM / 2.0
    for axis_y, clock, mouth_z, blind_z in patterns:
        for index in range(bmr_pod.TEBM_MOUNT_HOLE_COUNT):
            angle = math.radians(clock + 90.0 * index)
            x = radius * math.cos(angle)
            y = axis_y + radius * math.sin(angle)
            assert _material(solid, x, y, mouth_z, size=0.4) == 0.0, (
                f"M2 bore at y={axis_y:.3f} clock {clock} index {index} "
                "is missing")
            assert _material(solid, x, y, blind_z, size=0.4) > 0.0, (
                f"M2 bore at y={axis_y:.3f} clock {clock} index {index} "
                "is not blind")


def test_every_magnet_cavity_is_buried_behind_its_own_skin(
        variant: Variant) -> None:
    """Each station is void inside, solid outside, and opens nowhere.

    The shell count already says the cavities are sealed.  This walks each one
    in the exported solid: the cavity itself is empty, the qualified 0.45 mm
    face skin in front of it is material, the land behind it is material all
    the way to the driver pocket, and the flat that carries it really is the
    part's exterior at that station.
    """
    solid, _facts = _exported_solid(variant)
    if solid is None:
        return
    module = variant.module
    checked = 0
    for _land, axis_y in module.MAGNET_LANDS:
        for station in bmr_pod.land_magnet_faces(axis_y):
            face_x, face_y, face_z = station["face_xyz_mm"]
            sign = 1.0 if face_x > 0.0 else -1.0
            # The cavity, one magnet's depth in from the face skin.
            cavity_x = face_x - sign * (FACE_SKIN_MM + CAVITY_DEPTH_MM / 2.0)
            assert _material(solid, cavity_x, face_y, face_z,
                             size=0.4) == 0.0, (
                f"the captive cavity at x={face_x:.4f}, y={face_y:.3f} is "
                "not open")
            # The face skin between it and the flat.
            skin_x = face_x - sign * FACE_SKIN_MM / 2.0
            assert _material(solid, skin_x, face_y, face_z,
                             size=0.3) > 0.0, (
                f"the {FACE_SKIN_MM} mm face skin at x={face_x:.4f} is "
                "breached; the magnet would show")
            # Just outside the flat there is nothing: the flat is the skin.
            assert _material(solid, face_x + sign * 0.6, face_y, face_z,
                             size=0.4) == 0.0, (
                f"the magnet flat at x={face_x:.4f} is not the exterior")
            # And the land behind the cavity is solid all the way in, so no
            # station has broken into a driver pocket.
            inner_x = face_x - sign * (FACE_SKIN_MM + CAVITY_DEPTH_MM + 1.0)
            assert _material(solid, inner_x, face_y, face_z, size=0.4) > 0.0, (
                f"the land behind the cavity at x={face_x:.4f} is not solid")
            checked += 1
    assert checked == variant.magnet_count
    print(f"    {checked} captive cavit(ies) open inside, sealed behind "
          f"{FACE_SKIN_MM} mm of skin, D{MAGNET_DIAMETER_MM}x{MAGNET_DEPTH_MM} "
          f"magnet in a Ø{CAVITY_DIAMETER_MM}x{CAVITY_DEPTH_MM} cavity")


def test_mount_interface_is_geometrically_identical_to_the_released_ear(
        variant: Variant) -> None:
    """Ear for ear, this part's mount is the released crescent's mount.

    Neither variant is a superset of the released crescent, so a
    symmetric-difference-over-the-whole-silhouette gate cannot apply.  What
    has to be identical is the mount, so the comparison is confined to the
    joint authority's own ear footprints over the add-on's own Z span, where
    both parts must be exactly the complete D9.8 ear less the D3.4 passage and
    the blind D4.6 receiver.
    """
    solid, _facts = _exported_solid(variant)
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
        # confined to the plate band the two share -- the driver body would
        # otherwise swamp it -- and has to show a large difference in *both*
        # directions.  Material of the release that is absent here is what
        # rules out an inherited outline.
        band = Pos(0.0, bmr_pod.MOUNT_AXIS_XY[1],
                   (CORE_REAR_Z + THICKNESS_MM) / 2.0
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


def test_mate_simulation_against_the_staged_um_collar(
        variant: Variant) -> None:
    """Assemble on the real UM BREP: no interference, gap and screws intact."""
    solid, _facts = _exported_solid(variant)
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


def test_the_assembled_junction_has_no_window(variant: Variant) -> None:
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
    solid, _facts = _exported_solid(variant)
    if solid is None:
        return
    boss_edge = abs(TWEETER_JOINT_X[1]) + TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0
    depth = 4.0 + (variant.z_span[1] - variant.z_span[0])
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
            column = Pos(x, 0.0, 0.0) * Box(0.30, 400.0, depth)
            here = (solid & column)
            there = (um & column)
            if here is None or here.volume == 0.0:
                continue
            near = here.bounding_box().min.Y
            if there is None or there.volume == 0.0:
                continue
            far = there.bounding_box().max.Y
            widest = max(widest, near - far)
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
                column = Pos(x, 0.0, 0.0) * Box(0.30, 400.0, depth)
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


def test_free_t_cable_reaches_the_declared_entry_without_exposure(
        variant: Variant) -> None:
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
    solid, _facts = _exported_solid(variant)
    if solid is None:
        return
    low, high = variant.z_span
    mate = Pos(UM_CUTOUT[0], UM_CUTOUT[1], (low + high) / 2.0) * Cylinder(
        bmr_pod.UM_MATE_R_MM, (high - low) + 2.0)
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
        duct = bmr_pod.duct_cutter()
        crossing = cable - mate
        assert crossing is not None and crossing.volume > 0.0, (
            f"{state}: the modelled T cable never crosses the mate face")
        # Only the first 6 mm past the face: beyond that the modelled cable
        # keeps its released free-flight path instead of following the duct.
        throat = Pos(bmr_pod.CABLE_ENTRY_XY[0], bmr_pod.CABLE_ENTRY_XY[1],
                     bmr_pod.CABLE_DUCT_Z_MM) * Box(14.0, 14.0, 14.0)
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
        print(f"    free T cable at z={bmr_pod.TS_FREE_CABLE_Z:.2f} runs clear "
              f"to the mate face and into the Ø{bmr_pod.CABLE_DUCT_D_MM:.2f} "
              f"entry in {checked} staged state(s)")


def _filled_silhouette(solid, z: float, centre_y: float):
    """Unit prism of the part's exterior plan at one Z, holes filled.

    Only the outer wire is kept, so pockets, bores, receivers and buried
    magnet cavities -- which are interior, or open only to a declared face --
    never register as plan growth.
    """
    slab = solid & (Pos(0.0, centre_y, z + 0.05) * Box(400.0, 400.0, 0.1))
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


def test_exterior_never_grows_rearward(variant: Variant) -> None:
    """Front-face-down, every layer must sit on the one before it.

    Each part is printed with z=18.3 on the bed, so print height runs with
    decreasing Z.  Requiring the exterior plan at each Z to lie inside the
    plan just in front of it is exactly the no-overhang condition for the
    whole outside of the part -- the skirt, the ear step, the entry collar,
    the lands and, on the opposed variant, the waist between them.  The one
    declared mate-face entry is filled back in first: a Ø6 bore through a wall
    is a bridge, not plan growth.
    """
    solid, _facts = _exported_solid(variant)
    if solid is None:
        return
    module = variant.module
    low, high = variant.z_span
    if variant.key == "coaxial":
        lands = [bmr_pod.MOUNT_AXIS_XY[1]]
    else:
        lands = [module.LOWER_AXIS_XY[1], module.UPPER_AXIS_XY[1]]
    envelope = _plan_prism(bmr_pod.entry_collar_plan(), *bmr_pod.ENTRY_COLLAR_Z)
    for axis_y in lands:
        envelope = envelope.fuse(
            Pos(0.0, axis_y, (low + high) / 2.0)
            * Cylinder(bmr_pod.POD_OUTER_R_MM, high - low))
    probe = solid.fuse(bmr_pod.duct_cutter() & envelope).clean()

    centre_y = bmr_pod.MOUNT_AXIS_XY[1]
    front = None
    for z in variant.silhouette_ladder:
        here = _filled_silhouette(probe, z, centre_y)
        assert here is not None, f"no cross section at z={z}"
        if front is not None:
            leak = (here - front).volume
            assert leak < 1.0e-6, (
                f"the exterior plan at z={z} reaches {leak:.6f} mm3 outside "
                "the plan in front of it; that is an overhang in the "
                "front-face-down print")
        front = here
    print(f"    exterior silhouette never grows rearward over "
          f"{len(variant.silhouette_ladder)} sections; no support needed "
          "outside the declared blind pockets")


def test_wing_clearance_is_unchanged(variant: Variant) -> None:
    """Both wing families must clear this part exactly as they clear the ND.

    The dropped land and its skirt reach down into plan territory the struts
    never touched, and the wings top out at y=449, so this is a real check
    rather than a formality on either variant.
    """
    solid, _facts = _exported_solid(variant)
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


FAMILY_TESTS = (
    test_vase_authority_is_mirrored_exactly,
    test_the_two_variants_share_one_family_module,
)

VARIANT_TESTS = (
    test_mount_constants_equal_the_released_joint_authority,
    test_depth_stack_is_the_variant_it_claims,
    test_pod_outer_wall_is_the_driver_land,
    test_pod_is_dropped_as_far_as_the_mate_allows,
    test_skirt_fills_the_junction_and_outsections_the_struts,
    test_cable_path_is_one_hidden_entry_and_one_declared_pass,
    test_captive_stations_are_the_vase_s_own,
    test_inherited_tweeter_clamp_holes_are_gone,
    test_no_declared_opening_reaches_the_assembled_exterior,
    test_candidate_flags_are_set,
    test_part_is_not_wired_into_the_release,
    test_slicing_profile_is_the_base_profile_derived,
    test_pause_plan_follows_the_station_geometry,
    test_delivery_validator_accepts_the_sliced_project,
    test_exported_solid_is_one_body_with_only_its_magnet_voids,
    test_declared_openings_are_the_only_openings,
    test_every_magnet_cavity_is_buried_behind_its_own_skin,
    test_mount_interface_is_geometrically_identical_to_the_released_ear,
    test_mate_simulation_against_the_staged_um_collar,
    test_the_assembled_junction_has_no_window,
    test_free_t_cable_reaches_the_declared_entry_without_exposure,
    test_exterior_never_grows_rearward,
    test_wing_clearance_is_unchanged,
)


def main() -> None:
    for test in FAMILY_TESTS:
        test()
        print(f"  PASS {test.__name__}")
    for variant in VARIANTS:
        print(f"--- {variant.module.PART_NAME} ({variant.key})")
        for test in VARIANT_TESTS:
            test(variant)
            print(f"  PASS {test.__name__}")
    total = len(FAMILY_TESTS) + len(VARIANTS) * len(VARIANT_TESTS)
    suffix = f"; {len(SKIPPED)} skipped gate(s)" if SKIPPED else ""
    print(f"BMR pod family: {len(FAMILY_TESTS)} shared + "
          f"{len(VARIANT_TESTS)} per-variant gates over {len(VARIANTS)} "
          f"variants = {total} focused gates pass{suffix}")


if __name__ == "__main__":
    main()
