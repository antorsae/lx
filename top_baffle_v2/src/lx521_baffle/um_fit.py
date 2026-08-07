"""MU10 terminal/Faston service proxy for fit review, not print geometry.

The acoustic MU10RB-SL mesh omits its electrical tabs and the available
datasheet does not dimension them.  This module therefore keeps the
uncertainty explicit and separate from the driver STL. It clocks the
terminal carrier at 283 degrees, midway between the 238 and 328 degree
mounting screws, adds a closed D98/D80/D60 body obstruction derived from
the open reference mesh, and models 6.3 mm female Faston service space.

V1L deliberately brings its UM cable onto that 283-degree axis.  Its
installed cable therefore occupies the outer end of this conservative
withdrawal volume: the box is a *service-motion* envelope, not a claim
that an attached cable can be absent.  The V1L split TPU strain relief
must still remain outside the box and seats on V1L's real rear plane at
z=6.8 rather than the proud family's z=0 face.

The Obi-Wan printed UM passage ends flush at the LM carrier's native R113
boundary. Its physical Ø7 cable then runs free behind UM, crosses the
283-degree D82 reference with a circumferential tangent at z=2.7, and turns
through a true R20 (above the R14 minimum) to a Y breakout; two explicit Ø3.2,
R8-minimum slack leads enter non-overlapping low-profile flag-Faston
proxies and support independent one-at-a-time 12 mm pull sweeps while the
opposite terminal stays installed. A conservative stepped
W22 keepout screens those free loops against the adjacent LM driver. The
deleted inward-radial D7 continuation and overlapping straight boots were
both impossible.

Measure the real carrier radius, tab spacing, rear projection, polarity
order, chosen insulation boots and pull-off direction before committing a
full print.  ``PHYSICAL_MEASURE_REQUIRED`` must remain true until those
measurements replace the proxy constants.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import hashlib
from pathlib import Path

from build123d import (
    Box,
    Circle,
    Compound,
    Cylinder,
    Face,
    Line,
    Plane,
    Polyline,
    Pos,
    Rectangle,
    Rot,
    Spline,
    ThreePointArc,
    Wire,
    extrude,
    loft,
    make_face,
    sweep,
)

from .base import (
    L22_CUTOUT,
    STAND_FOOT,
    UM_CUTOUT,
    UM_TERMINAL_CLOCK_DEG,
    UM_TERMINAL_GAP_DEG,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHYSICAL_MEASURE_REQUIRED = True

FASTON_TAB_W = 6.3
FASTON_TAB_T = 0.8
FASTON_TAB_EXPOSED_L = 12.0
FASTON_RECEPTACLE_RADIAL_L = 16.0
FASTON_RECEPTACLE_TANGENTIAL_W = 8.5
FASTON_RECEPTACLE_Z = 5.0
FASTON_PAIR_PITCH = 11.0
FASTON_BOOT_RADIAL_L = 18.0
FASTON_BOOT_TANGENTIAL_W = 9.5
FASTON_BOOT_Z = 7.0
FASTON_FLAG_WIRE_ENTRY_R = 55.0
FASTON_PULL_DISTANCE = 12.0
FASTON_PULL_FLEX_Z = 9.0
FASTON_LEAD_D = 3.2
FASTON_LEAD_MIN_BEND_R = 8.0
FASTON_BREAKOUT_LENGTH = 8.0
FASTON_LEAD_INSERT = 2.0
FASTON_LEAD_PULL_STATES_MM = (0.0, 3.0, 6.0, 9.0, 12.0)
FASTON_TERMINAL_IDS = (1, 2)
FASTON_LEAD_1_START_HANDLE_MM = 45.0
FASTON_LEAD_1_END_HANDLE_MM = 72.5
FASTON_LEAD_2_START_HANDLE_MM = 75.0
FASTON_LEAD_2_END_HANDLE_MM = 47.5
FASTON_BREAKOUT_BUNDLE_OVERLAP_MM = 5.0
FASTON_BREAKOUT_BUNDLE_OD = 8.0
FASTON_BREAKOUT_LEAD_OD = 4.0
# A short OD8 collar straddles the nominal jacket/lead split.  The D7
# jacket enters from +Z and both D3.2 leads leave toward -Z, so a collar
# extending to both sides of the split gives all three heat-shrink legs a
# positive-volume junction rather than three solids that merely share a cap.
FASTON_BREAKOUT_JUNCTION_LENGTH_MM = 4.0

TERMINAL_CONTACT_RADIUS = 40.5
TERMINAL_CARRIER_RADIUS = TERMINAL_CONTACT_RADIUS
TERMINAL_CARRIER_TANGENTIAL_W = 24.0
TERMINAL_CARRIER_RADIAL_L = 8.0
TERMINAL_CARRIER_Z = 6.0
TERMINAL_SERVICE_Z = -10.2

# Complete connector/removal capsule: 32 tangential x 40 radial x 10
# rear-depth. It begins outside the known body and includes the installed
# receptacles plus a conservative pull region.
REMOVAL_TANGENTIAL_W = 32.0
REMOVAL_RADIAL_L = 40.0
REMOVAL_REAR_Z = 10.0
REMOVAL_INNER_RADIUS = TERMINAL_CONTACT_RADIUS
REMOVAL_Z_CENTER = TERMINAL_SERVICE_Z

GROMMET_CABLE_D = 7.0
GROMMET_BARREL_D = 8.0
GROMMET_FLANGE_D = 13.0
GROMMET_SPLIT_GAP = 0.20

# V1L terminal-axis handoff: the D8/D7 TPU shank straddles the physical
# rear face and reaches 2.5 mm into the R14 bore.  Its 2 mm flat flange
# is wholly behind the rear face, leaving a large z-gap to the Faston
# service-motion box (whose closest face is z=-5).
V1L_GROMMET_INSERT_DEPTH = 2.5
V1L_GROMMET_FLANGE_T = 2.0
V1L_GROMMET_BORE_RADIAL_CLEARANCE = 0.05
V1L_REAR_CABLE_Z_MIN = -18.0

# Public service-envelope semantics for checks and review tooling.
# V1L's installed cable is the functional handoff to the terminals and
# intentionally enters the removal corridor; its strain relief does not.
V1L_CABLE_REMOVAL_OVERLAP_INTENTIONAL = True
REMOVAL_ENVELOPE_CABLE_POLICY = {
    "proud": "must_clear",
    "v1l": "intentional_terminal_handoff_overlap",
    "obiwan": "independent_flag_faston_pull_with_slack_leads",
}
REMOVAL_ENVELOPE_GROMMET_POLICY = {
    "proud": "must_clear",
    "v1l": "must_clear",
}

OBIWAN_TERMINATED_HANDOFF_R = 20.0
OBIWAN_TERMINATED_HANDOFF_STEPS = 40
MU10_BODY_MODEL_TOLERANCE_MM = 0.40
MU10_MIN_PRINTED_BODY_CLEARANCE_MM = 0.25

# Dimensioned terminal-less body reference. The raw STL is an open acoustic
# surface asset, so exact collisions use a closed BREP derived from the same
# public D98/D80/D60/43.6-mm envelope and four conservative 12-degree struts.
MU10_REFERENCE_STL = (
    PROJECT_ROOT.parent
    / "linkwitz" / "H1658-04_MU10RB-SL_driver.stl")
MU10_REFERENCE_STL_SHA256 = (
    "bb92511ee12bed3aa7db942b43d1f0e10127dd692bac224fb56f2b1ca9dff0a1")
MU10_FLANGE_R = 49.0
MU10_FRAME_R = 40.0
MU10_FRAME_INNER_R = 36.5
MU10_MOTOR_R = 30.0
MU10_SEAT_Z = 14.3
MU10_FRONT_PROUD = 5.4
MU10_REAR_DEPTH = 38.2
MU10_INTERMEDIATE_OUTER_R = 31.0
MU10_INTERMEDIATE_INNER_R = 20.0
MU10_STRUT_WIDTH_DEG = 12.0
MU10_WORLD_STRUT_ANGLES = (13.0, 103.0, 193.0, 283.0)
MU10_RAW_TO_OBIWAN_ROT_Z_DEG = 58.0

# Conservative stepped LM-driver rear keepout derived from the local
# E0022_W22EX001 shrinkwrap bounds. It deliberately fills basket gaps: the
# terminal slack loops must clear this harder envelope, not merely the open
# triangles. The source model's axis is mapped to world Z with its front at
# the flush baffle plane (z=18.3).
W22_REFERENCE_STEP = PROJECT_ROOT.parent / "E0022_W22EX001.stp"
W22_REFERENCE_STEP_SHA256 = (
    "7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6")
# Source-space bounds and datum are cached facts from the hash-pinned STEP so
# lightweight routing/report callers do not import OCC geometry.  The guarded
# ``test_w22_reference_step_geometry`` phase independently imports the actual
# STEP and verifies both these bounds and the structured placement below.  The
# source driver axis is +Y; Rot(X=+90deg) maps it to world +Z while source +Z
# maps to world -Y. Translation places the source front datum on the common
# baffle front plane at z=18.3.
W22_NATIVE_BOUNDS_MM = (
    (-110.5, -37.0, -110.5),
    (110.5, 65.798931, 110.5),
)
W22_REFERENCE_BOUNDS_TOLERANCE_MM = 0.02
W22_NATIVE_FRONT_Y_MM = W22_NATIVE_BOUNDS_MM[1][1]
W22_WORLD_FRONT_Z_MM = 18.3
W22_NATIVE_TO_WORLD_ROT_X_DEG = 90.0
W22_NATIVE_TO_WORLD_TRANSLATION_MM = (
    L22_CUTOUT[0],
    L22_CUTOUT[1],
    W22_WORLD_FRONT_Z_MM - W22_NATIVE_FRONT_Y_MM,
)
W22_WORLD_BOUNDS_MM = (
    (L22_CUTOUT[0] + W22_NATIVE_BOUNDS_MM[0][0],
     L22_CUTOUT[1] - W22_NATIVE_BOUNDS_MM[1][2],
     W22_NATIVE_TO_WORLD_TRANSLATION_MM[2]
     + W22_NATIVE_BOUNDS_MM[0][1]),
    (L22_CUTOUT[0] + W22_NATIVE_BOUNDS_MM[1][0],
     L22_CUTOUT[1] - W22_NATIVE_BOUNDS_MM[0][2],
     W22_NATIVE_TO_WORLD_TRANSLATION_MM[2]
     + W22_NATIVE_BOUNDS_MM[1][1]),
)
W22_BODY_STEPS = (
    (61.0, -85.0, -47.5),
    (80.0, -47.5, -12.5),
    (83.0, -12.5, -2.5),
    (94.0, -2.5, 7.5),
)
# Six padded world-AABB keepouts cover the real basket-spoke material that
# escapes the compact radial steps between z=-18.73 and -2.5.  Keeping the
# six lobes separate is materially more faithful than filling their complete
# R94 annulus: the latter falsely blocked terminal 1's pull corridor even
# though that corridor lies in a real basket gap.  Bounds come from the
# guarded, hash-pinned transformed-STEP subtraction and include >=0.10 mm
# padding on every face.
W22_BASKET_SPOKE_KEEP_OUT_BOUNDS_MM = (
    ((34.70, 122.55, -18.84), (48.58, 134.93, -2.39)),
    ((79.68, 193.94, -18.84), (88.47, 208.03, -2.39)),
    ((-48.58, 122.55, -18.84), (-34.70, 134.93, -2.39)),
    ((-88.47, 193.94, -18.84), (-79.68, 208.03, -2.39)),
    ((-48.58, 267.03, -18.84), (-34.70, 279.42, -2.39)),
    ((34.70, 267.03, -18.84), (48.58, 279.42, -2.39)),
)
# Add the declared 0.02-mm validation tolerance to the non-manufacturing
# proxy so OCCT's 0.0011-mm imported front-face overshoot is contained too.
W22_FLANGE_STEP = (110.52, 7.5, 18.32)


def _polar_xy(radius: float, tangential_offset: float = 0.0):
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    return (UM_CUTOUT[0] + radius * ux + tangential_offset * vx,
            UM_CUTOUT[1] + radius * uy + tangential_offset * vy)


def _local_box(radius: float, tangential_offset: float,
               z: float, radial_l: float, tangential_w: float,
               depth_z: float):
    x, y = _polar_xy(radius, tangential_offset)
    return (Pos(x, y, z) * Rot(Z=UM_TERMINAL_CLOCK_DEG)
            * Box(radial_l, tangential_w, depth_z))


def _validated_pull_mm(pull_mm: float) -> float:
    pull_mm = float(pull_mm)
    if not 0.0 <= pull_mm <= FASTON_PULL_DISTANCE:
        raise ValueError(pull_mm)
    return pull_mm


def _pulls_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]) -> dict[int, float]:
    """Return a validated, ordered two-terminal pull composition."""
    if not isinstance(pull_by_terminal_mm, Mapping):
        raise TypeError("pull_by_terminal_mm must map terminal ids 1 and 2")
    keys = set(pull_by_terminal_mm)
    expected = set(FASTON_TERMINAL_IDS)
    if keys != expected:
        raise ValueError(
            f"pull_by_terminal_mm keys must be {sorted(expected)}; "
            f"received {sorted(keys, key=str)}")
    return {
        terminal_id: _validated_pull_mm(pull_by_terminal_mm[terminal_id])
        for terminal_id in FASTON_TERMINAL_IDS
    }


def _uniform_pull_state(pull_mm: float) -> dict[int, float]:
    pull_mm = _validated_pull_mm(pull_mm)
    return {terminal_id: pull_mm for terminal_id in FASTON_TERMINAL_IDS}


def obiwan_independent_pull_state(
        terminal_id: int, pull_mm: float) -> dict[int, float]:
    """One-terminal service composition; the opposite Faston stays home.

    The public independent service states are intentionally discrete.  They
    represent the five inspected removal stations, not an animation claim
    about unverified hardware between stations.  The legacy scalar APIs
    remain continuous over 0..12 mm and move both terminals together.
    """
    if terminal_id not in FASTON_TERMINAL_IDS:
        raise ValueError(terminal_id)
    pull_mm = _validated_pull_mm(pull_mm)
    if not any(math.isclose(pull_mm, station, abs_tol=1e-9)
               for station in FASTON_LEAD_PULL_STATES_MM):
        raise ValueError(
            f"independent pull must use a declared station "
            f"{FASTON_LEAD_PULL_STATES_MM}; received {pull_mm:g}")
    return {
        candidate: pull_mm if candidate == terminal_id else 0.0
        for candidate in FASTON_TERMINAL_IDS
    }


def obiwan_independent_pull_states():
    """Structured one-at-a-time states for both terminals and all stations."""
    states = []
    for terminal_id in FASTON_TERMINAL_IDS:
        other_id = 2 if terminal_id == 1 else 1
        for station_mm in FASTON_LEAD_PULL_STATES_MM:
            states.append({
                "name": (
                    f"terminal_{terminal_id}_pull_{station_mm:g}mm_"
                    f"terminal_{other_id}_installed"),
                "active_terminal_id": terminal_id,
                "installed_terminal_id": other_id,
                "station_mm": station_mm,
                "pull_by_terminal_mm": obiwan_independent_pull_state(
                    terminal_id, station_mm),
                "other_terminal_remains_installed": True,
                "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
            })
    return tuple(states)


def terminal_carrier_proxy():
    return _local_box(
        TERMINAL_CARRIER_RADIUS, 0.0, TERMINAL_SERVICE_Z,
        TERMINAL_CARRIER_RADIAL_L,
        TERMINAL_CARRIER_TANGENTIAL_W,
        TERMINAL_CARRIER_Z,
    )


def faston_proxy_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Tabs plus receptacles at independent per-terminal pull positions."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    parts = {}
    for idx, off in enumerate((-FASTON_PAIR_PITCH / 2.0,
                               FASTON_PAIR_PITCH / 2.0), 1):
        parts[f"terminal_tab_{idx}"] = _local_box(
            TERMINAL_CONTACT_RADIUS + FASTON_TAB_EXPOSED_L / 2.0,
            off, TERMINAL_SERVICE_Z, FASTON_TAB_EXPOSED_L,
            FASTON_TAB_W, FASTON_TAB_T)
        parts[f"faston_receptacle_{idx}"] = _local_box(
            TERMINAL_CONTACT_RADIUS + FASTON_RECEPTACLE_RADIAL_L / 2.0
            + pulls[idx],
            off, TERMINAL_SERVICE_Z, FASTON_RECEPTACLE_RADIAL_L,
            FASTON_RECEPTACLE_TANGENTIAL_W, FASTON_RECEPTACLE_Z)
    return parts


def faston_proxy_parts_for_terminal_pull(terminal_id: int, pull_mm: float):
    return faston_proxy_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def faston_proxy_parts(pull_mm: float = 0.0):
    """Legacy scalar composition: both receptacles move by ``pull_mm``."""
    return faston_proxy_parts_by_terminal(_uniform_pull_state(pull_mm))


def faston_boot_proxy_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Provisional flag boots at independent per-terminal pull positions."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        f"faston_insulation_boot_{idx}": _local_box(
            TERMINAL_CONTACT_RADIUS + FASTON_BOOT_RADIAL_L / 2.0
            + pulls[idx],
            off, TERMINAL_SERVICE_Z,
            FASTON_BOOT_RADIAL_L,
            FASTON_BOOT_TANGENTIAL_W,
            FASTON_BOOT_Z)
        for idx, off in enumerate((-FASTON_PAIR_PITCH / 2.0,
                                   FASTON_PAIR_PITCH / 2.0), 1)
    }


def faston_boot_proxy_parts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return faston_boot_proxy_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def faston_boot_proxy_parts(pull_mm: float = 0.0):
    """Legacy scalar composition: both insulation boots move together."""
    return faston_boot_proxy_parts_by_terminal(_uniform_pull_state(pull_mm))


def faston_flag_wire_entry_face_points_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Wire-entry face axes at independent per-terminal positions."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    entries = {}
    half = FASTON_BOOT_TANGENTIAL_W / 2.0
    for idx, off in enumerate((-FASTON_PAIR_PITCH / 2.0,
                               FASTON_PAIR_PITCH / 2.0), 1):
        side = -1.0 if off < 0.0 else 1.0
        x, y = _polar_xy(
            FASTON_FLAG_WIRE_ENTRY_R + pulls[idx], off + side * half)
        entries[f"terminal_lead_{idx}"] = (x, y, TERMINAL_SERVICE_Z)
    return entries


def faston_flag_wire_entry_face_points(pull_mm: float = 0.0):
    return faston_flag_wire_entry_face_points_by_terminal(
        _uniform_pull_state(pull_mm))


def faston_flag_lead_endpoints_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Lead centers after 2 mm engagement at independent positions."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    entries = {}
    half = FASTON_BOOT_TANGENTIAL_W / 2.0
    for idx, off in enumerate((-FASTON_PAIR_PITCH / 2.0,
                               FASTON_PAIR_PITCH / 2.0), 1):
        side = -1.0 if off < 0.0 else 1.0
        t = off + side * (half - FASTON_LEAD_INSERT)
        x, y = _polar_xy(FASTON_FLAG_WIRE_ENTRY_R + pulls[idx], t)
        entries[f"terminal_lead_{idx}"] = (x, y, TERMINAL_SERVICE_Z)
    return entries


def faston_flag_lead_endpoints(pull_mm: float = 0.0):
    return faston_flag_lead_endpoints_by_terminal(
        _uniform_pull_state(pull_mm))


def faston_pull_sweep_parts():
    """Independent 12-mm radial removal sweeps for the two flag Fastons."""
    radial_l = FASTON_BOOT_RADIAL_L + FASTON_PULL_DISTANCE
    return {
        f"faston_flag_pull_sweep_{idx}": _local_box(
            TERMINAL_CONTACT_RADIUS + radial_l / 2.0,
            off,
            TERMINAL_SERVICE_Z,
            radial_l,
            FASTON_BOOT_TANGENTIAL_W,
            FASTON_PULL_FLEX_Z)
        for idx, off in enumerate((-FASTON_PAIR_PITCH / 2.0,
                                   FASTON_PAIR_PITCH / 2.0), 1)
    }


def removal_envelope():
    return _local_box(
        REMOVAL_INNER_RADIUS + REMOVAL_RADIAL_L / 2.0,
        0.0,
        REMOVAL_Z_CENTER,
        REMOVAL_RADIAL_L,
        REMOVAL_TANGENTIAL_W,
        REMOVAL_REAR_Z,
    )


def terminal_contact_allowance_envelope():
    """Named driver-contact zone; never a general body-collision waiver."""
    parts = [terminal_carrier_proxy()]
    parts.extend(part for name, part in faston_proxy_parts().items()
                 if name.startswith("terminal_tab_"))
    return Compound(children=parts)


def _annular_cylinder(inner_r, outer_r, z0, z1):
    outer = Pos(
        UM_CUTOUT[0], UM_CUTOUT[1], (z0 + z1) / 2.0
    ) * Cylinder(outer_r, z1 - z0)
    inner = Pos(
        UM_CUTOUT[0], UM_CUTOUT[1], (z0 + z1) / 2.0
    ) * Cylinder(inner_r, z1 - z0)
    return outer - inner


def _mu10_strut(angle_deg):
    native_outline = (
        (37.0, -2.5),
        (35.5, -8.0),
        (26.0, -27.5),
        (20.5, -29.0),
        (22.5, -23.5),
        (31.5, -7.0),
    )
    outline = [(r, MU10_SEAT_Z + native_z)
               for r, native_z in native_outline]
    outline.append(outline[0])
    face = Plane.XZ * Face(Wire(Polyline(*outline).edges()))
    width = 2.0 * 37.0 * math.sin(
        math.radians(MU10_STRUT_WIDTH_DEG / 2.0))
    local = extrude(face, amount=width / 2.0, both=True)
    return Pos(UM_CUTOUT[0], UM_CUTOUT[1], 0.0) * Rot(Z=angle_deg) * local


def mu10_body_keepout(include_flange=False):
    """Closed BREP body obstruction derived from the reference mesh source."""
    children = [
        _annular_cylinder(
            MU10_FRAME_INNER_R, MU10_FRAME_R,
            MU10_SEAT_Z - 5.5, MU10_SEAT_Z),
        _annular_cylinder(
            MU10_INTERMEDIATE_INNER_R, MU10_INTERMEDIATE_OUTER_R,
            MU10_SEAT_Z - 28.0, MU10_SEAT_Z - 22.0),
        Pos(UM_CUTOUT[0], UM_CUTOUT[1],
            (MU10_SEAT_Z - 22.8 + MU10_SEAT_Z - MU10_REAR_DEPTH) / 2.0)
        * Cylinder(
            MU10_MOTOR_R,
            (MU10_SEAT_Z - 22.8) - (MU10_SEAT_Z - MU10_REAR_DEPTH)),
        *(_mu10_strut(angle) for angle in MU10_WORLD_STRUT_ANGLES),
    ]
    if include_flange:
        children.append(_annular_cylinder(
            39.0, MU10_FLANGE_R,
            MU10_SEAT_Z, MU10_SEAT_Z + MU10_FRONT_PROUD))
    return Compound(children=children)


def mu10_body_reference_facts():
    digest = hashlib.sha256(MU10_REFERENCE_STL.read_bytes()).hexdigest()
    return {
        "raw_stl": str(MU10_REFERENCE_STL),
        "raw_stl_sha256": digest,
        "expected_raw_stl_sha256": MU10_REFERENCE_STL_SHA256,
        "seat_transform": (
            "Pos(0,366.081,14.3)*Rot(Z=58)*Rot(X=90)"),
        "raw_world_bounds": (
            (-49.0, 317.081, -23.9),
            (49.0, 415.081, 19.7),
        ),
        "world_strut_angles_deg": MU10_WORLD_STRUT_ANGLES,
        "terminals_present_in_raw_stl": False,
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
    }


def w22_body_keepout(include_flange=False):
    """Closed conservative LM basket/motor obstruction for harness review."""
    children = [
        Pos(L22_CUTOUT[0], L22_CUTOUT[1], (z0 + z1) / 2.0)
        * Cylinder(radius, z1 - z0)
        for radius, z0, z1 in W22_BODY_STEPS
    ]
    for lower, upper in W22_BASKET_SPOKE_KEEP_OUT_BOUNDS_MM:
        size = tuple(upper[index] - lower[index] for index in range(3))
        center = tuple(
            (lower[index] + upper[index]) / 2.0 for index in range(3))
        children.append(Pos(*center) * Box(*size))
    if include_flange:
        radius, z0, z1 = W22_FLANGE_STEP
        children.append(
            Pos(L22_CUTOUT[0], L22_CUTOUT[1], (z0 + z1) / 2.0)
            * Cylinder(radius, z1 - z0))
    return Compound(children=children)


def load_w22_reference_step_native():
    """Import the hash-pinned W22 reference in its native STEP frame.

    This intentionally remains an explicit validation-only operation; normal
    routing and service-envelope generation use the conservative stepped
    keepout and therefore never pay the STEP import's OCC memory cost.
    """
    digest = hashlib.sha256(W22_REFERENCE_STEP.read_bytes()).hexdigest()
    if digest != W22_REFERENCE_STEP_SHA256:
        raise RuntimeError(
            "W22 reference STEP digest changed: "
            f"expected {W22_REFERENCE_STEP_SHA256}, got {digest}")
    from build123d import import_step
    return import_step(str(W22_REFERENCE_STEP))


def w22_native_to_world_location():
    """Return the declared +90-degree-X/front-z=18.3 W22 placement."""
    return (
        Pos(*W22_NATIVE_TO_WORLD_TRANSLATION_MM)
        * Rot(X=W22_NATIVE_TO_WORLD_ROT_X_DEG)
    )


def place_w22_reference_step(native_reference):
    """Place a native-frame W22 reference with the declared transform."""
    return w22_native_to_world_location() * native_reference


def w22_body_reference_facts():
    digest = hashlib.sha256(W22_REFERENCE_STEP.read_bytes()).hexdigest()
    return {
        "source_step": str(W22_REFERENCE_STEP),
        "source_step_sha256": digest,
        "expected_source_step_sha256": W22_REFERENCE_STEP_SHA256,
        "conservative_steps": W22_BODY_STEPS,
        "flange_step": W22_FLANGE_STEP,
        "units": "mm",
        "native_bounds_mm": W22_NATIVE_BOUNDS_MM,
        "transformed_world_bounds_mm": W22_WORLD_BOUNDS_MM,
        "native_axes": {
            "x": "radial axis in source flange plane",
            "y": "driver axis, positive toward source front datum",
            "z": "radial axis in source flange plane",
        },
        "world_axes": {
            "x": "baffle horizontal, positive right",
            "y": "baffle vertical, positive toward UM",
            "z": "baffle depth, positive toward front face",
        },
        "world_center_front_datum_mm": (
            L22_CUTOUT[0], L22_CUTOUT[1], W22_WORLD_FRONT_Z_MM),
        "world_front_datum_z_mm": W22_WORLD_FRONT_Z_MM,
        "native_to_world": {
            "rotation": {
                "axis": "+X",
                "degrees": W22_NATIVE_TO_WORLD_ROT_X_DEG,
            },
            "translation_mm": W22_NATIVE_TO_WORLD_TRANSLATION_MM,
            "axis_map": {
                "native_+X": "world_+X",
                "native_+Y_driver_front": "world_+Z_baffle_front",
                "native_+Z": "world_-Y",
            },
            "homogeneous_matrix_row_major": (
                (1.0, 0.0, 0.0, W22_NATIVE_TO_WORLD_TRANSLATION_MM[0]),
                (0.0, 0.0, -1.0,
                 W22_NATIVE_TO_WORLD_TRANSLATION_MM[1]),
                (0.0, 1.0, 0.0,
                 W22_NATIVE_TO_WORLD_TRANSLATION_MM[2]),
                (0.0, 0.0, 0.0, 1.0),
            ),
        },
        "provenance": {
            "source_kind": "hash_pinned_manufacturer_shrinkwrap_STEP",
            "source_name": "E0022_W22EX001.stp",
            "source_sha256": W22_REFERENCE_STEP_SHA256,
            "source_units": "mm",
            "bounds_basis": (
                "cached exact source bounds; guarded W22-only geometry "
                "phase verifies them by runtime STEP import"),
            "bounds_validation_phase": "test_w22_reference_step_geometry",
            "bounds_validation_tolerance_mm": (
                W22_REFERENCE_BOUNDS_TOLERANCE_MM),
            "front_datum_basis": (
                "native max-Y face placed at baffle front z=18.3"),
            "keepout_basis": (
                "radially conservative stepped envelope rounded from "
                "the transformed source bounds"),
            "reference_geometry_scope": "W22EX001_only",
            "installed_u22_geometry_verified": False,
            "terminals_or_leads_verified": False,
        },
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
    }


def rear_cable_envelope(routing_profile: str):
    """Conservative D7 cable geometry for fit/service review.

    Proud retains the vertical continuation behind its rear outlet.  V1L
    returns the exact complete keyed route -- planar main, terminal refit,
    circular spatial handoff and rear continuation.  That installed V1L continuation
    intentionally enters the Faston service-motion envelope. Obi-Wan has no
    UM-carrier outlet: return its complete physical harness, which leaves the
    LM-owned printed passage at R113 and continues freely behind UM through
    the D82 terminal reference and breakout.
    """
    if routing_profile == "proud":
        from .cables import UM_HANDOFF

        x, y, _z = UM_HANDOFF["proud"]["rear_end"]
        z0, z1 = -18.0, 3.0
        return Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
            GROMMET_CABLE_D / 2.0, z1 - z0)
    if routing_profile == "v1l":
        from .cables import (
            UM_HANDOFF,
            UM_V1L_HANDOFF_KEY,
            um_path_wire,
        )

        # Exact circular D7 sweep on the same keyed wire that cuts V1L.
        # Continue its final -Z tangent behind the modeled rear endpoint
        # so the installed cable's deliberate service-corridor occupancy
        # is represented rather than silently truncated at z=-2.
        path = um_path_wire(um_handoff_key=UM_V1L_HANDOFF_KEY)
        section = (Plane(origin=path @ 0, z_dir=path % 0)
                   * Circle(GROMMET_CABLE_D / 2.0))
        complete = sweep(section, path=path)
        spec = UM_HANDOFF[UM_V1L_HANDOFF_KEY]
        x, y, outlet_z = spec["outlet"]
        continuation = Pos(
            x, y, (V1L_REAR_CABLE_Z_MIN + outlet_z) / 2.0
        ) * Cylinder(
            GROMMET_CABLE_D / 2.0,
            outlet_z - V1L_REAR_CABLE_Z_MIN,
        )
        return complete.fuse(continuation)
    if routing_profile == "obiwan":
        return Compound(children=list(obiwan_terminal_harness_parts().values()))
    raise ValueError(routing_profile)


def obiwan_terminated_cable_points():
    """Complete D7 bundle through its G1 R20 terminal breakout.

    The printed duct ends flush at the LM carrier's native R113 boundary.
    The physical bundle continues freely behind the UM carrier, crosses the
    283-degree D82 reference with a circumferential tangent, then turns toward
    -Z on a true tangent/Z R20 quarter circle while remaining outside the
    known D60 motor and terminal carrier. Its jacket stops at a named two-lead
    heat-shrink breakout; the D7
    solid is never falsely intersected with both connector bodies.
    """
    from .obiwan.route import (
        UM_MOUTH_TANGENT,
        route_cable_points,
    )

    route = [tuple(map(float, point))
             for point in route_cable_points(spacing_mm=1.5)]
    start = route[-1]
    plan_length = math.hypot(*UM_MOUTH_TANGENT)
    vx, vy = (UM_MOUTH_TANGENT[0] / plan_length,
              UM_MOUTH_TANGENT[1] / plan_length)
    arc = []
    for index in range(1, OBIWAN_TERMINATED_HANDOFF_STEPS + 1):
        phi = (math.pi * 0.5 * index
               / OBIWAN_TERMINATED_HANDOFF_STEPS)
        z = (start[2] - OBIWAN_TERMINATED_HANDOFF_R
             * (1.0 - math.cos(phi)))
        tangential = OBIWAN_TERMINATED_HANDOFF_R * math.sin(phi)
        arc.append((start[0] + tangential * vx,
                    start[1] + tangential * vy, z))
    return route + arc


def _cubic_points(p0, p1, p2, p3, count=321):
    points = []
    for index in range(count):
        q = index / (count - 1)
        r = 1.0 - q
        points.append(tuple(
            r ** 3 * p0[axis]
            + 3.0 * r ** 2 * q * p1[axis]
            + 3.0 * r * q ** 2 * p2[axis]
            + q ** 3 * p3[axis]
            for axis in range(3)))
    return points


def _polyline_length(points):
    return sum(math.dist(a, b) for a, b in zip(points[:-1], points[1:]))


def _prefix_by_length(points, distance_mm: float):
    """Polyline prefix ending at an interpolated arc-length station."""
    if distance_mm <= 0.0:
        return [tuple(points[0]), tuple(points[1])]
    out = [tuple(points[0])]
    travelled = 0.0
    for a, b in zip(points[:-1], points[1:]):
        segment = math.dist(a, b)
        if travelled + segment >= distance_mm:
            q = (distance_mm - travelled) / max(segment, 1e-12)
            out.append(tuple(a[i] + q * (b[i] - a[i]) for i in range(3)))
            return out
        out.append(tuple(b))
        travelled += segment
    return out


def _suffix_by_length(points, distance_mm: float):
    return list(reversed(_prefix_by_length(
        list(reversed(points)), distance_mm)))


def _lead_local_specs():
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    vx, vy = -math.sin(a), math.cos(a)
    breakout = obiwan_terminated_cable_points()[-1]
    breakout_t = (
        (breakout[0] - UM_CUTOUT[0]) * vx
        + (breakout[1] - UM_CUTOUT[1]) * vy)
    return (
        # name, split t, wire-entry t, end tangent in local (r,t,z),
        # installed start/end Bezier handle lengths.
        ("terminal_lead_1", breakout_t - FASTON_LEAD_D / 2.0,
         -FASTON_PAIR_PITCH / 2.0
         - FASTON_BOOT_TANGENTIAL_W / 2.0 + FASTON_LEAD_INSERT,
         (0.0, 1.0, 0.0),
         FASTON_LEAD_1_START_HANDLE_MM, FASTON_LEAD_1_END_HANDLE_MM),
        ("terminal_lead_2", breakout_t + FASTON_LEAD_D / 2.0,
         FASTON_PAIR_PITCH / 2.0
         + FASTON_BOOT_TANGENTIAL_W / 2.0 - FASTON_LEAD_INSERT,
         (0.0, -1.0, 0.0),
         FASTON_LEAD_2_START_HANDLE_MM, FASTON_LEAD_2_END_HANDLE_MM),
    )


def _solved_lead_local_points(
        split_t, entry_t, end_tangent, installed_start_h, end_h, pull_mm):
    """Fixed-length cubic lead at one radial Faston pull station."""
    breakout_z = obiwan_terminated_cable_points()[-1][2]
    p0 = (TERMINAL_CONTACT_RADIUS, split_t, breakout_z)

    def curve(start_h, pull):
        p3 = (FASTON_FLAG_WIRE_ENTRY_R + pull,
              entry_t, TERMINAL_SERVICE_Z)
        p1 = (p0[0], p0[1], p0[2] - start_h)
        p2 = tuple(p3[i] - end_h * end_tangent[i] for i in range(3))
        return _cubic_points(p0, p1, p2, p3)

    installed = curve(installed_start_h, 0.0)
    target = _polyline_length(installed)
    if pull_mm == 0.0:
        return installed, installed_start_h, target

    low = 0.0
    high = installed_start_h
    low_length = _polyline_length(curve(low, pull_mm))
    high_length = _polyline_length(curve(high, pull_mm))
    if low_length > target + 1e-6:
        raise RuntimeError(
            f"Faston pull {pull_mm:g} mm cannot preserve lead length; "
            f"minimum {low_length:.3f} > installed {target:.3f}")
    while high_length < target:
        high *= 1.5
        high_length = _polyline_length(curve(high, pull_mm))
        if high > 250.0:
            raise RuntimeError("Faston fixed-length lead solve did not bracket")
    for _ in range(60):
        mid = (low + high) / 2.0
        if _polyline_length(curve(mid, pull_mm)) < target:
            low = mid
        else:
            high = mid
    solved = (low + high) / 2.0
    points = curve(solved, pull_mm)
    if abs(_polyline_length(points) - target) > 1e-6:
        raise RuntimeError("Faston fixed-length lead solve drifted")
    return points, solved, target


def obiwan_terminal_lead_points_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Two fixed-length D3.2 leads at independent pull positions.

    Local coordinates are radial/tangential/Z about the 283-degree axis.
    Both conductors begin inside the D7 breakout, move rearward into the
    open inter-driver service volume, and approach the outward boot side
    face with the flag wire-entry tangent. Each connector-side control may
    translate independently while its start handle is solved to preserve
    that conductor's installed length.
    """
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    # The exact R20 endpoint has local r=40.5, t=20.0. R20 is the minimum
    # whole-millimetre free-cable bend that preserves 0.8 mm to the current
    # conservative terminal-carrier proxy; the printed plan handoff is R15,
    # one millimetre above the required R14 minimum.
    leads = {}
    for terminal_id, (name, split_t, entry_t, end_tangent,
                      installed_start_h, end_h) in enumerate(
                          _lead_local_specs(), 1):
        local, _solved_start_h, _target = _solved_lead_local_points(
            split_t, entry_t, end_tangent,
            installed_start_h, end_h, pulls[terminal_id])
        leads[name] = [
            (UM_CUTOUT[0] + r * ux + t * vx,
             UM_CUTOUT[1] + r * uy + t * vy,
             z)
            for r, t, z in local
        ]
    return leads


def obiwan_terminal_lead_points_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return obiwan_terminal_lead_points_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_terminal_lead_points(pull_mm: float = 0.0):
    """Legacy scalar composition: both fixed-length leads move together."""
    return obiwan_terminal_lead_points_by_terminal(
        _uniform_pull_state(pull_mm))


OBIWAN_TERMINAL_HARNESS_PART_NAMES = (
    "obiwan_D7_bundle_to_Y_breakout",
    "obiwan_terminal_lead_1_D3p2",
    "obiwan_terminal_lead_2_D3p2",
)


def obiwan_terminal_harness_part_by_terminal(
        part_name: str, pull_by_terminal_mm: Mapping[int, float]):
    """Build one physical harness solid for bounded-memory validation."""
    from .cables import _tube_loft

    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    bundle_radius = ((GROMMET_CABLE_D / 2.0)
                     / math.cos(math.pi / 24.0))
    lead_radius = ((FASTON_LEAD_D / 2.0)
                   / math.cos(math.pi / 24.0))
    if part_name == "obiwan_D7_bundle_to_Y_breakout":
        return _tube_loft(
            obiwan_terminated_cable_points(), bundle_radius, sides=24)
    lead_name = part_name.removeprefix("obiwan_").removesuffix("_D3p2")
    if part_name not in OBIWAN_TERMINAL_HARNESS_PART_NAMES:
        raise ValueError(f"unknown Obi-Wan harness part: {part_name}")
    points = obiwan_terminal_lead_points_by_terminal(pulls)[lead_name]
    return _tube_loft(points, lead_radius, sides=24)


def obiwan_terminal_harness_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Physical D7 jacket plus independently positioned D3.2 leads."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        name: obiwan_terminal_harness_part_by_terminal(name, pulls)
        for name in OBIWAN_TERMINAL_HARNESS_PART_NAMES
    }


def obiwan_terminal_harness_parts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return obiwan_terminal_harness_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_terminal_harness_parts(pull_mm: float = 0.0):
    """Legacy scalar harness composition; both leads move together."""
    return obiwan_terminal_harness_parts_by_terminal(
        _uniform_pull_state(pull_mm))


def _obiwan_y_breakout_paths_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        "bundle": _suffix_by_length(
            obiwan_terminated_cable_points(),
            FASTON_BREAKOUT_BUNDLE_OVERLAP_MM),
        "leads": {
            name: _prefix_by_length(points, FASTON_BREAKOUT_LENGTH)
            for name, points in obiwan_terminal_lead_points_by_terminal(
                pulls).items()
        },
    }


def _obiwan_y_breakout_junction():
    """OD8 collar overlapping the incoming jacket and both branch legs."""
    x, y, z = obiwan_terminated_cable_points()[-1]
    return Pos(x, y, z) * Cylinder(
        FASTON_BREAKOUT_BUNDLE_OD / 2.0,
        FASTON_BREAKOUT_JUNCTION_LENGTH_MM,
    )


OBIWAN_Y_BREAKOUT_BOOT_PART_NAMES = (
    "obiwan_Y_breakout_bundle_heatshrink",
    "obiwan_Y_breakout_terminal_lead_1_heatshrink",
    "obiwan_Y_breakout_terminal_lead_2_heatshrink",
)


def obiwan_y_breakout_boot_part_by_terminal(
        part_name: str, pull_by_terminal_mm: Mapping[int, float]):
    """Build one Y-boot leg for bounded-memory validation."""
    from .cables import _tube_loft

    if part_name not in OBIWAN_Y_BREAKOUT_BOOT_PART_NAMES:
        raise ValueError(f"unknown Obi-Wan Y-boot part: {part_name}")
    paths = _obiwan_y_breakout_paths_by_terminal(pull_by_terminal_mm)
    bundle_r = ((FASTON_BREAKOUT_BUNDLE_OD / 2.0)
                / math.cos(math.pi / 24.0))
    lead_r = ((FASTON_BREAKOUT_LEAD_OD / 2.0)
              / math.cos(math.pi / 24.0))
    if part_name == "obiwan_Y_breakout_bundle_heatshrink":
        return _tube_loft(
            paths["bundle"], bundle_r, sides=24).fuse(
                _obiwan_y_breakout_junction()).clean()
    lead_name = part_name.removeprefix(
        "obiwan_Y_breakout_").removesuffix("_heatshrink")
    return _tube_loft(paths["leads"][lead_name], lead_r, sides=24)


def obiwan_y_breakout_boot_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Heat-shrink legs with a positive-volume OD8 fused Y junction.

    The named legacy bundle part owns the central collar.  Each OD4 branch
    penetrates that collar for half its length, so fusing the three returned
    parts produces one connected envelope rather than a cap-touching shell.
    """
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        name: obiwan_y_breakout_boot_part_by_terminal(name, pulls)
        for name in OBIWAN_Y_BREAKOUT_BOOT_PART_NAMES
    }


def obiwan_y_breakout_boot_parts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return obiwan_y_breakout_boot_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_y_breakout_boot_parts(pull_mm: float = 0.0):
    """Legacy scalar Y-boot composition; both branch leads move together."""
    return obiwan_y_breakout_boot_parts_by_terminal(
        _uniform_pull_state(pull_mm))


def obiwan_y_breakout_boot_envelope_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """One fused printable/proxy envelope for the complete Y boot."""
    parts = obiwan_y_breakout_boot_parts_by_terminal(pull_by_terminal_mm)
    names = tuple(parts)
    envelope = parts[names[0]]
    for name in names[1:]:
        envelope = envelope.fuse(parts[name])
    return envelope.clean()


def obiwan_y_breakout_boot_envelope(pull_mm: float = 0.0):
    return obiwan_y_breakout_boot_envelope_by_terminal(
        _uniform_pull_state(pull_mm))


def obiwan_y_breakout_boot_envelope_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return obiwan_y_breakout_boot_envelope_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_y_breakout_cable_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Underlying D7/D3.2 cable segments that the Y boot must contain."""
    from .cables import _tube_loft

    paths = _obiwan_y_breakout_paths_by_terminal(pull_by_terminal_mm)
    bundle_r = ((GROMMET_CABLE_D / 2.0)
                / math.cos(math.pi / 24.0))
    lead_r = ((FASTON_LEAD_D / 2.0)
              / math.cos(math.pi / 24.0))
    parts = {
        "obiwan_Y_underlying_D7_bundle": _tube_loft(
            paths["bundle"], bundle_r, sides=24),
    }
    for name, points in paths["leads"].items():
        parts[f"obiwan_Y_underlying_{name}_D3p2"] = _tube_loft(
            points, lead_r, sides=24)
    return parts


def obiwan_y_breakout_cable_parts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    return obiwan_y_breakout_cable_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_y_breakout_cable_parts(pull_mm: float = 0.0):
    return obiwan_y_breakout_cable_parts_by_terminal(
        _uniform_pull_state(pull_mm))


def obiwan_y_breakout_facts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Analytic Y-junction construction and cable-containment contract."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    paths = _obiwan_y_breakout_paths_by_terminal(pulls)
    breakout = tuple(obiwan_terminated_cable_points()[-1])
    a = math.radians(UM_TERMINAL_CLOCK_DEG)
    vx, vy = -math.sin(a), math.cos(a)
    breakout_t = (
        (breakout[0] - UM_CUTOUT[0]) * vx
        + (breakout[1] - UM_CUTOUT[1]) * vy)
    lead_offsets = {
        name: abs(split_t - breakout_t)
        for (name, split_t, _entry_t, _end_tangent,
             _installed_start_h, _end_h) in _lead_local_specs()
    }
    junction_cable_margin = min(
        FASTON_BREAKOUT_BUNDLE_OD / 2.0 - GROMMET_CABLE_D / 2.0,
        *(FASTON_BREAKOUT_BUNDLE_OD / 2.0
          - offset - FASTON_LEAD_D / 2.0
          for offset in lead_offsets.values()),
    )
    return {
        "units": "mm",
        "pull_by_terminal_mm": pulls,
        "breakout_center_world_mm": breakout,
        "bundle_boot_od_mm": FASTON_BREAKOUT_BUNDLE_OD,
        "branch_boot_od_mm": FASTON_BREAKOUT_LEAD_OD,
        "underlying_bundle_od_mm": GROMMET_CABLE_D,
        "underlying_branch_od_mm": FASTON_LEAD_D,
        "bundle_radial_wall_mm": (
            (FASTON_BREAKOUT_BUNDLE_OD - GROMMET_CABLE_D) / 2.0),
        "branch_radial_wall_mm": (
            (FASTON_BREAKOUT_LEAD_OD - FASTON_LEAD_D) / 2.0),
        "junction_length_mm": FASTON_BREAKOUT_JUNCTION_LENGTH_MM,
        "junction_overlap_each_side_mm": (
            FASTON_BREAKOUT_JUNCTION_LENGTH_MM / 2.0),
        "lead_center_offsets_from_bundle_mm": lead_offsets,
        "junction_min_underlying_cable_margin_mm": junction_cable_margin,
        "junction_construction": (
            "OD8 Z collar fused into incoming OD8 tail; both OD4 branch "
            "legs penetrate the centered collar with positive volume"),
        "bundle_path_endpoints_world_mm": (
            tuple(paths["bundle"][0]), tuple(paths["bundle"][-1])),
        "branch_path_endpoints_world_mm": {
            name: (tuple(points[0]), tuple(points[-1]))
            for name, points in paths["leads"].items()
        },
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
    }


def obiwan_y_breakout_facts(pull_mm: float = 0.0):
    facts = obiwan_y_breakout_facts_by_terminal(_uniform_pull_state(pull_mm))
    facts["pull_state_mm"] = float(pull_mm)
    return facts


def obiwan_y_breakout_facts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    facts = obiwan_y_breakout_facts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))
    facts.update({
        "active_terminal_id": terminal_id,
        "installed_terminal_id": 2 if terminal_id == 1 else 1,
        "pull_station_mm": float(pull_mm),
        "other_terminal_remains_installed": True,
    })
    return facts


def obiwan_separated_lead_part_by_terminal(
        terminal_id: int, pull_by_terminal_mm: Mapping[int, float]):
    """Build one physical lead after the intentional 8-mm Y overlap."""
    from .cables import _tube_loft

    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    if terminal_id not in FASTON_TERMINAL_IDS:
        raise ValueError(terminal_id)
    lead_radius = ((FASTON_LEAD_D / 2.0)
                   / math.cos(math.pi / 24.0))
    name = f"terminal_lead_{terminal_id}"
    points = obiwan_terminal_lead_points_by_terminal(pulls)[name]
    return _tube_loft(
        _suffix_by_length(points, FASTON_BREAKOUT_LENGTH),
        lead_radius, sides=24)


def obiwan_separated_lead_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Physical lead solids after the intentional 8-mm Y overlap."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        f"terminal_lead_{terminal_id}":
            obiwan_separated_lead_part_by_terminal(terminal_id, pulls)
        for terminal_id in FASTON_TERMINAL_IDS
    }


def obiwan_separated_lead_parts(pull_mm: float = 0.0):
    return obiwan_separated_lead_parts_by_terminal(
        _uniform_pull_state(pull_mm))


def obiwan_terminal_harness_facts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    leads = obiwan_terminal_lead_points_by_terminal(pulls)
    entries = faston_flag_lead_endpoints_by_terminal(pulls)
    face_points = faston_flag_wire_entry_face_points_by_terminal(pulls)
    solved_handles = {}
    installed_lengths = {}
    for terminal_id, (name, split_t, entry_t, end_tangent,
                      installed_start_h, end_h) in enumerate(
                          _lead_local_specs(), 1):
        _points, solved, target = _solved_lead_local_points(
            split_t, entry_t, end_tangent,
            installed_start_h, end_h, pulls[terminal_id])
        solved_handles[name] = solved
        installed_lengths[name] = target
    return {
        "bundle_breakout": tuple(obiwan_terminated_cable_points()[-1]),
        "lead_endpoints": {
            name: tuple(points[-1]) for name, points in leads.items()},
        "lead_engagement_points": entries,
        "wire_entry_face_points": face_points,
        "lead_lengths_mm": {name: _polyline_length(points)
                            for name, points in leads.items()},
        "installed_lead_lengths_mm": installed_lengths,
        "solved_start_handles_mm": solved_handles,
        "pull_by_terminal_mm": pulls,
        "pull_distance_mm": FASTON_PULL_DISTANCE,
        "breakout_length_mm": FASTON_BREAKOUT_LENGTH,
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
    }


def obiwan_terminal_harness_facts_for_terminal_pull(
        terminal_id: int, pull_mm: float):
    facts = obiwan_terminal_harness_facts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))
    facts.update({
        "active_terminal_id": terminal_id,
        "installed_terminal_id": 2 if terminal_id == 1 else 1,
        "pull_station_mm": float(pull_mm),
        "other_terminal_remains_installed": True,
    })
    return facts


def obiwan_terminal_harness_facts(pull_mm: float = 0.0):
    """Legacy scalar facts; retain the original ``pull_state_mm`` field."""
    facts = obiwan_terminal_harness_facts_by_terminal(
        _uniform_pull_state(pull_mm))
    facts["pull_state_mm"] = float(pull_mm)
    return facts


def obiwan_terminal_service_parts_by_terminal(
        pull_by_terminal_mm: Mapping[int, float]):
    """Compose connectors, boots, leads, harness and Y boot for a state."""
    pulls = _pulls_by_terminal(pull_by_terminal_mm)
    return {
        **faston_proxy_parts_by_terminal(pulls),
        **faston_boot_proxy_parts_by_terminal(pulls),
        **obiwan_terminal_harness_parts_by_terminal(pulls),
        **obiwan_y_breakout_boot_parts_by_terminal(pulls),
    }


def obiwan_terminal_service_state_parts(
        terminal_id: int, pull_mm: float):
    """Complete one-terminal pull state with the opposite side installed."""
    return obiwan_terminal_service_parts_by_terminal(
        obiwan_independent_pull_state(terminal_id, pull_mm))


def obiwan_terminal_service_state_facts(
        terminal_id: int, pull_mm: float):
    pulls = obiwan_independent_pull_state(terminal_id, pull_mm)
    other_id = 2 if terminal_id == 1 else 1
    return {
        "active_terminal_id": terminal_id,
        "installed_terminal_id": other_id,
        "station_mm": float(pull_mm),
        "pull_by_terminal_mm": pulls,
        "other_terminal_remains_installed": True,
        "connector_part_names": (
            "terminal_tab_1", "faston_receptacle_1",
            "terminal_tab_2", "faston_receptacle_2"),
        "boot_part_names": (
            "faston_insulation_boot_1", "faston_insulation_boot_2"),
        "harness": obiwan_terminal_harness_facts_by_terminal(pulls),
        "y_breakout": obiwan_y_breakout_facts_by_terminal(pulls),
        "physical_measure_required": PHYSICAL_MEASURE_REQUIRED,
    }


def obiwan_lm_cable_envelope():
    """Reference-only D7.8 short LM lead floating behind the carrier."""
    from .cables import _tube_loft
    from .obiwan.route import (
        LM_CABLE_D_EST,
        lm_cable_points,
    )

    points = [tuple(map(float, point))
              for point in lm_cable_points(spacing_mm=0.75)]
    radius = ((LM_CABLE_D_EST / 2.0)
              / math.cos(math.pi / 24.0))
    return _tube_loft(points, radius, sides=24)


def obiwan_ts_cable_envelope():
    """Reference-only tweeter cable through the crown-crossover D6 path."""
    from .cables import _tube_loft
    from .obiwan.route import (
        TS_CABLE_D_EST,
        ts_cable_points,
    )

    points = [tuple(map(float, point))
              for point in ts_cable_points(spacing_mm=1.25)]
    radius = ((TS_CABLE_D_EST / 2.0)
              / math.cos(math.pi / 24.0))
    return _tube_loft(points, radius, sides=24)


def _split_about_route_plane(full, routing_profile: str):
    """Split the proud insert through its R14 centerline/Z plane."""
    from .cables import UM_HANDOFF

    if routing_profile != "proud":
        raise ValueError(routing_profile)
    spec = UM_HANDOFF[routing_profile]
    x, y, _z = spec["rear_end"]
    tx, ty, _tz = spec["tangent"]
    nx, ny = -ty, tx
    angle_n = math.degrees(math.atan2(ny, nx))
    extent = 30.0
    width = 50.0
    gap = GROMMET_SPLIT_GAP

    def half(sign: float):
        cx = x + sign * nx * (extent / 2.0 + gap / 4.0)
        cy = y + sign * ny * (extent / 2.0 + gap / 4.0)
        clip = (Pos(cx, cy, -0.5) * Rot(Z=angle_n)
                * Box(extent - gap / 2.0, width, 30.0))
        return full & clip

    return half(1.0), half(-1.0)


def _proud_curved_grommet():
    """Short flexible shank following the final R14 segment.

    The proud outlet has no external straight socket.  A straight barrel
    placed behind it was floating clear of the baffle and could not enter
    the curved bore.  This D8.0/D7.0 split shank follows the actual last
    5.5 mm of centerline and terminates in a flange seated at rear z=0.
    """
    from .cables import UM_HANDOFF, UM_HANDOFF_R_MM

    spec = UM_HANDOFF["proud"]
    sx, sy, sz = spec["start"]
    tx, ty, _tz = spec["tangent"]
    radius = UM_HANDOFF_R_MM
    insert_z = 2.5
    phi0 = math.acos(1.0 - (sz - insert_z) / radius)

    def point(phi: float):
        return (sx + radius * math.sin(phi) * tx,
                sy + radius * math.sin(phi) * ty,
                sz - radius * (1.0 - math.cos(phi)))

    p0 = point(phi0)
    pm = point((phi0 + math.pi / 2.0) / 2.0)
    p1 = point(math.pi / 2.0)
    path = Wire([ThreePointArc(p0, pm, p1),
                 Line(p1, spec["rear_end"])])

    def tube(radius_mm: float):
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(radius_mm)
        return sweep(section, path=path)

    shank = tube(GROMMET_BARREL_D / 2.0) - tube(GROMMET_CABLE_D / 2.0)
    x, y, _z = spec["rear_end"]

    def cyl(radius_mm: float, z0: float, z1: float):
        return Pos(x, y, (z0 + z1) / 2.0) * Cylinder(radius_mm, z1 - z0)

    flange = (cyl(GROMMET_FLANGE_D / 2.0, -2.0, 0.0)
              - cyl(GROMMET_CABLE_D / 2.0, -2.2, 0.2))
    return shank + flange


def _v1l_terminal_curved_grommet():
    """D8/D7 split TPU shank for V1L's keyed circular handoff.

    The shank follows the exact analytic spatial arc inside the rear face and
    its tangent lead outside it.  A flat annular flange occupies
    z=(rear-2)..rear, so its seating face is exactly z=6.8 and no part
    of the strain relief enters the z=-15..-5 Faston motion volume.
    """
    from .cables import (
        UM_HANDOFF,
        UM_V1L_HANDOFF_KEY,
        UM_V1L_REAR_FACE_Z_MM,
    )

    spec = UM_HANDOFF[UM_V1L_HANDOFF_KEY]
    sx, sy, sz = spec["start"]
    tangent = tuple(spec["tangent"])
    normal = tuple(spec["arc_normal"])
    radius = spec["arc_radius_mm"]
    arc_angle = spec["arc_angle_rad"]
    face = tuple(spec["rear_face_axis_point"])
    face_tangent = tuple(spec["face_tangent"])
    rear_z = UM_V1L_REAR_FACE_Z_MM
    inner_z = rear_z + V1L_GROMMET_INSERT_DEPTH
    outer_z = rear_z - V1L_GROMMET_FLANGE_T

    def phi_at_z(z: float) -> float:
        normal_z = normal[2]
        if abs(normal_z) <= 1.0e-12:
            raise ValueError("V1L spatial arc has no rearward component")
        cos_phi = 1.0 - (z - sz) / (radius * normal_z)
        if not -1.0 <= cos_phi <= 1.0:
            raise ValueError(f"z={z:g} is outside the V1L circular handoff")
        phi = math.acos(cos_phi)
        if phi > arc_angle + 1.0e-9:
            raise ValueError(f"z={z:g} lies beyond the V1L rear face")
        return phi

    def point(phi: float):
        start = (sx, sy, sz)
        return tuple(
            start[index]
            + radius * math.sin(phi) * tangent[index]
            + radius * (1.0 - math.cos(phi)) * normal[index]
            for index in range(3)
        )

    def external_point_at_z(z: float):
        if face_tangent[2] >= -1.0e-12:
            raise ValueError("V1L face tangent does not leave through rear")
        scale = (z - face[2]) / face_tangent[2]
        return tuple(
            face[index] + scale * face_tangent[index]
            for index in range(3)
        )

    def path_from_inner_to_external(inner: float, external: float):
        phi0 = phi_at_z(inner)
        arc = ThreePointArc(
            point(phi0),
            point((phi0 + arc_angle) / 2.0),
            face,
        )
        return Wire([arc, Line(face, external_point_at_z(external))])

    body_path = path_from_inner_to_external(inner_z, outer_z)
    # Extend only the subtractive bore beyond both body ends; this avoids
    # coplanar inner caps and guarantees a through D7 passage after fuse.
    # Continue the subtractive bore to the tangent outlet.  Because the cable
    # crosses the flat flange obliquely, stopping its centerline only
    # 0.6 mm behind the flange leaves a clipped crescent at the rear face.
    # The full arc removes that cap artifact while adding no printable
    # material outside the flange.
    bore_path = path_from_inner_to_external(
        inner_z + 0.6, spec["outlet"][2])

    def swept(path, radius_mm: float):
        section = Plane(origin=path @ 0, z_dir=path % 0) * Circle(radius_mm)
        return sweep(section, path=path)

    shank_blank = swept(body_path, GROMMET_BARREL_D / 2.0)
    # D7 is the guaranteed cable envelope; add 0.05 mm radial fit
    # clearance so coincident swept faces cannot pinch a real cable or
    # leave numerical slivers after the split boolean.  The D8 body thus
    # retains a 0.45 mm minimum nominal TPU wall.
    cable_bore = swept(
        bore_path,
        GROMMET_CABLE_D / 2.0 + V1L_GROMMET_BORE_RADIAL_CLEARANCE,
    )
    rear_x, rear_y, rear_face_z = spec["rear_face_axis_point"]
    flange_blank = Pos(
        rear_x,
        rear_y,
        rear_face_z - V1L_GROMMET_FLANGE_T / 2.0,
    ) * Cylinder(GROMMET_FLANGE_D / 2.0, V1L_GROMMET_FLANGE_T)
    return shank_blank.fuse(flange_blank) - cable_bore


def _split_v1l_terminal_grommet(full):
    """Split the V1L insert in its route/Z plane with a 0.2 mm gap."""
    from .cables import (
        UM_HANDOFF,
        UM_V1L_HANDOFF_KEY,
        UM_V1L_REAR_FACE_Z_MM,
    )

    spec = UM_HANDOFF[UM_V1L_HANDOFF_KEY]
    x, y, _z = spec["rear_face_axis_point"]
    tx, ty, _tz = spec["tangent"]
    nx, ny = -ty, tx
    angle_n = math.degrees(math.atan2(ny, nx))
    extent = 30.0
    width = 50.0
    gap = GROMMET_SPLIT_GAP

    def half(sign: float):
        cx = x + sign * nx * (extent / 2.0 + gap / 4.0)
        cy = y + sign * ny * (extent / 2.0 + gap / 4.0)
        clip = (Pos(cx, cy, UM_V1L_REAR_FACE_Z_MM)
                * Rot(Z=angle_n)
                * Box(extent - gap / 2.0, width, 30.0))
        return full & clip

    return half(1.0), half(-1.0)


def split_grommet_parts(routing_profile: str = "proud"):
    """Profile-fitted two-piece TPU inserts for the R6P routes.

    Proud uses a short curved shank that follows the final R14 bore and
    seats its flange on z=0.  V1L follows its keyed spatial arc and seats on the
    physical z=6.8 rear face without entering the Faston motion volume.
    Obi-Wan deliberately has no printed grommet. The nominal split bores are
    D7 (proud) and D7.1 (V1L).
    """
    if routing_profile == "proud":
        full = _proud_curved_grommet()
        half_a, half_b = _split_about_route_plane(full, routing_profile)
        return {"um_grommet_half_a": half_a,
                "um_grommet_half_b": half_b}
    if routing_profile == "v1l":
        full = _v1l_terminal_curved_grommet()
        half_a, half_b = _split_v1l_terminal_grommet(full)
        return {"um_grommet_half_a": half_a,
                "um_grommet_half_b": half_b}
    raise ValueError(routing_profile)


def gen_step():
    carrier = terminal_carrier_proxy()
    carrier.label = "MU10_terminal_carrier_PROXY_measure_hardware"
    env = removal_envelope()
    env.label = "V1L_legacy_pair_outboard_removal_ENVELOPE_32x40x10"
    body = mu10_body_keepout(include_flange=True)
    body.label = (
        "REFERENCE_MU10_D98_D80_D60_BODY_TERMINALS_OMITTED_"
        "PHYSICAL_CHECK_REQUIRED")
    lm_body = w22_body_keepout(include_flange=True)
    lm_body.label = (
        "REFERENCE_W22_CONSERVATIVE_STEPPED_REAR_BODY_"
        "SERVICE_LOOP_KEEP_CLEAR")
    children = [body, lm_body, carrier, env]
    for label, part in faston_proxy_parts().items():
        part.label = label
        children.append(part)
    for label, part in faston_boot_proxy_parts().items():
        part.label = label
        children.append(part)
    for label, part in faston_pull_sweep_parts().items():
        part.label = f"{label}_12mm_SERVICE_ENVELOPE"
        children.append(part)
    for label, part in obiwan_terminal_harness_parts().items():
        part.label = f"{label}_PHYSICAL_CABLE_PROXY"
        children.append(part)
    for label, part in obiwan_y_breakout_boot_parts().items():
        part.label = f"{label}_PHYSICAL_HEATSHRINK_PROXY"
        children.append(part)
    for profile in ("proud", "v1l"):
        cable = rear_cable_envelope(profile)
        role = ("complete_INTENTIONAL_FASTON_HANDOFF_OVERLAP"
                if profile == "v1l" else "rear")
        cable.label = f"{profile}_{role}_D7_cable_ENVELOPE"
        children.append(cable)
        for label, part in split_grommet_parts(profile).items():
            part.label = f"{profile}_{label}_TPU_PRINT_PART"
            children.append(part)
    assembly = Compound(children=children)
    lo, hi = UM_TERMINAL_GAP_DEG
    state = "floor_stand" if STAND_FOOT else "no_floor_stand"
    assembly.label = (
        f"MU10_terminals_clock_{UM_TERMINAL_CLOCK_DEG:g}_deg_"
        f"between_{lo:g}_{hi:g}_{state}_PHYSICAL_CHECK_REQUIRED"
    )
    return assembly
