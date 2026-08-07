"""Stock/Slim vase family for two opposed Tectonic TEBM35C10-4 BMRs.

The lower vase, UM opening, UM insert pattern, seam-B keyed interface, and
proud cable routes remain on their released datums.  Above the lower vase the
Dayton crescent is replaced by two measured, overlapping D66 lands:
the lower BMR mounts on the acoustic front and the upper BMR mounts on the
rear.  A quintic Bezier rear surface grows the T zone smoothly from the
released 18.3-mm plate to the published 25.1-mm driver depth.  Both driver
pockets retain a thin blind wall opposite their mounting face.  The shared T
duct follows a tangent/arc/tangent clearance path around the UM opening.  A
separate D4.6 branch leaves it at the existing G1 arc junction, initially
sharing the complete 3D tangent, then wraps smoothly around the lower pocket
while rising toward the front-biased outlet behind the upper/rear driver.

Coordinate frame
----------------
X/Y use the released top-baffle drawing datum.  Z=0 is the released rear
plane and Z=18.3 is the acoustic front.  The grown T-zone rear plane is
Z=-6.8.  Stock retains the released rear plane Z=0 below the smooth growth;
Slim retains its released rear plane Z=6.8 there.  Both therefore reach the
same 25.1-mm local BMR depth without changing the acoustic/front plane or the
seam-B interface.  The print transform is deliberately not applied here; the
exporter owns the common X180 front-face-down contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from build123d import (
    Bezier,
    Box,
    Circle,
    Cylinder,
    Line,
    Plane,
    Pos,
    Spline,
    ThreePointArc,
    Wire,
    extrude,
    make_face,
    sweep,
)

from ..base import (
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_D_MM,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_PCD_MM,
    outline_face,
)
from ..cables import (
    TS_ROUTE_CAPTIVE,
    cable_cutters,
    seam_relief_cutters,
    ts_cutter_from_path,
    ts_section,
)
from ..magnets import CaptiveMagnetTools, apply_wall_cavity
from .top_baffle_nd25fw4_b import STANDARD_MAGNET_Z_MM
from .top_baffle_nd25fw4_b2 import OUTLINE_B2
from .top_baffle_nd25fw4_b2_split import (
    DOVETAILS_B,
    SEAM_B_M3_AXIS_X_MM,
    SEAM_B_M3_AXIS_Z_MM,
    SEAM_B_M3_INSERT_BORE_D_MM,
    SEAM_B_M3_INSERT_DEPTH_MM,
    SEAM_B_M3_INSERT_ENGAGEMENT_MM,
    SEAM_B_M3_INSERT_TIP_MARGIN_MM,
    SEAM_B_M3_RECOMMENDED_SCREW_LENGTH_MM,
    SEAM_B_M3_VASE_FACE_Y_MM,
    SEAM_B_Y,
    _below_region,
    _grown,
    _prism,
    seam_b_m3_vase_insert_cutter,
)


PRINT_ORIENTATION = "front-face-down"
PART_NAME = "vase_TEBM35C10-4"


@dataclass(frozen=True)
class VaseTEBMProfile:
    """One released lower-envelope using the regular seam-B dovetail."""

    key: str
    release_variant: str
    rear_surface_z_mm: float
    rear_ramp_start_y_mm: float

    @property
    def section_depth_mm(self) -> float:
        return THICKNESS_MM - self.rear_surface_z_mm

    @property
    def local_rear_growth_mm(self) -> float:
        return self.rear_surface_z_mm - REAR_T_MOUNT_Z_MM


STOCK_PROFILE = VaseTEBMProfile(
    key="stock",
    release_variant="Stock-TEBM35C10-4",
    rear_surface_z_mm=0.0,
    rear_ramp_start_y_mm=391.709,
)
SLIM_PROFILE = VaseTEBMProfile(
    key="slim",
    release_variant="Slim-TEBM35C10-4",
    rear_surface_z_mm=6.8,
    # Start on seam B with zero slope/curvature.  This preserves the exact
    # Slim interface plane while giving the descending shared T duct enough
    # rear cover before the common full-depth BMR zone.
    rear_ramp_start_y_mm=SEAM_B_Y,
)
VASE_TEBM_PROFILES = {
    STOCK_PROFILE.key: STOCK_PROFILE,
    SLIM_PROFILE.key: SLIM_PROFILE,
}


def vase_profile(
    profile: str | VaseTEBMProfile = "stock",
) -> VaseTEBMProfile:
    if isinstance(profile, VaseTEBMProfile):
        return profile
    try:
        return VASE_TEBM_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown TEBM vase profile {profile!r}") from exc

# Published Tectonic / distributor interface envelope, in millimetres.
TEBM_NOMINAL_D_MM = 52.0
TEBM_MAX_D_MM = 54.0
TEBM_BASKET_D_MM = 43.6
TEBM_DEPTH_MM = 25.1
TEBM_MASS_G = 51.3
TEBM_CUTOUT_D_MM = 1.69 * 25.4
TEBM_MOUNT_PCD_MM = 1.90 * 25.4
TEBM_MOUNT_HOLE_COUNT = 4

# Layout contract.  The lower/front flange keeps the released 2.10-mm face
# gap to the D97.5 UM flange.  Because each basket crosses the opposite face,
# the second pitch is governed by D54 flange to D43.6 body, plus 0.50 mm.
UM_FLANGE_D_MM = 97.5
PRESERVED_UM_TO_T_FACE_GAP_MM = 2.10
BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM = 0.50
LOWER_T_AXIS_Y_MM = (
    float(UM_CUTOUT[1])
    + UM_FLANGE_D_MM / 2.0
    + TEBM_MAX_D_MM / 2.0
    + PRESERVED_UM_TO_T_FACE_GAP_MM
)
PAIR_AXIS_PITCH_MM = (
    TEBM_MAX_D_MM / 2.0
    + TEBM_BASKET_D_MM / 2.0
    + BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM
)
UPPER_T_AXIS_Y_MM = LOWER_T_AXIS_Y_MM + PAIR_AXIS_PITCH_MM
T_AXIS_Y_MM = (LOWER_T_AXIS_Y_MM, UPPER_T_AXIS_Y_MM)

# D66 supplies the measured driver land and M2 insert ligaments.  A 0.10-mm
# straight-face margin is added beyond the captive helper's exact 6.40-mm
# qualified land, so each side magnet has a real planar interface rather
# than merely touching a circular silhouette at the land corners.
TEBM_LAND_D_MM = 66.0
TEBM_LAND_R_MM = TEBM_LAND_D_MM / 2.0
T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM = 3.20
T_MAGNET_FLAT_EDGE_MARGIN_MM = 0.10
T_MAGNET_FLAT_HALF_HEIGHT_MM = (
    T_MAGNET_REQUIRED_FLAT_HALF_HEIGHT_MM + T_MAGNET_FLAT_EDGE_MARGIN_MM
)
T_MAGNET_FACE_X_MM = math.sqrt(
    TEBM_LAND_R_MM ** 2 - T_MAGNET_FLAT_HALF_HEIGHT_MM ** 2
)
T_MAGNET_TOTAL = 4

# Requested M2 x 4 x D3.2 heat-set insert bores.  Opposite 45-degree clocks
# keep the two published four-hole patterns visually paired while separating
# every insert from the side-magnet chimney.
M2_INSERT_BORE_D_MM = 3.2
M2_INSERT_DEPTH_MM = 4.0
LOWER_T_MOUNT_CLOCK_DEG = 45.0
UPPER_T_MOUNT_CLOCK_DEG = -45.0

# The rear surface starts at the exact released B2 flare crest and reaches
# full growth at the lower opening's tangent.  Bezier Z control values
# [0,0,0,-g,-g,-g] are the exact degree-five smootherstep, giving zero first
# and second derivatives at both ends (C2 joins, no corner angle).
REAR_GROWTH_MM = TEBM_DEPTH_MM - THICKNESS_MM
REAR_T_MOUNT_Z_MM = -REAR_GROWTH_MM
REAR_RAMP_START_Y_MM = STOCK_PROFILE.rear_ramp_start_y_mm
REAR_RAMP_END_Y_MM = LOWER_T_AXIS_Y_MM - TEBM_CUTOUT_D_MM / 2.0
REAR_RAMP_LENGTH_MM = REAR_RAMP_END_Y_MM - REAR_RAMP_START_Y_MM
PRESERVED_LOWER_END_Y_MM = 421.0

# Each opposed driver pocket is blind at the face opposite its mounting
# flange.  Keeping the 1.2-mm wall inside the existing 25.1-mm envelope
# leaves 23.9 mm from the mounting face to the wall, accounting for the
# published driver's face/flange layer without changing the print datum.
T_BLIND_BACK_WALL_THICKNESS_MM = 1.20
T_CLEAR_POCKET_DEPTH_MM = TEBM_DEPTH_MM - T_BLIND_BACK_WALL_THICKNESS_MM
LOWER_T_POCKET_REAR_Z_MM = (
    REAR_T_MOUNT_Z_MM + T_BLIND_BACK_WALL_THICKNESS_MM
)
UPPER_T_POCKET_FRONT_Z_MM = THICKNESS_MM - T_BLIND_BACK_WALL_THICKNESS_MM

# The vase-specific shared route retains the exact released seam-B crossing
# and lower-T mouth but no longer follows the outer wall.  It stays on the
# R46.3 UM-clearance arc through 114.5 degrees, then uses one opposing G1 arc
# through the narrow lower-T/UM neck and into the fixed outlet.  The selected
# exit is the latest 0.5-degree sampled station that passes the full-section
# exterior gate with reserve; the path keeps a 15-mm-class minimum radius.
MAIN_T_ROUTE_NAME = "vase_tebm_um_contained_3d_g1_y"
MAIN_T_ROUTE_PRE_SEAM_XY_MM = (-34.082604, 313.202641)
MAIN_T_ROUTE_SEAM_XY_MM = (-33.3077, 315.95)
MAIN_T_ROUTE_POST_SEAM_XY_MM = (-32.914491, 317.145618)
MAIN_T_LOWER_OUTLET_XY_MM = (-3.4, 433.0)
MAIN_T_UM_CLEARANCE_RADIUS_MM = 46.30
MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG = 114.5
MAIN_T_UM_NOMINAL_INNER_LIGAMENT_MM = (
    MAIN_T_UM_CLEARANCE_RADIUS_MM
    - float(UM_CUTOUT[2]) / 2.0
    - ts_section(float(UM_CUTOUT[1]))[0]
)
MAIN_T_UM_PILOT_VERTICAL_LIGAMENT_MM = (
    THICKNESS_MM - UM_PILOT_DEPTH_MM
    - sum(ts_section(float(UM_CUTOUT[1]))[1:])
)

# One D4.6 pair branch leaves the shared main at its existing G1 junction
# between the R46.3 UM-clearance arc and the lower-pocket exit arc.  Both
# outgoing Z laws are endpoint-flat at that station, so the branch and
# lower leg share the main's full 3D tangent rather than merely meeting at one
# point.  A 25.5-mm guide runs 0.2 mm outside the qualified R25.3 keepout;
# fitting one 3D spline through that guide removes the former line/arc
# curvature jumps while retaining manufacturing reserve after interpolation.
UPPER_T_BRANCH_D_MM = 4.60
UPPER_T_BRANCH_SPLIT_Y_MM = (
    float(UM_CUTOUT[1])
    + MAIN_T_UM_CLEARANCE_RADIUS_MM
    * math.sin(math.radians(MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG))
)
# Z=2.5 is the lowest 0.5-mm search candidate whose complete expanded section
# stays within the analytic body envelope.  It minimizes the short lower
# leg's vertical bend while retaining the mandatory 0.8-mm guarded skin.
UPPER_T_BRANCH_SPLIT_Z_MM = 2.50
UPPER_T_BRANCH_CLEARANCE_RADIUS_MM = 25.30
UPPER_T_BRANCH_GUIDE_RADIUS_MM = 25.50
UPPER_T_BRANCH_GUIDE_JOIN_ANGLE_DEG = -66.50
UPPER_T_BRANCH_LEAD_START_HANDLE_MM = 10.0
UPPER_T_BRANCH_LEAD_END_HANDLE_MM = 15.0
UPPER_T_BRANCH_GUIDE_SAMPLE_SPACING_MM = 1.50
UPPER_T_BRANCH_OUTLET_XY_MM = (0.0, 476.0)

# Cable exits are deliberately biased toward the face opposite each driver's
# acoustic face.  The lower/front-facing T exits 6.8 mm from the grown rear
# face at Z=0; the upper/rear-facing T exits 6.8 mm from the front at Z=11.5.
# Both legs use minimum-jerk quintics over path length, with zero slope and
# zero second derivative at the split and pocket endpoints.
T_CABLE_FACE_INSET_MM = (
    THICKNESS_MM - ts_section(MAIN_T_LOWER_OUTLET_XY_MM[1])[2]
)
LOWER_T_OUTLET_Z_MM = REAR_T_MOUNT_Z_MM + T_CABLE_FACE_INSET_MM
UPPER_T_BRANCH_OUTLET_Z_MM = THICKNESS_MM - T_CABLE_FACE_INSET_MM
MAIN_T_ROUTE_ENTRY_Z_MM = ts_section(MAIN_T_ROUTE_PRE_SEAM_XY_MM[1])[2]
UPPER_T_BRANCH_MIN_INNER_LIGAMENT_MM = (
    UPPER_T_BRANCH_CLEARANCE_RADIUS_MM
    - TEBM_CUTOUT_D_MM / 2.0
    - UPPER_T_BRANCH_D_MM / 2.0
)
UPPER_T_BRANCH_MIN_OUTER_LIGAMENT_MM = (
    TEBM_LAND_R_MM
    - UPPER_T_BRANCH_GUIDE_RADIUS_MM
    - UPPER_T_BRANCH_D_MM / 2.0
)
UPPER_T_BRANCH_LOWER_INSERT_VERTICAL_LIGAMENT_MM = (
    THICKNESS_MM - M2_INSERT_DEPTH_MM
    - (UPPER_T_BRANCH_SPLIT_Z_MM + UPPER_T_BRANCH_D_MM / 2.0)
)

# Manufacturing containment contract.  Expanding every active local duct by
# this amount and subtracting the final, unbored exterior envelope must leave
# no volume.  The sole exception is the explicitly bounded seam-B entry mouth;
# the two driver-pocket ends remain inside the virtual unbored envelope.
DUCT_EXTERIOR_SKIN_GUARD_MM = 0.80
DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3 = 1.0e-5
DUCT_APPROVED_SEAM_MOUTH_SIZE_XYZ_MM = (12.0, 8.0, 10.0)
DUCT_APPROVED_SEAM_MOUTH_CENTER_XYZ_MM = (
    MAIN_T_ROUTE_SEAM_XY_MM[0],
    SEAM_B_Y - 1.0,
    11.5,
)


@dataclass(frozen=True)
class VaseTEBMModel:
    """Authoritative solid plus the exact four captive-station records."""

    solid: object
    magnet_tools: tuple[CaptiveMagnetTools, ...]


def _vertical_cylinder(
    x: float, y: float, diameter: float, z_min: float, z_max: float,
):
    if z_max <= z_min:
        raise ValueError("cylinder z_max must exceed z_min")
    return Pos(x, y, (z_min + z_max) / 2.0) * Cylinder(
        diameter / 2.0, z_max - z_min
    )


def _circle_tangent_angle(
    point_xy: tuple[float, float],
    center_xy: tuple[float, float],
    radius: float,
    offset_sign: float,
) -> float:
    """Angle of one tangent point from an external plan point."""
    dx = point_xy[0] - center_xy[0]
    dy = point_xy[1] - center_xy[1]
    distance = math.hypot(dx, dy)
    if distance <= radius:
        raise ValueError("tangent source must lie outside the clearance circle")
    base = math.atan2(dy, dx)
    offset = math.acos(radius / distance)
    return base + offset_sign * offset


def _circle_point(
    center_xy: tuple[float, float], radius: float, angle: float,
) -> tuple[float, float, float]:
    return (
        center_xy[0] + radius * math.cos(angle),
        center_xy[1] + radius * math.sin(angle),
        0.0,
    )


def _main_t_tangent_geometry() -> tuple[
    float, float, tuple[float, float, float], tuple[float, float, float],
]:
    center = (0.0, float(UM_CUTOUT[1]))
    start_angle = _circle_tangent_angle(
        MAIN_T_ROUTE_POST_SEAM_XY_MM,
        center,
        MAIN_T_UM_CLEARANCE_RADIUS_MM,
        -1.0,
    )
    end_angle = math.radians(MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG)
    return (
        start_angle,
        end_angle,
        _circle_point(center, MAIN_T_UM_CLEARANCE_RADIUS_MM, start_angle),
        _circle_point(center, MAIN_T_UM_CLEARANCE_RADIUS_MM, end_angle),
    )


def _main_t_neck_exit_geometry() -> tuple[
    tuple[float, float, float], float, float,
]:
    """Return midpoint, radius, and turn for the contained G1 exit arc."""
    _start_angle, exit_angle, _tangent_start, neck_start = (
        _main_t_tangent_geometry())
    heading = math.atan2(-math.cos(exit_angle), math.sin(exit_angle))
    left_normal = (-math.sin(heading), math.cos(heading))
    outlet = (*MAIN_T_LOWER_OUTLET_XY_MM, 0.0)
    delta = (outlet[0] - neck_start[0], outlet[1] - neck_start[1])
    denominator = 2.0 * (
        delta[0] * left_normal[0] + delta[1] * left_normal[1])
    if denominator <= 0.0:
        raise ValueError("neck exit arc cannot reach the lower-T outlet")
    second_radius = (
        delta[0] * delta[0] + delta[1] * delta[1]) / denominator
    second_center = (
        neck_start[0] + second_radius * left_normal[0],
        neck_start[1] + second_radius * left_normal[1],
    )
    second_start_angle = math.atan2(
        neck_start[1] - second_center[1],
        neck_start[0] - second_center[0],
    )
    second_end_angle = math.atan2(
        outlet[1] - second_center[1], outlet[0] - second_center[0])
    second_turn = (
        second_end_angle - second_start_angle) % (2.0 * math.pi)
    if not 0.0 < second_turn < math.pi:
        raise ValueError("neck exit arc must be the minor counter-clockwise arc")
    second_mid_angle = second_start_angle + second_turn / 2.0
    second_mid = (
        second_center[0] + second_radius * math.cos(second_mid_angle),
        second_center[1] + second_radius * math.sin(second_mid_angle),
        0.0,
    )
    return second_mid, second_radius, second_turn


def optimized_main_t_path():
    """G1 contained route from released seam B to the lower T pocket."""
    start_angle, end_angle, tangent_start, tangent_end = (
        _main_t_tangent_geometry())
    pre = (*MAIN_T_ROUTE_PRE_SEAM_XY_MM, 0.0)
    seam = (*MAIN_T_ROUTE_SEAM_XY_MM, 0.0)
    post = (*MAIN_T_ROUTE_POST_SEAM_XY_MM, 0.0)
    initial_tangent = tuple(
        seam[index] - pre[index] for index in range(3))
    clockwise_circle_tangent = (
        math.sin(start_angle), -math.cos(start_angle), 0.0)
    lead = Spline(
        pre,
        seam,
        post,
        tangent_start,
        tangents=(initial_tangent, clockwise_circle_tangent),
        tangent_scalars=(1.0, 1.0),
    )
    normalized_start = start_angle % (2.0 * math.pi)
    middle_angle = (normalized_start + end_angle) / 2.0
    center = (0.0, float(UM_CUTOUT[1]))
    clearance_arc = ThreePointArc(
        tangent_start,
        _circle_point(center, MAIN_T_UM_CLEARANCE_RADIUS_MM, middle_angle),
        tangent_end,
    )
    second_mid, _second_radius, _second_turn = (
        _main_t_neck_exit_geometry())
    neck_exit_arc = ThreePointArc(
        tangent_end, second_mid, (*MAIN_T_LOWER_OUTLET_XY_MM, 0.0))
    return Wire((lead, clearance_arc, neck_exit_arc))


def _main_t_branch_split_xy() -> tuple[float, float]:
    """Exact shared-main G1 station where the upper route bifurcates."""
    point = _main_t_tangent_geometry()[3]
    if not math.isclose(
        float(point[1]), UPPER_T_BRANCH_SPLIT_Y_MM, abs_tol=1.0e-9,
    ):
        raise RuntimeError("upper-T split no longer matches the main G1 join")
    return (float(point[0]), float(point[1]))


def _main_t_branch_split_parameter(path=None) -> float:
    """Normalized main-path length at the analytic G1 split station."""
    if path is None:
        path = optimized_main_t_path()
    low = 0.0
    high = 1.0
    for _iteration in range(64):
        middle = (low + high) / 2.0
        if float((path @ middle).Y) < UPPER_T_BRANCH_SPLIT_Y_MM:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


def _upper_t_branch_guide_geometry() -> tuple[
    float, float, tuple[float, float, float], tuple[float, float, float],
]:
    """Return the clearance-guide arc and its two tangent stations."""
    center = (0.0, LOWER_T_AXIS_Y_MM)
    start_angle = math.radians(UPPER_T_BRANCH_GUIDE_JOIN_ANGLE_DEG)
    end_angle = _circle_tangent_angle(
        UPPER_T_BRANCH_OUTLET_XY_MM,
        center,
        UPPER_T_BRANCH_GUIDE_RADIUS_MM,
        -1.0,
    )
    return (
        start_angle,
        end_angle,
        _circle_point(center, UPPER_T_BRANCH_GUIDE_RADIUS_MM, start_angle),
        _circle_point(center, UPPER_T_BRANCH_GUIDE_RADIUS_MM, end_angle),
    )


def _upper_t_branch_start_tangent() -> tuple[float, float, float]:
    """Unit tangent shared by both legs of the Y at the split."""
    angle = math.radians(MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG)
    return (math.sin(angle), -math.cos(angle), 0.0)


def upper_t_branch_plan_guide():
    """G1 plan guide used to seed the clearance-preserving 3D spline."""
    start_angle, end_angle, tangent_start, tangent_end = (
        _upper_t_branch_guide_geometry())
    split = (*_main_t_branch_split_xy(), 0.0)
    start_tangent = _upper_t_branch_start_tangent()
    circle_tangent = (
        -math.sin(start_angle), math.cos(start_angle), 0.0)
    first_control = tuple(
        split[index]
        + UPPER_T_BRANCH_LEAD_START_HANDLE_MM * start_tangent[index]
        for index in range(3)
    )
    second_control = tuple(
        tangent_start[index]
        - UPPER_T_BRANCH_LEAD_END_HANDLE_MM * circle_tangent[index]
        for index in range(3)
    )
    center = (0.0, LOWER_T_AXIS_Y_MM)
    middle_angle = (start_angle + end_angle) / 2.0
    return Wire((
        Bezier(
            split, first_control, second_control, tangent_start).edge(),
        ThreePointArc(
            tangent_start,
            _circle_point(
                center, UPPER_T_BRANCH_GUIDE_RADIUS_MM, middle_angle),
            tangent_end,
        ),
        Line(tangent_end, (*UPPER_T_BRANCH_OUTLET_XY_MM, 0.0)),
    ))


def _minimum_jerk01(value: float) -> float:
    """C2 endpoint-flat quintic used for the opposed-face Z transition."""
    value = min(1.0, max(0.0, float(value)))
    return value ** 3 * (10.0 + value * (-15.0 + 6.0 * value))


def _main_t_center_z_mm(
    path_fraction: float, split_fraction: float,
) -> float:
    """C2, endpoint-flat Z law for the shared main and lower T leg."""
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("main T split fraction must lie inside the path")
    path_fraction = min(1.0, max(0.0, float(path_fraction)))
    if path_fraction <= split_fraction:
        smooth = _minimum_jerk01(path_fraction / split_fraction)
        return (
            MAIN_T_ROUTE_ENTRY_Z_MM
            + (UPPER_T_BRANCH_SPLIT_Z_MM - MAIN_T_ROUTE_ENTRY_Z_MM)
            * smooth
        )
    smooth = _minimum_jerk01(
        (path_fraction - split_fraction) / (1.0 - split_fraction))
    return (
        UPPER_T_BRANCH_SPLIT_Z_MM
        + (LOWER_T_OUTLET_Z_MM - UPPER_T_BRANCH_SPLIT_Z_MM) * smooth
    )


def upper_t_branch_path():
    """Single tangent-constrained 3D spline into the upper/rear pocket."""
    guide = upper_t_branch_plan_guide()
    count = max(
        24,
        int(math.ceil(
            float(guide.length)
            / UPPER_T_BRANCH_GUIDE_SAMPLE_SPACING_MM)),
    )
    points = []
    for index in range(count + 1):
        u = index / count
        point = guide @ u
        smooth = _minimum_jerk01(u)
        z = (
            UPPER_T_BRANCH_SPLIT_Z_MM
            + (UPPER_T_BRANCH_OUTLET_Z_MM
               - UPPER_T_BRANCH_SPLIT_Z_MM) * smooth
        )
        points.append((float(point.X), float(point.Y), z))

    _start_angle, _end_angle, _tangent_start, tangent_end = (
        _upper_t_branch_guide_geometry())
    outlet = (*UPPER_T_BRANCH_OUTLET_XY_MM, 0.0)
    end_tangent = tuple(
        outlet[index] - tangent_end[index] for index in range(3))
    return Spline(
        *points,
        tangents=(_upper_t_branch_start_tangent(), end_tangent),
        tangent_scalars=(1.0, 1.0),
    )


def _sample_path(path, spacing_mm: float = 0.5):
    count = max(16, int(math.ceil(float(path.length) / spacing_mm)))
    return [tuple(path @ (index / count)) for index in range(count + 1)]


def optimized_main_t_centerline_points(spacing_mm: float = 0.5):
    """Manufacturing-space samples of the endpoint-flat 3D main route."""
    path = optimized_main_t_path()
    split_fraction = _main_t_branch_split_parameter(path)
    count = max(16, int(math.ceil(float(path.length) / spacing_mm)))
    points = []
    for index in range(count + 1):
        path_fraction = index / count
        point = path @ path_fraction
        points.append((
            float(point.X),
            float(point.Y),
            _main_t_center_z_mm(path_fraction, split_fraction),
        ))
    return points


def upper_t_branch_centerline_points(spacing_mm: float = 0.5):
    return [tuple(map(float, point))
            for point in _sample_path(upper_t_branch_path(), spacing_mm)]


def _accumulated_plan_turn_deg(path, spacing_mm: float = 0.25) -> float:
    count = max(32, int(math.ceil(float(path.length) / spacing_mm)))
    headings = []
    for index in range(count + 1):
        tangent = path % (index / count)
        headings.append(math.atan2(float(tangent.Y), float(tangent.X)))
    turn = 0.0
    for first, second in zip(headings, headings[1:]):
        delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
        turn += abs(delta)
    return math.degrees(turn)


def _minimum_plan_bend_radius_mm(path, spacing_mm: float = 0.10) -> float:
    points = _sample_path(path, spacing_mm)
    radii = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        a = math.dist(middle[:2], last[:2])
        b = math.dist(first[:2], last[:2])
        c = math.dist(first[:2], middle[:2])
        twice_area = abs(
            (middle[0] - first[0]) * (last[1] - first[1])
            - (middle[1] - first[1]) * (last[0] - first[0]))
        if twice_area > 1.0e-9:
            radii.append(a * b * c / (2.0 * twice_area))
    if not radii:
        return math.inf
    return min(radii)


def _accumulated_spatial_turn_deg(
    path, spacing_mm: float = 0.25,
) -> float:
    count = max(32, int(math.ceil(float(path.length) / spacing_mm)))
    tangents = []
    for index in range(count + 1):
        tangent = path % (index / count)
        vector = tuple(float(value) for value in tangent)
        magnitude = math.sqrt(sum(value * value for value in vector))
        tangents.append(tuple(value / magnitude for value in vector))
    return math.degrees(sum(
        math.acos(max(-1.0, min(1.0, sum(
            first[axis] * second[axis] for axis in range(3)
        ))))
        for first, second in zip(tangents, tangents[1:])
    ))


def _minimum_spatial_bend_radius_mm(
    path, spacing_mm: float = 0.10,
) -> float:
    points = _sample_path(path, spacing_mm)
    radii = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        side_a = math.dist(middle, last)
        side_b = math.dist(first, last)
        side_c = math.dist(first, middle)
        first_leg = tuple(
            middle[axis] - first[axis] for axis in range(3))
        second_leg = tuple(
            last[axis] - first[axis] for axis in range(3))
        cross = (
            first_leg[1] * second_leg[2]
            - first_leg[2] * second_leg[1],
            first_leg[2] * second_leg[0]
            - first_leg[0] * second_leg[2],
            first_leg[0] * second_leg[1]
            - first_leg[1] * second_leg[0],
        )
        twice_area = math.sqrt(sum(value * value for value in cross))
        if twice_area > 1.0e-9:
            radii.append(
                side_a * side_b * side_c / (2.0 * twice_area))
    return min(radii) if radii else math.inf


def _polyline_length_mm(points) -> float:
    """Length of an already sampled manufacturing-space centerline."""
    return sum(
        math.dist(first, second)
        for first, second in zip(points, points[1:])
    )


def _accumulated_spatial_turn_from_points_deg(points) -> float:
    """Accumulated 3D heading change of sampled centerline chords."""
    tangents = []
    for first, second in zip(points, points[1:]):
        vector = tuple(
            second[axis] - first[axis] for axis in range(3))
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude > 1.0e-12:
            tangents.append(tuple(value / magnitude for value in vector))
    return math.degrees(sum(
        math.acos(max(-1.0, min(1.0, sum(
            first[axis] * second[axis] for axis in range(3)
        ))))
        for first, second in zip(tangents, tangents[1:])
    ))


def _minimum_spatial_bend_radius_from_points_mm(points) -> float:
    """Minimum circumradius across sampled 3D centerline triplets."""
    radii = []
    for first, middle, last in zip(points, points[1:], points[2:]):
        side_a = math.dist(middle, last)
        side_b = math.dist(first, last)
        side_c = math.dist(first, middle)
        first_leg = tuple(
            middle[axis] - first[axis] for axis in range(3))
        second_leg = tuple(
            last[axis] - first[axis] for axis in range(3))
        cross = (
            first_leg[1] * second_leg[2]
            - first_leg[2] * second_leg[1],
            first_leg[2] * second_leg[0]
            - first_leg[0] * second_leg[2],
            first_leg[0] * second_leg[1]
            - first_leg[1] * second_leg[0],
        )
        twice_area = math.sqrt(sum(value * value for value in cross))
        if twice_area > 1.0e-9:
            radii.append(
                side_a * side_b * side_c / (2.0 * twice_area))
    return min(radii) if radii else math.inf


def upper_t_branch_split_tangent_error_deg() -> float:
    """Angular error between the shared-main and branch 3D tangents."""
    expected = _upper_t_branch_start_tangent()
    actual_vector = upper_t_branch_path() % 0.0
    actual = tuple(float(value) for value in actual_vector)
    magnitude = math.sqrt(sum(value * value for value in actual))
    dot = sum(expected[index] * actual[index] / magnitude
              for index in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def _plan_volume(z_min_mm: float, z_max_mm: float):
    """Extrude the exact B2/BMR plan over one explicit source-Z interval."""
    z_min = float(z_min_mm)
    z_max = float(z_max_mm)
    height = z_max - z_min
    if height <= 0.0:
        raise ValueError("TEBM vase plan volume requires positive height")
    released = extrude(
        Plane.XY.offset(z_min) * outline_face(OUTLINE_B2), amount=height)
    lower_clip_min_y = SEAM_B_Y - 20.0
    lower_clip = Pos(
        0.0,
        (lower_clip_min_y + PRESERVED_LOWER_END_Y_MM) / 2.0,
        (z_min + z_max) / 2.0,
    ) * Box(
        400.0,
        PRESERVED_LOWER_END_Y_MM - lower_clip_min_y,
        height + 2.0,
    )
    lower = released & lower_clip

    lands = None
    for axis_y in T_AXIS_Y_MM:
        land = Pos(0.0, axis_y, (z_min + z_max) / 2.0) * Cylinder(
            TEBM_LAND_R_MM, height
        )
        lands = land if lands is None else lands + land
    flat_clip = Pos(
        0.0,
        sum(T_AXIS_Y_MM) / 2.0,
        (z_min + z_max) / 2.0,
    ) * Box(
        2.0 * T_MAGNET_FACE_X_MM,
        200.0,
        height + 2.0,
    )
    return lower + (lands & flat_clip)


def _slab_without_functional_bores(
    profile: str | VaseTEBMProfile = "stock",
):
    """Build one released lower slab plus the two clipped circular lands."""
    spec = vase_profile(profile)
    return _plan_volume(spec.rear_surface_z_mm, THICKNESS_MM)


def _rear_growth_wedge(
    profile: str | VaseTEBMProfile = "stock",
):
    """Return the exact C2 rear-growth volume before plan intersection."""
    spec = vase_profile(profile)
    x_min = -100.0
    y0 = spec.rear_ramp_start_y_mm
    y1 = REAR_RAMP_END_Y_MM
    y_top = UPPER_T_AXIS_Y_MM + TEBM_LAND_R_MM + 14.0
    rear_start = spec.rear_surface_z_mm
    rear_end = REAR_T_MOUNT_Z_MM
    run = y1 - y0
    controls = (
        (x_min, y0, rear_start),
        (x_min, y0 + run / 5.0, rear_start),
        (x_min, y0 + 2.0 * run / 5.0, rear_start),
        (x_min, y0 + 3.0 * run / 5.0, rear_end),
        (x_min, y0 + 4.0 * run / 5.0, rear_end),
        (x_min, y1, rear_end),
    )
    section = make_face(Wire((
        Bezier(*controls).edge(),
        Line(controls[-1], (x_min, y_top, rear_end)).edge(),
        Line((x_min, y_top, rear_end), (x_min, y_top, rear_start)).edge(),
        Line((x_min, y_top, rear_start), controls[0]).edge(),
    )))
    return extrude(section, amount=200.0, dir=(1.0, 0.0, 0.0))


def external_envelope(
    profile: str | VaseTEBMProfile = "stock",
):
    """Final unbored exterior, including growth and seam-B pockets."""
    spec = vase_profile(profile)
    slab = _slab_without_functional_bores(spec)
    rear_growth = _rear_growth_wedge(spec) & _plan_volume(
        REAR_T_MOUNT_Z_MM, spec.rear_surface_z_mm)
    envelope = slab + rear_growth
    envelope -= _prism(_grown(_below_region(SEAM_B_Y, DOVETAILS_B)))
    return envelope


def _pilot_centers(
    center_y: float, pcd: float, angles_deg: tuple[float, ...],
) -> tuple[tuple[float, float], ...]:
    radius = pcd / 2.0
    return tuple((
        radius * math.cos(math.radians(angle)),
        center_y + radius * math.sin(math.radians(angle)),
    ) for angle in angles_deg)


def _apply_driver_interfaces(part):
    # UM stays exactly as released: D82 opening and four front-blind M3 insert
    # bores on D89.5.  The lower BMR pocket opens only from the front; the
    # upper BMR pocket opens only from the rear.  Each retains a 1.2-mm blind
    # wall at the opposite face inside the established 25.1-mm envelope.
    z_min = REAR_T_MOUNT_Z_MM - 1.0
    z_max = THICKNESS_MM + 1.0
    part -= _vertical_cylinder(
        0.0, float(UM_CUTOUT[1]), float(UM_CUTOUT[2]), z_min, z_max)
    part -= _vertical_cylinder(
        0.0, LOWER_T_AXIS_Y_MM, TEBM_CUTOUT_D_MM,
        LOWER_T_POCKET_REAR_Z_MM, z_max)
    part -= _vertical_cylinder(
        0.0, UPPER_T_AXIS_Y_MM, TEBM_CUTOUT_D_MM,
        z_min, UPPER_T_POCKET_FRONT_Z_MM)

    for x, y in _pilot_centers(
        float(UM_CUTOUT[1]), UM_PILOT_PCD_MM,
        tuple(float(value) for value in UM_PILOT_ANGLES_DEG),
    ):
        part -= _vertical_cylinder(
            x, y, UM_PILOT_D_MM,
            THICKNESS_MM - UM_PILOT_DEPTH_MM, THICKNESS_MM)

    patterns = (
        (LOWER_T_AXIS_Y_MM, LOWER_T_MOUNT_CLOCK_DEG,
         THICKNESS_MM - M2_INSERT_DEPTH_MM, THICKNESS_MM),
        (UPPER_T_AXIS_Y_MM, UPPER_T_MOUNT_CLOCK_DEG,
         REAR_T_MOUNT_Z_MM,
         REAR_T_MOUNT_Z_MM + M2_INSERT_DEPTH_MM),
    )
    for axis_y, clock, bore_z_min, bore_z_max in patterns:
        angles = tuple(clock + 90.0 * index
                       for index in range(TEBM_MOUNT_HOLE_COUNT))
        for x, y in _pilot_centers(axis_y, TEBM_MOUNT_PCD_MM, angles):
            part -= _vertical_cylinder(
                x, y, M2_INSERT_BORE_D_MM, bore_z_min, bore_z_max)
    return part


def _main_t_cable_duct(section_extra_mm: float = 0.0):
    """Oval/round shared route following its qualified 3D centerline."""
    if section_extra_mm < 0.0:
        raise ValueError("main T duct expansion must be non-negative")
    path = optimized_main_t_path()
    split_fraction = _main_t_branch_split_parameter(path)
    return ts_cutter_from_path(
        path,
        section_extra_mm=section_extra_mm,
        center_z_fn=lambda path_fraction, _x, _y, _default_z: (
            _main_t_center_z_mm(path_fraction, split_fraction)
        ),
        follow_3d_tangent=True,
    )


def _upper_t_cable_duct(radial_extra_mm: float = 0.0):
    """Circular one-pair branch from the shared main to upper/rear T."""
    if radial_extra_mm < 0.0:
        raise ValueError("upper-T duct expansion must be non-negative")
    path = upper_t_branch_path()
    section = Plane(origin=path @ 0.0, z_dir=path % 0.0) * Circle(
        UPPER_T_BRANCH_D_MM / 2.0 + radial_extra_mm)
    return sweep(section, path=path)


def _approved_seam_mouth_volume():
    """Only exterior exception allowed for the local T cable cutters."""
    return Pos(*DUCT_APPROVED_SEAM_MOUTH_CENTER_XYZ_MM) * Box(
        *DUCT_APPROVED_SEAM_MOUTH_SIZE_XYZ_MM)


def duct_exposure_residuals(
    envelope=None,
    skin_guard_mm: float = DUCT_EXTERIOR_SKIN_GUARD_MM,
    profile: str | VaseTEBMProfile = "stock",
) -> dict[str, object]:
    """BREP residuals outside the body after approved mouths are removed.

    A passing residual has effectively zero volume.  Expanding the cutter,
    rather than sampling its centerline, proves the full duct section retains
    ``skin_guard_mm`` of exterior cover.  Driver-pocket terminations require
    no mask because the unbored envelope virtually fills those approved exits.
    """
    if skin_guard_mm < 0.0:
        raise ValueError("duct exterior skin guard must be non-negative")
    if envelope is None:
        envelope = external_envelope(profile)
    main = _main_t_cable_duct(section_extra_mm=skin_guard_mm)
    branch = _upper_t_cable_duct(radial_extra_mm=skin_guard_mm)
    return {
        "shared_main_except_seam_b_entry": (
            (main - envelope) - _approved_seam_mouth_volume()),
        "upper_t_branch": branch - envelope,
    }


def duct_unapproved_opening_residuals(
    skin_guard_mm: float = DUCT_EXTERIOR_SKIN_GUARD_MM,
) -> dict[str, object]:
    """Guarded-duct intersections with every non-route exterior opening."""
    if skin_guard_mm < 0.0:
        raise ValueError("duct exterior skin guard must be non-negative")
    main = _main_t_cable_duct(section_extra_mm=skin_guard_mm)
    branch = _upper_t_cable_duct(radial_extra_mm=skin_guard_mm)
    z_min = REAR_T_MOUNT_Z_MM - 1.0
    z_max = THICKNESS_MM + 1.0
    um_opening = _vertical_cylinder(
        0.0, float(UM_CUTOUT[1]), float(UM_CUTOUT[2]), z_min, z_max)
    lower_pocket = _vertical_cylinder(
        0.0, LOWER_T_AXIS_Y_MM, TEBM_CUTOUT_D_MM,
        LOWER_T_POCKET_REAR_Z_MM, z_max)
    upper_pocket = _vertical_cylinder(
        0.0, UPPER_T_AXIS_Y_MM, TEBM_CUTOUT_D_MM,
        z_min, UPPER_T_POCKET_FRONT_Z_MM)

    insert_bores = None
    for x, y in _pilot_centers(
        float(UM_CUTOUT[1]), UM_PILOT_PCD_MM,
        tuple(float(value) for value in UM_PILOT_ANGLES_DEG),
    ):
        bore = _vertical_cylinder(
            x, y, UM_PILOT_D_MM,
            THICKNESS_MM - UM_PILOT_DEPTH_MM, THICKNESS_MM)
        insert_bores = bore if insert_bores is None else insert_bores + bore
    patterns = (
        (LOWER_T_AXIS_Y_MM, LOWER_T_MOUNT_CLOCK_DEG,
         THICKNESS_MM - M2_INSERT_DEPTH_MM, THICKNESS_MM),
        (UPPER_T_AXIS_Y_MM, UPPER_T_MOUNT_CLOCK_DEG,
         REAR_T_MOUNT_Z_MM,
         REAR_T_MOUNT_Z_MM + M2_INSERT_DEPTH_MM),
    )
    for axis_y, clock, bore_z_min, bore_z_max in patterns:
        angles = tuple(clock + 90.0 * index
                       for index in range(TEBM_MOUNT_HOLE_COUNT))
        for x, y in _pilot_centers(axis_y, TEBM_MOUNT_PCD_MM, angles):
            bore = _vertical_cylinder(
                x, y, M2_INSERT_BORE_D_MM, bore_z_min, bore_z_max)
            insert_bores = bore if insert_bores is None else insert_bores + bore

    insert_bores = insert_bores + seam_b_m3_vase_insert_cutter()

    return {
        "shared_main_to_um_opening": main & um_opening,
        "shared_main_to_upper_t_pocket": main & upper_pocket,
        "upper_branch_to_um_opening": branch & um_opening,
        "upper_branch_to_lower_t_pocket": branch & lower_pocket,
        "shared_main_to_insert_bores": main & insert_bores,
        "upper_branch_to_insert_bores": branch & insert_bores,
    }


def _validate_duct_exterior_containment(envelope) -> None:
    residuals = duct_exposure_residuals(envelope)
    residuals.update(duct_unapproved_opening_residuals())
    failures = {
        name: float(residual.volume)
        for name, residual in residuals.items()
        if float(residual.volume) > DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3
    }
    if failures:
        details = ", ".join(
            f"{name}={volume:.6f} mm^3"
            for name, volume in sorted(failures.items()))
        raise RuntimeError(
            f"{PART_NAME}: cable duct breaches exterior skin: {details}")


def _apply_cable_routes(part):
    # Keep the released UM and lower feeder/entry infrastructure.  The vase's
    # local shared T span then takes its own shortest-clearance route from the
    # exact seam-B handoff to the exact lower pocket outlet; its section still
    # comes from cables.ts_section.  The dedicated upper branch shares only
    # the upstream span and never passes through the lower driver pocket.
    for cutter in cable_cutters(
        route_names=("um", "t1f", "t2f"),
    ):
        part -= cutter
    for cutter in seam_relief_cutters(("ts",)):
        part -= cutter
    part -= _main_t_cable_duct()
    part -= _upper_t_cable_duct()
    return part


def _apply_t_magnets(part):
    records: list[CaptiveMagnetTools] = []
    for vertical, axis_y in (("lower", LOWER_T_AXIS_Y_MM),
                             ("upper", UPPER_T_AXIS_Y_MM)):
        for side, sign in (("left", -1.0), ("right", 1.0)):
            part, tools = apply_wall_cavity(
                part,
                name=f"tebm_{vertical}_{side}_base",
                face=(sign * T_MAGNET_FACE_X_MM,
                      axis_y, STANDARD_MAGNET_Z_MM),
                outward=(sign, 0.0, 0.0),
                owner="base",
                print_up=(0.0, 0.0, -1.0),
                bed_datum=(0.0, 0.0, THICKNESS_MM),
            )
            records.append(tools)
    return part, tuple(records)


def _validate_duct_magnet_separation(magnet_tools) -> None:
    guarded = {
        "shared_main": _main_t_cable_duct(
            section_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM),
        "upper_t_branch": _upper_t_cable_duct(
            radial_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM),
    }
    failures = {}
    for duct_name, duct in guarded.items():
        for tools in magnet_tools:
            volume = sum(float((duct & cutter).volume)
                         for cutter in tools.cutters)
            if volume > DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3:
                failures[f"{duct_name}_to_{tools.name}"] = volume
    if failures:
        details = ", ".join(
            f"{name}={volume:.6f} mm^3"
            for name, volume in sorted(failures.items()))
        raise RuntimeError(
            f"{PART_NAME}: cable duct reaches magnet opening: {details}")


def build_model(
    profile: str | VaseTEBMProfile = "stock",
) -> VaseTEBMModel:
    """Build and validate one monolithic keyed Stock/Slim replacement."""
    spec = vase_profile(profile)
    slab = _slab_without_functional_bores(spec)
    rear_growth = _rear_growth_wedge(spec) & _plan_volume(
        REAR_T_MOUNT_Z_MM, spec.rear_surface_z_mm)
    part = slab + rear_growth
    seam_cutter = _prism(_grown(_below_region(SEAM_B_Y, DOVETAILS_B)))
    _validate_duct_exterior_containment(part - seam_cutter)
    part = _apply_driver_interfaces(part)
    part = _apply_cable_routes(part)

    # Reuse the released top-side female pockets at seam B.  The two mids own
    # the matching full-through-thickness male dovetails, exactly as in B2/V1.
    part -= seam_cutter
    part -= seam_b_m3_vase_insert_cutter()
    part, magnet_tools = _apply_t_magnets(part)
    _validate_duct_magnet_separation(magnet_tools)
    part.label = f"{PART_NAME}_{spec.key}"

    solids = list(part.solids())
    if not part.is_valid or len(solids) != 1 or solids[0].volume <= 1.0:
        raise RuntimeError(
            f"{PART_NAME}_{spec.key}: expected one valid positive-volume solid")
    return VaseTEBMModel(part, magnet_tools)


def design_facts(
    profile: str | VaseTEBMProfile = "stock",
) -> dict[str, object]:
    """Stable, serializable manufacturing facts independent of tessellation."""
    spec = vase_profile(profile)
    main_path = optimized_main_t_path()
    branch_path = upper_t_branch_path()
    split_xy = _main_t_branch_split_xy()
    _second_mid, neck_exit_radius, _second_turn = (
        _main_t_neck_exit_geometry())
    main_points = optimized_main_t_centerline_points(0.25)
    main_minimum_um_radius = min(
        math.hypot(x, y - float(UM_CUTOUT[1]))
        for x, y, _z in main_points
        if y >= MAIN_T_ROUTE_SEAM_XY_MM[1]
    )
    branch_points = upper_t_branch_centerline_points(0.25)
    branch_minimum_lower_radius = min(
        math.hypot(x, y - LOWER_T_AXIS_Y_MM)
        for x, y, _z in branch_points
    )
    return {
        "part": PART_NAME,
        "profile": spec.key,
        "release_variant": spec.release_variant,
        "coordinate_system": {
            "units": "mm",
            "rear_plane_z_mm": spec.rear_surface_z_mm,
            "front_plane_z_mm": THICKNESS_MM,
            "grown_rear_plane_z_mm": REAR_T_MOUNT_Z_MM,
            "print_orientation": PRINT_ORIENTATION,
        },
        "released_interfaces": {
            "seam_b_y_mm": SEAM_B_Y,
            "seam_b_radial_m3": {
                "axis_xyz_mm": [
                    SEAM_B_M3_AXIS_X_MM,
                    SEAM_B_M3_VASE_FACE_Y_MM,
                    SEAM_B_M3_AXIS_Z_MM,
                ],
                "insert_bore_d_mm": SEAM_B_M3_INSERT_BORE_D_MM,
                "insert_depth_mm": SEAM_B_M3_INSERT_DEPTH_MM,
                "recommended_screw": (
                    f"M3x{SEAM_B_M3_RECOMMENDED_SCREW_LENGTH_MM:g} "
                    "socket-cap"),
                "insert_engagement_mm": SEAM_B_M3_INSERT_ENGAGEMENT_MM,
                "tip_margin_mm": SEAM_B_M3_INSERT_TIP_MARGIN_MM,
                "access": "LM cutout; hidden after W22 installation",
            },
            "um_axis_y_mm": float(UM_CUTOUT[1]),
            "um_opening_d_mm": float(UM_CUTOUT[2]),
            "um_insert_pcd_mm": UM_PILOT_PCD_MM,
            "um_insert_bore_d_mm": UM_PILOT_D_MM,
            "um_insert_depth_mm": UM_PILOT_DEPTH_MM,
            "cable_routes": ["um", "ts", "t1f", "t2f"],
            "upstream_ts_route": TS_ROUTE_CAPTIVE,
            "local_ts_route": MAIN_T_ROUTE_NAME,
        },
        "tebm35c10_4": {
            "quantity": 2,
            "nominal_d_mm": TEBM_NOMINAL_D_MM,
            "maximum_d_mm": TEBM_MAX_D_MM,
            "basket_d_mm": TEBM_BASKET_D_MM,
            "cutout_d_mm": TEBM_CUTOUT_D_MM,
            "depth_mm": TEBM_DEPTH_MM,
            "mass_g_each": TEBM_MASS_G,
            "axis_y_mm": list(T_AXIS_Y_MM),
            "axis_pitch_mm": PAIR_AXIS_PITCH_MM,
            "opening_web_mm": PAIR_AXIS_PITCH_MM - TEBM_CUTOUT_D_MM,
            "land_d_mm": TEBM_LAND_D_MM,
            "lower_mount_face": "front",
            "upper_mount_face": "rear",
        },
        "blind_back_walls": {
            "count": 2,
            "thickness_mm": T_BLIND_BACK_WALL_THICKNESS_MM,
            "clear_pocket_depth_mm": T_CLEAR_POCKET_DEPTH_MM,
            "lower_wall_z_range_mm": [
                REAR_T_MOUNT_Z_MM, LOWER_T_POCKET_REAR_Z_MM],
            "upper_wall_z_range_mm": [
                UPPER_T_POCKET_FRONT_Z_MM, THICKNESS_MM],
        },
        "t_cable_routing": {
            "shared_main": {
                "carries": ["lower_t_pair", "upper_t_pair"],
                "seam_b_crossing_xy_mm": list(MAIN_T_ROUTE_SEAM_XY_MM),
                "lower_t_outlet_xy_mm": list(MAIN_T_LOWER_OUTLET_XY_MM),
                "entry_xyz_mm": [
                    *MAIN_T_ROUTE_PRE_SEAM_XY_MM, MAIN_T_ROUTE_ENTRY_Z_MM],
                "split_xyz_mm": [
                    *split_xy, UPPER_T_BRANCH_SPLIT_Z_MM],
                "lower_t_outlet_xyz_mm": [
                    *MAIN_T_LOWER_OUTLET_XY_MM, LOWER_T_OUTLET_Z_MM],
                "plan_length_mm": float(main_path.length),
                "centerline_length_mm": _polyline_length_mm(main_points),
                "accumulated_plan_turn_deg": _accumulated_plan_turn_deg(
                    main_path),
                "accumulated_3d_turn_deg": (
                    _accumulated_spatial_turn_from_points_deg(main_points)),
                "minimum_plan_bend_radius_mm": (
                    _minimum_plan_bend_radius_mm(main_path)),
                "minimum_3d_bend_radius_mm": (
                    _minimum_spatial_bend_radius_from_points_mm(main_points)),
                "um_clearance_arc_radius_mm": (
                    MAIN_T_UM_CLEARANCE_RADIUS_MM),
                "um_clearance_arc_exit_angle_deg": (
                    MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG),
                "neck_exit_arc_radius_mm": neck_exit_radius,
                "minimum_center_radius_about_um_mm": (
                    main_minimum_um_radius),
                "minimum_um_plan_ligament_mm": (
                    main_minimum_um_radius
                    - float(UM_CUTOUT[2]) / 2.0
                    - ts_section(float(UM_CUTOUT[1]))[0]),
                "minimum_um_pilot_vertical_ligament_mm": (
                    MAIN_T_UM_PILOT_VERTICAL_LIGAMENT_MM),
                "lower_outlet_face_bias": "rear",
                "z_transition": (
                    "two endpoint-flat minimum-jerk quintics over "
                    "normalized path length; C2 at entry, split, and outlet"),
                "path_form": (
                    "G1 plan spline + UM-clearance arc + opposing exit arc; "
                    "normal-section ruled loft follows the 3D centerline"),
            },
            "upper_t_branch": {
                "carries": ["upper_t_pair"],
                "diameter_mm": UPPER_T_BRANCH_D_MM,
                "split_xyz_mm": [
                    *_main_t_branch_split_xy(),
                    UPPER_T_BRANCH_SPLIT_Z_MM,
                ],
                "lower_t_outlet_xyz_mm": [
                    *MAIN_T_LOWER_OUTLET_XY_MM, LOWER_T_OUTLET_Z_MM],
                "upper_t_outlet_xyz_mm": [
                    *UPPER_T_BRANCH_OUTLET_XY_MM,
                    UPPER_T_BRANCH_OUTLET_Z_MM,
                ],
                "opposed_mount_face_inset_mm": T_CABLE_FACE_INSET_MM,
                "upper_outlet_face_bias": "front",
                "centerline_length_mm": float(branch_path.length),
                "accumulated_plan_turn_deg": _accumulated_plan_turn_deg(
                    branch_path),
                "accumulated_3d_turn_deg": _accumulated_spatial_turn_deg(
                    branch_path),
                "minimum_plan_bend_radius_mm": (
                    _minimum_plan_bend_radius_mm(branch_path)),
                "minimum_3d_bend_radius_mm": (
                    _minimum_spatial_bend_radius_mm(branch_path)),
                "split_3d_tangent_error_deg": (
                    upper_t_branch_split_tangent_error_deg()),
                "minimum_required_lower_center_radius_mm": (
                    UPPER_T_BRANCH_CLEARANCE_RADIUS_MM),
                "clearance_guide_radius_mm": (
                    UPPER_T_BRANCH_GUIDE_RADIUS_MM),
                "actual_minimum_lower_center_radius_mm": (
                    branch_minimum_lower_radius),
                "minimum_inner_ligament_mm": (
                    branch_minimum_lower_radius
                    - TEBM_CUTOUT_D_MM / 2.0
                    - UPPER_T_BRANCH_D_MM / 2.0),
                "nominal_lower_land_outer_ligament_mm": (
                    UPPER_T_BRANCH_MIN_OUTER_LIGAMENT_MM),
                "minimum_lower_insert_vertical_ligament_mm": (
                    UPPER_T_BRANCH_LOWER_INSERT_VERTICAL_LIGAMENT_MM),
                "connects": ["shared_t_main", "upper_rear_t_pocket"],
                "z_transition": (
                    "minimum-jerk quintic over normalized path length"),
                "path_form": (
                    "single tangent-constrained 3D spline from a "
                    "G1 clearance guide"),
            },
            "exterior_containment": {
                "brep_gate": "expanded_cutter_minus_unbored_envelope",
                "forbidden_opening_gate": (
                    "expanded_cutter_intersection_with_unapproved_voids"),
                "magnet_gate": (
                    "expanded_cutter_intersection_with_magnet_tools"),
                "minimum_skin_guard_mm": DUCT_EXTERIOR_SKIN_GUARD_MM,
                "volume_tolerance_mm3": (
                    DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3),
                "approved_exterior_mouths": ["seam_b_ts_entry"],
                "driver_pocket_terminations": [
                    "lower_front_t_pocket", "upper_rear_t_pocket"],
            },
        },
        "rear_growth": {
            "growth_mm": spec.local_rear_growth_mm,
            "starts_at_released_rear_z_mm": spec.rear_surface_z_mm,
            "ends_at_bmr_rear_z_mm": REAR_T_MOUNT_Z_MM,
            "ramp_start_y_mm": spec.rear_ramp_start_y_mm,
            "ramp_end_y_mm": REAR_RAMP_END_Y_MM,
            "ramp_length_mm": (
                REAR_RAMP_END_Y_MM - spec.rear_ramp_start_y_mm),
            "continuity": "C2 degree-five smootherstep Bezier",
        },
        "m2_insert_bores": {
            "count": 2 * TEBM_MOUNT_HOLE_COUNT,
            "diameter_mm": M2_INSERT_BORE_D_MM,
            "depth_mm": M2_INSERT_DEPTH_MM,
            "pcd_mm": TEBM_MOUNT_PCD_MM,
            "lower_clock_deg": LOWER_T_MOUNT_CLOCK_DEG,
            "upper_clock_deg": UPPER_T_MOUNT_CLOCK_DEG,
        },
        "t_captive_magnets": {
            "count": T_MAGNET_TOTAL,
            "interface_face_x_mm": [-T_MAGNET_FACE_X_MM,
                                     T_MAGNET_FACE_X_MM],
            "axis_y_mm": list(T_AXIS_Y_MM),
            "axis_z_mm": STANDARD_MAGNET_Z_MM,
            "qualified_flat_height_mm": (
                2.0 * T_MAGNET_FLAT_HALF_HEIGHT_MM),
            "flat_edge_margin_mm": T_MAGNET_FLAT_EDGE_MARGIN_MM,
        },
    }


def gen_step(profile: str = "stock"):
    return build_model(profile).solid
