"""Minimal two-piece Obi-Wan LM/UM-to-Ae embedded-magnet coupon.

The source coordinate frame follows the released baffle convention: XY is
the installed front plane, local Z=0 is the Obi-Wan rear datum, and local Z=11.5
is the front face.  The print exports rotate the front face onto the bed.

Each magnet cavity has a curved printed cradle, printable axial retaining
skins, a full-width upper loading chimney, and a 45-degree closing roof.  The
cavity is closed in the final solid; it is accessible only while the print is
paused immediately before the roof begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

from build123d import (
    Align,
    Box,
    Compound,
    Cylinder,
    Face,
    Pos,
    Rot,
    Vector,
    Wire,
    extrude,
)


# Reuse the released Obi-Wan/Ae dimensional authorities instead of copying their
# radii, site heights, or receiver-root envelopes.  The physically tested
# coupon's zero interface gap is deliberately frozen locally below.
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gen_obiwan_wing_design_map import (  # noqa: E402
    PAD_LM_RADIAL_MM,
    PAD_LM_TANGENTIAL_MM,
    PAD_UM_RADIAL_MM,
    PAD_UM_TANGENTIAL_MM,
)
from top_baffle_nd25fw4 import THICKNESS_MM  # noqa: E402
from top_baffle_nd25fw4_obiwan import (  # noqa: E402
    CORE_REAR_Z,
    LM_CORE_R,
    LM_RECESS_R,
    SIDE_MAGNET_D,
    SIDE_MAGNET_POCKET_D,
    UM_CORE_R,
    UM_RECESS_R,
    side_magnet_sites,
)


DEPTH_MM = float(THICKNESS_MM - CORE_REAR_Z)
RELEASED_CARRIER_LIP_MM = float(LM_CORE_R - LM_RECESS_R)
assert math.isclose(
    RELEASED_CARRIER_LIP_MM,
    float(UM_CORE_R - UM_RECESS_R),
    abs_tol=1.0e-9,
)

# This physically tested process coupon predates the released production
# pair's 0.05-mm assembly clearance and intentionally has coincident plastic
# interface datums.  Keep this local and immutable: importing the production
# wing gap would silently turn the validated coupon's 0.90-mm magnet-face
# stack into the standard/Obi-Wan-LM-lower 0.95-mm stack when that separate
# authority moves. Obi-Wan ring pairs instead use the visible-surface datum
# and a 1.10-mm stack.
COUPON_INTERFACE_GAP_MM = 0.0

# The no-glue cavity removes the released pocket's 0.20-mm adhesive allowance.
# A D5 x 2 magnet therefore gets 0.10 mm of axial clearance, while the released
# D5.2 diametral clearance remains unchanged.
MAGNET_NOMINAL_DIAMETER_MM = float(SIDE_MAGNET_D)
CAVITY_DIAMETER_MM = float(SIDE_MAGNET_POCKET_D)
CAVITY_RADIAL_DEPTH_MM = 2.10

# These axial skins must survive a normal 0.4-mm-nozzle slice, not merely exist
# in the BREP.  A 0.30-mm first pass disappeared with Bambu Studio's Classic
# wall generator (0.42-mm outer / 0.45-mm inner line widths).  0.44 mm was the
# first tested value retained at both LM and UM; use 0.45 mm for one full line
# plus a small geometric margin.
FACE_SKIN_MM = 0.45
INNER_SKIN_MM = 0.45
CARRIER_MAGNET_LAND_MM = (
    FACE_SKIN_MM + CAVITY_RADIAL_DEPTH_MM + INNER_SKIN_MM
)
CARRIER_BACKING_ADD_MM = CARRIER_MAGNET_LAND_MM - RELEASED_CARRIER_LIP_MM

ROOF_ANGLE_DEG = 45.0
ROOF_HEIGHT_MM = (
    CAVITY_DIAMETER_MM / 2.0 / math.tan(math.radians(ROOF_ANGLE_DEG))
)
TARGET_LAYER_MM = 0.20
STATION_GAP_MM = 1.20
BOOLEAN_EPS_MM = 0.03


@dataclass(frozen=True)
class Station:
    key: str
    radius_mm: float
    tangential_mm: float
    wing_radial_mm: float
    y_mm: float
    axis_z_mm: float

    @property
    def raw_roof_start_print_z_mm(self) -> float:
        return DEPTH_MM - (self.axis_z_mm - CAVITY_DIAMETER_MM / 2.0)

    @property
    def bury_plane_print_z_mm(self) -> float:
        # Round upward to a real 0.20-mm project layer.  The common LM/UM
        # cavity top is exactly 5.80 mm.  This is the last fully open layer,
        # not the slicer's pause-marker layer: the pause is attached to the
        # following, first-closing layer.
        return (
            math.ceil((self.raw_roof_start_print_z_mm - 1.0e-9)
                      / TARGET_LAYER_MM)
            * TARGET_LAYER_MM
        )

    @property
    def roof_start_source_z_mm(self) -> float:
        return DEPTH_MM - self.bury_plane_print_z_mm

    @property
    def roof_apex_source_z_mm(self) -> float:
        return self.roof_start_source_z_mm - ROOF_HEIGHT_MM


_sites = {site["name"]: site for site in side_magnet_sites()}
_lm_axis_z = float(_sites["lm_upper_right"]["z_mm"] - CORE_REAR_Z)
_um_axis_z = float(_sites["um_right"]["z_mm"] - CORE_REAR_Z)

LM_Y_MM = -(STATION_GAP_MM / 2.0 + PAD_LM_TANGENTIAL_MM / 2.0)
UM_Y_MM = +(STATION_GAP_MM / 2.0 + PAD_UM_TANGENTIAL_MM / 2.0)

STATIONS = (
    Station(
        "lm",
        float(LM_CORE_R),
        float(PAD_LM_TANGENTIAL_MM),
        float(PAD_LM_RADIAL_MM),
        LM_Y_MM,
        _lm_axis_z,
    ),
    Station(
        "um",
        float(UM_CORE_R),
        float(PAD_UM_TANGENTIAL_MM),
        float(PAD_UM_RADIAL_MM),
        UM_Y_MM,
        _um_axis_z,
    ),
)


def _z_cylinder(cx: float, cy: float, radius: float, z0: float, z1: float):
    return Pos(cx, cy, (z0 + z1) / 2.0) * Cylinder(radius, z1 - z0)


def _crop_box(x0: float, x1: float, y0: float, y1: float):
    return Pos(x0, y0, -BOOLEAN_EPS_MM) * Box(
        x1 - x0,
        y1 - y0,
        DEPTH_MM + 2.0 * BOOLEAN_EPS_MM,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )


def _annular_station(
    station: Station,
    inner_radius_mm: float,
    outer_radius_mm: float,
    tangential_mm: float,
    x0: float,
    x1: float,
):
    """Exact cylindrical local arc, cropped to one compact root patch."""
    cx = -station.radius_mm
    outer = _z_cylinder(cx, station.y_mm, outer_radius_mm, 0.0, DEPTH_MM)
    inner = _z_cylinder(
        cx,
        station.y_mm,
        inner_radius_mm,
        -BOOLEAN_EPS_MM,
        DEPTH_MM + BOOLEAN_EPS_MM,
    )
    crop = _crop_box(
        x0,
        x1,
        station.y_mm - tangential_mm / 2.0,
        station.y_mm + tangential_mm / 2.0,
    )
    return ((outer - inner) & crop).clean()


def _carrier_station(station: Station):
    base = _annular_station(
        station,
        station.radius_mm - RELEASED_CARRIER_LIP_MM,
        station.radius_mm,
        station.tangential_mm,
        -CARRIER_MAGNET_LAND_MM - 0.5,
        0.25,
    )
    backing_width = CAVITY_DIAMETER_MM + 1.20
    backing = _annular_station(
        station,
        station.radius_mm - CARRIER_MAGNET_LAND_MM,
        station.radius_mm - RELEASED_CARRIER_LIP_MM + BOOLEAN_EPS_MM,
        backing_width,
        -CARRIER_MAGNET_LAND_MM - 0.5,
        -RELEASED_CARRIER_LIP_MM + 0.15,
    )
    return base.fuse(backing).clean()


def _wing_station(station: Station):
    inner_radius = station.radius_mm + COUPON_INTERFACE_GAP_MM
    outer_radius = inner_radius + station.wing_radial_mm
    return _annular_station(
        station,
        inner_radius,
        outer_radius,
        station.tangential_mm,
        -0.75,
        COUPON_INTERFACE_GAP_MM + station.wing_radial_mm + 0.35,
    )


def _x_axis_cylinder(x0: float, y: float, z: float, length: float):
    return (
        Pos(x0, y, z)
        * Rot(Y=90.0)
        * Cylinder(
            CAVITY_DIAMETER_MM / 2.0,
            length,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
    )


def _roof_wedge(x0: float, station: Station, radial_depth_mm: float):
    z0 = station.roof_start_source_z_mm
    half = CAVITY_DIAMETER_MM / 2.0
    wire = Wire.make_polygon(
        (
            Vector(x0, station.y_mm - half, z0),
            Vector(x0, station.y_mm + half, z0),
            Vector(x0, station.y_mm, z0 - ROOF_HEIGHT_MM),
        ),
        close=True,
    )
    return extrude(
        Face(wire),
        amount=radial_depth_mm,
        dir=Vector(1.0, 0.0, 0.0),
    )


def _cavity_cutters(station: Station, owner: str):
    if owner == "carrier":
        x0 = -FACE_SKIN_MM - CAVITY_RADIAL_DEPTH_MM
    elif owner == "wing":
        x0 = COUPON_INTERFACE_GAP_MM + FACE_SKIN_MM
    else:
        raise ValueError(f"unknown coupon owner: {owner!r}")

    z0 = station.roof_start_source_z_mm
    half = CAVITY_DIAMETER_MM / 2.0
    curved_cradle = _x_axis_cylinder(
        x0,
        station.y_mm,
        station.axis_z_mm,
        CAVITY_RADIAL_DEPTH_MM,
    )
    square_upper_half = Pos(x0, station.y_mm - half, z0) * Box(
        CAVITY_RADIAL_DEPTH_MM,
        CAVITY_DIAMETER_MM,
        station.axis_z_mm - z0 + BOOLEAN_EPS_MM,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    closing_roof = _roof_wedge(x0, station, CAVITY_RADIAL_DEPTH_MM)
    return curved_cradle, square_upper_half, closing_roof


def _cut_all_cavities(part, owner: str):
    result = part
    for station in STATIONS:
        for cutter in _cavity_cutters(station, owner):
            result = result - cutter
    return result.clean()


def _check_single_solid(shape, label: str):
    solids = list(shape.solids())
    if not shape.is_valid or len(solids) != 1 or solids[0].volume <= 0.01:
        raise RuntimeError(
            f"{label} must be one valid positive solid; "
            f"valid={shape.is_valid}, volumes={[solid.volume for solid in solids]}"
        )


def _single_solid(shape, label: str):
    _check_single_solid(shape, label)
    shape.label = label
    return shape


def carrier_coupon_installed():
    lm, um = (_carrier_station(station) for station in STATIONS)
    lm_inner_edge = LM_Y_MM + PAD_LM_TANGENTIAL_MM / 2.0
    um_inner_edge = UM_Y_MM - PAD_UM_TANGENTIAL_MM / 2.0
    # Bridge through the rear overlap shared by the released lip and the local
    # backing.  The original 0.50-mm bridge no longer reached both curved roots
    # after the skins were thickened; extending it toward the front beyond this
    # 1.00-mm band would intrude on the mating Ae curvature in the station gap.
    bridge = Pos(-CARRIER_MAGNET_LAND_MM + 0.02, lm_inner_edge - 0.12, 0.0) * Box(
        1.00,
        um_inner_edge - lm_inner_edge + 0.24,
        DEPTH_MM,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    part = _cut_all_cavities(lm.fuse(um, bridge).clean(), "carrier")
    return _single_solid(part, "obiwan_lm_um_carrier_embed_coupon")


def wing_coupon_installed():
    lm, um = (_wing_station(station) for station in STATIONS)
    lm_inner_edge = LM_Y_MM + PAD_LM_TANGENTIAL_MM / 2.0
    um_inner_edge = UM_Y_MM - PAD_UM_TANGENTIAL_MM / 2.0
    bridge = Pos(5.40, lm_inner_edge - 0.12, 0.0) * Box(
        0.75,
        um_inner_edge - lm_inner_edge + 0.24,
        DEPTH_MM,
        align=(Align.MIN, Align.MIN, Align.MIN),
    )
    part = _cut_all_cavities(lm.fuse(um, bridge).clean(), "wing")
    return _single_solid(part, "obiwan_ae_lm_um_embed_coupon")


def _front_down(part, label: str):
    rotated = Rot(X=180.0) * part
    bounds = rotated.bounding_box()
    printed = Pos(-bounds.min.X, -bounds.min.Y, -bounds.min.Z) * rotated
    return _single_solid(printed, label)


def carrier_coupon_print():
    return _front_down(
        carrier_coupon_installed(),
        "obiwan_lm_um_carrier_embed_coupon_front_down",
    )


def wing_coupon_print():
    return _front_down(
        wing_coupon_installed(),
        "obiwan_ae_lm_um_embed_coupon_front_down",
    )


def validate_coupon(carrier=None, wing=None) -> dict[str, float]:
    carrier = carrier or carrier_coupon_installed()
    wing = wing or wing_coupon_installed()
    _check_single_solid(carrier, "carrier validation")
    _check_single_solid(wing, "wing validation")

    gap = float(carrier.distance_to(wing))
    if not math.isclose(gap, COUPON_INTERFACE_GAP_MM, abs_tol=0.002):
        raise RuntimeError(
            f"coupon interface gap drifted: {gap:.6f} mm"
        )

    for station in STATIONS:
        carrier_cavity_x = -FACE_SKIN_MM - CAVITY_RADIAL_DEPTH_MM / 2.0
        wing_cavity_x = (
            COUPON_INTERFACE_GAP_MM
            + FACE_SKIN_MM
            + CAVITY_RADIAL_DEPTH_MM / 2.0
        )
        probes = (
            (carrier, (-FACE_SKIN_MM / 2.0, station.y_mm, station.axis_z_mm), True),
            (carrier, (carrier_cavity_x, station.y_mm, station.axis_z_mm), False),
            (carrier, (-CARRIER_MAGNET_LAND_MM + INNER_SKIN_MM / 2.0,
                       station.y_mm, station.axis_z_mm), True),
            (wing, (COUPON_INTERFACE_GAP_MM + FACE_SKIN_MM / 2.0,
                    station.y_mm, station.axis_z_mm), True),
            (wing, (wing_cavity_x, station.y_mm, station.axis_z_mm), False),
            (carrier, (carrier_cavity_x, station.y_mm,
                       station.roof_apex_source_z_mm / 2.0), True),
            (wing, (wing_cavity_x, station.y_mm,
                    station.roof_apex_source_z_mm / 2.0), True),
        )
        for shape, point, expected_inside in probes:
            actual = bool(shape.is_inside(point, tolerance=1.0e-5))
            if actual != expected_inside:
                raise RuntimeError(
                    f"{station.key} closure probe {point} expected "
                    f"inside={expected_inside}, got {actual}"
                )

    return {
        "interface_gap_mm": gap,
        "lm_bury_plane_print_z_mm": STATIONS[0].bury_plane_print_z_mm,
        "um_bury_plane_print_z_mm": STATIONS[1].bury_plane_print_z_mm,
        "carrier_backing_add_mm": CARRIER_BACKING_ADD_MM,
    }


def design_facts() -> dict[str, object]:
    station_facts = {}
    for station in STATIONS:
        sag = station.radius_mm - math.sqrt(
            station.radius_mm**2 - (CAVITY_DIAMETER_MM / 2.0) ** 2
        )
        station_facts[station.key] = {
            "released_radius_mm": station.radius_mm,
            "released_axis_z_from_rear_mm": station.axis_z_mm,
            "bury_plane_print_z_mm": station.bury_plane_print_z_mm,
            "roof_apex_source_z_mm": station.roof_apex_source_z_mm,
            "minimum_carrier_face_skin_at_cavity_edge_mm": FACE_SKIN_MM - sag,
            "released_ae_root_tangential_mm": station.tangential_mm,
            "released_ae_root_radial_mm": station.wing_radial_mm,
        }
    return {
        "piece_count": 2,
        "print_orientation": "front face down",
        "depth_mm": DEPTH_MM,
        "nominal_magnet_mm": [MAGNET_NOMINAL_DIAMETER_MM, 2.0],
        "cavity_diameter_mm": CAVITY_DIAMETER_MM,
        "cavity_radial_depth_mm": CAVITY_RADIAL_DEPTH_MM,
        "face_skin_nominal_mm": FACE_SKIN_MM,
        "inner_skin_nominal_mm": INNER_SKIN_MM,
        "interface_gap_mm": COUPON_INTERFACE_GAP_MM,
        "nominal_coupon_magnet_face_separation_mm": (
            2.0 * FACE_SKIN_MM + COUPON_INTERFACE_GAP_MM),
        "roof_angle_deg": ROOF_ANGLE_DEG,
        "target_layer_mm": TARGET_LAYER_MM,
        "carrier_released_lip_mm": RELEASED_CARRIER_LIP_MM,
        "carrier_local_backing_add_mm": CARRIER_BACKING_ADD_MM,
        "stations": station_facts,
    }


def gen_step():
    carrier = carrier_coupon_installed()
    wing = wing_coupon_installed()
    validate_coupon(carrier, wing)
    assembly = Compound(children=[carrier, wing])
    assembly.label = "obiwan_ae_lm_um_embedded_magnet_coupon_pair"
    return assembly
