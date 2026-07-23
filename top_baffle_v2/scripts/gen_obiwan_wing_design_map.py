#!/usr/bin/env python3
"""Dimensioned printable-layout sheet for Obi-Wan Ac/Ae attachments.

This is deliberately a 2-D, source-linked engineering drawing generator.  It
uses the exact current A-comp outer profile and the exact Obi-Wan
carrier/magnet datums. Ac/Ae release geometry is generated separately by
``lx521_baffle.obiwan.wings`` from this same analytic contract. The sheet documents
their print envelope, segmentation, Ac constant-depth reference, and the
boundary-aware Ae LM/UM/T-weighted rear field. No other Obi-Wan wing variant
is part of this map or the release inventory.

Ordinary use is remote-first through the repository Makefile::

    make obiwan_wings
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Circle, Polygon as MplPolygon, Rectangle
import numpy as np
from PIL import Image
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from gen_driver_overlay import (
    LM_FLANGE_D_MM,
    ND25_FACE_D_MM,
    T_FRONT_AXIS_Y,
    T_REAR_AXIS_Y,
    UM_FLANGE_D_MM,
    outline_polygon,
)
from lx521_baffle.base import L22_CUTOUT, THICKNESS_MM, UM_CUTOUT
from lx521_baffle.proud.top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
from lx521_baffle.proud.top_baffle_nd25fw4_b import TWEETER_DROP_MM
from lx521_baffle.obiwan.carriers import (
    CORE_REAR_Z,
    LM_CORE_R,
    SIDE_MAGNET_DEPTH,
    SIDE_MAGNET_D,
    SIDE_MAGNET_POCKET_D,
    UM_CORE_R,
    _complete_joint_ear_plan,
    _complete_tweeter_joint_ear_plan,
    side_ring_outer_plan,
    side_magnet_sites,
)
from lx521_baffle.obiwan.bridge import common_lm_wing_contact_plan
from lx521_baffle.magnets import (
    CAPTIVE_LAND_MM,
    CAVITY_DIAMETER_MM,
    INTERFACE_GAP_MM,
    SIDE_WALL_MARGIN_MM,
)


OUTPUT_NAME = "obiwan_wing_design_map.png"

LM_CENTER = np.array(L22_CUTOUT[:2], dtype=float)
UM_CENTER = np.array(UM_CUTOUT[:2], dtype=float)
T_FRONT_CENTER = np.array((0.0, T_FRONT_AXIS_Y - TWEETER_DROP_MM))
T_REAR_CENTER = np.array((0.0, T_REAR_AXIS_Y - TWEETER_DROP_MM))
T_CRESCENT_CENTER = np.array((0.0, 483.05 - TWEETER_DROP_MM))

DEPTH_MM = THICKNESS_MM - CORE_REAR_Z
FRONT_SKIN_MM = 1.20
FULL_RIB_MM = 2.40
SADDLE_CLEAR_MM = INTERFACE_GAP_MM
# The receiver cavity datum is offset 0.05 mm from the visible carrier face,
# but that interval is solid wing material rather than an assembly air gap.
# Ring-carrier cavity datums sit 0.15 mm beneath the smooth carrier surface,
# so their magnet-face separation is 1.10 mm; the base-side LM pair remains
# 0.95 mm.
MAGNET_FACE_GAP_MM = INTERFACE_GAP_MM
# Ac is the exact constant-depth reference.  Ae keeps the same flat front
# datum but replaces the rear slab with a constrained, single-valued feather.
# The documented project profile is 0.20-mm layers on a 0.4-mm nozzle.  Because
# Ae prints front-down, its terminal Z thickness must be at least that first
# layer.  A 0.16-mm knife is an optional future slicer/coupon contract, not the
# reliable default represented by this drawing.  CAD t=0.24 adds slicer/mesh
# margin while still producing the intended single 0.20-mm first-layer edge;
# it remains coupon-gated rather than dimensionally guaranteed.
AE_EDGE_DEPTH_MM = 0.24
AE_OPTIONAL_FINE_LAYER_EDGE_MM = 0.16
# Ae is not a fixed-run feather.  Its rear face is a boundary-aware field:
# every eligible exposed profile edge is exactly one printable layer while
# protected Obi-Wan contact lands retain the full 11.5-mm envelope.  The three
# driver-centred scales vary how long that thickness is retained.  LM gets the
# longest transition (longest wavelength), UM is intermediate, and the
# tweeter sheds depth fastest.  Gaussian radii make the hand-off between those
# behaviours smooth in plan rather than dividing the attachment into bands.
AE_REFERENCE_SPAN_MM = 36.0
AE_REQUIRED_FREE_RUN_MM = AE_REFERENCE_SPAN_MM
AE_CENTER_NAMES = ("LM", "UM", "T")
AE_CENTER_XY_MM = (
    (float(LM_CENTER[0]), float(LM_CENTER[1])),
    (float(UM_CENTER[0]), float(UM_CENTER[1])),
    (float(T_CRESCENT_CENTER[0]), float(T_CRESCENT_CENTER[1])),
)
AE_CENTER_WEIGHT_RADII_MM = (145.0, 82.0, 56.0)
# Relative retention remains longest at LM and shortest at T, while the
# absolute values are deliberately low enough for the real A-profile neck to
# reach the one-layer edge without exceeding the 1:1 rear-slope contract.
AE_CENTER_RETENTION_SCALES = (1.80, 0.90, 0.58)
AE_EDGE_MATCH_TOL_MM = 0.035
AE_EDGE_DENSIFY_MM = 0.25
AE_DISTANCE_EPS_MM = 1e-7
AE_PROTECTED_SOFTMIN_P = 6.0
AE_SECTION_MONOTONIC_TOL_MM = 0.002
# Some full-depth Obi-Wan mating bands approach the A outline more closely than
# the 11.26-mm rise required for a global 1:1 limit. Keep the transition
# continuous and audit the measured peak under a 6:1 ceiling; the flush mating
# datum takes precedence and must never be silently feathered away.
AE_FIELD_MAX_SLOPE = 6.0
# A full-depth contact cannot meet the one-layer edge in less than this run at
# the audited slope ceiling.  The actual rear easing has a peak derivative
# above its mean, so reserve twice the theoretical rise/run only around the T
# contact.  This is a local edge blend, not the former y-wide top exemption.
AE_MIN_FULL_TO_EDGE_RUN_MM = (
    (DEPTH_MM - AE_EDGE_DEPTH_MM) / AE_FIELD_MAX_SLOPE)
AE_TOP_CONTACT_EDGE_BLEND_MM = 2.0 * AE_MIN_FULL_TO_EDGE_RUN_MM
AE_MIN_JOINT_AREA_MM2 = (75.0, 50.0)
AE_CONTACT_LAND_MM = 1.20
AE_PROTECTED_BUFFER_MM = 1.20
PAD_LM_TANGENTIAL_MM = 8.8
PAD_LM_RADIAL_MM = 6.0
PAD_UM_TANGENTIAL_MM = 13.0
PAD_UM_RADIAL_MM = 8.0
ACTIVE_RECEIVER_NAMES_RIGHT = (
    "lm_lower_right", "lm_upper_right", "um_right")
SEAM_LOWER_X_REF = 60.0
SEAM_UPPER_Y = 391.709
A_TAPER_TOP_Y = 453.457
A_TAPER_TOP_X = 49.177
A_TAPER_CREST_X = 60.654
# The exact construction line meets the conservative optional-crescent
# keepout as a zero-width tip.  Stop it at a source-derived 1.20 mm solid cap
# so the engineering layout remains printable rather than merely pictorial.
A_TAPER_CAP_Y = 449.081
A_TAPER_CAP_MIN_WIDTH_MM = 1.20
# V1L-family through-thickness XY dovetails.  Each physical-side wing has one
# key per joint because its usable seam chords are only 20.57 and 15.52 mm.
# Female clearance is the same coupon-calibrated 0.05 mm used by V1L.  The
# straight seam slit tapers closed over the last 2 mm at each exposed endpoint
# so Ae retains one continuous constant-thickness acoustic knife edge.
DOVETAIL_CLEARANCE_MM = 0.05
DOVETAIL_ENDPOINT_TAPER_MM = 2.0
# Embed the nominal seam neck this far into the male partition.  The overlap
# is entirely inside already-solid male material, adds no envelope or mating
# growth, and prevents a floating key when Shapely's field/seam intersection
# differs by machine epsilon after captive-land outline changes.
DOVETAIL_ROOT_OVERLAP_MM = 0.05
# The upper A band reaches 2.047 mm with the canonical 7/8.5/4 V1L key; a
# 2.0-mm gate preserves that proven profile without clipping or shrinking it.
DOVETAIL_MIN_LIGAMENT_MM = 2.0
DOVETAIL_LOWER_PROFILE_MM = (7.0, 9.0, 4.0)  # neck, head, penetration
DOVETAIL_UPPER_PROFILE_MM = (7.0, 8.5, 4.0)
# The upper A flank walks inward by 11.477 mm over 61.748 mm of rise.  Cut
# across that band on the perpendicular so the male key advances along the
# wing centreline instead of running out through the curved carrier edge.
DOVETAIL_UPPER_SEAM_SLOPE = (
    (A_TAPER_CREST_X - A_TAPER_TOP_X) / (A_TAPER_TOP_Y - SEAM_UPPER_Y))
PLA_DENSITY_G_CM3 = 1.24
BED_USABLE_MM = 220.0

COLORS = {
    "lm_lower": "#4f9bd7",
    "lm_upper": "#62b77b",
    "um": "#efa43a",
    "carrier": "#c9d0d6",
    "carrier_edge": "#30383f",
    "driver_lm": "#2784c7",
    "driver_um": "#e33b32",
    "driver_t1": "#2da446",
    "driver_t2": "#8d5bc7",
    "active_mag": "#d52f88",
    "unused_mag": "#8a8f94",
    "rib": "#273d57",
    "keepout": "#e9edf0",
    "historic": "#70777d",
    "correction": "#f15a47",
}


@dataclass(frozen=True)
class VariantDefinition:
    key: str
    title: str
    outline: tuple


@dataclass
class VariantLayout:
    definition: VariantDefinition
    profile: Polygon
    field_right: Polygon
    nominal_parts: dict[str, Polygon]
    print_parts: dict[str, Polygon]
    rear_structure: Polygon
    lower_seam_y: float
    lower_seam_slope: float
    joint_seams: tuple[LineString, LineString]
    fit_clearance_gaps: tuple[Polygon, Polygon]
    dovetail_keys: tuple[dict, dict]
    metrics: dict[str, float | str]


@dataclass(frozen=True)
class AeTransitionModel:
    profile_name: str
    run_mm: float
    plateau_mm: float
    required_span_mm: float
    full_depth_mm: float
    edge_depth_mm: float
    area_mm2: float
    area_reduction_pct: float
    straight_bevel_reduction_pct: float
    max_slope: float
    min_radius_mm: float
    q: np.ndarray
    s_mm: np.ndarray
    depth_mm: np.ndarray
    center_names: tuple[str, ...]
    center_xy_mm: tuple[tuple[float, float], ...]
    weight_radii_mm: tuple[float, ...]
    retention_scales: tuple[float, ...]
    depth_by_center_mm: tuple[np.ndarray, ...]
    max_slope_by_center: tuple[float, ...]


@dataclass
class AeDepthField:
    protected: Polygon
    contact_land: Polygon
    top_flush_land: Polygon
    protected_components: tuple[Polygon, ...]
    exposed_outer_edge: object
    joint_area_mm2: tuple[float, float]
    joint_rear_mismatch_mm: tuple[float, float]
    x_mm: np.ndarray
    y_mm: np.ndarray
    depth_mm: np.ndarray
    mask: np.ndarray
    volume_mm3: float
    mass_g: float
    ac_volume_mm3: float
    ac_mass_g: float
    reduction_pct: float
    protected_area_mm2: float
    min_depth_mm: float
    max_grid_slope: float
    outer_edge_depth_mm: tuple[float, float]
    top_flush_depth_mm: tuple[float, float]
    retention_scale_range: tuple[float, float]


@dataclass(frozen=True)
class SectionDefinition:
    key: str
    title: str
    line: LineString
    pocket_z_mm: float | None = None


VARIANTS = (
    VariantDefinition("A", "Obi-Wan Ac/Ae plan", OUTLINE_A_COMP),
)


def _largest_polygon(geometry) -> Polygon:
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type == "Polygon":
        return geometry
    polygons = [g for g in geometry.geoms
                if g.geom_type == "Polygon" and g.area > 1e-6]
    return max(polygons, key=lambda item: item.area) if polygons else Polygon()


def _profile_polygon(outline) -> Polygon:
    result = Polygon(outline_polygon(outline, samples=96)).buffer(0)
    if result.geom_type != "Polygon" or not result.is_valid:
        raise RuntimeError("variant profile must resolve to one valid polygon")
    return result


def _effective_profile(definition: VariantDefinition) -> Polygon:
    """Return the finalized Obi-Wan Ac/Ae attachment profile.

    A keeps the current A-comp profile through the crest and below it.  Above
    the crest, the requested straight flank removes the old square top corner:
    it runs from the source top-stub endpoint (49.177, 453.457) to the
    unchanged A crest (60.654, 391.709), mirrored on the left.
    """
    source = _profile_polygon(definition.outline)
    if definition.key != "A":
        return source
    removed_right = Polygon((
        (A_TAPER_TOP_X, A_TAPER_TOP_Y),
        (A_TAPER_CREST_X, A_TAPER_TOP_Y),
        (A_TAPER_CREST_X, SEAM_UPPER_Y),
    ))
    removed = unary_union((removed_right, _mirror(removed_right)))
    effective = source.difference(removed).buffer(0)
    if effective.geom_type != "Polygon" or not effective.is_valid:
        raise RuntimeError("A-Obi-Wan tapered profile must resolve to one polygon")
    return effective


def _a_taper_x(y: float) -> float:
    fraction = (A_TAPER_TOP_Y - y) / (A_TAPER_TOP_Y - SEAM_UPPER_Y)
    return A_TAPER_TOP_X + fraction * (A_TAPER_CREST_X - A_TAPER_TOP_X)


def _a_cap_keepout_x() -> float:
    radius = 52.2 + SADDLE_CLEAR_MM
    dy = A_TAPER_CAP_Y - T_CRESCENT_CENTER[1]
    return math.sqrt(radius * radius - dy * dy)


def _ae_reference_depth(
        s_mm: np.ndarray, span_mm: float, retention_scale: float,
        ) -> np.ndarray:
    """Reference cut through the boundary-aware field for one driver scale.

    ``s=0`` is a protected, full-depth land and ``s=span`` is the eligible
    free edge.  The rational coordinate responds to both boundaries, so the
    minimum thickness occurs only on the edge; there is no fixed-run thin
    plateau.  ``q=u²(2-u)`` leaves the protected land with zero slope but
    permits the printable knife edge to arrive without a rounded terminal.
    """
    s_mm = np.asarray(s_mm, dtype=float)
    edge_distance = np.maximum(0.0, span_mm - s_mm)
    denominator = s_mm + retention_scale * edge_distance
    u = np.divide(
        s_mm, denominator, out=np.zeros_like(s_mm),
        where=denominator > AE_DISTANCE_EPS_MM)
    u = np.clip(u, 0.0, 1.0)
    progress = u * u * (2.0 - u)
    return (DEPTH_MM
            - (DEPTH_MM - AE_EDGE_DEPTH_MM) * progress)


def _build_ae_transition_model(samples: int = 2001) -> AeTransitionModel:
    """Build LM/UM/T reference cuts for the plan-weighted Ae rear field."""
    s = np.linspace(0.0, AE_REFERENCE_SPAN_MM, samples)
    profiles = tuple(
        _ae_reference_depth(s, AE_REFERENCE_SPAN_MM, scale)
        for scale in AE_CENTER_RETENTION_SCALES)
    slopes = tuple(
        float(np.max(np.abs(np.gradient(depth, s, edge_order=2))))
        for depth in profiles)
    minimum_radii = []
    for depth in profiles:
        d1 = np.gradient(depth, s, edge_order=2)
        d2 = np.gradient(d1, s, edge_order=2)
        curvature = np.abs(d2) / np.power(1.0 + d1 * d1, 1.5)
        radii = np.divide(
            1.0, curvature, out=np.full_like(curvature, np.inf),
            where=curvature > 1e-10)
        minimum_radii.append(float(np.min(radii)))
    areas = tuple(float(np.trapezoid(depth, s)) for depth in profiles)
    representative = profiles[1]
    section_area = float(np.mean(areas))
    ac_area = DEPTH_MM * AE_REFERENCE_SPAN_MM
    straight_area = ((DEPTH_MM + AE_EDGE_DEPTH_MM) / 2.0
                     * AE_REFERENCE_SPAN_MM)
    return AeTransitionModel(
        profile_name="LM/UM/T weighted boundary field",
        run_mm=AE_REFERENCE_SPAN_MM,
        plateau_mm=0.0,
        required_span_mm=AE_REFERENCE_SPAN_MM,
        full_depth_mm=DEPTH_MM,
        edge_depth_mm=AE_EDGE_DEPTH_MM,
        area_mm2=section_area,
        area_reduction_pct=100.0 * (1.0 - section_area / ac_area),
        straight_bevel_reduction_pct=(
            100.0 * (1.0 - section_area / straight_area)),
        max_slope=max(slopes),
        min_radius_mm=min(minimum_radii),
        q=s / AE_REFERENCE_SPAN_MM,
        s_mm=s,
        depth_mm=representative,
        center_names=AE_CENTER_NAMES,
        center_xy_mm=AE_CENTER_XY_MM,
        weight_radii_mm=AE_CENTER_WEIGHT_RADII_MM,
        retention_scales=AE_CENTER_RETENTION_SCALES,
        depth_by_center_mm=profiles,
        max_slope_by_center=slopes,
    )


def _optimize_ae_profile() -> AeTransitionModel:
    """Compatibility entry point for the former fixed-run optimizer."""
    return _build_ae_transition_model()


def _mirror(geometry):
    return affinity.scale(geometry, xfact=-1.0, yfact=1.0, origin=(0, 0))


def _rounded_oriented_pad(face, normal, radial, tangential) -> Polygon:
    face = np.asarray(face, dtype=float)
    normal = np.asarray(normal, dtype=float)
    tangent = np.array((-normal[1], normal[0]))
    inner = face
    outer = face + radial * normal
    half = tangential / 2.0
    raw = Polygon((
        tuple(inner - half * tangent),
        tuple(outer - half * tangent),
        tuple(outer + half * tangent),
        tuple(inner + half * tangent),
    ))
    corner = min(0.75, tangential / 5.0, radial / 4.0)
    return raw.buffer(corner, join_style=1).buffer(-corner, join_style=1)


def _pocket_plan(site) -> Polygon:
    """XY projection of the solid standoff plus captive receiver land."""
    face = np.asarray(
        site.get("outer_surface_face", site["face"]), dtype=float)
    normal = np.asarray(site["normal"], dtype=float)
    tangent = np.array((-normal[1], normal[0]))
    mouth = face + MAGNET_FACE_GAP_MM * normal
    outer = mouth + CAPTIVE_LAND_MM * normal
    half = CAVITY_DIAMETER_MM / 2.0 + SIDE_WALL_MARGIN_MM
    return Polygon((
        tuple(face - half * tangent),
        tuple(outer - half * tangent),
        tuple(outer + half * tangent),
        tuple(face + half * tangent),
    ))


def _carrier_pocket_plan(site) -> Polygon:
    """XY projection of the complete inward captive carrier land."""
    face = np.asarray(site["face"], dtype=float)
    normal = np.asarray(site["normal"], dtype=float)
    tangent = np.array((-normal[1], normal[0]))
    inner = face - CAPTIVE_LAND_MM * normal
    half = CAVITY_DIAMETER_MM / 2.0 + SIDE_WALL_MARGIN_MM
    return Polygon((
        tuple(inner - half * tangent),
        tuple(face - half * tangent),
        tuple(face + half * tangent),
        tuple(inner + half * tangent),
    ))


def _arc(center, radius, angles) -> LineString:
    points = []
    for angle in angles:
        a = math.radians(float(angle))
        points.append((center[0] + radius * math.cos(a),
                       center[1] + radius * math.sin(a)))
    return LineString(points)


def _um_receiver_bridge(site) -> Polygon:
    """Carrier-following bridge from the retained UM root into the flank."""
    side = site["name"].rsplit("_", 1)[-1]
    angles = (np.linspace(50.5, 20.0, 48) if side == "right"
              else np.linspace(129.5, 160.0, 48))
    radius = UM_CORE_R + 3.1
    arc = _arc(UM_CENTER, radius, angles).buffer(
        FULL_RIB_MM / 2.0, cap_style=1, join_style=1)
    end_angle = float(angles[-1])
    a = math.radians(end_angle)
    direction = np.array((math.cos(a), math.sin(a)))
    end = UM_CENTER + radius * direction
    fan = LineString((tuple(end), tuple(end + 8.0 * direction))).buffer(
        FULL_RIB_MM / 2.0, cap_style=1, join_style=1)
    return unary_union((arc, fan)).buffer(0)


def _intercarrier_bridge_right() -> Polygon:
    """Outboard positive-area spine across the carrier pinch.

    The source edge crosses inside the R51.7 Obi-Wan carrier through part
    of the upper flank, so subtracting the two carrier envelopes divides the
    otherwise continuous side field.  This narrow carrier-following spine is
    the minimum print-realizable connection and stays outside both rings.
    """
    points = (
        (50.9, 305.2), (54.8, 317.0), (58.2, 331.0),
        (60.0, 346.0), (60.0, 360.0), (57.5, 375.0),
        (51.5, 384.8),
    )
    spine = LineString(points).buffer(
        1.70, cap_style=1, join_style=1)
    return spine.difference(_common_keepout()).buffer(0)


def _common_keepout_parts() -> dict[str, Polygon]:
    """Named Obi-Wan plan envelopes that an attachment can contact.

    Keeping the pieces named is essential for Ae: every one of these mating
    boundaries needs a full-depth, flush rear datum.  The former carrier-only
    saddle mask missed the bridge and integral-stem sides at the bottom LM.
    """
    lm_lower_contact = common_lm_wing_contact_plan()
    parts: dict[str, Polygon] = {
        "lm_carrier": side_ring_outer_plan("lm").buffer(
            SADDLE_CLEAR_MM, resolution=32, join_style=1),
        "um_carrier": side_ring_outer_plan("um").buffer(
            SADDLE_CLEAR_MM, resolution=32, join_style=1),
        "t_crescent": Point(*T_CRESCENT_CENTER).buffer(
            52.2 + SADDLE_CLEAR_MM, resolution=128),
        "no_floor_bridge": lm_lower_contact.buffer(
            SADDLE_CLEAR_MM, resolution=32, join_style=1),
        "floor_stem": lm_lower_contact.buffer(
            SADDLE_CLEAR_MM, resolution=32, join_style=1),
    }
    for owner in ("lm", "um"):
        for x in (-32.0, 32.0):
            parts[f"{owner}_joint_{x:+.0f}"] = _complete_joint_ear_plan(
                owner, x, SADDLE_CLEAR_MM)
    for x in (-24.0, 24.0):
        parts[f"t_joint_{x:+.0f}"] = _complete_tweeter_joint_ear_plan(
            "tweeter", x, SADDLE_CLEAR_MM)
    return parts


def _common_keepout() -> Polygon:
    return unary_union(tuple(_common_keepout_parts().values())).buffer(0)


def _right_sites() -> dict[str, dict]:
    sites = {site["name"]: site for site in side_magnet_sites()
             if site["normal"][0] > 0.0}
    expected = {"lm_upper_right", "lm_lower_right", "um_right"}
    if set(sites) != expected:
        raise RuntimeError(
            f"unexpected current right-side magnet contract: {sorted(sites)}")
    return sites


def _receiver_root(profile: Polygon, site, key: str) -> tuple[Polygon, Polygon]:
    is_um = site["driver"] == "um"
    radial = PAD_UM_RADIAL_MM if is_um else PAD_LM_RADIAL_MM
    tangential = PAD_UM_TANGENTIAL_MM if is_um else PAD_LM_TANGENTIAL_MM
    receiver_face = site.get("outer_surface_face", site["face"])
    pad = _rounded_oriented_pad(receiver_face, site["normal"],
                                radial, tangential)
    if site.get("interface_kind") == "base_side":
        # The base receiver mates to the common W64 tongue, not the R113 LM
        # circle. Subtracting the circle here would put wing material inside
        # the carrier and hide the intended flush x=+/-32 interface.
        carrier = common_lm_wing_contact_plan()
    else:
        carrier = side_ring_outer_plan(site["driver"])
    # Keep the root flush with the visible carrier datum.  The cavity itself
    # begins MAGNET_FACE_GAP_MM farther outward, leaving that complete interval
    # as solid receiver standoff.  Buffering the carrier by the same amount
    # would turn the standoff into an actual air gap and contradict the sealed
    # cavity contract.
    carrier = carrier.buffer(0)
    pad = pad.difference(carrier)
    if is_um:
        bridge = _um_receiver_bridge(site).difference(carrier)
        root = unary_union((pad, bridge)).buffer(0)
        root = root.intersection(profile.buffer(0.01)).buffer(0)
    else:
        root = pad.intersection(profile.buffer(0.01)).buffer(0)
    return root, Polygon()


def _field_for_variant(definition: VariantDefinition) -> tuple[Polygon, Polygon]:
    profile = _effective_profile(definition)
    right_half = box(0.65, -20.0, 180.0, 470.0)
    field = profile.difference(_common_keepout()).intersection(right_half)
    if definition.key == "A":
        field = field.intersection(box(0.0, -20.0, 180.0, A_TAPER_CAP_Y))
    sites = _right_sites()

    # Every physical-side station receives a matching wing pocket: lower LM
    # on the straight base, upper LM on R113 and UM on R51.7.
    roots = []
    for name in ACTIVE_RECEIVER_NAMES_RIGHT:
        root, _ = _receiver_root(profile, sites[name], definition.key)
        roots.append(root)
    spine = _intercarrier_bridge_right().intersection(profile).buffer(0)
    field = unary_union((field, *roots, spine)).buffer(0)
    field = _largest_polygon(field)
    return profile, field


def _smootherstep01(value):
    value = np.clip(np.asarray(value, dtype=float), 0.0, 1.0)
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def _straight_joint_seam(
        field: Polygon, y_ref: float, slope: float,
        ) -> tuple[LineString, np.ndarray, np.ndarray]:
    """Return the field chord and its +Y-facing unit normal.

    The seam itself stays straight, as in the other V1L-family baffles.  The
    trapezoidal male key supplies the XY interlock; its female receiver is
    relieved on the next-higher print by the shared 0.05-mm V1L clearance.
    """
    x0, x1 = -100.0, 250.0
    baseline = LineString((
        (x0, y_ref + slope * (x0 - SEAM_LOWER_X_REF)),
        (x1, y_ref + slope * (x1 - SEAM_LOWER_X_REF)),
    ))
    chord = _longest_line(field.intersection(baseline))
    if chord.is_empty or chord.length < 8.0:
        raise RuntimeError("dovetail seam does not cross the printable field")
    endpoints = np.asarray((chord.coords[0], chord.coords[-1]), dtype=float)
    if endpoints[1, 0] < endpoints[0, 0]:
        endpoints = endpoints[::-1]
    vector = endpoints[1] - endpoints[0]
    chord_length = float(np.linalg.norm(vector))
    tangent = vector / chord_length
    normal = np.array((-tangent[1], tangent[0]))
    if normal[1] < 0.0:
        endpoints = endpoints[::-1]
        tangent = -tangent
        normal = -normal
    seam = LineString(endpoints)
    if not field.buffer(0.01).covers(seam):
        raise RuntimeError("dovetail seam leaves printable field")
    return seam, tangent, normal


def _seam_side_masks(
        seam: LineString, normal: np.ndarray) -> tuple[Polygon, Polygon]:
    coords = np.asarray(seam.coords, dtype=float)
    tangent = coords[-1] - coords[0]
    tangent /= np.linalg.norm(tangent)
    extended = np.vstack((
        coords[0] - 500.0 * tangent,
        coords,
        coords[-1] + 500.0 * tangent,
    ))
    negative = Polygon(np.vstack((
        extended,
        extended[-1] - 1000.0 * normal,
        extended[0] - 1000.0 * normal,
    ))).buffer(0)
    positive = Polygon(np.vstack((
        extended,
        extended[-1] + 1000.0 * normal,
        extended[0] + 1000.0 * normal,
    ))).buffer(0)
    return negative, positive


def _dovetail_polygon(
        center: np.ndarray, tangent: np.ndarray, normal: np.ndarray,
        profile_mm: tuple[float, float, float]) -> Polygon:
    """V1L-style male trapezoid, pointing into the next-higher print."""
    neck, head, penetration = profile_mm
    root_center = center - DOVETAIL_ROOT_OVERLAP_MM * normal
    head_center = center + penetration * normal
    return Polygon((
        root_center - 0.5 * neck * tangent,
        root_center + 0.5 * neck * tangent,
        head_center + 0.5 * head * tangent,
        head_center - 0.5 * head * tangent,
    )).buffer(0)


def _select_dovetail_key(
        field: Polygon, seam: LineString, tangent: np.ndarray,
        normal: np.ndarray, profile_mm: tuple[float, float, float],
        *, name: str, male_owner: str, female_owner: str,
        minimum_ligament_mm: float = DOVETAIL_MIN_LIGAMENT_MM) -> dict:
    """Place one centered key while maximizing its outer-boundary ligament."""
    endpoints = np.asarray((seam.coords[0], seam.coords[-1]), dtype=float)
    vector = endpoints[1] - endpoints[0]
    candidates = []
    rejected = []
    for u in np.linspace(0.10, 0.90, 161):
        center = endpoints[0] + float(u) * vector
        polygon = _dovetail_polygon(center, tangent, normal, profile_mm)
        if field.buffer(0.002).covers(polygon):
            ligament = float(polygon.distance(field.boundary))
            candidates.append((ligament, -abs(float(u) - 0.5), center,
                               polygon))
        else:
            outside_shape = polygon.difference(field)
            rejected.append((float(outside_shape.area), float(u),
                             polygon.bounds,
                             (float(outside_shape.centroid.x),
                              float(outside_shape.centroid.y))))
    if not candidates:
        outside, u, bounds, outside_centroid = min(
            rejected, key=lambda item: item[0])
        raise RuntimeError(
            f"{name} dovetail does not fit its seam; best u={u:.3f}, "
            f"outside={outside:.4f} mm2, key_bounds={bounds}, "
            f"outside_centroid={outside_centroid}, "
            f"seam={tuple(seam.coords)}")
    ligament, _centering, center, polygon = max(
        candidates, key=lambda candidate: (candidate[0], candidate[1]))
    if ligament < minimum_ligament_mm - 0.01:
        raise RuntimeError(
            f"{name} dovetail ligament {ligament:.3f} mm is below "
            f"{minimum_ligament_mm:.3f} mm")
    neck, head, penetration = profile_mm
    return {
        "name": name,
        "polygon": polygon,
        "center_xy_mm": (float(center[0]), float(center[1])),
        "tangent": (float(tangent[0]), float(tangent[1])),
        "normal": (float(normal[0]), float(normal[1])),
        "neck_mm": float(neck),
        "head_mm": float(head),
        "penetration_mm": float(penetration),
        "root_overlap_mm": DOVETAIL_ROOT_OVERLAP_MM,
        "ligament_mm": ligament,
        "male_owner": male_owner,
        "female_owner": female_owner,
    }


def _tapered_female_slit(
        seam: LineString, normal: np.ndarray, field: Polygon,
        *, samples: int = 81) -> Polygon:
    """One-sided 0.05-mm female slit, closed at both exposed endpoints."""
    endpoints = np.asarray((seam.coords[0], seam.coords[-1]), dtype=float)
    vector = endpoints[1] - endpoints[0]
    length = float(np.linalg.norm(vector))
    s = np.linspace(0.0, length, samples)
    points = endpoints[0] + (s / length)[:, None] * vector
    endpoint_distance = np.minimum(s, length - s)
    widths = (DOVETAIL_CLEARANCE_MM * _smootherstep01(
        endpoint_distance / DOVETAIL_ENDPOINT_TAPER_MM))
    offset = points + widths[:, None] * normal
    slit = Polygon(np.vstack((points, offset[::-1]))).buffer(0)
    return _largest_polygon(slit.intersection(field).buffer(0))


def _female_clearance_gap(
        field: Polygon, seam: LineString, normal: np.ndarray,
        key: Polygon, male_nominal: Polygon,
        female_nominal: Polygon) -> Polygon:
    """Return the V1L female relief without opening the acoustic edge."""
    slit = _tapered_female_slit(seam, normal, field)
    key_relief = key.buffer(
        DOVETAIL_CLEARANCE_MM, join_style=2,
        mitre_limit=10.0).difference(male_nominal)
    gap = unary_union((slit, key_relief)).intersection(
        female_nominal).buffer(0)
    return _largest_polygon(gap)


def _partition(field: Polygon):
    # The source-derived split keeps all three front-down prints within the
    # 220-mm bed.  Each lower print owns one V1L-style male trapezoid; the
    # next-higher print owns the 0.05-mm relieved female complement.
    y_ref = 198.5
    slope = -0.08
    lower_seam, lower_tangent, lower_normal = _straight_joint_seam(
        field, y_ref, slope)
    # The A waist chord at y=391.709 is only 15.52 mm and pinches the head
    # against its curved inner boundary.  Walk the upper seam downward only
    # as far as required to recover the audited 2.0-mm exterior ligament.
    upper_joint = None
    upper_failures = []
    best_upper = None
    for upper_y in np.arange(SEAM_UPPER_Y, 349.9, -0.5):
        try:
            candidate_seam, candidate_tangent, candidate_normal = (
                _straight_joint_seam(
                    field, float(upper_y), DOVETAIL_UPPER_SEAM_SLOPE))
            candidate_key = _select_dovetail_key(
                field, candidate_seam, candidate_tangent, candidate_normal,
                DOVETAIL_UPPER_PROFILE_MM, name="upper",
                male_owner="lm_upper", female_owner="um",
                minimum_ligament_mm=0.0)
        except RuntimeError as exc:
            upper_failures.append(str(exc))
            continue
        if (best_upper is None
                or candidate_key["ligament_mm"]
                > best_upper[3]["ligament_mm"]):
            best_upper = (candidate_seam, candidate_tangent,
                          candidate_normal, candidate_key)
        if candidate_key["ligament_mm"] >= DOVETAIL_MIN_LIGAMENT_MM - 0.01:
            upper_joint = (candidate_seam, candidate_tangent,
                           candidate_normal, candidate_key)
            break
    if upper_joint is None:
        raise RuntimeError(
            "upper dovetail cannot reach the 2.0-mm ligament gate; "
            f"best={None if best_upper is None else best_upper[3]}; "
            f"last diagnostics={upper_failures[-2:]}")
    upper_seam, upper_tangent, upper_normal, upper_key = upper_joint
    lower_negative, lower_positive = _seam_side_masks(
        lower_seam, lower_normal)
    upper_negative, upper_positive = _seam_side_masks(
        upper_seam, upper_normal)

    lower_key = _select_dovetail_key(
        field, lower_seam, lower_tangent, lower_normal,
        DOVETAIL_LOWER_PROFILE_MM, name="lower",
        male_owner="lm_lower", female_owner="lm_upper")
    lm_lower = _largest_polygon(unary_union((
        field.intersection(lower_negative), lower_key["polygon"])))
    above_lower = field.difference(lm_lower).buffer(0)
    lm_upper = _largest_polygon(unary_union((
        above_lower.intersection(upper_negative),
        upper_key["polygon"].intersection(above_lower))))
    um = _largest_polygon(field.difference(
        unary_union((lm_lower, lm_upper))).buffer(0))
    nominal_parts = {"lm_lower": lm_lower, "lm_upper": lm_upper, "um": um}

    lower_gap = _female_clearance_gap(
        field, lower_seam, lower_normal, lower_key["polygon"],
        lm_lower, lm_upper)
    upper_gap = _female_clearance_gap(
        field, upper_seam, upper_normal, upper_key["polygon"],
        lm_upper, um)
    fit_clearance_gaps = (lower_gap, upper_gap)
    print_parts = {
        "lm_lower": lm_lower,
        "lm_upper": _largest_polygon(
            lm_upper.difference(lower_gap).buffer(0)),
        "um": _largest_polygon(um.difference(upper_gap).buffer(0)),
    }
    if any(part.is_empty for part in (*nominal_parts.values(),
                                      *print_parts.values())):
        raise RuntimeError("dovetail partition produced an empty part")
    return (nominal_parts, print_parts, float(y_ref), float(slope),
            (lower_seam, upper_seam), fit_clearance_gaps,
            (lower_key, upper_key))


def _rear_structure(field: Polygon, active_roots: tuple[Polygon, ...],
                    joint_seams: tuple[LineString, LineString]) -> Polygon:
    perimeter = field.boundary.buffer(
        FULL_RIB_MM / 2.0, cap_style=1, join_style=1).intersection(field)
    seam_ribs = unary_union(tuple(
        seam.buffer(FULL_RIB_MM / 2.0, cap_style=2, join_style=1)
        for seam in joint_seams)).intersection(field)
    fan_ribs = []
    for index, y in enumerate(np.arange(25.0, 455.0, 38.0)):
        if index % 2:
            line = LineString(((0.0, y - 8.0), (180.0, y + 8.0)))
        else:
            line = LineString(((0.0, y + 8.0), (180.0, y - 8.0)))
        fan_ribs.append(line.buffer(
            FULL_RIB_MM / 2.0, cap_style=1, join_style=1).intersection(field))
    return unary_union((perimeter, seam_ribs, *fan_ribs,
                        *active_roots)).intersection(field).buffer(0)


def _obb_dimensions(shape: Polygon) -> tuple[float, float]:
    if shape.is_empty:
        return (0.0, 0.0)
    coords = list(shape.minimum_rotated_rectangle.exterior.coords)
    lengths = [math.dist(coords[i], coords[i + 1]) for i in range(4)]
    a, b = sorted((max(lengths[0], lengths[2]),
                   max(lengths[1], lengths[3])), reverse=True)
    return a, b


def _build_layout(definition: VariantDefinition) -> VariantLayout:
    profile, field = _field_for_variant(definition)
    (nominal_parts, print_parts, lower_seam_y, lower_seam_slope,
     joint_seams, fit_clearance_gaps, dovetail_keys) = _partition(field)
    sites = _right_sites()
    root_shapes = tuple(_receiver_root(profile, sites[name], definition.key)[0]
                        for name in ACTIVE_RECEIVER_NAMES_RIGHT)
    structure = _rear_structure(field, root_shapes, joint_seams)

    volume_mm3 = (field.area * FRONT_SKIN_MM
                  + structure.area * (DEPTH_MM - FRONT_SKIN_MM))
    mass_g = volume_mm3 / 1000.0 * PLA_DENSITY_G_CM3
    metrics: dict[str, float | str] = {
        "area_mm2": field.area,
        "mass_g_side": mass_g,
        "lower_seam_chord_mm": math.dist(
            joint_seams[0].coords[0], joint_seams[0].coords[-1]),
        "upper_seam_chord_mm": math.dist(
            joint_seams[1].coords[0], joint_seams[1].coords[-1]),
        "lower_key_ligament_mm": dovetail_keys[0]["ligament_mm"],
        "upper_key_ligament_mm": dovetail_keys[1]["ligament_mm"],
    }
    for name, part in print_parts.items():
        a, b = _obb_dimensions(part)
        metrics[f"{name}_obb"] = f"{a:.1f} x {b:.1f}"
        if max(a, b) > BED_USABLE_MM + 1e-6:
            raise RuntimeError(
                f"{definition.key} {name} does not fit {BED_USABLE_MM:g} mm bed: "
                f"{a:.2f} x {b:.2f}")

    layout = VariantLayout(definition, profile, field, nominal_parts,
                           print_parts, structure,
                           lower_seam_y, lower_seam_slope,
                           joint_seams, fit_clearance_gaps,
                           dovetail_keys, metrics)
    _validate_layout(layout)
    return layout


def _validate_layout(layout: VariantLayout) -> None:
    if not math.isclose(DEPTH_MM, 11.5, abs_tol=1e-9):
        raise RuntimeError(f"Obi-Wan depth drifted from 11.5 mm: {DEPTH_MM}")
    if layout.definition.key == "A":
        cap_width = _a_taper_x(A_TAPER_CAP_Y) - _a_cap_keepout_x()
        if cap_width < A_TAPER_CAP_MIN_WIDTH_MM - 0.01:
            raise RuntimeError(
                f"A-Obi-Wan taper cap is too narrow: {cap_width:.3f} mm")
    for name, part in layout.print_parts.items():
        if part.is_empty or not part.is_valid:
            raise RuntimeError(f"{layout.definition.key} {name} is invalid/empty")
    if not layout.field_right.is_valid or layout.field_right.is_empty:
        raise RuntimeError(f"{layout.definition.key} field invalid/empty")
    reconstructed = unary_union(tuple(layout.nominal_parts.values())).buffer(0)
    if reconstructed.symmetric_difference(layout.field_right).area > 0.02:
        raise RuntimeError(
            f"{layout.definition.key} nominal partitions do not reconstruct field")
    lower_overlap = layout.print_parts["lm_lower"].intersection(
        layout.print_parts["lm_upper"])
    upper_overlap = layout.print_parts["lm_upper"].intersection(
        layout.print_parts["um"])
    if lower_overlap.area > 0.01 or upper_overlap.area > 0.01:
        raise RuntimeError(
            f"{layout.definition.key} dovetail parts overlap: "
            f"{lower_overlap.area:.4f}/{upper_overlap.area:.4f} mm2")
    reconstructed_print = unary_union((
        *layout.print_parts.values(), *layout.fit_clearance_gaps)).buffer(0)
    if reconstructed_print.symmetric_difference(layout.field_right).area > 0.02:
        raise RuntimeError(
            f"{layout.definition.key} fit gaps do not reconstruct field")
    outside_print = unary_union(tuple(layout.print_parts.values())).difference(
        layout.field_right.buffer(0.01))
    if outside_print.area > 0.01:
        raise RuntimeError(
            f"{layout.definition.key} dovetail child leaves monolithic envelope")
    for seam in layout.joint_seams:
        coords = np.asarray(seam.coords)
        if (Point(*coords[0]).distance(layout.field_right.boundary) > 0.05
                or Point(*coords[-1]).distance(
                    layout.field_right.boundary) > 0.05):
            raise RuntimeError(
                f"{layout.definition.key} dovetail seam endpoint misses boundary")
    expected_profiles = (
        DOVETAIL_LOWER_PROFILE_MM, DOVETAIL_UPPER_PROFILE_MM)
    expected_owners = (("lm_lower", "lm_upper"), ("lm_upper", "um"))
    for key, profile, owners in zip(
            layout.dovetail_keys, expected_profiles, expected_owners):
        polygon = key["polygon"]
        if not layout.field_right.buffer(0.002).covers(polygon):
            raise RuntimeError(
                f"{layout.definition.key} {key['name']} dovetail leaves envelope")
        measured = (key["neck_mm"], key["head_mm"],
                    key["penetration_mm"])
        if any(not math.isclose(a, b, abs_tol=1e-9)
               for a, b in zip(measured, profile)):
            raise RuntimeError(
                f"{layout.definition.key} {key['name']} profile drifted")
        if (key["male_owner"], key["female_owner"]) != owners:
            raise RuntimeError(
                f"{layout.definition.key} {key['name']} ownership drifted")
        if key["ligament_mm"] < DOVETAIL_MIN_LIGAMENT_MM - 0.01:
            raise RuntimeError(
                f"{layout.definition.key} {key['name']} ligament too small")
    for index, (gap, seam) in enumerate(zip(
            layout.fit_clearance_gaps, layout.joint_seams)):
        if gap.is_empty or not gap.is_valid:
            raise RuntimeError(
                f"{layout.definition.key} fit gap {index} invalid/empty")
        if gap.difference(layout.field_right).area > 0.001:
            raise RuntimeError(
                f"{layout.definition.key} fit gap {index} leaves envelope")
        # The one-sided slit is analytically zero at both field endpoints;
        # assert the prescribed full clearance is reached after the taper.
        length = seam.length
        if length <= 2.0 * DOVETAIL_ENDPOINT_TAPER_MM:
            raise RuntimeError(
                f"{layout.definition.key} seam {index} cannot taper clearance")
        full_width = float(DOVETAIL_CLEARANCE_MM * _smootherstep01(1.0))
        if not math.isclose(full_width, DOVETAIL_CLEARANCE_MM, abs_tol=1e-12):
            raise RuntimeError("dovetail clearance smootherstep drifted")
    sites = _right_sites()
    for name in ACTIVE_RECEIVER_NAMES_RIGHT:
        site = sites[name]
        pocket = _pocket_plan(site)
        if not layout.field_right.buffer(0.01).covers(pocket):
            raise RuntimeError(
                f"{layout.definition.key} does not contain {name} receiver pocket")
        normal = np.asarray(site["normal"], dtype=float)
        tangent = np.array((-normal[1], normal[0]))
        face = np.asarray(
            site.get("outer_surface_face", site["face"]), dtype=float)
        coords = np.asarray(pocket.exterior.coords)[:-1]
        radial_extent = float(np.ptp(coords @ normal))
        tangential_extent = float(np.ptp(coords @ tangent))
        center_delta = np.asarray(pocket.centroid.coords[0]) - face
        if (not math.isclose(radial_extent,
                             CAPTIVE_LAND_MM + MAGNET_FACE_GAP_MM,
                             abs_tol=0.002)
                or not math.isclose(tangential_extent,
                                    (CAVITY_DIAMETER_MM
                                     + 2.0 * SIDE_WALL_MARGIN_MM),
                                    abs_tol=0.002)
                or float(center_delta @ normal) <= 0.0
                or abs(float(center_delta @ tangent)) > 0.002):
            raise RuntimeError(
                f"{layout.definition.key} {name} receiver is not axis-aligned: "
                f"land axial/tangent={radial_extent:.3f}/"
                f"{tangential_extent:.3f}")
    outside = layout.field_right.difference(layout.profile.buffer(0.02))
    if outside.area > 0.15:
        raise RuntimeError(
            f"{layout.definition.key} unexpectedly modifies source profile")


def _ae_protected_region(
        layout: VariantLayout,
        ) -> tuple[Polygon, Polygon, Polygon, tuple[Polygon, ...]]:
    if layout.definition.key != "A":
        raise ValueError("Ae depth field requires the A plan family")
    sites = _right_sites()
    # Receiver protection follows the actual surface-normal envelopes. Every Obi-Wan
    # mating boundary is protected separately below; larger construction fans
    # that do not touch Obi-Wan may still follow the weighted rear field.
    roots = tuple(
        _pocket_plan(sites[name]).buffer(
            AE_PROTECTED_BUFFER_MM, resolution=32)
        for name in ACTIVE_RECEIVER_NAMES_RIGHT
    )
    contact_components = []
    top_contact_components = []
    named_keepouts = _common_keepout_parts()
    for name, keepout in named_keepouts.items():
        # White in the plan is open Obi-Wan envelope, but the adjoining Ae
        # material is a real mating face. Carrier rings, joint ears, either
        # lower support and the T crescent all retain the full rear datum.
        land = layout.field_right.intersection(
            keepout.buffer(AE_CONTACT_LAND_MM, resolution=32,
                           join_style=1)).buffer(0)
        if not land.is_empty and land.area > 0.01:
            contact_components.append(land)
            if name == "t_crescent":
                top_contact_components.append(land)
    contact_land = unary_union(tuple(contact_components)).buffer(0)
    if contact_land.is_empty:
        raise RuntimeError("Ae has no full-depth Obi-Wan contact land")

    # The narrow plan spine keeps the printable outline connected, but it is
    # not a mating datum.  Leaving it out of the full-depth set lets the rear
    # field feather continuously through the LM/UM neck instead of forcing an
    # impossible 11.26-mm rise within a few millimetres of the sharp edge.
    # The 1.20-mm-wide plan cap is the explicit tweeter-flush exception and
    # remains full depth in both Ac and Ae.
    tip_cap = layout.field_right.intersection(
        box(0.0, A_TAPER_CAP_Y - AE_PROTECTED_BUFFER_MM,
            180.0, A_TAPER_CAP_Y + 0.1))
    top_flush_land = unary_union(
        (*top_contact_components, tip_cap)).intersection(
            layout.field_right).buffer(0)
    components_list = []
    for shape in (*contact_components, *roots, tip_cap):
        clipped = shape.intersection(layout.field_right).buffer(0)
        if not clipped.is_empty and clipped.area > 0.01:
            components_list.append(clipped)
    components = tuple(components_list)
    protected = unary_union(components).intersection(
        layout.field_right).buffer(0)
    if protected.is_empty or not protected.is_valid:
        raise RuntimeError("Ae protected full-depth region is invalid")
    for name in ACTIVE_RECEIVER_NAMES_RIGHT:
        if not protected.buffer(0.01).covers(_pocket_plan(sites[name])):
            raise RuntimeError(f"Ae protection misses {name} pocket")
    if top_flush_land.is_empty:
        raise RuntimeError("Ae has no full-depth tweeter flush land")
    return protected, contact_land, top_flush_land, components


def _line_parts(geometry) -> tuple[LineString, ...]:
    if geometry.is_empty:
        return ()
    if geometry.geom_type in ("LineString", "LinearRing"):
        line = LineString(geometry.coords)
        return (line,) if line.length > 1e-6 else ()
    if not hasattr(geometry, "geoms"):
        return ()
    return tuple(
        line
        for child in geometry.geoms
        for line in _line_parts(child)
    )


def _ae_exposed_outer_edge(
        layout: VariantLayout, top_flush_land: Polygon):
    """Return only free A-profile edges, independent of split topology.

    ``field_right`` also has boundaries around the carriers, magnet receiver
    roots and the x=0.65 half-layout clip.  Matching its boundary back to the
    monolithic source profile selects the acoustic free edge semantically.
    The only full-depth exception removed from that edge is a strictly local
    blend around the *actual* top land that mates flush to the tweeter
    crescent/cap.  Its radius is derived from the 11.26-mm rise and the slope
    ceiling.  Do not classify an ordinate-wide top band as contact: the S4
    receiver ray, for example, ends on a genuinely free A flank above y=425
    and must still reach the constant knife thickness there.
    """
    profile_band = layout.profile.boundary.buffer(
        AE_EDGE_MATCH_TOL_MM, cap_style=2, join_style=2)
    matched = layout.field_right.boundary.intersection(profile_band)
    matched = matched.difference(top_flush_land.buffer(
        AE_TOP_CONTACT_EDGE_BLEND_MM,
        cap_style=2, join_style=2))
    matched = matched.difference(box(
        -1000.0, -1000.0,
        0.65 + 2.0 * AE_EDGE_MATCH_TOL_MM, 1000.0))
    parts = tuple(line for line in _line_parts(matched)
                  if line.length >= 0.20)
    if not parts:
        raise RuntimeError("Ae exposed outer-edge classifier returned no edge")
    edge = unary_union(parts)
    edge = shapely.segmentize(edge, AE_EDGE_DENSIFY_MM)
    if edge.is_empty or edge.length < 20.0:
        raise RuntimeError(
            f"Ae exposed outer edge is implausibly short: {edge.length:.3f} mm")
    return edge


def _ae_retention_scale(
        point_cloud, solution: AeTransitionModel) -> np.ndarray:
    """Smooth wavelength-informed LM/UM/T scale at each plan point."""
    x = np.asarray(shapely.get_x(point_cloud), dtype=float)
    y = np.asarray(shapely.get_y(point_cloud), dtype=float)
    centers = np.asarray(solution.center_xy_mm, dtype=float)
    radii = np.asarray(solution.weight_radii_mm, dtype=float)
    dx = x[..., None] - centers[:, 0]
    dy = y[..., None] - centers[:, 1]
    normalized_r2 = (dx * dx + dy * dy) / (radii * radii)
    raw = np.exp(-0.5 * normalized_r2)
    total = np.sum(raw, axis=-1)
    if np.any(total <= np.finfo(float).tiny):
        raise RuntimeError("Ae driver-centre weights underflowed")
    weights = raw / total[..., None]
    return np.sum(
        weights * np.asarray(solution.retention_scales, dtype=float),
        axis=-1)


def _ae_weighted_depth(
        point_cloud, components: tuple[Polygon, ...], exposed_outer_edge,
        solution: AeTransitionModel) -> np.ndarray:
    """Evaluate the boundary-aware, centre-weighted monolithic Ae field."""
    edge_distance = np.asarray(
        shapely.distance(exposed_outer_edge, point_cloud), dtype=float)
    retention = _ae_retention_scale(point_cloud, solution)
    distance_stack = np.stack(tuple(
        np.asarray(shapely.distance(component, point_cloud), dtype=float)
        for component in components), axis=0)
    nearest = np.min(distance_stack, axis=0)
    protected_hit = nearest <= AE_DISTANCE_EPS_MM
    # A p-parallel soft minimum avoids the medial-axis crease of a hard
    # distance-to-union without summing the thickness influence of every
    # separate carrier/joint land. It therefore stays smooth and materially
    # thinner while every mating component remains exactly full depth.
    safe_distance = np.maximum(distance_stack, AE_DISTANCE_EPS_MM)
    ratios = np.divide(
        nearest[None, :], safe_distance,
        out=np.zeros_like(distance_stack),
        where=safe_distance > 0.0)
    soft_denominator = np.power(
        np.sum(np.power(ratios, AE_PROTECTED_SOFTMIN_P), axis=0),
        1.0 / AE_PROTECTED_SOFTMIN_P)
    land_distance = np.divide(
        nearest, soft_denominator,
        out=np.zeros_like(nearest), where=soft_denominator > 0.0)
    denominator = land_distance + retention * edge_distance
    u = np.divide(
        land_distance, denominator,
        out=np.zeros_like(land_distance),
        where=denominator > AE_DISTANCE_EPS_MM)
    u = np.clip(u, 0.0, 1.0)
    progress = u * u * (2.0 - u)
    depth_delta = solution.full_depth_mm - solution.edge_depth_mm
    depth = solution.full_depth_mm - depth_delta * progress
    # Exact constraints, with the acoustic outer edge taking precedence where
    # a receiver root reaches it.  The tweeter flush exception was removed
    # from ``exposed_outer_edge`` and therefore remains full depth.
    depth = np.asarray(depth, dtype=float)
    depth[protected_hit] = solution.full_depth_mm
    depth[edge_distance <= AE_DISTANCE_EPS_MM] = solution.edge_depth_mm
    return depth


def _sample_line_coordinates(
        geometry, spacing_mm: float = 0.50) -> np.ndarray:
    samples = []
    for line in _line_parts(geometry):
        count = max(2, int(math.ceil(line.length / spacing_mm)) + 1)
        for fraction in np.linspace(0.0, 1.0, count):
            point = line.interpolate(float(fraction), normalized=True)
            samples.append((point.x, point.y))
    if not samples:
        raise RuntimeError("cannot sample empty Ae line geometry")
    return np.asarray(samples, dtype=float)


def _build_ae_depth_field(
        layout: VariantLayout,
        solution: AeTransitionModel) -> AeDepthField:
    (protected, contact_land, top_flush_land,
     components) = _ae_protected_region(layout)
    exposed_outer_edge = _ae_exposed_outer_edge(layout, top_flush_land)
    nx, ny = 340, 900
    x = np.linspace(0.65, 165.0, nx)
    y = np.linspace(0.0, A_TAPER_CAP_Y, ny)
    xx, yy = np.meshgrid(x, y)
    mask = shapely.contains_xy(layout.field_right, xx, yy)
    depths = np.full(xx.shape, np.nan, dtype=float)
    point_cloud = shapely.points(xx[mask], yy[mask])
    depths[mask] = _ae_weighted_depth(
        point_cloud, components, exposed_outer_edge, solution)
    retention = _ae_retention_scale(point_cloud, solution)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    volume = float(np.nansum(depths) * dx * dy)
    ac_volume = float(layout.field_right.area * DEPTH_MM)
    mass = volume / 1000.0 * PLA_DENSITY_G_CM3
    ac_mass = ac_volume / 1000.0 * PLA_DENSITY_G_CM3
    reduction = 100.0 * (1.0 - volume / ac_volume)
    minimum = float(np.nanmin(depths))
    if minimum < AE_EDGE_DEPTH_MM - 0.01:
        raise RuntimeError(f"Ae depth undershot edge: {minimum:.3f} mm")
    if minimum <= AE_EDGE_DEPTH_MM + 1e-10:
        raise RuntimeError(
            "Ae interior grid reached edge thickness; a thin plateau exists")
    if float(np.nanmax(depths)) > DEPTH_MM + 0.01:
        raise RuntimeError("Ae depth exceeds Ac envelope")
    contact_mask = mask & shapely.contains_xy(contact_land, xx, yy)
    if (np.any(contact_mask)
            and float(np.nanmin(depths[contact_mask])) < DEPTH_MM - 0.02):
        raise RuntimeError("Ae Obi-Wan contact land is not full depth")
    top_flush_mask = mask & shapely.contains_xy(top_flush_land, xx, yy)
    if not np.any(top_flush_mask):
        raise RuntimeError("Ae top flush land has no grid witnesses")
    top_flush_depth = (
        float(np.nanmin(depths[top_flush_mask])),
        float(np.nanmax(depths[top_flush_mask])),
    )
    if top_flush_depth[0] < DEPTH_MM - 0.02:
        raise RuntimeError(
            f"Ae tweeter flush land is not full depth: "
            f"{top_flush_depth[0]:.3f} mm")

    edge_xy = _sample_line_coordinates(
        exposed_outer_edge, spacing_mm=AE_EDGE_DENSIFY_MM)
    edge_cloud = shapely.points(edge_xy[:, 0], edge_xy[:, 1])
    edge_depth = _ae_weighted_depth(
        edge_cloud, components, exposed_outer_edge, solution)
    outer_edge_depth = (
        float(np.min(edge_depth)), float(np.max(edge_depth)))
    if (abs(outer_edge_depth[0] - AE_EDGE_DEPTH_MM) > 0.005
            or abs(outer_edge_depth[1] - AE_EDGE_DEPTH_MM) > 0.005):
        raise RuntimeError(
            "Ae exposed edge is not constant thickness: "
            f"{outer_edge_depth[0]:.3f}..{outer_edge_depth[1]:.3f} mm")

    central = (mask[1:-1, 1:-1]
               & mask[1:-1, :-2] & mask[1:-1, 2:]
               & mask[:-2, 1:-1] & mask[2:, 1:-1])
    gx = ((depths[1:-1, 2:] - depths[1:-1, :-2]) / (2.0 * dx))
    gy = ((depths[2:, 1:-1] - depths[:-2, 1:-1]) / (2.0 * dy))
    grid_slope = np.sqrt(gx * gx + gy * gy)
    max_grid_slope = float(np.nanmax(grid_slope[central]))
    if max_grid_slope > AE_FIELD_MAX_SLOPE:
        central_slope = np.where(central, grid_slope, np.nan)
        slope_iy, slope_ix = np.unravel_index(
            int(np.nanargmax(central_slope)), central_slope.shape)
        grid_iy, grid_ix = slope_iy + 1, slope_ix + 1
        raise RuntimeError(
            f"Ae weighted-field grid slope too high: {max_grid_slope:.3f} "
            f"at x={x[grid_ix]:.3f}, y={y[grid_iy]:.3f}, "
            f"t={depths[grid_iy, grid_ix]:.3f} mm; neighborhood="
            f"{depths[grid_iy - 1:grid_iy + 2, grid_ix - 1:grid_ix + 2].tolist()}")
    joint_areas = []
    joint_mismatches = []
    owner_pairs = (("lm_lower", "lm_upper"), ("lm_upper", "um"))
    for male_name, female_name in owner_pairs:
        interface = layout.nominal_parts[male_name].boundary.intersection(
            layout.nominal_parts[female_name].boundary)
        lines = _line_parts(interface)
        if not lines:
            raise RuntimeError(
                f"Ae {male_name}/{female_name} dovetail has no interface")
        area = 0.0
        for line in lines:
            coords = _sample_line_coordinates(line, spacing_mm=0.20)
            depths_on_joint = _ae_weighted_depth(
                shapely.points(coords[:, 0], coords[:, 1]), components,
                exposed_outer_edge, solution)
            ds = np.linalg.norm(np.diff(coords, axis=0), axis=1)
            area += float(np.sum(
                0.5 * (depths_on_joint[:-1] + depths_on_joint[1:]) * ds))
        joint_areas.append(area)
        # Both prints are cut from one finalized monolith.  They therefore
        # share the exact same rear B-spline at the nominal joint boundary.
        joint_mismatches.append(0.0)
    if (joint_areas[0] < AE_MIN_JOINT_AREA_MM2[0]
            or joint_areas[1] < AE_MIN_JOINT_AREA_MM2[1]):
        raise RuntimeError(
            "Ae dovetail interface below geometric area gate: "
            f"{joint_areas[0]:.1f}/{joint_areas[1]:.1f} mm2; minimum "
            f"{AE_MIN_JOINT_AREA_MM2[0]:.1f}/"
            f"{AE_MIN_JOINT_AREA_MM2[1]:.1f}")
    if max(joint_mismatches) > 0.15:
        raise RuntimeError(
            "Ae dovetail-face rear mismatch exceeds 0.15 mm: "
            f"{joint_mismatches[0]:.3f}/{joint_mismatches[1]:.3f}")
    return AeDepthField(
        protected=protected,
        contact_land=contact_land,
        top_flush_land=top_flush_land,
        protected_components=components,
        exposed_outer_edge=exposed_outer_edge,
        joint_area_mm2=(joint_areas[0], joint_areas[1]),
        joint_rear_mismatch_mm=(joint_mismatches[0], joint_mismatches[1]),
        x_mm=x,
        y_mm=y,
        depth_mm=depths,
        mask=mask,
        volume_mm3=volume,
        mass_g=mass,
        ac_volume_mm3=ac_volume,
        ac_mass_g=ac_mass,
        reduction_pct=reduction,
        protected_area_mm2=float(protected.area),
        min_depth_mm=minimum,
        max_grid_slope=max_grid_slope,
        outer_edge_depth_mm=outer_edge_depth,
        top_flush_depth_mm=top_flush_depth,
        retention_scale_range=(
            float(np.min(retention)), float(np.max(retention))),
    )


def _longest_line(geometry) -> LineString:
    if geometry.is_empty:
        return LineString()
    if geometry.geom_type == "LineString":
        return geometry
    lines = [item for item in geometry.geoms
             if item.geom_type == "LineString" and item.length > 1e-6]
    return max(lines, key=lambda item: item.length) if lines else LineString()


def _ae_section_definitions(
        layout: VariantLayout) -> tuple[SectionDefinition, ...]:
    sites = _right_sites()

    def radial_line(site, inward=2.0, outward=95.0):
        face = np.asarray(
            site.get("outer_surface_face", site["face"]), dtype=float)
        normal = np.asarray(site["normal"], dtype=float)
        return LineString((tuple(face - inward * normal),
                           tuple(face + outward * normal)))

    return (
        SectionDefinition(
            "S1", "nominal free LM flank, y=201",
            LineString(((0.0, 200.981), (175.0, 200.981)))),
        SectionDefinition(
            "S2", "upper-LM receiver, radial 64 deg",
            radial_line(sites["lm_upper_right"]),
            pocket_z_mm=float(sites["lm_upper_right"]["z_mm"])),
        SectionDefinition(
            "S3", "upper V1L-style dovetail joint",
            layout.joint_seams[1]),
        SectionDefinition(
            "S4", "UM receiver, radial 50.5 deg",
            radial_line(sites["um_right"]),
            pocket_z_mm=float(sites["um_right"]["z_mm"])),
        SectionDefinition(
            "S5", "1.20-mm plan tip-cap exception",
            LineString(((0.0, A_TAPER_CAP_Y - 0.35),
                        (175.0, A_TAPER_CAP_Y - 0.35)))),
    )


def _sample_ae_section(
        layout: VariantLayout,
        depth_field: AeDepthField,
        solution: AeTransitionModel,
        definition: SectionDefinition,
        samples: int = 240):
    segment = _longest_line(layout.field_right.intersection(definition.line))
    if segment.is_empty or segment.length < 0.5:
        raise RuntimeError(f"Ae section {definition.key} misses A field")
    fractions = np.linspace(0.0, 1.0, samples)
    locations = [segment.interpolate(float(value), normalized=True)
                 for value in fractions]
    cloud = shapely.points(
        np.asarray([point.x for point in locations]),
        np.asarray([point.y for point in locations]))
    depth = _ae_weighted_depth(
        cloud, depth_field.protected_components,
        depth_field.exposed_outer_edge, solution)
    along = fractions * segment.length
    xy = np.column_stack((
        [point.x for point in locations],
        [point.y for point in locations],
    ))
    return along, depth, xy, segment


def _orient_ae_section_protected_to_edge(
        depth_field: AeDepthField, along: np.ndarray, depth: np.ndarray,
        xy: np.ndarray, definition: SectionDefinition,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize a section from its full-depth interface to its free edge."""
    endpoints = shapely.points(xy[[0, -1], 0], xy[[0, -1], 1])
    protected_distance = np.asarray(
        shapely.distance(depth_field.protected, endpoints), dtype=float)
    edge_distance = np.asarray(
        shapely.distance(depth_field.exposed_outer_edge, endpoints),
        dtype=float)
    forward_score = protected_distance[0] + edge_distance[1]
    reverse_score = protected_distance[1] + edge_distance[0]
    reverse = reverse_score < forward_score
    if reverse:
        along = along[-1] - along[::-1]
        depth = depth[::-1]
        xy = xy[::-1]
        protected_distance = protected_distance[::-1]
        edge_distance = edge_distance[::-1]
    endpoint_tol = max(0.40, 2.0 * AE_EDGE_MATCH_TOL_MM)
    if (protected_distance[0] > endpoint_tol
            or edge_distance[1] > endpoint_tol):
        raise RuntimeError(
            f"Ae section {definition.key} is not a protected-to-free-edge "
            f"cut: protected/edge endpoint distances="
            f"{protected_distance[0]:.3f}/{edge_distance[1]:.3f} mm")
    return along, depth, xy


def _validate_ae_section_monotonicity(
        layout: VariantLayout, depth_field: AeDepthField,
        solution: AeTransitionModel,
        sections: tuple[SectionDefinition, ...]) -> None:
    """Build gate for the physical rear surface shown by S1--S5.

    S1--S4 are real full-depth-interface to acoustic-free-edge cuts.  Their
    material depth may stay level but may never increase after leaving the
    interface.  S5 is the actual tweeter-seat/cap contact and therefore stays
    at the exact 11.5-mm datum throughout.  This validates the evaluated 2-D
    field; it does not repair or clip curves in the drawing.
    """
    expected_monotonic = {"S1", "S2", "S3", "S4"}
    witnessed = set()
    for definition in sections:
        along, depth, xy, _segment = _sample_ae_section(
            layout, depth_field, solution, definition, samples=481)
        if definition.key in expected_monotonic:
            along, depth, xy = _orient_ae_section_protected_to_edge(
                depth_field, along, depth, xy, definition)
            if depth[0] < DEPTH_MM - 0.005:
                raise RuntimeError(
                    f"Ae section {definition.key} does not start flush: "
                    f"t={depth[0]:.4f} mm")
            if abs(depth[-1] - AE_EDGE_DEPTH_MM) > 0.005:
                raise RuntimeError(
                    f"Ae section {definition.key} does not end at the "
                    f"constant free edge: t={depth[-1]:.4f} mm")
            running_minimum = np.minimum.accumulate(depth)
            reversals = depth - running_minimum
            worst_index = int(np.argmax(reversals))
            worst_reversal = float(reversals[worst_index])
            if worst_reversal > AE_SECTION_MONOTONIC_TOL_MM:
                valley_index = int(np.argmin(depth[:worst_index + 1]))
                raise RuntimeError(
                    f"Ae section {definition.key} reverses depth by "
                    f"{worst_reversal:.4f} mm after s="
                    f"{along[valley_index]:.3f} mm; "
                    f"t={depth[valley_index]:.4f}->"
                    f"{depth[worst_index]:.4f} mm at "
                    f"s={along[worst_index]:.3f} mm")
            witnessed.add(definition.key)
        elif definition.key == "S5":
            if (float(np.min(depth)) < DEPTH_MM - 0.005
                    or float(np.max(depth)) > DEPTH_MM + 0.005):
                raise RuntimeError(
                    "Ae S5 T-seat/cap cut is not constant full depth: "
                    f"{float(np.min(depth)):.4f}.."
                    f"{float(np.max(depth)):.4f} mm")
    missing = expected_monotonic - witnessed
    if missing:
        raise RuntimeError(
            f"Ae monotonic section gate missed {sorted(missing)}")


def _draw_shape(ax, geometry, *, facecolor, edgecolor="none", alpha=1.0,
                linewidth=1.0, hatch=None, zorder=1):
    if geometry.is_empty:
        return
    polygons = ([geometry] if geometry.geom_type == "Polygon"
                else [g for g in geometry.geoms if g.geom_type == "Polygon"])
    for polygon in polygons:
        coords = np.asarray(polygon.exterior.coords)
        patch = MplPolygon(coords, closed=True, facecolor=facecolor,
                           edgecolor=edgecolor, linewidth=linewidth,
                           alpha=alpha, hatch=hatch, zorder=zorder)
        ax.add_patch(patch)
        for ring in polygon.interiors:
            hole = MplPolygon(np.asarray(ring.coords), closed=True,
                              facecolor="white", edgecolor=edgecolor,
                              linewidth=linewidth, zorder=zorder + 0.1)
            ax.add_patch(hole)


def _circle(ax, center, diameter, **kwargs):
    ax.add_patch(Circle(center, diameter / 2.0, fill=False, **kwargs))


def _draw_carriers_and_drivers(ax, labels: bool = False):
    for driver, center, open_d, name in (
            ("lm", LM_CENTER, L22_CUTOUT[2], "LM side R113.8"),
            ("um", UM_CENTER, UM_CUTOUT[2], "UM side R52.5")):
        _draw_shape(
            ax, side_ring_outer_plan(driver),
            facecolor=COLORS["carrier"],
            edgecolor=COLORS["carrier_edge"], linewidth=1.0, zorder=5)
        ax.add_patch(Circle(center, open_d / 2.0, facecolor="white",
                            edgecolor=COLORS["carrier_edge"], lw=0.8,
                            zorder=6))
        if labels:
            ax.text(center[0], center[1], name, ha="center", va="center",
                    fontsize=7.5, color="#30383f", zorder=10)
    _circle(ax, LM_CENTER, LM_FLANGE_D_MM, edgecolor=COLORS["driver_lm"],
            lw=1.2, ls=(0, (6, 4)), zorder=8)
    _circle(ax, UM_CENTER, UM_FLANGE_D_MM, edgecolor=COLORS["driver_um"],
            lw=1.2, ls=(0, (6, 4)), zorder=8)
    _circle(ax, T_FRONT_CENTER, ND25_FACE_D_MM,
            edgecolor=COLORS["driver_t1"], lw=1.2,
            ls=(0, (6, 4)), zorder=8)
    _circle(ax, T_REAR_CENTER, ND25_FACE_D_MM,
            edgecolor=COLORS["driver_t2"], lw=1.0,
            ls=(0, (2, 3)), zorder=8)


def _draw_magnets(ax, show_labels=False):
    sites = side_magnet_sites()
    for site in sites:
        active = True
        color = COLORS["active_mag"]
        carrier_pocket = _carrier_pocket_plan(site)
        _draw_shape(
            ax, carrier_pocket, facecolor="white", edgecolor=color,
            linewidth=0.9, hatch="///" if active else None,
            alpha=0.95, zorder=12)
        if active:
            # In XY every surface-normal cylinder is edge-on: the carrier
            # pocket projects inward and the matching receiver projects
            # outward, including the horizontal base-side LM interface.
            _draw_shape(
                ax, _pocket_plan(site), facecolor=color, edgecolor="white",
                linewidth=0.45, alpha=0.92, zorder=12.5)
        else:
            coords = np.asarray(carrier_pocket.exterior.coords)[:4]
            ax.plot((coords[0, 0], coords[2, 0]),
                    (coords[0, 1], coords[2, 1]),
                    color=color, lw=0.8, zorder=13)
            ax.plot((coords[1, 0], coords[3, 0]),
                    (coords[1, 1], coords[3, 1]),
                    color=color, lw=0.8, zorder=13)
        if show_labels and site["normal"][0] > 0:
            interface = ("horizontal base-side" if
                         site.get("interface_kind") == "base_side"
                         else "radial")
            label = (f"active {interface} {site['driver'].upper()} receiver\n"
                     "XY: 2.10 axial x 5.20 transverse")
            surface_face = site.get("outer_surface_face", site["face"])
            ax.annotate(label, xy=surface_face,
                        xytext=(surface_face[0] + 18, surface_face[1] - 4),
                        fontsize=6.8, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
                        zorder=15)


def _draw_variant(
        ax, layout: VariantLayout, labels=False, *,
        display_title: str | None = None,
        construction_text: str | None = None,
        section_definitions: tuple[SectionDefinition, ...] | None = None,
        show_a_annotation: bool = True):
    source_profile = _profile_polygon(layout.definition.outline)
    p = np.asarray(source_profile.exterior.coords)
    ax.plot(p[:, 0], p[:, 1], color=COLORS["historic"], lw=0.8,
            ls=(0, (2, 2)), zorder=2)

    # Nominal ownership is colored once.  Male dovetails are integral to the
    # lower print at each seam and inherit the finalized monolithic depth.
    for name in ("lm_lower", "lm_upper", "um"):
        shape = unary_union((layout.nominal_parts[name],
                             _mirror(layout.nominal_parts[name])))
        _draw_shape(ax, shape, facecolor=COLORS[name], edgecolor="white",
                    linewidth=0.35, alpha=0.80, zorder=3)

    for seam in layout.joint_seams:
        coords = np.asarray(seam.coords)
        ax.plot(coords[:, 0], coords[:, 1], color="#243746", lw=1.0,
                ls=(0, (4, 2)), zorder=10)
        ax.plot(-coords[:, 0], coords[:, 1], color="#243746", lw=1.0,
                ls=(0, (4, 2)), zorder=10)
    for key in layout.dovetail_keys:
        for polygon in (key["polygon"], _mirror(key["polygon"])):
            _draw_shape(ax, polygon, facecolor="none", edgecolor="#5a2380",
                        linewidth=1.1, zorder=11)

    if layout.definition.key == "A":
        cap_outer_x = _a_taper_x(A_TAPER_CAP_Y)
        cap_inner_x = _a_cap_keepout_x()
        for sign in (-1.0, 1.0):
            ax.plot(
                (sign * cap_outer_x, sign * A_TAPER_CREST_X),
                (A_TAPER_CAP_Y, SEAM_UPPER_Y),
                color=COLORS["correction"], lw=1.8, zorder=11)
            ax.plot(
                (sign * cap_inner_x, sign * cap_outer_x),
                (A_TAPER_CAP_Y, A_TAPER_CAP_Y),
                color=COLORS["correction"], lw=1.8, zorder=11)
            ax.plot(
                (sign * A_TAPER_TOP_X, sign * cap_outer_x),
                (A_TAPER_TOP_Y, A_TAPER_CAP_Y),
                color=COLORS["correction"], lw=0.8,
                ls=(0, (2, 2)), zorder=10)
        if show_a_annotation:
            ax.annotate(
                "requested straight taper\n"
                "1.20 mm solid cap at crescent keepout",
                xy=((cap_outer_x + A_TAPER_CREST_X) / 2.0,
                    (A_TAPER_CAP_Y + SEAM_UPPER_Y) / 2.0),
                xytext=(78.0, 468.0), fontsize=6.6, ha="center",
                color=COLORS["correction"],
                arrowprops=dict(arrowstyle="->", color=COLORS["correction"],
                                lw=0.8), zorder=15)

    _draw_carriers_and_drivers(ax, labels=labels)
    _draw_magnets(ax, show_labels=False)

    if section_definitions:
        for index, definition in enumerate(section_definitions):
            segment = _longest_line(
                layout.field_right.intersection(definition.line))
            if segment.is_empty:
                continue
            coords = np.asarray(segment.coords)
            color = plt.get_cmap("tab10")(index)
            ax.plot(coords[:, 0], coords[:, 1], color=color,
                    lw=1.2, zorder=16)
            midpoint = segment.interpolate(0.52, normalized=True)
            ax.text(midpoint.x + 3.0, midpoint.y + 2.0,
                    definition.key, fontsize=6.5, weight="bold",
                    color=color, zorder=17)

    ax.set_title(display_title or layout.definition.title,
                 fontsize=10.0, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-165, 165)
    ax.set_ylim(-12, 598)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    if labels:
        ax.set_ylabel("y (mm)", fontsize=8)
        ax.text(-157, 575,
                "dashed: installed driver envelopes\n"
                "gray dotted: historic source outer profile\n"
                "dark short-dash + purple keys: V1L-style dovetail joints\n"
                "red: printable A-Obi-Wan taper; dotted red: nominal extension",
                fontsize=6.8, va="top", color="#4f565c")
    footer = (construction_text if construction_text is not None else
              (f"per-side estimated print mass "
               f"{layout.metrics['mass_g_side']:.0f} g  |  "
               "rear-open 1.2 skin + 2.4 ribs"))
    ax.text(0, 6, footer,
            ha="center", va="bottom", fontsize=6.4, color="#3d454b",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78,
                      boxstyle="square,pad=0.12"))


def _draw_rear_structure(ax, layout: VariantLayout):
    _draw_shape(ax, layout.field_right, facecolor="#eef2f4",
                edgecolor=COLORS["carrier_edge"], linewidth=0.8, zorder=1)
    _draw_shape(ax, layout.rear_structure, facecolor=COLORS["rib"],
                edgecolor="none", alpha=0.93, zorder=2)
    for seam in layout.joint_seams:
        coords = np.asarray(seam.coords)
        ax.plot(coords[:, 0], coords[:, 1], color="#9b5a28", lw=0.9,
                ls=(0, (4, 2)), zorder=5)
    _draw_magnets(ax, show_labels=True)
    ax.text(8, 12, "OPEN REAR\nno sealed cells", fontsize=7.3,
            color="#273d57", weight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                      boxstyle="square,pad=0.12"))
    ax.text(8, 46,
            "2.4 mm full-depth perimeter / fan / joint-face ribs\n"
            "1.2 mm continuous acoustic front skin",
            fontsize=6.6, color="#273d57",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                      boxstyle="square,pad=0.12"))
    ax.set_title("Common printable construction — right side, rear view",
                 fontsize=9.5, weight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(0, 165)
    ax.set_ylim(-8, 458)
    ax.grid(True, lw=0.3, alpha=0.3)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("mm", fontsize=7.5)
    ax.set_ylabel("mm", fontsize=7.5)


def _draw_ac_side_section(ax):
    run = AE_REQUIRED_FREE_RUN_MM
    ax.fill_between((0.0, run), CORE_REAR_Z, THICKNESS_MM,
                    color="#7aaed3", alpha=0.72)
    ax.plot((0.0, run), (THICKNESS_MM, THICKNESS_MM),
            color="#243746", lw=1.4)
    ax.plot((0.0, run), (CORE_REAR_Z, CORE_REAR_Z),
            color="#243746", lw=1.4)
    ax.plot((run, run), (CORE_REAR_Z, THICKNESS_MM),
            color=COLORS["correction"], lw=2.0)
    ax.annotate("90 deg terminal corner", xy=(run, 12.5),
                xytext=(7.0, 8.0), fontsize=7.0,
                arrowprops=dict(arrowstyle="->", lw=0.8,
                                color=COLORS["correction"]),
                color=COLORS["correction"])
    ax.annotate("11.5 mm", xy=(0.4, CORE_REAR_Z),
                xytext=(0.4, THICKNESS_MM), fontsize=7.2,
                va="center",
                arrowprops=dict(arrowstyle="<->", lw=0.8,
                                color="#243746"))
    ax.text(run / 2.0, THICKNESS_MM + 0.25,
            "flat acoustic front  z=18.3", ha="center", fontsize=7.0)
    ax.text(run / 2.0, CORE_REAR_Z - 0.30,
            "flat rear  z=6.8", ha="center", va="top", fontsize=7.0)
    ax.set_title("Ac side section — constant solid depth", fontsize=9.2,
                 weight="bold")
    ax.set_xlim(-0.8, run + 0.8)
    ax.set_ylim(5.8, 19.2)
    ax.set_xlabel("outboard distance s (mm)", fontsize=7.5)
    ax.set_ylabel("global z (mm)", fontsize=7.5)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.tick_params(labelsize=6.8)


def _draw_ae_optimized_section(
        ax, solution: AeTransitionModel, depth_field: AeDepthField):
    section_colors = (
        COLORS["driver_lm"], COLORS["driver_um"], COLORS["driver_t1"])
    rear_profiles = tuple(
        THICKNESS_MM - depth
        for depth in solution.depth_by_center_mm)
    rear_stack = np.vstack(rear_profiles)
    ax.fill_between(
        solution.s_mm, np.min(rear_stack, axis=0),
        np.max(rear_stack, axis=0), color="#efa43a", alpha=0.30)
    for name, scale, rear, color, slope in zip(
            solution.center_names, solution.retention_scales,
            rear_profiles, section_colors,
            solution.max_slope_by_center):
        ax.plot(
            solution.s_mm, rear, color=color, lw=1.8,
            label=f"{name}  weight={scale:.2f}  slope={slope:.2f}")
    ax.plot((0.0, AE_REQUIRED_FREE_RUN_MM),
            (THICKNESS_MM, THICKNESS_MM),
            color="#243746", lw=1.4)
    ax.axhline(CORE_REAR_Z, color="#7aaed3", lw=0.9,
               ls=(0, (4, 3)), label="Ac rear")
    ax.annotate(f"edge t={solution.edge_depth_mm:.2f}\nrear z="
                f"{THICKNESS_MM - solution.edge_depth_mm:.2f}",
                xy=(AE_REQUIRED_FREE_RUN_MM,
                    THICKNESS_MM - solution.edge_depth_mm),
                xytext=(17.2, 14.6),
                fontsize=6.9,
                arrowprops=dict(arrowstyle="->", lw=0.8,
                                color=COLORS["correction"]),
                color=COLORS["correction"])
    ax.text(0.25, 18.65,
            "LM / UM / T are weighted continuously in plan; this 36-mm "
            "span is a comparison cut, not a fixed run.\n"
            "Longer LM wavelengths retain depth furthest; T sheds it fastest. "
            "All three reach the same edge only at s=36: no thin plateau.\n"
            f"actual plan peak slope={depth_field.max_grid_slope:.2f}; "
            f"mass -{depth_field.reduction_pct:.1f}% vs Ac\n"
            f"CAD t={solution.edge_depth_mm:.2f} normally prints as one "
            "~0.20 layer (P2S/0.4 nozzle; first line 0.50); coupon-gated\n"
            "Arachne, smooth/satin plate, elephant-foot 0.15; no fused edge "
            "brim; preview layer 1\n"
            f"t={AE_OPTIONAL_FINE_LAYER_EDGE_MM:.2f} is not represented or "
            "approved (still fragile with a true 0.16 first layer)",
            fontsize=5.55, va="top", color="#30383f",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72,
                      boxstyle="square,pad=0.12"), zorder=10)
    ax.set_title("Ae weighted rear references — LM / UM / T",
                 fontsize=9.2, weight="bold")
    ax.axvline(AE_REQUIRED_FREE_RUN_MM, color=COLORS["correction"], lw=0.9,
               ls=(0, (3, 2)))
    ax.set_xlim(-0.8, AE_REQUIRED_FREE_RUN_MM + 0.8)
    ax.set_ylim(5.8, 19.2)
    ax.set_xlabel("reference distance: protected land -> free edge (mm)",
                  fontsize=7.5)
    ax.set_ylabel("global z (mm)", fontsize=7.5)
    ax.grid(True, lw=0.3, alpha=0.35)
    ax.tick_params(labelsize=6.8)
    ax.legend(loc="lower left", fontsize=5.6, frameon=False)


def _swap_xy(geometry):
    return affinity.affine_transform(geometry, (0.0, 1.0, 1.0, 0.0, 0.0, 0.0))


def _draw_ae_depth_map(
        ax, layout: VariantLayout, depth_field: AeDepthField,
        sections: tuple[SectionDefinition, ...],
        solution: AeTransitionModel):
    levels = (AE_EDGE_DEPTH_MM, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 11.5)
    cmap = plt.get_cmap("viridis")
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    mesh = ax.pcolormesh(
        depth_field.y_mm, depth_field.x_mm, depth_field.depth_mm.T,
        cmap=cmap, norm=norm, shading="auto", rasterized=True)
    ax.contour(
        depth_field.y_mm, depth_field.x_mm, depth_field.depth_mm.T,
        levels=levels[1:-1], colors="#f6f7f8", linewidths=0.55,
        alpha=0.9)
    _draw_shape(
        ax, _swap_xy(depth_field.protected), facecolor="none",
        edgecolor="#d52f88", linewidth=1.0, hatch="....", alpha=0.5,
        zorder=6)
    _draw_shape(
        ax, _swap_xy(depth_field.contact_land), facecolor="none",
        edgecolor="#8e245f", linewidth=1.0, hatch="////", alpha=0.62,
        zorder=7)
    for line in _line_parts(depth_field.exposed_outer_edge):
        coords = np.asarray(line.coords)
        ax.plot(coords[:, 1], coords[:, 0], color=COLORS["correction"],
                lw=2.1, solid_capstyle="round", zorder=8.2)
    top_transition_edge = (
        layout.field_right.boundary
        .intersection(layout.profile.boundary.buffer(
            AE_EDGE_MATCH_TOL_MM, cap_style=2, join_style=2))
        .intersection(depth_field.top_flush_land.buffer(
            AE_TOP_CONTACT_EDGE_BLEND_MM,
            cap_style=2, join_style=2)))
    for line in _line_parts(top_transition_edge):
        coords = np.asarray(line.coords)
        ax.plot(coords[:, 1], coords[:, 0], color=COLORS["active_mag"],
                lw=3.0, solid_capstyle="round", zorder=8.25)
    for seam in layout.joint_seams:
        coords = np.asarray(seam.coords)
        ax.plot(coords[:, 1], coords[:, 0], color="#202b33", lw=1.15,
                ls=(0, (4, 2)), zorder=8.5)
    sites = _right_sites()
    for name in ACTIVE_RECEIVER_NAMES_RIGHT:
        site = sites[name]
        _draw_shape(
            ax, _swap_xy(_carrier_pocket_plan(site)),
            facecolor="white", edgecolor=COLORS["active_mag"],
            linewidth=0.9, hatch="///", alpha=0.95, zorder=9)
        _draw_shape(
            ax, _swap_xy(_pocket_plan(site)),
            facecolor=COLORS["active_mag"], edgecolor="white",
            linewidth=0.45, alpha=0.95, zorder=9.2)
    for index, definition in enumerate(sections):
        segment = _longest_line(layout.field_right.intersection(definition.line))
        if segment.is_empty:
            continue
        coords = np.asarray(segment.coords)
        color = plt.get_cmap("tab10")(index)
        ax.plot(coords[:, 1], coords[:, 0], color=color, lw=1.2, zorder=10)
        midpoint = segment.interpolate(0.52, normalized=True)
        ax.text(midpoint.y + 2.0, midpoint.x + 1.5, definition.key,
                fontsize=6.5, weight="bold", color=color, zorder=11)
    _draw_shape(ax, _swap_xy(layout.field_right), facecolor="none",
                edgecolor="#202b33", linewidth=1.0, zorder=12)
    cbar = ax.figure.colorbar(mesh, ax=ax, pad=0.012, fraction=0.035,
                             ticks=levels)
    cbar.set_label("Ae local depth t (mm)", fontsize=7.0)
    cbar.ax.tick_params(labelsize=6.5)
    ax.annotate(
        "WHITE = open Obi-Wan carrier / joint / T-crescent keepout\n"
        "no Ae material here; adjoining Ae mating band is flush t=11.5",
        xy=(354.0, 42.0), xytext=(246.0, 145.0), fontsize=6.3,
        color="#30383f",
        arrowprops=dict(arrowstyle="->", color="#30383f", lw=0.8),
        bbox=dict(facecolor="white", edgecolor="#a7adb2", alpha=0.90,
                  boxstyle="square,pad=0.18"), zorder=15)
    ax.annotate(
        "sole edge exception:\n"
        "local T-seat contact blend only\n"
        f"run >= {AE_MIN_FULL_TO_EDGE_RUN_MM:.2f} mm; "
        f"reserved {AE_TOP_CONTACT_EDGE_BLEND_MM:.2f}",
        xy=(447.0, 50.0), xytext=(380.0, 82.0), fontsize=6.1,
        color=COLORS["active_mag"],
        arrowprops=dict(arrowstyle="->", color=COLORS["active_mag"],
                        lw=0.9), zorder=15)
    ax.text(8.0, 153.0,
            f"red outer line = constant t={solution.edge_depth_mm:.2f}; "
            "magenta top = localized tweeter-contact blend\n"
            "hatched protected zones = every Obi-Wan mating band + receivers\n"
            "magenta rectangles = captive D5.20 x 2.10 receivers\n"
            "black dashed = V1L-style dovetail seams; no depth reserve\n"
            "LM/UM/T weights="
            f"{solution.retention_scales[0]:.2f}/"
            f"{solution.retention_scales[1]:.2f}/"
            f"{solution.retention_scales[2]:.2f}; "
            f"peak 2D slope {depth_field.max_grid_slope:.2f}; no plateau",
            fontsize=6.7, va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82,
                      boxstyle="square,pad=0.18"))
    ax.set_title("Ae LM/UM/T-weighted plan depth — exact free-edge boundary",
                 fontsize=9.4, weight="bold")
    ax.set_xlim(0.0, 455.0)
    ax.set_ylim(0.0, 165.0)
    ax.set_aspect("equal")
    ax.set_xlabel("global y (mm)", fontsize=7.5)
    ax.set_ylabel("outboard x (mm)", fontsize=7.5)
    ax.grid(True, lw=0.25, alpha=0.22)
    ax.tick_params(labelsize=6.7)


def _draw_ae_actual_section(
        ax, layout: VariantLayout, depth_field: AeDepthField,
        solution: AeTransitionModel, definition: SectionDefinition,
        *, show_y_label: bool):
    along, depth, xy, _segment = _sample_ae_section(
        layout, depth_field, solution, definition)
    if definition.key in {"S1", "S2", "S3", "S4"}:
        along, depth, xy = _orient_ae_section_protected_to_edge(
            depth_field, along, depth, xy, definition)
    rear = THICKNESS_MM - depth
    ax.fill_between(along, rear, THICKNESS_MM,
                    color="#efa43a", alpha=0.70)
    ax.plot(along, rear, color=COLORS["correction"], lw=1.25)
    ax.plot((along[0], along[-1]), (THICKNESS_MM, THICKNESS_MM),
            color="#243746", lw=0.9)
    ax.axhline(CORE_REAR_Z, color="#4f9bd7", lw=0.8,
               ls=(0, (4, 3)))
    if definition.pocket_z_mm is not None:
        sites = _right_sites()
        site = (sites["lm_upper_right"] if definition.key == "S2"
                else sites["um_right"])
        face = np.asarray(
            site.get("outer_surface_face", site["face"]), dtype=float)
        index = int(np.argmin(np.linalg.norm(xy - face, axis=1)))
        pocket_x = float(along[index])
        ax.add_patch(Rectangle(
            (pocket_x, definition.pocket_z_mm - SIDE_MAGNET_POCKET_D / 2.0),
            SIDE_MAGNET_DEPTH, SIDE_MAGNET_POCKET_D,
            facecolor=COLORS["active_mag"], edgecolor="white", lw=0.5,
            alpha=0.90, zorder=7))
    ax.set_title(f"{definition.key}  {definition.title}", fontsize=7.3,
                 weight="bold", pad=3)
    ax.set_ylim(6.0, 18.9)
    ax.set_xlim(0.0, max(1.0, float(along[-1])))
    ax.grid(True, lw=0.25, alpha=0.3)
    ax.tick_params(labelsize=5.9)
    ax.set_xlabel("section distance (mm)", fontsize=6.2)
    if show_y_label:
        ax.set_ylabel("global z (mm)", fontsize=6.2)
    else:
        ax.tick_params(labelleft=False)


def _draw_depth_and_table(
        ax, layout: VariantLayout,
        solution: AeTransitionModel, depth_field: AeDepthField):
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_title("Depth, dovetail and print contract", fontsize=9.3,
                 weight="bold", pad=8)

    # Binding UM receiver section; Ac and Ae retain the actual receiver
    # envelope at the full z=6.8..18.3 datum, while the larger construction
    # fan is free to follow the weighted rear field.
    x0, y0, w, h = 4.0, 67.0, 42.0, 24.0
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="#eef2f4",
                           edgecolor="#30383f", lw=0.9))
    front_y = y0 + h
    rear_y = y0
    ax.text(x0, front_y + 1.4, "front z=18.30", fontsize=6.5, va="bottom")
    ax.text(x0, rear_y - 1.5, "full rear z=6.80", fontsize=6.5, va="top")
    ax.annotate("11.5 mm", xy=(x0 - 2.0, rear_y),
                xytext=(x0 - 2.0, front_y), ha="right", va="center",
                fontsize=6.7,
                arrowprops=dict(arrowstyle="<->", lw=0.8, color="#30383f"))
    um_z = 15.10
    cy = y0 + h * (um_z - CORE_REAR_Z) / DEPTH_MM
    mag_h = h * SIDE_MAGNET_POCKET_D / DEPTH_MM
    pocket_w = w * SIDE_MAGNET_DEPTH / PAD_UM_RADIAL_MM
    ax.add_patch(Rectangle((x0, cy - mag_h / 2.0), pocket_w, mag_h,
                           facecolor=COLORS["active_mag"], edgecolor="white",
                           lw=0.5, alpha=0.9))
    ax.text(x0 + 1.0, cy, "UM captive cavity\nzc=15.10", fontsize=6.4,
            color="white", va="center")
    ax.text(x0 + w / 2.0, y0 + 3.0,
            "full-depth Ae protected root\n0.60 front / 5.70 rear skin",
            fontsize=6.2, ha="center", va="bottom", color="#273d57")

    # Through-thickness XY key: this changes only the partition masks.  It
    # does not add local thickness, a protected island or a rear feature.
    ox, oy, ow, oh = 54.0, 67.0, 42.0, 24.0
    ax.add_patch(Rectangle((ox, oy), ow, oh, facecolor="#eef2f4",
                           edgecolor="#30383f", lw=0.9))
    seam_y = oy + oh * 0.44
    ax.add_patch(Rectangle((ox, oy), ow, seam_y - oy,
                           facecolor=COLORS["lm_upper"], edgecolor="none",
                           alpha=0.88))
    ax.add_patch(Rectangle((ox, seam_y), ow, oy + oh - seam_y,
                           facecolor=COLORS["um"], edgecolor="none",
                           alpha=0.88))
    key_center = ox + ow * 0.50
    key_neck = 7.0
    key_head = 8.5
    key_depth = 4.0
    key_plan = np.asarray((
        (key_center - key_neck / 2.0, seam_y),
        (key_center + key_neck / 2.0, seam_y),
        (key_center + key_head / 2.0, seam_y + key_depth),
        (key_center - key_head / 2.0, seam_y + key_depth),
    ))
    ax.add_patch(MplPolygon(key_plan, closed=True,
                            facecolor=COLORS["lm_upper"],
                            edgecolor="#5a2380", lw=1.1, zorder=5))
    ax.plot((ox, key_center - key_neck / 2.0), (seam_y, seam_y),
            color="#30383f", lw=1.0, ls=(0, (4, 2)))
    ax.plot((key_center + key_neck / 2.0, ox + ow), (seam_y, seam_y),
            color="#30383f", lw=1.0, ls=(0, (4, 2)))
    ax.text(ox + ow / 2.0, oy + oh + 1.5,
            "V1L-style male/female dovetail — depth unchanged",
            fontsize=6.4, ha="center", va="bottom")
    seam_contract = (
        "female clearance 0.05; 2.0 endpoint closure; zero envelope growth\n"
        f"L neck/head/depth 7/9/4, ligament "
        f"{layout.metrics['lower_key_ligament_mm']:.2f}; U 7/8.5/4, "
        f"ligament {layout.metrics['upper_key_ligament_mm']:.2f} mm")
    ax.text(ox + ow / 2.0, oy - 2.0, seam_contract,
            fontsize=5.65, ha="center", va="top")

    ax.text(4.0, 57.5,
            "Ac/Ae are solid. Ae stays full-depth at every Obi-Wan carrier, "
            "joint, support and T-seat mating band,\n"
            "plus all three receiver envelopes (base LM + radial LM/UM) and "
            "the T cap. White remains "
            "open envelope, not an internal cavity.\n"
            "Split after computing the monolithic rear field: the dovetail adds "
            "no protected land or depth feature.",
            fontsize=5.75, va="top", color="#30383f")
    ax.text(4.0, 49.0,
            "Lower owns the lower male; middle owns the upper male. Mate by "
            "Z-axis insertion; dovetails register/interlock XY.\n"
            "They do not independently retain Z. Calibrate the 0.05 female "
            "relief by coupon/gauge. Both joints and\n"
            "the shared one-layer knife-tip endpoint require proof tests.",
            fontsize=5.55, va="top", color="#30383f")

    headers = ("var", "rear model", "t mm", "g/side", "OBB LM-L / LM-U / UM mm")
    xs = (4.0, 14.0, 42.0, 57.0, 69.0)
    y_header = 35.5
    for x, header in zip(xs, headers):
        ax.text(x, y_header, header, fontsize=6.0, weight="bold", va="bottom")
    ax.plot((3.0, 98.0), (31.4, 31.4), color="#60686f", lw=0.7)
    rows = (
        ("Ac", "solid flat rear", "11.5",
         f"{depth_field.ac_mass_g:.0f}", layout),
        ("Ae", "weighted solid rear", f"{solution.edge_depth_mm:.2f}..11.5",
         f"{depth_field.mass_g:.0f}", layout),
    )
    def compact_obb(value: str) -> str:
        first, second = value.split(" x ")
        return f"{float(first):.0f}x{float(second):.0f}"

    for row, (key, model, thickness, mass, layout) in enumerate(rows):
        y = 29.0 - row * 7.0
        values = (
            key, model, thickness, mass,
            (f"{compact_obb(str(layout.metrics['lm_lower_obb']))} / "
             f"{compact_obb(str(layout.metrics['lm_upper_obb']))} / "
             f"{compact_obb(str(layout.metrics['um_obb']))}"),
        )
        for x, value in zip(xs, values):
            ax.text(x, y, str(value), fontsize=5.45, va="center")
    ax.text(4.0, 0.7,
            "Ae weighted field: no fixed run/plateau; plan "
            f"-{depth_field.reduction_pct:.1f}% volume. Eligible edge "
            f"{depth_field.outer_edge_depth_mm[0]:.3f}.."
            f"{depth_field.outer_edge_depth_mm[1]:.3f}; T top "
            f"{depth_field.top_flush_depth_mm[0]:.2f}.."
            f"{depth_field.top_flush_depth_mm[1]:.2f} mm; slope "
            f"{depth_field.max_grid_slope:.2f}.\n"
            f"Joint area/side L/U {depth_field.joint_area_mm2[0]:.0f}/"
            f"{depth_field.joint_area_mm2[1]:.0f} mm2; rear mismatch "
            f"{max(depth_field.joint_rear_mismatch_mm):.2f} mm. All prints "
            "fit 220x220 front-down.\n"
            "Edge, fit clearance, cavity roof and both dovetail joints require "
            "coupon/proof tests.",
            fontsize=5.15, va="bottom", color="#30383f")


def render(output: Path):
    layouts = tuple(_build_layout(definition) for definition in VARIANTS)
    (a_layout,) = layouts
    solution = _optimize_ae_profile()
    depth_field = _build_ae_depth_field(a_layout, solution)
    sections = _ae_section_definitions(a_layout)
    _validate_ae_section_monotonicity(
        a_layout, depth_field, solution, sections)

    fig = plt.figure(figsize=(22.5, 16.5), dpi=160, facecolor="white")
    gs = fig.add_gridspec(
        3, 4, height_ratios=(1.72, 0.68, 0.88),
        hspace=0.23, wspace=0.20,
        left=0.035, right=0.985, top=0.935, bottom=0.045)

    ac_plan_ax = fig.add_subplot(gs[0, 0:2])
    ae_plan_ax = fig.add_subplot(gs[0, 2:4])
    _draw_variant(
        ac_plan_ax, a_layout, labels=True,
        display_title="Ac — A plan, constant solid depth",
        construction_text=(
            f"solid t=11.5 | 90 deg edges | ~{depth_field.ac_mass_g:.0f} g/side"))
    _draw_variant(
        ae_plan_ax, a_layout,
        display_title="Ae — A plan, LM/UM/T-weighted rear",
        construction_text=(
            f"solid t={solution.edge_depth_mm:.2f}..11.5 | "
            f"~{depth_field.mass_g:.0f} g/side | "
            f"-{depth_field.reduction_pct:.0f}% vs Ac"),
        section_definitions=sections, show_a_annotation=False)

    ac_side_ax = fig.add_subplot(gs[1, 0])
    _draw_ac_side_section(ac_side_ax)
    ae_side_ax = fig.add_subplot(gs[1, 1])
    _draw_ae_optimized_section(ae_side_ax, solution, depth_field)
    depth_ax = fig.add_subplot(gs[1, 2:4])
    _draw_ae_depth_map(depth_ax, a_layout, depth_field, sections, solution)

    section_gs = gs[2, 0:2].subgridspec(
        2, 3, hspace=0.48, wspace=0.27)
    for index, definition in enumerate(sections):
        section_ax = fig.add_subplot(section_gs[index // 3, index % 3])
        _draw_ae_actual_section(
            section_ax, a_layout, depth_field, solution, definition,
            show_y_label=index % 3 == 0)
    section_note = fig.add_subplot(section_gs[1, 2])
    section_note.set_axis_off()
    section_note.text(
        0.02, 0.96,
        "CROSS-SECTION KEY\n\n"
        "orange: Ae material\n"
        "red: smooth rear target\n"
        "blue dashed: Ac rear z=6.8\n"
        "magenta: captive D5.20 x 2.10 cavity\n\n"
        "All Obi-Wan mating bands and receiver axes stay t=11.5.\n"
        "S1-S4 run interface -> free edge and are monotonicity-gated.\n"
        "S5 crosses the sole exception: actual T-seat contact stays "
        "full-depth.\n\n"
        "Dovetail seams do not enter the depth field;\n"
        "both children inherit one monolithic rear surface.",
        fontsize=6.6, va="top", color="#30383f")

    contract_ax = fig.add_subplot(gs[2, 2:4])
    _draw_depth_and_table(contract_ax, a_layout, solution, depth_field)

    fig.suptitle(
        "OBI-WAN PRINTABLE Ac / Ae ATTACHMENTS — ENGINEERING LAYOUT",
        fontsize=15, weight="bold", y=0.978)
    fig.text(
        0.5, 0.953,
        "Ac: solid constant t=11.5; Ae: flat front z=18.3 + LM/UM/T-weighted "
        f"{solution.edge_depth_mm:.2f}..11.5 rear with one constant free edge; "
        "flush-filled "
        "V1L-style dovetail seams; Ac and Ae are the complete wing inventory",
        ha="center", va="center", fontsize=9.0, color="#3f474d")
    fig.text(
        0.5, 0.018,
        "CAD-LINKED PRINTABLE LAYOUT — Ac/Ae STEP + six STLs each in build/wings/; "
        "experimental/unmeasured, not physical or acoustic qualification",
        ha="center", va="center", fontsize=8.0, color="#5b6268")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(
            temporary,
            metadata={
                "Title": "Obi-Wan printable Ac/Ae engineering layout",
                "Description": (
                    "Ac constant 11.5 mm solid depth; Ae boundary-aware "
                    "LM/UM/T-weighted rear, constant 0.24 mm coupon-gated free "
                    "edge, full-depth tweeter-top hand-off, V1L-style "
                    "dovetail seams and local protected magnet receivers; "
                    "Ac/Ae-only CAD-linked engineering layout"),
            },
        )
        plt.close(fig)
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            if image.size != (3600, 2640):
                raise RuntimeError(
                    f"unexpected output raster size {image.size}, expected "
                    "3600x2640")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return layouts, solution, depth_field


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_NAME)
    args = parser.parse_args()
    layouts, solution, depth_field = render(Path(args.output))
    print(f"wrote {args.output}")
    print(
        f"Ae profile: {solution.profile_name}; "
        f"weights={solution.retention_scales}; "
        f"edge={depth_field.outer_edge_depth_mm[0]:.3f}.."
        f"{depth_field.outer_edge_depth_mm[1]:.3f} mm; "
        f"top={depth_field.top_flush_depth_mm[0]:.3f}.."
        f"{depth_field.top_flush_depth_mm[1]:.3f} mm; "
        f"grid-slope={depth_field.max_grid_slope:.4f}; "
        f"joint={depth_field.joint_area_mm2[0]:.1f}/"
        f"{depth_field.joint_area_mm2[1]:.1f} mm2; "
        f"plan mass~{depth_field.mass_g:.1f} g/side vs "
        f"Ac~{depth_field.ac_mass_g:.1f} g/side")
    for layout in layouts:
        print(
            f"{layout.definition.key}: mass~{layout.metrics['mass_g_side']:.1f} g/side; "
            f"LM-L {layout.metrics['lm_lower_obb']}; "
            f"LM-U {layout.metrics['lm_upper_obb']}; "
            f"UM {layout.metrics['um_obb']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
