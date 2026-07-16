#!/usr/bin/env python3
"""Render deterministic, review-only V1LF acoustic-wing concept plates.

The generator is deliberately pure 2D: it imports neither build123d nor OCC.
It turns ``V1LF_ACOUSTIC_WINGS_SPEC.md`` into mechanically explicit concept
geometry.  Every side has three real receiver pads and one connected solid
load-path skeleton spanning the lower LM root, upper LM root, and UM root.
Acoustic apertures are actual Shapely footprints accepted only outside that
protected structure.  These PNGs are design comparators, not release CAD and
not acoustic qualification.

Ordinary use is through the repository's remote execution path::

    make wing_concepts
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, PathPatch, Polygon as MplPolygon, Rectangle
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageStat
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree


LM_CENTER = np.array((0.0, 200.981))
UM_CENTER = np.array((0.0, 366.081))
T_CENTER = np.array((0.0, 468.193))
LM_OUTER_R = 113.0
LM_OPEN_R = 95.0
UM_OUTER_R = 51.7
UM_OPEN_R = 41.0
T_OUTER_R = 52.2
FRONT_Z = 18.3
REAR_Z = 6.8
SEED = 521_406

BG = "#07131f"
PANEL = "#0d2030"
GRID = "#294052"
INK = "#f4f7fb"
MUTED = "#9eb1c1"
CARRIER = "#c7d0d8"
CARRIER_EDGE = "#263746"
WING_L = "#f4b942"
WING_R = "#ee9b2d"
STRUCT = "#63d99a"
CYAN = "#30c8d7"
MAGENTA = "#ee5acb"
RED = "#ff6b61"
BLUE = "#66a9ff"


@dataclass(frozen=True)
class VariantSpec:
    id: str
    slug: str
    title: str
    hypothesis: str
    facts: tuple[str, ...]
    wavelength: str
    family: str


@dataclass(frozen=True)
class MagnetSite:
    name: str
    driver: str
    side: str
    angle_deg: float
    face: tuple[float, float]
    normal: tuple[float, float]


@dataclass
class ConceptGeometry:
    full_outline: Polygon
    selected_outline: Polygon
    modules: tuple[Polygon, Polygon]
    structures: tuple[Polygon, Polygon]
    materials: tuple[Polygon, Polygon]
    apertures: tuple[list[Polygon], list[Polygon]]
    rear_apertures: tuple[list[Polygon], list[Polygon]]
    active_regions: tuple[Polygon, Polygon]
    metrics: dict[str, float | int | str]


VARIANTS = (
    VariantSpec(
        "W1", "solid_lx", "SOLID LX ENVELOPE",
        "Maximum thin-wing effect of restoring the complete B2 edge path.",
        ("full B2 plan", "continuous 1.20 mm front skin",
         "open rear reinforcement", "no acoustic apertures"),
        "Edge offsets are O(0.3 lambda) at 1 kHz and >1 lambda by 7 kHz.",
        "solid",
    ),
    VariantSpec(
        "W2", "solid_mid", "ROOT-PRESERVED MID ENVELOPE",
        "Separate edge-distance change from permeability while retaining every root.",
        ("50% radial envelope", "connected local root expansions",
         "same skin and Z envelope as W1", "no isolated UM islands"),
        "A fixed path change accrues 4x as much phase at 4 kHz as at 1 kHz; W2 halves only the geometric increment.",
        "solid",
    ),
    VariantSpec(
        "W3", "perforated_4k", "LOW-POROSITY STRAIGHT SHEET",
        "Test a fine, nearly homogeneous sheet whose inertance grows through 3-5 kHz.",
        ("pitch 4.0 mm", "diameter 1.40 mm",
         "2.6 mm straight open paths", "root/spine exclusion is geometric"),
        "4 mm pitch = lambda/21 at 4 kHz and lambda/12.3 at 7 kHz.",
        "straight",
    ),
    VariantSpec(
        "W4", "graded_1k_4k", "POROSITY + PATH GRADIENT",
        "Move the LM transition downward with longer, lower-porosity paths while keeping the upper field finer.",
        ("LM: 3-4% / 8 mm path", "UM: 8-10% / 3 mm path",
         "C1 blend y=315..335", "both variables change together"),
        "For comparable sheet reactance, l_eff/phi should fall roughly as 1/f.",
        "graded",
    ),
    VariantSpec(
        "W5", "open_honeycomb", "OPEN HONEYCOMB CONTROL",
        "Expose attachment and lattice effects with a real connected wall graph and no hidden skin.",
        ("actual polygonal openings", "0.8 mm minimum wall target",
         "LM 65-85% / constrained UM 45-65%", "common three-root skeleton remains solid"),
        "An approximately 7 mm cell is only lambda/7 at 7 kHz: this is a scattering control.",
        "honeycomb",
    ),
    VariantSpec(
        "W6", "tortuous_aperiodic", "OFFSET / APERIODIC PATHS",
        "Broaden the transition without direct front-to-rear line of sight or one identical pipe length.",
        ("offset entrance and rear exit", "LM D1.9/P4.4; UM D1.76/P3.2",
         "4-10 mm distributed paths",
         "open, inspectable, drainable; no closed volume"),
        "An 8-10 mm path has a quarter-wave feature near 8.6-10.7 kHz; lengths must be spread and measured.",
        "tortuous",
    ),
    VariantSpec(
        "W7", "vertical_aperture_gradient", "APERTURE-ONLY VERTICAL GRADIENT",
        "Hold pitch and path fixed; increase aperture upward to compensate part of the 1-to-4 kHz frequency rise.",
        ("constant 4.0 mm pitch", "constant 2.6 mm path",
         "LM: D0.90 / achieved phi about 4%", "UM/T: D2.05 / achieved phi about 12-13%"),
        "The ideal 4-to-15% scaling is clipped to about 4-to-12.5% by real UM structure; larger upward remains the sensible direction.",
        "aperture_gradient",
    ),
    VariantSpec(
        "W8", "vertical_path_gradient", "PATH-ONLY VERTICAL GRADIENT",
        "Hold aperture and porosity fixed; shorten the air slug from LM to UM/T.",
        ("diameter 1.20 mm", "pitch 4.0 mm / phi about 7%",
         "LM effective path 10-12 mm", "UM/T effective path 2.5-3.0 mm"),
        "The approximately 4:1 path ratio is the direct first-order counterpart to the 1-to-4 kHz frequency ratio.",
        "path_gradient",
    ),
    VariantSpec(
        "W9", "bimodal_apertures", "BIMODAL STRAIGHT APERTURES",
        "Spread viscous loss and end correction at fixed sheet depth and nearly fixed total porosity.",
        ("equal count small / large", "diameters 0.90 and 1.80 mm",
         "pitch 4.0 mm / ideal phi about 9.9%", "straight 2.6 mm paths"),
        "Both populations remain much smaller than lambda at 7 kHz; their loss/end corrections, not diffraction, are the variable.",
        "bimodal",
    ),
    VariantSpec(
        "W10", "radial_edge_loaded", "RADIAL EDGE-LOADED IMPEDANCE",
        "Keep leakage high near each carrier but make the B2 perimeter comparatively opaque.",
        ("inner phi about 16%", "outer phi about 4%",
         "constant 4.0 mm pitch / 2.6 mm path", "paired histogram with W11"),
        "Tests whether a low-porosity outer band preserves the long B2 acoustic edge at 4-7 kHz.",
        "radial_loaded",
    ),
    VariantSpec(
        "W11", "radial_edge_released", "RADIAL EDGE-RELEASED IMPEDANCE",
        "Reverse W10 spatially while preserving its aperture-size histogram.",
        ("inner phi about 4%", "outer phi about 16%",
         "constant 4.0 mm pitch / 2.6 mm path", "same global samples as W10"),
        "If the effective edge migrates inward, W10/W11 should separate most strongly above roughly 4 kHz.",
        "radial_released",
    ),
    VariantSpec(
        "W12", "solid_chirped_edge", "SOLID CHIRPED EDGE",
        "Distribute edge-arrival delays without introducing permeability.",
        ("continuous 1.20 mm skin", "C1 inward retreat 4-12 mm",
         "root islands remain full", "no acoustic apertures"),
        "The 8 mm retreat spread is about 8 degrees at 1 kHz, 34 degrees at 4 kHz, and 59 degrees at 7 kHz.",
        "chirped",
    ),
    VariantSpec(
        "W13", "perimeter_frame", "PERIMETER FRAME CONTROL",
        "Isolate the exact B2 outer edge with almost no pressure-supporting interior sheet.",
        ("8 mm nominal outer frame", "three-root spine + fan ties",
         "macro-open interior", "not a homogenized mesh"),
        "An 8 mm frame is only 8 degrees at 1 kHz but about 34/59 degrees at 4/7 kHz.",
        "frame",
    ),
    VariantSpec(
        "W14", "solid_radial_leak_slots", "SOLID DIRECTIONAL LEAK SLOTS",
        "Create deliberate secondary edge paths in a solid wing instead of pretending they form a sheet impedance.",
        ("6-8 C1 through-slots per side", "LM radial 8-28 mm / UM tangent 4-8 mm",
         "at least 1.2 mm structural ligaments", "slots avoid roots and perimeter"),
        "Projected LM slots span 0.09-0.33 lambda at 4 kHz; compact UM slots span 0.08-0.16 lambda at 7 kHz.",
        "slots",
    ),
)


def _cubic(p0, p1, p2, p3, count=28):
    t = np.linspace(0.0, 1.0, count, endpoint=False)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p0, p1, p2, p3))
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t**2 * p2 + t**3 * p3)


def _arc3(p1, p2, p3, count=28):
    (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
    d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    cx = ((x1*x1 + y1*y1) * (y2-y3) + (x2*x2 + y2*y2) * (y3-y1)
          + (x3*x3 + y3*y3) * (y1-y2)) / d
    cy = ((x1*x1 + y1*y1) * (x3-x2) + (x2*x2 + y2*y2) * (x1-x3)
          + (x3*x3 + y3*y3) * (x2-x1)) / d
    radius = math.hypot(x1 - cx, y1 - cy)
    a1, a2, a3 = (math.atan2(y - cy, x - cx) for x, y in (p1, p2, p3))
    ccw = (a2 - a1) % (2*math.pi) < (a3 - a1) % (2*math.pi)
    sweep = ((a3-a1) % (2*math.pi) if ccw else -((a1-a3) % (2*math.pi)))
    theta = a1 + sweep * np.linspace(0.0, 1.0, count, endpoint=False)
    return np.column_stack((cx + radius*np.cos(theta), cy + radius*np.sin(theta)))


def _mirror(geometry):
    return affinity.scale(geometry, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0))


def _largest_polygon(geometry) -> Polygon:
    geometry = geometry.buffer(0)
    if geometry.geom_type == "Polygon":
        return geometry
    return max(geometry.geoms, key=lambda item: item.area)


def b2_outline() -> Polygon:
    """Sample the exact post-drop B2 segment chain without importing OCC."""
    points: list[tuple[float, float]] = [
        (-76.199, 0.0), (-152.401, 256.155), (-38.122, 315.977),
        (-60.654, 391.709), (-10.081, 418.176),
    ]
    points.extend(map(tuple, _arc3(
        (-10.081, 418.176), (-24.570, 423.478), (-36.811, 432.879))[1:]))
    points.append((-36.811, 432.879))
    left_top = ((-36.811, 432.879), (-42.416, 438.602),
                (-46.699, 445.626), (-49.161, 453.457))
    points.extend(map(tuple, _cubic(*left_top)[1:]))
    points.extend(((-49.161, 453.457), (-36.468, 453.457)))
    top_curves = (
        ((-36.468, 453.457), (-35.847, 451.997), (-35.182, 450.378), (-34.388, 449.004)),
        ((-34.388, 449.004), (-27.556, 437.191), (-14.742, 428.951), (0.001, 428.947)),
        ((0.001, 428.947), (14.946, 428.943), (27.970, 437.256), (34.681, 449.510)),
        ((34.681, 449.510), (35.293, 450.628), (35.977, 452.266), (36.483, 453.457)),
    )
    for curve in top_curves:
        points.extend(map(tuple, _cubic(*curve)[1:]))
        points.append(curve[-1])
    points.extend(((36.483, 453.457), (49.177, 453.457)))
    right_top = ((49.177, 453.457), (46.712, 445.620),
                 (42.425, 438.592), (36.813, 432.866))
    points.extend(map(tuple, _cubic(*right_top)[1:]))
    points.append((36.813, 432.866))
    points.extend(map(tuple, _arc3(
        (36.813, 432.866), (24.570, 423.478), (10.081, 418.176))[1:]))
    points.extend(((10.081, 418.176), (60.654, 391.709),
                   (38.113, 315.947), (152.401, 256.120),
                   (76.201, 0.0)))
    raw = _largest_polygon(Polygon(points))
    # The drawing has micron-scale left/right rounding differences.  Concepts
    # use the exact common inside so the completed left field can be mirrored.
    return _largest_polygon(raw.intersection(_mirror(raw)))


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _local_core(y: float):
    if y <= 315.0:
        return LM_CENTER, LM_OUTER_R
    if y < 335.0:
        q = _smoothstep((y - 315.0) / 20.0)
        return (1-q)*LM_CENTER + q*UM_CENTER, (1-q)*LM_OUTER_R + q*UM_OUTER_R
    if y <= 400.0:
        return UM_CENTER, UM_OUTER_R
    if y < 430.0:
        q = _smoothstep((y - 400.0) / 30.0)
        return (1-q)*UM_CENTER + q*T_CENTER, (1-q)*UM_OUTER_R + q*T_OUTER_R
    return T_CENTER, T_OUTER_R


def _radial_outline(full: Polygon, fraction_fn) -> Polygon:
    points = []
    for x, y in np.asarray(full.exterior.coords)[:-1]:
        center, core_radius = _local_core(float(y))
        delta = np.array((x, y)) - center
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        fraction = float(fraction_fn(float(x), float(y), math.atan2(delta[1], delta[0])))
        radius = core_radius + fraction * max(0.0, distance - core_radius)
        points.append(tuple(center + delta * (radius / distance)))
    result = _largest_polygon(Polygon(points))
    return _largest_polygon(result.intersection(full))


def mid_outline(full: Polygon) -> Polygon:
    return _radial_outline(full, lambda _x, _y, _a: 0.5)


def chirped_outline(full: Polygon) -> Polygon:
    points = []
    for x, y in np.asarray(full.exterior.coords)[:-1]:
        center, core_radius = _local_core(float(y))
        delta = np.array((x, y)) - center
        distance = max(float(np.linalg.norm(delta)), 1e-6)
        angle = math.atan2(delta[1], delta[0])
        wave = (math.sin(14.0*angle + 0.035*y)
                + 0.35*math.sin(23.0*angle - 0.019*y + 1.2)) / 1.35
        retreat = 8.0 + 4.0*wave
        radius = max(core_radius + 3.5, distance - retreat)
        points.append(tuple(center + delta*(radius/distance)))
    return _largest_polygon(Polygon(points).buffer(0).intersection(full))


def core_keepout(gap: float):
    return unary_union((
        Point(*LM_CENTER).buffer(LM_OUTER_R + gap, resolution=128),
        Point(*UM_CENTER).buffer(UM_OUTER_R + gap, resolution=128),
        Point(*T_CENTER).buffer(T_OUTER_R + gap, resolution=128),
    ))


def um_realizability_lobes():
    """Common UM acoustic/structural lobes required by the retained roots.

    The R51.7 V1LF carrier exceeds the historic B2 outline through part of its
    side arc.  A design constrained literally to B2 has neither receiver depth
    nor an upper acoustic field.  These 18 mm annular side lobes are the small,
    common and explicitly reported exception used by every variant.
    """
    outer = Point(*UM_CENTER).buffer(UM_OUTER_R + 18.0, resolution=128)
    lobes = outer.difference(core_keepout(0.20)).intersection(
        box(-92.0, 323.0, 92.0, 426.0))
    return lobes.buffer(0)


def magnet_sites() -> tuple[MagnetSite, ...]:
    records = []
    radial_definitions = (
        ("lm_upper_left", "lm", "left", LM_CENTER, LM_OUTER_R, 116.0),
        ("um_left", "um", "left", UM_CENTER, UM_OUTER_R, 129.5),
        ("lm_upper_right", "lm", "right", LM_CENTER, LM_OUTER_R, 64.0),
        ("um_right", "um", "right", UM_CENTER, UM_OUTER_R, 50.5),
    )
    for name, driver, side, center, radius, angle in radial_definitions:
        radians = math.radians(angle)
        normal = np.array((math.cos(radians), math.sin(radians)))
        face = center + radius*normal
        records.append(MagnetSite(
            name, driver, side, angle, tuple(map(float, face)),
            tuple(map(float, normal))))
    # Lower LM stations mate through the straight W64 base sides shared by
    # the floor and no-floor carriers.  They are not synthetic points on the
    # R113 ring.
    records.extend((
        MagnetSite("lm_lower_left", "lm", "left", 180.0,
                   (-32.0, 18.0), (-1.0, 0.0)),
        MagnetSite("lm_lower_right", "lm", "right", 0.0,
                   (32.0, 18.0), (1.0, 0.0)),
    ))
    return tuple(records)


def _side_sites(side: str):
    return tuple(site for site in magnet_sites() if site.side == side)


def _root_pad(site: MagnetSite, full: Polygon):
    del full
    face = np.asarray(site.face)
    normal = np.asarray(site.normal)
    tangent = np.array((-normal[1], normal[0]))
    # The B2 outline is only about 0.5 mm outside the UM carrier at 129.5°.
    # A D5.2 receiver physically cannot live inside that acoustic outline.
    # The identical, rounded upper root pod is therefore a quantified common
    # structural exception; it is wider than the LM pads so its corridor into
    # the B2 field survives the 0.60 mm connectivity erosion.
    radial_length = 8.0 if site.driver == "um" else 6.0
    half_width = 6.5 if site.driver == "um" else 4.4
    inner = face - 0.02*normal
    outer = face + radial_length*normal
    pad = Polygon((
        tuple(inner - half_width*tangent),
        tuple(outer - half_width*tangent),
        tuple(outer + half_width*tangent),
        tuple(inner + half_width*tangent),
    )).buffer(0.75, join_style=1).buffer(-0.75, join_style=1)
    return pad.difference(core_keepout(0.0))


def _um_root_bridge(site: MagnetSite):
    """Common outboard arc that carries an UM root into the B2 field.

    At the exact 129.5/50.5-degree station the historic B2 outline lies
    inboard of the V1LF carrier, so a receiver pod alone has no positive-area
    overlap with B2.  The bridge follows the carrier exterior to the wider
    shoulder instead of cutting through the ring.
    """
    if site.driver != "um":
        return Polygon()
    if site.side == "left":
        angles = np.linspace(129.5, 160.0, 34)
    else:
        angles = np.linspace(50.5, 20.0, 34)
    arc = _arc_polyline(UM_CENTER, UM_OUTER_R + 3.1, angles).buffer(
        1.55, cap_style=1, join_style=1)
    terminal_angle = float(angles[-1])
    radians = math.radians(terminal_angle)
    direction = np.array((math.cos(radians), math.sin(radians)))
    endpoint = UM_CENTER + (UM_OUTER_R + 3.1)*direction
    # Eight millimetres reaches the real B2 outer rail in the wider shoulder.
    # This is a fan tie, not an acoustic-outline extension.
    fan_tie = LineString((tuple(endpoint), tuple(endpoint + 8.0*direction))).buffer(
        1.55, cap_style=1, join_style=1)
    return unary_union((arc, fan_tie)).difference(core_keepout(0.0)).buffer(0)


def _intercarrier_root_bridge(side: str):
    """Keep-out-safe solid spine joining the LM and UM root systems.

    There is no continuous B2-only corridor over part of the 0.4 mm carrier
    neck: the V1LF UM carrier itself lies outside the historic B2 comparator.
    A common outboard C1-like spine is therefore mandatory, not optional art.
    """
    left_points = (
        (-50.9, 305.2), (-54.8, 317.0), (-58.2, 331.0),
        (-60.0, 346.0), (-60.0, 360.0), (-57.5, 375.0),
        (-51.5, 384.8),
    )
    path = LineString(left_points).buffer(1.70, cap_style=1, join_style=1)
    if side == "right":
        path = _mirror(path)
    return path.difference(core_keepout(0.0)).buffer(0)


def _root_exceptions(full: Polygon):
    return unary_union([
        geometry
        for site in magnet_sites()
        for geometry in (_root_pad(site, full), _um_root_bridge(site))
        if not geometry.is_empty
    ] + [_intercarrier_root_bridge("left"),
         _intercarrier_root_bridge("right")]).buffer(0)


def _arc_polyline(center, radius, angles):
    points = []
    for angle in angles:
        radians = math.radians(float(angle))
        points.append((center[0] + radius*math.cos(radians),
                       center[1] + radius*math.sin(radians)))
    return LineString(points)


def _common_left_geometry(outline: Polygon, full: Polygon):
    left_box = box(-190.0, -20.0, -0.65, 530.0)
    exact_core = core_keepout(0.0)
    physical_outline = outline.union(um_realizability_lobes()).buffer(0)
    base = physical_outline.difference(core_keepout(0.20)).intersection(left_box)
    pads = [_root_pad(site, full) for site in _side_sites("left")]
    root_bridges = [_um_root_bridge(site) for site in _side_sites("left")
                    if site.driver == "um"]
    intercarrier_bridge = _intercarrier_root_bridge("left")

    # The outer rail is real material.  It connects the three receiver ties
    # without crossing either immutable carrier.
    outer_band = physical_outline.boundary.buffer(2.40, join_style=1).intersection(
        physical_outline).intersection(left_box).difference(exact_core)
    ties = []
    for site in _side_sites("left"):
        face = np.asarray(site.face)
        normal = np.asarray(site.normal)
        ray = LineString((face - 0.02*normal, face + 320.0*normal))
        ties.append(ray.buffer(1.55, cap_style=2, join_style=1)
                    .intersection(physical_outline).intersection(left_box)
                    .difference(exact_core))

    lower_face = np.asarray(next(
        site.face for site in _side_sites("left")
        if site.name == "lm_lower_left"))
    upper_face = np.asarray(next(
        site.face for site in _side_sites("left")
        if site.name == "lm_upper_left"))
    lm_arc = LineString((tuple(lower_face), tuple(upper_face))).buffer(
        1.55, cap_style=1, join_style=1)
    # Root preservation is structural, not an acoustic-outline variable.  The
    # LM arc may therefore occupy the full B2 comparator even for W2/W12;
    # otherwise their retreated outlines strand the lower LM receiver.
    backbone = lm_arc.intersection(full).intersection(left_box).difference(exact_core)

    structure = unary_union((outer_band, backbone, *ties, *pads,
                             *root_bridges, intercarrier_bridge)).buffer(0)
    # W2/W12 may retreat outside a root.  The pads are always retained inside
    # full B2 and are joined to the selected plan by the corresponding tie.
    # Only the named receiver pads may exceed the acoustic B2 comparator.
    # Ties, backbone, and outer rail remain clipped to the selected outline.
    structure = structure.difference(exact_core).buffer(0)
    module = base.union(structure).buffer(0)
    return _largest_polygon(module), _largest_polygon(structure)


def common_geometry(spec: VariantSpec):
    full = b2_outline()
    if spec.slug == "solid_mid":
        selected = mid_outline(full)
    elif spec.slug == "solid_chirped_edge":
        selected = chirped_outline(full)
    else:
        selected = full
    left_module, left_structure = _common_left_geometry(selected, full)
    right_module = _mirror(left_module)
    right_structure = _mirror(left_structure)
    return full, selected, (left_module, right_module), (left_structure, right_structure)


def _grid_centers(active: Polygon, pitch: float, jitter: float, seed: int):
    rng = np.random.default_rng(seed)
    prepared = prep(active)
    centers = []
    minx, miny, maxx, maxy = active.bounds
    for row, y in enumerate(np.arange(miny + pitch, maxy, pitch)):
        shift = 0.5*pitch if row % 2 else 0.0
        for x in np.arange(minx + pitch, min(-0.8, maxx), pitch):
            px, py = x + shift, y
            if jitter:
                px += rng.uniform(-jitter, jitter)
                py += rng.uniform(-jitter, jitter)
            if prepared.contains(Point(px, py)):
                centers.append((float(px), float(py), row))
    return centers


def _round_field(active: Polygon, pitch: float, diameter_fn, *,
                 jitter=0.0, seed=SEED, keep_fn=None):
    result = []
    for index, (x, y, row) in enumerate(_grid_centers(active, pitch, jitter, seed)):
        if keep_fn is not None and not keep_fn(index, x, y, row):
            continue
        diameter = float(diameter_fn(index, x, y, row))
        aperture = Point(x, y).buffer(diameter/2.0, resolution=10)
        if active.covers(aperture):
            result.append(aperture)
    return result


def _hexagon(x: float, y: float, radius: float):
    return Polygon([
        (x + radius*math.cos(math.radians(30.0 + 60.0*i)),
         y + radius*math.sin(math.radians(30.0 + 60.0*i)))
        for i in range(6)
    ])


def _honeycomb_field(active: Polygon):
    result = []
    # The B2-to-UM strip is much narrower than the LM field.  Use the same
    # real 0.8 mm wall rule at two printable scales instead of silently
    # deleting the upper honeycomb and pretending the concept spans UM.
    for region, radius in (
            (active.intersection(box(-200, -20, 0, 324.6)), 4.15),
            (active.intersection(box(-200, 325.4, 0, 530.0)), 2.13)):
        if region.is_empty:
            continue
        dx = math.sqrt(3.0)*radius + 0.80
        dy = 1.5*radius + 0.80
        minx, miny, maxx, maxy = region.bounds
        row = 0
        for y in np.arange(miny + radius, maxy, dy):
            shift = 0.5*dx if row % 2 else 0.0
            for x in np.arange(minx + radius + shift, min(-0.8, maxx), dx):
                hole = _hexagon(float(x), float(y), radius)
                # Boundary cells may be cleanly truncated by the already
                # eroded active field.  This preserves the 1.20 mm structural
                # reserve while avoiding an artificial low-porosity dead band
                # around every root and perimeter.  Reject tiny remnants.
                clipped = hole.intersection(region)
                if (not clipped.is_empty and clipped.geom_type == "Polygon"
                        and clipped.area >= 0.55*hole.area):
                    result.append(clipped)
            row += 1
    return result


def _bimodal_field(active: Polygon):
    """Equal-count D0.9/D1.8 population on one common safe center set."""
    safe = active.buffer(-0.90)
    centers = _grid_centers(safe, 4.0, 0.06, SEED + 9)
    if len(centers) % 2:
        centers = centers[:-1]
    result = []
    for index, (x, y, _row) in enumerate(centers):
        diameter = 1.80 if index % 2 == 0 else 0.90
        aperture = Point(x, y).buffer(diameter/2.0, resolution=10)
        if not active.covers(aperture):
            raise RuntimeError("bimodal safe-center construction escaped active field")
        result.append(aperture)
    return result


def _radial_coordinate(point: Point, full: Polygon):
    core_distance = core_keepout(0.20).boundary.distance(point)
    outer_distance = full.boundary.distance(point)
    total = core_distance + outer_distance
    return 0.5 if total <= 1e-9 else core_distance/total


def _tortuous_field(active: Polygon):
    fronts, rears = [], []

    def add_square_pair_grid(region, pitch, base_radius, seed):
        rng = np.random.default_rng(seed)
        minx, miny, maxx, maxy = region.bounds
        index = 0
        for y0 in np.arange(miny + pitch, maxy, pitch):
            for x0 in np.arange(minx + pitch, min(-0.8, maxx), pitch):
                x = float(x0 + rng.uniform(-0.08, 0.08))
                y = float(y0 + rng.uniform(-0.08, 0.08))
                index += 1
                radius = base_radius*(1.0 + 0.055*math.sin(index*0.71))
                front = Point(x, y).buffer(radius, resolution=10)
                rear = Point(x - 0.5*pitch, y + 0.5*pitch).buffer(
                    radius, resolution=4)
                if region.covers(front) and region.covers(rear):
                    fronts.append(front)
                    rears.append(rear)

    # Separate printable scales keep the narrow UM strip populated without
    # sacrificing the strict no-projected-line-of-sight construction.
    add_square_pair_grid(
        active.intersection(box(-200.0, -20.0, 0.0, 324.4)),
        4.40, 0.95, SEED + 60)
    add_square_pair_grid(
        active.intersection(box(-200.0, 325.6, 0.0, 530.0)),
        3.20, 0.88, SEED + 61)
    return fronts, rears


def _matched_radial_field(active: Polygon, full: Polygon, loaded: bool):
    """W10/W11 exact matched aperture multiset with reversed radial rank."""
    safe = active.buffer(-0.90)
    records = []
    for x, y, row in _grid_centers(safe, 4.0, 0.05, SEED + 10):
        coordinate = _radial_coordinate(Point(x, y), full)
        records.append((coordinate, x, y, row))
    records.sort(key=lambda item: item[0])
    diameters = [1.80 - 0.90*_smoothstep(item[0]) for item in records]
    assigned = diameters if loaded else list(reversed(diameters))
    result = []
    for (_coordinate, x, y, _row), diameter in zip(records, assigned):
        aperture = Point(x, y).buffer(diameter/2.0, resolution=10)
        if not active.covers(aperture):
            raise RuntimeError("radial matched-pair safe center escaped active field")
        result.append(aperture)
    return result


def _slot_field(active: Polygon):
    selected = []

    def add_region(center, radius, angles, lengths, target,
                   offsets=(6.0, 8.0, 10.0, 12.0)):
        added = 0
        for angle in angles:
            radians = math.radians(float(angle))
            direction = np.array((math.cos(radians), math.sin(radians)))
            accepted = None
            for start_offset in offsets:
                start = center + (radius + start_offset)*direction
                for length in lengths:
                    end = start + length*direction
                    width = 1.40 + 0.18*math.sin(math.radians(float(angle))*3.0)
                    slot = LineString((tuple(start), tuple(end))).buffer(
                        width/2.0, cap_style=1, join_style=1)
                    if (active.covers(slot)
                            and all(slot.distance(other) >= 6.0 for other in selected)):
                        accepted = slot
                        break
                if accepted is not None:
                    break
            if accepted is not None:
                selected.append(accepted)
                added += 1
            if added == target:
                break
        return added

    lm_added = add_region(
        LM_CENTER, LM_OUTER_R, np.linspace(126.0, 234.0, 37),
        (28.0, 24.0, 20.0, 16.0, 12.0, 8.0), 5)
    def add_um_arc_once():
        # The UM lobe is an annular strip interrupted by its receiver arc and
        # the mandatory inter-carrier spine.  A radial 4 mm capsule cannot
        # retain the required ligaments there, so search a compact
        # carrier-tangent arc rather than drawing an impossible horn.
        for radial_offset in (7.0, 9.0, 11.0, 13.0):
            arc_radius = UM_OUTER_R + radial_offset
            for center_angle in np.linspace(92.0, 268.0, 89):
                for length in (8.0, 7.0, 6.0, 5.0, 4.0):
                    half_angle = math.degrees(0.5*length/arc_radius)
                    angles = np.linspace(
                        center_angle-half_angle, center_angle+half_angle, 12)
                    slot = _arc_polyline(UM_CENTER, arc_radius, angles).buffer(
                        0.60, cap_style=1, join_style=1)
                    if (slot.centroid.y >= 335.0 and active.covers(slot)
                            and all(slot.distance(other) >= 6.0 for other in selected)):
                        selected.append(slot)
                        return 1
        return 0

    um_added = add_um_arc_once() + add_um_arc_once()
    if lm_added < 4 or um_added < 2 or len(selected) < 7:
        raise RuntimeError(
            "safe radial slots shortfall: "
            f"LM={lm_added}, UM={um_added}, total={len(selected)}")
    return selected


def _field_for_variant(spec: VariantSpec, active: Polygon, full: Polygon):
    slug = spec.slug
    rear = []
    if slug in {"solid_lx", "solid_mid", "solid_chirped_edge", "perimeter_frame"}:
        return [], rear
    if slug == "perforated_4k":
        return _round_field(
            active, 4.0,
            lambda _i, _x, _y, _r: 1.40,
            jitter=0.20, seed=SEED + 3), rear
    if slug == "graded_1k_4k":
        low = active.intersection(box(-200, -20, 0, 314.4))
        blend = active.intersection(box(-200, 315.6, 0, 334.4))
        high = active.intersection(box(-200, 335.6, 0, 530.0))
        field = _round_field(low, 5.0, lambda *_: 1.0,
                             jitter=0.10, seed=SEED + 40)
        field += _round_field(blend, 4.5, lambda i, x, y, r: 1.2,
                              jitter=0.10, seed=SEED + 41)
        field += _round_field(high, 4.0, lambda *_: 1.6,
                              jitter=0.10, seed=SEED + 42)
        return field, rear
    if slug == "open_honeycomb":
        return _honeycomb_field(active), rear
    if slug == "tortuous_aperiodic":
        return _tortuous_field(active)
    if slug == "vertical_aperture_gradient":
        def diameter(_i, _x, y, _row):
            q = _smoothstep((y - 315.0)/20.0)
            return math.sqrt(0.90**2 + (2.05**2 - 0.90**2)*q)
        return _round_field(active, 4.0, diameter,
                            jitter=0.08, seed=SEED + 7), rear
    if slug == "vertical_path_gradient":
        return _round_field(active, 4.0, lambda *_: 1.20,
                            jitter=0.08, seed=SEED + 8), rear
    if slug == "bimodal_apertures":
        return _bimodal_field(active), rear
    if slug in {"radial_edge_loaded", "radial_edge_released"}:
        return _matched_radial_field(active, full, slug.endswith("loaded")), rear
    if slug == "solid_radial_leak_slots":
        return _slot_field(active), rear
    raise KeyError(slug)


def _component_count(geometry) -> int:
    if geometry.is_empty:
        return 0
    if geometry.geom_type == "Polygon":
        return 1
    return sum(1 for item in geometry.geoms if item.geom_type == "Polygon" and item.area > 1e-4)


def _mirror_polygons(polygons):
    return [_mirror(polygon) for polygon in polygons]


def build_concept(spec: VariantSpec) -> ConceptGeometry:
    full, selected, modules, structures = common_geometry(spec)
    left_active = modules[0].difference(structures[0].buffer(1.20, join_style=1)).buffer(-0.05)
    # LM and UM acoustic fields are intentionally separate regions joined by
    # the protected solid spine.  Preserve every positive-area field component.
    left_active = left_active.buffer(0)
    apertures_left, rear_left = _field_for_variant(spec, left_active, full)
    apertures = (apertures_left, _mirror_polygons(apertures_left))
    rear_apertures = (rear_left, _mirror_polygons(rear_left))
    active_regions = (left_active, _mirror(left_active))

    if spec.slug == "perimeter_frame":
        physical_outline = full.union(um_realizability_lobes()).buffer(0)
        outer_frame = physical_outline.boundary.buffer(
            8.0, join_style=1).intersection(modules[0])
        left_frame = outer_frame.union(structures[0]).buffer(0)
        materials = (_largest_polygon(left_frame), _mirror(_largest_polygon(left_frame)))
    elif apertures_left:
        left_material = modules[0].difference(unary_union(apertures_left)).buffer(0)
        right_material = _mirror(left_material)
        materials = (_largest_polygon(left_material), _largest_polygon(right_material))
    else:
        materials = modules

    open_geometry = modules[0].difference(materials[0]).buffer(0)
    open_left = open_geometry.area
    active_porosity = (100.0*open_left/left_active.area
                       if left_active.area > 0 else 0.0)
    gross_open = 100.0*open_left/modules[0].area
    def upper_owned(item):
        point = np.array((item.centroid.x, item.centroid.y))
        lm_distance = float(np.linalg.norm(point - LM_CENTER))
        upper_distance = min(float(np.linalg.norm(point - UM_CENTER)),
                             float(np.linalg.norm(point - T_CENTER)))
        return upper_distance < lm_distance
    um_count = sum(1 for item in apertures_left if upper_owned(item))
    lm_count = len(apertures_left) - um_count
    zone_bounds = {
        "lm": box(-200.0, -20.0, 0.0, 315.0),
        "blend": box(-200.0, 315.0, 0.0, 335.0),
        "um": box(-200.0, 335.0, 0.0, 530.0),
    }
    zone_porosity = {}
    for name, zone in zone_bounds.items():
        zone_active = left_active.intersection(zone).area
        zone_open = open_geometry.intersection(zone).area
        zone_porosity[name] = 100.0*zone_open/zone_active if zone_active > 0 else 0.0
    metrics = {
        "apertures_per_side": len(apertures_left),
        "lm_apertures": lm_count,
        "um_apertures": um_count,
        "active_porosity_pct": active_porosity,
        "gross_open_pct": gross_open,
        "lm_active_porosity_pct": zone_porosity["lm"],
        "blend_active_porosity_pct": zone_porosity["blend"],
        "um_active_porosity_pct": zone_porosity["um"],
        "module_area_mm2": modules[0].area,
        "structure_area_mm2": structures[0].area,
    }
    concept = ConceptGeometry(
        full, selected, modules, structures, materials, apertures,
        rear_apertures, active_regions, metrics)
    validate_concept(spec, concept)
    return concept


def validate_concept(spec: VariantSpec, concept: ConceptGeometry):
    full = concept.full_outline
    if not full.is_valid or full.area <= 0:
        raise RuntimeError("invalid B2 comparator")
    if not full.buffer(1e-5).covers(concept.selected_outline):
        raise RuntimeError(f"{spec.slug}: selected outline exceeds B2")
    for index, side in enumerate(("left", "right")):
        module = concept.modules[index]
        structure = concept.structures[index]
        material = concept.materials[index]
        if _component_count(module) != 1:
            raise RuntimeError(f"{spec.slug} {side}: module is disconnected")
        if _component_count(structure) != 1:
            raise RuntimeError(f"{spec.slug} {side}: structural skeleton is disconnected")
        if _component_count(material) != 1:
            raise RuntimeError(f"{spec.slug} {side}: final material is disconnected")
        allowed = full.union(um_realizability_lobes()).union(
            _root_exceptions(full)).buffer(1e-5)
        if not allowed.covers(module):
            raise RuntimeError(f"{spec.slug} {side}: material exceeds B2/root exceptions")
        eroded = structure.buffer(-0.60)
        if _component_count(eroded) != 1:
            raise RuntimeError(f"{spec.slug} {side}: 0.60 mm eroded skeleton disconnected")
        for site in _side_sites(side):
            face = np.asarray(site.face)
            normal = np.asarray(site.normal)
            witness = Point(*(face + 0.80*normal)).buffer(0.15)
            if not structure.buffer(0.06).intersects(witness):
                raise RuntimeError(f"{spec.slug}: missing root material at {site.name}")
            if not eroded.buffer(0.06).intersects(witness):
                raise RuntimeError(
                    f"{spec.slug}: eroded skeleton no longer reaches {site.name}")
            pad = _root_pad(site, full)
            tangent = np.array((-normal[1], normal[0]))
            coordinates = np.asarray(pad.exterior.coords)
            tangential = coordinates @ tangent
            radial = coordinates @ normal
            tangential_span = float(tangential.max() - tangential.min())
            radial_span = float(radial.max() - radial.min())
            hard_t = 11.8 if site.driver == "um" else 7.6
            hard_r = 5.4 if site.driver == "um" else 3.4
            if tangential_span < hard_t or radial_span < hard_r:
                raise RuntimeError(
                    f"{spec.slug}: {site.name} pad below hard plan minimum "
                    f"({tangential_span:.3f} x {radial_span:.3f} mm)")
        protected = structure.buffer(1.199)
        for aperture in concept.apertures[index]:
            if not concept.active_regions[index].buffer(1e-6).covers(aperture):
                raise RuntimeError(f"{spec.slug} {side}: aperture leaves active field")
            if protected.intersects(aperture):
                raise RuntimeError(f"{spec.slug} {side}: aperture invades skeleton")
        apertures = concept.apertures[index]
        if len(apertures) > 1:
            minimum_gap = 0.79 if spec.slug == "open_honeycomb" else 1.19
            tree = STRtree(apertures)
            for aperture_index, aperture in enumerate(apertures):
                for other_index in tree.query(aperture.buffer(minimum_gap)):
                    other_index = int(other_index)
                    if other_index <= aperture_index:
                        continue
                    if aperture.distance(apertures[other_index]) < minimum_gap:
                        raise RuntimeError(
                            f"{spec.slug} {side}: adjacent aperture wall below "
                            f"{minimum_gap:.2f} mm")
    if concept.modules[0].intersection(concept.modules[1]).area > 1e-6:
        raise RuntimeError(f"{spec.slug}: left/right modules bridge")
    if concept.modules[1].symmetric_difference(_mirror(concept.modules[0])).area > 1e-5:
        raise RuntimeError(f"{spec.slug}: module mirror mismatch")
    if concept.structures[1].symmetric_difference(_mirror(concept.structures[0])).area > 1e-5:
        raise RuntimeError(f"{spec.slug}: structure mirror mismatch")
    if spec.family not in {"solid", "chirped", "frame"}:
        if (concept.metrics["lm_active_porosity_pct"] <= 0.0
                or concept.metrics["um_active_porosity_pct"] <= 0.0):
            raise RuntimeError(f"{spec.slug}: acoustic field does not span LM and UM")
    zone_targets = {
        "perforated_4k": ((8.0, 12.0), (7.0, 12.0)),
        "graded_1k_4k": ((2.0, 5.0), (8.0, 15.0)),
        "open_honeycomb": ((65.0, 85.0), (45.0, 65.0)),
        "tortuous_aperiodic": ((10.0, 20.0), (10.0, 20.0)),
        "vertical_aperture_gradient": ((3.0, 5.0), (12.0, 16.0)),
        "solid_radial_leak_slots": ((1.0, 3.0), (1.0, 3.0)),
    }
    if spec.slug in zone_targets:
        for label, value, limits in (
                ("LM", concept.metrics["lm_active_porosity_pct"],
                 zone_targets[spec.slug][0]),
                ("UM", concept.metrics["um_active_porosity_pct"],
                 zone_targets[spec.slug][1])):
            if not limits[0] <= value <= limits[1]:
                raise RuntimeError(
                    f"{spec.slug}: {label} active porosity {value:.2f}% "
                    f"outside {limits[0]:.1f}-{limits[1]:.1f}%")
    if spec.slug == "tortuous_aperiodic":
        for front, rear in zip(concept.apertures[0], concept.rear_apertures[0]):
            if front.intersects(rear):
                raise RuntimeError("W6 front/rear apertures retain line of sight")
        if unary_union(concept.apertures[0]).intersects(
                unary_union(concept.rear_apertures[0])):
            raise RuntimeError("W6 aperture planes retain cross-pair line of sight")


def _iter_polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    else:
        for item in geometry.geoms:
            yield from _iter_polygons(item)


def _draw_geometry(ax, geometry, *, facecolor, edgecolor=INK, alpha=1.0,
                   linewidth=0.7, zorder=2):
    for polygon in _iter_polygons(geometry):
        vertices = []
        codes = []
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords)
            vertices.extend(coordinates)
            codes.extend([MplPath.MOVETO]
                         + [MplPath.LINETO]*(len(coordinates)-2)
                         + [MplPath.CLOSEPOLY])
        path = MplPath(np.asarray(vertices, dtype=float), codes)
        ax.add_patch(PathPatch(
            path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            lw=linewidth, zorder=zorder, joinstyle="round"))


def _draw_apertures(ax, apertures, *, edgecolor=CYAN, facecolor=BG,
                    alpha=1.0, zorder=7, linewidth=0.22):
    patches = [MplPolygon(np.asarray(polygon.exterior.coords), closed=True)
               for polygon in apertures]
    if patches:
        ax.add_collection(PatchCollection(
            patches, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            linewidth=linewidth, zorder=zorder))


def _draw_carriers(ax, *, labels=True):
    for center, outer, inner, label in (
            (LM_CENTER, LM_OUTER_R, LM_OPEN_R, "LM"),
            (UM_CENTER, UM_OUTER_R, UM_OPEN_R, "UM")):
        ax.add_patch(Circle(center, outer, fc=CARRIER, ec=CARRIER_EDGE,
                            lw=1.05, zorder=12))
        ax.add_patch(Circle(center, inner, fc=BG, ec=CARRIER_EDGE,
                            lw=0.9, zorder=13))
        if labels:
            ax.text(center[0], center[1], label, color=MUTED, fontsize=8,
                    ha="center", va="center", weight="bold", zorder=14)
    ax.add_patch(Circle(T_CENTER, T_OUTER_R, fc="none", ec=CARRIER,
                        lw=0.9, ls=(0, (4, 3)), zorder=12))
    if labels:
        ax.text(0, T_CENTER[1], "T keep-out", color=MUTED, fontsize=6.5,
                ha="center", va="center", zorder=14)
    for site in magnet_sites():
        ax.add_patch(Circle(site.face, 3.55, fc=BG, ec=MAGENTA,
                            lw=0.95, zorder=17))
        ax.add_patch(Circle(site.face, 2.60, fc="none", ec=INK,
                            lw=0.65, zorder=18))


def _draw_concept_on_axis(ax, spec: VariantSpec, concept: ConceptGeometry,
                          xlim, ylim, *, labels=True, grid=True):
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    if grid:
        ax.grid(True, color=GRID, lw=0.35, alpha=0.35)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=6)
    full_xy = np.asarray(concept.full_outline.exterior.coords)
    ax.plot(full_xy[:, 0], full_xy[:, 1], color=MUTED, lw=0.75,
            ls=(0, (2, 3)), zorder=1)
    if spec.slug == "perimeter_frame":
        for module in concept.modules:
            _draw_geometry(ax, module, facecolor="none", edgecolor=WING_L,
                           alpha=0.25, linewidth=0.6, zorder=2)
    else:
        for material, color in zip(concept.materials, (WING_L, WING_R)):
            _draw_geometry(ax, material, facecolor=color, edgecolor=INK,
                           alpha=0.88, linewidth=0.65, zorder=3)
    if spec.slug != "perimeter_frame":
        for apertures in concept.apertures:
            _draw_apertures(ax, apertures,
                            edgecolor=(MAGENTA if spec.slug == "solid_radial_leak_slots" else CYAN),
                            linewidth=(0.55 if spec.slug == "solid_radial_leak_slots" else 0.20))
    for rear in concept.rear_apertures:
        _draw_apertures(ax, rear, edgecolor=MAGENTA, facecolor="none",
                        alpha=0.80, zorder=6, linewidth=0.40)
    for structure in concept.structures:
        _draw_geometry(ax, structure, facecolor=STRUCT, edgecolor=INK,
                       alpha=0.82, linewidth=0.45, zorder=9)
    # Repaint apertures above the structural ghost only where they are legal;
    # no accepted footprint intersects the protected mask.
    if spec.slug not in {"perimeter_frame"}:
        for apertures in concept.apertures:
            _draw_apertures(ax, apertures,
                            edgecolor=(MAGENTA if spec.slug == "solid_radial_leak_slots" else CYAN),
                            linewidth=(0.55 if spec.slug == "solid_radial_leak_slots" else 0.20),
                            zorder=10)
    _draw_carriers(ax, labels=labels)
    ax.axvline(0, color=INK, lw=0.45, ls=(0, (2, 4)), alpha=0.55, zorder=19)


def _draw_section(ax, spec: VariantSpec):
    ax.set_facecolor(PANEL)
    ax.set_xlim(6.0, 18.8); ax.set_ylim(-0.1, 3.25)
    ax.set_xticks((REAR_Z, 10.0, 14.0, FRONT_Z)); ax.set_yticks([])
    ax.tick_params(axis="x", colors=MUTED, labelsize=6)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.axvline(REAR_Z, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    ax.axvline(FRONT_Z, color=INK, lw=0.7)
    ax.set_title("THROUGH-Z CONSTRUCTION", loc="left", color=INK,
                 fontsize=8.4, weight="bold", pad=6)
    family = spec.family
    if family in {"solid", "chirped"}:
        ax.add_patch(Rectangle((17.1, 0.38), 1.2, 2.0, fc=WING_L, ec=INK, lw=0.7))
        ax.add_patch(MplPolygon(((6.8, 0.38), (17.1, 0.38),
                                (17.1, 0.78), (8.1, 0.62)),
                               fc="#b97a25", ec=INK, lw=0.6))
        ax.text(6.35, 2.62, "1.20 skin; open rear ribs; no trapped air",
                color=MUTED, fontsize=6.8)
    elif family == "frame":
        for x in (7.0, 17.1):
            ax.add_patch(Rectangle((x, 0.38), 1.2, 2.0, fc=STRUCT, ec=INK, lw=0.7))
        ax.annotate("", (18.55, 1.40), (6.45, 1.40),
                    arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.0))
        ax.text(6.35, 2.62, "macro-open core; only frame + ties + roots",
                color=MUTED, fontsize=6.8)
    elif family == "tortuous":
        for x, color in ((16.9, CYAN), (9.4, MAGENTA)):
            ax.add_patch(Rectangle((x, 0.38), 0.8, 2.0, fc=WING_L, ec=color, lw=0.7))
        ax.plot((18.45, 17.3, 13.5, 9.8, 6.55),
                (1.92, 1.92, 1.35, 0.86, 0.86), color=INK, lw=1.2)
        ax.text(6.35, 2.62, "offset open planes; no direct line of sight",
                color=MUTED, fontsize=6.8)
    elif family == "slots":
        ax.add_patch(Rectangle((17.1, 0.38), 1.2, 2.0, fc=WING_L, ec=INK, lw=0.7))
        ax.add_patch(Rectangle((16.9, 1.24), 1.6, 0.34, fc=BG, ec=MAGENTA, lw=0.7))
        ax.annotate("", (18.55, 1.41), (15.9, 1.41),
                    arrowprops=dict(arrowstyle="->", color=MAGENTA, lw=1.0))
        ax.text(6.35, 2.62, "solid skin with deliberate through-slot edge",
                color=MUTED, fontsize=6.8)
    elif family in {"graded", "path_gradient"}:
        lm_depth = 8.0 if family == "graded" else 11.0
        um_depth = 3.0 if family == "graded" else 2.8
        for y, depth, color, label in (
                (1.95, lm_depth, MAGENTA, "LM"),
                (0.78, um_depth, CYAN, "UM/T")):
            ax.add_patch(Rectangle((18.3-depth, y-0.27), depth, 0.54,
                                   fc=WING_L, ec=INK, lw=0.6))
            ax.add_patch(Rectangle((18.3-depth-0.15, y-0.07), depth+0.3, 0.14,
                                   fc=BG, ec=color, lw=0.5))
            ax.text(6.35, y, f"{label}: {depth:.1f} mm open path",
                    color=color, fontsize=6.7, va="center")
    else:
        depth = 2.6
        ax.add_patch(Rectangle((18.3-depth, 0.38), depth, 2.0,
                               fc=WING_L, ec=INK, lw=0.7))
        for y in (0.76, 1.38, 2.00):
            ax.add_patch(Rectangle((18.3-depth-0.15, y-0.07), depth+0.30, 0.14,
                                   fc=BG, ec=CYAN, lw=0.45))
            ax.annotate("", (18.55, y), (18.3-depth-0.25, y),
                        arrowprops=dict(arrowstyle="->", color=CYAN, lw=0.8))
        ax.text(6.35, 2.62, "straight, open both ends; support-free",
                color=MUTED, fontsize=6.8)
    ax.set_xlabel("world z (mm)", color=MUTED, fontsize=6.5, labelpad=2)


def _facts_block(fig, spec: VariantSpec, concept: ConceptGeometry):
    fig.text(0.650, 0.875, "PLAN REALIZABILITY + ACOUSTIC VARIABLE",
             color=INK, fontsize=9.2, weight="bold")
    fig.text(0.650, 0.849, spec.hypothesis, color=INK, fontsize=8.0,
             va="top", wrap=True)
    y = 0.797
    for fact in spec.facts:
        fig.text(0.658, y, f"•  {fact}", color=MUTED, fontsize=7.25, va="top")
        y -= 0.030
    metrics = concept.metrics
    fig.text(0.650, 0.672, "MEASURED FROM THIS 2D CONCEPT",
             color=INK, fontsize=8.2, weight="bold")
    fig.text(
        0.658, 0.648,
        f"apertures / side: {metrics['apertures_per_side']}   "
        f"LM: {metrics['lm_apertures']}   UM/T: {metrics['um_apertures']}\n"
        f"local active phi — LM {metrics['lm_active_porosity_pct']:.1f}%  /  "
        f"blend {metrics['blend_active_porosity_pct']:.1f}%  /  "
        f"UM/T {metrics['um_active_porosity_pct']:.1f}%\n"
        f"whole active phi: {metrics['active_porosity_pct']:.1f}%   "
        f"gross open: {metrics['gross_open_pct']:.1f}%\n"
        "3 receiver pads + continuous spine + outer rail per side\n"
        "D5.2 pocket plan keep-outs; 3D receiver/fit qualification pending",
        color=MUTED, fontsize=6.65, va="top", linespacing=1.20)
    fig.text(0.650, 0.575, "WAVELENGTH CHECK",
             color=INK, fontsize=8.2, weight="bold")
    fig.text(
        0.658, 0.552,
        "1 kHz: lambda 343 mm / 1.05 deg per mm\n"
        "3.5 kHz: lambda 98.0 mm / 3.67 deg per mm\n"
        "4 kHz: lambda 85.8 mm / 4.20 deg per mm\n"
        "7 kHz: lambda 49.0 mm / 7.35 deg per mm\n"
        "10 kHz: lambda 34.3 mm / 10.50 deg per mm\n"
        "LM R113 = 0.33 lambda @1k; UM R51.7 = 0.60 lambda @4k = 1.06 lambda @7k",
        color=MUTED, fontsize=6.35, va="top", linespacing=1.18)
    fig.text(0.658, 0.486, spec.wavelength, color=CYAN, fontsize=7.15,
             va="top", wrap=True)


def _save_atomic(fig, output: Path, spec: VariantSpec):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    token = f"LX521_V1LF_WING_{spec.id}_{spec.slug.upper()}_CONCEPT_R2"
    try:
        fig.savefig(temporary, dpi=160, facecolor=BG,
                    metadata={
                        "Title": token,
                        "Description": (f"{token}; status=CONCEPT_UNMEASURED; "
                                        f"seed={SEED}; no_OCC=1; roots=3_per_side"),
                    })
        with Image.open(temporary) as image:
            image.verify()
        with Image.open(temporary) as image:
            image.load()
            if image.size != (2400, 1600):
                raise RuntimeError(f"unexpected PNG size {image.size}: {temporary}")
            extrema = ImageStat.Stat(image.convert("RGB")).extrema
            if not any(high-low > 40 for low, high in extrema):
                raise RuntimeError(f"apparently blank concept PNG: {temporary}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def render_variant(spec: VariantSpec, output_dir: Path):
    concept = build_concept(spec)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.0,
        "axes.titleweight": "bold", "savefig.pad_inches": 0,
    })
    fig = plt.figure(figsize=(15, 10), dpi=160, facecolor=BG)

    main = fig.add_axes((0.035, 0.065, 0.585, 0.845), facecolor=BG)
    _draw_concept_on_axis(main, spec, concept, (-165, 165), (-18, 520))
    main.set_xticks(np.arange(-150, 151, 50))
    main.set_yticks(np.arange(0, 501, 50))
    main.set_xlabel("world x (mm)", color=MUTED, fontsize=7)
    main.set_ylabel("world y (mm)", color=MUTED, fontsize=7)
    main.text(-158, 18, "green = REAL SOLID root/spine/outer-rail topology",
              color=STRUCT, fontsize=7.2, rotation=73, ha="left", va="bottom")
    main.text(4, 18, "independent mirrored modules", color=MUTED,
              fontsize=6.8, rotation=90, va="bottom")

    lm_zoom = fig.add_axes((0.646, 0.255, 0.165, 0.180), facecolor=BG)
    _draw_concept_on_axis(lm_zoom, spec, concept, (-145, 145), (82, 325),
                          labels=False, grid=False)
    lm_zoom.set_xticks([]); lm_zoom.set_yticks([])
    lm_zoom.set_title("BOTTOM / LM — TWO ROOTS PER SIDE", color=INK,
                      fontsize=7.2, pad=4)
    um_zoom = fig.add_axes((0.823, 0.255, 0.153, 0.180), facecolor=BG)
    _draw_concept_on_axis(um_zoom, spec, concept, (-88, 88), (303, 445),
                          labels=False, grid=False)
    um_zoom.set_xticks([]); um_zoom.set_yticks([])
    um_zoom.set_title("TOP / UM — ONE ROOT PER SIDE", color=INK,
                      fontsize=7.2, pad=4)

    section = fig.add_axes((0.648, 0.055, 0.327, 0.165), facecolor=PANEL)
    _draw_section(section, spec)
    _facts_block(fig, spec, concept)

    fig.text(0.040, 0.955, spec.id, color=WING_L,
             fontsize=25, weight="bold", va="top")
    title_x = 0.112 if len(spec.id) == 3 else 0.092
    fig.text(title_x, 0.956, spec.title, color=INK,
             fontsize=17, weight="bold", va="top")
    fig.text(title_x, 0.926, spec.slug, color=MUTED,
             fontsize=9.2, family="monospace", va="top")
    fig.text(0.968, 0.950, "CONCEPT / UNMEASURED",
             color=RED, fontsize=10.0, weight="bold", ha="right", va="top")
    fig.text(0.968, 0.926, "NOT FULL-3D COLLISION CHECKED  •  NOT ACOUSTICALLY QUALIFIED",
             color=RED, fontsize=7.1, ha="right", va="top")
    fig.text(
        0.040, 0.020,
        "Pure-2D feasibility gate passed  •  front z=18.3  •  rear limit z=6.8  •  "
        "B2 field + common UM/root exceptions  •  0.60 mm eroded skeleton remains connected  •  seed 521406",
        color=MUTED, fontsize=7.1, va="bottom")

    output = output_dir / f"v1lf_wing_{spec.slug}_concept.png"
    _save_atomic(fig, output, spec)
    plt.close(fig)
    print(f"wrote {output}")
    return output


def render_index(outputs: list[Path], output_dir: Path):
    width, height = 2400, 1600
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except OSError:
        title_font = label_font = small_font = ImageFont.load_default()
    draw.text((54, 34), "V1LF ACOUSTIC WINGS — 2D PLAN-REALIZABLE LM + UM MATRIX",
              fill=INK, font=title_font)
    draw.text((54, 88),
              "Every side joins LM-lower + LM-upper + UM in solid XY structure; 3D pockets, assembly keep-outs and acoustics remain unqualified.",
              fill=MUTED, font=small_font)
    columns, rows = 4, 4
    gutter = 22
    cell_w = (width - 2*54 - (columns-1)*gutter)//columns
    cell_h = (height - 150 - 45 - (rows-1)*gutter)//rows
    thumb_h = cell_h - 42
    for index, (spec, path) in enumerate(zip(VARIANTS, outputs)):
        row, column = divmod(index, columns)
        x = 54 + column*(cell_w + gutter)
        y = 142 + row*(cell_h + gutter)
        with Image.open(path) as source:
            image = source.convert("RGB")
            # Crop to the dominant front specimen so the matrix compares geometry.
            crop = image.crop((45, 125, 1030, 1490))
            crop.thumbnail((cell_w, thumb_h), Image.Resampling.LANCZOS)
        px = x + (cell_w-crop.width)//2
        py = y + 34 + (thumb_h-crop.height)//2
        canvas.paste(crop, (px, py))
        draw.text((x, y), f"{spec.id}  {spec.slug}", fill=INK, font=label_font)
    draw.text((54, height-34),
              "Green = protected solid load path; amber = acoustic field; magenta rings = D5.2 receiver stations.",
              fill=MUTED, font=small_font)
    output = output_dir / "v1lf_wing_concepts_index.png"
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    canvas.save(temporary, pnginfo=None)
    with Image.open(temporary) as image:
        if image.size != (2400, 1600):
            raise RuntimeError(f"unexpected index size {image.size}")
    temporary.replace(output)
    print(f"wrote {output}")
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("review"))
    parser.add_argument("--variant", choices=[variant.slug for variant in VARIANTS])
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    selected = [variant for variant in VARIANTS
                if args.variant is None or variant.slug == args.variant]
    if args.validate_only:
        for spec in selected:
            concept = build_concept(spec)
            print(f"validated {spec.id} {spec.slug}: {concept.metrics}")
        return 0
    outputs = [render_variant(spec, args.output_dir) for spec in selected]
    if args.variant is None:
        render_index(outputs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
