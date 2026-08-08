"""STEP-first flat/graded acoustic wings for the extreme Obi-Wan carrier.

This module turns the approved analytic A-plan contract in
``gen_obiwan_wing_design_map.py`` into release geometry.  flat is the constant
11.5 mm reference.  graded has the same flat front datum and exact plan envelope,
but its rear is one directly controlled cubic tensor B-spline driven by the
approved LM/UM/tweeter-weighted depth field.  Full-depth Obi-Wan lands and the
one-layer free edge are enforced directly on that datum-clipped rear graph.

The canonical object is one monolithic wing per side.  The three printable
pieces are intersections of that finalized object with the approved V1L-style
dovetail masks; they are never independently surfaced or reconstructed.

Geometry construction is intentionally guarded and remote-first.  This module
imports OCC/build123d and therefore belongs inside the guarded worker; every
OCC-producing public function also requires ``run_memory_guarded.py``.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union

from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepGProp import BRepGProp
from OCP.BRepClass3d import BRepClass3d_SolidClassifier
from OCP.BRepLib import BRepLib
from OCP.GProp import GProp_GProps
from OCP.Geom import Geom_BSplineSurface
from OCP.Geom2d import Geom2d_Line
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Dir, gp_Dir2d, gp_Lin, gp_Pnt, gp_Pnt2d
from OCP.Precision import Precision
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TopAbs import TopAbs_IN

from build123d import (
    Axis,
    Compound,
    Face,
    Part,
    Plane,
    Wire,
    extrude,
    import_brep,
    mirror,
)

import gen_obiwan_wing_design_map as contract
from ..base import STAND_FOOT, THICKNESS_MM
from .carriers import (
    CORE_REAR_Z,
    SIDE_MAGNET_DEPTH,
    SIDE_MAGNET_POCKET_D,
    _axis_cylinder,
    _plan_prism,
    side_magnet_sites,
)
from .lm_split import (
    REGISTRATION_WING_CLEARANCE_MM,
    registration_wing_clearance_tools,
)
from ..magnets import (
    CAPTIVE_LAND_MM,
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    INNER_SKIN_MM,
    NOMINAL_PAIRED_FACE_SEPARATION_MM,
    wall_cavity_tools,
)


VARIANT_IDS = ("flat", "graded")
SIDE_NAMES = ("left", "right")
PRINT_PART_KEYS = ("lm_lower", "lm_upper", "um")
TWO_PIECE_PRINT_PART_KEYS = ("lm_lower", "lm_um_upper")

FRONT_Z_MM = float(THICKNESS_MM)
REAR_LIMIT_Z_MM = float(CORE_REAR_Z)
FULL_DEPTH_MM = float(contract.DEPTH_MM)
GRADED_EDGE_DEPTH_MM = float(contract.GRADED_EDGE_DEPTH_MM)
MAGNET_FACE_GAP_MM = float(contract.MAGNET_FACE_GAP_MM)
MAGNET_POCKET_DIAMETER_MM = float(CAVITY_DIAMETER_MM)
MAGNET_POCKET_DEPTH_MM = float(CAVITY_DEPTH_MM)
MAGNET_CAPTIVE_LAND_MM = float(CAPTIVE_LAND_MM)

GRADED_SURFACE_GRID_X = 145
GRADED_SURFACE_GRID_Y = 361
GRADED_SURFACE_GRID_PADDING_MM = 4.0
GRADED_SURFACE_SPLINE_DEGREE = 3
GRADED_SURFACE_OUTSIDE_EDGE_SLOPE = 1.0
GRADED_SURFACE_OUTSIDE_MIN_DEPTH_MM = -2.0
GRADED_EXACT_EDGE_BAND_MM = 0.12
GRADED_RELIEF_BOUNDARY_GUARD_MM = 0.08
# How far inside the plan the clamp samples the law when it continues the
# cutter floor across the perimeter overshoot. One boundary guard width: far
# enough in that the analytic relief is on solid ground, close enough that the
# value carried outward is the boundary value and not an interior one.
GRADED_SURFACE_CLAMP_PROBE_MM = GRADED_RELIEF_BOUNDARY_GUARD_MM
GRADED_EDGE_BOOLEAN_OVERSHOOT_MM = 0.20
# Preserve the 0.04-mm protected-land collar that keeps the fitted rear
# transition below the 0.03-mm C0 gate. The new G1 lower root creates one
# acute relief tip where this normal inset corresponds to 0.186 mm measured
# along the tip bisector. That tiny region gets an explicit, bounded exception
# below; the established 0.08-mm gate remains in force everywhere else.
GRADED_BOOLEAN_RELIEF_INSET_MM = 0.04
GRADED_BOOLEAN_RELIEF_SIMPLIFY_MM = 0.015
GRADED_BOOLEAN_RELIEF_GENERAL_MAX_HAUSDORFF_MM = 0.08
GRADED_BOOLEAN_RELIEF_ROOT_EXCEPTION_RADIUS_MM = 0.25
GRADED_BOOLEAN_RELIEF_ROOT_MAX_HAUSDORFF_MM = 0.20
# A protected receiver at the new, narrower G1 lower root encloses one tiny
# disconnected component of the *candidate* graded rear-relief plan.  Cutting a
# 3.87-mm2 island would create a fragile isolated rear pocket for negligible
# mass benefit.  Retain it at full depth, but fail if this narrowly scoped
# topological allowance ever grows into meaningful acoustic relief.
GRADED_RELIEF_DISCONNECTED_COMPONENT_MAX_AREA_MM2 = 4.0
GRADED_RELIEF_DISCONNECTED_TOTAL_MAX_AREA_MM2 = 4.0
GRADED_BOOLEAN_EDGE_EXTENSION_MM = 0.13
GRADED_BOOLEAN_EDGE_EXTENSION_INSET_MM = 0.003
GRADED_BOOLEAN_EDGE_EXTENSION_SIMPLIFY_MM = 0.001
# The relief cutter carries past the exact outline on every exposed boundary
# instead of stopping inside the perimeter guard. The blank owns the outline,
# so the part beyond it removes nothing; what it removes is the full-depth
# ledge the guard band used to leave standing. 0.20 mm matches the existing
# GRADED_EDGE_BOOLEAN_OVERSHOOT_MM precedent and comfortably clears the
# 0.08-mm guard plus the 0.04-mm normal inset. Measured on the right field:
# the uncut band inside the plan falls from 42.049 mm2 to 0.000595 mm2, and
# the result is unchanged from 0.15 mm to 0.50 mm of overshoot, so the healing
# is complete rather than marginal at this setting.
GRADED_RELIEF_PERIMETER_OVERSHOOT_MM = 0.20
# Residual uncut area allowed inside the plan outside the retained keep-out.
# The measured residue is a single 0.000595 mm2 speck at the plan's extreme-y
# corner, where the mitre cap cannot close a sub-degree wedge; 0.01 mm2 keeps
# a 17x margin over it while still catching any real ledge, which measured
# 42 mm2 before this change. Keeping only the connected cutter body leaves a
# 0.2721 mm2 residue at the acute lower root, inside the root exception the
# Hausdorff gate below already recognises. The bound is set just above that
# measurement; the ledge this gate exists to catch measured 42 mm2.
GRADED_RELIEF_MAX_UNCUT_BAND_MM2 = 0.35
# The cutter must never reach material the design retains at full depth.
# Measured 0.000000 mm2; this bound exists only to absorb Shapely predicate
# noise, not to license a bite.
GRADED_RELIEF_MAX_RETAINED_BITE_MM2 = 1.0e-6
# Overshoot corridor shed when protected land meeting the outline splits the
# ring. Measured 1.2657 mm2, all of it at the acute lower root.
GRADED_RELIEF_MAX_DISCARDED_OVERSHOOT_MM2 = 2.0
GRADED_CUTTER_CONTROL_MIN_DEPTH_MM = GRADED_EDGE_DEPTH_MM + 0.02
GRADED_CUTTER_MIN_PLAN_OVERLAP_MM = 0.02
GRADED_PROTECTED_BOUNDARY_SAMPLE_SPACING_MM = 0.50
GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM = 0.004
GRADED_PROTECTED_BOUNDARY_MAX_C0_JUMP_MM = 0.03
GRADED_PROTECTED_CONSTRAINT_SPACING_MM = 0.25
# The relief surface meets the exact full-depth blank with a deliberately tiny
# 0.005-mm C0 easing.  Protected land itself is never Booleaned: it remains
# untouched material in the exact blank.
GRADED_PROTECTED_CONSTRAINT_TARGET_MM = FULL_DEPTH_MM - 0.005
GRADED_PROTECTED_CONSTRAINT_TOL_MM = 0.004
GRADED_PROTECTED_COLLAR_OFFSETS_MM = (0.25, 0.50, 1.00, 2.00, 4.00)
# A print mask that ends exactly on the cubic graded exterior can leave OCC with a
# coincident face/edge classification and no intersection solid. Extend only
# the part of each mask that lies outside the finalized wing plan. Interior
# dovetail ownership and fit-clearance gaps therefore remain exact, while the
# Boolean tool crosses every exposed perimeter by a deterministic margin.
PRINT_MASK_EXTERIOR_OVERSHOOT_MM = 0.05
GRADED_PROTECTED_GHOST_MAX_DEPTH_MM = 2.0 * FULL_DEPTH_MM - GRADED_EDGE_DEPTH_MM
GRADED_PROTECTED_GHOST_REGULARIZATION = 1.0e-5
GRADED_PROTECTED_OUTWARD_REGULARIZATION = 1.0e-3
GRADED_PROTECTED_MAX_COLLAR_ERROR_MM = 0.75
GRADED_PROTECTED_COLLAR_ERROR_HARD_LIMIT_MM = 2.0
GRADED_PROTECTED_COLLAR_OPTIMUM_MARGIN_MM = 0.01
GRADED_PROTECTED_COLLAR_MAX_REVERSAL_MM = 0.02
ADAPTIVE_VOLUME_EPS = 1.0e-6
ADAPTIVE_VOLUME_MAX_REACHED_ERROR = 5.0e-6

_A_DEFINITION = next(item for item in contract.VARIANTS if item.key == "A")


def _require_guarded_build() -> None:
    """Reject accidental local/in-process OCC construction."""
    import run_memory_guarded as memory_guard
    memory_guard.require_guarded_build(
        "flat/graded Obi-Wan wing construction requires run_memory_guarded.py; "
        "use the remote-first Make/export workflow")


def _normalize_variant(variant_id: str) -> str:
    slug = str(variant_id).strip().lower()
    if slug not in VARIANT_IDS:
        raise ValueError(
            f"unknown Obi-Wan basic wing {variant_id!r}; expected "
            f"one of {VARIANT_IDS}")
    return slug


def adaptive_volume_mm3(shape) -> float:
    """Return a guarded adaptive exact-BREP volume in cubic millimetres.

    build123d's default ``Shape.volume`` uses OCC's non-adaptive integration.
    That is accurate for flat's planar prism, but it can miss roughly 0.8% on
    graded's densely trimmed tensor B-spline.  The adaptive BRepGProp overload is
    the release oracle for both serialized volume facts and STL parity.

    Compound inputs are deliberately evaluated one solid at a time.  This
    keeps the semantics additive and makes the closed-solid/error checks
    explicit for every member.
    """
    _require_guarded_build()
    solids = list(shape.solids())
    if not solids:
        return 0.0
    total = 0.0
    for solid in solids:
        properties = GProp_GProps()
        reached_error = float(BRepGProp.VolumeProperties_s(
            solid.wrapped, properties, ADAPTIVE_VOLUME_EPS, True, False))
        volume = float(properties.Mass())
        if (not np.isfinite(reached_error)
                or reached_error > ADAPTIVE_VOLUME_MAX_REACHED_ERROR):
            raise RuntimeError(
                "adaptive BREP volume integration missed its error gate: "
                f"reached={reached_error:.9g}, "
                f"limit={ADAPTIVE_VOLUME_MAX_REACHED_ERROR:.9g}")
        if not np.isfinite(volume) or volume <= 0.0:
            raise RuntimeError(
                f"adaptive BREP volume is not finite/positive: {volume!r}")
        total += volume
    return total


def _normalize_side(side: str) -> str:
    value = str(side).strip().lower()
    if value not in SIDE_NAMES:
        raise ValueError(f"unknown wing side {side!r}; expected {SIDE_NAMES}")
    return value


def _highs_thread_budget() -> int:
    """Return a bounded inner-solver budget for the guarded remote host."""
    configured = os.environ.get("LX_CAD_HIGHS_THREADS")
    if configured is not None:
        try:
            value = int(configured)
        except ValueError as exc:
            raise ValueError("LX_CAD_HIGHS_THREADS must be an integer") from exc
        if not 1 <= value <= 64:
            raise ValueError("LX_CAD_HIGHS_THREADS must be between 1 and 64")
        return value
    if os.environ.get("LX_CAD_MEMORY_PROFILE") != "osado-512g":
        return 1
    try:
        slots = max(1, int(os.environ.get("LX_CAD_GUARD_SLOTS", "4")))
    except ValueError:
        slots = 4
    return max(1, min(8, (os.cpu_count() or 1) // slots))


def _mirror_plan(geometry):
    return affinity.scale(geometry, xfact=-1.0, yfact=1.0, origin=(0.0, 0.0))


@lru_cache(maxsize=1)
def _layout():
    """Approved A-plan, receiver roots and both print partitions."""
    return contract._build_layout(_A_DEFINITION)


@lru_cache(maxsize=1)
def _graded_analytics():
    """Approved graded solution plus all drawing-time analytic build gates."""
    layout = _layout()
    solution = contract._optimize_graded_profile()
    depth_field = contract._build_graded_depth_field(layout, solution)
    sections = contract._graded_section_definitions(layout)
    contract._validate_graded_section_monotonicity(
        layout, depth_field, solution, sections)
    return solution, depth_field, sections


def wing_plan(variant_id: str, side: str):
    """Return the authoritative installed XY plan for one monolithic wing."""
    _normalize_variant(variant_id)  # flat and graded deliberately share the plan.
    side = _normalize_side(side)
    plan = _layout().field_right
    return plan if side == "right" else _mirror_plan(plan)


def wing_print_plan_parts(variant_id: str, side: str) -> dict[str, Polygon]:
    """Return the exact XY masks used to qualify the split print solids."""
    _normalize_variant(variant_id)
    side = _normalize_side(side)
    result = _layout().print_parts
    if side == "right":
        return dict(result)
    return {key: _mirror_plan(value) for key, value in result.items()}


def wing_two_piece_print_plan_parts(
        variant_id: str, side: str) -> dict[str, Polygon]:
    """Return the exact XY masks for the alternative two-piece wing."""
    _normalize_variant(variant_id)
    side = _normalize_side(side)
    result = _layout().two_piece_print_parts
    if side == "right":
        return dict(result)
    return {key: _mirror_plan(value) for key, value in result.items()}


def _selected_sites(side: str) -> tuple[dict, ...]:
    side = _normalize_side(side)
    suffix = f"_{side}"
    expected_names = {
        f"lm_lower{suffix}", f"lm_upper{suffix}", f"um{suffix}"}
    selected = tuple(
        site for site in side_magnet_sites()
        if site["name"] in expected_names)
    if len(selected) != 3 or {site["name"] for site in selected} != expected_names:
        raise RuntimeError(
            f"Obi-Wan basic receiver contract drifted for {side}: "
            f"{[site['name'] for site in selected]}")
    order = {name: index for index, name in enumerate((
        f"lm_lower{suffix}", f"lm_upper{suffix}", f"um{suffix}"))}
    return tuple(sorted(selected, key=lambda site: order[site["name"]]))


def _receiver_datum_face(site) -> tuple[float, float]:
    """Visible carrier surface used by the mating wing receiver."""
    face = site.get("outer_surface_face", site["face"])
    return float(face[0]), float(face[1])


def receiver_facts(side: str) -> tuple[dict, ...]:
    """Serializable captive contract for all three receivers on one side."""
    records = []
    for site in _selected_sites(side):
        receiver_datum = _receiver_datum_face(site)
        tools = wall_cavity_tools(
            name=str(site["name"]),
            face=receiver_datum,
            outward=(*site["normal"], 0.0),
            owner="wing",
            axis_z=float(site["z_mm"]),
            print_up=(0.0, 0.0, -1.0),
            front_z=FRONT_Z_MM,
            interface_gap_mm=MAGNET_FACE_GAP_MM,
        )
        record = dict(tools.facts())
        record.update({
            "driver": str(site["driver"]),
            "axis_normal_xy": [float(value) for value in site["normal"]],
            "carrier_face_xy_mm": list(receiver_datum),
            "carrier_cavity_datum_xy_mm": [
                float(value) for value in site["face"]],
            "receiver_cavity_face_xy_mm": list(tools.actual_face_xyz[:2]),
            "axis_z_mm": tools.actual_face_xyz[2],
            "cavity_diameter_mm": MAGNET_POCKET_DIAMETER_MM,
            "cavity_depth_mm": MAGNET_POCKET_DEPTH_MM,
            "captive_land_mm": MAGNET_CAPTIVE_LAND_MM,
            "receiver_solid_standoff_mm": MAGNET_FACE_GAP_MM,
            "physical_interface_gap_mm": 0.0,
            "receiver_spacing_standoff_is_solid": True,
            "paired_magnet_face_separation_mm": round(
                NOMINAL_PAIRED_FACE_SEPARATION_MM
                + float(site.get("carrier_cavity_face_inset_mm", 0.0)), 9),
            "orientation": (
                "shoulder_normal_axis_tangential_diameter"
                if site.get("interface_kind") == "shoulder"
                else "radial_axis_tangential_diameter"),
            "interface_kind": str(site.get("interface_kind", "ring")),
            "carrier_magnet_fully_buried": True,
            "receiver_magnet_fully_buried": True,
        })
        records.append(record)
    return tuple(records)


def receiver_pockets(side: str) -> dict[str, object]:
    """Return unioned coupon-style cutter groups for one side."""
    _require_guarded_build()
    result = {}
    for site in _selected_sites(side):
        tools = wall_cavity_tools(
            name=str(site["name"]),
            face=_receiver_datum_face(site),
            outward=(*site["normal"], 0.0),
            owner="wing",
            axis_z=float(site["z_mm"]),
            print_up=(0.0, 0.0, -1.0),
            front_z=FRONT_Z_MM,
            interface_gap_mm=MAGNET_FACE_GAP_MM,
        )
        cutter = tools.cutters[0]
        for component in tools.cutters[1:]:
            cutter = cutter.fuse(component)
        result[site["name"]] = cutter.clean()
    return result


def receiver_required_lands(side: str) -> dict[str, object]:
    """Return lands that the immutable receiver host must already contain."""
    _require_guarded_build()
    return {
        str(site["name"]): wall_cavity_tools(
            name=str(site["name"]),
            face=_receiver_datum_face(site),
            outward=(*site["normal"], 0.0),
            owner="wing",
            axis_z=float(site["z_mm"]),
            print_up=(0.0, 0.0, -1.0),
            front_z=FRONT_Z_MM,
            interface_gap_mm=MAGNET_FACE_GAP_MM,
        ).required_land
        for site in _selected_sites(side)
    }


def _normalize_xy(points: Iterable) -> np.ndarray:
    values = np.asarray(points, dtype=float)
    if values.shape == (2,):
        values = values.reshape((1, 2))
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("XY samples must be one (x,y) pair or an Nx2 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("XY samples must be finite")
    return values


def _right_depth_vector(slug: str, xy: np.ndarray) -> np.ndarray:
    if slug == "flat":
        return np.full(len(xy), FULL_DEPTH_MM, dtype=float)
    solution, depth_field, _sections = _graded_analytics()
    cloud = shapely.points(xy[:, 0], xy[:, 1])
    depth = contract._graded_weighted_depth(
        cloud, depth_field.protected_components,
        depth_field.exposed_outer_edge, solution)
    return np.clip(np.asarray(depth, dtype=float),
                   GRADED_EDGE_DEPTH_MM, FULL_DEPTH_MM)


def wing_depth_at(
        variant_id: str, side: str, xy: Iterable) -> tuple[float, ...]:
    """Evaluate the approved local material depth at installed XY points."""
    slug = _normalize_variant(variant_id)
    side = _normalize_side(side)
    samples = _normalize_xy(xy).copy()
    if side == "left":
        samples[:, 0] *= -1.0
    return tuple(float(value) for value in _right_depth_vector(slug, samples))


def wing_section_samples(
        variant_id: str, side: str, *, samples: int = 481) -> dict[str, dict]:
    """Return S1--S5 analytic sections used by the monotonic build gate."""
    slug = _normalize_variant(variant_id)
    side = _normalize_side(side)
    if samples < 9:
        raise ValueError("section sampling requires at least 9 points")
    layout = _layout()
    solution, depth_field, sections = _graded_analytics()
    result: dict[str, dict] = {}
    for definition in sections:
        along, graded_depth, xy, _segment = contract._sample_graded_section(
            layout, depth_field, solution, definition, samples=samples)
        if definition.key in ("S1", "S2", "S3", "S4"):
            along, graded_depth, xy = contract._orient_graded_section_protected_to_edge(
                depth_field, along, graded_depth, xy, definition)
        depth = (np.full_like(graded_depth, FULL_DEPTH_MM)
                 if slug == "flat" else np.asarray(graded_depth, dtype=float))
        if side == "left":
            xy = np.asarray(xy, dtype=float).copy()
            xy[:, 0] *= -1.0
        running_minimum = np.minimum.accumulate(depth)
        worst_reversal = float(np.max(depth - running_minimum))
        result[definition.key] = {
            "title": definition.title,
            "distance_mm": [float(value) for value in along],
            "xy_mm": [[float(x), float(y)] for x, y in xy],
            "depth_mm": [float(value) for value in depth],
            "start_depth_mm": float(depth[0]),
            "end_depth_mm": float(depth[-1]),
            "minimum_depth_mm": float(np.min(depth)),
            "maximum_depth_mm": float(np.max(depth)),
            "worst_depth_reversal_mm": worst_reversal,
            "monotonic_nonincreasing": (
                worst_reversal <= contract.GRADED_SECTION_MONOTONIC_TOL_MM),
            "pocket_axis_z_mm": (
                None if definition.pocket_z_mm is None
                else float(definition.pocket_z_mm)),
        }
    return result


def _open_uniform_spline_axis(
        count: int, degree: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Greville stations plus OCC knots/multiplicities.

    Placing X/Y poles at the Greville abscissae makes the clamped B-spline
    reproduce both Cartesian coordinates exactly.  Only Z is shaped by the
    local control net, so the rear remains a single-valued graph without the
    global interpolation overshoot of ``GeomAPI_PointsToBSplineSurface``.
    """
    if count <= degree:
        raise ValueError("B-spline control count must exceed its degree")
    last_knot = count - degree
    knots = np.arange(last_knot + 1, dtype=float)
    multiplicities = np.ones(last_knot + 1, dtype=int)
    multiplicities[0] = multiplicities[-1] = degree + 1
    full_knots = np.concatenate((
        np.zeros(degree + 1, dtype=float),
        np.arange(1, last_knot, dtype=float),
        np.full(degree + 1, float(last_knot)),
    ))
    greville = np.asarray([
        np.sum(full_knots[index + 1:index + degree + 1]) / degree
        for index in range(count)
    ], dtype=float)
    return greville / float(last_knot), knots, multiplicities


def _occ_real_array(values: np.ndarray) -> TColStd_Array1OfReal:
    result = TColStd_Array1OfReal(1, len(values))
    for index, value in enumerate(values, start=1):
        result.SetValue(index, float(value))
    return result


def _occ_integer_array(values: np.ndarray) -> TColStd_Array1OfInteger:
    result = TColStd_Array1OfInteger(1, len(values))
    for index, value in enumerate(values, start=1):
        result.SetValue(index, int(value))
    return result


def _graded_control_depths(xy: np.ndarray) -> np.ndarray:
    """Analytic depths with a one-sided continuation outside free edges.

    A rectangular control net necessarily contains poles just beyond the
    irregular A outline.  Continuing the free-edge slope outward prevents a
    symmetric rounded minimum at the knife edge.  Poles beyond carrier and
    joint boundaries remain full-depth so every Obi-Wan contact approaches the
    exact 11.5-mm rear datum.  The final exact-plan Boolean discards all of
    these conditioning poles.
    """
    layout = _layout()
    _solution, depth_field, _sections = _graded_analytics()
    depths = _right_depth_vector("graded", xy)
    inside = shapely.contains_xy(layout.field_right, xy[:, 0], xy[:, 1])
    # Full depth outside the plan is right only where the wing really is full
    # depth at that boundary: the carrier and joint contacts, which are exactly
    # the retained lands.  Everywhere else the cutter now overshoots the
    # outline, so a full-depth pole out there lifts the cutter floor and leaves
    # the boundary ridge back.  Continue the interior instead: sample the law
    # just inside the plan and carry that value outward, so the floor runs flat
    # through the trim and the plan edge is cut by the trim alone.
    retained_land = unary_union(
        (depth_field.protected, _graded_retained_full_depth_islands())
    ).buffer(0)
    clamp_indices: list[int] = []
    clamp_probes: list[tuple[float, float]] = []
    for index in np.flatnonzero(~inside):
        point = Point(float(xy[index, 0]), float(xy[index, 1]))
        boundary_point, _unused = nearest_points(
            layout.field_right.boundary, point)
        if boundary_point.distance(
                depth_field.exposed_outer_edge) <= 2.0 * contract.GRADED_EDGE_MATCH_TOL_MM:
            outside_distance = point.distance(boundary_point)
            depths[index] = max(
                GRADED_SURFACE_OUTSIDE_MIN_DEPTH_MM,
                GRADED_EDGE_DEPTH_MM
                - GRADED_SURFACE_OUTSIDE_EDGE_SLOPE * outside_distance)
        elif boundary_point.distance(retained_land) <= (
                GRADED_BOOLEAN_RELIEF_INSET_MM):
            depths[index] = FULL_DEPTH_MM
        else:
            clamp_indices.append(int(index))
            clamp_probes.append(
                _interior_probe_xy(point, boundary_point, layout.field_right))
    if clamp_indices:
        # One batched evaluation: the law is far too slow to call per pole and
        # this branch covers most of the conditioning net.
        depths[np.asarray(clamp_indices)] = _right_depth_vector(
            "graded", np.asarray(clamp_probes, dtype=float))
    return depths


def _interior_probe_xy(point: Point, boundary_point: Point,
                       plan) -> tuple[float, float]:
    """Return a point one guard width inside the plan from the boundary."""
    dx = boundary_point.x - point.x
    dy = boundary_point.y - point.y
    span = float(np.hypot(dx, dy))
    if span <= 1.0e-12:
        return float(boundary_point.x), float(boundary_point.y)
    step = GRADED_SURFACE_CLAMP_PROBE_MM / span
    probe_x = float(boundary_point.x + dx * step)
    probe_y = float(boundary_point.y + dy * step)
    if not shapely.contains_xy(plan, probe_x, probe_y):
        return float(boundary_point.x), float(boundary_point.y)
    return probe_x, probe_y


def _expanded_spline_knots(count: int, degree: int) -> np.ndarray:
    last_knot = count - degree
    return np.concatenate((
        np.zeros(degree + 1, dtype=float),
        np.arange(1, last_knot, dtype=float),
        np.full(degree + 1, float(last_knot)),
    ))


def _axis_spline_basis(
        coordinate: float, axis_min: float, axis_max: float,
        count: int, degree: int) -> tuple[tuple[int, float], ...]:
    """Evaluate the non-zero clamped basis functions on one axis."""
    last_knot = count - degree
    parameter = np.clip(
        (float(coordinate) - axis_min) / (axis_max - axis_min) * last_knot,
        0.0, float(last_knot))
    full_knots = _expanded_spline_knots(count, degree)
    if parameter >= last_knot:
        span = count - 1
    else:
        span = int(np.searchsorted(full_knots, parameter, side="right") - 1)
        span = min(count - 1, max(degree, span))
    basis = np.zeros(degree + 1, dtype=float)
    left = np.zeros(degree + 1, dtype=float)
    right = np.zeros(degree + 1, dtype=float)
    basis[0] = 1.0
    for order in range(1, degree + 1):
        left[order] = parameter - full_knots[span + 1 - order]
        right[order] = full_knots[span + order] - parameter
        saved = 0.0
        for index in range(order):
            denominator = right[index + 1] + left[order - index]
            term = 0.0 if abs(denominator) <= 1.0e-15 else (
                basis[index] / denominator)
            basis[index] = saved + right[index + 1] * term
            saved = left[order - index] * term
        basis[order] = saved
    indices = range(span - degree, span + 1)
    return tuple(
        (index, float(value))
        for index, value in zip(indices, basis, strict=True)
        if value > 1.0e-14
    )


def _tensor_spline_basis(
        x_mm: float, y_mm: float, x_axis: np.ndarray,
        y_axis: np.ndarray) -> tuple[tuple[int, int, int, float], ...]:
    x_basis = _axis_spline_basis(
        x_mm, float(x_axis[0]), float(x_axis[-1]),
        len(x_axis), GRADED_SURFACE_SPLINE_DEGREE)
    y_basis = _axis_spline_basis(
        y_mm, float(y_axis[0]), float(y_axis[-1]),
        len(y_axis), GRADED_SURFACE_SPLINE_DEGREE)
    return tuple(
        (ix * len(y_axis) + iy, ix, iy, bx * by)
        for ix, bx in x_basis
        for iy, by in y_basis
        if bx * by > 1.0e-14
    )


def _protected_boundary_records(
        spacing_mm: float,
        ) -> tuple[tuple[float, float, float, float], ...]:
    """Return boundary XY and the locally oriented protected-side normal."""
    _solution, depth_field, _sections = _graded_analytics()
    plan = _layout().field_right
    external_guard = plan.boundary.buffer(
        contract.GRADED_EDGE_MATCH_TOL_MM, cap_style=2, join_style=2)
    transition = depth_field.protected.boundary.difference(external_guard)
    records = []
    plan_with_tolerance = plan.buffer(1.0e-6)
    for line in _shapely_line_parts(transition):
        sample_count = max(1, int(np.ceil(line.length / spacing_mm)))
        for sample_index in range(sample_count):
            distance = line.length * (
                sample_index + 0.5) / float(sample_count)
            tangent_half_span = min(0.05, 0.20 * line.length)
            before = line.interpolate(max(0.0, distance - tangent_half_span))
            after = line.interpolate(min(line.length, distance + tangent_half_span))
            tx = float(after.x - before.x)
            ty = float(after.y - before.y)
            tangent_length = float(np.hypot(tx, ty))
            if tangent_length <= 1.0e-8:
                raise RuntimeError(
                    "graded protected fitting perimeter has a degenerate tangent")
            nx = -ty / tangent_length
            ny = tx / tangent_length
            boundary = line.interpolate(distance)
            plus = Point(
                boundary.x + GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary.y + GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * ny)
            minus = Point(
                boundary.x - GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary.y - GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * ny)
            plus_inside = depth_field.protected.covers(plus)
            minus_inside = depth_field.protected.covers(minus)
            if plus_inside == minus_inside:
                raise RuntimeError(
                    "graded protected fitting probe did not straddle the land at "
                    f"({boundary.x:.6f}, {boundary.y:.6f})")
            inside = plus if plus_inside else minus
            outside = minus if plus_inside else plus
            if (not plan_with_tolerance.covers(inside)
                    or not plan_with_tolerance.covers(outside)):
                raise RuntimeError(
                    "graded protected fitting probe left the plan at "
                    f"({boundary.x:.6f}, {boundary.y:.6f})")
            if not plus_inside:
                nx = -nx
                ny = -ny
            records.append((
                float(boundary.x), float(boundary.y), float(nx), float(ny)))
    if len(records) < 20:
        raise RuntimeError("graded protected fitting perimeter is implausibly short")
    return tuple(records)


def _fit_graded_protected_control_net(
        control_depth: np.ndarray, x_axis: np.ndarray,
        y_axis: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Fit local ghost controls so narrow full-depth lands stay non-flat.

    Analytic Greville samples are not interpolation constraints.  A narrow
    1.2-mm Obi-Wan land can therefore fall between stations and be several
    millimetres thinner in the evaluated spline.  Promoting every active pole
    to 11.5 mm would fix that mismatch only by creating a broad shelf.  This
    sparse bounded fit instead allows overshoot solely in controls that lie on
    the protected side of every boundary row they support.  Mixed/outward
    controls remain capped at the physical envelope and are penalized 100x.
    The evaluated surface, rather than its ghost ordinates, is subsequently
    clipped and audited.
    """
    from scipy.optimize import OptimizeWarning, linprog, lsq_linear
    from scipy.sparse import coo_matrix, hstack, vstack

    records = _protected_boundary_records(
        GRADED_PROTECTED_CONSTRAINT_SPACING_MM)
    plan = _layout().field_right
    _solution, depth_field, _sections = _graded_analytics()
    protected = depth_field.protected
    plan_with_tolerance = plan.buffer(1.0e-6)
    seed = np.asarray(control_depth, dtype=float).ravel().copy()
    ny_count = len(y_axis)
    observations: dict[int, list[float]] = {}
    active_indices: set[int] = set()
    boundary_entries = []
    for bx, by, nx, ny in records:
        entries = _tensor_spline_basis(bx, by, x_axis, y_axis)
        boundary_entries.append(entries)
        for flat_index, ix, iy, _coefficient in entries:
            active_indices.add(flat_index)
            projection = (
                (float(x_axis[ix]) - bx) * nx
                + (float(y_axis[iy]) - by) * ny)
            observations.setdefault(flat_index, []).append(projection)

    # Include enough neighboring knot rings to cover the complete 4-mm
    # analytic collar.  A fixed one-ring halo was adequate on the coarse net,
    # but became physically too narrow after local-resolution refinement.
    candidate_indices = set(active_indices)
    x_span_mm = (
        float(x_axis[-1] - x_axis[0])
        / float(len(x_axis) - GRADED_SURFACE_SPLINE_DEGREE))
    y_span_mm = (
        float(y_axis[-1] - y_axis[0])
        / float(len(y_axis) - GRADED_SURFACE_SPLINE_DEGREE))
    maximum_collar = max(GRADED_PROTECTED_COLLAR_OFFSETS_MM)
    x_ring_count = int(np.ceil(maximum_collar / x_span_mm)) + 1
    y_ring_count = int(np.ceil(maximum_collar / y_span_mm)) + 1
    for flat_index in tuple(active_indices):
        ix, iy = divmod(flat_index, ny_count)
        for dix in range(-x_ring_count, x_ring_count + 1):
            for diy in range(-y_ring_count, y_ring_count + 1):
                jx, jy = ix + dix, iy + diy
                if 0 <= jx < len(x_axis) and 0 <= jy < len(y_axis):
                    candidate_indices.add(jx * ny_count + jy)

    ghost_indices = {
        index for index, projections in observations.items()
        if projections and min(projections) >= -0.05
    }
    upper_by_index = {
        index: (GRADED_PROTECTED_GHOST_MAX_DEPTH_MM
                if index in ghost_indices else FULL_DEPTH_MM)
        for index in candidate_indices
    }
    # Protected-side ghosts need only rise to pull the evaluated boundary up.
    # Mixed/outward controls must also be allowed to fall, otherwise the only
    # way to reach the land is a broad thick shelf.  The final exact minimum
    # skin and horizontal envelope Booleans remain the physical Z bounds;
    # this lower control bound is only a spline-conditioning allowance.
    lower_by_index = {
        index: (seed[index] if index in ghost_indices
                else GRADED_SURFACE_OUTSIDE_MIN_DEPTH_MM)
        for index in candidate_indices
    }
    unknown_indices = tuple(sorted(
        index for index in candidate_indices
        if (upper_by_index[index] - seed[index] > 1.0e-8
            or seed[index] - lower_by_index[index] > 1.0e-8)
    ))
    if not unknown_indices or not ghost_indices:
        raise RuntimeError("graded protected spline fit has no adjustable ghosts")
    column_for = {index: column for column, index in enumerate(unknown_indices)}

    sample_specs: list[tuple[float, float, float | None, float]] = []
    collar_rays: list[tuple[tuple[float, float], ...]] = []
    for bx, by, nx, ny in records:
        sample_specs.append((
            bx, by, GRADED_PROTECTED_CONSTRAINT_TARGET_MM, 1000.0))
        ray = [(bx, by)]
        for offset, weight in zip(
                GRADED_PROTECTED_COLLAR_OFFSETS_MM,
                (0.25, 0.125, 0.05, 0.025, 0.0125), strict=True):
            px = bx - offset * nx
            py = by - offset * ny
            point = Point(px, py)
            if not plan_with_tolerance.covers(point):
                break
            # A normal leaving one protected component can enter a nearby
            # component at a larger offset.  Such a point belongs to a second
            # full-depth land, not to this boundary's outward feather.  Stop
            # the collar ray at the first re-entry so the LP never receives
            # contradictory full-depth and taper constraints.
            if protected.covers(point):
                break
            sample_specs.append((px, py, None, weight))
            ray.append((px, py))
        if len(ray) > 1:
            collar_rays.append(tuple(ray))

    analytic_indices = [
        index for index, spec in enumerate(sample_specs) if spec[2] is None
    ]
    if analytic_indices:
        analytic_xy = np.asarray([
            (sample_specs[index][0], sample_specs[index][1])
            for index in analytic_indices
        ], dtype=float)
        analytic_depth = _right_depth_vector("graded", analytic_xy)
        for index, target in zip(
                analytic_indices, analytic_depth, strict=True):
            px, py, _unused, weight = sample_specs[index]
            sample_specs[index] = (px, py, float(target), weight)
    collar_specs = tuple(
        (px, py, float(target))
        for px, py, target, weight in sample_specs
        if weight < 1000.0 and target is not None
    )

    row_indices: list[int] = []
    column_indices: list[int] = []
    matrix_values: list[float] = []
    rhs: list[float] = []
    for row, (px, py, target, weight) in enumerate(sample_specs):
        assert target is not None
        entries = _tensor_spline_basis(px, py, x_axis, y_axis)
        baseline = sum(
            coefficient * seed[flat_index]
            for flat_index, _ix, _iy, coefficient in entries)
        for flat_index, _ix, _iy, coefficient in entries:
            column = column_for.get(flat_index)
            if column is not None:
                row_indices.append(row)
                column_indices.append(column)
                matrix_values.append(weight * coefficient)
        rhs.append(weight * (float(target) - baseline))
    fitting_matrix = coo_matrix(
        (matrix_values, (row_indices, column_indices)),
        shape=(len(sample_specs), len(unknown_indices))).tocsr()

    regularization = coo_matrix((
        np.asarray([
            (GRADED_PROTECTED_GHOST_REGULARIZATION
             if index in ghost_indices
             else GRADED_PROTECTED_OUTWARD_REGULARIZATION)
            for index in unknown_indices
        ], dtype=float),
        (np.arange(len(unknown_indices)), np.arange(len(unknown_indices)))),
        shape=(len(unknown_indices), len(unknown_indices))).tocsr()
    system = vstack((fitting_matrix, regularization), format="csr")
    target_vector = np.concatenate((
        np.asarray(rhs, dtype=float),
        np.zeros(len(unknown_indices), dtype=float)))
    upper_delta = np.asarray([
        upper_by_index[index] - seed[index] for index in unknown_indices
    ], dtype=float)
    lower_delta = np.asarray([
        lower_by_index[index] - seed[index] for index in unknown_indices
    ], dtype=float)
    solution = lsq_linear(
        system, target_vector,
        bounds=(lower_delta, upper_delta),
        method="trf", tol=1.0e-7, lsmr_tol="auto", lsmr_maxiter=120,
        max_iter=120, verbose=0)
    # ``trf`` can exhaust its conservative active-set iteration budget after
    # producing a perfectly usable bounded vector.  The geometric residual,
    # collar error and monotonicity checks below are the authoritative stop
    # conditions; only a numerical failure/non-finite vector is fatal here.
    if (solution.status < 0
            or not np.all(np.isfinite(np.asarray(solution.x, dtype=float)))):
        raise RuntimeError(
            "graded protected spline fit failed: " + str(solution.message))
    fitted = seed.copy()
    fitted[np.asarray(unknown_indices, dtype=int)] += solution.x

    # The collar fit minimizes an aggregate L2 residual.  Close it with one
    # bounded global L-infinity solve so no isolated perimeter point can hide
    # behind thousands of good samples.  One signed bounded correction per
    # pole preserves the L2 collar solution as the origin without the doubled,
    # degenerate positive/negative variable formulation.
    boundary_rows: list[int] = []
    boundary_columns: list[int] = []
    boundary_values: list[float] = []
    boundary_rhs = []
    for row, entries in enumerate(boundary_entries):
        baseline = float(sum(
            coefficient * seed[flat_index]
            for flat_index, _ix, _iy, coefficient in entries))
        boundary_rhs.append(
            GRADED_PROTECTED_CONSTRAINT_TARGET_MM - baseline)
        for flat_index, _ix, _iy, coefficient in entries:
            column = column_for.get(flat_index)
            if column is not None:
                boundary_rows.append(row)
                boundary_columns.append(column)
                boundary_values.append(coefficient)
    boundary_matrix = coo_matrix(
        (boundary_values, (boundary_rows, boundary_columns)),
        shape=(len(boundary_entries), len(unknown_indices))).tocsr()
    residual_from_l2 = (
        np.asarray(boundary_rhs, dtype=float)
        - boundary_matrix @ np.asarray(solution.x, dtype=float))

    collar_rows: list[int] = []
    collar_columns: list[int] = []
    collar_values: list[float] = []
    collar_target_from_l2 = []
    for row, (px, py, target) in enumerate(collar_specs):
        entries = _tensor_spline_basis(px, py, x_axis, y_axis)
        baseline = float(sum(
            coefficient * seed[flat_index]
            for flat_index, _ix, _iy, coefficient in entries))
        current = baseline
        for flat_index, _ix, _iy, coefficient in entries:
            column = column_for.get(flat_index)
            if column is not None:
                collar_rows.append(row)
                collar_columns.append(column)
                collar_values.append(coefficient)
                current += coefficient * float(solution.x[column])
        collar_target_from_l2.append(target - current)
    collar_matrix = coo_matrix(
        (collar_values, (collar_rows, collar_columns)),
        shape=(len(collar_specs), len(unknown_indices))).tocsr()
    collar_target_from_l2 = np.asarray(collar_target_from_l2, dtype=float)
    l2_delta = np.asarray(solution.x, dtype=float)

    # Encode the required outward monotonic decrease in the LP itself.  Each
    # row is d(next)-d(previous) <= 0.02 mm along one protected-boundary ray.
    # The small allowance covers spline/STEP numerical tolerance without
    # permitting a visible outward rebound.
    monotonic_rows: list[int] = []
    monotonic_columns: list[int] = []
    monotonic_values: list[float] = []
    monotonic_rhs: list[float] = []
    monotonic_count = 0
    for ray in collar_rays:
        for previous_xy, next_xy in zip(ray[:-1], ray[1:], strict=True):
            coefficients: dict[int, float] = {}
            current_values = []
            for sign, (px, py) in ((-1.0, previous_xy), (1.0, next_xy)):
                current = 0.0
                for flat_index, _ix, _iy, coefficient in _tensor_spline_basis(
                        px, py, x_axis, y_axis):
                    current += coefficient * seed[flat_index]
                    column = column_for.get(flat_index)
                    if column is not None:
                        current += coefficient * l2_delta[column]
                        coefficients[column] = (
                            coefficients.get(column, 0.0)
                            + sign * coefficient)
                current_values.append(current)
            for column, value in coefficients.items():
                if abs(value) > 1.0e-14:
                    monotonic_rows.append(monotonic_count)
                    monotonic_columns.append(column)
                    monotonic_values.append(value)
            current_difference = current_values[1] - current_values[0]
            monotonic_rhs.append(
                GRADED_PROTECTED_COLLAR_MAX_REVERSAL_MM - current_difference)
            monotonic_count += 1
    monotonic_matrix = coo_matrix(
        (monotonic_values, (monotonic_rows, monotonic_columns)),
        shape=(monotonic_count, len(unknown_indices))).tocsr()
    monotonic_rhs_array = np.asarray(monotonic_rhs, dtype=float)
    correction_bounds = [
        (float(lower - current), float(upper - current))
        for lower, upper, current in zip(
            lower_delta, upper_delta, l2_delta, strict=True)
    ]
    highs_threads = _highs_thread_budget()
    # SciPy intentionally forwards these native HiGHS options but emits a
    # generic "unrecognized" warning because they are not part of linprog's
    # portable option surface.  They are valid in the pinned HiGHS 1.12
    # backend.  One remote Make slot gets up to eight of osado's 64 CPUs;
    # an explicit environment override remains available for diagnostics.
    def run_highs(objective, matrix, rhs_values, bounds):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Unrecognized options detected:.*",
                category=OptimizeWarning)
            return linprog(
                objective, A_ub=matrix, b_ub=rhs_values,
                bounds=bounds, method="highs-ds",
                options={
                    "time_limit": 120.0,
                    "presolve": True,
                    "threads": highs_threads,
                    "parallel": highs_threads > 1,
                    "simplex_strategy": 2,
                    "simplex_max_concurrency": highs_threads,
                    "random_seed": 0,
                })

    # First measure the best possible collar match under the actual control
    # bounds *while already enforcing the release boundary tolerance*.  The
    # ordering matters: minimizing collar error alone can consume the pole
    # freedom needed to make the exact fused land visually C0.  A free collar
    # L-infinity slack turns any remaining failure into either an impossible
    # boundary or a solver defect, never a guessed collar tolerance.
    collar_slack = coo_matrix(
        -np.ones((len(collar_specs), 1), dtype=float)).tocsr()
    collar_measure_rows = vstack((
        hstack((collar_matrix, collar_slack), format="csr"),
        hstack((-collar_matrix, collar_slack), format="csr"),
    ), format="csr")
    collar_measure_collar_rhs = np.concatenate((
        collar_target_from_l2, -collar_target_from_l2))
    boundary_zero_slack = coo_matrix(
        np.zeros((len(boundary_entries), 1), dtype=float)).tocsr()
    collar_measure_boundary_rows = vstack((
        hstack((boundary_matrix, boundary_zero_slack), format="csr"),
        hstack((-boundary_matrix, boundary_zero_slack), format="csr"),
    ), format="csr")
    collar_measure_boundary_rhs = np.concatenate((
        residual_from_l2 + GRADED_PROTECTED_CONSTRAINT_TOL_MM,
        -residual_from_l2 + GRADED_PROTECTED_CONSTRAINT_TOL_MM,
    ))
    monotonic_zero_slack = coo_matrix(
        np.zeros((monotonic_count, 1), dtype=float)).tocsr()
    collar_measure_monotonic_rows = hstack((
        monotonic_matrix, monotonic_zero_slack), format="csr")
    collar_measure_matrix = vstack((
        collar_measure_boundary_rows,
        collar_measure_monotonic_rows,
        collar_measure_rows,
    ), format="csr")
    collar_measure_rhs = np.concatenate((
        collar_measure_boundary_rhs,
        monotonic_rhs_array,
        collar_measure_collar_rhs,
    ))
    collar_measure_objective = np.concatenate((
        np.zeros(len(unknown_indices), dtype=float),
        np.asarray([1.0], dtype=float)))
    collar_measure = run_highs(
        collar_measure_objective, collar_measure_matrix,
        collar_measure_rhs, correction_bounds + [(0.0, None)])
    if not collar_measure.success:
        raise RuntimeError(
            "graded protected spline collar minimax measurement failed: "
            + collar_measure.message)
    collar_optimum = float(collar_measure.x[-1])
    if collar_optimum > GRADED_PROTECTED_COLLAR_ERROR_HARD_LIMIT_MM:
        raise RuntimeError(
            "graded protected spline collar cannot follow the analytic field: "
            f"minimum possible L-infinity error={collar_optimum:.6f} mm, "
            f"hard limit={GRADED_PROTECTED_COLLAR_ERROR_HARD_LIMIT_MM:.6f} mm")
    collar_limit = max(
        GRADED_PROTECTED_MAX_COLLAR_ERROR_MM,
        collar_optimum + GRADED_PROTECTED_COLLAR_OPTIMUM_MARGIN_MM)
    variable_count = len(unknown_indices)
    corrected_delta = l2_delta + collar_measure.x[:variable_count]
    fitted = seed.copy()
    fitted[np.asarray(unknown_indices, dtype=int)] += corrected_delta

    def evaluate(px: float, py: float) -> float:
        return float(sum(
            coefficient * fitted[flat_index]
            for flat_index, _ix, _iy, coefficient in _tensor_spline_basis(
                px, py, x_axis, y_axis)))

    boundary_errors = [
        abs(evaluate(bx, by) - GRADED_PROTECTED_CONSTRAINT_TARGET_MM)
        for bx, by, _nx, _ny in records
    ]
    boundary_error = max(boundary_errors)
    if boundary_error > GRADED_PROTECTED_CONSTRAINT_TOL_MM + 1.0e-6:
        worst_index = int(np.argmax(np.asarray(boundary_errors)))
        worst_record = records[worst_index]
        worst_entries = _tensor_spline_basis(
            worst_record[0], worst_record[1], x_axis, y_axis)
        upper_achievable = float(sum(
            coefficient * upper_by_index.get(flat_index, seed[flat_index])
            for flat_index, _ix, _iy, coefficient in worst_entries))
        active_ghosts = sum(
            flat_index in ghost_indices
            for flat_index, _ix, _iy, _coefficient in worst_entries)
        raise RuntimeError(
            "graded protected spline boundary fit missed its target: "
            f"max error={boundary_error:.6f} mm at "
            f"({worst_record[0]:.6f}, {worst_record[1]:.6f}); "
            f"max pole={float(np.max(fitted)):.6f} mm, "
            f"upper-achievable={upper_achievable:.6f} mm, "
            f"active ghosts={active_ghosts}, "
            f"ghosts={len(ghost_indices)}, unknowns={len(unknown_indices)}, "
            f"collar optimum={collar_optimum:.6f} mm, "
            f"collar limit={collar_limit:.6f} mm")

    maximum_collar_error = 0.0
    maximum_collar_reversal = 0.0
    for bx, by, nx, ny in records:
        depths = [evaluate(bx, by)]
        collar_xy = []
        for offset in GRADED_PROTECTED_COLLAR_OFFSETS_MM:
            px, py = bx - offset * nx, by - offset * ny
            point = Point(px, py)
            if not plan_with_tolerance.covers(point):
                break
            if protected.covers(point):
                break
            collar_xy.append((px, py))
            depths.append(evaluate(px, py))
        if collar_xy:
            analytic = _right_depth_vector(
                "graded", np.asarray(collar_xy, dtype=float))
            maximum_collar_error = max(
                maximum_collar_error,
                float(np.max(np.abs(np.asarray(depths[1:]) - analytic))))
            maximum_collar_reversal = max(
                maximum_collar_reversal,
                float(np.max(np.diff(np.asarray(depths)))))
    if maximum_collar_error > collar_limit + 1.0e-6:
        raise RuntimeError(
            "graded protected spline collar departs from the analytic field: "
            f"max error={maximum_collar_error:.6f} mm, "
            f"fitted limit={collar_limit:.6f} mm")
    if maximum_collar_reversal > GRADED_PROTECTED_COLLAR_MAX_REVERSAL_MM + 1.0e-6:
        raise RuntimeError(
            "graded protected spline collar is not monotonic: "
            f"reversal={maximum_collar_reversal:.6f} mm")

    return fitted.reshape(control_depth.shape), {
        "constraint_count": len(records),
        "adjusted_pole_count": len(unknown_indices),
        "ghost_pole_count": len(ghost_indices),
        "minimum_control_depth_mm": float(np.min(fitted)),
        "maximum_ghost_depth_mm": float(np.max(fitted)),
        "minimax_solver": "highs_parallel_dual_simplex_collar_with_boundary",
        "highs_thread_budget": highs_threads,
        "minimum_achievable_collar_error_mm": collar_optimum,
        "fitted_collar_error_limit_mm": collar_limit,
        "monotonic_constraint_count": monotonic_count,
        "minimax_boundary_optimum_mm": float(boundary_error),
        "maximum_boundary_error_mm": float(boundary_error),
        "maximum_collar_error_mm": float(maximum_collar_error),
        "maximum_collar_reversal_mm": float(maximum_collar_reversal),
    }


def _graded_smooth_body() -> Part:
    """Build one datum-clipped local cubic B-spline rear graph.

    The rear graph is a directly controlled, open-uniform degree-3 tensor
    B-spline.  Unlike a global point-cloud interpolation it is convex-hull
    bounded in Z, locally supported, deterministic and fast even with enough
    poles to resolve the LM/UM receiver lands.  The exact A plan, full-depth
    Obi-Wan lands and one-layer acoustic edge are applied as BREP constraints.
    """
    exact_plan = _layout().field_right
    _solution, depth_field, _sections = _graded_analytics()
    min_x, min_y, max_x, max_y = exact_plan.bounds
    unit_x, u_knots, u_mults = _open_uniform_spline_axis(
        GRADED_SURFACE_GRID_X, GRADED_SURFACE_SPLINE_DEGREE)
    unit_y, v_knots, v_mults = _open_uniform_spline_axis(
        GRADED_SURFACE_GRID_Y, GRADED_SURFACE_SPLINE_DEGREE)
    x = (min_x - GRADED_SURFACE_GRID_PADDING_MM
         + unit_x * (max_x - min_x + 2.0 * GRADED_SURFACE_GRID_PADDING_MM))
    y = (min_y - GRADED_SURFACE_GRID_PADDING_MM
         + unit_y * (max_y - min_y + 2.0 * GRADED_SURFACE_GRID_PADDING_MM))
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    control_depth = _graded_control_depths(xy).reshape(xx.shape)
    # The protected lands are retained from the exact full-depth blank and
    # excluded from the cutter by ``boolean_relief_plan`` below.  A ghost-pole
    # boundary fit is therefore neither necessary nor acceptable here: it can
    # pull the otherwise accurate dense analytic surface 4--6 mm away from the
    # approved S2--S4 profiles.  Greville-sampled analytic controls preserve
    # those profiles while the conservative trim supplies the exact lands.
    _fit_facts = {
        "method": "dense_analytic_greville_controls_with_exact_land_mask",
        "adjusted_pole_count": 0,
    }
    # A B-spline lies in the convex hull of its controls.  This CAD-only lower
    # clamp therefore proves the fitted removal surface leaves at least a
    # 0.26-mm skin everywhere; the exact acoustic-edge cutter subsequently
    # creates the intended constant 0.24-mm free edge without a tangent cap
    # Boolean on the spline solid.
    control_depth = np.maximum(
        control_depth, GRADED_CUTTER_CONTROL_MIN_DEPTH_MM)
    if float(np.min(control_depth)) < GRADED_CUTTER_CONTROL_MIN_DEPTH_MM - 1.0e-9:
        raise RuntimeError("graded fitted cutter control-depth floor was lost")
    fitted_flat = control_depth.ravel()

    def fitted_depth_at(px: float, py: float) -> float:
        return float(sum(
            coefficient * fitted_flat[flat_index]
            for flat_index, _ix, _iy, coefficient in _tensor_spline_basis(
                px, py, x, y)))

    fitted_section_errors = {}
    for definition in _sections:
        if definition.key not in {"S1", "S2", "S3", "S4"}:
            continue
        _along, expected_depth, section_xy, _segment = (
            contract._sample_graded_section(
                _layout(), depth_field, _solution, definition, samples=65))
        _along, expected_depth, section_xy = (
            contract._orient_graded_section_protected_to_edge(
                depth_field, _along, expected_depth, section_xy, definition))
        fitted_depth = np.asarray([
            fitted_depth_at(px, py) for px, py in section_xy
        ], dtype=float)
        fitted_section_errors[definition.key] = float(np.max(
            np.abs(fitted_depth - expected_depth)))
    if max(fitted_section_errors.values()) > 0.75:
        raise RuntimeError(
            "graded fitted B-spline departs from its analytic sections: "
            + ", ".join(
                f"{key}={value:.4f} mm"
                for key, value in fitted_section_errors.items()))
    rear_z = FRONT_Z_MM - control_depth

    poles = TColgp_Array2OfPnt(
        1, GRADED_SURFACE_GRID_X, 1, GRADED_SURFACE_GRID_Y)
    for ix in range(GRADED_SURFACE_GRID_X):
        for iy in range(GRADED_SURFACE_GRID_Y):
            poles.SetValue(
                ix + 1, iy + 1,
                gp_Pnt(float(xx[ix, iy]), float(yy[ix, iy]),
                       float(rear_z[ix, iy])))
    rear_surface = Geom_BSplineSurface(
        poles,
        _occ_real_array(u_knots), _occ_real_array(v_knots),
        _occ_integer_array(u_mults), _occ_integer_array(v_mults),
        GRADED_SURFACE_SPLINE_DEGREE, GRADED_SURFACE_SPLINE_DEGREE,
        False, False)
    relief_plan, _relief_component_facts = _graded_relief_plan()
    # OCC does not produce a valid solid when the fitted cutter is clipped by
    # the analytic plan's ~1,400 tessellated boundary edges.  Use a strictly
    # inward, sub-nozzle Boolean mask: it can only enlarge the exact full-depth
    # land, never cut it, and remains within 0.08 mm of the analytic relief.
    boolean_relief_plan, _boolean_relief_facts = _graded_boolean_relief_plan()

    # Retain a padded rectangle for cap-tool construction, but trim the fitted
    # rear itself to the conservative simplified Boolean relief outline. The
    # exact analytic outline has roughly 1,400 tessellated edges and is kept
    # for every depth/protected-land audit, not passed to OCC as a trim wire.
    surface_plan = orient(Polygon((
        (float(x[0]), float(y[0])),
        (float(x[-1]), float(y[0])),
        (float(x[-1]), float(y[-1])),
        (float(x[0]), float(y[-1])),
    )), sign=1.0)
    u_last = float(GRADED_SURFACE_GRID_X - GRADED_SURFACE_SPLINE_DEGREE)
    v_last = float(GRADED_SURFACE_GRID_Y - GRADED_SURFACE_SPLINE_DEGREE)

    def uv_point(px: float, py: float) -> tuple[float, float]:
        return (
            (float(px) - float(x[0])) / float(x[-1] - x[0]) * u_last,
            (float(py) - float(y[0])) / float(y[-1] - y[0]) * v_last,
        )

    def exact_uv_wire(ring) -> Wire:
        coordinates = list(ring.coords)
        if coordinates[0] != coordinates[-1]:
            coordinates.append(coordinates[0])
        wire_builder = BRepBuilderAPI_MakeWire()
        for (x1, y1), (x2, y2) in zip(
                coordinates[:-1], coordinates[1:], strict=True):
            u1, v1 = uv_point(x1, y1)
            u2, v2 = uv_point(x2, y2)
            du, dv = u2 - u1, v2 - v1
            length = float(np.hypot(du, dv))
            if length <= 1.0e-12:
                continue
            curve = Geom2d_Line(
                gp_Pnt2d(u1, v1), gp_Dir2d(du / length, dv / length))
            edge_builder = BRepBuilderAPI_MakeEdge(
                curve, rear_surface, 0.0, length)
            if not edge_builder.IsDone():
                raise RuntimeError(
                    "graded exact UV trim edge failed at "
                    f"({x1:.6f}, {y1:.6f})..({x2:.6f}, {y2:.6f})")
            wire_builder.Add(edge_builder.Edge())
            if not wire_builder.IsDone():
                raise RuntimeError(
                    "graded exact UV trim wire disconnected at "
                    f"({x1:.6f}, {y1:.6f})..({x2:.6f}, {y2:.6f})")
        return Wire(wire_builder.Wire())

    trim_plan = orient(boolean_relief_plan, sign=1.0)
    for ring in (trim_plan.exterior, *trim_plan.interiors):
        ring_xy = np.asarray(ring.coords, dtype=float)
        minimum_segment = float(np.min(np.linalg.norm(
            np.diff(ring_xy, axis=0), axis=1)))
        if minimum_segment < 0.005:
            raise RuntimeError(
                "graded conservative trim contains a sub-0.005-mm segment: "
                f"{minimum_segment:.6f} mm")
    outer_wire = exact_uv_wire(trim_plan.exterior)
    face_builder = BRepBuilderAPI_MakeFace(
        rear_surface, outer_wire.wrapped, False)
    for interior in trim_plan.interiors:
        face_builder.Add(exact_uv_wire(interior).wrapped)
    if not face_builder.IsDone():
        raise RuntimeError("graded conservative fitted rear face was not built")
    rear_face = Face(face_builder.Face()).fix()
    if not rear_face.is_valid:
        raise RuntimeError("graded conservative fitted rear trim is invalid")

    # Sweep the fitted rear downward.  The convex-hull control-depth floor
    # already proves a >0.24-mm skin, so no near-tangent plane cap is needed.
    # The trim is strictly inside the analytic relief, so protected lands and
    # the complete perimeter guard never participate in the final body cut.
    cutter_floor_z = REAR_LIMIT_Z_MM - 1.0
    relief_cutter = extrude(
        rear_face, amount=float(np.max(rear_z)) - cutter_floor_z,
        dir=(0.0, 0.0, -1.0), clean=True)
    relief_cutter = _strict_single_solid(
        relief_cutter, "graded conservative downward rear-relief cutter")
    sweep_mm = float(np.max(rear_z)) - cutter_floor_z
    expected_cutter_volume = float(boolean_relief_plan.area * sweep_mm)
    if not np.isclose(
            relief_cutter.volume, expected_cutter_volume,
            rtol=2.0e-4, atol=1.0):
        raise RuntimeError(
            "graded fitted face selected the wrong side of its trim wire: "
            f"volume={relief_cutter.volume:.3f} mm3, "
            f"projected-area expectation={expected_cutter_volume:.3f} mm3")
    relief_solid = list(relief_cutter.solids())[0]
    if not BRepLib.OrientClosedSolid_s(relief_solid.wrapped):
        raise RuntimeError("graded rear-relief cutter could not be outward-oriented")
    interior_xy = boolean_relief_plan.representative_point()
    interior_depth = fitted_depth_at(interior_xy.x, interior_xy.y)
    interior_top_z = FRONT_Z_MM - interior_depth
    interior_probe_z = interior_top_z - 0.5 * sweep_mm
    classifier = BRepClass3d_SolidClassifier(relief_solid.wrapped)
    classifier.Perform(
        gp_Pnt(interior_xy.x, interior_xy.y, interior_probe_z),
        Precision.Confusion_s())
    if classifier.State() != TopAbs_IN:
        relief_solid.wrapped.Complement()
        classifier = BRepClass3d_SolidClassifier(relief_solid.wrapped)
        classifier.Perform(
            gp_Pnt(interior_xy.x, interior_xy.y, interior_probe_z),
            Precision.Confusion_s())
    if classifier.State() != TopAbs_IN:
        raise RuntimeError(
            "graded rear-relief cutter does not contain its certified "
            "interior probe after orientation correction")
    relief_cutter = _strict_single_solid(
        Part([relief_solid]), "graded outward-oriented rear-relief cutter")
    # The edge cutter deliberately reaches 0.04 mm farther inward than the
    # relief guard.  That overlap prevents a coincident cutter seam while the
    # laterally oversized band crosses the blank's exterior sidewall.
    edge_band = depth_field.exposed_outer_edge.buffer(
        GRADED_EXACT_EDGE_BAND_MM, cap_style=2, join_style=2
    ).difference(depth_field.protected).buffer(0)
    edge_cutter = _plan_prism(
        edge_band,
        REAR_LIMIT_Z_MM - GRADED_EDGE_BOOLEAN_OVERSHOOT_MM,
        FRONT_Z_MM - GRADED_EDGE_DEPTH_MM)
    edge_cutter = _strict_single_solid(
        edge_cutter, "graded laterally oversized exact-edge cutter")
    blank = _plan_prism(exact_plan, REAR_LIMIT_Z_MM, FRONT_Z_MM)
    blank = _one_solid(blank, "graded pristine exact-plan blank")
    relief_probe = blank - relief_cutter
    if relief_probe is None:
        raise RuntimeError("graded smooth rear-relief subtraction returned no body")
    relief_probe = _one_solid(
        relief_probe.clean(), "graded blank after smooth rear-relief subtraction")
    relief_removed_volume = float(blank.volume - relief_probe.volume)
    if relief_removed_volume <= 100.0:
        cutter_bounds = relief_cutter.bounding_box()
        blank_bounds = blank.bounding_box()
        raise RuntimeError(
            "graded smooth rear-relief cutter removed no meaningful material: "
            f"removed={relief_removed_volume:.6f} mm3, "
            f"cutter_xy={cutter_bounds.min.X:.6f}.."
            f"{cutter_bounds.max.X:.6f}/"
            f"{cutter_bounds.min.Y:.6f}..{cutter_bounds.max.Y:.6f}, "
            f"cutter_z={cutter_bounds.min.Z:.6f}.."
            f"{cutter_bounds.max.Z:.6f}, "
            f"blank_xy={blank_bounds.min.X:.6f}.."
            f"{blank_bounds.max.X:.6f}/"
            f"{blank_bounds.min.Y:.6f}..{blank_bounds.max.Y:.6f}, "
            f"blank_z={blank_bounds.min.Z:.6f}.."
            f"{blank_bounds.max.Z:.6f}")
    # Subtract the positively overlapping rear and edge tools as one Boolean
    # set.  Sequential subtraction can leave a coincident BREP partition at
    # z=FRONT_Z-GRADED_EDGE_DEPTH when the protected captive-land outline changes;
    # union-before-cut is set-theoretically identical while preserving the
    # exact 0.24-mm edge and one connected printable body.
    combined_cutter = relief_cutter.fuse(edge_cutter)
    if combined_cutter is None:
        raise RuntimeError("graded combined rear/edge cutter fusion returned no tool")
    combined_cutter = _strict_single_solid(
        combined_cutter.clean(), "graded positively overlapping rear/edge cutter")
    carved = blank - combined_cutter
    if carved is None:
        raise RuntimeError("graded combined rear/edge subtraction returned no body")
    body = _one_solid(
        carved.clean(),
        "graded exact blank after protected rear/edge relief subtraction")
    if not 0.0 < body.volume < blank.volume:
        raise RuntimeError(
            "graded removal tools did not reduce the pristine blank volume")
    return _one_solid(body, "graded constrained monolith before receivers")


def _single_graded_relief_component(relief_geometry):
    """Return the one manufacturable graded relief component and audit facts.

    Any discarded component remains pristine full-depth blank material.  This
    operation can therefore only strengthen the wing; it can never remove a
    protected land or alter the exterior plan.
    """
    if relief_geometry.is_empty or not relief_geometry.is_valid:
        raise RuntimeError("graded rear relief complement is empty or invalid")
    if relief_geometry.geom_type == "Polygon":
        components = (relief_geometry,)
    elif relief_geometry.geom_type == "MultiPolygon":
        components = tuple(sorted(
            relief_geometry.geoms, key=lambda item: item.area, reverse=True))
    else:
        raise RuntimeError(
            "graded rear relief complement has unsupported topology: "
            f"{relief_geometry.geom_type}")

    primary = components[0]
    discarded_areas = tuple(float(piece.area) for piece in components[1:])
    discarded_total = float(sum(discarded_areas))
    discarded_maximum = max(discarded_areas, default=0.0)
    if (discarded_maximum
            > GRADED_RELIEF_DISCONNECTED_COMPONENT_MAX_AREA_MM2 + 1.0e-9
            or discarded_total
            > GRADED_RELIEF_DISCONNECTED_TOTAL_MAX_AREA_MM2 + 1.0e-9):
        raise RuntimeError(
            "graded rear relief developed a material disconnected component: "
            f"areas={discarded_areas}, "
            "per-component limit="
            f"{GRADED_RELIEF_DISCONNECTED_COMPONENT_MAX_AREA_MM2:.3f} mm2, "
            f"total limit={GRADED_RELIEF_DISCONNECTED_TOTAL_MAX_AREA_MM2:.3f} "
            "mm2")
    if primary.is_empty or not primary.is_valid or primary.interiors:
        raise RuntimeError(
            "graded primary rear relief must be one valid opening-free polygon")
    return primary, {
        "candidate_component_count": len(components),
        "cut_component_count": 1,
        "retained_full_depth_island_count": len(discarded_areas),
        "retained_full_depth_island_area_mm2": discarded_total,
        "largest_retained_island_area_mm2": discarded_maximum,
        "maximum_allowed_island_area_mm2": (
            GRADED_RELIEF_DISCONNECTED_COMPONENT_MAX_AREA_MM2),
        "maximum_allowed_total_island_area_mm2": (
            GRADED_RELIEF_DISCONNECTED_TOTAL_MAX_AREA_MM2),
    }


@lru_cache(maxsize=1)
def _graded_relief_candidate():
    """Return the exact relief complement before component selection.

    The perimeter guard keeps this plan off the exact outline so the analytic
    relief never has to be evaluated on the boundary itself.  It is a guard on
    the *analysis*, not on the cutter: ``_graded_boolean_relief_plan`` carries
    the cutter back across it and out of the plan, or the guard band survives
    the Boolean as a standing full-depth ledge.
    """
    exact_plan = _layout().field_right
    _solution, depth_field, _sections = _graded_analytics()
    perimeter_guard = exact_plan.boundary.buffer(
        GRADED_RELIEF_BOUNDARY_GUARD_MM, cap_style=2, join_style=2
    ).intersection(exact_plan).buffer(0)
    return exact_plan.difference(
        depth_field.protected.union(perimeter_guard)).buffer(0)


@lru_cache(maxsize=1)
def _graded_relief_plan():
    """Return the connected graded relief plan plus its topology facts."""
    return _single_graded_relief_component(_graded_relief_candidate())


@lru_cache(maxsize=1)
def _graded_retained_full_depth_islands():
    """Return relief islands the design deliberately leaves at full depth."""
    relief_plan, _component_facts = _graded_relief_plan()
    return _graded_relief_candidate().difference(relief_plan).buffer(0)


@lru_cache(maxsize=1)
def _graded_boolean_relief_plan():
    """Return the connected OCC trim and its localized tolerance audit."""
    relief_plan, _component_facts = _graded_relief_plan()
    _solution, depth_field, _sections = _graded_analytics()
    exact_plan = _layout().field_right
    # Everything the design keeps at full depth: the protected lands and the
    # deliberately retained relief islands.  The cutter is held off both by the
    # established normal inset, which is the C0 protected-land contract.
    retained_full_depth = unary_union(
        (depth_field.protected, _graded_retained_full_depth_islands())
    ).buffer(0)
    retained_keepout = retained_full_depth.buffer(
        GRADED_BOOLEAN_RELIEF_INSET_MM, join_style=2).buffer(0)
    # Carry the cutter across the perimeter guard and out of the plan on every
    # exposed boundary.  The blank already owns the exact outline, so the part
    # of the cutter outside the plan removes nothing and the plan trim keeps
    # defining the perimeter; what it does remove is the full-depth ledge the
    # guard band used to leave standing wherever the relief runs deep against
    # the outline.
    exposed_boundary = exact_plan.boundary.difference(retained_keepout)
    perimeter_overshoot = exposed_boundary.buffer(
        GRADED_RELIEF_PERIMETER_OVERSHOOT_MM, cap_style=2, join_style=2
    ).buffer(0)
    # Simplify last, after the keep-out difference: the difference is what
    # introduces the sub-micron segments that the trim-wire gate in
    # ``_graded_smooth_body`` rejects.  Any bite this simplification could take
    # out of the 0.04-mm keep-out collar is caught by the retained-bite gate
    # below.
    boolean_relief_plan = unary_union(
        (relief_plan, perimeter_overshoot)
    ).buffer(0).difference(retained_keepout).buffer(0).simplify(
        GRADED_BOOLEAN_RELIEF_SIMPLIFY_MM, preserve_topology=True
    ).buffer(0)
    # Protected land that meets the outline interrupts the overshoot ring, so
    # the union can arrive as a MultiPolygon.  The rear face is trimmed by one
    # wire, so keep the one body that carries the relief and account for what
    # that leaves behind; the discard sits at the acute lower root, the same
    # place the root Hausdorff exception below already covers.
    if boolean_relief_plan.geom_type == "MultiPolygon":
        parts = sorted(
            boolean_relief_plan.geoms, key=lambda item: item.area,
            reverse=True)
        discarded_overshoot = float(sum(part.area for part in parts[1:]))
        boolean_relief_plan = parts[0]
    else:
        discarded_overshoot = 0.0
    if discarded_overshoot > GRADED_RELIEF_MAX_DISCARDED_OVERSHOOT_MM2:
        raise RuntimeError(
            "graded Boolean relief mask shed too much overshoot corridor: "
            f"{discarded_overshoot:.6f} mm2")
    if (boolean_relief_plan.geom_type != "Polygon"
            or boolean_relief_plan.is_empty
            or not boolean_relief_plan.is_valid):
        raise RuntimeError(
            "graded conservative Boolean relief mask left its exact topology")
    # The cutter may leave the relief plan only outward, into the overshoot
    # corridor; it may never reach material the design retains at full depth.
    corridor = unary_union((relief_plan, perimeter_overshoot)).buffer(
        GRADED_BOOLEAN_RELIEF_SIMPLIFY_MM + 1.0e-9)
    if boolean_relief_plan.difference(corridor).area > 1.0e-6:
        raise RuntimeError(
            "graded Boolean relief mask escaped its overshoot corridor")
    retained_bite = boolean_relief_plan.intersection(retained_full_depth).area
    if retained_bite > GRADED_RELIEF_MAX_RETAINED_BITE_MM2:
        raise RuntimeError(
            "graded Boolean relief mask bit into retained full-depth land: "
            f"{retained_bite:.9f} mm2")
    # The contract this whole change exists to enforce: no non-protected
    # material inside the plan may survive the cutter as a standing ledge.
    uncut_band = exact_plan.difference(
        retained_keepout).difference(boolean_relief_plan).area
    if uncut_band > GRADED_RELIEF_MAX_UNCUT_BAND_MM2:
        raise RuntimeError(
            "graded Boolean relief mask left an uncut perimeter ledge: "
            f"{uncut_band:.6f} mm2")

    root_tip = min(
        relief_plan.exterior.coords, key=lambda coordinate: coordinate[1])
    root_exception = Point(*root_tip).buffer(
        GRADED_BOOLEAN_RELIEF_ROOT_EXCEPTION_RADIUS_MM, resolution=32)
    mask_tolerance = boolean_relief_plan.buffer(
        GRADED_BOOLEAN_RELIEF_GENERAL_MAX_HAUSDORFF_MM + 1.0e-9)
    general_boundary = relief_plan.boundary.difference(root_exception)
    if not general_boundary.difference(mask_tolerance).is_empty:
        raise RuntimeError(
            "graded conservative Boolean relief mask exceeded its general "
            "0.08-mm tolerance outside the lower-root exception")
    root_boundary = relief_plan.boundary.intersection(root_exception)
    root_tolerance = boolean_relief_plan.buffer(
        GRADED_BOOLEAN_RELIEF_ROOT_MAX_HAUSDORFF_MM + 1.0e-9)
    if not root_boundary.difference(root_tolerance).is_empty:
        raise RuntimeError(
            "graded conservative Boolean relief mask exceeded its localized "
            "lower-root tolerance")
    # The old absolute Hausdorff bound assumed the mask never left the relief
    # plan.  The mask now leaves it deliberately and only outward, so the
    # bound that still means something is on the outward reach: the corridor
    # check above proves the direction, this proves the distance.
    outside_plan = boolean_relief_plan.difference(exact_plan).buffer(0)
    overshoot_budget = exact_plan.boundary.buffer(
        GRADED_RELIEF_PERIMETER_OVERSHOOT_MM
        + GRADED_BOOLEAN_RELIEF_SIMPLIFY_MM + 1.0e-9,
        cap_style=2, join_style=2)
    if not outside_plan.difference(overshoot_budget).is_empty:
        raise RuntimeError(
            "graded Boolean relief mask overshot its perimeter budget of "
            f"{GRADED_RELIEF_PERIMETER_OVERSHOOT_MM:.3f} mm")
    outward_reach = max((
        float(exact_plan.distance(Point(*coordinate)))
        for geometry in getattr(outside_plan, "geoms", (outside_plan,))
        for coordinate in geometry.exterior.coords), default=0.0)

    edge_plan_overlap = (
        GRADED_EXACT_EDGE_BAND_MM
        - boolean_relief_plan.distance(depth_field.exposed_outer_edge))
    if edge_plan_overlap < GRADED_CUTTER_MIN_PLAN_OVERLAP_MM:
        raise RuntimeError(
            "graded edge/relief cutters lost positive plan overlap: "
            f"{edge_plan_overlap:.6f} mm")
    return boolean_relief_plan, {
        "normal_inset_mm": GRADED_BOOLEAN_RELIEF_INSET_MM,
        "general_maximum_hausdorff_mm": (
            GRADED_BOOLEAN_RELIEF_GENERAL_MAX_HAUSDORFF_MM),
        "lower_root_exception_radius_mm": (
            GRADED_BOOLEAN_RELIEF_ROOT_EXCEPTION_RADIUS_MM),
        "lower_root_maximum_hausdorff_mm": (
            GRADED_BOOLEAN_RELIEF_ROOT_MAX_HAUSDORFF_MM),
        "lower_root_tip_xy_mm": [float(value) for value in root_tip],
        "outside_exact_relief_area_mm2": float(
            boolean_relief_plan.difference(relief_plan).area),
        "edge_plan_overlap_mm": float(edge_plan_overlap),
        "perimeter_overshoot_mm": GRADED_RELIEF_PERIMETER_OVERSHOOT_MM,
        "measured_outward_reach_mm": outward_reach,
        "outside_exact_plan_area_mm2": float(
            boolean_relief_plan.difference(exact_plan).area),
        "uncut_perimeter_band_mm2": float(uncut_band),
        "maximum_uncut_perimeter_band_mm2": GRADED_RELIEF_MAX_UNCUT_BAND_MM2,
        "retained_full_depth_bite_mm2": float(retained_bite),
        "discarded_overshoot_area_mm2": float(discarded_overshoot),
    }


def _strict_single_solid(shape, label: str) -> Part:
    solids = list(shape.solids())
    if (not shape.is_valid or len(solids) != 1
            or solids[0].volume <= 1.0):
        raise RuntimeError(
            f"{label} must be one valid positive solid; valid={shape.is_valid}, "
            f"solid_count={len(solids)}, "
            f"volumes={[solid.volume for solid in solids]}")
    return Part([solids[0]])


def _one_solid(shape, label: str) -> Part:
    result = _strict_single_solid(shape, label)
    bounds = result.bounding_box()
    if bounds.min.Z < REAR_LIMIT_Z_MM - 0.03:
        raise RuntimeError(
            f"{label} exceeds the 11.5-mm rear envelope: "
            f"z_min={bounds.min.Z:.4f}")
    if bounds.max.Z > FRONT_Z_MM + 0.03:
        raise RuntimeError(
            f"{label} exceeds the flat front datum: z_max={bounds.max.Z:.4f}")
    return result


def _print_mask_plan(plan: Polygon) -> Polygon:
    """Cross the exterior perimeter without changing internal split seams."""
    field = _layout().field_right
    exterior_overshoot = plan.buffer(
        PRINT_MASK_EXTERIOR_OVERSHOOT_MM,
        join_style=2,
    ).difference(field)
    mask = unary_union((plan, exterior_overshoot)).buffer(0)
    if (mask.geom_type != "Polygon" or not mask.is_valid
            or mask.intersection(field).symmetric_difference(plan).area
            > 1.0e-6):
        raise RuntimeError(
            "outward-oversized print mask changed its in-plan ownership")
    return mask


def _cut_receivers(part, side: str) -> Part:
    for name, land in receiver_required_lands(side).items():
        missing = land - part
        missing_volume = 0.0 if missing is None else float(missing.volume)
        if missing_volume > 0.02:
            raise RuntimeError(
                f"{side} {name} immutable wing host misses "
                f"{missing_volume:.4f} mm3 of required captive land")
    for name, cutter in receiver_pockets(side).items():
        part = part - cutter
        if part is None:
            raise RuntimeError(f"{side} receiver cut {name} returned no shape")
    return _one_solid(part.clean(), f"{side} finalized receiver wing")


def _cut_optional_lm_key_clearance(part, side: str) -> Part:
    """Keep wing clearance around the optional ring-local keyed split."""
    side = _normalize_side(side)
    tool = registration_wing_clearance_tools()[side]
    result = part - tool
    if result is None:
        raise RuntimeError(
            f"{side} optional LM key-land clearance returned no shape")
    return _one_solid(
        result.clean(), f"{side} wing with optional LM key-land clearance")


@lru_cache(maxsize=2)
def _right_monolith_cached(slug: str) -> Part:
    _require_guarded_build()
    slug = _normalize_variant(slug)
    if slug == "flat":
        body = _plan_prism(
            _layout().field_right, REAR_LIMIT_Z_MM, FRONT_Z_MM)
        body = _one_solid(body, "flat constant-depth monolith")
    else:
        body = _graded_smooth_body()
    body = _cut_receivers(body, "right")
    body = _cut_optional_lm_key_clearance(body, "right")
    body.label = f"obiwan_wing_{slug}_right_monolithic"
    return body


def wing_monolithic(variant_id: str, side: str):
    """Return one finalized canonical wing; left is an exact final mirror."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    side = _normalize_side(side)
    right = deepcopy(_right_monolith_cached(slug))
    if side == "right":
        right.label = f"obiwan_wing_{slug}_right_monolithic"
        return right
    left = mirror(right, about=Plane.YZ)
    left = _one_solid(left, f"{slug} mirrored left monolith")
    left.label = f"obiwan_wing_{slug}_left_monolithic"
    return left


@lru_cache(maxsize=2)
def _right_print_parts_cached(slug: str) -> tuple[Part, ...]:
    """Intersect exact print masks only after finalizing the monolith."""
    monolith = deepcopy(_right_monolith_cached(slug))
    result = []
    for order, role in enumerate(PRINT_PART_KEYS, start=1):
        mask = _plan_prism(
            _print_mask_plan(_layout().print_parts[role]),
            REAR_LIMIT_Z_MM - 0.5, FRONT_Z_MM + 0.5)
        common = monolith & mask
        if common is None:
            raise RuntimeError(
                f"{slug} right {role} print-mask intersection is empty")
        piece = _one_solid(
            common.clean(), f"{slug} right {role} print piece")
        piece.label = f"obiwan_wing_{slug}_right_{order}of3_{role}"
        result.append(piece)
    return tuple(result)


def wing_print_parts(variant_id: str, side: str) -> dict[str, Part]:
    """Return three installed-position print pieces for one side."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    side = _normalize_side(side)
    right = tuple(deepcopy(piece) for piece in _right_print_parts_cached(slug))
    result: dict[str, Part] = {}
    for order, (role, piece) in enumerate(
            zip(PRINT_PART_KEYS, right, strict=True), start=1):
        if side == "left":
            piece = _one_solid(
                mirror(piece, about=Plane.YZ),
                f"{slug} mirrored left {role} print piece")
        piece.label = f"obiwan_wing_{slug}_{side}_{order}of3_{role}"
        result[role] = piece
    return result


@lru_cache(maxsize=2)
def _right_two_piece_print_parts_cached(slug: str) -> tuple[Part, ...]:
    """Build B from the finalized monolith while preserving A's lower."""
    monolith = deepcopy(_right_monolith_cached(slug))
    lower = deepcopy(_right_print_parts_cached(slug)[0])
    lower.label = f"obiwan_wing_{slug}_right_b_1of2_lm_lower"
    upper_mask = _plan_prism(
        _print_mask_plan(_layout().two_piece_print_parts["lm_um_upper"]),
        REAR_LIMIT_Z_MM - 0.5, FRONT_Z_MM + 0.5)
    common = monolith & upper_mask
    if common is None:
        raise RuntimeError(
            f"{slug} right two-piece LM/UM upper intersection is empty")
    upper = _one_solid(
        common.clean(), f"{slug} right two-piece LM/UM upper print piece")
    upper.label = f"obiwan_wing_{slug}_right_b_2of2_lm_um_upper"
    return lower, upper


def wing_two_piece_print_parts(
        variant_id: str, side: str) -> dict[str, Part]:
    """Return the alternative two installed-position print pieces."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    side = _normalize_side(side)
    right = tuple(
        deepcopy(piece)
        for piece in _right_two_piece_print_parts_cached(slug))
    result: dict[str, Part] = {}
    for order, (role, piece) in enumerate(
            zip(TWO_PIECE_PRINT_PART_KEYS, right, strict=True), start=1):
        if side == "left":
            piece = _one_solid(
                mirror(piece, about=Plane.YZ),
                f"{slug} mirrored left two-piece {role} print piece")
        piece.label = f"obiwan_wing_{slug}_{side}_b_{order}of2_{role}"
        result[role] = piece
    return result


def wing_monolithic_assembly(variant_id: str) -> Compound:
    """Canonical STEP authority: installed left/right monolithic pair."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    assembly = Compound(children=[
        wing_monolithic(slug, "left"),
        wing_monolithic(slug, "right"),
    ])
    assembly.label = f"lx521_obiwan_basic_wing_{slug}_monolithic_pair"
    return assembly


def wing_print_assembly(variant_id: str) -> Compound:
    """Installed six-piece assembly used for the assembled STEP review."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    children = []
    for side in SIDE_NAMES:
        children.extend(wing_print_parts(slug, side).values())
    assembly = Compound(children=children)
    assembly.label = f"lx521_obiwan_basic_wing_{slug}_print_assembly"
    return assembly


def wing_two_piece_print_assembly(variant_id: str) -> Compound:
    """Installed four-piece assembly for the alternative B wing split."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    children = []
    for side in SIDE_NAMES:
        children.extend(wing_two_piece_print_parts(slug, side).values())
    assembly = Compound(children=children)
    assembly.label = (
        f"lx521_obiwan_basic_wing_{slug}_two_piece_print_assembly")
    return assembly


def _normalize_state(state: str) -> str:
    aliases = {
        "floor": "floor_stand",
        "floor_stand": "floor_stand",
        "no_floor": "no_floor_stand",
        "no-floor": "no_floor_stand",
        "no_floor_stand": "no_floor_stand",
    }
    value = aliases.get(str(state).strip().lower())
    if value is None:
        raise ValueError(
            "state must be floor_stand or no_floor_stand")
    return value


def wing_review_context_parts(state: str) -> dict[str, Part]:
    """Exact installed LM/UM/T reference parts for one Obi-Wan stand state."""
    _require_guarded_build()
    state = _normalize_state(state)
    loaded_state = "floor_stand" if STAND_FOOT else "no_floor_stand"
    if state != loaded_state:
        raise RuntimeError(
            f"requested {state}, but imported Obi-Wan geometry is {loaded_state}; "
            "set LX_STAND_FOOT before starting the guarded worker")
    from .carriers import core_parts
    from .attachments import tweeter_crescent

    context = core_parts()
    context["addon_tweeter_crescent"] = tweeter_crescent()
    for name, shape in context.items():
        shape.label = f"reference_obiwan_{state}_{name}"
    return context


def wing_review_split_context_parts(
        no_floor_manifest: Path, floor_manifest: Path,
        ) -> dict[str, dict]:
    """Load exact staged LM split alternatives plus shared UM/T context.

    Both stage manifests are validated against their own explicit stand
    state.  This is deliberately artifact-backed: toggling the imported
    ``STAND_FOOT`` global would mix route, bridge and floor geometry from two
    incompatible module states in one worker.
    """
    _require_guarded_build()
    from export_obiwan_staged import load_stage_manifest, staged_part_paths

    no_floor_manifest = Path(no_floor_manifest)
    floor_manifest = Path(floor_manifest)
    no_floor_payload = load_stage_manifest(
        no_floor_manifest, stand_foot=False)
    floor_payload = load_stage_manifest(
        floor_manifest, stand_foot=True)
    no_floor_paths = staged_part_paths(
        no_floor_manifest, no_floor_payload)
    floor_paths = staged_part_paths(floor_manifest, floor_payload)

    specifications = (
        ("lm_lower_floor", floor_payload, floor_paths,
         "optional_lm_keyed_1_of_2_bottom", "floor_stand"),
        ("lm_lower_no_floor", no_floor_payload, no_floor_paths,
         "optional_lm_keyed_1_of_2_bottom", "no_floor_stand"),
        ("lm_upper", no_floor_payload, no_floor_paths,
         "optional_lm_keyed_2_of_2_top", "no_floor_stand"),
        ("um", no_floor_payload, no_floor_paths,
         "core_um_carrier", "no_floor_stand"),
        ("t", no_floor_payload, no_floor_paths,
         "addon_tweeter_crescent", "no_floor_stand"),
    )
    context: dict[str, dict] = {}
    for key, payload, paths, part_key, state in specifications:
        if part_key not in paths or part_key not in payload["parts"]:
            raise RuntimeError(
                f"Obi-Wan staged review context lacks {state}/{part_key}")
        shape = import_brep(str(paths[part_key]))
        solids = list(shape.solids())
        if (not shape.is_valid or len(solids) != 1
                or solids[0].volume <= 0.01):
            raise RuntimeError(
                f"invalid staged review BREP {state}/{part_key}: "
                f"valid={shape.is_valid} solids={len(solids)}")
        source_label = f"reference_obiwan_{state}_{part_key}"
        shape = Part([solids[0]])
        shape.label = source_label
        context[key] = {
            "shape": shape,
            "source_label": source_label,
            "source_sha256": payload["parts"][part_key]["sha256"],
            "state": state,
            "part_key": part_key,
        }
    if (context["lm_lower_floor"]["source_sha256"]
            == context["lm_lower_no_floor"]["source_sha256"]):
        raise RuntimeError(
            "floor and no-floor LM lower staged BREPs must be distinct")
    return context


def wing_review_assembly(variant_id: str, state: str) -> Compound:
    """Installed wing pair with the existing Obi-Wan core/crescent context.

    Obi-Wan state is selected at module import by ``LX_STAND_FOOT``.  Requiring
    the argument to agree prevents a mislabeled review assembly and reuses the
    authoritative carrier builders instead of copying their geometry here.
    """
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    state = _normalize_state(state)
    children = list(wing_review_context_parts(state).values())
    children.extend((
        wing_monolithic(slug, "left"),
        wing_monolithic(slug, "right"),
    ))
    assembly = Compound(children=children)
    assembly.label = f"lx521_obiwan_basic_wing_{slug}_{state}_review"
    return assembly


def _bounds_record(shape) -> list[list[float]]:
    bounds = shape.bounding_box()
    return [
        [float(bounds.min.X), float(bounds.min.Y), float(bounds.min.Z)],
        [float(bounds.max.X), float(bounds.max.Y), float(bounds.max.Z)],
    ]


def _section_fact_summary(slug: str) -> dict[str, dict]:
    sampled = wing_section_samples(slug, "right", samples=481)
    return {
        key: {
            "title": value["title"],
            "start_depth_mm": value["start_depth_mm"],
            "end_depth_mm": value["end_depth_mm"],
            "minimum_depth_mm": value["minimum_depth_mm"],
            "maximum_depth_mm": value["maximum_depth_mm"],
            "worst_depth_reversal_mm": value["worst_depth_reversal_mm"],
            "monotonic_nonincreasing": value["monotonic_nonincreasing"],
        }
        for key, value in sampled.items()
    }


def _shapely_line_parts(geometry) -> tuple:
    """Flatten line components without buffering their exact locus."""
    if geometry.is_empty:
        return ()
    if geometry.geom_type in ("LineString", "LinearRing"):
        return (geometry,) if geometry.length > 1.0e-6 else ()
    if not hasattr(geometry, "geoms"):
        return ()
    return tuple(
        line for child in geometry.geoms
        for line in _shapely_line_parts(child)
    )


def _brep_vertical_depth_mm(shape, x_mm: float, y_mm: float) -> float:
    """Measure one depth with the legacy Boolean probe.

    This deliberately remains available as a sparse cross-engine oracle.  The
    complete perimeter audit uses :class:`_VerticalDepthSampler` below: loading
    the exact BREP once avoids rebuilding a Boolean data structure for every
    one of the 2,196 paired probe lines.
    """
    hits = shape.intersect(
        Axis((float(x_mm), float(y_mm), REAR_LIMIT_Z_MM - 2.0),
             (0.0, 0.0, 1.0)),
        tolerance=1.0e-5, include_touched=True)
    if not hits:
        raise RuntimeError(
            f"vertical BREP probe missed x/y={x_mm:.6f}/{y_mm:.6f}")
    z_values = [
        float(vertex.Z)
        for hit in hits
        for vertex in hit.vertices()
    ]
    if len(z_values) < 2:
        raise RuntimeError(
            f"vertical BREP probe returned no span at "
            f"{x_mm:.6f}/{y_mm:.6f}")
    return max(z_values) - min(z_values)


class _VerticalDepthSampler:
    """Fast repeated vertical probes against one immutable exact BREP.

    ``IntCurvesFace_ShapeIntersector`` indexes the shape's faces once and then
    intersects finite vertical curves without constructing Boolean result
    topology.  It therefore measures the same front/rear BREP faces as the
    legacy Axis/Common probe while remaining practical at the required
    0.50-mm protected-perimeter spacing.  The intersector is mutable and must
    stay process-local; callers intentionally use it serially.
    """

    def __init__(self, shape, tolerance: float = 1.0e-5):
        self._shape = shape  # Keep the wrapped TopoDS_Shape alive.
        self._tolerance = float(tolerance)
        self._z_low = REAR_LIMIT_Z_MM - 2.0
        self._z_high = FRONT_Z_MM + 2.0
        self._intersector = IntCurvesFace_ShapeIntersector()
        self._intersector.Load(shape.wrapped, self._tolerance)
        self._cache: dict[tuple[float, float], float] = {}

    def depth_mm(self, x_mm: float, y_mm: float) -> float:
        x = float(x_mm)
        y = float(y_mm)
        key = (x, y)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        line = gp_Lin(
            gp_Pnt(x, y, self._z_low), gp_Dir(0.0, 0.0, 1.0))
        self._intersector.Perform(
            line, 0.0, self._z_high - self._z_low)
        if not self._intersector.IsDone():
            raise RuntimeError(
                f"vertical BREP intersector failed at x/y={x:.6f}/{y:.6f}")
        self._intersector.SortResult()

        z_values: list[float] = []
        for index in range(1, self._intersector.NbPnt() + 1):
            point = self._intersector.Pnt(index)
            if (abs(float(point.X()) - x) > self._tolerance
                    or abs(float(point.Y()) - y) > self._tolerance):
                raise RuntimeError(
                    "vertical BREP intersector returned an off-axis point at "
                    f"x/y={x:.6f}/{y:.6f}")
            z = float(point.Z())
            if not z_values or abs(z - z_values[-1]) > self._tolerance:
                z_values.append(z)

        if len(z_values) < 2:
            raise RuntimeError(
                "vertical BREP intersector returned no material span at "
                f"{x:.6f}/{y:.6f}; unique_z={z_values}")
        depth = max(z_values) - min(z_values)
        self._cache[key] = depth
        return depth

    def depths_mm(
            self, xy_points: Iterable[tuple[float, float]]) -> tuple[float, ...]:
        return tuple(
            self.depth_mm(x, y)
            for x, y in xy_points
        )


@lru_cache(maxsize=1)
def _graded_protected_perimeter_brep_facts() -> dict:
    """Measure the complete internal protected-land C0 transition."""
    _require_guarded_build()
    graded = wing_monolithic("graded", "right")
    _solution, depth_field, _definitions = _graded_analytics()
    protected = depth_field.protected
    plan = wing_plan("graded", "right")
    external_guard = plan.boundary.buffer(
        contract.GRADED_EDGE_MATCH_TOL_MM, cap_style=2, join_style=2)
    excluded_external = protected.boundary.intersection(external_guard)
    internal_transition = protected.boundary.difference(external_guard)
    plan_with_tolerance = plan.buffer(1.0e-6)

    probe_pairs: list[tuple[Point, Point, Point]] = []
    maximum_jump = 0.0
    maximum_jump_xy = (float("nan"), float("nan"))
    for line in _shapely_line_parts(internal_transition):
        sample_count = max(
            1, int(np.ceil(
                line.length / GRADED_PROTECTED_BOUNDARY_SAMPLE_SPACING_MM)))
        for sample_index in range(sample_count):
            distance = line.length * (
                sample_index + 0.5) / float(sample_count)
            tangent_half_span = min(0.05, 0.20 * line.length)
            before = line.interpolate(max(0.0, distance - tangent_half_span))
            after = line.interpolate(min(line.length, distance + tangent_half_span))
            tx = float(after.x - before.x)
            ty = float(after.y - before.y)
            tangent_length = float(np.hypot(tx, ty))
            if tangent_length <= 1.0e-8:
                raise RuntimeError(
                    "graded protected perimeter contains a degenerate tangent")
            nx = -ty / tangent_length
            ny = tx / tangent_length
            boundary = line.interpolate(distance)
            plus = Point(
                boundary.x + GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary.y + GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * ny)
            minus = Point(
                boundary.x - GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * nx,
                boundary.y - GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM * ny)
            plus_inside = protected.covers(plus)
            minus_inside = protected.covers(minus)
            if plus_inside == minus_inside:
                raise RuntimeError(
                    "graded protected perimeter probe did not straddle the land at "
                    f"({boundary.x:.6f}, {boundary.y:.6f})")
            inside = plus if plus_inside else minus
            outside = minus if plus_inside else plus
            if (not plan_with_tolerance.covers(inside)
                    or not plan_with_tolerance.covers(outside)):
                raise RuntimeError(
                    "graded internal protected perimeter probe left the wing plan at "
                    f"({boundary.x:.6f}, {boundary.y:.6f})")
            probe_pairs.append((boundary, inside, outside))

    perimeter_samples = len(probe_pairs)
    if perimeter_samples < 20:
        raise RuntimeError(
            "graded protected perimeter was not sampled completely: "
            f"total={perimeter_samples}")

    sampler = _VerticalDepthSampler(graded)
    flattened_xy = [
        (float(point.x), float(point.y))
        for _boundary, inside, outside in probe_pairs
        for point in (inside, outside)
    ]
    flattened_depths = sampler.depths_mm(flattened_xy)
    qualified_samples = 0
    for pair_index, (boundary, inside, outside) in enumerate(probe_pairs):
        inside_depth = flattened_depths[2 * pair_index]
        outside_depth = flattened_depths[2 * pair_index + 1]
        if not np.isclose(inside_depth, FULL_DEPTH_MM, atol=0.03):
            raise RuntimeError(
                "graded protected land lost exact full depth: "
                f"{inside_depth:.6f} mm at "
                f"({inside.x:.6f}, {inside.y:.6f})")
        jump = abs(inside_depth - outside_depth)
        qualified_samples += 1
        if qualified_samples == 1 or jump > maximum_jump:
            maximum_jump = jump
            maximum_jump_xy = (float(boundary.x), float(boundary.y))

    if qualified_samples != perimeter_samples:
        raise RuntimeError(
            "graded protected perimeter was not probed completely: "
            f"qualified={qualified_samples}, total={perimeter_samples}")
    if maximum_jump > GRADED_PROTECTED_BOUNDARY_MAX_C0_JUMP_MM:
        raise RuntimeError(
            "graded protected-land C0 rear step exceeds 0.03 mm: "
            f"jump={maximum_jump:.6f} at {maximum_jump_xy}")

    # Keep a sparse deterministic comparison with the former Boolean/Axis
    # engine.  This guards the faster face intersector's semantics without
    # paying for thousands of separately prepared BRepAlgoAPI_Common calls.
    calibration_indices = np.unique(np.linspace(
        0, perimeter_samples - 1, 9, dtype=int))
    maximum_engine_delta = 0.0
    calibration_probe_count = 0
    for pair_index in calibration_indices:
        _boundary, inside, outside = probe_pairs[int(pair_index)]
        for point, fast_depth in (
                (inside, flattened_depths[2 * int(pair_index)]),
                (outside, flattened_depths[2 * int(pair_index) + 1])):
            legacy_depth = _brep_vertical_depth_mm(graded, point.x, point.y)
            maximum_engine_delta = max(
                maximum_engine_delta, abs(legacy_depth - fast_depth))
            calibration_probe_count += 1
    if maximum_engine_delta > 0.002:
        raise RuntimeError(
            "graded vertical BREP probe engines disagree: "
            f"maximum delta={maximum_engine_delta:.6f} mm")
    return {
        "locus": (
            "complete depth_field.protected.boundary excluding only "
            "external-plan-coincident segments"),
        "maximum_sample_spacing_mm": (
            GRADED_PROTECTED_BOUNDARY_SAMPLE_SPACING_MM),
        "paired_probe_offset_mm": GRADED_PROTECTED_BOUNDARY_PROBE_OFFSET_MM,
        "maximum_allowed_c0_jump_mm": (
            GRADED_PROTECTED_BOUNDARY_MAX_C0_JUMP_MM),
        "internal_transition_length_mm": float(internal_transition.length),
        "excluded_external_boundary_length_mm": float(
            excluded_external.length),
        "paired_probe_count": qualified_samples,
        "maximum_measured_c0_jump_mm": maximum_jump,
        "maximum_jump_xy_mm": list(maximum_jump_xy),
        "probe_engine": "IntCurvesFace_ShapeIntersector",
        "legacy_boolean_calibration_probe_count": calibration_probe_count,
        "legacy_boolean_maximum_delta_mm": maximum_engine_delta,
    }


def wing_facts(variant_id: str) -> dict:
    """Return JSON-safe analytic and actual-BREP release facts."""
    _require_guarded_build()
    slug = _normalize_variant(variant_id)
    layout = _layout()
    solution, depth_field, _sections = _graded_analytics()
    right = wing_monolithic(slug, "right")
    left = wing_monolithic(slug, "left")
    parts = {
        side: wing_print_parts(slug, side)
        for side in SIDE_NAMES
    }
    two_piece_parts = {
        side: wing_two_piece_print_parts(slug, side)
        for side in SIDE_NAMES
    }
    if slug == "flat":
        analytic_volume = float(layout.field_right.area * FULL_DEPTH_MM)
        analytic_mass = (
            analytic_volume / 1000.0 * contract.PLA_DENSITY_G_CM3)
        depth_min = depth_max = FULL_DEPTH_MM
        edge_range = [FULL_DEPTH_MM, FULL_DEPTH_MM]
        reduction = 0.0
        max_slope = 0.0
    else:
        analytic_volume = float(depth_field.volume_mm3)
        analytic_mass = float(depth_field.mass_g)
        depth_min = float(GRADED_EDGE_DEPTH_MM)
        depth_max = FULL_DEPTH_MM
        edge_range = [float(value)
                      for value in depth_field.outer_edge_depth_mm]
        reduction = float(depth_field.reduction_pct)
        max_slope = float(depth_field.max_grid_slope)

    print_records = {}
    two_piece_print_records = {}
    for side in SIDE_NAMES:
        print_records[side] = {}
        for role, piece in parts[side].items():
            print_records[side][role] = {
                "label": piece.label,
                "volume_mm3": adaptive_volume_mm3(piece),
                "bounds_mm": _bounds_record(piece),
            }
        two_piece_print_records[side] = {}
        for role, piece in two_piece_parts[side].items():
            two_piece_print_records[side][role] = {
                "label": piece.label,
                "volume_mm3": adaptive_volume_mm3(piece),
                "bounds_mm": _bounds_record(piece),
            }
    protected_perimeter_brep = (
        None if slug == "flat" else _graded_protected_perimeter_brep_facts())
    return {
        "schema_version": 1,
        "artifact_family": "obiwan_wing_artifacts",
        "variant_slug": slug,
        "outline_family": "A-Obi-Wan straight upper taper",
        "qualification": {
            "status": "unmeasured_acoustic_experiment",
            "not_w1_w2_spec_compliance": True,
            "canonical_geometry": "monolithic_pair",
            "print_geometry_derivation": "post_boolean_plan_intersection",
        },
        "coordinate_frame": {
            "xy": "installed baffle plan",
            "front_z_mm": FRONT_Z_MM,
            "rear_limit_z_mm": REAR_LIMIT_Z_MM,
        },
        "depth_contract": {
            "model": ("constant" if slug == "flat"
                      else "LM_UM_T_weighted_smooth_rear"),
            "full_depth_mm": FULL_DEPTH_MM,
            "minimum_depth_mm": depth_min,
            "maximum_depth_mm": depth_max,
            "eligible_outer_edge_depth_mm": edge_range,
            "graded_optional_fine_layer_edge_mm": float(
                contract.GRADED_OPTIONAL_FINE_LAYER_EDGE_MM),
            "analytic_volume_mm3_per_side": analytic_volume,
            "analytic_mass_g_per_side": analytic_mass,
            "volume_reduction_from_flat_pct": reduction,
            "maximum_analytic_grid_slope": max_slope,
            "maximum_allowed_slope": float(contract.GRADED_FIELD_MAX_SLOPE),
            "surface_construction": (
                "plan_prism" if slug == "flat"
                else "direct_open_uniform_control_bspline"),
            "surface_control_net_xy": (
                None if slug == "flat"
                else [GRADED_SURFACE_GRID_X, GRADED_SURFACE_GRID_Y]),
            "surface_spline_degree": (
                0 if slug == "flat" else GRADED_SURFACE_SPLINE_DEGREE),
            "surface_control_padding_mm": (
                0.0 if slug == "flat" else GRADED_SURFACE_GRID_PADDING_MM),
            "surface_outside_edge_slope": (
                0.0 if slug == "flat" else GRADED_SURFACE_OUTSIDE_EDGE_SLOPE),
            "surface_outside_minimum_control_depth_mm": (
                None if slug == "flat" else GRADED_SURFACE_OUTSIDE_MIN_DEPTH_MM),
            "exact_edge_brep_band_mm": (
                0.0 if slug == "flat" else GRADED_EXACT_EDGE_BAND_MM),
            "conservative_relief_island_retention": (
                None if slug == "flat" else _graded_relief_plan()[1]),
            "conservative_boolean_relief_mask": (
                None if slug == "flat" else _graded_boolean_relief_plan()[1]),
            "protected_perimeter_brep_c0_gate": protected_perimeter_brep,
            "retention_centres": list(solution.center_names),
            "retention_scales": [float(value)
                                  for value in solution.retention_scales],
        },
        "interface_contract": {
            "selected_receiver_count_per_side": 3,
            "lower_floor_bend_integration": (
                contract._lower_wing_blend(layout.profile)[1]),
            "receivers": {
                side: list(receiver_facts(side)) for side in SIDE_NAMES
            },
            "protected_full_depth_area_mm2_per_side": float(
                depth_field.protected_area_mm2),
            "top_flush_depth_mm": [float(value)
                                   for value in depth_field.top_flush_depth_mm],
            "optional_lm_keyed_split": {
                "geometrically_compatible": True,
                "physical_fit_coupon_required": True,
                "ring_local_key_clearance_mm": float(
                    REGISTRATION_WING_CLEARANCE_MM),
                "pocket_location": "carrier_interface_between_front_and_rear",
                "right_pocket_uses_left_relief_worst_case": True,
                "left_is_exact_mirror": True,
                "carrier_exterior_growth_mm": 0.0,
                "primary_magnet_datums_unchanged": True,
            },
        },
        "dovetail_contract": {
            "method": "v1l_style_through_thickness_xy_dovetails",
            "part_roles": list(PRINT_PART_KEYS),
            "two_piece_part_roles": list(TWO_PIECE_PRINT_PART_KEYS),
            "key_count_per_side": len(layout.dovetail_keys),
            "clearance_mm": float(contract.DOVETAIL_CLEARANCE_MM),
            "endpoint_taper_mm": float(
                contract.DOVETAIL_ENDPOINT_TAPER_MM),
            "endpoint_taper_location": "both_seam_endpoints",
            "male_root_overlap_mm": float(
                contract.DOVETAIL_ROOT_OVERLAP_MM),
            "male_owners": [key["male_owner"]
                              for key in layout.dovetail_keys],
            "female_owners": [key["female_owner"]
                                for key in layout.dovetail_keys],
            "lower_profile_mm": {
                "neck": float(layout.dovetail_keys[0]["neck_mm"]),
                "head": float(layout.dovetail_keys[0]["head_mm"]),
                "depth": float(
                    layout.dovetail_keys[0]["penetration_mm"]),
            },
            "upper_profile_mm": {
                "neck": float(layout.dovetail_keys[1]["neck_mm"]),
                "head": float(layout.dovetail_keys[1]["head_mm"]),
                "depth": float(
                    layout.dovetail_keys[1]["penetration_mm"]),
            },
            "key_centres_xy_mm_right": [
                [float(value) for value in key["center_xy_mm"]]
                for key in layout.dovetail_keys
            ],
            "minimum_outer_ligament_mm": [
                float(key["ligament_mm"]) for key in layout.dovetail_keys
            ],
            "seam_chord_mm": [
                float(layout.metrics["lower_seam_chord_mm"]),
                float(layout.metrics["upper_seam_chord_mm"]),
            ],
            "no_envelope_growth": True,
            "through_local_thickness": True,
            "assembly_motion": "z_axis_slide",
            "z_retention": False,
            "coupon_qualification_required": True,
            "graded_joint_interface_area_mm2": [
                float(value) for value in depth_field.joint_area_mm2
            ],
            "graded_joint_rear_mismatch_mm": [
                float(value) for value in depth_field.joint_rear_mismatch_mm
            ],
        },
        "print_contract": {
            "usable_bed_xy_mm": float(contract.BED_USABLE_MM),
            "options": {
                "a": {
                    "piece_count_per_side": 3,
                    "part_roles": list(PRINT_PART_KEYS),
                },
                "b": {
                    "piece_count_per_side": 2,
                    "part_roles": list(TWO_PIECE_PRINT_PART_KEYS),
                    "lower_geometry_identical_to_a": True,
                    "former_upper_fit_gap_restored": True,
                },
            },
            "right_plan_obb_mm": {
                role: str(layout.metrics[f"{role}_obb"])
                for role in PRINT_PART_KEYS
            },
            "two_piece_right_plan_bed_fit_mm": {
                "lm_lower": str(layout.metrics["lm_lower_obb"]),
                "lm_um_upper": str(
                    layout.metrics["lm_um_upper_bed_fit"]),
            },
            "installed_piece_brep": print_records,
            "two_piece_installed_piece_brep": two_piece_print_records,
        },
        "analytic_sections_right": _section_fact_summary(slug),
        "actual_brep": {
            "right_monolith_label": right.label,
            "left_monolith_label": left.label,
            "volume_integration": {
                "method": "BRepGProp_adaptive_2d_Gauss",
                "requested_relative_error": ADAPTIVE_VOLUME_EPS,
                "maximum_reached_relative_error": (
                    ADAPTIVE_VOLUME_MAX_REACHED_ERROR),
                "only_closed": True,
                "skip_shared": False,
            },
            "right_volume_mm3": adaptive_volume_mm3(right),
            "left_volume_mm3": adaptive_volume_mm3(left),
            "right_bounds_mm": _bounds_record(right),
            "left_bounds_mm": _bounds_record(left),
            "valid_single_solid_each_side": True,
        },
    }


def gen_step(variant_id: str = "flat", assembled: bool = False):
    """Compatibility entry point for canonical or printable STEP assembly."""
    return (wing_print_assembly(variant_id) if assembled
            else wing_monolithic_assembly(variant_id))


__all__ = (
    "VARIANT_IDS",
    "SIDE_NAMES",
    "PRINT_PART_KEYS",
    "TWO_PIECE_PRINT_PART_KEYS",
    "receiver_facts",
    "receiver_pockets",
    "receiver_required_lands",
    "wing_plan",
    "wing_print_plan_parts",
    "wing_two_piece_print_plan_parts",
    "wing_depth_at",
    "wing_section_samples",
    "wing_monolithic",
    "wing_print_parts",
    "wing_two_piece_print_parts",
    "wing_monolithic_assembly",
    "wing_print_assembly",
    "wing_two_piece_print_assembly",
    "wing_review_assembly",
    "adaptive_volume_mm3",
    "wing_facts",
    "gen_step",
)
