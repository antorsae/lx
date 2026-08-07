"""Final R6F Obi-Wan acceptance checks.

Each OCC-heavy check runs in a fresh guarded process. These checks own
the final Obi-Wan contract; the proud-family regression module contains no
legacy Obi-Wan architecture assertions.
"""

from __future__ import annotations

# The suite-private test_harness module owns fresh-process guarded dispatch.
# Prevent a generic pytest invocation from collecting all OCC-heavy tests
# into one long-lived process and bypassing the local 8 GiB tree cap and
# fresh-process isolation.
__test__ = False

import fcntl
import math
import hashlib
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (
        Path(__file__).resolve().parent,
        PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import re
import subprocess
import sys
import tempfile
import time

import numpy as np
from test_harness import (
    GUARDED_CASE,
    SERVICE_ORCHESTRATOR_CASE,
    GuardedCase,
    run_selected_case,
    run_suite,
)

LM_CUTTER_GROUP_COUNT = 21
R6F_NATIVE_STAGE_SCHEMA_VERSION = 7
R6F_CHECK_LAUNCH_HEADROOM_MB = 2500.0
R6F_CABLE_WORKER_HEADROOM_MB = 3500.0
R6F_HEADROOM_WAIT_TIMEOUT_S = 300.0

# User-accepted actual-BREP baseline, scoped to the one measured interface.
# OCC distance evaluation is stable to the printed 0.001 mm precision but can
# vary slightly below that across kernels, so retain a 0.005 mm repeatability
# band while continuing to fail any material downward regression. Every other
# route/keepout pair remains on the general INSERT_COVER_CLEAR - 0.03 gate.
ACCEPTED_BREP_CLEARANCE_BASELINES_MM = {
    (False, "UM route / LM pads"): 0.260,
    (True, "UM route / LM pads"): 0.357,
}
ACCEPTED_BREP_CLEARANCE_REPEATABILITY_MM = 0.005


def _required_brep_clearance_mm(stand_foot, label, general_minimum_mm):
    baseline = ACCEPTED_BREP_CLEARANCE_BASELINES_MM.get(
        (stand_foot, label))
    if baseline is None:
        return general_minimum_mm
    return baseline - ACCEPTED_BREP_CLEARANCE_REPEATABILITY_MM


def _large_host_execution() -> bool:
    return (
        os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def _state(stand_foot: bool):
    os.environ["LX_STAND_FOOT"] = "1" if stand_foot else "0"
    os.environ["LX_ROUTING_PROFILE"] = "obiwan"


def _min_three_point_radius(points):
    points = np.asarray(points, dtype=float)
    values = []
    for a, b, c in zip(points[:-2], points[1:-1], points[2:]):
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        if area2 > 1e-10:
            values.append(ab * bc * ac / (2.0 * area2))
    return min(values) if values else math.inf


def _min_three_point_radius_record(points):
    """Return minimum sampled radius, center index and center point."""
    points = np.asarray(points, dtype=float)
    values = []
    for index, (a, b, c) in enumerate(
            zip(points[:-2], points[1:-1], points[2:]), start=1):
        ab = np.linalg.norm(b - a)
        bc = np.linalg.norm(c - b)
        ac = np.linalg.norm(c - a)
        area2 = np.linalg.norm(np.cross(b - a, c - a))
        if area2 > 1e-10:
            values.append((ab * bc * ac / (2.0 * area2), index))
    if not values:
        return math.inf, None, None
    radius, index = min(values)
    return float(radius), int(index), tuple(map(float, points[index]))


def _max_turn_deg(points):
    vectors = np.diff(np.asarray(points, dtype=float), axis=0)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    cosines = np.sum(vectors[:-1] * vectors[1:], axis=1)
    return float(np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0))).max())


def _shape_bounds_mm(shape):
    bounds = shape.bounding_box()
    return (
        (bounds.min.X, bounds.min.Y, bounds.min.Z),
        (bounds.max.X, bounds.max.Y, bounds.max.Z),
    )


def _assert_bounds_close(actual, expected, tolerance_mm, label):
    for side, (actual_xyz, expected_xyz) in enumerate(
            zip(actual, expected, strict=True)):
        for axis, (actual_value, expected_value) in enumerate(
                zip(actual_xyz, expected_xyz, strict=True)):
            assert math.isclose(
                actual_value, expected_value, abs_tol=tolerance_mm), (
                    f"{label} bound[{side}][{axis}] {actual_value:.6f} != "
                    f"{expected_value:.6f} mm")


def _trim_points_by_length(points, distance):
    """Return the sampled suffix after an arc-length breakout allowance."""
    points = np.asarray(points, dtype=float)
    lengths = np.concatenate((
        [0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    index = min(int(np.searchsorted(lengths, distance)), len(points) - 2)
    return points[index:]


def test_w22_reference_step_geometry():
    """Import and place the actual W22 STEP in one guarded OCC process."""
    _state(False)
    import run_memory_guarded as memory_guard

    assert memory_guard.is_guarded_process(), (
        "W22 STEP geometry validation must run under the CAD memory guard")
    # The fit module imports build123d/OCC, so keep this import after the
    # guard assertion: an accidental direct call must fail before allocating
    # any CAD-kernel state.
    import lx521_baffle.um_fit as fit
    facts = fit.w22_body_reference_facts()
    provenance = facts["provenance"]
    assert Path(facts["source_step"]).resolve() == (
        fit.W22_REFERENCE_STEP.resolve())
    assert Path(facts["source_step"]).name == "E0022_W22EX001.stp"
    assert facts["source_step_sha256"] == fit.W22_REFERENCE_STEP_SHA256
    assert facts["source_step_sha256"] == provenance["source_sha256"]
    assert facts["units"] == provenance["source_units"] == "mm"
    assert facts["native_to_world"]["rotation"] == {
        "axis": "+X", "degrees": 90.0}
    assert facts["native_to_world"]["axis_map"] == {
        "native_+X": "world_+X",
        "native_+Y_driver_front": "world_+Z_baffle_front",
        "native_+Z": "world_-Y",
    }
    assert provenance["bounds_validation_phase"] == (
        "test_w22_reference_step_geometry")
    assert provenance["reference_geometry_scope"] == "W22EX001_only"
    assert provenance["installed_u22_geometry_verified"] is False
    assert provenance["terminals_or_leads_verified"] is False
    assert facts["physical_measure_required"] is True

    tolerance = provenance["bounds_validation_tolerance_mm"]
    assert tolerance == fit.W22_REFERENCE_BOUNDS_TOLERANCE_MM
    native = fit.load_w22_reference_step_native()
    assert native.is_valid and len(native.solids()) > 0
    native_bounds = _shape_bounds_mm(native)
    _assert_bounds_close(
        native_bounds, facts["native_bounds_mm"], tolerance,
        "native W22 STEP")
    assert math.isclose(
        native_bounds[1][1], fit.W22_NATIVE_FRONT_Y_MM,
        abs_tol=tolerance)

    # Derive the expected world AABB from the imported native AABB and the
    # declared +90-degree X map (x,y,z)->(x,-z,y).  Comparing this independent
    # derivation with both the placed BREP and cached world facts proves that
    # the structured transform is actually applied, not merely documented.
    tx, ty, tz = facts["native_to_world"]["translation_mm"]
    native_min, native_max = native_bounds
    assert math.isclose(
        tz + native_max[1], facts["world_front_datum_z_mm"],
        abs_tol=tolerance)
    derived_world_bounds = (
        (tx + native_min[0], ty - native_max[2], tz + native_min[1]),
        (tx + native_max[0], ty - native_min[2], tz + native_max[1]),
    )
    placed = fit.place_w22_reference_step(native)
    del native
    assert placed.is_valid and len(placed.solids()) > 0
    world_bounds = _shape_bounds_mm(placed)
    _assert_bounds_close(
        world_bounds, derived_world_bounds, tolerance,
        "structured W22 placement")
    _assert_bounds_close(
        world_bounds, facts["transformed_world_bounds_mm"], tolerance,
        "cached W22 world bounds")
    assert math.isclose(
        world_bounds[1][2], facts["world_front_datum_z_mm"],
        abs_tol=tolerance)

    native_size = tuple(
        native_max[index] - native_min[index] for index in range(3))
    world_size = tuple(
        world_bounds[1][index] - world_bounds[0][index]
        for index in range(3))
    assert math.isclose(world_size[0], native_size[0], abs_tol=tolerance)
    assert math.isclose(world_size[1], native_size[2], abs_tol=tolerance)
    assert math.isclose(world_size[2], native_size[1], abs_tol=tolerance)

    # The service phases screen cables and connector motion against the
    # intentionally simpler stepped W22 envelope.  Prove on the real,
    # transformed STEP that this proxy is conservative; otherwise clearance
    # to the proxy would not imply clearance to the reference geometry.
    conservative = fit.w22_body_keepout(include_flange=True)
    outside = placed - conservative
    outside_volume = 0.0 if outside is None else sum(
        solid.volume for solid in outside.solids())
    outside_bounds = None
    if outside_volume > 1e-9:
        bb = outside.bounding_box()
        outside_bounds = (
            (bb.min.X, bb.min.Y, bb.min.Z),
            (bb.max.X, bb.max.Y, bb.max.Z),
        )
        outside_components = []
        for solid in outside.solids():
            solid_bb = solid.bounding_box()
            outside_components.append((
                solid.volume,
                (solid_bb.min.X, solid_bb.min.Y, solid_bb.min.Z),
                (solid_bb.max.X, solid_bb.max.Y, solid_bb.max.Z),
            ))
    else:
        outside_components = []
    assert outside_volume < 0.05, (
        "transformed W22 reference escapes conservative service keepout by "
        f"{outside_volume:.3f} mm3; bounds={outside_bounds}; "
        f"components={outside_components}")
    print(
        "  hash-pinned W22 STEP native/world bounds, +90deg-X "
        "front-z=18.3 placement, and conservative-proxy containment pass; "
        "installed U22 remains unqualified")


def test_route_contract():
    from shapely.geometry import LineString, Point, Polygon, box
    _state(False)
    from lx521_baffle.base import (
        BRIDGE_HOLE_XY,
        BRIDGE_INSERT_D_MM,
        L22_CUTOUT,
        L22_PILOT_ANGLES_DEG,
        L22_PILOT_PCD_MM,
        M5_INSERT_ENTRY_D_MM,
        UM_TERMINAL_CLOCK_DEG,
    )
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.bridge as bridge
    import lx521_baffle.cables as cables
    import lx521_baffle.obiwan.floor as floor
    import lx521_baffle.obiwan.route as route
    import run_memory_guarded as memory_guard
    import export_obiwan_staged as staged
    import write_obiwan_release_manifest as release_manifest
    from lx521_baffle.magnets import (
        CAPTIVE_LAND_MM,
        CAVITY_DEPTH_MM,
        CAVITY_DIAMETER_MM,
        DEFAULT_SPEC,
        FACE_SKIN_MM,
        INNER_SKIN_MM,
        INTERFACE_GAP_MM,
        NOMINAL_PAIRED_FACE_SEPARATION_MM,
        ROOF_ANGLE_DEG,
        wall_cavity_tools,
    )

    assert route.route_inner_cutter_group_count("lm") == LM_CUTTER_GROUP_COUNT

    assert L22_PILOT_ANGLES_DEG == (
        0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    assert flush.OBIWAN_LM_PILOT_ANGLES_DEG == L22_PILOT_ANGLES_DEG
    for xy in flush.LM_PILOT_XY:
        assert math.isclose(
            math.dist(xy, L22_CUTOUT[:2]), L22_PILOT_PCD_MM / 2.0,
            abs_tol=1e-8)
    magnet_sites = core.side_magnet_sites()
    assert len(magnet_sites) == 6
    assert sum(site["name"].endswith("left") for site in magnet_sites) == 3
    assert sum(site["name"].endswith("right") for site in magnet_sites) == 3
    assert {site["driver"] for site in magnet_sites} == {"lm", "um"}
    lm_magnets = [site for site in magnet_sites if site["driver"] == "lm"]
    assert {site["name"] for site in lm_magnets} == {
        "lm_upper_left", "lm_upper_right",
        "lm_lower_left", "lm_lower_right"}
    assert all(site["magnet_fully_buried"] for site in lm_magnets)
    lm_by_name = {site["name"]: site for site in lm_magnets}
    assert lm_by_name["lm_upper_left"]["angle_deg"] == 116.0
    assert lm_by_name["lm_upper_right"]["angle_deg"] == 64.0
    assert {name: lm_by_name[name]["z_mm"] for name in lm_by_name} == {
        "lm_upper_left": 15.10, "lm_upper_right": 15.10,
        "lm_lower_left": 15.10, "lm_lower_right": 15.10}
    expected_outer_right, expected_normal_right = (
        bridge.bridge_soft_blend_frame(0.5, "right"))
    expected_face_right = tuple(
        point - 0.15 * normal
        for point, normal in zip(
            expected_outer_right, expected_normal_right, strict=True))
    assert np.allclose(
        lm_by_name["lm_lower_right"]["outer_surface_face"],
        expected_outer_right, atol=1e-12)
    assert np.allclose(
        lm_by_name["lm_lower_right"]["face"],
        expected_face_right, atol=1e-12)
    assert np.allclose(
        lm_by_name["lm_lower_right"]["normal"],
        expected_normal_right, atol=1e-12)
    assert np.allclose(
        lm_by_name["lm_lower_left"]["outer_surface_face"],
        (-expected_outer_right[0], expected_outer_right[1]), atol=1e-12)
    assert np.allclose(
        lm_by_name["lm_lower_left"]["normal"],
        (-expected_normal_right[0], expected_normal_right[1]), atol=1e-12)
    assert all(
        lm_by_name[name]["interface_kind"] == "shoulder"
        for name in ("lm_lower_left", "lm_lower_right"))
    assert all(
        math.isclose(
            lm_by_name[name]["face_offset_mm"], -0.15, abs_tol=1e-12)
        for name in ("lm_lower_left", "lm_lower_right"))
    assert all(math.isclose(
        lm_by_name[name]["shoulder_parameter"], 0.5, abs_tol=1e-12)
        for name in ("lm_lower_left", "lm_lower_right"))
    assert all(math.isclose(
        lm_by_name[name]["carrier_cavity_face_inset_mm"], 0.15,
        abs_tol=1e-12)
        for name in ("lm_lower_left", "lm_lower_right"))
    assert all(lm_by_name[name]["continuous_flush_shoulder"] is True
               for name in ("lm_lower_left", "lm_lower_right"))
    assert all(
        math.isclose(
            lm_by_name[name]["face_offset_mm"], 0.65, abs_tol=1e-12)
        for name in ("lm_upper_left", "lm_upper_right"))
    assert all(
        math.isclose(
            lm_by_name[name]["local_captive_backing_boss_mm"], 0.0,
            abs_tol=1e-12)
        for name in ("lm_upper_left", "lm_upper_right"))
    assert all(math.isclose(
        lm_by_name[name]["continuous_flush_ring_fairing_mm"], 0.80,
        abs_tol=1e-12)
        for name in ("lm_upper_left", "lm_upper_right"))
    assert all(math.isclose(
        lm_by_name[name]["carrier_cavity_face_inset_mm"], 0.15,
        abs_tol=1e-12)
        for name in ("lm_upper_left", "lm_upper_right"))
    assert math.isclose(
        core.LM_SHOULDER_MAGNET_PARAMETER, 0.5, abs_tol=1e-12)
    assert math.isclose(
        core.THICKNESS_MM
        - (core.LM_SHOULDER_MAGNET_Z
           + core.SIDE_MAGNET_POCKET_D / 2.0),
        0.60, abs_tol=1e-12)

    # The complete carrier captive lands are above the Option-B vertical
    # tangent and contained by the shared upper shoulder.  This is the
    # geometric proof that floor mode needs no lower magnet rail.
    pocket_radius_mm = core.SIDE_MAGNET_POCKET_D / 2.0
    shoulder_plan = bridge.common_lm_wing_contact_plan()
    tangent_half_width = (
        pocket_radius_mm + DEFAULT_SPEC.side_wall_margin_mm)
    for site in (
            lm_by_name["lm_lower_left"],
            lm_by_name["lm_lower_right"]):
        face = np.asarray(site["face"], dtype=float)
        normal = np.asarray(site["normal"], dtype=float)
        tangent = np.asarray((-normal[1], normal[0]), dtype=float)
        inner = face - CAPTIVE_LAND_MM * normal
        land_plan = Polygon((
            tuple(inner - tangent_half_width * tangent),
            tuple(face - tangent_half_width * tangent),
            tuple(face + tangent_half_width * tangent),
            tuple(inner + tangent_half_width * tangent),
        ))
        assert land_plan.bounds[1] > floor.FLOOR_BEND_VERTICAL_TANGENT_Y_MM
        assert land_plan.difference(shoulder_plan).area < 1.0e-8
    um_magnets = [site for site in magnet_sites if site["driver"] == "um"]
    assert {site["angle_deg"] for site in um_magnets} == {50.5, 129.5}
    assert {site["clock_from_top_deg"] for site in um_magnets} == {
        -39.5, 39.5}
    assert all(site["magnet_fully_buried"] for site in um_magnets)
    assert all(not site["proud_ear_added"] for site in um_magnets)
    assert all(math.isclose(
        site["face_offset_mm"], 0.65, abs_tol=1e-12)
        for site in um_magnets)
    assert all(math.isclose(
        site["local_captive_backing_boss_mm"], 0.0, abs_tol=1e-12)
        for site in um_magnets)
    assert all(math.isclose(
        site["continuous_flush_ring_fairing_mm"], 0.80,
        abs_tol=1e-12) for site in um_magnets)
    assert all(math.isclose(
        site["carrier_cavity_face_inset_mm"], 0.15,
        abs_tol=1e-12) for site in um_magnets)
    assert all(site["z_mm"] == 15.10 for site in um_magnets)
    assert core.SIDE_MAGNET_D == 5.0
    assert core.SIDE_MAGNET_POCKET_D == CAVITY_DIAMETER_MM == 5.20
    assert core.SIDE_MAGNET_DEPTH == CAVITY_DEPTH_MM == 2.10
    assert math.isclose(
        core.SIDE_MAGNET_CAPTIVE_LAND, CAPTIVE_LAND_MM, abs_tol=1e-12)
    assert math.isclose(CAPTIVE_LAND_MM, 3.00, abs_tol=1e-12)
    assert core.SIDE_MAGNET_FACE_SKIN == FACE_SKIN_MM == 0.45
    assert core.SIDE_MAGNET_INNER_SKIN == INNER_SKIN_MM == 0.45
    assert core.SIDE_INTERFACE_GAP == INTERFACE_GAP_MM == 0.05
    assert math.isclose(
        NOMINAL_PAIRED_FACE_SEPARATION_MM, 0.95, abs_tol=1e-12)
    assert ROOF_ANGLE_DEG == 45.0
    assert math.isclose(
        core.LM_CORE_R + 0.65 - CAPTIVE_LAND_MM - flush.LM_RECESS_R,
        0.05, abs_tol=1e-12)
    assert math.isclose(
        core.UM_CORE_R + 0.65 - CAPTIVE_LAND_MM - flush.UM_RECESS_R,
        0.05, abs_tol=1e-12)
    assert math.isclose(core.LM_VISIBLE_RING_R, 113.80, abs_tol=1e-12)
    assert math.isclose(core.UM_VISIBLE_RING_R, 52.50, abs_tol=1e-12)
    assert math.isclose(
        core.UM_T_FAIRING_CUSP_HALF_WIDTH_MM,
        core.T_UM_WEB_BLEND_START_X, abs_tol=1e-12)
    lm_visible_plan = core.side_ring_outer_plan("lm")
    um_visible_plan = core.side_ring_outer_plan("um")
    assert lm_visible_plan.intersection(um_visible_plan).area < 1e-9
    assert math.isclose(
        lm_visible_plan.distance(um_visible_plan), core.CORE_RING_GAP,
        abs_tol=1e-9)
    um_nominal_plan = Point(*core.UM_CUTOUT[:2]).buffer(
        core.UM_CORE_R, resolution=256)
    um_t_cusp = box(
        -core.T_UM_CABLE_MOUTH_HALF_WIDTH,
        core.UM_CUTOUT[1],
        core.T_UM_CABLE_MOUTH_HALF_WIDTH,
        core.UM_CUTOUT[1] + core.UM_VISIBLE_RING_R + 1.0,
    )
    assert um_visible_plan.difference(um_nominal_plan).intersection(
        um_t_cusp).area < 1e-9
    for site in magnet_sites:
        if site["interface_kind"] != "ring":
            continue
        visible_plan = (
            lm_visible_plan if site["driver"] == "lm"
            else um_visible_plan)
        assert Point(*site["outer_surface_face"]).distance(
            visible_plan.boundary) < 0.001
    assert math.isclose(
        core.THICKNESS_MM
        - (core.SIDE_MAGNET_Z["um"]
           + core.SIDE_MAGNET_POCKET_D / 2.0),
        0.6, abs_tol=1e-12)
    # Every station is generated by the shared proven helper. Ring and lower
    # shoulder cavity datums sit 0.15 mm beneath their continuous visible
    # surfaces. Screen the full 3.00-mm land, not merely the cavity.
    for site in magnet_sites:
        tools = wall_cavity_tools(
            name=site["name"], face=site["face"],
            outward=(*site["normal"], 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP)
        facts = tools.facts()
        assert len(tools.cutters) == 3
        assert facts["closure_kind"] == "transverse_gable_45deg"
        assert facts["cavity_diameter_mm"] == 5.20
        assert facts["cavity_depth_mm"] == 2.10
        assert facts["face_skin_mm"] == facts["inner_skin_mm"] == 0.45
        assert math.isclose(
            facts["captive_land_mm"], 3.00, abs_tol=1e-12)
        assert facts["roof_angle_deg"] == 45.0
        assert math.isclose(
            facts["paired_magnet_face_separation_mm"], 0.95,
            abs_tol=1e-12)
        effective_pair_separation = (
            facts["paired_magnet_face_separation_mm"]
            + site["carrier_cavity_face_inset_mm"])
        expected_pair_separation = 1.10
        assert math.isclose(
            effective_pair_separation, expected_pair_separation,
            abs_tol=1e-12)
        assert facts["minimum_retaining_path_mm"] == 0.42
        assert facts["actual_face_xyz_mm"][:2] == list(site["face"])
        assert all(math.isclose(actual, expected, abs_tol=1e-12)
                   for actual, expected in zip(
                       facts["marked_pole_axis_xyz"],
                       (*site["normal"], 0.0), strict=True))
        assert facts["print_up_source_xyz"] == [0.0, 0.0, -1.0]
        expected_roof_start = 5.80
        assert math.isclose(
            facts["cavity_bury_roof_start_print_z_mm"],
            expected_roof_start, abs_tol=1e-9)
        assert math.isclose(
            facts["roof_apex_print_z_mm"] - expected_roof_start,
            DEFAULT_SPEC.roof_height_mm, abs_tol=1e-9)
        assert math.isclose(
            facts["required_min_part_top_print_z_mm"]
            - facts["roof_apex_print_z_mm"],
            INNER_SKIN_MM, abs_tol=1e-9)

    lm_magnet_insert_gaps = []
    for site in lm_magnets:
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        inner = face - CAPTIVE_LAND_MM * normal
        axis = LineString((inner, face))
        lm_magnet_insert_gaps.extend(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0 - flush.PAD_D_MM / 2.0
            for xy in flush.LM_PILOT_XY)
    assert min(lm_magnet_insert_gaps) >= 2.0
    for site in lm_magnets:
        if not site["name"].startswith("lm_lower"):
            continue
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        axis = LineString((
            face - CAPTIVE_LAND_MM * normal, face))
        # The shoulder pockets remain far from both the no-floor bridge M5
        # pattern and the LM flange inserts.
        bridge_bore_gap = min(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0
            - M5_INSERT_ENTRY_D_MM / 2.0
            for xy in BRIDGE_HOLE_XY)
        bridge_boss_gap = min(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0 - flush.PAD_D_MM / 2.0
            for xy in BRIDGE_HOLE_XY)
        assert bridge_bore_gap >= 25.6
        assert bridge_boss_gap >= 24.0
        assert min(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0 - flush.PAD_D_MM / 2.0
            for xy in flush.LM_PILOT_XY) >= 13.6

    # State-specific nearest-insert screen. The relocated shoulder sites are
    # governed by the same LM flange insert in both states; the no-floor
    # 40 x 50 bridge pattern is now farther away. Use the full
    # inward pocket axis plus the D5.2 radius, and screen both the maximum
    # D6.5 entry
    # and the conservative D9.6 load-bearing envelope.  Mirrored sites must
    # match exactly.
    lower_lm_sites = tuple(
        site for site in lm_magnets
        if site["interface_kind"] == "shoulder")
    insert_sets = {
        "floor": tuple(flush.LM_PILOT_XY),
        "no_floor": tuple(flush.LM_PILOT_XY) + tuple(BRIDGE_HOLE_XY),
    }
    state_insert_gaps = {}
    for state, insert_xy in insert_sets.items():
        bore_gaps = []
        boss_gaps = []
        for site in lower_lm_sites:
            normal = np.asarray(site["normal"], dtype=float)
            face = np.asarray(site["face"], dtype=float)
            axis = LineString((
                face - CAPTIVE_LAND_MM * normal, face))
            bore_gaps.append(min(
                axis.distance(Point(*xy))
                - pocket_radius_mm - M5_INSERT_ENTRY_D_MM / 2.0
                for xy in insert_xy))
            boss_gaps.append(min(
                axis.distance(Point(*xy))
                - pocket_radius_mm - flush.PAD_D_MM / 2.0
                for xy in insert_xy))
        assert math.isclose(bore_gaps[0], bore_gaps[1], abs_tol=1e-9)
        assert math.isclose(boss_gaps[0], boss_gaps[1], abs_tol=1e-9)
        state_insert_gaps[state] = (bore_gaps[0], boss_gaps[0])
    assert state_insert_gaps["floor"][0] >= 15.1
    assert state_insert_gaps["floor"][1] >= 13.6
    assert np.allclose(
        state_insert_gaps["floor"], state_insert_gaps["no_floor"],
        atol=1e-9)
    um_magnet_insert_gaps = []
    for site in um_magnets:
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        inner = face - CAPTIVE_LAND_MM * normal
        axis = LineString((inner, face))
        um_magnet_insert_gaps.extend(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0
            - core.UM_INSERT_BOSS_D / 2.0
            for xy in flush.UM_PILOT_XY)
    assert min(um_magnet_insert_gaps) >= 1.0
    assert core.JOINT_BOSS_D == core.TWEETER_JOINT_BOSS_D == 9.0
    assert (core.JOINT_FUNCTIONAL_BOSS_D
            == core.TWEETER_JOINT_FUNCTIONAL_BOSS_D == 9.8)
    assert core.JOINT_NECK_D == core.TWEETER_JOINT_NECK_D == 4.0
    assert core.LM_CORE_R == 113.0
    assert math.isclose(core.UM_CORE_R, 51.7, abs_tol=1e-12)
    assert math.isclose(core.CORE_RING_GAP, 0.4, abs_tol=1e-9)
    assert core.JOINT_EAR_X == (-32.0, 32.0)
    assert core.TWEETER_JOINT_X == (-24.0, 24.0)
    assert core.SEAT_MEMBRANE_T == 0.85
    assert route.DUCT_D == 8.2
    assert route.TS_DUCT_D == 6.0
    assert route.LM_CABLE_D_EST == 7.8
    assert route.CABLE_D_EST == 7.0
    assert route.TS_CABLE_D_EST == 5.2
    assert route.TUNNEL_SKIN == 0.8
    assert route.TUNNEL_ROOF_SKIN == 0.85
    assert route.LM_ROUTE_OWNER_CLEARANCE == 0.05
    assert route.BURIAL_WEB_OWNER_INSET == 0.05
    assert math.isclose(
        route.LM_VISIBLE_RING_R, core.LM_VISIBLE_RING_R, abs_tol=1e-12)
    assert math.isclose(
        route.MAIN_LM_ROUTE_R + route.CUTTER_R,
        core.LM_CORE_R - route.LM_ROUTE_OWNER_CLEARANCE,
        abs_tol=1e-12)
    assert math.isclose(
        route.TS_LM_ROUTE_R + route.TS_CUTTER_R,
        core.LM_CORE_R - route.LM_ROUTE_OWNER_CLEARANCE,
        abs_tol=1e-12)
    assert math.isclose(
        route.MAIN_LM_ROUTE_R + route.MAIN_OUTER_R,
        core.LM_VISIBLE_RING_R - route.LM_ROUTE_OWNER_CLEARANCE,
        abs_tol=1e-12)
    assert math.isclose(
        route.TS_LM_ROUTE_R + route.TS_OUTER_R,
        core.LM_VISIBLE_RING_R - route.LM_ROUTE_OWNER_CLEARANCE,
        abs_tol=1e-12)
    assert route.TUBE_SECTION_SPACING == 5.5
    assert route.TUBE_SECTION_SIDES == 8
    assert (20.0 - math.sqrt(
        20.0 ** 2 - (route.TUBE_SECTION_SPACING / 2.0) ** 2)) < 0.20
    assert memory_guard.MEMORY_PROFILES["local-macos"] == {
        "max_rss_mb": 8192,
        "min_free_mb": 0,
        "max_guard_slots": 1,
        "max_light_guard_slots": 0,
    }
    assert memory_guard.MEMORY_PROFILES["osado-512g"] == {
        "max_rss_mb": 512 * 1024,
        "min_free_mb": 64 * 1024,
        "max_guard_slots": 16,
        "max_light_guard_slots": 64,
    }
    assert memory_guard.MAX_RSS_MB <= memory_guard.PROFILE_MAX_RSS_MB
    assert memory_guard.MIN_FREE_MB >= memory_guard.PROFILE_MIN_FREE_MB
    assert memory_guard.GUARD_SLOTS <= memory_guard.PROFILE_MAX_GUARD_SLOTS
    assert staged.SCHEMA_VERSION == R6F_NATIVE_STAGE_SCHEMA_VERSION == 7
    assert release_manifest.FORMAT_VERSION == 12
    assert staged.ATTACHMENT_KEYS_BASE == ("addon_tweeter_crescent",)
    assert set(staged.PRINT_PART_SPECS) == {
        "core_lm_carrier",
        "core_um_carrier",
        "optional_lm_keyed_1_of_2_bottom",
        "optional_lm_keyed_2_of_2_top",
        "addon_tweeter_crescent",
    }
    assert staged._expected_print_keys(True) == (
        "core_lm_carrier",
        "core_um_carrier",
        "optional_lm_keyed_1_of_2_bottom",
        "optional_lm_keyed_2_of_2_top",
        "addon_tweeter_crescent",
    )
    assert staged.OPTIONAL_LM_SPLIT_KEYS == (
        "optional_lm_keyed_1_of_2_bottom",
        "optional_lm_keyed_2_of_2_top",
    )
    assert {
        staged.PRINT_PART_SPECS[key]["group"]
        for key in staged.OPTIONAL_LM_SPLIT_KEYS
    } == {"lm_split"}
    assert all(
        "grommet" not in key
        for stand_foot in (False, True)
        for key in staged._expected_print_keys(stand_foot))
    guard_policy = staged._guard_policy()
    assert guard_policy["memory_profile"] == memory_guard.MEMORY_PROFILE
    assert guard_policy["max_process_tree_rss_mib"] == memory_guard.MAX_RSS_MB
    assert guard_policy["min_immediately_reclaimable_mib"] == (
        memory_guard.MIN_FREE_MB)
    assert guard_policy["guard_slots"] == memory_guard.GUARD_SLOTS
    assert guard_policy["worker_launch_headroom_mib"] == (
        3200.0 if memory_guard.MIN_FREE_MB else 0.0)
    if memory_guard.MEMORY_PROFILE == "local-macos":
        assert 0 < memory_guard.MAX_RSS_MB <= 8192
        assert memory_guard.MIN_FREE_MB >= 0
        assert memory_guard.GUARD_SLOTS == 1
        assert guard_policy["aggregate_cgroup_max_mib"] is None
    else:
        assert memory_guard.MEMORY_PROFILE == "osado-512g"
        assert memory_guard.MIN_FREE_MB >= 64 * 1024
        assert guard_policy["aggregate_cgroup_max_mib"] == 512 * 1024
    runtime_identity = staged._runtime_identity()
    assert runtime_identity["python"] == sys.version
    assert set(runtime_identity["packages"]) == set(
        staged.RUNTIME_DISTRIBUTIONS)

    facts = route.route_facts()
    assert facts["open_bore_jump_count"] == 0
    assert math.isclose(facts["tunnel_floor_skin_mm"], 0.8, abs_tol=1e-12)
    assert facts["lm_roof_mm"] >= 0.84
    assert facts["ts_lm_roof_mm"] >= 0.84
    assert facts["ts_um_roof_mm"] >= 0.84
    assert facts["lm_seat_membrane_mm"] == 0.85
    assert facts["um_seat_membrane_mm"] == 0.85
    assert 82.0 <= facts["crossover_angle_deg"] <= 95.0
    assert facts["main_bridge_start_handle_mm"] == 11.0
    assert facts["main_bridge_end_handle_mm"] == 17.0
    assert facts["t_bridge_start_handle_mm"] == 30.0
    assert facts["t_bridge_end_handle_mm"] == 80.0
    assert facts["crossover_t_z_mm"] > facts["crossover_main_z_mm"]
    assert (facts["crossover_nominal_void_gap_mm"]
            >= route.CROSSOVER_MIN_CLEARANCE)
    assert facts["crossover_free_um_to_t_cover_gap_mm"] >= 0.25
    assert facts["crossover_physical_gap_mm"] >= 1.5
    assert facts["crossover_plan_overlap_mm"] <= 30.0
    assert facts["terminal_clock_deg"] == UM_TERMINAL_CLOCK_DEG == 283.0
    assert facts["terminal_plan_bend_radius_mm"] == 15.0
    assert facts["terminal_mouth_tangent"] == (
        "clockwise_circumferential_body_clear")
    assert facts["um_pilot_bump_names"] == ("um_pilot_328", "um_pilot_58")
    assert set(facts["covered_bump_names"]) == {
        "lm_pilot_300", "lm_pilot_0", "lm_pilot_60",
        "lm_pilot_240", "lm_pilot_180", "lm_pilot_120",
        "um_pilot_328", "um_pilot_58",
    }
    assert facts["solid_backfill_count"] == 8
    assert set(facts["solid_backfill_names"]) == set(
        facts["covered_bump_names"])
    assert facts["solid_backfill_tube_overlap_mm"] == 0.55
    assert facts["solid_backfill_added_rear_depth_mm"] == 0.0
    assert facts["solid_backfill_floor_hardware_exceptions"] == ()
    assert all(record["filled_height_mm"] > 0.5
               for record in facts["solid_backfill_records"])
    assert set(facts["lm_burial_webs"]) == {"UM", "T"}
    assert facts["lm_burial_web_count"] >= 2
    assert facts["lm_burial_web_growth_upper_bound_mm3"] > 0.0
    assert facts["lm_burial_web_floor_hardware_clear_d_mm"] is None
    assert math.isclose(
        facts["lm_burial_webs"]["UM"]["full_width_mm"],
        2.0 * (
            route.MAIN_OUTER_R + route.BURIAL_WEB_LATERAL_OVERLAP),
        abs_tol=1e-12)
    assert math.isclose(
        facts["lm_burial_webs"]["T"]["full_width_mm"],
        2.0 * (
            route.TS_OUTER_R + route.BURIAL_WEB_LATERAL_OVERLAP),
        abs_tol=1e-12)
    assert all(math.isclose(
        record["tube_center_overlap_mm"], route.TUNNEL_FUSE_OVERLAP,
        abs_tol=1e-12) for record in facts["lm_burial_webs"].values())
    assert all(math.isclose(
        record["lateral_fusion_overlap_mm"],
        route.BURIAL_WEB_LATERAL_OVERLAP, abs_tol=1e-12)
        for record in facts["lm_burial_webs"].values())
    assert set(facts["um_burial_webs"]) == {"T"}
    assert facts["um_burial_web_count"] >= 1
    assert facts["um_burial_web_growth_upper_bound_mm3"] > 0.0
    assert math.isclose(
        facts["um_burial_webs"]["T"]["full_width_mm"],
        2.0 * (
            route.TS_OUTER_R + route.BURIAL_WEB_LATERAL_OVERLAP),
        abs_tol=1e-12)
    assert math.isclose(
        facts["um_burial_webs"]["T"]["tube_center_overlap_mm"],
        route.TUNNEL_FUSE_OVERLAP, abs_tol=1e-12)
    assert math.isclose(
        facts["um_burial_webs"]["T"]["lateral_fusion_overlap_mm"],
        route.BURIAL_WEB_LATERAL_OVERLAP, abs_tol=1e-12)
    assert facts["functional_lm_feed_count"] == 2
    assert facts["functional_lm_feed_mode"] == (
        "bridge_rear_normal_entry_then_buried_rise")
    assert facts["central_owner_feed_state"] == "no_floor"
    assert np.allclose(
        facts["central_owner_feed_xy"],
        (cables.OBIWAN_NO_FLOOR_UM_ENTRY_XY,
         cables.OBIWAN_NO_FLOOR_T_ENTRY_XY), atol=1e-12)
    assert math.isclose(
        facts["central_owner_feed_rear_z_mm"], 5.3, abs_tol=1e-9)
    assert np.allclose(
        facts["functional_lm_feed_points"],
        ((*cables.OBIWAN_NO_FLOOR_UM_ENTRY_XY, 5.3),
         (*cables.OBIWAN_NO_FLOOR_T_ENTRY_XY, 5.3)), atol=1e-9)
    assert np.allclose(
        facts["buried_route_start_points"],
        ((*cables.OBIWAN_NO_FLOOR_UM_ENTRY_XY,
          route.NO_FLOOR_MAIN_FEED_START_Z),
         (*cables.OBIWAN_NO_FLOOR_T_ENTRY_XY,
          route.NO_FLOOR_T_FEED_START_Z)), atol=1e-9)
    assert np.allclose(
        facts["no_floor_route_rear_skin_mm"], (0.80, 0.80),
        atol=1e-12)
    assert np.allclose(
        facts["no_floor_entry_route_overlap_mm"], (0.10, 1.20),
        atol=1e-12)
    assert facts["functional_lm_feed_web_omitted"]
    ports = facts["no_floor_rear_entry_ports"]
    assert facts["no_floor_entry_layout"] == (
        "d20_lm_top_t_lower_left_um_lower_right")
    assert np.allclose(
        facts["no_floor_entry_window_center_xy_mm"],
        cables.SUPPORT_WINDOW[:2], atol=1e-12)
    assert facts["no_floor_entry_window_diameter_mm"] == 20.0
    assert facts["no_floor_lm_entry_buried_relief_radial_mm"] == 0.005
    assert facts["no_floor_lm_entry_buried_relief_rear_skin_mm"] == 0.10
    assert facts["no_floor_t_entry_buried_relief_radial_mm"] == 0.005
    assert facts["no_floor_t_entry_buried_relief_rear_skin_mm"] == 0.10
    assert tuple(port["name"] for port in ports) == ("lm", "t", "um")
    assert np.allclose(
        tuple(port["xy_mm"] for port in ports),
        (cables.OBIWAN_NO_FLOOR_LM_ENTRY_XY,
         cables.OBIWAN_NO_FLOOR_T_ENTRY_XY,
         cables.OBIWAN_NO_FLOOR_UM_ENTRY_XY), atol=1e-12)
    assert tuple(port["diameter_mm"] for port in ports) == (9.0, 6.0, 8.2)
    vestibules = facts["no_floor_rear_entry_vestibules"]
    assert tuple(item["name"] for item in vestibules) == ("t", "um")
    assert tuple(item["diameter_mm"] for item in vestibules) == (6.0, 8.2)
    assert np.allclose(
        tuple(item["xy_mm"] for item in vestibules),
        (cables.OBIWAN_NO_FLOOR_T_ENTRY_XY,
         cables.OBIWAN_NO_FLOOR_UM_ENTRY_XY), atol=1e-12)
    assert np.allclose(
        tuple(item["rear_skin_mm"] for item in vestibules),
        (0.80, 0.80), atol=1e-12)
    assert facts["no_floor_um_entry_cap_relief_half_length_mm"] == 0.60
    assert facts["no_floor_um_entry_cap_relief_radial_inset_mm"] == 0.010
    assert facts["no_floor_entry_vestibule_rear_skin_mm"] == 0.80
    cap_relief = route.no_floor_rear_entry_cap_relief_cutters()[0]
    assert cap_relief.bounding_box().min.Z >= 6.10 - 1.0e-8
    assert cap_relief.distance_to(
        route.no_floor_rear_entry_bore_cutters()[0]) >= 0.80
    window_center = np.asarray(
        facts["no_floor_entry_window_center_xy_mm"], dtype=float)
    window_radius = facts["no_floor_entry_window_diameter_mm"] / 2.0
    for port in ports:
        radial_rim = (
            window_radius
            - np.linalg.norm(np.asarray(port["xy_mm"]) - window_center)
            - port["diameter_mm"] / 2.0)
        assert radial_rim >= 0.72
    for index, first in enumerate(ports):
        for second in ports[index + 1:]:
            web = (
                math.dist(first["xy_mm"], second["xy_mm"])
                - first["diameter_mm"] / 2.0
                - second["diameter_mm"] / 2.0)
            assert web >= 0.80
    buried_lm_radius = (
        ports[0]["diameter_mm"] / 2.0
        + facts["no_floor_lm_entry_buried_relief_radial_mm"])
    buried_lm_rim = (
        window_radius
        - np.linalg.norm(np.asarray(ports[0]["xy_mm"]) - window_center)
        - buried_lm_radius)
    assert buried_lm_rim >= 0.72
    for neighbour in ports[1:]:
        buried_web = (
            math.dist(ports[0]["xy_mm"], neighbour["xy_mm"])
            - buried_lm_radius - neighbour["diameter_mm"] / 2.0)
        assert buried_web >= 0.80
    buried_t_radius = (
        ports[1]["diameter_mm"] / 2.0
        + facts["no_floor_t_entry_buried_relief_radial_mm"])
    buried_t_rim = (
        window_radius
        - np.linalg.norm(np.asarray(ports[1]["xy_mm"]) - window_center)
        - buried_t_radius)
    assert buried_t_rim >= 0.72
    buried_t_um_web = (
        math.dist(ports[1]["xy_mm"], ports[2]["xy_mm"])
        - buried_t_radius - ports[2]["diameter_mm"] / 2.0)
    assert buried_t_um_web >= 0.80
    assert all(math.isclose(port["rear_z_mm"], 5.05, abs_tol=1e-12)
               for port in ports)
    assert math.isclose(ports[0]["inner_z_mm"], 13.8, abs_tol=1e-12)
    assert all(math.isclose(port["inner_z_mm"], 10.3, abs_tol=1e-12)
               for port in ports[1:])
    assert facts["functional_all_cable_feed_count"] == 3
    assert facts["functional_all_cable_feed_names"] == ("lm", "um", "t")
    assert route.NO_FLOOR_MAIN_ENTRY_JOIN_XY == (12.9, 78.0)
    assert route.NO_FLOOR_T_ENTRY_JOIN_XY == (-12.0, 74.0)
    assert route.NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG == 34.75
    assert route.NO_FLOOR_T_ENTRY_START_BEARING_DEG == 126.0
    assert route.NO_FLOOR_LM_EXIT_PLAN_BEARING_DEG == 120.0
    # At the upper bridge-insert row, both outer feeds pass toward x=0:
    # T to the right of the x=-20 insert and UM to the left of x=+20.
    # Their complete swept envelopes are checked against the exact insert
    # BREPs in the final mouth contract below.
    main_route = route.route_cable_points(0.05)
    t_route = route.ts_cable_points(0.05)
    assert facts["no_floor_service_patch_margin_mm"] == 6.0
    assert np.allclose(
        facts["no_floor_service_patch_bounds_mm"],
        (-26.0, 14.0, 26.0, 76.0), atol=1e-12)
    assert facts["no_floor_service_patch_release_mode"] == (
        "hold_through_flat_service_patch_then_descend_behind_lm_recess")
    assert set(facts["no_floor_service_patch_routes"]) == {"um", "t"}
    for label, points, inner_radius, outer_radius in (
            ("um", main_route, route.CUTTER_R, route.MAIN_OUTER_R),
            ("t", t_route, route.TS_CUTTER_R, route.TS_OUTER_R)):
        record = facts["no_floor_service_patch_routes"][label]
        assert record["transition_length_mm"] >= 15.0, (label, record)
        assert record["release_station_mm"] > record["guard_end_station_mm"]
        assert math.isclose(
            record["ring_entry_center_radius_mm"],
            route.LM_VISIBLE_RING_R, abs_tol=1e-12)
        assert math.isclose(
            record["ring_entry_overlap_mm"],
            route.NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM, abs_tol=1e-12)
        assert record["ring_deep_owner_entry_station_mm"] >= (
            record["patch_guard_end_station_mm"])
        assert record["guard_end_station_mm"] >= (
            record["ring_deep_owner_entry_station_mm"]
            + route.NO_FLOOR_RING_ENTRY_BURIAL_OVERLAP_MM - 1e-9)
        expected_flat_guard = record["patch_guard_end_station_mm"]
        assert math.isclose(
            record["flat_face_guard_end_station_mm"],
            expected_flat_guard, abs_tol=0.21)
        assert record["protects_lm_recess"] is True
        assert record["min_lumen_rear_skin_mm"] >= 0.80 - 1.0e-9
        assert record["min_outer_cover_rear_clearance_mm"] >= -1.0e-9
        stations = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(np.diff(
                points[:, :2], axis=0), axis=1))))
        point_guard = route._no_floor_burial_guard_stations(
            stations, points[:, :2], outer_radius)
        assert abs(
            point_guard["guard_end_station_mm"]
            - record["guard_end_station_mm"]
        ) <= 0.21
        guarded = (
            stations
            <= record["flat_face_guard_end_station_mm"] + 1.0e-9)
        assert np.min(
            points[guarded, 2] - inner_radius - route.PAD_FACE_Z
        ) >= 0.80 - 1.0e-9
        assert np.min(
            points[guarded, 2] - outer_radius - route.PAD_FACE_Z
        ) >= -1.0e-9
    main_at_insert_y = float(np.interp(70.0, main_route[:, 1], main_route[:, 0]))
    t_at_insert_y = float(np.interp(70.0, t_route[:, 1], t_route[:, 0]))
    assert main_at_insert_y < 20.0
    assert t_at_insert_y > -20.0
    assert main_at_insert_y <= 10.9
    assert t_at_insert_y >= -10.2

    # Protect the *recessed* LM front surface, not only the outer z=18.3
    # plane.  The production ducts are circumscribed octagons, so use their
    # true corner radii rather than the nominal circular cable diameters.
    def recess_surface_wall(points, nominal_radius):
        points = np.asarray(points, dtype=float)
        radial = np.linalg.norm(
            points[:, :2] - np.asarray(route.L22_CUTOUT[:2]), axis=1)
        center_distance = np.hypot(
            np.maximum(radial - flush.LM_RECESS_R, 0.0),
            np.maximum(flush.LM_SEAT_Z - points[:, 2], 0.0),
        )
        cutter_corner_radius = (
            nominal_radius / math.cos(math.pi / route.TUBE_SECTION_SIDES))
        return float(np.min(center_distance) - cutter_corner_radius)

    main_recess_wall = recess_surface_wall(main_route, route.CUTTER_R)
    t_recess_wall = recess_surface_wall(t_route, route.TS_CUTTER_R)
    assert main_recess_wall >= route.TUNNEL_ROOF_SKIN - 0.02, (
        f"UM duct retains only {main_recess_wall:.3f} mm at LM recess")
    assert t_recess_wall >= route.TUNNEL_ROOF_SKIN - 0.02, (
        f"T duct retains only {t_recess_wall:.3f} mm at LM recess")
    assert np.allclose(
        facts["no_floor_lm_entry_xy_mm"],
        cables.OBIWAN_NO_FLOOR_LM_ENTRY_XY, atol=1e-12)
    assert facts["printed_lm_tunnel_count"] == 1
    assert facts["lm_lead_mode"] == (
        "d20_cluster_entry_buried_d9_tunnel_then_r14_rear_handoff")
    assert facts["lm_internal_diameter_mm"] == 9.0
    assert facts["lm_internal_center_z_mm"] == 13.0
    assert facts["lm_internal_front_skin_mm"] == 0.80
    assert math.isclose(
        facts["lm_internal_rear_skin_mm"], 3.20, abs_tol=1e-12)
    assert facts["lm_internal_to_um_lumen_wall_mm"] >= 0.80
    assert facts["lm_internal_to_t_lumen_wall_mm"] >= 0.80
    assert facts["lm_internal_lumen_wall_measurement"] == (
        "sampled_3d_centerline_surface")
    assert facts["lm_internal_lumen_wall_sample_spacing_mm"] == 0.20
    # Both outer routes clear LM in plan and retain the design 0.80-mm 3-D
    # wall after their service-patch prefixes move onto fully buried layers.
    # Keep both diagnostics explicit so a future edit cannot restore the
    # rejected exterior detours merely to improve a plan projection.
    assert facts["lm_internal_to_um_plan_lumen_wall_mm"] >= 0.40
    assert facts["lm_internal_to_t_plan_lumen_wall_mm"] >= 0.20
    assert facts["lm_internal_to_um_lumen_wall_mm"] >= 0.85
    assert facts["lm_internal_to_t_lumen_wall_mm"] >= 1.20
    # Keep an exact ruled-sweep gate beside the sampled centerline oracle.
    # Coarse production section phase can consume several tenths at a bend;
    # closest-point coordinates make any future correction local and
    # auditable instead of encouraging a blind global tolerance reduction.
    lm_exact = route._round_tube(
        route.lm_internal_duct_points(0.20), route.LM_INTERNAL_DUCT_R)
    for label, points, radius in (
            ("UM", main_route, route.CUTTER_R),
            ("T", t_route, route.TS_CUTTER_R)):
        neighbour_exact = route._round_tube(points, radius)
        exact_wall, lm_point, neighbour_point = (
            lm_exact.distance_to_with_closest_points(neighbour_exact))
        assert exact_wall >= 0.79, (
            f"LM/{label} exact swept-lumen wall {exact_wall:.3f} mm; "
            f"LM={tuple(round(value, 3) for value in lm_point)} "
            f"{label}={tuple(round(value, 3) for value in neighbour_point)}")
    lm_internal_points = route.lm_internal_duct_points(0.20)
    lm_internal_radius = _min_three_point_radius(lm_internal_points)
    assert lm_internal_radius >= 14.0, (
        f"LM internal duct bend radius {lm_internal_radius:.3f} mm")
    assert _max_turn_deg(lm_internal_points) <= 2.0
    assert math.isclose(
        facts["lm_visible_ring_radius_mm"], core.LM_VISIBLE_RING_R,
        abs_tol=1e-12)
    assert math.isclose(
        facts["lm_route_owner_clearance_mm"], 0.05,
        abs_tol=1e-12)
    assert math.isclose(
        facts["main_lm_lumen_outer_radius_mm"], core.LM_CORE_R - 0.05,
        abs_tol=1e-12)
    assert math.isclose(
        facts["t_lm_lumen_outer_radius_mm"], core.LM_CORE_R - 0.05,
        abs_tol=1e-12)
    assert math.isclose(
        facts["main_lm_cover_outer_radius_mm"],
        core.LM_VISIBLE_RING_R - 0.05,
        abs_tol=1e-12)
    assert math.isclose(
        facts["t_lm_cover_outer_radius_mm"],
        core.LM_VISIBLE_RING_R - 0.05,
        abs_tol=1e-12)
    assert math.isclose(
        facts["lm_ring_min_exterior_skin_mm"], 0.85, abs_tol=1e-12)
    assert math.isclose(facts["lm_ring_route_groove_mm"], 0.0, abs_tol=1e-12)
    assert math.isclose(
        facts["burial_web_owner_inset_mm"], 0.05, abs_tol=1e-12)
    assert facts["lm_rear_exit_kind"] == (
        "continuous_r14_d9_rear_face_handoff")
    assert math.isclose(
        facts["lm_rear_exit_bend_radius_mm"], 14.0, abs_tol=1e-12)
    assert facts["lm_rear_exit_min_qualified_radius_mm"] >= 13.9
    assert np.allclose(
        facts["lm_rear_port_xy_mm"],
        (
            cables.LM_DUCT_OUT_X_MM,
            route.NO_FLOOR_LM_DUCT_OUT_Y_MM,
        ),
        atol=1e-12,
    )
    assert math.isclose(
        facts["lm_rear_port_clearance_from_aperture_mm"],
        17.8,
        abs_tol=1e-12,
    )
    assert math.isclose(
        facts["lm_rear_port_diameter_mm"], 9.0, abs_tol=1e-12)
    assert math.isclose(
        facts["lm_rear_port_rear_z_mm"], route.PAD_FACE_Z,
        abs_tol=1e-12)
    assert math.isclose(
        facts["lm_rear_port_inner_z_mm"],
        route.LM_INTERNAL_CENTER_Z_MM,
        abs_tol=1e-12)
    assert facts["lm_external_cable_follows_handoff_tangent"]
    assert math.isclose(
        facts["lm_rear_handoff_plan_bearing_deg"], 120.0, abs_tol=1e-12)
    expected_lm_plan_tangent = np.asarray((
        math.cos(math.radians(120.0)),
        math.sin(math.radians(120.0)),
        0.0,
    ))
    assert np.allclose(
        facts["lm_rear_handoff_plan_tangent_xyz"],
        expected_lm_plan_tangent, atol=1e-12)
    assert np.allclose(
        facts["lm_rear_face_mouth_xyz_mm"],
        (cables.LM_DUCT_OUT_X_MM, route.NO_FLOOR_LM_DUCT_OUT_Y_MM,
         route.PAD_FACE_Z), atol=1e-12)
    assert facts["lm_rear_face_angle_deg_from_normal"] < 27.0
    lm_outer = route.lm_cable_points(0.5)[0]
    assert np.allclose(
        lm_outer,
        (cables.LM_DUCT_OUT_X_MM, route.NO_FLOOR_LM_DUCT_OUT_Y_MM,
         route.PAD_FACE_Z), atol=1e-12)
    assert math.isclose(
        lm_outer[1], route.L22_CUTOUT[1] - route.L22_CUTOUT[2] / 2.0
        - route.NO_FLOOR_LM_DUCT_OUT_CLEARANCE_FROM_APERTURE_MM,
        abs_tol=1e-12)
    lm_free = route.lm_cable_points(0.5)
    free_direction = lm_free[-1] - lm_free[0]
    free_direction /= np.linalg.norm(free_direction)
    assert np.allclose(
        free_direction, facts["lm_rear_face_tangent_xyz"], atol=1e-12)
    assert np.all(np.diff(lm_free[:, 1]) > 0.0)
    assert np.all(np.diff(lm_free[:, 0]) < 0.0)
    assert np.all(np.diff(lm_free[:, 2]) < 0.0)
    assert math.isclose(
        facts["lm_external_cable_end_z_mm"],
        route.LM_EXTERNAL_LEAD_END_Z,
        abs_tol=1e-12)
    # The exact D9 port must stay clear of every captive LM cavity and its
    # qualified backing land, while eliminating the former oblique ring bite.
    lm_port = route.lm_rear_exit_port_cutter()
    port_plan_radius = math.dist(
        route.LM_REAR_PORT_XY, route.L22_CUTOUT[:2])
    assert (
        port_plan_radius - route.LM_REAR_PORT_R
        > route.L22_CUTOUT[2] / 2.0 + 1.0), (
        "D9 LM rear port reaches the acoustic aperture")
    # The 17.8-mm no-floor datum protects both the planar front and the flange
    # recess.  First retain the legacy exact-BREP outer-front witness, then
    # independently screen the complete R14 centerline against the recessed
    # face using the production octagon's true corner radius.
    from build123d import Cylinder, Pos
    front_skin = Pos(
        route.L22_CUTOUT[0],
        route.L22_CUTOUT[1],
        core.THICKNESS_MM - 0.40,
    ) * Cylinder(core.LM_CORE_R, 0.80)
    front_skin -= Pos(
        route.L22_CUTOUT[0],
        route.L22_CUTOUT[1],
        core.THICKNESS_MM - 0.50,
    ) * Cylinder(flush.LM_RECESS_R, 1.00)
    front_breach = lm_port & front_skin
    front_breach_volume = (
        0.0 if front_breach is None else front_breach.volume)
    assert front_breach_volume < 0.01, (
        "D9/R14 no-floor outlet breaches the retained 0.8-mm front skin "
        f"by {front_breach_volume:.6f} mm3")
    assert recess_surface_wall(
        route.lm_rear_handoff_points(0.05),
        route.LM_REAR_PORT_R,
    ) >= route.LM_INTERNAL_FRONT_SKIN_MM - 0.02
    for site in lm_magnets:
        tools = wall_cavity_tools(
            name=site["name"], face=site["face"],
            outward=(*site["normal"], 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP)
        keepouts = (
            ("qualified land", tools.required_land),
            ("nominal magnet", tools.nominal_magnet),
            *((f"cavity cutter {index}", cutter)
              for index, cutter in enumerate(tools.cutters)),
        )
        for label, keepout in keepouts:
            overlap = lm_port & keepout
            overlap_volume = 0.0 if overlap is None else overlap.volume
            assert overlap_volume < 0.01, (
                f"LM rear port intersects {site['name']} {label} "
                f"by {overlap_volume:.6f} mm3")
    assert facts["anchor_legs"] == {}
    assert 12.14 <= facts["main_max_rear_protrusion_mm"] <= 12.25
    assert 9.90 <= facts["ts_max_rear_protrusion_mm"] <= 10.00
    assert facts["um_terminal_reference_opening_radius_mm"] == 41.0
    assert facts["um_printed_owner"] == "lm_only"
    assert not facts["um_carrier_main_duct"]
    assert facts["um_terminal_lead_mode"] == (
        "free_from_lm_r113_mouth_behind_um")
    assert facts["um_telescoping_handoff_count"] == 0
    assert facts["t_handoff_mode"] == "lm_um_core_then_free_behind_tweeter"
    assert facts["t_lower_lm_flush_radius_mm"] == core.LM_CORE_R
    assert math.isclose(
        facts["t_lower_um_flush_radius_mm"], core.UM_CORE_R,
        abs_tol=1e-12)
    assert math.isclose(
        facts["t_upper_um_flush_radius_mm"], core.UM_CORE_R,
        abs_tol=1e-12)
    assert math.isclose(
        facts["t_crescent_clear_radius_mm"], 51.9, abs_tol=1e-12)
    assert not facts["t_tweeter_printed_duct"]
    assert math.isclose(facts["t_free_cable_z_mm"], 3.8, abs_tol=1e-12)
    assert math.isclose(
        facts["t_free_cable_rear_clearance_mm"], 0.4, abs_tol=1e-12)
    assert facts["t_telescoping_handoff_count"] == 0

    paths = {
        "UM": route.route_cable_points(0.20),
        "LM": route.lm_rear_handoff_points(0.20),
        "T": route.ts_cable_points(0.20),
    }
    for name, points in paths.items():
        radius, radius_index, radius_point = (
            _min_three_point_radius_record(points))
        required_radius = (
            route.LM_EXIT_MIN_BEND_R_MM if name == "LM" else 14.0)
        assert radius >= required_radius, (
            f"{name} minimum bend radius {radius:.3f} at "
            f"index={radius_index}, xyz={radius_point}")
        assert _max_turn_deg(points) <= 2.0, f"{name} has a non-G1 kink"

    mouth = np.asarray(paths["UM"][-1], dtype=float)
    mouth_angle = math.degrees(math.atan2(
        mouth[1] - core.UM_CUTOUT[1],
        mouth[0] - core.UM_CUTOUT[0])) % 360.0
    assert math.isclose(mouth_angle, 283.0, abs_tol=1e-5)
    assert math.isclose(mouth[2], 2.70, abs_tol=1e-8)
    mouth_tangent = np.asarray(paths["UM"][-1]) - np.asarray(paths["UM"][-2])
    mouth_tangent /= np.linalg.norm(mouth_tangent)
    a = math.radians(283.0)
    expected_tangent = np.asarray((*route.UM_MOUTH_TANGENT, 0.0))
    assert float(np.dot(mouth_tangent, expected_tangent)) > 0.999

    # The complete D9.8 functional ears clear the T lumen. The tight +X UM
    # station overlaps the route in plan, so only the exact 3-D cutter-to-ear
    # distance can qualify it; a sampled plan-height deduction previously
    # and incorrectly treated the ear as tunneled.
    t_line = LineString(np.asarray(paths["T"])[:, :2])
    for x in core.JOINT_EAR_X:
        lm_ear_plan = core._complete_joint_ear_plan(
            "lm", x, core.JOINT_RECEIVER_RADIAL_CLEAR)
        wall = t_line.distance(lm_ear_plan) - route.TS_CUTTER_R
        assert wall >= route.TUNNEL_SKIN - 0.02, (
            f"T void to LM standalone-ear wall {wall:.3f} mm")

    t_inner_cutter = route._round_tube(
        route._owner_cutter_points(
            route.ts_cable_points(1.8), "um"),
        route.TS_CUTTER_R)
    um_ear_clearances = {}
    for x in core.JOINT_EAR_X:
        complete_um_ear = core._complete_joint_ear("um", x)
        overlap = _intersection_volume(t_inner_cutter, complete_um_ear)
        assert overlap < 0.01, (
            f"T inner cutter intersects complete UM ear x={x:g} by "
            f"{overlap:.4f} mm3")
        um_ear_clearances[x] = t_inner_cutter.distance_to(complete_um_ear)
        assert um_ear_clearances[x] >= 0.44, (
            f"T inner-cutter clearance to complete UM ear x={x:g} is "
            f"only {um_ear_clearances[x]:.3f} mm")
    assert min(um_ear_clearances.values()) >= 0.44

    # Analytic plan-intent erosion/normal distance catches horizontal route
    # escape early. The final manufactured-BREP normal wall is independently
    # bracketed in every state/owner shell test below.
    plan = route.route_plan_containment_facts()
    assert set(plan) == {"UM", "T"}
    for name, record in plan.items():
        assert record["contained"], (
            f"{name} failed exact eroded material-outline containment: "
            f"{record}")
        assert record["min_normal_wall_mm"] >= route.TUNNEL_SKIN - 0.04

    print(
        "  Obi-Wan route: rotated crown, zero windows, "
        f"{facts['crossover_angle_deg']:.2f} deg crossover, "
        "free-UM/printed-T crossover gap "
        f"{facts['crossover_free_um_to_t_cover_gap_mm']:.2f} mm; "
        f"normal wall {facts['min_plan_normal_wall_mm']:.2f} mm")


def _insert_bump_clearance(stand_foot):
    _state(stand_foot)
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.route as route

    main_s = np.linspace(0.0, route.ROUTE_LENGTH,
                         len(route.route_cable_points(0.10)))
    main = np.asarray(route.route_cable_points(0.10))
    ts_s = np.linspace(0.0, route.TS_ROUTE_LENGTH,
                       len(route.ts_cable_points(0.10)))
    ts = np.asarray(route.ts_cable_points(0.10))

    def clearance(points, stations, jump, centers, solid_r, floor_z, outer_r):
        name, center_s, _low, _half = jump
        i = int(np.argmin(np.abs(stations - center_s)))
        center = min(centers, key=lambda xy: math.dist(xy, points[i, :2]))
        dxy = np.linalg.norm(points[:, :2] - np.asarray(center), axis=1)
        distance = np.hypot(
            np.maximum(dxy - solid_r, 0.0),
            np.maximum(floor_z - points[:, 2], 0.0),
        )
        margin = float(distance.min() - outer_r)
        assert margin >= route.INSERT_COVER_CLEAR - 0.03, (
            f"{name} outer cover clearance {margin:.3f}")

    for bump in route.MAIN_COVERED_BUMPS:
        clearance(main, main_s,
                  (bump.name, bump.station, bump.low_z, bump.half_length),
                  flush.LM_PILOT_XY,
                  flush.PAD_D_MM / 2.0, flush.PAD_FACE_Z,
                  route.MAIN_OUTER_R)
    for bump in route.T_COVERED_BUMPS:
        record = (bump.name, bump.station, bump.low_z, bump.half_length)
        if bump.name.startswith("lm_"):
            clearance(ts, ts_s, record, flush.LM_PILOT_XY,
                      flush.PAD_D_MM / 2.0, flush.PAD_FACE_Z,
                      route.TS_OUTER_R)
        else:
            clearance(ts, ts_s, record, flush.UM_PILOT_XY,
                      4.6 / 2.0, route.UM_PILOT_FLOOR_Z,
                      route.TS_OUTER_R)
    state = "floor" if stand_foot else "no-floor"
    print(f"  {state}: every LM pad and UM 328/58 bore clears its cover")






def test_floor_route_smoothness():
    _state(True)
    import lx521_baffle.obiwan.route as route

    facts = route.route_facts()
    assert facts["open_bore_jump_count"] == 0
    assert facts["solid_backfill_count"] == 8
    assert facts["solid_backfill_floor_hardware_exceptions"] == ()
    assert facts["lm_burial_web_floor_hardware_clear_d_mm"] is None
    assert facts["solid_backfill_added_rear_depth_mm"] == 0.0
    assert facts["functional_lm_feed_mode"] == (
        "integrated_stem_rear_face_shallow_rise")
    assert facts["central_owner_feed_state"] == "floor"
    assert np.allclose(
        facts["functional_lm_feed_points"],
        ((*route.FLOOR_MAIN_FEED_XY, route.PAD_FACE_Z),
         (*route.FLOOR_T_FEED_XY, route.PAD_FACE_Z)), atol=1e-9)
    assert np.allclose(
        facts["central_owner_feed_xy"],
        (route.FLOOR_MAIN_FEED_XY, route.FLOOR_T_FEED_XY), atol=1e-12)
    assert (facts["crossover_nominal_void_gap_mm"]
            >= route.CROSSOVER_MIN_CLEARANCE)
    assert facts["crossover_free_um_to_t_cover_gap_mm"] >= 0.25
    for name, points in (
            ("UM", route.route_cable_points(0.20)),
            ("LM", route.lm_rear_handoff_points(0.20)),
            ("T", route.ts_cable_points(0.20))):
        radius = _min_three_point_radius(points)
        required_radius = (
            route.LM_EXIT_MIN_BEND_R_MM if name == "LM" else 14.0)
        assert radius >= required_radius, (
            f"floor {name} bend radius {radius:.3f}")
        assert _max_turn_deg(points) <= 2.0
    assert math.isclose(
        facts["main_min_z_mm"],
        min(bump.low_z for bump in route.MAIN_COVERED_BUMPS),
        abs_tol=0.02)
    assert math.isclose(
        facts["ts_min_z_mm"],
        min(bump.low_z for bump in route.T_COVERED_BUMPS),
        abs_tol=0.02)
    assert facts["main_max_rear_protrusion_mm"] > 0.0
    assert facts["ts_max_rear_protrusion_mm"] > 0.0
    print(
        "  floor-state central stem feeds and all covered bumps retain "
        "complete-route R14/G1")


def test_no_floor_route_smoothness():
    """Keep the minimal carrier's Z bypasses independently R14/G1-safe."""
    _state(False)
    import lx521_baffle.obiwan.route as route

    facts = route.route_facts()
    assert facts["open_bore_jump_count"] == 0
    assert facts["solid_backfill_count"] == 8
    assert (facts["crossover_nominal_void_gap_mm"]
            >= route.CROSSOVER_MIN_CLEARANCE)
    assert facts["crossover_free_um_to_t_cover_gap_mm"] >= 0.25
    for name, points in (
            ("UM", route.route_cable_points(0.20)),
            ("LM", route.lm_rear_handoff_points(0.20)),
            ("T", route.ts_cable_points(0.20))):
        radius = _min_three_point_radius(points)
        required_radius = (
            route.LM_EXIT_MIN_BEND_R_MM if name == "LM" else 14.0)
        assert radius >= required_radius, (
            f"no-floor {name} bend radius {radius:.3f}")
        assert _max_turn_deg(points) <= 2.0
    print("  no-floor covered bumps retain complete-route R14/G1")


def _bump_brep_clearance(stand_foot):
    _state(stand_foot)
    from build123d import Compound, Cylinder, Pos
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    main_outer = route._round_tube(
        route.route_cable_points(1.2), route.MAIN_OUTER_R)
    t_outer = route._round_tube(
        route.ts_cable_points(1.2), route.TS_OUTER_R)

    pad_keepouts = Compound(children=[
        Pos(x, y, (flush.PAD_FACE_Z + core.CORE_REAR_Z) / 2.0)
        * Cylinder(flush.PAD_D_MM / 2.0,
                   core.CORE_REAR_Z - flush.PAD_FACE_Z)
        for x, y in flush.LM_PILOT_XY
    ])
    um_insert_keepouts = Compound(children=[
        Pos(x, y, (flush.UM_SEAT_Z - 2.0))
        * Cylinder(2.3, 4.0)
        for x, y in flush.UM_PILOT_XY
    ])
    for cover, keepout, label in (
            (main_outer, pad_keepouts, "UM route / LM pads"),
            (t_outer, pad_keepouts, "T route / LM pads"),
            (t_outer, um_insert_keepouts, "T route / UM inserts")):
        clearance = cover.distance_to(keepout)
        general_minimum = route.INSERT_COVER_CLEAR - 0.03
        accepted_baseline = ACCEPTED_BREP_CLEARANCE_BASELINES_MM.get(
            (stand_foot, label))
        required = _required_brep_clearance_mm(
            stand_foot, label, general_minimum)
        baseline_note = (
            ""
            if accepted_baseline is None
            else f" (accepted baseline {accepted_baseline:.3f} mm)"
        )
        print(f"    {label}: {clearance:.3f} mm{baseline_note}")
        assert clearance >= required, (
            f"{label} BREP clearance {clearance:.3f} mm below "
            f"{required:.3f} mm{baseline_note}")

    state = "floor" if stand_foot else "no-floor"
    print(f"  {state}: exact outer-cover BREPs clear all driver inserts")






def _final_bump_backfill_contract(stand_foot):
    """Every intended roof-to-bore fill survives in the final carriers."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Compound, Cylinder, Pos, import_brep
    import lx521_baffle.obiwan.route as route

    owners = {
        "lm": import_brep(staged["lm"]),
        "um": import_brep(staged["um"]),
    }
    specs = route.bump_backfill_specs()
    assert len(specs) == 8
    # The lowest 0.55 mm of each saddle deliberately overlaps its printed
    # duct roof so the two fuse robustly.  The final carrier then receives
    # the exact UM/T clearance cutter through that overlap.  Remove only that
    # authoritative clearance BREP from the expected-solid contract; a broad
    # tolerance here could hide the very roof-to-bore hollow this test guards.
    duct_clearance_voids = {
        "UM": route._round_tube(
            route._owner_cutter_points(
                route.route_cable_points(1.8), "lm"),
            route.CUTTER_R),
        "T": route._round_tube(
            route._owner_cutter_points(
                route.ts_cable_points(1.8), "lm"),
            route.TS_CUTTER_R),
    }
    for spec in specs:
        components = tuple(route._bump_backfill(spec))
        assert components
        contract = Compound(children=list(components))
        # The local fill begins in the tube roof and ends exactly at the
        # blind-bore floor; it cannot be a detached decorative cap.
        assert math.isclose(
            spec.route_xyz[2] + spec.route_outer_radius - spec.bottom_z,
            route.BUMP_BACKFILL_TUBE_OVERLAP, abs_tol=1e-9)
        assert spec.top_z > spec.bottom_z
        assert (contract.bounding_box().min.Z
                > spec.route_xyz[2] - spec.route_outer_radius)
        expected_solid = contract - duct_clearance_voids[spec.route_name]
        assert expected_solid is not None and expected_solid.volume > 0.1
        missing = expected_solid - owners[spec.owner]
        missing_volume = 0.0 if missing is None else sum(
            solid.volume for solid in missing.solids())
        assert missing_volume < 0.05, (
            f"{spec.name} final {spec.owner} carrier lost "
            f"{missing_volume:.3f} mm3 of solid backfill")
        assert _intersection_volume(
            owners[spec.owner], expected_solid
        ) > 0.98 * expected_solid.volume

        # Independent cross-section probes span the route-to-pilot saddle;
        # they do not reuse the convex-hull BREP builder.
        route_xy = np.asarray(spec.route_xyz[:2], dtype=float)
        pilot_xy = np.asarray(spec.pilot_xy, dtype=float)
        for fraction in (0.15, 0.50, 0.85):
            xy = route_xy + fraction * (pilot_xy - route_xy)
            z0 = spec.bottom_z + 0.10
            z1 = spec.top_z - 0.10
            probe = Pos(float(xy[0]), float(xy[1]), (z0 + z1) / 2.0) * (
                Cylinder(0.18, z1 - z0))
            hit = _intersection_volume(owners[spec.owner], probe)
            assert hit > 0.97 * probe.volume, (
                f"{spec.name} hollow at saddle fraction {fraction:.2f}: "
                f"{hit:.4f}/{probe.volume:.4f} mm3")

    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state}: all eight final-BREP duct bumps are solid-backed to "
        "their blind-bore floors; only exact duct-clearance envelopes "
        "remain")






def _final_lm_burial_web_contract(stand_foot):
    """All six LM bumps retain closed full-width longitudinal burial."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Cylinder, Pos, import_brep
    from lx521_baffle.magnets import wall_cavity_tools
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    lm = import_brep(staged["lm"])
    main = np.asarray(route.route_cable_points(0.20), dtype=float)
    ts = np.asarray(route.ts_cable_points(0.20), dtype=float)
    route_records = {
        "UM": (
            main, np.linspace(0.0, route.ROUTE_LENGTH, len(main)),
            route.MAIN_COVERED_BUMPS,
            route.MAIN_OUTER_R, route.CUTTER_R, False),
        "T": (
            ts, np.linspace(0.0, route.TS_ROUTE_LENGTH, len(ts)),
            tuple(bump for bump in route.T_COVERED_BUMPS
                  if bump.name.startswith("lm_pilot_")),
            route.TS_OUTER_R, route.TS_CUTTER_R, True),
    }
    # The centered bridge feed now reaches the 300-degree insert bypass only
    # after a long shallow rise, so both of that bypass's longitudinal ends
    # are real buried-web transitions. The independent feed-lumen probe below
    # proves that station zero itself remains an open rear-facing mouth.
    pilot_by_name = {
        f"lm_pilot_{int(angle)}": np.asarray(xy, dtype=float)
        for angle, xy in zip(
            (0.0, 60.0, 120.0, 180.0, 240.0, 300.0),
            flush.LM_PILOT_XY)
    }
    probe_radius = 0.14
    functional_voids = []
    for pilot_xy in pilot_by_name.values():
        x, y = map(float, pilot_xy)
        bore_z0 = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM
        functional_voids.append(
            Pos(
                x, y, (bore_z0 + flush.LM_SEAT_Z + 0.15) / 2.0
            ) * Cylinder(
                flush.L22_PILOT_D_MM / 2.0,
                flush.LM_SEAT_Z + 0.15 - bore_z0))
    # The near-exterior UM and T covers deliberately pass below the two upper
    # LM magnet sites in plan.  Their Z bumps keep the cable lumen and round
    # cover clear, while the full-height burial web is subsequently recut by
    # the sealed coupon-style loading chimney and 45-degree roof.  Exempt the
    # exact production cutters—not a bounding box—from the shoulder witness;
    # otherwise this test reports the intended captive cavity as a conduit
    # pinhole after the routes are moved radially out to the visible ring.
    for site in core.side_magnet_sites("lm"):
        tools = wall_cavity_tools(
            name=site["name"],
            face=site["face"],
            outward=(*site["normal"], 0.0),
            owner="carrier",
            axis_z=site["z_mm"],
            print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP,
        )
        functional_voids.extend(tools.cutters)

    tested = {}
    tested_ends = {}
    boundary_probe_count = 0
    for route_name, (
            points, stations, bumps, outer_radius, cutter_radius,
            omit_crossover) in route_records.items():
        duct_void = route._round_tube(
            route._extended_points(points), cutter_radius)

        def assert_shoulder(station, label):
            nonlocal boundary_probe_count
            xyz = np.asarray([
                np.interp(station, stations, points[:, axis])
                for axis in range(3)
            ])
            index = int(np.searchsorted(stations, station))
            index = min(max(index, 2), len(points) - 3)
            tangent = points[index + 2, :2] - points[index - 2, :2]
            tangent /= np.linalg.norm(tangent)
            normal = np.asarray((-tangent[1], tangent[0]))
            count = 0
            for normal_fraction in (-0.90, -0.70, 0.70, 0.90):
                xy = xyz[:2] + normal_fraction * outer_radius * normal
                radial = float(np.linalg.norm(
                    xy - np.asarray(route.L22_CUTOUT[:2], dtype=float)))
                # The burial web is owner-cropped to the LM support annulus.
                # Do not ask a finite-radius witness to prove material beyond
                # that outline (notably beside the two intentional feeds).
                # Eroding by the probe radius keeps every accepted cylinder
                # wholly inside the independently stated support domain.
                if not (
                    route.L22_CUTOUT[2] / 2.0 + probe_radius
                    <= radial <= 113.0 - probe_radius
                ):
                    continue
                z0 = (
                    xyz[2] - route.BURIAL_WEB_TUBE_OVERLAP + 0.12)
                z1 = (
                    route.LM_SEAT_MEMBRANE_BOTTOM_Z
                    + route.TUNNEL_FUSE_OVERLAP - 0.12)
                assert z1 > z0 + 0.20
                probe = Pos(
                    float(xy[0]), float(xy[1]), (z0 + z1) / 2.0
                ) * Cylinder(probe_radius, z1 - z0)
                # The requested shoulder is solid over the complete web
                # height except for the exact cable lumen, final pilot insert
                # and sealed captive-magnet loading cavity. Subtract those
                # functional voids;
                # never move the probe above the shoulder defect.
                expected = probe - duct_void
                for functional_void in functional_voids:
                    expected = expected - functional_void
                expected_volume = sum(
                    solid.volume for solid in expected.solids())
                if expected_volume <= 0.01:
                    continue
                missing = expected - lm
                missing_volume = sum(
                    solid.volume for solid in missing.solids())
                assert missing_volume < max(0.012, 0.04 * expected_volume), (
                    f"{label} {route_name} shoulder open at station "
                    f"{station:.2f}, normal {normal_fraction:+.2f}: "
                    f"missing {missing_volume:.4f}/"
                    f"{expected_volume:.4f} mm3")
                count += 1
                boundary_probe_count += 1
            assert count >= 2, f"{label} retained only {count} shoulder probes"
            return count

        for bump in bumps:
            pilot_xy = pilot_by_name[bump.name]
            keepout_half = (
                outer_radius + flush.PAD_D_MM / 2.0
                + route.INSERT_COVER_CLEAR)
            count = 0
            # These stations lie beyond the old station-local saddle but
            # inside the old omega-leg keepout. They directly target both
            # longitudinal mouths visible in the reported rear view.
            ends = set()
            for direction in (-1, 1):
                direction_count = 0
                for fraction in (0.70, 0.88):
                    station = (
                        bump.station + direction * fraction * keepout_half)
                    # Never clamp an absent end into the route. In particular,
                    # lm_pilot_300 owns station zero, which is the intentional
                    # main-feed mouth rather than a failed burial-web end.
                    if station <= stations[0] or station >= stations[-1]:
                        continue
                    direction_count += assert_shoulder(
                        station,
                        f"{bump.name} transition {direction:+d}")
                if direction_count:
                    ends.add(direction)
                    count += direction_count
            assert ends, f"{bump.name} has no applicable longitudinal end"
            expected_ends = {
                direction for direction in (-1, 1)
                if (stations[0]
                    < bump.station + direction * 0.70 * keepout_half
                    < stations[-1])
            }
            assert ends == expected_ends, (
                f"{bump.name} ends {ends} != applicable {expected_ends}")
            assert count >= 4 * len(ends), (
                f"{bump.name} had only {count} shoulder probes")
            tested[bump.name] = count
            tested_ends[bump.name] = ends

            bore_floor_z = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM
            floor = Pos(
                float(pilot_xy[0]), float(pilot_xy[1]),
                (flush.PAD_FACE_Z + 0.12 + bore_floor_z - 0.12) / 2.0
            ) * Cylinder(
                0.55, bore_floor_z - flush.PAD_FACE_Z - 0.24)
            retained = _intersection_volume(lm, floor)
            assert retained > 0.96 * floor.volume, (
                f"{bump.name} lost its blind-bore floor: "
                f"{retained:.4f}/{floor.volume:.4f} mm3")

        # Probe each actual low-run boundary as well as the bump transitions.
        # At a natural-burial transition the production web includes one
        # neighboring station; at an owner/crossover boundary it stops on the
        # last allowed station.
        web_points = np.asarray(
            route.route_cable_points(1.8) if route_name == "UM"
            else route.ts_cable_points(1.8), dtype=float)
        web_stations = np.linspace(
            0.0, stations[-1], len(web_points))
        keep, allowed = route._burial_web_masks(
            web_points, outer_radius, omit_crossover=omit_crossover,
            support_domains=("lm",))
        transitions = np.flatnonzero(keep[1:] != keep[:-1]) + 1
        for index in transitions:
            if keep[index - 1] and not keep[index]:
                # Center the finite-radius witness on a fully interior web
                # section. Natural low/high transitions have one expanded
                # neighbor; owner/crossover exits do not, so step inward an
                # additional section there.
                probe_index = index - 1 if allowed[index] else index - 2
            else:
                # Mirror the same interior rule at an entry.
                probe_index = index if allowed[index - 1] else index + 1
            probe_index = min(
                max(int(probe_index), 1), len(web_points) - 2)
            assert_shoulder(
                float(web_stations[probe_index]),
                f"computed web boundary {probe_index}")

        # No-floor's explicit rear-normal bore owns the external mouth; the
        # oblique sweep starts behind the rear skin and positively overlaps
        # that bore.  Floor retains its direct station-zero service mouth.
        facts = route.route_facts()
        feed_index = 0 if route_name == "UM" else 1
        assert np.allclose(
            points[0], facts["buried_route_start_points"][feed_index],
            atol=1e-9)
        tangent = points[1] - points[0]
        tangent /= np.linalg.norm(tangent)
        feed_probe = route._round_tube(
            np.vstack((points[0] - 0.35 * tangent,
                       points[0] + 0.35 * tangent)),
            cutter_radius - 0.20)
        assert _intersection_volume(lm, feed_probe) < 0.02, (
            f"{route_name} functional feed mouth was capped")

    assert set(tested) == {
        "lm_pilot_0", "lm_pilot_60", "lm_pilot_120",
        "lm_pilot_180", "lm_pilot_240", "lm_pilot_300",
    }
    assert facts["functional_lm_feed_count"] == 2
    assert facts["functional_lm_feed_web_omitted"]
    # The per-bump check above is authoritative: every geometrically
    # applicable longitudinal end must be present. Both carrier states now
    # use the same centered feed geometry, so both ends of the 300-degree
    # bypass remain buried in both states.
    assert tested_ends["lm_pilot_300"] == {-1, 1}
    assert boundary_probe_count > sum(len(ends) for ends in tested_ends.values())
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state}: all six LM bump runs are closed full-width at every "
        "applicable longitudinal end; station-zero feeds and only exact "
        "blind insert bores remain")






def _final_um_burial_web_contract(stand_foot):
    """Both UM T-bypass bumps retain solid longitudinal shoulders."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "T", tempfile.gettempdir(), shell_keys=())
    from build123d import (
        Cylinder, Face, Polyline, Pos, Wire, import_brep, loft)
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route
    from lx521_baffle.magnets import wall_cavity_tools

    um = import_brep(staged["um"])
    points = np.asarray(route.ts_cable_points(0.20), dtype=float)
    stations = np.linspace(0.0, route.TS_ROUTE_LENGTH, len(points))
    bumps = tuple(
        bump for bump in route.T_COVERED_BUMPS
        if bump.name.startswith("um_pilot_"))
    assert {bump.name for bump in bumps} == {
        "um_pilot_328", "um_pilot_58"}

    # The contract permits exactly the D6 cable lumen and the blind insert
    # bores.  Everything else between the rear half of the covered bump and
    # the UM seat membrane must be solid, including both longitudinal ends
    # that were formerly open between the two narrow omega anchor legs.
    duct_void = route._round_tube(
        route._owner_cutter_points(
            route.ts_cable_points(1.8), "um"),
        route.TS_CUTTER_R)
    functional_voids = []
    bore_z0 = flush.UM_SEAT_Z - flush.UM_PILOT_DEPTH_MM
    bore_z1 = flush.UM_SEAT_Z + 0.15
    for x, y in flush.UM_PILOT_XY:
        functional_voids.append(
            Pos(x, y, (bore_z0 + bore_z1) / 2.0) * Cylinder(
                flush.UM_PILOT_D_MM / 2.0, bore_z1 - bore_z0))
    # The 50.5-degree captive station lies close to the 58-degree bypass. It
    # is a legitimate service void, so subtract its exact cradle, chimney,
    # and 45-degree roof rather than an obsolete open blind cylinder.
    for site in core.side_magnet_sites("um"):
        tools = wall_cavity_tools(
            name=site["name"], face=site["face"],
            outward=(*site["normal"], 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP)
        functional_voids.extend(tools.cutters)
    # The 58-degree run reaches the lower edge of the direct tweeter
    # half-lap receiver. That open mating volume begins at z=12.30 and is
    # filled by the optional crescent's complementary half when assembled;
    # it is neither a trapped duct cavity nor permission to omit the solid
    # web below it. Subtract both exact interface families from the witness.
    for x in core.JOINT_EAR_X:
        z0, z1 = core.LM_JOINT_Z
        functional_voids.append(core._joint_ear(
            "lm", x,
            (z0 - core.JOINT_RECEIVER_RADIAL_CLEAR,
             core.UM_JOINT_Z[0]),
            core.JOINT_RECEIVER_RADIAL_CLEAR))
        functional_voids.append(core._cylinder_at(
            x, core.JOINT_EAR_Y, core.JOINT_INSERT_BORE_D / 2.0,
            *core.JOINT_INSERT_BORE_Z))
    for x in core.TWEETER_JOINT_X:
        functional_voids.append(core._plan_prism(
            core._complete_tweeter_joint_ear_plan(
                "tweeter", x, core.TWEETER_JOINT_CLEAR),
            core.TWEETER_CORE_JOINT_Z[1],
            core.TWEETER_ADDON_JOINT_Z[1]
            + core.TWEETER_JOINT_CLEAR))
        functional_voids.append(core._cylinder_at(
            x, core.TWEETER_JOINT_Y, core.TWEETER_JOINT_HOLE_D / 2.0,
            core.TWEETER_CORE_JOINT_Z[0] - 0.2,
            core.TWEETER_CORE_BORE_TOP_Z))

    probe_radius = 0.14
    tested = {}
    um_center = np.asarray(route.UM_CUTOUT[:2], dtype=float)
    owner_outer = Pos(*route.UM_CUTOUT[:2], -40.0) * Cylinder(
        core.UM_CORE_R - 0.05, 100.0)
    owner_inner = Pos(*route.UM_CUTOUT[:2], -40.0) * Cylinder(
        route.UM_CUTOUT[2] / 2.0 + 0.05, 100.0)
    owner_annulus = owner_outer - owner_inner

    def xyz_at(station):
        return np.asarray([
            np.interp(station, stations, points[:, axis])
            for axis in range(3)
        ])

    def frame_at(station):
        xyz = xyz_at(station)
        index = int(np.searchsorted(stations, station))
        index = min(max(index, 2), len(points) - 3)
        tangent = points[index + 2, :2] - points[index - 2, :2]
        tangent /= np.linalg.norm(tangent)
        return xyz, np.asarray((-tangent[1], tangent[0]))

    for bump in bumps:
        applicable_ends = set()
        probe_count = 0
        keepout_half = (
            route.TS_OUTER_R + route.UM_PAD_D_MM / 2.0
            + route.INSERT_COVER_CLEAR)
        for direction in (-1, 1):
            direction_count = 0
            for fraction in (0.70, 0.88):
                station = bump.station + direction * fraction * keepout_half
                assert stations[0] < station < stations[-1]
                xyz, normal = frame_at(station)
                for normal_fraction in (
                        -0.90, -0.70, -0.35, 0.0, 0.35, 0.70, 0.90):
                    xy = xyz[:2] + (
                        normal_fraction * route.TS_OUTER_R * normal)
                    radial = float(np.linalg.norm(xy - um_center))
                    if not (
                        route.UM_CUTOUT[2] / 2.0 + probe_radius
                        <= radial <= core.UM_CORE_R - probe_radius
                    ):
                        continue
                    z0 = (
                        xyz[2] - route.BURIAL_WEB_TUBE_OVERLAP + 0.12)
                    z1 = (
                        route.UM_SEAT_MEMBRANE_BOTTOM_Z
                        + route.TUNNEL_FUSE_OVERLAP - 0.12)
                    assert z1 > z0 + 0.20
                    probe = Pos(
                        float(xy[0]), float(xy[1]), (z0 + z1) / 2.0
                    ) * Cylinder(probe_radius, z1 - z0)
                    expected = probe - duct_void
                    for functional_void in functional_voids:
                        expected = expected - functional_void
                    expected_volume = sum(
                        solid.volume for solid in expected.solids())
                    if expected_volume <= 0.01:
                        continue
                    missing = expected - um
                    missing_volume = sum(
                        solid.volume for solid in missing.solids())
                    missing_limit = max(0.012, 0.04 * expected_volume)
                    if missing_volume >= missing_limit:
                        missing_bb = missing.bounding_box()
                        raise AssertionError(
                            f"{bump.name} UM shoulder open at station "
                            f"{station:.2f}, direction {direction:+d}, "
                            f"fraction {fraction:.2f}, normal "
                            f"{normal_fraction:+.2f}: missing "
                            f"{missing_volume:.4f}/"
                            f"{expected_volume:.4f} mm3; probe xyz="
                            f"({xy[0]:.3f},{xy[1]:.3f},"
                            f"{(z0 + z1) / 2.0:.3f}) radial={radial:.3f}; "
                            f"missing bbox=("
                            f"{missing_bb.min.X:.3f},"
                            f"{missing_bb.min.Y:.3f},"
                            f"{missing_bb.min.Z:.3f})..("
                            f"{missing_bb.max.X:.3f},"
                            f"{missing_bb.max.Y:.3f},"
                            f"{missing_bb.max.Z:.3f})")
                    direction_count += 1
                    probe_count += 1
            assert direction_count >= 4, (
                f"{bump.name} retained only {direction_count} probes on "
                f"longitudinal side {direction:+d}")
            applicable_ends.add(direction)
        assert applicable_ends == {-1, 1}
        assert probe_count >= 8

        angle = float(bump.name.rsplit("_", 1)[-1])
        pilot_xy = route._UM_PILOT_BY_ANGLE[angle]
        floor_z0 = route.UM_PILOT_FLOOR_Z + 0.12
        floor_z1 = bore_z0 - 0.12
        floor_probe = Pos(
            pilot_xy[0], pilot_xy[1], (floor_z0 + floor_z1) / 2.0
        ) * Cylinder(0.55, floor_z1 - floor_z0)
        retained_floor = _intersection_volume(um, floor_probe)
        assert retained_floor > 0.96 * floor_probe.volume, (
            f"{bump.name} lost its blind-bore floor: "
            f"{retained_floor:.4f}/{floor_probe.volume:.4f} mm3")

        # A filled shoulder must not be achieved by plugging the cable. Use
        # an independently resampled reduced-radius lumen through the full
        # checked interval and require it to remain empty in the final UM.
        local_stations = np.linspace(
            bump.station - 0.88 * keepout_half,
            bump.station + 0.88 * keepout_half, 17)
        local_points = np.vstack([xyz_at(value) for value in local_stations])
        local_lumen = (
            route._round_tube(local_points, route.TS_CUTTER_R - 0.20)
            & owner_annulus)
        assert local_lumen is not None and local_lumen.volume > 1.0
        assert _intersection_volume(um, local_lumen) < 0.02, (
            f"{bump.name} D6 lumen was plugged by the burial web")

        # Continuous independent sentinel over the two point-probed sides.
        # This loft intentionally does not call the production burial-web
        # builder, so a shared mask/section bug cannot make both pass.
        sections = []
        for station in np.linspace(
                bump.station - 0.88 * keepout_half,
                bump.station + 0.88 * keepout_half, 9):
            xyz, normal = frame_at(station)
            half_width = route.TS_OUTER_R - 0.14
            q0 = xyz[:2] - half_width * normal
            q1 = xyz[:2] + half_width * normal
            z0 = xyz[2] - route.BURIAL_WEB_TUBE_OVERLAP + 0.14
            z1 = (route.UM_SEAT_MEMBRANE_BOTTOM_Z
                  + route.TUNNEL_FUSE_OVERLAP - 0.14)
            corners = (
                (float(q0[0]), float(q0[1]), float(z0)),
                (float(q1[0]), float(q1[1]), float(z0)),
                (float(q1[0]), float(q1[1]), float(z1)),
                (float(q0[0]), float(q0[1]), float(z1)),
                (float(q0[0]), float(q0[1]), float(z0)),
            )
            sections.append(Face(Wire(Polyline(*corners).edges())))
        expected_web = loft(sections, ruled=True).clean() & owner_annulus
        expected_web -= duct_void
        for functional_void in functional_voids:
            expected_web -= functional_void
        expected_web = expected_web.clean()
        expected_volume = sum(
            solid.volume for solid in expected_web.solids())
        assert expected_volume > 1.0
        missing_web = expected_web - um
        missing_volume = sum(
            solid.volume for solid in missing_web.solids())
        retained_volume = _intersection_volume(um, expected_web)
        assert missing_volume < 0.05, (
            f"{bump.name} continuous UM burial web is missing "
            f"{missing_volume:.4f}/{expected_volume:.4f} mm3")
        assert retained_volume > 0.995 * expected_volume, (
            f"{bump.name} retained only "
            f"{retained_volume:.4f}/{expected_volume:.4f} mm3 of its "
            "continuous UM burial web")
        tested[bump.name] = probe_count

    assert set(tested) == {"um_pilot_328", "um_pilot_58"}
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state}: both UM T-bypass bumps have full-width solid "
        "longitudinal shoulders; only the D6 lumen and exact blind-bore, "
        "sealed captive-magnet and half-lap interface voids remain")






def _final_feed_and_flush_mouth_contract(stand_foot):
    """Final carriers own only buried skins up to native mouth planes."""
    _state(stand_foot)
    staging = tempfile.TemporaryDirectory(prefix="lx-obiwan-mouths-")
    staged = _stage_shell_contract_breps(
        stand_foot, "T", staging.name, shell_keys=())
    from build123d import Box, Cylinder, Pos, import_brep
    from shapely.geometry import LineString, Point, box
    from lx521_baffle.base import (
        BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM, L22_CUTOUT,
        M5_INSERT_ENTRY_D_MM, UM_CUTOUT)
    import lx521_baffle.cables as cables
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.bridge as bridge
    import lx521_baffle.obiwan.route as route

    owners = {
        "lm": import_brep(staged["lm"]),
        "um": import_brep(staged["um"]),
        "tweeter": import_brep(staged["tweeter"]),
    }
    facts = route.route_facts()
    main = np.asarray(route.route_cable_points(0.20), dtype=float)
    ts = np.asarray(route.ts_cable_points(0.20), dtype=float)

    # Cutter extension is a global ruled-loft phase input, not merely an
    # endpoint allowance. Floor retains its internal-stem backreach; the
    # explicit no-floor Z bores require zero swept-cutter backreach so T
    # cannot erode the neighbouring UM feed wall.
    phase_source = route.ts_cable_points(1.8)
    for owner in ("lm", "um"):
        phased = route._owner_cutter_points(phase_source, owner)
        expected_extension = route._owner_cutter_extension(owner)
        assert math.isclose(
            np.linalg.norm(phased[0] - phase_source[0]),
            expected_extension, abs_tol=1e-9)
        assert math.isclose(
            np.linalg.norm(phased[-1] - phase_source[-1]),
            expected_extension, abs_tol=1e-9)

    expected_mode = (
        "integrated_stem_rear_face_shallow_rise" if stand_foot
        else "bridge_rear_normal_entry_then_buried_rise")
    assert facts["functional_lm_feed_mode"] == expected_mode
    expected_main_feed = (
        route.FLOOR_MAIN_FEED_XY if stand_foot
        else route.NO_FLOOR_MAIN_FEED_XY)
    expected_t_feed = (
        route.FLOOR_T_FEED_XY if stand_foot
        else route.NO_FLOOR_T_FEED_XY)
    expected_main_z = (
        route.PAD_FACE_Z if stand_foot
        else route.NO_FLOOR_MAIN_FEED_START_Z)
    expected_t_z = (
        route.PAD_FACE_Z if stand_foot
        else route.NO_FLOOR_T_FEED_START_Z)
    assert np.allclose(
        main[0], (*expected_main_feed, expected_main_z), atol=1e-9)
    assert np.allclose(
        ts[0], (*expected_t_feed, expected_t_z), atol=1e-9)
    assert tuple(facts["central_owner_feed_rise_lengths_mm"]) == (
        24.0, 45.0 if stand_foot else 27.5)
    if stand_foot:
        assert route.CENTRAL_MAIN_FEED_START_BEARING_DEG == 65.0
        assert route.CENTRAL_T_FEED_START_BEARING_DEG == 116.0
    else:
        assert math.isclose(
            route.CENTRAL_MAIN_FEED_START_BEARING_DEG,
            route.NO_FLOOR_MAIN_ENTRY_START_BEARING_DEG, abs_tol=1e-12)
        assert math.isclose(
            route.CENTRAL_T_FEED_START_BEARING_DEG,
            route.NO_FLOOR_T_ENTRY_START_BEARING_DEG, abs_tol=1e-12)
    if stand_foot:
        import lx521_baffle.obiwan.floor as floor

        floor_facts = floor.integrated_floor_facts()["floor_lanes"]
        for name, outer_radius in (("um", route.MAIN_OUTER_R),
                                   ("t", route.TS_OUTER_R)):
            record = floor_facts[name]
            assert record["rear_mouth_relief_z_mm"] == (
                floor.FLOOR_REAR_FACE_SKIN_MM,
                route.NO_FLOOR_FEED_REAR_Z)
            assert math.isclose(
                record["rear_face_skin_mm"],
                floor.FLOOR_REAR_FACE_SKIN_MM, abs_tol=1e-12)
            assert math.isclose(
                record["rear_mouth_relief_radius_mm"],
                outer_radius
                + floor.FLOOR_FEED_MOUTH_CONTRACT_CLEARANCE_MM
                + floor.FLOOR_FEED_MOUTH_BOOLEAN_MARGIN_MM,
                abs_tol=1e-12)

    feed_specs = (
        ("UM", main, route.CUTTER_R, route.MAIN_OUTER_R,
         route.route_cable_points(1.8),
         route.LM_MAIN_CUTTER_SEGMENT_COUNT),
        ("T", ts, route.TS_CUTTER_R, route.TS_OUTER_R,
         route.ts_cable_points(1.8),
         route.LM_T_CUTTER_SEGMENT_COUNT),
    )
    front_domain = Pos(0.0, 220.0, 52.65) * Box(
        600.0, 600.0, 94.7)
    no_floor_port_by_name = {
        bore.name.upper(): bore
        for bore in route.no_floor_rear_entry_bores()}
    for (label, points, cutter_radius, outer_radius,
         production_points, segment_count) in feed_specs:
        local = points[:20]
        lumen = route._round_tube_global_segment(
            route._owner_cutter_points(production_points, "lm"),
            cutter_radius - 0.15, 0, segment_count)
        assert _intersection_volume(owners["lm"], lumen) < 0.02, (
            f"{label} central rear feed is capped")
        production_lumen = route._round_tube_global_segment(
            route._owner_cutter_points(production_points, "lm"),
            cutter_radius, 0, segment_count)
        outer = route._round_tube(local, cutter_radius + 0.55)
        inner = route._round_tube(
            route._extended_points(local, 1.0),
            cutter_radius + 0.15)
        skin = (outer - inner) & front_domain
        # The production cutter uses the authoritative coarse global loft
        # and its route-wide endpoint phase, while this witness uses a dense
        # local centerline. Deduct their exact set difference so a valid lumen
        # is never misreported as missing cover after a feed-arc adjustment.
        skin -= production_lumen
        if not stand_foot:
            port = no_floor_port_by_name[label]
            skin -= route._z_axis_bore(
                port.xy, port.radius_mm,
                port.rear_z_mm, port.inner_z_mm)
            # The D9 LM feed intentionally passes beside these annular
            # witnesses. It is a functional void, not missing UM/T cover, so
            # deduct the exact one-piece production cutter. The independent
            # route contract still enforces the wall between all three lumens.
            skin -= route.no_floor_lm_internal_cutter()
        retained = _intersection_volume(owners["lm"], skin)
        missing_skin = skin - owners["lm"]
        missing_skin_components = []
        if missing_skin is not None:
            for component in missing_skin.solids():
                bounds = component.bounding_box()
                missing_skin_components.append({
                    "volume_mm3": float(component.volume),
                    "min_xyz_mm": (
                        float(bounds.min.X), float(bounds.min.Y),
                        float(bounds.min.Z)),
                    "max_xyz_mm": (
                        float(bounds.max.X), float(bounds.max.Y),
                        float(bounds.max.Z)),
                })
        assert retained > 0.995 * skin.volume, (
            f"{label} central feed skin incomplete: "
            f"{retained:.4f}/{skin.volume:.4f} mm3; "
            f"missing={missing_skin_components}")
        rear = route._polygon_prism(
            Point(*points[0, :2]).buffer(
                outer_radius + 0.30, resolution=32),
            -20.0, route.NO_FLOOR_FEED_REAR_Z - 0.02)
        if stand_foot:
            # The integral floor deliberately retains its 0.45-mm exterior
            # rear skin over each buried service lane.  That skin is solid
            # (and therefore correctly intersects this low-Z witness), but
            # it is not a projecting route cover.  Exclude only the declared
            # face skin before proving the central feed/cover does not extend
            # behind the common baffle rear datum.
            rear -= route._polygon_prism(
                Point(*points[0, :2]).buffer(
                    outer_radius + 0.30, resolution=32),
                -20.0, floor.FLOOR_REAR_FACE_SKIN_MM + 0.01)
        assert _intersection_volume(owners["lm"], rear) < 0.02, (
            f"{label} central feed projects behind z=5.3")

    # The UM/T central lumens do not touch each other. In no-floor state all
    # three D20-fed branches also clear the four blind bridge inserts.
    feed_lumen_web = (
        np.linalg.norm(main[0] - ts[0])
        - route.CUTTER_R - route.TS_CUTTER_R)
    assert feed_lumen_web >= route.CROSSOVER_MIN_CLEARANCE
    if not stand_foot:
        # A horizontal route tangent to the rear face can pass the old lumen
        # test yet leave no usable circular entry in the printed bridge.  The
        # three explicit rear-normal bores must be fully open in the final LM
        # owner and overlap their buried route voids by positive volume.
        port_cutters = route.no_floor_rear_entry_bore_cutters()
        ports_by_name = {
            port.name.upper(): (port, cutter)
            for port, cutter in zip(
                route.no_floor_rear_entry_bores(), port_cutters,
                strict=True)
        }
        assert set(ports_by_name) == {"LM", "UM", "T"}
        for (label, points, cutter_radius, _outer_radius,
             _production_points, _segment_count) in feed_specs:
            port, port_cutter = ports_by_name[label]
            assert _intersection_volume(owners["lm"], port_cutter) < 0.02, (
                f"{label} rear-normal entry is blocked")
            rear_disk = route._z_axis_bore(
                port.xy, port.radius_mm - 0.15,
                route.NO_FLOOR_FEED_REAR_Z - 0.02,
                route.NO_FLOOR_FEED_REAR_Z + 0.04)
            assert _intersection_volume(owners["lm"], rear_disk) < 0.02, (
                f"{label} rear entry is not a full open circle")
            buried = route._round_tube(points[:20], cutter_radius)
            overlap = port_cutter & buried
            overlap_volume = (
                0.0 if overlap is None else sum(
                    solid.volume for solid in overlap.solids()))
            assert overlap_volume > 10.0, (
                f"{label} rear entry does not connect to its buried duct; "
                f"overlap={overlap_volume:.6f} mm3")

        # The user-visible service face is the four-insert rectangle plus
        # 6 mm.  It must be one planar solid sheet: only the four insert bores
        # and the three D20-packed cable mouths may interrupt it, and no route
        # cover may protrude behind z=5.3 anywhere in that patch.
        x0, y0, x1, y1 = route.NO_FLOOR_SERVICE_PATCH_BOUNDS
        patch_thickness = 0.24
        rear_patch = Pos(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            route.PAD_FACE_Z + 0.06 + patch_thickness / 2.0,
        ) * Box(x1 - x0, y1 - y0, patch_thickness)
        allowed_openings = [
            route._z_axis_bore(
                xy, M5_INSERT_ENTRY_D_MM / 2.0 + 0.02,
                route.PAD_FACE_Z - 0.10,
                route.PAD_FACE_Z + patch_thickness + 0.30)
            for xy in BRIDGE_HOLE_XY
        ]
        allowed_openings.extend(
            route._z_axis_bore(
                port.xy, port.radius_mm + 0.02,
                route.PAD_FACE_Z - 0.10,
                route.PAD_FACE_Z + patch_thickness + 0.30)
            for port in route.no_floor_rear_entry_bores()
        )
        required_patch = rear_patch
        for opening in allowed_openings:
            required_patch = required_patch - opening
        missing_patch = required_patch - owners["lm"]
        missing_patch_volume = (
            0.0 if missing_patch is None
            else sum(solid.volume for solid in missing_patch.solids()))
        assert missing_patch_volume < 0.05, (
            "no-floor rear service patch has a visible route opening: "
            f"missing={missing_patch_volume:.6f} mm3")

        behind_patch = Pos(
            (x0 + x1) / 2.0,
            (y0 + y1) / 2.0,
            route.PAD_FACE_Z - 0.16,
        ) * Box(x1 - x0, y1 - y0, 0.20)
        rear_protrusion = owners["lm"] & behind_patch
        rear_protrusion_volume = (
            0.0 if rear_protrusion is None else rear_protrusion.volume)
        assert rear_protrusion_volume < 0.02, (
            "no-floor route cover protrudes behind the planar service face: "
            f"volume={rear_protrusion_volume:.6f} mm3")

        # The front view looks through the D221.2 flange recess onto its
        # z=12.3 floor and R110.6 cylindrical wall.  Protect the lower central
        # sector containing both the LM R14 handoff and UM bridge transition;
        # no driver pilot, acoustic opening or other intentional feature lies
        # in this box.  This exact final-BREP witness catches the two windows
        # that a planar z=18.3-only check cannot see.
        witness_inset = 0.05
        crown_domain = Pos(
            0.0, (72.0 + 101.0) / 2.0, core.THICKNESS_MM / 2.0,
        ) * Box(80.0, 29.0, core.THICKNESS_MM)
        floor_z0 = (
            route.LM_SEAT_Z - route.TUNNEL_ROOF_SKIN + witness_inset)
        floor_z1 = route.LM_SEAT_Z - witness_inset
        recess_floor_skin = Pos(
            *route.L22_CUTOUT[:2], (floor_z0 + floor_z1) / 2.0,
        ) * Cylinder(
            route.LM_RECESS_R - witness_inset, floor_z1 - floor_z0)
        recess_floor_skin -= Pos(
            *route.L22_CUTOUT[:2], (floor_z0 + floor_z1) / 2.0,
        ) * Cylinder(
            route.L22_CUTOUT[2] / 2.0 + witness_inset,
            floor_z1 - floor_z0)
        wall_z0 = route.LM_SEAT_Z + witness_inset
        wall_z1 = core.THICKNESS_MM - witness_inset
        recess_wall_skin = Pos(
            *route.L22_CUTOUT[:2], (wall_z0 + wall_z1) / 2.0,
        ) * Cylinder(
            route.LM_RECESS_R + route.TUNNEL_ROOF_SKIN - witness_inset,
            wall_z1 - wall_z0)
        recess_wall_skin -= Pos(
            *route.L22_CUTOUT[:2], (wall_z0 + wall_z1) / 2.0,
        ) * Cylinder(
            route.LM_RECESS_R + witness_inset,
            wall_z1 - wall_z0)
        recess_floor_skin &= crown_domain
        recess_wall_skin &= crown_domain
        protected_recess_skin = recess_floor_skin.fuse(
            recess_wall_skin).clean()
        missing_recess_skin = protected_recess_skin - owners["lm"]
        missing_recess_components = []
        if missing_recess_skin is not None:
            for solid in missing_recess_skin.solids():
                bounds = solid.bounding_box()
                missing_recess_components.append({
                    "volume_mm3": float(solid.volume),
                    "min_xyz_mm": (
                        float(bounds.min.X), float(bounds.min.Y),
                        float(bounds.min.Z)),
                    "max_xyz_mm": (
                        float(bounds.max.X), float(bounds.max.Y),
                        float(bounds.max.Z)),
                })
        missing_recess_skin_volume = sum(
            item["volume_mm3"] for item in missing_recess_components)
        assert missing_recess_skin_volume < 0.05, (
            "LM recessed front surface has a visible LM/UM route window: "
            f"missing={missing_recess_skin_volume:.6f} mm3; "
            f"components={missing_recess_components}")

        # The rectangular bridge cover is cropped at the z=5.3 rear datum,
        # while the circular LM owner takes over farther along each route.
        # Probe the entire bridge-to-ring handoff at a depth that must remain
        # solid rear skin.  This catches a real tunnel opening even when the
        # STEP viewer does not colour the support-blocker visible through it.
        route_records = facts["no_floor_service_patch_routes"]
        for label, points, cutter_radius in (
                ("UM", main, route.CUTTER_R),
                ("T", ts, route.TS_CUTTER_R)):
            record = route_records[label.lower()]
            # UM is deliberately allowed to form a closed rear belly after
            # its complete cover clears the flat patch; only T retains the
            # former z=5.3 bridge-to-ring spine.  The exact rear-patch and
            # recess-skin witnesses above govern UM's two visible surfaces.
            if record["flat_face_guard_end_station_mm"] <= (
                    record["patch_guard_end_station_mm"] + 0.21):
                continue
            stations = np.concatenate((
                [0.0], np.cumsum(np.linalg.norm(np.diff(
                    points[:, :2], axis=0), axis=1))))
            handoff = (
                (stations >= record["patch_guard_end_station_mm"] - 1e-9)
                & (stations <= record[
                    "flat_face_guard_end_station_mm"] + 1e-9))
            handoff_xy = points[handoff, :2]
            assert len(handoff_xy) >= 3
            rear_skin_spine = LineString(handoff_xy).buffer(
                min(0.50, cutter_radius - 0.20),
                resolution=12, cap_style=2, join_style=1)
            rear_skin_probe = route._polygon_prism(
                rear_skin_spine,
                route.PAD_FACE_Z + 0.15,
                route.PAD_FACE_Z + 0.55,
            )
            missing_handoff_skin = rear_skin_probe - owners["lm"]
            missing_handoff_skin_volume = (
                0.0 if missing_handoff_skin is None
                else sum(solid.volume
                         for solid in missing_handoff_skin.solids()))
            assert missing_handoff_skin_volume < 0.05, (
                f"{label} bridge-to-ring handoff opens through the rear face: "
                f"missing={missing_handoff_skin_volume:.6f} mm3")

        lm_points = route.lm_internal_duct_points(0.20)
        assert np.allclose(
            lm_points[0, :2], cables.OBIWAN_NO_FLOOR_LM_ENTRY_XY,
            atol=1e-12)
        assert np.allclose(
            lm_points[-1], route.LM_REAR_HANDOFF_SPEC["start"],
            atol=1e-12)
        # Use the exact fused production void for the nominal D9 gate. A
        # separately lofted equal-radius reference has different octagon
        # section phase and therefore invents facet slivers that are not a
        # physical obstruction. Independently pass the estimated D7.8 cable
        # through the functional (non-overtravel) centerline.
        lm_lumen = route.no_floor_lm_internal_cutter()
        assert _intersection_volume(owners["lm"], lm_lumen) < 0.03, (
            "no-floor exact LM D9 entry/tunnel/exit cutter is blocked")
        lm_complete = route.lm_complete_duct_points(0.20)
        assert _min_three_point_radius(
            route.lm_rear_handoff_points(0.20)) >= 13.99
        assert _max_turn_deg(
            route.lm_rear_handoff_points(0.20)) <= 1.0
        lm_cable = route._round_tube(
            lm_complete, route.LM_CABLE_D_EST / 2.0)
        assert _intersection_volume(owners["lm"], lm_cable) < 0.03, (
            "no-floor estimated D7.8 LM cable cannot traverse the D9 duct")
        entry_cutter = ports_by_name["LM"][1]
        overlap = entry_cutter & lm_lumen
        overlap_volume = (
            0.0 if overlap is None else sum(
                solid.volume for solid in overlap.solids()))
        assert overlap_volume > 20.0, (
            "LM entry is disconnected from its buried tunnel; "
            f"overlap={overlap_volume:.6f} mm3")

        # Equal-radius polygon tubes can make OCC report an empty Boolean
        # intersection when their coincident side faces use different loft
        # station phases.  Prove the exit splice with a smaller physical core
        # inside the declared tangent pre-fusion interval instead.  Complete
        # containment in both one-piece solids demonstrates positive-volume
        # overlap without depending on coincident-face classification.
        exit_start = np.asarray(
            route.LM_REAR_HANDOFF_SPEC["start"], dtype=float)
        entry_tangent = np.asarray(
            route.LM_REAR_HANDOFF_SPEC["plan_tangent"], dtype=float)
        exit_overlap_core = route._round_tube(
            np.asarray((
                exit_start
                - 1.50 * route.LM_REAR_PORT_PREFUSION_MM
                / 2.0 * entry_tangent,
                exit_start
                - 0.50 * route.LM_REAR_PORT_PREFUSION_MM
                / 2.0 * entry_tangent,
            )),
            route.LM_REAR_PORT_R - 0.50,
        )
        for label, host in (
                ("buried tunnel", lm_lumen),
                ("R14 exit", route.lm_rear_exit_port_cutter())):
            missing = exit_overlap_core - host
            missing_volume = (
                0.0 if missing is None else sum(
                    solid.volume for solid in missing.solids()))
            assert missing_volume < 0.02, (
                f"LM exit overlap core is missing from {label}; "
                f"missing={missing_volume:.6f} mm3")
        # These are deliberately Z-separated internal lanes.  Measure the
        # swept nominal lumens in 3-D; their XY projections overlap by design
        # and are not a physical wall metric.
        lm_functional_lumen = route._round_tube(
            lm_points, route.LM_INTERNAL_DUCT_R)
        for label, points, radius in (
                ("UM", main, route.CUTTER_R),
                ("T", ts, route.TS_CUTTER_R)):
            neighbouring_lumen = route._round_tube(points, radius)
            exact_wall = lm_functional_lumen.distance_to(neighbouring_lumen)
            assert exact_wall >= 0.79, (
                f"LM/{label} exact swept-lumen wall {exact_wall:.3f} mm")
        for label, points, outer_radius in (
                ("UM", main, route.MAIN_OUTER_R),
                ("T", ts, route.TS_OUTER_R)):
            bridge_points = points[points[:, 1] <= 90.0]
            envelope = route._round_tube(
                bridge_points, outer_radius + 0.05)
            for index, insert in enumerate(bridge.bridge_insert_envelopes()):
                assert _intersection_volume(envelope, insert) < 0.01, (
                    f"{label} bridge feed reaches insert {index}")
        assert tuple(BRIDGE_HOLE_XY) == (
            (-20.0, 20.0), (20.0, 20.0),
            (-20.0, 70.0), (20.0, 70.0))
        assert BRIDGE_INSERT_D_MM > 0.0

    # LM now has one gradual D9/R14 port through the actual rear plane.  In
    # the no-floor state it is part of the same buried D9 bridge tunnel.
    # The nearby protected lower-ring witness is deliberately where the
    # former oblique side-relief left the reported visible crescent bite.
    # The production cutter is nominal D9.  Do not compare its equal-radius
    # faceted loft against the independently staged final part: different
    # global section phase produces harmless polygon-corner slivers.  The
    # release requirement is that the estimated D7.8 cable itself remains
    # unobstructed along the complete R14 exit.
    lm_exit_cable = route._round_tube(
        route.lm_rear_handoff_points(0.20),
        route.LM_CABLE_D_EST / 2.0,
        section_spacing_mm=route.LM_EXIT_TUBE_SECTION_SPACING_MM,
    )
    assert _intersection_volume(owners["lm"], lm_exit_cable) < 0.02, (
        "D7.8 LM cable is blocked in the gradual rear exit")
    face = np.asarray(route.LM_REAR_HANDOFF_SPEC["face"], dtype=float)
    tangent = np.asarray(
        route.LM_REAR_HANDOFF_SPEC["face_tangent"], dtype=float)
    lm_port_face = route._round_tube(
        np.asarray((face - 0.30 * tangent, face + 0.30 * tangent)),
        route.LM_CABLE_D_EST / 2.0)
    assert _intersection_volume(owners["lm"], lm_port_face) < 0.02, (
        "LM rear exit is not open along the R14 face tangent")
    former_bite = route._z_axis_bore(
        (-9.5, 105.0), 0.45,
        core.CORE_REAR_Z, core.CORE_REAR_Z + 0.75)
    retained_bite_witness = _intersection_volume(owners["lm"], former_bite)
    assert retained_bite_witness > 0.995 * former_bite.volume, (
        "LM lower-ring material remains missing beside the gradual port")

    world = Pos(0.0, 220.0, 0.0) * Cylinder(500.0, 200.0)

    def radial_domains(center, radius, owner_inside):
        inside = Pos(*center, 0.0) * Cylinder(radius, 200.0)
        if owner_inside:
            allowed = inside
            forbidden = world - (
                Pos(*center, 0.0) * Cylinder(radius + 0.03, 200.0))
        else:
            allowed = world - inside
            forbidden = (
                Pos(*center, 0.0) * Cylinder(radius - 0.03, 200.0))
        return allowed, forbidden

    def transition_indices(mask):
        return list(np.flatnonzero(mask[1:] != mask[:-1]) + 1)

    declared_closure_allowances = {
        "lm": (core._junction_closure_web("lm_um", "lm"),),
        "um": (
            core._junction_closure_web("lm_um", "um"),
            core._junction_closure_web("t_um", "um"),
        ),
    }

    def assert_flush_mouth(
            label, owner_name, points, index, cutter_radius, outer_radius,
            allowed, forbidden):
        lo = max(0, index - 14)
        hi = min(len(points), index + 15)
        local = points[lo:hi]
        assert len(local) >= 8, f"{label} lacks a local route witness"
        corridor = route._round_tube(local, outer_radius + 0.20)
        protrusion = corridor & forbidden
        protruding_owner = owners[owner_name] & protrusion
        # The new full-depth junction webs intentionally occupy part of the
        # former outside-of-ring domain.  Remove only those exact source-owned
        # solids from this legacy horn diagnostic; the lumen and middle-wall
        # checks below still prove that the route handoff itself is flush and
        # open, while any undeclared cover tongue remains a failure.
        for allowance in declared_closure_allowances.get(owner_name, ()):
            if (protruding_owner is None
                    or protruding_owner.volume <= 1.0e-9):
                protruding_owner = None
                break
            protruding_owner = protruding_owner - allowance
        hit = 0.0 if protruding_owner is None else protruding_owner.volume
        protruding_facts = []
        if hit >= 0.02:
            for solid in protruding_owner.solids():
                bounds = solid.bounding_box()
                protruding_facts.append({
                    "volume_mm3": solid.volume,
                    "min": (bounds.min.X, bounds.min.Y, bounds.min.Z),
                    "max": (bounds.max.X, bounds.max.Y, bounds.max.Z),
                })
        assert hit < 0.02, (
            f"{label} retains a tongue/horn beyond its native boundary: "
            f"{hit:.4f} mm3; components={protruding_facts}")

        # Use the manufactured owner's complete cutter path so the ruled
        # octagons retain their production section phase.  Re-lofting only
        # the local points rotates/repositions the polygonal sections and can
        # report a false cap despite the exact nominal cutter being open.
        cutter_owner = owner_name
        lumen = route._round_tube(
            route._owner_cutter_points(points, cutter_owner),
            cutter_radius - 0.15)
        lumen = lumen & corridor
        assert _intersection_volume(owners[owner_name], lumen) < 0.02, (
            f"{label} final lumen is capped")

        # Probe a conservative middle 0.40 mm of the nominal 0.80 mm wall.
        # Cropping this witness by the native owner domain makes the test
        # sensitive to a missing mouth skin without requiring material in
        # the intentionally free cable gap.
        # Build the conservative middle-wall witness on the exact global
        # section phases used by production: unextended points for the outer
        # cover and the owner-specific extended path for its lumen.  A local
        # re-loft changes octagon phase and falsely asks the final carrier for
        # material outside its manufactured 0.8-mm shell.
        shell_outer = route._round_tube(
            points, cutter_radius + 0.60)
        shell_inner = route._round_tube(
            route._owner_cutter_points(points, owner_name),
            cutter_radius + 0.20)
        shell = (shell_outer - shell_inner) & allowed & corridor
        retained = _intersection_volume(owners[owner_name], shell)
        assert retained > 0.995 * shell.volume, (
            f"{label} skin incomplete at flush mouth: "
            f"{retained:.4f}/{shell.volume:.4f} mm3")

    lm_owner_plan = route._lm_positive_owner_plan()
    lm_allowed = route._polygon_prism(lm_owner_plan, -100.0, 100.0)
    lm_forbidden = world - route._polygon_prism(
        lm_owner_plan.buffer(0.03, join_style=1), -100.0, 100.0)
    main_lm_inside = np.asarray([
        lm_owner_plan.covers(Point(float(x), float(y)))
        for x, y in main[:, :2]
    ])
    main_lm_crossings = transition_indices(
        main_lm_inside)
    assert main_lm_crossings
    # The user-visible upper outlet remains a flush handoff.  Both route
    # covers now finish beneath the continuous R113.8 exterior; beyond it only
    # the exact LM-owned full-depth closure web is legitimate.
    assert_flush_mouth(
        "LM UM free-cable exit", "lm", main, main_lm_crossings[-1],
        route.CUTTER_R, route.MAIN_OUTER_R,
        lm_allowed, lm_forbidden)

    t_lm_inside = np.asarray([
        lm_owner_plan.covers(Point(float(x), float(y)))
        for x, y in ts[:, :2]
    ])
    lm_crossings = transition_indices(
        t_lm_inside)
    # Both states use the same central lower feed. The removed point horn is
    # always the final outward crossing.
    assert lm_crossings
    assert_flush_mouth(
        "LM T outer mouth", "lm", ts, lm_crossings[-1],
        route.TS_CUTTER_R, route.TS_OUTER_R,
        lm_allowed, lm_forbidden)

    t_um_r = np.linalg.norm(
        ts[:, :2] - np.asarray(UM_CUTOUT[:2], dtype=float), axis=1)
    um_crossings = transition_indices(t_um_r <= core.UM_CORE_R)
    assert len(um_crossings) == 2
    um_inside, um_outside = radial_domains(
        UM_CUTOUT[:2], core.UM_CORE_R, True)
    for label, index in zip(
            ("UM T lower mouth", "UM T upper mouth"), um_crossings,
            strict=True):
        assert_flush_mouth(
            label, "um", ts, index,
            route.TS_CUTTER_R, route.TS_OUTER_R,
            um_inside, um_outside)

    # Both deleted suffixes remain physical cables only. Their conservative
    # cable envelopes must clear the final UM and tweeter bodies; this is the
    # positive absence witness for the former printed arcs.
    main_free = main[main_lm_crossings[-1] + 3:]
    main_free_cable = route._round_tube(
        main_free, route.CABLE_R_EST + 0.05)
    assert _intersection_volume(owners["um"], main_free_cable) < 0.02
    t_free_start = um_crossings[-1] + 3
    t_free = ts[t_free_start:]
    t_free_cable = route._round_tube(
        t_free, route.TS_CABLE_D_EST / 2.0 + 0.05)
    assert _intersection_volume(owners["tweeter"], t_free_cable) < 0.02

    assert route.required_handoff_shell_components("UM") == ()
    assert route.required_handoff_shell_components("T") == ()
    assert facts["um_telescoping_handoff_count"] == 0
    assert facts["t_telescoping_handoff_count"] == 0
    state = "floor" if stand_foot else "no-floor"
    staging.cleanup()
    print(
        f"  {state}: central feeds and native LM/UM T mouths retain "
        "full lumens/skins; LM UM/T covers retain a 0.85-mm R113.8 exterior "
        "skin and both deleted "
        "suffixes remain collision-free cable only")






def _crossover_brep(stand_foot):
    _state(stand_foot)
    import lx521_baffle.obiwan.route as route

    main_void = route._round_tube(
        route.route_cable_points(1.2), route.CUTTER_R)
    t_void = route._round_tube(
        route.ts_cable_points(1.2), route.TS_CUTTER_R)
    void_distance = main_void.distance_to(t_void)
    assert void_distance >= route.CROSSOVER_MIN_CLEARANCE - 0.02, (
        f"OCC nominal route-void distance {void_distance:.3f}")
    main_cable = route._round_tube(
        route.route_cable_points(1.2), route.CABLE_R_EST)
    t_cable = route._round_tube(
        route.ts_cable_points(1.2), route.TS_CABLE_D_EST / 2.0)
    cable_distance = main_cable.distance_to(t_cable)
    assert cable_distance >= 1.5
    t_outer = route._round_tube(
        route.ts_cable_points(1.2), route.TS_OUTER_R)
    free_to_cover = main_cable.distance_to(t_outer)
    assert free_to_cover >= 0.25
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state} OCC crossover: nominal void gap {void_distance:.2f} mm, "
        f"{cable_distance:.2f} mm cable gap, free UM cable to printed T "
        f"cover {free_to_cover:.2f} mm")






def test_bridge_contract():
    _state(False)
    from lx521_baffle.obiwan.bridge import (
        BRIDGE_BORE_FLOOR_MM,
        BRIDGE_FACE_Z,
        BRIDGE_FUSION_INTERFACE_T,
        BRIDGE_FUSION_INTERFACE_Z,
        BRIDGE_GOVERNING_NECK_WIDTH_MM,
        BRIDGE_MIN_FUSION_SF_5G,
        BRIDGE_MIN_MEMBER_SF_5G,
        BRIDGE_ROUTE_SECTION_Y_RANGE,
        BRIDGE_WEB_T,
        BRIDGE_WEB_TUNNEL_DEDUCTION_MM,
        BRIDGE_WEB_WIDTH,
        BRIDGE_WEB_X,
        BRIDGE_WEB_Y,
        BRIDGE_BLEND_START_Y,
        LM_WING_CONTACT_FUSION_OVERLAP_MM,
        LM_WING_CONTACT_Z,
        bridge_load_facts,
        bridge_face_plan,
        bridge_plan_facts,
        common_lm_wing_contact_plan,
        floor_wing_contact_profile_addition_plan,
        native_bridge_face_plan,
    )
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    from lx521_baffle.obiwan.floor import integral_stem_plan_points
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    plan = bridge_plan_facts()
    assert plan["holes"] == (
        (-20.0, 20.0), (20.0, 20.0), (-20.0, 70.0), (20.0, 70.0))
    assert plan["vectors_from_lm"] == (
        (-20.0, -180.981), (20.0, -180.981),
        (-20.0, -130.981), (20.0, -130.981))
    assert math.isclose(plan["pattern_width_mm"], 40.0, abs_tol=1e-12)
    assert math.isclose(plan["pattern_height_mm"], 50.0, abs_tol=1e-12)
    assert math.isclose(plan["pattern_center_offset_mm"], 155.981,
                        abs_tol=1e-9)
    assert all(math.isclose(r, expected, abs_tol=1e-6)
               for r, expected in zip(
                   plan["radii_from_lm"],
                   (182.082735, 182.082735, 132.499141, 132.499141)))
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(BRIDGE_FACE_Z, (5.3, 18.3)))
    assert math.isclose(BRIDGE_WEB_T, 13.0, abs_tol=1e-12)
    assert math.isclose(BRIDGE_BORE_FLOOR_MM, 6.2, abs_tol=1e-12)
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(plan["web_z"], BRIDGE_FACE_Z))
    assert math.isclose(plan["rear_insert_entry_z_mm"], 5.3,
                        abs_tol=1e-12)
    assert math.isclose(plan["insert_front_floor_mm"], 6.2,
                        abs_tol=1e-12)
    assert plan["solid_web_bounds"] == (-31.0, 14.0, 31.0, 90.25)
    assert route.NO_FLOOR_BRIDGE_CORE_BOUNDS == (
        BRIDGE_WEB_X[0], BRIDGE_WEB_Y[0],
        BRIDGE_WEB_X[1], BRIDGE_WEB_Y[1])
    assert plan["solid_web_width_mm"] == BRIDGE_WEB_WIDTH == 62.0
    assert plan["solid_web_height_mm"] == 76.25
    assert plan["solid_web_corner_radius_mm"] == 4.0
    assert math.isclose(
        plan["governing_neck_width_mm"], BRIDGE_GOVERNING_NECK_WIDTH_MM,
        abs_tol=1e-12)
    assert math.isclose(
        BRIDGE_GOVERNING_NECK_WIDTH_MM, 38.8, abs_tol=1e-12)
    route_section = plan["route_section"]
    assert BRIDGE_ROUTE_SECTION_Y_RANGE == (73.30, 90.25)
    assert route_section["y_range_mm"] == BRIDGE_ROUTE_SECTION_Y_RANGE
    assert route_section["sample_step_max_mm"] <= 0.01
    assert route_section["minimum_net_width_mm"] >= 38.8
    assert plan["face_opening_count"] == 0
    assert math.isclose(plan["rear_rib_depth_mm"], 0.0, abs_tol=1e-12)
    assert plan["fusion_interface_z"] == BRIDGE_FUSION_INTERFACE_Z
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(
                   BRIDGE_FUSION_INTERFACE_Z, (6.8, 18.3)))
    assert math.isclose(BRIDGE_FUSION_INTERFACE_T, 11.5, abs_tol=1e-12)
    assert math.isclose(
        plan["face_exterior_area_mm2"], plan["face_plan_area_mm2"],
        abs_tol=1e-6)

    # The common 2D envelope still defines the no-floor web and the upper
    # floor shoulder, but the floor addition must not export the old stem's
    # nominal fusion offset as a visible lower rectangular box. It owns no
    # material at all below the common shoulder's y=60 tangent.
    floor_stem = Polygon(integral_stem_plan_points()).buffer(0)
    native_bridge = native_bridge_face_plan()
    universal = common_lm_wing_contact_plan()
    raw_union = unary_union((floor_stem, native_bridge)).buffer(0)
    assert raw_union.geom_type == "Polygon" and len(raw_union.interiors) == 2
    expected = Polygon(raw_union.exterior)
    assert universal.symmetric_difference(expected).area <= 1e-8
    assert bridge_face_plan().symmetric_difference(universal).area <= 1e-8
    floor_effective = unary_union((
        floor_stem, floor_wing_contact_profile_addition_plan())).buffer(0)
    assert floor_effective.symmetric_difference(universal).area <= 1e-8
    assert floor_effective.hausdorff_distance(universal) <= 1e-9
    floor_addition = floor_wing_contact_profile_addition_plan()
    assert len(floor_addition.interiors) == 0
    assert math.isclose(
        floor_addition.bounds[1], BRIDGE_BLEND_START_Y,
        abs_tol=1e-12)
    forbidden_lower_box = box(-200.0, 0.0, 200.0, BRIDGE_BLEND_START_Y)
    assert floor_addition.intersection(forbidden_lower_box).area <= 1e-8
    for actual, expected_bound in zip(
            universal.bounds,
            (-80.59730075442252, 0.0,
             80.59730075442253, 121.77825313411685), strict=True):
        assert math.isclose(actual, expected_bound, abs_tol=1e-9)
    assert plan["universal_wing_contact_profile"] is False
    assert plan["no_floor_wing_contact_profile"] == (
        "common_bridge_plus_floor_stem")
    assert plan["floor_wing_contact_profile"] == (
        "upper_common_shoulder_only")
    assert plan["floor_exposed_perimeter_box"] is False
    assert plan["floor_lower_magnet_rails"] is False
    assert plan["universal_wing_contact_bounds"] == tuple(
        map(float, universal.bounds))
    assert plan["native_bridge_bounds"][1] == 14.0
    assert plan["native_floor_stem_bounds"][1] == 0.0
    assert plan["floor_profile_added_area_mm2"] > 599.0
    assert plan["no_floor_profile_added_area_mm2"] > 2164.0
    assert plan["transition_pocket_fill_count"] == 2
    assert 36.0 < plan["transition_pocket_fill_area_mm2"] < 38.0
    assert plan["wing_contact_z"] == LM_WING_CONTACT_Z == (6.8, 18.3)
    assert math.isclose(
        plan["wing_contact_fusion_overlap_mm"],
        LM_WING_CONTACT_FUSION_OVERLAP_MM, abs_tol=1e-12)
    assert math.isclose(
        plan["floor_profile_min_y_mm"], BRIDGE_BLEND_START_Y,
        abs_tol=1e-12)

    load = bridge_load_facts()
    assert load["design_mass_kg"] == 4.0
    assert load["design_y_cg_mm"] == 230.0
    assert load["rear_cg_mm"] == 70.0
    assert load["root_y_mm"] == 90.25
    assert 73.25 <= load["governing_section_y_mm"] <= 90.25
    assert load["normal_root_lever_mm"] == 139.75
    assert math.isclose(
        load["member_normal_lever_mm"],
        load["design_y_cg_mm"] - load["governing_section_y_mm"],
        abs_tol=1e-12)
    assert load["fusion_normal_root_lever_mm"] == 139.75
    assert load["creep_allow_mpa"] == 8.0
    assert load["short_allow_mpa"] == 18.0
    assert load["insert_pullout_n"] == 600.0
    assert math.isclose(load["web_rear_z_mm"], 5.3, abs_tol=1e-12)
    assert math.isclose(load["web_front_z_mm"], 18.3, abs_tol=1e-12)
    assert math.isclose(load["web_depth_mm"], 13.0, abs_tol=1e-12)
    assert math.isclose(
        load["rear_depth_protrusion_mm"], 0.0, abs_tol=1e-12)
    assert load["gross_web_width_mm"] == 62.0
    assert load["deducted_central_tunnel_width_mm"] == 23.2
    assert BRIDGE_WEB_TUNNEL_DEDUCTION_MM == 23.2
    assert math.isclose(load["net_web_width_mm"], 38.8, abs_tol=1e-12)
    assert math.isclose(
        load["governing_neck_width_mm"], BRIDGE_GOVERNING_NECK_WIDTH_MM,
        abs_tol=1e-12)
    member_section = load["member_section"]
    assert member_section["model"] == (
        "sampled_axis_aligned_lumen_bounding_rectangles")
    assert member_section["sample_step_max_mm"] <= 0.05
    assert member_section["centerline_step_max_mm"] <= 0.02
    assert member_section["void_bounding_margin_mm"] >= 0.05
    assert member_section["lumen_bounding_rectangle_count"] in {2, 3}
    assert set(member_section["lumen_names"]).issubset({"lm", "um", "t"})
    assert load["area_mm2"] < load["gross_web_width_mm"] * load["web_depth_mm"]
    assert load["magnet_load_credit_n"] == 0.0
    assert load["combined_insert_5g_n"] > load["normal_insert_5g_n"]
    assert load["member_sf_1g_creep"] >= 2.0
    assert load["member_sf_3g"] >= 1.5
    assert load["member_sf_5g"] >= BRIDGE_MIN_MEMBER_SF_5G
    assert load["insert_sf_5g"] >= 1.35
    assert load["fusion_interface"]["span_deg"] == 68.0
    assert load["fusion_interface"]["deducted_lm_tunnel_count"] == 1
    assert load["fusion_interface"]["deducted_lm_tunnel_width_mm"] == 9.0
    assert load["fusion_interface"]["deducted_um_tunnel_count"] == 1
    assert load["fusion_interface"]["deducted_um_tunnel_width_mm"] == 8.2
    assert load["fusion_interface"]["deducted_t_tunnel_width_mm"] == 6.0
    assert load["fusion_interface"]["deducted_tunnel_width_mm"] == 23.2
    assert load["fusion_interface"]["effective_width_mm"] > 109.0
    assert load["fusion_interface"]["interface_z"] == (
        BRIDGE_FUSION_INTERFACE_Z)
    assert math.isclose(
        load["fusion_interface"]["interface_height_mm"], 11.5,
        abs_tol=1e-12)
    assert load["fusion_sf_1g_creep"] >= 3.0
    assert load["fusion_sf_3g"] >= BRIDGE_MIN_FUSION_SF_5G
    assert load["fusion_sf_5g"] >= BRIDGE_MIN_FUSION_SF_5G
    assert load["fusion_shear_sf_1g"] >= 10.0
    assert load["fusion_shear_sf_3g"] >= 5.0
    assert load["fusion_shear_sf_5g"] >= 5.0
    print(
        f"  front-flush bridge plate: conservative sampled X/Z route "
        f"section (plan lower bound {BRIDGE_GOVERNING_NECK_WIDTH_MM:.2f} "
        f"x {BRIDGE_WEB_T:.1f} mm), "
        f"5g SF {load['member_sf_5g']:.2f}; no depth beyond LM pads")


def test_bridge_geometry():
    _state(False)
    from build123d import Box, Cylinder, Pos, Vertex
    from lx521_baffle.base import (
        BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM, BRIDGE_INSERT_DEPTH_MM,
        M5_INSERT_BODY_D_MM, M5_INSERT_ENTRY_DEPTH_MM,
        M5_INSERT_ENTRY_D_MM)
    from lx521_baffle.obiwan.bridge import (
        BRIDGE_FACE_Z,
        BRIDGE_WEB_FRONT_Z,
        BRIDGE_WEB_REAR_Z,
        bridge_fastener_head_envelopes,
        bridge_face_plan,
        fused_bridge_tail,
    )

    tail = fused_bridge_tail()
    assert tail.is_valid and len(tail.solids()) == 1
    bb = tail.bounding_box()
    assert math.isclose(bb.min.Y, 0.0, abs_tol=1e-6)
    assert math.isclose(bb.min.Z, BRIDGE_WEB_REAR_Z, abs_tol=1e-6)
    assert math.isclose(bb.max.Z, BRIDGE_WEB_FRONT_Z, abs_tol=1e-6)
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(BRIDGE_FACE_Z, (5.3, 18.3)))
    assert math.isclose(M5_INSERT_ENTRY_D_MM, 6.5, abs_tol=1e-12)
    assert math.isclose(M5_INSERT_ENTRY_DEPTH_MM, 2.0, abs_tol=1e-12)
    assert len(bridge_face_plan().interiors) == 0

    # Every representative region that used to be an open frame/X cell is
    # now solid at the front face.
    for x, y in (
            (0.0, 5.0), (0.0, 45.0),
            (-12.0, 35.0), (12.0, 55.0), (0.0, 82.0)):
        probe = Pos(x, y, 17.5) * Cylinder(0.45, 0.8)
        assert _intersection_volume(tail, probe) > 0.95 * probe.volume

    # Four bores open from the web rear, stop after 6.8 mm, and retain a
    # positive solid front floor. Screw heads have an unobstructed approach.
    for x, y in BRIDGE_HOLE_XY:
        bore = Pos(
            x, y, BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM / 2.0
        ) * Cylinder(BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
        assert _intersection_volume(tail, bore) < 0.02
        entry_radius = Vertex(
            x, y, BRIDGE_WEB_REAR_Z + 1.0
        ).distance_to(tail)
        body_radius = Vertex(
            x, y, BRIDGE_WEB_REAR_Z + 3.0
        ).distance_to(tail)
        assert math.isclose(
            entry_radius, M5_INSERT_ENTRY_D_MM / 2.0, abs_tol=2.0e-5)
        assert math.isclose(
            body_radius, M5_INSERT_BODY_D_MM / 2.0, abs_tol=2.0e-5)
        floor_z0 = BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM + 0.1
        floor = Pos(
            x, y, (floor_z0 + BRIDGE_WEB_FRONT_Z - 0.1) / 2.0
        ) * Cylinder(1.0, BRIDGE_WEB_FRONT_Z - 0.1 - floor_z0)
        assert _intersection_volume(tail, floor) > 0.98 * floor.volume
    assert _intersection_volume(tail, bridge_fastener_head_envelopes()) < 0.02

    # Exercise the actual Obi-Wan carrier mount-hole function on one minimal
    # production-sized boss.  This independently pins the recessed 6.2-mm
    # LM total: only its first 2.0 mm grows to D6.5, with D6.4 below and the
    # existing 0.8-mm printed floor retained above PAD_FACE_Z.
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    x, y = flush.LM_PILOT_XY[0]
    boss = Pos(
        x, y, (flush.PAD_FACE_Z + flush.LM_SEAT_Z) / 2.0
    ) * Cylinder(
        flush.PAD_D_MM / 2.0,
        flush.LM_SEAT_Z - flush.PAD_FACE_Z,
    )
    boss = core._cut_lm_mount_holes(boss).clean()
    assert boss.is_valid and len(boss.solids()) == 1
    assert math.isclose(flush.LM_BORE_DEPTH_MM, 6.2, abs_tol=1.0e-12)
    assert math.isclose(
        Vertex(x, y, flush.LM_SEAT_Z - 1.0).distance_to(boss),
        M5_INSERT_ENTRY_D_MM / 2.0,
        abs_tol=2.0e-5,
    )
    assert math.isclose(
        Vertex(x, y, flush.LM_SEAT_Z - 3.0).distance_to(boss),
        M5_INSERT_BODY_D_MM / 2.0,
        abs_tol=2.0e-5,
    )
    lm_floor_z0 = flush.PAD_FACE_Z + 0.1
    lm_floor_z1 = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM - 0.1
    lm_floor = Pos(
        x, y, (lm_floor_z0 + lm_floor_z1) / 2.0
    ) * Cylinder(1.0, lm_floor_z1 - lm_floor_z0)
    assert _intersection_volume(boss, lm_floor) > 0.95 * lm_floor.volume
    print(
        "  bridge BREP: four rear D6.5x2/D6.4x6.8 bores; "
        "Obi-Wan LM boss: front D6.5x2/D6.4x6.2 bore"
    )


def test_joint_load_contract():
    _state(False)
    from lx521_baffle.obiwan.carriers import (
        carrier_spoke_load_facts,
        joint_load_facts,
    )

    load = joint_load_facts()
    assert load["design_mass_kg"] == 0.85
    assert load["creep_allow_mpa"] == 8.0
    assert load["short_allow_mpa"] == 18.0
    assert load["m3_shear_allow_mpa"] == 100.0
    assert load["m3_tension_allow_mpa"] == 100.0
    assert load["plan_lever_mm"] == 120.0
    assert load["rear_lever_mm"] == 70.0
    assert load["magnet_load_credit_n"] == 0.0
    assert math.isclose(
        load["lm_um_clearance_bore_d_mm"], 3.4, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_insert_receiver_d_mm"], 4.6, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_insert_receiver_depth_mm"], 4.0, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_insert_front_floor_mm"], 1.9, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_functional_boss_d_mm"], 9.8, abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_functional_boss_d_mm"], 9.8,
        abs_tol=1e-12)
    assert math.isclose(load["lm_um_net_width_mm"], 5.2, abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_net_width_mm"], 5.2, abs_tol=1e-12)
    assert math.isclose(
        load["minimum_half_thickness_mm"], 5.4, abs_tol=1e-12)
    assert math.isclose(
        load["full_half_thickness_mm"], 5.4, abs_tol=1e-12)
    assert math.isclose(
        load["nearby_route_front_ligament_mm"], 5.35, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_axial_gap_mm"], 0.20, abs_tol=1e-12)
    assert math.isclose(
        load["lm_um_bore_overlap_mm"], 0.35, abs_tol=1e-12)
    assert load["lm_um_standalone_ear_ownership_required"] is True
    assert load["lm_um_full_360_wall_required"] is True
    assert load["lm_um_cross_owner_material_allowed"] is False
    # The analytic printed-ear/M3 screen deliberately does not certify
    # polymer-specific insert retention. Release still requires a physical
    # pullout qualification in the chosen material, orientation and process.
    assert load["lm_um_insert_pullout_qualification_required"] is True
    assert math.isclose(
        load["um_tweeter_clearance_bore_d_mm"], 3.4,
        abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_insert_receiver_d_mm"], 4.6,
        abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_insert_receiver_depth_mm"], 4.0,
        abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_insert_front_floor_mm"], 1.9,
        abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_axial_gap_mm"], 0.20, abs_tol=1e-12)
    assert math.isclose(
        load["um_tweeter_bore_overlap_mm"], 0.35, abs_tol=1e-12)
    assert load["um_tweeter_standalone_ear_ownership_required"] is True
    assert load["um_tweeter_full_360_wall_required"] is True
    assert load["um_tweeter_cross_owner_material_allowed"] is False
    assert load["um_tweeter_insert_pullout_qualification_required"] is True
    assert load["pla_sf_1g_creep"] >= 3.0
    assert load["pla_sf_3g"] >= 2.0
    assert load["pla_sf_5g"] >= 1.5
    assert load["m3_shear_sf_5g"] >= 4.0
    moment_thresholds = {
        "moment_1g": 2.7,
        "moment_3g": 2.0,
        "moment_5g": 1.25,
    }
    for case, threshold in moment_thresholds.items():
        moment = load[case]
        assert moment["governing_interface"] in {"lm_um", "um_tweeter"}
        assert set(moment["interfaces"]) == {"lm_um", "um_tweeter"}
        lm_um = moment["interfaces"]["lm_um"]
        um_tweeter = moment["interfaces"]["um_tweeter"]
        assert math.isclose(
            lm_um["contact_lever_mm"], 7.35, abs_tol=1e-12)
        assert math.isclose(
            um_tweeter["contact_lever_mm"], 7.35, abs_tol=1e-12)
        assert math.isclose(
            lm_um["net_area_per_ear_mm2"], 5.2 * 5.4,
            abs_tol=1e-12)
        assert math.isclose(
            um_tweeter["net_area_per_ear_mm2"], 5.2 * 5.4,
            abs_tol=1e-12)
        assert lm_um["contact_sf"] >= threshold
        assert um_tweeter["contact_sf"] >= threshold
        assert math.isclose(lm_um["contact_sf"],
                            um_tweeter["contact_sf"], abs_tol=1e-12)
        assert math.isclose(
            moment["contact_sf"], min(
                lm_um["contact_sf"], um_tweeter["contact_sf"]),
            abs_tol=1e-12)
        assert math.isclose(
            moment["lm_um_insert_pullout_required_n"],
            lm_um["contact_force_per_ear_n"], abs_tol=1e-12)
        assert math.isclose(
            moment["um_tweeter_insert_pullout_required_n"],
            um_tweeter["contact_force_per_ear_n"], abs_tol=1e-12)
        assert "lm_um_insert_pullout_sf" not in moment
    assert load["moment_5g"]["m3_tension_sf"] >= 1.15
    spokes = carrier_spoke_load_facts()
    assert spokes["design_mass_kg"] == 4.0
    assert spokes["creep_allow_mpa"] == 8.0
    assert spokes["short_allow_mpa"] == 18.0
    assert spokes["lm_sf_1g"] >= 6.0
    assert spokes["lm_sf_3g"] >= 4.0
    assert spokes["lm_sf_5g"] >= 4.0
    assert spokes["um_sf_1g"] >= 8.0
    assert spokes["um_sf_3g"] >= 4.0
    assert spokes["um_sf_5g"] >= 4.0
    print(
        f"  both two-ear interfaces screen 0.85 kg upper mass: PLA SF "
        f"1g/3g/5g {load['pla_sf_1g_creep']:.2f}/"
        f"{load['pla_sf_3g']:.2f}/{load['pla_sf_5g']:.2f}; "
        f"M3 shear SF5g {load['m3_shear_sf_5g']:.2f}")


def test_um_driver_spoke_is_separate_from_lm_um_insert_ear():
    """The adjacent MU10 support must not grow a visible M3-ear bridge."""
    _state(False)
    from shapely.geometry import Point
    from lx521_baffle.base import UM_CUTOUT, UM_PILOT_ANGLES_DEG
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core

    assert core.UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG == {238.0: 5.0}
    assert math.isclose(
        core.UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z,
        core.UM_JOINT_Z[0] - 0.01, abs_tol=1.0e-12)
    # The dense closure sweep samples 0.03 mm on both sides of the ear
    # transition.  The lower radial root must cover the final lower sample,
    # yet leave the ear-height support fully deflected and separate.
    assert (core.UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z
            >= core.UM_JOINT_Z[0] - 0.03)
    assert (core.UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z
            < core.UM_JOINT_Z[0])
    index = UM_PILOT_ANGLES_DEG.index(238.0)
    pilot = flush.UM_PILOT_XY[index]
    um_boss_floor = (
        flush.UM_SEAT_Z - core.UM_PILOT_DEPTH_MM
        - flush.UM_PAD_FLOOR_MM)
    # This must be a Z handoff, not a lower-Z fan of radial and deflected
    # supports.  The latter brackets a small sealed void against the driver
    # recess; the former keeps each slice a single load path.
    assert core.um_pilot_spoke_z_segments(
        238.0, um_boss_floor, flush.UM_SEAT_Z) == (
            (0.0, um_boss_floor,
             core.UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z),
            (5.0, core.UM_PILOT_LOWER_RADIAL_SPOKE_TOP_Z,
             flush.UM_SEAT_Z),
        )
    assert core.um_pilot_spoke_z_segments(
        58.0, um_boss_floor, flush.UM_SEAT_Z) == (
            (0.0, um_boss_floor, flush.UM_SEAT_Z),)
    closure_land = core._um_pilot_recess_closure_land(
        UM_CUTOUT[:2], pilot)
    assert math.isclose(
        core.UM_PILOT_RECESS_CLOSURE_LAND_EXPANSION_MM, 0.45,
        abs_tol=1.0e-12)
    assert math.isclose(
        core.UM_PILOT_RECESS_CLOSURE_LAND_DEPTH_MM, 1.20,
        abs_tol=1.0e-12)
    # The crescent-closing land is internal to the driver recess/ring
    # interface: it resolves an inaccessible tiny void but does not create
    # an exterior boss, locator, or M3-ear bridge.
    recess_contact = Point(*UM_CUTOUT[:2]).buffer(
        core.UM_RECESS_R + 0.25, resolution=128)
    assert closure_land.difference(recess_contact).area < 1.0e-8
    assert closure_land.area < 6.1
    spoke = core._radial_spoke_plan(
        UM_CUTOUT[:2], pilot, core.UM_RECESS_R + 0.25,
        core.UM_STRUCT_SPOKE_W,
        core.UM_PILOT_SPOKE_TANGENTIAL_OFFSETS_DEG[238.0])
    boss = Point(*pilot).buffer(core.UM_INSERT_BOSS_D / 2.0,
                                resolution=32)
    ear = core._complete_joint_ear_plan("um", core.JOINT_EAR_X[0])
    support = boss.union(spoke)
    assert ear.intersection(support).area < 1.0e-8
    assert ear.distance(support) >= core.UM_JOINT_EAR_SPOKE_CLEAR_MM
    assert ear.intersection(closure_land).area < 1.0e-8
    assert closure_land.intersection(support).area > 0.5

    # The rerouted support still has a positive overlap with the structural
    # outer lip, so the D4.6 driver insert remains a connected load path.
    outer_lip = Point(*UM_CUTOUT[:2]).buffer(
        core.UM_CORE_R, resolution=128).difference(
            Point(*UM_CUTOUT[:2]).buffer(
                core.UM_RECESS_R, resolution=128))
    assert spoke.intersection(outer_lip).area > 1.0
    print("  238-degree UM support clears LM--UM M3 ear by "
          f"{ear.distance(support):.3f} mm while remaining lip-connected")


def _assert_lm_mount_bores(lm, core, flush):
    """Probe all six exact rotated mount axes, including retained floors."""
    from build123d import Cylinder, Pos, Vertex
    from lx521_baffle.base import (
        L22_PILOT_D_MM,
        M5_INSERT_BODY_D_MM,
        M5_INSERT_ENTRY_DEPTH_MM,
        M5_INSERT_ENTRY_D_MM,
    )

    assert math.isclose(M5_INSERT_ENTRY_DEPTH_MM, 2.0, abs_tol=1.0e-12)
    for angle, (x, y) in zip(
            flush.OBIWAN_LM_PILOT_ANGLES_DEG, flush.LM_PILOT_XY):
        bore_z0 = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM
        # Keep the clearance witness wholly inside the specified blind bore.
        # Extending it below ``bore_z0`` would classify the deliberately
        # retained printed floor as an obstruction.
        bore_z1 = flush.LM_SEAT_Z + 0.1
        bore = Pos(x, y, (bore_z0 + bore_z1) / 2.0) * Cylinder(
            L22_PILOT_D_MM / 2.0,
            bore_z1 - bore_z0)
        assert _intersection_volume(lm, bore) < 0.02, (
            f"{angle:g}deg blind insert bore is obstructed")
        entry_radius = Vertex(
            x, y, flush.LM_SEAT_Z - 1.0
        ).distance_to(lm)
        body_radius = Vertex(
            x, y, flush.LM_SEAT_Z - 3.0
        ).distance_to(lm)
        assert math.isclose(
            entry_radius, M5_INSERT_ENTRY_D_MM / 2.0, abs_tol=2.0e-5), (
            angle, entry_radius, "M5 entry relief")
        assert math.isclose(
            body_radius, M5_INSERT_BODY_D_MM / 2.0, abs_tol=2.0e-5), (
            angle, body_radius, "M5 body bore")
        floor_h = bore_z0 - flush.PAD_FACE_Z - 0.2
        floor = Pos(
            x, y, flush.PAD_FACE_Z + 0.1 + floor_h / 2.0
        ) * Cylinder(1.0, floor_h)
        assert _intersection_volume(lm, floor) > 0.90 * floor.volume, (
            f"{angle:g}deg blind insert lost its printed floor")


def test_floor_lm_core():
    _state(True)
    staged = _stage_shell_contract_breps(
        True, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Cylinder, Pos, Rot, import_brep
    from export_piece_stls import BED_MM, BED_ROT_Z
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.floor as floor
    import lx521_baffle.flush as flush

    lm = import_brep(staged["lm"])
    assert lm.is_valid and len(lm.solids()) == 1
    minimum_edge_mm = min(float(edge.length) for edge in lm.edges())
    assert minimum_edge_mm >= 1.0e-5, (
        "floor LM contains a micron-scale coincident-feature edge: "
        f"{minimum_edge_mm:.9g} mm")
    bounds = lm.bounding_box()
    assert math.isclose(bounds.min.Y, floor.FLOOR_Y_MM, abs_tol=0.02)
    assert math.isclose(bounds.min.Z, floor.FOOT_REAR_Z_MM, abs_tol=0.02)
    assert math.isclose(bounds.max.Z, floor.FOOT_FRONT_Z_MM, abs_tol=0.02)
    assert lm.volume > 150000.0, (
        f"floor LM lost its integral full-depth stem/foot: {lm.volume:.1f}")
    # Empty witness below the 0.85-mm seat membrane, halfway between the
    # 0/60-degree insert spokes and 5 mm outside the driver opening in the
    # deleted annular slab.  The former fixed 90-degree witness became
    # route-owned when the upper feed moved; this relationship-based sector
    # remains away from all three routes and from both adjacent spokes.  A
    # retained annular slab fills this probe.
    witness_angle_deg = sum(flush.OBIWAN_LM_PILOT_ANGLES_DEG[:2]) / 2.0
    witness_radius_mm = core.L22_CUTOUT[2] / 2.0 + 5.0
    witness_angle_rad = math.radians(witness_angle_deg)
    membrane_void = Pos(
        core.L22_CUTOUT[0] + witness_radius_mm * math.cos(witness_angle_rad),
        core.L22_CUTOUT[1] + witness_radius_mm * math.sin(witness_angle_rad),
        9.0,
    ) * Cylinder(1.0, 2.0)
    assert _intersection_volume(lm, membrane_void) < 0.01
    assert BED_MM == 256.0
    _assert_lm_mount_bores(lm, core, flush)
    rotated = Rot(Z=BED_ROT_Z["obiwan_core_1_of_2_lm_carrier"]) * lm
    bb = rotated.bounding_box().size
    assert max(bb.X, bb.Y, bb.Z) > BED_MM, (
        "canonical monolithic floor LM unexpectedly fits the standard bed; "
        "the separately checked keyed option is the <=220-mm print path")
    print(
        f"  floor LM: integral Y=0 foot; intentional large-format reference "
        f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f}")


def test_no_floor_lm_core():
    _state(False)
    staged = _stage_shell_contract_breps(
        False, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Cylinder, Pos, Rot, import_brep
    from shapely.geometry import LineString, Point
    from export_piece_stls import BED_MM, BED_ROT_Z
    from lx521_baffle.base import BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.route as route
    from lx521_baffle.obiwan.bridge import (
        BRIDGE_FACE_Z,
        BRIDGE_FUSION_CRADLE_Z,
        BRIDGE_INSERT_DEPTH_MM,
        BRIDGE_WEB_T,
        BRIDGE_WEB_FRONT_Z,
        BRIDGE_WEB_REAR_Z,
        bridge_fastener_head_envelopes,
        bridge_face_plan,
        bridge_fusion_cradle_plan,
        bridge_fusion_interface_facts,
        bridge_load_facts,
        bridge_plan_facts,
        bridge_solid_web_plan,
    )

    lm = import_brep(staged["lm"])
    assert lm.is_valid and len(lm.solids()) == 1
    # The measured pre-compatibility pinned-osado baseline is 102,784.6 mm3.
    # Permit only the conservative gross burial-web volume plus the exact
    # common-profile growth above that narrow 103,000 mm3 bracket; actual
    # growth remains smaller because neither estimate subtracts overlaps.
    burial_growth_ceiling = route.route_facts()[
        "lm_burial_web_growth_upper_bound_mm3"]
    profile_growth_ceiling = (
        bridge_plan_facts()["no_floor_profile_added_area_mm2"]
        * BRIDGE_WEB_T)
    assert lm.volume < (
            103000.0 + burial_growth_ceiling + profile_growth_ceiling), (
        f"no-floor LM exceeds minimal-material budget: {lm.volume:.1f} mm3")
    assert BED_MM == 256.0
    assert math.isclose(lm.bounding_box().min.Y, 0.0, abs_tol=0.02), (
        "no-floor LM lost the universal Y=0 front tongue")
    _assert_lm_mount_bores(lm, core, flush)
    for x, y in BRIDGE_HOLE_XY:
        bore = Pos(
            x, y, BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM / 2.0
        ) * Cylinder(
            BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
        hit = lm & bore
        assert hit is None or hit.volume < 0.02
        floor_z0 = BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM + 0.1
        floor_probe = Pos(
            x, y, (floor_z0 + BRIDGE_WEB_FRONT_Z - 0.1) / 2.0
        ) * Cylinder(
            1.0, BRIDGE_WEB_FRONT_Z - 0.1 - floor_z0)
        floor_hit = lm & floor_probe
        assert floor_hit is not None and floor_hit.volume > 1.0
    head_hit = lm & bridge_fastener_head_envelopes()
    assert head_hit is None or head_hit.volume < 0.02
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(
                   BRIDGE_FACE_Z,
                   (flush.PAD_FACE_Z, core.THICKNESS_MM)))
    web = core._plan_prism(bridge_solid_web_plan(), *BRIDGE_FACE_Z)
    # The full panel is present except for its four intentional bores. The
    # deleted LM micro-duct leaves no void; broad front witnesses stay solid.
    for x, y in ((-24.0, 40.0), (0.0, 45.0), (24.0, 60.0)):
        witness = Pos(x, y, 17.4) * Cylinder(0.5, 1.0)
        assert _intersection_volume(lm, witness) > 0.95 * witness.volume
    assert math.isclose(
        web.bounding_box().min.Z, BRIDGE_WEB_REAR_Z, abs_tol=1e-9)
    assert math.isclose(
        web.bounding_box().max.Z, BRIDGE_WEB_FRONT_Z, abs_tol=1e-9)
    # `bridge_face_plan` is the universal *outer* profile.  Remove the
    # immutable R110.6 flange recess before asking for solid material: its
    # intersection is 1195.63 mm2 and is deliberately cut from both states.
    front_material_plan = bridge_face_plan().difference(
        Point(*core.L22_CUTOUT[:2]).buffer(
            core.LM_RECESS_R, resolution=256)).buffer(0)
    front_contract = core._plan_prism(
        front_material_plan, BRIDGE_WEB_FRONT_Z - 0.35,
        BRIDGE_WEB_FRONT_Z - 0.05)
    front_missing = front_contract - lm
    front_missing_volume = 0.0 if front_missing is None else sum(
        solid.volume for solid in front_missing.solids())
    assert front_missing_volume < 0.05, (
        f"front-flush bridge face has {front_missing_volume:.3f} mm3 missing")
    lip = core._cylinder_at(
        core.L22_CUTOUT[0], core.L22_CUTOUT[1], core.LM_CORE_R,
        core.CORE_REAR_Z, core.THICKNESS_MM)
    lip -= core._cylinder_at(
        core.L22_CUTOUT[0], core.L22_CUTOUT[1], core.LM_RECESS_R,
        core.CORE_REAR_Z - 0.1, core.THICKNESS_MM + 0.1)
    cradle = core._plan_prism(
        bridge_fusion_cradle_plan(), *BRIDGE_FUSION_CRADLE_Z)
    interface = cradle & lip
    assert interface is not None and interface.volume > 2500.0, (
        "solid-web cradle does not overlap the LM structural lip")
    # The two printed route footprints cross the critical cradle sector in
    # plan: UM at 300 deg and tweeter at 240 deg. Their centerlines dip
    # beneath the structural slice while bypassing insert bores; the screen
    # nevertheless deducts both complete projected widths. The LM lead is a
    # free cable below the web and has no printed void to deduct.
    critical_paths = {
        "UM_D8p2": (route.route_cable_points(1.5), route.CUTTER_R),
        "T_D6": (route.ts_cable_points(1.5), route.TS_CUTTER_R),
    }
    cradle_plan = bridge_fusion_cradle_plan()
    for name, (points, radius) in critical_paths.items():
        footprint = LineString(points[:, :2]).buffer(
            radius, resolution=16, cap_style=1, join_style=1)
        assert footprint.intersection(cradle_plan).area > 0.01, (
            f"{name} does not cross the calculated bridge cradle in plan")
    # Subtract the actual 3-D duct BREP—not the analytical deductions—and
    # measure the retained interface section from the resulting solid.
    for cutter in route.route_inner_cutters("lm"):
        interface = interface - cutter
    interface = interface.clean()
    interface_hit = lm & interface
    assert interface_hit is not None
    assert interface_hit.volume > 0.98 * interface.volume, (
        "final LM lost the routed cradle-to-structural-lip interface")
    interface_facts = bridge_fusion_interface_facts()
    assert interface_facts["interface_z"] == (
        core.CORE_REAR_Z, core.THICKNESS_MM)
    overlap_radial = core.LM_CORE_R - core.LM_RECESS_R
    retained_arc_width = (
        interface.volume
        / (overlap_radial * interface_facts["interface_height_mm"]))
    assert retained_arc_width >= interface_facts["effective_width_mm"] - 0.2
    retained_section = (
        retained_arc_width * interface_facts["interface_height_mm"] ** 2
        / 6.0)
    assert retained_section >= interface_facts[
        "rear_section_modulus_mm3"] - 5.0
    assert bridge_load_facts()["fusion_sf_5g"] >= 1.4
    front_down = (
        Rot(Z=BED_ROT_Z["obiwan_core_1_of_2_lm_carrier"])
        * Rot(X=180.0) * lm)
    bb = front_down.bounding_box().size
    assert max(bb.X, bb.Y, bb.Z) > 0.0
    print(
        "  no-floor LM: fused front-flush four-hole solid web; "
        "front-down footprint "
        f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f}")


def _load_lm_keyed_parts(stand_foot):
    """Load the exact hash-validated release stage for the split gate."""
    from build123d import import_brep

    paths = _validated_obiwan_stage_paths(stand_foot)
    lm = import_brep(str(paths["core_lm_carrier"]))
    um = import_brep(str(paths["core_um_carrier"]))
    parts = {
        key: import_brep(str(paths[key]))
        for key in (
            "optional_lm_keyed_1_of_2_bottom",
            "optional_lm_keyed_2_of_2_top",
        )
    }
    return lm, um, parts


def _validated_obiwan_stage_paths(stand_foot):
    """Return only the Make-owned, hash-validated Obi-Wan native stage.

    Every R6F Make node depends on ``validate_obiwan_stages``.  Consumers must
    therefore reuse that single source/runtime/guard-bound transaction rather
    than constructing a second private copy of the LM, UM, or tweeter BREP.
    """
    _state(stand_foot)
    from export_obiwan_staged import load_stage_manifest, staged_part_paths

    state_name = "floor_stand" if stand_foot else "no_floor_stand"
    manifest = PROJECT_ROOT / "build" / state_name / (
        ".obiwan_stage/manifest.json")
    # These tests consume an already promoted, hash-bound native stage.  Its
    # recorded osado runtime/guard identity is part of the source fingerprint,
    # but a portable post-promotion audit must not pretend that the local
    # workstation has that active build environment.
    payload = load_stage_manifest(
        manifest, stand_foot=stand_foot,
        require_active_environment=False)
    return staged_part_paths(manifest, payload)


def _assert_lower_shoulder_magnet_split_ownership(
        lm, lm_lower, lm_upper, core, lm_split, state):
    """Keep both lower captive stations wholly in the lower LM print.

    The split pieces are derived from the finalized canonical carrier, so a
    void-only check is insufficient: an accidentally detached neighborhood
    would also appear unobstructed.  Compare the complete positive qualified
    land after cradle/chimney/roof subtraction, both 0.45-mm axial skins, and
    the seated-magnet void against the canonical BREP.  The upper print must
    own none of this station.
    """
    from lx521_baffle.magnets import DEFAULT_SPEC, wall_cavity_tools

    lower_sites = {
        site["name"]: site
        for site in core.side_magnet_sites("lm")
        if site["interface_kind"] == "shoulder"
    }
    assert set(lower_sites) == {"lm_lower_left", "lm_lower_right"}

    seam = lm_split.LM_SPLIT_SEAM_Y
    for name, site in lower_sites.items():
        nx, ny = site["normal"]
        tangent_half_width = (
            DEFAULT_SPEC.cavity_radius_mm
            + DEFAULT_SPEC.side_wall_margin_mm)
        assert site["face"][1] + tangent_half_width < seam
        assert math.isclose(site["face_offset_mm"], -0.15, abs_tol=1e-12)
        tools = wall_cavity_tools(
            name=name, face=site["face"],
            outward=(nx, ny, 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP)

        for cutter_index, cutter in enumerate(tools.cutters):
            assert _intersection_volume(lm, cutter) < 0.03, (
                f"{state} canonical {name} cutter {cutter_index} blocked")
            assert _intersection_volume(lm_lower, cutter) < 0.03, (
                f"{state} lm_lower lost {name} cavity/roof cutter "
                f"{cutter_index}")
            assert _intersection_volume(lm_upper, cutter) < 0.03, (
                f"{state} lm_upper intrudes into {name} cutter "
                f"{cutter_index}")
        assert _intersection_volume(lm, tools.nominal_magnet) < 0.02
        assert _intersection_volume(lm_lower, tools.nominal_magnet) < 0.02

        qualified_solid = tools.required_land
        for cutter in tools.cutters:
            qualified_solid = qualified_solid - cutter
        canonical_land = _intersection_volume(lm, qualified_solid)
        lower_land = _intersection_volume(lm_lower, qualified_solid)
        upper_land = _intersection_volume(lm_upper, qualified_solid)
        assert canonical_land > 0.98 * qualified_solid.volume, (
            f"{state} canonical {name} lost qualified captive land")
        assert abs(lower_land - canonical_land) < 0.04, (
            f"{state} {name} captive land is not wholly in lm_lower: "
            f"{lower_land:.3f}/{canonical_land:.3f} mm3")
        assert upper_land < 0.02, (
            f"{state} {name} captive land leaked into lm_upper by "
            f"{upper_land:.3f} mm3")

        # Independently probe the sealed interface and inner axial skins.
        skin_diameter = core.SIDE_MAGNET_D - 0.4
        face_skin = core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"], skin_diameter,
            DEFAULT_SPEC.face_skin_mm - 0.03, 0.0)
        inner_face = (
            site["face"][0]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * nx,
            site["face"][1]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * ny,
        )
        inner_skin = core._axis_cylinder(
            inner_face, site["normal"], site["z_mm"], skin_diameter,
            DEFAULT_SPEC.inner_skin_mm - 0.03, 0.0)
        for label, skin in (("interface", face_skin),
                            ("inner", inner_skin)):
            assert _intersection_volume(lm, skin) > 0.97 * skin.volume, (
                f"{state} canonical {name} lost sealed {label} skin")
            assert _intersection_volume(
                lm_lower, skin) > 0.97 * skin.volume, (
                    f"{state} lm_lower lost {name} sealed {label} skin")
            assert _intersection_volume(lm_upper, skin) < 0.02, (
                f"{state} lm_upper owns {name} {label} skin")


def _assert_lm_keyed_split(stand_foot):
    _state(stand_foot)
    from build123d import Rot
    from shapely.geometry import box
    from export_piece_stls import (
        BED_ROT_Z,
        OBIWAN_OPTIONAL_LM_SPLIT_BED_MM,
    )
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.lm_split as lm_split
    import lx521_baffle.obiwan.route as route
    import lx521_baffle.um_fit as fit

    lm, um, parts = _load_lm_keyed_parts(stand_foot)
    expected_keys = {
        "optional_lm_keyed_1_of_2_bottom",
        "optional_lm_keyed_2_of_2_top",
    }
    assert set(parts) == expected_keys
    bottom = parts["optional_lm_keyed_1_of_2_bottom"]
    top = parts["optional_lm_keyed_2_of_2_top"]
    for name, part in parts.items():
        assert part.is_valid and len(part.solids()) == 1, name
        assert part.volume > 0.01
    assert _intersection_volume(bottom, top) < 0.02

    # The hidden registration now lives wholly inside the native smooth ring.
    # No source material is added and neither keyed half may grow beyond the
    # canonical monolithic carrier.  Do not subtract independently imported
    # parent/child BREPs to prove that relationship: OCCT can classify their
    # exactly coincident faces as disjoint and return the entire child as the
    # difference.  Instead gate the designed local tools, seam ownership,
    # parent envelope and exact volume balance.
    augmented_lm = lm_split.registration_augmented_carrier(lm)
    support_tool = lm_split.registration_support_land_tool()
    assert math.isclose(augmented_lm.volume, lm.volume, abs_tol=0.03)
    lm_bounds = lm.bounding_box()
    for name, part in parts.items():
        bounds = part.bounding_box()
        assert bounds.min.X >= lm_bounds.min.X - 0.02, name
        assert bounds.min.Y >= lm_bounds.min.Y - 0.02, name
        assert bounds.min.Z >= lm_bounds.min.Z - 0.02, name
        assert bounds.max.X <= lm_bounds.max.X + 0.02, name
        assert bounds.max.Y <= lm_bounds.max.Y + 0.02, name
        assert bounds.max.Z <= lm_bounds.max.Z + 0.02, name
    support_outside = support_tool - lm
    assert (0.0 if support_outside is None
            else support_outside.volume) < 0.03

    # Even the conservative native-wall witness must not reach the actual W22
    # STEP or service proxy; the production split adds no support material.
    w22_native = fit.load_w22_reference_step_native()
    w22_placed = fit.place_w22_reference_step(w22_native)
    assert _intersection_volume(support_tool, w22_placed) < 0.03
    assert _intersection_volume(
        support_tool, fit.w22_body_keepout(include_flange=True)) < 0.03

    state = "floor" if stand_foot else "no-floor"
    _assert_standalone_lm_joint_brep(
        lm, core, f"{state} canonical LM")
    _assert_standalone_lm_joint_brep(
        top, core, f"{state} optional keyed LM top")
    _assert_standalone_um_joint_brep(
        um, core, f"{state} canonical UM")
    _assert_lm_um_joint_stls(stand_foot, core)
    _assert_um_tweeter_joint_stls(stand_foot, core)
    _assert_lower_shoulder_magnet_split_ownership(
        lm, bottom, top, core, lm_split, state)

    # The optional top owns both upper LM ring stations.  Gate its released
    # BREP directly: the exterior must retain the same continuous cylindrical
    # skin as the canonical carrier, with neither a local outward boss nor a
    # pocket-shaped dent that could reveal the buried magnet position.
    for site in core.side_magnet_sites("lm"):
        if site["interface_kind"] != "ring":
            continue
        visible_face = site["outer_surface_face"]
        outside = core._axis_cylinder(
            visible_face, site["normal"], site["z_mm"],
            core.SIDE_EAR_D, 0.0, core.SIDE_EAR_OUT)
        assert _intersection_volume(top, outside) < 0.05, (
            f"{state} optional keyed LM top has a proud surface at "
            f"{site['name']}")

        annular_skin = _assert_no_visible_ring_magnet_cue(
            top, site, core, f"{state} optional keyed LM top")
        assert _intersection_volume(bottom, annular_skin) < 0.02, (
            f"{state} lower keyed LM piece unexpectedly owns the upper "
            f"ring skin at {site['name']}")

    # The only intentional source-volume loss is the two local female fit
    # reliefs not occupied by their male pins.  Compare scalar volume balance
    # against that independently constructed local relief; unlike a global
    # parent-minus-imported-child Boolean, this remains stable across BREP
    # serialization and still catches any missing bulk material.
    male_tool = lm_split.male_registration_key_tool()
    socket_tool = lm_split.female_registration_socket_tool()
    top_clip = core._plan_prism(
        box(-400.0, lm_split.LM_SPLIT_SEAM_Y, 400.0, 600.0),
        -200.0, 200.0)
    expected_relief = augmented_lm & socket_tool
    expected_relief = expected_relief & top_clip
    expected_relief = expected_relief - male_tool
    mass_deficit = augmented_lm.volume - bottom.volume - top.volume
    assert mass_deficit >= -0.03
    # The tall floor carrier's independently serialized Boolean result varies
    # by less than 1 mm3 over 309,000 mm3 (3.3 ppm); no-floor agrees to below
    # 0.001 mm3.  Keep the bound far below either pin's 5.83-mm3 volume.
    assert abs(mass_deficit - expected_relief.volume) < 1.0, (
        f"{state} keyed split mass deficit {mass_deficit:.6f} mm3 does not "
        f"match designed socket relief {expected_relief.volume:.6f} mm3")
    male_protrusions = male_tool & top_clip
    bottom_above_seam = bottom & top_clip
    unexpected_bottom_above = bottom_above_seam - male_protrusions
    assert (0.0 if unexpected_bottom_above is None
            else unexpected_bottom_above.volume) < 0.03
    assert _intersection_volume(bottom, male_tool) >= (
        male_tool.volume - 0.03)
    assert _intersection_volume(top, male_tool) < 0.03
    assert _intersection_volume(top, socket_tool) < 0.03
    male_outside_socket = male_protrusions - socket_tool
    assert (0.0 if male_outside_socket is None
            else male_outside_socket.volume) < 0.03

    # Because the split is derived after hollowing, every exact nominal
    # cutter remains unobstructed on both sides of the butt handoff. This is
    # the diameter/normal-section gate: no internal alignment sleeve is
    # permitted in the already tight D8.2 and D6 lumens.
    for cutter in route.route_inner_cutters("lm"):
        assert _intersection_volume(bottom, cutter) < 0.03
        assert _intersection_volume(top, cutter) < 0.03
        assert _intersection_volume(socket_tool, cutter) < 0.03
        assert _intersection_volume(support_tool, cutter) < 0.03
    # Independently generated nominal acoustic shells must also remain
    # untouched: a socket that misses the lumen but thins its cover is still
    # an unacceptable hidden resonance/leak risk.
    for route_name in ("UM", "T"):
        for shell in route.required_assembled_shell_components(route_name):
            assert _intersection_volume(socket_tool, shell) < 0.03, (
                f"LM registration socket cuts the {route_name} route shell")
    seam = lm_split.LM_SPLIT_SEAM_Y
    for label, points in (
            ("UM", route.route_cable_points(0.5)),
            ("T", route.ts_cable_points(0.5))):
        local = points[np.abs(points[:, 1] - seam) <= 5.0]
        assert len(local) >= 3, f"{label} has no sampled seam handoff"
        assert local[:, 1].min() < seam - lm_split.LM_SPLIT_GAP_MM / 2.0
        assert local[:, 1].max() > seam + lm_split.LM_SPLIT_GAP_MM / 2.0

    facts = lm_split.registration_fit_facts()
    assert facts["registration_pair_count"] == 2
    assert facts["registration_sides"] == ("left", "right")
    assert facts["registration_is_keyed"] is True
    assert facts["registration_form"] == (
        "two_symmetric_cylindrical_pins_round_plus_relief_sockets")
    assert facts["registration_axis_world_xyz"] == (0.0, 1.0, 0.0)
    assert facts["registration_axis_normal_to_horizontal_seam"] is True
    assert facts["registration_symmetry_error_mm"] < 1e-9
    assert 217.6 <= facts["registration_center_spacing_mm"] <= 217.8
    assert facts["assembly_motion"] == (
        "top_half_approaches_along_negative_world_y")
    assert facts["assembly_gap_mm"] == 0.0
    assert facts["buried_route_joint"] == "closed_zero_gap_planar_butt"
    assert facts["pin_diameter_mm"] == 1.60
    assert facts["pin_root_overlap_mm"] == 0.50
    assert facts["male_pin_length_mm"] == 2.90
    assert 11.66 <= facts["male_total_volume_mm3"] <= 11.67
    assert facts["engagement_depth_mm"] == 2.40
    assert math.isclose(
        facts["socket_round_diameter_mm"], 1.80, abs_tol=1e-12)
    assert facts["socket_radial_clearance_mm"] == 0.10
    assert facts["socket_end_clearance_mm"] == 0.25
    assert facts["socket_blind_depth_mm"] == 2.65
    assert facts["round_socket_side"] == "right"
    assert facts["relieved_socket_side"] == "left"
    assert facts["relieved_socket_x_extra_each_side_mm"] == 0.02
    assert math.isclose(
        facts["relieved_socket_x_span_mm"], 1.84, abs_tol=1e-12)
    assert math.isclose(
        facts["registered_round_diametral_play_mm"], 0.20,
        abs_tol=1e-12)
    assert math.isclose(
        facts["relative_pin_pitch_error_capacity_mm"], 0.22,
        abs_tol=1e-12)
    assert facts["round_socket_inner_wall_mm"] >= 0.38
    assert facts["round_socket_outer_wall_mm"] >= 0.38
    assert facts["relieved_socket_inner_wall_mm"] >= 0.38
    assert facts["relieved_socket_outer_wall_mm"] >= 0.38
    assert facts["minimum_socket_radial_wall_mm"] >= 0.38
    assert facts["socket_blind_end_wall_mm"] >= 0.50
    assert math.isclose(
        facts["support_land_length_mm"], 2.70, abs_tol=1e-12)
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(
                   facts["support_land_z_range_mm"], (13.02, 15.58)))
    assert facts["support_land_clearance_above_rear_mm"] >= 6.21
    assert facts["support_land_clearance_below_front_mm"] >= 2.71
    assert facts["support_land_driver_recess_plan_clearance_mm"] >= 0.0
    assert facts["support_land_driver_flange_plan_clearance_mm"] >= 0.079
    assert facts["exterior_support_land"] is False
    assert facts["registration_wall_source"] == "native_r113p8_smooth_ring"
    assert facts["support_land_outer_radius_mm"] <= 113.8
    assert 0.77 <= facts[
        "support_land_plan_growth_from_structural_ring_mm"] <= 0.80
    assert facts[
        "support_land_plan_growth_beyond_visible_fairing_mm"] == 0.0
    assert facts["support_land_plan_outline_growth_mm"] == 0.0
    assert facts["inward_support_land_rejected_for_driver_collision"] is False
    assert facts["two_round_socket_design_rejected"] is True
    assert facts["tolerance_strategy"] == (
        "right_round_locator_left_x_relief_round_and_diamond")
    assert "bind" in facts["binding_drawback"]
    assert facts["nominal_nozzle_diameter_mm"] == 0.40
    assert facts["pin_nominal_nozzle_width_count"] == 4.0
    assert facts["pin_and_socket_slicer_gate_required"] is True
    assert "no load credit" in facts["printability_drawback"]
    assert facts["envelope_growth_mm"] == 0.0
    assert facts["installed_structural_load_credit_n"] == 0.0
    assert facts["standalone_retention_credit_n"] == 0.0
    assert facts["physical_coupon_required"] is True
    assert facts["physical_load_qualification_required"] is True
    assert OBIWAN_OPTIONAL_LM_SPLIT_BED_MM == facts[
        "target_square_bed_mm"] == 220.0
    assert facts["floor_bottom_print_rotation_x_deg"] == 180.0
    assert facts["print_orientation"] == "front_face_down_all_pieces"
    assert facts["floor_bottom_in_bed_rotation_deg"] == 26.0

    footprints = {}
    for name, part in parts.items():
        oriented = (
            Rot(Z=BED_ROT_Z[f"obiwan_{name}"])
            * Rot(X=180.0) * part)
        size = oriented.bounding_box().size
        footprints[name] = (size.X, size.Y, size.Z)
        assert max(size.X, size.Y, size.Z) <= 220.0, (
            f"{name} footprint {size.X:.2f} x {size.Y:.2f} x "
            f"{size.Z:.2f} exceeds the 220-mm option contract")

    if stand_foot:
        import lx521_baffle.obiwan.floor as floor

        bottom_bounds = bottom.bounding_box()
        assert math.isclose(
            bottom_bounds.min.Y, floor.FLOOR_Y_MM, abs_tol=0.02)
        assert math.isclose(
            bottom_bounds.min.Z, floor.FOOT_REAR_Z_MM, abs_tol=0.02)
        assert math.isclose(
            bottom_bounds.max.Z, floor.FOOT_FRONT_Z_MM, abs_tol=0.02)
        assert top.bounding_box().min.Y > floor.STEM_TOP_Y_MM
    else:
        assert math.isclose(
            bottom.bounding_box().min.Y, 0.0, abs_tol=0.02), (
                "no-floor LM lower lost the universal Y=0 front tongue")
    print(
        f"  {state} optional LM keyed split: zero-gap route butt, "
        f"two concealed +Y pins with round+relieved sockets entirely inside "
        f"the native smooth ring; "
        f"220-mm footprints {footprints}")






def _intersection_volume(a, b):
    if (a is None or b is None or a.is_null or b.is_null
            or a.volume <= 1e-12 or b.volume <= 1e-12):
        return 0.0
    hit = a & b
    return 0.0 if hit is None else hit.volume


def _assert_no_visible_ring_magnet_cue(part, site, core, label):
    """Gate the exact cylinder complement and its retained outer skin."""
    assert site["interface_kind"] == "ring"
    visible_face = site["outer_surface_face"]
    visible_r = site["outer_surface_radius_mm"]
    half_width = core.SIDE_EAR_D / 2.0
    tangent_recession = (
        visible_r - math.sqrt(visible_r ** 2 - half_width ** 2))
    local_window = core._axis_cylinder(
        visible_face, site["normal"], site["z_mm"], core.SIDE_EAR_D,
        tangent_recession + 0.10, core.SIDE_EAR_OUT)
    ideal_cylinder = core._cylinder_at(
        *site["center"], visible_r,
        core.CORE_REAR_Z - core.SIDE_EAR_D,
        core.THICKNESS_MM + core.SIDE_EAR_D)
    radial_complement = local_window - ideal_cylinder
    overshoot = _intersection_volume(part, radial_complement)
    assert overshoot < 0.01, (
        f"{label} has {overshoot:.4f} mm3 outside the exact "
        f"R{visible_r:.2f} visible cylinder at {site['name']}")

    skin_depth = 0.10
    assert site["carrier_cavity_face_inset_mm"] >= skin_depth + 0.04
    outer = core._cylinder_at(
        *site["center"], visible_r, core.CORE_REAR_Z + 0.05,
        core.THICKNESS_MM - 0.05)
    inner = core._cylinder_at(
        *site["center"], visible_r - skin_depth,
        core.CORE_REAR_Z, core.THICKNESS_MM)
    annular_skin = outer - inner
    annular_skin &= core._axis_cylinder(
        visible_face, site["normal"], site["z_mm"],
        8.0, 1.0, 0.20)
    retained = _intersection_volume(part, annular_skin)
    assert retained > 0.995 * annular_skin.volume, (
        f"{label} reveals {site['name']}: {retained:.4f}/"
        f"{annular_skin.volume:.4f} mm3 of the continuous "
        "0.10-mm exterior skin remains")
    return annular_skin


def _joint_annulus_witness(core, x, bore_diameter, z0, z1):
    """A full-circle wall witness inside the complete functional boss."""
    outer = core._cylinder_at(
        x, core.JOINT_EAR_Y,
        core.JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.20, z0, z1)
    inner = core._cylinder_at(
        x, core.JOINT_EAR_Y, bore_diameter / 2.0 + 0.20,
        z0 - 0.10, z1 + 0.10)
    return outer - inner


def _assert_standalone_lm_joint_brep(lm, core, label):
    """LM alone owns two complete ears with rear-driven D3.4 bores."""
    for x in core.JOINT_EAR_X:
        complete_ear = core._complete_joint_ear("lm", x)
        clearance_bore = core._cylinder_at(
            x, core.JOINT_EAR_Y,
            core.JOINT_CLEARANCE_BORE_D / 2.0,
            core.CORE_REAR_Z - core.JOINT_BORE_REAR_OVERSHOOT,
            core.JOINT_CLEARANCE_BORE_TOP_Z)
        required_ear = complete_ear - clearance_bore
        retained = _intersection_volume(lm, required_ear)
        assert retained > 0.995 * required_ear.volume, (
            f"{label} x={x:g} retains only {retained:.3f}/"
            f"{required_ear.volume:.3f} mm3 of its complete standalone ear")
        obstruction = _intersection_volume(lm, clearance_bore)
        assert obstruction < 0.02, (
            f"{label} x={x:g} obstructs its D3.4 clearance bore by "
            f"{obstruction:.3f} mm3")

        wall = _joint_annulus_witness(
            core, x, core.JOINT_CLEARANCE_BORE_D,
            core.LM_JOINT_Z[0] + 0.20,
            core.LM_JOINT_Z[1] - 0.20)
        wall_fill = _intersection_volume(lm, wall)
        assert wall_fill > 0.995 * wall.volume, (
            f"{label} x={x:g} lacks a complete 360-degree D3.4 bore wall: "
            f"{wall_fill:.3f}/{wall.volume:.3f} mm3")

        opposing_ear = core._complete_joint_ear("um", x)
        cross_owner = _intersection_volume(lm, opposing_ear)
        assert cross_owner < 0.02, (
            f"{label} x={x:g} contains {cross_owner:.3f} mm3 of UM-owned "
            "ear material")


def _assert_standalone_um_joint_brep(um, core, label):
    """UM alone owns two complete ears with rear-open D4.6 receivers."""
    for x in core.JOINT_EAR_X:
        complete_ear = core._complete_joint_ear("um", x)
        insert_receiver = core._cylinder_at(
            x, core.JOINT_EAR_Y, core.JOINT_INSERT_BORE_D / 2.0,
            *core.JOINT_INSERT_BORE_Z)
        required_ear = complete_ear - insert_receiver
        retained = _intersection_volume(um, required_ear)
        assert retained > 0.995 * required_ear.volume, (
            f"{label} x={x:g} retains only {retained:.3f}/"
            f"{required_ear.volume:.3f} mm3 of its complete standalone ear")
        obstruction = _intersection_volume(um, insert_receiver)
        assert obstruction < 0.02, (
            f"{label} x={x:g} obstructs its rear-open D4.6 x 4.0 receiver "
            f"by {obstruction:.3f} mm3")

        wall = _joint_annulus_witness(
            core, x, core.JOINT_INSERT_BORE_D,
            core.UM_JOINT_Z[0] + 0.20,
            core.JOINT_INSERT_BORE_Z[1] - 0.20)
        wall_fill = _intersection_volume(um, wall)
        assert wall_fill > 0.995 * wall.volume, (
            f"{label} x={x:g} lacks a complete 360-degree D4.6 receiver "
            f"wall: {wall_fill:.3f}/{wall.volume:.3f} mm3")

        floor_z0 = core.JOINT_INSERT_BORE_Z[1] + 0.15
        floor_z1 = core.THICKNESS_MM - 0.15
        floor = core._cylinder_at(
            x, core.JOINT_EAR_Y,
            core.JOINT_INSERT_BORE_D / 2.0 - 0.25,
            floor_z0, floor_z1)
        floor_fill = _intersection_volume(um, floor)
        assert floor_fill > 0.98 * floor.volume, (
            f"{label} x={x:g} lost its 1.9-mm front receiver floor: "
            f"{floor_fill:.3f}/{floor.volume:.3f} mm3")

        opposing_ear = core._complete_joint_ear("lm", x)
        cross_owner = _intersection_volume(um, opposing_ear)
        assert cross_owner < 0.02, (
            f"{label} x={x:g} contains {cross_owner:.3f} mm3 of LM-owned "
            "ear material")


def _tweeter_joint_annulus_witness(
        core, x, bore_diameter, z0, z1):
    """Full-circle wall witness inside one complete D9.8 T boss."""
    outer = core._cylinder_at(
        x, core.TWEETER_JOINT_Y,
        core.TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.20,
        z0, z1)
    inner = core._cylinder_at(
        x, core.TWEETER_JOINT_Y, bore_diameter / 2.0 + 0.20,
        z0 - 0.10, z1 + 0.10)
    return outer - inner


def _assert_standalone_um_tweeter_joint_brep(
        um, tweeter, core, label):
    """Each T--UM print owns a complete functional ear and usable bore."""
    assert math.isclose(
        core.TWEETER_ADDON_JOINT_Z[0]
        - core.TWEETER_CORE_JOINT_Z[1], 0.20, abs_tol=1e-12)
    assert math.isclose(
        core.TWEETER_CORE_BORE_TOP_Z
        - core.TWEETER_JOINT_INSERT_BORE_Z[0],
        0.35, abs_tol=1e-12)
    assert math.isclose(
        core.TWEETER_JOINT_INSERT_BORE_Z[1]
        - core.TWEETER_ADDON_JOINT_Z[0],
        4.0, abs_tol=1e-12)
    assert math.isclose(
        core.TWEETER_JOINT_INSERT_FRONT_FLOOR_MM,
        1.9, abs_tol=1e-12)

    for x in core.TWEETER_JOINT_X:
        core_ear = core._complete_tweeter_joint_ear("um", x)
        core_bore = core._cylinder_at(
            x, core.TWEETER_JOINT_Y,
            core.TWEETER_JOINT_HOLE_D / 2.0,
            core.TWEETER_CORE_JOINT_Z[0] - 0.2,
            core.TWEETER_CORE_BORE_TOP_Z)
        core_required = core_ear - core_bore
        retained = _intersection_volume(um, core_required)
        assert retained > 0.995 * core_required.volume, (
            f"{label} UM x={x:g} retains only {retained:.3f}/"
            f"{core_required.volume:.3f} mm3 of its complete T ear")
        obstruction = _intersection_volume(um, core_bore)
        assert obstruction < 0.02, (
            f"{label} UM x={x:g} obstructs its D3.4 bore by "
            f"{obstruction:.3f} mm3")
        core_wall = _tweeter_joint_annulus_witness(
            core, x, core.TWEETER_JOINT_HOLE_D,
            core.TWEETER_CORE_JOINT_Z[0] + 0.20,
            core.TWEETER_CORE_JOINT_Z[1] - 0.20)
        core_wall_fill = _intersection_volume(um, core_wall)
        assert core_wall_fill > 0.995 * core_wall.volume, (
            f"{label} UM x={x:g} lacks a complete 360-degree D3.4 wall: "
            f"{core_wall_fill:.3f}/{core_wall.volume:.3f} mm3")
        cross_owner = _intersection_volume(
            um, core._complete_tweeter_joint_ear("tweeter", x))
        assert cross_owner < 0.02, (
            f"{label} UM x={x:g} contains {cross_owner:.3f} mm3 of "
            "crescent-owned T ear material")

        addon_ear = core._complete_tweeter_joint_ear("tweeter", x)
        receiver = core._cylinder_at(
            x, core.TWEETER_JOINT_Y,
            core.TWEETER_JOINT_INSERT_BORE_D / 2.0,
            *core.TWEETER_JOINT_INSERT_BORE_Z)
        addon_required = addon_ear - receiver
        retained = _intersection_volume(tweeter, addon_required)
        assert retained > 0.995 * addon_required.volume, (
            f"{label} crescent x={x:g} retains only {retained:.3f}/"
            f"{addon_required.volume:.3f} mm3 of its complete T ear")
        obstruction = _intersection_volume(tweeter, receiver)
        assert obstruction < 0.02, (
            f"{label} crescent x={x:g} obstructs its D4.6 x 4.0 receiver "
            f"by {obstruction:.3f} mm3")
        addon_wall = _tweeter_joint_annulus_witness(
            core, x, core.TWEETER_JOINT_INSERT_BORE_D,
            core.TWEETER_ADDON_JOINT_Z[0] + 0.20,
            core.TWEETER_JOINT_INSERT_BORE_Z[1] - 0.20)
        addon_wall_fill = _intersection_volume(tweeter, addon_wall)
        assert addon_wall_fill > 0.995 * addon_wall.volume, (
            f"{label} crescent x={x:g} lacks a complete 360-degree D4.6 "
            f"receiver wall: {addon_wall_fill:.3f}/"
            f"{addon_wall.volume:.3f} mm3")
        floor = core._cylinder_at(
            x, core.TWEETER_JOINT_Y, 2.05,
            core.TWEETER_JOINT_INSERT_BORE_Z[1] + 0.15,
            core.THICKNESS_MM - 0.15)
        floor_fill = _intersection_volume(tweeter, floor)
        assert floor_fill > 0.995 * floor.volume, (
            f"{label} crescent x={x:g} lost its 1.9-mm front floor: "
            f"{floor_fill:.3f}/{floor.volume:.3f} mm3")
        cross_owner = _intersection_volume(
            tweeter, core._complete_tweeter_joint_ear("um", x))
        assert cross_owner < 0.02, (
            f"{label} crescent x={x:g} contains {cross_owner:.3f} mm3 of "
            "UM-owned T ear material")


def _stl_world_point_membership(stl_path, world_points):
    """Classify installed-frame points against one hash-bound release STL."""
    from lx521_baffle.print_contract import (
        validate_front_down_transform,
        validate_print_sidecar,
    )
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkPolyData
    from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints
    from vtkmodules.vtkIOGeometry import vtkSTLReader

    payload = validate_print_sidecar(stl_path)
    matrix = np.asarray(validate_front_down_transform(payload), dtype=float)
    world = np.asarray(world_points, dtype=float)
    assert world.ndim == 2 and world.shape[1] == 3
    homogeneous = np.column_stack((world, np.ones(len(world))))
    transformed = (homogeneous @ matrix.T)[:, :3]

    points = vtkPoints()
    points.SetDataTypeToDouble()
    for point in transformed:
        points.InsertNextPoint(*(float(value) for value in point))
    cloud = vtkPolyData()
    cloud.SetPoints(points)

    reader = vtkSTLReader()
    reader.SetFileName(str(stl_path))
    reader.Update()
    surface = reader.GetOutput()
    assert surface.GetNumberOfCells() > 0, f"empty release STL: {stl_path}"

    enclosed = vtkSelectEnclosedPoints()
    enclosed.SetInputData(cloud)
    enclosed.SetSurfaceData(surface)
    enclosed.SetTolerance(1.0e-6)
    enclosed.Update()
    return tuple(bool(enclosed.IsInside(index)) for index in range(len(world)))


def _assert_stl_point_groups(
        stl_path, point_groups, *, required_groups, forbidden_groups):
    names = tuple(dict.fromkeys((*required_groups, *forbidden_groups)))
    points = []
    ranges = {}
    for name in names:
        start = len(points)
        points.extend(point_groups[name])
        ranges[name] = range(start, len(points))
    membership = _stl_world_point_membership(stl_path, points)
    for name in required_groups:
        missing = [
            (index - ranges[name].start, points[index])
            for index in ranges[name] if not membership[index]
        ]
        assert not missing, (
            f"{stl_path.name} misses required {name} material at "
            f"{missing[:4]}")
    for name in forbidden_groups:
        present = [
            (index - ranges[name].start, points[index])
            for index in ranges[name] if membership[index]
        ]
        assert not present, (
            f"{stl_path.name} contains forbidden {name} material at "
            f"{present[:4]}")


def _assert_stl_ring_surfaces(stl_path, sites, core):
    """Audit the hash-bound release mesh at every buried ring magnet."""
    from bambu_3mf_audit import read_stl_triangles
    from lx521_baffle.print_contract import (
        validate_front_down_transform,
        validate_print_sidecar,
    )

    payload = validate_print_sidecar(stl_path)
    source_to_stl = np.asarray(
        validate_front_down_transform(payload), dtype=float)
    facets = np.asarray(read_stl_triangles(stl_path), dtype=float)
    flat = facets.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    source = (
        homogeneous @ np.linalg.inv(source_to_stl).T)[:, :3]

    probes = []
    fairing_indices = []
    outer_indices = []
    for site in sites:
        center = np.asarray(site["center"], dtype=float)
        normal = np.asarray(site["normal"], dtype=float)
        tangent = np.asarray((-normal[1], normal[0]), dtype=float)
        rel = source[:, :2] - center
        radius = np.linalg.norm(rel, axis=1)
        visible_r = site["outer_surface_radius_mm"]
        half_width = core.SIDE_EAR_D / 2.0
        recession = visible_r - math.sqrt(
            visible_r ** 2 - half_width ** 2)
        local_vertices = (
            (np.abs(rel @ tangent) <= half_width + 0.10)
            & (rel @ normal >= visible_r - recession - 0.10))
        assert np.any(local_vertices), (
            f"{stl_path.name} has no exterior facets at {site['name']}")
        excess = radius[local_vertices] - visible_r
        assert not np.any(excess > 0.02), (
            f"{stl_path.name} has a facet vertex "
            f"{float(excess.max()):.4f} mm outside R{visible_r:.2f} "
            f"at {site['name']}")

        angle0 = math.atan2(normal[1], normal[0])

        # STL cylinders are faceted inward by their chord tolerance.  Do not
        # misclassify that ordinary tessellation as a missing 0.10-mm skin.
        # The exact BREP gate above proves the skin itself; these deeper corner
        # witnesses prove that the exported mesh retains the continuous
        # R113.8/R52.5 fairing around (and not a local pad behind) the cavity.
        # Their tangent/Z distance is safely outside the D5.20 cavity aperture.
        for arc_mm in (-3.4, 3.4):
            angle = angle0 + arc_mm / visible_r
            for dz in (-2.7, 2.7):
                r = visible_r - 0.35
                fairing_indices.append(len(probes))
                probes.append((
                    center[0] + r * math.cos(angle),
                    center[1] + r * math.sin(angle),
                    site["z_mm"] + dz))

        # A tangent station-local flat can be radially proud away from its
        # centre even when the centre itself lies on the nominal cylinder.
        # Probe that exterior and also retain the stronger facet-vertex radial
        # bound above, which catches the full local D7.8 pad width.
        for arc_mm in (-3.6, -1.8, 0.0, 1.8, 3.6):
            angle = angle0 + arc_mm / visible_r
            for dz in (-2.7, 0.0, 2.7):
                r = visible_r + 0.05
                outer_indices.append(len(probes))
                probes.append((
                    center[0] + r * math.cos(angle),
                    center[1] + r * math.sin(angle),
                    site["z_mm"] + dz))
    membership = _stl_world_point_membership(stl_path, probes)
    assert all(membership[index] for index in fairing_indices), (
        f"{stl_path.name} lacks its continuous visible-ring fairing")
    assert not any(membership[index] for index in outer_indices), (
        f"{stl_path.name} has material outside its visible ring")


def _assert_lm_um_joint_stls(stand_foot, core):
    """Semantic point gate on each independently printable release mesh."""
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
    lm_wall_radius = (
        core.JOINT_CLEARANCE_BORE_D + core.JOINT_FUNCTIONAL_BOSS_D) / 4.0
    um_wall_radius = (
        core.JOINT_INSERT_BORE_D + core.JOINT_FUNCTIONAL_BOSS_D) / 4.0

    def rings(radius, z):
        return tuple(
            (x + radius * math.cos(angle),
             core.JOINT_EAR_Y + radius * math.sin(angle), z)
            for x in core.JOINT_EAR_X for angle in angles)

    def bore_points(radius, z_values):
        offsets = (
            (0.0, 0.0),
            (radius, 0.0), (-radius, 0.0),
            (0.0, radius), (0.0, -radius),
        )
        return tuple(
            (x + dx, core.JOINT_EAR_Y + dy, z)
            for x in core.JOINT_EAR_X for z in z_values
            for dx, dy in offsets)

    groups = {
        "LM 360-degree wall": rings(lm_wall_radius, 9.50),
        "UM 360-degree wall": rings(um_wall_radius, 14.40),
        "LM clearance bore": bore_points(1.45, (7.20, 9.50, 12.00)),
        "UM insert receiver": bore_points(
            2.00, (12.65, 14.40, 16.15)),
        "UM receiver front floor": bore_points(2.00, (17.25,)),
    }
    state_dir = "floor_stand" if stand_foot else "no_floor_stand"
    stl_dir = PROJECT_ROOT / "build" / state_dir / "stl"
    lm_paths = (
        stl_dir / "obiwan_core_1_of_2_lm_carrier.stl",
        stl_dir / "obiwan_optional_lm_keyed_2_of_2_top.stl",
    )
    for stl_path in lm_paths:
        _assert_stl_point_groups(
            stl_path, groups,
            required_groups=("LM 360-degree wall",),
            forbidden_groups=(
                "LM clearance bore", "UM 360-degree wall"))
    lm_ring_sites = tuple(
        site for site in core.side_magnet_sites("lm")
        if site["interface_kind"] == "ring")
    for stl_path in lm_paths:
        _assert_stl_ring_surfaces(stl_path, lm_ring_sites, core)
    um_path = stl_dir / "obiwan_core_2_of_2_um_carrier.stl"
    _assert_stl_point_groups(
        um_path, groups,
        required_groups=(
            "UM 360-degree wall", "UM receiver front floor"),
        forbidden_groups=(
            "UM insert receiver", "LM 360-degree wall"))
    _assert_stl_ring_surfaces(
        um_path, tuple(core.side_magnet_sites("um")), core)
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state} release STLs: complete standalone LM/keyed D3.4 ears; "
        "complete UM D4.6 x 4.0 blind receivers with 1.9-mm front floors")


def _assert_um_tweeter_joint_stls(stand_foot, core):
    """Semantic 360-degree wall/floor gate on both T--UM release meshes."""
    angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)

    def rings(radius, z):
        return tuple(
            (x + radius * math.cos(angle),
             core.TWEETER_JOINT_Y + radius * math.sin(angle), z)
            for x in core.TWEETER_JOINT_X for angle in angles)

    def disk_points(radii, z):
        points = [
            (x, core.TWEETER_JOINT_Y, z)
            for x in core.TWEETER_JOINT_X
        ]
        points.extend(
            (x + radius * math.cos(angle),
             core.TWEETER_JOINT_Y + radius * math.sin(angle), z)
            for x in core.TWEETER_JOINT_X for radius in radii
            for angle in angles)
        return tuple(points)

    def bore_points(radius, z_values):
        offsets = (
            (0.0, 0.0),
            (radius, 0.0), (-radius, 0.0),
            (0.0, radius), (0.0, -radius),
        )
        return tuple(
            (x + dx, core.TWEETER_JOINT_Y + dy, z)
            for x in core.TWEETER_JOINT_X for z in z_values
            for dx, dy in offsets)

    groups = {
        "UM T-joint 360-degree wall": rings(3.30, 9.50),
        "UM T-joint clearance bore": bore_points(
            1.45, (7.20, 9.50, 12.00)),
        "crescent 360-degree receiver wall": rings(3.40, 14.40),
        "crescent insert receiver": bore_points(
            2.00, (12.65, 14.40, 16.15)),
        "crescent 1.9-mm receiver floor": disk_points(
            (1.20, 2.00), 17.25),
    }
    state_dir = "floor_stand" if stand_foot else "no_floor_stand"
    stl_dir = PROJECT_ROOT / "build" / state_dir / "stl"
    um_path = stl_dir / "obiwan_core_2_of_2_um_carrier.stl"
    tweeter_path = (
        stl_dir / "obiwan_addon_tweeter_crescent.stl")
    _assert_stl_point_groups(
        um_path, groups,
        required_groups=("UM T-joint 360-degree wall",),
        forbidden_groups=(
            "UM T-joint clearance bore",
            "crescent 360-degree receiver wall"))
    _assert_stl_point_groups(
        tweeter_path, groups,
        required_groups=(
            "crescent 360-degree receiver wall",
            "crescent 1.9-mm receiver floor"),
        forbidden_groups=(
            "crescent insert receiver",
            "UM T-joint 360-degree wall"))
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state} release STLs: complete standalone UM D3.4 T ears; "
        "complete crescent D4.6 x 4.0 receivers with 1.9-mm front floors")


def _complete_tweeter_joint_witnesses(core, x):
    """Independent complete-ear authority for the T--UM service joint."""
    core_ear = core._complete_tweeter_joint_ear("um", x)
    addon_ear = core._complete_tweeter_joint_ear("tweeter", x)
    core_bolt = core._cylinder_at(
        x, core.TWEETER_JOINT_Y,
        core.TWEETER_JOINT_HOLE_D / 2.0,
        core.TWEETER_CORE_JOINT_Z[0] - 0.2,
        core.TWEETER_CORE_BORE_TOP_Z)
    insert = core._cylinder_at(
        x, core.TWEETER_JOINT_Y,
        core.TWEETER_JOINT_INSERT_BORE_D / 2.0,
        *core.TWEETER_JOINT_INSERT_BORE_Z)
    return {
        "core_ear": core_ear,
        "addon_ear": addon_ear,
        "core_required": core_ear - core_bolt,
        "addon_required": addon_ear - insert,
        "core_bolt": core_bolt,
        "insert": insert,
    }


def _assert_core_interface_breps(lm, um, core):
    import lx521_baffle.obiwan.route as route
    import lx521_baffle.obiwan.bridge as bridge
    from lx521_baffle.magnets import DEFAULT_SPEC, wall_cavity_tools
    from shapely.geometry import Point

    assert math.isclose(
        core.UM_JOINT_Z[0] - core.LM_JOINT_Z[1],
        0.20, abs_tol=1e-12)
    assert math.isclose(
        core.JOINT_CLEARANCE_BORE_TOP_Z - core.JOINT_INSERT_BORE_Z[0],
        0.35, abs_tol=1e-12)
    assert math.isclose(
        core.JOINT_INSERT_BORE_Z[1] - core.UM_JOINT_Z[0],
        4.0, abs_tol=1e-12)
    assert math.isclose(
        core.UM_JOINT_Z[0] - core.JOINT_INSERT_BORE_Z[0],
        0.20, abs_tol=1e-12)
    assert math.isclose(
        core.JOINT_INSERT_FRONT_FLOOR_MM, 1.9, abs_tol=1e-12)
    _assert_standalone_lm_joint_brep(lm, core, "canonical LM")
    _assert_standalone_um_joint_brep(um, core, "canonical UM")

    owner = {"lm": lm, "um": um}
    route_covers = {
        driver: tuple(core.route_outer_covers(driver))
        for driver in owner
    }
    assert len(route_covers["lm"]) == 2
    assert len(route_covers["um"]) == 1
    for site in core.side_magnet_sites():
        nx, ny = site["normal"]
        face_offset = site["face_offset_mm"]
        expected_offset = (
            -0.15 if site["interface_kind"] == "shoulder" else 0.65)
        assert math.isclose(face_offset, expected_offset, abs_tol=1e-12)
        assert site["magnet_fully_buried"]
        assert not site["proud_ear_added"]
        assert math.isclose(
            site["local_captive_backing_boss_mm"], 0.0,
            abs_tol=1e-12)
        if site["interface_kind"] == "ring":
            assert math.isclose(
                math.dist(site["face"], site["center"]),
                site["radius"] + expected_offset, abs_tol=1e-9)
            assert math.isclose(
                site["continuous_flush_ring_fairing_mm"], 0.80,
                abs_tol=1e-12)
            assert math.isclose(
                site["carrier_cavity_face_inset_mm"], 0.15,
                abs_tol=1e-12)
            assert math.isclose(
                math.dist(site["outer_surface_face"], site["center"]),
                site["outer_surface_radius_mm"], abs_tol=1e-9)
        else:
            assert site["interface_kind"] == "shoulder"
            assert math.isclose(
                math.dist(site["face"], site["outer_surface_face"]),
                0.15, abs_tol=1e-12)
            assert math.isclose(
                math.dist(site["outer_surface_face"], site["center"]),
                site["radius"], abs_tol=1e-9)

        tools = wall_cavity_tools(
            name=site["name"], face=site["face"],
            outward=(nx, ny, 0.0), owner="carrier",
            axis_z=site["z_mm"], print_up=(0.0, 0.0, -1.0),
            front_z=core.THICKNESS_MM,
            interface_gap_mm=core.SIDE_INTERFACE_GAP)
        assert tools.closure_kind == "transverse_gable_45deg"
        assert tools.spec.cavity_diameter_mm == 5.20
        assert tools.spec.cavity_depth_mm == 2.10
        assert tools.spec.face_skin_mm == tools.spec.inner_skin_mm == 0.45
        assert math.isclose(
            tools.spec.captive_land_mm, 3.00, abs_tol=1e-12)
        assert tools.spec.roof_angle_deg == 45.0

        # Nothing may project beyond the visible carrier surface. Ring sites
        # use one continuous cylindrical fairing, never a local flat boss.
        # The lower sites live on a curved cubic shoulder, so a tangent-plane
        # cylinder is not its exterior: adjacent native shoulder material
        # legitimately crosses that plane. Gate those sites against the exact
        # shared shoulder plan in a local three-dimensional window instead.
        visible_face = site.get("outer_surface_face", site["face"])
        if site["interface_kind"] == "shoulder":
            window_radius = core.SIDE_EAR_D / 2.0 + core.SIDE_EAR_OUT
            local_plan = Point(*visible_face).buffer(
                window_radius, resolution=96)
            outside_plan = local_plan.difference(
                bridge.common_lm_wing_contact_plan()).buffer(0)
            outside = core._plan_prism(
                outside_plan,
                site["z_mm"] - core.SIDE_EAR_D / 2.0,
                site["z_mm"] + core.SIDE_EAR_D / 2.0)
        else:
            outside = core._axis_cylinder(
                visible_face, site["normal"], site["z_mm"],
                core.SIDE_EAR_D, 0.0, core.SIDE_EAR_OUT)
        assert _intersection_volume(
            owner[site["driver"]], outside) < 0.05, (
                f"{site['name']} projects beyond its sealed interface")

        if site["interface_kind"] == "ring":
            _assert_no_visible_ring_magnet_cue(
                owner[site["driver"]], site, core,
                f"canonical {site['driver'].upper()} carrier")

        for index, cutter in enumerate(tools.cutters):
            overlap = _intersection_volume(owner[site["driver"]], cutter)
            assert overlap < 0.03, (
                f"{site['name']} captive cutter {index} is obstructed by "
                f"{overlap:.4f} mm3")
        assert _intersection_volume(
            owner[site["driver"]], tools.nominal_magnet) < 0.02

        # The exact helper land minus its three functional voids must remain
        # solid.  This simultaneously gates the circular retaining cradle,
        # the two printable side walls, both 0.45-mm axial skins, the gable
        # roof, and the post-roof sealing layer.
        qualified_solid = tools.required_land
        for cutter in tools.cutters:
            qualified_solid = qualified_solid - cutter
        retained = _intersection_volume(
            owner[site["driver"]], qualified_solid)
        assert retained > 0.98 * qualified_solid.volume, (
            f"{site['name']} retains only {retained:.3f}/"
            f"{qualified_solid.volume:.3f} mm3 of captive land")

        # Independent axial probes make accidental external access or loss
        # of the inner backstop fail even if helper/production code changed
        # together.  Probe inside the D5 magnet projection, away from the
        # circular edge tolerance.
        skin_diameter = core.SIDE_MAGNET_D - 0.4
        face_skin = core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"], skin_diameter,
            DEFAULT_SPEC.face_skin_mm - 0.03, 0.0)
        inner_face = (
            site["face"][0]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * nx,
            site["face"][1]
            - (DEFAULT_SPEC.face_skin_mm
               + DEFAULT_SPEC.cavity_depth_mm) * ny,
        )
        inner_skin = core._axis_cylinder(
            inner_face, site["normal"], site["z_mm"], skin_diameter,
            DEFAULT_SPEC.inner_skin_mm - 0.03, 0.0)
        for label, skin in (("interface", face_skin),
                            ("inner", inner_skin)):
            fill = _intersection_volume(owner[site["driver"]], skin)
            assert fill > 0.97 * skin.volume, (
                f"{site['name']} lost sealed {label} skin: "
                f"{fill:.4f}/{skin.volume:.4f} mm3")

        cavity_void = tools.cutters[0]
        for cutter in tools.cutters[1:]:
            cavity_void = cavity_void.fuse(cutter)
        if site["driver"] == "um":
            route_lumens = (route._round_tube(
                route._owner_cutter_points(
                    route.ts_cable_points(1.8), "um"),
                route.TS_CUTTER_R),)
            required_route_gap = route.TS_SIDE_WALL + 0.8
        else:
            route_lumens = (
                route._round_tube(
                    route._owner_cutter_points(
                        route.route_cable_points(1.8), "lm"),
                    route.CUTTER_R),
                route._round_tube(
                    route._owner_cutter_points(
                        route.ts_cable_points(1.8), "lm"),
                    route.TS_CUTTER_R),
            )
            required_route_gap = route.TUNNEL_SKIN
        for route_index, lumen in enumerate(route_lumens):
            lumen_gap = cavity_void.distance_to(lumen)
            assert lumen_gap >= required_route_gap - 0.03, (
                f"{site['name']} cavity-to-route-{route_index} gap "
                f"{lumen_gap:.3f} < {required_route_gap:.3f} mm")


def _wait_for_worker_headroom(label, minimum_mb=2500.0):
    """Require launch headroom only when host-free monitoring is enabled."""
    import run_memory_guarded as memory_guard

    if memory_guard.MIN_FREE_MB == 0:
        return
    deadline = time.monotonic() + R6F_HEADROOM_WAIT_TIMEOUT_S
    while True:
        free_mb = memory_guard._free_memory_mib()
        assert free_mb is not None, (
            f"cannot measure memory before isolated {label} build")
        if free_mb >= minimum_mb:
            return
        assert time.monotonic() < deadline, (
            f"only {free_mb:.0f} MiB immediately reclaimable; refusing "
            f"to start isolated {label} CAD worker")
        time.sleep(0.5)


def _cad_worker_command(script):
    """Keep nested workers in a live guard or establish a fresh one.

    Most checks deliberately run inside one outer guarded process. The long
    terminal-service matrix is different: dozens of short OCC imports can
    leave file-cache pages inactive for the lifetime of that outer session,
    exhausting the conservative free+speculative+purgeable counter even
    though every individual worker is small. Its lightweight orchestrator
    therefore stays outside OCC and gives every substep a fresh authenticated
    guard lifecycle. Local service work remains serial; osado admits separate
    matrices through its bounded guard slots.
    """
    import run_memory_guarded as memory_guard

    command = [sys.executable, str(script)]
    if memory_guard.is_guarded_process():
        return command
    guard = PROJECT_ROOT / "scripts/run_memory_guarded.py"
    return [sys.executable, str(guard), "--", *command]


def _stage_shell_contract_breps_unlocked(
        stand_foot, route_name, directory,
        shell_keys=("nominal", "lower", "upper"), *, seed_targets):
    """Build only route-shell chunks in isolated descendants.

    Carrier and tweeter BREPs come exclusively from the Make-owned validated
    stage supplied in ``seed_targets``.  The private content-addressed cache
    remains useful for these test-only shell chunks, which are not release
    artifacts.  Native BREP keeps the validation handoff lossless in both
    profiles, and every descendant remains inside the active memory guard.
    """
    script_path = Path(__file__).resolve()
    script = str(script_path)
    digest = hashlib.sha256()
    digest.update(
        f"lx-obiwan-r6f-native-stage-v{R6F_NATIVE_STAGE_SCHEMA_VERSION}"
        .encode("ascii"))
    digest.update(script_path.read_bytes())
    digest.update(sys.version.encode("utf-8"))
    digest.update(b"floor" if stand_foot else b"no_floor")
    from importlib.metadata import PackageNotFoundError, version
    for distribution in (
            "build123d", "cadquery-ocp", "numpy", "shapely"):
        try:
            package_version = version(distribution)
        except PackageNotFoundError:
            package_version = "missing"
        digest.update(distribution.encode("utf-8"))
        digest.update(package_version.encode("utf-8"))
    source_paths = sorted((PROJECT_ROOT / "src/lx521_baffle").rglob("*.py"))
    source_paths.extend(sorted((PROJECT_ROOT / "scripts").glob("*.py")))
    for source_path in source_paths:
        digest.update(source_path.relative_to(PROJECT_ROOT).as_posix().encode(
            "utf-8"))
        digest.update(source_path.read_bytes())
    state_name = "floor" if stand_foot else "no_floor"
    large_host = _large_host_execution()
    digest.update(b"\0full-host" if large_host else b"\0segmented-host")
    cache_root = (Path(tempfile.gettempdir()) / "lx-obiwan-r6f-brep-cache"
                  / digest.hexdigest()[:24] / state_name)
    shell_cache = cache_root / (
        f"shell_{route_name}_full" if large_host
        else f"shell_{route_name}_segmented")
    shell_cache.mkdir(parents=True, exist_ok=True)

    def cached_shell_chunks(manifest):
        if not manifest.is_file() or manifest.stat().st_size == 0:
            return ()
        chunks = tuple(Path(line) for line in manifest.read_text(
            encoding="utf-8").splitlines() if line)
        if not chunks or not all(
                path.is_file() and path.stat().st_size > 0
                for path in chunks):
            return ()
        return chunks

    targets = dict(seed_targets)
    assert {"lm", "um", "tweeter"} <= set(targets)
    shell_segment_count = (
        1 if large_host else {"LM": 2, "UM": 8, "T": 12}[route_name])
    shell_jobs = {}
    for shell_key, wall_text in (
            ("nominal", "nominal"),
            ("lower", "0.76"),
            ("upper", "0.90")):
        entries = []
        for index in range(shell_segment_count):
            updates = {
                "LX_R6F_EXPORT_SHELL": route_name,
                "LX_R6F_EXPORT_SHELL_WALL": wall_text,
            }
            if not large_host:
                updates.update({
                    "LX_R6F_EXPORT_SHELL_SEGMENT": str(index),
                    "LX_R6F_EXPORT_SHELL_SEGMENT_COUNT": str(
                        shell_segment_count),
                })
            label = (f"shell_{shell_key}_full" if large_host
                     else f"shell_{shell_key}_seg_{index:02d}")
            entries.append((label, updates))
        shell_jobs[shell_key] = tuple(entries)
    assert set(shell_keys) <= set(shell_jobs)
    jobs = [job for key in shell_keys for job in shell_jobs[key]]
    for label, updates in jobs:
        assert "LX_R6F_EXPORT_SHELL" in updates
        suffix = ".manifest"
        target = shell_cache / f"{label}{suffix}"
        chunks = cached_shell_chunks(target)
        if chunks:
            targets[label] = chunks
            print(f"isolated {label}: reused hash-keyed native BREP cache",
                  flush=True)
            continue

        # Shell workers export each already-disjoint required solid directly
        # (no compound re-intersection). The selected profile guard
        # continuously samples the complete process tree. This admission
        # threshold matters on the local profile and is trivially satisfied
        # behind osado's mandatory host-free floor.
        launch_headroom = 3200.0
        _wait_for_worker_headroom(label, launch_headroom)
        temporary = target.with_name(
            f".{target.stem}.{os.getpid()}.{time.time_ns()}.partial{suffix}")
        env = os.environ.copy()
        env.pop("LX_R6F_CASE_ID", None)
        env.update(updates)
        env["LX_R6F_EXPORT_PATH"] = str(temporary)
        env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
        proc = subprocess.run(
            _cad_worker_command(script), env=env, text=True,
            capture_output=True)
        assert proc.returncode == 0, (
            f"isolated {label} build failed\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        assert temporary.is_file() and temporary.stat().st_size > 0
        chunks = tuple(Path(line) for line in temporary.read_text(
            encoding="utf-8").splitlines() if line)
        assert chunks and all(
            path.is_file() and path.stat().st_size > 0
            for path in chunks)
        temporary.replace(target)
        targets[label] = chunks
        print(proc.stdout, end="", flush=True)
    for shell_key in shell_keys:
        chunks = []
        for label, _updates in shell_jobs[shell_key]:
            chunks.extend(targets[label])
        targets[f"shell_{shell_key}"] = tuple(chunks)
    return targets


def _stage_shell_contract_breps(
        stand_foot, route_name, directory,
        shell_keys=("nominal", "lower", "upper")):
    """Seed from the validated release stage, then cache test-only shells."""
    paths = _validated_obiwan_stage_paths(stand_foot)
    carriers = {
        "lm": paths["core_lm_carrier"],
        "um": paths["core_um_carrier"],
        "tweeter": paths["addon_tweeter_crescent"],
    }
    if not shell_keys:
        return carriers

    lock_root = (Path(tempfile.gettempdir())
                 / "lx-obiwan-r6f-brep-cache" / "locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    state_name = "floor" if stand_foot else "no_floor"
    mode = "full" if _large_host_execution() else "segmented"
    shell_lock_path = lock_root / (
        f"{state_name}-{route_name.lower()}-{mode}-shells.lock")
    with shell_lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _stage_shell_contract_breps_unlocked(
            stand_foot, route_name, directory, shell_keys=shell_keys,
            seed_targets=carriers)


def _validate_staged_shell(staged, shell_key, route_name):
    """Return exact shell volume not contained by its assembled owners."""
    # Both printed routes are mandatory-core owned. The tweeter is staged for
    # mechanical/service checks, never credited toward T-shell containment.
    part_keys = ["lm", "um"]
    if _large_host_execution():
        # Osado can keep every full shell component and all final owners in
        # one guarded address space. The local path below retains its
        # one-carrier-at-a-time native-BREP subtraction chain.
        _wait_for_worker_headroom(
            f"{route_name} {shell_key} complete-shell validation", 3200.0)
        env = os.environ.copy()
        env.pop("LX_R6F_CASE_ID", None)
        env["LX_R6F_VALIDATE_SHELL"] = f"{route_name}/{shell_key}"
        env["LX_R6F_VALIDATE_SHELL_PATHS"] = os.pathsep.join(
            str(path) for path in staged[shell_key])
        env["LX_R6F_VALIDATE_PART_PATHS"] = os.pathsep.join(
            str(staged[key]) for key in part_keys)
        proc = subprocess.run(
            _cad_worker_command(Path(__file__).resolve()),
            env=env, text=True, capture_output=True)
        assert proc.returncode == 0, (
            f"isolated complete {route_name} {shell_key} validation failed\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        match = re.search(r"missing=([0-9.eE+-]+)", proc.stdout)
        assert match, f"missing shell result absent from: {proc.stdout!r}"
        print(proc.stdout, end="", flush=True)
        return float(match.group(1))

    total_missing = 0.0
    for chunk_index, chunk_path in enumerate(staged[shell_key]):
        current = Path(chunk_path)
        remaining_volume = math.inf
        for index, part_key in enumerate(part_keys):
            _wait_for_worker_headroom(
                f"{route_name} {shell_key} chunk {chunk_index} "
                f"minus {part_key}", 1800.0)
            output = current.with_name(
                f"{shell_key}_chunk_{chunk_index:02d}_minus_"
                f"{index + 1}_{part_key}.brep")
            env = os.environ.copy()
            env.pop("LX_R6F_CASE_ID", None)
            env["LX_R6F_SUBTRACT_INPUT"] = str(current)
            env["LX_R6F_SUBTRACT_PART"] = str(staged[part_key])
            env["LX_R6F_SUBTRACT_OUTPUT"] = str(output)
            proc = subprocess.run(
                _cad_worker_command(Path(__file__).resolve()),
                env=env, text=True, capture_output=True)
            assert proc.returncode == 0, (
                f"isolated {route_name} {shell_key} chunk {chunk_index} "
                f"minus {part_key} failed\nstdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}")
            match = re.search(r"remaining=([0-9.eE+-]+)", proc.stdout)
            assert match, (
                f"remaining shell result absent from: {proc.stdout!r}")
            remaining_volume = float(match.group(1))
            print(proc.stdout, end="", flush=True)
            if remaining_volume <= 1e-9:
                break
            assert output.is_file() and output.stat().st_size > 0
            current = output
        total_missing += max(0.0, remaining_volume)
    return total_missing


def _handoff_missing_volume(owners, route_name, route, *, verbose=False):
    """All Obi-Wan cable handoffs are native flush mouths/free cable."""
    components = route.required_handoff_shell_components(route_name)
    assert components == (), (
        f"{route_name} retained a stale telescoping tongue/socket contract")
    return 0.0


def _assembled_shell_contract(stand_foot, route_name):
    _state(stand_foot)
    # Stage the final carrier BREPs before importing OCC/core/route in this
    # process; otherwise those resident modules consume the headroom needed
    # by the isolated construction child.
    staging = tempfile.TemporaryDirectory(prefix="lx-obiwan-r6f-")
    staged = _stage_shell_contract_breps(
        stand_foot, route_name, staging.name)
    missing_volume = _validate_staged_shell(
        staged, "shell_nominal", route_name)
    lower_missing = _validate_staged_shell(
        staged, "shell_lower", route_name)
    upper_missing = _validate_staged_shell(
        staged, "shell_upper", route_name)
    from build123d import Cylinder, Pos, import_brep
    lm = import_brep(staged["lm"])
    um = import_brep(staged["um"])
    print("  starting assembled shell BREP contract", flush=True)
    from lx521_baffle.base import L22_CUTOUT, THICKNESS_MM, UM_CUTOUT
    import lx521_baffle.flush as flush
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    assert um.volume < 35000.0, (
        f"UM exceeds minimal-material budget: {um.volume:.1f} mm3")
    um_membrane_void = Pos(
        UM_CUTOUT[0] - 45.0, UM_CUTOUT[1], 11.5) * Cylinder(0.8, 1.0)
    assert _intersection_volume(um, um_membrane_void) < 0.01
    for x, y in flush.UM_PILOT_XY:
        bore_z0 = flush.UM_SEAT_Z - core.UM_PILOT_DEPTH_MM
        bore_z1 = flush.UM_SEAT_Z + 0.1
        bore = Pos(x, y, (bore_z0 + bore_z1) / 2.0) * Cylinder(
            core.UM_PILOT_D_MM / 2.0,
            bore_z1 - bore_z0)
        assert _intersection_volume(um, bore) < 0.02
        floor = Pos(
            x, y, bore_z0 - flush.UM_PAD_FLOOR_MM / 2.0
        ) * Cylinder(0.8, flush.UM_PAD_FLOOR_MM - 0.1)
        assert _intersection_volume(um, floor) > 0.90 * floor.volume
    actual_parts = [lm, um]
    if route_name == "T":
        tweeter = import_brep(staged["tweeter"])
        actual_parts.append(tweeter)
        state = "floor" if stand_foot else "no-floor"
        _assert_standalone_um_tweeter_joint_brep(
            um, tweeter, core, f"{state} canonical T--UM")
    if route_name == "UM":
        _assert_core_interface_breps(lm, um, core)

    # Cover fusions are followed by functional recuts; neither driver flange
    # seat may contain re-added crossover or route-cover material.
    for part, cutout, radius, seat_z, label in (
            (lm, L22_CUTOUT, flush.LM_RECESS_R, flush.LM_SEAT_Z, "LM"),
            (um, UM_CUTOUT, flush.UM_RECESS_R, flush.UM_SEAT_Z, "UM")):
        keepout = Pos(
            cutout[0], cutout[1], (seat_z + THICKNESS_MM + 0.2) / 2.0
        ) * Cylinder(radius, THICKNESS_MM + 0.2 - seat_z)
        hit = part & keepout
        assert hit is None or hit.volume < 0.02, (
            f"{label} flange seat refilled by {hit.volume:.3f} mm3")

    assert missing_volume < 0.05, (
        f"{route_name} assembled shell missing {missing_volume:.3f} mm3")
    # Exact final-BREP normal-distance bracket. The lower-radius swept shell
    # must be wholly contained in the manufactured LM+UM parts, while a
    # shell 0.10 mm beyond the 0.8-mm design wall must expose positive
    # missing volume. This is independent of the analytic XY route buffer.
    lower_wall = 0.76
    upper_wall = 0.90
    assert lower_missing < 0.05, (
        f"{route_name} final-BREP {lower_wall:.2f} wall containment "
        f"missing {lower_missing:.3f} mm3")
    assert upper_missing > 0.10, (
        f"{route_name} final-BREP wall bracket failed; {upper_wall:.2f} "
        f"shell missing only {upper_missing:.3f} mm3")
    owners = {"lm": lm, "um": um}
    if route_name == "T":
        owners["tweeter"] = actual_parts[-1]
    handoff_missing = _handoff_missing_volume(
        owners, route_name, route, verbose=True)
    assert handoff_missing < 0.05, (
        f"{route_name} required owner-seam shell missing "
        f"{handoff_missing:.3f} mm3")

    overlap = lm & um
    assert overlap is None or overlap.volume < 0.02, (
        f"LM/UM prints collide by {overlap.volume:.3f} mm3")
    if route_name == "T":
        overlap = um & actual_parts[-1]
        assert overlap is None or overlap.volume < 0.02, (
            f"UM/tweeter prints collide by {overlap.volume:.3f} mm3")
        overlap = lm & actual_parts[-1]
        assert overlap is None or overlap.volume < 0.02, (
            f"LM/tweeter prints collide by {overlap.volume:.3f} mm3")
    state = "floor" if stand_foot else "no-floor"
    staging.cleanup()
    print(
        f"  {state} {route_name} shell present to each native mouth/seam; "
        f"final-BREP normal wall bracket {lower_wall:.2f}..{upper_wall:.2f} mm")










def _physical_tubes(route):
    return {
        "UM_D7": route._round_tube(
            route.route_cable_points(1.5), route.CABLE_R_EST),
        "LM_D7p8": route._round_tube(
            route.lm_complete_duct_points(1.0),
            route.LM_CABLE_D_EST / 2.0),
        "T_D5p2": route._round_tube(
            route.ts_cable_points(1.5), route.TS_CABLE_D_EST / 2.0),
    }


def _carrier_cable_clearance(owner, stand_foot=False):
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "T", tempfile.gettempdir(), shell_keys=())
    if _large_host_execution():
        label = f"{owner} complete three-cable clearance matrix"
        _wait_for_worker_headroom(label, R6F_CABLE_WORKER_HEADROOM_MB)
        env = os.environ.copy()
        env.pop("LX_R6F_CASE_ID", None)
        env["LX_R6F_VALIDATE_CABLE"] = "ALL"
        env["LX_R6F_VALIDATE_CARRIER_PATH"] = str(staged[owner])
        env["LX_R6F_VALIDATE_CARRIER_OWNER"] = owner
        env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
        proc = subprocess.run(
            _cad_worker_command(Path(__file__).resolve()),
            env=env, text=True, capture_output=True)
        assert proc.returncode == 0, (
            f"isolated {label} failed\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")
        print(proc.stdout, end="", flush=True)
        state = "floor" if stand_foot else "no-floor"
        print(
            f"  exact D7.8/D7/D5.2 cables fit {state} {owner.upper()} carrier")
        return

    # OCC retains the loft/Boolean allocator after ``del``. Keep one cable
    # pair per guarded worker in both profiles, but validate the full cable in
    # each osado worker instead of the local 12-segment tweeter workaround.
    for name in ("UM_D7", "LM_D7p8", "T_D5p2"):
        segment_count = (
            1 if _large_host_execution()
            else 12 if name == "T_D5p2" else 1)
        modes = ("overflow", "collision", "witness")
        for segment_index in range(segment_count):
            for mode in modes:
                label = (
                    f"{owner}/{name} cable {mode} segment "
                    f"{segment_index + 1}/{segment_count}")
                _wait_for_worker_headroom(
                    label, R6F_CABLE_WORKER_HEADROOM_MB)
                env = os.environ.copy()
                env.pop("LX_R6F_CASE_ID", None)
                env["LX_R6F_VALIDATE_CABLE"] = name
                env["LX_R6F_VALIDATE_CABLE_MODE"] = mode
                env["LX_R6F_VALIDATE_CABLE_SEGMENT"] = str(segment_index)
                env["LX_R6F_VALIDATE_CABLE_SEGMENT_COUNT"] = str(
                    segment_count)
                env["LX_R6F_VALIDATE_CARRIER_PATH"] = str(staged[owner])
                env["LX_R6F_VALIDATE_CARRIER_OWNER"] = owner
                env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
                proc = subprocess.run(
                    _cad_worker_command(Path(__file__).resolve()),
                    env=env, text=True, capture_output=True)
                assert proc.returncode == 0, (
                    f"isolated {label} validation failed\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                print(proc.stdout, end="", flush=True)
    state = "floor" if stand_foot else "no-floor"
    print(f"  exact D7.8/D7/D5.2 cables fit {state} {owner.upper()} carrier")










def test_floor_integrated_mount():
    """Validate the one-piece floor load path, services and strength gate."""
    _state(True)
    staged = _stage_shell_contract_breps(
        True, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Box, Cylinder, Plane, Pos, Rot, import_brep
    from export_piece_stls import (
        BED_MM,
        BED_ROT_Z,
        OBIWAN_OPTIONAL_LM_SPLIT_BED_MM,
    )
    import lx521_baffle.obiwan.floor as floor
    import lx521_baffle.obiwan.floor_strength as strength
    import lx521_baffle.obiwan.lm_split as lm_split
    import lx521_baffle.obiwan.route as route
    import lx521_baffle.obiwan.carriers as core
    from shapely.geometry import Point, Polygon
    from lx521_baffle.base import L22_CUTOUT
    from lx521_baffle.flush import LM_RECESS_R
    from lx521_baffle.obiwan.bridge import (
        LM_WING_CONTACT_Z,
        common_lm_wing_contact_plan,
    )
    from lx521_baffle.obiwan.floor import integral_stem_plan_points

    lm = import_brep(staged["lm"])
    assert lm.is_valid and len(lm.solids()) == 1
    facts = floor.integrated_floor_facts()
    assert facts["ownership"] == (
        "floor_core_lm_and_optional_keyed_bottom")
    assert facts["feature_group_count"] == 5
    assert facts["floor_y_mm"] == 0.0
    assert facts["lm_axis_y_mm"] == 200.981
    assert facts["lm_axis_to_floor_mm"] == 200.981
    assert facts["foot_width_mm"] == 64.0
    assert facts["foot_height_mm"] == 18.3
    assert facts["foot_z_mm"] == (-150.0, 18.3)
    assert facts["stem_z_mm"] == (0.0, 18.3)
    assert facts["root_fillet_r_mm"] is None
    bend = facts["floor_bend"]
    assert bend["profile"] == "option_b_tangent_cubic"
    assert bend["wall_thickness_mm"] == 18.3
    assert bend["rear_span_mm"] == 75.0
    assert bend["rise_mm"] == 65.0
    assert bend["horizontal_handle_mm"] == 67.5
    assert math.isclose(
        bend["vertical_handle_mm"], 33.91836734693878,
        abs_tol=1e-12)
    assert bend["minimum_centerline_radius_mm"] >= 41.0
    assert bend["curvature_reversals"] == 0
    assert bend["horizontal_tangent_xyz_mm"] == (
        0.0, 9.15, -65.85)
    assert bend["vertical_tangent_xyz_mm"] == (0.0, 74.15, 9.15)
    assert math.isclose(
        facts["rear_flat_end_z_mm"],
        -65.85 + bend["fusion_overlap_mm"], abs_tol=1e-12)
    assert math.isclose(
        facts["upright_start_y_mm"],
        74.15 - bend["fusion_overlap_mm"], abs_tol=1e-12)
    assert facts["panel_z_mm"] == (-150.0, -146.0)
    assert facts["panel_height_mm"] == 44.0
    assert facts["nl8_center_y_mm"] == 22.0
    assert facts["nl8_cutout_d_mm"] == 31.0
    assert facts["nl8_screw_d_mm"] == 3.2
    assert facts["nl8_screw_pitch_mm"] == 29.2
    assert facts["floor_lane_count"] == 3
    assert set(facts["floor_lanes"]) == {"lm", "um", "t"}
    assert {
        name: record["diameter_mm"]
        for name, record in facts["floor_lanes"].items()
    } == {"lm": 9.0, "um": 8.2, "t": 6.0}
    assert all(
        record["bend_radius_mm"] >= 14.0
        for record in facts["floor_lanes"].values())

    bounds = lm.bounding_box()
    assert math.isclose(bounds.min.Y, 0.0, abs_tol=0.02)
    assert math.isclose(bounds.min.Z, -150.0, abs_tol=0.02)
    assert math.isclose(bounds.max.Z, 18.3, abs_tol=0.02)

    # Probe only the newly added universal-profile shoulder.  The complete
    # common plan also contains legitimate driver, route and carrier cavities;
    # demanding it all be solid would test a filled baffle, not exterior wing
    # compatibility.  The live flat/graded gate independently compares the exact
    # final front-section exteriors of both stand states.
    native_floor_plan = Polygon(integral_stem_plan_points()).buffer(0)
    shared_shoulder_plan = common_lm_wing_contact_plan().difference(
        native_floor_plan).buffer(0)
    # The two historic transition pockets are wholly inside the R110.6 driver
    # flange recess and are therefore removed from both finished carriers by
    # the ordinary seat cutter.  Wing compatibility concerns the material
    # outside that immutable driver keepout.
    shared_shoulder_plan = shared_shoulder_plan.difference(
        Point(*L22_CUTOUT[:2]).buffer(LM_RECESS_R, resolution=256)
    ).buffer(0)
    shared_front = core._plan_prism(
        shared_shoulder_plan,
        LM_WING_CONTACT_Z[1] - 0.35,
        LM_WING_CONTACT_Z[1] - 0.05)
    shared_front_missing = shared_front - lm
    shared_front_missing_volume = (
        0.0 if shared_front_missing is None else sum(
            solid.volume for solid in shared_front_missing.solids()))
    assert shared_front_missing_volume < 0.05, (
        "floor LM lacks universal wing-contact shoulder material: "
        f"{shared_front_missing_volume:.3f} mm3")

    # Independent final-BREP material probes cover the retained rear flat,
    # the exact Option-B mid-bend and the retained upper upright without
    # reusing the solid builders.  A negative witness in the former straight
    # upright proves the old hard-corner envelope was actually removed.
    rear_flat_witness = Pos(28.0, 9.15, -100.0) * Box(2.0, 16.0, 20.0)
    bend_mid_witness = Pos(28.0, 28.93061224489796, -3.0375) * Box(
        2.0, 2.0, 2.0)
    upper_upright_witness = Pos(28.0, 80.0, 9.15) * Box(2.0, 8.0, 16.0)
    for label, witness in (
            ("rear flat", rear_flat_witness),
            ("Option-B mid-bend", bend_mid_witness),
            ("upper upright", upper_upright_witness)):
        retained = _intersection_volume(lm, witness)
        assert retained > 0.97 * witness.volume, (
            f"integral {label} retained only "
            f"{retained / witness.volume:.1%}")
    former_upright = Pos(28.0, 29.0, 15.0) * Box(2.0, 2.0, 2.0)
    assert _intersection_volume(lm, former_upright) < 0.02, (
        "former hard-corner upright material remains outside Option B")

    # The old universal-profile fusion offset was evaluated against a stem
    # that Option B had already removed below y=71.15.  At front depth it
    # therefore became a freestanding rectangular perimeter around the bend.
    # Reject its bottom crossbar, lower side wall, and former magnet rail.
    # The relocated captive pair is wholly in the upper shoulder.
    old_box_crossbar = Pos(0.0, 0.05, 12.55) * Box(40.0, 0.06, 4.0)
    old_box_side = Pos(31.95, 8.0, 12.55) * Box(0.06, 8.0, 4.0)
    for label, witness in (
            ("bottom crossbar", old_box_crossbar),
            ("lower side wall", old_box_side)):
        assert _intersection_volume(lm, witness) < 0.01, (
            f"obsolete floor-profile rectangular {label} remains")
    former_magnet_rail = Pos(30.5, 18.0, 12.55) * Box(
        1.0, 2.0, 4.0)
    assert _intersection_volume(lm, former_magnet_rail) < 0.01, (
        "obsolete lower-LM magnet rail remains below the curved stand")

    # The rear connector panel is real, with a bounded service scoop behind
    # it and the exact NL8 plus four-hole pattern open through the panel.
    cavity_x, cavity_y, cavity_z = facts["service_cavity_xyz_mm"]
    cavity_probe = Pos(
        sum(cavity_x) / 2.0,
        sum(cavity_y) / 2.0,
        sum(cavity_z) / 2.0,
    ) * Box(
        cavity_x[1] - cavity_x[0] - 0.4,
        cavity_y[1] - cavity_y[0] - 0.4,
        cavity_z[1] - cavity_z[0] - 0.4,
    )
    assert _intersection_volume(lm, cavity_probe) < 0.05
    panel_z = sum(facts["panel_z_mm"]) / 2.0
    panel_h = facts["panel_z_mm"][1] - facts["panel_z_mm"][0]
    panel_voids = [
        Pos(0.0, facts["nl8_center_y_mm"], panel_z)
        * Cylinder(facts["nl8_cutout_d_mm"] / 2.0 - 0.15, panel_h + 0.2),
    ]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            panel_voids.append(
                Pos(
                    sx * facts["nl8_screw_pitch_mm"] / 2.0,
                    facts["nl8_center_y_mm"]
                    + sy * facts["nl8_screw_pitch_mm"] / 2.0,
                    panel_z,
                ) * Cylinder(
                    facts["nl8_screw_d_mm"] / 2.0 - 0.10,
                    panel_h + 0.2))
    assert all(_intersection_volume(lm, void) < 0.02
               for void in panel_voids)

    # Each service continuation is buried through the floor member, retains
    # a local wall and opens at full diameter into the connector cavity. The
    # radius check samples every edge of the production composite wire--not
    # its dependency-light preview/control polygon.
    lane_radii = {}
    lane_centerlines = {}
    for name, record in facts["floor_lanes"].items():
        x = record["x_mm"]
        y = record["floor_y_mm"]
        radius = record["diameter_mm"] / 2.0
        lumen = Pos(x, y, -90.0) * Cylinder(radius - 0.15, 10.0)
        assert _intersection_volume(lm, lumen) < 0.02, (
            f"{name} floor lumen is obstructed")
        wall = (
            Pos(x, y, -90.0) * Cylinder(radius + 0.55, 8.0)
            - Pos(x, y, -90.0) * Cylinder(radius + 0.10, 8.0)
        )
        retained = _intersection_volume(lm, wall)
        assert retained > 0.97 * wall.volume, (
            f"{name} floor lumen is not fully buried: "
            f"{retained / wall.volume:.1%} wall retained")

        connector_margin = min(
            x - radius - cavity_x[0],
            cavity_x[1] - (x + radius),
            y - radius - cavity_y[0],
            cavity_y[1] - (y + radius),
        )
        assert connector_margin >= 0.49, (
            f"{name} full connector opening margin "
            f"{connector_margin:.3f} mm")
        connector_opening = Pos(
            x, y, cavity_z[1] + 0.50
        ) * Cylinder(radius - 0.10, 0.60)
        assert _intersection_volume(lm, connector_opening) < 0.02, (
            f"{name} connector-side full-diameter opening is capped")

        path = floor.floor_lane_path(name)
        assert path.is_valid and not path.is_closed
        edges = tuple(path.edges())
        assert len(edges) == (4 if name == "lm" else 2)

        def xyz(edge, parameter):
            point = edge @ parameter
            return np.asarray((point.X, point.Y, point.Z), dtype=float)

        def tangent(edge, parameter):
            vector = edge % parameter
            out = np.asarray((vector.X, vector.Y, vector.Z), dtype=float)
            return out / np.linalg.norm(out)

        actual_samples = []
        edge_radii = []
        for edge_index, edge in enumerate(edges):
            points = np.asarray([
                xyz(edge, index / 160.0) for index in range(161)])
            edge_radii.append(_min_three_point_radius(points))
            actual_samples.extend(
                points if edge_index == 0 else points[1:])
        for left, right in zip(edges[:-1], edges[1:]):
            gap = float(np.linalg.norm(xyz(left, 1.0) - xyz(right, 0.0)))
            alignment = float(np.dot(
                tangent(left, 1.0), tangent(right, 0.0)))
            assert gap <= 1.0e-6, (
                f"{name} composite wire has {gap:.9f}-mm edge gap")
            assert alignment >= 0.999999, (
                f"{name} composite wire join is not G1: {alignment:.9f}")
        actual_samples = np.asarray(actual_samples)
        lane_centerlines[name] = actual_samples
        lane_radii[name] = min(
            min(edge_radii), _min_three_point_radius(actual_samples))
        assert lane_radii[name] >= 14.0 - 0.03, (
            f"{name} integral-floor lane radius "
            f"{lane_radii[name]:.3f} mm")
        assert _max_turn_deg(actual_samples) <= 2.0, (
            f"{name} composite wire has a sampled direction cusp")

        if name == "lm":
            assert record["handoff_mode"] == "buried_r14_rear_exit"
            assert record["route_overlap_mm"] == 0.0
            assert record["prefusion_handoff_gap_mm"] == 0.0
            assert record["owner_cutter_backreach_mm"] == 0.0
            assert np.allclose(
                xyz(edges[-1], 1.0), record["feed_xyz_mm"], atol=1e-9)
            lm_free = route.lm_cable_points(0.10)
            assert np.allclose(
                lm_free[0],
                record["rear_face_mouth_xyz_mm"],
                atol=1e-9)
            assert np.allclose(
                record["rear_face_mouth_xyz_mm"],
                (*route.LM_REAR_PORT_XY, route.LM_REAR_PORT_REAR_Z),
                atol=1e-9)
            assert record["exit_bend_radius_mm"] >= 14.0
            lm_port = route.lm_rear_exit_port_cutter()
            bounds = lm_port.bounding_box()
            assert (bounds.min.Z <= record["feed_xyz_mm"][2] <= bounds.max.Z
                    and bounds.min.Y <= record["feed_xyz_mm"][1]
                    <= bounds.max.Y), (
                        "floor LM lane does not overlap the R14 rear port")
            assert np.allclose(
                tangent(edges[-1], 1.0),
                record["external_tangent_xyz"], atol=1e-6)
            # The three former lower-stem holes remain closed.  UM/T are on
            # the retained vertical rear face.  The old LM witness at y=38.5
            # now lies in the Option-B bend, so probe both true offset skins
            # along the cubic normal instead of pretending z=0 is still an
            # exterior surface there.  The one allowed LM outlet lives beside
            # the lower ring at the shared Stock/Slim datum.
            rear_skin = floor.FLOOR_REAR_FACE_SKIN_MM - 0.05
            for label, center, probe_radius in (
                    ("former UM stem mouth", (8.0, 82.0), 3.8),
                    ("former T stem mouth", (-8.0, 82.0), 2.7)):
                rear_plug = Pos(*center, rear_skin / 2.0) * Cylinder(
                    probe_radius, rear_skin)
                assert _intersection_volume(lm, rear_plug) > (
                    0.97 * rear_plug.volume), (
                        f"{label} is still visible at the rear face")

            from lx521_baffle.floor_bend import (
                WALL_HALF_THICKNESS_MM,
                centerline_controls,
                cubic_derivatives,
                cubic_point,
            )
            controls = centerline_controls()
            low, high = 0.0, 1.0
            for _index in range(80):
                parameter = 0.5 * (low + high)
                if cubic_point(controls, parameter)[1] < 38.5:
                    low = parameter
                else:
                    high = parameter
            parameter = 0.5 * (low + high)
            center = np.asarray(cubic_point(controls, parameter), dtype=float)
            tangent = np.asarray(
                cubic_derivatives(controls, parameter)[0], dtype=float)
            normal = np.asarray(
                (0.0, tangent[2], -tangent[1]), dtype=float)
            normal /= np.linalg.norm(normal)
            for side in (-1.0, 1.0):
                outward = side * normal
                surface = center + WALL_HALF_THICKNESS_MM * outward
                inward = -outward
                # Probe a 0.20-mm band wholly inside the specified 0.45-mm
                # skin.  ``Cylinder`` is centre-aligned, so locate its centre
                # 0.20 mm inward from the analytic offset face.  The 0.10-mm
                # exterior allowance also absorbs the small tangent-plane /
                # curved-surface chord error across the full D8.4 witness.
                probe_depth = 0.20
                origin = surface + 0.20 * inward
                curved_plug = Plane(
                    origin=tuple(origin), z_dir=tuple(inward)
                ) * Cylinder(4.2, probe_depth)
                assert _intersection_volume(lm, curved_plug) > (
                    0.97 * curved_plug.volume), (
                        "former LM stem mouth is exposed through the "
                        f"Option-B {'outer' if side < 0 else 'inner'} skin")
            outlet = Pos(
                *record["rear_face_mouth_xyz_mm"][:2], rear_skin / 2.0
            ) * Cylinder(radius - 0.10, rear_skin)
            assert _intersection_volume(lm, outlet) < 0.02, (
                "LM lower-ring outlet is capped")
            continue

        assert record["handoff_mode"] == "buried_route_overlap"
        assert record["prefusion_handoff_gap_mm"] == 0.8
        assert record["owner_cutter_backreach_mm"] == 2.0
        assert math.isclose(
            record["owner_cutter_backreach_mm"],
            route.FLOOR_FEED_CUTTER_EXTENSION, abs_tol=1e-9)
        assert math.isclose(
            record["route_overlap_mm"], 1.2, abs_tol=1e-9)
        annular = (
            route.route_cable_points(0.02) if name == "um"
            else route.ts_cable_points(0.02))
        feed = np.asarray(record["feed_xyz_mm"], dtype=float)
        assert np.allclose(annular[0], feed, atol=1e-9)
        body_endpoint = xyz(edges[-1], 1.0)
        assert math.isclose(
            float(np.linalg.norm(feed - body_endpoint)),
            record["prefusion_handoff_gap_mm"], abs_tol=1e-6)
        floor_tangent = tangent(edges[-1], 1.0)
        annular_tangent = annular[1] - annular[0]
        annular_tangent /= np.linalg.norm(annular_tangent)
        assert float(np.dot(floor_tangent, annular_tangent)) >= 0.999, (
            f"{name} floor/annular handoff is not G1")
        owner_start = (
            feed
            - record["owner_cutter_backreach_mm"] * annular_tangent)
        overlap_vector = body_endpoint - owner_start
        axial_overlap = float(np.dot(overlap_vector, annular_tangent))
        lateral_gap = float(np.linalg.norm(
            overlap_vector - axial_overlap * annular_tangent))
        assert math.isclose(
            axial_overlap, record["route_overlap_mm"], abs_tol=0.03), (
            f"{name} effective final overlap is {axial_overlap:.3f} mm")
        assert lateral_gap <= 0.03, (
            f"{name} setback/owner cutter axes miss by "
            f"{lateral_gap:.3f} mm")

    # Screen complete production centerlines, not just their deceptively
    # well-separated endpoints.  The original LM y=82 turn and a trial
    # y=74.5 turn both crossed the UM/T approach cubics.  The immediate
    # second R14 LM turn plus x=+/-12 service tracks retain a real wall along
    # every complete floor-body lane pair.
    lane_radii_mm = {
        name: record["diameter_mm"] / 2.0
        for name, record in facts["floor_lanes"].items()
    }
    for left, right in (("lm", "um"), ("lm", "t"), ("um", "t")):
        center_distance = float(np.min(np.linalg.norm(
            lane_centerlines[left][:, None, :]
            - lane_centerlines[right][None, :, :], axis=2)))
        lumen_wall = (
            center_distance - lane_radii_mm[left] - lane_radii_mm[right])
        assert lumen_wall >= route.TUNNEL_SKIN - 0.02, (
            f"{left.upper()}/{right.upper()} complete floor lanes leave "
            f"only {lumen_wall:.3f} mm wall")

    # Also screen every non-mating floor lane against the first 12 mm of
    # each annular feed.  Own lane/feed pairs intentionally overlap by the
    # separately gated 1.2 mm handoff; all other pairs must remain distinct.
    annular_feeds = {}
    for name, annular, radius in (
            ("um", route.route_cable_points(0.10), route.CUTTER_R),
            ("t", route.ts_cable_points(0.10), route.TS_CUTTER_R)):
        annular = np.asarray(annular, dtype=float)
        stations = np.concatenate((
            [0.0], np.cumsum(np.linalg.norm(
                np.diff(annular, axis=0), axis=1))))
        annular_feeds[name] = (annular[stations <= 12.0 + 1e-9], radius)
    for lane_name, lane_points in lane_centerlines.items():
        for feed_name, (feed_points, feed_radius) in annular_feeds.items():
            if lane_name == feed_name:
                continue
            center_distance = float(np.min(np.linalg.norm(
                lane_points[:, None, :] - feed_points[None, :, :], axis=2)))
            lumen_wall = (
                center_distance - lane_radii_mm[lane_name] - feed_radius)
            assert lumen_wall >= route.TUNNEL_SKIN - 0.02, (
                f"{lane_name.upper()} floor lane leaves only "
                f"{lumen_wall:.3f} mm to {feed_name.upper()} annular feed")
    feed_distance = float(np.min(np.linalg.norm(
        annular_feeds["um"][0][:, None, :]
        - annular_feeds["t"][0][None, :, :], axis=2)))
    feed_wall = (
        feed_distance - annular_feeds["um"][1] - annular_feeds["t"][1])
    assert feed_wall >= route.TUNNEL_SKIN - 0.02, (
        f"UM/T nominal annular feeds leave only {feed_wall:.3f} mm wall")

    # The globally phased 2.0-mm owner backreach is part of the installed
    # floor lumens, not merely a Boolean allowance.  Sample both complete
    # backreach segments so the two state-owned floor mouths remain
    # independently printable passages rather than merging behind the face.
    backreach_segments = {}
    for name, points in (
            ("um", route.route_cable_points(1.8)),
            ("t", route.ts_cable_points(1.8))):
        extended = route._owner_cutter_points(points, "lm")
        u = np.linspace(0.0, 1.0, 161)[:, None]
        backreach_segments[name] = (
            extended[0] + u * (extended[1] - extended[0]))
    backreach_distance = float(np.min(np.linalg.norm(
        backreach_segments["um"][:, None, :]
        - backreach_segments["t"][None, :, :], axis=2)))
    backreach_wall = (
        backreach_distance - route.CUTTER_R - route.TS_CUTTER_R)
    assert backreach_wall >= route.TUNNEL_SKIN - 0.02, (
        f"UM/T 2.0-mm owner backreaches leave only "
        f"{backreach_wall:.3f} mm wall")

    # The monolithic carrier is intentionally retained as the canonical
    # large-format reference. The bottom keyed option owns the complete stand
    # and, like every released part, prints front-face-down.  Its in-bed Z
    # rotation is allowed to minimize XY without changing that common datum.
    canonical = (
        Rot(Z=BED_ROT_Z["obiwan_core_1_of_2_lm_carrier"])
        * Rot(X=180.0) * lm)
    canonical_size = canonical.bounding_box().size
    assert BED_MM == 256.0
    assert max(canonical_size.X, canonical_size.Y, canonical_size.Z) > BED_MM
    keyed = lm_split.lm_carrier_split_parts(lm)
    bottom = keyed["optional_lm_keyed_1_of_2_bottom"]
    top = keyed["optional_lm_keyed_2_of_2_top"]
    assert math.isclose(bottom.bounding_box().min.Y, 0.0, abs_tol=0.02)
    assert math.isclose(bottom.bounding_box().min.Z, -150.0, abs_tol=0.02)
    assert top.bounding_box().min.Y > facts["stem_top_y_mm"]
    bottom_print = (
        Rot(Z=BED_ROT_Z["obiwan_optional_lm_keyed_1_of_2_bottom"])
        * Rot(X=180.0) * bottom)
    bottom_size = bottom_print.bounding_box().size
    assert OBIWAN_OPTIONAL_LM_SPLIT_BED_MM == 220.0
    assert max(bottom_size.X, bottom_size.Y, bottom_size.Z) <= 220.0, (
        f"integral keyed bottom footprint {bottom_size.X:.2f} x "
        f"{bottom_size.Y:.2f} x {bottom_size.Z:.2f} exceeds 220 mm")

    # Closed-form simulation uses the exact net tangent-wall section, an
    # explicit curved-wall/lumen stress factor, and vertical/lateral 1/3/5g
    # cases. It never substitutes for the still-pending physical gate.
    screen = strength.integral_floor_strength_facts()
    assert screen["schema_version"] == 3
    assert screen["analysis_kind"] == (
        "closed_form_net_section_screen_not_fea")
    geometry = screen["geometry"]
    assert geometry["floor_y_mm"] == 0.0
    assert geometry["lm_axis_y_mm"] == 200.981
    assert geometry["lm_axis_to_floor_mm"] == 200.981
    assert geometry["foot_width_mm"] == 64.0
    assert geometry["foot_height_mm"] == 18.3
    assert geometry["foot_z_mm"] == (-150.0, 18.3)
    assert geometry["root_fillet_r_mm"] is None
    assert geometry["floor_bend"] == bend
    assert geometry["bend_min_centerline_radius_mm"] >= 41.0
    assert geometry["root_stress_concentration_factor"] == 1.25
    assert {
        item["name"]: item["diameter_mm"]
        for item in screen["net_root_section"]["lumens"]
    } == {"lm": 9.0, "um": 8.2, "t": 6.0}
    assert set(screen["loads"]) == {"1", "3", "5"}
    assert set(screen["anchored_lateral_loads"]) == {"1", "3", "5"}
    assert all(
        screen["anchored_lateral_loads"][str(g)][
            "bending_stress_design_mpa"] > 0.0
        for g in (1, 3, 5))
    shoulder = screen["shoulder_ring_junction"]
    assert shoulder["analysis_kind"] == (
        "conservative_unreinforced_outer_lip_lower_bound_diagnostic")
    shoulder_section = shoulder["section"]
    assert shoulder_section["plane_y_mm"] == 105.981
    assert shoulder_section["radial_ordinate_mm"] == 95.0
    assert shoulder_section["lip_inner_radius_mm"] == 110.6
    assert shoulder_section["lip_outer_radius_mm"] == 113.0
    assert shoulder_section["ligament_count"] == 2
    assert math.isclose(
        shoulder_section["half_ligament_width_mm"],
        4.554675207314091, abs_tol=1e-12)
    assert shoulder_section["depth_mm"] == 11.5
    assert math.isclose(
        shoulder_section["net_area_mm2"],
        104.75752976822409, abs_tol=1e-10)
    assert math.isclose(
        shoulder_section["second_moment_x_mm4"],
        1154.5152759873029, abs_tol=1e-9)
    assert math.isclose(
        shoulder_section["governing_section_modulus_mm3"],
        200.78526538909614, abs_tol=1e-10)
    shoulder_interpretation = shoulder["interpretation"]
    assert shoulder_interpretation[
        "changes_root_analytical_screen_pass"] is False
    assert shoulder_interpretation[
        "complete_assembly_failure_prediction"] is False
    assert shoulder_interpretation["installed_lm_flange_required"] is True
    assert shoulder_interpretation[
        "physical_proof_and_creep_gate_required"] is True
    assert shoulder_interpretation["threshold_result"] == (
        "DIAGNOSTIC_LOWER_BOUND_BELOW_THRESHOLDS")
    assert set(shoulder["materials"]) == set(screen["materials"])
    assert all(
        record["lower_bound_meets_root_thresholds"] is False
        for record in shoulder["materials"].values())
    thresholds = screen["acceptance_thresholds"]
    assert thresholds["min_sf_1g_sustained"] == 2.0
    assert thresholds["min_sf_3g_transient"] == 1.5
    assert thresholds["min_sf_5g_transient"] == 1.05
    expected_result = {
        "Bambu PLA Tough+": True,
        "Bambu PLA Basic": True,
        "Bambu PLA Lite": False,
        "Bambu PLA Matte": True,
        "Bambu PLA Silk+": True,
    }
    assert set(screen["materials"]) == set(expected_result)
    for name, expected_pass in expected_result.items():
        material = screen["materials"][name]
        assert material["sf_1g_sustained"] >= (
            thresholds["min_sf_1g_sustained"])
        assert material["sf_3g_transient"] >= (
            thresholds["min_sf_3g_transient"])
        if expected_pass:
            assert material["sf_5g_transient"] >= (
                thresholds["min_sf_5g_transient"])
            assert material["anchored_lateral_sf_1g_sustained"] >= (
                thresholds["min_sf_1g_sustained"])
            assert material["anchored_lateral_sf_3g_transient"] >= (
                thresholds["min_sf_3g_transient"])
            assert material["anchored_lateral_sf_5g_transient"] >= (
                thresholds["min_sf_5g_transient"])
        else:
            assert name == "Bambu PLA Lite"
            assert material["provisional"] is True
            assert material["sf_5g_transient"] < (
                thresholds["min_sf_5g_transient"])
        assert material["analytical_screen_pass"] is expected_pass
        assert material["physical_qualification"] == "PENDING"
    assert screen["physical_gate"]["status"] == "PENDING"
    assert screen["design_load"]["magnet_load_credit_n"] == 0.0
    assert screen["design_load"][
        "concealed_split_key_load_credit_n"] == 0.0
    assert screen["split_configuration"][
        "concealed_key_structural_credit_n"] == 0.0
    assert screen["split_configuration"][
        "installed_driver_flange_required_to_bridge_seam"] is True
    stability = screen["stability"]
    assert math.isclose(
        stability["lateral_tip_acceleration_g"],
        32.0 / 230.0, abs_tol=1e-12)
    assert stability["lateral_tip_acceleration_g"] < 1.0
    assert "anchor" in stability["warning"].lower()
    print(
        "  integral floor: exact Y=0 / 200.981-mm datum, W64 x H18.3 "
        "full-depth root, NL8 service panel and three buried R14 lanes; "
        f"keyed bottom {bottom_size.X:.2f} x {bottom_size.Y:.2f} x "
        f"{bottom_size.Z:.2f}; 4/5 PLA analytical passes, physical gate "
        "PENDING and anti-tip required")


def _tweeter_and_service_legacy_monolithic(stand_foot):
    _state(stand_foot)
    staging = tempfile.TemporaryDirectory(prefix="lx-obiwan-service-")
    staged = _stage_shell_contract_breps(
        stand_foot, "T", staging.name, shell_keys=())
    from build123d import import_brep
    import lx521_baffle.um_fit as fit
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    assert fit.PHYSICAL_MEASURE_REQUIRED is True
    body_facts = fit.mu10_body_reference_facts()
    assert body_facts["raw_stl_sha256"] == body_facts[
        "expected_raw_stl_sha256"]
    assert body_facts["terminals_present_in_raw_stl"] is False
    assert body_facts["world_strut_angles_deg"] == (
        13.0, 103.0, 193.0, 283.0)
    full_body = fit.mu10_body_keepout(include_flange=True)
    full_bb = full_body.bounding_box()
    assert math.isclose(full_bb.min.X, -49.0, abs_tol=0.02)
    assert math.isclose(full_bb.max.X, 49.0, abs_tol=0.02)
    assert math.isclose(full_bb.min.Y, 317.081, abs_tol=0.02)
    assert math.isclose(full_bb.max.Y, 415.081, abs_tol=0.02)
    assert math.isclose(full_bb.min.Z, -23.9, abs_tol=0.02)
    assert math.isclose(full_bb.max.Z, 19.7, abs_tol=0.02)
    lm = import_brep(staged["lm"])
    tweeter = import_brep(staged["tweeter"])
    assert tweeter.is_valid and len(tweeter.solids()) == 1
    cable = route._round_tube(
        route.ts_cable_points(1.5), route.TS_CABLE_D_EST / 2.0)
    hit = tweeter & cable
    assert hit is None or hit.volume < 0.10

    um = import_brep(staged["um"])
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    w22_facts = fit.w22_body_reference_facts()
    assert w22_facts["source_step_sha256"] == w22_facts[
        "expected_source_step_sha256"]
    fastons = fit.faston_proxy_parts()
    boots = fit.faston_boot_proxy_parts()
    pull_sweeps = fit.faston_pull_sweep_parts()
    service_parts = {
        "terminal_carrier": fit.terminal_carrier_proxy(),
        **fastons,
        **boots,
        **pull_sweeps,
    }
    for printed_name, printed in (("LM", lm), ("UM", um),
                                  ("tweeter", tweeter)):
        body_collision = printed & body
        body_volume = (0.0 if body_collision is None
                       else body_collision.volume)
        assert body_volume < 0.10, (
            f"{printed_name}/known MU10 body collision {body_volume:.3f}")
        body_clearance = printed.distance_to(body)
        required_clearance = (
            fit.MU10_BODY_MODEL_TOLERANCE_MM
            + fit.MU10_MIN_PRINTED_BODY_CLEARANCE_MM)
        assert body_clearance >= required_clearance - 0.03, (
            f"{printed_name}/MU10 positive clearance {body_clearance:.3f}")
        for service_name, service_part in service_parts.items():
            collision = printed & service_part
            volume = 0.0 if collision is None else collision.volume
            assert volume < 0.10, (
                f"{printed_name}/{service_name} collision {volume:.3f}")

    # The provisional low-profile flag bodies fit simultaneously; no test
    # may hide an impossible pair behind aggregate-envelope semantics.
    assert fastons["faston_receptacle_1"].distance_to(
        fastons["faston_receptacle_2"]) >= 2.49
    assert boots["faston_insulation_boot_1"].distance_to(
        boots["faston_insulation_boot_2"]) >= 1.49
    assert pull_sweeps["faston_flag_pull_sweep_1"].distance_to(
        pull_sweeps["faston_flag_pull_sweep_2"]) >= 1.49
    assert (fit.REMOVAL_ENVELOPE_CABLE_POLICY["obiwan"]
            == "independent_flag_faston_pull_with_slack_leads")

    harness = fit.obiwan_terminal_harness_parts()
    bundle = harness["obiwan_D7_bundle_to_Y_breakout"]
    leads = {
        1: harness["obiwan_terminal_lead_1_D3p2"],
        2: harness["obiwan_terminal_lead_2_D3p2"],
    }
    for printed in (lm, um, tweeter):
        for cable_name, cable_part in harness.items():
            collision = printed & cable_part
            assert collision is None or collision.volume < 0.10, (
                f"{cable_name}/printed collision")
    assert bundle.distance_to(body) >= 1.0
    assert bundle.distance_to(lm_body) >= 1.0
    assert bundle.distance_to(service_parts["terminal_carrier"]) >= 0.75
    for lead in leads.values():
        assert lead.distance_to(body) >= 1.0
        assert lead.distance_to(lm_body) >= 1.0
        assert lead.distance_to(service_parts["terminal_carrier"]) >= 0.75

    # D7 jacket ends before every connector. Each D3.2 lead enters only its
    # matching flag boot; the opposite connector remains service-clear.
    for part in (*fastons.values(), *boots.values()):
        assert _intersection_volume(bundle, part) < 0.01
    for index, lead in leads.items():
        assert _intersection_volume(
            lead, boots[f"faston_insulation_boot_{index}"]) > 0.05
        other = 2 if index == 1 else 1
        assert _intersection_volume(
            lead, boots[f"faston_insulation_boot_{other}"]) < 0.01
        assert _intersection_volume(
            lead, fastons[f"faston_receptacle_{other}"]) < 0.01

    # Separate pull sweeps clear the body, all printed parts, the opposite
    # installed connector and opposite lead. Matching lead/boot occupancy is
    # intentional and its long path supplies more than the 12-mm pull slack.
    for index in (1, 2):
        sweep = pull_sweeps[f"faston_flag_pull_sweep_{index}"]
        other = 2 if index == 1 else 1
        assert sweep.distance_to(body) >= 0.20
        assert sweep.distance_to(lm_body) >= 1.0
        assert _intersection_volume(
            sweep, fastons[f"faston_receptacle_{other}"]) < 0.01
        assert _intersection_volume(
            sweep, fastons[f"terminal_tab_{other}"]) < 0.01
        assert _intersection_volume(
            sweep, boots[f"faston_insulation_boot_{other}"]) < 0.01
        assert _intersection_volume(sweep, leads[other]) < 0.01

    for service_name, service_part in service_parts.items():
        if service_name in {"terminal_carrier", "terminal_tab_1",
                            "terminal_tab_2"}:
            continue
        collision = service_part & body
        volume = 0.0 if collision is None else collision.volume
        assert volume < 0.10, (
            f"{service_name}/known MU10 body collision {volume:.3f}")
    t_cable = fit.obiwan_ts_cable_envelope()
    t_body = t_cable & body
    assert t_body is None or t_body.volume < 0.10
    lm_cable_body = fit.obiwan_lm_cable_envelope() & body
    assert lm_cable_body is None or lm_cable_body.volume < 0.10
    for service_name, service_part in service_parts.items():
        collision = t_cable & service_part
        volume = 0.0 if collision is None else collision.volume
        assert volume < 0.10, (
            f"T cable/{service_name} collision {volume:.3f}")
    terminated_points = fit.obiwan_terminated_cable_points()
    assert fit.OBIWAN_TERMINATED_HANDOFF_R == 20.0
    assert _min_three_point_radius(terminated_points) >= 13.9
    assert _max_turn_deg(terminated_points) <= 2.5
    # Prove the free continuation itself is the declared G1 R20 quarter arc;
    # a whole-route radius check would otherwise be governed by printed R15.
    free_arc = np.asarray(terminated_points[-41:], dtype=float)
    route_before = np.asarray(terminated_points[-42], dtype=float)
    angle = math.radians(283.0)
    u = np.asarray((math.cos(angle), math.sin(angle), 0.0))
    v = np.asarray((*route.UM_MOUTH_TANGENT, 0.0))
    center = np.asarray((core.UM_CUTOUT[0], core.UM_CUTOUT[1], 0.0))
    radial = (free_arc - center) @ u
    tangential = (free_arc - center) @ v
    phi = np.linspace(0.0, math.pi / 2.0, 41)
    assert np.max(np.abs(radial - route.UM_MOUTH_R)) < 1e-6
    assert np.max(np.abs(
        tangential - fit.OBIWAN_TERMINATED_HANDOFF_R * np.sin(phi))) < 1e-6
    assert np.max(np.abs(
        free_arc[:, 2] - (route.UM_MOUTH_Z
                          - fit.OBIWAN_TERMINATED_HANDOFF_R
                          * (1.0 - np.cos(phi))))) < 1e-6
    assert _min_three_point_radius(free_arc) >= 19.95
    incoming = free_arc[0] - route_before
    outgoing = free_arc[1] - free_arc[0]
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    # Chords straddle two finite samplings of an analytically shared tangent
    # (1.5-mm R15 route stations and 40-step R20 arc).  Their half-step angle
    # is bounded separately from the exact declared tangent above.
    assert float(np.dot(incoming, outgoing)) > 0.998
    end_tangent = free_arc[-1] - free_arc[-2]
    end_tangent /= np.linalg.norm(end_tangent)
    assert float(np.dot(end_tangent, np.asarray((0.0, 0.0, -1.0)))) > 0.999

    installed_facts = fit.obiwan_terminal_harness_facts(0.0)
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        lead_points = fit.obiwan_terminal_lead_points(pull_mm)
        harness_facts = fit.obiwan_terminal_harness_facts(pull_mm)
        assert harness_facts["pull_state_mm"] == pull_mm
        for index, (name, points) in enumerate(lead_points.items(), 1):
            assert _min_three_point_radius(points) >= 8.40
            assert _max_turn_deg(points) <= 2.5
            assert abs(
                harness_facts["lead_lengths_mm"][name]
                - installed_facts["lead_lengths_mm"][name]) < 1e-6
            assert np.linalg.norm(
                np.asarray(points[-1])
                - np.asarray(
                    harness_facts["lead_engagement_points"][name])) < 1e-6
            tangent = np.asarray(points[-1]) - np.asarray(points[-2])
            tangent /= np.linalg.norm(tangent)
            expected = v if index == 1 else -v
            assert float(np.dot(tangent, expected)) > 0.999
        assert (harness_facts["solved_start_handles_mm"]["terminal_lead_1"]
                > 37.0)
        assert (harness_facts["solved_start_handles_mm"]["terminal_lead_2"]
                > 70.0)

    separated = fit.obiwan_separated_lead_parts(0.0)
    assert separated["terminal_lead_1"].distance_to(
        separated["terminal_lead_2"]) >= 0.49
    breakout_parts = fit.obiwan_y_breakout_boot_parts(0.0)
    assert set(breakout_parts) == {
        "obiwan_Y_breakout_bundle_heatshrink",
        "obiwan_Y_breakout_terminal_lead_1_heatshrink",
        "obiwan_Y_breakout_terminal_lead_2_heatshrink",
    }
    assert all(part.is_valid and part.volume > 0.01
               for part in breakout_parts.values())
    for boot_name, boot in breakout_parts.items():
        assert boot.distance_to(body) >= 0.20, boot_name
        assert boot.distance_to(lm_body) >= 0.75, boot_name
        for printed in (lm, um, tweeter):
            collision = boot & printed
            assert collision is None or collision.volume < 0.10, boot_name
    try:
        fit.split_grommet_parts("obiwan")
    except ValueError:
        pass
    else:
        raise AssertionError("Obi-Wan must not generate printed grommet parts")

    for x in core.TWEETER_JOINT_X:
        witness = _complete_tweeter_joint_witnesses(core, x)
        assert (_intersection_volume(um, witness["core_required"])
                > 0.995 * witness["core_required"].volume)
        assert _intersection_volume(tweeter, witness["core_ear"]) < 0.02
        assert (_intersection_volume(tweeter, witness["addon_required"])
                > 0.995 * witness["addon_required"].volume)
        assert _intersection_volume(um, witness["addon_ear"]) < 0.02
        bolt_hit = _intersection_volume(tweeter, witness["core_bolt"])
        insert_hit = _intersection_volume(tweeter, witness["insert"])
        assert bolt_hit < 0.02, (
            f"tweeter refills rear M3 bore at x={x:g} by "
            f"{bolt_hit:.4f} mm3")
        assert insert_hit < 0.02, (
            f"tweeter refills blind insert receiver at x={x:g} by "
            f"{insert_hit:.4f} mm3")
    state = "floor" if stand_foot else "no-floor"
    staging.cleanup()
    print(
        f"  {state}: tweeter ears, free rear T cable, named Faston/boot/"
        "removal proxies and strain relief pass")


_SERVICE_PRINTED_COMPONENT_SPECS = {
    f"printed_component_{owner}_{token}": (owner, family, part_name)
    for owner in ("lm", "um", "tweeter")
    for token, family, part_name in (
        ("harness_bundle", "harness", "obiwan_D7_bundle_to_Y_breakout"),
        ("harness_lead1", "harness", "obiwan_terminal_lead_1_D3p2"),
        ("harness_lead2", "harness", "obiwan_terminal_lead_2_D3p2"),
        ("boot_bundle", "boot", "obiwan_Y_breakout_bundle_heatshrink"),
        ("boot_lead1", "boot",
         "obiwan_Y_breakout_terminal_lead_1_heatshrink"),
        ("boot_lead2", "boot",
         "obiwan_Y_breakout_terminal_lead_2_heatshrink"),
    )
}
_SERVICE_INDEPENDENT_PRINTED_SPECS = {
    f"independent_component_{owner}_t{terminal_id}_p{int(pull_mm)}": (
        owner, terminal_id, pull_mm)
    for owner in ("lm", "um", "tweeter")
    for terminal_id in (1, 2)
    for pull_mm in (0.0, 3.0, 6.0, 9.0, 12.0)
}


_SERVICE_PHASES = (
    "references_numeric",
    "installed_connectors",
    "installed_harness",
    "pull_clearance_1",
    "pull_clearance_2",
    "independent_global_1",
    "independent_global_2",
    "breakout_boot",
    "printed_body_lm",
    "printed_body_um",
    "printed_body_tweeter",
    *_SERVICE_PRINTED_COMPONENT_SPECS,
    *_SERVICE_INDEPENDENT_PRINTED_SPECS,
    "joint_um",
    "joint_tweeter",
)


def _service_paths_from_environment():
    paths = {}
    for name in ("lm", "um", "tweeter"):
        value = os.environ.get(f"LX_R6F_SERVICE_{name.upper()}_PATH")
        if not value:
            raise SystemExit(f"missing isolated service {name} BREP path")
        path = Path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"invalid isolated service {name} BREP: {path}")
        paths[name] = path
    return paths


def _service_reference_numeric(fit):
    import lx521_baffle.obiwan.carriers as core
    import lx521_baffle.obiwan.route as route

    assert fit.PHYSICAL_MEASURE_REQUIRED is True
    body_facts = fit.mu10_body_reference_facts()
    assert body_facts["raw_stl_sha256"] == body_facts[
        "expected_raw_stl_sha256"]
    assert body_facts["terminals_present_in_raw_stl"] is False
    assert body_facts["world_strut_angles_deg"] == (
        13.0, 103.0, 193.0, 283.0)
    full_body = fit.mu10_body_keepout(include_flange=True)
    full_bb = full_body.bounding_box()
    assert math.isclose(full_bb.min.X, -49.0, abs_tol=0.02)
    assert math.isclose(full_bb.max.X, 49.0, abs_tol=0.02)
    assert math.isclose(full_bb.min.Y, 317.081, abs_tol=0.02)
    assert math.isclose(full_bb.max.Y, 415.081, abs_tol=0.02)
    assert math.isclose(full_bb.min.Z, -23.9, abs_tol=0.02)
    assert math.isclose(full_bb.max.Z, 19.7, abs_tol=0.02)

    w22_facts = fit.w22_body_reference_facts()
    assert w22_facts["source_step_sha256"] == w22_facts[
        "expected_source_step_sha256"]
    assert w22_facts["units"] == "mm"
    assert w22_facts["native_to_world"]["rotation"] == {
        "axis": "+X", "degrees": 90.0}
    assert w22_facts["native_to_world"]["axis_map"] == {
        "native_+X": "world_+X",
        "native_+Y_driver_front": "world_+Z_baffle_front",
        "native_+Z": "world_-Y",
    }
    assert w22_facts["world_front_datum_z_mm"] == 18.3
    assert w22_facts["provenance"]["terminals_or_leads_verified"] is False
    assert w22_facts["physical_measure_required"] is True

    terminated_points = fit.obiwan_terminated_cable_points()
    assert fit.OBIWAN_TERMINATED_HANDOFF_R == 20.0
    assert _min_three_point_radius(terminated_points) >= 13.9
    # The physical loft uses 1.5-mm stations to bound OCC memory; its R15
    # chord turn is therefore about six degrees.  Smoothness itself is
    # proved on the 0.2-mm printed path in ``test_route_contract`` and by the
    # exact G1/R20 checks below, rather than conflating chord density with a
    # geometric kink.
    assert _max_turn_deg(terminated_points) <= 6.1
    arc_count = fit.OBIWAN_TERMINATED_HANDOFF_STEPS + 1
    free_arc = np.asarray(terminated_points[-arc_count:], dtype=float)
    route_before = np.asarray(terminated_points[-arc_count - 1], dtype=float)
    phi = np.linspace(0.0, math.pi / 2.0, len(free_arc))
    incoming_plan = free_arc[0] - route_before
    incoming_plan[2] = 0.0
    incoming_plan /= np.linalg.norm(incoming_plan)
    declared_tangent = np.asarray((*route.UM_MOUTH_TANGENT, 0.0))
    declared_tangent /= np.linalg.norm(declared_tangent)
    assert float(np.dot(incoming_plan, declared_tangent)) > 0.998
    free_v = declared_tangent
    plan_normal = np.asarray((-free_v[1], free_v[0], 0.0))
    delta = free_arc - free_arc[0]
    tangential = delta @ free_v
    transverse = delta @ plan_normal
    assert np.max(np.abs(transverse)) < 1e-6
    assert np.max(np.abs(
        tangential - fit.OBIWAN_TERMINATED_HANDOFF_R * np.sin(phi))) < 1e-6
    assert np.max(np.abs(
        free_arc[:, 2] - (route.UM_MOUTH_Z
                          - fit.OBIWAN_TERMINATED_HANDOFF_R
                          * (1.0 - np.cos(phi))))) < 1e-6
    assert _min_three_point_radius(free_arc) >= 19.95
    incoming = free_arc[0] - route_before
    outgoing = free_arc[1] - free_arc[0]
    incoming /= np.linalg.norm(incoming)
    outgoing /= np.linalg.norm(outgoing)
    # The source tangent is exact; these are finite chords on the adjacent
    # R15/R20 samplings and therefore include their bounded half-step angle.
    assert float(np.dot(incoming, outgoing)) > 0.998
    end_tangent = free_arc[-1] - free_arc[-2]
    end_tangent /= np.linalg.norm(end_tangent)
    assert float(np.dot(
        end_tangent, np.asarray((0.0, 0.0, -1.0)))) > 0.999

    terminal_angle = math.radians(283.0)
    v = np.asarray((
        -math.sin(terminal_angle), math.cos(terminal_angle), 0.0))
    installed_facts = fit.obiwan_terminal_harness_facts(0.0)
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        lead_points = fit.obiwan_terminal_lead_points(pull_mm)
        harness_facts = fit.obiwan_terminal_harness_facts(pull_mm)
        assert harness_facts["pull_state_mm"] == pull_mm
        for index, (name, points) in enumerate(lead_points.items(), 1):
            assert _min_three_point_radius(points) >= 8.40
            assert _max_turn_deg(points) <= 2.5
            assert abs(
                harness_facts["lead_lengths_mm"][name]
                - installed_facts["lead_lengths_mm"][name]) < 1e-6
            assert np.linalg.norm(
                np.asarray(points[-1]) - np.asarray(
                    harness_facts["lead_engagement_points"][name])) < 1e-6
            tangent = np.asarray(points[-1]) - np.asarray(points[-2])
            tangent /= np.linalg.norm(tangent)
            expected = v if index == 1 else -v
            assert float(np.dot(tangent, expected)) > 0.999
        assert (harness_facts["solved_start_handles_mm"]["terminal_lead_1"]
                > 37.0)
        assert (harness_facts["solved_start_handles_mm"]["terminal_lead_2"]
                > 70.0)

    states = fit.obiwan_independent_pull_states()
    assert len(states) == 2 * len(fit.FASTON_LEAD_PULL_STATES_MM)
    for state in states:
        active = state["active_terminal_id"]
        installed = state["installed_terminal_id"]
        pulls = state["pull_by_terminal_mm"]
        assert pulls[active] == state["station_mm"]
        assert pulls[installed] == 0.0
        assert state["other_terminal_remains_installed"] is True
        assert state["physical_measure_required"] is True


def _service_installed_connectors(fit):
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    fastons = fit.faston_proxy_parts()
    boots = fit.faston_boot_proxy_parts()
    pull_sweeps = fit.faston_pull_sweep_parts()
    service_parts = {
        "terminal_carrier": fit.terminal_carrier_proxy(),
        **fastons,
        **boots,
        **pull_sweeps,
    }
    assert fastons["faston_receptacle_1"].distance_to(
        fastons["faston_receptacle_2"]) >= 2.49
    assert boots["faston_insulation_boot_1"].distance_to(
        boots["faston_insulation_boot_2"]) >= 1.49
    assert pull_sweeps["faston_flag_pull_sweep_1"].distance_to(
        pull_sweeps["faston_flag_pull_sweep_2"]) >= 1.49
    assert (fit.REMOVAL_ENVELOPE_CABLE_POLICY["obiwan"]
            == "independent_flag_faston_pull_with_slack_leads")
    for service_name, service_part in service_parts.items():
        if service_name in {
                "terminal_carrier", "terminal_tab_1", "terminal_tab_2"}:
            continue
        assert _intersection_volume(service_part, body) < 0.10, (
            f"{service_name}/known MU10 body collision")
    t_cable = fit.obiwan_ts_cable_envelope()
    assert _intersection_volume(t_cable, body) < 0.10
    lm_cable = fit.obiwan_lm_cable_envelope()
    assert _intersection_volume(lm_cable, body) < 0.10
    assert lm_cable.distance_to(lm_body) >= 0.0
    for service_name, service_part in service_parts.items():
        assert _intersection_volume(t_cable, service_part) < 0.10, (
            f"T cable/{service_name} collision")


def _service_installed_harness(fit):
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    carrier = fit.terminal_carrier_proxy()
    fastons = fit.faston_proxy_parts()
    boots = fit.faston_boot_proxy_parts()
    harness = fit.obiwan_terminal_harness_parts()
    bundle = harness["obiwan_D7_bundle_to_Y_breakout"]
    leads = {
        1: harness["obiwan_terminal_lead_1_D3p2"],
        2: harness["obiwan_terminal_lead_2_D3p2"],
    }
    assert bundle.distance_to(body) >= 1.0
    assert bundle.distance_to(lm_body) >= 1.0
    assert bundle.distance_to(carrier) >= 0.75
    for lead in leads.values():
        assert lead.distance_to(body) >= 1.0
        assert lead.distance_to(lm_body) >= 1.0
        assert lead.distance_to(carrier) >= 0.75
    for part in (*fastons.values(), *boots.values()):
        assert _intersection_volume(bundle, part) < 0.01
    for index, lead in leads.items():
        assert _intersection_volume(
            lead, boots[f"faston_insulation_boot_{index}"]) > 0.05
        other = 2 if index == 1 else 1
        assert _intersection_volume(
            lead, boots[f"faston_insulation_boot_{other}"]) < 0.01
        assert _intersection_volume(
            lead, fastons[f"faston_receptacle_{other}"]) < 0.01
    separated = fit.obiwan_separated_lead_parts(0.0)
    assert separated["terminal_lead_1"].distance_to(
        separated["terminal_lead_2"]) >= 0.49


def _service_pull_clearance(fit, terminal_id):
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    fastons = fit.faston_proxy_parts()
    boots = fit.faston_boot_proxy_parts()
    sweep = fit.faston_pull_sweep_parts()[
        f"faston_flag_pull_sweep_{terminal_id}"]
    harness = fit.obiwan_terminal_harness_parts()
    other = 2 if terminal_id == 1 else 1
    other_lead = harness[f"obiwan_terminal_lead_{other}_D3p2"]
    assert sweep.distance_to(body) >= 0.20
    lm_distance = sweep.distance_to(lm_body)
    assert lm_distance >= 1.0, (
        f"terminal {terminal_id} pull sweep/W22 keepout clearance "
        f"{lm_distance:.3f} mm")
    assert _intersection_volume(
        sweep, fastons[f"faston_receptacle_{other}"]) < 0.01
    assert _intersection_volume(
        sweep, fastons[f"terminal_tab_{other}"]) < 0.01
    assert _intersection_volume(
        sweep, boots[f"faston_insulation_boot_{other}"]) < 0.01
    assert _intersection_volume(sweep, other_lead) < 0.01


def _service_independent_global(fit, terminal_id):
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    sweep = fit.faston_pull_sweep_parts()[
        f"faston_flag_pull_sweep_{terminal_id}"]
    other = 2 if terminal_id == 1 else 1
    installed = fit.obiwan_terminal_harness_facts_for_terminal_pull(
        terminal_id, 0.0)
    installed_length = installed["lead_lengths_mm"][
        f"terminal_lead_{terminal_id}"]
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        pulls = fit.obiwan_independent_pull_state(terminal_id, pull_mm)
        facts = fit.obiwan_terminal_service_state_facts(
            terminal_id, pull_mm)
        assert facts["active_terminal_id"] == terminal_id
        assert facts["installed_terminal_id"] == other
        assert facts["pull_by_terminal_mm"] == pulls
        assert facts["other_terminal_remains_installed"] is True
        assert facts["physical_measure_required"] is True
        assert abs(
            facts["harness"]["lead_lengths_mm"][
                f"terminal_lead_{terminal_id}"] - installed_length) < 1e-6
        assert facts["harness"]["pull_by_terminal_mm"][other] == 0.0

        fastons = fit.faston_proxy_parts_by_terminal(pulls)
        boots = fit.faston_boot_proxy_parts_by_terminal(pulls)
        separated = fit.obiwan_separated_lead_parts_by_terminal(pulls)
        moving_lead = separated[f"terminal_lead_{terminal_id}"]
        installed_lead = separated[f"terminal_lead_{other}"]
        receptacle = fastons[f"faston_receptacle_{terminal_id}"]
        insulation = boots[f"faston_insulation_boot_{terminal_id}"]
        assert _intersection_volume(
            receptacle, fastons[f"faston_receptacle_{other}"]) < 0.01
        assert _intersection_volume(
            receptacle, fastons[f"terminal_tab_{other}"]) < 0.01
        assert _intersection_volume(
            receptacle, boots[f"faston_insulation_boot_{other}"]) < 0.01
        assert _intersection_volume(receptacle, installed_lead) < 0.01
        assert _intersection_volume(insulation, installed_lead) < 0.01
        assert _intersection_volume(moving_lead, installed_lead) < 0.01
        assert moving_lead.distance_to(body) >= 1.0
        assert moving_lead.distance_to(lm_body) >= 1.0

        # The static pull envelope must contain every sampled connector and
        # boot position.  At 12 mm the receptacle has zero solid overlap with
        # its provisional 12-mm tab; physical qualification remains required
        # because the modeled release margin is deliberately not invented.
        outside = receptacle - sweep
        assert outside is None or outside.volume < 0.01
        outside = insulation - sweep
        assert outside is None or outside.volume < 0.01
        if math.isclose(pull_mm, fit.FASTON_PULL_DISTANCE, abs_tol=1e-9):
            assert _intersection_volume(
                receptacle, fastons[f"terminal_tab_{terminal_id}"]) < 0.01

        y_parts = fit.obiwan_y_breakout_boot_parts_by_terminal(pulls)
        y_branch = y_parts[
            f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        y_envelope = fit.obiwan_y_breakout_boot_envelope_by_terminal(pulls)
        assert y_envelope.is_valid and len(y_envelope.solids()) == 1
        assert y_branch.distance_to(body) >= 0.20
        assert y_branch.distance_to(lm_body) >= 0.75
        assert _intersection_volume(y_branch, installed_lead) < 0.01
        assert _intersection_volume(
            y_branch, fastons[f"faston_receptacle_{other}"]) < 0.01
        for cable_name, cable in (
                fit.obiwan_y_breakout_cable_parts_by_terminal(pulls).items()):
            remainder = cable - y_envelope
            missing = 0.0 if remainder is None else remainder.volume
            assert missing < 0.02, (
                f"Y state {terminal_id}/{pull_mm:g} misses "
                f"{cable_name} by {missing:.4f} mm3")


def _service_independent_printed(fit, paths, name, terminal_id):
    from build123d import import_brep

    printed = import_brep(paths[name])
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        pulls = fit.obiwan_independent_pull_state(terminal_id, pull_mm)
        lead = fit.obiwan_separated_lead_parts_by_terminal(pulls)[
            f"terminal_lead_{terminal_id}"]
        branch = fit.obiwan_y_breakout_boot_parts_by_terminal(pulls)[
            f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        assert _intersection_volume(printed, lead) < 0.10, (
            f"{name}/terminal {terminal_id} lead at {pull_mm:g} mm")
        assert _intersection_volume(printed, branch) < 0.10, (
            f"{name}/terminal {terminal_id} Y branch at {pull_mm:g} mm")


def _service_breakout_boot(fit):
    from lx521_baffle.cables import _tube_loft

    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    breakout_parts = fit.obiwan_y_breakout_boot_parts(0.0)
    assert set(breakout_parts) == {
        "obiwan_Y_breakout_bundle_heatshrink",
        "obiwan_Y_breakout_terminal_lead_1_heatshrink",
        "obiwan_Y_breakout_terminal_lead_2_heatshrink",
    }
    assert all(part.is_valid and part.volume > 0.01
               for part in breakout_parts.values())
    paths = fit._obiwan_y_breakout_paths_by_terminal(
        fit._uniform_pull_state(0.0))
    bundle_radius = ((fit.FASTON_BREAKOUT_BUNDLE_OD / 2.0)
                     / math.cos(math.pi / 24.0))
    incoming_bundle = _tube_loft(
        paths["bundle"], bundle_radius, sides=24)
    junction = fit._obiwan_y_breakout_junction()
    assert _intersection_volume(incoming_bundle, junction) > 1.0
    for terminal_id in (1, 2):
        branch = breakout_parts[
            f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        assert _intersection_volume(branch, junction) > 1.0
    for boot_name, boot in breakout_parts.items():
        assert boot.distance_to(body) >= 0.20, boot_name
        assert boot.distance_to(lm_body) >= 0.75, boot_name
    breakout = fit.obiwan_y_breakout_boot_envelope(0.0)
    assert breakout.is_valid and len(breakout.solids()) == 1
    cable_parts = fit.obiwan_y_breakout_cable_parts(0.0)
    missing = 0.0
    for cable_name, cable in cable_parts.items():
        remainder = cable - breakout
        volume = 0.0 if remainder is None else remainder.volume
        missing += volume
        assert volume < 0.02, f"Y boot misses {cable_name} by {volume:.4f}"
    facts = fit.obiwan_y_breakout_facts(0.0)
    assert facts["junction_length_mm"] > 0.0
    assert facts["junction_overlap_each_side_mm"] > 0.0
    assert facts["junction_min_underlying_cable_margin_mm"] >= -1e-9
    assert missing < 0.04

    try:
        fit.split_grommet_parts("obiwan")
    except ValueError:
        pass
    else:
        raise AssertionError("Obi-Wan must not generate printed grommet parts")


def _service_printed_body(fit, paths, name):
    from build123d import import_brep
    import lx521_baffle.obiwan.route as route

    printed = import_brep(paths[name])
    assert printed.is_valid and len(printed.solids()) == 1
    body = fit.mu10_body_keepout()
    collision = _intersection_volume(printed, body)
    assert collision < 0.10, f"{name}/known MU10 body collision {collision:.3f}"
    required_clearance = (
        fit.MU10_BODY_MODEL_TOLERANCE_MM
        + fit.MU10_MIN_PRINTED_BODY_CLEARANCE_MM)
    assert printed.distance_to(body) >= required_clearance - 0.03
    service_parts = {
        "terminal_carrier": fit.terminal_carrier_proxy(),
        **fit.faston_proxy_parts(),
        **fit.faston_boot_proxy_parts(),
        **fit.faston_pull_sweep_parts(),
    }
    for service_name, service_part in service_parts.items():
        volume = _intersection_volume(printed, service_part)
        assert volume < 0.10, f"{name}/{service_name} collision {volume:.3f}"
    if name == "tweeter":
        cable = route._round_tube(
            route.ts_cable_points(1.5), route.TS_CABLE_D_EST / 2.0)
        assert _intersection_volume(printed, cable) < 0.10


def _service_printed_harness(fit, paths, name):
    from build123d import import_brep

    printed = import_brep(paths[name])
    for cable_name, cable in fit.obiwan_terminal_harness_parts().items():
        assert _intersection_volume(printed, cable) < 0.10, (
            f"{cable_name}/{name} collision")
    for boot_name, boot in fit.obiwan_y_breakout_boot_parts(0.0).items():
        assert _intersection_volume(printed, boot) < 0.10, (
            f"{boot_name}/{name} collision")


def _service_printed_component(
        fit, paths, name, family, part_name):
    """Check one final carrier against one harness/boot loft per worker."""
    from build123d import import_brep

    printed = import_brep(paths[name])
    pulls = fit._uniform_pull_state(0.0)
    if family == "harness":
        component = fit.obiwan_terminal_harness_part_by_terminal(
            part_name, pulls)
    elif family == "boot":
        component = fit.obiwan_y_breakout_boot_part_by_terminal(
            part_name, pulls)
    else:
        raise ValueError(family)
    assert _intersection_volume(printed, component) < 0.10, (
        f"{part_name}/{name} collision")


def _service_independent_printed_state(
        fit, paths, name, terminal_id, pull_mm):
    """Check one connector pull station without retaining adjacent states."""
    from build123d import import_brep

    printed = import_brep(paths[name])
    pulls = fit.obiwan_independent_pull_state(terminal_id, pull_mm)
    lead = fit.obiwan_separated_lead_part_by_terminal(terminal_id, pulls)
    branch_name = (
        f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
    branch = fit.obiwan_y_breakout_boot_part_by_terminal(branch_name, pulls)
    assert _intersection_volume(printed, lead) < 0.10, (
        f"{name}/terminal {terminal_id} lead at {pull_mm:g} mm")
    assert _intersection_volume(printed, branch) < 0.10, (
        f"{name}/terminal {terminal_id} Y branch at {pull_mm:g} mm")


def _service_component_for_phase(fit, phase, selector):
    """Build exactly one service solid, never a face-rich carrier beside it."""
    if phase in _SERVICE_PRINTED_COMPONENT_SPECS:
        if selector != "component":
            raise ValueError(selector)
        _name, family, part_name = _SERVICE_PRINTED_COMPONENT_SPECS[phase]
        pulls = fit._uniform_pull_state(0.0)
        if family == "harness":
            return fit.obiwan_terminal_harness_part_by_terminal(
                part_name, pulls)
        if family == "boot":
            return fit.obiwan_y_breakout_boot_part_by_terminal(
                part_name, pulls)
        raise ValueError(family)
    if phase in _SERVICE_INDEPENDENT_PRINTED_SPECS:
        _name, terminal_id, pull_mm = (
            _SERVICE_INDEPENDENT_PRINTED_SPECS[phase])
        pulls = fit.obiwan_independent_pull_state(terminal_id, pull_mm)
        if selector == "lead":
            return fit.obiwan_separated_lead_part_by_terminal(
                terminal_id, pulls)
        if selector == "branch":
            part_name = (
                f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
            return fit.obiwan_y_breakout_boot_part_by_terminal(
                part_name, pulls)
        raise ValueError(selector)
    raise ValueError(phase)


def _service_component_chunk_count(phase, selector):
    """Measured bounded-memory tiling for carrier/component intersections."""
    if _large_host_execution():
        return 1
    if phase in _SERVICE_PRINTED_COMPONENT_SPECS:
        _owner, family, part_name = _SERVICE_PRINTED_COMPONENT_SPECS[phase]
        if family == "harness" and part_name.endswith("bundle_to_Y_breakout"):
            return 8
        if family == "harness":
            return 2
        return 1
    if phase in _SERVICE_INDEPENDENT_PRINTED_SPECS:
        return 1
    raise ValueError(phase)


def _run_staged_service_component_phase(phase, staged, stand_foot):
    """Build component and test carrier collision in separate OCC children."""
    if phase in _SERVICE_PRINTED_COMPONENT_SPECS:
        owner = _SERVICE_PRINTED_COMPONENT_SPECS[phase][0]
        selectors = ("component",)
    else:
        owner = _SERVICE_INDEPENDENT_PRINTED_SPECS[phase][0]
        selectors = ("lead", "branch")
    script = str(Path(__file__).resolve())
    with tempfile.TemporaryDirectory(
            prefix="lx-obiwan-service-component-") as directory:
        for selector in selectors:
            chunk_count = _service_component_chunk_count(phase, selector)
            component_path = Path(directory) / (
                f"{selector}.manifest" if chunk_count > 1
                else f"{selector}.brep")
            _wait_for_worker_headroom(
                f"terminal-service {phase}/{selector} build", 3200.0)
            env = os.environ.copy()
            env.pop("LX_R6F_CASE_ID", None)
            env["LX_R6F_SERVICE_COMPONENT_EXPORT"] = phase
            env["LX_R6F_SERVICE_COMPONENT_SELECTOR"] = selector
            env["LX_R6F_SERVICE_COMPONENT_PATH"] = str(component_path)
            env["LX_R6F_SERVICE_COMPONENT_CHUNK_COUNT"] = str(chunk_count)
            env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
            proc = subprocess.run(
                _cad_worker_command(script), env=env, text=True,
                capture_output=True)
            assert proc.returncode == 0, (
                f"service component build {phase}/{selector} failed\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
            assert component_path.is_file() and component_path.stat().st_size
            print(proc.stdout, end="", flush=True)
            if chunk_count > 1:
                component_paths = tuple(
                    Path(line) for line in component_path.read_text(
                        encoding="utf-8").splitlines() if line)
                assert len(component_paths) == chunk_count
                assert all(path.is_file() and path.stat().st_size
                           for path in component_paths)
            else:
                component_paths = (component_path,)

            for chunk_index, chunk_path in enumerate(component_paths):
                _wait_for_worker_headroom(
                    f"terminal-service {phase}/{selector} collision "
                    f"{chunk_index + 1}/{len(component_paths)}", 3200.0)
                env = os.environ.copy()
                env.pop("LX_R6F_CASE_ID", None)
                env["LX_R6F_SERVICE_COMPONENT_COLLISION"] = phase
                env["LX_R6F_SERVICE_COMPONENT_SELECTOR"] = selector
                env["LX_R6F_SERVICE_COMPONENT_PATH"] = str(chunk_path)
                env["LX_R6F_SERVICE_PRINTED_PATH"] = str(staged[owner])
                env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
                proc = subprocess.run(
                    _cad_worker_command(script), env=env, text=True,
                    capture_output=True)
                assert proc.returncode == 0, (
                    f"service component collision {phase}/{selector} "
                    f"chunk {chunk_index + 1} failed\n"
                    f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
                print(proc.stdout, end="", flush=True)


def _service_joint(fit, paths, name):
    from build123d import import_brep
    import lx521_baffle.obiwan.carriers as core

    part = import_brep(paths[name])
    if name == "tweeter":
        um = import_brep(paths["um"])
        actual_overlap = _intersection_volume(part, um)
        assert actual_overlap < 0.02, (
            f"actual UM/tweeter print collision {actual_overlap:.4f} mm3")
    for x in core.TWEETER_JOINT_X:
        witness = _complete_tweeter_joint_witnesses(core, x)
        if name == "um":
            core_hit = _intersection_volume(
                part, witness["core_required"])
            assert core_hit > 0.995 * witness["core_required"].volume, (
                f"UM/core owned tweeter ear at x={x:g}: {core_hit:.4f}/"
                f"{witness['core_required'].volume:.4f} mm3")
            addon_hit = _intersection_volume(part, witness["addon_ear"])
            assert addon_hit < 0.02, (
                f"UM enters add-on tweeter ear at x={x:g} by "
                f"{addon_hit:.4f} mm3")
        else:
            core_hit = _intersection_volume(part, witness["core_ear"])
            assert core_hit < 0.02, (
                f"tweeter owned core-ear intrusion at x={x:g} is "
                f"{core_hit:.4f} mm3")
            addon_hit = _intersection_volume(
                part, witness["addon_required"])
            assert addon_hit > 0.995 * witness["addon_required"].volume, (
                f"tweeter/add-on owned ear at x={x:g}: {addon_hit:.4f}/"
                f"{witness['addon_required'].volume:.4f} mm3")
            bolt_hit = _intersection_volume(part, witness["core_bolt"])
            insert_hit = _intersection_volume(part, witness["insert"])
            assert bolt_hit < 0.02, (
                f"tweeter refills rear M3 bore at x={x:g} by "
                f"{bolt_hit:.4f} mm3")
            assert insert_hit < 0.02, (
                f"tweeter refills blind insert receiver at x={x:g} by "
                f"{insert_hit:.4f} mm3")


def _service_phase_worker(phase, paths, stand_foot):
    _state(stand_foot)
    import lx521_baffle.um_fit as fit

    if phase == "references_numeric":
        _service_reference_numeric(fit)
    elif phase == "installed_connectors":
        _service_installed_connectors(fit)
    elif phase == "installed_harness":
        _service_installed_harness(fit)
    elif phase.startswith("pull_clearance_"):
        _service_pull_clearance(fit, int(phase.rsplit("_", 1)[1]))
    elif phase.startswith("independent_global_"):
        _service_independent_global(fit, int(phase.rsplit("_", 1)[1]))
    elif phase == "breakout_boot":
        _service_breakout_boot(fit)
    elif phase.startswith("printed_body_"):
        _service_printed_body(fit, paths, phase.removeprefix("printed_body_"))
    elif phase.startswith("joint_"):
        _service_joint(fit, paths, phase.removeprefix("joint_"))
    else:
        raise SystemExit(f"unknown terminal-service phase: {phase}")
    print(f"isolated terminal-service phase passed: {phase}", flush=True)


def _run_service_phase(phase, staged, stand_foot):
    if (phase in _SERVICE_PRINTED_COMPONENT_SPECS
            or phase in _SERVICE_INDEPENDENT_PRINTED_SPECS):
        _run_staged_service_component_phase(phase, staged, stand_foot)
        print(
            f"isolated staged terminal-service phase passed: {phase}",
            flush=True)
        return
    if phase == "references_numeric":
        launch_headroom = 1800.0
    else:
        launch_headroom = 2200.0
    _wait_for_worker_headroom(
        f"terminal-service {phase}",
        launch_headroom)
    env = os.environ.copy()
    env.pop("LX_R6F_CASE_ID", None)
    env["LX_R6F_SERVICE_PHASE"] = phase
    env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
    for name in ("lm", "um", "tweeter"):
        env[f"LX_R6F_SERVICE_{name.upper()}_PATH"] = str(staged[name])
    proc = subprocess.run(
        _cad_worker_command(Path(__file__).resolve()),
        env=env, text=True, capture_output=True)
    assert proc.returncode == 0, (
        f"isolated terminal-service phase {phase} failed\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    print(proc.stdout, end="", flush=True)


def _run_large_host_service_matrix(staged, stand_foot):
    """Run the complete terminal matrix once in a large guarded process.

    The local implementation below deliberately rebuilds/imports one state
    at a time to protect the workstation floor. Osado instead reuses the
    three final carriers and each unique service component, while preserving
    every collision and pull-state assertion.
    """
    _state(stand_foot)
    from build123d import import_brep
    import lx521_baffle.um_fit as fit

    _service_reference_numeric(fit)
    _service_installed_connectors(fit)
    _service_installed_harness(fit)
    for terminal_id in (1, 2):
        _service_pull_clearance(fit, terminal_id)
        _service_independent_global(fit, terminal_id)
    _service_breakout_boot(fit)
    for name in ("lm", "um", "tweeter"):
        _service_printed_body(fit, staged, name)

    printed = {
        name: import_brep(staged[name])
        for name in ("lm", "um", "tweeter")
    }
    pulls = fit._uniform_pull_state(0.0)
    component_specs = {
        (family, part_name)
        for _owner, family, part_name
        in _SERVICE_PRINTED_COMPONENT_SPECS.values()
    }
    for family, part_name in sorted(component_specs):
        if family == "harness":
            component = fit.obiwan_terminal_harness_part_by_terminal(
                part_name, pulls)
        else:
            assert family == "boot"
            component = fit.obiwan_y_breakout_boot_part_by_terminal(
                part_name, pulls)
        assert component.is_valid and component.volume > 0.01
        for owner, carrier in printed.items():
            collision = _intersection_volume(carrier, component)
            assert collision < 0.10, (
                f"{owner}/{part_name} collision {collision:.3f} mm3")

    for terminal_id in (1, 2):
        branch_name = (
            f"obiwan_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
        for pull_mm in (0.0, 3.0, 6.0, 9.0, 12.0):
            independent = fit.obiwan_independent_pull_state(
                terminal_id, pull_mm)
            lead = fit.obiwan_separated_lead_part_by_terminal(
                terminal_id, independent)
            branch = fit.obiwan_y_breakout_boot_part_by_terminal(
                branch_name, independent)
            for owner, carrier in printed.items():
                lead_collision = _intersection_volume(carrier, lead)
                branch_collision = _intersection_volume(carrier, branch)
                assert lead_collision < 0.10, (
                    f"{owner}/terminal {terminal_id} lead at {pull_mm:g} "
                    f"mm collision {lead_collision:.3f} mm3")
                assert branch_collision < 0.10, (
                    f"{owner}/terminal {terminal_id} branch at {pull_mm:g} "
                    f"mm collision {branch_collision:.3f} mm3")

    _service_joint(fit, staged, "um")
    _service_joint(fit, staged, "tweeter")
    print(
        "complete large-host terminal service matrix passed: installed, "
        "12 mm independent pulls, all printed owners, Y boot and joints",
        flush=True)


def _tweeter_and_service(stand_foot):
    _state(stand_foot)
    phases = _SERVICE_PHASES
    start_phase = os.environ.get("LX_R6F_SERVICE_START_PHASE")
    if start_phase:
        if start_phase not in phases:
            raise ValueError(
                f"unknown LX_R6F_SERVICE_START_PHASE: {start_phase}")
        phases = phases[phases.index(start_phase):]
    stop_phase = os.environ.get("LX_R6F_SERVICE_STOP_AFTER_PHASE")
    if stop_phase:
        if stop_phase not in phases:
            raise ValueError(
                "unknown/out-of-order LX_R6F_SERVICE_STOP_AFTER_PHASE: "
                f"{stop_phase}")
        phases = phases[:phases.index(stop_phase) + 1]
    with tempfile.TemporaryDirectory(prefix="lx-obiwan-service-") as directory:
        staged = _stage_shell_contract_breps(
            stand_foot, "T", directory, shell_keys=())
        if _large_host_execution() and phases == _SERVICE_PHASES:
            _run_large_host_service_matrix(staged, stand_foot)
        else:
            for phase in phases:
                _run_service_phase(phase, staged, stand_foot)
    state = "floor" if stand_foot else "no-floor"
    if phases == _SERVICE_PHASES:
        print(
            f"  {state}: isolated tweeter/body/harness/Faston/Y-boot/"
            "joint service contract passes")
    else:
        print(
            f"  {state}: isolated service phase slice passes: "
            f"{phases[0]}..{phases[-1]}")









def _case(
    case_id, function, *args, stand_state,
    service_orchestrator_class=GUARDED_CASE,
):
    return GuardedCase(
        case_id=case_id,
        function=function,
        args=tuple(args),
        stand_state=stand_state,
        service_orchestrator_class=service_orchestrator_class,
        make_stamp=case_id,
        legacy_selector=f"test_{case_id}",
    )


CASES = (
    _case("route_contract", test_route_contract, stand_state=False),
    _case("w22_reference_step_geometry", test_w22_reference_step_geometry,
          stand_state=False),
    _case("insert_bump_clearance", _insert_bump_clearance, False,
          stand_state=False),
    _case("floor_insert_bump_clearance", _insert_bump_clearance, True,
          stand_state=True),
    _case("no_floor_route_smoothness", test_no_floor_route_smoothness,
          stand_state=False),
    _case("floor_route_smoothness", test_floor_route_smoothness,
          stand_state=True),
    _case("bump_brep_clearance", _bump_brep_clearance, False,
          stand_state=False),
    _case("floor_bump_brep_clearance", _bump_brep_clearance, True,
          stand_state=True),
    _case("bump_backfill_contract", _final_bump_backfill_contract, False,
          stand_state=False),
    _case("floor_bump_backfill_contract", _final_bump_backfill_contract, True,
          stand_state=True),
    _case("lm_burial_web_contract", _final_lm_burial_web_contract, False,
          stand_state=False),
    _case("floor_lm_burial_web_contract", _final_lm_burial_web_contract, True,
          stand_state=True),
    _case("um_burial_web_contract", _final_um_burial_web_contract, False,
          stand_state=False),
    _case("floor_um_burial_web_contract", _final_um_burial_web_contract, True,
          stand_state=True),
    _case("feed_and_flush_mouth_contract",
          _final_feed_and_flush_mouth_contract, False, stand_state=False),
    _case("floor_feed_and_flush_mouth_contract",
          _final_feed_and_flush_mouth_contract, True, stand_state=True),
    _case("crossover_brep", _crossover_brep, False, stand_state=False),
    _case("floor_crossover_brep", _crossover_brep, True, stand_state=True),
    _case("bridge_contract", test_bridge_contract, stand_state=False),
    _case("bridge_geometry", test_bridge_geometry, stand_state=False),
    _case("joint_load_contract", test_joint_load_contract, stand_state=False),
    _case("um_driver_spoke_is_separate_from_lm_um_insert_ear",
          test_um_driver_spoke_is_separate_from_lm_um_insert_ear,
          stand_state=False),
    _case("floor_lm_core", test_floor_lm_core, stand_state=True),
    _case("no_floor_lm_core", test_no_floor_lm_core, stand_state=False),
    _case("floor_lm_keyed_split", _assert_lm_keyed_split, True,
          stand_state=True),
    _case("no_floor_lm_keyed_split", _assert_lm_keyed_split, False,
          stand_state=False),
    _case("floor_um_shell", _assembled_shell_contract, True, "UM",
          stand_state=True),
    _case("floor_t_shell", _assembled_shell_contract, True, "T",
          stand_state=True),
    _case("no_floor_um_shell", _assembled_shell_contract, False, "UM",
          stand_state=False),
    _case("no_floor_t_shell", _assembled_shell_contract, False, "T",
          stand_state=False),
    _case("lm_cable_clearance", _carrier_cable_clearance, "lm", False,
          stand_state=False),
    _case("um_cable_clearance", _carrier_cable_clearance, "um", False,
          stand_state=False),
    _case("floor_lm_cable_clearance", _carrier_cable_clearance, "lm", True,
          stand_state=True),
    _case("floor_um_cable_clearance", _carrier_cable_clearance, "um", True,
          stand_state=True),
    _case("floor_integrated_mount", test_floor_integrated_mount,
          stand_state=True),
    _case("tweeter_and_service", _tweeter_and_service, False,
          stand_state=False,
          service_orchestrator_class=SERVICE_ORCHESTRATOR_CASE),
    _case("floor_tweeter_and_service", _tweeter_and_service, True,
          stand_state=True,
          service_orchestrator_class=SERVICE_ORCHESTRATOR_CASE),
)


def main():
    component_export_phase = os.environ.get(
        "LX_R6F_SERVICE_COMPONENT_EXPORT")
    if component_export_phase:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "service-component export escaped the CAD memory guard")
        target = os.environ.get("LX_R6F_SERVICE_COMPONENT_PATH")
        selector = os.environ.get("LX_R6F_SERVICE_COMPONENT_SELECTOR")
        chunk_count = int(os.environ.get(
            "LX_R6F_SERVICE_COMPONENT_CHUNK_COUNT", "1"))
        if not target or not selector:
            raise SystemExit("service-component export paths are required")
        if chunk_count < 1:
            raise SystemExit("service-component chunk count must be positive")
        _state(os.environ.get("LX_STAND_FOOT") == "1")
        from build123d import Box, Pos, export_brep
        import lx521_baffle.um_fit as fit
        component = _service_component_for_phase(
            fit, component_export_phase, selector)
        if not component.is_valid:
            raise SystemExit("invalid service component")
        if chunk_count == 1:
            if not export_brep(component, target):
                raise SystemExit(
                    f"failed service component export: "
                    f"{component_export_phase}/{selector}")
        else:
            bb = component.bounding_box()
            lower = [bb.min.X, bb.min.Y, bb.min.Z]
            upper = [bb.max.X, bb.max.Y, bb.max.Z]
            spans = [upper[index] - lower[index] for index in range(3)]
            axis = max(range(3), key=spans.__getitem__)
            step = spans[axis] / chunk_count
            chunk_paths = []
            target_path = Path(target)
            for index in range(chunk_count):
                tile_lower = [value - 1.0 for value in lower]
                tile_upper = [value + 1.0 for value in upper]
                tile_lower[axis] = (
                    lower[axis] + index * step
                    - (0.05 if index else 0.0))
                tile_upper[axis] = (
                    lower[axis] + (index + 1) * step
                    + (0.05 if index + 1 < chunk_count else 0.0))
                size = [tile_upper[i] - tile_lower[i] for i in range(3)]
                center = [
                    (tile_lower[i] + tile_upper[i]) / 2.0
                    for i in range(3)]
                chunk = component & (Pos(*center) * Box(*size))
                if chunk is None or chunk.volume <= 1e-9:
                    raise SystemExit(
                        f"empty service component chunk {index + 1}")
                chunk_path = target_path.with_name(
                    f"{target_path.stem}_chunk_{index:02d}.brep")
                if not export_brep(chunk, chunk_path):
                    raise SystemExit(
                        f"failed service component chunk {index + 1}")
                chunk_paths.append(chunk_path)
            target_path.write_text(
                "".join(f"{path}\n" for path in chunk_paths),
                encoding="utf-8")
        print(
            f"isolated service component exported: "
            f"{component_export_phase}/{selector} "
            f"{component.volume:.3f} mm3 in {chunk_count} chunk(s)",
            flush=True)
        return

    component_collision_phase = os.environ.get(
        "LX_R6F_SERVICE_COMPONENT_COLLISION")
    if component_collision_phase:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "service-component collision escaped the CAD memory guard")
        component_path = os.environ.get("LX_R6F_SERVICE_COMPONENT_PATH")
        printed_path = os.environ.get("LX_R6F_SERVICE_PRINTED_PATH")
        selector = os.environ.get("LX_R6F_SERVICE_COMPONENT_SELECTOR")
        if not component_path or not printed_path or not selector:
            raise SystemExit("service-component collision paths are required")
        from build123d import import_brep
        component = import_brep(component_path)
        printed = import_brep(printed_path)
        collision = _intersection_volume(component, printed)
        if collision >= 0.10:
            raise SystemExit(
                f"{component_collision_phase}/{selector} collision "
                f"{collision:.3f} mm3")
        print(
            f"isolated service component collision clear: "
            f"{component_collision_phase}/{selector} "
            f"{collision:.6f} mm3", flush=True)
        return

    service_phase = os.environ.get("LX_R6F_SERVICE_PHASE")
    if service_phase:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "isolated terminal-service phase escaped the CAD memory guard")
        _service_phase_worker(
            service_phase,
            _service_paths_from_environment(),
            os.environ.get("LX_STAND_FOOT") == "1")
        return

    carrier_export = os.environ.get("LX_R6F_EXPORT_CARRIER")
    shell_export = os.environ.get("LX_R6F_EXPORT_SHELL")
    tweeter_export = os.environ.get("LX_R6F_EXPORT_TWEETER")
    subtract_input = os.environ.get("LX_R6F_SUBTRACT_INPUT")
    shell_validation = os.environ.get("LX_R6F_VALIDATE_SHELL")
    cable_validation = os.environ.get("LX_R6F_VALIDATE_CABLE")
    if cable_validation == "ALL":
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "complete cable validation escaped the CAD memory guard")
        carrier_path = os.environ.get("LX_R6F_VALIDATE_CARRIER_PATH")
        owner = os.environ.get("LX_R6F_VALIDATE_CARRIER_OWNER")
        if not carrier_path or owner not in {"lm", "um"}:
            raise SystemExit("complete cable validation inputs are required")
        _state(os.environ.get("LX_STAND_FOOT") == "1")
        from build123d import Box, Pos, import_brep
        import lx521_baffle.obiwan.route as route

        carrier = import_brep(carrier_path)
        specs = {
            "UM_D7": (
                route.route_cable_points(1.5),
                route.CABLE_R_EST, route.CUTTER_R),
            "LM_D7p8": (
                route.lm_complete_duct_points(1.0),
                route.LM_CABLE_D_EST / 2.0, route.LM_REAR_PORT_R),
            "T_D5p2": (
                route.ts_cable_points(1.5),
                route.TS_CABLE_D_EST / 2.0, route.TS_CUTTER_R),
        }
        for cable_name, (points, physical_radius, nominal_radius) in (
                specs.items()):
            # LM_D7p8 now covers the complete printed D9 path, including its
            # R14 rear handoff.  Overflow and both carrier-collision witnesses
            # are mandatory so an open-but-unfishable L-junction cannot pass.
            modes = (("collision", "witness") if nominal_radius is None
                     else ("overflow", "collision", "witness"))
            for mode in modes:
                if mode == "overflow":
                    cable = route._round_tube(points, physical_radius)
                    nominal = route._round_tube(points, nominal_radius)
                    overflow = cable - nominal
                    volume = sum(
                        solid.volume for solid in overflow.solids())
                    limit = 0.01
                    description = "nominal overflow"
                else:
                    radius = (physical_radius if mode == "collision"
                              else physical_radius + 0.05)
                    cable = route._round_tube(points, radius)
                    hit = carrier & cable
                    volume = 0.0 if hit is None else hit.volume
                    limit = 0.10
                    description = (
                        "carrier collision" if mode == "collision"
                        else "+0.05 mm witness collision")
                if volume >= limit:
                    raise SystemExit(
                        f"{owner}/{cable_name} {description} "
                        f"{volume:.6f} mm3 >= {limit:.2f} mm3")
                print(
                    f"complete cable clear: {owner}/{cable_name}/{mode} "
                    f"volume={volume:.6f} mm3", flush=True)
        if owner == "lm":
            # The D7.8 lead exits through the gradual D9/R14 handoff.  It must
            # be empty and open through the applicable rear exterior plane,
            # proving a usable cable opening rather than a sealed lumen.
            port = route.lm_rear_exit_port_cutter()
            port_hit = carrier & port
            port_hit_volume = (
                0.0 if port_hit is None else port_hit.volume)
            if port_hit_volume >= 0.10:
                raise SystemExit(
                    f"lm/R14 rear port retained {port_hit_volume:.6f} "
                    "mm3 of carrier material")
            stand_foot = os.environ.get("LX_STAND_FOOT") == "1"
            rear_z = (route.STEM_Z_MM[0]
                      if stand_foot else route.PAD_FACE_Z)
            bounds = port.bounding_box()
            rear_plane_slab = Pos(
                (bounds.min.X + bounds.max.X) / 2.0,
                (bounds.min.Y + bounds.max.Y) / 2.0,
                rear_z,
            ) * Box(
                bounds.max.X - bounds.min.X + 2.0,
                bounds.max.Y - bounds.min.Y + 2.0,
                0.04,
            )
            rear_aperture = port & rear_plane_slab
            rear_aperture_volume = (
                0.0 if rear_aperture is None else rear_aperture.volume)
            blocked_aperture = carrier & rear_aperture
            blocked_aperture_volume = (
                0.0 if blocked_aperture is None
                else blocked_aperture.volume)
            if rear_aperture_volume <= 0.10:
                raise SystemExit(
                    f"lm/R14 rear port does not cross the {rear_z:.2f} "
                    "mm rear exterior plane")
            if blocked_aperture_volume >= 0.01:
                raise SystemExit(
                    f"lm/R14 rear aperture blocked by "
                    f"{blocked_aperture_volume:.6f} mm3")
            state = "floor" if stand_foot else "no-floor"
            print(
                f"complete LM R14 rear port clear: {state}; rear-plane "
                f"aperture={rear_aperture_volume:.6f} mm3", flush=True)
        return
    if cable_validation:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "isolated cable validation escaped the CAD memory guard")
        carrier_path = os.environ.get("LX_R6F_VALIDATE_CARRIER_PATH")
        owner = os.environ.get("LX_R6F_VALIDATE_CARRIER_OWNER")
        mode = os.environ.get("LX_R6F_VALIDATE_CABLE_MODE")
        segment_text = os.environ.get("LX_R6F_VALIDATE_CABLE_SEGMENT")
        segment_count_text = os.environ.get(
            "LX_R6F_VALIDATE_CABLE_SEGMENT_COUNT")
        if not carrier_path or owner not in {"lm", "um"}:
            raise SystemExit("isolated cable validation inputs are required")
        if mode not in {"overflow", "collision", "witness"}:
            raise SystemExit("isolated cable validation mode is required")
        try:
            segment_index = int(segment_text)
            segment_count = int(segment_count_text)
        except (TypeError, ValueError):
            raise SystemExit(
                "isolated cable validation segment is required") from None
        if not 0 <= segment_index < segment_count:
            raise SystemExit("isolated cable validation segment is invalid")
        _state(os.environ.get("LX_STAND_FOOT") == "1")
        from build123d import import_brep
        import lx521_baffle.obiwan.route as route

        specs = {
            "UM_D7": (
                route.route_cable_points(1.5),
                route.CABLE_R_EST, route.CUTTER_R),
            "LM_D7p8": (
                route.lm_complete_duct_points(1.0),
                route.LM_CABLE_D_EST / 2.0, route.LM_REAR_PORT_R),
            "T_D5p2": (
                route.ts_cable_points(1.5),
                route.TS_CABLE_D_EST / 2.0, route.TS_CUTTER_R),
        }
        if cable_validation not in specs:
            raise SystemExit(f"unknown cable validation: {cable_validation}")
        points, physical_radius, nominal_radius = specs[cable_validation]
        if mode == "overflow" and nominal_radius is None:
            raise SystemExit("selected cable has no nominal overflow contract")
        def cable_segment(radius):
            if segment_count == 1:
                return route._round_tube(points, radius)
            return route._round_tube_global_segment(
                points, radius, segment_index, segment_count)

        if mode == "overflow":
            cable = cable_segment(physical_radius)
            nominal = cable_segment(nominal_radius)
            overflow = cable - nominal
            volume = sum(solid.volume for solid in overflow.solids())
            limit = 0.01
            description = "nominal overflow"
        else:
            carrier = import_brep(carrier_path)
            radius = (physical_radius if mode == "collision"
                      else physical_radius + 0.05)
            cable = cable_segment(radius)
            hit = carrier & cable
            volume = 0.0 if hit is None else hit.volume
            limit = 0.10
            description = ("carrier collision" if mode == "collision"
                           else "+0.05 mm witness collision")
        if volume >= limit:
            raise SystemExit(
                f"{owner}/{cable_validation} {description} "
                f"{volume:.6f} mm3 >= {limit:.2f} mm3")
        print(
            f"isolated cable clear: {owner}/{cable_validation}/{mode} "
            f"segment={segment_index + 1}/{segment_count} "
            f"volume={volume:.6f} mm3",
            flush=True)
        return
    if subtract_input:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "isolated BREP subtraction escaped the CAD memory guard")
        part_path = os.environ.get("LX_R6F_SUBTRACT_PART")
        output_path = os.environ.get("LX_R6F_SUBTRACT_OUTPUT")
        if not part_path or not output_path:
            raise SystemExit("isolated BREP subtraction paths are required")
        from build123d import Compound, export_brep, import_brep
        source = import_brep(subtract_input)
        part = import_brep(part_path)
        remaining = []
        for solid in source.solids():
            difference = solid - part
            if difference is not None:
                remaining.extend(
                    candidate for candidate in difference.solids()
                    if candidate.volume > 1e-9)
        remaining_volume = sum(solid.volume for solid in remaining)
        bbox_text = ""
        if remaining:
            result = Compound(children=remaining)
            if not export_brep(result, output_path):
                raise SystemExit("failed to export isolated BREP remainder")
            bb = result.bounding_box()
            bbox_text = (
                f" bbox=({bb.min.X:.3f},{bb.min.Y:.3f},{bb.min.Z:.3f}).."
                f"({bb.max.X:.3f},{bb.max.Y:.3f},{bb.max.Z:.3f})")
        print(
            f"isolated subtraction: remaining={remaining_volume:.9g}"
            f"{bbox_text}", flush=True)
        return

    if shell_validation:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "isolated shell validation escaped the CAD memory guard")
        from build123d import import_brep
        shell_text = os.environ.get("LX_R6F_VALIDATE_SHELL_PATHS")
        if not shell_text:
            shell_text = os.environ.get("LX_R6F_VALIDATE_SHELL_PATH")
        part_text = os.environ.get("LX_R6F_VALIDATE_PART_PATHS", "")
        if not shell_text or not part_text:
            raise SystemExit("isolated shell validation paths are required")
        shell_paths = shell_text.split(os.pathsep)
        parts = [import_brep(path) for path in part_text.split(os.pathsep)]
        missing_volume = 0.0
        missing_bounds = []
        for shell_path in shell_paths:
            shell_shape = import_brep(shell_path)
            for shell in shell_shape.solids():
                remaining = [shell]
                for part in parts:
                    next_remaining = []
                    for solid in remaining:
                        difference = solid - part
                        if difference is not None:
                            next_remaining.extend(
                                candidate
                                for candidate in difference.solids()
                                if candidate.volume > 1e-9)
                    remaining = next_remaining
                    if not remaining:
                        break
                missing_volume += sum(
                    solid.volume for solid in remaining)
                missing_bounds.extend(
                    solid.bounding_box() for solid in remaining)
        bbox_text = ""
        if missing_bounds:
            bbox_text = (
                " bbox=("
                f"{min(bb.min.X for bb in missing_bounds):.3f},"
                f"{min(bb.min.Y for bb in missing_bounds):.3f},"
                f"{min(bb.min.Z for bb in missing_bounds):.3f})..("
                f"{max(bb.max.X for bb in missing_bounds):.3f},"
                f"{max(bb.max.Y for bb in missing_bounds):.3f},"
                f"{max(bb.max.Z for bb in missing_bounds):.3f})")
        print(
            f"isolated {shell_validation} validation: "
            f"missing={missing_volume:.9g}{bbox_text}", flush=True)
        return

    if carrier_export or shell_export or tweeter_export:
        import run_memory_guarded as memory_guard
        if not memory_guard.is_guarded_process():
            raise SystemExit(
                "isolated BREP export escaped the active CAD memory guard")
        target = os.environ.get("LX_R6F_EXPORT_PATH")
        if not target:
            raise SystemExit("LX_R6F_EXPORT_PATH is required")
        _state(os.environ.get("LX_STAND_FOOT") == "1")
        from build123d import export_brep
        if carrier_export:
            valid_carrier_exports = {
                "lm", "um", "lm_outer", "lm_finalize",
                *(f"lm_cut_{index}"
                  for index in range(LM_CUTTER_GROUP_COUNT)),
            }
            if carrier_export not in valid_carrier_exports:
                raise SystemExit(f"unknown carrier export: {carrier_export}")
            import lx521_baffle.obiwan.carriers as core
            if carrier_export == "lm":
                part = core.lm_carrier()
                label = "LM carrier"
            elif carrier_export == "um":
                part = core.um_carrier()
                label = "UM carrier"
            elif carrier_export == "lm_outer":
                part = core.lm_carrier_outer_blank()
                label = "LM solid outer blank"
            elif carrier_export.startswith("lm_cut_"):
                input_path = os.environ.get("LX_R6F_LM_INPUT_PATH")
                if not input_path:
                    raise SystemExit("LX_R6F_LM_INPUT_PATH is required")
                from build123d import import_brep
                index = int(carrier_export.rsplit("_", 1)[1])
                part = core.apply_lm_route_cutter(
                    import_brep(input_path), index)
                label = f"LM cutter group {index}"
            else:
                input_path = os.environ.get("LX_R6F_LM_INPUT_PATH")
                if not input_path:
                    raise SystemExit("LX_R6F_LM_INPUT_PATH is required")
                from build123d import import_brep
                part = core.finalize_lm_carrier(
                    import_brep(input_path), routes_already_cut=True)
                label = "LM finalized carrier"
        elif shell_export:
            if shell_export not in ("LM", "UM", "T"):
                raise SystemExit(f"unknown shell export: {shell_export}")
            import lx521_baffle.obiwan.route as route
            wall_text = os.environ.get(
                "LX_R6F_EXPORT_SHELL_WALL", "nominal")
            wall = None if wall_text == "nominal" else float(wall_text)
            segment_text = os.environ.get("LX_R6F_EXPORT_SHELL_SEGMENT")
            segment_count_text = os.environ.get(
                "LX_R6F_EXPORT_SHELL_SEGMENT_COUNT")
            if segment_text is None or segment_count_text is None:
                components = route.required_assembled_shell_components(
                    shell_export, normal_wall_mm=wall)
                segment_label = "full"
            else:
                segment_index = int(segment_text)
                segment_count = int(segment_count_text)
                components = (
                    route.required_assembled_shell_segment_components(
                        shell_export, segment_index, segment_count,
                        normal_wall_mm=wall))
                segment_label = f"{segment_index + 1}/{segment_count}"
            label = f"{shell_export} shell {wall_text}"
            # The contract already returns disjoint exact solids.  Exporting
            # them directly avoids rebuilding their union and intersecting it
            # with a tile grid, which doubled the OCC peak without adding any
            # geometric information.  Validation still subtracts every
            # exact solid from each final owner in a short-lived worker.
            chunk_paths = []
            chunk_volume = 0.0
            target_path = Path(target)
            for component in components:
                chunk_path = target_path.with_name(
                    f"{target_path.stem}_chunk_{len(chunk_paths):02d}.brep")
                if not export_brep(component, chunk_path):
                    raise SystemExit(
                        f"failed to export isolated {label} chunk")
                chunk_paths.append(chunk_path)
                chunk_volume += component.volume
            if not chunk_paths:
                raise SystemExit(f"isolated {label} produced no shell chunks")
            target_path.write_text(
                "".join(f"{path}\n" for path in chunk_paths),
                encoding="utf-8")
            print(
                f"isolated {label} segment {segment_label}: "
                f"{chunk_volume:.1f} mm3 in "
                f"{len(chunk_paths)} exact chunks", flush=True)
            return
        elif tweeter_export:
            import lx521_baffle.obiwan.attachments as addons
            part = addons.tweeter_crescent()
            label = "tweeter crescent"
        if not export_brep(part, target):
            raise SystemExit(f"failed to export isolated {label} BREP")
        print(f"isolated {label}: {part.volume:.1f} mm3", flush=True)
        return

    case_id = os.environ.get("LX_R6F_CASE_ID")
    guard = PROJECT_ROOT / "scripts/run_memory_guarded.py"
    if case_id:
        import run_memory_guarded as memory_guard

        def before_case(case):
            # Starting at a profile's hard floor can force the guard to kill
            # a later BREP comparison. This is launch hysteresis, not a
            # relaxation of the selected profile's kill floor.
            _wait_for_worker_headroom(
                f"R6F case {case.case_id}",
                R6F_CHECK_LAUNCH_HEADROOM_MB)

        run_selected_case(
            CASES,
            case_id,
            script=Path(__file__).resolve(),
            guard=guard,
            is_guarded_process=memory_guard.is_guarded_process,
            large_host=_large_host_execution(),
            before_case=before_case,
        )
        return

    run_suite(
        CASES,
        script=Path(__file__).resolve(),
        guard=guard,
        selector_env="LX_R6F_CASE_ID",
        private_env_prefix="LX_R6F_",
        requested_workers=int(os.environ.get("LX_CAD_GUARD_SLOTS", "1")),
        large_host=_large_host_execution(),
        suite_label="R6F",
        success_message="all final Obi-Wan R6F checks passed",
    )


if __name__ == "__main__":
    main()
