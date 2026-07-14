"""Final R6F V1LF acceptance checks.

Each OCC-heavy check runs in a fresh guarded process. These checks own
the final V1LF contract; the proud-family regression module contains no
legacy V1LF architecture assertions.
"""

from __future__ import annotations

# The module owns its own fresh-process, memory-guarded runner below.
# Prevent a generic pytest invocation from collecting all OCC-heavy tests
# into one long-lived process and bypassing both the 8 GiB tree cap and the
# 0.5 GiB immediately-reclaimable macOS floor.
__test__ = False

from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import math
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

LM_CUTTER_GROUP_COUNT = 20
R6F_NATIVE_STAGE_SCHEMA_VERSION = 6
R6F_CHECK_LAUNCH_HEADROOM_MB = 2500.0
R6F_CABLE_WORKER_HEADROOM_MB = 3500.0
R6F_HEADROOM_WAIT_TIMEOUT_S = 300.0


def _large_host_execution() -> bool:
    return (
        os.environ.get("LX_CAD_MEMORY_PROFILE") == "osado-512g"
        and os.environ.get("LX_CAD_ALLOW_PARALLEL") == "1"
    )


def _state(stand_foot: bool):
    os.environ["LX_STAND_FOOT"] = "1" if stand_foot else "0"
    os.environ["LX_ROUTING_PROFILE"] = "v1lf"


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
    import top_baffle_nd25fw4_um_fit as fit
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
    from shapely.geometry import LineString, Point
    _state(False)
    from top_baffle_nd25fw4 import (
        L22_CUTOUT,
        L22_PILOT_ANGLES_DEG,
        L22_PILOT_PCD_MM,
        UM_TERMINAL_CLOCK_DEG,
    )
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_b as proud
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route
    import run_memory_guarded as memory_guard
    import export_v1lf_staged as staged
    import write_v1lf_release_manifest as release_manifest

    assert route.route_inner_cutter_group_count("lm") == LM_CUTTER_GROUP_COUNT

    assert L22_PILOT_ANGLES_DEG == (30.0, 90.0, 150.0, 210.0, 270.0, 330.0)
    assert flush.V1LF_LM_PILOT_ANGLES_DEG == (
        0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
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
    assert {site["angle_deg"] for site in lm_magnets} == {
        64.0, 116.0, 224.0, 316.0}
    assert {site["clock_from_top_deg"] for site in lm_magnets} == {
        -26.0, 26.0, -134.0, -226.0}
    assert all(site["flush_buried"] for site in lm_magnets)
    assert all(site["face_offset_mm"] == 0.0 for site in lm_magnets)
    assert {
        site["angle_deg"]: site["z_mm"] for site in lm_magnets
    } == {64.0: 12.55, 116.0: 12.55, 224.0: 12.55, 316.0: 15.40}
    assert math.isclose(
        core.THICKNESS_MM
        - (core.SIDE_MAGNET_Z_OVERRIDES[("lm", 316.0)]
           + core.SIDE_MAGNET_POCKET_D / 2.0),
        0.30, abs_tol=1e-12)
    um_magnets = [site for site in magnet_sites if site["driver"] == "um"]
    assert {site["angle_deg"] for site in um_magnets} == {50.5, 129.5}
    assert {site["clock_from_top_deg"] for site in um_magnets} == {
        -39.5, 39.5}
    assert all(site["flush_buried"] for site in um_magnets)
    assert all(not site["proud_ear_added"] for site in um_magnets)
    assert all(site["face_offset_mm"] == 0.0 for site in um_magnets)
    assert all(site["z_mm"] == 15.10 for site in um_magnets)
    assert core.SIDE_MAGNET_D == 5.0
    assert core.SIDE_MAGNET_POCKET_D == 5.2
    assert core.SIDE_MAGNET_DEPTH == 2.2
    assert (core.SIDE_MAGNET_D == proud.MAGNET_D_MM
            and core.SIDE_MAGNET_POCKET_D
            == proud.MAG_POCKET_D_MM
            == proud.MAG_RECEIVER_D_MM
            and core.SIDE_MAGNET_DEPTH
            == proud.MAG_FLUSH_DEPTH_MM
            == proud.MAG_PIN_BASE_DEPTH_MM
            == proud.MAG_PIN_RECEIVER_DEPTH_MM)
    assert math.isclose(
        core.SIDE_MAGNET_DEPTH - 2.0, 0.2, abs_tol=1e-12)
    assert math.isclose(
        core.LM_CORE_R - core.SIDE_MAGNET_DEPTH - flush.LM_RECESS_R,
        0.2, abs_tol=1e-12)
    assert math.isclose(
        core.UM_CORE_R - core.SIDE_MAGNET_DEPTH - flush.UM_RECESS_R,
        0.2, abs_tol=1e-12)
    assert math.isclose(
        core.THICKNESS_MM
        - (core.SIDE_MAGNET_Z["um"]
           + core.SIDE_MAGNET_POCKET_D / 2.0),
        0.6, abs_tol=1e-12)
    # At the route-safe +/-26-degree LM clocks, the buried D5.2 pocket remains
    # well separated from every D9.6 threaded-insert boss. This is the actual
    # capsule-to-disc edge gap, not only angular separation.
    lm_magnet_insert_gaps = []
    for site in lm_magnets:
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        inner = face - core.SIDE_MAGNET_DEPTH * normal
        axis = LineString((inner, face))
        lm_magnet_insert_gaps.extend(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0 - flush.PAD_D_MM / 2.0
            for xy in flush.LM_PILOT_XY)
    assert min(lm_magnet_insert_gaps) >= 2.2
    for site in lm_magnets:
        if site["angle_deg"] not in (224.0, 316.0):
            continue
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        axis = LineString((
            face - core.SIDE_MAGNET_DEPTH * normal, face))
        assert min(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0 - flush.PAD_D_MM / 2.0
            for xy in flush.LM_PILOT_XY) >= 23.0
    um_magnet_insert_gaps = []
    for site in um_magnets:
        normal = np.asarray(site["normal"], dtype=float)
        face = np.asarray(site["face"], dtype=float)
        inner = face - core.SIDE_MAGNET_DEPTH * normal
        axis = LineString((inner, face))
        um_magnet_insert_gaps.extend(
            axis.distance(Point(*xy))
            - core.SIDE_MAGNET_POCKET_D / 2.0
            - core.UM_INSERT_BOSS_D / 2.0
            for xy in flush.UM_PILOT_XY)
    assert min(um_magnet_insert_gaps) >= 1.0
    assert core.JOINT_BOSS_D == core.TWEETER_JOINT_BOSS_D == 9.0
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
    assert route.TUBE_SECTION_SPACING == 5.5
    assert route.TUBE_SECTION_SIDES == 8
    assert (20.0 - math.sqrt(
        20.0 ** 2 - (route.TUBE_SECTION_SPACING / 2.0) ** 2)) < 0.20
    assert memory_guard.MEMORY_PROFILES["local-macos"] == {
        "max_rss_mb": 8192,
        "min_free_mb": 512,
        "max_guard_slots": 1,
    }
    assert memory_guard.MEMORY_PROFILES["osado-512g"] == {
        "max_rss_mb": 512 * 1024,
        "min_free_mb": 64 * 1024,
        "max_guard_slots": 16,
    }
    assert memory_guard.MAX_RSS_MB <= memory_guard.PROFILE_MAX_RSS_MB
    assert memory_guard.MIN_FREE_MB >= memory_guard.PROFILE_MIN_FREE_MB
    assert memory_guard.GUARD_SLOTS <= memory_guard.PROFILE_MAX_GUARD_SLOTS
    assert staged.SCHEMA_VERSION == R6F_NATIVE_STAGE_SCHEMA_VERSION == 6
    assert release_manifest.FORMAT_VERSION == 6
    assert staged.ATTACHMENT_KEYS_BASE == ("addon_tweeter_crescent",)
    assert staged.OPTIONAL_LM_SPLIT_KEYS == (
        "optional_lm_keyed_1of2_bottom",
        "optional_lm_keyed_2of2_top",
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
    assert guard_policy["worker_launch_headroom_mib"] == 3200.0
    if memory_guard.MEMORY_PROFILE == "local-macos":
        assert 0 < memory_guard.MAX_RSS_MB <= 8192
        assert memory_guard.MIN_FREE_MB >= 512
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
    assert facts["t_bridge_start_handle_mm"] == 65.0
    assert facts["t_bridge_end_handle_mm"] == 60.0
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
    assert all(not record["floor_hardware_clearance"]
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
        "bridge_rear_face_shallow_rise")
    assert facts["no_floor_bridge_feed_xy"] == (
        (5.0, 82.0), (-5.0, 82.0))
    assert math.isclose(
        facts["no_floor_bridge_feed_rear_z_mm"], 5.3, abs_tol=1e-9)
    assert np.allclose(
        facts["functional_lm_feed_points"],
        ((5.0, 82.0, 5.3), (-5.0, 82.0, 5.3)), atol=1e-9)
    assert facts["functional_lm_feed_web_omitted"]
    assert facts["printed_lm_tunnel_count"] == 0
    assert facts["lm_lead_mode"] == "short_free_span_no_micro_duct"
    assert route.LM_LEAD_ANGLE_DEG == 269.5
    assert math.isclose(route.LM_LEAD_INNER_R, 93.85, abs_tol=1e-12)
    assert math.isclose(facts["lm_free_cable_outer_z_mm"], 0.40,
                        abs_tol=1e-12)
    assert facts["lm_free_cable_inner_z_mm"] == 3.80
    assert math.isclose(facts["lm_free_cable_rear_clearance_mm"], 1.00,
                        abs_tol=1e-12)
    # The complete free D7.8 envelope stays below the deepest carrier/web
    # rear datum until its circular rise starts at R103.
    outer_top = (facts["lm_free_cable_outer_z_mm"]
                 + route.LM_CABLE_D_EST / 2.0)
    assert math.isclose(outer_top, route.PAD_FACE_Z - 1.00,
                        abs_tol=1e-12)
    assert (route.L22_CUTOUT[2] / 2.0
            < route.LM_FREE_CABLE_RISE_START_R < route.LM_RECESS_R)
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
        "LM": route.lm_cable_points(0.20),
        "T": route.ts_cable_points(0.20),
    }
    for name, points in paths.items():
        radius = _min_three_point_radius(points)
        assert radius >= 14.0, f"{name} minimum bend radius {radius:.3f}"
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

    # The minimally outward-shifted LM half-laps retain the full T-void
        # side wall. The upper T tunnel deliberately passes through only the
        # bottom of the complementary front UM half-lap; its closed cover,
        # M3 clearance and retained upper ligament are all explicit.
    t_line = LineString(np.asarray(paths["T"])[:, :2])
    for x in core.JOINT_EAR_X:
        lm_half_lap = core.joint_ear_polygon(
            "lm", x, core.JOINT_RECEIVER_RADIAL_CLEAR)
        wall = t_line.distance(lm_half_lap) - route.TS_CUTTER_R
        assert wall >= route.TUNNEL_SKIN - 0.02, (
            f"T void to LM half-lap wall {wall:.3f} mm")

        um_half_lap = core.joint_ear_polygon(
            "um", x, core.JOINT_RECEIVER_RADIAL_CLEAR)
        near = [point for point in np.asarray(paths["T"])
                if Point(float(point[0]), float(point[1])).distance(
                    um_half_lap) <= route.TS_OUTER_R]
        if near:
            top_ligament = core.UM_JOINT_Z[1] - max(
                float(point[2]) + route.TS_CUTTER_R for point in near)
            assert top_ligament >= 4.80
            bolt_wall = (t_line.distance(Point(x, core.JOINT_EAR_Y))
                         - route.TS_CUTTER_R - core.JOINT_HOLE_D / 2.0)
            assert bolt_wall >= route.TUNNEL_SKIN
            assert math.isclose(
                core.UM_JOINT_TUNNEL_LIGAMENT, 5.35, abs_tol=1e-8)

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
        "  V1LF route: rotated crown, zero windows, "
        f"{facts['crossover_angle_deg']:.2f} deg crossover, "
        "free-UM/printed-T crossover gap "
        f"{facts['crossover_free_um_to_t_cover_gap_mm']:.2f} mm; "
        f"normal wall {facts['min_plan_normal_wall_mm']:.2f} mm")


def _insert_bump_clearance(stand_foot):
    _state(stand_foot)
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf_route as route

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


def test_insert_bump_clearance():
    _insert_bump_clearance(False)


def test_floor_insert_bump_clearance():
    _insert_bump_clearance(True)


def test_floor_route_smoothness():
    _state(True)
    import top_baffle_nd25fw4_v1lf_route as route

    facts = route.route_facts()
    assert facts["open_bore_jump_count"] == 0
    assert facts["solid_backfill_count"] == 8
    assert set(facts["solid_backfill_floor_hardware_exceptions"]) == {
        "lm_pilot_300", "lm_pilot_240", "lm_pilot_180"}
    assert math.isclose(
        facts["lm_burial_web_floor_hardware_clear_d_mm"],
        route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D, abs_tol=1e-12)
    assert facts["solid_backfill_added_rear_depth_mm"] == 0.0
    assert (facts["crossover_nominal_void_gap_mm"]
            >= route.CROSSOVER_MIN_CLEARANCE)
    assert facts["crossover_free_um_to_t_cover_gap_mm"] >= 0.25
    for name, points in (
            ("UM", route.route_cable_points(0.20)),
            ("LM", route.lm_cable_points(0.20)),
            ("T", route.ts_cable_points(0.20))):
        radius = _min_three_point_radius(points)
        assert radius >= 14.0, f"floor {name} bend radius {radius:.3f}"
        assert _max_turn_deg(points) <= 2.0
    assert math.isclose(
        min(bump.low_z for bump in route.MAIN_COVERED_BUMPS),
        -6.35, abs_tol=1e-9)
    assert math.isclose(
        min(bump.low_z for bump in route.T_COVERED_BUMPS),
        -5.45, abs_tol=1e-9)
    assert 18.00 <= facts["main_max_rear_protrusion_mm"] <= 18.10
    assert 16.00 <= facts["ts_max_rear_protrusion_mm"] <= 16.10
    print("  floor-state deep covered bumps retain complete-route R14/G1")


def test_no_floor_route_smoothness():
    """Keep the minimal carrier's Z bypasses independently R14/G1-safe."""
    _state(False)
    import top_baffle_nd25fw4_v1lf_route as route

    facts = route.route_facts()
    assert facts["open_bore_jump_count"] == 0
    assert facts["solid_backfill_count"] == 8
    assert (facts["crossover_nominal_void_gap_mm"]
            >= route.CROSSOVER_MIN_CLEARANCE)
    assert facts["crossover_free_um_to_t_cover_gap_mm"] >= 0.25
    for name, points in (
            ("UM", route.route_cable_points(0.20)),
            ("LM", route.lm_cable_points(0.20)),
            ("T", route.ts_cable_points(0.20))):
        radius = _min_three_point_radius(points)
        assert radius >= 14.0, f"no-floor {name} bend radius {radius:.3f}"
        assert _max_turn_deg(points) <= 2.0
    print("  no-floor covered bumps retain complete-route R14/G1")


def _bump_brep_clearance(stand_foot):
    _state(stand_foot)
    from build123d import Compound, Cylinder, Pos
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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
        print(f"    {label}: {clearance:.3f} mm")
        assert clearance >= route.INSERT_COVER_CLEAR - 0.03, (
            f"{label} BREP clearance {clearance:.3f} mm")

    if stand_foot:
        import top_baffle_nd25fw4_v1lf_attachments as addons
        hardware = addons.structural_hardware_proxies()
        for cover, label in ((main_outer, "UM"), (t_outer, "T")):
            clearance = cover.distance_to(hardware)
            assert clearance >= route.SUPPORT_TUNNEL_GAP - 0.03, (
                f"floor {label} cover/hardware clearance "
                f"{clearance:.3f} mm")
    state = "floor" if stand_foot else "no-floor"
    print(f"  {state}: exact outer-cover BREPs clear inserts and hardware")


def test_bump_brep_clearance():
    _bump_brep_clearance(False)


def test_floor_bump_brep_clearance():
    _bump_brep_clearance(True)


def _final_bump_backfill_contract(stand_foot):
    """Every intended roof-to-bore fill survives in the final carriers."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Compound, Cylinder, Pos, import_brep
    import top_baffle_nd25fw4_v1lf_route as route

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
        # they do not reuse the convex-hull BREP builder. Skip only the exact
        # floor hardware cylinder, never a broad plan band.
        route_xy = np.asarray(spec.route_xyz[:2], dtype=float)
        pilot_xy = np.asarray(spec.pilot_xy, dtype=float)
        for fraction in (0.15, 0.50, 0.85):
            xy = route_xy + fraction * (pilot_xy - route_xy)
            if (spec.floor_hardware_clearance
                    and np.linalg.norm(xy - pilot_xy)
                    < route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D / 2.0):
                continue
            z0 = spec.bottom_z + 0.10
            z1 = spec.top_z - 0.10
            probe = Pos(float(xy[0]), float(xy[1]), (z0 + z1) / 2.0) * (
                Cylinder(0.18, z1 - z0))
            hit = _intersection_volume(owners[spec.owner], probe)
            assert hit > 0.97 * probe.volume, (
                f"{spec.name} hollow at saddle fraction {fraction:.2f}: "
                f"{hit:.4f}/{probe.volume:.4f} mm3")

    expected_exceptions = (
        route.FLOOR_SUPPORT_BUMP_NAMES if stand_foot else frozenset())
    assert {spec.name for spec in specs if spec.floor_hardware_clearance} == (
        expected_exceptions)
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state}: all eight final-BREP duct bumps are solid-backed to "
        "their bore floors; only exact duct-clearance and floor-hardware "
        "envelopes remain")


def test_bump_backfill_contract():
    _final_bump_backfill_contract(False)


def test_floor_bump_backfill_contract():
    _final_bump_backfill_contract(True)


def _final_lm_burial_web_contract(stand_foot):
    """All six LM bumps retain closed full-width longitudinal burial."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Box, Cylinder, Pos, import_brep
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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
    structural = (
        route.FLOOR_SUPPORT_BUMP_NAMES if stand_foot else frozenset())
    probe_radius = 0.14
    functional_voids = []
    for name, pilot_xy in pilot_by_name.items():
        x, y = map(float, pilot_xy)
        if name in structural:
            z0, z1 = route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_Z
            functional_voids.append(
                Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
                    route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D / 2.0,
                    z1 - z0))
            z0, z1 = route.FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_Z
            functional_voids.append(
                Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
                    route.FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_D / 2.0,
                    z1 - z0))
            functional_voids.append(
                Pos(
                    x, y,
                    (flush.PAD_FACE_Z - 0.20
                     + flush.LM_SEAT_Z + 0.15) / 2.0
                ) * Cylinder(
                    core.STRUCT_MOUNT_CLEAR_D / 2.0,
                    flush.LM_SEAT_Z + 0.15
                    - (flush.PAD_FACE_Z - 0.20)))
        else:
            bore_z0 = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM
            functional_voids.append(
                Pos(
                    x, y, (bore_z0 + flush.LM_SEAT_Z + 0.15) / 2.0
                ) * Cylinder(
                    flush.L22_PILOT_D_MM / 2.0,
                    flush.LM_SEAT_Z + 0.15 - bore_z0))

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
                # height except for the exact cable lumen and exact final
                # pilot/support hardware. Subtract those functional voids;
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

            if bump.name not in structural:
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

        # The unburied station-zero cap remains a deliberate service mouth.
        facts = route.route_facts()
        feed_index = 0 if route_name == "UM" else 1
        assert np.allclose(
            points[0], facts["functional_lm_feed_points"][feed_index],
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
    # applicable longitudinal end must be present.  In no-floor state the
    # centered bridge feed reaches the 300-degree bypass before either end,
    # while the floor route begins at that same bypass and therefore has no
    # predecessor-side shoulder to bury.
    assert tested_ends["lm_pilot_300"] == (
        {1} if stand_foot else {-1, 1})
    assert boundary_probe_count > sum(len(ends) for ends in tested_ends.values())
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state}: all six LM bump runs are closed full-width at every "
        "applicable longitudinal end; station-zero feeds and only exact "
        "blind-bores/floor hardware remain")


def test_lm_burial_web_contract():
    _final_lm_burial_web_contract(False)


def test_floor_lm_burial_web_contract():
    _final_lm_burial_web_contract(True)


def _final_um_burial_web_contract(stand_foot):
    """Both UM T-bypass bumps retain solid longitudinal shoulders."""
    _state(stand_foot)
    staged = _stage_shell_contract_breps(
        stand_foot, "T", tempfile.gettempdir(), shell_keys=())
    from build123d import (
        Cylinder, Face, Polyline, Pos, Wire, import_brep, loft)
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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
    # The 50.5-degree pocket lies close to the 58-degree bypass. It is a
    # legitimate radial service void, so subtract its exact production
    # cutter rather than weakening or relocating the solid-web witness.
    for site in core.side_magnet_sites("um"):
        functional_voids.append(core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"],
            core.SIDE_MAGNET_POCKET_D, core.SIDE_MAGNET_DEPTH, 1.0))
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
             z1 + core.JOINT_RECEIVER_RADIAL_CLEAR),
            core.JOINT_RECEIVER_RADIAL_CLEAR))
        functional_voids.append(core._cylinder_at(
            x, core.JOINT_EAR_Y, core.JOINT_HOLE_D / 2.0,
            core.CORE_REAR_Z - core.JOINT_BORE_REAR_OVERSHOOT,
            core.THICKNESS_MM + 0.2))
    for x in core.TWEETER_JOINT_X:
        functional_voids.append(core._plan_prism(
            core.tweeter_joint_polygon(x, core.TWEETER_JOINT_CLEAR),
            core.TWEETER_ADDON_JOINT_Z[0] - core.TWEETER_JOINT_CLEAR,
            core.TWEETER_ADDON_JOINT_Z[1] + 0.2))
        functional_voids.append(core._cylinder_at(
            x, core.TWEETER_JOINT_Y, core.TWEETER_JOINT_HOLE_D / 2.0,
            core.TWEETER_CORE_JOINT_Z[0] - 0.2,
            core.TWEETER_CORE_JOINT_Z[1] + 0.3))

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
        "flush-magnet and half-lap interface voids remain")


def test_um_burial_web_contract():
    _final_um_burial_web_contract(False)


def test_floor_um_burial_web_contract():
    _final_um_burial_web_contract(True)


def _final_feed_and_flush_mouth_contract(stand_foot):
    """Final carriers own only buried skins up to native mouth planes."""
    _state(stand_foot)
    staging = tempfile.TemporaryDirectory(prefix="lx-v1lf-mouths-")
    staged = _stage_shell_contract_breps(
        stand_foot, "T", staging.name, shell_keys=())
    from build123d import Box, Cylinder, Pos, import_brep
    from shapely.geometry import Point
    from top_baffle_nd25fw4 import (
        BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM, L22_CUTOUT, UM_CUTOUT)
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_bridge as bridge
    import top_baffle_nd25fw4_v1lf_route as route

    owners = {
        "lm": import_brep(staged["lm"]),
        "um": import_brep(staged["um"]),
        "tweeter": import_brep(staged["tweeter"]),
    }
    facts = route.route_facts()
    main = np.asarray(route.route_cable_points(0.20), dtype=float)
    ts = np.asarray(route.ts_cable_points(0.20), dtype=float)

    # Cutter extension is a global ruled-loft phase input, not merely an
    # endpoint allowance. Only the two core owners retain printed T cover.
    phase_source = route.ts_cable_points(1.8)
    for owner, expected_extension in (
            ("lm", (route.NO_FLOOR_FEED_CUTTER_EXTENSION
                    if not stand_foot
                    else route.STANDARD_CUTTER_EXTENSION)),
            ("um", (route.NO_FLOOR_FEED_CUTTER_EXTENSION
                    if not stand_foot
                    else route.STANDARD_CUTTER_EXTENSION))):
        phased = route._owner_cutter_points(phase_source, owner)
        assert math.isclose(
            np.linalg.norm(phased[0] - phase_source[0]),
            expected_extension, abs_tol=1e-9)
        assert math.isclose(
            np.linalg.norm(phased[-1] - phase_source[-1]),
            expected_extension, abs_tol=1e-9)

    if stand_foot:
        assert facts["functional_lm_feed_mode"] == "supported_ring_lateral"
        expected_main_xy = np.asarray(
            L22_CUTOUT[:2], dtype=float) + 114.0 * np.asarray((
                math.cos(math.radians(300.0)),
                math.sin(math.radians(300.0))))
        expected_t_xy = np.asarray(
            L22_CUTOUT[:2], dtype=float) + 114.0 * np.asarray((
                math.cos(math.radians(240.0)),
                math.sin(math.radians(240.0))))
        assert np.allclose(main[0, :2], expected_main_xy, atol=1e-8)
        assert np.allclose(ts[0, :2], expected_t_xy, atol=1e-8)
        assert not np.allclose(main[0], (5.0, 82.0, 5.3), atol=1e-8)
        assert not np.allclose(ts[0], (-5.0, 82.0, 5.3), atol=1e-8)
    else:
        assert facts["functional_lm_feed_mode"] == (
            "bridge_rear_face_shallow_rise")
        assert np.allclose(main[0], (5.0, 82.0, 5.3), atol=1e-9)
        assert np.allclose(ts[0], (-5.0, 82.0, 5.3), atol=1e-9)
        assert tuple(facts["no_floor_bridge_feed_rise_lengths_mm"]) == (
            24.0, 27.5)
        assert route.NO_FLOOR_FEED_START_BEARING_DEG == 65.0

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
        for (label, points, cutter_radius, outer_radius,
             production_points, segment_count) in feed_specs:
            local = points[:20]
            lumen = route._round_tube_global_segment(
                route._owner_cutter_points(production_points, "lm"),
                cutter_radius - 0.15, 0, segment_count)
            assert _intersection_volume(owners["lm"], lumen) < 0.02, (
                f"{label} bridge rear feed is capped")
            outer = route._round_tube(local, cutter_radius + 0.55)
            inner = route._round_tube(
                route._extended_points(local, 1.0),
                cutter_radius + 0.15)
            skin = (outer - inner) & front_domain
            retained = _intersection_volume(owners["lm"], skin)
            assert retained > 0.995 * skin.volume, (
                f"{label} bridge feed skin incomplete: "
                f"{retained:.4f}/{skin.volume:.4f} mm3")
            rear = route._polygon_prism(
                Point(*points[0, :2]).buffer(
                    outer_radius + 0.30, resolution=32),
                -20.0, route.NO_FLOOR_FEED_REAR_Z - 0.02)
            assert _intersection_volume(owners["lm"], rear) < 0.02, (
                f"{label} bridge feed projects behind z=5.3")

        # The two lumens share the bridge center without touching each other
        # or any of the immutable four blind insert envelopes.
        feed_lumen_web = (
            np.linalg.norm(main[0] - ts[0])
            - route.CUTTER_R - route.TS_CUTTER_R)
        assert feed_lumen_web >= route.CROSSOVER_MIN_CLEARANCE
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

    lm_allowed, lm_forbidden = radial_domains(
        L22_CUTOUT[:2], core.LM_CORE_R, True)
    main_lm_r = np.linalg.norm(
        main[:, :2] - np.asarray(L22_CUTOUT[:2], dtype=float), axis=1)
    main_lm_crossings = transition_indices(main_lm_r <= core.LM_CORE_R)
    assert main_lm_crossings
    if stand_foot:
        # Floor retains the legacy R114 centerline, but printed cover stops
        # at the native R113 ring; the outer millimetre is a free entry span.
        assert_flush_mouth(
            "floor UM native-ring feed", "lm", main,
            main_lm_crossings[0], route.CUTTER_R, route.MAIN_OUTER_R,
            lm_allowed, lm_forbidden)
    # The user-visible upper outlet is a native R113 flush mouth. No socket,
    # tongue, collar or point horn may survive beyond the LM ring.
    assert_flush_mouth(
        "LM UM free-cable exit", "lm", main, main_lm_crossings[-1],
        route.CUTTER_R, route.MAIN_OUTER_R,
        lm_allowed, lm_forbidden)

    t_lm_r = np.linalg.norm(
        ts[:, :2] - np.asarray(L22_CUTOUT[:2], dtype=float), axis=1)
    lm_crossings = transition_indices(t_lm_r <= core.LM_CORE_R)
    # No-floor begins inside R113 at the bridge rear mouth; floor begins at
    # its legacy R114 ring feed and therefore has an additional inward
    # crossing. The removed point horn is always the final outward crossing.
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
        f"  {state}: bridge/ring feeds and native LM/UM T mouths retain "
        "full lumens/skins; LM UM outlet is R113-flush and both deleted "
        "suffixes remain collision-free cable only")


def test_feed_and_flush_mouth_contract():
    _final_feed_and_flush_mouth_contract(False)


def test_floor_feed_and_flush_mouth_contract():
    _final_feed_and_flush_mouth_contract(True)


def _crossover_brep(stand_foot):
    _state(stand_foot)
    import top_baffle_nd25fw4_v1lf_route as route

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


def test_crossover_brep():
    _crossover_brep(False)


def test_floor_crossover_brep():
    _crossover_brep(True)


def test_bridge_contract():
    _state(False)
    from top_baffle_nd25fw4_v1lf_bridge import (
        BRIDGE_BORE_FLOOR_MM,
        BRIDGE_FACE_Z,
        BRIDGE_FUSION_INTERFACE_T,
        BRIDGE_FUSION_INTERFACE_Z,
        BRIDGE_GOVERNING_NECK_WIDTH_MM,
        BRIDGE_MIN_FUSION_SF_5G,
        BRIDGE_MIN_MEMBER_SF_5G,
        BRIDGE_WEB_T,
        BRIDGE_WEB_TUNNEL_DEDUCTION_MM,
        BRIDGE_WEB_WIDTH,
        BRIDGE_WEB_X,
        BRIDGE_WEB_Y,
        bridge_load_facts,
        bridge_plan_facts,
    )
    import top_baffle_nd25fw4_v1lf_route as route

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
        BRIDGE_GOVERNING_NECK_WIDTH_MM, 47.8, abs_tol=1e-12)
    route_section = plan["route_section"]
    assert route_section["y_range_mm"] == (73.25, 90.25)
    assert route_section["sample_step_max_mm"] <= 0.01
    assert route_section["minimum_net_width_mm"] >= 47.8
    assert route_section["minimum_net_width_mm"] > 53.0
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

    load = bridge_load_facts()
    assert load["design_mass_kg"] == 4.0
    assert load["design_y_cg_mm"] == 230.0
    assert load["rear_cg_mm"] == 70.0
    assert load["root_y_mm"] == 90.25
    assert load["governing_section_y_mm"] == 73.25
    assert load["normal_root_lever_mm"] == 139.75
    assert load["member_normal_lever_mm"] == 156.75
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
    assert load["deducted_central_tunnel_width_mm"] == 14.2
    assert BRIDGE_WEB_TUNNEL_DEDUCTION_MM == 14.2
    assert math.isclose(load["net_web_width_mm"], 47.8, abs_tol=1e-12)
    assert math.isclose(
        load["governing_neck_width_mm"], BRIDGE_GOVERNING_NECK_WIDTH_MM,
        abs_tol=1e-12)
    assert load["magnet_load_credit_n"] == 0.0
    assert load["combined_insert_5g_n"] > load["normal_insert_5g_n"]
    assert load["member_sf_1g_creep"] >= 2.0
    assert load["member_sf_3g"] >= 1.5
    assert load["member_sf_5g"] >= BRIDGE_MIN_MEMBER_SF_5G
    assert load["insert_sf_5g"] >= 1.35
    assert load["fusion_interface"]["span_deg"] == 68.0
    assert load["fusion_interface"]["deducted_um_tunnel_count"] == 1
    assert load["fusion_interface"]["deducted_um_tunnel_width_mm"] == 8.2
    assert load["fusion_interface"]["deducted_t_tunnel_width_mm"] == 6.0
    assert load["fusion_interface"]["deducted_tunnel_width_mm"] == 14.2
    assert load["fusion_interface"]["effective_width_mm"] > 118.0
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
        f"  front-flush bridge plate: conservative route-net section "
        f"{BRIDGE_GOVERNING_NECK_WIDTH_MM:.2f} x {BRIDGE_WEB_T:.1f} mm, "
        f"5g SF {load['member_sf_5g']:.2f}; no depth beyond LM pads")


def test_bridge_geometry():
    _state(False)
    from build123d import Cylinder, Pos
    from top_baffle_nd25fw4 import (
        BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM, BRIDGE_INSERT_DEPTH_MM)
    from top_baffle_nd25fw4_v1lf_bridge import (
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
    assert math.isclose(bb.min.Z, BRIDGE_WEB_REAR_Z, abs_tol=1e-6)
    assert math.isclose(bb.max.Z, BRIDGE_WEB_FRONT_Z, abs_tol=1e-6)
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(BRIDGE_FACE_Z, (5.3, 18.3)))
    assert len(bridge_face_plan().interiors) == 0

    # Every representative region that used to be an open frame/X cell is
    # now solid at the front face.
    for x, y in ((0.0, 45.0), (-12.0, 35.0), (12.0, 55.0), (0.0, 82.0)):
        probe = Pos(x, y, 17.5) * Cylinder(0.45, 0.8)
        assert _intersection_volume(tail, probe) > 0.95 * probe.volume

    # Four bores open from the web rear, stop after 6.8 mm, and retain a
    # positive solid front floor. Screw heads have an unobstructed approach.
    for x, y in BRIDGE_HOLE_XY:
        bore = Pos(
            x, y, BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM / 2.0
        ) * Cylinder(BRIDGE_INSERT_D_MM / 2.0, BRIDGE_INSERT_DEPTH_MM)
        assert _intersection_volume(tail, bore) < 0.02
        floor_z0 = BRIDGE_WEB_REAR_Z + BRIDGE_INSERT_DEPTH_MM + 0.1
        floor = Pos(
            x, y, (floor_z0 + BRIDGE_WEB_FRONT_Z - 0.1) / 2.0
        ) * Cylinder(1.0, BRIDGE_WEB_FRONT_Z - 0.1 - floor_z0)
        assert _intersection_volume(tail, floor) > 0.98 * floor.volume
    assert _intersection_volume(tail, bridge_fastener_head_envelopes()) < 0.02
    print("  bridge BREP is one front-flush solid web; four rear blind bores")


def test_joint_load_contract():
    _state(False)
    from top_baffle_nd25fw4_v1lf import (
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
    assert load["pla_sf_1g_creep"] >= 3.0
    assert load["pla_sf_3g"] >= 2.0
    assert load["pla_sf_5g"] >= 1.5
    assert load["m3_shear_sf_5g"] >= 4.0
    assert load["moment_1g"]["contact_sf"] >= 2.7
    assert load["moment_3g"]["contact_sf"] >= 2.0
    assert load["moment_5g"]["contact_sf"] >= 1.25
    assert load["moment_5g"]["m3_tension_sf"] >= 1.15
    spokes = carrier_spoke_load_facts()
    assert spokes["design_mass_kg"] == 4.0
    assert spokes["creep_allow_mpa"] == 8.0
    assert spokes["short_allow_mpa"] == 18.0
    assert spokes["lm_sf_1g"] >= 6.0
    assert spokes["lm_sf_3g"] >= 4.0
    assert spokes["lm_sf_5g"] >= 4.0
    assert spokes["lm_floor_three_spoke_area_mm2"] == 84.0
    assert spokes["lm_floor_sf_1g"] >= 17.0
    assert spokes["lm_floor_sf_3g"] >= 12.0
    assert spokes["lm_floor_sf_5g"] >= 7.5
    assert spokes["um_sf_1g"] >= 8.0
    assert spokes["um_sf_3g"] >= 4.0
    assert spokes["um_sf_5g"] >= 4.0
    print(
        f"  both two-ear interfaces screen 0.85 kg upper mass: PLA SF "
        f"1g/3g/5g {load['pla_sf_1g_creep']:.2f}/"
        f"{load['pla_sf_3g']:.2f}/{load['pla_sf_5g']:.2f}; "
        f"M3 shear SF5g {load['m3_shear_sf_5g']:.2f}")


def _assert_lm_mount_bores(lm, core, flush, stand_foot):
    """Probe all six exact rotated mount axes, including retained floors."""
    from build123d import Cylinder, Pos

    structural = set(core.STRUCT_MOUNT_ANGLES) if stand_foot else set()
    for angle, (x, y) in zip(
            flush.V1LF_LM_PILOT_ANGLES_DEG, flush.LM_PILOT_XY):
        if angle in structural:
            void = Pos(
                x, y, (flush.PAD_FACE_Z + flush.LM_SEAT_Z) / 2.0
            ) * Cylinder(
                core.STRUCT_MOUNT_CLEAR_D / 2.0,
                flush.LM_SEAT_Z - flush.PAD_FACE_Z + 0.3)
            assert _intersection_volume(lm, void) < 0.02, (
                f"{angle:g}deg structural clearance is obstructed")
        else:
            bore_z0 = flush.LM_SEAT_Z - flush.LM_BORE_DEPTH_MM
            # Keep the clearance witness wholly inside the specified blind
            # bore.  Extending it below ``bore_z0`` would classify the
            # deliberately retained printed floor as an obstruction.
            bore_z1 = flush.LM_SEAT_Z + 0.1
            bore = Pos(x, y, (bore_z0 + bore_z1) / 2.0) * Cylinder(
                core.L22_PILOT_D_MM / 2.0,
                bore_z1 - bore_z0)
            assert _intersection_volume(lm, bore) < 0.02, (
                f"{angle:g}deg blind insert bore is obstructed")
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
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

    lm = import_brep(staged["lm"])
    assert lm.is_valid and len(lm.solids()) == 1
    minimum_edge_mm = min(float(edge.length) for edge in lm.edges())
    assert minimum_edge_mm >= 1.0e-5, (
        "no-floor LM contains a micron-scale coincident-feature edge: "
        f"{minimum_edge_mm:.9g} mm")
    burial_growth_ceiling = route.route_facts()[
        "lm_burial_web_growth_upper_bound_mm3"]
    # 45 cm3 brackets the prior minimal floor carrier. Add only the gross
    # full-width-web loft volume, which deliberately overcounts its overlap
    # with the round covers, carrier membrane and existing bore saddles.
    assert lm.volume < 45000.0 + burial_growth_ceiling, (
        f"floor LM exceeds minimal-material budget: {lm.volume:.1f} mm3")
    assert lm.bounding_box().min.Y > 80.0, "floor LM inherited bridge tail"
    # Empty witness below the 0.85-mm seat membrane, away from spokes and
    # all three routes. A retained annular slab fills this probe.
    membrane_void = Pos(0.0, 300.981, 9.0) * Cylinder(1.0, 2.0)
    assert _intersection_volume(lm, membrane_void) < 0.01
    assert BED_MM == 256.0
    import top_baffle_nd25fw4_flush as flush
    _assert_lm_mount_bores(lm, core, flush, True)
    rotated = Rot(Z=BED_ROT_Z["v1lf_core_1of2_lm_carrier"]) * lm
    bb = rotated.bounding_box().size
    assert max(bb.X, bb.Y, bb.Z) <= BED_MM, (
        f"floor LM bed footprint {bb.X:.2f} x {bb.Y:.2f} x "
        f"{bb.Z:.2f} exceeds {BED_MM:.1f} mm")
    print(f"  floor LM: no tail; flat-bed footprint {bb.X:.2f} x {bb.Y:.2f}")


def test_no_floor_lm_core():
    _state(False)
    staged = _stage_shell_contract_breps(
        False, "LM", tempfile.gettempdir(), shell_keys=())
    from build123d import Cylinder, Pos, Rot, import_brep
    from shapely.geometry import LineString
    from export_piece_stls import BED_MM, V1LF_NO_FLOOR_LM_TILT_X_DEG
    from top_baffle_nd25fw4 import BRIDGE_HOLE_XY, BRIDGE_INSERT_D_MM
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf_route as route
    from top_baffle_nd25fw4_v1lf_bridge import (
        BRIDGE_FACE_Z,
        BRIDGE_FUSION_CRADLE_Z,
        BRIDGE_INSERT_DEPTH_MM,
        BRIDGE_WEB_FRONT_Z,
        BRIDGE_WEB_REAR_Z,
        bridge_fastener_head_envelopes,
        bridge_face_plan,
        bridge_fusion_cradle_plan,
        bridge_fusion_interface_facts,
        bridge_load_facts,
        bridge_solid_web_plan,
    )

    lm = import_brep(staged["lm"])
    assert lm.is_valid and len(lm.solids()) == 1
    # The measured pinned-osado baseline is 102,784.6 mm3. Permit only the
    # conservative gross burial-web volume above that narrow 103,000 mm3
    # bracket; actual growth must be smaller because the gross estimate does
    # not subtract its round-cover/backfill/carrier overlap.
    burial_growth_ceiling = route.route_facts()[
        "lm_burial_web_growth_upper_bound_mm3"]
    assert lm.volume < 103000.0 + burial_growth_ceiling, (
        f"no-floor LM exceeds minimal-material budget: {lm.volume:.1f} mm3")
    assert BED_MM == 256.0
    assert lm.bounding_box().min.Y < 16.0, "no-floor LM lost fused tail"
    _assert_lm_mount_bores(lm, core, flush, False)
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
    front_contract = core._plan_prism(
        bridge_face_plan(), BRIDGE_WEB_FRONT_Z - 0.35,
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
    tilted = Rot(X=V1LF_NO_FLOOR_LM_TILT_X_DEG) * lm
    bb = tilted.bounding_box().size
    assert bb.X <= BED_MM and bb.Y <= BED_MM and bb.Z <= BED_MM
    print(
        "  no-floor LM: fused front-flush four-hole solid web; "
        f"X{V1LF_NO_FLOOR_LM_TILT_X_DEG:g} footprint "
        f"{bb.X:.2f} x {bb.Y:.2f} x {bb.Z:.2f}")


def _load_lm_keyed_parts(stand_foot):
    """Load the exact hash-validated release stage for the split gate."""
    _state(stand_foot)
    from build123d import import_brep
    from export_v1lf_staged import load_stage_manifest, staged_part_paths

    state_name = "floor_stand" if stand_foot else "no_floor_stand"
    manifest = Path(__file__).resolve().parent / state_name / (
        ".v1lf_stage/manifest.json")
    payload = load_stage_manifest(manifest)
    paths = staged_part_paths(manifest, payload)
    lm = import_brep(str(paths["core_lm_carrier"]))
    parts = {
        key: import_brep(str(paths[key]))
        for key in (
            "optional_lm_keyed_1of2_bottom",
            "optional_lm_keyed_2of2_top",
        )
    }
    return lm, parts


def _assert_lm_keyed_split(stand_foot):
    _state(stand_foot)
    from build123d import Rot
    from shapely.geometry import box
    from export_piece_stls import (
        BED_ROT_Z,
        V1LF_OPTIONAL_LM_SPLIT_BED_MM,
    )
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_lm_split as lm_split
    import top_baffle_nd25fw4_v1lf_route as route

    lm, parts = _load_lm_keyed_parts(stand_foot)
    expected_keys = {
        "optional_lm_keyed_1of2_bottom",
        "optional_lm_keyed_2of2_top",
    }
    assert set(parts) == expected_keys
    bottom = parts["optional_lm_keyed_1of2_bottom"]
    top = parts["optional_lm_keyed_2of2_top"]
    for name, part in parts.items():
        assert part.is_valid and len(part.solids()) == 1, name
        assert part.volume > 0.01
    assert _intersection_volume(bottom, top) < 0.02

    # The hidden registration is made only by reassigning/cutting source
    # material. Neither half may grow beyond the monolithic LM envelope.
    for name, part in parts.items():
        extra = part - lm
        assert (0.0 if extra is None else extra.volume) < 0.03, name

    # The only intentional source loss is the local female fit relief not
    # occupied by the male key. Everywhere else, including both route-cover
    # sections, the union equals the monolithic carrier exactly.
    male_tool = lm_split.male_registration_key_tool()
    socket_tool = lm_split.female_registration_socket_tool()
    top_clip = core._plan_prism(
        box(-400.0, lm_split.LM_SPLIT_SEAM_Y, 400.0, 600.0),
        -100.0, 100.0)
    expected_relief = lm & socket_tool
    expected_relief = expected_relief & top_clip
    expected_relief = expected_relief - male_tool
    missing = lm - bottom
    missing = missing - top
    unexpected_missing = missing - expected_relief
    uncut_relief = expected_relief - missing
    assert (0.0 if unexpected_missing is None
            else unexpected_missing.volume) < 0.05
    assert (0.0 if uncut_relief is None
            else uncut_relief.volume) < 0.05
    assert _intersection_volume(bottom, male_tool) >= (
        male_tool.volume - 0.03)
    assert _intersection_volume(top, male_tool) < 0.03
    male_outside_socket = male_tool - socket_tool
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
    assert facts["registration_pair_count"] == 1
    assert facts["registration_side"] == "right"
    assert facts["registration_is_keyed"] is True
    assert facts["registration_form"] == (
        "single_straight_rounded_tongue")
    assert facts["registration_tongue_width_mm"] == 0.80
    assert 70.0 <= facts["registration_insertion_angle_deg"] <= 80.0
    assert facts["assembly_motion"] == (
        "top_half_approaches_along_negative_registration_axis")
    assert facts["sampled_insertion_max_plan_overlap_mm2"] < 1e-7
    assert facts["assembly_gap_mm"] == 0.0
    assert facts["buried_route_joint"] == "closed_zero_gap_planar_butt"
    assert facts["male_volume_mm3"] >= 15.0
    assert facts["engagement_depth_mm"] >= 3.4
    assert facts["socket_inner_wall_mm"] >= 0.65
    assert facts["socket_outer_wall_mm"] >= 0.60
    assert facts["driver_radial_clearance_mm"] >= 0.74
    assert facts["registered_plan_play_mm"] <= 0.25
    assert facts["registered_z_play_mm"] <= 0.30
    assert facts["envelope_growth_mm"] == 0.0
    assert facts["installed_structural_load_credit_n"] == 0.0
    assert facts["standalone_retention_credit_n"] == 0.0
    assert facts["physical_coupon_required"] is True
    assert facts["physical_load_qualification_required"] is True
    assert V1LF_OPTIONAL_LM_SPLIT_BED_MM == facts[
        "target_square_bed_mm"] == 220.0

    orientations = {
        "optional_lm_keyed_1of2_bottom":
            BED_ROT_Z["v1lf_optional_lm_keyed_1of2_bottom"],
        "optional_lm_keyed_2of2_top":
            BED_ROT_Z["v1lf_optional_lm_keyed_2of2_top"],
    }
    footprints = {}
    for name, part in parts.items():
        oriented = Rot(Z=orientations[name]) * part
        size = oriented.bounding_box().size
        footprints[name] = (size.X, size.Y, size.Z)
        assert max(size.X, size.Y, size.Z) <= 220.0, (
            f"{name} footprint {size.X:.2f} x {size.Y:.2f} x "
            f"{size.Z:.2f} exceeds the 220-mm option contract")

    if stand_foot:
        assert bottom.bounding_box().min.Y > 80.0
    else:
        assert bottom.bounding_box().min.Y < 16.0
    state = "floor" if stand_foot else "no-floor"
    print(
        f"  {state} optional LM keyed split: zero-gap route butt, "
        f"one concealed source-volume registration key, no envelope growth; "
        f"220-mm footprints {footprints}")


def test_floor_lm_keyed_split():
    _assert_lm_keyed_split(True)


def test_no_floor_lm_keyed_split():
    _assert_lm_keyed_split(False)


def _intersection_volume(a, b):
    hit = a & b
    return 0.0 if hit is None else hit.volume


def _assert_core_interface_breps(lm, um, core):
    import top_baffle_nd25fw4_v1lf_route as route

    for x in core.JOINT_EAR_X:
        lm_ear = core._joint_ear("lm", x, core.LM_JOINT_Z)
        um_ear = core._joint_ear("um", x, core.UM_JOINT_Z)
        bore = core._cylinder_at(
            x, core.JOINT_EAR_Y, core.JOINT_HOLE_D / 2.0,
            core.CORE_REAR_Z - core.JOINT_BORE_REAR_OVERSHOOT,
            core.THICKNESS_MM + 0.2)
        expected_lm_ear = lm_ear - bore
        expected_um_ear = um_ear - bore
        assert (_intersection_volume(lm, expected_lm_ear)
                > 0.995 * expected_lm_ear.volume), (
                    f"LM half-lap at x={x:g} was gouged outside its bore")
        overlap = _intersection_volume(um, lm_ear)
        assert overlap < 0.02, (
            f"UM overlaps LM half-lap at x={x:g} by {overlap:.3f} mm3")
        assert (_intersection_volume(um, expected_um_ear)
                > 0.995 * expected_um_ear.volume), (
                    f"UM half-lap at x={x:g} was gouged outside its bore")
        overlap = _intersection_volume(lm, um_ear)
        assert overlap < 0.02, (
            f"LM overlaps UM half-lap at x={x:g} by {overlap:.3f} mm3")

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
        assert face_offset == 0.0
        assert site["flush_buried"] and not site["proud_ear_added"]
        outside = core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"],
            core.SIDE_EAR_D, 0.0, core.SIDE_EAR_OUT)
        assert _intersection_volume(
            owner[site["driver"]], outside) < 0.05, (
                f"{site['name']} retained proud magnet-ear material")
        pocket = core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"],
            core.SIDE_MAGNET_POCKET_D, core.SIDE_MAGNET_DEPTH, 0.8)
        assert _intersection_volume(owner[site["driver"]], pocket) < 0.02

        # The 2.2-mm radial pocket leaves only a 0.2-mm floor before the
        # driver recess. Probe the center of that floor in the final BREP so
        # a coincident Boolean cannot silently delete it.
        nx, ny = site["normal"]
        floor_face = (
            site["face"][0] - (core.SIDE_MAGNET_DEPTH + 0.10) * nx,
            site["face"][1] - (core.SIDE_MAGNET_DEPTH + 0.10) * ny,
        )
        floor_probe = core._axis_cylinder(
            floor_face, site["normal"], site["z_mm"],
            core.SIDE_MAGNET_POCKET_D - 0.4, 0.05, 0.05)
        floor_fill = _intersection_volume(
            owner[site["driver"]], floor_probe)
        assert floor_fill > 0.90 * floor_probe.volume, (
            f"{site['name']} lost its 0.2-mm radial pocket floor: "
            f"{floor_fill:.4f}/{floor_probe.volume:.4f} mm3 remains")

        if site["driver"] == "um":
            # The raised UM pocket also leaves a 0.6-mm front skin. Sample
            # that skin on the pocket axis, inside the R51.7 lip.
            from build123d import Cylinder, Pos
            axis_xy = (
                site["face"][0] - core.SIDE_MAGNET_DEPTH / 2.0 * nx,
                site["face"][1] - core.SIDE_MAGNET_DEPTH / 2.0 * ny,
            )
            front_probe = Pos(axis_xy[0], axis_xy[1], 17.95) * Cylinder(
                0.4, 0.30)
            front_fill = _intersection_volume(
                owner[site["driver"]], front_probe)
            assert front_fill > 0.90 * front_probe.volume, (
                f"{site['name']} lost its 0.6-mm front skin: "
                f"{front_fill:.4f}/{front_probe.volume:.4f} mm3 remains")

            # The requested solid burial web may legitimately occupy the
            # old empty-cover keepout: it is solid carrier material, not the
            # cable void. Measure the actual safety invariant instead. The
            # pocket must retain the complete 0.8-mm tunnel shell plus the
            # original 0.8-mm isolation wall to the exact D6 lumen.
            um_lumen = route._round_tube(
                route._owner_cutter_points(
                    route.ts_cable_points(1.8), "um"),
                route.TS_CUTTER_R)
            lumen_gap = pocket.distance_to(um_lumen)
            required_gap = route.TS_SIDE_WALL + 0.8
            assert lumen_gap >= required_gap - 0.03, (
                f"{site['name']} pocket-to-D6-lumen gap "
                f"{lumen_gap:.3f} < {required_gap:.3f} mm")
            continue

        # A final pocket recut must not turn a buried cable tunnel into an
        # opening. LM does not use the new UM full-width web, so retain its
        # original 0.8-mm outer-cover keepout contract.
        pocket_keepout = core._axis_cylinder(
            site["face"], site["normal"], site["z_mm"],
            core.SIDE_MAGNET_POCKET_D + 1.6,
            core.SIDE_MAGNET_DEPTH + 0.8, 1.0)
        for index, cover in enumerate(route_covers[site["driver"]]):
            overlap = _intersection_volume(pocket_keepout, cover)
            assert overlap < 0.02, (
                f"{site['name']} 0.8-mm pocket keepout intersects "
                f"{site['driver']} route cover {index} by "
                f"{overlap:.3f} mm3")


def _wait_for_worker_headroom(label, minimum_mb=2500.0):
    """Require measured launch headroom without relaxing the floor guard."""
    import run_memory_guarded as memory_guard

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
    guard = Path(__file__).with_name("run_memory_guarded.py")
    return [sys.executable, str(guard), "--", *command]


def _stage_shell_contract_breps_unlocked(
        stand_foot, route_name, directory,
        shell_keys=("nominal", "lower", "upper"), *,
        include_carriers=True, seed_targets=None):
    """Build carriers and shell contracts in isolated descendants.

    Explicit local execution retains the workstation-safe cutter/segment
    chain. The osado profile has enough guarded memory to construct the final
    LM and each complete shell contract directly. Native BREP keeps the
    validation handoff lossless in both profiles, and every descendant
    remains inside the active memory guard.
    """
    script_path = Path(__file__).resolve()
    script = str(script_path)
    digest = hashlib.sha256()
    digest.update(
        f"lx-v1lf-r6f-native-stage-v{R6F_NATIVE_STAGE_SCHEMA_VERSION}"
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
    source_paths = sorted(script_path.parent.glob("top_baffle_nd25fw4*.py"))
    for source_path in source_paths:
        digest.update(source_path.name.encode("utf-8"))
        digest.update(source_path.read_bytes())
    state_name = "floor" if stand_foot else "no_floor"
    large_host = _large_host_execution()
    digest.update(b"\0full-host" if large_host else b"\0segmented-host")
    cache_root = (Path(tempfile.gettempdir()) / "lx-v1lf-r6f-brep-cache"
                  / digest.hexdigest()[:24] / state_name)
    carrier_cache = cache_root / (
        "carriers_full" if large_host else "carriers_segmented")
    shell_cache = cache_root / (
        f"shell_{route_name}_full" if large_host
        else f"shell_{route_name}_segmented")
    carrier_cache.mkdir(parents=True, exist_ok=True)
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

    targets = dict(seed_targets or {})
    if not include_carriers:
        jobs = []
    elif large_host:
        # osado builds the final carrier directly. The outer+20-cut+finalize
        # chain exists only to cap retained OCC memory on a workstation.
        jobs = [
            ("um", {"LX_R6F_EXPORT_CARRIER": "um"}),
            ("lm", {"LX_R6F_EXPORT_CARRIER": "lm"}),
        ]
    else:
        jobs = [
            ("um", {"LX_R6F_EXPORT_CARRIER": "um"}),
            ("lm_outer", {"LX_R6F_EXPORT_CARRIER": "lm_outer"}),
            *((f"lm_cut_{index}", {
                "LX_R6F_EXPORT_CARRIER": f"lm_cut_{index}"})
              for index in range(LM_CUTTER_GROUP_COUNT)),
            ("lm", {"LX_R6F_EXPORT_CARRIER": "lm_finalize"}),
        ]
    if include_carriers and route_name == "T":
        jobs.insert(1, ("tweeter", {"LX_R6F_EXPORT_TWEETER": "1"}))
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
    jobs.extend(
        job for key in shell_keys for job in shell_jobs[key])
    for label, updates in jobs:
        is_shell = "LX_R6F_EXPORT_SHELL" in updates
        suffix = ".manifest" if is_shell else ".brep"
        cache_dir = shell_cache if is_shell else carrier_cache
        target = cache_dir / f"{label}{suffix}"
        if is_shell:
            chunks = cached_shell_chunks(target)
            if chunks:
                targets[label] = chunks
                print(f"isolated {label}: reused hash-keyed native BREP cache",
                      flush=True)
                continue
        elif target.is_file() and target.stat().st_size > 0:
            targets[label] = target
            print(f"isolated {label}: reused hash-keyed native BREP cache",
                  flush=True)
            continue

        # Carrier construction is the dominant isolated peak. Shell workers
        # export each already-disjoint required solid directly (no compound
        # re-intersection). The selected profile guard continuously samples
        # the complete process tree. This admission threshold matters on the
        # local 8 GiB/0.5 GiB profile and is trivially satisfied behind
        # osado's 64 GiB host-free floor.
        launch_headroom = 3200.0
        _wait_for_worker_headroom(label, launch_headroom)
        temporary = target.with_name(
            f".{target.stem}.{os.getpid()}.{time.time_ns()}.partial{suffix}")
        env = os.environ.copy()
        env.pop("LX_R6F_SINGLE_CHECK", None)
        env.update(updates)
        if label.startswith("lm_cut_"):
            index = int(label.rsplit("_", 1)[1])
            input_label = "lm_outer" if index == 0 else f"lm_cut_{index - 1}"
            input_path = carrier_cache / f"{input_label}.brep"
            assert input_path.is_file() and input_path.stat().st_size > 0
            env["LX_R6F_LM_INPUT_PATH"] = str(input_path)
        elif label == "lm" and not large_host:
            input_path = carrier_cache / (
                f"lm_cut_{LM_CUTTER_GROUP_COUNT - 1}.brep")
            assert input_path.is_file() and input_path.stat().st_size > 0
            env["LX_R6F_LM_INPUT_PATH"] = str(input_path)
        env["LX_R6F_EXPORT_PATH"] = str(temporary)
        env["LX_STAND_FOOT"] = "1" if stand_foot else "0"
        proc = subprocess.run(
            _cad_worker_command(script), env=env, text=True,
            capture_output=True)
        assert proc.returncode == 0, (
            f"isolated {label} build failed\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
        assert temporary.is_file() and temporary.stat().st_size > 0
        if is_shell:
            chunks = tuple(Path(line) for line in temporary.read_text(
                encoding="utf-8").splitlines() if line)
            assert chunks and all(
                path.is_file() and path.stat().st_size > 0
                for path in chunks)
            temporary.replace(target)
            targets[label] = chunks
        else:
            temporary.replace(target)
            targets[label] = target
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
    """Publish shared carriers once, then stage each route concurrently."""
    lock_root = (Path(tempfile.gettempdir())
                 / "lx-v1lf-r6f-brep-cache" / "locks")
    lock_root.mkdir(parents=True, exist_ok=True)
    state_name = "floor" if stand_foot else "no_floor"
    carrier_lock_path = lock_root / f"{state_name}-carriers.lock"
    with carrier_lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        carriers = _stage_shell_contract_breps_unlocked(
            stand_foot, route_name, directory, shell_keys=())
    if not shell_keys:
        return carriers
    mode = "full" if _large_host_execution() else "segmented"
    shell_lock_path = lock_root / (
        f"{state_name}-{route_name.lower()}-{mode}-shells.lock")
    with shell_lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _stage_shell_contract_breps_unlocked(
            stand_foot, route_name, directory, shell_keys=shell_keys,
            include_carriers=False, seed_targets=carriers)


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
        env.pop("LX_R6F_SINGLE_CHECK", None)
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
            env.pop("LX_R6F_SINGLE_CHECK", None)
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
    """All V1LF cable handoffs are native flush mouths/free cable."""
    components = route.required_handoff_shell_components(route_name)
    assert components == (), (
        f"{route_name} retained a stale telescoping tongue/socket contract")
    return 0.0


def _assembled_shell_contract(stand_foot, route_name):
    _state(stand_foot)
    # Stage the final carrier BREPs before importing OCC/core/route in this
    # process; otherwise those resident modules consume the headroom needed
    # by the isolated construction child.
    staging = tempfile.TemporaryDirectory(prefix="lx-v1lf-r6f-")
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
    from top_baffle_nd25fw4 import L22_CUTOUT, THICKNESS_MM, UM_CUTOUT
    import top_baffle_nd25fw4_flush as flush
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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
        actual_parts.append(import_brep(staged["tweeter"]))
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


def test_floor_um_shell():
    _assembled_shell_contract(True, "UM")


def test_floor_t_shell():
    _assembled_shell_contract(True, "T")


def test_no_floor_um_shell():
    _assembled_shell_contract(False, "UM")


def test_no_floor_t_shell():
    _assembled_shell_contract(False, "T")


def _physical_tubes(route):
    return {
        "UM_D7": route._round_tube(
            route.route_cable_points(1.5), route.CABLE_R_EST),
        "LM_D7p8": route._round_tube(
            route.lm_cable_points(1.0), route.LM_CABLE_D_EST / 2.0),
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
        env.pop("LX_R6F_SINGLE_CHECK", None)
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
        modes = (("collision", "witness") if name == "LM_D7p8"
                 else ("overflow", "collision", "witness"))
        for segment_index in range(segment_count):
            for mode in modes:
                label = (
                    f"{owner}/{name} cable {mode} segment "
                    f"{segment_index + 1}/{segment_count}")
                _wait_for_worker_headroom(
                    label, R6F_CABLE_WORKER_HEADROOM_MB)
                env = os.environ.copy()
                env.pop("LX_R6F_SINGLE_CHECK", None)
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


def test_lm_cable_clearance():
    _carrier_cable_clearance("lm")


def test_um_cable_clearance():
    _carrier_cable_clearance("um")


def test_floor_lm_cable_clearance():
    _carrier_cable_clearance("lm", True)


def test_floor_um_cable_clearance():
    _carrier_cable_clearance("um", True)


def test_floor_support():
    _state(True)
    staged = _stage_shell_contract_breps(
        True, "LM", tempfile.gettempdir(), shell_keys=())
    support_path = Path(staged["lm"]).parent / "floor_support.brep"
    if not support_path.is_file() or support_path.stat().st_size == 0:
        _wait_for_worker_headroom("floor support", 3200.0)
        temporary = support_path.with_name(
            f".{support_path.stem}.{os.getpid()}.partial.brep")
        env = os.environ.copy()
        env.pop("LX_R6F_SINGLE_CHECK", None)
        env["LX_R6F_EXPORT_SUPPORT"] = "1"
        env["LX_R6F_EXPORT_PATH"] = str(temporary)
        env["LX_STAND_FOOT"] = "1"
        proc = subprocess.run(
            _cad_worker_command(Path(__file__).resolve()),
            env=env, text=True, capture_output=True)
        assert proc.returncode == 0, (
            f"isolated floor support build failed\nstdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}")
        assert temporary.is_file() and temporary.stat().st_size > 0
        temporary.replace(support_path)
        print(proc.stdout, end="", flush=True)
    from build123d import Rot, import_brep
    from export_piece_stls import BED_MM, BED_ROT_Z
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_attachments as addons
    import top_baffle_nd25fw4_v1lf_route as route

    support = import_brep(support_path)
    assert support.is_valid and len(support.solids()) == 1
    assert tuple(site["angle_deg"] for site in core.structural_mount_sites()) == (
        180.0, 240.0, 300.0)
    lm = import_brep(staged["lm"])
    collision = support & lm
    assert collision is None or collision.volume < 0.02, (
        f"floor support collides with LM by {collision.volume:.3f} mm3")
    assert math.isclose(
        addons.STRUCT_INSERT_FRONT_Z - addons.STRUCT_INSERT_REAR_Z,
        addons.STRUCT_INSERT_DEPTH, abs_tol=1e-12)
    assert math.isclose(
        addons.STRUCT_INSERT_HARDWARE_FRONT_Z
        - addons.STRUCT_INSERT_REAR_Z,
        addons.STRUCT_INSERT_HARDWARE_LENGTH, abs_tol=1e-12)
    assert addons.STRUCT_INSERT_HARDWARE_LENGTH == 5.8
    assert route.FLOOR_SUPPORT_INSERT_REAR_Z == addons.STRUCT_INSERT_REAR_Z
    assert (route.FLOOR_SUPPORT_INSERT_HARDWARE_LENGTH
            == addons.STRUCT_INSERT_HARDWARE_LENGTH)
    assert route.FLOOR_SUPPORT_SHANK_D == addons.STRUCT_SCREW_SHANK_D
    assert math.isclose(
        route.FLOOR_SUPPORT_INSERT_BOSS_D,
        addons.STRUCT_INSERT_BOSS_D, abs_tol=1e-12)
    assert math.isclose(
        route.FLOOR_SUPPORT_INSERT_BOSS_FRONT_Z,
        addons.SUPPORT_FLANGE_Z[1], abs_tol=1e-12)
    assert math.isclose(
        route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D,
        addons.STRUCT_INSERT_BOSS_D + 2.0 * route.SUPPORT_TUNNEL_GAP,
        abs_tol=1e-12)
    assert all(math.isclose(actual, expected, abs_tol=1e-12)
               for actual, expected in zip(
                   route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_Z,
                   (addons.STRUCT_INSERT_REAR_Z
                    - route.SUPPORT_TUNNEL_GAP,
                    addons.SUPPORT_FLANGE_Z[1]
                    + route.SUPPORT_TUNNEL_GAP)))
    assert math.isclose(
        (addons.STRUCT_INSERT_BOSS_D - addons.STRUCT_INSERT_D) / 2.0,
        2.6, abs_tol=1e-12)
    assert math.isclose(
        route.FLOOR_SUPPORT_BACKFILL_SHANK_CLEAR_D,
        addons.STRUCT_SCREW_SHANK_D + 2.0 * route.SUPPORT_TUNNEL_GAP,
        abs_tol=1e-12)
    assert math.isclose(
        addons.STRUCT_RECEIVER_D - core.STRUCT_MOUNT_KEY_D,
        0.4, abs_tol=1e-12)
    engagement = (min(core.CORE_REAR_Z, addons.SUPPORT_FLANGE_Z[1])
                  - max(core.STRUCT_MOUNT_KEY_Z[0],
                        addons.STRUCT_RECEIVER_FLOOR_Z))
    assert engagement >= 1.25
    from build123d import Cylinder, Pos
    # The lower routes deliberately cross all three support axes and the
    # carrier owns an outward radial spoke at each key.  Compare the final
    # support with that exact, intentionally relieved envelope; an unrelieved
    # D20-minus-D10 annulus is not physically available in this assembly.
    exact_route_covers = tuple(route.route_outer_covers("lm"))
    route_clearance_cutters = tuple(route.support_floor_clearance_cutters())
    spoke_clearance_cutters = addons.structural_spoke_clearance_cutters()
    intentional_socket_reliefs = (
        exact_route_covers
        + route_clearance_cutters
        + spoke_clearance_cutters)
    boss_wall_fractions = []
    socket_ring_fractions = []
    relieved_socket_fractions = []
    for site in core.structural_mount_sites():
        x, y = site["xy"]
        receiver = Pos(
            x, y,
            (addons.STRUCT_RECEIVER_FLOOR_Z
             + addons.SUPPORT_FLANGE_Z[1] + 0.2) / 2.0
        ) * Cylinder(
            addons.STRUCT_RECEIVER_D / 2.0,
            addons.SUPPORT_FLANGE_Z[1]
            - addons.STRUCT_RECEIVER_FLOOR_Z + 0.2)
        assert _intersection_volume(support, receiver) < 0.02
        socket_ring = Pos(
            x, y,
            (addons.STRUCT_RECEIVER_FLOOR_Z
             + addons.SUPPORT_FLANGE_Z[1]) / 2.0
        ) * Cylinder(
            addons.STRUCT_LUG_D / 2.0,
            addons.SUPPORT_FLANGE_Z[1]
            - addons.STRUCT_RECEIVER_FLOOR_Z)
        socket_ring -= receiver
        socket_retained = _intersection_volume(support, socket_ring)
        socket_ring_fractions.append(socket_retained / socket_ring.volume)
        relieved_socket = socket_ring
        for cutter in intentional_socket_reliefs:
            relieved_socket -= cutter
        assert relieved_socket is not None and relieved_socket.volume > 0.01
        relieved_retained = _intersection_volume(support, relieved_socket)
        relieved_socket_fractions.append(
            relieved_retained / relieved_socket.volume)
        # Independent final-BREP witnesses: the carrier must expose nearly
        # the complete D12.4 boss envelope below its z=5.3 pad, while the
        # support must retain a substantial connected annulus around the
        # D6.4 heat-set cavity. These do not reuse the backfill-plan builder.
        z0 = addons.STRUCT_INSERT_REAR_Z + 0.2
        z1 = addons.STRUCT_INSERT_HARDWARE_FRONT_Z - 0.2
        carrier_clearance = Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
            route.FLOOR_SUPPORT_BACKFILL_INSERT_CLEAR_D / 2.0 - 0.1,
            z1 - z0)
        assert _intersection_volume(lm, carrier_clearance) < 0.02, (
            f"floor carrier failed D12.4 boss clearance at {site}")
        boss_wall = Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
            addons.STRUCT_INSERT_BOSS_D / 2.0 - 0.1, z1 - z0)
        boss_wall -= Pos(x, y, (z0 + z1) / 2.0) * Cylinder(
            addons.STRUCT_INSERT_D / 2.0 + 0.1, z1 - z0)
        retained = _intersection_volume(support, boss_wall)
        boss_wall_fractions.append(retained / boss_wall.volume)
        assert retained > 0.75 * boss_wall.volume, (
            f"support heat-set boss wall retained only "
            f"{retained / boss_wall.volume:.1%} at {site}")
    assert min(relieved_socket_fractions) > 0.98, (
        "support retention of intentionally relieved D20 key-socket "
        "envelopes by 180/240/300-degree site: "
        f"{[f'{fraction:.1%}' for fraction in relieved_socket_fractions]}; "
        "raw unrelieved fractions (diagnostic only): "
        f"{[f'{fraction:.1%}' for fraction in socket_ring_fractions]}")
    for cable in _physical_tubes(route).values():
        hit = support & cable
        assert hit is None or hit.volume < 0.10
    for keepout in route_clearance_cutters:
        hit = support & keepout
        assert hit is None or hit.volume < 0.05
    hardware = addons.structural_hardware_proxies()
    for printed, label in ((support, "support"), (lm, "LM")):
        hit = hardware & printed
        assert hit is None or hit.volume < 0.05, (
            f"floor hardware envelope intersects {label} by "
            f"{hit.volume:.3f} mm3")
    for cable in _physical_tubes(route).values():
        hit = hardware & cable
        assert hit is None or hit.volume < 0.05
    load = addons.structural_load_facts()
    assert load["design_mass_kg"] == 4.0
    assert load["design_y_cg_mm"] == 230.0
    assert load["rear_cg_mm"] == 70.0
    assert load["creep_allow_mpa"] == 8.0
    assert load["short_allow_mpa"] == 18.0
    assert load["insert_pullout_n"] == 600.0
    assert load["member_span_mm"] == 110.0
    assert load["magnet_load_credit_n"] == 0.0
    assert load["combined_insert_5g_n"] > load["normal_insert_5g_n"]
    assert load["actual_section_mm3"] >= load["required_section_5g_mm3"]
    assert load["member_sf_static_creep"] >= 2.0
    assert load["member_sf_3g"] >= 1.5
    assert load["member_sf_5g"] >= 1.05
    assert load["insert_sf_5g"] >= 1.9
    rotated = Rot(Z=BED_ROT_Z["v1lf_addon_mount_floor_support"]) * support
    bb = rotated.bounding_box().size
    assert BED_MM == 256.0
    assert max(bb.X, bb.Y, bb.Z) <= BED_MM, (
        f"floor support bed footprint {bb.X:.2f} x {bb.Y:.2f} x "
        f"{bb.Z:.2f} exceeds {BED_MM:.1f} mm")
    print(
        f"  floor support: three support heat-sets, min boss-wall witness "
        f"{min(boss_wall_fractions):.1%}, 1/3/5g section screen, "
        f"bed {bb.X:.2f} x {bb.Y:.2f}")


def _tweeter_and_service_legacy_monolithic(stand_foot):
    _state(stand_foot)
    staging = tempfile.TemporaryDirectory(prefix="lx-v1lf-service-")
    staged = _stage_shell_contract_breps(
        stand_foot, "T", staging.name, shell_keys=())
    from build123d import import_brep
    import top_baffle_nd25fw4_um_fit as fit
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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
    assert (fit.REMOVAL_ENVELOPE_CABLE_POLICY["v1lf"]
            == "independent_flag_faston_pull_with_slack_leads")

    harness = fit.v1lf_terminal_harness_parts()
    bundle = harness["v1lf_D7_bundle_to_Y_breakout"]
    leads = {
        1: harness["v1lf_terminal_lead_1_D3p2"],
        2: harness["v1lf_terminal_lead_2_D3p2"],
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
    t_cable = fit.v1lf_ts_cable_envelope()
    t_body = t_cable & body
    assert t_body is None or t_body.volume < 0.10
    lm_cable_body = fit.v1lf_lm_cable_envelope() & body
    assert lm_cable_body is None or lm_cable_body.volume < 0.10
    for service_name, service_part in service_parts.items():
        collision = t_cable & service_part
        volume = 0.0 if collision is None else collision.volume
        assert volume < 0.10, (
            f"T cable/{service_name} collision {volume:.3f}")
    terminated_points = fit.v1lf_terminated_cable_points()
    assert fit.V1LF_TERMINATED_HANDOFF_R == 20.0
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
        tangential - fit.V1LF_TERMINATED_HANDOFF_R * np.sin(phi))) < 1e-6
    assert np.max(np.abs(
        free_arc[:, 2] - (route.UM_MOUTH_Z
                          - fit.V1LF_TERMINATED_HANDOFF_R
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

    installed_facts = fit.v1lf_terminal_harness_facts(0.0)
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        lead_points = fit.v1lf_terminal_lead_points(pull_mm)
        harness_facts = fit.v1lf_terminal_harness_facts(pull_mm)
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

    separated = fit.v1lf_separated_lead_parts(0.0)
    assert separated["terminal_lead_1"].distance_to(
        separated["terminal_lead_2"]) >= 0.49
    breakout_parts = fit.v1lf_y_breakout_boot_parts(0.0)
    assert set(breakout_parts) == {
        "v1lf_Y_breakout_bundle_heatshrink",
        "v1lf_Y_breakout_terminal_lead_1_heatshrink",
        "v1lf_Y_breakout_terminal_lead_2_heatshrink",
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
        fit.split_grommet_parts("v1lf")
    except ValueError:
        pass
    else:
        raise AssertionError("V1LF must not generate printed grommet parts")

    for x in core.TWEETER_JOINT_X:
        core_ear = core._plan_prism(
            core.tweeter_joint_polygon(x), *core.TWEETER_CORE_JOINT_Z)
        addon_ear = core._plan_prism(
            core.tweeter_joint_polygon(x), *core.TWEETER_ADDON_JOINT_Z)
        assert _intersection_volume(um, core_ear) > 0.45 * core_ear.volume
        assert _intersection_volume(tweeter, core_ear) < 0.02
        assert _intersection_volume(tweeter, addon_ear) > 0.45 * addon_ear.volume
        assert _intersection_volume(um, addon_ear) < 0.02
        insert = core._cylinder_at(
            x, core.TWEETER_JOINT_Y,
            core.TWEETER_JOINT_INSERT_BORE_D / 2.0,
            core.TWEETER_ADDON_JOINT_Z[0] - 0.1,
            core.TWEETER_ADDON_JOINT_Z[0] + 4.0)
        assert _intersection_volume(tweeter, insert) < 0.02
    state = "floor" if stand_foot else "no-floor"
    staging.cleanup()
    print(
        f"  {state}: tweeter ears, free rear T cable, named Faston/boot/"
        "removal proxies and strain relief pass")


_SERVICE_PRINTED_COMPONENT_SPECS = {
    f"printed_component_{owner}_{token}": (owner, family, part_name)
    for owner in ("lm", "um", "tweeter")
    for token, family, part_name in (
        ("harness_bundle", "harness", "v1lf_D7_bundle_to_Y_breakout"),
        ("harness_lead1", "harness", "v1lf_terminal_lead_1_D3p2"),
        ("harness_lead2", "harness", "v1lf_terminal_lead_2_D3p2"),
        ("boot_bundle", "boot", "v1lf_Y_breakout_bundle_heatshrink"),
        ("boot_lead1", "boot",
         "v1lf_Y_breakout_terminal_lead_1_heatshrink"),
        ("boot_lead2", "boot",
         "v1lf_Y_breakout_terminal_lead_2_heatshrink"),
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
    import top_baffle_nd25fw4_v1lf as core
    import top_baffle_nd25fw4_v1lf_route as route

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

    terminated_points = fit.v1lf_terminated_cable_points()
    assert fit.V1LF_TERMINATED_HANDOFF_R == 20.0
    assert _min_three_point_radius(terminated_points) >= 13.9
    # The physical loft uses 1.5-mm stations to bound OCC memory; its R15
    # chord turn is therefore about six degrees.  Smoothness itself is
    # proved on the 0.2-mm printed path in ``test_route_contract`` and by the
    # exact G1/R20 checks below, rather than conflating chord density with a
    # geometric kink.
    assert _max_turn_deg(terminated_points) <= 6.1
    arc_count = fit.V1LF_TERMINATED_HANDOFF_STEPS + 1
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
        tangential - fit.V1LF_TERMINATED_HANDOFF_R * np.sin(phi))) < 1e-6
    assert np.max(np.abs(
        free_arc[:, 2] - (route.UM_MOUTH_Z
                          - fit.V1LF_TERMINATED_HANDOFF_R
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
    installed_facts = fit.v1lf_terminal_harness_facts(0.0)
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        lead_points = fit.v1lf_terminal_lead_points(pull_mm)
        harness_facts = fit.v1lf_terminal_harness_facts(pull_mm)
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

    states = fit.v1lf_independent_pull_states()
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
    assert (fit.REMOVAL_ENVELOPE_CABLE_POLICY["v1lf"]
            == "independent_flag_faston_pull_with_slack_leads")
    for service_name, service_part in service_parts.items():
        if service_name in {
                "terminal_carrier", "terminal_tab_1", "terminal_tab_2"}:
            continue
        assert _intersection_volume(service_part, body) < 0.10, (
            f"{service_name}/known MU10 body collision")
    t_cable = fit.v1lf_ts_cable_envelope()
    assert _intersection_volume(t_cable, body) < 0.10
    lm_cable = fit.v1lf_lm_cable_envelope()
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
    harness = fit.v1lf_terminal_harness_parts()
    bundle = harness["v1lf_D7_bundle_to_Y_breakout"]
    leads = {
        1: harness["v1lf_terminal_lead_1_D3p2"],
        2: harness["v1lf_terminal_lead_2_D3p2"],
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
    separated = fit.v1lf_separated_lead_parts(0.0)
    assert separated["terminal_lead_1"].distance_to(
        separated["terminal_lead_2"]) >= 0.49


def _service_pull_clearance(fit, terminal_id):
    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    fastons = fit.faston_proxy_parts()
    boots = fit.faston_boot_proxy_parts()
    sweep = fit.faston_pull_sweep_parts()[
        f"faston_flag_pull_sweep_{terminal_id}"]
    harness = fit.v1lf_terminal_harness_parts()
    other = 2 if terminal_id == 1 else 1
    other_lead = harness[f"v1lf_terminal_lead_{other}_D3p2"]
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
    installed = fit.v1lf_terminal_harness_facts_for_terminal_pull(
        terminal_id, 0.0)
    installed_length = installed["lead_lengths_mm"][
        f"terminal_lead_{terminal_id}"]
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        pulls = fit.v1lf_independent_pull_state(terminal_id, pull_mm)
        facts = fit.v1lf_terminal_service_state_facts(
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
        separated = fit.v1lf_separated_lead_parts_by_terminal(pulls)
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

        y_parts = fit.v1lf_y_breakout_boot_parts_by_terminal(pulls)
        y_branch = y_parts[
            f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        y_envelope = fit.v1lf_y_breakout_boot_envelope_by_terminal(pulls)
        assert y_envelope.is_valid and len(y_envelope.solids()) == 1
        assert y_branch.distance_to(body) >= 0.20
        assert y_branch.distance_to(lm_body) >= 0.75
        assert _intersection_volume(y_branch, installed_lead) < 0.01
        assert _intersection_volume(
            y_branch, fastons[f"faston_receptacle_{other}"]) < 0.01
        for cable_name, cable in (
                fit.v1lf_y_breakout_cable_parts_by_terminal(pulls).items()):
            remainder = cable - y_envelope
            missing = 0.0 if remainder is None else remainder.volume
            assert missing < 0.02, (
                f"Y state {terminal_id}/{pull_mm:g} misses "
                f"{cable_name} by {missing:.4f} mm3")


def _service_independent_printed(fit, paths, name, terminal_id):
    from build123d import import_brep

    printed = import_brep(paths[name])
    for pull_mm in fit.FASTON_LEAD_PULL_STATES_MM:
        pulls = fit.v1lf_independent_pull_state(terminal_id, pull_mm)
        lead = fit.v1lf_separated_lead_parts_by_terminal(pulls)[
            f"terminal_lead_{terminal_id}"]
        branch = fit.v1lf_y_breakout_boot_parts_by_terminal(pulls)[
            f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        assert _intersection_volume(printed, lead) < 0.10, (
            f"{name}/terminal {terminal_id} lead at {pull_mm:g} mm")
        assert _intersection_volume(printed, branch) < 0.10, (
            f"{name}/terminal {terminal_id} Y branch at {pull_mm:g} mm")


def _service_breakout_boot(fit):
    from top_baffle_nd25fw4_cables import _tube_loft

    body = fit.mu10_body_keepout()
    lm_body = fit.w22_body_keepout()
    breakout_parts = fit.v1lf_y_breakout_boot_parts(0.0)
    assert set(breakout_parts) == {
        "v1lf_Y_breakout_bundle_heatshrink",
        "v1lf_Y_breakout_terminal_lead_1_heatshrink",
        "v1lf_Y_breakout_terminal_lead_2_heatshrink",
    }
    assert all(part.is_valid and part.volume > 0.01
               for part in breakout_parts.values())
    paths = fit._v1lf_y_breakout_paths_by_terminal(
        fit._uniform_pull_state(0.0))
    bundle_radius = ((fit.FASTON_BREAKOUT_BUNDLE_OD / 2.0)
                     / math.cos(math.pi / 24.0))
    incoming_bundle = _tube_loft(
        paths["bundle"], bundle_radius, sides=24)
    junction = fit._v1lf_y_breakout_junction()
    assert _intersection_volume(incoming_bundle, junction) > 1.0
    for terminal_id in (1, 2):
        branch = breakout_parts[
            f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink"]
        assert _intersection_volume(branch, junction) > 1.0
    for boot_name, boot in breakout_parts.items():
        assert boot.distance_to(body) >= 0.20, boot_name
        assert boot.distance_to(lm_body) >= 0.75, boot_name
    breakout = fit.v1lf_y_breakout_boot_envelope(0.0)
    assert breakout.is_valid and len(breakout.solids()) == 1
    cable_parts = fit.v1lf_y_breakout_cable_parts(0.0)
    missing = 0.0
    for cable_name, cable in cable_parts.items():
        remainder = cable - breakout
        volume = 0.0 if remainder is None else remainder.volume
        missing += volume
        assert volume < 0.02, f"Y boot misses {cable_name} by {volume:.4f}"
    facts = fit.v1lf_y_breakout_facts(0.0)
    assert facts["junction_length_mm"] > 0.0
    assert facts["junction_overlap_each_side_mm"] > 0.0
    assert facts["junction_min_underlying_cable_margin_mm"] >= -1e-9
    assert missing < 0.04

    try:
        fit.split_grommet_parts("v1lf")
    except ValueError:
        pass
    else:
        raise AssertionError("V1LF must not generate printed grommet parts")


def _service_printed_body(fit, paths, name):
    from build123d import import_brep
    import top_baffle_nd25fw4_v1lf_route as route

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
    for cable_name, cable in fit.v1lf_terminal_harness_parts().items():
        assert _intersection_volume(printed, cable) < 0.10, (
            f"{cable_name}/{name} collision")
    for boot_name, boot in fit.v1lf_y_breakout_boot_parts(0.0).items():
        assert _intersection_volume(printed, boot) < 0.10, (
            f"{boot_name}/{name} collision")


def _service_printed_component(
        fit, paths, name, family, part_name):
    """Check one final carrier against one harness/boot loft per worker."""
    from build123d import import_brep

    printed = import_brep(paths[name])
    pulls = fit._uniform_pull_state(0.0)
    if family == "harness":
        component = fit.v1lf_terminal_harness_part_by_terminal(
            part_name, pulls)
    elif family == "boot":
        component = fit.v1lf_y_breakout_boot_part_by_terminal(
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
    pulls = fit.v1lf_independent_pull_state(terminal_id, pull_mm)
    lead = fit.v1lf_separated_lead_part_by_terminal(terminal_id, pulls)
    branch_name = (
        f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
    branch = fit.v1lf_y_breakout_boot_part_by_terminal(branch_name, pulls)
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
            return fit.v1lf_terminal_harness_part_by_terminal(
                part_name, pulls)
        if family == "boot":
            return fit.v1lf_y_breakout_boot_part_by_terminal(
                part_name, pulls)
        raise ValueError(family)
    if phase in _SERVICE_INDEPENDENT_PRINTED_SPECS:
        _name, terminal_id, pull_mm = (
            _SERVICE_INDEPENDENT_PRINTED_SPECS[phase])
        pulls = fit.v1lf_independent_pull_state(terminal_id, pull_mm)
        if selector == "lead":
            return fit.v1lf_separated_lead_part_by_terminal(
                terminal_id, pulls)
        if selector == "branch":
            part_name = (
                f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
            return fit.v1lf_y_breakout_boot_part_by_terminal(
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
            prefix="lx-v1lf-service-component-") as directory:
        for selector in selectors:
            chunk_count = _service_component_chunk_count(phase, selector)
            component_path = Path(directory) / (
                f"{selector}.manifest" if chunk_count > 1
                else f"{selector}.brep")
            _wait_for_worker_headroom(
                f"terminal-service {phase}/{selector} build", 3200.0)
            env = os.environ.copy()
            env.pop("LX_R6F_SINGLE_CHECK", None)
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
                env.pop("LX_R6F_SINGLE_CHECK", None)
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
    import top_baffle_nd25fw4_v1lf as core

    part = import_brep(paths[name])
    if name == "tweeter":
        um = import_brep(paths["um"])
        actual_overlap = _intersection_volume(part, um)
        assert actual_overlap < 0.02, (
            f"actual UM/tweeter print collision {actual_overlap:.4f} mm3")
    for x in core.TWEETER_JOINT_X:
        core_ear = core._plan_prism(
            core.tweeter_joint_polygon(x), *core.TWEETER_CORE_JOINT_Z)
        addon_ear = core._plan_prism(
            core.tweeter_joint_polygon(x), *core.TWEETER_ADDON_JOINT_Z)
        core_hit = _intersection_volume(part, core_ear)
        addon_hit = _intersection_volume(part, addon_ear)
        if name == "um":
            assert core_hit > 0.45 * core_ear.volume, (
                f"UM/core tweeter ear at x={x:g}: {core_hit:.4f}/"
                f"{core_ear.volume:.4f} mm3")
            assert addon_hit < 0.02, (
                f"UM enters add-on tweeter ear at x={x:g} by "
                f"{addon_hit:.4f} mm3")
        else:
            # The closed T conduit may occupy a tiny rear corner of the
            # analytic ear prism where the final UM has already been radially
            # relieved. Exact print/print and bolt-envelope checks govern;
            # retain a hard bound so that this cannot grow into the joint.
            assert core_hit < 1.0, (
                f"tweeter nominal core-ear intrusion at x={x:g} is "
                f"{core_hit:.4f} mm3")
            assert addon_hit > 0.45 * addon_ear.volume, (
                f"tweeter/add-on ear at x={x:g}: {addon_hit:.4f}/"
                f"{addon_ear.volume:.4f} mm3")
            core_bolt = core._cylinder_at(
                x, core.TWEETER_JOINT_Y,
                core.TWEETER_JOINT_HOLE_D / 2.0,
                core.TWEETER_CORE_JOINT_Z[0] - 0.2,
                core.TWEETER_CORE_JOINT_Z[1] + 0.3)
            assert _intersection_volume(part, core_bolt) < 0.02
            insert = core._cylinder_at(
                x, core.TWEETER_JOINT_Y,
                core.TWEETER_JOINT_INSERT_BORE_D / 2.0,
                core.TWEETER_ADDON_JOINT_Z[0] - 0.1,
                core.TWEETER_ADDON_JOINT_Z[0] + 4.0)
            assert _intersection_volume(part, insert) < 0.02


def _service_phase_worker(phase, paths, stand_foot):
    _state(stand_foot)
    import top_baffle_nd25fw4_um_fit as fit

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
    env.pop("LX_R6F_SINGLE_CHECK", None)
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
    import top_baffle_nd25fw4_um_fit as fit

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
            component = fit.v1lf_terminal_harness_part_by_terminal(
                part_name, pulls)
        else:
            assert family == "boot"
            component = fit.v1lf_y_breakout_boot_part_by_terminal(
                part_name, pulls)
        assert component.is_valid and component.volume > 0.01
        for owner, carrier in printed.items():
            collision = _intersection_volume(carrier, component)
            assert collision < 0.10, (
                f"{owner}/{part_name} collision {collision:.3f} mm3")

    for terminal_id in (1, 2):
        branch_name = (
            f"v1lf_Y_breakout_terminal_lead_{terminal_id}_heatshrink")
        for pull_mm in (0.0, 3.0, 6.0, 9.0, 12.0):
            independent = fit.v1lf_independent_pull_state(
                terminal_id, pull_mm)
            lead = fit.v1lf_separated_lead_part_by_terminal(
                terminal_id, independent)
            branch = fit.v1lf_y_breakout_boot_part_by_terminal(
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
    with tempfile.TemporaryDirectory(prefix="lx-v1lf-service-") as directory:
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


def test_tweeter_and_service():
    _tweeter_and_service(False)


def test_floor_tweeter_and_service():
    _tweeter_and_service(True)


CHECKS = [
    test_route_contract,
    test_w22_reference_step_geometry,
    test_insert_bump_clearance,
    test_floor_insert_bump_clearance,
    test_no_floor_route_smoothness,
    test_floor_route_smoothness,
    test_bump_brep_clearance,
    test_floor_bump_brep_clearance,
    test_bump_backfill_contract,
    test_floor_bump_backfill_contract,
    test_lm_burial_web_contract,
    test_floor_lm_burial_web_contract,
    test_um_burial_web_contract,
    test_floor_um_burial_web_contract,
    test_feed_and_flush_mouth_contract,
    test_floor_feed_and_flush_mouth_contract,
    test_crossover_brep,
    test_floor_crossover_brep,
    test_bridge_contract,
    test_bridge_geometry,
    test_joint_load_contract,
    test_floor_lm_core,
    test_no_floor_lm_core,
    test_floor_lm_keyed_split,
    test_no_floor_lm_keyed_split,
    test_floor_um_shell,
    test_floor_t_shell,
    test_no_floor_um_shell,
    test_no_floor_t_shell,
    test_lm_cable_clearance,
    test_um_cable_clearance,
    test_floor_lm_cable_clearance,
    test_floor_um_cable_clearance,
    test_floor_support,
    test_tweeter_and_service,
    test_floor_tweeter_and_service,
]

_SERVICE_ORCHESTRATOR_CHECKS = {
    "test_tweeter_and_service",
    "test_floor_tweeter_and_service",
}


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
        import top_baffle_nd25fw4_um_fit as fit
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
    support_export = os.environ.get("LX_R6F_EXPORT_SUPPORT")
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
        from build123d import import_brep
        import top_baffle_nd25fw4_v1lf_route as route

        carrier = import_brep(carrier_path)
        specs = {
            "UM_D7": (
                route.route_cable_points(1.5),
                route.CABLE_R_EST, route.CUTTER_R),
            "LM_D7p8": (
                route.lm_cable_points(1.0),
                route.LM_CABLE_D_EST / 2.0, None),
            "T_D5p2": (
                route.ts_cable_points(1.5),
                route.TS_CABLE_D_EST / 2.0, route.TS_CUTTER_R),
        }
        for cable_name, (points, physical_radius, nominal_radius) in (
                specs.items()):
            # LM_D7p8 is an intentionally free short lead: there is no
            # printed nominal D8.2 tunnel against which an overflow could be
            # meaningful.  Its physical and +0.05-mm witness collisions stay
            # mandatory against both carriers.
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
        import top_baffle_nd25fw4_v1lf_route as route

        specs = {
            "UM_D7": (
                route.route_cable_points(1.5),
                route.CABLE_R_EST, route.CUTTER_R),
            "LM_D7p8": (
                route.lm_cable_points(1.0),
                route.LM_CABLE_D_EST / 2.0, None),
            "T_D5p2": (
                route.ts_cable_points(1.5),
                route.TS_CABLE_D_EST / 2.0, route.TS_CUTTER_R),
        }
        if cable_validation not in specs:
            raise SystemExit(f"unknown cable validation: {cable_validation}")
        points, physical_radius, nominal_radius = specs[cable_validation]
        if mode == "overflow" and nominal_radius is None:
            raise SystemExit(
                "LM_D7p8 is a free lead with no nominal tunnel overflow "
                "contract")
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

    if carrier_export or shell_export or tweeter_export or support_export:
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
            import top_baffle_nd25fw4_v1lf as core
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
            import top_baffle_nd25fw4_v1lf_route as route
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
            import top_baffle_nd25fw4_v1lf_attachments as addons
            part = addons.tweeter_crescent()
            label = "tweeter crescent"
        else:
            import top_baffle_nd25fw4_v1lf_attachments as addons
            part = addons.floor_support()
            label = "floor support"
        if not export_brep(part, target):
            raise SystemExit(f"failed to export isolated {label} BREP")
        print(f"isolated {label}: {part.volume:.1f} mm3", flush=True)
        return

    single = os.environ.get("LX_R6F_SINGLE_CHECK")
    if single:
        import run_memory_guarded as memory_guard
        local_service_orchestrator = (
            single in _SERVICE_ORCHESTRATOR_CHECKS
            and not _large_host_execution())
        if (not memory_guard.is_guarded_process()
                and not local_service_orchestrator):
            guard = Path(__file__).with_name("run_memory_guarded.py")
            proc = subprocess.run(
                [sys.executable, str(guard), "--", sys.executable,
                 str(Path(__file__).resolve())],
                env=os.environ.copy())
            raise SystemExit(proc.returncode)
        check = next((fn for fn in CHECKS if fn.__name__ == single), None)
        if check is None:
            raise SystemExit(f"unknown R6F check: {single}")
        # Starting a complete check exactly at a profile's hard floor can
        # force the guard to kill a later in-process BREP comparison. Admit
        # every check only after the host has recovered a measured 2500 MiB;
        # carrier/shell/support workers retain their 3200 MiB gate and the
        # segmented cable workers retain 3500 MiB. This is launch hysteresis,
        # not a relaxation of the selected profile's kill floor.
        _wait_for_worker_headroom(
            f"R6F check {single}", R6F_CHECK_LAUNCH_HEADROOM_MB)
        print(f"{single}:", flush=True)
        check()
        return

    guard = Path(__file__).with_name("run_memory_guarded.py")

    def run_check(check):
        env = os.environ.copy()
        # Private worker selectors are control-flow inputs handled before
        # LX_R6F_SINGLE_CHECK in main(). Never let stale shell state turn a
        # requested acceptance check into an unrelated successful worker.
        for name in tuple(env):
            if name.startswith("LX_R6F_"):
                env.pop(name)
        env["LX_R6F_SINGLE_CHECK"] = check.__name__
        local_service_orchestrator = (
            check.__name__ in _SERVICE_ORCHESTRATOR_CHECKS
            and not _large_host_execution())
        command = (
            [sys.executable, str(Path(__file__).resolve())]
            if local_service_orchestrator
            else [sys.executable, str(guard), "--", sys.executable,
                  str(Path(__file__).resolve())]
        )
        proc = subprocess.run(
            command,
            env=env, text=True, capture_output=True)
        return check.__name__, proc.returncode, proc.stdout, proc.stderr

    requested_workers = int(os.environ.get("LX_CAD_GUARD_SLOTS", "1"))
    workers = (min(requested_workers, len(CHECKS))
               if _large_host_execution() else 1)
    results = []
    if workers == 1:
        for check in CHECKS:
            results.append(run_check(check))
            name, returncode, stdout, stderr = results[-1]
            print(stdout, end="", flush=True)
            if stderr:
                print(stderr, end="", file=sys.stderr, flush=True)
    else:
        print(
            f"R6F remote runner: {workers} concurrent isolated checks; "
            "shared per-state BREP staging is lock-serialized",
            flush=True)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_check, check): check.__name__
                for check in CHECKS
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                _name, _returncode, stdout, stderr = result
                print(stdout, end="", flush=True)
                if stderr:
                    print(stderr, end="", file=sys.stderr, flush=True)
    failures = {name for name, returncode, _stdout, _stderr in results
                if returncode}
    failed = [check.__name__ for check in CHECKS
              if check.__name__ in failures]
    if failed:
        raise SystemExit("R6F FAILED: " + ", ".join(failed))
    print("\nall final V1LF R6F checks passed")


if __name__ == "__main__":
    main()
