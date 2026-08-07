"""Focused source/BREP contract for the Stock/Slim TEBM35C10-4 family."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from lx521_baffle.base import THICKNESS_MM, UM_CUTOUT
from lx521_baffle.cables import (
    TS_ROUTE_CAPTIVE,
    route_points,
)
from lx521_baffle.magnet_contract import CAPTIVE_LAND_MM
from lx521_baffle.proud.top_baffle_nd25fw4_b2_split import (
    CLEARANCE_MM,
    DOVETAILS_B,
    SEAM_B_M3_AXIS_X_MM,
    SEAM_B_M3_AXIS_Z_MM,
    SEAM_B_M3_INSERT_BORE_D_MM,
    SEAM_B_M3_INSERT_DEPTH_MM,
    SEAM_B_M3_INSERT_ENGAGEMENT_MM,
    SEAM_B_M3_INSERT_TIP_MARGIN_MM,
    SEAM_B_M3_VASE_FACE_Y_MM,
    SEAM_B_Y,
    seam_b_m3_vase_insert_cutter,
)
from lx521_baffle.proud.vase_tebm35c10_4 import (
    BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM,
    DUCT_APPROVED_SEAM_MOUTH_CENTER_XYZ_MM,
    DUCT_APPROVED_SEAM_MOUTH_SIZE_XYZ_MM,
    DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3,
    DUCT_EXTERIOR_SKIN_GUARD_MM,
    LOWER_T_AXIS_Y_MM,
    LOWER_T_MOUNT_CLOCK_DEG,
    LOWER_T_OUTLET_Z_MM,
    LOWER_T_POCKET_REAR_Z_MM,
    M2_INSERT_BORE_D_MM,
    M2_INSERT_DEPTH_MM,
    MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG,
    MAIN_T_LOWER_OUTLET_XY_MM,
    MAIN_T_ROUTE_ENTRY_Z_MM,
    MAIN_T_ROUTE_NAME,
    MAIN_T_ROUTE_SEAM_XY_MM,
    MAIN_T_UM_CLEARANCE_RADIUS_MM,
    MAIN_T_UM_NOMINAL_INNER_LIGAMENT_MM,
    MAIN_T_UM_PILOT_VERTICAL_LIGAMENT_MM,
    PAIR_AXIS_PITCH_MM,
    PART_NAME,
    REAR_GROWTH_MM,
    REAR_RAMP_END_Y_MM,
    REAR_RAMP_LENGTH_MM,
    REAR_RAMP_START_Y_MM,
    REAR_T_MOUNT_Z_MM,
    SLIM_PROFILE,
    STOCK_PROFILE,
    T_BLIND_BACK_WALL_THICKNESS_MM,
    T_CLEAR_POCKET_DEPTH_MM,
    TEBM_BASKET_D_MM,
    TEBM_CUTOUT_D_MM,
    TEBM_DEPTH_MM,
    TEBM_LAND_D_MM,
    TEBM_MAX_D_MM,
    TEBM_MOUNT_HOLE_COUNT,
    TEBM_MOUNT_PCD_MM,
    T_MAGNET_FACE_X_MM,
    T_MAGNET_FLAT_EDGE_MARGIN_MM,
    T_CABLE_FACE_INSET_MM,
    UPPER_T_BRANCH_CLEARANCE_RADIUS_MM,
    UPPER_T_BRANCH_D_MM,
    UPPER_T_BRANCH_GUIDE_RADIUS_MM,
    UPPER_T_BRANCH_MIN_INNER_LIGAMENT_MM,
    UPPER_T_BRANCH_LOWER_INSERT_VERTICAL_LIGAMENT_MM,
    UPPER_T_BRANCH_MIN_OUTER_LIGAMENT_MM,
    UPPER_T_BRANCH_OUTLET_XY_MM,
    UPPER_T_BRANCH_OUTLET_Z_MM,
    UPPER_T_BRANCH_SPLIT_Y_MM,
    UPPER_T_BRANCH_SPLIT_Z_MM,
    UPPER_T_AXIS_Y_MM,
    UPPER_T_MOUNT_CLOCK_DEG,
    UPPER_T_POCKET_FRONT_Z_MM,
    build_model,
    design_facts,
    duct_exposure_residuals,
    duct_unapproved_opening_residuals,
    external_envelope,
    _main_t_branch_split_parameter,
    _main_t_cable_duct,
    _main_t_center_z_mm,
    _upper_t_cable_duct,
    optimized_main_t_centerline_points,
    optimized_main_t_path,
    upper_t_branch_centerline_points,
    upper_t_branch_path,
    upper_t_branch_split_tangent_error_deg,
    vase_profile,
)


def _close(actual: float, expected: float, tolerance: float = 1.0e-6) -> None:
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=0.0), (
        actual, expected)


def _mount_centers(axis_y: float, clock: float):
    radius = TEBM_MOUNT_PCD_MM / 2.0
    for index in range(TEBM_MOUNT_HOLE_COUNT):
        angle = math.radians(clock + 90.0 * index)
        yield radius * math.cos(angle), axis_y + radius * math.sin(angle)


def test_dimension_contract() -> None:
    assert vase_profile("stock") is STOCK_PROFILE
    assert vase_profile("slim") is SLIM_PROFILE
    _close(STOCK_PROFILE.rear_surface_z_mm, 0.0)
    _close(SLIM_PROFILE.rear_surface_z_mm, 6.8)
    _close(STOCK_PROFILE.section_depth_mm, 18.3)
    _close(SLIM_PROFILE.section_depth_mm, 11.5)
    _close(STOCK_PROFILE.local_rear_growth_mm, 6.8)
    _close(SLIM_PROFILE.local_rear_growth_mm, 13.6)
    _close(STOCK_PROFILE.rear_ramp_start_y_mm, REAR_RAMP_START_Y_MM)
    _close(SLIM_PROFILE.rear_ramp_start_y_mm, SEAM_B_Y)
    _close(LOWER_T_AXIS_Y_MM, 443.931)
    _close(UPPER_T_AXIS_Y_MM, 493.231)
    _close(PAIR_AXIS_PITCH_MM, 49.3)
    _close(TEBM_CUTOUT_D_MM, 42.926)
    _close(PAIR_AXIS_PITCH_MM - TEBM_CUTOUT_D_MM, 6.374)
    _close(
        PAIR_AXIS_PITCH_MM - TEBM_MAX_D_MM / 2.0
        - TEBM_BASKET_D_MM / 2.0,
        BODY_TO_OPPOSITE_FLANGE_CLEARANCE_MM,
    )
    _close(REAR_GROWTH_MM, TEBM_DEPTH_MM - THICKNESS_MM)
    _close(REAR_T_MOUNT_Z_MM, -6.8)
    assert REAR_RAMP_START_Y_MM < REAR_RAMP_END_Y_MM < LOWER_T_AXIS_Y_MM
    _close(REAR_RAMP_LENGTH_MM,
           REAR_RAMP_END_Y_MM - REAR_RAMP_START_Y_MM)
    assert TEBM_LAND_D_MM == 66.0
    assert M2_INSERT_BORE_D_MM == 3.2
    assert M2_INSERT_DEPTH_MM == 4.0
    assert T_BLIND_BACK_WALL_THICKNESS_MM == 1.2
    _close(T_CLEAR_POCKET_DEPTH_MM, 23.9)
    _close(LOWER_T_POCKET_REAR_Z_MM, -5.6)
    _close(UPPER_T_POCKET_FRONT_Z_MM, 17.1)
    assert MAIN_T_ROUTE_NAME == "vase_tebm_um_contained_3d_g1_y"
    _close(MAIN_T_UM_CLEARANCE_RADIUS_MM, 46.3)
    _close(MAIN_T_CLEARANCE_ARC_EXIT_ANGLE_DEG, 114.5)
    _close(MAIN_T_UM_NOMINAL_INNER_LIGAMENT_MM, 2.0)
    _close(MAIN_T_UM_PILOT_VERTICAL_LIGAMENT_MM, 1.65)
    assert UPPER_T_BRANCH_D_MM == 4.6
    assert UPPER_T_BRANCH_SPLIT_Y_MM < LOWER_T_AXIS_Y_MM
    _close(UPPER_T_BRANCH_SPLIT_Y_MM, 408.21220684158396)
    _close(UPPER_T_BRANCH_SPLIT_Z_MM, 2.5)
    _close(UPPER_T_BRANCH_CLEARANCE_RADIUS_MM, 25.3)
    _close(UPPER_T_BRANCH_GUIDE_RADIUS_MM, 25.5)
    assert UPPER_T_BRANCH_MIN_INNER_LIGAMENT_MM > 1.5
    assert UPPER_T_BRANCH_MIN_OUTER_LIGAMENT_MM > 5.0
    assert UPPER_T_BRANCH_LOWER_INSERT_VERTICAL_LIGAMENT_MM > 1.3
    _close(MAIN_T_ROUTE_ENTRY_Z_MM, 11.5)
    _close(LOWER_T_OUTLET_Z_MM, 0.0)
    _close(T_CABLE_FACE_INSET_MM, 6.8)
    _close(UPPER_T_BRANCH_OUTLET_Z_MM, 11.5)
    _close(
        LOWER_T_OUTLET_Z_MM - REAR_T_MOUNT_Z_MM,
        THICKNESS_MM - UPPER_T_BRANCH_OUTLET_Z_MM,
    )
    assert T_MAGNET_FLAT_EDGE_MARGIN_MM == 0.10
    assert T_MAGNET_FACE_X_MM < TEBM_LAND_D_MM / 2.0
    _close(DUCT_EXTERIOR_SKIN_GUARD_MM, 0.8)
    assert DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3 <= 1.0e-5
    assert (
        DUCT_APPROVED_SEAM_MOUTH_CENTER_XYZ_MM[1]
        + DUCT_APPROVED_SEAM_MOUTH_SIZE_XYZ_MM[1] / 2.0
        < 320.0
    )


def _sample_plan_metrics(path, spacing_mm: float = 0.10):
    count = max(32, int(math.ceil(float(path.length) / spacing_mm)))
    points = [tuple(path @ (index / count)) for index in range(count + 1)]
    headings = []
    for index in range(count + 1):
        tangent = path % (index / count)
        headings.append(math.atan2(float(tangent.Y), float(tangent.X)))
    length = sum(math.dist(first, second)
                 for first, second in zip(points, points[1:]))
    turn = 0.0
    for first, second in zip(headings, headings[1:]):
        delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
        turn += abs(delta)
    return points, length, math.degrees(turn)


def test_t_route_optimization_contract() -> None:
    main = optimized_main_t_path()
    branch = upper_t_branch_path()
    main_points, main_length, main_turn = _sample_plan_metrics(main)
    branch_points, branch_length, branch_turn = _sample_plan_metrics(branch)
    main_xyz = optimized_main_t_centerline_points(0.10)

    _close(main_points[0][0], -34.082604)
    _close(main_points[0][1], 313.202641)
    _close(main_points[-1][0], MAIN_T_LOWER_OUTLET_XY_MM[0])
    _close(main_points[-1][1], MAIN_T_LOWER_OUTLET_XY_MM[1])
    _close(main_xyz[0][2], MAIN_T_ROUTE_ENTRY_Z_MM)
    _close(main_xyz[-1][2], LOWER_T_OUTLET_Z_MM)
    assert min(math.dist(point[:2], MAIN_T_ROUTE_SEAM_XY_MM)
               for point in main_points) < 0.11

    # Compare the same local interval against the shared Stock/Slim route.
    # That baseline now owns its own tangent-circle/R12 optimization, so the
    # larger TEBM envelope no longer receives credit for deleting the former
    # wall-following defect.  It must remain shorter and lower-turn than the
    # already-optimized common route while preserving its 15-mm-class radius.
    from build123d import Spline
    legacy = Spline(*route_points("ts", ts_route_key=TS_ROUTE_CAPTIVE))
    legacy_points, _legacy_length_all, _legacy_turn_all = (
        _sample_plan_metrics(legacy))
    first_y = main_points[0][1]
    legacy_local = [point for point in legacy_points if point[1] >= first_y]
    legacy_length = sum(math.dist(first, second)
                        for first, second in zip(
                            legacy_local, legacy_local[1:]))
    # Dense legacy tangent samples are filtered on the same monotone Y span.
    count = len(legacy_points) - 1
    legacy_headings = []
    for index, point in enumerate(legacy_points):
        if point[1] >= first_y:
            tangent = legacy % (index / count)
            legacy_headings.append(
                math.atan2(float(tangent.Y), float(tangent.X)))
    legacy_turn = 0.0
    for first, second in zip(legacy_headings, legacy_headings[1:]):
        delta = (second - first + math.pi) % (2.0 * math.pi) - math.pi
        legacy_turn += abs(delta)
    legacy_turn = math.degrees(legacy_turn)
    assert main_length < legacy_length - 0.50
    assert main_turn < 0.97 * legacy_turn

    main_min_radius = min(
        math.hypot(point[0], point[1] - float(UM_CUTOUT[1]))
        for point in main_points
        if point[1] >= MAIN_T_ROUTE_SEAM_XY_MM[1]
    )
    assert main_min_radius > 46.20

    # The shared main descends with two endpoint-flat minimum-jerk laws:
    # entry -> Y split -> rear-biased lower outlet.  At the split, both the
    # continuing lower leg and the upper branch therefore inherit the same
    # horizontal 3D tangent without a Z kink.
    split_fraction = _main_t_branch_split_parameter(main)
    _close(
        _main_t_center_z_mm(split_fraction, split_fraction),
        UPPER_T_BRANCH_SPLIT_Z_MM,
    )
    assert all(
        second[2] <= first[2] + 1.0e-7
        for first, second in zip(main_xyz, main_xyz[1:])
    )
    endpoint_probe = 1.0e-5
    for station, expected_z in (
        (0.0, MAIN_T_ROUTE_ENTRY_Z_MM),
        (split_fraction, UPPER_T_BRANCH_SPLIT_Z_MM),
        (1.0, LOWER_T_OUTLET_Z_MM),
    ):
        left = max(0.0, station - endpoint_probe)
        right = min(1.0, station + endpoint_probe)
        assert abs(
            _main_t_center_z_mm(left, split_fraction) - expected_z
        ) < 1.0e-7
        assert abs(
            _main_t_center_z_mm(right, split_fraction) - expected_z
        ) < 1.0e-7

    assert 100.0 < branch_length < 110.0
    assert branch_turn < 130.0
    _close(branch_points[0][1], UPPER_T_BRANCH_SPLIT_Y_MM)
    _close(branch_points[0][2], UPPER_T_BRANCH_SPLIT_Z_MM)
    _close(branch_points[-1][0], UPPER_T_BRANCH_OUTLET_XY_MM[0])
    _close(branch_points[-1][1], UPPER_T_BRANCH_OUTLET_XY_MM[1])
    _close(branch_points[-1][2], UPPER_T_BRANCH_OUTLET_Z_MM)

    # This is a real 3D Y, not two centerlines that merely intersect.  The
    # outgoing branch has the same XYZ tangent as the shared main, and its Z
    # law rises monotonically to the front-biased upper outlet, with flat
    # approaches at both the shared split and the opposed pocket face.
    assert upper_t_branch_split_tangent_error_deg() < 1.0e-6
    branch_start_tangent = branch % 0.0
    branch_end_tangent = branch % 1.0
    assert abs(float(branch_start_tangent.Z)) < 1.0e-9
    assert abs(float(branch_end_tangent.Z)) < 1.0e-9
    dense_branch = upper_t_branch_centerline_points(0.20)
    assert all(
        second[2] >= first[2] - 1.0e-7
        for first, second in zip(dense_branch, dense_branch[1:])
    )
    assert min(
        math.hypot(x, y - LOWER_T_AXIS_Y_MM)
        for x, y, _z in dense_branch
    ) >= UPPER_T_BRANCH_CLEARANCE_RADIUS_MM


def test_ducts_retain_exterior_skin_except_declared_mouths() -> None:
    guarded_main = _main_t_cable_duct(
        section_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM)
    guarded_branch = _upper_t_cable_duct(
        radial_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM)

    for profile in ("stock", "slim"):
        envelope = external_envelope(profile)
        # The shared cutter deliberately enters through seam B; that is the
        # only portion allowed outside either unbored envelope.
        main_outside_before_exception = guarded_main - envelope
        assert float(main_outside_before_exception.volume) > 1.0
        assert (
            float(main_outside_before_exception.bounding_box().max.Y)
            < 320.0
        )
        assert float((guarded_branch - envelope).volume) <= (
            DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3)

        residuals = duct_exposure_residuals(envelope, profile=profile)
        assert set(residuals) == {
            "shared_main_except_seam_b_entry", "upper_t_branch"}
        assert all(
            float(residual.volume) <= DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3
            for residual in residuals.values()
        ), profile
    # Finite swept-volume overlap proves a connected Y throat; a point or
    # cap-only touch would be geometrically unusable even with equal tangents.
    assert float((guarded_main & guarded_branch).volume) > 1.0

    forbidden = duct_unapproved_opening_residuals()
    assert set(forbidden) == {
        "shared_main_to_um_opening",
        "shared_main_to_upper_t_pocket",
        "upper_branch_to_um_opening",
        "upper_branch_to_lower_t_pocket",
        "shared_main_to_insert_bores",
        "upper_branch_to_insert_bores",
    }
    assert all(
        float(residual.volume) <= DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3
        for residual in forbidden.values()
    )


def test_model_geometry() -> None:
    model = build_model()
    part = model.solid
    solids = list(part.solids())
    assert part.is_valid
    assert len(solids) == 1
    assert solids[0].volume > 150_000.0
    bounds = part.bounding_box()
    _close(bounds.size.X, 121.308, 2.0e-4)
    # The regular top piece starts above the 0.05-mm seam gap and contains
    # the two through-thickness female pockets for the mid-piece males.
    _close(bounds.min.Y, SEAM_B_Y + CLEARANCE_MM, 2.0e-4)
    _close(bounds.size.Y, 210.231, 2.0e-4)
    _close(bounds.size.Z, 25.1, 2.0e-4)
    _close(bounds.min.Z, -6.8, 2.0e-4)
    _close(bounds.max.Z, 18.3, 2.0e-4)
    assert max(bounds.size.X, bounds.size.Y, bounds.size.Z) < 256.0

    # UM remains through.  Each opposed T pocket is open only from its mount
    # face and retains a real 1.2-mm wall at the opposite face.
    for z in (-6.0, 0.0, THICKNESS_MM - 0.2):
        assert not part.is_inside(
            (0.0, float(UM_CUTOUT[1]), z), tolerance=1.0e-5)
    for x in (-10.0, 0.0, 10.0):
        assert part.is_inside(
            (x, LOWER_T_AXIS_Y_MM,
             REAR_T_MOUNT_Z_MM + T_BLIND_BACK_WALL_THICKNESS_MM / 2.0),
            tolerance=1.0e-5,
        )
        assert not part.is_inside(
            (x, LOWER_T_AXIS_Y_MM, LOWER_T_POCKET_REAR_Z_MM + 0.2),
            tolerance=1.0e-5,
        )
        assert part.is_inside(
            (x, UPPER_T_AXIS_Y_MM,
             UPPER_T_POCKET_FRONT_Z_MM
             + T_BLIND_BACK_WALL_THICKNESS_MM / 2.0),
            tolerance=1.0e-5,
        )
        assert not part.is_inside(
            (x, UPPER_T_AXIS_Y_MM, UPPER_T_POCKET_FRONT_Z_MM - 0.2),
            tolerance=1.0e-5,
        )

    # The former B2 crescent survives nowhere outside the lower D66 land.
    for sign in (-1.0, 1.0):
        assert not part.is_inside(
            (sign * 32.0, 431.0, THICKNESS_MM - 0.3),
            tolerance=1.0e-5,
        )
        assert part.is_inside(
            (sign * 29.0, 431.0, THICKNESS_MM - 0.3),
            tolerance=1.0e-5,
        )

    # Both the optimized shared main and the independent upper branch are
    # continuously void.  The removed straight lower-to-upper bore is solid
    # again at the inter-driver web; the new branch occupies the right side.
    for x, y, z in optimized_main_t_centerline_points(1.0):
        if y < MAIN_T_ROUTE_SEAM_XY_MM[1] + 0.25:
            continue
        assert not part.is_inside(
            (x, y, z), tolerance=1.0e-5)
    for x, y, z in upper_t_branch_centerline_points(1.0):
        assert not part.is_inside((x, y, z), tolerance=1.0e-5)
    assert part.is_inside(
        (-17.5, (LOWER_T_AXIS_Y_MM + UPPER_T_AXIS_Y_MM) / 2.0, 5.0),
        tolerance=1.0e-5,
    )
    # Real plastic survives on both sides of the rising D4.6 branch at
    # the binding station beside the lower pocket.
    lower_station = min(
        upper_t_branch_centerline_points(0.10),
        key=lambda point: abs(point[1] - LOWER_T_AXIS_Y_MM),
    )
    assert abs(lower_station[1] - LOWER_T_AXIS_Y_MM) < 0.11
    for x in (
        lower_station[0] - UPPER_T_BRANCH_D_MM / 2.0 - 0.60,
        lower_station[0] + UPPER_T_BRANCH_D_MM / 2.0 + 0.60,
    ):
        assert part.is_inside(
            (x, lower_station[1], lower_station[2]),
            tolerance=1.0e-5,
        )
    # The deliberate 6.374-mm bridge between the two BMR apertures survives.
    assert part.is_inside(
        (0.0, (LOWER_T_AXIS_Y_MM + UPPER_T_AXIS_Y_MM) / 2.0, 10.0),
        tolerance=1.0e-5,
    )

    # Four front-opening and four rear-opening D3.2 x 4.0 insert bores.
    for x, y in _mount_centers(LOWER_T_AXIS_Y_MM, LOWER_T_MOUNT_CLOCK_DEG):
        assert not part.is_inside((x, y, THICKNESS_MM - 2.0),
                                  tolerance=1.0e-5)
        assert part.is_inside((x, y, THICKNESS_MM - 4.25),
                              tolerance=1.0e-5)
    for x, y in _mount_centers(UPPER_T_AXIS_Y_MM, UPPER_T_MOUNT_CLOCK_DEG):
        assert not part.is_inside((x, y, REAR_T_MOUNT_Z_MM + 2.0),
                                  tolerance=1.0e-5)
        assert part.is_inside((x, y, REAR_T_MOUNT_Z_MM + 4.25),
                              tolerance=1.0e-5)

    assert len(model.magnet_tools) == 4
    assert {tools.name for tools in model.magnet_tools} == {
        "tebm_lower_left_base", "tebm_lower_right_base",
        "tebm_upper_left_base", "tebm_upper_right_base",
    }
    for tools in model.magnet_tools:
        _close(tools.roof_start_print_z_mm, 5.8)
        _close(tools.required_min_part_top_print_z_mm, 8.85)
        assert not part.is_inside(tools.cavity_center_xyz, tolerance=1.0e-5)
        # The midpoint of the qualified interface skin remains real plastic.
        sign = 1.0 if tools.pair_axis_xyz[0] > 0.0 else -1.0
        face_x, face_y, face_z = tools.actual_face_xyz
        assert part.is_inside(
            (face_x - sign * 0.225, face_y, face_z), tolerance=1.0e-5)
        assert tools.spec.captive_land_mm == CAPTIVE_LAND_MM

    # Neither revised T cutter intersects any of the four qualified captive
    # magnet cutter stacks.  This guards against turning a cable bore into an
    # unintended side opening even when both features remain individually
    # valid booleans.
    branch_cutter = _upper_t_cable_duct(
        radial_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM)
    main_cutter = _main_t_cable_duct(
        section_extra_mm=DUCT_EXTERIOR_SKIN_GUARD_MM)
    for tools in model.magnet_tools:
        assert sum(float((branch_cutter & cutter).volume)
                   for cutter in tools.cutters) < 1.0e-7
        assert sum(float((main_cutter & cutter).volume)
                   for cutter in tools.cutters) < 1.0e-7

    facts = design_facts()
    assert facts["part"] == PART_NAME
    assert facts["m2_insert_bores"]["count"] == 8
    assert facts["t_captive_magnets"]["count"] == 4
    assert facts["blind_back_walls"]["count"] == 2
    routing = facts["t_cable_routing"]
    assert routing["shared_main"]["carries"] == [
        "lower_t_pair", "upper_t_pair"]
    assert routing["upper_t_branch"]["connects"] == [
        "shared_t_main", "upper_rear_t_pocket"]
    assert routing["shared_main"]["minimum_plan_bend_radius_mm"] > 15.0
    assert routing["shared_main"]["minimum_3d_bend_radius_mm"] > 15.0
    assert routing["shared_main"]["centerline_length_mm"] > (
        routing["shared_main"]["plan_length_mm"])
    assert routing["shared_main"]["lower_outlet_face_bias"] == "rear"
    assert routing["upper_t_branch"]["upper_outlet_face_bias"] == "front"
    assert routing["shared_main"]["lower_t_outlet_xyz_mm"] == [
        *MAIN_T_LOWER_OUTLET_XY_MM, LOWER_T_OUTLET_Z_MM]
    assert routing["upper_t_branch"]["upper_t_outlet_xyz_mm"] == [
        *UPPER_T_BRANCH_OUTLET_XY_MM, UPPER_T_BRANCH_OUTLET_Z_MM]
    assert routing["shared_main"]["minimum_um_plan_ligament_mm"] > 1.9
    assert routing["shared_main"][
        "minimum_um_pilot_vertical_ligament_mm"] > 1.6
    assert routing["upper_t_branch"][
        "minimum_lower_insert_vertical_ligament_mm"] > 1.3
    assert routing["upper_t_branch"]["split_3d_tangent_error_deg"] < 1.0e-6
    assert routing["upper_t_branch"]["minimum_3d_bend_radius_mm"] > 20.0
    assert routing["upper_t_branch"][
        "actual_minimum_lower_center_radius_mm"] >= (
            UPPER_T_BRANCH_CLEARANCE_RADIUS_MM)
    _close(
        routing["upper_t_branch"]["opposed_mount_face_inset_mm"],
        T_CABLE_FACE_INSET_MM,
    )
    assert routing["exterior_containment"] == {
        "brep_gate": "expanded_cutter_minus_unbored_envelope",
        "forbidden_opening_gate": (
            "expanded_cutter_intersection_with_unapproved_voids"),
        "magnet_gate": (
            "expanded_cutter_intersection_with_magnet_tools"),
        "minimum_skin_guard_mm": DUCT_EXTERIOR_SKIN_GUARD_MM,
        "volume_tolerance_mm3": DUCT_EXPOSURE_VOLUME_TOLERANCE_MM3,
        "approved_exterior_mouths": ["seam_b_ts_entry"],
        "driver_pocket_terminations": [
            "lower_front_t_pocket", "upper_rear_t_pocket"],
    }
    assert facts["released_interfaces"]["cable_routes"] == [
        "um", "ts", "t1f", "t2f"]


def test_stock_slim_seams_are_exact_production_interfaces() -> None:
    from build123d import Box, Pos

    records = (("stock", STOCK_PROFILE), ("slim", SLIM_PROFILE))
    volumes = {}
    for profile, envelope_spec in records:
        model = build_model(profile)
        part = model.solid
        bounds = part.bounding_box()
        assert part.is_valid and len(part.solids()) == 1
        volumes[profile] = float(part.volume)
        _close(bounds.min.Z, REAR_T_MOUNT_Z_MM, 2.0e-4)
        _close(bounds.max.Z, THICKNESS_MM, 2.0e-4)
        _close(bounds.min.Y, SEAM_B_Y + CLEARANCE_MM, 2.0e-4)

        # The BMR vase is the regular seam-B receiver: both clearance-grown
        # trapezoid pockets remain open through the complete local section.
        z_mid = (envelope_spec.rear_surface_z_mm + THICKNESS_MM) / 2.0
        for cx, _neck, _head, depth in DOVETAILS_B:
            pocket_probe = Pos(
                cx, SEAM_B_Y + depth / 2.0, z_mid
            ) * Box(0.40, 0.40, envelope_spec.section_depth_mm - 0.40)
            assert float((part & pocket_probe).volume) <= 1.0e-7, (
                profile, cx, "regular seam-B pocket is obstructed")

        receiver = seam_b_m3_vase_insert_cutter()
        receiver_overlap = part & receiver
        assert (receiver_overlap is None
                or float(receiver_overlap.volume) <= 1.0e-7), (
            profile, "radial M3 receiver is obstructed")
        blind_floor_probe = Pos(
            SEAM_B_M3_AXIS_X_MM,
            SEAM_B_M3_VASE_FACE_Y_MM + SEAM_B_M3_INSERT_DEPTH_MM + 0.10,
            SEAM_B_M3_AXIS_Z_MM,
        ) * Box(0.40, 0.10, 0.40)
        assert float((part & blind_floor_probe).volume) >= 0.015, (
            profile, "radial M3 receiver broke through its blind floor")

        # The smooth Slim growth begins at seam B with zero slope and
        # curvature.  Nothing may protrude behind that variant's released
        # rear interface in a narrow seam datum band.
        below_rear = Pos(
            0.0,
            SEAM_B_Y,
            envelope_spec.rear_surface_z_mm - 2.5,
        ) * Box(300.0, 0.002, 4.8)
        assert float((part & below_rear).volume) <= 1.0e-7, profile

        facts = design_facts(profile)
        assert facts["profile"] == profile
        assert facts["release_variant"] == envelope_spec.release_variant
        radial_m3 = facts["released_interfaces"]["seam_b_radial_m3"]
        assert radial_m3["axis_xyz_mm"] == [
            SEAM_B_M3_AXIS_X_MM,
            SEAM_B_M3_VASE_FACE_Y_MM,
            SEAM_B_M3_AXIS_Z_MM,
        ]
        _close(radial_m3["insert_bore_d_mm"], SEAM_B_M3_INSERT_BORE_D_MM)
        _close(radial_m3["insert_depth_mm"], SEAM_B_M3_INSERT_DEPTH_MM)
        _close(radial_m3["insert_engagement_mm"],
               SEAM_B_M3_INSERT_ENGAGEMENT_MM)
        _close(radial_m3["tip_margin_mm"],
               SEAM_B_M3_INSERT_TIP_MARGIN_MM)
        _close(
            facts["coordinate_system"]["rear_plane_z_mm"],
            envelope_spec.rear_surface_z_mm,
        )
        _close(
            facts["rear_growth"]["growth_mm"],
            envelope_spec.local_rear_growth_mm,
        )

    assert volumes["slim"] < volumes["stock"]


def test_auxiliary_profile_contract() -> None:
    from gen_vase_tebm35c10_4_slicing_profile import generate
    from release_validation import _validate_support_override_policy

    with tempfile.TemporaryDirectory() as directory:
        for key, variant in (
            ("stock", "Stock-TEBM35C10-4"),
            ("slim", "Slim-TEBM35C10-4"),
        ):
            output = Path(directory) / f"{key}.profile.json"
            profile = generate(
                PROJECT_ROOT / "captive_magnet_slicing_profile.json",
                output,
                key,
            )
            _validate_support_override_policy(profile)
            assert profile["catalog_mode"] == "auxiliary"
            assert profile["artifact_overrides"] == []
            assert profile["artifact_scope"] == [{
                "state": "shared",
                "variant": variant,
                "part": PART_NAME,
            }]
            process = profile["repo_overrides"]["process"]
            assert {process[name] for name in (
                "enable_support", "support_on_build_plate_only",
                "support_critical_regions_only",
                "support_remove_small_overhang",
            )} == {"0"}
            assert json.loads(output.read_text(encoding="utf-8")) == profile


def test_make_and_remote_contracts_are_first_class() -> None:
    makefile = (PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    remote = (PROJECT_ROOT / "scripts" / "remote_cad.py").read_text(
        encoding="utf-8")
    slicer = (PROJECT_ROOT / "scripts" / "slice_captive_magnets.py").read_text(
        encoding="utf-8")
    delivery = (PROJECT_ROOT / "scripts" /
                "validate_vase_tebm35c10_4_delivery.py").read_text(
                    encoding="utf-8")
    assert "vase_tebm35c10_4_cad:" in makefile
    assert "vase_tebm35c10_4_stock_cad:" in makefile
    assert "vase_tebm35c10_4_slim_cad:" in makefile
    assert "vase_tebm35c10_4_3mf:" in makefile
    assert "vase_tebm35c10_4_stock_3mf:" in makefile
    assert "vase_tebm35c10_4_slim_3mf:" in makefile
    assert "--auxiliary-catalog" in makefile
    assert 'test -e "$(VASE_TEBM_CAD_STAMP)"' in makefile
    assert "$(VASE_TEBM_STOCK_SLICE_STAMP):" in makefile
    assert "$(VASE_TEBM_SLIM_SLICE_STAMP):" in makefile
    assert "--profile stock" in makefile
    assert "--profile slim" in makefile
    assert '"vase_tebm35c10_4_cad"' in remote
    assert '"tebm35c10_4"' in remote
    assert '"--auxiliary-catalog"' in slicer
    assert "support_disabled_no_support_feature_blocks" in delivery


def main() -> None:
    tests = (
        test_dimension_contract,
        test_t_route_optimization_contract,
        test_ducts_retain_exterior_skin_except_declared_mouths,
        test_model_geometry,
        test_stock_slim_seams_are_exact_production_interfaces,
        test_auxiliary_profile_contract,
        test_make_and_remote_contracts_are_first_class,
    )
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"{PART_NAME}: {len(tests)} focused gates pass")


if __name__ == "__main__":
    main()
