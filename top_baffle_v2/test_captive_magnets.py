"""Focused geometry gates for the shared captive-magnet helper."""

from __future__ import annotations

import inspect
import math

from build123d import Align, Box, Pos

import captive_magnets as captive


def _one_valid_solid(shape, label: str) -> None:
    solids = list(shape.solids())
    assert shape.is_valid, label
    assert len(solids) == 1, (label, [solid.volume for solid in solids])
    assert solids[0].volume > 1.0, label


def test_nominal_authority() -> None:
    facts = captive.design_facts()
    assert facts["magnet_diameter_mm"] == 5.0
    assert facts["magnet_depth_mm"] == 2.0
    assert facts["cavity_diameter_mm"] == 5.2
    assert facts["cavity_depth_mm"] == 2.1
    assert facts["face_skin_mm"] == facts["inner_skin_mm"] == 0.45
    assert facts["interface_gap_mm"] == 0.05
    assert facts["roof_angle_deg"] == 45.0
    assert math.isclose(facts["roof_height_mm"], 2.6, abs_tol=1.0e-12)
    # This is a frozen JSON-schema fact, not merely a geometrically close
    # value: 0.45 + 2.10 + 0.45 must serialize as exactly 3.0.
    assert facts["captive_land_mm"] == 3.0
    assert captive.CAPTIVE_LAND_MM == 3.0
    assert facts["paired_magnet_face_separation_mm"] == 0.95
    assert captive.NOMINAL_PAIRED_FACE_SEPARATION_MM == 0.95
    # A generic positive-backing hook could restore an exterior bevel or
    # create a visible pocket boss.  Production cavity application is
    # deliberately subtractive-only.
    assert "backing_additions" not in inspect.signature(
        captive.apply_wall_cavity).parameters
    assert "backing_additions" not in inspect.signature(
        captive.apply_axial_cavity).parameters


def test_coupon_regression_planes_and_pair() -> None:
    # Coupon local coordinates: rear=0, front=11.5. Production LM and UM now
    # share the front-biased 8.30-mm local axis. Its CAD roof plane at 5.80
    # produces the common Bambu 0.16-profile pause at 5.96 mm.
    common = {
        "face": (0.0, 0.0),
        "outward": (1.0, 0.0, 0.0),
        "front_z": 11.5,
        "print_up": (0.0, 0.0, -1.0),
    }
    lm_base = captive.wall_cavity_tools(
        name="lm_base", owner="base", axis_z=8.30, **common)
    lm_receiver = captive.wall_cavity_tools(
        name="lm_receiver", owner="receiver", axis_z=8.30, **common)
    um_base = captive.wall_cavity_tools(
        name="um_base", owner="base", axis_z=8.30, **common)
    assert math.isclose(
        lm_base.roof_start_print_z_mm, 5.80, abs_tol=1.0e-12)
    assert math.isclose(
        um_base.roof_start_print_z_mm, 5.80, abs_tol=1.0e-12)
    assert math.isclose(
        lm_base.required_min_part_top_print_z_mm, 8.85,
        abs_tol=1.0e-12)
    pair = captive.pair_facts(lm_base, lm_receiver)
    assert math.isclose(pair["interface_gap_mm"], 0.05, abs_tol=1.0e-12)
    assert math.isclose(
        pair["nominal_magnet_face_separation_mm"], 0.95,
        abs_tol=1.0e-12)


def test_coupon_style_wall_solids() -> None:
    # Two simple 4-mm lands exercise both owner directions and guarantee that
    # the new helper leaves real interface/back skins rather than reopening a
    # legacy glue pocket.
    base = Pos(-4.0, -4.0, 0.0) * Box(
        4.0, 8.0, 11.5, align=(Align.MIN, Align.MIN, Align.MIN))
    # Deliberately begin the receiver at the shared datum.  The 0.05-mm pair-
    # spacing allowance must remain solid in front of the qualified 0.45-mm
    # receiver skin; cutting it as a local air gap leaves a visible pocket-
    # width notch on the exterior.
    receiver = Pos(0.0, -4.0, 0.0) * Box(
        4.0, 8.0, 11.5, align=(Align.MIN, Align.MIN, Align.MIN))
    kwargs = {
        "name": "coupon_lm",
        "face": (0.0, 0.0, 5.75),
        "outward": (1.0, 0.0, 0.0),
        "front_z": 11.5,
        "print_up": (0.0, 0.0, -1.0),
    }
    base, base_tools = captive.apply_wall_cavity(
        base, owner="base", **kwargs)
    receiver, receiver_tools = captive.apply_wall_cavity(
        receiver, owner="receiver", **kwargs)
    _one_valid_solid(base, "base captive coupon")
    _one_valid_solid(receiver, "receiver captive coupon")
    assert base.is_inside((-0.225, 0.0, 5.75), tolerance=1.0e-5)
    assert not base.is_inside(
        base_tools.cavity_center_xyz, tolerance=1.0e-5)
    assert base.is_inside((-2.775, 0.0, 5.75), tolerance=1.0e-5)
    assert len(receiver_tools.cutters) == 3
    assert receiver.is_inside((0.025, 0.0, 5.75), tolerance=1.0e-5)
    assert receiver.is_inside((0.275, 0.0, 5.75), tolerance=1.0e-5)
    assert receiver.is_inside((0.475, 0.0, 5.75), tolerance=1.0e-5)
    assert not receiver.is_inside((0.525, 0.0, 5.75), tolerance=1.0e-5)
    assert not receiver.is_inside(
        receiver_tools.cavity_center_xyz, tolerance=1.0e-5)
    assert receiver.is_inside((2.825, 0.0, 5.75), tolerance=1.0e-5)


def test_application_rejects_missing_host_land() -> None:
    undersized = Pos(-1.0, -1.0, 0.0) * Box(
        1.0, 2.0, 11.5, align=(Align.MIN, Align.MIN, Align.MIN))
    try:
        captive.apply_wall_cavity(
            undersized,
            name="missing_land_probe",
            face=(0.0, 0.0, 5.75),
            outward=(1.0, 0.0, 0.0),
            owner="base",
            front_z=11.5,
            print_up=(0.0, 0.0, -1.0),
        )
    except captive.CaptiveMagnetGeometryError as exc:
        assert "immutable host misses" in str(exc)
    else:
        raise AssertionError("cavity application accepted incomplete land")


def test_axis_parallel_and_front_down_opposed_cones() -> None:
    block = Pos(-4.0, -4.0, 0.0) * Box(
        8.0, 8.0, 6.0, align=(Align.MIN, Align.MIN, Align.MIN))
    block, tools = captive.apply_axial_cavity(
        block,
        name="v0_rear_down",
        face=(0.0, 0.0, 0.0),
        inward=(0.0, 0.0, 1.0),
        pair_axis=(0.0, 0.0, -1.0),
        print_up=(0.0, 0.0, 1.0),
        bed_datum=(0.0, 0.0, 0.0),
    )
    _one_valid_solid(block, "axis-parallel captive coupon")
    assert tools.closure_kind == "axis_parallel_conical_45deg"
    assert math.isclose(tools.roof_start_print_z_mm, 2.6, abs_tol=1e-12)
    assert math.isclose(
        tools.required_min_part_top_print_z_mm, 5.65, abs_tol=1e-12)
    assert block.is_inside((0.0, 0.0, 0.225), tolerance=1.0e-5)
    assert not block.is_inside((0.0, 0.0, 1.50), tolerance=1.0e-5)

    v0 = captive.axial_cavity_tools(
        name="v0_front_down",
        face=(0.0, 0.0, 0.0),
        inward=(0.0, 0.0, 1.0),
        pair_axis=(0.0, 0.0, -1.0),
        print_up=(0.0, 0.0, -1.0),
        bed_datum=(0.0, 0.0, 18.3),
    )
    assert v0.closure_kind == "axis_opposed_conical_45deg"
    assert all(math.isclose(a, b, abs_tol=1.0e-12) for a, b in zip(
        v0.cavity_center_xyz, (0.0, 0.0, 4.10)))
    assert all(math.isclose(a, b, abs_tol=1.0e-12) for a, b in zip(
        v0.seated_magnet_center_xyz, (0.0, 0.0, 4.10)))
    assert math.isclose(v0.roof_start_print_z_mm, 15.25, abs_tol=1e-12)
    assert math.isclose(v0.roof_apex_print_z_mm, 17.85, abs_tol=1e-12)
    assert math.isclose(
        v0.required_min_part_top_print_z_mm, 18.3, abs_tol=1e-12)


def main() -> None:
    tests = (
        test_nominal_authority,
        test_coupon_regression_planes_and_pair,
        test_coupon_style_wall_solids,
        test_application_rejects_missing_host_land,
        test_axis_parallel_and_front_down_opposed_cones,
    )
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"captive magnets: {len(tests)} focused gates pass")


if __name__ == "__main__":
    main()
