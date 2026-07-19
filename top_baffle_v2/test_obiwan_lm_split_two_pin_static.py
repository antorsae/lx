"""Pure/static contract for the optional Obi-Wan LM two-pin split.

This test intentionally does not import build123d or the CAD module.  It is a
fast guard for the normal-to-seam axis, symmetric male pins, round+relieved
socket tolerance strategy, and the driver-clear exterior support-land budget.
Exact BREP containment and route-shell checks remain in
``test_obiwan_r6f.py`` and run on the remote CAD host.
"""

from __future__ import annotations

import ast
from math import sqrt
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "top_baffle_nd25fw4_obiwan_lm_split.py"


def _source_tree():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    return source, ast.parse(source)


def _numeric_constants(tree):
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        sign = 1.0
        if (isinstance(value, ast.UnaryOp)
                and isinstance(value.op, ast.USub)):
            sign = -1.0
            value = value.operand
        if isinstance(value, ast.Constant) and isinstance(
                value.value, (int, float)):
            values[target.id] = sign * float(value.value)
    return values


def _function(tree, name):
    return next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name)


def test_two_symmetric_pins_and_normal_axis_are_source_contracts():
    source, tree = _source_tree()
    constants = _numeric_constants(tree)
    assert constants["REGISTRATION_PIN_DIAMETER_MM"] == 1.60
    assert constants["REGISTRATION_PIN_ENGAGEMENT_MM"] == 2.40
    assert constants["REGISTRATION_PIN_ROOT_OVERLAP_MM"] == 0.50
    assert constants["REGISTRATION_CENTER_Z_MM"] == 14.30

    centers = ast.unparse(_function(tree, "registration_pin_centers_xyz"))
    assert "'left'" in centers and "'right'" in centers
    assert "cx - x" in centers and "cx + x" in centers
    assert "LM_SPLIT_SEAM_Y" in centers

    cylinder = ast.unparse(_function(tree, "_y_axis_cylinder"))
    assert "Rot(X=-90.0)" in cylinder
    assert "Align.MIN" in cylinder
    assert '"registration_axis_world_xyz": (0.0, 1.0, 0.0)' in source
    assert '"top_half_approaches_along_negative_world_y"' in source


def test_round_plus_relief_socket_avoids_wide_pitch_binding():
    source, tree = _source_tree()
    constants = _numeric_constants(tree)
    assert constants["REGISTRATION_SOCKET_RADIAL_CLEAR_MM"] == 0.12
    assert constants["REGISTRATION_SOCKET_END_CLEAR_MM"] == 0.25
    assert constants["REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM"] == 0.06
    assert constants["REGISTRATION_MIN_RADIAL_WALL_MM"] == 0.50
    assert constants["REGISTRATION_SUPPORT_END_WALL_MM"] == 0.50
    assert constants["REGISTRATION_SUPPORT_RECESS_CLEAR_MM"] == 0.05
    assert constants["REGISTRATION_DRIVER_FLANGE_R_MM"] == 110.52
    assert constants["REGISTRATION_WING_CLEARANCE_MM"] == 0.25

    sockets = ast.unparse(
        _function(tree, "female_registration_socket_tools"))
    assert "side == 'left'" in sockets
    assert "REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM" in sockets
    assert "if side == 'left' else 0.0" in sockets
    assert '"round_socket_side": "right"' in source
    assert '"relieved_socket_side": "left"' in source
    assert '"two_round_socket_design_rejected": True' in source
    assert "two round sockets across the wide pitch can bind" in source
    assert '"pin_and_socket_slicer_gate_required": True' in source
    assert "horizontal D1.6 pins are four nominal 0.4-mm nozzle" in source

    # The round locator permits +/-0.12 mm X float and the relieved locator
    # +/-0.18 mm.  Their differential pitch capacity is therefore 0.30 mm.
    pitch_capacity = (
        2.0 * constants["REGISTRATION_SOCKET_RADIAL_CLEAR_MM"]
        + constants["REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM"])
    assert abs(pitch_capacity - 0.30) < 1e-12


def test_relief_uses_driver_clear_exterior_support_lands():
    _, tree = _source_tree()
    constants = _numeric_constants(tree)

    # Source-authority dimensions: LM centre Y=200.981, driver recess R110.6,
    # outer carrier R113.0, rear z=6.8 and front z=18.3.  Repeat the module's
    # support-land calculation without importing OCC/build123d.
    cy = 200.981
    recess_r = 110.6
    outer_r = 113.0
    rear_z = 6.8
    seat_z = 12.3
    front_z = 18.3
    seam_y = cy + constants["LM_SPLIT_SEAM_OFFSET_Y"]
    support_r = (
        constants["REGISTRATION_PIN_DIAMETER_MM"] / 2.0
        + constants["REGISTRATION_SOCKET_RADIAL_CLEAR_MM"]
        + constants["REGISTRATION_MIN_RADIAL_WALL_MM"])
    support_half_x = (
        support_r
        + constants["REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM"])
    support_start_y = (
        seam_y - constants["REGISTRATION_PIN_ROOT_OVERLAP_MM"])
    support_length = (
        constants["REGISTRATION_PIN_ROOT_OVERLAP_MM"]
        + constants["REGISTRATION_PIN_ENGAGEMENT_MM"]
        + constants["REGISTRATION_SOCKET_END_CLEAR_MM"]
        + constants["REGISTRATION_SUPPORT_END_WALL_MM"])
    end_dy = support_start_y + support_length - cy
    root_dy = support_start_y - cy
    inner_authority_r = (
        recess_r + constants["REGISTRATION_SUPPORT_RECESS_CLEAR_MM"])
    x = sqrt(inner_authority_r ** 2 - end_dy ** 2) + support_half_x
    recess_clearance = sqrt(
        (x - support_half_x) ** 2 + end_dy ** 2) - recess_r
    outline_growth = sqrt(
        (x + support_half_x) ** 2 + root_dy ** 2) - outer_r
    driver_flange_clearance = (
        recess_r + recess_clearance
        - constants["REGISTRATION_DRIVER_FLANGE_R_MM"])
    support_z_min = constants["REGISTRATION_CENTER_Z_MM"] - support_r
    support_z_max = constants["REGISTRATION_CENTER_Z_MM"] + support_r

    assert 218.3 < 2.0 * x < 218.5
    assert abs(recess_clearance - 0.05) < 1e-9
    assert driver_flange_clearance >= 0.129999
    assert 1.39 < outline_growth < 1.42
    assert support_z_min > seat_z
    assert support_z_min - rear_z > 6.0
    assert front_z - support_z_max > 2.5


def test_both_pins_are_reassigned_and_both_sockets_are_cut():
    _, tree = _source_tree()
    split = ast.unparse(_function(tree, "lm_carrier_split_parts"))
    supports = ast.unparse(_function(tree, "registration_support_land_tools"))
    assert "REGISTRATION_MIN_RADIAL_WALL_MM" in supports
    assert "REGISTRATION_SUPPORT_END_WALL_MM" in supports
    assert "carrier = registration_augmented_carrier(carrier)" in split
    assert "male_registration_pin_tools().items()" in split
    assert "female_registration_socket_tools().values()" in split
    assert "outside_source = male_tool - carrier" in split
    assert "bottom = _fuse_attached" in split
    assert "top -= socket_tool" in split


def test_ac_ae_clearance_tools_offset_the_worst_case_land():
    source, tree = _source_tree()
    clearance = ast.unparse(
        _function(tree, "registration_wing_clearance_tools"))
    assert "REGISTRATION_WING_CLEARANCE_MM" in clearance
    assert "2.0 * clearance" in clearance
    assert "REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM" in clearance
    assert '"wing_clearance_compatible_variants": ("ac", "ae")' in source
    assert '"wing_clearance_pocket_between_front_and_rear": True' in source


CHECKS = (
    test_two_symmetric_pins_and_normal_axis_are_source_contracts,
    test_round_plus_relief_socket_avoids_wide_pitch_binding,
    test_relief_uses_driver_clear_exterior_support_lands,
    test_both_pins_are_reassigned_and_both_sockets_are_cut,
    test_ac_ae_clearance_tools_offset_the_worst_case_land,
)


def main() -> int:
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
