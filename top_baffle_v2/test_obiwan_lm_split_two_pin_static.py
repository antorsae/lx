"""Pure/static contract for the optional Obi-Wan LM two-pin split.

This test intentionally does not import build123d or the CAD module.  It is a
fast guard for the normal-to-seam axis, symmetric male pins, round+relieved
socket tolerance strategy, and the very small annular-wall budget.  Exact BREP
containment and route-shell checks remain in ``test_obiwan_r6f.py`` and run on
the remote CAD host.
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
    assert constants["REGISTRATION_PIN_DIAMETER_MM"] == 0.80
    assert constants["REGISTRATION_PIN_ENGAGEMENT_MM"] == 0.80
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
    assert "horizontal D0.8 pins are only two nominal 0.4-mm nozzle" in source

    # The round locator permits +/-0.12 mm X float and the relieved locator
    # +/-0.18 mm.  Their differential pitch capacity is therefore 0.30 mm.
    pitch_capacity = (
        2.0 * constants["REGISTRATION_SOCKET_RADIAL_CLEAR_MM"]
        + constants["REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM"])
    assert abs(pitch_capacity - 0.30) < 1e-12


def test_relief_stays_inside_one_printable_annular_wall_path():
    _, tree = _source_tree()
    constants = _numeric_constants(tree)

    # Source-authority dimensions: LM centre Y=200.981, driver recess
    # R110.6, outer carrier R113.0.  Repeat the module's balanced-wall
    # calculation without importing OCC/build123d.
    cy = 200.981
    inner_r = 110.6
    outer_r = 113.0
    seam_y = cy + constants["LM_SPLIT_SEAM_OFFSET_Y"]
    dy0 = seam_y - cy
    depth = (constants["REGISTRATION_PIN_ENGAGEMENT_MM"]
             + constants["REGISTRATION_SOCKET_END_CLEAR_MM"])
    dy1 = dy0 + depth
    half_x = (
        constants["REGISTRATION_PIN_DIAMETER_MM"] / 2.0
        + constants["REGISTRATION_SOCKET_RADIAL_CLEAR_MM"]
        + constants["REGISTRATION_RELIEVED_SOCKET_X_EXTRA_MM"])
    inner_limit = sqrt(inner_r ** 2 - dy1 ** 2) + half_x
    outer_limit = sqrt(outer_r ** 2 - dy0 ** 2) - half_x
    x = (inner_limit + outer_limit) / 2.0
    inner_wall = sqrt((x - half_x) ** 2 + dy1 ** 2) - inner_r
    outer_wall = outer_r - sqrt((x + half_x) ** 2 + dy0 ** 2)

    assert 216.0 < 2.0 * x < 217.0
    assert inner_wall >= constants["REGISTRATION_MIN_RADIAL_WALL_MM"]
    assert outer_wall >= constants["REGISTRATION_MIN_RADIAL_WALL_MM"]
    assert 0.50 < min(inner_wall, outer_wall) < 0.51


def test_both_pins_are_reassigned_and_both_sockets_are_cut():
    _, tree = _source_tree()
    split = ast.unparse(_function(tree, "lm_carrier_split_parts"))
    assert "male_registration_pin_tools().items()" in split
    assert "female_registration_socket_tools().values()" in split
    assert "outside_source = male_tool - carrier" in split
    assert "bottom = _fuse_attached" in split
    assert "top -= socket_tool" in split


CHECKS = (
    test_two_symmetric_pins_and_normal_axis_are_source_contracts,
    test_round_plus_relief_socket_avoids_wide_pitch_binding,
    test_relief_stays_inside_one_printable_annular_wall_path,
    test_both_pins_are_reassigned_and_both_sockets_are_cut,
)


def main() -> int:
    for check in CHECKS:
        check()
        print(f"PASS {check.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
