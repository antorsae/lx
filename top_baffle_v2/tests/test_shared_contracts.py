#!/usr/bin/env python3
"""Pure regressions for the small shared Stage 1 contract helpers."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)
import subprocess
import struct
import sys
import tempfile
import types

from lx521_baffle.magnet_contract import DEFAULT_SPEC as DEFAULT_MAGNET_SPEC
from lx521_baffle.print_contract import (
    FrontDownContractError,
    front_down_transform_record,
    validate_front_down_transform,
)
from lx521_baffle.geom import point_segment_distance, smoothstep01
from lx521_baffle.io import pretty_json_bytes, sha256_bytes, sha256_file
from lx521_baffle.stl_export import (
    BinaryStlLayoutError,
    canonicalize_near_zero_stl_coordinates,
    validate_binary_stl_length,
)


ROOT = PROJECT_ROOT
EXPECTED_CAPTIVE_MAGNET_RELEASE_SITE_GEOMETRY_MM = {
    "magnet_diameter_mm": 5.0,
    "magnet_depth_mm": 2.0,
    "cavity_diameter_mm": 5.2,
    "cavity_depth_mm": 2.1,
    "face_skin_mm": 0.45,
    "inner_skin_mm": 0.45,
    "captive_land_mm": 3.0,
    "interface_gap_mm": 0.05,
    "roof_angle_deg": 45.0,
    "minimum_retaining_path_mm": 0.42,
}


def _canonical_subprocess_environment() -> dict[str, str]:
    """Expose the installed-style source and CLI roots to ``python -c``."""
    environment = os.environ.copy()
    roots = [str(ROOT / "src"), str(ROOT / "scripts")]
    existing = environment.get("PYTHONPATH")
    if existing:
        roots.append(existing)
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment


def test_hash_primitives() -> None:
    data = b"abc"
    expected = (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )
    assert sha256_bytes(data) == expected
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "payload.bin"
        path.write_bytes(data)
        assert sha256_file(path) == expected
        assert sha256_file(str(path)) == expected


def test_pretty_json_bytes_freeze_strict_and_permissive_modes() -> None:
    assert pretty_json_bytes(
        {"z": 2, "a": [1]}, allow_nan=False
    ) == b'{\n  "a": [\n    1\n  ],\n  "z": 2\n}\n'
    permissive = pretty_json_bytes({"value": math.nan}, allow_nan=True)
    assert permissive == b'{\n  "value": NaN\n}\n'
    assert math.isnan(json.loads(permissive)["value"])
    try:
        pretty_json_bytes({"value": math.nan}, allow_nan=False)
    except ValueError:
        pass
    else:
        raise AssertionError("strict JSON unexpectedly accepted NaN")


def test_front_down_transform_constructor() -> None:
    record = front_down_transform_record(
        (-2.5, 4.0, -7.25), z_rotation_deg=90.0)
    assert record["pre_translation_bbox_min_mm"] == [-2.5, 4.0, -7.25]
    assert record["stl_origin_translation_mm"] == [2.5, -4.0, 7.25]
    matrix = validate_front_down_transform(record)
    expected = (
        (0.0, 1.0, 0.0, 2.5),
        (1.0, 0.0, 0.0, -4.0),
        (0.0, 0.0, -1.0, 7.25),
        (0.0, 0.0, 0.0, 1.0),
    )
    for actual_row, expected_row in zip(matrix, expected, strict=True):
        for actual, wanted in zip(actual_row, expected_row, strict=True):
            assert math.isclose(actual, wanted, abs_tol=1.0e-12)


def test_front_down_transform_constructor_rejects_malformed_input() -> None:
    malformed = (
        ((1.0, 2.0), 0.0),
        ((1.0, 2.0, 3.0), math.inf),
        ((1.0, False, 3.0), 0.0),
    )
    for minimum, angle in malformed:
        try:
            front_down_transform_record(minimum, z_rotation_deg=angle)
        except FrontDownContractError:
            pass
        else:
            raise AssertionError(
                f"malformed transform input accepted: {minimum!r}, {angle!r}")


def test_captive_magnet_contract_matches_independent_release_golden() -> None:
    facts = DEFAULT_MAGNET_SPEC.facts()
    actual = {
        key: facts[key]
        for key in EXPECTED_CAPTIVE_MAGNET_RELEASE_SITE_GEOMETRY_MM
    }
    assert actual == EXPECTED_CAPTIVE_MAGNET_RELEASE_SITE_GEOMETRY_MM
    import slice_captive_magnets as slicer
    assert (slicer.RELEASE_SITE_GEOMETRY_MM
            == EXPECTED_CAPTIVE_MAGNET_RELEASE_SITE_GEOMETRY_MM)


def test_contract_and_slicer_import_without_cad_kernel() -> None:
    probe = (
        "import json, sys; "
        "import lx521_baffle.magnet_contract, slice_captive_magnets; "
        "bad=sorted(name for name in sys.modules "
        "if name == 'build123d' or name.startswith('build123d.') "
        "or name == 'OCP' or name.startswith('OCP.')); "
        "print(json.dumps(bad)); "
        "raise SystemExit(1 if bad else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        cwd=ROOT,
        env=_canonical_subprocess_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def _guard_probe_source() -> str:
    return """\
import json
import os
import sys
from pathlib import Path
import run_memory_guarded as guard

count = int(os.environ.get("LX_GUARD_PROBE_COUNT", "0")) + 1
os.environ["LX_GUARD_PROBE_COUNT"] = str(count)
guard.reexec_under_guard(Path(__file__))
print(json.dumps({"count": count, "argv": sys.argv[1:]}), flush=True)
exit_index = sys.argv.index("--exit")
raise SystemExit(int(sys.argv[exit_index + 1]))
"""


def _guard_probe_environment(count: int) -> dict[str, str]:
    environment = _canonical_subprocess_environment()
    environment.pop("LX_CAD_MEMORY_GUARDED", None)
    environment.pop("LX_CAD_MEMORY_GUARD_PID", None)
    environment["LX_GUARD_PROBE_COUNT"] = str(count)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def test_guard_reexec_preserves_arguments_exit_and_wraps_once() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        probe = Path(temporary) / "guard_probe.py"
        probe.write_text(_guard_probe_source(), encoding="utf-8")
        arguments = ["alpha", "two words", "--exit", "23"]
        completed = subprocess.run(
            [sys.executable, "-B", str(probe), *arguments],
            cwd=ROOT,
            env=_guard_probe_environment(0),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    assert completed.returncode == 23, completed.stderr
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert records == [{"count": 2, "argv": arguments}]


def test_guarded_reexec_is_a_noop_without_recursive_wrapper() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        probe = Path(temporary) / "guarded_probe.py"
        probe.write_text(_guard_probe_source(), encoding="utf-8")
        arguments = ["already-guarded", "--exit", "0"]
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/run_memory_guarded.py"),
                "--",
                sys.executable,
                "-B",
                str(probe),
                *arguments,
            ],
            cwd=ROOT,
            env=_guard_probe_environment(40),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    assert completed.returncode == 0, completed.stderr
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert records == [{"count": 41, "argv": arguments}]


def test_guard_requirement_preserves_caller_error_and_import_order() -> None:
    import run_memory_guarded as guard

    try:
        guard.require_guarded_build("caller-specific guard failure")
    except RuntimeError as exc:
        assert str(exc) == "caller-specific guard failure"
    else:
        raise AssertionError("unguarded process unexpectedly passed guard gate")

    eager_imports = {
        "scripts/export_piece_stls.py": "from build123d import",
        "scripts/gen_cable_routing.py": "import numpy as np",
        "scripts/gen_driver_overlay.py": "import numpy as np",
        "scripts/gen_obiwan_wing_design_map.py": "import matplotlib",
        "scripts/polar_index_base.py": "from build123d import",
        "src/lx521_baffle/obiwan/carriers.py": "from build123d import",
    }
    for filename, eager_import in eager_imports.items():
        lines = (ROOT / filename).read_text(encoding="utf-8").splitlines()
        gate_line = next(
            index for index, line in enumerate(lines)
            if "reexec_under_guard(" in line)
        import_line = next(
            index for index, line in enumerate(lines)
            if line.startswith(eager_import))
        assert gate_line < import_line, filename


def test_scalar_geometry_primitives_freeze_clamping_and_distance() -> None:
    assert smoothstep01(-1.0) == 0.0
    assert smoothstep01(0.0) == 0.0
    assert smoothstep01(0.5) == 0.5
    assert smoothstep01(1.0) == 1.0
    assert smoothstep01(2.0) == 1.0
    assert math.isclose(
        point_segment_distance((1.0, 2.0), (0.0, 0.0), (4.0, 0.0)),
        2.0,
    )
    assert math.isclose(
        point_segment_distance((-3.0, 4.0), (0.0, 0.0), (4.0, 0.0)),
        5.0,
    )


def _one_triangle_stl_bytes(*, noisy_coordinate: float = 0.0) -> bytes:
    payload = bytearray(b"shared STL fixture".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", 1))
    payload.extend(struct.pack(
        "<12fH",
        0.0,
        0.0,
        1.0,
        noisy_coordinate,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0,
    ))
    return bytes(payload)


def test_shared_stl_primitives_and_exporter_adapters() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stl = root / "one.stl"
        stl.write_bytes(_one_triangle_stl_bytes(noisy_coordinate=1.0e-8))
        assert validate_binary_stl_length(stl) == 1
        assert canonicalize_near_zero_stl_coordinates(stl, 2.0e-7) == 1
        assert canonicalize_near_zero_stl_coordinates(stl, 2.0e-7) == 0
        malformed = root / "malformed.stl"
        malformed.write_bytes(_one_triangle_stl_bytes() + b"x")
        try:
            validate_binary_stl_length(malformed)
        except BinaryStlLayoutError as exc:
            assert exc.triangle_count == 1
            assert exc.actual_bytes == 135
            assert exc.expected_bytes == 134
        else:
            raise AssertionError("malformed binary STL length was accepted")

    sources = {
        name: (ROOT / name).read_text(encoding="utf-8")
        for name in (
            "scripts/export_piece_stls.py",
            "scripts/export_coupon.py",
            "scripts/export_obiwan_wings.py",
        )
    }
    positive_message = "STL transform-zero epsilon must be positive"
    assert positive_message in sources["scripts/export_piece_stls.py"]
    assert positive_message in sources["scripts/export_coupon.py"]
    assert positive_message not in sources["scripts/export_obiwan_wings.py"]
    assert "return _jsonable(facts)" in sources["scripts/export_obiwan_wings.py"]
    assert "_remove_collapsed_apex_facets(temporary)" in sources[
        "scripts/export_piece_stls.py"]
    assert "_modifier_mesh_facts(temporary)" in sources[
        "scripts/export_piece_stls.py"]

    wing_tree = ast.parse(sources["scripts/export_obiwan_wings.py"])
    exporter = next(
        node for node in wing_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_export_variant"
    )
    wanted = (
        "export_stl",
        "_validate_binary_stl",
        "_canonicalize_transform_zeros",
        "_strict_mesh_facts",
        "_promote_transaction",
    )
    lines: dict[str, list[int]] = {name: [] for name in wanted}
    for node in ast.walk(exporter):
        if not isinstance(node, ast.Call):
            continue
        name = (
            node.func.id if isinstance(node.func, ast.Name)
            else node.func.attr if isinstance(node.func, ast.Attribute)
            else None
        )
        if name in lines:
            lines[name].append(node.lineno)
    assert all(len(lines[name]) == 1 for name in wanted), lines
    assert [lines[name][0] for name in wanted] == sorted(
        lines[name][0] for name in wanted)


def test_ordered_assembly_helper_and_narrow_adoption() -> None:
    class FakeCompound:
        def __init__(self, *, children):
            self.children = children
            self.label = None

    class FakeSolid:
        def __init__(self):
            self.label = None

    stub = types.ModuleType("build123d")
    stub.Compound = FakeCompound
    module_name = "_shared_assembly_contract_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "src/lx521_baffle/assembly.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    prior = sys.modules.get("build123d")
    sys.modules["build123d"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        if prior is None:
            sys.modules.pop("build123d", None)
        else:
            sys.modules["build123d"] = prior

    left, right = FakeSolid(), FakeSolid()
    assembly = module.ordered_labeled_compound(
        {"left": left, "right": right}, label="ordered")
    assert assembly.children == [left, right]
    assert [left.label, right.label] == ["left", "right"]
    assert assembly.label == "ordered"

    for specialized in (
        "src/lx521_baffle/obiwan/assembled.py",
        "src/lx521_baffle/um_fit.py",
        "src/lx521_baffle/obiwan/wings.py",
    ):
        assert "ordered_labeled_compound" not in (
            ROOT / specialized).read_text(encoding="utf-8")
    assert 'if "b1_wing" in k' in (
        ROOT / "src/lx521_baffle/proud/b1_assembled.py").read_text(
            encoding="utf-8")
    assert 'if "a_shoulder" in k' in (
        ROOT / "src/lx521_baffle/proud/a_comp_assembled.py").read_text(
            encoding="utf-8")


def test_to_print_shelf_waits_for_release_catalog() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    rule = makefile.split("check_to_print_shelf:", 1)[1].split("\n\t", 1)[0]
    assert "$(TO_PRINT_CATALOG_PREREQ)" in rule
    prerequisite = makefile.split(
        "TO_PRINT_CATALOG_PREREQ :=", 1)[1].split("\n", 2)[:2]
    prerequisite = " ".join(line.strip(" \\") for line in prerequisite)
    assert "$(filter remote-worker,$(LX_CAD_EXECUTION))" in prerequisite
    assert "$(CAPTIVE_MAGNET_CATALOG)" in prerequisite
    assert "$(abspath $(CAPTIVE_MAGNET_CATALOG))" in prerequisite


def main() -> None:
    tests = (
        test_hash_primitives,
        test_pretty_json_bytes_freeze_strict_and_permissive_modes,
        test_front_down_transform_constructor,
        test_front_down_transform_constructor_rejects_malformed_input,
        test_captive_magnet_contract_matches_independent_release_golden,
        test_contract_and_slicer_import_without_cad_kernel,
        test_guard_reexec_preserves_arguments_exit_and_wraps_once,
        test_guarded_reexec_is_a_noop_without_recursive_wrapper,
        test_guard_requirement_preserves_caller_error_and_import_order,
        test_scalar_geometry_primitives_freeze_clamping_and_distance,
        test_shared_stl_primitives_and_exporter_adapters,
        test_ordered_assembly_helper_and_narrow_adoption,
        test_to_print_shelf_waits_for_release_catalog,
    )
    for test in tests:
        test()
    print(f"shared contracts: {len(tests)} pure gates pass")


if __name__ == "__main__":
    main()
