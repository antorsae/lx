#!/usr/bin/env python3
"""Contracts for the Obi-Wan 01a PETG-GF local-solid modifier."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for import_root in (ROOT / "src", ROOT / "scripts"):
    text = str(import_root)
    if text not in sys.path:
        sys.path.insert(0, text)

from bambu_3mf_audit import mesh_bounds, read_stl_triangles
import build_obiwan_bridge_root_modifier as modifier


def test_modifier_is_front_down_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "bridge_root.modifier.stl"
    contract_path = tmp_path / "bridge_root.modifier.json"
    contract = modifier.build_modifier(
        output_stl=output,
        contract_path=contract_path,
    )

    triangles = read_stl_triangles(output)
    bounds = mesh_bounds(triangles)
    assert len(triangles) == 28
    assert math.isclose(bounds.minimum[2], 0.0, abs_tol=2.0e-4)
    assert math.isclose(bounds.maximum[2], 13.0, abs_tol=2.0e-4)
    assert contract["triangle_count"] == 28
    assert contract["subtype"] == "modifier_part"
    assert contract["role"] == "bridge_root_local_solid"
    assert contract["artifact_match"] == modifier.ARTIFACT_MATCH
    assert contract["process"] == {
        "sparse_infill_density": "100%",
        "sparse_infill_pattern": "zig-zag",
    }
    assert contract_path.is_file()


def test_modifier_covers_bridge_and_both_lower_lm_boss_axes() -> None:
    plan = modifier.SOURCE_PLAN_XY
    assert min(x for x, _y in plan) <= -52.375
    assert max(x for x, _y in plan) >= 52.375
    assert min(y for _x, y in plan) <= 14.0
    assert max(y for _x, y in plan) >= 110.265
    assert modifier.SOURCE_Z_MM == (5.3, 18.3)


def main() -> None:
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        parameters = tuple(inspect.signature(test).parameters)
        if not parameters:
            test()
        elif parameters == ("tmp_path",):
            (ROOT / "build").mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(
                    prefix=f".{test.__name__}-",
                    dir=ROOT / "build") as directory:
                test(Path(directory))
        else:
            raise RuntimeError(
                f"unsupported test signature for {test.__name__}")
        print(f"PASS {test.__name__}")
    print(f"all {len(tests)} bridge/root modifier tests passed")


if __name__ == "__main__":
    main()
