"""Low-level binary-STL transaction primitives shared by exporters.

Callers retain their public error wording, tolerance policy, mesh-fact
normalization, optional repairs, and publication transaction ordering.
"""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Any, Iterable


class BinaryStlLayoutError(RuntimeError):
    """A binary STL header/count does not match its exact byte length."""

    def __init__(
        self,
        *,
        path: Path,
        actual_bytes: int,
        triangle_count: int | None,
        expected_bytes: int | None,
    ) -> None:
        super().__init__(str(path))
        self.path = path
        self.actual_bytes = actual_bytes
        self.triangle_count = triangle_count
        self.expected_bytes = expected_bytes

    @property
    def truncated_header(self) -> bool:
        return self.triangle_count is None


def _validated_triangle_count(path: Path, data: bytes | bytearray) -> int:
    actual = len(data)
    if actual < 84:
        raise BinaryStlLayoutError(
            path=path,
            actual_bytes=actual,
            triangle_count=None,
            expected_bytes=None,
        )
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangles
    if triangles < 1 or actual != expected:
        raise BinaryStlLayoutError(
            path=path,
            actual_bytes=actual,
            triangle_count=triangles,
            expected_bytes=expected,
        )
    return triangles


def validate_binary_stl_length(path: str | Path) -> int:
    """Validate exact binary-STL length and return its triangle count."""
    stl = Path(path)
    with stl.open("rb") as stream:
        header = stream.read(84)
    if len(header) < 84:
        raise BinaryStlLayoutError(
            path=stl,
            actual_bytes=stl.stat().st_size,
            triangle_count=None,
            expected_bytes=None,
        )
    triangles = struct.unpack_from("<I", header, 80)[0]
    actual = stl.stat().st_size
    expected = 84 + 50 * triangles
    if triangles < 1 or actual != expected:
        raise BinaryStlLayoutError(
            path=stl,
            actual_bytes=actual,
            triangle_count=triangles,
            expected_bytes=expected,
        )
    return triangles


def canonicalize_near_zero_stl_coordinates(
    path: str | Path,
    epsilon_mm: float,
) -> int:
    """Rewrite nonzero vertex coordinates within ``epsilon_mm`` as +0.0."""
    stl = Path(path)
    data = bytearray(stl.read_bytes())
    triangles = _validated_triangle_count(stl, data)
    changed = 0
    for triangle in range(triangles):
        vertex_base = 84 + 50 * triangle + 12
        for coordinate in range(9):
            offset = vertex_base + 4 * coordinate
            value = struct.unpack_from("<f", data, offset)[0]
            if value != 0.0 and abs(value) <= epsilon_mm:
                struct.pack_into("<f", data, offset, 0.0)
                changed += 1
    if changed:
        stl.write_bytes(data)
    return changed


DEFAULT_TOPOLOGY_DEFECT_KEYS = (
    "open",
    "over_shared",
    "winding",
    "degenerate",
    "duplicates",
    "nonfinite",
    "zero_volume",
    "negative_volume",
    "component_error",
)


def stl_topology_defects(
    path: str | Path,
    *,
    defect_keys: Iterable[str] = DEFAULT_TOPOLOGY_DEFECT_KEYS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return authoritative checker facts and their nonzero defect subset."""
    from check_manifold import stl_diagnostics

    facts = stl_diagnostics(Path(path))
    defects = {key: facts[key] for key in defect_keys if facts.get(key)}
    return facts, defects
