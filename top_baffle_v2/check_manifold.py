"""Manifold sweep over every exported STL: each edge of a watertight
mesh must be shared by exactly TWO triangles. Open or over-shared
edges are how an exposed duct, dropped sliver face or self-touching
boolean shows up in the mesh long before a slicer complains.

Run: python check_manifold.py [dir ...]   (default: both stl trees)
Exit 1 on any defect; prints per-file edge parity. Wired into `make
manifold` and the default `all` flow.
"""

from __future__ import annotations

import struct
import sys
from collections import Counter
from pathlib import Path


def stl_edge_parity(path: Path) -> tuple[int, int, int]:
    """(triangles, open edges, over-shared edges) for a binary STL."""
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    edges: Counter = Counter()
    off = 84
    for _ in range(n):
        vs = struct.unpack_from("<9f", data, off + 12)
        tri = [tuple(round(c, 4) for c in vs[i:i + 3]) for i in (0, 3, 6)]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edges[frozenset((tri[a], tri[b]))] += 1
        off += 50
    open_e = sum(1 for c in edges.values() if c == 1)
    over_e = sum(1 for c in edges.values() if c > 2)
    return n, open_e, over_e


def main() -> int:
    roots = ([Path(a) for a in sys.argv[1:]] if sys.argv[1:] else
             [Path(__file__).parent / d / "stl"
              for d in ("floor_stand", "no_floor_stand")])
    files = sorted(f for r in roots if r.is_dir() for f in r.glob("*.stl"))
    if not files:
        print("no STLs found", file=sys.stderr)
        return 1
    bad = 0
    for f in files:
        tris, open_e, over_e = stl_edge_parity(f)
        status = "ok" if not (open_e or over_e) else "DEFECT"
        if open_e or over_e:
            bad += 1
        print(f"  {status:6s} {f.parent.parent.name}/{f.name}: "
              f"{tris} tris, {open_e} open, {over_e} over-shared")
    print(f"{len(files) - bad}/{len(files)} STLs manifold")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
