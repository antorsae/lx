#!/usr/bin/env python3
"""Transactional STEP/STL/review exporter for one Obi-Wan Ac/Ae wing family.

The geometry authority is :mod:`obiwan_wings_cad`.  This module owns only
artifact assembly, print orientation, strict mesh validation, review renders,
hash manifests, and last-known-good promotion.  Build it through the normal
remote CAD path; direct invocation receives the project memory guard.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any, Iterable

from front_down_contract import (
    sidecar_path_for_stl,
    validate_print_sidecar,
    write_print_sidecar,
)


SCRIPT_DIR = Path(__file__).resolve().parent
GEOMETRY_MODULE = "obiwan_wings_cad"
GEOMETRY_SOURCE = SCRIPT_DIR / f"{GEOMETRY_MODULE}.py"
CONTRACT_SOURCE = SCRIPT_DIR / "gen_obiwan_wing_design_map.py"
NO_FLOOR_STAGE_MANIFEST = (
    SCRIPT_DIR / "no_floor_stand/.obiwan_stage/manifest.json")
FLOOR_STAGE_MANIFEST = SCRIPT_DIR / "floor_stand/.obiwan_stage/manifest.json"
INTERFACE_SOURCES = (
    SCRIPT_DIR / "cad-remote-requirements.lock",
    SCRIPT_DIR / "captive_magnets.py",
    SCRIPT_DIR / "front_down_contract.py",
    SCRIPT_DIR / "obiwan_acoustic_wings_spec.md",
    SCRIPT_DIR / "gen_driver_overlay.py",
    SCRIPT_DIR / "export_obiwan_staged.py",
    SCRIPT_DIR / "top_baffle_nd25fw4.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_b.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_b1.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_b2.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_cables.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_flush.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_v1.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_attachments.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_a_comp.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_bridge.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_floor.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_floor_strength.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_lm_split.py",
    SCRIPT_DIR / "top_baffle_nd25fw4_obiwan_route.py",
    NO_FLOOR_STAGE_MANIFEST,
    FLOOR_STAGE_MANIFEST,
)

FIXED_TIMESTAMP = "2020-01-01T00:00:00"
VARIANTS = ("ac", "ae")
SIDES = ("left", "right")
PART_ORDER = ("lm_lower", "lm_upper", "um")
BED_LIMIT_MM = 220.0
FRONT_Z_MM = 18.3
REAR_Z_MM = 6.8
SIDE_SECTION_X_MM = 55.0
MESH_TOLERANCE_MM = 0.01
MESH_ANGULAR_TOLERANCE = 0.08
AE_MESH_TOLERANCE_MM = 0.002
AE_MESH_ANGULAR_TOLERANCE = 0.03
REVIEW_MESH_TOLERANCE_MM = 0.22
REVIEW_MESH_ANGULAR_TOLERANCE = 0.30
STL_TRANSFORM_ZERO_EPSILON_MM = 2.0e-7
CONTEXT_COLOR = "#67737e"
CONTEXT_LINESTYLE = (0, (1.2, 2.2))
CONTEXT_LINEWIDTH = 1.15
NO_FLOOR_CONTEXT_COLOR = "#2878b5"
NO_FLOOR_CONTEXT_LINESTYLE = (0, (7.0, 2.2, 1.3, 2.2))
FLOOR_CONTEXT_COLOR = "#2e8b57"
FLOOR_CONTEXT_LINESTYLE = (0, (1.0, 2.2))
STATE_CONTEXT_LINEWIDTH = 1.45

CANONICAL_STEP_TEMPLATE = "top_baffle_nd25fw4_obiwan_wing_{slug}.step"
ASSEMBLED_STEP_TEMPLATE = (
    "top_baffle_nd25fw4_obiwan_wing_{slug}_assembled.step")
FACTS_TEMPLATE = "obiwan_wing_{slug}_facts.json"
MANIFEST_TEMPLATE = "obiwan_wing_{slug}_print_manifest.json"
REVIEW_KINDS = (
    "front", "rear", "side_section", "split_exploded", "magnet_roots")


def _guard_or_reexec() -> None:
    """Give direct CLI use the same guard as Make/remote-worker builds."""
    if __name__ != "__main__":
        return
    import run_memory_guarded as memory_guard

    if memory_guard.is_guarded_process():
        return
    guard = SCRIPT_DIR / "run_memory_guarded.py"
    completed = subprocess.run(
        [sys.executable, str(guard), "--", sys.executable,
         str(Path(__file__).resolve()), *sys.argv[1:]],
        check=False,
    )
    raise SystemExit(completed.returncode)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_attestation() -> dict[str, Any]:
    paths = (Path(__file__).resolve(), GEOMETRY_SOURCE, CONTRACT_SOURCE,
             *INTERFACE_SOURCES)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(
            "basic-wing source attestation is incomplete: "
            + ", ".join(str(path) for path in missing))
    records = []
    aggregate = hashlib.sha256()
    for path in sorted(
            paths,
            key=lambda item: item.relative_to(SCRIPT_DIR).as_posix()):
        relative = path.relative_to(SCRIPT_DIR).as_posix()
        digest = _sha256(path)
        records.append({"path": relative, "sha256": digest})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "geometry_module": GEOMETRY_MODULE,
        "geometry_path": GEOMETRY_SOURCE.name,
        "geometry_sha256": _sha256(GEOMETRY_SOURCE),
        "exporter_path": Path(__file__).name,
        "exporter_sha256": _sha256(Path(__file__).resolve()),
        "contract_generator_path": CONTRACT_SOURCE.name,
        "contract_generator_sha256": _sha256(CONTRACT_SOURCE),
        "combined_sha256": aggregate.hexdigest(),
        "files": records,
    }


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value}")
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_jsonable(payload), indent=2, sort_keys=True)
               + "\n").encode("utf-8")
    path.write_bytes(encoded)
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if parsed.get("schema_version") != payload.get("schema_version"):
        raise RuntimeError(f"JSON transaction failed round trip: {path}")


def _bbox_facts(shape) -> dict[str, list[float]]:
    bbox = shape.bounding_box()
    return {
        "min_mm": [float(bbox.min.X), float(bbox.min.Y), float(bbox.min.Z)],
        "max_mm": [float(bbox.max.X), float(bbox.max.Y), float(bbox.max.Z)],
        "size_mm": [float(bbox.size.X), float(bbox.size.Y),
                    float(bbox.size.Z)],
    }


def _require_solids(shape, expected: int, label: str) -> list[Any]:
    solids = list(shape.solids())
    volumes = [float(solid.volume) for solid in solids]
    if (not shape.is_valid or len(solids) != expected
            or any(volume <= 0.01 for volume in volumes)):
        raise RuntimeError(
            f"{label}: expected {expected} valid positive solids; "
            f"valid={shape.is_valid} volumes={volumes}")
    return solids


def _validate_step_transaction(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError(f"temporary STEP is missing/truncated: {path}")
    with path.open("rb") as stream:
        header = stream.read(32)
        stream.seek(max(0, path.stat().st_size - 4096))
        tail = stream.read()
    if (not header.startswith(b"ISO-10303-21;")
            or not tail.rstrip().endswith(b"END-ISO-10303-21;")):
        raise RuntimeError(f"temporary STEP transaction is incomplete: {path}")


def _validate_binary_stl(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise RuntimeError(f"temporary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", header, 80)[0]
    expected = 84 + 50 * triangles
    if triangles < 1 or path.stat().st_size != expected:
        raise RuntimeError(
            f"temporary STL transaction invalid: triangles={triangles} "
            f"bytes={path.stat().st_size} expected={expected}")


def _canonicalize_transform_zeros(
        path: Path, epsilon_mm: float = STL_TRANSFORM_ZERO_EPSILON_MM) -> int:
    data = bytearray(path.read_bytes())
    if len(data) < 84:
        raise RuntimeError(f"temporary STL is truncated: {path}")
    triangles = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangles
    if len(data) != expected:
        raise RuntimeError(
            f"temporary STL transaction invalid: triangles={triangles} "
            f"bytes={len(data)} expected={expected}")
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
        path.write_bytes(data)
    return changed


def _strict_mesh_facts(path: Path) -> dict[str, Any]:
    from check_manifold import stl_diagnostics

    facts = stl_diagnostics(path)
    defect_keys = (
        "open", "over_shared", "winding", "degenerate", "duplicates",
        "nonfinite", "zero_volume", "negative_volume", "component_error",
    )
    defects = {key: facts[key] for key in defect_keys if facts[key]}
    if defects:
        raise RuntimeError(
            f"temporary STL fails strict manifold contract: {path.name}: "
            f"{defects}; triangles={facts['triangles']} "
            f"components={facts['components']}")
    return _jsonable(facts)


def _part_label(slug: str, side: str, order: int, role: str) -> str:
    return f"obiwan_wing_{slug}_{side}_{order}of3_{role}"


def _stl_name(slug: str, side: str, order: int, role: str) -> str:
    return f"lx521_top_obiwan_wing_{slug}_{side}_{order}of3_{role}.stl"


def _best_print_orientation(shape, Rot) -> tuple[Any, float, dict[str, Any]]:
    """Return a deterministic front-down minimum-XY orientation.

    The former half-degree brute-force sweep transformed the full BREP 381
    times per piece.  The print footprint is a planar projection, so its
    convex minimum rectangle gives the same candidate angle directly; only
    the two orthogonal BREP transforms are evaluated.
    """
    from shapely.geometry import MultiPoint

    front_down = Rot(X=180.0) * shape

    xy = [(float(vertex.X), float(vertex.Y))
          for vertex in front_down.vertices()]
    if len(xy) < 3:
        raise RuntimeError("print part has too few XY witnesses")
    rectangle = MultiPoint(xy).convex_hull.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    edges = []
    for index in range(4):
        dx = coordinates[index + 1][0] - coordinates[index][0]
        dy = coordinates[index + 1][1] - coordinates[index][1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))))
    _length, rectangle_angle = max(edges, key=lambda item: item[0])
    candidate_angles = tuple(sorted({
        (-rectangle_angle) % 180.0,
        (90.0 - rectangle_angle) % 180.0,
    }))

    def score(angle: float):
        candidate = Rot(Z=angle) * front_down
        bbox = candidate.bounding_box()
        size = bbox.size
        return ((max(float(size.X), float(size.Y)),
                 float(size.X) * float(size.Y), abs(angle)),
                candidate, bbox)

    angle, best_score, best_shape, best_bbox = min(
        ((angle, *score(angle)) for angle in candidate_angles),
        key=lambda item: item[1])
    size = best_bbox.size
    if (float(size.X) > BED_LIMIT_MM + 1e-6
            or float(size.Y) > BED_LIMIT_MM + 1e-6
            or float(size.Z) > BED_LIMIT_MM + 1e-6):
        raise RuntimeError(
            "wing print part exceeds 220 mm bed after rotation: "
            f"{size.X:.3f} x {size.Y:.3f} x {size.Z:.3f} mm at "
            f"X180/Z{angle:.2f}")
    translation = (
        -float(best_bbox.min.X),
        -float(best_bbox.min.Y),
        -float(best_bbox.min.Z),
    )
    moved = (importlib.import_module("build123d").Pos(
        *translation) * best_shape)
    _require_solids(moved, 1, "print-oriented part")
    radians = math.radians(float(angle))
    cosine, sine = math.cos(radians), math.sin(radians)
    transform = {
        "print_orientation": "front_face_down",
        "rotation_deg": {"x": 180.0, "z": float(angle)},
        "pre_translation_bbox_min_mm": [
            float(best_bbox.min.X), float(best_bbox.min.Y),
            float(best_bbox.min.Z),
        ],
        "stl_origin_translation_mm": list(translation),
        "source_to_stl_matrix": [
            [cosine, sine, 0.0, translation[0]],
            [sine, -cosine, 0.0, translation[1]],
            [0.0, 0.0, -1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }
    return moved, float(angle), {
        "bbox_mm": _bbox_facts(moved),
        "transform": transform,
    }


def _mesh_records(parts: dict[tuple[str, str], Any]) -> list[dict[str, Any]]:
    import numpy as np

    records = []
    colors = {
        "lm_lower": "#4f9bd7",
        "lm_upper": "#62b77b",
        "um": "#efa43a",
    }
    for (side, role), shape in parts.items():
        vertices, triangles = shape.tessellate(
            REVIEW_MESH_TOLERANCE_MM, REVIEW_MESH_ANGULAR_TOLERANCE)
        xyz = np.asarray(
            [[float(vertex.X), float(vertex.Y), float(vertex.Z)]
             for vertex in vertices], dtype=float)
        indices = np.asarray(triangles, dtype=int)
        if xyz.size == 0 or indices.size == 0:
            raise RuntimeError(f"review tessellation is empty: {side}/{role}")
        records.append({
            "side": side,
            "role": role,
            "color": colors[role],
            "triangles": xyz[indices],
        })
    return records


def _context_mesh_records(parts: dict[str, dict]) -> list[dict[str, Any]]:
    """Tessellate exact staged split-state LM plus no-floor UM/T BREPs."""
    import numpy as np

    identities = {
        "lm_lower_floor": {
            "label": "Obi-Wan LM lower — floor stand",
            "color": FLOOR_CONTEXT_COLOR,
            "linestyle": FLOOR_CONTEXT_LINESTYLE,
            "linewidth": STATE_CONTEXT_LINEWIDTH,
            "style_name": "dotted",
            "legend_label": "LM lower — floor (green dotted)",
            "side_main": False,
        },
        "lm_lower_no_floor": {
            "label": "Obi-Wan LM lower — no-floor",
            "color": NO_FLOOR_CONTEXT_COLOR,
            "linestyle": NO_FLOOR_CONTEXT_LINESTYLE,
            "linewidth": STATE_CONTEXT_LINEWIDTH,
            "style_name": "dash_dot",
            "legend_label": "LM lower — no-floor (blue dash-dot)",
            "side_main": True,
        },
        "lm_upper": {
            "label": "Obi-Wan LM upper — no-floor reference",
            "color": CONTEXT_COLOR,
            "linestyle": CONTEXT_LINESTYLE,
            "linewidth": CONTEXT_LINEWIDTH,
            "style_name": "dotted_neutral",
            "legend_label": "LM upper / UM / T — no-floor (gray dotted)",
            "side_main": True,
        },
        "um": {
            "label": "Obi-Wan UM — no-floor reference",
            "color": CONTEXT_COLOR,
            "linestyle": CONTEXT_LINESTYLE,
            "linewidth": CONTEXT_LINEWIDTH,
            "style_name": "dotted_neutral",
            "legend_label": None,
            "side_main": True,
        },
        "t": {
            "label": "Obi-Wan T crescent — no-floor reference",
            "color": CONTEXT_COLOR,
            "linestyle": CONTEXT_LINESTYLE,
            "linewidth": CONTEXT_LINEWIDTH,
            "style_name": "dotted_neutral",
            "legend_label": None,
            "side_main": True,
        },
    }
    if set(parts) != set(identities):
        raise RuntimeError(
            "Obi-Wan review context must contain two LM-lower states plus "
            "LM-upper, UM and T")
    records = []
    for key, style in identities.items():
        entry = parts[key]
        shape = entry["shape"]
        vertices, triangles = shape.tessellate(
            REVIEW_MESH_TOLERANCE_MM, REVIEW_MESH_ANGULAR_TOLERANCE)
        xyz = np.asarray(
            [[float(vertex.X), float(vertex.Y), float(vertex.Z)]
             for vertex in vertices], dtype=float)
        indices = np.asarray(triangles, dtype=int)
        if xyz.size == 0 or indices.size == 0:
            raise RuntimeError(f"empty Obi-Wan review context mesh: {key}")
        records.append({
            "key": key,
            **style,
            "source_label": entry["source_label"],
            "source_sha256": entry["source_sha256"],
            "state": entry["state"],
            "part_key": entry["part_key"],
            "triangles": xyz[indices],
            "projection_loops": {},
        })
    return records


def _context_legend_handles(context_records: list[dict[str, Any]]):
    """One unambiguous legend entry per state/style family."""
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0], [0], color=record["color"],
            lw=record["linewidth"], ls=record["linestyle"],
            label=record["legend_label"])
        for record in context_records
        if record.get("legend_label")
    ]


def _projected_context_loops(
        record: dict[str, Any], axes: tuple[int, int],
        ) -> tuple[Any, ...]:
    """Exact silhouette loops from the union of projected BREP triangles."""
    import numpy as np
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    cache = record["projection_loops"]
    if axes in cache:
        return cache[axes]
    polygons = []
    for triangle in record["triangles"]:
        projected = np.asarray(triangle[:, axes], dtype=float)
        polygon = Polygon(projected)
        if polygon.is_valid and polygon.area > 1.0e-7:
            polygons.append(polygon)
    if not polygons:
        raise RuntimeError(
            f"{record['label']} has no nondegenerate {axes} projection")
    merged = unary_union(polygons).buffer(0)
    polygon_parts = (
        [merged] if merged.geom_type == "Polygon"
        else [part for part in getattr(merged, "geoms", ())
              if part.geom_type == "Polygon" and part.area > 1.0e-6])
    loops = []
    for polygon in polygon_parts:
        loops.append(np.asarray(polygon.exterior.coords, dtype=float))
        loops.extend(
            np.asarray(ring.coords, dtype=float)
            for ring in polygon.interiors)
    if not loops:
        raise RuntimeError(
            f"{record['label']} projection produced no silhouette loops")
    cache[axes] = tuple(loops)
    return cache[axes]


def _draw_mesh_review(
        path: Path, records: list[dict[str, Any]], *, title: str,
        metadata_variant: str,
        elev: float, azim: float, selection: Iterable[tuple[str, str]] | None = None,
        exploded: bool = False, z_scale: float = 1.0,
        hide_z_axis: bool = False,
        context_records: list[dict[str, Any]] | None = None,
        focus: tuple[tuple[float, float], tuple[float, float],
                     tuple[float, float]] | None = None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    selected = set(selection) if selection is not None else None
    plotted = []
    fig = plt.figure(figsize=(12.0, 8.0), dpi=150, facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_proj_type("ortho")
    for record in records:
        key = (record["side"], record["role"])
        if selected is not None and key not in selected:
            continue
        triangles = np.array(record["triangles"], copy=True)
        if z_scale != 1.0:
            triangles[..., 2] = (
                FRONT_Z_MM
                + z_scale * (triangles[..., 2] - FRONT_Z_MM))
        if exploded:
            role_index = PART_ORDER.index(record["role"])
            direction = -1.0 if record["side"] == "left" else 1.0
            triangles[..., 0] += direction * 9.0 * role_index
            triangles[..., 1] += 10.0 * (role_index - 1)
            triangles[..., 2] += 2.5 * role_index
        collection = Poly3DCollection(
            triangles, facecolor=record["color"], edgecolor="#25313a",
            linewidth=0.08, alpha=0.96)
        ax.add_collection3d(collection)
        plotted.append(triangles.reshape(-1, 3))
    if context_records:
        context_z = (
            FRONT_Z_MM + 0.16 if elev >= 0.0 else REAR_Z_MM - 0.16)
        display_z = (
            FRONT_Z_MM + z_scale * (context_z - FRONT_Z_MM))
        for context in context_records:
            for loop in _projected_context_loops(context, (0, 1)):
                xyz = np.column_stack((
                    loop,
                    np.full(len(loop), display_z, dtype=float),
                ))
                ax.plot(
                    xyz[:, 0], xyz[:, 1], xyz[:, 2],
                    color=context["color"], lw=context["linewidth"],
                    ls=context["linestyle"], alpha=0.94, zorder=20)
                plotted.append(xyz)
    if not plotted:
        raise RuntimeError(f"review selection has no geometry: {title}")
    cloud = np.vstack(plotted)
    if focus is None:
        mins = cloud.min(axis=0)
        maxs = cloud.max(axis=0)
        padding = np.maximum((maxs - mins) * 0.06, (2.0, 2.0, 1.0))
        mins -= padding
        maxs += padding
    else:
        mins = np.asarray([axis[0] for axis in focus], dtype=float)
        maxs = np.asarray([axis[1] for axis in focus], dtype=float)
    ax.set_xlim(float(mins[0]), float(maxs[0]))
    ax.set_ylim(float(mins[1]), float(maxs[1]))
    ax.set_zlim(float(mins[2]), float(maxs[2]))
    spans = np.maximum(maxs - mins, 1.0)
    ax.set_box_aspect(tuple(float(value) for value in spans))
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("world X (mm)", fontsize=9, labelpad=5)
    ax.set_ylabel("world Y (mm)", fontsize=9, labelpad=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=1)
    if hide_z_axis:
        # A plan view has no readable Z dimension.  Hiding it also prevents
        # Matplotlib's projected Z label/ticks from colliding with the title.
        ax.set_zticks([])
        ax.set_zlabel("")
    else:
        ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_zlabel(
            "display Z (mm)" if z_scale != 1.0 else "world Z (mm)",
            fontsize=9, labelpad=7)
        ax.tick_params(axis="z", which="major", labelsize=8, pad=1)
    fig.suptitle(title, fontsize=14, weight="bold", y=0.96)
    if z_scale != 1.0:
        fig.text(
            0.06, 0.045, f"Rear depth shown at Z x{z_scale:g}; XY exact",
            fontsize=9, color="#7b2d26")
    legend_handles = [
        Patch(facecolor="#4f9bd7", label="LM lower"),
        Patch(facecolor="#62b77b", label="LM upper"),
        Patch(facecolor="#efa43a", label="UM / top"),
    ]
    if context_records:
        legend_handles.extend(_context_legend_handles(context_records))
    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0,
        fontsize=9, framealpha=0.96)
    ax.grid(True, linewidth=0.35, alpha=0.45)
    fig.subplots_adjust(left=0.055, right=0.82, bottom=0.10, top=0.90)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=150, facecolor="white",
        metadata={"Title": f"{title} [{metadata_variant}]",
                  "Description": (
                      "Project-native Obi-Wan Ac/Ae CAD QA render with dual-"
                      "state Obi-Wan LM-lower silhouettes: coincident common "
                      "profile, blue dash-dot no-floor and green dotted "
                      "floor stand")})
    plt.close(fig)


def _mesh_x_plane_segments(
        triangles, x_mm: float, tolerance: float = 1.0e-7,
        ) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Intersect tessellated faces with one true world-X section plane.

    Returned coordinates are world ``(Y, Z)``.  A triangle lying exactly in
    the plane is reduced to its longest edge, which keeps the plot stable
    without inventing a projected silhouette.
    """
    import numpy as np

    segments = []
    for triangle in triangles:
        distances = triangle[:, 0] - float(x_mm)
        points = []

        def add_point(point) -> None:
            yz = np.asarray((point[1], point[2]), dtype=float)
            if not any(np.linalg.norm(yz - existing) <= tolerance
                       for existing in points):
                points.append(yz)

        for first, second in ((0, 1), (1, 2), (2, 0)):
            p0, p1 = triangle[first], triangle[second]
            d0, d1 = distances[first], distances[second]
            if abs(d0) <= tolerance:
                add_point(p0)
            if abs(d1) <= tolerance:
                add_point(p1)
            if d0 * d1 < -(tolerance * tolerance):
                fraction = d0 / (d0 - d1)
                add_point(p0 + fraction * (p1 - p0))
        if len(points) < 2:
            continue
        pair = max(
            ((first, second)
             for index, first in enumerate(points)
             for second in points[index + 1:]),
            key=lambda item: float(np.linalg.norm(item[1] - item[0])))
        if np.linalg.norm(pair[1] - pair[0]) > tolerance:
            segments.append((tuple(pair[0]), tuple(pair[1])))
    return segments


def _draw_side_section_review(
        path: Path, records: list[dict[str, Any]], *, title: str,
        metadata_variant: str,
        x_mm: float = SIDE_SECTION_X_MM,
        context_records: list[dict[str, Any]] | None = None) -> None:
    """Draw a real Y/Z plane cut instead of an edge-on 3-D projection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MultipleLocator

    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=150, facecolor="white")
    all_segments = []
    role_labels = {
        "lm_lower": "LM lower",
        "lm_upper": "LM upper",
        "um": "UM / top",
    }
    legend_handles = []
    for record in records:
        if record["side"] != "right":
            continue
        segments = _mesh_x_plane_segments(record["triangles"], x_mm)
        if not segments:
            continue
        all_segments.extend(segments)
        ax.add_collection(LineCollection(
            segments, colors=record["color"], linewidths=1.25,
            alpha=0.98, capstyle="round", joinstyle="round"))
        legend_handles.append(Line2D(
            [0], [0], color=record["color"], lw=3.0,
            label=role_labels[record["role"]]))
    if context_records:
        for context in context_records:
            context_loops = list(
                _projected_context_loops(context, (1, 2)))
            if context.get("side_main"):
                all_segments.extend(
                    (first, second)
                    for loop in context_loops
                    for first, second in zip(loop[:-1], loop[1:]))
            # The floor LM lower is intentionally excluded from main-view
            # autoscaling because its exact foot reaches Z=-150 mm.  The
            # explicit limits below clip it to the acoustic-depth window;
            # the comparison inset renders its complete Y/Z silhouette.
            ax.add_collection(LineCollection(
                context_loops, colors=context["color"],
                linewidths=context["linewidth"],
                linestyles=context["linestyle"], alpha=0.94))
        legend_handles.extend(_context_legend_handles(context_records))
    if not all_segments:
        raise RuntimeError(
            f"side-section plane x={x_mm:.3f} misses all right wing pieces")

    y_values = [coordinate[0] for segment in all_segments for coordinate in segment]
    z_values = [coordinate[1] for segment in all_segments for coordinate in segment]
    y_span = max(max(y_values) - min(y_values), 1.0)
    z_span = max(max(z_values) - min(z_values), 1.0)
    ax.set_xlim(
        min(y_values) - max(5.0, 0.025 * y_span),
        max(y_values) + max(5.0, 0.025 * y_span))
    ax.set_ylim(
        min(z_values) - max(0.6, 0.06 * z_span),
        max(z_values) + max(0.6, 0.06 * z_span))
    ax.axhline(FRONT_Z_MM, color="#273746", lw=1.1, ls="--",
               label="flat acoustic front")
    ax.axhline(REAR_Z_MM, color="#7b2d26", lw=1.0, ls=":",
               label="Ac rear datum")
    ax.xaxis.set_major_locator(MultipleLocator(50.0))
    ax.yaxis.set_major_locator(MultipleLocator(1.0))
    ax.set_xlabel("world Y along baffle (mm)", fontsize=11)
    ax.set_ylabel("true global Z (mm)", fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.38)
    ax.set_title(title, fontsize=14, weight="bold", pad=16)
    ax.legend(
        handles=legend_handles + [
            Line2D([0], [0], color="#273746", lw=1.1, ls="--",
                   label="flat front z=18.3"),
            Line2D([0], [0], color="#7b2d26", lw=1.0, ls=":",
                   label="rear limit z=6.8"),
        ], loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1,
        borderaxespad=0.0, fontsize=8.0, framealpha=0.96)

    if context_records:
        lower_records = [
            context for context in context_records
            if context["key"] in {"lm_lower_no_floor", "lm_lower_floor"}
        ]
        inset = fig.add_axes((0.765, 0.15, 0.215, 0.31))
        inset_cloud = []
        for context in lower_records:
            loops = list(_projected_context_loops(context, (1, 2)))
            inset.add_collection(LineCollection(
                loops, colors=context["color"],
                linewidths=context["linewidth"],
                linestyles=context["linestyle"], alpha=0.96))
            inset_cloud.extend(loops)
        if inset_cloud:
            import numpy as np

            cloud = np.vstack(inset_cloud)
            spans = np.maximum(cloud.max(axis=0) - cloud.min(axis=0), 1.0)
            padding = np.maximum(0.04 * spans, (2.0, 2.0))
            inset.set_xlim(
                cloud[:, 0].min() - padding[0],
                cloud[:, 0].max() + padding[0])
            inset.set_ylim(
                cloud[:, 1].min() - padding[1],
                cloud[:, 1].max() + padding[1])
        inset.set_aspect("equal", adjustable="box")
        inset.grid(True, linewidth=0.35, alpha=0.35)
        inset.tick_params(labelsize=6.5)
        inset.set_xlabel("world Y (mm)", fontsize=7)
        inset.set_ylabel("world Z (mm)", fontsize=7)
        inset.set_title(
            "LM lower alternatives — complete depth", fontsize=8,
            weight="bold", pad=4)
    fig.text(
        0.42, 0.035,
        f"Colored wing = true cut at world X={x_mm:.1f} mm. Main view keeps "
        "acoustic-depth scale; green floor geometry is clipped there and "
        "shown complete in the inset.",
        ha="center", fontsize=8.5, color="#44515c")
    fig.subplots_adjust(left=0.10, right=0.72, bottom=0.13, top=0.88)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=150, facecolor="white",
        metadata={"Title": f"{title} [{metadata_variant}]",
                  "Description": (
                      "True world-X section of Obi-Wan Ac/Ae CAD with dual-"
                      "state Obi-Wan LM-lower silhouettes: coincident common "
                      "profile, blue dash-dot no-floor and green dotted "
                      "floor stand")})
    plt.close(fig)


def _draw_magnet_root_review(
        path: Path, records: list[dict[str, Any]], receiver_records,
        *, title: str, metadata_variant: str,
        context_records: list[dict[str, Any]] | None = None) -> None:
    """Dimension the base-LM and two radial receiver pockets in XY."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Polygon as PolygonPatch

    receiver_order = {
        "lm_lower_right": 0, "lm_upper_right": 1, "um_right": 2}
    receivers = sorted(
        receiver_records, key=lambda item: receiver_order.get(item["name"], 99))
    if (len(receivers) != 3
            or {item["name"] for item in receivers} != set(receiver_order)):
        raise RuntimeError(
            "magnet-root review requires right base-LM, radial-LM and UM "
            "receivers")
    by_role = {(record["side"], record["role"]): record for record in records}
    context_by_key = {
        record["key"]: record for record in (context_records or [])}
    fig, axes = plt.subplots(
        1, 3, figsize=(16.0, 7.5), dpi=150, facecolor="white")
    receiver_color = "#d83b91"
    cavity_color = "#d9f0f3"
    retaining_color = "#183b4e"
    for ax, receiver in zip(axes, receivers, strict=True):
        role = receiver["name"].removesuffix("_right")
        record = by_role.get(("right", role))
        if record is None:
            raise RuntimeError(f"magnet-root review lacks right/{role} mesh")
        ax.add_collection(PolyCollection(
            record["triangles"][..., :2], facecolor=record["color"],
            edgecolor="#25313a", linewidth=0.08, alpha=0.82))
        context_keys = {
            "lm_lower": ("lm_lower_floor", "lm_lower_no_floor"),
            "lm_upper": ("lm_upper",),
            "um": ("um",),
        }[role]
        for context_key in context_keys:
            context = context_by_key.get(context_key)
            if context is None:
                continue
            for loop in _projected_context_loops(context, (0, 1)):
                ax.plot(
                    loop[:, 0], loop[:, 1], color=context["color"],
                    lw=context["linewidth"], ls=context["linestyle"],
                    alpha=0.96, zorder=8)

        normal = np.asarray(receiver["axis_normal_xy"], dtype=float)
        normal /= np.linalg.norm(normal)
        tangent = np.asarray((-normal[1], normal[0]), dtype=float)
        mouth = np.asarray(receiver["receiver_mouth_xy_mm"], dtype=float)
        carrier_face = np.asarray(receiver["carrier_face_xy_mm"], dtype=float)
        # Receiver facts are the same captive-cavity authority consumed by
        # the release catalog.  Do not retain aliases for the retired exposed
        # glue-pocket schema: a stale review must fail with the production
        # field names rather than silently describing D5.2 x 2.2 geometry.
        diameter = float(receiver["cavity_diameter_mm"])
        depth = float(receiver["cavity_depth_mm"])
        magnet_diameter = float(receiver["magnet_diameter_mm"])
        magnet_depth = float(receiver["magnet_depth_mm"])
        face_skin = float(receiver["face_skin_mm"])
        inner_skin = float(receiver["inner_skin_mm"])
        captive_land = float(receiver["captive_land_mm"])
        gap = float(receiver["carrier_to_receiver_face_gap_mm"])
        cavity_start = mouth + face_skin * normal
        cavity_end = cavity_start + depth * normal
        land_end = mouth + captive_land * normal
        inner_end = cavity_end + inner_skin * normal
        if not np.allclose(inner_end, land_end, atol=1.0e-9, rtol=0.0):
            raise RuntimeError(
                f"{receiver['name']}: captive land does not equal face skin "
                "+ cavity depth + inner skin")
        magnet_start = cavity_start
        magnet_end = magnet_start + magnet_depth * normal
        half_diameter = 0.5 * diameter
        half_magnet = 0.5 * magnet_diameter

        def axial_band(start, end, half_width):
            return np.vstack((
                start - half_width * tangent,
                start + half_width * tangent,
                end + half_width * tangent,
                end - half_width * tangent,
            ))

        cavity = axial_band(cavity_start, cavity_end, half_diameter)
        face_retainer = axial_band(mouth, cavity_start, half_diameter)
        inner_retainer = axial_band(cavity_end, inner_end, half_diameter)
        magnet = axial_band(magnet_start, magnet_end, half_magnet)
        ax.add_patch(PolygonPatch(
            cavity, closed=True, facecolor=cavity_color, alpha=0.82,
            edgecolor="#2b7a87", linestyle="--", linewidth=1.4, zorder=7))
        # These two 0.45-mm strips are the physically printed retaining walls.
        # The previous plot covered the full land with a magenta cavity/magnet
        # envelope and therefore hid both skins even though the STEP contained
        # them.  Keep them opaque and above both the wing mesh and cavity.
        for wall in (face_retainer, inner_retainer):
            ax.add_patch(PolygonPatch(
                wall, closed=True, facecolor=retaining_color, alpha=0.96,
                edgecolor="white", linewidth=0.75, hatch="////", zorder=9))
        ax.add_patch(PolygonPatch(
            magnet, closed=True, facecolor=receiver_color, alpha=0.55,
            edgecolor="#8d145c", linewidth=1.5, zorder=8))
        ax.plot(
            [land_end[0] - half_diameter * tangent[0],
             land_end[0] + half_diameter * tangent[0]],
            [land_end[1] - half_diameter * tangent[1],
             land_end[1] + half_diameter * tangent[1]],
            color=retaining_color, lw=1.2, zorder=10)
        ax.scatter(*carrier_face, s=72, facecolor="white", edgecolor="#1e2a33",
                   linewidth=1.4, zorder=9)
        ax.scatter(*mouth, s=42, facecolor=receiver_color, edgecolor="white",
                   linewidth=1.0, zorder=10)
        ax.annotate(
            "", xy=mouth + (depth + 4.0) * normal, xytext=carrier_face,
            arrowprops={"arrowstyle": "->", "color": "#8d145c",
                        "lw": 1.8}, zorder=10)

        centre = 0.5 * (mouth + land_end)
        half_view = 12.0
        ax.set_xlim(centre[0] - half_view, centre[0] + half_view)
        ax.set_ylim(centre[1] - half_view, centre[1] + half_view)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("world X (mm)", fontsize=10)
        ax.set_ylabel("world Y (mm)", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, linewidth=0.45, alpha=0.42)
        ax.set_title(
            f"{role.replace('_', ' ').upper()} receiver — "
            f"{receiver['name']}",
            fontsize=11, weight="bold", pad=10)
        ax.text(
            0.035, 0.965,
            f"Ø{diameter:.1f} × {depth:.1f} mm\n"
            f"skins {face_skin:.2f} / {inner_skin:.2f} mm\n"
            f"carrier gap {gap:.2f} mm\n"
            f"axis z={float(receiver['axis_z_mm']):.2f} mm",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                  "edgecolor": receiver_color, "alpha": 0.94}, zorder=12)

    if context_records:
        inset = fig.add_axes((0.435, 0.695, 0.13, 0.15))
        all_xy = []
        for context in context_records:
            for loop in _projected_context_loops(context, (0, 1)):
                inset.plot(
                    loop[:, 0], loop[:, 1], color=context["color"],
                    lw=max(0.8, 0.75 * context["linewidth"]),
                    ls=context["linestyle"], alpha=0.96)
                all_xy.append(loop)
        if all_xy:
            cloud = np.vstack(all_xy)
            span = np.maximum(cloud.max(axis=0) - cloud.min(axis=0), 1.0)
            pad = 0.04 * span
            inset.set_xlim(cloud[:, 0].min() - pad[0],
                           cloud[:, 0].max() + pad[0])
            inset.set_ylim(cloud[:, 1].min() - pad[1],
                           cloud[:, 1].max() + pad[1])
        inset.set_aspect("equal", adjustable="box")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_title(
            "exact LM lower states + no-floor upper/UM/T",
            fontsize=7.1, pad=2)
        for spine in inset.spines.values():
            spine.set_color("#a3abb2")
            spine.set_linewidth(0.6)

    fig.suptitle(title, fontsize=14, weight="bold", y=0.95)
    legend_handles = [
            Patch(facecolor="#4f9bd7", label="LM-lower wing material"),
            Patch(facecolor="#62b77b", label="LM-upper wing material"),
            Patch(facecolor="#efa43a", label="UM/top wing material"),
            Patch(facecolor=cavity_color, edgecolor="#2b7a87",
                  label="D5.20 × 2.10 internal cavity"),
            Patch(facecolor=retaining_color, edgecolor="white", hatch="////",
                  label="0.45-mm captive retaining skin"),
            Patch(facecolor=receiver_color, alpha=0.55,
                  label="nominal D5 × 2 magnet"),
        ]
    if context_records:
        legend_handles.extend(_context_legend_handles(context_records))
    legend_handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor="#1e2a33", label="carrier magnet face"))
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4,
        fontsize=7.8, framealpha=0.96,
        bbox_to_anchor=(0.5, 0.045))
    fig.text(
        0.5, 0.105,
        "Dark hatched strips are the two continuous 0.45-mm retaining skins; "
        "cyan is the cavity, magenta the seated magnet. The 45° roof closes "
        "in Z and is therefore outside this XY plan view.",
        ha="center", fontsize=9, color="#44515c")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.18, top=0.87, wspace=0.24)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path, dpi=150, facecolor="white",
        metadata={"Title": f"{title} [{metadata_variant}]",
                  "Description": (
                      "Dimensioned Obi-Wan captive D5.20 x 2.10 receivers, visible "
                      "0.45-mm axial retaining skins, and dual-"
                      "state Obi-Wan LM-lower silhouettes: coincident common "
                      "profile, blue dash-dot no-floor and green dotted "
                      "floor stand")})
    plt.close(fig)


def _validate_png(path: Path) -> None:
    from PIL import Image

    if not path.is_file() or path.stat().st_size < 4096:
        raise RuntimeError(f"review PNG is missing/truncated: {path}")
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        if image.width < 1200 or image.height < 800:
            raise RuntimeError(
                f"review PNG is too small: {path}: {image.size}")


def _render_reviews(
        slug: str, parts: dict[tuple[str, str], Any], review_dir: Path,
        receiver_records, geometry,
        ) -> tuple[list[Path], dict[str, Any]]:
    display_slug = {"ac": "Ac", "ae": "Ae"}[slug]
    metadata_variant = slug.upper()
    records = _mesh_records(parts)
    context_parts = geometry.wing_review_split_context_parts(
        NO_FLOOR_STAGE_MANIFEST, FLOOR_STAGE_MANIFEST)
    context_records = _context_mesh_records(context_parts)
    outputs = {kind: review_dir / f"obiwan_wing_{slug}_{kind}.png"
               for kind in REVIEW_KINDS}
    _draw_mesh_review(
        outputs["front"], records,
        title=(f"Obi-Wan {display_slug} — acoustic front / common LM-lower "
               "profile"),
        metadata_variant=metadata_variant,
        elev=90.0, azim=-90.0, hide_z_axis=True,
        context_records=context_records)
    _draw_mesh_review(
        outputs["rear"], records,
        title=f"Obi-Wan {display_slug} — rear surface / dovetail seams",
        metadata_variant=metadata_variant,
        elev=-90.0, azim=90.0, hide_z_axis=True,
        context_records=context_records)
    _draw_side_section_review(
        outputs["side_section"], records,
        title=(f"Obi-Wan {display_slug} — true right-wing rear-depth profile "
               "(Y/Z section)"),
        metadata_variant=metadata_variant,
        context_records=context_records)
    _draw_mesh_review(
        outputs["split_exploded"], records,
        title=f"Obi-Wan {display_slug} — six-piece exploded print assembly",
        metadata_variant=metadata_variant,
        elev=28.0, azim=-62.0, exploded=True,
        context_records=context_records)
    _draw_magnet_root_review(
        outputs["magnet_roots"], records,
        title=f"Obi-Wan {display_slug} — right LM/UM magnetic roots",
        metadata_variant=metadata_variant,
        receiver_records=receiver_records,
        context_records=context_records)
    for path in outputs.values():
        _validate_png(path)
    context_facts = {
        "state_contract": "dual_lm_lower_with_no_floor_upper_um_t",
        "records": [
            {
                "key": record["key"],
                "state": record["state"],
                "part_key": record["part_key"],
                "source_label": record["source_label"],
                "source_sha256": record["source_sha256"],
                "color": record["color"],
                "line_style": record["style_name"],
                "xy_path_count": len(
                    _projected_context_loops(record, (0, 1))),
                "yz_path_count": len(
                    _projected_context_loops(record, (1, 2))),
                "z_bounds_mm": [
                    float(record["triangles"][..., 2].min()),
                    float(record["triangles"][..., 2].max()),
                ],
            }
            for record in context_records
        ],
    }
    return [outputs[kind] for kind in REVIEW_KINDS], context_facts


def _artifact_record(slug_stage: Path, relative: Path, kind: str) -> dict[str, Any]:
    staged = slug_stage / relative
    if not staged.is_file() or staged.stat().st_size <= 0:
        raise RuntimeError(f"staged artifact is missing: {relative}")
    return {
        "path": relative.as_posix(),
        "kind": kind,
        "size_bytes": staged.stat().st_size,
        "sha256": _sha256(staged),
    }


def _promote_transaction(staged_slug: Path, final_slug: Path) -> None:
    """Promote one complete slug directory with rollback on failure."""
    if not staged_slug.is_dir():
        raise RuntimeError(f"staged slug directory is missing: {staged_slug}")
    final_slug.parent.mkdir(parents=True, exist_ok=True)
    backup_slug = (
        final_slug.parent / f".{final_slug.name}.{os.getpid()}.backup")
    slug_promoted = False
    try:
        if backup_slug.exists():
            shutil.rmtree(backup_slug)
        if final_slug.exists():
            final_slug.replace(backup_slug)
        staged_slug.replace(final_slug)
        slug_promoted = True
    except Exception:
        if slug_promoted and final_slug.exists():
            shutil.rmtree(final_slug)
        if backup_slug.exists():
            backup_slug.replace(final_slug)
        raise
    else:
        if backup_slug.exists():
            shutil.rmtree(backup_slug)


def _output_root(path: Path) -> Path:
    root = path if path.is_absolute() else SCRIPT_DIR / path
    root = root.resolve()
    if root == SCRIPT_DIR or SCRIPT_DIR not in root.parents:
        raise ValueError(
            f"output root must be a child of {SCRIPT_DIR}, got {root}")
    return root


def _result_path(path: Path) -> str:
    try:
        return path.relative_to(SCRIPT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def _export_variant(slug: str, output_root_arg: Path) -> dict[str, Any]:
    os.environ.setdefault("LX_ROUTING_PROFILE", "obiwan")
    from build123d import Compound, Rot, export_step, export_stl

    geometry = importlib.import_module(GEOMETRY_MODULE)
    for api_name in (
            "wing_monolithic", "wing_print_parts", "wing_facts",
            "receiver_facts", "wing_review_split_context_parts"):
        if not callable(getattr(geometry, api_name, None)):
            raise RuntimeError(
                f"{GEOMETRY_MODULE} lacks required API {api_name}()")

    source = _source_attestation()
    output_root = _output_root(output_root_arg)
    final_slug = output_root / slug
    stage_root = (output_root.parent
                  / f".{output_root.name}_{slug}.{os.getpid()}.transaction")
    if stage_root.exists():
        shutil.rmtree(stage_root)
    slug_stage = stage_root / slug
    stl_stage = slug_stage / "stl"
    review_stage = slug_stage / "review"
    stl_stage.mkdir(parents=True, exist_ok=True)
    review_stage.mkdir(parents=True, exist_ok=True)

    try:
        monoliths: dict[str, Any] = {}
        for side in SIDES:
            shape = geometry.wing_monolithic(slug, side)
            _require_solids(shape, 1, f"{slug}/{side} monolith")
            shape.label = f"obiwan_wing_{slug}_{side}_monolithic"
            monoliths[side] = shape

        print_parts: dict[tuple[str, str], Any] = {}
        for side in SIDES:
            mapping = geometry.wing_print_parts(slug, side)
            if set(mapping) != set(PART_ORDER):
                raise RuntimeError(
                    f"{slug}/{side}: print roles {sorted(mapping)} do not "
                    f"match {list(PART_ORDER)}")
            for order, role in enumerate(PART_ORDER, start=1):
                shape = mapping[role]
                _require_solids(shape, 1, f"{slug}/{side}/{role}")
                shape.label = _part_label(slug, side, order, role)
                print_parts[(side, role)] = shape

        canonical_label = f"lx521_obiwan_basic_wing_{slug}_monolithic_pair"
        canonical = Compound(children=[monoliths[side] for side in SIDES])
        canonical.label = canonical_label
        _require_solids(canonical, 2, canonical_label)
        assembly_label = f"lx521_obiwan_basic_wing_{slug}_print_assembly"
        assembled_children = [
            print_parts[(side, role)]
            for side in SIDES for role in PART_ORDER
        ]
        assembled = Compound(children=assembled_children)
        assembled.label = assembly_label
        _require_solids(assembled, 6, assembly_label)

        canonical_rel = Path(CANONICAL_STEP_TEMPLATE.format(slug=slug))
        assembled_rel = Path(ASSEMBLED_STEP_TEMPLATE.format(slug=slug))
        canonical_path = slug_stage / canonical_rel
        assembled_path = slug_stage / assembled_rel
        export_step(canonical, str(canonical_path), timestamp=FIXED_TIMESTAMP)
        export_step(assembled, str(assembled_path), timestamp=FIXED_TIMESTAMP)
        _validate_step_transaction(canonical_path)
        _validate_step_transaction(assembled_path)

        part_facts = []
        stl_relatives = []
        sidecar_relatives = []
        for side in SIDES:
            for order, role in enumerate(PART_ORDER, start=1):
                shape = print_parts[(side, role)]
                mesh_tolerance = (
                    AE_MESH_TOLERANCE_MM
                    if slug == "ae" else MESH_TOLERANCE_MM)
                mesh_angular_tolerance = (
                    AE_MESH_ANGULAR_TOLERANCE
                    if slug == "ae" else MESH_ANGULAR_TOLERANCE)
                moved, z_angle, print_facts = _best_print_orientation(shape, Rot)
                name = _stl_name(slug, side, order, role)
                relative = Path("stl") / name
                path = slug_stage / relative
                export_stl(
                    moved, str(path), tolerance=mesh_tolerance,
                    angular_tolerance=mesh_angular_tolerance)
                _validate_binary_stl(path)
                zero_fixes = _canonicalize_transform_zeros(path)
                mesh_facts = _strict_mesh_facts(path)
                sidecar_relative = relative.with_suffix(".print.json")
                sidecar_path = sidecar_path_for_stl(path)
                if sidecar_path != slug_stage / sidecar_relative:
                    raise RuntimeError(
                        f"non-canonical print sidecar path for {relative}: "
                        f"{sidecar_path}")
                write_print_sidecar(
                    path,
                    part=path.stem,
                    transform=print_facts["transform"],
                    extra={
                        "artifact_family": "obiwan_wing_artifacts",
                        "variant_slug": slug,
                        "assembly_label": shape.label,
                        "side": side,
                        "order": order,
                        "role": role,
                        "mesh": {
                            "tolerance_mm": mesh_tolerance,
                            "angular_tolerance": mesh_angular_tolerance,
                        },
                    },
                )
                sidecar_payload = validate_print_sidecar(
                    path, sidecar_path)
                sidecar_transform = {
                    key: sidecar_payload.get(key)
                    for key in print_facts["transform"]
                }
                if sidecar_transform != print_facts["transform"]:
                    raise RuntimeError(
                        f"print sidecar transform drifted from exporter: "
                        f"{sidecar_path}")
                if sidecar_payload.get("stl_sha256") != _sha256(path):
                    raise RuntimeError(
                        f"print sidecar does not bind its STL: "
                        f"{sidecar_path}")
                entry = {
                    "label": shape.label,
                    "path": relative.as_posix(),
                    "side": side,
                    "order": order,
                    "role": role,
                    "assembly_bbox_mm": _bbox_facts(shape),
                    "print_bbox_mm": print_facts["bbox_mm"],
                    "volume_mm3": geometry.adaptive_volume_mm3(shape),
                    "volume_integration": "BRepGProp_adaptive_2d_Gauss",
                    "bed_limit_mm": BED_LIMIT_MM,
                    "print_transform_deg": {"x": 180.0, "z": z_angle},
                    "print_transform": print_facts["transform"],
                    "print_sidecar": sidecar_relative.as_posix(),
                    "print_sidecar_sha256": _sha256(sidecar_path),
                    "mesh_tolerance_mm": mesh_tolerance,
                    "mesh_angular_tolerance": mesh_angular_tolerance,
                    "transform_zero_fixes": zero_fixes,
                    "stl_diagnostics": mesh_facts,
                }
                part_facts.append(entry)
                stl_relatives.append(relative)
                sidecar_relatives.append(sidecar_relative)

        review_paths, review_context = _render_reviews(
            slug, print_parts, review_stage, geometry.receiver_facts("right"),
            geometry)
        review_relatives = [Path("review") / path.name for path in review_paths]

        source_geometry = geometry.wing_facts(slug)
        facts_rel = Path(FACTS_TEMPLATE.format(slug=slug))
        facts_path = slug_stage / facts_rel
        facts_payload = {
            "schema_version": 2,
            "artifact_family": "obiwan_wing_artifacts",
            "variant_slug": slug,
            "source": source,
            "geometry": _jsonable(source_geometry),
            "exports": {
                "canonical_step": {
                    "path": canonical_rel.as_posix(),
                    "label": canonical_label,
                    "solid_count": 2,
                    "bbox_mm": _bbox_facts(canonical),
                },
                "assembled_step": {
                    "path": assembled_rel.as_posix(),
                    "label": assembly_label,
                    "solid_count": 6,
                    "bbox_mm": _bbox_facts(assembled),
                },
                "print_parts": part_facts,
                "review_pngs": [path.as_posix() for path in review_relatives],
                "review_context": review_context,
            },
        }
        _write_json(facts_path, facts_payload)

        artifact_specs = [
            (canonical_rel, "canonical_step"),
            (assembled_rel, "assembled_step"),
            *((relative, "print_stl") for relative in stl_relatives),
            *((relative, "print_sidecar")
              for relative in sidecar_relatives),
            (facts_rel, "facts_json"),
            *((relative, "review_png") for relative in review_relatives),
        ]
        artifacts = sorted(
            (_artifact_record(slug_stage, relative, kind)
             for relative, kind in artifact_specs),
            key=lambda record: record["path"],
        )
        manifest_rel = Path(MANIFEST_TEMPLATE.format(slug=slug))
        manifest_path = slug_stage / manifest_rel
        manifest_payload = {
            "schema_version": 2,
            "artifact_family": "obiwan_wing_artifacts",
            "variant_slug": slug,
            "source": source,
            "facts_path": facts_rel.as_posix(),
            "facts_sha256": _sha256(facts_path),
            "canonical_step_path": canonical_rel.as_posix(),
            "assembled_step_path": assembled_rel.as_posix(),
            "print_parts": [entry["path"] for entry in part_facts],
            "print_sidecars": [
                entry["print_sidecar"] for entry in part_facts],
            "review_pngs": [path.as_posix() for path in review_relatives],
            "artifacts": artifacts,
        }
        _write_json(manifest_path, manifest_payload)

        _promote_transaction(slug_stage, final_slug)
        result = {
            "variant_slug": slug,
            "canonical_step": _result_path(final_slug / canonical_rel),
            "assembled_step": _result_path(final_slug / assembled_rel),
            "stl_count": len(stl_relatives),
            "print_sidecar_count": len(sidecar_relatives),
            "facts": _result_path(final_slug / facts_rel),
            "manifest": _result_path(final_slug / manifest_rel),
            "review_pngs": [
                _result_path(final_slug / path) for path in review_relatives],
            "source_sha256": source["combined_sha256"],
        }
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export one complete Obi-Wan Ac/Ae wing artifact family")
    parser.add_argument(
        "--slug", "--variant", dest="slug", required=True, choices=VARIANTS)
    parser.add_argument("--output-root", type=Path, default=Path("wings"))
    args = parser.parse_args()
    result = _export_variant(args.slug, args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    _guard_or_reexec()
    raise SystemExit(main())
