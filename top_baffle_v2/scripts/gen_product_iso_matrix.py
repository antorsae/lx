#!/usr/bin/env python3
"""Standardized ISO render set for the README product-comparison matrix.

Every cell is rendered from an already-built STEP file with one shared camera
(`ISO_ELEV_DEG`/`ISO_AZIM_DEG`), one shared background, and a frame declared as
a constant rather than fitted to the input.  A declared frame is what makes the
matrix comparable: rendering a single cell with ``--cell`` produces exactly the
same pixels as rendering it inside a complete sweep, and the floor-stand and
no-floor-stand cells of one product stay registered against each other.

Two scale groups exist.  The six product cells share one absolute world frame,
so their parts are directly comparable in size and position.  The two
tweeter-option cells share one fixed *span* but are centred on their own
geometry, because a tweeter carrier drawn inside the whole-product frame would
be a few dozen pixels tall.  Scale is exact within a group and deliberately
different between groups; each render states its group in the corner label.

Geometry is read from the promoted build tree, never regenerated here.  Build
the inputs first with the ordinary remote targets (``make``, ``make
obiwan_wings``, ``make vase_tebm35c10_4_cad``); this script fails closed and
names the missing target when an input is absent.

The renderer is the project-native one used by ``scripts/export_obiwan_wings.py``:
build123d tessellation into a Matplotlib ``Poly3DCollection`` under an
orthographic projection.  Output is deterministic -- no timestamps and no
Matplotlib ``Software`` tag -- so an unchanged build re-renders byte-identical
PNGs.

Ordinary use is the local packaging target::

    make iso_matrix
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
for _canonical_import_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _canonical_import_text = str(_canonical_import_root)
    if _canonical_import_text not in sys.path:
        sys.path.insert(0, _canonical_import_text)

if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))


DEFAULT_OUTDIR = Path("images/generated/iso")

# One camera for every cell.  Geometry is remapped to display coordinates
# `(world X, -world Z, world Y)` first, so Matplotlib's own vertical axis
# carries the baffle height and the default azimuth quadrant looks at the
# acoustic front from above-right.  Changing either angle invalidates the
# comparison and must be done for the whole matrix at once.
ISO_ELEV_DEG = 22.0
ISO_AZIM_DEG = -58.0

# One declared light for the whole matrix, in the same display coordinates,
# placed roughly over the camera's left shoulder. `AMBIENT_LIGHT` is the
# floor brightness so a face turned fully away is still legible.
LIGHT_DIRECTION = (-0.35, -0.72, 0.60)
AMBIENT_LIGHT = 0.45

FIGURE_INCHES = (6.4, 6.4)
FIGURE_DPI = 150
BACKGROUND = "#f4f6f8"

# Tessellation is a review resolution: fine enough that the driver apertures
# and fairings read as curves, far coarser than any release mesh. These images
# document product shape, not manufactured wall thickness.
MESH_TOLERANCE_MM = 0.12
MESH_ANGULAR_TOLERANCE = 0.20

ROLE_COLORS = {
    "base": "#4f9bd7",
    "top": "#efa43a",
    "perimeter": "#62b77b",
}

# Display-space frames.  `span` is `(dx, dy, dz)` in millimetres and is what
# fixes the scale; `center` pins the frame absolutely, or is None to centre the
# frame on each cell's own bounding box.  `zoom` only trims Matplotlib's own
# margin around the projected box and is constant inside a group, so it does
# not affect comparability.
#
# The product frame is absolute: floor-stand and stock-bridge cells of one
# product stay registered, and the floor stem visibly occupies depth the
# stock-bridge cell leaves empty.  It contains every product cell with margin
# (widest is Stock/Slim at X ±152.401, deepest is either floor stand at display
# Y 150, tallest is Stock/Slim at display Z 453.457).
SCALE_GROUPS = {
    "product": {
        "label": "product scale",
        "span": (320.0, 180.0, 490.0),
        "center": (0.0, 66.0, 235.0),
        "zoom": 1.12,
    },
    "tweeter_option": {
        "label": "tweeter-option scale",
        # Wide enough for the taller opposed-BMR vase; both tweeter cells
        # share the frame so their relative sizes stay honest.
        "span": (170.0, 60.0, 215.0),
        "center": None,
        "zoom": 1.08,
    },
}


def _state_sources(state: str) -> dict[str, Path]:
    root = Path("build") / state
    return {
        "stock": root / "b2_split.step",
        "slim_base": root / "v1l_split.step",
        "slim_top": root / "v1_split.step",
        "obiwan_core": root / "obiwan_split.step",
        "obiwan_crescent": root / "obiwan_attachments.step",
    }


OBIWAN_WINGS = Path("build/wings/flat/obiwan_wing_flat_assembled.step")
TEBM_VASE = Path("build/vase_TEBM35C10-4/stock/vase_TEBM35C10-4.step")

MAKE_TARGET_FOR_PREFIX = (
    (Path("build/floor_stand"), "make floor_stand"),
    (Path("build/no_floor_stand"), "make no_floor_stand"),
    (Path("build/wings"), "make obiwan_wings"),
    (Path("build/vase_TEBM35C10-4"), "make vase_tebm35c10_4_cad"),
)


def _cell(key: str, title: str, scale_group: str,
          parts: tuple[tuple[str, Path], ...]) -> dict:
    return {
        "key": key,
        "title": title,
        "scale_group": scale_group,
        "parts": parts,
    }


def _product_cells() -> tuple[dict, ...]:
    cells = []
    for state, state_label in (
            ("no_floor_stand", "stock bridge (no floor stand)"),
            ("floor_stand", "floor stand")):
        source = _state_sources(state)
        cells.append(_cell(
            f"stock_{state}", f"Stock R6P — {state_label}", "product",
            (("base", source["stock"]),)))
        cells.append(_cell(
            f"slim_{state}", f"Slim R6P — {state_label}", "product",
            (("base", source["slim_base"]),
             ("top", source["slim_top"]))))
        # Obi-Wan's mandatory geometry is two bare collars.  The optional flat
        # wings and tweeter crescent are included so this cell shows a
        # comparable acoustic baffle rather than two rings in empty space; the
        # legend keeps the mandatory/optional split explicit.
        cells.append(_cell(
            f"obiwan_{state}", f"Obi-Wan R6F — {state_label}", "product",
            (("base", source["obiwan_core"]),
             ("top", source["obiwan_crescent"]),
             ("perimeter", OBIWAN_WINGS))))
    return tuple(cells)


CELLS = (
    *_product_cells(),
    _cell(
        "tweeter_nd25fw4_crescent",
        "ND25FW-4 face-to-face crescent", "tweeter_option",
        (("top", _state_sources("no_floor_stand")["obiwan_crescent"]),)),
    _cell(
        "tweeter_tebm35c10_4_vase",
        "TEBM35C10-4 opposed BMR vase", "tweeter_option",
        (("top", TEBM_VASE),)),
)

ROLE_LEGEND = {
    "base": "base / carrier",
    "top": "vase / tweeter carrier",
    "perimeter": "optional perimeter",
}


def _make_target_hint(path: Path) -> str:
    for prefix, target in MAKE_TARGET_FOR_PREFIX:
        if prefix == path or prefix in path.parents:
            return target
    return "make"


def _resolve(path: Path) -> Path:
    absolute = PROJECT_ROOT / path
    if not absolute.is_file():
        raise SystemExit(
            f"missing ISO-matrix input: {path}\n"
            f"build it first with '{_make_target_hint(path)}'")
    return absolute


def _display_triangles(path: Path):
    """Tessellate one STEP file into display-space triangles.

    Display coordinates are `(world X, -world Z, world Y)`: Matplotlib's third
    axis becomes the baffle height and its second axis becomes rear-to-front
    depth, so the shared camera angles read as an ordinary product ISO.
    """
    import numpy as np
    from build123d import import_step

    shape = import_step(str(_resolve(path)))
    vertices, triangles = shape.tessellate(
        MESH_TOLERANCE_MM, MESH_ANGULAR_TOLERANCE)
    xyz = np.asarray(
        [[float(vertex.X), float(vertex.Y), float(vertex.Z)]
         for vertex in vertices], dtype=float)
    indices = np.asarray(triangles, dtype=int)
    if xyz.size == 0 or indices.size == 0:
        raise SystemExit(f"ISO-matrix input tessellated to nothing: {path}")
    display = np.column_stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1]))
    return display[indices]


def _shaded_facecolors(color: str, triangles):
    """Per-facet Lambert shading of one base colour.

    Matplotlib's own ``Poly3DCollection(shade=True)`` has moved around across
    releases; doing the Lambert term here keeps the matrix reproducible on any
    supported Matplotlib and lets every cell share one declared light.
    """
    import numpy as np
    from matplotlib.colors import to_rgb

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    normals /= lengths[:, None]
    # Two-sided: a tessellated STEP mixes facet windings and a signed term
    # would blacken half of an otherwise continuous surface.
    lambert = np.abs(normals @ np.asarray(LIGHT_DIRECTION, dtype=float))
    intensity = AMBIENT_LIGHT + (1.0 - AMBIENT_LIGHT) * lambert
    rgb = np.asarray(to_rgb(color), dtype=float)[None, :] * intensity[:, None]
    return np.clip(
        np.column_stack((rgb, np.ones(len(triangles)))), 0.0, 1.0)


def _frame(group: dict, cloud):
    """Return `(mins, maxs)` for one cell under its scale group."""
    import numpy as np

    span = np.asarray(group["span"], dtype=float)
    if group["center"] is None:
        center = (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0
    else:
        center = np.asarray(group["center"], dtype=float)
    return center - span / 2.0, center + span / 2.0


def _render(cell: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    group = SCALE_GROUPS[cell["scale_group"]]
    fig = plt.figure(
        figsize=FIGURE_INCHES, dpi=FIGURE_DPI, facecolor=BACKGROUND)
    axes = fig.add_subplot(111, projection="3d")
    axes.set_proj_type("ortho")
    axes.set_facecolor(BACKGROUND)

    roles = []
    cloud = []
    for role, path in cell["parts"]:
        triangles = _display_triangles(path)
        # Facet shading rather than per-triangle wireframe: a review-resolution
        # mesh drawn with visible edges reads as noise on the large flat
        # acoustic faces, and hides the shape these images exist to show.
        axes.add_collection3d(Poly3DCollection(
            triangles, facecolors=_shaded_facecolors(ROLE_COLORS[role],
                                                     triangles),
            edgecolors="none", alpha=1.0))
        cloud.append(triangles.reshape(-1, 3))
        if role not in roles:
            roles.append(role)

    mins, maxs = _frame(group, np.vstack(cloud))
    axes.set_xlim(float(mins[0]), float(maxs[0]))
    axes.set_ylim(float(mins[1]), float(maxs[1]))
    axes.set_zlim(float(mins[2]), float(maxs[2]))
    axes.set_box_aspect(
        tuple(float(value) for value in group["span"]), zoom=group["zoom"])
    axes.view_init(elev=ISO_ELEV_DEG, azim=ISO_AZIM_DEG)
    axes.set_axis_off()

    fig.text(
        0.035, 0.965, cell["title"], fontsize=13, weight="bold",
        color="#25313a", va="top")
    fig.text(
        0.035, 0.925,
        f"ISO elev {ISO_ELEV_DEG:g}° / azim {ISO_AZIM_DEG:g}° · "
        f"{group['label']}",
        fontsize=9, color="#66717e", va="top")
    axes.legend(
        handles=[Patch(facecolor=ROLE_COLORS[role], label=ROLE_LEGEND[role])
                 for role in roles],
        loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.0,
        fontsize=8, framealpha=0.92)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(
            temporary, dpi=FIGURE_DPI, facecolor=BACKGROUND,
            metadata={
                "Software": None,
                "Title": f"LX521.4 top baffle ISO matrix — {cell['key']}",
                "Description": (
                    f"{cell['title']}; elev={ISO_ELEV_DEG:g}; "
                    f"azim={ISO_AZIM_DEG:g}; "
                    f"scale_group={cell['scale_group']}; "
                    f"span_mm={group['span']}"),
            })
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR,
        help=f"output directory (default: {DEFAULT_OUTDIR})")
    parser.add_argument(
        "--cell", action="append", default=None, metavar="KEY",
        help="render only this cell; repeatable")
    parser.add_argument(
        "--list", action="store_true",
        help="print the cell keys and their sources, then exit")
    arguments = parser.parse_args(argv)

    if arguments.list:
        for cell in CELLS:
            sources = " ".join(str(path) for _, path in cell["parts"])
            print(f"{cell['key']}\t{cell['scale_group']}\t{sources}")
        return 0

    known = {cell["key"] for cell in CELLS}
    selected = set(arguments.cell) if arguments.cell else known
    unknown = sorted(selected - known)
    if unknown:
        raise SystemExit(f"unknown ISO-matrix cell(s): {', '.join(unknown)}")

    outdir = arguments.outdir
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    # Resolve every input before drawing anything: a matrix half-written
    # because its last cell was never built is worse than one not written.
    for cell in CELLS:
        if cell["key"] in selected:
            for _role, path in cell["parts"]:
                _resolve(path)
    for cell in CELLS:
        if cell["key"] not in selected:
            continue
        output = outdir / f"{cell['key']}.png"
        _render(cell, output)
        reported = (
            output.relative_to(PROJECT_ROOT)
            if output.is_relative_to(PROJECT_ROOT) else output)
        print(f"wrote {reported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
