#!/usr/bin/env python3
"""Standardized ISO render set for the README product-comparison matrix.

Every cell is rendered from an already-built STEP file with one shared camera
(`ISO_ELEV_DEG`/`ISO_AZIM_DEG`), one shared background, and a frame declared as
a constant rather than fitted to the input.  A declared frame is what makes the
matrix comparable: rendering a single cell with ``--cell`` produces exactly the
same pixels as rendering it inside a complete sweep, and the floor-stand and
no-floor-stand cells of one product stay registered against each other.

Two scale groups exist.  The six product cells share one absolute world frame,
so their parts are directly comparable in size and position.  The three
tweeter-option cells share one fixed *span* but are centred on their own
geometry, because a tweeter carrier drawn inside the whole-product frame would
be a few dozen pixels tall.  Scale is exact within a group and deliberately
different between groups; each render states its group in the corner label.

The same cells are written in two shapes.  ``images/generated/iso/`` holds one
square PNG per cell, which the product docs embed individually.
``images/generated/iso/rows/`` holds the four wide images the README stacks
full-width: one row per product carrying that product's two stand states side
by side, and one row carrying the three tweeter options.  A row panel is drawn
by the same code, under the same camera and the same declared frame, as the
matching single cell, so a row and its cells cannot disagree.

Geometry is read from the promoted build tree, never regenerated here.  Build
the inputs first with the ordinary remote targets (``make``, ``make
obiwan_wings``, ``make vase_tebm35c10_4_cad``) plus the local candidate target
``make obiwan_bmr_crescent_cad``; this script fails closed and names the
missing target when an input is absent.

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
import math
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

# A row panel is one full cell: the same square axes, so geometry is drawn at
# exactly the cell's inches-per-millimetre.  Matplotlib fits a 3-D box by the
# smaller side of its axes, so a panel cannot simply be made narrower — it
# would shrink the part instead of trimming margin.  Rows are therefore
# composed from rendered panels: every panel of a row is cropped with one
# shared box, so the parts stay registered and the empty depth a stock-bridge
# cell leaves for the floor stem survives the crop.
ROW_PANEL_INCHES = FIGURE_INCHES[1]
ROW_PAD_PX = 40
ROW_GUTTER_PX = 72
# Header band above the panels (row title, the shared camera/scale line, and
# the per-panel captions) and footer band below them for the role legend.
ROW_HEADER_INCHES = 1.20
ROW_FOOTER_INCHES = 0.38

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
        # Sized to the extreme of each axis across all three options with a
        # small margin: the opposed-BMR vase is the widest (121.3) and tallest
        # (210.2), the coaxial BMR crescent is by far the deepest (50.2).  All
        # three cells share the frame, so their relative sizes stay honest —
        # the ND25FW-4 crescent really is that much smaller.
        "span": (132.0, 56.0, 215.0),
        "center": None,
        "zoom": 1.08,
    },
}


def _state_sources(state: str) -> dict[str, Path]:
    root = Path("build") / state
    return {
        "stock": root / "b2_split.step",
        # The V1L export already bundles the unchanged V1 vase, so this one
        # file is the complete four-piece Slim product.  `v1_split.step` is
        # NOT its other half -- it is the separate V1 variant, full-depth
        # bottom/mids under the same vase -- and drawing it here superimposed
        # a second baffle over Slim's own.
        "slim": root / "v1l_split.step",
        "obiwan_core": root / "obiwan_split.step",
        "obiwan_crescent": root / "obiwan_attachments.step",
    }


OBIWAN_WINGS = Path("build/wings/flat/obiwan_wing_flat_assembled.step")
TEBM_VASE = Path("build/vase_TEBM35C10-4/stock/vase_TEBM35C10-4.step")
TEBM_CRESCENT = Path(
    "build/bmr_crescent_TEBM35C10-4/obiwan_bmr_crescent_TEBM35C10-4.step")

MAKE_TARGET_FOR_PREFIX = (
    (Path("build/floor_stand"), "make floor_stand"),
    (Path("build/no_floor_stand"), "make no_floor_stand"),
    (Path("build/wings"), "make obiwan_wings"),
    (Path("build/vase_TEBM35C10-4"), "make vase_tebm35c10_4_cad"),
    (Path("build/bmr_crescent_TEBM35C10-4"), "make obiwan_bmr_crescent_cad"),
)


def _cell(key: str, title: str, caption: str, scale_group: str,
          parts: tuple[tuple[str, Path], ...], note: str | None = None) -> dict:
    """One render.  `title` heads its own PNG; `caption`/`note` head its row
    panel, where the row title already carries the product name."""
    return {
        "key": key,
        "title": title,
        "caption": caption,
        "note": note,
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
            f"stock_{state}", f"Stock — {state_label}", state_label,
            "product", (("base", source["stock"]),)))
        cells.append(_cell(
            f"slim_{state}", f"Slim — {state_label}", state_label,
            "product", (("base", source["slim"]),)))
        # Obi-Wan's mandatory geometry is two bare collars.  The optional flat
        # wings and tweeter crescent are included so this cell shows a
        # comparable acoustic baffle rather than two rings in empty space; the
        # legend keeps the mandatory/optional split explicit.
        cells.append(_cell(
            f"obiwan_{state}", f"Obi-Wan — {state_label}", state_label,
            "product",
            (("base", source["obiwan_core"]),
             ("top", source["obiwan_crescent"]),
             ("perimeter", OBIWAN_WINGS))))
    return tuple(cells)


CELLS = (
    *_product_cells(),
    _cell(
        "tweeter_nd25fw4_crescent",
        "ND25FW-4 face-to-face crescent",
        "ND25FW-4 face-to-face pair",
        "tweeter_option",
        (("top", _state_sources("no_floor_stand")["obiwan_crescent"]),),
        note="default on every product · Obi-Wan crescent shown"),
    _cell(
        "tweeter_tebm35c10_4_vase",
        "TEBM35C10-4 opposed BMR vase",
        "TEBM35C10-4 opposed BMR vase",
        "tweeter_option",
        (("top", TEBM_VASE),),
        note="Stock and Slim only · replaces the standard vase"),
    _cell(
        "tweeter_tebm35c10_4_crescent",
        "TEBM35C10-4 coaxial BMR crescent (candidate)",
        "TEBM35C10-4 coaxial BMR crescent",
        "tweeter_option",
        (("top", TEBM_CRESCENT),),
        note="Obi-Wan only · candidate, not release-authorized"),
)

# One row per README block.  Every panel of a row belongs to one scale group,
# so a row is internally comparable by construction; `_render_row` asserts it.
ROWS = (
    {
        "key": "stock_row",
        "title": "Stock — full-depth 18.3 mm proud baffle",
        "cells": ("stock_no_floor_stand", "stock_floor_stand"),
    },
    {
        "key": "slim_row",
        "title": "Slim — 11.5 mm front-flush proud baffle",
        "cells": ("slim_no_floor_stand", "slim_floor_stand"),
    },
    {
        "key": "obiwan_row",
        "title": "Obi-Wan — two bare collars, optional crescent and wings",
        "cells": ("obiwan_no_floor_stand", "obiwan_floor_stand"),
    },
    {
        "key": "tweeter_row",
        "title": "Tweeter options — two driver choices, three implementations",
        "cells": ("tweeter_nd25fw4_crescent", "tweeter_tebm35c10_4_vase",
                  "tweeter_tebm35c10_4_crescent"),
    },
)

ROLE_LEGEND = {
    "base": "base / carrier",
    "top": "vase / tweeter carrier",
    "perimeter": "optional perimeter",
}

# Rows live one level down so `images/generated/iso/*.png` stays exactly the
# per-cell set the product docs embed.
ROW_SUBDIR = Path("rows")

CELL_BY_KEY = {cell["key"]: cell for cell in CELLS}
for _row in ROWS:
    _unknown_panels = [key for key in _row["cells"] if key not in CELL_BY_KEY]
    if _unknown_panels:
        raise SystemExit(
            f"row {_row['key']} names unknown cell(s): "
            f"{', '.join(_unknown_panels)}")


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


def _draw(axes, cell: dict) -> list[str]:
    """Draw one cell into a prepared 3-D axes; return the roles it used."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    group = SCALE_GROUPS[cell["scale_group"]]
    axes.set_proj_type("ortho")
    axes.set_facecolor(BACKGROUND)

    roles = []
    cloud = []
    facets = []
    colors = []
    for role, path in cell["parts"]:
        triangles = _display_triangles(path)
        facets.append(triangles)
        colors.append(_shaded_facecolors(ROLE_COLORS[role], triangles))
        cloud.append(triangles.reshape(-1, 3))
        if role not in roles:
            roles.append(role)

    # Every part goes into ONE collection with per-facet colours.  Matplotlib
    # depth-sorts whole collections against each other by a single value, so
    # separate collections let a later part occlude an earlier one outright --
    # Slim's vase covered its own V1L base that way.  Inside one collection
    # the sort is per facet, which is what interleaves the parts correctly.
    # Facet shading rather than a wireframe: a review-resolution mesh drawn
    # with visible edges reads as noise on the large flat acoustic faces, and
    # hides the shape these images exist to show.
    axes.add_collection3d(Poly3DCollection(
        np.concatenate(facets), facecolors=np.concatenate(colors),
        edgecolors="none", alpha=1.0))

    mins, maxs = _frame(group, np.vstack(cloud))
    axes.set_xlim(float(mins[0]), float(maxs[0]))
    axes.set_ylim(float(mins[1]), float(maxs[1]))
    axes.set_zlim(float(mins[2]), float(maxs[2]))
    axes.set_box_aspect(
        tuple(float(value) for value in group["span"]), zoom=group["zoom"])
    axes.view_init(elev=ISO_ELEV_DEG, azim=ISO_AZIM_DEG)
    axes.set_axis_off()
    return roles


def _camera_line(group: dict) -> str:
    return (f"ISO elev {ISO_ELEV_DEG:g}° / azim {ISO_AZIM_DEG:g}° · "
            f"{group['label']}")


def _legend_handles(roles):
    from matplotlib.patches import Patch

    return [Patch(facecolor=ROLE_COLORS[role], label=ROLE_LEGEND[role])
            for role in roles]


def _save(fig, output: Path, title: str, description: str) -> None:
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(
            temporary, dpi=FIGURE_DPI, facecolor=BACKGROUND,
            metadata={
                "Software": None,
                "Title": title,
                "Description": description,
            })
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(fig)


def _render(cell: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    group = SCALE_GROUPS[cell["scale_group"]]
    fig = plt.figure(
        figsize=FIGURE_INCHES, dpi=FIGURE_DPI, facecolor=BACKGROUND)
    axes = fig.add_subplot(111, projection="3d")
    roles = _draw(axes, cell)

    fig.text(
        0.035, 0.965, cell["title"], fontsize=13, weight="bold",
        color="#25313a", va="top")
    fig.text(
        0.035, 0.925, _camera_line(group),
        fontsize=9, color="#66717e", va="top")
    axes.legend(
        handles=_legend_handles(roles),
        loc="lower left", bbox_to_anchor=(0.0, 0.0), borderaxespad=0.0,
        fontsize=8, framealpha=0.92)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    _save(
        fig, output,
        f"LX521.4 top baffle ISO matrix — {cell['key']}",
        f"{cell['title']}; elev={ISO_ELEV_DEG:g}; azim={ISO_AZIM_DEG:g}; "
        f"scale_group={cell['scale_group']}; span_mm={group['span']}")


def _panel_pixels(cell: dict):
    """Render one cell into a bare square panel and return its RGB pixels."""
    import matplotlib.pyplot as plt
    import numpy as np

    figure = plt.figure(
        figsize=(ROW_PANEL_INCHES, ROW_PANEL_INCHES), dpi=FIGURE_DPI,
        facecolor=BACKGROUND)
    axes = figure.add_axes((0.0, 0.0, 1.0, 1.0), projection="3d")
    roles = _draw(axes, cell)
    figure.canvas.draw()
    pixels = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return pixels, roles


def _shared_crop(panels) -> tuple[int, int, int, int]:
    """One crop box covering the drawn geometry of every panel in a row.

    Cropping each panel by the same box is what keeps a row honest: the panels
    stay registered against each other, so the depth a floor stem occupies and
    the matching stock-bridge cell leaves empty is still visible as empty.
    """
    import numpy as np
    from matplotlib.colors import to_rgb

    background = np.asarray(to_rgb(BACKGROUND), dtype=float) * 255.0
    height, width = panels[0].shape[:2]
    drawn = np.zeros((height, width), dtype=bool)
    for pixels in panels:
        drawn |= np.abs(pixels.astype(float) - background).sum(axis=2) > 12.0
    rows = np.flatnonzero(drawn.any(axis=1))
    columns = np.flatnonzero(drawn.any(axis=0))
    if rows.size == 0 or columns.size == 0:
        raise SystemExit("ISO-matrix row rendered nothing to crop")
    return (max(int(rows[0]) - ROW_PAD_PX, 0),
            min(int(rows[-1]) + 1 + ROW_PAD_PX, height),
            max(int(columns[0]) - ROW_PAD_PX, 0),
            min(int(columns[-1]) + 1 + ROW_PAD_PX, width))


def _render_row(row: dict, output: Path) -> None:
    """Compose one README row from its own cells' panels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [CELL_BY_KEY[key] for key in row["cells"]]
    groups = {cell["scale_group"] for cell in cells}
    if len(groups) != 1:
        raise SystemExit(
            f"row {row['key']} mixes scale groups {sorted(groups)}; "
            "a row must be comparable within itself")
    group = SCALE_GROUPS[cells[0]["scale_group"]]

    rendered = []
    roles = []
    for cell in cells:
        pixels, cell_roles = _panel_pixels(cell)
        rendered.append(pixels)
        for role in cell_roles:
            if role not in roles:
                roles.append(role)
    top, bottom, left, right = _shared_crop(rendered)
    cropped = [pixels[top:bottom, left:right] for pixels in rendered]

    panels = len(cropped)
    panel_height, panel_width = cropped[0].shape[:2]
    header = round(ROW_HEADER_INCHES * FIGURE_DPI)
    footer = round(ROW_FOOTER_INCHES * FIGURE_DPI)
    width = (panels * panel_width + (panels - 1) * ROW_GUTTER_PX
             + 2 * ROW_PAD_PX)
    height = header + panel_height + footer
    fig = plt.figure(
        figsize=(width / FIGURE_DPI, height / FIGURE_DPI), dpi=FIGURE_DPI,
        facecolor=BACKGROUND)

    def from_top(inches: float) -> float:
        return 1.0 - inches * FIGURE_DPI / height

    for index, (cell, pixels) in enumerate(zip(cells, cropped)):
        offset = ROW_PAD_PX + index * (panel_width + ROW_GUTTER_PX)
        fig.figimage(pixels, xo=offset, yo=footer, origin="upper")
        centre = (offset + panel_width / 2.0) / width
        fig.text(
            centre, from_top(0.84), cell["caption"], fontsize=13,
            weight="bold", color="#25313a", ha="center", va="top")
        if cell["note"]:
            fig.text(
                centre, from_top(1.06), cell["note"], fontsize=10,
                color="#66717e", ha="center", va="top")

    fig.text(
        ROW_PAD_PX / width, from_top(0.26), row["title"], fontsize=17,
        weight="bold", color="#25313a", va="top")
    fig.text(
        ROW_PAD_PX / width, from_top(0.58), _camera_line(group),
        fontsize=10, color="#66717e", va="top")
    fig.legend(
        handles=_legend_handles(roles), loc="lower left",
        bbox_to_anchor=(ROW_PAD_PX / width, 0.0), borderaxespad=0.0,
        fontsize=9, ncols=len(roles), frameon=False)

    _save(
        fig, output,
        f"LX521.4 top baffle ISO row — {row['key']}",
        f"{row['title']}; panels={'|'.join(row['cells'])}; "
        f"elev={ISO_ELEV_DEG:g}; azim={ISO_AZIM_DEG:g}; "
        f"scale_group={cells[0]['scale_group']}; span_mm={group['span']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR,
        help=f"output directory (default: {DEFAULT_OUTDIR})")
    parser.add_argument(
        "--cell", action="append", default=None, metavar="KEY",
        help="render only this cell; repeatable")
    parser.add_argument(
        "--row", action="append", default=None, metavar="KEY",
        help="render only this README row; repeatable")
    parser.add_argument(
        "--list", action="store_true",
        help="print the cell and row keys and their sources, then exit")
    arguments = parser.parse_args(argv)

    if arguments.list:
        for cell in CELLS:
            sources = " ".join(str(path) for _, path in cell["parts"])
            print(f"cell\t{cell['key']}\t{cell['scale_group']}\t{sources}")
        for row in ROWS:
            print(f"row\t{row['key']}\t{' '.join(row['cells'])}")
        return 0

    # Naming either kind selects only what was named, so `--cell` alone stays
    # the single-cell shortcut it has always been.
    explicit = arguments.cell is not None or arguments.row is not None
    known_cells = {cell["key"] for cell in CELLS}
    known_rows = {row["key"] for row in ROWS}
    cells_selected = (
        set(arguments.cell) if arguments.cell
        else (set() if explicit else known_cells))
    rows_selected = (
        set(arguments.row) if arguments.row
        else (set() if explicit else known_rows))
    unknown = sorted(cells_selected - known_cells)
    if unknown:
        raise SystemExit(f"unknown ISO-matrix cell(s): {', '.join(unknown)}")
    unknown = sorted(rows_selected - known_rows)
    if unknown:
        raise SystemExit(f"unknown ISO-matrix row(s): {', '.join(unknown)}")

    outdir = arguments.outdir
    if not outdir.is_absolute():
        outdir = PROJECT_ROOT / outdir
    # Resolve every input before drawing anything: a matrix half-written
    # because its last cell was never built is worse than one not written.
    required = set(cells_selected)
    for row in ROWS:
        if row["key"] in rows_selected:
            required.update(row["cells"])
    for cell in CELLS:
        if cell["key"] in required:
            for _role, path in cell["parts"]:
                _resolve(path)

    def report(output: Path) -> None:
        reported = (
            output.relative_to(PROJECT_ROOT)
            if output.is_relative_to(PROJECT_ROOT) else output)
        print(f"wrote {reported}")

    for cell in CELLS:
        if cell["key"] not in cells_selected:
            continue
        output = outdir / f"{cell['key']}.png"
        _render(cell, output)
        report(output)
    for row in ROWS:
        if row["key"] not in rows_selected:
            continue
        output = outdir / ROW_SUBDIR / f"{row['key']}.png"
        _render_row(row, output)
        report(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
