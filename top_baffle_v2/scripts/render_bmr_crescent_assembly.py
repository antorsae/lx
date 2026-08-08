#!/usr/bin/env python3
"""Assembled front views of both candidate BMR pods on the staged UM collar.

``tests/test_bmr_crescent.py`` proves each junction has no window by
projecting both parts and walking sight lines across it.  This is the same
claim as a picture: an orthographic front elevation, looking straight down -Z
at the two parts installed, so the flush skirt can be read by eye instead of
trusted.  Four panels are drawn in a grid -- one row per candidate variant,
one column per stand state, because the UM collar differs between them -- all
under one declared frame, so the coaxial pod and the much taller opposed pod
are directly comparable and their identical junction reads as identical.

A second figure frames the cable entry alone, looking up at the mate face from
where the UM sits, which is the only angle the collar's shape can be judged
from.  Two elevations of the same frame, so a flat face could not hide edge-on
in one of them.  It is drawn from the coaxial pod alone: inside that section
box the two variants are the same geometry, built by the same shared skirt,
duct and collar helpers, which ``test_the_two_variants_share_one_family_module``
asserts by object identity rather than by eye.

The renderer is the project-native one -- build123d tessellation into a
Matplotlib ``Poly3DCollection`` -- with the ISO matrix's own light, colours and
background, so the output is deterministic and looks like the rest of the
review set.

Inputs are read from the build tree and never regenerated::

    LX_CAD_EXECUTION=local make obiwan_bmr_crescent_cad
    python scripts/render_bmr_crescent_assembly.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    _text = str(_root)
    if _text not in sys.path:
        sys.path.insert(0, _text)

if __name__ == "__main__":
    import run_memory_guarded as memory_guard
    memory_guard.reexec_under_guard(Path(__file__))

from gen_product_iso_matrix import (  # noqa: E402
    AMBIENT_LIGHT,
    BACKGROUND,
    FIGURE_DPI,
    LIGHT_DIRECTION,
    MESH_ANGULAR_TOLERANCE,
    MESH_TOLERANCE_MM,
    ROLE_COLORS,
)

DEFAULT_OUTPUT = Path("review/bmr_crescent_assembled_front.png")
DEFAULT_ENTRY_OUTPUT = Path("review/bmr_crescent_entry_closeup.png")
BUILD_ROOT = Path("build") / "bmr_crescent_TEBM35C10-4"
POD_BREP = BUILD_ROOT / "obiwan_bmr_crescent_TEBM35C10-4.brep"
VARIANTS = (
    ("coaxial", "coaxial — two BMRs back to back, 50.2 mm deep", POD_BREP),
    ("opposed", "opposed — the vase layout, 25.1 mm deep and much taller",
     BUILD_ROOT / "obiwan_bmr_crescent_opposed_TEBM35C10-4.brep"),
)
STATES = ("floor_stand", "no_floor_stand")

# Straight-on front elevation: the camera looks along the display Y axis, so
# display X is world X and display Z is world Y.  This is the view the
# assembled-front feedback was written against.
FRONT_ELEV_DEG = 0.0
FRONT_AZIM_DEG = -90.0

# Framed on the junction and the pods rather than on the whole assembly: the
# point of the picture is the seam between the collar and the pod, and a frame
# that fitted the UM's full 310..426 mm reach would leave it a few dozen
# pixels tall.  The top is set by the opposed pod's upper land at y=534.79 --
# Matplotlib does not clip 3-D collections, so a shorter frame would draw that
# land over the titles rather than cropping it -- and every panel shares the
# frame, which is what keeps the two variants comparable.
FRAME_X = (-52.0, 52.0)
FRAME_Y = (401.0, 540.0)
PANEL_INCHES = (5.0, 5.4)

# The entry close-up looks up at the mate face from where the UM sits: in
# front of the part and below it, which is the angle the collar's shape has to
# be judged from.  Two elevations of the same frame, so a flat face could not
# hide edge-on in one of them.
ENTRY_VIEWS = (
    (-32.0, -90.0, "from the UM side, 32° below the mate face"),
    (-58.0, -68.0, "same frame, steeper and off to one side"),
)
# Matplotlib does not clip 3-D collections, so framing alone cannot produce a
# close-up: the rest of the pod is still drawn over it.  The pod is therefore
# sectioned to this world box first.  Every cut plane is at least 7 mm from
# the collar, so the flat faces at the edges of the picture are the section
# and cannot be mistaken for the collar's own surfaces.
ENTRY_SECTION_X = (-14.0, 22.0)
ENTRY_SECTION_Y = (410.0, 434.0)
ENTRY_SECTION_Z = (-6.0, 14.0)
ENTRY_ZOOM = 1.05
ENTRY_PANEL_INCHES = (5.2, 4.8)


def _resolve(path: Path) -> Path:
    absolute = PROJECT_ROOT / path
    if not absolute.is_file():
        raise SystemExit(
            f"missing input: {path}\n"
            "build it first with 'LX_CAD_EXECUTION=local make "
            "obiwan_bmr_crescent_cad' and the staged Obi-Wan exporter")
    return absolute


def _display_triangles(path: Path):
    """Tessellate one BREP into (world X, -world Z, world Y) triangles."""
    import numpy as np
    from build123d import import_brep

    shape = import_brep(str(_resolve(path)))
    vertices, triangles = shape.tessellate(
        MESH_TOLERANCE_MM, MESH_ANGULAR_TOLERANCE)
    xyz = np.asarray(
        [[float(vertex.X), float(vertex.Y), float(vertex.Z)]
         for vertex in vertices], dtype=float)
    indices = np.asarray(triangles, dtype=int)
    if xyz.size == 0 or indices.size == 0:
        raise SystemExit(f"input tessellated to nothing: {path}")
    return np.column_stack((xyz[:, 0], -xyz[:, 2], xyz[:, 1]))[indices]


def _shaded(color: str, triangles):
    import numpy as np
    from matplotlib.colors import to_rgb

    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    lengths[lengths == 0.0] = 1.0
    normals /= lengths[:, None]
    lambert = np.abs(normals @ np.asarray(LIGHT_DIRECTION, dtype=float))
    intensity = AMBIENT_LIGHT + (1.0 - AMBIENT_LIGHT) * lambert
    rgb = np.asarray(to_rgb(color), dtype=float)[None, :] * intensity[:, None]
    return np.clip(
        np.column_stack((rgb, np.ones(len(triangles)))), 0.0, 1.0)


def _draw(axes, state: str, pod: Path) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    collar = (Path("build") / state / ".obiwan_stage"
              / "core_um_carrier.brep")
    facets, colors = [], []
    for role, path in (("base", collar), ("top", pod)):
        triangles = _display_triangles(path)
        facets.append(triangles)
        colors.append(_shaded(ROLE_COLORS[role], triangles))

    axes.set_proj_type("ortho")
    axes.set_facecolor(BACKGROUND)
    axes.add_collection3d(Poly3DCollection(
        np.concatenate(facets), facecolors=np.concatenate(colors),
        edgecolors="none", alpha=1.0))
    depth = 120.0
    axes.set_xlim(*FRAME_X)
    axes.set_ylim(-depth / 2.0, depth / 2.0)
    axes.set_zlim(*FRAME_Y)
    axes.set_box_aspect(
        (FRAME_X[1] - FRAME_X[0], depth, FRAME_Y[1] - FRAME_Y[0]), zoom=1.35)
    axes.view_init(elev=FRONT_ELEV_DEG, azim=FRONT_AZIM_DEG)
    axes.set_axis_off()


def _entry_section_triangles():
    """The pod sectioned to the entry box, in display coordinates."""
    import numpy as np
    from build123d import Box, Part, Pos, import_brep

    pod = Part(import_brep(str(_resolve(POD_BREP))).solids())
    spans = (ENTRY_SECTION_X, ENTRY_SECTION_Y, ENTRY_SECTION_Z)
    centre = [(low + high) / 2.0 for low, high in spans]
    sizes = [high - low for low, high in spans]
    piece = pod & (Pos(*centre) * Box(*sizes))
    vertices, triangles = piece.tessellate(
        MESH_TOLERANCE_MM, MESH_ANGULAR_TOLERANCE)
    xyz = np.asarray(
        [[float(vertex.X), float(vertex.Y), float(vertex.Z)]
         for vertex in vertices], dtype=float)
    return np.column_stack(
        (xyz[:, 0], -xyz[:, 2], xyz[:, 1]))[np.asarray(triangles, dtype=int)]


def _draw_entry(axes, triangles, elev: float, azim: float) -> None:
    """The sectioned pod, framed on the cable entry, seen from the UM's side."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    axes.set_proj_type("ortho")
    axes.set_facecolor(BACKGROUND)
    axes.add_collection3d(Poly3DCollection(
        triangles, facecolors=_shaded(ROLE_COLORS["top"], triangles),
        edgecolors="none", alpha=1.0))
    # Display coordinates are (world X, -world Z, world Y).
    axes.set_xlim(*ENTRY_SECTION_X)
    axes.set_ylim(-ENTRY_SECTION_Z[1], -ENTRY_SECTION_Z[0])
    axes.set_zlim(*ENTRY_SECTION_Y)
    axes.set_box_aspect((
        ENTRY_SECTION_X[1] - ENTRY_SECTION_X[0],
        ENTRY_SECTION_Z[1] - ENTRY_SECTION_Z[0],
        ENTRY_SECTION_Y[1] - ENTRY_SECTION_Y[0]), zoom=ENTRY_ZOOM)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()


def render_entry(output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(
        figsize=(ENTRY_PANEL_INCHES[0] * len(ENTRY_VIEWS),
                 ENTRY_PANEL_INCHES[1] + 0.7),
        facecolor=BACKGROUND)
    figure.suptitle(
        "Cable entry — the Ø6.00 bore and its collar, seen from the UM side",
        fontsize=11, y=0.965)
    figure.text(
        0.5, 0.918,
        "pod alone, orthographic, sectioned to a box round the entry · the "
        "collar is the bore's own sweep offset by one 1.20 mm wall",
        ha="center", fontsize=8, color="#54606b")
    figure.text(
        0.5, 0.891,
        "so it has no flat face and no corner — the flat planes at the edges "
        "of each picture are the section, 7 mm or more away from it",
        ha="center", fontsize=8, color="#54606b")
    figure.text(
        0.5, 0.866,
        "shown on the coaxial pod; both variants build this entry from the "
        "same shared helpers",
        ha="center", fontsize=8, color="#54606b")
    triangles = _entry_section_triangles()
    for index, (elev, azim, caption) in enumerate(ENTRY_VIEWS):
        axes = figure.add_subplot(
            1, len(ENTRY_VIEWS), index + 1, projection="3d")
        _draw_entry(axes, triangles, elev, azim)
        axes.set_title(caption, fontsize=8, y=0.99)
    return _save(figure, output)


def _save(figure, output: Path) -> Path:
    import matplotlib.pyplot as plt

    absolute = PROJECT_ROOT / output
    absolute.parent.mkdir(parents=True, exist_ok=True)
    temporary = absolute.with_name(f".{absolute.stem}.{os.getpid()}.tmp.png")
    try:
        figure.savefig(
            temporary, dpi=FIGURE_DPI, facecolor=BACKGROUND,
            metadata={"Software": None})
        temporary.replace(absolute)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    return absolute


def render(output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, columns = len(VARIANTS), len(STATES)
    figure = plt.figure(
        figsize=(PANEL_INCHES[0] * columns,
                 PANEL_INCHES[1] * rows + 1.1),
        facecolor=BACKGROUND)
    figure.suptitle(
        "Both candidate BMR pods assembled on the Obi-Wan UM collar — front "
        "elevation", fontsize=12, y=0.992)
    # ``suptitle`` anchors its top at ``y``; these must do the same or their
    # baselines land inside it.
    figure.text(
        0.5, 0.971,
        "orthographic, looking down -Z, one shared frame for all four panels",
        ha="center", va="top", fontsize=8, color="#54606b")
    figure.text(
        0.5, 0.959,
        "the junction between the collar and the pod is solid on both · "
        "drivers not fitted, so the driver pockets are open",
        ha="center", va="top", fontsize=8, color="#54606b")
    for row, (key, caption, pod) in enumerate(VARIANTS):
        left = None
        for column, state in enumerate(STATES):
            axes = figure.add_subplot(
                rows, columns, row * columns + column + 1, projection="3d")
            _draw(axes, state, pod)
            axes.set_title(state.replace("_", " "), fontsize=9, y=0.97)
            left = left or axes
        # One caption per row rather than one per panel: the two stand states
        # differ only in the collar, so repeating the variant's description
        # over each of them would be noise.
        figure.text(
            0.5, left.get_position().y1 + 0.004, caption,
            ha="center", fontsize=10, weight="bold", color="#25313a")
    figure.legend(
        handles=[Patch(facecolor=ROLE_COLORS["base"], label="UM collar"),
                 Patch(facecolor=ROLE_COLORS["top"], label="BMR pod")],
        loc="lower center", ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 0.004))
    return _save(figure, output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--entry-output", type=Path, default=DEFAULT_ENTRY_OUTPUT)
    args = parser.parse_args(argv)
    print(render(args.output))
    print(render_entry(args.entry_output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
