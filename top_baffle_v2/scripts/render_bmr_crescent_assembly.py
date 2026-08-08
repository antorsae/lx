#!/usr/bin/env python3
"""Assembled front view of the candidate BMR pod on the staged UM collar.

``tests/test_bmr_crescent.py`` proves the junction has no window by projecting
both parts and walking sight lines across it.  This is the same claim as a
picture: an orthographic front elevation, looking straight down -Z at the two
parts installed, so the flush skirt can be read by eye instead of trusted.

Two panels are drawn side by side, one per stand state, because the UM collar
differs between them.  The renderer is the project-native one -- build123d
tessellation into a Matplotlib ``Poly3DCollection`` -- with the ISO matrix's
own light, colours and background, so the output is deterministic and looks
like the rest of the review set.

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
POD_BREP = (Path("build") / "bmr_crescent_TEBM35C10-4"
            / "obiwan_bmr_crescent_TEBM35C10-4.brep")
STATES = ("floor_stand", "no_floor_stand")

# Straight-on front elevation: the camera looks along the display Y axis, so
# display X is world X and display Z is world Y.  This is the view the
# assembled-front feedback was written against.
FRONT_ELEV_DEG = 0.0
FRONT_AZIM_DEG = -90.0

# Framed on the junction rather than on the whole assembly: the point of the
# picture is the seam between the collar and the pod, and a frame that fitted
# the UM's full 310..426 mm reach would leave it a few dozen pixels tall.
FRAME_X = (-52.0, 52.0)
FRAME_Y = (401.0, 492.0)
PANEL_INCHES = (5.0, 4.6)


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


def _draw(axes, state: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import numpy as np

    collar = (Path("build") / state / ".obiwan_stage"
              / "core_um_carrier.brep")
    facets, colors = [], []
    for role, path in (("base", collar), ("top", POD_BREP)):
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


def render(output: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    figure = plt.figure(
        figsize=(PANEL_INCHES[0] * len(STATES), PANEL_INCHES[1] + 0.9),
        facecolor=BACKGROUND)
    figure.suptitle(
        "Candidate BMR pod assembled on the Obi-Wan UM collar — front "
        "elevation", fontsize=11, y=0.965)
    figure.text(
        0.5, 0.915,
        "orthographic, looking down -Z · the junction between the collar and "
        "the pod is solid · drivers not fitted, so the front pocket is open "
        "and the partition pass shows through it",
        ha="center", fontsize=8, color="#54606b")
    for index, state in enumerate(STATES):
        axes = figure.add_subplot(
            1, len(STATES), index + 1, projection="3d")
        _draw(axes, state)
        axes.set_title(state.replace("_", " "), fontsize=9, y=0.98)
    figure.legend(
        handles=[Patch(facecolor=ROLE_COLORS["base"], label="UM collar"),
                 Patch(facecolor=ROLE_COLORS["top"], label="BMR pod")],
        loc="lower center", ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, 0.005))

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(render(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
