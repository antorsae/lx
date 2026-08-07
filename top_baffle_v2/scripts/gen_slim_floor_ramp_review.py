#!/usr/bin/env python3
"""Side-section review of the slim floor bottom's rear-thickness ramp.

The floor-state ``piece_bottom`` used to hold the thin 11.5 mm field down to
y=96 and then swell back to the full 18.3 mm section over an 18 mm smoothstep
ending at y=78 -- four millimetres above the Option-B vertical tangent, so the
plate visibly bulged just before it met its stand.  It now runs one quintic
ramp over the whole span the stand leaves free: slim from the seam-A dovetails
down to 2 mm below their root, then a single smooth thickening that reaches
full depth exactly where the stand arc completes.

This renders both rear profiles from the exported meshes so the change can be
inspected against real geometry rather than the thickness law alone.  Pass the
two STLs and an output path; the section plane defaults to a constant-X lane
clear of every duct, driver pilot and the D190 aperture.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lx521_baffle.base import L22_CUTOUT, THICKNESS_MM  # noqa: E402
from lx521_baffle.floor_bend import (  # noqa: E402
    BEND_REAR_SPAN_MM,
    BEND_RISE_MM,
    centerline_controls,
    cubic_point,
)
from lx521_baffle.geom import smootherstep01  # noqa: E402
from lx521_baffle.proud.v1l import RAMP_Y0, RAMP_Y1, REAR_MM  # noqa: E402
from lx521_baffle.proud.v1l_split import (  # noqa: E402
    FLOOR_RAMP_FULL_DEPTH_Y_MM,
    FLOOR_RAMP_SLIM_MARGIN_MM,
    FLOOR_RAMP_SLIM_Y_MM,
    SEAM_A_Y,
)

# Overview lane: far enough outboard of the LM/UM/TS ducts and their foot
# lanes to cut solid material, still inboard of the foot's plan taper so the
# whole stand -- arc, rear foot and connector panel -- appears in section.
SECTION_X_MM = -40.0
# Rear-profile lane.  The overview lane enters the D190 acoustic aperture at
# y=114.8, so its rear face stops being a rear face before the seam.  Take the
# profile on the -66 mm seam-A dovetail axis instead: solid all the way up,
# and it carries the section through the dovetail itself, which is exactly the
# interface that has to stay at the slim 11.5 mm section.
PROFILE_X_MM = -66.0

OLD_COLOR = "0.55"
NEW_COLOR = "tab:blue"


def _inverse_print_transform(stl_path: Path):
    """Map printed STL coordinates back to baffle model coordinates.

    Exported meshes are laid down front-face-down, so their vertices are in
    bed space.  The print sidecar records the exact source-to-STL affine that
    put them there; invert it rather than re-deriving the orientation.
    """
    sidecar = stl_path.with_suffix(".print.json")
    matrix = json.loads(sidecar.read_text())["source_to_stl_matrix"]
    scale = [matrix[axis][axis] for axis in range(3)]
    offset = [matrix[axis][3] for axis in range(3)]
    for axis in range(3):
        row = matrix[axis]
        if abs(scale[axis]) != 1.0 or any(
                row[other] != 0.0 for other in range(3) if other != axis):
            raise SystemExit(
                f"{sidecar.name} is not an axis-aligned print transform")
    return lambda p: tuple(
        (p[axis] - offset[axis]) / scale[axis] for axis in range(3))


def read_binary_stl(path: Path):
    to_model = _inverse_print_transform(path)
    data = path.read_bytes()
    count = struct.unpack_from("<I", data, 80)[0]
    triangles = []
    offset = 84
    for _ in range(count):
        values = struct.unpack_from("<12fH", data, offset)
        triangles.append(tuple(
            to_model(values[start:start + 3])
            for start in (3, 6, 9)))
        offset += 50
    return triangles


def section_segments(triangles, plane_x: float):
    """Intersect a mesh with the plane x=``plane_x``; return (y, z) pairs."""
    segments = []
    for tri in triangles:
        distances = [vertex[0] - plane_x for vertex in tri]
        if min(distances) > 0.0 or max(distances) < 0.0:
            continue
        crossings = []
        for index in range(3):
            a, b = tri[index], tri[(index + 1) % 3]
            da, db = distances[index], distances[(index + 1) % 3]
            if da == 0.0:
                crossings.append((a[1], a[2]))
            if (da > 0.0) != (db > 0.0) and da != db:
                weight = da / (da - db)
                crossings.append((
                    a[1] + weight * (b[1] - a[1]),
                    a[2] + weight * (b[2] - a[2]),
                ))
        unique = []
        for point in crossings:
            if not any(abs(point[0] - other[0]) < 1e-9
                       and abs(point[1] - other[1]) < 1e-9
                       for other in unique):
                unique.append(point)
        if len(unique) == 2:
            segments.append(tuple(unique))
    return segments


def rear_profile(segments, y_lo: float, y_hi: float, samples: int = 900):
    """Rearmost material Z per Y station, restricted to the upright plate."""
    profile = []
    for index in range(samples + 1):
        y = y_lo + (y_hi - y_lo) * index / samples
        best = None
        for (y0, z0), (y1, z1) in segments:
            if y0 == y1:
                continue
            if not (min(y0, y1) - 1e-9 <= y <= max(y0, y1) + 1e-9):
                continue
            z = z0 + (y - y0) * (z1 - z0) / (y1 - y0)
            # Ignore the stand's own rearward surfaces; this curve is the
            # plate's rear face, which never runs behind the bend tangent.
            if z < -12.0 or z > THICKNESS_MM + 0.5:
                continue
            best = z if best is None else min(best, z)
        if best is not None:
            profile.append((y, best))
    return profile


def draw_section(axis, segments, color, label, linewidth=1.0,
                 z_window=None):
    """Draw section segments; ``z_window`` keeps the rear-face zoom free of
    the duct bores that cross these planes further forward."""
    first = True
    for (y0, z0), (y1, z1) in segments:
        if z_window is not None and (
                min(z0, z1) < z_window[0] or max(z0, z1) > z_window[1]):
            continue
        axis.plot([y0, y1], [z0, z1], color=color, lw=linewidth,
                  solid_capstyle="round",
                  label=label if first else None, zorder=2)
        first = False


def bend_centerline():
    controls = centerline_controls()
    return [cubic_point(controls, index / 200.0)[1:] for index in range(201)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True,
                        help="floor-state piece_bottom STL before the change")
    parser.add_argument("--new", type=Path, required=True,
                        help="floor-state piece_bottom STL after the change")
    parser.add_argument("--out", type=Path, required=True,
                        help="output PNG path")
    parser.add_argument("--section-x", type=float, default=SECTION_X_MM)
    parser.add_argument("--profile-x", type=float, default=PROFILE_X_MM)
    args = parser.parse_args()

    old_mesh = read_binary_stl(args.old)
    new_mesh = read_binary_stl(args.new)
    old_segments = section_segments(old_mesh, args.section_x)
    new_segments = section_segments(new_mesh, args.section_x)
    old_profile_segments = section_segments(old_mesh, args.profile_x)
    new_profile_segments = section_segments(new_mesh, args.profile_x)
    if not (old_segments and new_segments
            and old_profile_segments and new_profile_segments):
        raise SystemExit(
            f"the x={args.section_x:g}/{args.profile_x:g} planes miss one "
            "of the meshes")

    old_rear = rear_profile(old_profile_segments, 60.0, SEAM_A_Y + 3.0)
    new_rear = rear_profile(new_profile_segments, 60.0, SEAM_A_Y + 3.0)

    arc_y = FLOOR_RAMP_FULL_DEPTH_Y_MM
    aperture_y = L22_CUTOUT[1] - L22_CUTOUT[2] / 2.0

    fig, (ax_side, ax_zoom) = plt.subplots(
        2, 1, figsize=(12.6, 12.2),
        gridspec_kw={"height_ratios": [2.35, 1.0]})

    # -- whole side section ------------------------------------------------
    draw_section(ax_side, old_segments, OLD_COLOR,
                 "before: 18 mm smoothstep y=78..96")
    draw_section(ax_side, new_segments, NEW_COLOR,
                 "after: 43.85 mm quintic y=74.15..118")
    rear_window = (-3.4, 7.4)
    draw_section(ax_zoom, old_profile_segments, OLD_COLOR, None,
                 linewidth=1.1, z_window=rear_window)
    draw_section(ax_zoom, new_profile_segments, NEW_COLOR, None,
                 linewidth=1.1, z_window=rear_window)

    bend = bend_centerline()
    ax_side.plot([p[0] for p in bend], [p[1] for p in bend],
                 color="tab:orange", lw=1.1, ls="-.", zorder=3,
                 label=(f"Option-B stand arc centreline "
                        f"({BEND_REAR_SPAN_MM:g} x {BEND_RISE_MM:g} mm, "
                        f"Rmin 41)"))
    ax_side.axvline(arc_y, color="tab:orange", lw=1.0, ls="--", zorder=1)
    ax_side.axvline(FLOOR_RAMP_SLIM_Y_MM, color=NEW_COLOR, lw=1.0, ls="--",
                    zorder=1)
    ax_side.axvline(SEAM_A_Y, color="tab:green", lw=1.0, ls="--", zorder=1)
    ax_side.set_xlim(-4.0, 132.0)
    ax_side.set_ylim(-106.0, 28.0)
    ax_side.set_aspect("equal", adjustable="box")
    ax_side.set_xlabel("y  (up from the floor, mm)")
    ax_side.set_ylabel("z  (depth; front face at 18.3, rear datum 0)")
    ax_side.set_title(
        f"slim floor-stand piece_bottom -- side section at x = "
        f"{args.section_x:g} mm", fontsize=11)
    ax_side.grid(True, lw=0.3, alpha=0.4)
    ax_side.legend(loc="lower left", fontsize=8, framealpha=0.94)

    # -- rear-face zoom ----------------------------------------------------
    for y, text, color in (
            (arc_y,
             f"arc completes into the stand\ny={arc_y:.2f} -> full 18.3 mm",
             "tab:orange"),
            (RAMP_Y0, f"old ramp start\ny={RAMP_Y0:g}", OLD_COLOR),
            (RAMP_Y1, f"old ramp end (slim)\ny={RAMP_Y1:g}", OLD_COLOR),
            (aperture_y, f"D190 aperture edge\ny={aperture_y:.2f}",
             "tab:red"),
            (FLOOR_RAMP_SLIM_Y_MM,
             f"ramp starts, still slim\ny={FLOOR_RAMP_SLIM_Y_MM:g}"
             f"  (seam - {FLOOR_RAMP_SLIM_MARGIN_MM:g})", NEW_COLOR),
            (SEAM_A_Y, f"seam A / dovetail root\ny={SEAM_A_Y:g}",
             "tab:green")):
        ax_zoom.axvline(y, color=color, lw=0.9, ls="--", zorder=1,
                        alpha=0.85)

    ax_zoom.plot([p[0] for p in old_rear], [p[1] for p in old_rear],
                 color=OLD_COLOR, lw=2.6, alpha=0.75, zorder=4,
                 label="before: rear face measured off the mesh")
    ax_zoom.plot([p[0] for p in new_rear], [p[1] for p in new_rear],
                 color=NEW_COLOR, lw=2.6, alpha=0.9, zorder=4,
                 label="after: rear face measured off the mesh")
    law = [(y, REAR_MM * smootherstep01(
        (y - arc_y) / (FLOOR_RAMP_SLIM_Y_MM - arc_y)))
        for y in [arc_y + (FLOOR_RAMP_SLIM_Y_MM - arc_y) * i / 400.0
                  for i in range(401)]]
    ax_zoom.plot([p[0] for p in law], [p[1] for p in law],
                 color="k", lw=0.9, ls=":", zorder=5,
                 label="quintic smootherstep law (analytic)")
    ax_zoom.axhline(REAR_MM, color="0.3", lw=0.7, ls=":", zorder=1)
    ax_zoom.axhline(0.0, color="0.3", lw=0.7, ls=":", zorder=1)
    ax_zoom.annotate(f"slim rear plane z={REAR_MM:g}  (11.5 mm field)",
                     xy=(126.0, REAR_MM), ha="right", va="bottom",
                     fontsize=7.6, color="0.25")
    ax_zoom.annotate("full-depth rear plane z=0  (18.3 mm)",
                     xy=(126.0, 0.0), ha="right", va="bottom",
                     fontsize=7.6, color="0.25")
    for y, text, color, offset in (
            (arc_y, f"arc completes\ny={arc_y:.2f}", "tab:orange", 7.2),
            (RAMP_Y0, f"old start\ny={RAMP_Y0:g}", "0.35", 4.6),
            (RAMP_Y1, f"old end\ny={RAMP_Y1:g}", "0.35", 7.2),
            (aperture_y, f"D190 edge\ny={aperture_y:.2f}", "tab:red", 3.0),
            (FLOOR_RAMP_SLIM_Y_MM,
             f"new start (slim)\ny={FLOOR_RAMP_SLIM_Y_MM:g}", NEW_COLOR,
             4.6),
            (SEAM_A_Y, f"seam A\ny={SEAM_A_Y:g}", "tab:green", 7.2)):
        ax_zoom.annotate(text, xy=(y, offset), ha="center", va="bottom",
                         fontsize=7.2, color=color,
                         bbox={"fc": "white", "ec": color, "lw": 0.5,
                               "boxstyle": "round,pad=0.22", "alpha": 0.93})
    ax_zoom.annotate(
        "", xy=(FLOOR_RAMP_SLIM_Y_MM, -1.35), xytext=(arc_y, -1.35),
        arrowprops={"arrowstyle": "<->", "color": NEW_COLOR, "lw": 1.3})
    ax_zoom.annotate(
        f"one ramp, {FLOOR_RAMP_SLIM_Y_MM - arc_y:.2f} mm",
        xy=((arc_y + FLOOR_RAMP_SLIM_Y_MM) / 2.0, -1.9), ha="center",
        va="top", fontsize=8.4, color=NEW_COLOR)
    ax_zoom.annotate(
        "", xy=(RAMP_Y1, 8.15), xytext=(RAMP_Y0, 8.15),
        arrowprops={"arrowstyle": "<->", "color": "0.35", "lw": 1.1})
    ax_zoom.annotate(
        f"old ramp, {RAMP_Y1 - RAMP_Y0:.2f} mm",
        xy=((RAMP_Y0 + RAMP_Y1) / 2.0, 8.35), ha="center", va="bottom",
        fontsize=8.0, color="0.35")
    ax_zoom.annotate(
        "dovetail into the mids:\nslim in both states",
        xy=(SEAM_A_Y + 1.4, REAR_MM), xytext=(122.0, 3.4), ha="center",
        va="top", fontsize=7.2, color="tab:green",
        arrowprops={"arrowstyle": "->", "color": "tab:green", "lw": 0.8},
        bbox={"fc": "white", "ec": "tab:green", "lw": 0.5,
              "boxstyle": "round,pad=0.22", "alpha": 0.93})
    ax_zoom.set_xlim(68.0, 126.0)
    ax_zoom.set_ylim(-3.4, 9.6)
    ax_zoom.set_xlabel("y  (up from the floor, mm)")
    ax_zoom.set_ylabel("z  (rear face depth, mm)")
    ax_zoom.set_title(
        f"rear face, seam to stand -- the slim to full transition "
        f"(section at x = {args.profile_x:g} mm, the -66 dovetail axis)",
        fontsize=10.5)
    ax_zoom.grid(True, lw=0.3, alpha=0.4)
    ax_zoom.legend(loc="upper left", fontsize=8, framealpha=0.94)

    fig.suptitle(
        "V1L slim floor stand: piece_bottom rear-thickness transition, "
        "before vs after", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, dpi=170, metadata={
            "Title": "LX521_PROUD_R6P_floor_stand_v1l_bottom_rear_ramp",
            "Description": (
                "slim floor piece_bottom rear-thickness ramp; "
                "LX_STAND_FOOT=1; LX_ROUTING_PROFILE=proud; "
                f"section x={args.section_x:g} mm; "
                f"ramp {FLOOR_RAMP_FULL_DEPTH_Y_MM:.2f}.."
                f"{FLOOR_RAMP_SLIM_Y_MM:g} mm quintic"),
        })
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  section plane      x = {args.section_x:g} mm")
    print(f"  old ramp           y = {RAMP_Y0:g}..{RAMP_Y1:g} "
          f"({RAMP_Y1 - RAMP_Y0:g} mm, cubic smoothstep)")
    print(f"  new ramp           y = {arc_y:.2f}..{FLOOR_RAMP_SLIM_Y_MM:g} "
          f"({FLOOR_RAMP_SLIM_Y_MM - arc_y:.2f} mm, quintic smootherstep)")


if __name__ == "__main__":
    main()
