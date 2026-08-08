#!/usr/bin/env python3
"""Review of the slim floor bottom's rear-thickness ramp through the stand.

The floor-state ``piece_bottom`` used to reach its full 18.3 mm section at
the Option-B vertical tangent, so the plate was already back to full depth at
the station where the stand arc has only started to turn.  The ramp now runs
on ONE quintic in PATH LENGTH along the whole combined profile -- slim 2 mm
below the seam-A dovetail root, down the flat plate, and on along the bend
centreline as it sweeps -- reaching full depth exactly at the HORIZONTAL
tangent, where the arc has finished turning and the foot begins.

Carrying the ramp through the bend thins the stand's concave face, which is
the face the three cable lanes hug around mid-arc, so those lanes are
rerouted convex-ward as quintics.  Their covers are the real cost of the
change and are drawn here beside the law.

Panels:
  1. side section through the whole stand, before and after, off the meshes;
  2. wall thickness against path length s -- the law, the measured plate and
     bend sections, and both tangents;
  3. concave/convex cover for each rerouted lane, with the cover forced at
     each lane's FIXED plate-side join called out.
"""

from __future__ import annotations

import argparse
import json
import math
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
from lx521_baffle.cables import (  # noqa: E402
    FOOT_LANES,
    TS_ROUTE_CAPTIVE,
    UM_V1L_HANDOFF_KEY,
    proud_floor_entry_controls,
)
from lx521_baffle.floor_bend import (  # noqa: E402
    BEND_CENTERLINE_LENGTH_MM,
    BEND_REAR_SPAN_MM,
    BEND_RISE_MM,
    WALL_HALF_THICKNESS_MM,
    bezier_point,
    centerline_controls,
    cubic_derivatives,
    cubic_point,
)
from lx521_baffle.proud.v1l import RAMP_Y0, RAMP_Y1  # noqa: E402
from lx521_baffle.proud.v1l_split import (  # noqa: E402
    FLOOR_RAMP_FLAT_LENGTH_MM,
    FLOOR_RAMP_SLIM_MARGIN_MM,
    FLOOR_RAMP_SLIM_Y_MM,
    FLOOR_RAMP_TOTAL_LENGTH_MM,
    FLOOR_RAMP_VERTICAL_TANGENT_Y_MM,
    SEAM_A_Y,
    floor_ramp_rear_cut_mm,
    floor_ramp_thickness_mm,
    floor_ramp_wall_thickness_law,
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
LANE_COLORS = {"lm": "tab:green", "um": "tab:purple", "ts": "tab:orange"}
DUCT_SKIN_RULE_MM = 1.6


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


def _ray_hit(segments, origin, direction, limit: float = 30.0):
    """Nearest crossing distance from ``origin`` along ``direction``."""
    ox, oy = origin
    dx, dy = direction
    best = None
    for (x0, y0), (x1, y1) in segments:
        ex, ey = x1 - x0, y1 - y0
        denominator = dx * ey - dy * ex
        if abs(denominator) < 1e-12:
            continue
        t = ((x0 - ox) * ey - (y0 - oy) * ex) / denominator
        u = ((x0 - ox) * dy - (y0 - oy) * dx) / denominator
        if t <= 1e-7 or t > limit or not -1e-9 <= u <= 1.0 + 1e-9:
            continue
        best = t if best is None else min(best, t)
    return best


def measured_bend_thickness(segments, samples: int = 160):
    """Wall thickness normal to the arc, read straight off the mesh section."""
    controls = centerline_controls()
    measured = []
    for index in range(samples + 1):
        parameter = index / samples
        point = cubic_point(controls, parameter)
        first, _second = cubic_derivatives(controls, parameter)
        norm = math.hypot(first[1], first[2])
        normal = (-first[2] / norm, first[1] / norm)
        origin = (point[1], point[2])
        outward = _ray_hit(segments, origin, normal)
        inward = _ray_hit(segments, origin, (-normal[0], -normal[1]))
        if outward is None or inward is None:
            continue
        travelled = BEND_CENTERLINE_LENGTH_MM - _arc_length(parameter)
        measured.append(
            (FLOOR_RAMP_FLAT_LENGTH_MM + travelled, outward + inward))
    return measured


def _arc_length(parameter: float) -> float:
    from lx521_baffle.floor_bend import centerline_arc_length

    return centerline_arc_length(parameter)


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


def lane_covers(samples: int = 1200):
    """Concave/convex cover against path length for each rerouted lane."""
    controls = centerline_controls()
    horizontal_z = controls[0][2]
    stations = []
    for index in range(samples + 1):
        parameter = index / samples
        point = cubic_point(controls, parameter)
        first, _second = cubic_derivatives(controls, parameter)
        norm = math.hypot(first[1], first[2])
        normal = (-first[2] / norm, first[1] / norm)
        thickness = floor_ramp_wall_thickness_law(parameter)
        stations.append((point[1], point[2], normal, thickness,
                         FLOOR_RAMP_FLAT_LENGTH_MM
                         + BEND_CENTERLINE_LENGTH_MM - _arc_length(parameter)))
    results = {}
    for name, lane in FOOT_LANES.items():
        lane_controls = proud_floor_entry_controls(
            name, um_handoff_key=UM_V1L_HANDOFF_KEY,
            ts_route_key=TS_ROUTE_CAPTIVE)
        radius = lane[4] / 2.0
        series = []
        for index in range(samples + 1):
            px, py, pz = bezier_point(lane_controls, index / samples)
            if pz < horizontal_z:
                continue
            if py > FLOOR_RAMP_VERTICAL_TANGENT_Y_MM:
                # Above the tangent the lane is in the ramped plate, whose
                # rear face is the same law read at that station.
                station = FLOOR_RAMP_SLIM_Y_MM - py
                series.append((
                    station,
                    pz - radius - floor_ramp_rear_cut_mm(max(0.0, station)),
                    THICKNESS_MM - pz - radius,
                ))
                continue
            best = None
            for cy, cz, normal, thickness, station in stations:
                distance = math.hypot(py - cy, pz - cz)
                if best is None or distance < best[0]:
                    best = (distance, cy, cz, normal, thickness, station)
            _d, cy, cz, normal, thickness, station = best
            offset = (py - cy) * normal[0] + (pz - cz) * normal[1]
            series.append((
                station,
                offset - radius - (WALL_HALF_THICKNESS_MM - thickness),
                WALL_HALF_THICKNESS_MM - (offset + radius),
            ))
        series.sort()
        join = lane_controls[-1]
        results[name] = {
            "series": series,
            "join_y": join[1],
            "join_s": FLOOR_RAMP_SLIM_Y_MM - join[1],
            "join_cover": join[2] - radius - floor_ramp_rear_cut_mm(
                FLOOR_RAMP_SLIM_Y_MM - join[1]),
            "diameter": lane[4],
        }
    return results


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
    measured_bend = measured_bend_thickness(new_segments)

    arc_y = FLOOR_RAMP_VERTICAL_TANGENT_Y_MM
    aperture_y = L22_CUTOUT[1] - L22_CUTOUT[2] / 2.0
    total = FLOOR_RAMP_TOTAL_LENGTH_MM

    fig, (ax_side, ax_law, ax_cover) = plt.subplots(
        3, 1, figsize=(13.0, 16.4),
        gridspec_kw={"height_ratios": [2.0, 1.15, 1.15]})

    # -- 1. whole side section --------------------------------------------
    draw_section(ax_side, old_segments, OLD_COLOR,
                 "before: full depth at the vertical tangent")
    draw_section(ax_side, new_segments, NEW_COLOR,
                 f"after: one quintic over s=0..{total:.2f} mm of path")
    bend = bend_centerline()
    ax_side.plot([p[0] for p in bend], [p[1] for p in bend],
                 color="tab:orange", lw=1.1, ls="-.", zorder=3,
                 label=(f"Option-B stand arc centreline "
                        f"({BEND_REAR_SPAN_MM:g} x {BEND_RISE_MM:g} mm, "
                        f"Rmin 41, arc {BEND_CENTERLINE_LENGTH_MM:.2f} mm)"))
    ax_side.axvline(arc_y, color="tab:orange", lw=1.0, ls="--", zorder=1)
    ax_side.axvline(FLOOR_RAMP_SLIM_Y_MM, color=NEW_COLOR, lw=1.0, ls="--",
                    zorder=1)
    ax_side.axvline(SEAM_A_Y, color="tab:green", lw=1.0, ls="--", zorder=1)
    controls = centerline_controls()
    ax_side.plot([controls[0][1]], [controls[0][2]], marker="o", ms=6,
                 color="k", zorder=6)
    ax_side.annotate(
        f"horizontal tangent: full 18.3 mm\ny={controls[0][1]:g}, "
        f"z={controls[0][2]:g}  (s={total:.2f} mm)",
        xy=(controls[0][1], controls[0][2]),
        xytext=(24.0, -86.0), fontsize=8.0,
        arrowprops={"arrowstyle": "->", "color": "k", "lw": 0.9},
        bbox={"fc": "white", "ec": "k", "lw": 0.6,
              "boxstyle": "round,pad=0.25", "alpha": 0.94})
    ax_side.annotate(
        f"vertical tangent\ny={arc_y:g}  (s={FLOOR_RAMP_FLAT_LENGTH_MM:g} mm)"
        f"\nwall {floor_ramp_thickness_mm(FLOOR_RAMP_FLAT_LENGTH_MM):.3f} mm",
        xy=(arc_y + 1.2, -34.0), fontsize=8.0, color="tab:orange")
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

    # -- 2. the law against path length ------------------------------------
    law_s = [total * index / 600.0 for index in range(601)]
    ax_law.plot(law_s, [floor_ramp_thickness_mm(s) for s in law_s],
                color=NEW_COLOR, lw=2.4,
                label="one quintic smootherstep in path length (analytic)")
    plate_band = [
        (FLOOR_RAMP_SLIM_Y_MM - y, THICKNESS_MM - z) for y, z in new_rear
        if FLOOR_RAMP_VERTICAL_TANGENT_Y_MM <= y <= FLOOR_RAMP_SLIM_Y_MM]
    ax_law.plot([p[0] for p in plate_band], [p[1] for p in plate_band],
                color="k", lw=0.0, marker=".", ms=2.4, zorder=5,
                label="measured plate section (mesh, x = -66)")
    if measured_bend:
        ax_law.plot([p[0] for p in measured_bend],
                    [p[1] for p in measured_bend],
                    color="tab:red", lw=0.0, marker=".", ms=2.4, zorder=5,
                    label="measured bend section normal to the arc (mesh)")
    old_band = [
        (FLOOR_RAMP_SLIM_Y_MM - y, THICKNESS_MM - z) for y, z in old_rear
        if FLOOR_RAMP_VERTICAL_TANGENT_Y_MM <= y <= FLOOR_RAMP_SLIM_Y_MM]
    ax_law.plot([p[0] for p in old_band], [p[1] for p in old_band],
                color=OLD_COLOR, lw=1.6, ls="--", zorder=4,
                label="before: plate section (mesh)")
    ax_law.axvline(FLOOR_RAMP_FLAT_LENGTH_MM, color="tab:orange", lw=1.1,
                   ls="--")
    ax_law.axvline(total, color="k", lw=1.1, ls="--")
    ax_law.axhline(THICKNESS_MM, color="0.35", lw=0.7, ls=":")
    ax_law.axhline(11.5, color="0.35", lw=0.7, ls=":")
    ax_law.annotate(
        f"vertical tangent\ns={FLOOR_RAMP_FLAT_LENGTH_MM:g} mm "
        f"(flat plate run)\nwall "
        f"{floor_ramp_thickness_mm(FLOOR_RAMP_FLAT_LENGTH_MM):.3f} mm",
        xy=(FLOOR_RAMP_FLAT_LENGTH_MM, 12.0),
        xytext=(FLOOR_RAMP_FLAT_LENGTH_MM + 6.0, 12.3), fontsize=8.0,
        color="tab:orange",
        arrowprops={"arrowstyle": "->", "color": "tab:orange", "lw": 0.9},
        bbox={"fc": "white", "ec": "tab:orange", "lw": 0.6,
              "boxstyle": "round,pad=0.22", "alpha": 0.94})
    ax_law.annotate(
        f"horizontal tangent\ns={total:.2f} mm  (total path)\nfull 18.3 mm,"
        " zero slope",
        xy=(total, 17.4), xytext=(total - 46.0, 15.5), fontsize=8.0,
        arrowprops={"arrowstyle": "->", "color": "k", "lw": 0.9},
        bbox={"fc": "white", "ec": "k", "lw": 0.6,
              "boxstyle": "round,pad=0.22", "alpha": 0.94})
    ax_law.annotate(
        f"flat plate {FLOOR_RAMP_FLAT_LENGTH_MM:g} mm", xy=(20.0, 11.15),
        ha="center", fontsize=8.0, color="0.3")
    ax_law.annotate(
        f"bend sweep {BEND_CENTERLINE_LENGTH_MM:.2f} mm",
        xy=(101.0, 11.15), ha="center", fontsize=8.0, color="0.3")
    ax_law.set_xlim(-2.0, total + 2.0)
    ax_law.set_ylim(11.0, 18.9)
    ax_law.set_xlabel(
        "s  (path length from the slim field at y=118, down the plate, then "
        "along the bend centreline, mm)")
    ax_law.set_ylabel("wall thickness normal to\nthe swept surface (mm)")
    ax_law.set_title(
        "thickness against path length -- one law, no knee at the vertical "
        "tangent", fontsize=10.5)
    ax_law.grid(True, lw=0.3, alpha=0.4)
    ax_law.legend(loc="lower right", fontsize=8, framealpha=0.94)

    # -- 3. what the rerouted lanes keep -----------------------------------
    covers = lane_covers()
    for name, record in covers.items():
        color = LANE_COLORS[name]
        series = record["series"]
        ax_cover.plot([p[0] for p in series], [p[1] for p in series],
                      color=color, lw=1.9,
                      label=f"{name.upper()} D{record['diameter']:g} concave "
                            f"(rear/inner) cover")
        ax_cover.plot([p[0] for p in series], [p[2] for p in series],
                      color=color, lw=1.0, ls=":", alpha=0.8,
                      label=f"{name.upper()} convex (front/outer) cover")
        ax_cover.plot([record["join_s"]], [record["join_cover"]],
                      marker="v", ms=8, color=color, zorder=6)
    ax_cover.axhline(DUCT_SKIN_RULE_MM, color="k", lw=1.4, ls="--",
                     label=f"slim duct-skin rule {DUCT_SKIN_RULE_MM} mm")
    ax_cover.axvline(FLOOR_RAMP_FLAT_LENGTH_MM, color="tab:orange", lw=1.0,
                     ls="--")
    join_text = "  ".join(
        f"{name.upper()} {record['join_cover']:.3f}"
        for name, record in covers.items())
    ax_cover.annotate(
        "cover forced at each lane's FIXED plate-side join (y=84.7 / 82.0 / "
        f"90.0):\n{join_text} mm -- no reroute can beat these",
        xy=(6.0, 7.4), fontsize=8.2,
        bbox={"fc": "white", "ec": "0.4", "lw": 0.6,
              "boxstyle": "round,pad=0.3", "alpha": 0.95})
    ax_cover.set_xlim(-2.0, total + 2.0)
    ax_cover.set_ylim(0.0, 9.4)
    ax_cover.set_xlabel("s  (path length, mm)")
    ax_cover.set_ylabel("cover to the wall face (mm)")
    ax_cover.set_title(
        "rerouted floor lanes through the ramped bend -- concave cover is "
        "the cost of the full ramp", fontsize=10.5)
    ax_cover.grid(True, lw=0.3, alpha=0.4)
    ax_cover.legend(loc="upper right", fontsize=7.4, framealpha=0.94, ncol=2)

    fig.suptitle(
        "V1L slim floor stand: piece_bottom rear-thickness ramp carried "
        "through the Option-B bend", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.stem}.{os.getpid()}.tmp.png")
    try:
        fig.savefig(temporary, dpi=170, metadata={
            "Title": "LX521_PROUD_R6P_floor_stand_v1l_bottom_rear_ramp",
            "Description": (
                "slim floor piece_bottom rear-thickness ramp; "
                "LX_STAND_FOOT=1; LX_ROUTING_PROFILE=proud; "
                f"section x={args.section_x:g} mm; one quintic over "
                f"s=0..{total:.3f} mm reaching full depth at the horizontal "
                "tangent; floor lanes rerouted convex-ward"),
        })
        temporary.replace(args.out)
    finally:
        temporary.unlink(missing_ok=True)
    plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  section plane      x = {args.section_x:g} mm")
    print(f"  old ramp           y = {RAMP_Y0:g}..{RAMP_Y1:g} "
          f"(no-floor state, cubic smoothstep)")
    print(f"  path-length ramp   s = 0..{total:.3f} mm "
          f"({FLOOR_RAMP_FLAT_LENGTH_MM:g} mm plate + "
          f"{BEND_CENTERLINE_LENGTH_MM:.3f} mm arc), slim margin "
          f"{FLOOR_RAMP_SLIM_MARGIN_MM:g} mm, rear cut at the tangent "
          f"{floor_ramp_rear_cut_mm(FLOOR_RAMP_FLAT_LENGTH_MM):.4f} mm")
    for name, record in covers.items():
        concave = min(p[1] for p in record["series"])
        convex = min(p[2] for p in record["series"])
        print(f"  {name} lane          concave {concave:.4f} / convex "
              f"{convex:.4f} mm, join cover {record['join_cover']:.4f} mm")


if __name__ == "__main__":
    main()
