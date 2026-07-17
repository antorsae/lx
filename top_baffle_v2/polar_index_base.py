"""Polar-index base: a two-part printed turntable under the floor-
stand foot, giving repeatable-to-~1-deg azimuth steps for polar
measurements across sessions (the science-entry rig).

  BASE (on the stand/table, Ø216 x 4): center spigot O14, annular
    glide track R76..84, TWO flex arms at R60 (180 deg apart) with
    45-deg cone noses UP, and an engraved rim scale (5-deg ticks,
    10-deg majors, numerals every 30 deg).
  ROTOR (carries the speaker, 169 x 185 x 4): a 3-tall fence pockets
    the foot's exact floor footprint (plate 152.4 wide x 18.3 +
    150 foot blade; +0.8 clearance; 46-wide rear gap for the NL8
    plug, whose barrel bottom rides ~8 above the plate), center bore
    O14.4, and 72 cone sockets on the UNDERSIDE at R60 -- O3.8 at
    10-deg positions (firm majors), O3.0 at 5-deg (light minors).
    A pointer notch marks the front-center rim.

Rotation axis = the foot footprint center: (x=0, z=-65.85) in baffle
plan coordinates, i.e. 84.15 mm BEHIND the front baffle plane. Polars
therefore sweep the front plane on an 84.15 mm eccentric -- constant
and known, so correct source-mic distance per angle in post:
d(theta) = sqrt(d0^2 + r^2 - 2*d0*r*cos(theta)), r = 84.15. At 1.5 m
and +-60 deg the raw bias is ~0.25 dB / ~0.12 ms -- identical across
variants, so comparisons are unaffected even uncorrected.

Detents: two arms x 72 sockets = 5-deg steps, both arms engaged at
every step (even ring). Break-away ~0.4 N-m: firm under the ~3.2 kg
module, easy by hand. Print both parts flat, no supports; PLA+.

This measurement jig has no acoustic/front datum and is not a baffle piece.
Its functional orientation is therefore outside the baffle front-face-down
texture contract: X180 would put the spigot/noses or fence into the bed.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import subprocess
import struct
import sys

if __name__ == "__main__":
    import run_memory_guarded as memory_guard

    if not memory_guard.is_guarded_process():
        guard = Path(__file__).with_name("run_memory_guarded.py")
        raise SystemExit(subprocess.run(
            [sys.executable, str(guard), "--", sys.executable,
             str(Path(__file__).resolve()), *sys.argv[1:]],
            check=False).returncode)

from build123d import (Box, Cone, Cylinder, Plane, Pos, Rot, Text,
                       export_stl, extrude, mirror)

# shared numbers
SPIGOT_D = 14.0
SPIGOT_CLR = 0.4          # rotor bore = 14.4
DETENT_R = 60.0
ARM_L, ARM_W = 22.0, 5.0
NOSE_H = 0.8              # 45-deg cone noses / sockets
PLATE_T = 4.0

# foot floor footprint (from the b2_split foot: plate bottom edge
# 152.4 wide x 18.3 deep + foot blade 150 deep; shared floor plane)
FOOT_W = 152.4
FOOT_D = 168.3
POCKET_CLR = 0.8          # each side total (0.4/side)
FENCE_W, FENCE_H = 4.0, 3.0
NL8_GAP_W = 46.0          # rear-center fence gap for the NL8 plug

ROTOR_W = FOOT_W + POCKET_CLR + 2 * FENCE_W + 8.0    # 169.2
ROTOR_D = FOOT_D + POCKET_CLR + 2 * FENCE_W + 8.0    # 185.1
BASE_D = 216.0


def _validate_binary_stl(path: Path) -> None:
    with path.open("rb") as stream:
        header = stream.read(84)
    if len(header) != 84:
        raise RuntimeError(f"temporary polar STL is truncated: {path}")
    triangles = struct.unpack_from("<I", header, 80)[0]
    expected = 84 + 50 * triangles
    if triangles < 1 or path.stat().st_size != expected:
        raise RuntimeError(
            f"temporary polar STL invalid: triangles={triangles} "
            f"bytes={path.stat().st_size} expected={expected}")


def base_plate():
    part = Pos(0, 0, PLATE_T / 2.0) * Cylinder(BASE_D / 2.0, PLATE_T)
    # glide track ring + center spigot
    part += Pos(0, 0, PLATE_T + 0.6) * (
        Cylinder(84.0, 1.2) - Cylinder(76.0, 1.4))
    part += Pos(0, 0, PLATE_T + 2.5) * Cylinder(SPIGOT_D / 2.0, 5.0)
    # two flex arms: U-slot cut through, 1.5 under-gap so the arm can
    # deflect down while the base sits on a table; 45-deg cone nose UP
    for sgn in (1.0, -1.0):
        cx = sgn * DETENT_R
        # slot outline: three sides of the arm freed (root at -y side)
        slot = (Pos(cx, ARM_L / 2.0 - 2.0, PLATE_T / 2.0)
                * Box(ARM_W + 2.4, ARM_L + 2.4, PLATE_T + 2.0))
        keep = (Pos(cx, ARM_L / 2.0 - 2.0 - 1.2, PLATE_T / 2.0)
                * Box(ARM_W, ARM_L + 2.4, PLATE_T + 2.0))
        part -= (slot - keep)
        # under-gap: thin the arm to its top 2.5
        part -= (Pos(cx, ARM_L / 2.0 - 2.0 - 1.2, 0.75)
                 * Box(ARM_W + 0.02, ARM_L + 2.4, 1.5))
        # nose at the arm tip, ON the detent ring radius (45-deg cone)
        part += Pos(cx, 0.0, PLATE_T + NOSE_H / 2.0) * Cone(
            NOSE_H + 0.9, 0.9, NOSE_H)
    # rim scale: ticks every 5 (short) / 10 (long), numerals every 30
    for k in range(72):
        a = k * 5.0
        long_t = (k % 2 == 0)
        tick = Pos((92.0 + (0 if long_t else 4.0)), 0, PLATE_T - 0.25) \
            * Box(12.0 if long_t else 8.0, 1.0, 0.5)
        part -= Rot(Z=a) * tick
    for k in range(12):
        txt = Text(f"{k * 30}", font_size=7.0)
        part -= (Rot(Z=-k * 30.0) * Pos(0, 104.0, PLATE_T - 0.5)
                 * extrude(Plane.XY * txt, amount=0.7))
    return part


def rotor_plate():
    part = Pos(0, 0, PLATE_T / 2.0) * Box(ROTOR_W, ROTOR_D, PLATE_T)
    # fence (walls around the foot pocket), rear gap for the NL8 plug
    pw, pd = FOOT_W + POCKET_CLR, FOOT_D + POCKET_CLR
    fence = (Pos(0, 0, PLATE_T + FENCE_H / 2.0)
             * Box(pw + 2 * FENCE_W, pd + 2 * FENCE_W, FENCE_H))
    fence -= Pos(0, 0, PLATE_T + FENCE_H / 2.0) * Box(pw, pd, FENCE_H + 0.2)
    fence -= (Pos(0, -(pd / 2.0 + FENCE_W / 2.0), PLATE_T + FENCE_H / 2.0)
              * Box(NL8_GAP_W, FENCE_W + 0.4, FENCE_H + 0.2))
    part += fence
    # center bore riding the spigot
    part -= Pos(0, 0, PLATE_T / 2.0) * Cylinder(
        (SPIGOT_D + SPIGOT_CLR) / 2.0, PLATE_T + 0.2)
    # 72 detent cone sockets on the UNDERSIDE (O3.8 majors at 10 deg,
    # O3.0 minors at 5 deg; 45-deg cones print as clean hole ceilings)
    for k in range(72):
        a = math.radians(k * 5.0)
        r_mouth = 1.9 if k % 2 == 0 else 1.5
        sx, sy = DETENT_R * math.cos(a), DETENT_R * math.sin(a)
        # 45-deg TRUNCATED cone entering 0.5 below the face: a base
        # coplanar with the surface leaves sliver mouth rings, and a
        # sharp apex meshes into degenerate zero-length edges -- both
        # read as STL defects
        h = r_mouth + 0.5 - 0.25
        part -= Pos(sx, sy, -0.5 + h / 2.0) * Cone(h + 0.25, 0.25, h)
    # pointer notch, front-center rim (reads against the base scale)
    part -= (Pos(0, ROTOR_D / 2.0, PLATE_T / 2.0)
             * Rot(Z=45.0) * Box(6.0, 6.0, PLATE_T + 0.2))
    return part


def main():
    out_dir = Path(__file__).parent / "floor_stand" / "stl"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, solid in (("lx521_polar_base_1of2_base", base_plate()),
                        ("lx521_polar_base_2of2_rotor", rotor_plate())):
        path = out_dir / f"{name}.stl"
        temporary = path.with_name(
            f".{path.stem}.{os.getpid()}.tmp.stl")
        try:
            export_stl(solid, str(temporary), tolerance=0.05,
                       angular_tolerance=0.2)
            _validate_binary_stl(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
