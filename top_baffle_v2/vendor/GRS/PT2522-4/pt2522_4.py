"""Parametric envelope model of the GRS PT2522-4 3-1/2" planar tweeter (4 ohm, open-back dipole).

GRS publishes no CAD model. Values come from the Parts Express spec sheet in this folder
(272-122--grs-pt2522-4-spec-sheet.pdf, "drawing") and from the product photos ("photo": front, rear and side
views; scaled from the 67.5 mm plate width and the 89.3 mm plate height). Photo values are estimates: confirm
them against a physical driver before release.

Construction (from the photos): a thin steel flange carries a pill-shaped grille housing on BOTH faces. The
"thin" face (grille only, screw heads) bulges 2.4 mm; the "thick" face carries the motor housing (6.0 mm), a black
terminal insulator block below the grille and two L-shaped solder lugs that stand 13.5 mm proud of the flange,
further than the housing. Six assembly screws on the flange have heads on both faces. The four corner-notch holes
(62 x 79) are the mounting holes.

Frame: x across the plate, y along it (terminals at -y), z normal to it. z = 0 is the thin-face side of the flange
(the face that seats on a front-mounting ledge); the thin face is +z (towards the listener when mounted this way)
and the thick face with the terminals is -z.

usage: python pt2522_4.py OUT_DIR    (writes PT2522-4.step, PT2522-4.stl, PT2522-4.json)
"""
import json
import sys
from pathlib import Path

from build123d import (Box, BuildPart, BuildSketch, Cylinder, Locations, Mode, Plane, RectangleRounded, extrude,
                       export_step, export_stl)

DRAWING = dict(
    plate_w=67.5, plate_h=89.3,                 # flange outline
    hole_d=3.5, hole_pitch=(62.0, 79.0),        # four corner-notch mounting holes
    corner_notch_r=5.5,
    grille_w=57.7, grille_h=65.5,               # grille housing footprint
    total_depth=9.4,                            # thin-face housing to thick-face housing
    lug_w=5.2,                                  # solder lug width
)
PHOTO = dict(
    plate_t=1.0,                                # flange thickness
    thin_bulge=2.4,                             # grille housing proud of the flange, thin face
    thick_bulge=6.0,                            # motor housing proud of the flange, thick face (2.4 + 1.0 + 6.0 = 9.4)
    housing_r=4.0,                              # housing corner radius in plan
    grille_center_y=2.35,                       # housing centre above the plate centre
    terminal_block=(40.0, 9.0, 3.0),            # insulator block w, h, proud of the flange (thick face)
    terminal_center_y=-35.35,
    lug_x=16.6, lug_h=5.0, lug_proud=13.5,      # L-lugs at x = +/-lug_x, standing lug_proud from the flange
    screw_head_d=5.0, screw_head_h=1.8,         # six flange screws, heads on both faces
    screws=((-23.0, 40.5), (23.0, 40.5), (-31.5, 0.0), (31.5, 0.0), (-20.5, -40.0), (20.5, -40.0)),
)


def build():
    d, p = DRAWING, PHOTO
    t = p['plate_t']
    with BuildPart() as tw:
        with BuildSketch():
            RectangleRounded(d['plate_w'], d['plate_h'], 2.0)
        extrude(amount=-t)
        with Locations(*[(sx * d['hole_pitch'][0] / 2, sy * d['hole_pitch'][1] / 2, -t / 2) for sx in (-1, 1) for sy in (-1, 1)]):
            Cylinder(d['hole_d'] / 2, t + 2, mode=Mode.SUBTRACT)
        for z0, h in ((0.0, p['thin_bulge']), (-t, -p['thick_bulge'])):     # housing on both faces, rounded in plan
            with BuildSketch(Plane.XY.offset(z0)):
                with Locations((0, p['grille_center_y'])):
                    RectangleRounded(d['grille_w'], d['grille_h'], p['housing_r'])
            extrude(amount=h)
        w, h, proud = p['terminal_block']
        with Locations((0, p['terminal_center_y'], -t - proud / 2)):
            Box(w, h, proud)
        with Locations(*[(sx * p['lug_x'], p['terminal_center_y'], -t - p['lug_proud'] / 2) for sx in (-1, 1)]):
            Box(d['lug_w'], p['lug_h'], p['lug_proud'])
        for z, dz in ((0.0, 1), (-t, -1)):                                  # screw heads on both faces
            with Locations(*[(x, y, z + dz * p['screw_head_h'] / 2) for x, y in p['screws']]):
                Cylinder(p['screw_head_d'] / 2, p['screw_head_h'])
    return tw.part


def facts():
    return dict(
        model='GRS PT2522-4', kind='planar tweeter, open back (dipole)', impedance_ohm=4, re_ohm=3.4, le_mH_10kHz=0.01,
        sd_cm2=20.2, spl_db_2p83v_1m=93, rms_power_w=20, usable_hz=[800, 20000], qms=4.60, qts=4.46,
        drawing=DRAWING, photo_estimates=PHOTO,
        frame='x across, y along (terminals at -y), z normal; z=0 is the thin-face side of the flange; thin face +z, '
              'thick face with the terminal lugs -z',
        source='vendor/GRS/PT2522-4/272-122--grs-pt2522-4-spec-sheet.pdf (Parts Express 272-122) and product photos')


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    part = build()
    export_step(part, str(out / 'PT2522-4.step')); export_stl(part, str(out / 'PT2522-4.stl'))
    (out / 'PT2522-4.json').write_text(json.dumps(facts(), indent=2) + '\n')
    bb = part.bounding_box()
    print('volume mm3', round(part.volume, 1), 'bbox', [round(v, 2) for v in (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)])
