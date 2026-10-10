"""Parametric envelope model of the Mundorf hifiAMT17D2.1 (4 ohm dipole AMT, no faceplate, free-standing).

Mundorf publishes no CAD model. Values come from the Mundorf datasheet in this folder (pdf_Mundorf_AMT17D2.1_1.pdf,
"drawing"), the dimensioned AMT17D2.1 drawing published by dealers (AMT17D2.1-drawing.jpg: depth 10, hole pitch
47 x 69) and product photos ("photo"). Photo values are estimates: confirm them against a physical driver before
release. The successor AMT17D2.2 (mundorf-amt17d2.2-specification.pdf) shares the ear plate and hole pattern but
is deeper (12) and wider behind the plate (60): it does NOT fit this envelope.

Construction: a flat black front plate (55 x 60, the slotted grille itself) with four ears top and bottom carrying
the D4.5 mounting holes; the AMT motor body sits behind it on the same footprint. Both faces radiate (dipole). The
leads are flying leads leaving one short edge of the body near a corner (photos of the D2.x series); there are no
terminals.

Frame: x across the plate, y along it (ears at +/-y), z normal to it. z = 0 is the front face of the plate; the
plate is -PLATE_T..0 and the body behind it down to -DEPTH. The flying leads leave the -y edge at x = LEAD_X.

usage: python amt17d2_1.py OUT_DIR    (writes AMT17D2.1.step, AMT17D2.1.stl, AMT17D2.1.json)
"""
import json
import sys
from pathlib import Path

from build123d import (BuildPart, BuildSketch, Circle, Cylinder, Locations, Mode, Plane, Rectangle, RectangleRounded,
                       SlotCenterToCenter, extrude, export_step, export_stl)

DRAWING = dict(
    plate_w=55.0, plate_h=60.0,                 # front plate (grille) outline
    overall_h=77.0,                             # over the ears
    hole_d=4.5, hole_pitch=(47.0, 69.0),        # four ear holes
    depth=10.0,                                 # front plate face to rear of the body (AMT17D2.1 drawing)
)
PHOTO = dict(
    plate_t=1.5,                                # front plate thickness (side view of the drawing)
    plate_corner_r=1.5,
    ear_w=8.0,                                  # ear width; the ear end is a half-circle about the hole
    body_w=55.0, body_h=60.0,                   # body behind the plate: taken as the plate footprint (conservative)
    lead_edge='-y', lead_x=14.0,                # flying leads leave this edge, just inboard of an ear
    lead_d=3.0, lead_stub=3.0,                  # grommet + leads, and how far they stand off the edge before bending
    slot_count=7, slot_w=4.2, slot_len=(30.0, 36.0),   # grille slots (cosmetic)
)


def build():
    d, p = DRAWING, PHOTO
    t = p['plate_t']
    hx, hy = d['hole_pitch'][0] / 2, d['hole_pitch'][1] / 2
    ear_len = d['overall_h'] / 2 - hy                    # hole centre to the ear end (= ear end radius)
    with BuildPart() as tw:
        with BuildSketch(Plane.XY.offset(-t)):
            RectangleRounded(d['plate_w'], d['plate_h'], p['plate_corner_r'])
            for sy in (-1, 1):                          # ears: a bar from the plate edge to the hole, rounded end
                with Locations(*[(sx * hx, sy * (d['plate_h'] / 2 + hy) / 2) for sx in (-1, 1)]):
                    Rectangle(p['ear_w'], hy - d['plate_h'] / 2 + 1.0)
                with Locations(*[(sx * hx, sy * hy) for sx in (-1, 1)]):
                    Circle(ear_len)
        extrude(amount=t)
        with Locations(*[(sx * hx, sy * hy, -t / 2) for sx in (-1, 1) for sy in (-1, 1)]):
            Cylinder(d['hole_d'] / 2, t + 2, mode=Mode.SUBTRACT)
        with BuildSketch(Plane.XY.offset(-d['depth'])):
            RectangleRounded(p['body_w'], p['body_h'], p['plate_corner_r'])
        extrude(amount=d['depth'] - t)
        # flying-lead grommet on the -y edge, mid-depth of the body
        with Locations((p['lead_x'], -p['body_h'] / 2 - p['lead_stub'] / 2, -(d['depth'] + t) / 2)):
            Cylinder(p['lead_d'] / 2, p['lead_stub'] + 0.5, rotation=(90, 0, 0))
        # cosmetic grille slots in the plate face (0.4 deep)
        with BuildSketch(Plane.XY):
            n = p['slot_count']; pitch = (d['plate_h'] - 14.0) / (n - 1)
            for i in range(n):
                L = p['slot_len'][i % 2]
                with Locations((0, -((n - 1) / 2) * pitch + i * pitch)):
                    SlotCenterToCenter(L - p['slot_w'], p['slot_w'])
        extrude(amount=-0.4, mode=Mode.SUBTRACT)
    return tw.part


def facts():
    return dict(
        model='Mundorf hifiAMT17D2.1', kind='AMT, open back (dipole), no faceplate', impedance_ohm=4, re_ohm=3.9,
        z0_ohm=4.2, zmin_ohm=3.9, spl_db_2p83v=91, power_w=60, fs_hz=2000, response_hz_minus6db=[1500, 35000],
        crossover_hz=dict(first_order=5000, second_order=4000), dispersion_h_15khz_deg=50, diaphragm_mm2=1260,
        magnet='neodymium', weight_g=160, drawing=DRAWING, photo_estimates=PHOTO,
        frame='x across, y along (ears at +/-y), z normal; z=0 is the front face of the plate, body behind to -depth; '
              'flying leads leave the -y edge at x=lead_x',
        source='vendor/Mundorf/AMT17D2.1/pdf_Mundorf_AMT17D2.1_1.pdf (doc.soundimports.nl), AMT17D2.1-drawing.jpg '
               '(dealer drawing: depth 10, pitch 47 x 69) and product photos')


if __name__ == '__main__':
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent)
    part = build()
    export_step(part, str(out / 'AMT17D2.1.step')); export_stl(part, str(out / 'AMT17D2.1.stl'))
    (out / 'AMT17D2.1.json').write_text(json.dumps(facts(), indent=2) + '\n')
    bb = part.bounding_box()
    print('volume mm3', round(part.volume, 1), 'bbox', [round(v, 2) for v in (bb.min.X, bb.min.Y, bb.min.Z, bb.max.X, bb.max.Y, bb.max.Z)])
