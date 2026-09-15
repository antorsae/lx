"""Current LM pin/socket qualification coupon; millimetres.

Crop both native seam ends from the promoted STEP at their actual pitch.
Connect each pair with a 2 mm sacrificial brace on the acoustic-front side,
8 mm away from the seam. Braces are test fixtures, not a structural model.
The production pin/socket surfaces are unmodified. Print front-face down.
"""
from pathlib import Path
import os
import json
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
from build123d import Box, Compound, Pos, Rot, import_step, export_step, export_stl
from lx521_baffle.obiwan.lm_split import LM_SPLIT_SEAM_Y, registration_fit_facts


def gen_step():
    assembly = import_step(str(ROOT / 'build/no_floor_stand/obiwan_lm_split.step'))
    pieces = []
    for index, child in enumerate(assembly.children):
        bottom = 'bottom' in child.label
        cy = LM_SPLIT_SEAM_Y + (-2.75 if bottom else 4.0)
        crop = Pos(0, cy, 9.15) * Box(240, 10.5 if bottom else 8.0, 18.3)
        ends = child & crop
        # Full-width brace ties the two native ends at the exact source pitch.
        brace_y = LM_SPLIT_SEAM_Y + (-7.0 if bottom else 7.0)
        brace = Pos(0, brace_y, 17.3) * Box(218, 2.0, 2.0)
        joined = ends.fuse(brace)
        if not joined.is_valid or len(joined.solids()) != 1:
            raise RuntimeError('coupon brace failed to join the native seam ends')
        source_name = 'obiwan_optional_lm_keyed_1_of_2_bottom' if bottom else 'obiwan_optional_lm_keyed_2_of_2_top'
        authority = json.loads((ROOT / 'build/no_floor_stand/stl' / (source_name + '.print.json')).read_text())
        laid = Rot(Z=authority['rotation_deg']['z']) * Rot(X=180) * joined
        bb = laid.bounding_box()
        laid = Pos(index * 240 - bb.min.X, -bb.min.Y, -bb.min.Z) * laid
        laid.label = 'male_pin_pair' if bottom else 'female_socket_pair'
        pieces.append(laid)
    return Compound(children=pieces, label='LM registration qualification fixture')


if __name__ == '__main__':
    import hashlib, json
    part = gen_step()
    out = Path(__file__).parent
    export_step(part, str(out / 'lm_registration.step'))
    for child in part.children:
        # Each separate STL starts at the bed origin; STEP offsets are review-only.
        bb = child.bounding_box()
        printable = Pos(-bb.min.X, -bb.min.Y, -bb.min.Z) * child
        export_stl(printable, str(out / (child.label + '.stl')), tolerance=0.005, angular_tolerance=0.08)
    facts = registration_fit_facts()
    facts['print_rotations_z_deg'] = {label: json.loads((ROOT/'build/no_floor_stand/stl'/filename).read_text())['rotation_deg']['z'] for label, filename in [('male_pin_pair', 'obiwan_optional_lm_keyed_1_of_2_bottom.print.json'), ('female_socket_pair', 'obiwan_optional_lm_keyed_2_of_2_top.print.json')]}
    facts['purpose'] = 'Fit/path coupon only; sacrificial braces change stiffness. Full LM assembly/load tests remain mandatory.'
    facts['source_step_sha256'] = hashlib.sha256((ROOT/'build/no_floor_stand/obiwan_lm_split.step').read_bytes()).hexdigest()
    facts['files'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'male_pin_pair.stl', out/'female_socket_pair.stl')}
    facts['solids'] = [{'name': c.label, 'valid': c.is_valid, 'volume_mm3': c.volume,
                       'size_mm': list(c.bounding_box().size)} for c in part.children]
    (out/'lm_registration.json').write_text(json.dumps(facts, indent=2)+'\n')
    print('Exported two valid, positive-volume registration fixture halves')
