#!/usr/bin/env python3
"""Generate the H2C file guide from its live job manifest and print policies."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lx521_baffle.h2c.printing import policy


def link(label,path,start):return f'[{label}]({os.path.relpath(path,start)})'


def main():
    out=ROOT/'to_print/h2c';catalog=json.loads((out/'catalog.json').read_text());cfg=policy()
    jobs=catalog['jobs'];passed=sum(j['status']=='sliced_audited' for j in jobs)
    lines=['# H2C print files','',f'{passed} of {len(jobs)} jobs digitally audited. Printer: **H2C, two 0.6 mm High Flow nozzles**.',
        '', 'The pre-migration project is preserved in commit `c261f32`. These projects use native H2C machine programs. '+
        'The P2S files are a separate earlier release.', '',
        '**Choose one family, one stand state and one tweeter/wing alternative per speaker.** '+
        'Print twice for a stereo pair. The V4 accessories plate already contains two caps and two retainers.', '',
        '| Configuration | H2C main pieces per speaker | Previous P2S arrangement |',
        '|---|---|---|',
        '| Stock / Slim, no floor stand | 1 LM + 1 upper module | 3 LM + 1 upper |',
        '| Stock / Slim, floor stand | 1 lower LM with stand + 1 upper LM + 1 upper module | 3 LM + 1 upper |',
        '| Regular Obiwan, either stand state | 1 LM + 1 UM + 1 crescent + 2 optional wings | 2 LM + 1 UM + 1 crescent + 4 wing sections |',
        '| V4 Obiwan, either stand state | 1 LM + 1 fused body + 2 optional wings | 2 LM + 1 fused body + 4 wing sections |',
        '', 'Stock/Slim shoulders or side wings remain optional. The upper module and BMR alternative are shared between stand states. '+
        'A-style shoulders remain four pieces: their upper and lower outlines meet only at a tangent point, so joining them would require an outline change. '+
        'The B1 alternative already uses one continuous wing per side. '+
        'Use the new H2C upper module with the new contained-pin LM joint; its two Ø3 mm pins replace the old upward dovetails. '+
        'The hidden M3 × 20 clamp and Hanglife M3, Ø5 × 4 mm insert remain. Floor-stand LM halves retain the established lower dovetail joint.', '',
        'The unchanged one-piece Stock/Slim floor-stand trials failed the actual PLA-nozzle reach check. '+
        'They are retained under `build/h2c/experiments/rejected_one_piece_floor/`, outside this shelf. '+
        'The current floor-stand pair removes the old vertical LM seam and keeps the original outline.', '',
        '## Materials and support', '',
        '- **Model: left nozzle. PLA support interface: right nozzle.** Configure the corresponding feeds on the H2C. '+
        'The previous P2S AMS slot numbers are not nozzle assignments.',
        '- Default lane: Tinmorry PETG-GF + PLA. V4 also has its own PETG Translucent + PLA Translucent lane.',
        '- Engineering Plate with glue, 70 °C for both materials, no raft.',
        '- 0.16 mm layers, 0.20 mm first layer, six walls. Structural LM/UM: 100% infill. Wings: 10% gyroid. '+
        'V4 tweeter modifier: 15% gyroid; caps/retainers: 15% gyroid. Regular crescent: its existing 30% policy.',
        '- PETG support bases with three dense, zero-gap PLA interface layers above and below. '+
        'Caps have PLA directly beneath their ceilings; retainers print without supports.',
        '- Stock/Slim complete no-floor LM uses a 2 mm brim and no support; other parts use a 5 mm brim. '+
        'The narrow brim needs an adhesion trial on the actual H2C plate.',
        '- Dedicated nozzles retain native H2C priming, retraction and standby programs. '+
        'The P2S same-nozzle purge-volume workaround is not transferred to every H2C switch. Flushing into model, infill and supports is disabled.',
        '- V4 translucent body keeps Classic outer-first walls to reduce changes in surface texture around buried magnets.',
        '', '## Magnets and hardware', '',
        'Regular interfaces retain Ø5 × 2 mm N52 magnets. V4 UM/wing contacts retain Ø6 × 3 mm N45; '+
        'V4 wings use the regular Ø5 × 2 mm magnets at their two LM contacts. A full regular wing has three magnets; '+
        'a full V4 wing has four. All remain buried. Magnet pauses are recalculated from actual H2C extrusion and embedded in the projects.', '',
        'Retain existing driver, LM/UM, stand and wing hardware. V4 uses the project’s M3/Ø4.6 mm printed-bore convention '+
        'for Hanglife M3 inserts, Ø5 mm outside × 4 mm long. The optional Tectonic BMR mounts retain their documented M2 hardware exception.', '',
        '## File list', '',
        '`.3mf` is the editable project with its geometry, settings and measured magnet pauses. '+
        '`.gcode.3mf` contains the audited slice. STLs carry geometry only; importing an STL alone loses supports, infill modifiers and insertion pauses.', '',
        '| Family / part | Material lane | Infill | Magnets | Files | Status |','|---|---|---|---:|---|---|']
    for job in jobs:
        project=ROOT/job['project'];work=ROOT/job['work']
        process=json.loads((work/'process.json').read_text())
        meshes=list(dict.fromkeys(s['path'] for s in job['preparation']['sources'] if s['subtype']=='normal_part'))
        files=[link('Project',project,out)]
        files += [link('STL' if len(meshes)==1 else Path(p).stem,ROOT/p,out) for p in meshes]
        if job.get('sliced_project'):files.insert(1,link('Sliced',ROOT/job['sliced_project'],out))
        if job.get('audit'):files.append(link('Audit',ROOT/job['audit'],out))
        status='Digital pass; physical trial pending' if job['status']=='sliced_audited' else job['status'].replace('_',' ')
        name=job['name'].removeprefix('h2c_')+(' (candidate)' if job.get('candidate') else '')
        density=process['sparse_infill_density']+(' + 15% tweeter region' if job['role']=='crescent_body' else '')
        lines.append(f"| {name} | {job['lane']} | {density} | {job['magnet_count']} | {' · '.join(files)} | {status} |")
    lines+=['','## Policy exceptions','']
    lines += ['- '+entry for entry in cfg['exceptions']]
    lines+=['','## Geometry review','',
        link('Stock/Slim orthographic assembly sheet',ROOT/'build/h2c/views/primary_review_contact_sheet.png',out)+' · '+
        link('V4 with continuous wings — front',ROOT/'build/h2c/views/H2C_V4_graded_front.png',out)+' · '+
        link('V4 with continuous wings — rear',ROOT/'build/h2c/views/H2C_V4_flat_rear_oblique.png',out), '',
        'The views use the actual exported parts in installed coordinates. Drivers and service caps are omitted. '+
        'Native STEP assemblies and individual parts are in `../../build/h2c/review_models/` and `../../build/h2c/STEP/` '+
        'from this guide. The installed CAD Viewer launcher was unavailable; these PNGs provide the checked visual review.', '',
        link('Native Stock/Slim joint checks',ROOT/'build/h2c/geometry_validation.json',out)+' · '+
        link('Continuous Obiwan LM / regular UM / V4 interface checks',ROOT/'build/h2c/obiwan_interface_validation.json',out),
        '', '## Verification and limits','',
        'The audit checks sliced geometry and modifier identity, machine/material settings, actual nozzle assignments, '+
        'deposition bounds including bead widths and arc extrema, support clearance, cap-ceiling coverage, '+
        'captive-wall extrusion and insertion timing, plus static G-code validation. '+
        'Native CAD checks cover the new Stock/Slim pin joint, solid interference and preserved driver openings.', '',
        'Passing those checks does not establish physical adhesion, magnet retention, loaded stand strength or acoustic performance. '+
        'The first H2C build still needs the project’s hardware/material qualification. BMR and V4 retain their candidate status.', '',
        '## Rebuild','', '```sh','make                    # H2C geometry, projects, slicing, audit and generated guide',
        'make h2c_prepare        # prepare editable projects from existing authorities',
        'make h2c_validate       # slice/audit prepared projects; reuse hash-matched outputs',
        'make h2c_review         # regenerate native assembly and mesh review images',
        'make h2c_docs           # regenerate this guide',
        'make PRINTER=P2S all    # earlier printer pipeline','```','',
        'Policy sources: '+link('H2C printer/material policy',ROOT/'print_policy_h2c.json',out)+' and '+
        link('shared role/hardware policy',ROOT/'print_policy.json',out)+'. The build does not connect to a printer.','']
    (out/'README.md').write_text('\n'.join(lines))
    print(f'H2C guide: {passed}/{len(jobs)} digitally audited',flush=True)


if __name__=='__main__':main()
