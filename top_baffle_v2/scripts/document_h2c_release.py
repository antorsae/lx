#!/usr/bin/env python3
"""Generate the H2C file guide from its live job manifest and print policies."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lx521_baffle.h2c.printing import policy
from lx521_baffle.tweeter_options import TWEETER_FAMILIES, ND25FN


def link(label,path,start):return f'[{label}]({os.path.relpath(path,start)})'


def tweeter_table():
    rows=['| Tweeter family | Construction | Compatible baffles |', '|---|---|---|']
    for family in TWEETER_FAMILIES:
        products=', '.join(p.title() for p in family['products'])
        rows.append(f"| {family['label']} | {family['construction']} | {products} |")
    return rows


def write_selection_guide():
    rows=['# Three tweeter families', '',
        'Generated from `src/lx521_baffle/tweeter_options.py` by `make h2c_docs`. '+
        'Stock, Slim and Obiwan are baffle designs; the following are the three tweeter families.', '',
        '![Three tweeter families and five arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)', '',
        'Left to right: ND25FW-4 crescent, BMR Stock/Slim vase, BMR Obiwan coaxial crescent, '+
        'BMR Obiwan opposed crescent, and ND25FN-4 integrated UM/waveguide. '+
        'Actual printed carriers share one scale; drivers and removable service caps are omitted.', '',
        *tweeter_table(), '',
        '## What to change', '',
        '![Obiwan upper choices aligned at the common LM joint](../images/generated/iso/rows/obiwan_upper_row.png)', '',
        '| Baffle | ND25FW-4 | TEBM35C10-4 BMR | ND25FN-4 waveguide |', '|---|---|---|---|',
        '| Stock / Slim | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | No compatible upper currently provided |',
        '| Obiwan | Regular UM + separate crescent + regular wings | Regular UM + coaxial or opposed BMR crescent + regular wings | Fused UM/waveguide body + matching ND25FN-4 wings |', '',
        'BMR coaxial and opposed mounts use the same driver family. On Obiwan they share the regular UM half-lap mount. '+
        'The ND25FN-4 body instead includes the UM carrier: its new surround and buried upper magnets need the matching wings. '+
        'All three Obiwan selections retain the common LM interface. Choose one upper arrangement per speaker.', '',
        '![Regular and ND25FN-4 uppers with their flat and graded H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)', '',
        '| Obiwan upper | Buried upper contacts | Each matching H2C wing | Hardware / assembly |',
        '|---|---|---|---|',
        '| ND25FW-4 | 2 × Ø5 × 2 mm N52 in regular UM | 3 × Ø5 × 2 mm N52 | [Regular crescent](obiwan.md#dayton-nd25fw-4-crescent) |',
        '| TEBM35C10-4 BMR | Same regular UM contacts | Same regular wings | [Coaxial](obiwan.md#candidate-coaxial-tebm35c10-4-bmr-crescent) / [opposed](obiwan.md#candidate-opposed-tebm35c10-4-bmr-crescent) |',
        '| ND25FN-4 waveguide | 4 × Ø6 × 3 mm N45 in fused body | 2 × Ø6 × 3 mm N45 at UM + 2 × Ø5 × 2 mm N52 at LM | [Body, caps and M3 retainers](DAYTON_ND25FN4_WAVEGUIDE.md) |', '',
        'Stock/Slim standard shoulders and B1 wings are for ND25FW-4 only: BMR upper magnet seats differ and no matching '+
        'BMR perimeter is supplied. The extra side magnets on the Obiwan BMR pods also have no supplied mate; '+
        'regular Obiwan wings retain their existing LM/UM contacts.', '',
        '## Current H2C deliverables', '',
        'The [H2C print catalog](../to_print/h2c/README.md) includes every family and records compatibility on each job. '+
        'Choose a stand state, an upper arrangement and one optional wing style. Stock/Slim use one LM without the stand '+
        'or two LM pieces with the stand. Obiwan uses one LM and one continuous optional wing per side.', '',
        'The [Dayton ND25FN-4 waveguide guide](DAYTON_ND25FN4_WAVEGUIDE.md) covers its fused body, two caps, '+
        'two M3 retainers, matching wings, magnet types and separate PETG-GF/PLA and PETG Translucent/PLA Translucent jobs.', '',
        'For detailed geometry and BMR arrangements, see [Stock](stock.md#tweeter-options), '+
        '[Slim](slim.md#tweeter-options), [Obiwan](obiwan.md#tweeter-options) and [Variants](VARIANTS.md). '+
        'The earlier P2S shelf and CAD-only BMR-slim topology keep their separate qualification boundaries.', '',
        'ND25FN-4 waveguide is the current product name for the former retained-package revision. '+
        'Historical revision strings survive only as source/provenance identifiers; use the current H2C catalog for printing.', '',
        '## Qualification', '',
        'All three families are selection options; that does not certify identical acoustics or interchangeable drivers. '+
        'BMR and ND25FN-4 remain qualification candidates. CAD and slicer checks do not establish physical fit, loaded retention, '+
        'directivity or crossover suitability. The current H2C files still require a hardware/material trial.', '']
    (ROOT/'docs/TWEETER_OPTIONS.md').write_text('\n'.join(rows))


def main():
    out=ROOT/'to_print/h2c';catalog=json.loads((out/'catalog.json').read_text());cfg=policy()
    jobs=catalog['jobs'];passed=sum(j['status']=='sliced_audited' for j in jobs)
    lines=['# H2C print files','',f'{passed} of {len(jobs)} jobs digitally audited. Printer: **H2C, two 0.6 mm High Flow nozzles**.',
        '', 'The pre-migration source checkpoint is recorded in the [generated-file history guide](../../docs/GENERATED_HISTORY.md). These projects use native H2C machine programs. '+
        'The P2S files are a separate earlier release.', '',
        '**Choose one family, one stand state and one tweeter/wing alternative per speaker.** '+
        'Print twice for a stereo pair. The Dayton ND25FN-4 waveguide accessories plate already contains two caps and two retainers.', '',
        '| Configuration | H2C main pieces per speaker | Previous P2S arrangement |',
        '|---|---|---|',
        '| Stock / Slim, no floor stand | 1 LM + 1 upper module | 3 LM + 1 upper |',
        '| Stock / Slim, floor stand | 1 lower LM with stand + 1 upper LM + 1 upper module | 3 LM + 1 upper |',
        '| Regular Obiwan, either stand state | 1 LM + 1 UM + 1 crescent + 2 optional wings | 2 LM + 1 UM + 1 crescent + 4 wing sections |',
        '| Obiwan with ND25FN-4 waveguide, either stand state | 1 LM + 1 fused body + 2 optional wings | 2 LM + 1 fused body + 4 wing sections |',
        '', 'Stock/Slim standard shoulders and B1 wings are optional for ND25FW-4; no matching BMR perimeter is supplied. The upper module and BMR alternative are shared between stand states. '+
        'A-style shoulders remain four pieces: their upper and lower outlines meet only at a tangent point, so joining them would require an outline change. '+
        'The B1 alternative already uses one continuous wing per side. '+
        'Use the new H2C upper module with the new contained-pin LM joint; its two Ø3 mm pins replace the old upward dovetails. '+
        'The hidden M3 × 20 clamp and Hanglife M3, Ø5 × 4 mm insert remain. Floor-stand LM halves retain the established lower dovetail joint.', '',
        'The unchanged one-piece Stock/Slim floor-stand trials failed the actual PLA-nozzle reach check. '+
        'Their reports are retained under `build/h2c/experiments/rejected_one_piece_floor/`; trial meshes and slices remain local, outside this shelf. '+
        'The current floor-stand pair removes the old vertical LM seam and keeps the original outline.', '',
        '## Tweeter families', '',
        '![Three tweeter families and five arrangements at one scale](../../images/generated/iso/rows/tweeter_row.png)', '',
        *tweeter_table(), '',
        '![Complete Obiwan upper choices at the same LM datum](../../images/generated/iso/rows/obiwan_upper_row.png)', '',
        'The ND25FN-4 selection replaces the entire regular UM and crescent. Use its matching continuous wings; '+
        'the same fused body fits both stand states. Stock/Slim currently have no ND25FN-4 upper. '+
        link('Three-family selection guide',ROOT/'docs/TWEETER_OPTIONS.md',out)+' · '+
        link('ND25FN-4 assembly, hardware and print guide',ROOT/'docs/DAYTON_ND25FN4_WAVEGUIDE.md',out), '',
        '## Materials and support', '',
        '- **Model: left nozzle. PLA support interface: right nozzle.** Configure the corresponding feeds on the H2C. '+
        'The previous P2S AMS slot numbers are not nozzle assignments.',
        '- Default lane: Tinmorry PETG-GF + PLA. Dayton ND25FN-4 waveguide also has its own PETG Translucent + PLA Translucent lane.',
        '- Engineering Plate with glue, 70 °C for both materials, no raft.',
        '- 0.16 mm layers, 0.20 mm first layer, six walls. Structural LM/UM: 100% infill. Wings: 10% gyroid. '+
        'Dayton ND25FN-4 waveguide tweeter modifier: 15% gyroid; caps/retainers: 15% gyroid. Regular crescent: its existing 30% policy.',
        '- PETG support bases with three dense, zero-gap PLA interface layers above and below. '+
        'Caps have PLA directly beneath their ceilings; retainers print without supports.',
        '- Stock/Slim complete no-floor LM uses a 2 mm brim and no support; other parts use a 5 mm brim. '+
        'The narrow brim needs an adhesion trial on the actual H2C plate.',
        '- Dedicated nozzles retain native H2C priming, retraction and standby programs. '+
        'The P2S same-nozzle purge-volume workaround is not transferred to every H2C switch. Flushing into model, infill and supports is disabled.',
        '- Dayton ND25FN-4 waveguide translucent body keeps Classic outer-first walls to reduce changes in surface texture around buried magnets.',
        '', '## Magnets and hardware', '',
        'Regular interfaces retain Ø5 × 2 mm N52 magnets. Dayton ND25FN-4 waveguide UM/wing contacts retain Ø6 × 3 mm N45; '+
        'Dayton ND25FN-4 waveguide wings use the regular Ø5 × 2 mm magnets at their two LM contacts. A full regular wing has three magnets; '+
        'a full Dayton ND25FN-4 waveguide wing has four. All remain buried. Magnet pauses are recalculated from actual H2C extrusion and embedded in the projects.', '',
        'Retain existing driver, LM/UM, stand and wing hardware. Dayton ND25FN-4 waveguide uses the project’s M3/Ø4.6 mm printed-bore convention '+
        'for Hanglife M3 inserts, Ø5 mm outside × 4 mm long. The optional Tectonic BMR mounts retain their documented M2 hardware exception.', '',
        '## File list', '',
        '`.3mf` is the editable project with its geometry, settings and measured magnet pauses. '+
        '`.gcode.3mf` contains the audited slice. STLs carry geometry only; importing an STL alone loses supports, infill modifiers and insertion pauses.', '',
        '| Baffle / part | Tweeter compatibility | Material lane | Infill | Magnets | Files | Status |','|---|---|---|---|---:|---|---|']
    for job in jobs:
        project=ROOT/job['project'];work=ROOT/job['work']
        process=json.loads((work/'process.json').read_text())
        meshes=list(dict.fromkeys(s['path'] for s in job['preparation']['sources'] if s['subtype']=='normal_part'))
        files=[link('Project',project,out)]
        files += [link('STL' if len(meshes)==1 else Path(p).stem,ROOT/p,out) for p in meshes]
        if job.get('sliced_project'):files.insert(1,link('Sliced',ROOT/job['sliced_project'],out))
        if job.get('audit'):files.append(link('Audit',ROOT/job['audit'],out))
        status='Digital pass; physical trial pending' if job['status']=='sliced_audited' else job['status'].replace('_',' ')
        name=job['name'].removeprefix('h2c_')
        if job['family']==ND25FN:
            name='Obiwan / ND25FN-4 '+name.removeprefix('dayton_nd25fn4_').replace('_',' ')
        name+=(' (candidate)' if job.get('candidate') else '')
        compatibility=', '.join(next(r['driver'].removeprefix('Dayton Audio ').removeprefix('Tectonic ') for r in TWEETER_FAMILIES if r['id']==t) for t in job['tweeter_families'])
        density=process['sparse_infill_density']+(' + 15% tweeter region' if job['role']=='crescent_body' else '')
        lines.append(f"| {name} | {compatibility} | {job['lane']} | {density} | {job['magnet_count']} | {' · '.join(files)} | {status} |")
    lines+=['','## Policy exceptions','']
    lines += ['- '+entry for entry in cfg['exceptions']]
    lines+=['','## Geometry review','',
        '![Regular and ND25FN-4 waveguide assemblies with matching flat or graded H2C wings](../../images/generated/iso/rows/obiwan_wing_row.png)', '',
        'The four panels use one camera and scale, with the common no-floor-stand LM. '+
        'Regular wings also fit the regular-UM BMR selections. Drivers and service caps are omitted.', '',
        link('Stock/Slim orthographic assembly sheet',ROOT/'build/h2c/views/primary_review_contact_sheet.png',out)+' · '+
        link('Dayton ND25FN-4 waveguide with continuous wings — front',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_graded_front.png',out)+' · '+
        link('Dayton ND25FN-4 waveguide with continuous wings — rear',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_flat_rear_oblique.png',out), '',
        'The views use the actual exported parts in installed coordinates. Drivers and service caps are omitted. '+
        'Native STEP assemblies and individual parts are in `../../build/h2c/review_models/` and `../../build/h2c/STEP/` '+
        'from this guide. The installed CAD Viewer launcher was unavailable; these PNGs provide the checked visual review.', '',
        link('Native Stock/Slim joint checks',ROOT/'build/h2c/geometry_validation.json',out)+' · '+
        link('Continuous Obiwan LM / regular UM / Dayton ND25FN-4 waveguide interface checks',ROOT/'build/h2c/obiwan_interface_validation.json',out),
        '', '## Verification and limits','',
        'The audit checks sliced geometry and modifier identity, machine/material settings, actual nozzle assignments, '+
        'deposition bounds including bead widths and arc extrema, support clearance, cap-ceiling coverage, '+
        'captive-wall extrusion and insertion timing, plus static G-code validation. '+
        'Native CAD checks cover the new Stock/Slim pin joint, solid interference and preserved driver openings.', '',
        'Passing those checks does not establish physical adhesion, magnet retention, loaded stand strength or acoustic performance. '+
        'The first H2C build still needs the project’s hardware/material qualification. BMR and Dayton ND25FN-4 waveguide retain their candidate status.', '',
        '## Rebuild','', 'On a fresh checkout, first follow the [retained-source setup](../../docs/GENERATED_HISTORY.md#fresh-checkout-validation).', '',
        '```sh','make                    # H2C geometry, projects, slicing, audit and generated guide',
        'make h2c_prepare        # prepare editable projects from existing authorities',
        'make h2c_validate       # slice/audit prepared projects; reuse hash-matched outputs',
        'make h2c_review         # regenerate native assembly and mesh review images',
        'make h2c_docs           # regenerate this guide',
        'make PRINTER=P2S all    # earlier printer pipeline','```','',
        'Policy sources: '+link('H2C printer/material policy',ROOT/'print_policy_h2c.json',out)+' and '+
        link('shared role/hardware policy',ROOT/'print_policy.json',out)+'. The build does not connect to a printer.','']
    (out/'README.md').write_text('\n'.join(lines))
    write_selection_guide()
    print(f'H2C guide: {passed}/{len(jobs)} digitally audited',flush=True)


if __name__=='__main__':main()
