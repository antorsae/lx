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
    rows=['# Five tweeter families', '',
        'Generated from `src/lx521_baffle/tweeter_options.py` by `make h2c_docs`. '+
        'Stock, Slim and Obiwan are baffle designs; the following are the five tweeter families.', '',
        '![Five tweeter families, ten arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)', '',
        'Left to right: ND25FW-4 crescent, BMR Stock/Slim vase, BMR Obiwan coaxial crescent, '+
        'BMR Obiwan opposed crescent, ND25FN-4 Stock vase, ND25FN-4 Obiwan vase top, and the PT2522-4 and AMT17D2.1 dipoles '+
        '(T above, upright and rotated; all four layouts of each below). '+
        'Actual printed carriers share one scale; drivers and removable service caps are omitted.', '',
        *tweeter_table(), '',
        '## What to change', '',
        '![Obiwan upper choices aligned at the common LM joint](../images/generated/iso/rows/obiwan_upper_row.png)', '',
        '| Baffle | ND25FW-4 | TEBM35C10-4 BMR | ND25FN-4 waveguide | PT2522-4 / AMT17D2.1 dipole |', '|---|---|---|---|---|',
        '| Stock | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | ND25FN-4 vase replaces the whole standard upper; two twist-lock caps; no wings | Not provided |',
        '| Slim | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | No compatible upper currently provided | Not provided |',
        '| Obiwan | Regular UM + separate crescent + regular wings | Regular UM + coaxial or opposed BMR crescent + regular wings | Keyed LM bottom (either stand state) + vase top (LM top + UM + waveguides) + two caps + matching ND25FN-4 wings | T above: any LM + UM-with-tweeter piece. T below: keyed LM bottom + LM-top-with-tweeter-and-UM piece (PT2522 upright: two prints). Upright or rotated. No wings |', '',
        'The dipole parts carry one open-back driver each, upright or rotated 90°, above or below the raised MU10, with hidden '+
        'T and UM ducts (T below: on opposite sides). Drivers fix with M3 screws into M3 heat-set inserts. '+
        '[Dipole tweeter guide](DIPOLE_TWEETERS.md).', '',
        '![Dipole tweeters: PT2522-4 and AMT17D2.1, T above / T below, upright / rotated](../images/generated/iso/rows/dipole_row.png)', '',
        'BMR coaxial and opposed mounts use the same driver family. On Obiwan they share the regular UM half-lap mount. '+
        'The ND25FN-4 Obiwan vase top replaces the LM top and the UM together; it keys onto the released keyed LM bottom of either stand state, '+
        'so it does not use the one-piece Obiwan LM. Its upper LM wing magnets sit at 9/171° and need the matching wings. '+
        'The ND25FN-4 Stock vase uses the H2C Stock LM in either stand state. Choose one upper arrangement per speaker.', '',
        '![Regular and ND25FN-4 uppers with their flat and graded H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)', '',
        '| Obiwan upper | Buried upper contacts | Each matching H2C wing | Hardware / assembly |',
        '|---|---|---|---|',
        '| ND25FW-4 | 2 × Ø5 × 2 mm N52 in regular UM | 3 × Ø5 × 2 mm N52 | [Regular crescent](obiwan.md#dayton-nd25fw-4-crescent) |',
        '| TEBM35C10-4 BMR | Same regular UM contacts | Same regular wings | [Coaxial](obiwan.md#candidate-coaxial-tebm35c10-4-bmr-crescent) / [opposed](obiwan.md#candidate-opposed-tebm35c10-4-bmr-crescent) |',
        '| ND25FN-4 vase top | 2 × Ø5 × 2 mm N52 at 9/171° in the vase top + 2 in the keyed LM bottom | 2 × Ø5 × 2 mm N52 (LM lower + LM upper) | [Vase top, caps and wings](DAYTON_ND25FN4_WAVEGUIDE.md) |',
        '| PT2522-4 / AMT17D2.1 dipole | None (T below: the LM top pockets are filled) | No wings | [Dipole tweeter guide](DIPOLE_TWEETERS.md) |', '',
        'Stock/Slim standard shoulders and B1 wings are for ND25FW-4 only: BMR upper magnet seats differ and no matching '+
        'BMR perimeter is supplied. The extra side magnets on the Obiwan BMR pods also have no supplied mate; '+
        'regular Obiwan wings retain their existing LM/UM contacts.', '',
        '## Current H2C deliverables', '',
        'The [H2C print catalog](../to_print/README.md) includes every family and records compatibility on each job. '+
        'Choose a stand state, an upper arrangement and one optional wing style. Stock/Slim use one LM without the stand '+
        'or two LM pieces with the stand. Obiwan uses one LM and one continuous optional wing per side.', '',
        'The [Dayton ND25FN-4 waveguide guide](DAYTON_ND25FN4_WAVEGUIDE.md) covers both vases, their twist-lock caps, '+
        'the Obiwan keyed LM bottom and matching wings, and their PETG-GF/PLA jobs.', '',
        'For detailed geometry and BMR arrangements, see [Stock](stock.md#tweeter-options), '+
        '[Slim](slim.md#tweeter-options), [Obiwan](obiwan.md#tweeter-options) and [Variants](VARIANTS.md). '+
        'H2C is the only supported printer. Archived small-bed files and CAD-only BMR-slim topology are not current print selections.', '',
        'The ND25FN-4 vases replace the former fused body with M3 retainers; that package is archived. '+
        'Historical revision strings survive only as source/provenance identifiers; use the current H2C catalog for printing.', '',
        '## Qualification', '',
        'All five families are selection options; that does not certify identical acoustics or interchangeable drivers. '+
        'BMR, ND25FN-4 and the dipoles remain qualification candidates. CAD and slicer checks do not establish physical fit, loaded retention, '+
        'directivity or crossover suitability. The current H2C files still require a hardware/material trial.', '']
    (ROOT/'docs/TWEETER_OPTIONS.md').write_text('\n'.join(rows))


LANE_LABELS={'petg_gf_pla':'PETG-GF'}
GROUP_TITLES={'lm':'LM (lower module)','upper':'Upper','dipole':'Dipole tweeters (GRS PT2522-4, Mundorf AMT17D2.1)','wings':'Wings','addons':'Add-ons'}
PRODUCT_INTRO={
    'stock':['Stock baffle. Choose the LM for your stand state and one upper:',
        '- **No floor stand:** `lm/stock_lm_no_floor_stand`.',
        '- **Floor stand:** `lm/stock_lm_lower_floor_stand` + `lm/stock_lm_upper_floor_stand`.',
        '- **Upper, one of:** `stock_upper` (ND25FW-4), `stock_upper_bmr` (TEBM35C10-4 BMR, candidate), or the ND25FN-4 vase '+
        '`stock_nd25fn4_tweeters_{right,left}_upper` + its two caps (candidate; pick the hand, see below).',
        '- **Wings (optional, ND25FW-4 upper only):** the four A shoulders or the two B1 wings.'],
    'slim':['Slim baffle. Choose the LM for your stand state and one upper:',
        '- **No floor stand:** `lm/slim_lm_no_floor_stand`.',
        '- **Floor stand:** `lm/slim_lm_lower_floor_stand` + `lm/slim_lm_upper_floor_stand`.',
        '- **Upper, one of:** `slim_upper` (ND25FW-4) or `slim_upper_bmr` (TEBM35C10-4 BMR, candidate). Slim has no ND25FN-4 upper.',
        '- **Wings (optional, ND25FW-4 upper only):** the four A shoulders or the two B1 wings.'],
    'obiwan':['Obiwan baffle. Choose an LM for your stand state, then one upper arrangement:',
        '- **LM, one of:** the one-piece `lm/obiwan_lm_carrier_{floor_stand,no_floor_stand}`, or the keyed halves '+
        '`lm/obiwan_lm_keyed_bottom_{floor_stand,no_floor_stand}` + `lm/obiwan_lm_keyed_top` (the keyed top is the same part for both stand states).',
        '- **ND25FW-4 or BMR upper:** `upper/obiwan_um_carrier` + one crescent (`obiwan_crescent_nd25fw4`, or the BMR candidates '+
        '`obiwan_crescent_bmr` / `obiwan_crescent_bmr_opposed`); optional regular wings `wings/obiwan_wing_{flat,graded}_{left,right}`. Any LM above fits.',
        '- **ND25FN-4 waveguide (candidate):** the keyed bottom for your stand state + `upper/obiwan_nd25fn4_tweeters_{right,left}_top` '+
        '(it replaces the keyed top, the UM and the crescent) + its two caps; optional matching wings '+
        '`wings/obiwan_nd25fn4_tweeters_<hand>_wing_*`. Regular wings do not fit it.',
        '- **Dipole tweeter (candidate; GRS PT2522-4 planar or Mundorf AMT17D2.1), one driver, upright or rotated:** '+
        '*T above* — any LM above + `dipole/obiwan_<pt2522|amt17d2>_t_above_<upright|rotated>` (UM + tweeter frame in one piece, '+
        'replaces the UM carrier and crescent). *T below* — the keyed bottom for your stand state + '+
        '`dipole/obiwan_<driver>_t_below_<orientation>` (keyed LM top + tweeter + UM in one piece); the PT2522 upright comes as '+
        'two prints, a custom keyed LM top `dipole/obiwan_pt2522_t_below_upright_lm_top` + its T + UM piece `_um_t`, joined like the '+
        'regular LM top and UM (they only fit each other). '+
        'Drivers fix with M3 screws into M3 heat-set inserts. No wings for the dipole parts.',
        '- **Floor stand add-on:** `addons/obiwan_nl8_service_lid` closes the underside NL8 bay.'],
}


def part_rows(jobs,start):
    """One table row per part; its material lanes share the row."""
    from lx521_baffle.h2c.shelf import location
    parts={}
    for job in jobs:parts.setdefault(job['name'],[]).append(job)
    rows=['| Part | Tweeters | Infill | Magnets | Files | Status |','|---|---|---|---:|---|---|']
    for name,lanes in parts.items():
        lanes.sort(key=lambda j:list(LANE_LABELS).index(j['lane']))
        first=lanes[0];stem=location(name)[2]
        files=[link('STL',ROOT/first['stl'],start)]
        for job in lanes:
            label=LANE_LABELS[job['lane']]
            files.append(link(f'{label} project',ROOT/job['project'],start))
            if job.get('sliced_project'):files.append(link(f'{label} sliced',ROOT/job['sliced_project'],start))
            if job.get('audit'):files.append(link(f'{label} audit',ROOT/job['audit'],start))
        status='Digital pass; physical trial pending' if all(j['status']=='sliced_audited' for j in lanes) else \
            ', '.join(sorted({j['status'].replace('_',' ') for j in lanes}))
        label=f'`{stem}`'+(' (candidate)' if first.get('candidate') else '')
        compatibility=', '.join(next(r['driver'].removeprefix('Dayton Audio ').removeprefix('Tectonic ') for r in TWEETER_FAMILIES if r['id']==t) for t in first['tweeter_families'])
        density=json.loads((ROOT/first['work']/'process.json').read_text())['sparse_infill_density']
        rows.append(f"| {label} | {compatibility} | {density} | {first['magnet_count']} | {' · '.join(files)} | {status} |")
    return rows


def write_product_readme(product,jobs):
    from lx521_baffle.h2c.shelf import SHELF,GROUPS,location
    out=SHELF/product
    mine=[j for j in jobs if location(j['name'])[0]==product]
    passed=sum(j['status']=='sliced_audited' for j in mine)
    lines=[f'# {product.title()} — H2C print files','',
        f'{passed} of {len(mine)} jobs digitally audited. '+link('All products, materials and policies',SHELF/'README.md',out)+'.','',
        *PRODUCT_INTRO[product],'',
        'Each part keeps its files side by side: `<part>.stl`, `<part>.petg-gf.3mf` (editable project) and '+
        '`<part>.petg-gf.gcode.3mf` (audited slice). '+
        'Print from the `.3mf` or `.gcode.3mf`: an STL alone loses the supports, blockers, infill modifiers and magnet pauses.','']
    if product in ('stock','obiwan'):
        lines+=['**ND25FN-4 hands.** The waveguides are not left/right symmetric. `tweeters_right` has the tweeter column on the '+
            'right of the MU10 seen from the front; `tweeters_left` is its mirror. For a mirror-imaged pair with the tweeters '+
            'inboard, the left speaker takes `tweeters_right` and the right speaker `tweeters_left`. Caps and wings are hand-specific. '+
            link('ND25FN-4 guide',ROOT/'docs/DAYTON_ND25FN4_WAVEGUIDE.md',out)+'.','']
    plates_path=SHELF/'plates.json'
    combined={k:v for k,v in (json.loads(plates_path.read_text())['files'] if plates_path.exists() else {}).items() if v['product']==product}
    if combined:
        lines+=['## Combined files: one file per stand state','',
            'Each file holds every part for that stand state, arranged on named plates. Plates are grouped by option: print the '+
            'shared plates plus the plates of the one tweeter option (A to F) and wing style you chose. Each part appears once; '+
            'every plate is sliced and audited like the single-part jobs below (per-part settings, blockers, seam placement and '+
            'magnet pauses are kept).','']
        for key,entry in sorted(combined.items()):
            lines+=[f"### {entry['title']}",'',
                link('Project (.3mf)',ROOT/entry['project'],out)+' · '+link('Sliced, all plates (.gcode.3mf)',ROOT/entry['sliced_project'],out)+' · '+
                link('Audit',ROOT/entry['audit'],out),'','| Plate | Contents | Parts | Magnet pauses (mm) |','|---:|---|---|---|']
            for plate in entry['plates']:
                pauses=', '.join(f'{z:.2f}' for z in plate['magnet_pauses_mm']) or 'none'
                lines.append(f"| {plate['plate']} | {plate['name']} | {', '.join('`'+x+'`' for x in plate['parts'])} | {pauses} |")
            lines.append('')
        lines+=['## Individual parts','']
    for group in GROUPS:
        members=[j for j in mine if location(j['name'])[1]==group]
        if not members:continue
        lines+=[f'## {GROUP_TITLES[group]}','',*part_rows(members,out),'']
    (out/'README.md').write_text('\n'.join(lines))


def main():
    from lx521_baffle.h2c.shelf import SHELF,CATALOG,PRODUCTS
    out=SHELF;catalog=json.loads(CATALOG.read_text());cfg=policy()
    jobs=catalog['jobs'];passed=sum(j['status']=='sliced_audited' for j in jobs)
    lines=['# H2C print files','',f'{passed} of {len(jobs)} jobs digitally audited. Printer: **H2C, two 0.6 mm High Flow nozzles**.',
        '', 'The pre-migration source checkpoint is recorded in the [generated-file history guide](../docs/GENERATED_HISTORY.md). These projects use native H2C machine programs. '+
        'H2C is the only supported printer; superseded P2S files are under `archive/p2s/` for provenance only.', '',
        '| Product | Files |','|---|---|',
        *[f"| {p.title()} | {link(p+'/', out/p/'README.md', out)} — `lm/`, `upper/`, `wings/`"+(', `addons/`' if p=='obiwan' else '')+' |' for p in PRODUCTS], '',
        '**Choose one product, one stand state and one tweeter/wing alternative per speaker.** '+
        'Print twice for a stereo pair (each ND25FN-4 cap job is one cap).', '',
        '| Configuration | H2C main pieces per speaker |',
        '|---|---|',
        '| Stock / Slim, no floor stand | 1 LM + 1 upper module |',
        '| Stock / Slim, floor stand | 1 lower LM with stand + 1 upper LM + 1 upper module |',
        '| Regular Obiwan, either stand state | 1 LM (one piece, or keyed bottom + keyed top) + 1 UM + 1 crescent + 2 optional wings |',
        '| Stock with ND25FN-4 vase, either stand state | Stock LM (1 or 2 pieces) + 1 ND25FN-4 vase + 2 caps |',
        '| Obiwan with ND25FN-4 vase top, either stand state | 1 keyed LM bottom (floor or no-floor) + 1 vase top + 2 caps + 2 optional wings |',
        '', 'Stock/Slim standard shoulders and B1 wings are optional for ND25FW-4; no matching BMR perimeter is supplied. The upper module and BMR alternative are shared between stand states. '+
        'A-style shoulders remain four pieces: their upper and lower outlines meet only at a tangent point, so joining them would require an outline change. '+
        'The B1 alternative already uses one continuous wing per side. '+
        'Use the new H2C upper module with the new contained-pin LM joint; its two Ø3 mm pins replace the old upward dovetails. '+
        'The hidden M3 × 20 clamp and Hanglife M3, Ø5 × 4 mm insert remain. Floor-stand LM halves retain the established lower dovetail joint.', '',
        'The Obiwan keyed LM bottom and keyed top are the released two-piece LM. They take every Obiwan upper, and the keyed bottom '+
        'is also what the ND25FN-4 vase top keys onto. The keyed top is one part for both stand states.', '',
        'The unchanged one-piece Stock/Slim floor-stand trials failed the actual PLA-nozzle reach check. '+
        'Their reports are retained under `build/h2c/experiments/rejected_one_piece_floor/`; trial meshes and slices remain local, outside this shelf. '+
        'The current floor-stand pair removes the old vertical LM seam and keeps the original outline.', '',
        '## Tweeter families', '',
        '![Five tweeter families, ten arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)', '',
        *tweeter_table(), '',
        '![Complete Obiwan upper choices at the same LM datum](../images/generated/iso/rows/obiwan_upper_row.png)', '',
        'The ND25FN-4 Obiwan vase top replaces the LM top, the regular UM and the crescent; it needs a keyed LM bottom (floor or no-floor stand) '+
        'and, if wings are wanted, its matching wings. The ND25FN-4 Stock vase replaces the Stock upper module. Slim has no ND25FN-4 upper. '+
        'The PT2522-4 / AMT17D2.1 dipoles (Obiwan only, `obiwan/dipole/`) replace the UM (T above) or the keyed LM top and UM (T below). '+
        link('Dipole tweeter guide',ROOT/'docs/DIPOLE_TWEETERS.md',out)+' · '+
        link('Tweeter family selection guide',ROOT/'docs/TWEETER_OPTIONS.md',out)+' · '+
        link('ND25FN-4 assembly, hardware and print guide',ROOT/'docs/DAYTON_ND25FN4_WAVEGUIDE.md',out), '',
        '## Materials and support', '',
        '- **Model: left nozzle. PLA support interface: right nozzle.** Configure the corresponding feeds on the H2C. '+
        'Use the current H2C nozzle assignments for both material lanes.',
        '- Material: Tinmorry PETG-GF + PLA (`.petg-gf` files) for every part.',
        '- Engineering Plate with glue, 70 °C for both materials, no raft.',
        '- 0.16 mm layers, 0.20 mm first layer, six walls. Structural LM/UM: 100% infill. Wings: 10% gyroid. '+
        'Dayton ND25FN-4 vase bodies: 100%, with their T and t waveguides at 15% gyroid (a modifier volume; the MU10 seat and the LM/seam joints stay 100%); their caps: 15% gyroid. Regular crescent: its existing 30% policy.',
        '- PETG support bases with three dense, zero-gap PLA interface layers above and below. '+
        'ND25FN-4 caps print skin-up with support under their sleeves; vase duct blockers stop short of the driver-seat ceilings.',
        '- Obiwan floor-stand LMs mask their underside NL8 service bay, so its short lid-rebate lip bridges instead of carrying a support column through the lead bay.',
        '- Stock/Slim complete no-floor LM uses a 2 mm brim and no support; other parts use a 5 mm brim. '+
        'The narrow brim needs an adhesion trial on the actual H2C plate.',
        '- Dedicated nozzles retain native H2C priming, retraction and standby programs. '+
        'The dedicated-nozzle process uses its own H2C purge policy. Flushing into model, infill and supports is disabled.',
        '', '## Magnets and hardware', '',
        'All interfaces use Ø5 × 2 mm N52 magnets. A full regular wing has three; an ND25FN-4 Obiwan wing has two (LM lower, LM upper at 9/171°) '+
        'and no UM contact. The ND25FN-4 Stock vase has no wing magnets. All remain buried. Magnet pauses are recalculated from actual H2C extrusion and embedded in the projects.', '',
        'Retain existing driver, LM/UM, stand and wing hardware. The ND25FN-4 vases keep the MU10 M3/Ø4.6 mm printed bores '+
        'for Hanglife M3 inserts, Ø5 mm outside × 4 mm long; the tweeters need no screws (twist-lock caps, soldered leads). '+
        'The optional Tectonic BMR mounts retain their documented M2 hardware exception.', '',
        '## Files', '',
        'Each product folder lists its parts with their STL, editable project (`.3mf`: geometry, settings and measured magnet pauses), '+
        'audited slice (`.gcode.3mf`) and audit report. STLs carry geometry only; importing an STL alone loses supports, blockers, '+
        'infill modifiers and insertion pauses. `catalog.json` is the machine-readable job list.', '',
        ' · '.join(link(p.title(),out/p/'README.md',out) for p in PRODUCTS)]
    lines+=['','## Policy exceptions','']
    lines += ['- '+entry for entry in cfg['exceptions']]
    lines+=['','## Geometry review','',
        '![Regular and ND25FN-4 waveguide assemblies with matching flat or graded H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)', '',
        'The four panels use one camera and scale. Regular panels use the no-floor-stand LM; ND25FN-4 panels use the keyed floor-stand LM bottom and the vase top (it fits the no-floor keyed bottom as well). '+
        'Regular wings also fit the regular-UM BMR selections. Drivers and service caps are omitted.', '',
        link('Stock/Slim orthographic assembly sheet',ROOT/'build/h2c/views/primary_review_contact_sheet.png',out)+' · '+
        link('Obiwan ND25FN-4 vase top with graded wings — front',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_obiwan_tweeters_right_graded_front.png',out)+' · '+
        link('Obiwan ND25FN-4 vase top with flat wings — rear',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_obiwan_tweeters_right_flat_rear_oblique.png',out)+' · '+
        link('Stock ND25FN-4 vase — front',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_stock_tweeters_right_front.png',out)+' · '+
        link('Stock ND25FN-4 vase — rear',ROOT/'build/h2c/views/H2C_Dayton_ND25FN4_stock_tweeters_right_rear_oblique.png',out), '',
        'The views use the actual exported parts in installed coordinates, including the ND25FN-4 caps. Drivers are omitted. '+
        'Native STEP assemblies and individual parts are in `../build/h2c/review_models/` and `../build/h2c/STEP/` '+
        'from this guide. The installed CAD Viewer launcher was unavailable; these PNGs provide the checked visual review.', '',
        link('Native Stock/Slim joint checks',ROOT/'build/h2c/geometry_validation.json',out)+' · '+
        link('Obiwan LM / regular UM and ND25FN-4 vase interface checks',ROOT/'build/h2c/obiwan_interface_validation.json',out),
        '', '## Verification and limits','',
        'The audit checks sliced geometry and modifier identity, machine/material settings, actual nozzle assignments, '+
        'deposition bounds including bead widths and arc extrema, support clearance from every duct contract, '+
        'captive-wall extrusion and insertion timing, plus static G-code validation. '+
        'Native CAD checks cover the new Stock/Slim pin joint, solid interference and preserved driver openings.', '',
        'Passing those checks does not establish physical adhesion, magnet retention, loaded stand strength or acoustic performance. '+
        'The first H2C build still needs the project’s hardware/material qualification. BMR and the Dayton ND25FN-4 vases retain their candidate status.', '',
        '## Rebuild','', 'On a fresh checkout, first follow the [retained-source setup](../docs/GENERATED_HISTORY.md#fresh-checkout-validation).', '',
        '```sh','make                    # H2C geometry, projects, slicing, audit and generated guide',
        'make h2c_prepare        # prepare editable projects from existing authorities',
        'make h2c_validate       # slice/audit prepared projects; reuse hash-matched outputs',
        'make h2c_review         # regenerate native assembly and mesh review images',
        'make h2c_docs           # regenerate this guide',
        '```','',
        'Policy sources: '+link('H2C printer/material policy',ROOT/'print_policy_h2c.json',out)+' and '+
        link('shared role/hardware policy',ROOT/'print_policy.json',out)+'. The build does not connect to a printer.','']
    (out/'README.md').write_text('\n'.join(lines))
    for product in PRODUCTS:write_product_readme(product,jobs)
    write_selection_guide()
    print(f'H2C guide: {passed}/{len(jobs)} digitally audited',flush=True)


if __name__=='__main__':main()
