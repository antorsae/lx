"""Generate the material-lane guide and a preview from validated real toolpaths."""
import json
from pathlib import Path
import numpy as np
from translucent_print import HERE, ROOT, OUT, WORK, POLICY, write_json, sha, verify


def main():
    verify()
    manifest=json.loads((OUT/'manifest.json').read_text())
    config=json.loads(POLICY.read_text())
    prep=json.loads((WORK/'preparation.json').read_text())
    reports={r['key']:json.loads((ROOT/r['audit']).read_text()) for r in manifest['projects']}
    names={'body':'01 — fused UM + V4 body','accessories':'02 — two caps + two M3 retainers',
           'V4_flat_left_UPPER':'Flat left upper wing','V4_flat_right_UPPER':'Flat right upper wing',
           'V4_graded_left_UPPER':'Graded left upper wing','V4_graded_right_UPPER':'Graded right upper wing'}
    table=[]
    for row in manifest['projects']:
        minutes=round(row['seconds']/60)
        grams={r['id']:r['total_used_g'] for r in row['filaments']}
        table.append(f"| [{names[row['key']]}]({Path(row['path']).name}) | {minutes//60} h {minutes%60:02} min | {grams[1]:.1f} g | {grams[2]:.1f} g |")
    profile=[json.loads((WORK/'profiles'/n).read_text()) for n in ['resolved_filament.json','resolved_support_interface_filament.json']]
    def selected(p,key):
        values=p[key];return values[-1] if isinstance(values,list) else values
    materials=[]
    for role,p in zip(['model','interface'],profile):
        r=config[role]
        materials.append(f"| {r['slicer_filament']} | {r['preset'].split(' @')[0]} | **{r['ams_slot']}** | {selected(p,'nozzle_temperature_initial_layer')} / {selected(p,'nozzle_temperature')} °C | {selected(p,'filament_flow_ratio')} | {selected(p,'filament_max_volumetric_speed')} mm³/s |")
    body_pauses=', '.join(f'{v:.2f}' for v in reports['body']['magnet_pauses_mm'])
    wing_pauses=', '.join(f'{v:.2f}' for v in reports['V4_flat_left_UPPER']['magnet_pauses_mm'])
    minimum=min(x['inner_ceiling_coverage'] for c in reports['accessories']['cap_ceilings']['caps'] for x in c['interface_layers'])
    doc='''# V4 — PETG Translucent + PLA Translucent

Separate **sliced Bambu Studio 3MF files for the P2S with a 0.6 mm High Flow nozzle**. The installed Bambu translucent filament recipes have been resolved in full and the toolpaths regenerated. This lane has its own material and changeover policy; later PETG-GF/PLA calibration changes do not alter these translucent print files.

## AMS assignment

Open a file as a **project**, use its prepared slice, and map these materials in Bambu Studio's Send dialog:

| Filament in project | Material | AMS slot | First / other layers | Flow | Volumetric limit |
|---|---|---:|---:|---:|---:|
'''+ '\n'.join(materials)+'''

The native files use one physical High Flow nozzle. Their internal nozzle-map values are not AMS slot numbers; **material 1 → slot 4, material 2 → slot 2** is the confirmed print-send assignment. PLA is used for removable support interfaces. Model parts and support bodies use PETG Translucent. Display colours follow the selected Bambu Studio presets.

**Textured PEI bed: 70 °C throughout.** This lane explicitly sets both materials' bed values to 70 °C. PLA's installed preset normally uses 55/60 °C; the first body slice showed a 55 °C command during a material change despite `by_first_filament`. The shared-bed override prevents that drop. PLA's nozzle, flow, cooling and retraction settings retain the installed translucent recipe.

## Files

For one speaker: print the body once, the accessories plate once, and one matching left/right upper-wing pair. Flat and graded wings are alternatives; retain the existing matching lower wings.

| File | Estimated time | PETG Translucent | PLA Translucent |
|---|---:|---:|---:|
'''+ '\n'.join(table)+'''

Estimates include the native slicer's support and purge consumption and exclude time spent inserting magnets.

## Retained geometry and print policy

- Same approved print meshes, placement, support blockers, driver seats and LM interfaces. One body fits both floor-stand states; the unchanged geometry retains the [LM assembly check](../print/LM_assembly_check.json).
- UM: **100% zig-zag**. Tweeter region: **15% gyroid**, using the existing modifier above installed Y421 mm. Upper wings: **10% gyroid**. Caps/retainers: **15% gyroid**.
- 0.16 mm layers; 0.20 mm first layer; six walls, ten top and five bottom layers. Existing widths, ironing, 5 mm outer brim and print orientations retained.
- PETG support bodies and **three dense PLA top-interface layers**, zero top Z gap, zero interface spacing. Existing bottom-contact exception remains 0.18 mm gap / 0.5 mm spacing.
- Purge: **298 mm³ PETG → PLA; 575 mm³ PLA → PETG**, matching the selected translucent colours and exceeding the shared material-separation minimum. Flushing into model, infill and supports is disabled.
- All 12 insert bores, cable routes and captive magnet cavities retain their nonprinting support blockers. Native geometry and actual support paths are independently checked.

The material overlay and its explicit exceptions live in [translucent_material_policy.json](../translucent_material_policy.json). Geometry roles, support rules and wall acceptance still come from the project-wide [print policy](../../../print_policy.json). Material temperature, flow, cooling, retraction and speed settings come from the complete frozen installed translucent presets, rather than the GF recipe.

Bambu's native `--allow-mix-temp` option is used for the intentional PETG/PLA support pair. The real PETG and PLA identities are preserved. If you reslice in the GUI, retain the mixed-material support permission and recheck support coverage and magnet timing.

## Body surface revision

Use the body file named **SMOOTH_WALLS**. The previous Arachne slice changed the visible outer-wall width from 0.52 mm to as much as 1.09 mm over the magnet pockets at a constant 60 mm/s. That changes the requested extrusion rate locally and is a plausible source of the reported pocket outline or texture change.

The revised body uses **Classic, thin-wall detection, outer walls first**, retaining **0.52 mm outer walls at 60 mm/s** across all four magnet covers. `Precise outer wall` is disabled because Bambu ignores that option for outer-first ordering. Both the complete retaining-wall boundary and the visible exterior contour are checked against actual extrusion footprints. Magnet positions, cover geometry, loading paths and LM interfaces are unchanged. Caps and wings retain their existing Arachne process.

![Measured exterior paths and line widths before and after](magnet_surface_comparison.png)

The measured exterior width change is removed. Confirm the physical appearance with the [small body magnet surface test](qualification/README.md), which uses the actual curved body station and the revised material/process settings. Complete cosmetic invisibility has not been demonstrated by a physical print.

## Magnet pauses and material appearance

'''+f'The body pauses before **Z{body_pauses} mm** for its four Ø6×3 mm magnets. Each upper wing pauses before **Z{wing_pauses} mm** for one Ø5×2 mm LM magnet and then two Ø6×3 mm UM magnets. These timings were rediscovered from the translucent-material slices.\n'+'''
Match polarity, fully seat each disc in its inclined pocket and keep it below the printing surface before resuming. The pause moves the bed down for access and restores the printing height. Changing pose or layer height requires reslicing and checking insertion timing again.

The buried cavities remain closed. Translucent plastic can reveal the infill and embedded magnets; this recipe preserves the structural infill rather than optimizing optical clarity. Physical adhesion, support removal and magnet retention with this material pair remain unmeasured.

## Cap supports and verification

'''+f'Both caps have three PLA ceiling interfaces at Z8.68, Z8.84 and Z9.00 mm, followed by the first PETG ceiling at Z9.16 mm. Measured inner coverage is at least **{minimum*100:.2f}%**. Neither retainer has generated support.\n'+'''
![Actual PLA interfaces and first PETG ceiling layers](cap_ceiling_support.png)

All six jobs passed full oriented-triangle/placement comparison, native material recipe checks, actual T0 model/support and T1 interface extrusion checks, bed-temperature checks, support clearance, D6 wall continuity, insertion timing and static G-code validation. Native plate warnings are empty. Standard Bambu firmware commands remain intact; static checking does not simulate the printer.

[manifest.json](manifest.json) binds the six files to the preparation, frozen profiles, source meshes and passing reports. [views_manifest.json](views_manifest.json) binds the ceiling preview to the actual sliced G-code. Rebuild from the project root:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_print.py prepare
# Inspect review/nd25fn4_translucent/dry_run.json before slicing.
../.venv/bin/python candidates/nd25fn4_crescent/translucent_print.py slice
../.venv/bin/python candidates/nd25fn4_crescent/translucent_print.py publish
../.venv/bin/python candidates/nd25fn4_crescent/translucent_surface_coupon.py prepare
# Inspect review/nd25fn4_translucent/surface_coupon/dry_run.json.
../.venv/bin/python candidates/nd25fn4_crescent/translucent_surface_coupon.py run
../.venv/bin/python candidates/nd25fn4_crescent/translucent_print_docs.py
```

Verify the current files without reslicing:

```bash
../.venv/bin/python candidates/nd25fn4_crescent/translucent_print.py verify
```
'''
    (OUT/'README.md').write_text(doc)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    from cap_support_check import local_segments
    from gcode_analysis import parse_gcode
    gcode=WORK/'accessories/plate_1.gcode';parsed=parse_gcode(gcode)
    fig,axes=plt.subplots(2,2,figsize=(10,10),layout='constrained')
    for column,cap in enumerate(reports['accessories']['cap_ceilings']['caps']):
        center=np.asarray(cap['center_bed_xy_mm'])
        support=min(parsed.layers,key=lambda l:abs(l.z-cap['interface_layers'][-1]['z_mm']))
        roof=min(parsed.layers,key=lambda l:abs(l.z-cap['first_ceiling_layer_z_mm']))
        for row,layer in enumerate([support,roof]):
            ax=axes[row,column]
            below=[np.array([[s.x0,s.y0],[s.x1,s.y1]])-center for s in local_segments(support,center,'Support interface')]
            ax.add_collection(LineCollection(below,colors='#BA3344',linewidths=1.,alpha=1. if row==0 else .35))
            if row:
                lines=[np.array([[s.x0,s.y0],[s.x1,s.y1]])-center for s in local_segments(layer,center)
                       if not s.feature.startswith(('Support','Prime','Brim','Custom'))]
                ax.add_collection(LineCollection(lines,colors='#5B6D78',linewidths=.7))
            ax.add_patch(Circle((0,0),cap['ceiling_diameter_mm']/2,fill=False,color='#172f40',lw=1,ls='--'))
            label='PLA Translucent interface' if row==0 else 'First PETG Translucent ceiling'
            ax.set(xlim=(-29,29),ylim=(-29,29),aspect='equal',xlabel='Local X (mm)',ylabel='Local Y (mm)',
                title=f'Cap {column+1} — {label}\nZ = {layer.z:.2f} mm',facecolor='#FAFBFC')
            ax.grid(alpha=.12)
    fig.suptitle('V4 caps — actual translucent-material toolpaths\nRed: PLA interface (AMS 2) · Grey: PETG ceiling (AMS 4)\nThree interface layers per cap · Zero gap to ceiling',fontsize=13)
    image=OUT/'cap_ceiling_support.png';fig.savefig(image,dpi=160);plt.close(fig)
    baseline_path=WORK/'surface_trials/baseline/surface.json'
    final_path=WORK/'body/surface_paths.json'
    baseline=json.loads(baseline_path.read_text());final=json.loads(final_path.read_text())
    assert sha(ROOT/baseline['gcode'])==baseline['gcode_sha256']
    assert final['gcode_sha256']==reports['body']['gcode_sha256']
    old=baseline['sites'][2];new=final['sites'][2]
    before=max(old['layers'],key=lambda l:l['width_range_mm'][1])
    after=min(new['layers'],key=lambda l:abs(l['z_mm']-before['z_mm']))
    center=np.asarray(old['center_bed_mm'][:2])
    fig=plt.figure(figsize=(11,10),layout='constrained')
    grid=fig.add_gridspec(2,2,height_ratios=[1.4,1])
    from shapely.geometry import LineString
    for col,(layer,title,colour) in enumerate([(before,'Before: Arachne','#DC862E'),(after,'Revised: Classic, outer first','#276A97')]):
        ax=fig.add_subplot(grid[0,col])
        for p in layer['paths']:
            polygon=LineString(np.array([p['a'],p['b']])-center).buffer(p['width']/2,quad_segs=8)
            x,y=polygon.exterior.xy;ax.fill(x,y,color=colour,lw=0)
        for edge in layer['exterior_section']:
            edge=np.asarray(edge)-center;ax.plot(edge[:,0],edge[:,1],color='#223844',lw=1.,ls='--')
        ax.set(aspect='equal',xlim=(-7,7),ylim=(-7,7),xlabel='Local X (mm)',ylabel='Local Y (mm)',
            title=f'{title}\nActual outside-wall footprints at Z{layer["z_mm"]:.2f} mm')
        ax.grid(alpha=.15)
        ax.text(.03,.03,'Dashed: CAD outside boundary',transform=ax.transAxes,fontsize=9)
    ax=fig.add_subplot(grid[1,:])
    heights=sorted({l['z_mm'] for s in baseline['sites'] for l in s['layers']})
    oldmax=[max(l['width_range_mm'][1] for s in baseline['sites'] for l in s['layers'] if abs(l['z_mm']-z)<.001) for z in heights]
    ax.fill_between(heights,.52,oldmax,color='#DC862E',alpha=.25)
    ax.plot(heights,oldmax,color='#C97825',lw=2,label='Before — maximum across four covers')
    newz=sorted({l['z_mm'] for s in final['sites'] for l in s['layers']})
    newmax=[max(l['width_range_mm'][1] for s in final['sites'] for l in s['layers'] if abs(l['z_mm']-z)<.001) for z in newz]
    ax.plot(newz,newmax,color='#276A97',lw=2.5,label='Revised — all four covers stay at 0.52 mm')
    ax.set(xlabel='Printed layer Z (mm)',ylabel='Visible outer-wall width (mm)',ylim=(.4,1.2),
        title='Measured over the seated-magnet region; outside speed remains 60 mm/s')
    ax.legend(loc='upper right');ax.grid(alpha=.15)
    fig.suptitle('V4 magnet covers — removing the exterior width change\nActual sliced paths; surface appearance still requires a physical test',fontsize=15)
    surface_image=OUT/'magnet_surface_comparison.png';fig.savefig(surface_image,dpi=160);plt.close(fig)
    write_json(OUT/'views_manifest.json',dict(script_sha256=sha(__file__),project_sha256=reports['accessories']['project_sha256'],
        gcode_sha256=sha(gcode),image_sha256=sha(image),image=image.name,
        cap_support_measurements=reports['accessories']['cap_ceilings'],
        surface_comparison=dict(image=surface_image.name,image_sha256=sha(surface_image),
            baseline_report_sha256=sha(baseline_path),final_report_sha256=sha(final_path),
            baseline_gcode_sha256=baseline['gcode_sha256'],final_gcode_sha256=final['gcode_sha256'])))
    print('Generated',OUT/'README.md','and',image,flush=True)


if __name__=='__main__': main()
