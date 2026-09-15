#!/usr/bin/env python3
"""Import a GUI-exported slice into review storage after complete static audits.

This never promotes into the print shelf and never contacts a printer. Physical
qualification and GUI preview evidence remain separate from a static pass.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'), str(ROOT/'scripts')]
from delivery_contract import LANES, canonical_source, is_gui
from gui_project_audit import audit_gui_geometry, sha256
from manage_delivery import require, validate
import artifact_emit as emit
import slice_captive_magnets as captive
import build_obiwan_combo_plate as combo


def material_extrusions(gcode: Path, supported: bool) -> dict:
    """Track actual positive extrusion under each material, not config alone."""
    tool, feature, relative, previous_e = None, '', True, 0.0
    counts={'model':0,'support':0,'interface':0}
    failures=[]
    number_pattern=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
    for number,raw in enumerate(gcode.open(),1):
        line=raw.split(';',1)[0].strip()
        if raw.startswith('; FEATURE: '): feature=raw[11:].strip()
        if re.fullmatch(r'T\d+',line): tool=int(line[1:])
        if line=='M83': relative=True
        if line=='M82': relative=False
        value=re.search(r'(?:^|\s)E('+number_pattern+r')',line)
        if line.startswith('G92') and value: previous_e=float(value[1])
        if not re.match(r'^G[123](?:\s|$)',line) or not value: continue
        e=float(value[1]);delta=e if relative else e-previous_e;previous_e=e
        if delta<=0 or not re.search(r'\b[XY]'+number_pattern,line): continue
        if not feature or feature.lower() in {'custom','undefined'} or 'tower' in feature.lower() or 'purge' in feature.lower(): continue
        kind='interface' if feature.lower().startswith('support interface') else 'support' if feature.lower().startswith('support') else 'model'
        expected=1 if kind=='interface' and supported else 0
        if tool!=expected: failures.append(f'line {number}: {kind} extrudes with T{tool}, expected T{expected}')
        counts[kind]+=1
    require(not failures,'material assignment failed: '+ '; '.join(failures[:8]))
    require(counts['model']>0,'no model extrusion found')
    if supported:
        require(counts['support']>0 and counts['interface']>0,'supported project emitted no support body/interface')
    else:
        require(counts['support']==counts['interface']==0,'support-free project emitted support')
    return counts


def audit(name: str, project: Path, output: Path):
    shelf=ROOT/'to_print'
    manifest=validate(shelf,geometry=False)
    entry=next((e for e in json.loads((shelf/'catalog.json').read_text())['entries'] if e['name']==name),None)
    require(entry is not None and is_gui(entry),'name must identify a GUI-delivered shelf entry')
    authority=next(r for r in manifest['projects'] if r['name']==name and r['kind']=='gui_project')
    original=shelf/authority['path']
    with zipfile.ZipFile(project) as z:
        members=[n for n in z.namelist() if n.endswith('.gcode')]
        require(members==['Metadata/plate_1.gcode'],'export must contain exactly one sliced plate')
        payload=z.read(members[0])
        settings=json.loads(z.read('Metadata/project_settings.config'))
    with zipfile.ZipFile(original) as z:
        baseline=json.loads(z.read('Metadata/project_settings.config'))
    for key in ('nozzle_diameter','nozzle_volume_type','filament_settings_id','wall_loops','layer_height',
                'sparse_infill_density','sparse_infill_pattern','enable_support','support_type','support_style',
                'support_filament','support_interface_filament','support_top_z_distance','flush_volumes_matrix',
                'flush_multiplier','filament_flush_volumetric_speed'):
        require(settings.get(key)==baseline.get(key),f'GUI changed pinned setting {key}')
    geometry=audit_gui_geometry(ROOT,entry,project)
    supported=str(settings.get('enable_support'))=='1'
    # Resolve the source recipe in disposable storage. Do not trust the edited
    # GUI archive's claims when checking the actual G-code configuration.
    with tempfile.TemporaryDirectory(prefix='lx521-gui-slice-') as temporary:
        work=Path(temporary);gcode=work/'plate_1.gcode';gcode.write_bytes(payload)
        catalog=captive.normalize_catalog(ROOT/'review/captive_magnet_release_catalog.json')
        artifacts={a['id']:a for a in catalog['artifacts']}
        base=captive.prepare_profiles(ROOT/'captive_magnet_slicing_profile_petg_gf_06hf.json',work/'base',
                                     system_root=None,bambu_binary=captive._find_bambu_binary(None))
        if entry.get('composite_plate'):
            api=combo.get_variant(entry['state'])
            parts=[(p.artifact_id,combo._placement_matrix(p)) for p in api.PARTS if p.artifact_id]
            first=artifacts[parts[0][0]]
        elif entry.get('catalog_artifact_id'):
            first=artifacts[entry['catalog_artifact_id']]
            parts=[(first['id'],geometry['stl_to_bed_matrix'])]
        else:
            first=None;parts=[]
        bundle=captive._artifact_profile_bundle(first,base,work/'effective') if first else base
        expected_pauses=[5.96] if parts else []
        archive=emit._validate_ready_project_archive(project,gcode,expected_pause_z=expected_pauses,profile_bundle=bundle)
        parsed=captive.parse_gcode(gcode,retain_feature_prefixes=('Support',))
        errors=emit._validate_actual_gcode_profile(parsed,bundle)
        require(not errors,'G-code profile mismatch: '+'; '.join(errors))
        materials=material_extrusions(gcode,supported)
        from audit_changeover import audit_changeover
        changeover=audit_changeover(gcode) if supported else {'status':'not_applicable_no_support_material'}
        enclosed=emit._enclosed_support_audit(gcode)
        require(enclosed['status']=='pass','enclosed support: '+'; '.join(enclosed['failures']))
        cavities={};ducts={}
        for artifact_id,matrix in parts:
            artifact=artifacts[artifact_id]
            require(sha256(artifact['stl'])==artifact['stl_catalog_sha256'],f'current geometry differs from cavity authority: {artifact_id}')
            discovery=combo._discovery_record(ROOT/'review/captive_magnet_slice_audit',artifact_id)
            cavity_parsed,cavities[artifact_id]=emit._validate_ready_cavity_toolpaths(
                artifact=artifact,discovery_record=discovery,gcode=gcode,stl_to_bed_matrix=matrix)
            emit._assert_pauses_precede_layer_extrusion(cavity_parsed,archive['gcode_pause_events'])
            ducts[artifact_id]=captive.audit_support_toolpaths_vs_ducts(gcode=gcode,
                contract=artifact['duct_collision_contract'],source_to_stl_matrix=artifact['source_to_stl_matrix'],stl_to_bed_matrix=matrix)
        static=emit._validate_with_gcode_skill(gcode,work,bundle)
        require(static.get('ok') is True,'G-code skill static validation failed or unavailable')
        report=dict(status='pass',scope='static geometry/settings/material extrusion/cavities/pauses/ducts/enclosed support/machine bounds',
                    name=name,project_sha256=sha256(project),gui_source_sha256=sha256(original),geometry=geometry,
                    material_extrusions=materials,changeover=changeover,enclosed_support=enclosed,cavities=cavities,ducts=ducts,
                    physical_qualification='pending',release_authorized=False)
        destination=output/name/sha256(project)[:16]
        require(not destination.exists(),f'import already exists: {destination}')
        destination.mkdir(parents=True)
        shutil.copy2(project,destination/'gui_slice.gcode.3mf')
        (destination/'plate_1.gcode').write_bytes(payload)
        (destination/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Static audit passed; imported for review: {destination}')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name',required=True)
    p.add_argument('--project',required=True,type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'review/gui_slices')
    a=p.parse_args();audit(a.name,a.project.resolve(),a.output.resolve())


if __name__=='__main__':
    try: main()
    except (ValueError,OSError,KeyError,captive.AuditError) as exc:
        print(f'GUI slice rejected: {exc}',file=sys.stderr);raise SystemExit(2)
