"""Separate native V4 translucent material lane; never rewrite PETG-GF jobs.

Prepare freezes complete installed filament recipes and clones the approved
native geometry. Slice discovers insertion timing from fresh paths, slices the
pauses, and runs the existing geometry/support/wall gates. Publication requires
all six jobs to pass; verification binds delivered bytes to that evidence.
"""
from copy import deepcopy
from pathlib import Path
from xml.etree import ElementTree as ET
import argparse
import json
import re
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'src')]
from prepare_print import sha
from finalize_print import KEYS, add_pauses, SKILL
from release_validation import PresetResolver, _standalone_cli_config
from lx521_baffle.print_policy import policy, policy_sha256, role_settings, native_material_values

BASE = ROOT/'review/nd25fn4_print'
WORK = ROOT/'review/nd25fn4_translucent'
OUT = HERE/'print_translucent'
POLICY = HERE/'translucent_material_policy.json'
BBL = Path.home()/'Library/Application Support/BambuStudio/system/BBL'
BIN = Path('/Applications/BambuStudio.app/Contents/MacOS/BambuStudio')
PROFILE_NAMES = ['resolved_filament.json', 'resolved_support_interface_filament.json']
META = {'name', 'type', 'from', 'instantiation', 'setting_id', 'filament_id', 'inherits',
        'include', 'description', 'version', 'compatible_printers', 'compatible_prints',
        'compatible_printers_condition', 'compatible_prints_condition',
        'filament_settings_id', 'filament_ingredients_safe', 'filament_emission_safe',
        'filament_contact_safe'}


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def rel(path):
    return str(path.relative_to(ROOT))


def read_settings(path):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('Metadata/project_settings.config'))


def rewrite_project(source, target, process):
    """Copy every geometry/component byte; remove previous material pauses."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():
            if name == 'Metadata/custom_gcode_per_layer.xml':
                continue
            data = json.dumps(process, indent=2).encode() if name == 'Metadata/project_settings.config' else old.read(name)
            new.writestr(name, data)


def material_settings(original, profiles, config):
    result = deepcopy(original)
    a, b = profiles
    # Remove old material-only extension fields so a GF-specific override
    # cannot survive merely because the new system preset omits that key.
    old_profiles = [json.loads((BASE/'profiles'/n).read_text()) for n in PROFILE_NAMES]
    new_keys = set(a) | set(b)
    for key in (set(old_profiles[0]) | set(old_profiles[1])) - new_keys - META:
        result.pop(key, None)
    for key in new_keys - META:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            # The one asymmetric native setting is the first-layer AUX fan;
            # PLA's system default is zero. It is inactive after layer three.
            assert key == 'first_x_layer_fan_speed', key
            va = va if va is not None else ['0']
            vb = vb if vb is not None else ['0']
        if isinstance(va, list):
            shape = original.get(key)
            if not isinstance(shape, list) or len(shape) not in (2, 4):
                shape = [''] * (4 if len(va) == 2 else 2)
            # AMS drying vectors are not extruder-variant vectors. Native
            # projects store only the active setting for each material.
            if key.startswith('filament_dev_ams_drying_') or len(va) > 2 or len(vb) > 2:
                va, vb = va[:1], vb[:1]
            result[key] = native_material_values(shape, va, vb)
        else:
            assert va == vb, (key, va, vb)
            result[key] = va
    result.update(
        print_settings_id='LX521 V4 0.6HF 0.16mm PETG Translucent + PLA Translucent',
        filament_settings_id=[p['name'] for p in profiles],
        filament_ids=[p['filament_id'] for p in profiles],
        filament_colour=[config[k]['colour'] for k in ('model', 'interface')],
        default_filament_colour=[config[k]['colour'] for k in ('model', 'interface')],
        flush_volumes_matrix=[str(v) for v in config['flush_volumes_matrix_mm3']],
        flush_volumes_vector=['140']*4,
        filament_map_mode='Manual', nozzle_volume_type=['High Flow'],
        extruder_nozzle_stats=['High Flow#1'],
        bed_temperature_formula='by_first_filament',
    )
    for key in ['filament_map', 'filament_map_2', 'filament_nozzle_map', 'filament_volume_map']:
        result[key] = ['1', '1']
    return result


def slice_command(project, output):
    return [str(BIN), '--debug', '2', '--slice', '1', '--arrange', '0', '--orient', '0',
            '--mtcpp', '5000000', '--allow-mix-temp', '--load-filaments',
            ';'.join(str(WORK/'profiles'/n) for n in PROFILE_NAMES),
            '--outputdir', str(project.parent), '--export-3mf', output, str(project)]


def prepare(keys=None):
    config = json.loads(POLICY.read_text())
    assert config['model']['ams_slot'] == 4 and config['interface']['ams_slot'] == 2
    assert all(x >= policy()['materials']['purge_each_direction_mm3']
               for x in config['flush_volumes_matrix_mm3'][1:3])
    WORK.mkdir(parents=True, exist_ok=True)
    preserved = {rel(p): sha(p) for p in sorted((HERE/'print').rglob('*')) if p.is_file()}
    frozen = WORK/'preserved_GF_files.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == preserved, 'Existing GF outputs changed'
    else:
        write_json(frozen, preserved)
    profiles = []
    resolver = PresetResolver(BBL)
    for kind, filename in zip(['model', 'interface'], PROFILE_NAMES):
        source = BBL/'filament'/(config[kind]['preset']+'.json')
        profile = _standalone_cli_config('filament', resolver.resolve(source))
        assert profile['filament_id'] == config[kind]['filament_id']
        profile.update(config[kind].get('profile_overrides', {}))
        profile['filament_settings_id'] = [profile['name']]
        profiles.append(profile)
        write_json(WORK/'profiles'/filename, profile)
    for source in resolver.dependencies:
        target = WORK/'profiles/system_sources'/source.relative_to(BBL)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    shutil.copy2(BASE/'profiles/resolved_machine.json', WORK/'profiles/resolved_machine.json')
    wrapper = json.loads((BASE/'profiles/validation_wrapper.json').read_text())
    wrapper.update(native_config=str(WORK/'profiles/resolved_machine.json'),
                   native_settings=[str(WORK/'profiles/resolved_machine.json')],
                   native_filaments=[str(WORK/'profiles'/n) for n in PROFILE_NAMES],
                   filament=dict(type='PETG', nozzle_temp_c=245, bed_temp_c=70))
    write_json(WORK/'profiles/validation_wrapper.json', wrapper)
    # Route/bore facts are immutable inputs to the reused physical-clearance audit.
    (WORK/'inputs').mkdir(exist_ok=True)
    for name in ['ducts.json', 'insert_bores.json']:
        shutil.copy2(BASE/'inputs'/name, WORK/'inputs'/name)
    source_prep = json.loads((BASE/'preparation.json').read_text())
    existing=WORK/'preparation.json'
    prep=json.loads(existing.read_text()) if existing.exists() else {}
    for key in keys or KEYS:
        row = deepcopy(source_prep[key]); source = ROOT/row['path']
        for part in row['sources']:
            assert sha(ROOT/part['path']) == part['sha256'], part['path']
        target = WORK/key/(source.stem+'_TRANSLUCENT.3mf')
        process = material_settings(read_settings(source), profiles, config)
        if key=='body': process.update(config['body_surface_finish'])
        rewrite_project(source, target, process)
        (target.parent/'audit.json').unlink(missing_ok=True)
        row.update(path=rel(target), sha256=sha(target),
                   original_prepared_project=rel(source), original_prepared_sha256=sha(source))
        prep[key] = row
        write_json(target.parent/'dry_run_command.json', slice_command(target, 'discovery.gcode.3mf'))
        print(key, 'prepared:', rel(target), flush=True)
    prep['authority'] = dict(material_policy=rel(POLICY), material_policy_sha256=sha(POLICY),
        shared_policy_sha256=policy_sha256(), original_preparation=rel(BASE/'preparation.json'),
        original_preparation_sha256=sha(BASE/'preparation.json'),
        profile_files={rel(p):sha(p) for p in (WORK/'profiles').rglob('*.json')},
        script_sha256=sha(__file__))
    write_json(WORK/'preparation.json', prep)
    write_json(WORK/'dry_run.json', dict(status='prepared_not_sliced', slicer=str(BIN),
        printer='Bambu Lab P2S 0.6 mm High Flow',
        commands={k:json.loads((ROOT/prep[k]['path']).with_name('dry_run_command.json').read_text()) for k in KEYS}))


def extract(project, directory):
    with zipfile.ZipFile(project) as z:
        (directory/'plate_1.gcode').write_bytes(z.read('Metadata/plate_1.gcode'))


def run_slice(project, output):
    command = slice_command(project, output)
    write_json(project.parent/(output+'.command.json'), command)
    with (project.parent/(output+'.log')).open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    extract(project.parent/output, project.parent)


def check_recipe(process):
    cfg = json.loads(POLICY.read_text())
    assert process['filament_settings_id'] == [cfg[k]['preset'] for k in ('model','interface')]
    assert process['filament_ids'] == ['GFG01', 'GFA17']
    assert process['filament_type'] == ['PETG', 'PLA']
    assert process['nozzle_diameter'] == ['0.6'] and process['nozzle_volume_type'] == ['High Flow']
    for key in ['filament_map','filament_map_2','filament_nozzle_map','filament_volume_map']:
        assert process[key] == ['1','1'], key
    assert process['flush_volumes_matrix'] == [str(x) for x in cfg['flush_volumes_matrix_mm3']]
    recipes = [json.loads((WORK/'profiles'/n).read_text()) for n in PROFILE_NAMES]
    checked = []
    for key in sorted(set(recipes[0]) & set(recipes[1]) - META):
        if key not in process: continue  # Preset metadata need not be saved by native projects.
        a,b = recipes[0][key], recipes[1][key]
        if isinstance(a,list):
            if key.startswith('filament_dev_ams_drying_') or len(a)>2 or len(b)>2: a,b=a[:1],b[:1]
            expected=native_material_values(process[key],a,b)
        else: expected=a
        # Bambu normalizes numeric spellings; compare data numerically when possible.
        def canonical(v):
            if isinstance(v,list): return [canonical(x) for x in v]
            if v == '0%': return 0.0
            try: return float(v)
            except (ValueError,TypeError): return v
        assert canonical(process[key]) == canonical(expected), (key, process[key], expected)
        checked.append(key)
    for key,value in policy()['support'].items(): assert process[key] == value, key
    assert not any(x in json.dumps(process['filament_settings_id']) for x in ['GF','Tinmorry','Basic'])
    return checked


def check_pauses(gcode, records):
    lines=gcode.read_text().splitlines(); actual=[]
    for i,line in enumerate(lines):
        if line != '; ND25FN_MAGNET_INSERTION': continue
        block=lines[i:i+11]
        assert 'G1 Z250 F1200' in block and 'M400 U1' in block
        resume=float([s for s in block if s.startswith('G1 Z')][-1].split()[1][1:])
        prior=next(s for s in reversed(lines[:i]) if s.startswith('; Z_HEIGHT:'))
        assert abs(float(prior.split(':')[1])-resume)<1e-6
        actual.append(resume)
    expected=sorted(set(r['pause_before_z_mm'] for r in records))
    assert actual == expected, (actual,expected)
    # Confirm native custom XML retained real newlines on a future GUI reload.
    return actual


def audit(key, row, records):
    import audit_print
    from audit_gui_slice import material_extrusions
    from captive_wall_audit import audit_captive_walls
    audit_print.WORK=WORK
    prepared=ROOT/row['path']; directory=prepared.parent; final=directory/'ready.gcode.3mf'
    gcode=directory/'plate_1.gcode'
    process=audit_print.settings_check(final, prepared)
    checked=check_recipe(process)
    role='crescent_body' if key=='body' else 'crescent_accessories' if key=='accessories' else 'crescent_wing'
    for field,wanted in role_settings(role).items(): assert process[field]==wanted
    if key=='body':
        for field,wanted in json.loads(POLICY.read_text())['body_surface_finish'].items():
            assert process[field]==wanted,(field,process[field],wanted)
    report=dict(project_sha256=sha(final), gcode_sha256=sha(gcode),
        prepared_project_sha256=sha(prepared), audit_script_sha256=sha(__file__),
        material_policy_sha256=sha(POLICY), shared_policy_sha256=policy_sha256(),
        settings=process, native_recipe_fields_checked=checked,
        materials=material_extrusions(gcode, process['enable_support']=='1'))
    print(key,'native recipes and actual material roles pass',flush=True)
    report['geometry']=audit_print.geometry(final,row)
    print(key,'complete oriented mesh inventory and placement pass',flush=True)
    # Bed is shared: the native by_first_filament policy must really hold 70 C.
    bed=sorted(set(float(m.group(1)) for m in re.finditer(r'^[ \t]*M(?:140|190) S([\d.]+)',gcode.read_text(),re.M)))
    assert 70 in bed and set(bed)<={0.,70.}, bed
    report['actual_bed_temperature_commands_c']=bed
    if key=='body':
        report['ducts']=audit_print.support_ducts(gcode,row)
        from magnet_surface_paths import inspect as inspect_surface
        surface=inspect_surface(gcode,directory/'surface_paths.json')
        for site in surface['sites']:
            assert max(abs(v-.52) for v in site['width_range_mm'])<.001,site['site']
            assert site['max_exterior_boundary_gap_mm']<.08,site['site']
        report['exterior_surface']=dict(report=rel(directory/'surface_paths.json'),
            sha256=sha(directory/'surface_paths.json'),
            sites=[{k:v for k,v in s.items() if k!='layers'} for s in surface['sites']])
    if key=='accessories':
        from cap_support_check import ceiling_supports
        report['cap_ceilings']=ceiling_supports(gcode,row)
    if records:
        assert all(r['magnet_below_resume_plane'] and r['opening_clear_through_prior_layers'] for r in records)
        report['D6_retaining_walls']=audit_captive_walls(ROOT/row['sources'][0]['path'],row['offset'],records,gcode)
    report['magnet_pauses_mm']=check_pauses(gcode,records)
    with zipfile.ZipFile(final) as z:
        if records:
            custom=ET.fromstring(z.read('Metadata/custom_gcode_per_layer.xml'))
            for layer in custom.findall('.//layer'):
                assert '\nG90\n' in layer.attrib['gcode'] and '\nM400 U1\n' in layer.attrib['gcode']
    result=json.loads((directory/'result.json').read_text())
    assert result['return_code']==0 and not result['sliced_plates'][0]['warning_message'],result
    report['slicer_result']=result['sliced_plates'][0]
    with (directory/'gcode_static.json').open('w') as stream:
        subprocess.run([sys.executable,str(SKILL),'validate','--gcode',str(gcode),
            '--profile',str(WORK/'profiles/validation_wrapper.json'),'--json'],stdout=stream,check=True)
    static=json.loads((directory/'gcode_static.json').read_text())
    assert static['ok'] and not static['errors']
    report['status']='pass'
    write_json(directory/'audit.json',report)
    print(key,'support clearance / D6 walls / pauses / static gates passed',flush=True)


def slice_job(key):
    prep=json.loads((WORK/'preparation.json').read_text());row=prep[key]
    project=ROOT/row['path'];directory=project.parent
    assert sha(project)==row['sha256']
    # A rerun must rediscover timing with the previous custom pause removed.
    temporary=project.with_suffix('.staging.3mf')
    rewrite_project(project,temporary,read_settings(project))
    temporary.replace(project)
    row['sha256']=sha(project)
    write_json(WORK/'preparation.json',prep)
    if (directory/'audit.json').exists():
        (directory/'audit.json').unlink()
    run_slice(project,'discovery.gcode.3mf')
    records=[]
    if key!='accessories':
        from print_magnets import discover
        source=ROOT/row['sources'][0]['path']
        authority=json.loads(source.with_suffix('.print.json').read_text())
        records=discover(source,authority,row['offset'],directory/'plate_1.gcode')
        write_json(directory/'magnet_discovery.json',records)
        add_pauses(project,records)
        row['sha256']=sha(project)
        # Each key is sliced sequentially; preparation writes cannot race.
        write_json(WORK/'preparation.json',prep)
        print(key,'fresh insertion pauses:',sorted(set(r['pause_before_z_mm'] for r in records)),flush=True)
        run_slice(project,'ready.gcode.3mf')
        from artifact_emit import _encode_ready_project_custom_gcode_newlines
        _encode_ready_project_custom_gcode_newlines(directory/'ready.gcode.3mf')
    else:
        shutil.copy2(directory/'discovery.gcode.3mf',directory/'ready.gcode.3mf')
    extract(directory/'ready.gcode.3mf',directory)
    audit(key,row,records)


def preserved_check():
    old=json.loads((WORK/'preserved_GF_files.json').read_text())
    assert all((ROOT/p).is_file() and sha(ROOT/p)==digest for p,digest in old.items())
    return len(old)


def publish():
    prep=json.loads((WORK/'preparation.json').read_text());rows=[]
    for key in KEYS:
        directory=(ROOT/prep[key]['path']).parent
        report=json.loads((directory/'audit.json').read_text())
        native=directory/'ready.gcode.3mf'
        assert report['status']=='pass' and sha(native)==report['project_sha256']
        assert report['prepared_project_sha256']==sha(ROOT/prep[key]['path'])==prep[key]['sha256']
        assert report['audit_script_sha256']==sha(__file__)
        static=json.loads((directory/'gcode_static.json').read_text())
        assert static['ok'] and not static['errors']
        base={'body':'01_UM_Crescent_V4_SMOOTH_WALLS','accessories':'02_Caps_TWO_Retainers_TWO'}.get(key,key)
        target=OUT/(base+'_06HF_PETG_TRANSLUCENT_PLA_TRANSLUCENT.gcode.3mf')
        OUT.mkdir(exist_ok=True);shutil.copy2(native,target)
        rows.append(dict(key=key,path=rel(target),sha256=sha(target),
            audit=rel(directory/'audit.json'),audit_sha256=sha(directory/'audit.json'),
            static_audit=rel(directory/'gcode_static.json'),static_audit_sha256=sha(directory/'gcode_static.json'),
            seconds=report['slicer_result']['total_predication'],filaments=report['slicer_result']['filaments'],
            pauses_mm=report['magnet_pauses_mm']))
    files=[Path(__file__),POLICY,ROOT/'print_policy.json',HERE/'prepare_print.py',HERE/'finalize_print.py',
        HERE/'print_magnets.py',HERE/'audit_print.py',HERE/'cap_support_check.py',ROOT/'scripts/release_validation.py',
        ROOT/'scripts/captive_wall_audit.py',ROOT/'scripts/gcode_analysis.py',ROOT/'scripts/audit_gui_slice.py',
        ROOT/'src/lx521_baffle/print_policy.py',ROOT/'scripts/artifact_emit.py',HERE/'magnet_surface_paths.py']
    manifest=dict(status='sliced_static_checks_passed',physical_print_test='not performed with these materials',
        slicer='Bambu Studio 02.07.01.62',printer='Bambu Lab P2S 0.6 mm High Flow',
        material_policy=rel(POLICY),material_policy_sha256=sha(POLICY),
        preparation=rel(WORK/'preparation.json'),preparation_sha256=sha(WORK/'preparation.json'),
        unchanged_GF_files=preserved_check(),source_files={rel(p):sha(p) for p in files},projects=rows)
    write_json(OUT/'manifest.json',manifest)
    print('Published',len(rows),'separate translucent jobs',flush=True)


def verify():
    manifest=json.loads((OUT/'manifest.json').read_text())
    assert sha(ROOT/manifest['preparation'])==manifest['preparation_sha256']
    prep=json.loads((ROOT/manifest['preparation']).read_text())
    for path,digest in manifest['source_files'].items(): assert sha(ROOT/path)==digest,path
    for path,digest in prep['authority']['profile_files'].items(): assert sha(ROOT/path)==digest,path
    for row in manifest['projects']:
        for key,hash_key in [('path','sha256'),('audit','audit_sha256'),('static_audit','static_audit_sha256')]:
            assert sha(ROOT/row[key])==row[hash_key],row[key]
        a=json.loads((ROOT/row['audit']).read_text());assert a['status']=='pass' and a['project_sha256']==row['sha256']
        assert a['audit_script_sha256']==sha(__file__)
        assert a['prepared_project_sha256']==sha(ROOT/prep[row['key']]['path'])==prep[row['key']]['sha256']
        check_recipe(read_settings(ROOT/row['path']))
        for source in a['geometry']: assert sha(ROOT/source['source'])==source['sha256']
        if 'exterior_surface' in a:
            assert sha(ROOT/a['exterior_surface']['report'])==a['exterior_surface']['sha256']
    print('Verified',len(manifest['projects']),'translucent jobs;',preserved_check(),'GF files unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','slice','audit','publish','verify'])
    parser.add_argument('keys',nargs='*',choices=KEYS)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.keys)
    elif args.action=='slice':
        for key in args.keys or KEYS: slice_job(key)
    elif args.action=='audit':
        prep=json.loads((WORK/'preparation.json').read_text())
        for key in args.keys or KEYS:
            discovery=(ROOT/prep[key]['path']).parent/'magnet_discovery.json'
            audit(key,prep[key],json.loads(discovery.read_text()) if discovery.exists() else [])
    elif args.action=='publish': publish()
    else: verify()
