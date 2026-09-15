#!/usr/bin/env python3
"""Audit actual H2C deposition, then insert measured captive-magnet pauses."""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'candidates/nd25fn4_crescent')]
import numpy as np
import trimesh
from lx521_baffle.io import sha256_file
from lx521_baffle.h2c.printing import write_json,policy,job_settings
from print_magnets import magnet_geometry, discover_specs
from captive_wall_audit import audit_captive_walls

NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
ARG=re.compile(r'(?:^|\s)([XYZEFIJKR])('+NUMBER+r')')
AUDIT_VERSION=4


def qualification_inputs(job):
    """Bind qualification to the actual inputs and the code that checked it."""
    paths={ROOT/job['project'], ROOT/job['work']/'process.json', ROOT/job['work']/'dry_run.json',
        ROOT/'print_policy_h2c.json',ROOT/'print_policy.json'}
    paths.update(ROOT/s['path'] for s in job['preparation']['sources'])
    paths.update(ROOT/p for p in job.get('blockers',[]))
    paths.update(ROOT/s['path'] for s in job.get('blocker_sources',[]))
    if job.get('source'):paths.add(ROOT/job['source'])
    for p in ('scripts/audit_h2c_print.py','scripts/captive_wall_audit.py','scripts/gcode_analysis.py',
              'scripts/artifact_emit.py','src/lx521_baffle/h2c/printing.py',
              'candidates/nd25fn4_crescent/print_magnets.py','candidates/nd25fn4_crescent/audit_print.py',
              'candidates/nd25fn4_crescent/cap_support_check.py'):
        paths.add(ROOT/p)
    profiles=ROOT/'build/h2c/profiles'/job['lane']
    paths.update(profiles/(p+'.json') for p in ('machine','model','interface'))
    contract=ROOT/'build/h2c/contracts'/(job['name']+'.bores.json')
    if contract.exists():paths.add(contract)
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}


def audit_is_current(job):
    if job.get('status')!='sliced_audited' or job.get('audit_version')!=AUDIT_VERSION:return False
    try:
        audit=json.loads((ROOT/job['audit']).read_text())
        return (audit.get('inputs')==qualification_inputs(job)
            and audit['sliced_project_sha256']==sha256_file(ROOT/job['sliced_project'])
            and audit['gcode_sha256']==sha256_file(ROOT/job['work']/'plate_1.gcode'))
    except (KeyError,OSError):return False


def deposition(path):
    """Track Tn Hm material commands separately from physical T1000/T1001.

    Include arc extrema and bead half-width in each physical nozzle's bounds.
    Native startup/purge service motion is excluded from build-volume tests.
    """
    pos=np.zeros(3);absolute=True;relative_e=True;e=0.;tool=None;feature='';width=.62
    cfg=policy()
    failures=[];counts=Counter();bounds={};interface_layers=defaultdict(Counter);bed=[];layer_z=0.
    with path.open() as stream:
        for line_no,raw in enumerate(stream,1):
            if raw.startswith('; FEATURE: '):feature=raw[11:].strip().lower()
            if raw.startswith('; LINE_WIDTH:'):width=float(raw.split(':',1)[1])
            if raw.startswith('; Z_HEIGHT:'):layer_z=float(raw.split(':',1)[1])
            text=raw.split(';',1)[0].strip()
            if not text:continue
            command=text.split()[0]
            match=re.fullmatch(r'T(\d+)',command)
            if match and int(match[1]) in (0,1):tool=int(match[1])
            if command=='G90':absolute=True
            if command=='G91':absolute=False
            if command=='M83':relative_e=True
            if command=='M82':relative_e=False
            if command in ('M140','M190'):
                match=re.search(r'\bS('+NUMBER+r')',text)
                if match and float(match[1])!=0:bed.append(float(match[1]))
            args={k:float(v) for k,v in ARG.findall(text)}
            if command=='G92':
                if 'E' in args:e=args['E']
                for i,k in enumerate('XYZ'):
                    if k in args:pos[i]=args[k]
                continue
            if command not in ('G0','G1','G2','G3'):continue
            old=pos.copy()
            for i,k in enumerate('XYZ'):
                if k in args:pos[i]=args[k] if absolute else pos[i]+args[k]
            delta=args.get('E',0.) if relative_e else args.get('E',e)-e
            if 'E' in args:e=args['E'] if not relative_e else e+args['E']
            if delta<=0 or not any(k in args for k in ('X','Y','I','J')):continue
            if not feature or feature in {'custom','undefined'} or 'purge' in feature:continue
            kind='interface' if feature.startswith('support interface') else 'support' if feature.startswith('support') else 'tower' if 'tower' in feature else 'model'
            expected=1 if kind=='interface' else 0
            if tool not in (0,1) or (kind!='tower' and tool!=expected):
                failures.append(dict(line=line_no,kind=kind,material_tool=tool,expected=expected));continue
            counts[kind]+=1
            if kind=='interface':interface_layers[round(layer_z,5)][tool]+=1
            points=[old.copy(),pos.copy()]
            if command in ('G2','G3') and ('I' in args or 'J' in args):
                center=old[:2]+[args.get('I',0.),args.get('J',0.)]
                radius=np.linalg.norm(old[:2]-center)
                a=math.atan2(old[1]-center[1],old[0]-center[0]);b=math.atan2(pos[1]-center[1],pos[0]-center[0])
                sweep=(b-a)%(2*math.pi) if command=='G3' else (a-b)%(2*math.pi)
                if np.linalg.norm(old[:2]-pos[:2])<1e-7:sweep=2*math.pi
                for q in (0.,math.pi/2,math.pi,3*math.pi/2):
                    distance=(q-a)%(2*math.pi) if command=='G3' else (a-q)%(2*math.pi)
                    if distance<=sweep+1e-8:points.append(np.array([center[0]+radius*math.cos(q),center[1]+radius*math.sin(q),pos[2]]))
            points=np.asarray(points);lo=points.min(axis=0);hi=points.max(axis=0)
            lo[:2]-=width/2;hi[:2]+=width/2
            previous=bounds.get(tool,(lo.copy(),hi.copy()))
            bounds[tool]=(np.minimum(previous[0],lo),np.maximum(previous[1],hi))
            region=cfg['build_regions_mm']['left' if tool==0 else 'right']
            if lo[0]<region[0]-.05 or lo[1]<region[1]-.05 or hi[0]>region[2]+.05 or hi[1]>region[3]+.05 or hi[2]>320.05:
                failures.append(dict(line=line_no,kind=kind,material_tool=tool,bounds=[lo.tolist(),hi.tolist()],region=region))
    if not counts['model']:failures.append('no model extrusion')
    if not bed or any(abs(t-cfg['bed_temperature_c'])>.01 for t in bed):failures.append(dict(bed_commands=sorted(set(bed))))
    if counts['support'] and not counts['interface']:failures.append('support has no PLA interface')
    result=dict(status='pass' if not failures else 'fail',counts=dict(counts),
        deposition_bounds_mm={str(k):[a.tolist(),b.tolist()] for k,(a,b) in bounds.items()},
        interface_layers_mm=sorted(interface_layers),bed_temperatures_c=sorted(set(bed)),failures=failures[:20])
    if failures:raise ValueError(json.dumps(result))
    return result


def regular_specs(job):
    stl=ROOT/job['stl'];auth=json.loads(stl.with_suffix('.print.json').read_text())
    matrix=np.asarray(auth['source_to_stl_matrix']);matrix[:3,3]+=job['offset']
    mesh=trimesh.load_mesh(stl,process=True)
    installed=mesh.copy();installed.apply_transform(np.linalg.inv(np.asarray(auth['source_to_stl_matrix'])))
    catalogs=[ROOT/'review/captive_magnet_release_catalog.json']
    catalogs+=list((ROOT/'build/bmr_crescent_TEBM35C10-4').glob('*.catalog.json'))
    catalogs+=list((ROOT/'build/vase_TEBM35C10-4').glob('*/*.catalog.json'))
    sites=[]
    for path in catalogs:
        for artifact in json.loads(path.read_text()).get('artifacts',[]):
            p=artifact['part'];name=job['name'];family=job['family']
            match=False
            if 'upper_bmr' in name:match='vase_TEBM' in p and family in str(path)
            elif 'bmr_crescent' in name:match=p in name
            elif family in {'stock','slim'}:
                # Shelf names number assembly choices differently from the
                # canonical magnet catalog. Match the actual source part.
                original=json.loads((ROOT/job['source']).with_suffix('.print.json').read_text())
                match=(p.startswith(family) and (('upper' in name and 'vase' in p) or p==original['part']))
            elif 'wing' in name:
                match='wing' in p and all(t in p for t in ('flat' if 'flat' in name else 'graded','left' if 'left' in name else 'right'))
            elif 'core_lm_carrier' in name:match='optional_lm_keyed' in p and artifact['state']==job['state']
            else:match=p in name
            if match:
                for s in artifact['sites']:sites.append((artifact['id'],s))
    specs=[]
    for i,cavity in enumerate(installed.split(only_watertight=False)):
        if cavity.volume>=0:continue
        if not sites:raise ValueError(f"No magnet authority for {job['name']}")
        source,s=min(sites,key=lambda r:np.linalg.norm(np.asarray(r[1]['cavity_center_xyz_mm'])-cavity.center_mass))
        distance=float(np.linalg.norm(np.asarray(s['cavity_center_xyz_mm'])-cavity.center_mass))
        if distance>4:raise ValueError((job['name'],'unmatched cavity',distance,cavity.center_mass))
        center=np.asarray(s['seated_magnet_center_xyz_mm']);axis=np.asarray(s['installed_marked_pole_axis_xyz'])
        basis=trimesh.geometry.align_vectors([0,0,1],axis);basis[:3,3]=center
        magnet=trimesh.creation.cylinder(radius=s['magnet_diameter_mm']/2,height=s['magnet_depth_mm'],sections=128,transform=basis)
        magnet.apply_transform(matrix)
        approach=matrix[:3,:3]@np.array([0,0,-1.]);approach/=np.linalg.norm(approach)
        sweep=trimesh.convex.convex_hull(np.r_[magnet.vertices,magnet.vertices+approach*45])
        specs.append(dict(name=s['name'],diameter_mm=s['magnet_diameter_mm'],depth_mm=s['magnet_depth_mm'],
            center_bed_mm=trimesh.transform_points([center],matrix)[0].tolist(),pole_axis_bed=(matrix[:3,:3]@axis).tolist(),
            approach_axis_bed=approach.tolist(),seated_top_z_mm=float(magnet.bounds[1,2]),seated_bottom_z_mm=float(magnet.bounds[0,2]),
            source_site=source,site_registration_error_mm=distance,_magnet=magnet,_sweep=sweep))
    if len(specs)!=job['magnet_count']:raise ValueError('Magnet count mismatch')
    return specs


def insert_pauses(project,records):
    groups=defaultdict(list)
    for r in records:
        if not r['magnet_below_resume_plane'] or not r['opening_clear_through_prior_layers']:raise ValueError(r)
        groups[r['pause_before_z_mm']].append(r['name'])
    root=ET.Element('custom_gcodes_per_layer');plate=ET.SubElement(root,'plate');ET.SubElement(plate,'plate_info',id='1')
    for z,names in sorted(groups.items()):
        program=(f'; LX521_H2C_MAGNET_INSERTION\n; Insert {len(names)}: '+', '.join(names)+
            f'\nG90\nM400\nG1 Z300 F1200\nM400\nM400 U1\nG1 Z{z:.2f} F1200\nM400\n; LX521_H2C_MAGNET_RESUME')
        ET.SubElement(plate,'layer',top_z=f'{z:.2f}',type='4',extruder='1',color='',extra=program,gcode=program)
    ET.SubElement(plate,'mode',value='MultiExtruder')
    temporary=project.with_suffix('.tmp.3mf')
    with zipfile.ZipFile(project) as old,zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as new:
        for n in old.namelist():
            if n!='Metadata/custom_gcode_per_layer.xml':new.writestr(n,old.read(n))
        if groups:new.writestr('Metadata/custom_gcode_per_layer.xml',ET.tostring(root,encoding='utf-8'))
    temporary.replace(project)
    return sorted(groups)


def pause_audit(gcode,expected):
    from gcode_analysis import parse_gcode
    from artifact_emit import _assert_pauses_precede_layer_extrusion
    lines=gcode.read_text().splitlines();events=[];z=None
    for i,line in enumerate(lines):
        if line.startswith('; Z_HEIGHT:'):z=float(line.split(':',1)[1])
        if line!='; LX521_H2C_MAGNET_INSERTION':continue
        end=next((k for k in range(i+1,min(i+20,len(lines))) if lines[k]=='; LX521_H2C_MAGNET_RESUME'),None)
        if end is None:raise ValueError('Incomplete magnet park/pause/restore block')
        block=lines[i:end]
        if block.count('M400 U1')!=1:raise ValueError('Expected one operator pause')
        park=block.index('G1 Z300 F1200');pause=block.index('M400 U1')
        restore=next(k for k,s in enumerate(block) if re.fullmatch(r'G1 Z'+re.escape(f'{z:.2f}')+r'(?: F1200)?',s))
        if not park<pause<restore:raise ValueError('Magnet pause order changed')
        events.append(dict(z_mm=z,command_line_number=i+pause+1))
    if [e['z_mm'] for e in events]!=expected:raise ValueError(('Actual magnet pause mismatch',events,expected))
    # Retain no model segments, but count all extrusion and layer boundaries.
    parsed=parse_gcode(gcode,retain_regions=[(-100,-100,-99,-99)])
    return _assert_pauses_precede_layer_extrusion(parsed,events)


def support_audit(job,gcode):
    from gcode_analysis import audit_support_toolpaths_vs_ducts
    report={}
    if job['role']=='crescent_body':
        from audit_print import support_ducts
        prep=dict(job['preparation'],source_to_stl_matrix=job['source_to_stl_matrix'])
        report['ducts_and_insert_bores']=support_ducts(gcode,prep)
    elif job['role']=='crescent_accessories':
        from cap_support_check import ceiling_supports
        report['cap_ceilings']=ceiling_supports(gcode,job['preparation'])
    else:
        for source in job.get('blockers',[]):
            if not source.endswith('.json'):continue
            contract=json.loads((ROOT/source).read_text()).get('duct_collision_contract')
            if not contract:continue
            shift=np.eye(4);shift[:3,3]=job['offset']
            report[source]=audit_support_toolpaths_vs_ducts(gcode=gcode,contract=contract,
                source_to_stl_matrix=job['source_to_stl_matrix'],stl_to_bed_matrix=shift.tolist())
        definition=ROOT/'build/h2c/contracts'/(job['name']+'.bores.json')
        if job['family'] in {'stock','slim'} and job['role'] not in {'regular_wing'}:
            if not definition.exists():raise ValueError(f'Missing native bore audit authority: {definition}')
            from gcode_analysis import parse_gcode
            data=json.loads(definition.read_text());matrix=np.asarray(job['source_to_stl_matrix']).copy();matrix[:3,3]+=job['offset']
            step=ROOT/'build/h2c/STEP'/(job['name']+'.step')
            if sha256_file(step)!=data['step_sha256']:raise ValueError('Stale native bore authority')
            bores=[]
            for bore in data['bores']:
                a,b=trimesh.transform_points([bore['start'],bore['end']],matrix);axis=b-a
                if np.linalg.norm(axis)<1e-6:continue
                bores.append((bore,a,axis,np.dot(axis,axis)))
            parsed=parse_gcode(gcode,retain_feature_prefixes=('support',));minimum=None;count=0
            for layer in parsed.layers:
                chunks=[];widths=[]
                for s in layer.segments:
                    a=np.array([s.x0,s.y0,layer.z-(layer.layer_height or .16)/2]);b=np.array([s.x1,s.y1,a[2]])
                    n=max(2,int(np.ceil(s.length/.25))+1)
                    chunks.append(a+(b-a)*np.linspace(0,1,n)[:,None]);widths.extend([(s.line_width or .62)/2]*n)
                if not chunks:continue
                points=np.concatenate(chunks);widths=np.asarray(widths)
                for bore,a,axis,length2 in bores:
                    t=(points-a)@axis/length2;inside=(t>1e-5)&(t<1-1e-5)
                    if not inside.any():continue
                    delta=points[inside]-a-t[inside,None]*axis
                    distance=np.linalg.norm(delta,axis=1)-bore['radius_mm']-widths[inside]
                    m=float(distance.min());minimum=m if minimum is None else min(m,minimum);count+=int(inside.sum())
                    if m<-.03:raise ValueError(('Support enters native bore',job['name'],bore['name'],layer.z,m))
            report['native_bores']=dict(status='pass',bore_count=len(bores),support_samples=count,minimum_clearance_mm=minimum,
                definition_sha256=sha256_file(definition),method='0.25 mm support samples, bead width and native cylinder interiors at each layer midplane')
    return report


def static_validation(gcode,job):
    from artifact_emit import _gcode_tool_path
    profiles=ROOT/'build/h2c/profiles'/job['lane'];work=ROOT/job['work']
    # Y=-16 is the native H2C trash-bin clearance in machine_start_gcode.
    # This service bound is separate from the per-nozzle deposition bounds.
    wrapper=dict(backend='orcaslicer',native_config=str(profiles/'machine.json'),
        native_settings=[str(profiles/'machine.json'),str(work/'process.json')],
        native_filaments=[str(profiles/'model.json'),str(profiles/'interface.json')],
        machine=dict(name='Bambu Lab H2C 0.6 High Flow',bed_size_mm=[330,320],z_height_mm=325,
            motion_bounds_mm=dict(x=[0,330],y=[-16,320],z=[0,325])),
        filament=dict(type='PETG',nozzle_temp_c=260,bed_temp_c=70))
    write_json(work/'gcode_validation_profile.json',wrapper)
    tool=_gcode_tool_path()
    if tool is None:raise ValueError('Required static G-code validator is unavailable')
    cmd=[sys.executable,str(tool),'validate','--gcode',str(gcode),'--profile',str(work/'gcode_validation_profile.json'),'--json']
    run=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    result=json.loads(run.stdout);write_json(work/'gcode_skill_validation.json',result)
    if not result.get('ok'):raise ValueError(('Static G-code check failed',result))
    return result


def run_job(job):
    work=ROOT/job['work'];project=ROOT/job['project'];command=json.loads((work/'dry_run.json').read_text())
    for source in job['preparation']['sources']:
        if sha256_file(ROOT/source['path'])!=source['sha256']:
            raise ValueError(('Prepared geometry changed; regenerate the project',source['path']))
    def run(output):
        cmd=command.copy();cmd[cmd.index('--export-3mf')+1]=output
        write_json(work/(output+'.command.json'),cmd)
        path=work/output
        profiles=ROOT/'build/h2c/profiles'/job['lane']
        inputs={str(p.relative_to(ROOT)):sha256_file(p) for p in [project,work/'process.json',work/'dry_run.json',
            *(profiles/(n+'.json') for n in ('machine','model','interface'))]}
        binding=work/(output+'.inputs.json')
        cached=json.loads(binding.read_text()) if binding.exists() else {}
        if not path.exists() or cached.get('inputs')!=inputs or cached.get('output_sha256')!=sha256_file(path):
            with (work/(output+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            write_json(binding,dict(inputs=inputs,output_sha256=sha256_file(path)))
        with zipfile.ZipFile(path) as z:(work/'plate_1.gcode').write_bytes(z.read('Metadata/plate_1.gcode'))
        return path
    print('AUDIT',job['id'],flush=True)
    discovery=run('ready.gcode.3mf' if (work/'ready.gcode.3mf').exists() else 'discovery.gcode.3mf');gcode=work/'plate_1.gcode'
    paths=deposition(gcode)
    specs=[];records=[];pause_z=[]
    if job['magnet_count']:
        if job['family']=='v4':
            stl=ROOT/job['stl'];auth=json.loads(stl.with_suffix('.print.json').read_text())
            specs=magnet_geometry(stl,auth,job['offset'],owner='body' if job['role']=='crescent_body' else 'wing')
        else:specs=regular_specs(job)
        records=discover_specs(specs,gcode)
        write_json(work/'magnet_discovery.json',records)
        pause_z=insert_pauses(project,records)
        sliced=run('ready.gcode.3mf');paths=deposition(gcode)
        repeated=discover_specs(specs,gcode)
        if [r['pause_before_z_mm'] for r in repeated]!=[r['pause_before_z_mm'] for r in records]:raise ValueError('Pauses changed closure timing')
    else:sliced=discovery
    with zipfile.ZipFile(sliced) as z:settings=json.loads(z.read('Metadata/project_settings.config'))
    for key,value in dict(printer_model='Bambu Lab H2C',nozzle_diameter=['0.6','0.6'],nozzle_volume_type=['High Flow','High Flow'],
        filament_map=['1','2'],support_filament='1',support_interface_filament='2',flush_into_objects='0',flush_into_infill='0',flush_into_support='0').items():
        if settings.get(key)!=value:raise ValueError((key,settings.get(key),value))
    with zipfile.ZipFile(project) as z:expected=json.loads(z.read('Metadata/project_settings.config'))
    directory=ROOT/'build/h2c/profiles'/job['lane']
    bundle=[json.loads((directory/'machine.json').read_text()),json.loads((directory/'process.json').read_text()),
        [json.loads((directory/'model.json').read_text()),json.loads((directory/'interface.json').read_text())]]
    current_policy=job_settings(bundle,job['lane'],job['role'],job['name'],
        brim=job.get('brim_mm',5),tower=job.get('tower',(270,255)))
    for key in ('sparse_infill_density','sparse_infill_pattern','wall_generator','support_top_z_distance',
                'support_bottom_z_distance','support_interface_top_layers','support_interface_bottom_layers','eng_plate_temp',
                'eng_plate_temp_initial_layer','enable_support','support_type','support_style','support_interface_spacing',
                'support_bottom_interface_spacing','support_object_xy_distance','filament_type','filament_settings_id',
                'nozzle_temperature','nozzle_temperature_initial_layer','filament_flow_ratio','filament_max_volumetric_speed',
                'layer_height','initial_layer_print_height','wall_loops','wall_sequence','top_shell_layers','bottom_shell_layers','brim_width'):
        wanted=expected.get(key,policy()['native_process_defaults'].get(key))
        if wanted!=current_policy.get(key):raise ValueError(('Prepared project differs from current policy',key,wanted,current_policy.get(key)))
        if settings.get(key)!=wanted:raise ValueError((key,settings.get(key),wanted))
    walls=audit_captive_walls(ROOT/job['stl'],job['offset'],records,gcode,diameters=(5,6)) if records else None
    # Geometry equivalence checks the oriented triangle soup, including every
    # nonprinting modifier/blocker, after Bambu's internal recentering.
    from audit_print import geometry
    geometry_report=geometry(sliced,job['preparation'])
    supports=support_audit(job,gcode)
    pauses=pause_audit(gcode,pause_z)
    static=static_validation(gcode,job)
    target=project.with_name(project.stem+'.gcode.3mf');shutil.copy2(sliced,target)
    report=dict(status='pass',qualification='digital; physical H2C qualification pending',
        project_sha256=sha256_file(project),sliced_project_sha256=sha256_file(target),gcode_sha256=sha256_file(gcode),
        audit_version=AUDIT_VERSION,inputs=qualification_inputs(job),deposition=paths,magnet_pauses_mm=pause_z,magnet_discovery=records,retaining_walls=walls,
        geometry=geometry_report,supports=supports,pause_program=pauses,static_validation=static)
    write_json(work/'audit.json',report)
    job.pop('error',None)
    job.update(status='sliced_audited',audit_version=AUDIT_VERSION,project_sha256=sha256_file(project),
        sliced_project=str(target.relative_to(ROOT)),audit=str((work/'audit.json').relative_to(ROOT)))
    print('PASS',job['id'],flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--only',action='append')
    ap.add_argument('--worker-job');ap.add_argument('--workers',type=int,choices=(1,2),default=1)
    args=ap.parse_args()
    path=ROOT/'to_print/h2c/catalog.json';catalog=json.loads(path.read_text())
    if args.worker_job:
        job=next(j for j in catalog['jobs'] if j['id']==args.worker_job)
        try:run_job(job)
        except Exception as error:
            import traceback
            traceback.print_exc()
            job.update(status='slice_or_audit_failed',error=str(error))
            # A rejected revision must not leave an older, apparently ready
            # slice beside its editable project on the manufacturing shelf.
            delivered=(ROOT/job['project']).with_name(Path(job['project']).stem+'.gcode.3mf')
            if delivered.exists():
                rejected=ROOT/job['work']/'rejected_delivery'/(sha256_file(delivered)[:12]+'_'+delivered.name)
                rejected.parent.mkdir(parents=True,exist_ok=True);delivered.replace(rejected)
            for key in ('sliced_project','audit','audit_version'):job.pop(key,None)
            write_json(ROOT/job['work']/'failure.json',dict(error=str(error),project_sha256=sha256_file(ROOT/job['project'])))
            print('FAIL',job['id'],str(error)[:1500],flush=True)
        write_json(ROOT/job['work']/'job_result.json',job)
        return
    jobs=[]
    for job in catalog['jobs']:
        if args.only and job['name'] not in args.only and job['id'] not in args.only:continue
        if audit_is_current(job):continue
        jobs.append(job)
    def execute(job):
        result=ROOT/job['work']/'job_result.json'
        if result.exists():result.unlink()
        subprocess.run([sys.executable,str(Path(__file__)),'--worker-job',job['id']],check=True)
        return json.loads(result.read_text())
    # Separate processes avoid Python's GIL. Two workers stay below the
    # workstation's 8 GiB aggregate budget; all catalog writes stay here.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(execute,j):j for j in jobs}
        for future in as_completed(futures):
            job=futures[future]
            try:job.update(future.result())
            except Exception as error:job.update(status='slice_or_audit_failed',error=str(error))
            write_json(path,catalog)
    if any(j['status']!='sliced_audited' for j in jobs):sys.exit(1)


if __name__=='__main__':main()
