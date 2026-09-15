#!/usr/bin/env python3
"""Build the H2C shelf from canonical CAD and approved mesh authorities.

No printer calls. Preparation, slicing and qualification have separate status
fields. All geometry transforms and each source hash are recorded explicitly.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
import hashlib
import math
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT / 'candidates/nd25fn4_crescent')]
import numpy as np
import trimesh
from shapely.geometry import MultiPoint, box
from lx521_baffle.io import sha256_file
from lx521_baffle.print_contract import front_down_transform_record, validate_print_sidecar
from lx521_baffle.h2c.printing import resolve_profiles, project_settings, job_settings, job_process, write_json, policy
from prepare_print import write_project

OUT = ROOT / 'to_print/h2c'
WORK = ROOT / 'build/h2c'
BIN = '/Applications/BambuStudio.app/Contents/MacOS/BambuStudio'


def rel(p): return str(Path(p).relative_to(ROOT))


def preparation_inputs(paths, bundle, definition):
    files=set(paths)
    for path in list(files):
        sidecar=path.with_suffix('.print.json')
        if sidecar.exists():files.add(sidecar)
    files.update(ROOT/p for p in ('print_policy_h2c.json','print_policy.json',
        'scripts/build_h2c_release.py','src/lx521_baffle/h2c/printing.py',
        'candidates/nd25fn4_crescent/prepare_print.py','candidates/nd25fn4_crescent/mesh_ops.py'))
    return dict(files={rel(p):sha256_file(p) for p in sorted(files)},definition=definition,
        profiles_sha256=hashlib.sha256(json.dumps(bundle,sort_keys=True).encode()).hexdigest())


def prepared_is_current(job, inputs):
    if not job or job.get('prepare_inputs')!=inputs:return False
    try:
        return (sha256_file(ROOT/job['project'])==job.get('project_sha256')
            and all(sha256_file(ROOT/s['path'])==s['sha256'] for s in job['preparation']['sources']))
    except OSError:return False


def source_mesh(path):
    """Read a hash-checked source and restore its installed coordinates."""
    authority = json.loads(path.with_suffix('.print.json').read_text())
    if authority['stl_sha256'] != sha256_file(path): raise ValueError(f'Stale source {path}')
    mesh = trimesh.load_mesh(path, process=True)
    mesh.apply_transform(np.linalg.inv(np.asarray(authority['source_to_stl_matrix'])))
    return mesh, authority


def v4_wings():
    """Replace the old lower split with the actual canonical continuous wing.

    The join lies below the V4 clearance cutter's Y308.4 start, above all
    preserved LM magnets. The 0.2 mm overlap is in unchanged source material.
    Buried cavities keep their inward winding throughout the Boolean.
    """
    from export_piece_stls import _strict_mesh_facts
    from mesh_ops import solid, to_trimesh, preserve_void_winding
    for slug in ('flat', 'graded'):
        for side in ('left', 'right'):
            a = WORK / f'STL/h2c_obiwan_wing_{slug}_{side}.stl'
            b = ROOT / f'candidates/nd25fn4_crescent/print/geometry/V4_{slug}_{side}_UPPER_PRINT.stl'
            name = f'h2c_v4_wing_{slug}_{side}'
            target = WORK / 'STL' / (name + '.stl')
            inputs = {rel(p):sha256_file(p) for p in (a,b)}
            if target.exists() and target.with_suffix('.print.json').exists():
                previous=json.loads(target.with_suffix('.print.json').read_text())
                if previous.get('source_hashes') == inputs and previous.get('boolean_version') == 2:continue
            lower, _ = source_mesh(a); upper, _ = source_mesh(b)
            below = trimesh.creation.box([500, 808.2, 500], transform=trimesh.transformations.translation_matrix([0, (308.2-500)/2, 0]))
            above = trimesh.creation.box([500, 292, 500], transform=trimesh.transformations.translation_matrix([0, (308+600)/2, 0]))
            low = trimesh.boolean.intersection([lower, below], engine='manifold')
            high = trimesh.boolean.intersection([upper, above], engine='manifold')
            mesh = trimesh.boolean.union([low, high], engine='manifold')
            mesh = to_trimesh(solid(mesh).simplify(.0001))
            if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume <= 0:
                raise ValueError(f'Invalid V4 monolithic wing {name}')
            components = mesh.split(only_watertight=False)
            if sum(c.volume > 0 for c in components) != 1 or sum(c.volume < 0 for c in components) != 4:
                raise ValueError(f'{name}: expected one body and four buried magnet cavities')
            posed = mesh.copy(); posed.apply_transform(np.diag([1., -1., -1., 1.]))
            minimum = posed.bounds[0].copy(); posed.apply_translation(-minimum)
            posed=to_trimesh(solid(posed).simplify(.001))
            preserve_void_winding(posed).export(target)
            facts = _strict_mesh_facts(target)
            authority = dict(schema_version=1, part=name, stl=target.name,
                stl_bytes=target.stat().st_size, stl_sha256=sha256_file(target),
                source_hashes=inputs, join_y_mm=[308.,308.2], mesh=facts,
                boolean_simplification_tolerance_mm=.001,boolean_version=2,
                **front_down_transform_record(minimum.tolist()))
            write_json(target.with_suffix('.print.json'), authority)
            print('V4 continuous wing:', name, flush=True)


def inventory():
    rows = []
    def add(name, family, role, source, *, state='shared', candidate=False, blockers=(), angle=None):
        angle=policy().get('print_orientation_z_deg',{}).get(name,angle)
        rows.append(dict(name=name, family=family, role=role, source=rel(source),
            state=state, candidate=candidate, blockers=list(blockers), angle=angle))
    for family in ('stock', 'slim'):
        for state in ('no_floor_stand',):
            name=f'h2c_{family}_lm_{state}'
            add(name,family,'lm_bottom',WORK/'STL'/f'{name}.stl',state=state,
                blockers=[rel(WORK/'support_blockers'/f'{name}.stl')],angle=90.)
        for owner in ('lm_lower','lm_upper'):
            name=f'h2c_{family}_{owner}_floor_stand'
            add(name,family,'lm_bottom' if owner=='lm_lower' else 'lm_top',WORK/'STL'/f'{name}.stl',state='floor_stand',
                blockers=[rel(WORK/'support_blockers'/f'{name}.stl')])
        name=f'h2c_{family}_upper'
        add(name,family,'um',WORK/'STL'/f'{name}.stl',blockers=[rel(WORK/'support_blockers'/f'{name}.stl')],angle=0.)
        name=f'h2c_{family}_upper_bmr'
        add(name,family,'um',WORK/'STL'/f'{name}.stl',candidate=True,blockers=[rel(WORK/'support_blockers'/f'{name}.stl')],angle=0.)
    old=json.loads((ROOT/'to_print/catalog.json').read_text())
    for e in old['entries']:
        family=e['family']; name=e['name']; src=e['source_stl']
        if family in {'stock','slim'} and any(k in name for k in ('shoulder','B1_wing')):
            source=ROOT/('build/'+src)
            add('h2c_'+name,family,'regular_wing',source,angle=0.)
    for state in ('no_floor_stand','floor_stand'):
        name=f'h2c_obiwan_core_lm_carrier_{state}'
        blockers=[]
        for part in ('optional_lm_keyed_1_of_2_bottom','optional_lm_keyed_2_of_2_top'):
            path=ROOT/f'build/{state}/support_blockers/obiwan_{part}.support_blocker.json'
            blockers.append(rel(path))
        add(name,'obiwan','lm_bottom',WORK/'STL'/f'{name}.stl',state=state,blockers=blockers)
    for owner, role, state in [('core_2_of_2_um_carrier','um','no_floor_stand'),
                               ('addon_tweeter_crescent','regular_tweeter','no_floor_stand'),
                               ('addon_nl8_service_lid','lm_bottom','floor_stand')]:
        blockers=[]
        path=ROOT/f'build/{state}/support_blockers/obiwan_{owner}.support_blocker.json'
        if path.exists():blockers=[rel(path)]
        add(f'h2c_obiwan_{owner}','obiwan',role,ROOT/f'build/{state}/stl/obiwan_{owner}.stl',
            state='floor_stand' if owner=='addon_nl8_service_lid' else 'shared',blockers=blockers,angle=0.)
    for slug in ('flat','graded'):
        for side in ('left','right'):
            name=f'h2c_obiwan_wing_{slug}_{side}'
            add(name,'obiwan','regular_wing',WORK/'STL'/f'{name}.stl')
            name=f'h2c_v4_wing_{slug}_{side}'
            add(name,'v4','crescent_wing',WORK/'STL'/f'{name}.stl')
    for stem in ('obiwan_bmr_crescent_TEBM35C10-4','obiwan_bmr_crescent_opposed_TEBM35C10-4'):
        add('h2c_'+stem,'obiwan','regular_tweeter',ROOT/'build/bmr_crescent_TEBM35C10-4'/f'{stem}.stl',candidate=True)
    return rows


def placement(mesh, brim, fixed_angle=None, full_left=False):
    """Conservative hull/brim/tower layout; toolpaths are checked separately."""
    cfg=policy();margin=cfg['edge_margin_mm']
    region=cfg['build_regions_mm']['left' if full_left else 'shared']
    area=[region[0]+margin,region[1]+margin,region[2]-margin,region[3]-margin]
    width,height=area[2]-area[0],area[3]-area[1]
    tower_side=cfg['prime_tower_reserve_mm']['side'];tower_inset=cfg['prime_tower_reserve_mm']['origin_inset']
    shared=cfg['build_regions_mm']['shared']
    tower_area=[shared[0]+margin,shared[1]+margin,shared[2]-margin,shared[3]-margin]
    candidates=[]
    outline=np.c_[np.asarray(MultiPoint(mesh.vertices[:,:2]).convex_hull.exterior.coords)[:,:2],
                  np.zeros(len(MultiPoint(mesh.vertices[:,:2]).convex_hull.exterior.coords))]
    for angle in ([fixed_angle] if fixed_angle is not None else range(-90,91,1)):
        rot=trimesh.transformations.rotation_matrix(math.radians(angle),[0,0,1]) @ np.diag([1.,-1.,-1.,1.])
        hull=MultiPoint(trimesh.transform_points(outline,rot)[:,:2]).convex_hull
        lo=np.array(hull.bounds[:2]); hi=np.array(hull.bounds[2:]); size=hi-lo
        if size[0]+2*brim>width+1e-3 or size[1]+2*brim>height+1e-3:continue
        offset=np.array([(area[0]+area[2]-size[0])/2,(area[1]+area[3]-size[1])/2,0])
        from shapely.affinity import translate
        occupied=translate(hull,xoff=offset[0]-lo[0],yoff=offset[1]-lo[1]).buffer(brim+2)
        # H2C can prime outside the configured tower origin. Reserve 60 mm
        # with a 7 mm inset, including the measured 4.375 mm lead-in bead.
        tower=None
        if full_left:
            # The large LM has a through driver opening. A tower can sit in
            # that empty aperture; prove the entire reserve clear against
            # every projected model triangle, including the stand at rear.
            import shapely
            center=trimesh.transform_points([[0.,200.981,0.]],rot)[0,:2]+offset[:2]-lo
            reserve=box(*(center-tower_side/2),*(center+tower_side/2))
            xy=trimesh.transform_points(mesh.vertices,rot)[:,:2]+offset[:2]-lo
            triangles=xy[mesh.faces]
            clearance=reserve.buffer(2)
            hit=(triangles.max(axis=1)>=np.array(clearance.bounds[:2])).all(axis=1)&(triangles.min(axis=1)<=np.array(clearance.bounds[2:])).all(axis=1)
            if box(*tower_area).contains(reserve) and not shapely.intersects(shapely.polygons(triangles[hit]),clearance).any():
                tower=tuple(float(v-tower_side/2+tower_inset) for v in center)
        for x in np.arange(tower_area[0],tower_area[2]-tower_side+.1,5):
            if tower:break
            for y in np.arange(tower_area[1],tower_area[3]-tower_side+.1,5):
                if not occupied.intersects(box(x,y,x+tower_side,y+tower_side)):
                    tower=(float(x+tower_inset),float(y+tower_inset));break
            if tower:break
        if tower is None:continue
        minimum=np.array([lo[0],lo[1],-mesh.bounds[1,2]])
        score=min(width-size[0]-2*brim,height-size[1]-2*brim)
        candidates.append((score,float(angle),minimum,offset,tower))
    if not candidates: raise ValueError('No H2C hull/brim/tower layout found')
    return max(candidates,key=lambda c:c[0])


def export_posed(row):
    path=ROOT/row['source']; mesh,source_authority=source_mesh(path)
    large=row['family'] in {'stock','slim'} and row['name'].endswith('_lm_no_floor_stand')
    brim=policy()['brim_mm']['stock_slim_no_floor_lm' if large else 'default']
    _,angle,minimum,offset,tower=placement(mesh,brim,row['angle'],large)
    transform=front_down_transform_record(minimum.tolist(),z_rotation_deg=angle)
    matrix=np.asarray(transform['source_to_stl_matrix'])
    mesh.apply_transform(matrix)
    target=OUT/'STL'/(row['name']+'.stl');target.parent.mkdir(parents=True,exist_ok=True)
    mesh.export(target)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError(target)
    components=mesh.split(only_watertight=False)
    if sum(c.volume>0 for c in components)!=1:raise ValueError(f'{target}: disconnected material')
    authority=dict(schema_version=1,part=target.stem,stl=target.name,stl_bytes=target.stat().st_size,
        stl_sha256=sha256_file(target),source_stl=row['source'],source_sha256=sha256_file(path),
        **transform)
    write_json(target.with_suffix('.print.json'),authority)
    validate_print_sidecar(target)
    group=[(target,'normal_part',{},[0,0,0])]
    blockers=[]; blocker_sources=[]
    for ref in row['blockers']:
        src=ROOT/ref
        if src.suffix=='.json':
            cfg=json.loads(src.read_text())
            # These blockers share the source print transform in the record.
            source=src.with_suffix('.stl')
            old_matrix=np.asarray(cfg['source_to_stl_matrix'])
        else:
            source=src;old_matrix=np.asarray(source_authority['source_to_stl_matrix'])
        part=trimesh.load_mesh(source,process=True)
        # Native cutter tessellation can contain zero-area apex triangles.
        # Remove only those collapsed facets before a transformed Boolean.
        part.update_faces(part.nondegenerate_faces(height=1.e-12))
        part.remove_unreferenced_vertices()
        # A blocker ending exactly at a duct ceiling can miss its overhang
        # facets. Dilate the nonprinting tools, preserving every model facet.
        clearance=policy()['support_blocker_extra_xyz_mm']
        from mesh_ops import solid, to_trimesh
        import manifold3d as manifold
        solids=[solid(c).simplify(.01) for c in part.split(only_watertight=False)]
        merged=manifold.Manifold.batch_boolean(solids,manifold.OpType.Add)
        part=to_trimesh(merged.minkowski_sum(manifold.Manifold.cube([2*clearance]*3,center=True)).simplify(.01))
        if not part.is_watertight or not part.is_winding_consistent:raise ValueError('Invalid dilated support blocker')
        part.apply_transform(matrix @ np.linalg.inv(old_matrix))
        blockers.append(part)
        blocker_sources.append(dict(path=rel(source),sha256=sha256_file(source)))
    if row['family'] in {'stock','slim'} and '_lm_lower_floor_stand' in row['name']:
        # Bambu blockers suppress overhang sources, not every support bead
        # passing through their volume. The rear service panel can otherwise
        # seed long support columns down the vertical cable entries. Carry
        # each entry's keepout to the rear-most panel surface so those small
        # spans bridge between their surrounding supported material.
        from lx521_baffle.cables import FOOT_LANES
        from lx521_baffle.floor_bend import canonical_lane_controls
        installed=mesh.copy();installed.apply_transform(np.linalg.inv(matrix))
        rear=float(installed.bounds[0,2])-1
        shadows=[]
        for name,(x,upright_z,y,_,diameter) in FOOT_LANES.items():
            top=canonical_lane_controls(x,y,upright_z)[0][2]
            radius=diameter/2+policy()['support_blocker_extra_xyz_mm']
            shadow=trimesh.creation.cylinder(radius=radius,height=top-rear,sections=96,
                transform=trimesh.transformations.translation_matrix([x,y,(top+rear)/2]))
            shadow.apply_transform(matrix);blockers.append(shadow)
            shadows.append(dict(route=name,center_xy_mm=[x,y],radius_mm=radius,rear_z_mm=rear,front_z_mm=float(top)))
        if len(shadows)!=3:raise ValueError(('Expected three floor cable-entry support shadows',shadows))
        row['floor_cable_support_shadows']=shadows
    # Preserve inward winding in the model; reverse a COPY only for the
    # non-printing positive blocker volumes inside the magnet cavities.
    for cavity in components:
        if cavity.volume < 0:
            cavity=cavity.copy();cavity.invert();blockers.append(cavity)
    if blockers:
        blocker=WORK/'inputs'/(row['name']+'_support_blocker.stl');blocker.parent.mkdir(parents=True,exist_ok=True)
        if row.get('floor_cable_support_shadows'):
            # Make overlapping entry/shadow masks one explicit volume.
            merged=manifold.Manifold.batch_boolean([solid(b) for b in blockers],manifold.OpType.Add)
            mask=to_trimesh(merged.simplify(.005))
            if not mask.is_watertight:raise ValueError('Open floor support mask')
            mask.export(blocker)
        else:trimesh.util.concatenate(blockers).export(blocker)
        group.append((blocker,'support_blocker',{},[0,0,0]))
    row.update(stl=rel(target),authority=rel(target.with_suffix('.print.json')),
        source_to_stl_matrix=matrix.tolist(),offset=offset.tolist(),tower=list(tower),
        brim_mm=brim,magnet_count=int(sum(c.volume<0 for c in components)),
        blocker_sources=blocker_sources,support_blocker_extra_xyz_mm=policy()['support_blocker_extra_xyz_mm'],
        print_size_mm=mesh.extents.tolist())
    return [group]


def slice_command(project, lane, work):
    profiles=WORK/'profiles'/lane
    return [BIN,'--debug','2','--slice','1','--arrange','0','--orient','0',
        '--mtcpp','8000000','--allow-mix-temp','--filament-map','1,2',
        '--load-settings',';'.join(map(str,[profiles/'machine.json',work/'process.json'])),
        '--load-filaments',';'.join(map(str,[profiles/'model.json',profiles/'interface.json'])),
        '--outputdir',str(work),'--export-3mf','discovery.gcode.3mf',str(project)]


def prepare(selected=None):
    v4_wings()
    bundles={lane:resolve_profiles(lane,WORK/'profiles'/lane) for lane in policy()['lanes']}
    rows=inventory()
    manifest_path=OUT/'catalog.json'
    previous=json.loads(manifest_path.read_text())['jobs'] if manifest_path.exists() else []
    retired={f'h2c_{f}_lm_floor_stand__petg_gf_pla' for f in ('stock','slim')}
    jobs={j['id']:j for j in previous if j['id'] not in retired}
    for job in previous:
        if job['id'] not in retired:continue
        for key in ('project','sliced_project','stl','authority'):
            path=ROOT/job[key] if key in job else None
            if path and path.exists() and path.is_relative_to(OUT):
                target=WORK/'experiments/rejected_one_piece_floor'/path.relative_to(OUT)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,target)
    for row in rows:
        if selected and row['name'] not in selected:continue
        lanes=tuple(bundles) if row['family']=='v4' else ('petg_gf_pla',)
        inputs={}
        sources=[ROOT/row['source']]
        contract=WORK/'contracts'/(row['name']+'.bores.json')
        if contract.exists():sources.append(contract)
        for p in row['blockers']:
            sources.append(ROOT/p)
            if p.endswith('.json'):sources.append((ROOT/p).with_suffix('.stl'))
        for lane in lanes:inputs[lane]=preparation_inputs(sources,bundles[lane],deepcopy(row))
        pending=[lane for lane in lanes if not prepared_is_current(jobs.get(row['name']+'__'+lane),inputs[lane])]
        if not pending:continue
        groups=export_posed(row)
        for lane in pending:
            settings=job_settings(bundles[lane],lane,row['role'],row['name'],brim=row['brim_mm'],tower=row['tower'])
            job_id=row['name']+'__'+lane
            work=WORK/'jobs'/job_id;work.mkdir(parents=True,exist_ok=True)
            project=OUT/row['family']/lane/(row['name']+'.3mf')
            prep=write_project(project,groups,settings,row['offset'])
            write_json(work/'process.json',job_process(bundles[lane],settings))
            command=slice_command(project,lane,work);write_json(work/'dry_run.json',command)
            jobs[job_id]=dict(id=job_id,lane=lane,status='prepared',**row,preparation=prep,project=rel(project),work=rel(work),
                prepare_inputs=inputs[lane],project_sha256=sha256_file(project))
            print('PREPARED',job_id,flush=True)
            write_json(manifest_path,dict(schema_version=1,printer='Bambu Lab H2C 0.6 High Flow',checkpoint='c261f32',jobs=list(jobs.values())))
    # Approved V4 body and accessory groups retain their exact local meshes.
    prep_old=json.loads((ROOT/'review/nd25fn4_print/preparation.json').read_text())
    for key,role in [('body','crescent_body'),('accessories','crescent_accessories')]:
        name='h2c_v4_'+key
        if selected and name not in selected:continue
        r=deepcopy(prep_old[key])
        inputs={lane:preparation_inputs([ROOT/s['path'] for s in r['sources']],bundle,dict(key=key,preparation=r))
            for lane,bundle in bundles.items()}
        pending=[lane for lane in bundles if not prepared_is_current(jobs.get(name+'__'+lane),inputs[lane])]
        if not pending:continue
        groups=[]
        for s in r['sources']:
            if sha256_file(ROOT/s['path'])!=s['sha256']:raise ValueError('V4 source changed')
            source=ROOT/s['path']
            if s['subtype']=='normal_part':
                target=OUT/'STL/v4'/source.name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
                if source.with_suffix('.print.json').exists():shutil.copy2(source.with_suffix('.print.json'),target.with_suffix('.print.json'))
                source=target
            item=(source,s['subtype'],s['overrides'],s['translation'])
            if key=='accessories' or not groups:groups.append([item])
            else:groups[0].append(item)
        # Move the body into the shared area; XY shape/height remain exact.
        offset=[65.,35.,0.] if key=='body' else [0.,0.,0.]
        for lane in pending:
            bundle=bundles[lane]
            job_id=name+'__'+lane;work=WORK/'jobs'/job_id;work.mkdir(parents=True,exist_ok=True)
            settings=job_settings(bundle,lane,role,name,tower=(270.,255.))
            if key=='body':
                # Its existing pose is diagonally arranged for 256 mm. Use a
                # small shift of the complete group instead of new geometry.
                offset=[35.,35.,0.]
            project=OUT/'v4'/lane/(name+'.3mf')
            prep=write_project(project,groups,settings,offset)
            write_json(work/'process.json',job_process(bundle,settings))
            write_json(work/'dry_run.json',slice_command(project,lane,work))
            matrix=json.loads((ROOT/r['sources'][0]['path']).with_suffix('.print.json').read_text()).get('source_to_stl_matrix') if key=='body' else None
            jobs[job_id]=dict(id=job_id,name=name,family='v4',role=role,lane=lane,state='shared',candidate=True,status='prepared',
                preparation=prep,project=rel(project),work=rel(work),offset=offset,source_to_stl_matrix=matrix,
                stl=rel(groups[0][0][0]),magnet_count=4 if key=='body' else 0,
                prepare_inputs=inputs[lane],project_sha256=sha256_file(project))
            print('PREPARED',job_id,flush=True)
            write_json(manifest_path,dict(schema_version=1,printer='Bambu Lab H2C 0.6 High Flow',checkpoint='c261f32',jobs=list(jobs.values())))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('prepare','v4-wings'))
    ap.add_argument('--only',action='append')
    args=ap.parse_args()
    if args.command=='v4-wings':v4_wings()
    else:prepare(args.only)


if __name__=='__main__':main()
