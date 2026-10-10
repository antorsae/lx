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
from lx521_baffle.tweeter_options import ND25FN, PT2522, AMT17, TWEETER_FAMILIES, compatible_tweeters
from prepare_print import write_project
from lx521_baffle.h2c import shelf

OUT = shelf.SHELF
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
        'src/lx521_baffle/tweeter_options.py','src/lx521_baffle/h2c/shelf.py',
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


def inventory():
    rows = []
    def add(name, family, role, source, *, state='shared', candidate=False, blockers=(), angle=None, product=None):
        angle=policy().get('print_orientation_z_deg',{}).get(name,angle)
        rows.append(dict(name=name, family=family, product_family=product or family,
            tweeter_families=compatible_tweeters(name,family,role),role=role, source=rel(source),
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
    old=json.loads((shelf.P2S_ARCHIVE/'catalog.json').read_text())
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
        # The released keyed LM halves: an alternative to the one-piece carrier for every Obiwan upper, and the
        # bottom the ND25FN-4 vase top keys onto. The keyed top is one part for both stand states.
        part='optional_lm_keyed_1_of_2_bottom'
        add(f'h2c_obiwan_lm_keyed_bottom_{state}','obiwan','lm_bottom',ROOT/f'build/{state}/stl/obiwan_{part}.stl',state=state,
            blockers=[rel(ROOT/f'build/{state}/support_blockers/obiwan_{part}.support_blocker.json')])
    part='optional_lm_keyed_2_of_2_top'
    add('h2c_obiwan_lm_keyed_top','obiwan','lm_top',ROOT/f'build/no_floor_stand/stl/obiwan_{part}.stl',
        blockers=[rel(ROOT/f'build/no_floor_stand/support_blockers/obiwan_{part}.support_blocker.json')])
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
    for stem in ('obiwan_bmr_crescent_TEBM35C10-4','obiwan_bmr_crescent_opposed_TEBM35C10-4'):
        add('h2c_'+stem,'obiwan','regular_tweeter',ROOT/'build/bmr_crescent_TEBM35C10-4'/f'{stem}.stl',candidate=True)
    # Dayton ND25FN-4 vases: each replaces its product's whole upper module.
    # Approved design-frame meshes, authorities and duct contracts live in
    # candidates/nd25fn4_vase/{obiwan,stock} (see their build_package.py).
    # The waveguides are not left/right symmetric: each product comes in two hands, named by the side of the MU10
    # the tweeters sit on, seen from the front. The left hand mirrors only the rebuilt upper; the parts it mates
    # (keyed LM bottom, Stock LM) are shared and unchanged.
    for product in ('obiwan','stock'):
        pkg=ROOT/'candidates/nd25fn4_vase'/product
        for hand in ('right','left'):
            def pkg_add(part,role,**kw):
                stem=f'{product}_tweeters_{hand}_{part}'
                blocker=pkg/'support_blockers'/f'nd25fn4_vase_{stem}.support_blocker.json'
                add(f'h2c_dayton_nd25fn4_{stem}',ND25FN,role,pkg/'STL'/f'nd25fn4_vase_{stem}.stl',
                    candidate=True,product=product,blockers=[rel(blocker)] if blocker.exists() else [],**kw)
            if product=='obiwan':
                pkg_add('top','lm_top')                     # fits both keyed LM bottoms (identical keyed tops since the lane unification)
                for slug in ('flat','graded'):
                    for side in ('left','right'):
                        pkg_add(f'wing_{slug}_{side}','crescent_wing')
            else:
                pkg_add('upper','um')
            for part in ('cap_front_tweeter','cap_rear_tweeter'):
                pkg_add(part,'crescent_accessories')
    # Dipole tweeters (GRS PT2522-4 planar, Mundorf AMT17D2.1), Obiwan only, one driver upright or rotated:
    #   T above: the UM carrier + tweeter frame in one piece (replaces the UM carrier, on the LM carrier or keyed LM);
    #   T below: the keyed LM top + tweeter + raised MU10 in one piece (on the keyed LM bottom), or -- PT2522 upright,
    #   too long for the nozzle reach -- a keyed LM top variant plus a T + UM piece on the released LM--UM joint.
    # Approved installed-frame meshes, authorities and blockers: candidates/dipole_tweeter (build_package.py).
    pkg=ROOT/'candidates/dipole_tweeter'
    for family,layout,orient,part,role in DIPOLE_SHELF:
        stem=f'{family}_obiwan_{layout}_{orient}'+(f'_{part}' if part else '')
        blocker=pkg/'support_blockers'/f'{stem}.support_blocker.json'
        add(f'h2c_{stem}',family,role,pkg/'STL'/f'{stem}.stl',candidate=True,product='obiwan',
            blockers=[rel(blocker)] if blocker.exists() else [])
    return rows


DIPOLE_SHELF=[(f,'t_above',o,None,'um') for f in (PT2522,AMT17) for o in ('upright','rotated')]+[
    (PT2522,'t_below','upright','lm_top','lm_top'),(PT2522,'t_below','upright','um_t','um'),(PT2522,'t_below','rotated',None,'lm_top'),
    (AMT17,'t_below','upright',None,'lm_top'),(AMT17,'t_below','rotated',None,'lm_top')]


def row_lanes(row):
    return list(policy()['lanes'])


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


def seam_mask(name,target,matrix):
    """Per-face seam enforcer mask for the posed STL at `target` (face order of a process=True load), or (None, None)."""
    seam_rule=policy().get('seam_enforcers',{}).get(name)
    if not seam_rule:return None,None
    # Paint the faces that face away from view in the installed pose (e.g. the floor) as seam enforcers, so every
    # outer-wall loop that passes them starts there instead of on a visible face.
    painted=trimesh.load_mesh(target,process=True);painted.apply_transform(np.linalg.inv(matrix))
    mask=np.zeros(len(painted.faces),bool)
    if 'installed_direction' in seam_rule:
        facing=painted.face_normals@np.asarray(seam_rule['installed_direction'],float)>=seam_rule['min_cos']
        if 'below_y_mm' in seam_rule:facing&=painted.triangles_center[:,1]<=seam_rule['below_y_mm']
        mask|=facing
    if seam_rule.get('mates'):
        # Faces touching a mating part are covered once assembled; seams there are hidden in the joint.
        from scipy.spatial import cKDTree
        sources={r['name']:r['source'] for r in inventory()}
        cloud=[]
        for mate in seam_rule['mates']:
            other,_=source_mesh(ROOT/sources[mate])
            cloud.append(trimesh.sample.sample_surface_even(other,int(other.area/0.04),seed=0)[0])
        gap,_=cKDTree(np.vstack(cloud)).query(painted.triangles_center)
        side=np.abs(painted.face_normals[:,2])<0.5          # vertical in the front-down print pose
        mask|=(gap<=seam_rule['max_gap_mm'])&side
    if not mask.any():raise ValueError(('Empty seam enforcer',name))
    return mask,dict(seam_rule,faces=int(mask.sum()),area_mm2=round(float(painted.area_faces[mask].sum()),1))


def export_posed(row):
    path=ROOT/row['source']; mesh,source_authority=source_mesh(path)
    large=row['family'] in {'stock','slim'} and row['name'].endswith('_lm_no_floor_stand')
    brim=policy()['brim_mm']['stock_slim_no_floor_lm' if large else 'default']
    brim=policy()['brim_mm'].get('by_part',{}).get(row['name'],brim)
    _,angle,minimum,offset,tower=placement(mesh,brim,row['angle'],large)
    transform=front_down_transform_record(minimum.tolist(),z_rotation_deg=angle)
    matrix=np.asarray(transform['source_to_stl_matrix'])
    mesh.apply_transform(matrix)
    target=shelf.stl(row['name']);target.parent.mkdir(parents=True,exist_ok=True)
    mesh.export(target)
    if not mesh.is_watertight or not mesh.is_winding_consistent:raise ValueError(target)
    components=mesh.split(only_watertight=False)
    if sum(c.volume>0 for c in components)!=1:raise ValueError(f'{target}: disconnected material')
    authority=dict(schema_version=1,part=target.stem,stl=target.name,stl_bytes=target.stat().st_size,
        stl_sha256=sha256_file(target),source_stl=row['source'],source_sha256=sha256_file(path),
        **transform)
    write_json(target.with_suffix('.print.json'),authority)
    validate_print_sidecar(target)
    part_overrides={}
    mask,info=seam_mask(row['name'],target,matrix)
    if mask is not None:
        part_overrides['__paint_seam__']=mask;row['seam_enforcer']=info
    group=[(target,'normal_part',part_overrides,[0,0,0])]
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
    if row['family']=='obiwan' and row['state']=='floor_stand' and row['role']=='lm_bottom' and 'service_lid' not in row['name']:
        # The underside NL8 service bay is the open slot the leads run through. Its crown and flanks were shaped to
        # print unsupported front-face down, but the lid rebate's 1-mm lip at the crown seeded a support column
        # the full length of the bay. Mask the bay and its rebate so that lip bridges.
        from shapely.geometry import Polygon
        from lx521_baffle.obiwan.floor import (_trough_plan_points,_trough_plan_offset,BAY_REBATE_OFF_MM,
            BAY_CEILING_Y_MM,BAY_REBATE_DEPTH_Y_MM)
        clearance=policy()['support_blocker_extra_xyz_mm']
        prisms=[]
        for outline,top in ((_trough_plan_points(),BAY_CEILING_Y_MM),
                            (_trough_plan_offset(BAY_REBATE_OFF_MM,BAY_REBATE_OFF_MM),BAY_REBATE_DEPTH_Y_MM)):
            prism=trimesh.creation.extrude_polygon(Polygon(outline).buffer(clearance),top+1+clearance)
            # Plan (x, z) extruded along y from 1 mm below the floor face.
            prism.apply_transform(np.array([[1.,0,0,0],[0,0,1,-1],[0,1,0,0],[0,0,0,1]]))
            if prism.volume<0:prism.invert()
            prism.apply_transform(matrix);prisms.append(prism)
        blockers+=prisms
        row['floor_bay_support_shadow']=dict(plan='obiwan.floor._trough_plan_points + lid rebate',
            y_mm=[-1.,BAY_CEILING_Y_MM+clearance],clearance_mm=clearance)
    if row['name'] in policy().get('magnet_insertion_support_blockers',[]):
        # A steeper support threshold can seed support above a captive magnet; keep every insertion column clear.
        top=float(mesh.bounds[1,2])
        for cavity in components:
            if cavity.volume>=0:continue
            (x0,y0,z0),(x1,y1,_)=cavity.bounds
            radius=max(x1-x0,y1-y0)/2+1.0
            column=trimesh.creation.cylinder(radius=radius,height=top-z0,sections=48,
                transform=trimesh.transformations.translation_matrix([(x0+x1)/2,(y0+y1)/2,(z0+top)/2]))
            blockers.append(column);row['magnet_insertion_support_blockers']=row.get('magnet_insertion_support_blockers',0)+1
    modifiers=[]
    for zone in policy().get('infill_modifiers_installed_mm',{}).get(row['name'],[]):
        # A nonprinting modifier volume: plan disks (installed x, y) minus keep-out disks and a y floor, through the
        # part's full depth. Only the part's own solid inside it takes the zone's settings; walls are unchanged.
        from shapely.geometry import Point, box as sbox
        from shapely.ops import unary_union
        region=unary_union([Point(x,y).buffer(r,resolution=64) for x,y,r in zone['disks']])
        for x,y,r in zone.get('keep_disks',[]):region=region.difference(Point(x,y).buffer(r,resolution=64))
        if 'y_min' in zone:region=region.intersection(sbox(-1e4,zone['y_min'],1e4,1e4))
        installed=mesh.copy();installed.apply_transform(np.linalg.inv(matrix))
        z0,z1=float(installed.bounds[0,2])-1,float(installed.bounds[1,2])+1
        volume=trimesh.creation.extrude_polygon(region,z1-z0);volume.apply_translation([0,0,z0])
        volume.apply_transform(matrix)
        target_mod=WORK/'inputs'/(row['name']+'_'+zone['name']+'_modifier.stl');target_mod.parent.mkdir(parents=True,exist_ok=True)
        volume.export(target_mod)
        modifiers.append((target_mod,'modifier_part',dict(zone['settings']),[0,0,0]))
        row.setdefault('infill_modifiers',[]).append(dict(zone,modifier_volume_mm3=round(float(volume.volume))))
    for zone in policy().get('support_blocker_boxes_installed_mm',{}).get(row['name'],[]):
        # Named installed-frame regions that must print without support (e.g. the Obiwan NL8 boss).
        lo,hi=np.asarray(zone['min'],float),np.asarray(zone['max'],float)
        box_=trimesh.creation.box(extents=hi-lo,transform=trimesh.transformations.translation_matrix((lo+hi)/2))
        box_.apply_transform(matrix);blockers.append(box_)
        row.setdefault('support_blocker_boxes',[]).append(zone)
    # Preserve inward winding in the model; reverse a COPY only for the
    # non-printing positive blocker volumes inside the magnet cavities.
    for cavity in components:
        if cavity.volume < 0:
            cavity=cavity.copy();cavity.invert();blockers.append(cavity)
    if blockers:
        blocker=WORK/'inputs'/(row['name']+'_support_blocker.stl');blocker.parent.mkdir(parents=True,exist_ok=True)
        if row.get('floor_cable_support_shadows') or row.get('floor_bay_support_shadow') or row.get('support_blocker_boxes'):
            # Make overlapping entry/shadow masks one explicit volume.
            merged=manifold.Manifold.batch_boolean([solid(b) for b in blockers],manifold.OpType.Add)
            mask=to_trimesh(merged.simplify(.005))
            if not mask.is_watertight:raise ValueError('Open floor support mask')
            mask.export(blocker)
        else:trimesh.util.concatenate(blockers).export(blocker)
        group.append((blocker,'support_blocker',{},[0,0,0]))
    group.extend(modifiers)
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
    bundles={lane:resolve_profiles(lane,WORK/'profiles'/lane) for lane in policy()['lanes']}
    rows=inventory()
    manifest_path=OUT/'catalog.json'
    previous=json.loads(manifest_path.read_text())['jobs'] if manifest_path.exists() else []
    retired={f'h2c_{f}_lm_floor_stand__petg_gf_pla' for f in ('stock','slim')}
    renamed={j['id'] for j in previous if j['id'].startswith('h2c_v4_')}
    retired.update(renamed)
    # The retained V4 fused body, its accessory plate and its wings are
    # replaced by the ND25FN-4 vases (their meshes stay in candidates/nd25fn4_crescent).
    current={row['name'] for row in rows}
    replaced={j['id'] for j in previous if j['family']==ND25FN and j['name'] not in current}
    retired.update(replaced)
    # The PETG Translucent + PLA Translucent lane is retired; PETG-GF + PLA is the only material lane.
    dropped_lanes={j['id'] for j in previous if j['lane'] not in policy()['lanes']}
    retired.update(dropped_lanes)
    jobs={j['id']:j for j in previous if j['id'] not in retired}
    for job in previous:
        if job['id'] not in retired:continue
        for key in ('project','sliced_project','stl','authority'):
            path=ROOT/job[key] if key in job else None
            if path and path.exists() and path.is_relative_to(OUT):
                v4=job['name'] in {'h2c_dayton_nd25fn4_body','h2c_dayton_nd25fn4_accessories'} or job['name'].startswith('h2c_dayton_nd25fn4_wing_')
                archive=('retired_translucent_lane' if job['id'] in dropped_lanes else
                    'retired_dayton_revision_names' if job['id'] in renamed else
                    ('retired_dayton_nd25fn4_v4' if v4 else 'renamed_dayton_nd25fn4') if job['id'] in replaced else 'rejected_one_piece_floor')
                target=WORK/'experiments'/archive/path.relative_to(OUT)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,target)
    if renamed:
        # The accessory job has two distinct source STLs. Archive the whole
        # old folder so its second mesh cannot remain as an orphan deliverable.
        for path in [OUT/'v4',OUT/'STL/v4']:
            if path.exists():
                target=WORK/'experiments/retired_dayton_revision_names'/path.relative_to(OUT)
                shutil.copytree(path,target,dirs_exist_ok=True);shutil.rmtree(path)
    if replaced and (OUT/'STL/dayton_nd25fn4').exists():
        # Retained V4 meshes lived in their own folder; archive it whole.
        target=WORK/'experiments/retired_dayton_nd25fn4_v4/STL/dayton_nd25fn4'
        shutil.copytree(OUT/'STL/dayton_nd25fn4',target,dirs_exist_ok=True);shutil.rmtree(OUT/'STL/dayton_nd25fn4')
    def save_manifest():
        write_json(manifest_path,dict(schema_version=2,printer='Bambu Lab H2C 0.6 High Flow',
            checkpoint='c261f32',tweeter_families=TWEETER_FAMILIES,jobs=list(jobs.values())))
    for row in rows:
        if selected and row['name'] not in selected:continue
        lanes=[lane for lane in row_lanes(row) if lane in bundles]
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
            project=shelf.project(row['name'],lane)
            prep=write_project(project,groups,settings,row['offset'])
            write_json(work/'process.json',job_process(bundles[lane],settings))
            command=slice_command(project,lane,work);write_json(work/'dry_run.json',command)
            jobs[job_id]=dict(id=job_id,lane=lane,status='prepared',**row,preparation=prep,project=rel(project),work=rel(work),
                prepare_inputs=inputs[lane],project_sha256=sha256_file(project))
            print('PREPARED',job_id,flush=True)
            save_manifest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=('prepare',))
    ap.add_argument('--only',action='append')
    args=ap.parse_args()
    prepare(args.only)


if __name__=='__main__':main()
