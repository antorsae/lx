"""Prepare native Bambu projects from the approved STLs, without changing CAD.

The frozen regular UM project is the process authority. Infill modifiers are
nonprinting volumes. Slicing and its independent audit are separate steps.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from xml.etree import ElementTree as ET

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))
from lx521_baffle.print_policy import apply_supported_policy, role_settings, policy_sha256
WORK = ROOT / 'review/nd25fn4_print'
OUT = HERE / 'print'
BASE = ROOT / 'to_print/obiwan/3mf_06hf_petg-gf_pla/obiwan_03_UM_carrier_1_of_1_GUI.3mf'
FROZEN_PROFILES = ROOT / 'review/petg_gui_project_workspace/obiwan_01_02_03_04_LM_UM_combo_floor_stand/base_profile/profiles'
SPLIT_Y = 421.0
CORE = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
PROD = 'http://schemas.microsoft.com/3dmanufacturing/production/2015/06'
ET.register_namespace('', CORE)
ET.register_namespace('p', PROD)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def print_mesh(source):
    target=HERE/'print/geometry'/(source.stem+'_PRINT.stl')
    if target.exists():
        authority=json.loads(target.with_suffix('.print.json').read_text())
        assert authority['approved_source_sha256']==sha(source)
        assert authority['stl_sha256']==sha(target)
        return target
    return source


def metadata(parent, key, value):
    return ET.SubElement(parent, 'metadata', key=key, value=str(value))


def settings(supported=True, infill='100%', pattern='zig-zag'):
    with zipfile.ZipFile(BASE) as archive:
        value = json.loads(archive.read('Metadata/project_settings.config'))
    # Complete all per-filament mapping vectors. The older exporter leaves
    # several at length one and the purge matrix at 4x4 for two materials.
    value.update(
        print_settings_id='LX521 ND25FN V4 Tinmorry 0.6HF 0.16mm',
        filament_colour=['#456C86', '#EEEEEE'],
        default_filament_colour=['#456C86', '#EEEEEE'],
        filament_map_mode='Manual', filament_map=['1', '1'],
        filament_map_2=['1', '1'], filament_nozzle_map=['1', '1'],
        filament_volume_map=['1', '1'], extruder_nozzle_stats=['High Flow#1'],
        flush_volumes_matrix=['0', '280', '280', '0'],
        flush_volumes_vector=['140', '140', '140', '140'],
        sparse_infill_density=infill, sparse_infill_pattern=pattern,
        enable_support='1' if supported else '0',
        wipe_tower_x=['12'], wipe_tower_y=['12'],
        # Keep different materials out of all model walls and infill.
        flush_into_objects='0', flush_into_infill='0', flush_into_support='0',
        brim_type='outer_only', brim_width='5',
    )
    return apply_supported_policy(value)


def slice_command(project,output_name='discovery.gcode.3mf'):
    return ['/Applications/BambuStudio.app/Contents/MacOS/BambuStudio','--debug','2',
            '--slice','1','--arrange','0','--orient','0','--mtcpp','5000000',
            '--load-filaments',';'.join(str(WORK/'profiles'/n) for n in
                ['resolved_filament.json','resolved_support_interface_filament.json']),
            '--outputdir',str(project.parent),'--export-3mf',output_name,str(project)]


def write_project(path, parts, process, offset=(0, 0, 0), pauses=()):
    """Write float32 STL vertices verbatim into a standard Bambu component tree."""
    root = ET.Element(f'{{{CORE}}}model', unit='millimeter')
    ET.SubElement(root, f'{{{CORE}}}metadata', name='Application').text = 'BambuStudio-02.07.01.62'
    ET.SubElement(root, f'{{{CORE}}}metadata', name='BambuStudio:3mfVersion').text = '1'
    ET.SubElement(root, f'{{{CORE}}}metadata', name='Title').text = path.stem
    resources = ET.SubElement(root, f'{{{CORE}}}resources')
    config = ET.Element('config')
    object_ids = []
    next_id = 1
    mesh_sources = []
    # Every group is one independently printable object. Body modifiers stay
    # in the body's component tree and never become a second printed body.
    for group in parts:
        child_ids = []
        entries = []
        for source, subtype, overrides, translation in group:
            mesh = trimesh.load_mesh(source, process=True)
            oid = next_id; next_id += 1
            child_ids.append((oid, translation))
            entries.append((oid, source, subtype, overrides, len(mesh.faces)))
            mesh_sources.append((oid, mesh))
        parent_id = next_id; next_id += 1
        object_ids.append(parent_id)
        obj = ET.SubElement(resources, f'{{{CORE}}}object', id=str(parent_id), type='model')
        components = ET.SubElement(obj, f'{{{CORE}}}components')
        for oid, t in child_ids:
            ET.SubElement(components, f'{{{CORE}}}component', objectid=str(oid),
                          transform='1 0 0 0 1 0 0 0 1 '+ ' '.join(map(str,t)))
        oc = ET.SubElement(config, 'object', id=str(parent_id))
        metadata(oc, 'name', Path(group[0][0]).stem)
        for oid, source, subtype, overrides, faces in entries:
            part = ET.SubElement(oc, 'part', id=str(oid), subtype=subtype)
            metadata(part, 'name', Path(source).stem)
            metadata(part, 'source_file', Path(source).name)
            metadata(part, 'matrix', '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1')
            if subtype == 'normal_part': metadata(part, 'extruder', '1')
            for key, value in overrides.items(): metadata(part, key, value)
            ET.SubElement(part, 'mesh_stat', face_count=str(faces), edges_fixed='0',
                          degenerate_facets='0', facets_removed='0', facets_reversed='0', backwards_edges='0')
    build = ET.SubElement(root, f'{{{CORE}}}build')
    plate = ET.SubElement(config, 'plate')
    for key, value in dict(plater_id='1', plater_name=path.stem, locked='true',
                           filament_map_mode='Manual', filament_maps='1 1',
                           filament_volume_maps='1 1', gcode_file='').items(): metadata(plate,key,value)
    for oid in object_ids:
        ET.SubElement(build, f'{{{CORE}}}item', objectid=str(oid), printable='1',
                      transform='1 0 0 0 1 0 0 0 1 '+' '.join(map(str,offset)))
        mi = ET.SubElement(plate, 'model_instance')
        metadata(mi,'object_id',oid);metadata(mi,'instance_id',0);metadata(mi,'identify_id',oid*10)
    ET.SubElement(config,'assemble')
    # Stream large meshes instead of constructing millions of XML Elements.
    shell = ET.tostring(root,encoding='utf-8',xml_declaration=True)
    prefix, suffix = shell.split(b'</resources>')
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        with archive.open('3D/3dmodel.model','w') as out:
            out.write(prefix)
            for oid,mesh in mesh_sources:
                out.write(f'<object id="{oid}" type="model"><mesh><vertices>'.encode())
                for start in range(0,len(mesh.vertices),10000):
                    out.write(''.join(f'<vertex x="{x:.9g}" y="{y:.9g}" z="{z:.9g}"/>' for x,y,z in mesh.vertices[start:start+10000]).encode())
                out.write(b'</vertices><triangles>')
                for start in range(0,len(mesh.faces),10000):
                    out.write(''.join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a,b,c in mesh.faces[start:start+10000]).encode())
                out.write(b'</triangles></mesh></object>')
            out.write(b'</resources>'+suffix)
        archive.writestr('Metadata/model_settings.config',ET.tostring(config,encoding='utf-8',xml_declaration=True))
        archive.writestr('Metadata/project_settings.config',json.dumps(process,indent=2)+'\n')
        with zipfile.ZipFile(BASE) as baseline:
            for name in ['[Content_Types].xml','_rels/.rels']:
                archive.writestr(name,baseline.read(name))
        if pauses:
            pause_root=ET.Element('custom_gcodes_per_layer');p=ET.SubElement(pause_root,'plate')
            ET.SubElement(p,'plate_info',id='1')
            for z,names in pauses:
                program=(f'; ND25FN_MAGNET_INSERTION\n; Insert {names}\nG90\nM400\n'
                         f'G1 Z250 F1200\nM400\nM400 U1\nG1 Z{z:.2f} F1200\nM400')
                ET.SubElement(p,'layer',top_z=f'{z:.2f}',type='4',extruder='1',color='',extra=program,gcode=program)
            ET.SubElement(p,'mode',value='SingleExtruder')
            archive.writestr('Metadata/custom_gcode_per_layer.xml',ET.tostring(pause_root,encoding='utf-8'))
    return dict(path=str(path.relative_to(ROOT)),sha256=sha(path),offset=list(offset),
                sources=[dict(path=str(Path(s).relative_to(ROOT)),sha256=sha(s),subtype=kind,
                              overrides=overrides,translation=list(t)) for g in parts for s,kind,overrides,t in g])


def insert_bores():
    """The twelve real insert sites, in installed coordinates (mm).

    Match the regular Obi-Wan bore blockers: cover mouth to blind floor
    with 0.25 mm clearance plus 0.02 mm boolean margin. M3 pilots have
    a flat floor at 4 mm depth; no extra blind-tip cavity is needed.
    """
    import v4_model as model
    rows=[]
    for i,(x,y) in enumerate(model.interface.UM_PILOT_XY):
        rows.append(dict(name=f'UM_M3_{i+1}',start=[x,y,model.interface.UM_SEAT_Z-model.interface.UM_PILOT_DEPTH_MM],
            end=[x,y,model.interface.UM_SEAT_Z],radius=model.interface.UM_PILOT_D_MM/2))
    for i,x in enumerate(model.interface.JOINT_EAR_X):
        rows.append(dict(name=f'LM_receiver_M3_{i+1}',start=[x,model.interface.JOINT_EAR_Y,12.4],
            end=[x,model.interface.JOINT_EAR_Y,model.interface.JOINT_INSERT_BORE_Z[1]],
            radius=model.interface.JOINT_INSERT_BORE_D/2))
    m=model.retained.M
    for which in (0,1):
        sign=1 if which==0 else -1
        for i,angle in enumerate(model.retained.SCREW_ANGLES):
            y=m.screw_circle_radius*np.cos(angle);z=m.screw_circle_radius*np.sin(angle)
            for suffix,low,high,radius in [('',m.service_end_x,m.service_end_x+m.insert_length,m.insert_pilot_radius)]:
                ends=trimesh.transform_points([[sign*low,y,sign*(z-model.retained.P.center_spacing/2)],
                    [sign*high,y,sign*(z-model.retained.P.center_spacing/2)]],model.PACKAGE_TO_INSTALLED)
                rows.append(dict(name=f'T{which+1}_M3_{i+1}'+suffix,start=ends[0].tolist(),end=ends[1].tolist(),radius=radius))
    return rows


def body_assets():
    import v4_model as model
    inputs = WORK/'inputs';inputs.mkdir(parents=True,exist_ok=True)
    stl = print_mesh(HERE/'STL/01_UM_Crescent_V4.stl')
    authority=json.loads(stl.with_suffix('.print.json').read_text())
    assert sha(stl)==authority['stl_sha256']
    matrix=np.asarray(authority['source_to_stl_matrix'])
    modifier=trimesh.creation.box([180,180,70],transform=trimesh.transformations.translation_matrix([0,SPLIT_Y+90,0]))
    modifier.apply_transform(matrix);modifier.export(inputs/'tweeters_15pct_gyroid.stl')
    blockers=[];routes=[];inlet_columns=[]
    routed,_,_,_=model.wiring()
    for name,(centers,radii) in routed.items():
        points=trimesh.transform_points(centers,model.PACKAGE_TO_INSTALLED)
        routes.append(dict(name=name,centers_mm=points.tolist(),radii_mm=np.asarray(radii).tolist()))
        mesh=model.tube_mesh(points[::3],np.asarray(radii)[::3]+.45,step=.6)
        blockers.append(mesh)
        # A local blocker inside a bore suppresses its own roof supports but
        # neighbouring service-pocket supports may still grow sideways into
        # the inlet. Block the short inlet's complete support column too.
        # Follow the entire mouth until it leaves the service chamber.
        # A fixed first-16-sample slice covered barely 1.2 mm and let the
        # enlarged M3 chamber grow support sideways into the next bend.
        pod_y=model.INSTALLED_Y_OFFSET+(31 if name=='upper_and_trunk' else -31)
        radial=np.hypot(points[:,0],points[:,1]-pod_y)
        limit=model.retained.M.service_radius+float(np.max(radii))+.5
        outside=np.flatnonzero(radial>limit)
        stop=int(outside[0])+1 if len(outside) else len(points)
        for point in points[:stop:5]:
            bed=trimesh.transform_points([point],matrix)[0]
            inlet_columns.append(trimesh.creation.cylinder(radius=3.8,height=50,sections=64,
                transform=trimesh.transformations.translation_matrix([bed[0],bed[1],20])))
    for name,points,radius in [('T_inlet',model.route.ts_cable_points(.15),3.35),
                               ('UM_lead',model.route.route_cable_points(.15),4.2)]:
        points=points[(points[:,1]>309)&(points[:,1]<326.5)]
        routes.append(dict(name=name,centers_mm=points.tolist(),radii_mm=[radius]*len(points)))
        blockers.append(model.tube_mesh(points,radius+.45,step=.5))
    body=trimesh.load_mesh(stl,process=True)
    for part in body.split(only_watertight=False):
        if part.volume<0:
            part.invert();part.apply_transform(np.linalg.inv(matrix));blockers.append(part)
    combined=model.solid(blockers[0])
    for part in blockers[1:]:combined+=model.solid(part)
    blocker=model.to_trimesh(combined);blocker.apply_transform(matrix)
    combined=model.solid(blocker)
    for column in inlet_columns:combined+=model.solid(column)
    blocker=model.to_trimesh(combined)
    blocker.export(inputs/'duct_and_magnet_support_blocker.stl')
    (inputs/'ducts.json').write_text(json.dumps(dict(frame='installed',routes=routes))+'\n')
    bores=insert_bores();insert_tools=[];bore_gauges=[]
    for bore in bores:
        a,b=np.asarray(bore['start']),np.asarray(bore['end'])
        center=(a+b)/2
        insert_tools.append(trimesh.creation.cylinder(radius=bore['radius']+.27,
            height=np.linalg.norm(b-a)+.54,sections=96,
            transform=trimesh.transformations.translation_matrix(center)))
        # Check each registered exclusion against a real bore in the exported
        # print mesh. Leave 0.2 mm for tessellation and rounded blind floors.
        gauge=trimesh.creation.cylinder(radius=bore['radius']-.2,
            height=np.linalg.norm(b-a)-.4,sections=64,
            transform=trimesh.transformations.translation_matrix(center))
        gauge.apply_transform(matrix);bore_gauges.append(gauge)
    gauge_union=trimesh.boolean.union(bore_gauges,engine='manifold')
    gauge_overlap=abs((model.solid(body)^model.solid(gauge_union)).volume())
    assert gauge_overlap<1e-5,('Insert blockers do not register with actual mesh bores',gauge_overlap)
    insert_blocker=trimesh.boolean.union(insert_tools,engine='manifold')
    assert insert_blocker.is_watertight
    insert_blocker.apply_transform(matrix)
    insert_blocker.export(inputs/'insert_bore_support_blocker.stl')
    (inputs/'insert_bores.json').write_text(json.dumps(dict(frame='installed',sites=bores,
        insert_site_count=12,clearance_mm=.27,actual_mesh_sha256=sha(stl),
        bore_registration_gauge_inset_mm=.2,bore_registration_overlap_mm3=gauge_overlap,
        reference='scripts/obiwan_support_blocker.py:_insert_bore_blockers'),indent=2)+'\n')
    return body,stl,inputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--body',action='store_true');parser.add_argument('--accessories',action='store_true')
    parser.add_argument('--wings',action='store_true');args=parser.parse_args()
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(exist_ok=True)
    profile_dir=WORK/'profiles';profile_dir.mkdir(exist_ok=True)
    # The installed user preset has been rewritten into an aggregate. Preserve
    # the exact resolved filament files that produced the regular UM instead.
    for name in ['resolved_filament.json','resolved_support_interface_filament.json','resolved_machine.json']:
        shutil.copy2(FROZEN_PROFILES/name,profile_dir/name)
    records={}
    if args.body:
        body,stl,inputs=body_assets()
        group=[(stl,'normal_part',{},(0,0,0)),
               (inputs/'tweeters_15pct_gyroid.stl','modifier_part',dict(sparse_infill_density='15%',sparse_infill_pattern='gyroid'),(0,0,0)),
               (inputs/'duct_and_magnet_support_blocker.stl','support_blocker',{},(0,0,0)),
               (inputs/'insert_bore_support_blocker.stl','support_blocker',{},(0,0,0))]
        offset=[float((256-v)/2) for v in body.extents[:2]]+[0]
        records['body']=write_project(WORK/'body/ND25FN_V4_body_discovery.3mf',[group],settings(),offset)
        records['body'].update(source_to_stl_matrix=json.loads(stl.with_suffix('.print.json').read_text())['source_to_stl_matrix'],infill_split_installed_y_mm=SPLIT_Y)
    if args.accessories:
        groups=[]
        for name,positions in [('02_Closed_Cap_PRINT_TWO.stl',[(74,97,0),(144,97,0)]),
                               ('03_Tweeter_Retainer_PRINT_TWO.stl',[(74,166,0),(144,166,0)])]:
            # Source caps/rings are centred on XY; positions are centre datums.
            for pos in positions:groups.append([(HERE/'STL'/name,'normal_part',{},pos)])
        # The caps open toward the bed and have a 44.6 mm cavity below the
        # closed ceiling. Support that ceiling with the same GF/PLA process
        # as the main body; a support-free recipe bridges the entire cavity.
        process=settings(True)
        process.update(role_settings('crescent_accessories'))
        records['accessories']=write_project(WORK/'accessories/ND25FN_V4_caps_retainers.3mf',groups,process)
    if args.wings:
        for approved in sorted((HERE/'STL/wings').glob('*.stl')):
            stl=print_mesh(approved)
            m=trimesh.load_mesh(stl,process=True);offset=[float((256-v)/2) for v in m.extents[:2]]+[0]
            voids=[]
            for cavity in m.split(only_watertight=False):
                if cavity.volume<0:cavity.invert();voids.append(cavity)
            blocker=WORK/'inputs'/(approved.stem+'_magnet_blocker.stl')
            trimesh.util.concatenate(voids).export(blocker)
            groups=[[(stl,'normal_part',{},(0,0,0)),(blocker,'support_blocker',{},(0,0,0))]]
            process=settings(True)
            process.update(role_settings('crescent_wing'))
            if approved.stem=='V4_flat_left_UPPER':process['wipe_tower_x']=['209']
            records[approved.stem]=write_project(WORK/approved.stem/(approved.stem+'.3mf'),groups,process,offset)
    manifest=WORK/'preparation.json'
    old=json.loads(manifest.read_text()) if manifest.exists() else {}
    old.update(records)
    for record in records.values():
        project=ROOT/record['path']
        (project.parent/'command.json').write_text(json.dumps(slice_command(project),indent=2)+'\n')
    old['authority']=dict(project=str(BASE.relative_to(ROOT)),sha256=sha(BASE),script_sha256=sha(__file__),policy_sha256=policy_sha256(),
                          profile_files={str(p.relative_to(ROOT)):sha(p) for p in profile_dir.glob('*.json')},
                          purge_mm3_each_direction=280,flush_into_model=False,flush_into_support=False)
    manifest.write_text(json.dumps(old,indent=2)+'\n')
    print(json.dumps({k:v.get('path') for k,v in records.items()},indent=2))


if __name__=='__main__':main()
