"""Independently bind sliced projects to meshes, process and actual toolpaths."""
from pathlib import Path
from xml.etree import ElementTree as ET
import gc
import json
import sys
import zipfile
import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];WORK=ROOT/'review/nd25fn4_print'
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from audit_gui_slice import material_extrusions
from audit_changeover import audit_changeover
from gcode_analysis import parse_gcode
from prepare_print import sha
from lx521_baffle.print_policy import validate_material_mapping, role_settings, policy_sha256


def transform(text):
    a=np.array([float(s) for s in text.split()]).reshape(4,3)
    out=np.eye(4);out[:3,:]=a.T
    return out


def leaf_mesh(z,path,oid):
    vertices=[];faces=[];active=False
    with z.open(path.lstrip('/')) as stream:
        for event,node in ET.iterparse(stream,events=('start','end')):
            tag=node.tag.rsplit('}',1)[-1]
            if event=='start' and tag=='object':active=int(node.attrib['id'])==oid
            if event!='end':continue
            if active and tag=='vertex':vertices.append(tuple(float(node.attrib[k]) for k in ['x','y','z']))
            if active and tag=='triangle':faces.append(tuple(int(node.attrib[k]) for k in ['v1','v2','v3']))
            if active and tag=='object':break
            if tag in {'vertex','triangle'}:node.clear()
    assert vertices and faces
    return np.asarray(vertices),np.asarray(faces,dtype=np.int32)


def mesh_equivalence(source,vertices,faces,source_translation):
    mesh=trimesh.load_mesh(source,process=True)
    source_vertices=np.asarray(mesh.vertices)+np.asarray(source_translation)
    assert len(mesh.faces)==len(faces),'Triangle count changed'
    # Bambu recentres geometry in float32. Cluster only vertices separated by
    # <0.02 micrometres before comparing the complete oriented triangle soup.
    # This avoids false identity failures on CAD's near-coincident seam nodes.
    tolerance=.00002
    tree=cKDTree(source_vertices);error,index=tree.query(vertices)
    assert error.max()<tolerance,(source,float(error.max()))
    parent=np.arange(len(source_vertices))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in tree.query_pairs(tolerance):
        a,b=root(a),root(b);parent[max(a,b)]=min(a,b)
    for i in np.flatnonzero(parent!=np.arange(len(parent))):parent[i]=root(i)
    def soup(f):
        f=parent[f];first=np.argmin(f,axis=1);r=np.arange(len(f))[:,None]
        f=f[r,(first[:,None]+np.arange(3))%3]
        return f[np.lexsort(f.T[::-1])]
    assert np.array_equal(soup(mesh.faces),soup(index[faces])),'Oriented triangle connectivity changed'
    return dict(triangles=len(faces),maximum_vertex_error_mm=float(error.max()),winding_and_connectivity_match=True)


def geometry(project,prep):
    rows=[]
    with zipfile.ZipFile(project) as z:
        root=ET.fromstring(z.read('3D/3dmodel.model'));ns={'c':root.tag.split('}')[0][1:]}
        conf=ET.fromstring(z.read('Metadata/model_settings.config'))
        expected=list(prep['sources']);source_index=0
        for item in root.findall('c:build/c:item',ns):
            matrix=transform(item.attrib['transform'])
            assert np.max(np.abs(matrix[:3,:3]-np.eye(3)))<1e-7
            oid=item.attrib['objectid'];obj=root.find(f'c:resources/c:object[@id="{oid}"]',ns)
            settings_obj=conf.find(f'object[@id="{oid}"]')
            for part in settings_obj.findall('part'):
                source=expected[source_index];source_index+=1
                assert part.attrib['subtype']==source['subtype']
                meta={m.attrib['key']:m.attrib['value'] for m in part.findall('metadata') if 'key' in m.attrib}
                assert meta['source_file']==Path(source['path']).name
                for key,value in source['overrides'].items():assert meta[key]==value
                child=obj.find(f'c:components/c:component[@objectid="{part.attrib["id"]}"]',ns)
                component_matrix=transform(child.attrib['transform'])
                path=next((v for k,v in child.attrib.items() if k.endswith('}path')),'3D/3dmodel.model')
                vertices,faces=leaf_mesh(z,path,int(part.attrib['id']))
                vertices=trimesh.transform_points(vertices,matrix@component_matrix)
                expected_translation=np.asarray(source['translation'])+np.asarray(prep['offset'])
                facts=mesh_equivalence(ROOT/source['path'],vertices,faces,expected_translation)
                rows.append(dict(source=source['path'],sha256=sha(ROOT/source['path']),subtype=source['subtype'],**facts))
                del vertices,faces;gc.collect()
        assert source_index==len(expected)
    return rows


def settings_check(project,prepared):
    with zipfile.ZipFile(project) as z:d=json.loads(z.read('Metadata/project_settings.config'))
    with zipfile.ZipFile(prepared) as z:expected=json.loads(z.read('Metadata/project_settings.config'))
    keys=['printer_model','nozzle_diameter','nozzle_volume_type','filament_settings_id','filament_map',
          'layer_height','initial_layer_print_height','wall_loops','sparse_infill_density','sparse_infill_pattern',
          'top_shell_layers','bottom_shell_layers','ironing_type','nozzle_temperature','filament_flow_ratio',
          'filament_max_volumetric_speed','textured_plate_temp','curr_bed_type','enable_support','support_type',
          'support_style','support_filament','support_interface_filament','support_top_z_distance',
          'support_interface_spacing','support_interface_top_layers','support_interface_bottom_layers',
          'support_object_xy_distance','flush_volumes_matrix','flush_multiplier','filament_flush_volumetric_speed',
          'flush_into_infill','flush_into_objects','flush_into_support']
    for key in keys:assert d.get(key)==expected.get(key),(key,d.get(key),expected.get(key))
    return d


def support_ducts(gcode,prep):
    routes=json.loads((WORK/'inputs/ducts.json').read_text())['routes']
    matrix=np.asarray(prep['source_to_stl_matrix']);matrix[:3,3]+=prep['offset']
    centers=np.concatenate([trimesh.transform_points(r['centers_mm'],matrix) for r in routes])
    radii=np.concatenate([r['radii_mm'] for r in routes]);tree=cKDTree(centers)
    bore_data=json.loads((WORK/'inputs/insert_bores.json').read_text())
    assert bore_data['actual_mesh_sha256']==prep['sources'][0]['sha256']
    assert bore_data['bore_registration_overlap_mm3']<1e-5
    bores=[]
    for row in bore_data['sites']:
        ends=trimesh.transform_points([row['start'],row['end']],matrix)
        assert np.linalg.norm(ends[0,:2]-ends[1,:2])<1e-6
        bores.append(dict(name=row['name'],center_xy=ends[0,:2],
            low_z=float(ends[:,2].min()),high_z=float(ends[:,2].max()),radius=row['radius'],
            minimum_support_clearance_mm=float('inf'),support_samples=0))
    parsed=parse_gcode(gcode,retain_feature_prefixes=('Support',))
    minimum=float('inf');count=0;failures=[];boundary_contacts=[]
    for layer in parsed.layers:
        if not layer.segments:continue
        chunks=[];widths=[]
        for s in layer.segments:
            a=np.array([s.x0,s.y0,(s.z0+s.z1)/2-(layer.layer_height or .16)/2]);b=np.array([s.x1,s.y1,a[2]])
            n=max(2,int(np.ceil(s.length/.25))+1)
            chunks.append(a+(b-a)*np.linspace(0,1,n)[:,None]);widths.extend([(s.line_width or .62)/2]*n)
        points=np.concatenate(chunks);distance,index=tree.query(points)
        clearance=distance-radii[index]-np.asarray(widths)
        minimum=min(minimum,float(clearance.min()));count+=len(points)
        if clearance.min()<-.03:
            i=int(np.argmin(clearance))
            failures.append(dict(z=layer.z,clearance=float(clearance.min()),point=points[i].tolist(),
                                 nearest_lumen_center=centers[index[i]].tolist(),radius=float(radii[index[i]])))
        for bore in bores:
            height=layer.layer_height or .16
            overlap=min(layer.z,bore['high_z'])-max(layer.z-height,bore['low_z'])
            if overlap<=0:continue
            clearance=np.linalg.norm(points[:,:2]-bore['center_xy'],axis=1)-bore['radius']-np.asarray(widths)
            minimum_bore=float(clearance.min())
            # A slicer intersects geometry at the layer midplane. For example,
            # the rear service floor is nominally Z24.7799, its PLA interface
            # ends at Z24.84, and its first printed insert wall is at Z25.00.
            # Count that sub-half-layer boundary contact separately; do not
            # treat the full extruded bead as support inside a printed bore.
            midplane=layer.z-height/2
            if not bore['low_z']<midplane<bore['high_z']:
                if minimum_bore<-.03:
                    assert overlap<=height/2+.0001
                    boundary_contacts.append(dict(name=bore['name'],layer_z_mm=layer.z,
                        layer_midplane_z_mm=midplane,nominal_boundary_overlap_mm=overlap,
                        maximum_allowed_boundary_overlap_mm=height/2,
                        classification='surface contact outside the sliced bore interior'))
                continue
            bore['minimum_support_clearance_mm']=min(bore['minimum_support_clearance_mm'],minimum_bore)
            bore['support_samples']+=len(points)
            assert minimum_bore>=-.03,('Support enters insert bore',bore['name'],layer.z,minimum_bore)
    assert not failures,failures[:10]
    return dict(support_samples=count,minimum_lumen_clearance_mm=minimum,pass_=True,
        insert_site_count=bore_data['insert_site_count'],
        insert_boundary_contacts=boundary_contacts,
        insert_definition_sha256=sha(WORK/'inputs/insert_bores.json'),insert_bores=[
            {k:(None if isinstance(v,float) and not np.isfinite(v) else v)
             for k,v in bore.items() if k!='center_xy'} for bore in bores])


def main():
    key=sys.argv[1];prep=json.loads((WORK/'preparation.json').read_text())[key]
    prepared=ROOT/prep['path'];directory=prepared.parent
    project=directory/'ready.gcode.3mf' if (directory/'ready.gcode.3mf').exists() else directory/'discovery.gcode.3mf'
    process=settings_check(project,prepared)
    validate_material_mapping(process)
    role = 'crescent_body' if key=='body' else 'crescent_accessories' if key=='accessories' else 'crescent_wing'
    for field, wanted in role_settings(role).items():
        assert process.get(field)==wanted, (key, field, process.get(field), wanted)
    with zipfile.ZipFile(project) as z:gcode=z.read('Metadata/plate_1.gcode')
    (directory/'plate_1.gcode').write_bytes(gcode)
    report=dict(project_sha256=sha(project),settings=process,policy_sha256=policy_sha256(),
                materials=material_extrusions(directory/'plate_1.gcode',process['enable_support']=='1'))
    report['changeover']=audit_changeover(directory/'plate_1.gcode')
    print(key,'settings and actual materials pass',flush=True)
    report['geometry']=geometry(project,prep)
    print(key,'mesh inventory, geometry and placement pass',flush=True)
    if key=='body':report['ducts']=support_ducts(directory/'plate_1.gcode',prep)
    if key=='accessories':
        from cap_support_check import ceiling_supports
        report['cap_ceilings']=ceiling_supports(directory/'plate_1.gcode',prep)
    discovery=directory/'magnet_discovery.json'
    if discovery.exists():
        rows=json.loads(discovery.read_text());assert all(r['magnet_below_resume_plane'] for r in rows)
        from captive_wall_audit import audit_captive_walls
        source=next(r for r in prep['sources'] if r['subtype']=='normal_part')
        report['D6_retaining_walls']=audit_captive_walls(ROOT/source['path'], prep['offset'], rows, directory/'plate_1.gcode')
        expected=sorted(set(r['pause_before_z_mm'] for r in rows))
        text=gcode.decode();pauses=[];lines=text.splitlines()
        for i,line in enumerate(lines):
            if line=='; ND25FN_MAGNET_INSERTION':
                block='\n'.join(lines[i:i+11]);assert 'G1 Z250 F1200' in block and 'M400 U1' in block
                restore=[s for s in lines[i:i+11] if s.startswith('G1 Z')][-1]
                pauses.append(float(restore.split()[1][1:]))
                # The slicer places custom layer G-code before layer extrusion.
                prev=next((s for s in reversed(lines[:i]) if s.startswith('; Z_HEIGHT:')),None)
                assert prev and abs(float(prev.split(':')[1])-pauses[-1])<1e-6
        assert pauses==expected,(pauses,expected)
        report['magnet_pauses_mm']=pauses
    result=json.loads((directory/'result.json').read_text())
    assert result['return_code']==0
    assert not result['sliced_plates'][0]['warning_message'],result['sliced_plates'][0]['warning_message']
    report['slicer_result']=result['sliced_plates'][0]
    report['status']='pass';(directory/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(key,'audit passed',flush=True)


if __name__=='__main__':main()
