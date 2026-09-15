"""Validate the delivered STL bytes, retained interfaces and installed UM fit."""
import itertools
import json
from pathlib import Path

import manifold3d
import numpy as np
import trimesh

import v4_model as model
from rebuild import sha, verify_package

HERE = Path(__file__).resolve().parent
g = model.retained


from mesh_ops import solid


def restored_print_parts():
    manifest = json.loads((HERE/"build_manifest.json").read_text())
    parts = {}
    for name, facts in manifest["files"].items():
        path = HERE/"STL"/name
        assert sha(path) == facts["sha256"], (name, "changed STL")
        parts[name] = trimesh.load_mesh(path, process=True)
    housing = parts[model.BODY_FILE].copy()
    housing.apply_transform(np.linalg.inv(np.array(
        manifest["files"][model.BODY_FILE]["assembly_to_print_matrix"])))
    cap = parts["02_Closed_Cap_PRINT_TWO.stl"].copy()
    p = cap.vertices.copy()
    cap.vertices = np.c_[-6.05-p[:,2], p[:,0], -31-p[:,1]]
    ring = parts["03_Tweeter_Retainer_PRINT_TWO.stl"].copy()
    p = ring.vertices.copy()
    ring.vertices = np.c_[p[:,2]+g.M.retainer_back_x, p[:,0], p[:,1]-31]
    opposite = np.diag([-1.,1.,-1.,1.])
    cap_upper = cap.copy(); cap_upper.apply_transform(opposite)
    ring_upper = ring.copy(); ring_upper.apply_transform(opposite)
    return parts, {"housing": housing, "cap_lower": cap, "cap_upper": cap_upper,
                   "retainer_lower": ring, "retainer_upper": ring_upper}


def um_mesh(state):
    return model.restored_core(state)


def ray_distances(mesh, origin, direction):
    origin = np.array(origin, dtype=float)
    direction = np.array(direction, dtype=float)
    points, _, _ = mesh.ray.intersects_location([origin], [direction], multiple_hits=True)
    return np.sort((points-origin)@direction)


def geometry_signature(mesh):
    # Order-independent float32 vertex/triangle geometry; normals are checked
    # separately. This detects changed mating geometry without byte-order noise.
    vertices, inverse = np.unique(np.asarray(mesh.vertices, dtype=np.float32), axis=0, return_inverse=True)
    triangles = np.sort(inverse[mesh.faces], axis=1)
    order = np.lexsort(triangles.T[::-1])
    import hashlib
    return hashlib.sha256(vertices.tobytes()+triangles[order].tobytes()).hexdigest()


def box_solid(lo,hi):
    lo,hi = np.array(lo,float),np.array(hi,float)
    return solid(trimesh.creation.box(hi-lo,
        transform=trimesh.transformations.translation_matrix((lo+hi)/2)))


def cylinder_region(x,y,r,lo,hi):
    import manifold3d as md
    return md.Manifold.cylinder(hi-lo,r,r,160).translate((x,y,lo))


def difference_volume(a,b):
    return abs((a-b).volume())+abs((b-a).volume())


def sampled_profile_curvature(height,spacing=.25):
    """Macroscopic STL curvature, filtering only the 0.3 mm tessellation.

    A 1.5 mm fitting window retains the former abrupt arc (about R1 mm)
    but does not mistake individual faceted triangle boundaries for ribs.
    Exclude the endpoint fitting region and intentional bore/cap edges.
    """
    from scipy.signal import savgol_filter
    slope=savgol_filter(height,7,3,deriv=1,delta=spacing)
    second=savgol_filter(height,7,3,deriv=2,delta=spacing)
    curvature=np.abs(second)/(1+slope*slope)**1.5
    maximum=float(curvature[3:-3].max())
    return {'maximum_curvature_per_mm':maximum,'minimum_radius_mm':1/maximum}


def rear_profile_facts(mesh):
    """Measure the exposed rear join directly from STL ray intersections.

    Stay outside the UM opening and the removable T cap. A raised ridge
    appears as a rearward excursion followed by a forward return while
    walking up toward the tweeter; the new sweep must not have that return.
    Curvature in both directions also catches an abrupt monotone arc or
    lateral ribs which a reversal-only check misses.
    """
    rows=[]
    for lateral in [-35.,-30.,-20.,-10.,0.,10.,20.,30.,35.]:
        start=max(398.,model.interface.UM_CUTOUT[1]+np.sqrt(42.25**2-lateral**2)+.25)
        stop=(model.LOWER_AXIS_Y-np.sqrt(28.5**2-lateral**2)-.15
              if abs(lateral)<28.5 else 440.)
        vertical=np.arange(start,stop,.25)
        origins=np.c_[np.full(len(vertical),lateral),vertical,np.full(len(vertical),-30.)]
        points,indices,_=mesh.ray.intersects_location(origins,np.tile([0.,0.,1.],(len(vertical),1)),multiple_hits=True)
        height=np.full(len(vertical),np.inf)
        np.minimum.at(height,indices,points[:,2])
        assert np.all(np.isfinite(height)),('open rear transition',lateral,vertical[~np.isfinite(height)])
        reversal=max(float(np.max(height[j:j+33])-height[j]) for j in range(len(height)))
        rows.append({'lateral_x_mm':lateral,'samples':len(vertical),
                     'surface_region':'rear_loft' if abs(lateral)<=20 else 'sloping_perimeter',
                     'maximum_forward_return_over_8_mm':reversal,
                     **sampled_profile_curvature(height),
                     'vertical_y_mm':vertical.tolist(),'rear_z_mm':height.tolist()})
    transverse=[]
    for vertical in [408.5,412.,416.,420.]:
        lateral=np.arange(-35.,35.01,.25)
        origins=np.c_[lateral,np.full(len(lateral),vertical),np.full(len(lateral),-30.)]
        points,indices,_=mesh.ray.intersects_location(origins,np.tile([0.,0.,1.],(len(lateral),1)),multiple_hits=True)
        height=np.full(len(lateral),np.inf)
        np.minimum.at(height,indices,points[:,2])
        assert np.all(np.isfinite(height)),('open rear transverse profile',vertical)
        # At the reference's narrow waist, X=+/-30..35 now samples the
        # inclined edge and its small rim roll. Keep those measurements,
        # but distinguish them from the broad central rear thickness loft.
        core=abs(lateral)<=20.
        transverse.append({'vertical_y_mm':vertical,'samples':len(lateral),
            **sampled_profile_curvature(height[core]),
            'broad_loft_lateral_range_mm':[-20,20],
            'complete_profile_curvature':sampled_profile_curvature(height),
            'lateral_x_mm':lateral.tolist(),'rear_z_mm':height.tolist()})
    central=[r for r in rows if r['surface_region']=='rear_loft']
    perimeter=[r for r in rows if r['surface_region']=='sloping_perimeter']
    return {'profiles':rows,'transverse_profiles':transverse,
            'maximum_forward_return_mm':max(r['maximum_forward_return_over_8_mm'] for r in central),
            'minimum_longitudinal_radius_mm':min(r['minimum_radius_mm'] for r in central),
            'minimum_transverse_radius_mm':min(r['minimum_radius_mm'] for r in transverse),
            'minimum_complete_perimeter_profile_radius_mm':min(
                [r['minimum_radius_mm'] for r in perimeter]+
                [r['complete_profile_curvature']['minimum_radius_mm'] for r in transverse]),
            'curvature_sample_spacing_mm':.25,'curvature_fit_window_mm':1.5,
            'required_minimum_longitudinal_radius_mm':10.,
            'required_minimum_transverse_radius_mm':50.}


def handoff_facts(state, body, lm):
    """Check the shared body against each independently configured LM route."""
    import os
    import subprocess
    import sys
    # Import each original routing state in isolation: the shared candidate
    # deliberately fixes its own upper gallery profile.
    code = '''import json
from lx521_baffle.obiwan import route as r
t=r.ts_cable_points(.08);t=t[(t[:,1]>305)&(t[:,1]<331)]
u=r.route_cable_points(.08);u=u[u[:,1]>309]
print(json.dumps({'T':t.tolist(),'UM':u.tolist()}))'''
    env={**os.environ,'LX_ROUTING_PROFILE':'obiwan',
         'LX_STAND_FOOT':'1' if state=='floor_stand' else '0',
         'PYTHONPATH':str(model.ROOT/'src')}
    routes=json.loads(subprocess.check_output([sys.executable,'-c',code],env=env,text=True))
    joined=body+solid(lm)
    result={}
    for name,radius,low,high in [('T',2.8,310.,326.),('UM',3.9,311.,328.)]:
        points=np.asarray(routes[name])
        gauge=solid(model.tube_mesh(points,radius))
        # Clip away artificial spherical ends at sampled-route truncations.
        # The UM interval starts below the shared LM/UM interface and ends
        # inside the open driver aperture.
        gauge^=box_solid([-50,low,-20],[60,high,25])
        overlap=abs((joined^gauge).volume())
        assert overlap<1e-5,(state,name,'LM/UM cable handoff obstruction',overlap)
        result[name]={'gauge_diameter_mm':2*radius,'installed_y_interval_mm':[low,high],
                      'shared_body_and_LM_overlap_mm3':overlap}
    return result


def validate_wings(assembled,facts):
    wings=json.loads((HERE/'wing_validation.json').read_text())
    assert wings['source_sha256']==sha(HERE/'build_wings.py')
    assert wings['model_source_sha256']==sha(HERE/'v4_model.py')
    assert wings['housing_stl_sha256']==sha(HERE/'STL'/model.BODY_FILE)
    for filename,row in wings['parts'].items():
        path=HERE/'STL/wings'/filename
        assert sha(path)==row['stl_sha256']
        wing=trimesh.load_mesh(path,process=True)
        wing.apply_transform(np.linalg.inv(row['source_to_stl_matrix']))
        fits={};w=solid(wing)
        for state,(_,_,body) in assembled.items():
            overlap=abs((body^w).volume())
            assert overlap<1e-5,(filename,state,'wing collision',overlap)
            fits[state]={'overlap_mm3':overlap}
        magnets=[]
        for pocket in row['relocated_UM_magnets']:
            body_magnet=min(facts['configurations']['no_floor_stand']['relocated_magnets'],
                            key=lambda r:abs(r['angle_deg']-pocket['angle_deg']))
            normal=np.array(pocket['normal'])
            delta=np.array(pocket['center_mm'])-body_magnet['new_center_mm']
            separation=float(delta@normal);error=np.linalg.norm(delta-separation*normal)
            assert error<.002,(filename,'magnet axis misalignment',delta)
            origin=np.array(pocket['contact_point_mm'])+(pocket['pocket_face_offset_mm']+model.MAGNET_CAVITY_DEPTH/2)*normal
            hits=ray_distances(wing,origin,-normal)
            assert len(hits)>=2 and hits[1]-hits[0]>model.MAGNET_SKIN-.04,(filename,'wing magnet skin',hits)
            magnets.append({'angle_deg':pocket['angle_deg'],'axis_error_mm':float(error),
                'magnet_center_separation_mm':separation,'measured_face_skin_mm':float(hits[1]-hits[0])})
        facts['matching_upper_wings'][filename]={'stl_sha256':row['stl_sha256'],'fits':fits,
            'preserved_LM_magnet':True,'closed_magnet_voids':3,'UM_magnets':magnets,
            'minimum_upper_shoulder_gap_mm':row['minimum_upper_shoulder_gap_mm']}
    facts['limits']=['Digital geometry checks; physical fit and print process remain untested.',
        'The lower acoustic flare is changed; acoustic performance is not established.',
        'Closed magnet pockets require pause-and-bury installation.',
        'The 223.5 mm diagonal footprint excludes brim, supports and printer exclusions.',
        'V4 driver retention, gaskets and O-rings retain the supplied package assumptions.']
    facts['status']='passed'
    (HERE/'validation.json').write_text(json.dumps(facts,indent=2)+'\n')
    print('Validation passed:',HERE/'validation.json',flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--geometry-only',action='store_true')
    mode.add_argument('--wings-only',action='store_true',help='Reuse a body report only when its exact source and build hashes still match')
    args=parser.parse_args()
    from mesh_ops import to_trimesh
    verify_package()
    manifest=json.loads((HERE/'build_manifest.json').read_text())
    for key,file in [('model_source_sha256','v4_model.py'),('builder_source_sha256','rebuild.py'),
                     ('mesh_ops_source_sha256','mesh_ops.py')]:
        assert manifest[key]==sha(HERE/file),(key,'stale build')
    assert manifest['route_source_sha256']==sha(model.route.__file__)
    assert manifest['compatible_LM_configurations']==list(model.LM_STATES)
    assert [name for name in manifest['files'] if name.startswith('01_')]==[model.BODY_FILE]
    canonical_um,canonical_provenance=model.restored_core(model.UM_SOURCE_STATE)
    assert canonical_provenance==manifest['UM_source']
    if args.wings_only:
        facts=json.loads((HERE/'geometry_validation.json').read_text())
        assert facts['status']=='geometry_passed_wings_not_checked'
        assert facts['validation_source_sha256']==sha(__file__)
        assert facts['source_build_manifest_sha256']==sha(HERE/'build_manifest.json')
        _,parts=restored_print_parts()
        mesh=model.installed(parts['housing']);body=solid(mesh)
        assembled={state:(parts,mesh,body) for state in model.LM_STATES}
        validate_wings(assembled,facts)
        return
    printed,parts=restored_print_parts()
    facts={'status':'checking','units':'mm','physical_fit':'not tested',
           'validation_source_sha256':sha(__file__),'source_build_manifest_sha256':sha(HERE/'build_manifest.json'),'STL':{},'unchanged_companions':{},
           'configurations':{},'matching_upper_wings':{},'shared_body_file':model.BODY_FILE}
    for name,mesh in printed.items():
        assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
        assert np.all(mesh.area_faces>0) and np.all(mesh.unique_faces())
        shells=mesh.split(only_watertight=False)
        assert sum(v.volume>0 for v in shells)==1
        void_count=sum(bool(v.volume<0) for v in shells)
        assert void_count==(model.UM_MAGNET_COUNT if name.startswith('01_') else 0),(name,'unexpected sealed void',void_count)
        solid(mesh)
        facts['STL'][name]={'sha256':sha(HERE/'STL'/name),'watertight':True,'material_components':1,
            'closed_void_shells':void_count,'consistent_winding':True,'zero_area_facets':0,
            'volume_mm3':float(mesh.volume),'size_mm':mesh.extents.tolist()}
    for name in ['02_Closed_Cap_PRINT_TWO.stl']:
        original=trimesh.load_mesh(model.PACKAGE/'cad'/name,process=True)
        assert geometry_signature(original)==geometry_signature(printed[name])
        facts['unchanged_companions'][name]={'identical_triangle_geometry_to_package':True}
    from validate_hardware import validate_hardware
    facts['M3_hardware'] = validate_hardware()
    print('Shared body: one material solid, four closed magnet cavities; cap unchanged; M3 hardware checked',flush=True)
    # Topology checks populate large adjacency caches on the print-frame
    # copy. Subsequent measurements use only the installed-frame meshes.
    for checked in printed.values():
        checked._cache.clear()
    import gc
    gc.collect()

    from lx521_baffle.um_fit import mu10_body_keepout,mu10_body_reference_facts
    mu10=[]
    for part in mu10_body_keepout(include_flange=True).solids():
        vertices,faces=part.tessellate(.08,.1)
        mu10.append(solid(trimesh.Trimesh(np.array([tuple(v) for v in vertices]),np.array(faces),process=True)))
    facts['MU10_reference']=mu10_body_reference_facts()
    assembled={}
    canonical_core=solid(canonical_um)
    mesh=model.installed(parts['housing']);body=solid(mesh)
    for state in model.LM_STATES:
        assembled[state]=(parts,mesh,body)
        original,provenance=model.restored_core(state)
        old=solid(original)
        lm,lm_provenance=model.restored_core(state,'lm')
        overlap=abs((body^solid(lm)).volume())
        assert overlap<1e-5,(state,'LM interference',overlap)
        mu_overlaps=[abs((body^ref).volume()) for ref in mu10]
        assert max(mu_overlaps)<.002,(state,'MU10 intrusion',mu_overlaps)
        # Compare actual functional regions with the independently released UM.
        # The old duct roof below the seat is intentionally backfilled. The
        # actual 14.3 mm contact surface and complete flange recess remain.
        regions={'driver_recess_and_seat':cylinder_region(0,model.interface.UM_CUTOUT[1],49.25,14.25,22),
                 'LM_center_tie':box_solid([-20.2,310,6.81],[ -13.8,327,18.3])}
        for index,(x,y) in enumerate(model.interface.UM_PILOT_XY):
            regions[f'UM_pilot_{index}']=cylinder_region(x,y,3,
                model.interface.UM_SEAT_Z-model.interface.UM_PILOT_DEPTH_MM-.1,22)
        for index,x in enumerate(model.interface.JOINT_EAR_X):
            regions[f'LM_receiver_{index}']=cylinder_region(x,model.interface.JOINT_EAR_Y,4.8,6,18.3)
        preservation={}
        for name,region in regions.items():
            delta=difference_volume(body^region,old^region)
            assert delta<.015,(state,name,'functional region changed',delta)
            preservation[name]={'symmetric_difference_volume_mm3':delta}
        under_seat=cylinder_region(0,model.interface.UM_CUTOUT[1],49.25,13.3,14.25)
        # This non-mating duct roof follows the one canonical UM source.
        # Its two former profiles cannot both be exact references for one
        # shared body. The contact seat/recess and every mounting pilot are
        # still checked above against BOTH independent UM configurations.
        added_roof=abs(((body-canonical_core)^under_seat).volume())
        removed_roof=abs(((canonical_core-body)^under_seat).volume())
        assert removed_roof<.015,(state,'seat roof lost material',removed_roof)
        receivers=[]
        for x in model.interface.JOINT_EAR_X:
            y=model.interface.JOINT_EAR_Y
            floor=ray_distances(mesh,[x,y,-30],[0,0,1])-30
            shell=ray_distances(mesh,[x+3.5,y,-30],[0,0,1])-30
            # Cosmetic material may continue ahead of the native front
            # lands. The receiver floor, mating face and single solid
            # cover must remain exact and uninterrupted.
            assert len(floor)==2 and abs(floor[0]-16.4)<.001 and floor[1]>=18.299
            assert len(shell)==2 and abs(shell[0]-12.4)<.001 and shell[1]>=18.299
            radii=[ray_distances(mesh,[x,y,14],[np.cos(a),np.sin(a),0])[0]
                   for a in np.linspace(0,2*np.pi,32,endpoint=False)]
            assert min(radii)>2.28 and max(radii)<2.31
            receivers.append({'axis_xy_mm':[x,y],'diameter_range_mm':[2*min(radii),2*max(radii)],
                              'floor_z_mm':floor[0],'mating_face_z_mm':shell[0],
                              'continuous_cover_front_z_mm':floor[1],'LM_half_lap_gap_mm':.2})
        raised_apron=body^box_solid([-50,300,18.301],[50,321.99,30])
        assert abs(raised_apron.volume())<.002,(state,'raised lower apron',raised_apron.volume())
        lm_front=solid(lm)^box_solid([-60,300,18.28],[60,340,18.30])
        um_front=body^box_solid([-60,300,18.299],[60,340,48.3])
        front_occlusion=float((um_front.project()^lm_front.project()).area())
        assert front_occlusion<.002,(state,'LM front face occluded',front_occlusion)
        # The previously exposed notches are now continuous material above
        # the LM ears. These probes are outside the retained screw bores.
        notch_points=[[sign*x,y,17.5] for sign in [-1,1] for x,y in [(33,323),(35,324),(36,326)]]
        assert np.all(mesh.contains(notch_points)),(state,'LM ear fairing notch')
        relocated=model.magnet_pockets(original,'body')
        cavities=[v for v in mesh.split(only_watertight=False) if v.volume<0]
        magnets=[]
        for old_void,pocket,transform in relocated:
            actual=min(cavities,key=lambda v:np.linalg.norm(v.center_mass-pocket.center_mass))
            assert np.linalg.norm(actual.center_mass-pocket.center_mass)<.0002
            assert abs(actual.volume+pocket.volume)<.003
            site=min(model.magnet_sites(),key=lambda s:np.linalg.norm(s['contact']-pocket.center_mass))
            normal=site['normal']
            # The external skin stays curved. Check actual centre thickness
            # against the solved internal datum; the full pocket/roof cover
            # is independently sampled in the depth/magnet finish check.
            face=model.magnet_burial(site['angle_deg'],'body')
            hits=ray_distances(mesh,site['contact']+(face-model.MAGNET_CAVITY_DEPTH/2)*normal,normal)
            assert len(hits)>=2 and abs(hits[1]-hits[0]+face)<.015,(state,'magnet skin',hits)
            magnets.append({'old_center_mm':old_void.center_mass.tolist(),'new_center_mm':actual.center_mass.tolist(),
                            'transform':transform.tolist(),'face_skin_mm':float(hits[1]-hits[0]),
                            'angle_deg':site['angle_deg'],'normal':normal.tolist(),'pocket_face_offset_mm':face,
                            'contact_point_mm':site['contact'].tolist()})
        authority_path=(HERE/'STL'/model.BODY_FILE).with_suffix('.print.json')
        authority=json.loads(authority_path.read_text())
        assert authority['stl_sha256']==sha(HERE/'STL'/model.BODY_FILE)
        assert authority['compatible_LM_configurations']==list(model.LM_STATES)
        check=printed[model.BODY_FILE].copy()
        check.apply_transform(np.linalg.inv(authority['source_to_stl_matrix']))
        assert np.allclose(check.vertices,mesh.vertices,atol=1e-8)
        facts['configurations'][state]={'UM_source':provenance,'LM_source':lm_provenance,
            'body_file':model.BODY_FILE,'body_sha256':sha(HERE/'STL'/model.BODY_FILE),
            'LM_overlap_mm3':overlap,'functional_region_preservation':preservation,'LM_receivers':receivers,
            'raised_apron_above_LM_face_mm3':float(raised_apron.volume()),
            'LM_front_occluded_area_mm2':front_occlusion,
            'front_interface_nominal_z_mm':model.UM_LM_FRONT_Z,
            'MU10_keepout_overlap_mm3':mu_overlaps,'relocated_magnets':magnets,
            'old_duct_backfill_below_seating_surface_mm3':added_roof,
            'duct_roof_preservation_reference':canonical_provenance,
            'removed_canonical_duct_roof_mm3':removed_roof,
            'LM_ear_notch_probes_filled':True,'installed_bounds_mm':mesh.bounds.tolist(),
            'print_authority_sha256':sha(authority_path),'LM_cable_handoffs':handoff_facts(state,body,lm)}
        print(state,'LM, driver seat/pilots, magnet skins and cable handoffs checked with the shared body',flush=True)

    for state,(parts,mesh,body) in assembled.items():
        meshes={**parts,'driver_lower':g.driver_mesh(0),'driver_upper':g.driver_mesh(1)}
        solids={name:solid(part) for name,part in meshes.items()}
        overlaps={}
        for a,b in itertools.combinations(solids,2):
            if np.any(meshes[a].bounds[1]<meshes[b].bounds[0]) or np.any(meshes[b].bounds[1]<meshes[a].bounds[0]):
                volume=0.
            else: volume=abs((solids[a]^solids[b]).volume())
            assert volume<1e-5,(state,a,b,'interference',volume)
            overlaps[a+' / '+b]=volume
        facts['configurations'][state]['V4_assembly_overlap_mm3']=overlaps
    mesh=assembled['no_floor_stand'][1]
    profile=[]
    for forward in [10.5,12.5,14.5,16.5,18.29]:
        _,old_radius=g.v3.inner_radius(forward-model.INSTALLED_Z_OFFSET,g.P)
        expected=model.THROAT_R+(old_radius-model.THROAT_R)/1.5
        actual=ray_distances(mesh,[0,model.LOWER_AXIS_Y,forward],[0,-1,0])[0]
        assert abs(actual-expected)<.025,('lower flare',forward,actual,expected)
        profile.append({'forward_z_mm':forward,'measured_lower_reach_mm':float(actual),'target_mm':float(expected)})
    # Cap boundary continuity: no external fairing step at the lower rear cap.
    angles=np.linspace(0,2*np.pi,181)
    xx=np.linspace(-17.8,-6.05,21)[:,None]
    yy=28.2*np.cos(angles)[None,:];zz=-31+28.2*np.sin(angles)[None,:]
    cap_delta=np.max(abs(model.envelope_field(xx,yy,zz)-model.tweeter_outer(xx,yy,zz)))
    assert cap_delta<1e-6,('cap rim exterior changed',cap_delta)
    facts['layout']={'previous_front_tweeter_axis_y_mm':470.781,'front_tweeter_axis_y_mm':model.LOWER_AXIS_Y,
        'rear_tweeter_axis_y_mm':model.LOWER_AXIS_Y+62,'lowering_mm':model.LOWERING_MM,
        'UM_to_front_tweeter_axis_spacing_mm':model.LOWER_AXIS_Y-model.interface.UM_CUTOUT[1],
        'lower_flare_slope_factor':1.5,'lower_mouth_reach_mm':model.LOWER_MOUTH_REACH,
        'minimum_nominal_front_ligament_mm':model.FRONT_LIGAMENT,'measured_flare_profile':profile,
        'cap_rim_maximum_field_change':float(cap_delta),'opposed_tweeter_pitch_mm':62,
        'throat_diameter_mm':37.2,'waveguide_depth_mm':8.8,'tweeter_depth_mm':35.8,
        'overall_fused_body_depth_mm':float(mesh.extents[2]),
        'UM_bowl_maximum_rise_above_driver_rim_mm':float(mesh.bounds[1,2]-18.3)}
    print('Closer layout, steeper lower flare and flush service-cap boundary checked',flush=True)

    rear=rear_profile_facts(mesh)
    assert rear['maximum_forward_return_mm']<.04,('rear fusion ridge',rear['maximum_forward_return_mm'])
    assert rear['minimum_longitudinal_radius_mm']>10.,('abrupt rear thickness arc',rear['minimum_longitudinal_radius_mm'])
    assert rear['minimum_transverse_radius_mm']>50.,('rear transverse ribs',rear['minimum_transverse_radius_mm'])
    assert rear['minimum_complete_perimeter_profile_radius_mm']>.6,('rear perimeter crease',rear['minimum_complete_perimeter_profile_radius_mm'])
    for state in model.LM_STATES:
        facts['configurations'][state]['rear_transition']=rear
    print('Rear STL profiles checked: no raised ridge; broad curvature along and across the join',flush=True)
    del printed,check,solids,meshes,mu10
    mesh._cache.clear()
    for part in parts.values():
        part._cache.clear()
    gc.collect()

    # An independent, undersized solid cable gauge must clear both bodies.
    gauge=g.extract(lambda x,y,z:np.maximum(model.ducts_field(x,y,z)+.1,
        model.CONNECT_START_Y-1-(z+model.INSTALLED_Y_OFFSET)),
        [[-7,16],[10,59],[model.CONNECT_OVERLAP_Y-1-model.INSTALLED_Y_OFFSET,30]],.30)
    gauge_solid=solid(model.installed(gauge))
    for state,(_,_,body) in assembled.items():
        volume=abs((body^gauge_solid).volume())
        assert volume<1e-5,(state,'cable obstruction',volume)
        facts['configurations'][state]['wire_gauge_overlap_mm3']=volume
    _,curve=model.connector_curve()
    ts=np.linspace(model.CONNECT_START_Y,model.CONNECT_END_Y,2001)
    d=curve.derivative()(ts);dd=curve.derivative(2)(ts)
    velocity=np.c_[d[:,0],np.ones(len(ts)),d[:,1]]
    acceleration=np.c_[dd[:,0],np.zeros(len(ts)),dd[:,1]]
    curvature=np.linalg.norm(np.cross(velocity,acceleration),axis=1)/np.linalg.norm(velocity,axis=1)**3
    minimum_radius=float(1/curvature.max())
    assert minimum_radius>12
    # Independently recover the join's taper from the immutable V4 route
    # samples. Its local diameter is smaller than the maximum trunk size.
    retained_centers,retained_radii=g.ROUTES['upper_and_trunk']
    retained_y=retained_centers[:,2]+model.INSTALLED_Y_OFFSET
    near=abs(retained_y-model.CONNECT_END_Y)<1.
    polynomial=np.polyfit(retained_y[near]-model.CONNECT_END_Y,retained_radii[near],3)
    expected_jet=np.array([np.polyval(np.polyder(polynomial,k),0.) for k in range(3)])
    h=.05;y=model.CONNECT_END_Y
    minus,center,plus=model.connector_radius(np.array([y-h,y,y+h]))
    actual_jet=np.array([center,(plus-minus)/(2*h),(plus-2*center+minus)/h**2])
    jet_error=abs(actual_jet-expected_jet)
    assert np.all(jet_error<[.0001,.001,.001]),('cable diameter step at retained join',jet_error)
    # Sample the new enclosed section beyond its overlap with the original
    # thin UM cover. Check material 0.8 mm outside the cable lumen in each
    # normal plane, avoiding the intentional branch/chamber entries.
    wall_points=[]
    for y in np.linspace(333,425,93):
        xz=curve(y);derivative=curve.derivative()(y)
        tangent=np.array([derivative[0],1,derivative[1]]);tangent/=np.linalg.norm(tangent)
        u=np.cross(tangent,[0,0,1]);u/=np.linalg.norm(u);v=np.cross(tangent,u)
        r=model.connector_radius(y)
        for angle in np.linspace(0,2*np.pi,24,endpoint=False):
            wall_points.append(np.array([xz[0],y,xz[1]])+(r+.8)*(u*np.cos(angle)+v*np.sin(angle)))
    # Both LM checks use the same body bytes, so measure its cover once.
    # Keep exact occupancy inside a closed local crop. Raycasting batches
    # against the complete two-million-face body can exceed the CAD guard
    # even though only this small gallery region is being measured.
    points=np.asarray(wall_points)
    cover_mesh=to_trimesh(body^box_solid(points.min(axis=0)-2.,points.max(axis=0)+2.))
    hits=np.concatenate([cover_mesh.contains(wall_points[i:i+16])
                         for i in range(0,len(wall_points),16)])
    assert np.all(hits),('thin/open new cable cover',np.array(wall_points)[~hits][:8].tolist())
    magnet_gaps=[]
    original,_=model.restored_core('no_floor_stand')
    for _,pocket,_ in model.magnet_pockets(original,'body'):
        gap=solid(pocket).min_gap(gauge_solid,10.)
        assert gap>1.,('magnet/cable gallery clearance',pocket.center_mass.tolist(),gap)
        magnet_gaps.append(gap)
    facts['wiring']={'connection':'C2 from preserved LM handoff through the outer UM gallery into the V4 side trunk',
        'minimum_connector_centerline_radius_mm':minimum_radius,'gauge_branch_diameter_mm':4.6,
        'gauge_connection_diameter_range_mm':[float(2*(model.connector_radius(ts).min()-.1)),float(2*(model.connector_radius(ts).max()-.1))],'sampled_new_cover_skin_mm':.8,
        'retained_join_radius_mm':float(expected_jet[0]),
        'radius_join_value_slope_curvature_errors':jet_error.tolist(),
        'cover_samples_checked':len(wall_points),'old_free_tail_removed':True,
        'old_UM_surface_route_filled':True,'magnet_to_cable_gauge_minimum_gaps_mm':magnet_gaps}
    print('Cable gauge, bend radius and enclosed cover checked',flush=True)

    if args.geometry_only:
        facts['status']='geometry_passed_wings_not_checked'
        (HERE/'geometry_validation.json').write_text(json.dumps(facts,indent=2)+'\n')
        return

    validate_wings(assembled,facts)



if __name__=='__main__':
    main()
