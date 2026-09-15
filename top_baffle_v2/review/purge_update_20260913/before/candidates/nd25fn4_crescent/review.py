"""Review the actual fused print meshes, LM interface and registered drivers."""
import json
from pathlib import Path
import numpy as np
import trimesh
import v4_model as model
from validate import restored_print_parts,box_solid
from mesh_ops import solid,to_trimesh
from rebuild import sha

HERE=Path(__file__).resolve().parent


def mu10_reference():
    path=model.ROOT/'vendor/SEAS/MU10RB-SL/H1658-04_MU10RB-SL_driver.stl'
    from lx521_baffle.um_fit import MU10_REFERENCE_STL_SHA256
    assert sha(path)==MU10_REFERENCE_STL_SHA256
    mesh=trimesh.load_mesh(path,process=True)
    transform=trimesh.transformations.translation_matrix([0,model.interface.UM_CUTOUT[1],14.3])
    transform=transform @ trimesh.transformations.rotation_matrix(np.deg2rad(58),[0,0,1])
    transform=transform @ trimesh.transformations.rotation_matrix(np.pi/2,[1,0,0])
    mesh.apply_transform(transform)
    return mesh


def render_object(mesh,color,opacity=1.):
    mesh=mesh.copy();p=mesh.vertices.copy()
    mesh.vertices=np.c_[p[:,2],p[:,0],p[:,1]-421.5]
    return mesh,tuple(c/255 for c in color[:3]),opacity


def main():
    import render_mesh as renderer
    original_actor=renderer.actor
    def exact_surface_actor(*args,**kwargs):
        actor=original_actor(*args,**kwargs)
        # Flat surface lighting prevents micron-scale boolean slivers from
        # contaminating normals across large planar triangles.
        actor.GetProperty().SetInterpolationToFlat()
        return actor
    renderer.actor=exact_surface_actor
    render=renderer.render
    output=HERE/'views';output.mkdir(exist_ok=True)
    _,parts=restored_print_parts()
    meshes=[]
    for name,mesh in parts.items():
        color=([66,141,189,255] if name=='housing' else
               [52,108,153,255] if name.startswith('cap') else [100,173,204,255])
        meshes.append(('fused_UM_V4' if name=='housing' else name,model.installed(mesh),color,
                       220000 if name=='housing' else 40000))
    for i in [0,1]:
        meshes.append((f'Dayton_ND25FN4_{i}_reference',model.installed(model.retained.driver_mesh(i)),
                       [238,148,43,255],110000))
    meshes.append(('SEAS_MU10_reference_terminals_omitted',mu10_reference(),[213,137,49,255],120000))
    lm,lm_provenance=model.restored_core('no_floor_stand','lm')
    meshes.append(('existing_LM_carrier',lm,[157,170,177,255],90000))
    scene=trimesh.Scene()
    for name,mesh,color,limit in meshes:
        preview=model.retained.v3.vtk_decimate(mesh,limit) if len(mesh.faces)>limit else mesh.copy()
        preview.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
            name=name,baseColorFactor=color,metallicFactor=.04,roughnessFactor=.68))
        preview.apply_scale(.001)
        scene.add_geometry(preview,node_name=name,geom_name=name)
    path=output/'nd25fn4_color_review.glb';scene.export(path,include_normals=True)
    assert len(trimesh.load_scene(path).geometry)==9
    manifest={'glb_sha256':sha(path),'source_build_manifest_sha256':sha(HERE/'build_manifest.json'),
        'source_review_sha256':sha(__file__),'units':'metres','up':'+Y',
        'scene_parts':list(scene.geometry),'LM_reference':lm_provenance,
        'note':'Blue is one fused UM + V4 body with separate service caps/retainers. Orange drivers are references. Grey LM is separate. Review-only decimation.'}
    objects=[render_object(mesh,color) for _,mesh,color,_ in meshes]
    render(objects,output/'installed.png',camera=(240,-115,70),focus=(2,0,4),
           scale=140,size=(1200,1500))
    render(objects,output/'rear_installed.png',camera=(-240,-100,62),focus=(0,0,4),
           scale=140,size=(1200,1500))
    render(objects,output/'mount_detail.png',camera=(220,-75,30),focus=(9,0,-8),
           scale=30,size=(1450,1050))
    render(objects,output/'rear_transition.png',camera=(-200,-80,30),focus=(0,0,-8),
           scale=37,size=(1450,1050))
    render(objects,output/'LM_interface.png',camera=(-180,-40,-105),focus=(12,0,-104),
           scale=46,size=(1450,1050))
    # A transparent crop reveals the complete three-dimensional route.
    # A single planar section obscures the rearward ends and can misleadingly
    # make this continuous gallery appear to stop inside the shell.
    body=model.installed(parts['housing'])
    render([render_object(body,[66,141,189,255])],output/'rear_transition_bare.png',
           camera=(-200,-80,30),focus=(0,0,-8),scale=37,size=(1450,1050))
    render([render_object(body,[66,141,189,255])],output/'rear_transition_side.png',
           camera=(0,240,-10),focus=(0,0,-10),scale=35,size=(1100,1300))
    render([render_object(body,[66,141,189,255])],output/'UM_solid_underside.png',
           camera=(-130,-30,-135),focus=(8,0,322-421.5),scale=48,size=(1400,1000))
    render([render_object(body,[66,141,189,255])],output/'UM_lower_band_rear.png',
           camera=(-190,0,321-421.5),focus=(7,0,321-421.5),scale=43,size=(1450,900))
    render([render_object(body,[66,141,189,255])],output/'UM_lower_band_oblique.png',
           camera=(-95,70,-145),focus=(7,0,321-421.5),scale=44,size=(1450,1050))
    render([render_object(body,[66,141,189,255]),render_object(lm,[157,170,177,255])],
           output/'duct_handoff_closed.png',camera=(-155,50,-125),
           focus=(8,14,316-421.5),scale=12,size=(1300,1050))
    render([render_object(body,[66,141,189,255])],output/'M2_tie_rounded_mouth.png',
           camera=(-50,-65,-50),focus=(10,-17,329-421.5),scale=10,size=(1000,1000))
    cut=to_trimesh(solid(body)^box_solid([18,324,-18],[59,437,18.3]))
    routes,_,_,_=model.wiring()
    gauge=model.retained.extract(lambda x,y,z:np.maximum(model.ducts_field(x,y,z)+.15,
        model.CONNECT_START_Y-1-(z+model.INSTALLED_Y_OFFSET)),
        [[-7,16],[10,59],[model.CONNECT_OVERLAP_Y-1-model.INSTALLED_Y_OFFSET,30]],.4)
    gauge=model.installed(gauge)
    gauge=to_trimesh(solid(gauge)^box_solid([18,324,-18],[59,437,18.3]))
    render([render_object(cut,[66,141,189,255],.22),render_object(gauge,[224,153,47,255])],
           output/'wire_connection_section.png',camera=(250,39,-30),
           focus=(model.GALLERY_Z,39,-40.5),scale=62,size=(1150,1600))
    wing_report=json.loads((HERE/'wing_validation.json').read_text())
    assert wing_report['housing_stl_sha256']==sha(HERE/'STL'/model.BODY_FILE)
    manifest['wing_scenes']={}
    for variant in ['flat','graded']:
        complete=scene.copy();wing_objects=[];installed_wings=[]
        for side in ['left','right']:
            name=f'V4_{variant}_{side}_UPPER.stl';row=wing_report['parts'][name]
            target=HERE/'STL/wings'/name;assert sha(target)==row['stl_sha256']
            wing=trimesh.load_mesh(target,process=True)
            wing.apply_transform(np.linalg.inv(row['source_to_stl_matrix']))
            installed_wings.append(wing)
            wing_objects.append(render_object(wing,[145,168,179,255]))
            preview=model.retained.v3.vtk_decimate(wing,70000) if len(wing.faces)>70000 else wing.copy()
            preview.visual=trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
                name=f'V4_{variant}_{side}_wing',baseColorFactor=[145,168,179,255],roughnessFactor=.7))
            preview.apply_scale(.001)
            complete.add_geometry(preview,geom_name=f'V4_{variant}_{side}_upper_wing',node_name=f'V4_{variant}_{side}_upper_wing')
        glb=output/f'nd25fn4_{variant}_wings.glb';complete.export(glb,include_normals=True)
        assert len(trimesh.load_scene(glb).geometry)==11
        render(objects+wing_objects,output/f'with_{variant}_wings.png',
               camera=(260,-110,70),focus=(3,0,4),scale=143,size=(1350,1500))
        render(objects+wing_objects,output/f'front_{variant}_wings.png',
               camera=(240,0,4),focus=(0,0,4),scale=143,size=(1350,1500))
        if variant=='flat':
            render(objects+wing_objects,output/'UM_outline_front.png',
                   camera=(250,0,-55.419),focus=(5,0,-55.419),scale=73,size=(1450,1450))
            render(objects+wing_objects,output/'UM_outline_oblique.png',
                   camera=(240,-85,-20),focus=(5,0,-55.419),scale=73,size=(1450,1450))
        if variant=='flat':
            # Four paired shoulder sections from the actual printed meshes. Purple
            # shows cavity volumes, not an assumed purchased magnet model.
            section=box_solid([-75,321,-20],[75,414,model.MAGNET_Z])
            cavity_objects=[]
            for part in [body,*installed_wings]:
                for cavity in part.split(only_watertight=False):
                    if cavity.volume<0 and cavity.center_mass[1]>325:
                        cavity_objects.append(render_object(
                            to_trimesh(solid(model.positive_void(cavity))^section),[175,86,181,255]))
            render([render_object(to_trimesh(solid(body)^section),[66,141,189,255]),
                    *[render_object(to_trimesh(solid(w)^section),[145,168,179,255]) for w in installed_wings]]+cavity_objects,
                   output/'magnet_alignment_section.png',camera=(210,0,-55.419),
                   focus=(model.MAGNET_Z,0,-55.419),scale=53,size=(1450,1100))
        manifest['wing_scenes'][variant]={'path':glb.name,'sha256':sha(glb)}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Review:',path,flush=True)


if __name__=='__main__':
    main()
