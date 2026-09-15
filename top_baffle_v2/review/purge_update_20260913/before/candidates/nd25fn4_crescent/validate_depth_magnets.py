"""Actual STL depth sections, uninterrupted exterior and buried-magnet cover."""
import json
import numpy as np
import trimesh
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

import v4_model as model
from rebuild import sha
from validate import restored_print_parts


def nearest_surface(mesh,points):
    vertices=vtk.vtkPoints();vertices.SetData(numpy_to_vtk(np.asarray(mesh.vertices),deep=True))
    cells=vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces),numpy_to_vtkIdTypeArray(
        np.c_[np.full(len(mesh.faces),3),mesh.faces].ravel().astype(np.int64),deep=True))
    data=vtk.vtkPolyData();data.SetPoints(vertices);data.SetPolys(cells)
    distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(data)
    return np.array([abs(distance.EvaluateFunction(p)) for p in points])


def full_cover(mesh):
    shells=mesh.split(only_watertight=False)
    outer=max(shells,key=lambda p:p.volume)
    cavities=[p for p in shells if p.volume<0 and p.center_mass[1]>330.]
    results=[]
    for cavity in cavities:
        points=np.r_[cavity.vertices,cavity.triangles_center]
        # Retain every cavity vertex and face centre. Include all nearby
        # external, cable and bore faces; never include the cavity itself.
        lo=cavity.bounds[0]-7.;hi=cavity.bounds[1]+7.
        tri=outer.triangles
        keep=np.all(tri.max(axis=1)>lo,axis=1)&np.all(tri.min(axis=1)<hi,axis=1)
        local=outer.submesh([np.flatnonzero(keep)],append=True,repair=False)
        distances=nearest_surface(local,points)
        minimum=float(distances.min())
        assert minimum>.75,('insufficient buried cover',cavity.center_mass,minimum)
        results.append({'cavity_center_mm':cavity.center_mass.tolist(),
            'sample_count':len(points),'minimum_material_to_open_surface_mm':minimum,
            'minimum_witness_mm':points[np.argmin(distances)].tolist()})
    return results


def depth_profile(mesh,angle):
    direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.])
    radii=np.arange(49.45,66.001,.10)
    points=np.array([0.,model.interface.UM_CUTOUT[1],40.])+radii[:,None]*direction
    hits,index,_=mesh.ray.intersects_location(points,np.tile([0.,0.,-1.],(len(points),1)),multiple_hits=True)
    front=np.full(len(points),-np.inf);rear=np.full(len(points),np.inf)
    np.maximum.at(front,index,hits[:,2]);np.minimum.at(rear,index,hits[:,2])
    valid=np.isfinite(front)
    return {'angle_deg':angle,'radius_mm':radii[valid].tolist(),
            'front_z_mm':front[valid].tolist(),'rear_z_mm':rear[valid].tolist(),
            'maximum_front_z_mm':float(front[valid].max())}


def surface_at_magnets(mesh):
    """Ray-check entire former rectangular-pad areas against free surface."""
    results=[]
    for site in model.magnet_sites():
        n=site['normal'];t=site['tangent'];up=site['up']
        u,v=np.meshgrid(np.arange(-6.,6.01,.4),np.arange(-6.4,3.61,.4),indexing='ij')
        centre=site['contact']+u.reshape(-1,1)*t+v.reshape(-1,1)*up
        # Find the outermost inside point along each inclined ray; very
        # deep points can pass out through the opposite front/rear face.
        trial=np.arange(6.,-16.01,-.25)
        q=centre[:,None,:]+trial[None,:,None]*n
        inside=model.envelope_field(q[:,:,2]-model.INSTALLED_Z_OFFSET,q[:,:,0],q[:,:,1]-model.INSTALLED_Y_OFFSET)<0
        valid=inside.any(axis=1);centre=centre[valid];inside=inside[valid]
        assert len(centre)>600,('insufficient exterior witnesses',site['angle_deg'])
        first=inside.argmax(axis=1)
        low=trial[first];high=low+.25
        for _ in range(30):
            middle=(low+high)/2;q=centre+middle[:,None]*n
            inside=model.envelope_field(q[:,2]-model.INSTALLED_Z_OFFSET,q[:,0],q[:,1]-model.INSTALLED_Y_OFFSET)<0
            low=np.where(inside,middle,low);high=np.where(inside,high,middle)
        expected=(low+high)/2
        origins=centre+8*n
        # A finite local triangle patch contains every exterior witness
        # in this normal band. It avoids testing distant tweeter facets
        # against thousands of oblique rays on the UM shoulder.
        cloud=np.r_[centre-16*n,origins]
        lo=cloud.min(axis=0)-1.;hi=cloud.max(axis=0)+1.
        triangles=mesh.triangles
        keep=np.all(triangles.max(axis=1)>lo,axis=1)&np.all(triangles.min(axis=1)<hi,axis=1)
        local=mesh.submesh([np.flatnonzero(keep)],append=True,repair=False)
        actual=np.full(len(origins),-np.inf)
        # Bound trimesh's candidate-triangle matrix. Oblique batches over
        # the full detailed body can otherwise allocate many gigabytes.
        for first in range(0,len(origins),24):
            batch=origins[first:first+24]
            points,indices,_=local.ray.intersects_location(batch,np.tile(-n,(len(batch),1)),multiple_hits=True)
            indices+=first
            np.maximum.at(actual,indices,np.einsum('ij,j->i',points-centre[indices],n))
        assert np.all(np.isfinite(actual)),('missing magnetic exterior',site['angle_deg'])
        error=float(np.max(abs(actual-expected)))
        assert error<.045,('exterior differs from the uninterrupted surface',site['angle_deg'],error)
        results.append({'angle_deg':site['angle_deg'],'sample_count':len(origins),
            'maximum_free_surface_error_mm':error,
            'axis_inclination_deg':float(np.degrees(np.arctan2(-n[2],np.linalg.norm(n[:2])))),
            'pocket_face_separation_mm':model.magnet_burial(site['angle_deg'],'wing')-model.magnet_burial(site['angle_deg'],'body')})
    return results


def shoulder_depths(mesh, *, validate_shape=True):
    """Measure a swept side, with no upright wall in the visible depth band."""
    results=[]
    for name,y in [('lower_neck',333.),('middle',model.interface.UM_CUTOUT[1]),('upper_shoulder',400.)]:
        z=np.arange(3.,22.01,.2)
        origins=np.c_[np.full(len(z),80.),np.full(len(z),y),z]
        points,indices,_=mesh.ray.intersects_location(origins,np.tile([-1.,0.,0.],(len(z),1)),multiple_hits=True)
        width=np.full(len(z),-np.inf);np.maximum.at(width,indices,points[:,0])
        valid=np.isfinite(width)
        assert np.all(np.isfinite(np.interp([8.,18.],z,width)))
        rear,front=np.interp([8.,18.],z,width)
        widest_depth=float(z[np.argmax(width)])
        if validate_shape:
            assert front-rear>4.,(name,'outer wall must lean across depth',rear,front)
            assert widest_depth>18.,(name,'front outline must be widest',widest_depth)
        # Rounded front/rear endpoints are deliberate. The exposed side
        # between those endpoints must incline consistently toward front.
        derivative=np.full(len(z),np.nan)
        derivative[valid]=np.gradient(width[valid],z[valid])
        face=(z>=10.01)&(z<=17.99)
        lean=np.degrees(np.arctan(derivative[face]))
        if validate_shape:
            assert np.min(lean)>20.,(name,'upright outer wall',float(np.min(lean)))
        average_lean=float(np.degrees(np.arctan((front-rear)/10.)))
        results.append({'section':name,'vertical_y_mm':y,'depth_z_mm':z[valid].tolist(),
            'lateral_half_width_mm':width[valid].tolist(),'widest_depth_z_mm':widest_depth,
            'rear_half_width_at_z8_mm':float(rear),'front_half_width_at_z18_mm':float(front),
            'front_minus_rear_full_width_mm':float(2*(front-rear)),
            'average_edge_lean_from_vertical_deg':average_lean,
            'minimum_edge_lean_z10_to_18_deg':float(np.min(lean)),
            'front_visible_band_at_z18_mm':float(front-np.sqrt(model.interface.UM_RECESS_R**2-(y-model.interface.UM_CUTOUT[1])**2))})
    return results


def constant_draft(mesh):
    """Measure draft normal to the local outline at 68 heights of the STL."""
    heights=np.arange(333.,400.01,1.)
    depths=np.arange(8.,16.01,1.)
    yy,zz=np.meshgrid(heights,depths,indexing='ij')
    origins=np.c_[np.full(yy.size,80.),yy.ravel(),zz.ravel()]
    points,indices,_=mesh.ray.intersects_location(origins,np.tile([-1.,0.,0.],(len(origins),1)),multiple_hits=True)
    width=np.full(len(origins),-np.inf);np.maximum.at(width,indices,points[:,0])
    assert np.all(np.isfinite(width)), 'missing tapered exterior'
    width=width.reshape(yy.shape)
    dz=np.gradient(width,depths,axis=1)
    dy=np.gradient(width,heights,axis=0)
    angles=np.degrees(np.arctan(dz/np.sqrt(1+dy*dy)))
    # Exclude only the finite-difference endpoint rows/columns.
    angles=angles[1:-1,1:-1]
    assert angles.min()>23. and angles.max()<30.,('inconsistent local draft',angles.min(),angles.max())
    assert np.ptp(angles)<5.,('draft varies too much through UM',np.ptp(angles))
    return {'sample_count':angles.size,'vertical_y_range_mm':[333.,400.],
        'depth_z_range_mm':[8.,16.], 'target_local_draft_deg':float(np.degrees(np.arctan(model.UM_DRAFT_SLOPE))),
        'minimum_actual_draft_deg':float(angles.min()),'maximum_actual_draft_deg':float(angles.max()),
        'average_actual_draft_deg':float(angles.mean()),
        'front_wider_than_rear':True,
        'scope':'Local plan-normal draft from actual triangle intersections; excludes rounded lip and exact LM receiver lands.'}


def main():
    _,parts=restored_print_parts();body=model.installed(parts['housing'])
    profiles=[depth_profile(body,a) for a in [0.,45.,-45.]]
    assert 19.<profiles[0]['maximum_front_z_mm']<=18.3+model.UM_RIM_RISE+.025,'The new lip must be a small forward flare above Z18.3'
    shoulders=shoulder_depths(body)
    cover={'body':full_cover(body)}
    assert len(cover['body'])==4
    surface=surface_at_magnets(body)
    wings=json.loads((model.HERE/'wing_validation.json').read_text())
    for name,row in wings['parts'].items():
        mesh=trimesh.load_mesh(model.HERE/'STL/wings'/name,process=True)
        mesh.apply_transform(np.linalg.inv(row['source_to_stl_matrix']))
        cover[name]=full_cover(mesh)
        assert len(cover[name])==2,(name,'UM pocket count')
    report={'status':'passed','source_sha256':sha(__file__),
        'body_sha256':sha(model.HERE/'STL'/model.BODY_FILE),
        'model_source_sha256':sha(model.HERE/'v4_model.py'),
        'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'source_wing_validation_sha256':sha(model.HERE/'wing_validation.json'),
        'magnet_size_mm':[model.MAGNET_DIAMETER,model.MAGNET_DEPTH],
        'pocket_size_mm':[model.MAGNET_CAVITY_DIAMETER,model.MAGNET_CAVITY_DEPTH],
        'minimum_required_cover_mm':.75,'full_cavity_cover':cover,
        'uninterrupted_exterior':surface,'radial_depth_profiles':profiles,
        'shoulder_depth_profiles':shoulders,'constant_draft':constant_draft(body),
        'scope':'Actual STL surfaces, including complete closed pocket chimneys and roofs. No magnetic flat/pad exemption. Physical pull and printing remain unqualified.'}
    (model.HERE/'depth_magnet_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Depth and buried magnets passed; minimum cover',min(r['minimum_material_to_open_surface_mm'] for rows in cover.values() for r in rows),flush=True)


if __name__=='__main__':main()
