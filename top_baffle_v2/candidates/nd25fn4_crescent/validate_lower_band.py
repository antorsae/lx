"""Measure the exported rear surface across the former rectangular seams."""
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree

import v4_model as model
from rebuild import sha
from mesh_ops import solid, to_trimesh
from validate import restored_print_parts, box_solid


def main():
    _,parts=restored_print_parts()
    body=model.installed(parts['housing'])
    # A closed geometric crop makes ray tests cheap without changing any
    # surface in the measured area. All rays enter from behind the part.
    mesh=to_trimesh(solid(body)^box_solid([-45,310,-20],[45,339,20]))
    lead=model.route.route_cable_points(.08)
    lead_tree=cKDTree(lead[lead[:,1]>309])
    profiles={}
    for y in (317.,320.,320.6,320.9,323.,326.,327.):
        x=np.arange(-35.,35.0001,.10)
        profiles[f'across_Y_{y:g}']=np.c_[x,np.full(len(x),y)]
    for x in (-32.,-29.,-27.1,-27.,-24.,24.,27.,27.1,29.,32.):
        y=np.arange(316.,334.0001,.10)
        profiles[f'along_X_{x:g}']=np.c_[np.full(len(y),x),y]
    results={}
    total=0
    for name,xy in profiles.items():
        # The sloping side now rounds into the rear near its perimeter.
        # Include that designed turnover in the target, while still testing
        # actual height and tangent continuity across all former partitions.
        datum=model.um_rear(xy[:,1],xy[:,0])
        low=datum-.1;high=datum+8.
        for _ in range(28):
            middle=(low+high)/2
            inside=model.envelope_field(middle-model.INSTALLED_Z_OFFSET,xy[:,0],xy[:,1]-model.INSTALLED_Y_OFFSET)<0
            low=np.where(inside,low,middle);high=np.where(inside,middle,high)
        expected=(low+high)/2
        q=np.c_[xy,expected]
        # Functional openings are tested by independent cable/driver/fastener
        # gauges. They must not be confused with a thickness discontinuity.
        eligible=np.hypot(xy[:,0],xy[:,1]-model.interface.UM_CUTOUT[1])>41.6
        for x in model.interface.JOINT_EAR_X:
            eligible &= np.linalg.norm(xy-[x,model.interface.JOINT_EAR_Y],axis=1)>5.25
        eligible &= lead_tree.query(q)[0]>4.55
        origin=np.c_[xy,np.full(len(xy),-20.)]
        points,rays,_=mesh.ray.intersects_location(origin,
            np.tile([0.,0.,1.],(len(xy),1)),multiple_hits=True)
        z=np.full(len(xy),np.inf)
        np.minimum.at(z,rays,points[:,2])
        assert np.isfinite(z[eligible]).all(),(name,'missing rear surface',xy[eligible&~np.isfinite(z)].tolist())
        error=np.abs(z-expected)
        assert error[eligible].max()<.035,(name,'step or exposed partition',
            np.c_[xy[eligible&(error>=.035)],z[eligible&(error>=.035)],
                  expected[eligible&(error>=.035)]].tolist())
        # Two 0.4-mm secants expose an abrupt normal turn at an old mask
        # boundary even if a single height witness happens to land on it.
        k=4
        good=eligible[:-2*k]&eligible[k:-k]&eligible[2*k:]
        surface=np.where(np.isfinite(z),z,0.)
        left=np.arctan2(surface[k:-k]-surface[:-2*k],.4)
        right=np.arctan2(surface[2*k:]-surface[k:-k],.4)
        turns=np.degrees(abs(right-left))[good]
        assert not len(turns) or turns.max()<20.,(name,'abrupt rear tangent',float(turns.max()))
        total+=int(eligible.sum())
        results[name]={'surface_witnesses':int(eligible.sum()),
            'maximum_height_error_mm':float(error[eligible].max()),
            'maximum_tangent_change_per_0_4mm_deg':float(turns.max()) if len(turns) else None}
    report={'status':'passed','body_sha256':sha(model.HERE/'STL'/model.BODY_FILE),
        'model_sha256':sha(model.HERE/'v4_model.py'),'source_sha256':sha(__file__),
        'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'surface_witnesses':total,'profiles':results,
        'limits':{'height_error_mm':.035,'tangent_change_over_0_4mm_deg':20.},
        'target':'Continuous envelope including the rounded rear perimeter of the front-wide taper.',
        'intent':'Continuous curved rear surface across former X=+/-27.1 and Y=327 web partitions; exact LM receiver lands and declared service openings excluded.'}
    (model.HERE/'lower_band_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Rear lower band:',total,'surface witnesses across',len(results),'profiles passed',flush=True)


if __name__=='__main__':
    main()
