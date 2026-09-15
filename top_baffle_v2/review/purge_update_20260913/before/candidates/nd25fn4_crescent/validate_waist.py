"""Check the actual outer UM/T waist through depth, including both former dents."""
import json
import numpy as np
from scipy.signal import savgol_filter
import v4_model as model
from mesh_ops import solid
from rebuild import sha
from validate import restored_print_parts


def section_facts(body):
    rows=[]
    vertical=np.arange(410.,452.001,.1)
    for depth in [-8.,-4.,0.,4.,8.,12.,16.,18.29]:
        polygons=body.slice(depth).to_polygons()
        a=np.vstack(polygons)
        b=np.vstack([np.roll(p,-1,axis=0) for p in polygons])
        limits=[]
        for y in vertical:
            selected=(np.minimum(a[:,1],b[:,1])<=y)&(np.maximum(a[:,1],b[:,1])>y)
            aa,bb=a[selected],b[selected]
            x=aa[:,0]+(y-aa[:,1])*(bb[:,0]-aa[:,0])/(bb[:,1]-aa[:,1])
            limits.append([x.min(),x.max()] if len(x) else [np.nan,np.nan])
        limits=np.array(limits)
        valid=np.isfinite(limits).all(axis=1)
        symmetry=float(np.max(abs(limits[valid].sum(axis=1))))
        for side in [-1,1]:
            width=side*limits[:,0 if side<0 else 1]
            # Only supply endpoint values to the polynomial fitter where this
            # depth has not yet reached the waist. Those points and their full
            # fitting neighbourhood are excluded from the measured surface.
            filled=np.interp(vertical,vertical[valid],width[valid])
            fitted=savgol_filter(filled,15,3)
            slope=savgol_filter(filled,15,3,deriv=1,delta=.1)
            curvature=savgol_filter(filled,15,3,deriv=2,delta=.1)/(1+slope*slope)**1.5
            eligible=(vertical>=414)&(vertical<=450)&(width>=35)
            eligible &= np.convolve(valid.astype(int),np.ones(15,dtype=int),'same')==15
            assert eligible.sum()>80,('insufficient waist coverage',depth,side)
            error=float(np.max(abs(fitted[eligible]-width[eligible])))
            radius=float(1/np.max(abs(curvature[eligible])))
            step=4
            turn=np.degrees(abs(np.arctan2(fitted[step:-step]-fitted[:-2*step],.4)
                             -np.arctan2(fitted[2*step:]-fitted[step:-step],.4)))
            turn_mask=eligible[:-2*step]&eligible[step:-step]&eligible[2*step:]
            rows.append(dict(depth_z_mm=depth,side=side,samples=int(eligible.sum()),
                checked_y_mm=[float(vertical[eligible].min()),float(vertical[eligible].max())],
                maximum_symmetry_error_mm=symmetry,minimum_radius_mm=radius,
                maximum_fit_departure_mm=error,maximum_tangent_change_per_0_4mm_deg=float(turn[turn_mask].max()),
                y_mm=vertical[eligible].tolist(),half_width_mm=width[eligible].tolist()))
    return rows


def main():
    _,parts=restored_print_parts()
    rows=section_facts(solid(model.installed(parts['housing'])))
    for row in rows:
        assert row['maximum_symmetry_error_mm']<.04,row
        assert row['maximum_fit_departure_mm']<.05,row
        assert row['minimum_radius_mm']>3.,row
        assert row['maximum_tangent_change_per_0_4mm_deg']<10.,row
        if row['depth_z_mm']<=16.:
            assert row['minimum_radius_mm']>5.,row
    # The former out-of-domain spline produced approximately -625 mm here.
    lateral=np.arange(-66.,66.001,.25)
    vertical=np.arange(378.,464.001,.25)
    heights=model.rear_loft_table()(lateral,vertical)
    low,high=float(heights.min()),float(heights.max())
    assert low>=model.INSTALLED_Z_OFFSET-model.retained.P.overall_depth/2-.001
    assert high<=18.3
    report=dict(status='passed',body_sha256=sha(model.HERE/'STL'/model.BODY_FILE),
        source_sha256=sha(__file__),model_source_sha256=sha(model.HERE/'v4_model.py'),
        source_build_manifest_sha256=sha(model.HERE/'build_manifest.json'),
        section_spacing_mm=.1,curvature_fit_window_mm=1.5,
        rear_height_range_mm=[low,high],profiles=rows,
        minimum_outer_radius_mm=min(r['minimum_radius_mm'] for r in rows),
        maximum_tangent_change_per_0_4mm_deg=max(r['maximum_tangent_change_per_0_4mm_deg'] for r in rows),
        scope='Actual STL sections on both outer sides, X magnitude >=35 mm, Y414..450, eight depths from rear to front. This includes both former dents; central service-bore edges are outside this exterior-surface test.')
    (model.HERE/'waist_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Waist passed:',len(rows),'section profiles; minimum radius',report['minimum_outer_radius_mm'],flush=True)


if __name__=='__main__':
    main()
