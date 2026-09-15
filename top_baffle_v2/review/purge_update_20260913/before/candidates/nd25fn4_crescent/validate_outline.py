"""Measure the visible UM silhouette from the exported body's projection."""
import json
import numpy as np
from shapely.geometry import Polygon,LineString
from shapely.ops import unary_union
from scipy.signal import savgol_filter

import v4_model as model
from mesh_ops import solid
from rebuild import sha
from validate import restored_print_parts


def silhouette(mesh):
    # The front silhouette is the projection of all actual material, not
    # one arbitrary depth section or the nominal construction curve.
    polygons=solid(mesh).project().to_polygons()
    return unary_union([Polygon(p) for p in polygons if len(p)>2])


def sample_widths(shape,vertical):
    limits=[]
    for y in vertical:
        cut=shape.intersection(LineString([(-80.,y),(80.,y)]))
        assert not cut.is_empty,('missing silhouette',float(y))
        limits.append([cut.bounds[0],cut.bounds[2]])
    return np.asarray(limits)


def main():
    _,parts=restored_print_parts()
    mesh=model.installed(parts['housing'])
    shape=silhouette(mesh)
    vertical=np.arange(311.,427.0001,.1)
    limits=sample_widths(shape,vertical)
    width=limits[:,1]
    symmetry=np.max(abs(limits.sum(axis=1)))
    assert symmetry<.04,('asymmetric exterior',symmetry)
    # A broad central lobe, and an integrated lower skirt; these targets
    # distinguish the requested outline from the old round ring plus ears.
    mid=(vertical>350)&(vertical<380)
    maximum=float(width[mid].max())
    widest_y=float(vertical[mid][np.argmax(width[mid])])
    assert model.UM_SCULPTED_HALF_WIDTH-1.<maximum<model.UM_SCULPTED_HALF_WIDTH+.1,('central half-width',maximum)
    assert 2*maximum<125.2,('UM surround no longer slender',2*maximum)
    assert abs(widest_y-model.interface.UM_CUTOUT[1])<.6
    lower=(vertical>=318)&(vertical<=340)
    assert np.min(width[lower])>38.,'lower skirt breaks into separate lobes'
    # The outer mesh uses 0.3 mm cells. Fit across five cells rather than
    # differentiating sub-cell triangle edges. Independently limit the
    # departure from the actual silhouette to 0.06 mm, so fitting cannot
    # conceal a meaningful corner or alter the delivered geometry.
    fit_window=15
    fitted=savgol_filter(width,fit_window,3)
    slope=savgol_filter(width,fit_window,3,deriv=1,delta=.1)
    curvature=savgol_filter(width,fit_window,3,deriv=2,delta=.1)/(1+slope*slope)**1.5
    free=(vertical>318)&(vertical<424)
    fit_error=float(np.max(abs(fitted[free]-width[free])))
    assert fit_error<.06,('outline detail lost by curvature fit',fit_error)
    minimum_radius=float(1/np.max(abs(curvature[free])))
    assert minimum_radius>3.,('abrupt free-outline corner',minimum_radius)
    step=4
    left=np.arctan2(fitted[step:-step]-fitted[:-2*step],.4)
    right=np.arctan2(fitted[2*step:]-fitted[step:-step],.4)
    eligible=free[:-2*step]&free[step:-step]&free[2*step:]
    max_turn=float(np.degrees(abs(right-left))[eligible].max())
    assert max_turn<10.,('visible silhouette kink',max_turn)
    report={'status':'passed','body_sha256':sha(model.HERE/'STL'/model.BODY_FILE),
        'model_source_sha256':sha(model.HERE/'v4_model.py'),
        'source_sha256':sha(__file__),
        'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'sample_spacing_mm':.1,'sample_count':len(vertical),
        'maximum_left_right_difference_mm':float(symmetry),
        'UM_maximum_width_mm':2*maximum,'widest_y_mm':widest_y,
        'minimum_free_outline_radius_mm':minimum_radius,
        'maximum_free_outline_tangent_change_per_0_4mm_deg':max_turn,
        'curvature_fit_window_mm':fit_window*.1,
        'maximum_fit_departure_from_actual_outline_mm':fit_error,
        'vertical_y_mm':vertical.tolist(),'lateral_bounds_mm':limits.tolist(),
        'scope':'Front-projected actual material; both side outlines, including every buried magnet station. Exact bottom receiver datums excluded only from curvature checks.'}
    (model.HERE/'outline_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('UM outline:',len(vertical),'sections, width',round(2*maximum,3),
          'mm; minimum free radius',round(minimum_radius,3),'mm',flush=True)


if __name__=='__main__':
    main()
