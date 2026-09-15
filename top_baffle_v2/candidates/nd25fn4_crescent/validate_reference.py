"""Driver-registered comparison of the exported STL with the supplied concept."""
import json
import numpy as np
import trimesh
from PIL import Image,ImageDraw,ImageFont
from shapely.geometry import LineString
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v4_model as model
from reference_outline import trace,REFERENCE
from validate_outline import silhouette,sample_widths
from validate import restored_print_parts
from rebuild import sha


def main():
    data,curve,mask=trace()
    _,parts=restored_print_parts();body=model.installed(parts['housing'])
    shape=silhouette(body)
    previous=json.loads((model.HERE/'superseded.json').read_text())['previous_front_wide_taper']
    archived=model.ROOT/previous/'STL'/model.BODY_FILE
    authority=json.loads(archived.with_suffix('.print.json').read_text())
    assert sha(archived)==authority['stl_sha256']
    old=trimesh.load_mesh(archived,process=True);old.apply_transform(np.linalg.inv(authority['source_to_stl_matrix']))
    old_shape=silhouette(old)
    # The latest flush/non-occluding LM interface supersedes the raster's
    # overlapping base below Y322. Continue checking the approved free UM.
    y=np.arange(322.,427.001,.25)
    target=curve(y);limits=sample_widths(shape,y);before=sample_widths(old_shape,y)
    error=(limits[:,1]-limits[:,0])/2-target
    old_error=(before[:,1]-before[:,0])/2-target
    rms=float(np.sqrt(np.mean(error**2)));old_rms=float(np.sqrt(np.mean(old_error**2)))
    maximum=float(np.max(abs(error)))
    # This is an undimensioned raster concept. Allow up to 2 mm at its
    # cropped upper continuation into the retained T flare; the full UM
    # outline must still meet the much tighter RMS target below.
    assert rms<.8 and maximum<2.,('reference silhouette mismatch',rms,maximum)
    assert rms<old_rms*.3,('insufficient improvement over rejected shape',rms,old_rms)
    base=[];cx,cy=data['driver_pixel_center'];scale=data['pixels_per_mm']
    for x in np.arange(-36.,36.001,2.):
        px=round(cx+x*scale)
        lowest=np.flatnonzero(mask[:,px])[-1]
        target_bottom=model.interface.UM_CUTOUT[1]-(lowest-cy)/scale
        cut=shape.intersection(LineString([(x,300.),(x,320.)]))
        actual=float(cut.bounds[1]);base.append([float(x),float(target_bottom),actual,actual-target_bottom])
    base=np.array(base)
    # Keep the base differences visible in the report; the LM assembly
    # check, not the concept image, now constrains this mating boundary.
    # Overlay the real external silhouette only. No smoothing or warping
    # of the STL; registration uses the driver flange, not the part edges.
    picture=Image.open(REFERENCE).convert('RGB');draw=ImageDraw.Draw(picture)
    boundary=max(getattr(shape,'geoms',[shape]),key=lambda p:p.area).exterior
    points=[(cx+x*scale,cy-(yy-model.interface.UM_CUTOUT[1])*scale) for x,yy in boundary.coords]
    draw.line(points,fill='#ff6171',width=2)
    path=model.HERE/'views/UM_reference_overlay.png';picture.save(path)
    fig,axes=plt.subplots(1,2,figsize=(12,8),layout='constrained')
    for ax,values,title in [(axes[0],before,'Previous STL'),(axes[1],limits,'New STL')]:
        for side in (-1,1):
            ax.plot(side*target,y,color='#30383d',lw=3,label='Reference' if side==1 else None)
            ax.plot(values[:,0 if side<0 else 1],y,color='#ee7358' if title.startswith('Previous') else '#258bbd',lw=1.5,label=title if side==1 else None)
        ax.set(title=title,xlabel='Lateral X (mm)',ylabel='Installed height Y (mm)',xlim=(-70,70),ylim=(305,432),aspect='equal')
        ax.grid(alpha=.15);ax.legend(loc='lower center',fontsize=9)
    fig.suptitle(f'Outline registered by the 98.6 mm driver flange — RMS error {old_rms:.2f} → {rms:.2f} mm',fontsize=14)
    chart=model.HERE/'views/UM_reference_outline_comparison.png';fig.savefig(chart,dpi=150);plt.close(fig)
    report={'status':'passed','body_sha256':sha(model.HERE/'STL'/model.BODY_FILE),
        'model_source_sha256':sha(model.HERE/'v4_model.py'),'source_sha256':sha(__file__),
        'trace_source_sha256':sha(model.HERE/'reference_outline.py'),
        'source_build_manifest_sha256':sha(model.HERE/'build_manifest.json'),
        'reference_path':str(REFERENCE.relative_to(model.ROOT)),'reference_sha256':sha(REFERENCE),
        'previous_stl_path':str(archived.relative_to(model.ROOT)),'previous_stl_sha256':sha(archived),
        'registration':data,'sample_count':len(y),'sample_height_range_mm':[322,427],
        'lower_base_constraint':'Flush Z18.3 and no LM-front occlusion supersede the raster base below Y322; checked by verify_lm_assembly.py',
        'rms_outline_error_mm':rms,'maximum_outline_error_mm':maximum,
        'previous_rms_outline_error_mm':old_rms,'error_reduction_percent':100*(1-rms/old_rms),
        'base_x_target_y_actual_y_error_mm':base.tolist(),
        'maximum_base_error_mm':float(np.max(abs(base[:,3]))),
        'outputs':{p.name:sha(p) for p in [path,chart]},
        'scope':'Actual exported STL projection versus symmetric image outline. Pixel scale is a design reference, not a measured dimension of a physical prototype.'}
    (model.HERE/'reference_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Reference comparison passed:',rms,'mm RMS;',maximum,'mm max;',100*(1-rms/old_rms),'percent improvement',flush=True)

if __name__=='__main__':main()
