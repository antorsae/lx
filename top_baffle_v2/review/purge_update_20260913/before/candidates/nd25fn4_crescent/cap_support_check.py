"""Measure cap-ceiling support from the exported mesh and actual G-code."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import Point, LineString
from shapely.ops import unary_union

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
WORK=ROOT/'review/nd25fn4_print'
sys.path.insert(0,str(ROOT/'scripts'))
from gcode_analysis import parse_gcode
from prepare_print import sha


def local_segments(layer,center,prefix=None,radius=27.):
    x,y=center
    return [s for s in layer.segments
            if (prefix is None or s.feature.startswith(prefix))
            and abs((s.x0+s.x1)/2-x)<radius and abs((s.y0+s.y1)/2-y)<radius]


def interface_mask(layer,center):
    return unary_union([LineString([(s.x0,s.y0),(s.x1,s.y1)]).buffer(
        (s.line_width or .62)/2,quad_segs=6)
        for s in local_segments(layer,center,'Support interface')])


def ceiling_supports(gcode,prep):
    parsed=parse_gcode(gcode)
    cap_sources=[s for s in prep['sources'] if Path(s['path']).name=='02_Closed_Cap_PRINT_TWO.stl']
    assert len(cap_sources)==2
    cap=trimesh.load_mesh(ROOT/cap_sources[0]['path'],process=True)
    points,_,_=cap.ray.intersects_location([[0,0,-1]],[[0,0,1]],multiple_hits=True)
    ceiling=float(points[:,2].min())
    assert 8.9<ceiling<9.2,('unexpected cap ceiling',ceiling)
    down=(cap.face_normals[:,2]<-.999)&(abs(cap.triangles_center[:,2]-ceiling)<.001)
    radius=float(np.linalg.norm(cap.vertices[np.unique(cap.faces[down])][:,:2],axis=1).max())
    assert 22.2<radius<22.4
    rows=[]
    for source in cap_sources:
        center=np.asarray(source['translation'][:2],float)
        roof=[]
        for layer in parsed.layers:
            for s in local_segments(layer,center):
                if s.feature.startswith(('Support','Prime','Brim','Skirt','Custom','Undefined')):
                    continue
                if LineString([(s.x0,s.y0),(s.x1,s.y1)]).distance(Point(*center))<1.:
                    roof.append(layer);break
        assert roof,('missing cap ceiling toolpaths',center.tolist())
        first=roof[0]
        assert abs(first.z-(first.layer_height or .16)/2-ceiling)<.1
        below=[l for l in parsed.layers if l.z<first.z-.01
               and local_segments(l,center,'Support interface')]
        assert len(below)>=3,('missing PLA beneath cap ceiling',center.tolist())
        layers=below[-3:]
        bottom=first.z-(first.layer_height or .16)
        assert abs(bottom-layers[-1].z)<.02,('unsupported gap beneath cap ceiling',bottom,layers[-1].z)
        disk=Point(*center).buffer(radius,quad_segs=128)
        # The preset deliberately leaves 0.7 mm object XY clearance. Check
        # the full roof and its inner area separately instead of hiding that
        # perimeter allowance in a single coverage percentage.
        core=Point(*center).buffer(radius-1.,quad_segs=128)
        coverage=[]
        for layer in layers:
            mask=interface_mask(layer,center)
            full=float(mask.intersection(disk).area/disk.area)
            inner=float(mask.intersection(core).area/core.area)
            assert full>.90 and inner>.98,('incomplete cap ceiling support',center.tolist(),layer.z,full,inner)
            coverage.append(dict(z_mm=layer.z,full_ceiling_coverage=full,inner_ceiling_coverage=inner,
                                 interface_segments=len(local_segments(layer,center,'Support interface'))))
        assert all(abs(b.z-a.z-.16)<.002 for a,b in zip(layers,layers[1:]))
        rows.append(dict(center_bed_xy_mm=center.tolist(),ceiling_cad_z_mm=ceiling,
            ceiling_diameter_mm=2*radius,first_ceiling_layer_z_mm=first.z,
            interface_to_ceiling_layer_gap_mm=bottom-layers[-1].z,
            inner_test_perimeter_allowance_mm=1.,interface_layers=coverage))
    retainers=[]
    for source in prep['sources']:
        if Path(source['path']).name!='03_Tweeter_Retainer_PRINT_TWO.stl':continue
        center=source['translation'][:2]
        count=sum(len(local_segments(l,center,'Support')) for l in parsed.layers)
        assert count==0,('unnecessary support on retainer or in clearance holes',center,count)
        retainers.append(dict(center_bed_xy_mm=center,support_segments=count))
    assert len(retainers)==2
    return dict(status='pass',gcode_sha256=sha(gcode),source_sha256=sha(__file__),
        cap_stl_sha256=sha(ROOT/cap_sources[0]['path']),caps=rows,retainers=retainers,
        method='Actual extrusion-width footprints in three consecutive PLA interface layers immediately below each printed ceiling; material identity is independently checked by audit_print.py.')


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    prep=json.loads((WORK/'preparation.json').read_text())['accessories']
    gcode=WORK/'accessories/plate_1.gcode'
    report=ceiling_supports(gcode,prep)
    published=HERE/'print/02_Caps_TWO_Retainers_TWO_06HF_PETG_GF_PLA.gcode.3mf'
    audit=json.loads((WORK/'accessories/audit.json').read_text())
    assert audit['status']=='pass' and audit['project_sha256']==sha(published)
    assert audit['cap_ceilings']==report
    parsed=parse_gcode(gcode)
    fig,axes=plt.subplots(2,2,figsize=(10,10),layout='constrained')
    for column,cap in enumerate(report['caps']):
        center=np.asarray(cap['center_bed_xy_mm'])
        support_z=cap['interface_layers'][-1]['z_mm']
        support=min(parsed.layers,key=lambda l:abs(l.z-support_z))
        roof=min(parsed.layers,key=lambda l:abs(l.z-cap['first_ceiling_layer_z_mm']))
        for row,layer in enumerate([support,roof]):
            ax=axes[row,column]
            below=[np.array([[s.x0,s.y0],[s.x1,s.y1]])-center
                   for s in local_segments(support,center,'Support interface')]
            ax.add_collection(LineCollection(below,colors='#249466',linewidths=1.,alpha=1. if row==0 else .35))
            if row:
                lines=[np.array([[s.x0,s.y0],[s.x1,s.y1]])-center for s in local_segments(layer,center)
                       if not s.feature.startswith(('Support','Prime','Brim','Custom'))]
                ax.add_collection(LineCollection(lines,colors='#355e83',linewidths=.65))
            ax.add_patch(Circle((0,0),cap['ceiling_diameter_mm']/2,fill=False,color='#172f40',lw=1.,ls='--'))
            label='PLA interface' if row==0 else 'First PETG-GF ceiling layer'
            ax.set(xlim=(-29,29),ylim=(-29,29),aspect='equal',xlabel='Local X (mm)',ylabel='Local Y (mm)',
                title=f'Cap {column+1} — {label}\nZ = {layer.z:.2f} mm')
            ax.grid(alpha=.12)
    fig.suptitle('Cap ceilings — actual sliced support paths\nGreen: removable PLA interface · Blue: PETG-GF ceiling\nThree interface layers per cap; zero gap to the first ceiling layer',fontsize=13)
    output=HERE/'print/cap_ceiling_support.png'
    fig.savefig(output,dpi=160);plt.close(fig)
    report.update(project_sha256=sha(published),image=output.name,image_sha256=sha(output))
    (HERE/'print/cap_support_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(output,flush=True)


if __name__=='__main__':main()
