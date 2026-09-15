"""Inspect the delivered print body against both exact LM top meshes.

Brief: read-only assembly inspection, installed millimetres (X lateral, Y up,
Z forward). Restore all print transforms; measure overlap, receiver axes,
half-lap clearance and the visible apron offset. Flush means coplanar outer
surfaces, independently of mounting compatibility. No geometry is regenerated.
"""
from pathlib import Path
import hashlib
import json
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mesh_ops import solid, to_trimesh
import manifold3d as md

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def installed(path):
    sidecar=path.with_suffix('.print.json')
    authority=json.loads(sidecar.read_text())
    assert authority['stl_sha256']==sha(path),path
    mesh=trimesh.load_mesh(path,process=True)
    mesh.apply_transform(np.linalg.inv(authority['source_to_stl_matrix']))
    return mesh,dict(path=str(path.relative_to(ROOT)),sha256=sha(path),sidecar_sha256=sha(sidecar))


def hits(mesh,origin,direction):
    points,_,_=mesh.ray.intersects_location([origin],[direction],multiple_hits=True)
    if not len(points):return points
    t=(points-np.asarray(origin))@np.asarray(direction)
    return points[np.argsort(t)]


def hole(mesh,x,y,z):
    points=[]
    for angle in np.linspace(0,2*np.pi,48,endpoint=False):
        p=hits(mesh,[x,y,z],[np.cos(angle),np.sin(angle),0])
        assert len(p)
        points.append(p[0,:2])
    points=np.asarray(points)
    # Fit in local coordinates to avoid cancellation at Y=315.77.
    q=points-[x,y]
    fit=np.linalg.lstsq(np.c_[2*q,np.ones(len(q))],np.sum(q*q,axis=1),rcond=None)[0]
    center=fit[:2]+[x,y]
    radii=np.linalg.norm(points-center,axis=1)
    return dict(center_xy_mm=center.tolist(),diameter_range_mm=(2*np.array([radii.min(),radii.max()])).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',action='store_true',help='Check the rebuilt design before preparing its print mesh')
    args=parser.parse_args()
    body_path=HERE/('STL/01_UM_Crescent_V4.stl' if args.design else 'print/geometry/01_UM_Crescent_V4_PRINT.stl')
    body,body_source=installed(body_path)
    crop=md.Manifold.cube((120,40,60)).translate((-60,300,-25))
    body_solid=solid(body)^crop
    body_crop=to_trimesh(body_solid)
    prefix='LM_design_assembly' if args.design else 'LM_assembly'
    output=HERE/'print'/f'{prefix}_check.json'
    report=dict(scope='exact print-body and LM top meshes in installed coordinates',
                physical_fit='not tested',source_script_sha256=sha(Path(__file__)),
                body=body_source,configurations={})
    section_meshes=None
    for state in ('no_floor_stand','floor_stand'):
        lm,lm_source=installed(ROOT/f'build/{state}/stl/obiwan_optional_lm_keyed_2_of_2_top.stl')
        lm_solid=solid(lm);lm_crop=to_trimesh(lm_solid^crop)
        overlap=abs((body_solid^lm_solid).volume())
        assert overlap<.002,(state,overlap)
        receivers=[]
        for x in (-32.,32.):
            y=315.770102
            a=hole(body_crop,x,y,14.);b=hole(lm_crop,x,y,10.)
            axis_error=float(np.linalg.norm(np.array(a['center_xy_mm'])-b['center_xy_mm']))
            assert axis_error<.015,(state,x,axis_error)
            um_face=float(hits(body_crop,[x+3.5,y,-25],[0,0,1])[0,2])
            lm_face=float(hits(lm_crop,[x+3.5,y,30],[0,0,-1])[0,2])
            gap=um_face-lm_face
            assert abs(gap-.2)<.001,(state,x,gap)
            receivers.append(dict(UM=a,LM=b,axis_offset_mm=axis_error,
                                  UM_rear_mating_z_mm=um_face,LM_front_mating_z_mm=lm_face,
                                  half_lap_gap_mm=gap))
        # Check physical front-face occlusion, not merely volumetric overlap.
        # A raised apron can pass the latter while hiding the LM in front view.
        lm_front=lm_solid^md.Manifold.cube((120,40,.02)).translate((-60,300,18.28))
        um_front=body_solid^md.Manifold.cube((120,40,30)).translate((-60,300,18.299))
        occluded_area=float((um_front.project()^lm_front.project()).area())
        assert occluded_area<.002,(state,'UM occludes the LM front',occluded_area)
        maximum_lower_z=float(body_crop.vertices[body_crop.vertices[:,1]<322.,2].max())
        assert abs(maximum_lower_z-18.3)<.002,(state,'lower front plane',maximum_lower_z)
        seam=[]
        for x in np.linspace(-30.,30.,13):
            lm_boundary=hits(lm_crop,[x,335.,18.299],[0,-1,0])
            assert len(lm_boundary),(state,x,'missing LM front boundary')
            lm_y=float(lm_boundary[0,1])
            um_boundary=hits(body_crop,[x,lm_y-.01,18.299],[0,1,0])
            assert len(um_boundary),(state,x,'missing UM seam')
            um_y=float(um_boundary[0,1]);gap=um_y-lm_y
            assert -.002<=gap<.3,(state,x,'visible seam gap',gap)
            um_z=float(hits(body_crop,[x,um_y+.2,35],[0,0,-1])[0,2])
            lm_z=float(hits(lm_crop,[x,lm_y-.2,35],[0,0,-1])[0,2])
            assert abs(um_z-lm_z)<.002,(state,x,'front-face step',um_z-lm_z)
            seam.append(dict(x_mm=float(x),LM_edge_y_mm=lm_y,UM_edge_y_mm=um_y,
                             assembly_gap_mm=gap,UM_front_z_mm=um_z,LM_front_z_mm=lm_z,
                             front_surface_step_mm=um_z-lm_z))
        report['configurations'][state]=dict(LM=lm_source,overlap_mm3=overlap,
            receivers=receivers,seam_samples=seam,mount_compatible=True,
            LM_front_occluded_area_mm2=occluded_area,maximum_lower_front_z_mm=maximum_lower_z,
            exterior_flush=True)
        if state=='no_floor_stand':section_meshes=(body_crop,lm_crop)
    # Confirm that the inspected LM top is the one represented by the existing
    # PETG-GF GUI project, and that the body is the mesh bound to the sliced job.
    shelf=json.loads((ROOT/'to_print/delivery_manifest.json').read_text())
    row=next(r for r in shelf['projects'] if r['path']=='obiwan/3mf_06hf_petg-gf_pla/obiwan_02_LM_top_keyed_2_of_2_GUI.3mf')
    assert sha(ROOT/'to_print'/row['path'])==row['sha256']
    assert row['source_stl_sha256']==report['configurations']['no_floor_stand']['LM']['sha256']
    assert row['geometry_audit']['status']=='pass'
    report['LM_project']=dict(path='to_print/'+row['path'],sha256=row['sha256'])
    if not args.design:
        current=json.loads((HERE/'print/manifest.json').read_text())
        row=next(r for r in current['projects'] if r['key']=='body')
        assert sha(ROOT/row['path'])==row['sha256']
        audit=json.loads((ROOT/row['audit']).read_text())
        assert audit['project_sha256']==row['sha256']
        assert next(r for r in audit['geometry'] if r['subtype']=='normal_part')['sha256']==body_source['sha256']
        report['body_project']=dict(path=row['path'],sha256=row['sha256'])
    report['result']='Mounts compatible; front faces flush at Z18.30; no LM front-face occlusion.'
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,x in zip(axes,[0.,32.]):
        for mesh,color,label in zip(section_meshes,['#267ba5','#737c83'],['UM + crescent','LM top']):
            section=mesh.section(plane_origin=[x,0,0],plane_normal=[1,0,0])
            assert section is not None
            for i,line in enumerate(section.discrete):
                ax.plot(line[:,1],line[:,2],color=color,lw=2,label=label if i==0 else None)
        ax.axhline(18.3,color='#737c83',ls=':',lw=1)
        ax.set(xlim=(307,322),ylim=(0,22.5),xlabel='Installed Y (mm)',ylabel='Frontward Z (mm)',
               title='Centre section, X = 0' if x==0 else 'Right receiver section, X = 32 mm')
        ax.set_aspect('equal');ax.grid(alpha=.15);ax.legend(loc='lower left')
    axes[0].annotate('Both faces at Z18.30 mm',xy=(313.7,18.3),xytext=(307.6,21.5),
                     fontsize=10,color='#27744b',arrowprops=dict(arrowstyle='-',color='#27744b'))
    axes[1].text(307.5,14,'Half-lap clearance: 0.20 mm',fontsize=10)
    fig.suptitle('Actual mesh assembled with LM top\nFlush front interface; LM face remains uncovered',fontsize=15)
    png=HERE/'print'/f'{prefix}_sections.png';fig.savefig(png,dpi=160);plt.close(fig)
    report['section_image']=dict(path=str(png.relative_to(ROOT)),sha256=sha(png))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(report['result'])
    for state,row in report['configurations'].items():
        print(state,'overlap',row['overlap_mm3'],'max screw-axis offset',max(r['axis_offset_mm'] for r in row['receivers']),
              'maximum front step',max(abs(r['front_surface_step_mm']) for r in row['seam_samples']),
              'occluded LM area',row['LM_front_occluded_area_mm2'])
    print(output)


if __name__=='__main__':main()
