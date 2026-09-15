#!/usr/bin/env python3
"""Check the continuous H2C LM against the retained UM and V4 interfaces."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'candidates/nd25fn4_crescent')]
import numpy as np
import manifold3d as md
from mesh_ops import solid,to_trimesh
from verify_lm_assembly import installed,hole,hits
from lx521_baffle.h2c.printing import write_json
from lx521_baffle.io import sha256_file


def main():
    crop=md.Manifold.cube((120,40,60)).translate((-60,300,-25))
    body,body_source=installed(ROOT/'candidates/nd25fn4_crescent/print/geometry/01_UM_Crescent_V4_PRINT.stl')
    v4=solid(body)^crop;body_crop=to_trimesh(v4)
    regular,regular_source=installed(ROOT/'build/no_floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl')
    regular=solid(regular)^crop
    rows=[]
    for state in ('no_floor_stand','floor_stand'):
        lm,lm_source=installed(ROOT/f'build/h2c/STL/h2c_obiwan_core_lm_carrier_{state}.stl')
        lm=solid(lm)^crop;lm_crop=to_trimesh(lm)
        overlap=abs((v4^lm).volume());regular_overlap=abs((regular^lm).volume())
        assert max(overlap,regular_overlap)<.002,(state,overlap,regular_overlap)
        receivers=[]
        for x in (-32.,32.):
            y=315.770102
            a=hole(body_crop,x,y,14.);b=hole(lm_crop,x,y,10.)
            axis=float(np.linalg.norm(np.array(a['center_xy_mm'])-b['center_xy_mm']))
            assert axis<.015,(state,axis)
            gap=float(hits(body_crop,[x+3.5,y,-25],[0,0,1])[0,2]-hits(lm_crop,[x+3.5,y,30],[0,0,-1])[0,2])
            assert abs(gap-.2)<.001,(state,gap)
            receivers.append(dict(x_mm=x,axis_error_mm=axis,half_lap_gap_mm=gap))
        lm_front=lm^md.Manifold.cube((120,40,.02)).translate((-60,300,18.28))
        um_front=v4^md.Manifold.cube((120,40,30)).translate((-60,300,18.299))
        occlusion=float((um_front.project()^lm_front.project()).area())
        assert occlusion<.002,(state,occlusion)
        samples=[]
        for x in np.linspace(-30,30,13):
            ly=float(hits(lm_crop,[x,335,18.299],[0,-1,0])[0,1])
            uy=float(hits(body_crop,[x,ly-.01,18.299],[0,1,0])[0,1])
            gap=uy-ly;assert -.002<=gap<.3,(state,x,gap)
            step=float(hits(body_crop,[x,uy+.2,35],[0,0,-1])[0,2]-hits(lm_crop,[x,ly-.2,35],[0,0,-1])[0,2])
            assert abs(step)<.002,(state,x,step)
            samples.append(dict(x_mm=float(x),seam_gap_mm=gap,front_face_step_mm=step))
        rows.append(dict(state=state,LM=lm_source,V4_overlap_mm3=overlap,regular_UM_overlap_mm3=regular_overlap,
            receivers=receivers,V4_LM_front_occlusion_mm2=occlusion,seam_samples=samples))
    write_json(ROOT/'build/h2c/obiwan_interface_validation.json',dict(status='pass',V4=body_source,regular_UM=regular_source,
        source_sha256=sha256_file(Path(__file__)),checks=rows,physical_fit='hardware trial pending'))
    print('H2C full LM: regular UM and V4 clear; V4 faces flush and LM uncovered in both stand states.',flush=True)


if __name__=='__main__':main()
