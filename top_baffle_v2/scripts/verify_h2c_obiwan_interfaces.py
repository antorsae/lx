#!/usr/bin/env python3
"""Check the H2C LM interfaces: regular Obiwan UM, and both Dayton ND25FN-4 vases.

* Regular Obiwan UM on the continuous LM carrier, both stand states: no overlap, receivers coaxial,
  front faces flush and the LM front uncovered.
* Obiwan ND25FN-4 vase top on the released floor-stand keyed LM bottom: no overlap. Up to y 278 the vase top
  contains all of the released keyed top (nothing removed: key, seam, ducts, seats, inserts carry over); below
  y 180 it is identical; between them it only adds the rear taper from the vase down the ring, behind the rear.
* Stock ND25FN-4 vase on both H2C Stock LM states: no overlap (the two D3 pins sit in their D3.3 sockets), and
  its cable vestibule contains the LM's D6 duct mouth at the seam.
"""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'candidates/nd25fn4_crescent')]
import numpy as np
import manifold3d as md
from shapely.geometry import Polygon, box
from mesh_ops import solid,to_trimesh
from verify_lm_assembly import installed,hole,hits
from lx521_baffle.h2c.printing import write_json
from lx521_baffle.io import sha256_file

VASE=ROOT/'candidates/nd25fn4_vase'


def regular_um():
    crop=md.Manifold.cube((120,40,60)).translate((-60,300,-25))
    um,um_source=installed(ROOT/'build/no_floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl')
    um=solid(um)^crop;um_crop=to_trimesh(um)
    rows=[]
    for state in ('no_floor_stand','floor_stand'):
        lm,lm_source=installed(ROOT/f'build/h2c/STL/h2c_obiwan_core_lm_carrier_{state}.stl')
        lm=solid(lm)^crop;lm_crop=to_trimesh(lm)
        overlap=abs((um^lm).volume())
        assert overlap<.002,(state,overlap)
        receivers=[]
        for x in (-32.,32.):
            y=315.770102
            a=hole(um_crop,x,y,14.);b=hole(lm_crop,x,y,10.)
            axis=float(np.linalg.norm(np.array(a['center_xy_mm'])-b['center_xy_mm']))
            assert axis<.015,(state,axis)
            receivers.append(dict(x_mm=x,axis_error_mm=axis))
        lm_front=lm^md.Manifold.cube((120,40,.02)).translate((-60,300,18.28))
        um_front=um^md.Manifold.cube((120,40,30)).translate((-60,300,18.299))
        occlusion=float((um_front.project()^lm_front.project()).area())
        assert occlusion<.002,(state,occlusion)
        samples=[]
        for x in np.linspace(-30,30,13):
            ly=float(hits(lm_crop,[x,335,18.299],[0,-1,0])[0,1])
            uy=float(hits(um_crop,[x,ly-.01,18.299],[0,1,0])[0,1])
            gap=uy-ly;assert -.002<=gap<.3,(state,x,gap)
            step=float(hits(um_crop,[x,uy+.2,35],[0,0,-1])[0,2]-hits(lm_crop,[x,ly-.2,35],[0,0,-1])[0,2])
            assert abs(step)<.002,(state,x,step)
            samples.append(dict(x_mm=float(x),seam_gap_mm=gap,front_face_step_mm=step))
        rows.append(dict(state=state,LM=lm_source,overlap_mm3=overlap,receivers=receivers,
            LM_front_occlusion_mm2=occlusion,seam_samples=samples))
    return dict(UM=um_source,checks=rows)


def obiwan_vase(hand,state):
    top,top_source=installed(VASE/f'obiwan/STL/nd25fn4_vase_obiwan_tweeters_{hand}_top.stl')
    bottom,bottom_source=installed(ROOT/f'build/{state}/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl')
    released,released_source=installed(ROOT/f'build/{state}/stl/obiwan_optional_lm_keyed_2_of_2_top.stl')
    seam=md.Manifold.cube((400,40,80)).translate((-200,155,-40))
    top_m=solid(top);overlap=abs((top_m^seam^solid(bottom)).volume())
    assert overlap<.002,('vase top overlaps the keyed LM bottom',overlap)
    rel_m=solid(released)
    kept=md.Manifold.cube((400,278-172.3,80)).translate((-200,172.3,-40))
    # the two LM wing magnet cavities moved from 64/116 to 40/140 deg (y 273): compare the outer shell, count those separately
    shells=top.split(only_watertight=False);outer=max(shells,key=lambda q:abs(q.volume))
    cavities=[q for q in shells if q.volume<0 and q.bounds[0,1]<278]
    assert len(cavities)==2 and all(abs(abs(q.volume)-64.9)<1 for q in cavities),[round(q.volume,1) for q in cavities]
    raw_removed=((rel_m^kept)-solid(outer))
    # the vase top re-cuts the released T and MU10 lumens along their own polylines through the y 262-282 junction
    # (a capsule tube a few hundredths wider than the released cut): allow that 0.15 mm band, nothing else
    contract=json.loads((ROOT/f'build/{state}/support_blockers/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.json').read_text())
    band=None
    for r in contract['duct_collision_contract']['regions']:
        if r['name'] not in ('um_route_lumen','t_route_lumen'):continue
        P=np.array(r['points_xyz_mm']);k=np.flatnonzero((P[:,1]>262)&(P[:,1]<282))
        for a_,b_ in zip(P[max(k[0]-1,0):k[-1]+1],P[max(k[0]-1,0)+1:k[-1]+2]):
            L=float(np.linalg.norm(b_-a_))
            if L<1e-6:continue
            c=md.Manifold.cylinder(L+.4,r['radius_mm']+.15,r['radius_mm']+.15,32,True)
            import trimesh as _tm
            T=_tm.geometry.align_vectors([0,0,1],(b_-a_)/L);T[:3,3]=(a_+b_)/2
            c=c.transform(T[:3,:]);band=c if band is None else band+c
    removed=abs((raw_removed-band).volume())
    # A Boolean volume picks up sub-0.01 mm films between two finely tessellated copies of the same surface, so the pass
    # criterion is sampled: released material 0.02 mm under its own surface (outside the junction re-cut band) must be
    # inside the vase top.
    import trimesh as _tm
    rel_t=to_trimesh(rel_m^kept);pts,fi=_tm.sample.sample_surface_even(rel_t,8000)
    k=~((pts[:,1]>260)&(pts[:,1]<284))&(pts[:,1]>172.8);q=pts[k]-rel_t.face_normals[fi[k]]*.02   # the seam face itself is covered by the y 172.48-180 identity check
    band_top=to_trimesh(solid(outer)^md.Manifold.cube((400,278-170,80)).translate((-200,170,-40)))   # ring band only: far lighter ray queries
    out=int(sum((~band_top.contains(q[i:i+1000])).sum() for i in range(0,len(q),1000)))
    assert out==0,('vase top lacks released keyed-top material below y 278',out,removed)
    seam_band=md.Manifold.cube((400,180-172.3,80)).translate((-200,172.3,-40))
    a=top_m^seam_band;b=rel_m^seam_band
    differs=abs((a-b).volume())+abs((b-a).volume())
    # sampled both ways (volume picks up tessellation films): each band's material 0.02 mm under its surface lies in the other
    import trimesh as _tm2
    at,bt=to_trimesh(a),to_trimesh(b);seam_miss=0
    for X,Y in ((at,bt),(bt,at)):
        p_,f_=_tm2.sample.sample_surface_even(X,4000);k_=(p_[:,1]>172.8)&(p_[:,1]<179.7)
        Q_=p_[k_]-X.face_normals[f_[k_]]*.02
        seam_miss+=int(sum((~Y.contains(Q_[i:i+1000])).sum() for i in range(0,len(Q_),1000)))
    assert seam_miss==0,('vase top differs from the released keyed top next to the seam',seam_miss,differs)
    added=(top_m^kept)-rel_m
    # the taper may fill behind the 0.85 mm L22 membrane, but never reach the seat plane, the R95 driver cutout or the wing face R113.94
    lm_c=(0.,200.981)
    def disc(r,z0,z1):return md.Manifold.cylinder(z1-z0,r,r,256).translate((lm_c[0],lm_c[1],z0))
    forbidden=(md.Manifold.cube((400,278-172.3,40)).translate((-200,172.3,12.0))+disc(95.0,-40,40)
        +(md.Manifold.cube((400,278-172.3,80)).translate((-200,172.3,-40))-disc(113.94,-40,40)))
    added_front=abs((added^forbidden).volume())
    hit=to_trimesh(added^forbidden) if added_front>0 else None
    print('taper in forbidden zones: %.3f mm3' % added_front, None if hit is None or hit.is_empty else [round(float(v),2) for v in hit.bounds.ravel()], flush=True)
    assert added_front<.25,('rear taper reaches the L22 seat plane, the driver cutout or the wing face',added_front)   # remeshing noise along shared released faces is ~0.1 mm3 here
    return dict(top=top_source,LM_bottom=bottom_source,released_top=released_source,overlap_with_LM_bottom_mm3=overlap,
        released_material_removed_below_y278_mm3=removed,released_samples_missing_0p02mm_deep=out,junction_lumen_recut_mm3=abs(raw_removed.volume())-removed,relocated_magnet_cavities_below_y278=len(cavities),difference_to_released_top_y172_180_mm3=differs,seam_band_samples_off_0p02mm=seam_miss,
        rear_taper_added_below_y278_mm3=abs(added.volume()),rear_taper_in_seat_cutout_or_wing_zone_mm3=added_front)


def keyed_tops_identical():
    """The floor and no-floor keyed LM tops must be the same part: every surface point of each lies within 0.05 mm of
    the other (their BREPs share one construction; only STEP->STL meshing differs, measured 0.037 mm)."""
    import trimesh as _tm
    A=installed(ROOT/'build/floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.stl')[0]
    B=installed(ROOT/'build/no_floor_stand/stl/obiwan_optional_lm_keyed_2_of_2_top.stl')[0]
    worst=0.0
    for X,Y in ((A,B),(B,A)):
        p_=_tm.sample.sample_surface_even(X,60000)[0];worst=max(worst,float(_tm.proximity.closest_point(Y,p_)[1].max()))
    assert worst<.05,('the released keyed LM tops differ between stand states',worst)
    return worst


def stock_vase(hand):
    upper,upper_source=installed(VASE/f'stock/STL/nd25fn4_vase_stock_tweeters_{hand}_upper.stl')
    crop=md.Manifold.cube((400,40,80)).translate((-200,300,-40))
    upper_m=solid(upper)^crop
    section=upper.section(plane_origin=[0,316.05,0],plane_normal=[0,1,0])
    holes=[Polygon(np.asarray(l)[:,[0,2]]) for l in section.discrete]
    mouth=box(-33.31-3,11.5-3,-33.31+3,11.5+3)
    assert any(h.buffer(.05).contains(mouth) for h in holes),'Stock vase vestibule misses the LM D6 duct mouth'
    rows=[]
    for state in ('no_floor_stand','upper_floor_stand'):
        lm,lm_source=installed(ROOT/f'build/h2c/STL/h2c_stock_lm_{state}.stl')
        overlap=abs((upper_m^solid(lm)).volume())
        assert overlap<.002,(state,overlap)
        rows.append(dict(LM=lm_source,overlap_mm3=overlap))
    return dict(upper=upper_source,checks=rows,vestibule_contains_LM_duct_mouth=True)


def _isolated(fn,argsets):
    """Run each heavy mesh comparison in a fresh worker (one at a time) so memory is returned between them."""
    import concurrent.futures as cf,multiprocessing as mp
    out={}
    with cf.ProcessPoolExecutor(max_workers=1,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as ex:
        for args in argsets:out['_on_'.join(args)]=ex.submit(fn,*args).result()
    return out


def main():
    result=dict(status='pass',regular_obiwan_UM=regular_um(),dayton_nd25fn4_obiwan=_isolated(obiwan_vase,[(h,s) for h in ('right','left') for s in ('floor_stand','no_floor_stand')]),
        released_keyed_top_floor_vs_no_floor_max_mm=keyed_tops_identical(),
        dayton_nd25fn4_stock={h:stock_vase(h) for h in ('right','left')},source_sha256=sha256_file(Path(__file__)),physical_fit='hardware trial pending')
    write_json(ROOT/'build/h2c/obiwan_interface_validation.json',result)
    print('H2C LM interfaces: regular Obiwan UM, Obiwan and Stock ND25FN-4 vases clear and registered.',flush=True)


if __name__=='__main__':main()
