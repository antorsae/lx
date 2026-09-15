"""Add an entirely internal loading relief for tilted pause-and-bury magnets.

The approved cosmetic STL is immutable. Print variants remove only the part
of a tilted chimney roof that otherwise closes before its magnet is below
the nozzle. The original seats, disc sizes, axes and exterior are retained.
"""
from pathlib import Path
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from print_magnets import magnet_geometry
from mesh_ops import solid,to_trimesh,preserve_void_winding
from prepare_print import HERE,ROOT,WORK,sha


def prepare(stl):
    authority=json.loads(stl.with_suffix('.print.json').read_text())
    source=trimesh.load_mesh(stl,process=True)
    outer=max(source.split(only_watertight=False),key=lambda m:m.volume)
    # Only the four/two relocated D6 pockets need the relief. The retained
    # upright LM pocket still uses its original D5 loading chimney.
    specs=[s for s in magnet_geometry(stl,authority,[0,0,0]) if s['diameter_mm']==6]
    reliefs=[];rows=[]
    for spec in specs:
        magnet=spec['_magnet'].copy();center=np.asarray(spec['center_bed_mm'])
        axis=np.asarray(spec['pole_axis_bed']);v=magnet.vertices-center
        axial=np.outer(v@axis,axis)
        magnet.vertices=center+(v-axial)*(6.10/6)+axial*(3.06/3)
        sweep=trimesh.convex.convex_hull(np.r_[magnet.vertices,magnet.vertices+np.asarray(spec['approach_axis_bed'])*25])
        roof=float(np.ceil((spec['seated_top_z_mm']+.12)*100)/100)
        relief=sweep.slice_plane([0,0,roof],[0,0,-1],cap=True)
        # Deterministic subdivision witnesses include planar face interiors.
        vertices,faces=trimesh.remesh.subdivide_to_size(relief.vertices,relief.faces,max_edge=.20,max_iter=9)
        # Triangulated circular caps contain many long, skinny fan triangles.
        # Collapse their redundant witnesses into 0.10-mm spatial cells.
        _,unique=np.unique(np.round(vertices/.10).astype(np.int32),axis=0,return_index=True)
        points=vertices[unique]
        print(stl.name,spec['name'],'cover witnesses',len(points),flush=True)
        _,distance,_=trimesh.proximity.closest_point(outer,points)
        assert distance.min()>=.75,(stl.name,spec['name'],distance.min())
        rows.append(dict(site=spec['name'],flat_loading_roof_z_mm=roof,
                         seated_magnet_top_z_mm=spec['seated_top_z_mm'],
                         minimum_exterior_cover_mm=float(distance.min()),cover_witnesses=len(points)))
        reliefs.append(relief)
    original=solid(source);cut=solid(trimesh.util.concatenate(reliefs));result=original-cut
    removed=original^cut
    assert removed.volume()>0
    # The cutter is wholly buried and cannot alter mounting surfaces or ducts.
    removed_mesh=to_trimesh(removed)
    assert outer.contains(removed_mesh.vertices).all()
    assert abs((original.volume()-result.volume())-removed.volume())<.01
    final=preserve_void_winding(to_trimesh(result.set_tolerance(.00005)))
    output=HERE/'print/geometry';output.mkdir(parents=True,exist_ok=True)
    target=output/(stl.stem+'_PRINT.stl');final.export(target)
    reloaded=trimesh.load_mesh(target,process=True);components=reloaded.split(only_watertight=False)
    assert reloaded.is_watertight and reloaded.is_winding_consistent
    assert sum(m.volume>0 for m in components)==1
    assert sum(m.volume<0 for m in components)==len(specs)+(0 if stl.name.startswith('01_') else 1)
    authority.update(stl=target.name,stl_sha256=sha(target),approved_source=str(stl.relative_to(ROOT)),
                     approved_source_sha256=sha(stl),print_preparation='internal tilted-magnet loading relief; exterior, seats and interfaces retained',
                     qualification='slice audit pending')
    target.with_suffix('.print.json').write_text(json.dumps(authority,indent=2)+'\n')
    report=dict(source=str(stl.relative_to(ROOT)),source_sha256=sha(stl),print_stl=str(target.relative_to(ROOT)),
                print_stl_sha256=sha(target),removed_mm3=float(removed.volume()),added_mm3=0,
                exterior_unchanged=True,watertight=True,positive_material_components=1,
                closed_magnet_cavities=int(sum(m.volume<0 for m in components)),sites=rows)
    (output/(stl.stem+'_loading.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(stl.name,json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    import sys
    paths=[HERE/'STL/01_UM_Crescent_V4.stl'] if '--body' in sys.argv else sorted((HERE/'STL/wings').glob('*.stl'))
    for path in paths:prepare(path)
