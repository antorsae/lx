"""Measure visible outside wall paths separately from hidden cavity walls."""
from pathlib import Path
import argparse
import json
import re
import numpy as np
import trimesh
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from translucent_print import ROOT, WORK, write_json, sha
from gcode_analysis import parse_gcode


def inspect(gcode, output):
    prep=json.loads((WORK/'preparation.json').read_text())['body']
    specs=json.loads((WORK/'body/magnet_discovery.json').read_text())
    mesh=trimesh.load_mesh(ROOT/prep['sources'][0]['path'],process=True)
    mesh.apply_translation(prep['offset'])
    outer=max(mesh.split(only_watertight=False),key=lambda m:m.volume)
    rois=[(s['center_bed_mm'][0]-10,s['center_bed_mm'][1]-10,s['center_bed_mm'][0]+10,s['center_bed_mm'][1]+10) for s in specs]
    parsed=parse_gcode(gcode,retain_regions=rois,retain_feature_prefixes=('Outer wall',))
    relevant={s.line_number for l in parsed.layers for s in l.segments}
    feed=0.;feeds={}
    for i,line in enumerate(gcode.open(),1):
        command=line.split(';')[0].strip()
        if re.match(r'^G[0123](?:\s|$)',command):
            f=re.search(r'(?:^|\s)F([\d.]+)',command)
            if f: feed=float(f[1])/60
        if i in relevant:feeds[i]=feed
    centers=outer.triangles_center;normals=outer.face_normals
    rows=[]
    for spec in specs:
        center=np.asarray(spec['center_bed_mm']);n=np.asarray(spec['pole_axis_bed'])
        select=(np.abs(centers[:,0]-center[0])<10)&(np.abs(centers[:,1]-center[1])<10)&(normals@n>.65)
        skin=outer.submesh([select],append=True,repair=False)
        layers=[];all_widths=[];all_speeds=[]
        for layer in parsed.layers:
            z=layer.z-(layer.layer_height or .16)/2
            if not spec['seated_bottom_z_mm']+.5<z<spec['seated_top_z_mm']-.5: continue
            edges=trimesh.intersections.mesh_plane(skin,[0,0,1],[0,0,z])
            edges=[e for e in edges if np.linalg.norm(e.mean(axis=0)[:2]-center[:2])<8]
            if not edges:continue
            boundary=unary_union([LineString(e[:,:2]) for e in edges])
            paths=[]
            for s in layer.segments:
                p=np.array([(s.x0+s.x1)/2,(s.y0+s.y1)/2])
                if np.linalg.norm(p-center[:2])>8:continue
                line=LineString([(s.x0,s.y0),(s.x1,s.y1)])
                w=s.line_width or .52
                distance=line.distance(boundary)
                if abs(distance-w/2)>.07:continue
                paths.append(dict(line=s.line_number,width=w,speed=feeds[s.line_number],
                    a=[s.x0,s.y0],b=[s.x1,s.y1],length=s.length))
            assert paths,(spec['name'],layer.z,'no visible outside wall')
            widths=[p['width'] for p in paths];speeds=[p['speed'] for p in paths]
            footprints=unary_union([LineString([p['a'],p['b']]).buffer(p['width']/2,quad_segs=5) for p in paths])
            witnesses=[]
            for edge in edges:
                a,b=edge[:,:2];count=max(2,int(np.ceil(np.linalg.norm(b-a)/.05))+1)
                witnesses.extend(p for p in a+(b-a)*np.linspace(0,1,count)[:,None]
                                 if np.linalg.norm(p-center[:2])<6.)
            gap=max((footprints.distance(Point(p)) for p in witnesses),default=0.)
            all_widths.extend(widths);all_speeds.extend(speeds)
            layers.append(dict(z_mm=layer.z,width_range_mm=[min(widths),max(widths)],
                speed_range_mm_s=[min(speeds),max(speeds)],paths=paths,
                exterior_witnesses=len(witnesses),max_exterior_boundary_gap_mm=gap,
                exterior_section=[e[:,:2].tolist() for e in edges]))
        assert len(layers)>25
        rows.append(dict(site=spec['name'],center_bed_mm=spec['center_bed_mm'],
            width_range_mm=[min(all_widths),max(all_widths)],speed_range_mm_s=[min(all_speeds),max(all_speeds)],
            max_exterior_boundary_gap_mm=max(l['max_exterior_boundary_gap_mm'] for l in layers),layers=layers))
    report=dict(gcode=str(gcode.relative_to(ROOT)),gcode_sha256=sha(gcode),script_sha256=sha(__file__),
        method='Only model outer-wall paths that touch the outward-facing exterior mesh section near each pocket; hidden cavity-wall paths are excluded.',
        scope='Measured toolpaths; cannot establish physical surface appearance',sites=rows)
    write_json(output,report)
    print(json.dumps([{k:v for k,v in s.items() if k!='layers'} for s in rows],indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('gcode',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();inspect(a.gcode.resolve(),a.output.resolve())
