from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh
from shapely.geometry import LineString,Point
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'scripts')]
from gcode_analysis import parse_gcode
stl=ROOT/'candidates/nd25fn4_crescent/print/geometry/01_UM_Crescent_V4_PRINT.stl';gcode=ROOT/'review/nd25fn4_print/body/plate_1.gcode'
prep=json.loads((ROOT/'review/nd25fn4_print/preparation.json').read_text())['body'];sites=json.loads((ROOT/'review/nd25fn4_print/body/magnet_discovery.json').read_text())
mesh=trimesh.load_mesh(stl,process=True);mesh.apply_translation(prep['offset']);cavities=[p for p in mesh.split(only_watertight=False) if p.volume<0]
roi=[(s['center_bed_mm'][0]-9,s['center_bed_mm'][1]-9,s['center_bed_mm'][0]+9,s['center_bed_mm'][1]+9) for s in sites]
parsed=parse_gcode(gcode,retain_regions=roi);reports=[]
for i,s in enumerate(sites):
 c=np.asarray(s['center_bed_mm']);cavity=min(cavities,key=lambda p:np.linalg.norm(p.center_mass-c));bounds=roi[i];rows=[]
 for layer in parsed.layers:
  z=layer.z-(layer.layer_height or .16)/2
  if z<cavity.bounds[0,2] or z>cavity.bounds[1,2]:continue
  edges=trimesh.intersections.mesh_plane(cavity,[0,0,1],[0,0,z]);samples=[]
  for a,b in edges[:,:,:2]:
   num=max(2,int(np.linalg.norm(b-a)/.06)+1);samples.extend(a+(b-a)*np.linspace(0,1,num)[:,None])
  if not samples:continue
  beads=[];paths=[]
  for p in layer.segments:
   if p.feature.lower() in ['custom','undefined','prime tower'] or p.feature.lower().startswith('support'):continue
   if not(bounds[0]-1<(p.x0+p.x1)/2<bounds[2]+1 and bounds[1]-1<(p.y0+p.y1)/2<bounds[3]+1):continue
   path=LineString([(p.x0,p.y0),(p.x1,p.y1)]);beads.append(path.buffer((p.line_width or .62)/2,quad_segs=4));paths.append(p)
  cover=unary_union(beads);distance=np.array([cover.distance(Point(p)) for p in samples]);edge_length=float(np.linalg.norm(edges[:,1,:2]-edges[:,0,:2],axis=1).sum())
  n=np.asarray(s['pole_axis_bed'])[:2];n/=np.linalg.norm(n)
  center=np.mean(np.asarray(samples),axis=0);ray=LineString([center-n*12,center+n*12]);poly=unary_union([LineString(x[:,:2]) for x in edges]);hits=poly.intersection(ray)
  hits=[hits] if hits.geom_type=='Point' else list(getattr(hits,'geoms',[]));hits=[p for p in hits if p.geom_type=='Point']
  exposed=None
  if hits:
   mouth=max(hits,key=lambda p:np.asarray(p.coords[0])@n);pos=np.asarray(mouth.coords[0]);near=[]
   for p in paths:
    line=LineString([(p.x0,p.y0),(p.x1,p.y1)]);hit=line.intersection(LineString([pos,pos+n*2.0]))
    if not hit.is_empty:near.append({'feature':p.feature,'width_mm':p.line_width,'path_id':p.path_id})
   exposed=near
  rows.append({'z':layer.z,'edge_length_mm':edge_length,'sample_count':len(samples),'max_edge_distance_mm':float(distance.max()),'fraction_within_0p08':float(np.mean(distance<=.08)),'interface_face_crossings':exposed})
 r={'site':s['name'],'cavity_z_bounds_mm':cavity.bounds[:,2].tolist(),'layers':rows,'worst_edge_distance_mm':max(x['max_edge_distance_mm'] for x in rows),'sampled_layers':len(rows)}
 reports.append(r);print(s['name'],r['sampled_layers'],'worst',r['worst_edge_distance_mm'],'exceptions',[(x['z'],round(x['max_edge_distance_mm'],3)) for x in rows if x['max_edge_distance_mm']>.08],flush=True)
report={'scope':'All cavity-intersecting sliced bead midplanes; samples along the cavity boundary at <=0.06 mm spacing; model bead footprints only. This measures boundary presence, not physical extrusion, adhesion or retention.','gcode_sha256':hashlib.sha256(gcode.read_bytes()).hexdigest(),'stl_sha256':hashlib.sha256(stl.read_bytes()).hexdigest(),'sites':reports}
(OUT/'crescent_magnet_fullheight.json').write_text(json.dumps(report,indent=2)+'\n')
