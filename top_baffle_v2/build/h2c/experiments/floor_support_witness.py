from pathlib import Path
import sys,json,numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[3];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from gcode_analysis import parse_gcode
j=next(j for j in json.loads((ROOT/'to_print/h2c/catalog.json').read_text())['jobs'] if j['name']=='h2c_slim_lm_lower_floor_stand')
d=json.loads((ROOT/'build/h2c/contracts'/f"{j['name']}.bores.json").read_text())
bore=next(b for b in d['bores'] if b['name']=='concave_cylinder_5')
mat=np.asarray(j['source_to_stl_matrix']);mat[:3,3]+=j['offset'];inverse=np.linalg.inv(mat)
a,b=trimesh.transform_points([bore['start'],bore['end']],mat);axis=b-a
parsed=parse_gcode(ROOT/j['work']/'plate_1.gcode',retain_feature_prefixes=('support',));records=[]
for layer in parsed.layers:
 if not 80.6<layer.z<81.1:continue
 for s in layer.segments:
  p=np.array([s.x0,s.y0,layer.z-(layer.layer_height or .16)/2]);q=np.array([s.x1,s.y1,p[2]])
  pts=p+(q-p)*np.linspace(0,1,max(2,int(np.ceil(s.length/.25))+1))[:,None]
  t=(pts-a)@axis/np.dot(axis,axis);dist=np.linalg.norm(pts-a-t[:,None]*axis,axis=1)-bore['radius_mm']-(s.line_width or .62)/2
  eligible=(t>1e-5)&(t<1-1e-5)&(dist<-.03)
  for i in np.flatnonzero(eligible):records.append(dict(line=s.line_number,feature=s.feature,print_z=layer.z,installed=trimesh.transform_points([pts[i]],inverse)[0].tolist(),distance=float(dist[i]),t=float(t[i])))
records=sorted(records,key=lambda x:x['distance']);print('bore',bore,'n',len(records));print(records[:3]);
points=np.array([r['installed'] for r in records[:3]])
# Test geometry at the same installed points, independent of cylinder bounds.
from build123d import import_step,Vector
shape=import_step(str(ROOT/f"build/h2c/STEP/{j['name']}.step"))
print('in material', [shape.is_inside(Vector(*p)) for p in points])
raw=ROOT/j['blockers'][0];auth=json.loads((ROOT/j['source']).with_suffix('.print.json').read_text());blocker=trimesh.load_mesh(raw);blocker.apply_transform(np.linalg.inv(auth['source_to_stl_matrix']))
print('in original blocker', np.any([c.contains(points) for c in blocker.split()],axis=0))
new=trimesh.load_mesh(ROOT/f"build/h2c/inputs/{j['name']}_support_blocker.stl");new.apply_transform(np.linalg.inv(np.asarray(j['source_to_stl_matrix'])))
print('in expanded blocker',np.any([c.contains(points) for c in new.split()],axis=0))
(ROOT/'build/h2c/experiments/floor_support_witness.json').write_text(json.dumps(records[:30],indent=2))
