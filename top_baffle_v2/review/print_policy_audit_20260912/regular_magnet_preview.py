from pathlib import Path
import sys,json,hashlib
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString,Point
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;sys.path[:0]=[str(ROOT/'scripts')]
from gcode_analysis import parse_gcode
stl=ROOT/'build/no_floor_stand/stl/obiwan_core_2_of_2_um_carrier.stl';gcode=OUT/'um_slice/plate_1.gcode'
r=next(x for x in json.loads((ROOT/'to_print/obiwan/3mf_06hf_petg-gf_pla/gui_projects.json').read_text())['projects'] if x['name']=='obiwan_03_UM_carrier_1_of_1')
mesh=trimesh.load_mesh(stl,process=True);mesh.apply_transform(r['geometry_audit']['stl_to_bed_matrix']);cavities=sorted([p for p in mesh.split(only_watertight=False) if p.volume<0],key=lambda p:p.center_mass[0])
roi=[(m.center_mass[0]-7,m.center_mass[1]-7,m.center_mass[0]+7,m.center_mass[1]+7) for m in cavities];parsed=parse_gcode(gcode,retain_regions=roi)
fig,axes=plt.subplots(2,3,figsize=(12,8));rows=[]
for i,cav in enumerate(cavities):
 c=cav.center_mass;bounds=roi[i]
 for j,want in enumerate([.84,3.4,5.8]):
  layer=min(parsed.layers,key=lambda l:abs(l.z-want));z=layer.z-(layer.layer_height or .16)/2;ax=axes[i,j]
  lines=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z]);ax.add_collection(LineCollection(lines[:,:,:2],colors='#20242c',lw=.8,zorder=4));beads=[];widths=[]
  edges=trimesh.intersections.mesh_plane(cav,[0,0,1],[0,0,z]);edge=unary_union([LineString(e[:,:2]) for e in edges])
  for p in layer.segments:
   if p.feature.lower() in ['custom','undefined','prime tower'] or p.feature.lower().startswith('support'):continue
   line=LineString([(p.x0,p.y0),(p.x1,p.y1)]);poly=line.buffer((p.line_width or .62)/2,quad_segs=4);beads.append(poly)
   x,y=poly.exterior.xy;ax.fill(x,y,color='#14578b' if p.feature=='Outer wall' else '#71a7cc' if p.feature=='Inner wall' else '#c4ae8e',alpha=.7,lw=0)
   if p.feature=='Outer wall' and line.distance(edge)<.39:widths.append(p.line_width)
  cover=unary_union(beads);sample=[]
  for a,b in edges[:,:,:2]:sample.extend(a+(b-a)*np.linspace(0,1,max(2,int(np.linalg.norm(b-a)/.06)+1))[:,None])
  dist=[cover.distance(Point(p)) for p in sample]
  rows.append({'site':i,'z':layer.z,'max_cavity_boundary_distance_mm':max(dist),'sampled_near_cavity_outer_wall_width_range_mm':[min(widths),max(widths)] if widths else None})
  ax.set(xlim=(c[0]-4.5,c[0]+4.5),ylim=(c[1]-4.5,c[1]+4.5),aspect='equal',title=f"Regular UM {'left' if i==0 else 'right'} — Z {layer.z:.2f}");ax.grid(alpha=.1)
fig.suptitle('Regular Obi-Wan Ø5×2 magnet walls — current mesh, fresh review slice\nBlack: actual STL section | dark blue: outer-wall beads | light blue: inner-wall beads',fontsize=14);fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'regular_UM_magnet_toolpaths.png',dpi=180)
(OUT/'regular_UM_magnet_toolpaths.json').write_text(json.dumps({'scope':'Three representative bead-midplane sections per cavity, separate from the failed release-catalog provenance gate. Not a release certification.','stl_sha256':hashlib.sha256(stl.read_bytes()).hexdigest(),'gcode_sha256':hashlib.sha256(gcode.read_bytes()).hexdigest(),'rows':rows},indent=2)+'\n');print(json.dumps(rows),flush=True)
