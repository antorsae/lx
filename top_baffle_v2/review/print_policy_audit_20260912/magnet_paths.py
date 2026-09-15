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
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts')]
from gcode_analysis import parse_gcode
OUT=Path(__file__).parent
stl=ROOT/'candidates/nd25fn4_crescent/print/geometry/01_UM_Crescent_V4_PRINT.stl'
gcode=ROOT/'review/nd25fn4_print/body/plate_1.gcode'
prep=json.loads((ROOT/'review/nd25fn4_print/preparation.json').read_text())['body']
sites=json.loads((ROOT/'review/nd25fn4_print/body/magnet_discovery.json').read_text())
mesh=trimesh.load_mesh(stl,process=True);mesh.apply_translation(prep['offset'])
parts=mesh.split(only_watertight=False)
cavities=[p for p in parts if p.volume<0]
roi=[(s['center_bed_mm'][0]-9,s['center_bed_mm'][1]-9,s['center_bed_mm'][0]+9,s['center_bed_mm'][1]+9) for s in sites]
parsed=parse_gcode(gcode,retain_regions=roi)
print('Loaded mesh and toolpaths',flush=True)
colors={'Outer wall':'#14578b','Inner wall':'#5599cb','Internal solid infill':'#b8a386','Sparse infill':'#daab40','Gap infill':'#cc784b','Support':'#839786','Support interface':'#a443b8','Bridge':'#36a282'}
fig,axes=plt.subplots(4,4,figsize=(16,16))
rows=[]
for i,s in enumerate(sites):
 c=np.array(s['center_bed_mm']);cavity=min(cavities,key=lambda p:np.linalg.norm(p.center_mass-c))
 for j,wanted in enumerate([3.32,6.12,9.48,9.64]):
  layer=min(parsed.layers,key=lambda l:abs(l.z-wanted));z=layer.z-(layer.layer_height or .16)/2
  bounds=roi[i];inside=(mesh.triangles_center[:,0]>bounds[0]-3)&(mesh.triangles_center[:,0]<bounds[2]+3)&(mesh.triangles_center[:,1]>bounds[1]-3)&(mesh.triangles_center[:,1]<bounds[3]+3)
  faces=np.flatnonzero(inside)
  lines=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z],local_faces=faces)
  edges=trimesh.intersections.mesh_plane(cavity,[0,0,1],[0,0,z])
  ax=axes[i,j];ax.add_collection(LineCollection(lines[:,:,:2],colors='#222222',linewidths=.75,zorder=4))
  segs=[p for p in layer.segments if bounds[0]-1<((p.x0+p.x1)/2)<bounds[2]+1 and bounds[1]-1<((p.y0+p.y1)/2)<bounds[3]+1 and p.feature not in ['Custom','Undefined','Prime tower']]
  allbeads=[];features={}
  for p in segs:
   path=LineString([(p.x0,p.y0),(p.x1,p.y1)]);bead=path.buffer((p.line_width or .62)/2,quad_segs=4)
   if not p.feature.lower().startswith('support'):allbeads.append(bead)
   x,y=bead.exterior.xy;ax.fill(x,y,color=colors.get(p.feature,'#bbbbbb'),alpha=.7,lw=0)
   features[p.feature]=features.get(p.feature,0)+p.length
  covered=unary_union(allbeads)
  samples=[]
  for a,b in edges[:,:,:2]:
   n=max(2,int(np.linalg.norm(b-a)/.06)+1);samples.extend(a+(b-a)*np.linspace(0,1,n)[:,None])
  distances=np.array([covered.distance(Point(p)) for p in samples])
  row={'site':s['name'],'layer_z_mm':layer.z,'section_z_mm':z,'boundary_samples':len(samples),'max_cavity_boundary_distance_to_model_bead_mm':float(distances.max()) if len(distances) else None,'boundary_within_0p08_mm_fraction':float(np.mean(distances<=.08)) if len(distances) else None,'feature_length_in_roi_mm':features}
  rows.append(row)
  ax.set(xlim=(c[0]-7,c[0]+7),ylim=(c[1]-7,c[1]+7),aspect='equal',title=f"{s['name']} | layer Z {layer.z:.2f}")
  ax.grid(alpha=.12)
 print(s['name'],flush=True)
fig.suptitle('Crescent magnet cavities — actual supplied body toolpaths\nBlack: STL boundaries at bead mid-height | blue: walls | tan: solid infill | green: support',fontsize=15)
fig.tight_layout(rect=(0,0,1,.955));fig.savefig(OUT/'crescent_magnet_toolpaths.png',dpi=180);plt.close(fig)
report={'scope':'Four representative layers per cavity; CAD cross-section versus deposited model bead footprints. Not a physical adhesion or strength test.','stl_sha256':hashlib.sha256(stl.read_bytes()).hexdigest(),'gcode_sha256':hashlib.sha256(gcode.read_bytes()).hexdigest(),'rows':rows}
(OUT/'crescent_magnet_toolpaths.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(rows,indent=1),flush=True)
