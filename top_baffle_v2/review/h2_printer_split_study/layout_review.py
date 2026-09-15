"""Draw true-scale screening layouts; do not emit printer or CAD files."""
from pathlib import Path
import json,hashlib
import numpy as np
import trimesh
from shapely.geometry import Polygon,box
from shapely.affinity import rotate,translate
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
REPORT=json.loads((OUT/'measurements.json').read_text());PARTS=REPORT['parts']
BED=(300,320);BRIM=5;EDGE=2;TOWER=50;GAP=5

def layout(name):
 part=PARTS[name];outline=Polygon(part['convex_outline_xy_mm'])
 angle=part['beds']['H2_dual_shared']['angle_deg']
 poly=rotate(outline,angle,origin=(0,0));b=poly.bounds
 offset=((BED[0]-(b[2]-b[0]))/2-b[0],(BED[1]-(b[3]-b[1]))/2-b[1])
 poly=translate(poly,*offset);expanded=poly.buffer(BRIM)
 tower=None
 if part['beds']['H2_dual_shared']['planning_allowances_fit']:
  candidates=[]
  for x in range(EDGE,BED[0]-TOWER-EDGE+1,4):
   for y in range(EDGE,BED[1]-TOWER-EDGE+1,4):
    tile=box(x,y,x+TOWER,y+TOWER)
    distance=tile.distance(expanded)
    if distance>=GAP:candidates.append((distance,x,y))
  if candidates:
   distance,x,y=max(candidates);tower={'x_mm':x,'y_mm':y,'width_mm':TOWER,'depth_mm':TOWER,'distance_to_5mm_brim_mm':distance}
 return {'angle_deg':angle,'translation_xy_mm':offset,'tower':tower},poly,expanded

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
fig,axs=plt.subplots(2,2,figsize=(14,16.5),facecolor='#f6f8fb')
fig.subplots_adjust(left=.055,right=.965,top=.905,bottom=.15,hspace=.36,wspace=.17)
fig.suptitle('H2C / H2D — which joins can disappear?',x=.055,y=.976,ha='left',fontsize=23,fontweight='bold',color='#182d43')
fig.text(.055,.946,'Actual project meshes · same scale · 300 × 320 mm shared dual-nozzle area',fontsize=13,color='#41566b')
selected=[('stock_no_floor_stand_LM_merged','Stock / Slim — complete lower section','Too tight for a normal brim in the shared area',False),('stock_no_floor_stand_LM_mids_merged','Stock / Slim — join the two middle pieces','Fits; keep a separate bottom/base piece',True),('obiwan_no_floor_stand_LM','Obiwan — complete LM carrier','Fits as one piece; floor-stand version also fits',True),('obiwan_flat_wing_left_whole','Obiwan — full wing, one per side','Fits; flat and graded have the same footprint',True)]
proof={}
for ax,(name,title,subtitle,good) in zip(axs.flat,selected):
 item=PARTS[name];witness,poly,expanded=layout(name);proof[name]=witness
 angle=np.deg2rad(witness['angle_deg']);rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
 colour='#347daf' if good else '#b56762'
 ax.set_facecolor('white');ax.add_patch(Rectangle((0,0),*BED,facecolor='#eef2f6',edgecolor='#40566e',lw=1.5,zorder=0))
 ax.add_patch(Rectangle((EDGE,EDGE),BED[0]-2*EDGE,BED[1]-2*EDGE,fill=False,edgecolor='#a0aeba',ls=':',lw=.8))
 xs,ys=expanded.exterior.xy;ax.fill(xs,ys,color=colour,alpha=.16,zorder=1)
 for path in item['sources']:
  p=ROOT/path;metadata=json.loads(p.with_suffix('.print.json').read_text())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==metadata['stl_sha256']
  mesh=trimesh.load_mesh(p,process=False)
  vertices=trimesh.transform_points(mesh.vertices,np.linalg.inv(metadata['source_to_stl_matrix']))
  xy=(vertices[:,:2]*np.array([1,-1]))@rotation.T+np.array(witness['translation_xy_mm'])
  ax.add_collection(PolyCollection(xy[mesh.faces],facecolors=colour,edgecolors='none',rasterized=True,zorder=2))
 if witness['tower']:
  t=witness['tower'];ax.add_patch(Rectangle((t['x_mm'],t['y_mm']),TOWER,TOWER,facecolor='#f2c978',edgecolor='#aa741e',lw=1,zorder=3))
  ax.text(t['x_mm']+TOWER/2,t['y_mm']+TOWER/2,'Prime\ntower\nreserve',ha='center',va='center',fontsize=8.5,color='#674811',zorder=4)
 elif good:subtitle+='; prime-tower placement still to resolve'
 ax.set_title(title,loc='left',fontsize=14,fontweight='bold',pad=30,color='#182d43')
 ax.text(0,1.025,subtitle,transform=ax.transAxes,fontsize=10.8,color='#2b7160' if good else '#a44742')
 ax.set(xlim=(-12,312),ylim=(-12,332),aspect='equal',xlabel='Bed X (mm)',ylabel='Bed Y (mm)')
 ax.set_xticks([0,100,200,300]);ax.set_yticks([0,100,200,300]);ax.grid(alpha=.1,zorder=0)
 for spine in ax.spines.values():spine.set_visible(False)
 f=item['beds']['H2_dual_shared'];dims=f['part_width_depth_mm']
 ax.text(0,-.145,f"Rotated part: {dims[0]:.1f} × {dims[1]:.1f} mm   |   Z rotation {f['angle_deg']:.1f}°",transform=ax.transAxes,color='#41566b',fontsize=10.5)
fig.text(.055,.035,'Blue/red: actual mesh projection. Pale surround: 5 mm brim allowance. Dashed line: 2 mm bed-edge margin.\nYellow: 50 × 50 mm planning reserve, ≥5 mm from the part brim. No supports or G-code generated.\nJoining the Stock/Slim mids and wing segments requires CAD regeneration; this is a purchasing study.',fontsize=10.5,color='#41566b',linespacing=1.55)
fig.savefig(OUT/'H2_split_comparison.png',dpi=175,facecolor=fig.get_facecolor());plt.close(fig)
additional = [
 'obiwan_floor_stand_LM',
 *[f'{family}_{state}_LM_mids_merged' for family in ['stock','slim'] for state in ['no_floor_stand','floor_stand']],
 *[f'{family}_{state}_LM_above_80' for family in ['stock','slim'] for state in ['no_floor_stand','floor_stand']],
 *[f'{prefix}_{family}_wing_{side}_whole' for prefix in ['obiwan','V4'] for family in ['flat','graded'] for side in ['left','right']],
]
for name in additional:
 if name not in proof:
  witness,_,_=layout(name);proof[name]=witness
checks={}
for name,witness in proof.items():
 _,poly,expanded=layout(name)
 good=PARTS[name]['beds']['H2_dual_shared']['planning_allowances_fit']
 check={'part_and_brim_inside_2mm_edge':box(EDGE,EDGE,BED[0]-EDGE,BED[1]-EDGE).covers(expanded)}
 if witness['tower']:
  t=witness['tower'];tile=box(t['x_mm'],t['y_mm'],t['x_mm']+TOWER,t['y_mm']+TOWER)
  check.update(tower_inside_2mm_edge=box(EDGE,EDGE,BED[0]-EDGE,BED[1]-EDGE).covers(tile),tower_at_least_5mm_from_brim=tile.distance(expanded)>=GAP)
 if good:
  assert check['part_and_brim_inside_2mm_edge'],name
  assert witness['tower'] and check['tower_inside_2mm_edge'] and check['tower_at_least_5mm_from_brim'],name
 checks[name]=check
(OUT/'layout_witnesses.json').write_text(json.dumps({'bed_xy_mm':BED,'coordinate_frame':'Local shared-nozzle area: add 25 mm to X to obtain the physical bed coordinate.','prime_tower_size_is_planning_allowance':True,'layouts':proof,'geometric_checks':checks},indent=2)+'\n')
for name,witness in proof.items():print(name,witness)
