from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from shapely.geometry import LineString
from shapely.ops import unary_union
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent;sys.path[:0]=[str(ROOT/'scripts')]
from gcode_analysis import parse_gcode
fig,axes=plt.subplots(2,3,figsize=(13,9));reports=[]
for row,(name,path,zs) in enumerate([('Obi-Wan floor',OUT/'floor_slice/plate_1.gcode',[5.64,5.8,5.96]),('Regular Obi-Wan UM',OUT/'um_slice/plate_1.gcode',[3.72,3.88,4.04])]):
 parsed=parse_gcode(path,retain_feature_prefixes=['Support interface']);layers=[min(parsed.layers,key=lambda l:abs(l.z-z)) for z in zs];polys=[]
 for l in layers:polys.append(unary_union([LineString([(s.x0,s.y0),(s.x1,s.y1)]).buffer((s.line_width or .62)/2,quad_segs=3) for s in l.segments if s.feature=='Support interface']))
 common=polys[0].intersection(polys[1]).intersection(polys[2]);geoms=list(common.geoms) if hasattr(common,'geoms') else [common];region=max(geoms,key=lambda p:p.area);pt=region.representative_point();cx,cy=pt.x,pt.y
 reports.append({'part':name,'PLA_top_interface_z_mm':zs,'common_interface_area_mm2':common.area,'witness_xy_mm':[cx,cy],'witness_covered_on_each_layer':[p.covers(pt) for p in polys]})
 for col,(l,p) in enumerate(zip(layers,polys)):
  ax=axes[row,col]
  for poly in list(p.geoms) if hasattr(p,'geoms') else [p]:
   x,y=poly.exterior.xy;ax.fill(x,y,color='#9e449d',lw=0)
   for hole in poly.interiors:
    x,y=hole.xy;ax.fill(x,y,color='white',lw=0)
  ax.scatter([cx],[cy],s=30,color='#222222',marker='+',zorder=4)
  ax.set(xlim=(cx-10,cx+10),ylim=(cy-10,cy+10),aspect='equal',title=f'{name}\nPLA interface {col+1}/3 — Z {l.z:.2f} mm',xlabel='Bed X (mm)',ylabel='Bed Y (mm)');ax.grid(alpha=.1)
fig.suptitle('Actual PLA toolpaths in fresh review slices\nSame XY window across three consecutive 0.16 mm layers; + is a common coverage witness',fontsize=14)
fig.tight_layout(rect=(0,0,1,.93));fig.savefig(OUT/'obiwan_PLA_three_layers.png',dpi=180)
(OUT/'support_layer_witnesses.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports),flush=True)
