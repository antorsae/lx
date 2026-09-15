"""Geometric follow-up: joint envelopes and optional permanent wing layouts."""
from pathlib import Path
import hashlib,json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon,box
from shapely.ops import unary_union
from shapely.affinity import rotate,translate,scale

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
REPORT=json.loads((OUT/'measurements.json').read_text());PARTS=REPORT['parts']
ANGLES=np.arange(0,180,.1)
C=np.cos(np.deg2rad(ANGLES))[:,None];S=np.sin(np.deg2rad(ANGLES))[:,None]
BEDS={'shared':(300,320),'right_H2C':(305,320),'left_both_or_right_H2D':(325,320)}
CACHE={};CASES={}

def hull(paths):
 polygons=[]
 for relative in paths:
  if relative not in CACHE:
   p=ROOT/relative;a=json.loads(p.with_suffix('.print.json').read_text())
   assert hashlib.sha256(p.read_bytes()).hexdigest()==a['stl_sha256']
   m=trimesh.load_mesh(p,process=False)
   v=trimesh.transform_points(m.vertices,np.linalg.inv(a['source_to_stl_matrix']))
   xy=v[:,:2];CACHE[relative]=Polygon(xy[ConvexHull(xy).vertices])
  polygons.append(CACHE[relative])
 return unary_union(polygons).convex_hull

def current(name):
 return hull(PARTS[name]['sources'])

def fit(shape,bed,brim=5.,edge=2.):
 p=np.asarray(shape.convex_hull.exterior.coords)[:-1]*[1,-1]
 x=C*p[:,0]-S*p[:,1];y=S*p[:,0]+C*p[:,1]
 spans=np.column_stack([np.ptp(x,axis=1),np.ptp(y,axis=1)])
 margins=np.min((np.array(bed)-spans)/2,axis=1)-brim-edge
 k=int(margins.argmax())
 return {'angle_deg':float(ANGLES[k]),'part_xy_mm':spans[k].tolist(),
         'with_brim_xy_mm':(spans[k]+2*brim).tolist(),
         'remaining_edge_margin_mm':float(margins[k]),'fit':bool(margins[k]>=0)}

def add(name,shape,note):
 b=shape.bounds
 CASES[name]={'note':note,'installed_bbox_xy_mm':list(b),
  'installed_dimensions_xy_mm':[b[2]-b[0],b[3]-b[1]],
  'outline_xy_mm':list(map(list,shape.convex_hull.exterior.coords)),
  'fits':{f'{bed}_brim{brim}':fit(shape,size,brim=brim) for bed,size in BEDS.items() for brim in [2,3,5]}}

for family in ['stock','slim']:
 for state in ['no_floor_stand','floor_stand']:
  lm=current(f'{family}_{state}_LM_merged')
  reverse=lm.intersection(box(-1000,-1000,1000,315.95))
  add(f'{family}_{state}_inward_top_joint',reverse,
      'Replace 6 mm upward keys with a joint contained in the Y315.95 envelope; same outside silhouette and driver coordinates. Do not simply flip existing keys into the driver seat: registration, rear lap and cable details need redesign.')
  add(f'{family}_{state}_joint_y311',lm.intersection(box(-1000,-1000,1000,311.0)),
      'Envelope only; NOT recommended: clips the R110.6 driver seat whose top is Y311.581.')

# Maximum retained half-width with top joint at the current Y315.95 and
# no upward protrusions. This is a local outline clip, never a scale transform.
thresholds={}
shape=current('stock_no_floor_stand_LM_merged').intersection(box(-1000,-1000,1000,315.95))
for bed,size in BEDS.items():
 for brim in [2,3,5]:
  low,high=110.6,152.40101
  if fit(shape,size,brim)['fit']:
   half=high
  elif not fit(shape.intersection(box(-low,-1000,low,1000)),size,brim)['fit']:
   thresholds[f'{bed}_brim{brim}']={'fit_without_cutting_driver_seat':False};continue
  else:
   for _ in range(24):
    mid=(low+high)/2
    if fit(shape.intersection(box(-mid,-1000,mid,1000)),size,brim)['fit']:low=mid
    else:high=mid
   half=low
  clipped=shape.intersection(box(-half,-1000,half,1000))
  thresholds[f'{bed}_brim{brim}']={'maximum_retained_width_mm':2*half,'trim_per_side_mm':152.40101-half,'fit':fit(clipped,size,brim)}

# Restored wing segments can be clipped at an alternative carrier-relative
# split. This checks only envelopes; it does not prove continuity or access.
fusion_sweeps={}
for upper_kind in ['regular','V4']:
 for state in ['no_floor_stand','floor_stand']:
  lm=current(f'obiwan_{state}_LM')
  upper=current(f'obiwan_{state}_upper') if upper_kind=='regular' else current('V4_body')
  prefix='obiwan' if upper_kind=='regular' else 'V4'
  wings=[current(f'{prefix}_flat_wing_{side}_whole') for side in ['left','right']]
  add(f'{upper_kind}_{state}_upper_with_full_wings',unary_union([upper,*wings]).convex_hull,
      'Whole wings fused to upper, without a new wing split; removes detachability.')
  rows=[]
  for seam in np.arange(170,325.1,1):
   lower_wings=[w.intersection(box(-1000,-1000,1000,float(seam))) for w in wings]
   upper_wings=[w.intersection(box(-1000,float(seam),1000,1000)) for w in wings]
   lower_shape=unary_union([lm,*lower_wings]).convex_hull
   upper_shape=unary_union([upper,*upper_wings]).convex_hull
   lf=fit(lower_shape,BEDS['shared']);uf=fit(upper_shape,BEDS['shared'])
   rows.append({'wing_seam_y_mm':float(seam),'lower':lf,'upper':uf,'both_fit':lf['fit'] and uf['fit'],'minimum_margin_mm':min(lf['remaining_edge_margin_mm'],uf['remaining_edge_margin_mm'])})
  best=max(rows,key=lambda row:row['minimum_margin_mm']);fusion_sweeps[f'{upper_kind}_{state}']={'best':best,'feasible_seam_y_mm':[r['wing_seam_y_mm'] for r in rows if r['both_fit']]}
  seam=best['wing_seam_y_mm']
  for owner,carrier,bounds in [('lower',lm,(-1000,seam)),('upper',upper,(seam,1000))]:
   shape=unary_union([carrier,*[w.intersection(box(-1000,bounds[0],1000,bounds[1])) for w in wings]]).convex_hull
   add(f'{upper_kind}_{state}_fused_wings_{owner}_y{int(seam)}',shape,
       'Permanent wing regions incorporated into their carrier; separate wing seam and all joins need CAD design. Full driver spacing retained.')
  # A common cut keeps each upper module identical across stand variants.
  seam={'regular':222.,'V4':257.}[upper_kind]
  for owner,carrier,bounds in [('lower',lm,(-1000,seam)),('upper',upper,(seam,1000))]:
   shape=unary_union([carrier,*[w.intersection(box(-1000,bounds[0],1000,bounds[1])) for w in wings]]).convex_hull
   add(f'{upper_kind}_{state}_common_fused_wings_{owner}_y{int(seam)}',shape,
       'Common wing cut for both stand states; upper body envelope stays identical. Permanent wings require new attachment and seam design, with driver spacing retained.')

# Optimistic lower bound: can even the two protected circular driver-seat
# envelopes fit if everything around them is removed? Existing spacing stays.
from shapely.geometry import Point
driver_pair=unary_union([Point(0,200.981).buffer(110.6,quad_segs=128),Point(0,366.081).buffer(49.3,quad_segs=128)]).convex_hull
add('two_seat_envelopes_only',driver_pair,'Optimistic envelope only: excludes stand, collar walls, joints and tweeter. Must not be confused with an actual printable part.')

reserves={}
selected={k:('left_both_or_right_H2D_brim2',325,2) for k in CASES if k.endswith('_inward_top_joint')}
selected.update({k:('shared_brim5',300,5) for k in CASES if k.startswith('regular_') and '_common_fused_wings_' in k})
selected.update({k:('left_both_or_right_H2D_brim2',325,2) for k in CASES if k.startswith('V4_') and '_common_fused_wings_' in k})
for name,(profile,width,brim) in selected.items():
 a=CASES[name];f=a['fits'][profile];assert f['fit'],name
 poly=rotate(Polygon(a['outline_xy_mm']),-f['angle_deg'],origin=(0,0))
 poly=scale(poly,yfact=-1,origin=(0,0));b=poly.bounds
 offset=((width-(b[2]-b[0]))/2-b[0],(320-(b[3]-b[1]))/2-b[1])
 poly=translate(poly,*offset);expanded=poly.buffer(brim)
 assert box(2,2,width-2,318).covers(expanded),name
 candidates=[]
 # X25..325 is the physical shared area. The 300-wide layout uses local X0.
 for x in range(27 if width==325 else 2,width-50-2+1,2):
  for y in range(2,320-50-2+1,2):
   d=box(x,y,x+50,y+50).distance(expanded)
   if d>=5:candidates.append((d,x,y))
 witness=max(candidates) if candidates else None
 reserves[name]={'profile':profile,'part_and_brim_inside_edge':True,
  'tower_50mm_inside_shared_area':bool(witness),
  'tower_distance_x_y_mm':list(witness) if witness else None,
  'angle_deg':f['angle_deg'],'translation_xy_mm':list(offset),
  'support_reach_validated':False}

output={'purpose':'Read-only purchasing refinement; no printable CAD generated.',
 'assumptions':{'edge_mm':2,'rotation_step_deg':.1,'driver_spacing_mm':165.1,'driver_seat_edge_gap_mm':165.1-110.6-49.3,'support_and_prime_tower':'Not included in this first envelope sweep.'},
 'cases':CASES,'stock_outline_trim_thresholds':thresholds,'permanent_wing_seam_sweeps':fusion_sweeps,
 'common_wing_seam_y_mm':{'regular':222,'V4':257},'layout_reserve_checks':reserves}
(OUT/'adjustments.json').write_text(json.dumps(output,indent=2)+'\n')
for k,v in thresholds.items():print('TRIM',k,v)
for k,v in fusion_sweeps.items():print('FUSION',k,v)
for k in ['stock_no_floor_stand_inward_top_joint','two_seat_envelopes_only']:
 print('CASE',k,CASES[k]['fits'])
