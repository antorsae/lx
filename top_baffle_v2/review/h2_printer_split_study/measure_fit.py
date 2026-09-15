"""Read-only rectangular-bed footprint screen of hash-bound project meshes."""
from pathlib import Path
import json,hashlib
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon,box

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
# The inherited Bambu profile limits the fixed left nozzle to Z320, versus
# the headline Z325/right-nozzle limit. None of the studied parts approaches it.
BEDS={'P2S':(256,256,256),'H2_dual_shared':(300,320,320),'H2C_right_Vortek':(305,320,325),'H2_fixed_left':(325,320,320)}
BRIM=5.;EDGE=2.
cache={};sources={};shapes={};results={}
angles=np.arange(0,180,.1)
cs=np.cos(np.deg2rad(angles));sn=np.sin(np.deg2rad(angles))

def load(relative):
 p=ROOT/relative
 if relative not in cache:
  authority=p.with_suffix('.print.json');a=json.loads(authority.read_text())
  sha=hashlib.sha256(p.read_bytes()).hexdigest()
  assert sha==a['stl_sha256'],p
  assert a['print_orientation'].startswith(('front_face_down','front-face-down')),a
  m=trimesh.load_mesh(p,process=False)
  v=trimesh.transform_points(m.vertices,np.linalg.inv(a['source_to_stl_matrix']))
  cache[relative]=v
  sources[relative]={'sha256':sha,'authority':str(authority.relative_to(ROOT)),'authority_sha256':hashlib.sha256(authority.read_bytes()).hexdigest(),'stl_dimensions_mm':m.extents.tolist(),'world_dimensions_mm':np.ptp(v,axis=0).tolist()}
 return cache[relative]

def add(name,paths,clip=None,note=''):
 v=np.concatenate([load(p) for p in paths]);xy=v[:,:2]
 hull=Polygon(xy[ConvexHull(xy).vertices])
 if clip is not None:hull=hull.intersection(box(-1000,clip[0],1000,clip[1]))
 installed_points=np.asarray(hull.exterior.coords)[:-1]
 points=installed_points*np.array([1,-1])  # X180 front-face-down projection
 px=cs[:,None]*points[:,0]-sn[:,None]*points[:,1]
 py=sn[:,None]*points[:,0]+cs[:,None]*points[:,1]
 spans=np.stack([np.ptp(px,axis=1),np.ptp(py,axis=1)],axis=1)
 choices={}
 for bed,(w,h,z) in BEDS.items():
  raw_margin=np.minimum((w-spans[:,0])/2,(h-spans[:,1])/2)
  k=int(raw_margin.argmax());margin=float(raw_margin[k]-BRIM-EDGE)
  choices[bed]={'angle_deg':float(angles[k]),'part_width_depth_mm':spans[k].tolist(),'with_5mm_brim_mm':(spans[k]+2*BRIM).tolist(),'raw_edge_margin_mm':float(raw_margin[k]),'remaining_margin_after_brim_and_2mm_edge_mm':margin,'bare_envelope_fits':bool(raw_margin[k]>=0 and np.ptp(v[:,2])<=z),'planning_allowances_fit':bool(margin>=0 and np.ptp(v[:,2])<=z)}
 results[name]={'sources':paths,'world_bbox_min_mm':v.min(axis=0).tolist(),'world_bbox_max_mm':v.max(axis=0).tolist(),'world_dimensions_mm':np.ptp(v,axis=0).tolist(),'screened_plan_dimensions_mm':list(np.array(hull.bounds)[2:]-np.array(hull.bounds)[:2]),'build_height_mm':float(np.ptp(v[:,2])),'height_is_unclipped_upper_bound':clip is not None,'note':note,'beds':choices,'convex_outline_xy_mm':points.tolist()}
 shapes[name]=hull

for state in ['no_floor_stand','floor_stand']:
 root=f'build/{state}/stl/'
 for family in ['stock','slim']:
  lower=[root+f'{family}_{n}_of_4_{part}.stl' for n,part in [(1,'bottom'),(2,'mid_left'),(3,'mid_right')]]
  top=root+f'{family}_4_of_4_vase_b2.stl'
  add(f'{family}_{state}_LM_merged',lower,note='Envelope of all three existing LM prints reassembled; internal joint removal would require CAD regeneration.')
  add(f'{family}_{state}_LM_mids_merged',lower[1:],note='Merge existing mid-left and mid-right; retain current lower seam and top module.')
  add(f'{family}_{state}_LM_flat_waist',lower,clip=(0,315.95),note='Concept: LM ending at seam B without upward dovetail keys. Requires a redesigned top joint.')
  add(f'{family}_{state}_LM_above_80',lower,clip=(80,1000),note='Concept: new base seam near Y80, below driver seat; joint/route clearance not designed.')
  add(f'{family}_{state}_whole',lower+[top],note='Entire baffle, excluding optional perimeter attachments.')
  add(f'{family}_{state}_upper',[top])
  add(f'{family}_{state}_bottom',[lower[0]])
 lm=root+'obiwan_core_1_of_2_lm_carrier.stl';um=root+'obiwan_core_2_of_2_um_carrier.stl';t=root+'obiwan_addon_tweeter_crescent.stl'
 add(f'obiwan_{state}_LM',[lm],note='Existing unsplit canonical LM print mesh, including state-specific bridge/floor stand.')
 add(f'obiwan_{state}_LM_UM',[lm,um])
 add(f'obiwan_{state}_whole',[lm,um,t])
 add(f'obiwan_{state}_upper',[um,t],note='Potential combined UM/tweeter envelope; serviceability and geometry still separate.')
for family in ['stock','slim']:
 for side in ['left','right']:
  n=1 if side=='left' else 2
  add(f'{family}_wing_{side}',[f'build/no_floor_stand/stl/{family}_wing_{n}_of_2_{side}.stl'])
  shoulders=sorted(str(p.relative_to(ROOT)) for p in (ROOT/'build/no_floor_stand/stl').glob(f'{family}_shoulder*_{side}.stl'))
  add(f'{family}_shoulder_{side}_merged',shoulders,note='Optional perimeter shoulder pair in its assembled frame.')
for family in ['flat','graded']:
 for side in ['left','right']:
  paths=[f'build/wings/{family}/stl/obiwan_wing_{family}_{side}_split2_{n}_of_2_{part}.stl' for n,part in [(1,'lm_lower'),(2,'lm_um_upper')]]
  add(f'obiwan_{family}_wing_{side}_whole',paths,note='Restored outline of two delivered segments; monolithic CAD exists separately.')
  v4=[paths[0],f'candidates/nd25fn4_crescent/STL/wings/V4_{family}_{side}_UPPER.stl']
  add(f'V4_{family}_wing_{side}_whole',v4,note='Current lower wing plus matched V4 upper wing, in assembled coordinates; CAD fusion still required.')
add('V4_body',['candidates/nd25fn4_crescent/STL/01_UM_Crescent_V4.stl'],note='Already one printable upper body on P2S; no size-driven consolidation gain.')
report={'status':'geometric_screen_only','assumptions':{'bed_dimensions_xyz_mm':BEDS,'brim_mm':BRIM,'edge_margin_mm':EDGE,'rotation_step_deg':.1,'orientation':'front face down; no out-of-plane tilt','prime_tower':'Not included in this envelope gate; plan separately.','support_footprint':'Not sliced. Five mm brim allowance is not proof that all generated supports fit.'},'source_files':sources,'parts':results}
(OUT/'measurements.json').write_text(json.dumps(report,indent=2)+'\n')
for name,item in results.items():
 if '_no_floor_' in name or 'wing_' in name:
  c=item['beds']['H2_dual_shared'];l=item['beds']['H2_fixed_left'];p=item['beds']['P2S']
  print(name,'raw=',[round(v,1) for v in item['screened_plan_dimensions_mm']], 'H2dual=',[round(v,1) for v in c['part_width_depth_mm']], 'dual margin=',round(c['remaining_margin_after_brim_and_2mm_edge_mm'],2),'left margin=',round(l['remaining_margin_after_brim_and_2mm_edge_mm'],2),'P2S margin=',round(p['remaining_margin_after_brim_and_2mm_edge_mm'],2))
