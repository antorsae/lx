from pathlib import Path
import re,json,hashlib
from collections import defaultdict,Counter
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
NUM=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
def scan(path):
 tool=None;feature='';z=None;rel=True;prev_e=0.;rows=defaultdict(Counter);n=0
 for n,raw in enumerate(path.open(),1):
  s=raw.split(';',1)[0].strip()
  if raw.startswith('; Z_HEIGHT:'):z=float(raw.split(':',1)[1])
  if raw.startswith('; FEATURE:'):feature=raw.split(':',1)[1].strip()
  if re.fullmatch(r'T\d+',s):tool=int(s[1:])
  if s=='M83':rel=True
  if s=='M82':rel=False
  m=re.search(r'(?:^|\s)E('+NUM+r')',s)
  if s.startswith('G92') and m:prev_e=float(m[1])
  if not re.match(r'^G[123](?:\s|$)',s) or not m:continue
  e=float(m[1]);delta=e if rel else e-prev_e;prev_e=e
  if delta<=0 or not re.search(r'\b[XY]'+NUM,s):continue
  if not feature or feature.lower() in ['custom','undefined'] or 'tower' in feature.lower() or 'purge' in feature.lower():continue
  kind='interface' if feature.lower().startswith('support interface') else 'support' if feature.lower().startswith('support') else 'model'
  rows[z][f'{kind}_T{tool}']+=1
 counts=sum(rows.values(),Counter())
 return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),extrusion_move_counts=dict(counts),layers=[dict(z=k,counts=dict(v)) for k,v in rows.items() if any('interface' in t for t in v)],num_lines=n)
if __name__ == '__main__':
 paths={'floor':OUT/'floor_slice/plate_1.gcode','um':OUT/'um_slice/plate_1.gcode','body':ROOT/'review/nd25fn4_print/body/plate_1.gcode','caps':ROOT/'review/nd25fn4_print/accessories/plate_1.gcode'}
 reports={}
 for k,p in paths.items():
  if not p.exists():continue
  r=scan(p);reports[k]=r;print(k,r['extrusion_move_counts']);print('interfaces',[(x['z'],{k:v for k,v in x['counts'].items() if 'interface' in k}) for x in r['layers']][:80],flush=True)
 (OUT/'support_materials.json').write_text(json.dumps(reports,indent=2)+'\n')
