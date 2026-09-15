from pathlib import Path
import sys,json,subprocess,zipfile
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),str(ROOT/'candidates/nd25fn4_crescent')]
import build_h2c_release as b
from print_magnets import magnet_geometry,discover_specs
from captive_wall_audit import audit_captive_walls
from audit_h2c_print import deposition

def main():
 angle=float(sys.argv[1]);sequence=sys.argv[2] if len(sys.argv)>2 else 'inner wall/outer wall'
 directory=ROOT/'build/h2c/experiments'/f'wing_arachne_{angle}_{sequence.split()[0]}'
 directory.mkdir(parents=True,exist_ok=True);b.OUT=directory
 row=next(r for r in b.inventory() if r['name']=='h2c_v4_wing_flat_right');row['angle']=angle
 groups=b.export_posed(row);lane='petg_gf_pla';profile=ROOT/'build/h2c/profiles'/lane
 bundle=[json.loads((profile/'machine.json').read_text()),json.loads((profile/'process.json').read_text()),[json.loads((profile/'model.json').read_text()),json.loads((profile/'interface.json').read_text())]]
 settings=b.project_settings(bundle,lane,row['role'],brim=row['brim_mm'],tower=row['tower']);settings.update(wall_generator='arachne',wall_sequence=sequence)
 project=directory/'trial.3mf';b.write_project(project,groups,settings,row['offset']);b.write_json(directory/'process.json',b.job_process(bundle,settings))
 cmd=b.slice_command(project,lane,directory)
 with (directory/'slice.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 with zipfile.ZipFile(directory/'discovery.gcode.3mf') as z:(directory/'plate_1.gcode').write_bytes(z.read('Metadata/plate_1.gcode'))
 stl=ROOT/row['stl'];specs=magnet_geometry(stl,json.loads(stl.with_suffix('.print.json').read_text()),row['offset'],owner='wing')
 for i,s in enumerate(specs):
  if s['name']=='preserved_LM':s['name']+=f'_{i}'
 records=discover_specs(specs,directory/'plate_1.gcode');b.write_json(directory/'discovery.json',records)
 status=all(r['magnet_below_resume_plane'] and r['opening_clear_through_prior_layers'] for r in records)
 print('TRIAL',angle,sequence,status,[(r['name'],r['first_obstructing_layer_z_mm'],r['seated_top_z_mm']) for r in records],flush=True)
 if status:
  b.write_json(directory/'wall_audit.json',audit_captive_walls(stl,row['offset'],records,directory/'plate_1.gcode',diameters=(5,6)))
  b.write_json(directory/'deposition.json',deposition(directory/'plate_1.gcode'))
  print('QUALIFIED',angle,sequence,flush=True)
if __name__=='__main__':main()
