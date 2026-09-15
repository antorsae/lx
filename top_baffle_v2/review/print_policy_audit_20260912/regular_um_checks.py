from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import artifact_emit as emit
import slice_captive_magnets as captive
import build_obiwan_combo_plate as combo
from support_materials import scan
report={'scope':'Attempt the standard release-catalog provenance gate before current UM cavity validation.'}
try:
 cat=captive.normalize_catalog(ROOT/'review/captive_magnet_release_catalog.json')
 report['catalog_status']='pass'
except Exception as e:
 report.update(catalog_status='failed',error=str(e))
print(json.dumps(report,indent=2),flush=True)
(OUT/'regular_um_magnet_checks.json').write_text(json.dumps(report,indent=2)+'\n')
variant=OUT/'um_variant_slice/plate_1.gcode'
if variant.exists():
 v=scan(variant);(OUT/'um_variant_materials.json').write_text(json.dumps(v,indent=2)+'\n');print('Variant-only material paths',v['extrusion_move_counts'])
