"""Export the integrated part at a search stage: STAGE=pre_ducts (-> isolid_<key>_um) | pre_t (-> isolid_<key>_t)."""
import os, sys
os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'; os.environ.setdefault('LX_ROUTING_PROFILE', 'obiwan')
sys.argv = [sys.argv[0]]; sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
import build_dipole_tweeter as b
import lx521_baffle.obiwan.lm_um_t as L
stage = os.environ.get('STAGE', 'pre_ducts')
m = L.lm_um_t_mesh(b.released_installed('h2c_obiwan_lm_keyed_top'), b.released_um_installed(), b.tessellate,
                   b.released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand'), stage=stage)
key = f'{L.d.DRIVER_KEY}_{L.d.ORIENTATION}'
out = os.environ.get('SC', '/tmp/') + f"isolid_{key}_{'um' if stage == 'pre_ducts' else 't'}{os.environ.get('RAISE_TAG', '')}.stl"
m.export(out); print('wrote', out, round(m.volume / 1e3, 1), m.is_watertight, m.bounds.round(1).tolist())
