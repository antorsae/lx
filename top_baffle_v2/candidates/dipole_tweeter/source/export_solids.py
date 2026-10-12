import os,sys;sys.argv=[sys.argv[0]]
sys.path[:0]=['scripts']
import build_dipole_tweeter as b
pl=b.pl
m=pl.um_t_mesh(b.released_um_installed(),b.tessellate,b.released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand'),stage=os.environ.get('STAGE','pre_t'))
out=os.environ.get('SC','/tmp/')+os.environ.get('OUT_PREFIX','solid')+f'_{pl.DRIVER_KEY}_{pl.ORIENTATION}'+os.environ.get('RAISE_TAG','')+'.stl'
m.export(out);print('wrote',out,round(m.volume/1e3,1),m.is_watertight)
