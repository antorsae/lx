import os
import sys;sys.argv=[sys.argv[0]];sys.path[:0]=['scripts']
import build_dipole_tweeter as b
pl=b.pl
m=pl.um_t_above_mesh(b.released_um_installed(),b.tessellate,solid_only=True)
out=os.environ.get('SC','/tmp/')+f'above_solid_{pl.DRIVER_KEY}_{pl.ORIENTATION}'+os.environ.get('RAISE_TAG','')+'.stl'
m.export(out);print('wrote',out,round(m.volume/1e3,1),'bodies',len(m.split(only_watertight=False)))
