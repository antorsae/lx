#!/bin/bash
set -eu
W=~/temp/claude-vase-recovery/stockm; cd $W; mkdir -p final
PY=~/temp/claude-vase-recovery/py; export LX_HAND=mirror LX_H2C_UPPER=$W/h2c_stock_upper_design.stl
flt() { grep -v -i "warn\|center_mass\|denominator\|barycentric\|^$" || true; }
SRC=~/temp/claude-vase-recovery/session/scratchpad/vase25
cp $SRC/{_body_block.stl,_body_nolane.stl,front_surface.npz,rear_surface.npz} .
$PY mirror_io_v12.py tweeter $SRC/nd25fn4_vendor_rimcentred.stl nd25fn4_vendor_rimcentred.stl
echo "== lanes"; $PY mesh_lanes.py $W 2>&1 | flt | tail -2
echo "== conform"; $PY conform.py $W 2>&1 | flt | tail -4
echo "== placement"; $PY assemble25.py $W 2>&1 | flt
echo "== key ring"; $PY key_ring25.py $W 2>&1 | flt | tail -2
echo "== key check"; $PY key_check25.py $W 2>&1 | flt | tail -6
echo "== caps"; $PY caps25.py $W 2>&1 | flt | grep -v "^   " | tail -6
echo "== cap check"; $PY cap_check25.py $W 2>&1 | flt | tail -8
echo "== mirror back"
$PY mirror_io_v12.py stl um_vase_nd25fn4_c25_body.stl final/um_vase_nd25fn4_c25m_body.stl
$PY mirror_io_v12.py stl cap_front_tweeter_REAR_FACE.stl final/cap_front_tweeter_REAR_FACE.stl
$PY mirror_io_v12.py stl cap_rear_tweeter_FRONT_FACE.stl final/cap_rear_tweeter_FRONT_FACE.stl
echo ALLDONE
