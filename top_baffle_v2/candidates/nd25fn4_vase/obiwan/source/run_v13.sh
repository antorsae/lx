#!/bin/bash
# usage: run_v13.sh std|mirror
set -eu
HAND=$1; W=~/temp/claude-obiwan-concept/v13/$HAND; mkdir -p $W/final; cd $W
PY=~/temp/claude-vase-recovery/py; export LX_HAND=$HAND LX_REL_CONTRACT=~/temp/claude-obiwan-concept/pkg/rel/obiwan_optional_lm_keyed_2_of_2_top.support_blocker.json
flt() { grep -v -i "warn\|center_mass\|denominator\|barycentric\|^$" || true; }
cp ../src/*.py .
if [ $HAND = mirror ]; then
  $PY mirror_io_v12.py stl ~/temp/claude-obiwan-concept/v11d/released_lm_keyed_top_floor_design.stl released_lm_keyed_top_floor_design.stl >/dev/null
  $PY mirror_io_v12.py tweeter ~/temp/claude-obiwan-concept/v11d/nd25fn4_vendor_rimcentred.stl nd25fn4_vendor_rimcentred.stl >/dev/null
else
  cp ~/temp/claude-obiwan-concept/v11d/released_lm_keyed_top_floor_design.stl ~/temp/claude-obiwan-concept/v11d/nd25fn4_vendor_rimcentred.stl .
fi
V=obiwan_v13_top
echo "== build"; $PY build_v12.py $PWD 46 $V 2>&1 | flt | grep -i "V13\|UM lead\|station\|BODY\|Error\|Traceback\|assert" || true
[ -f $V.stl ] || { echo BUILD FAILED; exit 1; }
echo "== ring blend"; $PY ring_blend_v12.py $V.stl obiwan_v12_top_unkeyed.stl ring_blend_report.json 2>&1 | flt | tail -1
echo "== placement"; $PY place_tweeters_v12.py $PWD $V 2>&1 | flt
echo "== key ring"; $PY key_ring_v12.py $PWD 2>&1 | flt | tail -1
cp obiwan_v12_top.stl ${V}_keyed.stl
echo "== caps"; $PY caps_v12.py $PWD $V tweeter_placement.json 2>&1 | flt | grep -v "^   " || true
if [ $HAND = std ]; then echo "== wings"; LX_REPO_INPUTS=~/temp/claude-obiwan-concept/repo_inputs2/ $PY wings_v7.py $PWD $V 2>&1 | flt | sed 's/h2c_obiwan_wing_//' | cut -c1-230; fi
echo "== checks"
cp obiwan_v12_top.stl obiwan_v11_top.stl; $PY key_check_v11.py $PWD 2>&1 | flt | grep -i "seated overlap\|insertion\|rotated   [-+]2\|rotated +90" | head -5
cp ${V}_build.json chk_build.json; cp ${V}_keyed.stl chk.stl; cp ${V}_cap_frontT.stl chk_cap_frontT.stl; cp ${V}_cap_reart.stl chk_cap_reart.stl
$PY lock_check_m3.py $PWD chk 2>&1 | flt | grep "+16\|+22" 
$PY ductwall.py ${V}_keyed.stl ${V}_build.json 2>&1 | flt
$PY magduct.py ${V}_keyed.stl $LX_REL_CONTRACT $HAND 2>&1 | flt
echo "== to the design frame"
if [ $HAND = mirror ]; then
  for f in ${V}_keyed.stl ${V}_cap_frontT.stl ${V}_cap_reart.stl; do $PY mirror_io_v12.py stl $f final/$f >/dev/null; done
  $PY mirror_io_v12.py json ${V}_build.json final/${V}_build.json >/dev/null
else cp ${V}_keyed.stl ${V}_cap_frontT.stl ${V}_cap_reart.stl ${V}_build.json ${V}_wing_*.stl final/; fi
echo ALLDONE
