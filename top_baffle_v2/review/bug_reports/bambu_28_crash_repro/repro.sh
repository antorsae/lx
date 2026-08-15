#!/bin/sh
# Bambu Studio 02.08.02.60 CLI: segmentation fault when slicing a large model
# with TWO filaments and supports enabled. Same inputs with one filament, or
# with supports disabled, or with a smaller model, all succeed.
BIN="/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "== 1. two filaments, supports ON  -> expect SEGFAULT (139)"
"$BIN" --debug 2 --slice 0 --arrange 1 --orient 0 --allow-rotations=0 \
  --export-3mf out.3mf \
  --load-settings "$HERE/machine.json;$HERE/process.json" \
  --load-filaments "$HERE/filament_1_petg_cf.json;$HERE/filament_2_pla_basic.json" \
  --outputdir "$HERE/out_crash" "$HERE/model.stl" >"$HERE/out_crash.log" 2>&1
echo "   exit=$?"

echo "== 2. same, ONE filament            -> expect 0"
"$BIN" --debug 2 --slice 0 --arrange 1 --orient 0 --allow-rotations=0 \
  --export-3mf out.3mf \
  --load-settings "$HERE/machine.json;$HERE/process.json" \
  --load-filaments "$HERE/filament_1_petg_cf.json" \
  --outputdir "$HERE/out_one" "$HERE/model.stl" >"$HERE/out_one.log" 2>&1
echo "   exit=$?"
