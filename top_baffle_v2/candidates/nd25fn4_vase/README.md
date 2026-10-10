# Dayton ND25FN-4 vases — approved H2C inputs

Two printed-waveguide uppers for the Dayton ND25FN-4. Each carries a front tweeter (T) on the listener side and a
rear tweeter (t) firing backwards, next to the MU10 midrange. They are the shelf's `dayton_nd25fn4` family
(`to_print/stock/upper/`, `to_print/obiwan/upper/` and `to_print/obiwan/wings/`). They replace the retained V4 fused body, its cap/retainer plate and its wings,
which are archived under `build/h2c/experiments/retired_dayton_nd25fn4_v4/` (sources remain in
`candidates/nd25fn4_crescent/`).

| Product | Parts on the shelf | Mates with |
|---|---|---|
| **Obiwan** (either stand state) | vase top, front-tweeter cap, rear-tweeter cap, flat or graded wings (L + R) | the regular Obiwan keyed LM bottom of either stand state (released `build/{floor_stand,no_floor_stand}/stl/obiwan_optional_lm_keyed_1_of_2_bottom.stl`, on the shelf as `to_print/obiwan/lm/obiwan_lm_keyed_bottom_{floor_stand,no_floor_stand}`) |
| **Stock** (either stand state) | upper, front-tweeter cap, rear-tweeter cap | the H2C Stock LM (`h2c_stock_lm_no_floor_stand`, or `h2c_stock_lm_lower_floor_stand` + `h2c_stock_lm_upper_floor_stand`) |

Each product has two hands, `tweeters_right` (as built first) and `tweeters_left`. The left hand is built in the
x-mirrored frame and mirrored back, so the released parts it mates stay exactly as released. Inside that frame:
- the released inputs are mirrored;
- the ND25FN-4 model is mirrored in its own local z, so the key rings, caps and tab pockets fit the real tweeter;
- the MU10 pilots are drilled at 180 − a, so they land back on 58/148/238/328°;
- the ducts are rerouted: `LX_HAND=mirror` in `obiwan/source/build_v12.py`, and new lanes plus the H2C pocket fill
  before the vestibule cut in the Stock `conform.py`.

Status: digital checks pass; no physical print yet.

## Layout

```
obiwan/ STL/*.stl + *.print.json      approved meshes in the installed frame (identity print authority)
        support_blockers/*.json|.stl  top's duct contract + blocker (shelf dilates it by 0.8 mm)
        magnets.catalog.json          LM wing magnet sites (upper pair moved to 9/171 deg), from magnet_catalog.py
        build_package.py              packages the approved meshes, derives blocker + contract (run on Osado)
        source/                       concept generator scripts and their JSON state (V11 + V12 changes)
stock/  STL/, support_blockers/, build_package.py   (H2C seam-B pin joint applied here)
        source/                       vase25 scripts and their JSON state
```

Rear-tweeter caps are stored in a cap frame (installed pose rotated 180° about X; `installed_to_source_matrix` in
their `.print.json`). The shelf's front-down pose then puts every visible cap skin on top instead of on supports.

## Obiwan vase top (concept V13 = V11 + owner fixes)

- The top contains all of the released keyed LM top below y 278 (checked: nothing removed; identical
  within y 172.48–180). The released keyed LM top is the same part in both stand states (within 0.032 mm), so
  the top fits either keyed LM bottom and the released LM key, seam, ducts, seats and inserts carry over. Only the
  rear gets new material: a C2 taper from the vase down each side of the ring to y 180 (`source/ring_blend_v12.py`). V11 ended the
  vase depth in a 6 mm cliff onto the 0.85 mm L22 membrane at y 278. Above y 278 the top is rebuilt: MU10 flush seat,
  T and t waveguides, razor edge, buried D6 T duct, D6 T→t link, D8.2 MU10 lead. V12 (`source/build_v12.py`)
  reroutes the MU10 lead under the T duct (≥ 7.9 mm apart) and out of the rear at the MU10 rim at 285°, 14 mm from
  the terminal strip at 283°. V11 left it behind the T cluster, 46 mm away.
- ND25FN-4 placement uses the vendor STL (`vendor/DAYTON/ND25FN-4`), re-centred on its rim/lug circle
  (XF 6.17). A 4-fold key ring in each R21.8 bore locks tweeter rotation using the four front-face lugs. The tab
  midpoint is clocked to the duct mouth, so both solder runs are equal.
- The LM wing magnets move from 64/116° to 9/171° on the LM ring, and the wings move theirs by the same rotation.
  V11/V12 used 40/140°, but those stations sit over the released T and MU10 lumens. The magnets' 45° closing roofs
  reached 0.65/1.10 mm into the lumens. Backing blocks unioned after the duct cut had partly filled the lumens and
  hidden this. The owner saw the strangled duct.
  - 9/171° is the highest ring position with ≥ 1.5 mm to both lumens, where they dive behind the L22 insert pads
    (`source/magscan.py`). The backing blocks are gone.
- V13 also re-cuts the released lumens along their own polylines through the y 262–282 junction. Since V11, the vase
  ducts' burial shells had partly filled them there.
- V13 lengthens the T lead's descent into T's wiring chamber: it was 13.5 mm over 8 mm (59°), and is now ≤ 27° on the
  right hand (a spiral round T's west side, entry 140°) and ≤ 34° on the left.
- The ring blend's add-on top is back to rear + 0.4 mm. A local-max top had reached into the lumens.
  `key_ring_v11.py` originally kept only the largest shell and dropped these two buried cavities. It now carries
  them through.
- Caps: twist-lock, screwless, no holes. Each has an R10 printed boss and a felt pad to preload the motor, plus
  1 mm tab-tip pockets swept over the twist.
- Support blocker: the released keyed-top blocker below y 278, the vase ducts (+0.25 mm), and the insert bores
  (4 × MU10 M3, 2 × L22 M5). Duct blockers stop 1.3 mm short of the MU10 and L22 seat ceilings, so the ceilings
  keep their support.

## Stock vase (concept vase25)

- The vase24 body was recovered by replaying the concept transcript on Osado (`source/replay_osado.py`; the
  vase..vase24 chain itself is not in this repository). The V11 lessons were then applied: vendor model, measured
  land (T 10.89 / t −9.99), key rings, re-derived caps, and a lane mouth midway between the tabs.
- Seam B: the concept used the P2S dovetail pockets. `build_package.py` fills both pockets and adds the two D3
  pins from the released H2C Stock upper. Checks: identical to the released upper inside both pocket boxes, zero
  overlap with both H2C Stock LM states, M3 clamp bore unchanged, and the cable vestibule contains the LM's D6
  duct mouth. The P2S dovetail version stays in the concept scratch.
- The vase has no stand-off magnets: its outline differs from B2, so the Stock shoulder/B1 wings do not fit.

## Rebuild

Heavy mesh work runs on Osado under the memory-capped wrapper:

```
# Obiwan V12 top: build_v12.py DIR 46 obiwan_v12_top_prering -> ring_blend_v12.py -> key_ring_v12.py (V11 caps/wings reused:
# front face and cap footprints are unchanged). Then package (inputs: top/caps/wings/build json + released keyed-top blocker)
py obiwan/build_package.py OUT TOP CAP_FRONT CAP_REAR BUILD_JSON RELEASED_TOP_BLOCKER_JSON WING_DIR
python obiwan/magnet_catalog.py            # light, local
# Stock (inputs: vase25 body/caps/c7.json + H2C Stock upper/LM meshes and the upper's blocker)
py stock/build_package.py OUT VASE_DIR H2C_STL_DIR H2C_BLOCKER_DIR
```

Then run `make -f Makefile.h2c h2c_geometry h2c_prepare h2c_validate` locally. The interface checks are in
`scripts/verify_h2c_obiwan_interfaces.py`.
