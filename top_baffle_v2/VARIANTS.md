# Variant catalog

The baffle prints as four pieces joined at three seams: **piece_bottom**
(1of4, carries the fused stand foot or the bridge holes),
**piece_mid_left / piece_mid_right** (2-3of4) — together the "LM
section" — and **piece_top** (4of4, the "vase": 10F + tweeter pair).
Every variant below shares ONE routing (round-4 front-datum: LM O8.2 @
z=12.55, UM O7.8 @ 12.55, shared T O6.0 @ 11.5 + strip feeders), one
seam system (A: y=120 keys +-63; B: y=315.95 keys -19/+28; C: x=-5.6),
and one fastener set — that is what makes the pieces interchangeable.
All exist in `floor_stand/` and `no_floor_stand/` builds.

## Base variants

| Variant | Replaces | Geometry | STLs |
|---|---|---|---|
| **B2** | (baseline, all 4) | Full 18.3 everywhere. Constant-wall mini-vase (walls tangent to r=50.83 about the UM). | `lx521_top_base_1..4of4` |
| **C7** | bottom + mids (+B2 vase) | LM knife taper: REAR-side smoothstep 18.3 -> 0.5 over 19 mm from the flank/chamfer edges; recovery lands at both seams; full bottom strip. Front plane intact. | `lx521_top_c7base_1..4of4` |
| **V0** | vase | Rear knife band: REAR-side 18.3 -> 0.5 over the last 2.8 mm of the vase outline (same sculpted side as C7); front intact. | `lx521_top_v0_4of4_vase` |
| **V1** | vase | Thin FLUSH vase: 11.5 (material z 6.8..18.3). Crescent re-derived (4.0 clamp seat at stock z); tweeter septum 11.5 (shorter standoffs, pair spacing -6.8); one shared front plane. | `lx521_top_v1_4of4_vase` |
| **V1L** | bottom + mids | Thin FLUSH LM section: 11.5 (z 6.8..18.3 -- SAME plane as the V1 vase: no seam-B step), smoothstep ramp y=78..96 to the full strip. O8.2 LM duct is the 11.5 binder (snug 2x2.5 fishing). | `lx521_top_v1l_1..3of4` (its `--variant v1l` export bundles the V1 vase = the complete ~12 mm baffle) |

**V1 vs V1L:** V1 = thin TOP piece; V1L = thin BOTTOM+MIDS. Pair them
for the full front-flush thin baffle; either also works alone on the
other family's pieces (see matrix).

## Add-ons (outline experiments)

| Family | Pieces | Fits | Anchoring |
|---|---|---|---|
| **A-comp shoulders** (18.3) | 4: `addonA_1..4of4` | B2 vase only | 2 pin magnets/side: flare wall zc=5.0 + crescent arc zc=10.7; outline kinks register |
| **B1 wings** (18.3) | 2: `addonB1_1..2of2` | B2 vase only | same two sites |
| **V1 A-shoulders** (11.5) | 4: `v1addonA_*` | V1 vase (V1L sets) | TWO pins/side: lower zc=12.5, upper zc=14.4 (in-wall, no bosses) |
| **V1 B1-wings** (11.5) | 2: `v1addonB1_*` | V1 vase (V1L sets) | two pins/side: lower zc=12.5, upper zc=14.4 (in-wall, no bosses) |
| V0 scarf family | (designed, not built) | V0 | would scarf onto the knife band; pending |

B2 addons on V0/V1: NO (knife/thin walls — no receiver seats).
V1 addons on B2/V0: NO (the zc=12.5/14.4 pockets exist only on the V1 vase).

## Compatibility matrix (bottom+mids x vase)

Any of {B2, C7, V1L} bottom+mids joins any of {B2, V0, V1} vase — the
front plane is always continuous at z=18.3 and the seams/keys/ducts
are identical. Rear-side steps at seam B land on the hidden side:

| bottom+mids \ vase | B2 (18.3) | V0 (knife band) | V1 (11.5) |
|---|---|---|---|
| **B2** (18.3) | stock reference | edge experiment | vase-thickness experiment (rear step 6.8) |
| **C7** (LM knife) | LM-edge experiment | full knife-edge baffle | knife LM + thin vase |
| **V1L** (11.5) | thin LM only (vase protrudes 6.8 rearward) | thin LM + knife vase | **complete UNIFORM 11.5 front-flush baffle** |

Notes: mixed-thickness key joints mate on the thinner piece's depth
(the through-pockets leave a shallow open notch on the hidden rear —
cosmetic). Tweeter through-bolt length follows the vase septum (18.3:
~M4x35; 11.5: ~M4x30).

## Common hardware (all variants)

* W22: 6x M5 x 6 x O7 heat-sets (bore O6.4 x 7.0).
* 10F: 4x M3 x 3 x O5 heat-sets (bore O4.6 x 4.0), pattern clocked to
  (58/148/238/328) — a square rotated +13 deg from 45; 45/90 grids are
  geometrically impossible (the notch dive crosses the pilot ring at
  azimuth ~137 deg; 180/270 sit on the lane/UM tail).
* Tweeter pair: M4 through-bolts + nyloc, clamping the crescent.
* Magnets: D5 x 2 N52 pins everywhere (base pockets 1.0 deep, magnet
  1.0 proud; receivers 3.2 deep in the attachments).

See PRINTING.md for print settings and torques; `make check` guards
clearances, seam keys, route smoothness, and cutter health.
