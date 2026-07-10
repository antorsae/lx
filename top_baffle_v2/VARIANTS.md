# Variant catalog

The baffle prints as four pieces joined at three seams: **piece_bottom**
(1of4, carries the fused stand foot or the bridge holes),
**piece_mid_left / piece_mid_right** (2-3of4) — together the "LM
section" — and **piece_top** (4of4, the "vase": 10F + tweeter pair).
Every variant below shares ONE routing (round-5 flush-ready: LM O8.2 @
z=12.55 ending y=84, UM O7.8 @ 12.55 riding its r=119.5 arc to a rear
exit at (78.4, 291), shared T @ 11.5 up the r=116.5 arc — O6.0 except
a W6.6 x H4.4 oval at zc=10.45 through the vase, where it passes UNDER
the MU10 flange seat + strip feeders), one seam system (A: y=120 keys
+-66/+-103; B: y=315.95 keys -19/+28; C: x=-5.6), and one fastener set
— that is what makes the pieces interchangeable. All exist in
`floor_stand/` and `no_floor_stand/` builds.

Drivers (LX521.4 production): LM = SEAS **U22REX/P-SL** (H1659-08,
flange O220.6 x 6.0 measured); UM = SEAS **MU10RB-SL** (H1658-04,
flange O98 x 4.0 measured). Older comments naming the LX521 prototype
drivers (W22EX001 / 10F) refer to the same cutout/pilot geometry.

## Base variants

| Variant | Replaces | Geometry | STLs |
|---|---|---|---|
| **B2** | (baseline, all 4) | Full 18.3 everywhere. Constant-wall mini-vase (walls tangent to r=50.83 about the UM). | `lx521_top_base_1..4of4` |
| **C7** | bottom + mids (+B2 vase) | LM knife taper: REAR-side smoothstep 18.3 -> 0.5 over 19 mm from the flank/chamfer edges; recovery lands at both seams; full bottom strip. Front plane intact. | `lx521_top_c7base_1..4of4` |
| **V0** | vase | Rear knife band: REAR-side 18.3 -> 0.5 over the last 2.8 mm of the vase outline (same sculpted side as C7); front intact. | `lx521_top_v0_4of4_vase` |
| **V1** | vase | Thin FLUSH vase: 11.5 (material z 6.8..18.3). Crescent re-derived (4.0 clamp seat at stock z); tweeter septum 11.5 (shorter standoffs, pair spacing -6.8); one shared front plane. | `lx521_top_v1_4of4_vase` |
| **V1L** | bottom + mids | Thin FLUSH LM section: 11.5 (z 6.8..18.3 -- SAME plane as the V1 vase: no seam-B step), smoothstep ramp y=78..96 to the full strip. O8.2 LM duct is the 11.5 binder (snug 2x2.5 fishing). | `lx521_top_v1l_1..3of4` (its `--variant v1l` export bundles the V1 vase = the complete ~12 mm baffle) |
| **V1LF** | all 4 | **V1L + FLUSH drivers**: front flange recesses (U22 O221.2 x 6.0, MU10 O98.6 x 4.0 — depths = owner-measured flange thicknesses, `RECESS_CLR_MM` 0.6) so both mids sit dead-level with the front plane. Insert bores re-cut from the seats at 6.2 (the owner's M5 x 5.8 inserts + 0.4 settle; use **M5 x 12** screws here); six straight O9.6 pad buttons on the rear (1.5 proud, chamfered rim, uncut plate material, concentric with the bores) restore the U22 insert stack -- irreducible: the 5.5 wall under the seat cannot swallow a 5.8 insert. Seam-A inner keys and the seat floor share the flange band — the steel flange clamps across seam A on assembly. Seam-B keys are FLIPPED in this set (tabs hang down from the vase into mid pockets BELOW the seam, left tooth at x=-23.6 head 9): the stock up-keys straddle the MU10 seat ring and would put the flange on the dovetail joint. SL: mids "must not be recessed" — this is a deliberate experiment arm vs that voicing (DSP re-EQ per configuration). | `lx521_top_v1lf_1..4of4` |

**V1 vs V1L:** V1 = thin TOP piece; V1L = thin BOTTOM+MIDS. Pair them
for the full front-flush thin baffle; either also works alone on the
other family's pieces (see matrix).

## Add-ons (outline experiments)

| Family | Pieces | Fits | Anchoring |
|---|---|---|---|
| **A-comp shoulders** (18.3) | 4: `addonA_1..4of4` | B2 vase only | 2 FLUSH magnets/side: flare wall zc=5.0 + crescent arc zc=10.7; outline kinks register |
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

**V1LF is a complete 4-piece set** (recessed seats span seam A and
live on both mids + vase), so it does NOT mix with other families:
its seam-A inner keys are height-shaved under the U22 seat and its
pilot bores sit deeper. Everything proud-mounted above mixes freely.
Flush vs proud on the SAME V1L geometry (V1L set vs V1LF set) is the
cleanest flange-diffraction A/B this project offers — SL voiced the
LX521 with proud flanges and straight un-chamfered holes, so treat
V1L as the as-designed reference arm.

## Common hardware (all variants)

* W22: 6x M5 x 5.8 x O6.3 heat-sets (bore O6.4 x 6.8).
* 10F: 4x M3 x 3 x O5 heat-sets (bore O4.6 x 4.0), pattern clocked to
  (58/148/238/328) — a square rotated +13 deg from 45; 45/90 grids are
  geometrically impossible (the notch dive crosses the pilot ring at
  azimuth ~137 deg; 180/270 sit on the lane/UM tail).
* Tweeter pair: M4 through-bolts + nyloc, clamping the crescent.
* Magnets: D5 x 2 N52 pins everywhere (base pockets 1.0 deep, magnet
  flush; receivers also 2.0 deep -- the two magnets meet level).

See PRINTING.md for print settings and torques; `make check` guards
clearances, seam keys, route smoothness, and cutter health.
