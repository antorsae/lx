# Dayton ND25FN-4 waveguide — Stock vase and Obiwan vase top

The **Dayton ND25FN-4 waveguide** is one of the project's five tweeter families,
alongside the Dayton ND25FW-4 face-to-face pair, the Tectonic TEBM35C10-4
BMR and the GRS PT2522-4 / Mundorf AMT17D2.1 dipoles. Two faceplate-free ND25FN-4 domes sit in printed waveguides fused with
the MU10 upper-mid seat. The front tweeter (**T**) faces the listener; the
rear tweeter (**t**), higher up, fires backwards. Each tweeter is closed from
behind by a screwless twist-lock cap whose skin is cut from the body's own
surface.

Two products carry it:

| Product | What it replaces | Printed parts per speaker |
|---|---|---|
| **Stock ND25FN-4 vase** | The Stock upper module (either stand state) | 1 vase + 2 caps, on the normal H2C Stock LM |
| **Obiwan ND25FN-4 vase top** | The LM top, the regular UM and the crescent (either stand state) | 1 keyed LM bottom (floor stand or no-floor stand) + 1 vase top + 2 caps + optional wing pair |

![Tweeter options at one scale, including both ND25FN-4 vases](../images/generated/iso/rows/tweeter_row.png)

The ND25FN-4 is a 4-ohm, 25 mm silk-dome element supplied without a
faceplate for custom mounting: Ø41 mm, 21 mm deep, Ø34 mm cutout. The ND25FW-4
mounts differently; the drivers are not interchangeable in these parts.
[Dayton Audio specifications and 3D reference](https://www.daytonaudio.com/product/1194/nd25fn-4-1-neo-silk-dome-tweeter-element-4-ohm).
The vendor STL (`vendor/DAYTON/ND25FN-4/`) positions each tweeter in the model.

## Left and right hands

The waveguides are not left/right symmetric, so each product comes in two hands. They are named by the side of the MU10
that the tweeter column sits on, seen from the front: `tweeters_right` and `tweeters_left`. For a mirror-imaged
stereo pair, print one of each. With the tweeters inboard, toward the centre, the **left** speaker takes
`tweeters_right` and the **right** speaker takes `tweeters_left`. Swap them for tweeters outboard.

- The hands mirror only the rebuilt upper. What they mate with is shared and unchanged: the Obiwan keyed LM bottom
  (key, seam, ducts, magnets) and the H2C Stock LM (pin joint, M3 clamp, single Ø6 mm cable duct).
- The ducts are rerouted per hand, so each tweeter lead still starts from the LM's duct.
- The MU10 screw pilots stay at 58/148/238/328° in both hands, so the MU10 fits the same way round.
- The Obiwan MU10 lead exits near the terminal strip in both hands.
- The ND25FN-4 itself is not symmetric. Each hand's key rings, caps and tab pockets are generated for the real
  tweeter, not mirrored from the other hand.
- **Caps, wings and bodies are hand-specific. The twist-lock caps lock in opposite directions on the two hands.**

## Stock ND25FN-4 vase

![Stock ND25FN-4 vase on the H2C Stock LM](../build/h2c/views/H2C_Dayton_ND25FN4_stock_front.png)

- Replaces `h2c_stock_upper` on the unchanged H2C Stock LM (either hand):
  `h2c_stock_lm_no_floor_stand`, or `h2c_stock_lm_lower_floor_stand` +
  `h2c_stock_lm_upper_floor_stand`.
- Seam B is the H2C contained-pin joint: two Ø3 mm pins on the vase enter the LM's Ø3.3 mm
  sockets. The hidden M3 × 20 clamp and its Hanglife insert are unchanged.
  Left of the MU10 the front is the released Stock upper; the MU10 flange is
  flush with the raised front on the tweeter side.
- MU10 at the released datum (0, 366.08). T at (±100, 350.08), t at (±100, 415.88) (+ for `tweeters_right`).
  Envelope 214.2 × 197.6 × 41.1 mm, 535.8 cm³.
- One cable lane per tweeter. Both leave the LM's single Ø6 mm duct through an
  open ramped vestibule; each lane mouth sits midway between its tweeter's two
  solder tabs.
- The Stock shoulders and B1 wings do not fit (different outline), and the vase has no wing magnets.

## Obiwan ND25FN-4 vase top

![Obiwan ND25FN-4 vase top with graded wings](../build/h2c/views/H2C_Dayton_ND25FN4_obiwan_graded_front.png)

- Replaces the LM top, the regular UM and its crescent with one piece that keys
  onto the released **keyed LM bottom of either stand state**
  (`obiwan_lm_keyed_bottom_floor_stand` or `obiwan_lm_keyed_bottom_no_floor_stand`, the regular Obiwan keyed LM
  bottom). The vase top contains
  the whole released keyed LM top below Y278 and is identical to it next to the
  seam, so the LM key, seam, L22 seat and inserts are the released ones. The
  released keyed LM top is the same part in both stand states (within 0.032 mm),
  so one vase top per hand fits both bottoms. Its
  rear tapers smoothly (C2) from the vase down each side of the LM ring to Y180.
- `tweeters_right`: MU10 at (−47.32, 361.74), T at (47.32, 354.81), t at (22.5, 423.0); `tweeters_left` is the
  mirror (x negated) above the released keyed LM region. Envelope
  227.9 × 296.5 × 41.3 mm, 440.5 cm³. It fits the H2C bed with 28 mm to spare.
- Buried ducts: T lead Ø6 mm and T → t link Ø6 mm, plus the MU10 lead Ø8.2 mm. The T lead descends into T's wiring
  chamber at no more than about 30°. Both released ring ducts run at full bore into the vase ducts. The MU10 lead
  passes under the T duct and dives out of the rear at the MU10 rim, at 285°. That is about 14 mm from
  the MU10 terminal strip (283°).
- Matching flat or graded wings (one continuous piece per side). Their upper
  magnet and the vase top's LM magnet sit at 9/171° instead of the regular
  64/116°. **Regular Obiwan wings do not fit.**

## Tweeter mounting and caps

- Each tweeter bore has a 4-fold key ring that takes the four lugs on the
  tweeter's front rim. Rotation is locked within ±2°, and the tab midpoint faces the
  cable mouth, so both solder runs are equal. **Solder** the leads;
  there is no room for Faston receptacles.
- The twist-lock cap has a printed R10 boss. With a 1.5 mm foam/felt pad,
  compressed to 1 mm, it preloads the motor on its axis. On the Stock vase, a
  foam ring under the cap sleeve also seats the rim flange. Tab-tip pockets in the cap
  clear the soldered tabs through the whole twist. The caps have no holes or
  screws: turn with a rubber grip pad, lift with a small suction cup (the pad pushes an
  unlocked cap about 0.5 mm proud).
- In the concept checks, the caps clear the tweeters in the locked, mid and entry poses.

## H2C print files

The files are in the product folders of the [H2C shelf](../to_print/README.md): the Stock vase and its caps in
[`to_print/stock/upper/`](../to_print/stock/README.md), the Obiwan vase top and its caps in
[`to_print/obiwan/upper/`](../to_print/obiwan/README.md), its wings in `to_print/obiwan/wings/` and the keyed LM bottom in
`to_print/obiwan/lm/`. Every part has an editable project and an audited slice side by side:
`<part>.petg-gf.3mf` / `.petg-gf.gcode.3mf` (Tinmorry PETG-GF + PLA).

| Part | Role / infill |
|---|---|
| `stock/upper/stock_nd25fn4_tweeters_{right,left}_upper` | Upper module, 100% zig-zag; T and t waveguides 15% gyroid |
| `obiwan/upper/obiwan_nd25fn4_tweeters_{right,left}_top` | LM top, 100% zig-zag; T and t waveguides 15% gyroid (either stand state) |
| `obiwan/lm/obiwan_lm_keyed_bottom_{floor_stand,no_floor_stand}` | Keyed LM bottom, 100% zig-zag (the regular Obiwan keyed LM bottom, shared by both hands and by every Obiwan upper; print the one for your stand state) |
| `{stock,obiwan}/upper/*_nd25fn4_tweeters_{right,left}_cap_{front,rear}` | Cap, 15% gyroid, supports |
| `obiwan/wings/obiwan_nd25fn4_tweeters_{right,left}_wing_{flat,graded}_{left,right}` | Wing, 10% gyroid, supports |

Each cap job holds one cap. Print both caps of the matching hand for each speaker. Caps print with the visible skin on top.
Use the `.3mf` or `.gcode.3mf`: they carry the material mapping, the duct
support blockers and the measured magnet pauses, and an STL alone does not.
Duct blockers stop 1.3 mm short of the MU10 and L22 seat ceilings so those
ceilings keep their support. After slicing, the audit checks every support toolpath against the
duct contracts.

See the [H2C policies and exceptions](../to_print/README.md#policy-exceptions).

## Magnets and hardware

| Location | Magnets (all Ø5 × 2 mm N52, buried) |
|---|---|
| Obiwan vase top | 2, LM ring at 9/171° |
| Obiwan keyed LM bottom | 2, released LM-lower sites |
| Each Obiwan ND25FN-4 wing | 2 (LM lower + LM upper at 9/171°) |
| Stock vase | none |

Insert magnets at the embedded, slice-derived pauses, checking polarity
against their mates.

- MU10: 4 × Hanglife HLTI-M3-001 (M3, Ø5 × 4 mm) in Ø4.6 × 4 mm printed pilots, as in the released UMs.
- Obiwan L22: the released M5 insert pilots. The keyed LM seam is assembled like
  the released keyed LM halves ([Obiwan guide](obiwan.md)).
- Stock seam B: the released H2C pin joint and the hidden M3 × 20 clamp.
- Tweeters: no screws. One 1.5 mm pad per cap (plus a foam ring on the Stock vase), soldered leads.

## Verification

- Interface checks ([report](../build/h2c/obiwan_interface_validation.json)):
  - Obiwan vase top (each hand) against both keyed LM bottoms, floor and no-floor stand: no overlap. Nothing is removed
    from the released keyed top below Y278, and the seam band is identical. The released floor and no-floor keyed tops
    agree within 0.032 mm. The rear taper stays clear of the L22 seat plane, the driver cutout and the wing faces.
  - Stock vase against both Stock LM states: no overlap, pins in their sockets, vestibule around the LM duct mouth.
- Driver fit (concept checks): no tweeter–body or cap interference in any
  cap pose. The key ring locks rotation and lets the tweeter in.
- Digital print qualification comes from the H2C audit. A physical print,
  loaded retention and acoustic behaviour are still pending. The waveguide
  shape alone does not establish directivity or crossover suitability.

Sources, generator scripts and rebuild commands:
[`candidates/nd25fn4_vase/`](../candidates/nd25fn4_vase/README.md). The
former fused body with M3 retainers (“V4”) is archived: its sources stay in
[`candidates/nd25fn4_crescent/`](../candidates/nd25fn4_crescent/README.md)
and its retired shelf files are under `build/h2c/experiments/retired_dayton_nd25fn4_v4/`.
