# H2C print files

65 of 65 jobs digitally audited. Printer: **H2C, two 0.6 mm High Flow nozzles**.

The pre-migration source checkpoint is recorded in the [generated-file history guide](../docs/GENERATED_HISTORY.md). These projects use native H2C machine programs. H2C is the only supported printer; superseded P2S files are under `archive/p2s/` for provenance only.

| Product | Files |
|---|---|
| Stock | [stock/](stock/README.md) — `lm/`, `upper/`, `wings/` |
| Slim | [slim/](slim/README.md) — `lm/`, `upper/`, `wings/` |
| Obiwan | [obiwan/](obiwan/README.md) — `lm/`, `upper/`, `wings/`, `addons/` |

**Choose one product, one stand state and one tweeter/wing alternative per speaker.** Print twice for a stereo pair (each ND25FN-4 cap job is one cap).

| Configuration | H2C main pieces per speaker |
|---|---|
| Stock / Slim, no floor stand | 1 LM + 1 upper module |
| Stock / Slim, floor stand | 1 lower LM with stand + 1 upper LM + 1 upper module |
| Regular Obiwan, either stand state | 1 LM (one piece, or keyed bottom + keyed top) + 1 UM + 1 crescent + 2 optional wings |
| Stock with ND25FN-4 vase, either stand state | Stock LM (1 or 2 pieces) + 1 ND25FN-4 vase + 2 caps |
| Obiwan with ND25FN-4 vase top, either stand state | 1 keyed LM bottom (floor or no-floor) + 1 vase top + 2 caps + 2 optional wings |

Stock/Slim standard shoulders and B1 wings are optional for ND25FW-4; no matching BMR perimeter is supplied. The upper module and BMR alternative are shared between stand states. A-style shoulders remain four pieces: their upper and lower outlines meet only at a tangent point, so joining them would require an outline change. The B1 alternative already uses one continuous wing per side. Use the new H2C upper module with the new contained-pin LM joint; its two Ø3 mm pins replace the old upward dovetails. The hidden M3 × 20 clamp and Hanglife M3, Ø5 × 4 mm insert remain. Floor-stand LM halves retain the established lower dovetail joint.

The Obiwan keyed LM bottom and keyed top are the released two-piece LM. They take every Obiwan upper, and the keyed bottom is also what the ND25FN-4 vase top keys onto. The keyed top is one part for both stand states.

The unchanged one-piece Stock/Slim floor-stand trials failed the actual PLA-nozzle reach check. Their reports are retained under `build/h2c/experiments/rejected_one_piece_floor/`; trial meshes and slices remain local, outside this shelf. The current floor-stand pair removes the old vertical LM seam and keeps the original outline.

## Tweeter families

![Five tweeter families and eight arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)

| Tweeter family | Construction | Compatible baffles |
|---|---|---|
| Dayton ND25FW-4 face-to-face | Two domes with factory waveguide faceplates clamping the crescent | Stock, Slim, Obiwan |
| Tectonic TEBM35C10-4 BMR | Two BMR drivers in an opposed vase or a coaxial/opposed crescent | Stock, Slim, Obiwan |
| Dayton ND25FN-4 waveguide | Two faceplate-free domes in printed front/rear waveguides, fused with the MU10 upper | Stock, Obiwan |
| GRS PT2522-4 dipole planar | One open-back planar on a frame above (T above) or below (T below) the raised MU10, upright or rotated, M3 heat-set inserts | Obiwan |
| Mundorf AMT17D2.1 dipole AMT | One open-back AMT on a frame above (T above) or below (T below) the raised MU10, upright or rotated, M3 heat-set inserts | Obiwan |

![Complete Obiwan upper choices at the same LM datum](../images/generated/iso/rows/obiwan_upper_row.png)

The ND25FN-4 Obiwan vase top replaces the LM top, the regular UM and the crescent; it needs a keyed LM bottom (floor or no-floor stand) and, if wings are wanted, its matching wings. The ND25FN-4 Stock vase replaces the Stock upper module. Slim has no ND25FN-4 upper. The PT2522-4 / AMT17D2.1 dipoles (Obiwan only, `obiwan/dipole/`) replace the UM (T above) or the keyed LM top and UM (T below). [Dipole tweeter guide](../docs/DIPOLE_TWEETERS.md) · [Tweeter family selection guide](../docs/TWEETER_OPTIONS.md) · [ND25FN-4 assembly, hardware and print guide](../docs/DAYTON_ND25FN4_WAVEGUIDE.md)

## Materials and support

- **Model: left nozzle. PLA support interface: right nozzle.** Configure the corresponding feeds on the H2C. Use the current H2C nozzle assignments for both material lanes.
- Material: Tinmorry PETG-GF + PLA (`.petg-gf` files) for every part.
- Engineering Plate with glue, 70 °C for both materials, no raft.
- 0.16 mm layers, 0.20 mm first layer, six walls. Structural LM/UM: 100% infill. Wings: 10% gyroid. Dayton ND25FN-4 vase bodies: 100%, with their T and t waveguides at 15% gyroid (a modifier volume; the MU10 seat and the LM/seam joints stay 100%); their caps: 15% gyroid. Regular crescent: its existing 30% policy.
- PETG support bases with three dense, zero-gap PLA interface layers above and below. ND25FN-4 caps print skin-up with support under their sleeves; vase duct blockers stop short of the driver-seat ceilings.
- Obiwan floor-stand LMs mask their underside NL8 service bay, so its short lid-rebate lip bridges instead of carrying a support column through the lead bay.
- Stock/Slim complete no-floor LM uses a 2 mm brim and no support; other parts use a 5 mm brim. The narrow brim needs an adhesion trial on the actual H2C plate.
- Dedicated nozzles retain native H2C priming, retraction and standby programs. The dedicated-nozzle process uses its own H2C purge policy. Flushing into model, infill and supports is disabled.

## Magnets and hardware

All interfaces use Ø5 × 2 mm N52 magnets. A full regular wing has three; an ND25FN-4 Obiwan wing has two (LM lower, LM upper at 9/171°) and no UM contact. The ND25FN-4 Stock vase has no wing magnets. All remain buried. Magnet pauses are recalculated from actual H2C extrusion and embedded in the projects.

Retain existing driver, LM/UM, stand and wing hardware. The ND25FN-4 vases keep the MU10 M3/Ø4.6 mm printed bores for Hanglife M3 inserts, Ø5 mm outside × 4 mm long; the tweeters need no screws (twist-lock caps, soldered leads). The optional Tectonic BMR mounts retain their documented M2 hardware exception.

## Files

Each product folder lists its parts with their STL, editable project (`.3mf`: geometry, settings and measured magnet pauses), audited slice (`.gcode.3mf`) and audit report. STLs carry geometry only; importing an STL alone loses supports, blockers, infill modifiers and insertion pauses. `catalog.json` is the machine-readable job list.

[Stock](stock/README.md) · [Slim](slim/README.md) · [Obiwan](obiwan/README.md)

## Policy exceptions

- Use dedicated left/right nozzles; old P2S AMS slot numbers are not nozzle assignments. Configure a model feed to the left and a PLA feed to the right before printing.
- Retain native H2C retraction, temperature standby, nozzle changing, start/end and priming G-code. Transfer only explicit Tinmorry thermal/flow calibrations. Do not apply the P2S 560 mm3 same-nozzle purge to every dual-nozzle switch.
- PETG-GF and PLA use a shared 70 C Engineering Plate with glue. No raft. This is the user's plate process and requires a first-print adhesion check, especially the 2 mm Stock/Slim brim.
- Use three dense zero-gap PLA interface layers above and below support for H2C. Keep PETG as the support base. Disable flushing into model, infill and support.
- The native H2C firmware programs and generated material/nozzle assignments must be checked from the actual slice; prepared projects alone are not print qualification.
- Obiwan ND25FN-4 vase wings retain Arachne and supports. Each D5 captive magnet is rechecked against the actual toolpaths.
- Expand regular cable/fastener support blockers by 0.8 mm along each print axis. Exact surface-only blockers allowed support in duct ceilings; the model geometry is unchanged. Dayton ND25FN-4 vase blockers follow the same rule, and their duct blockers stop 1.3 mm short of the MU10 and L22 seat ceilings so those ceilings keep their support.
- Stock/Slim floor lower pieces extend the three cable-entry support keepouts through the rear service panel. This prevents supports for the panel from being routed down the cable entries. The small regions above those entries bridge; surrounding panel supports remain enabled.

## Geometry review

![Regular and ND25FN-4 waveguide assemblies with matching flat or graded H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)

The four panels use one camera and scale. Regular panels use the no-floor-stand LM; ND25FN-4 panels use the keyed floor-stand LM bottom and the vase top (it fits the no-floor keyed bottom as well). Regular wings also fit the regular-UM BMR selections. Drivers and service caps are omitted.

[Stock/Slim orthographic assembly sheet](../build/h2c/views/primary_review_contact_sheet.png) · [Obiwan ND25FN-4 vase top with graded wings — front](../build/h2c/views/H2C_Dayton_ND25FN4_obiwan_tweeters_right_graded_front.png) · [Obiwan ND25FN-4 vase top with flat wings — rear](../build/h2c/views/H2C_Dayton_ND25FN4_obiwan_tweeters_right_flat_rear_oblique.png) · [Stock ND25FN-4 vase — front](../build/h2c/views/H2C_Dayton_ND25FN4_stock_tweeters_right_front.png) · [Stock ND25FN-4 vase — rear](../build/h2c/views/H2C_Dayton_ND25FN4_stock_tweeters_right_rear_oblique.png)

The views use the actual exported parts in installed coordinates, including the ND25FN-4 caps. Drivers are omitted. Native STEP assemblies and individual parts are in `../build/h2c/review_models/` and `../build/h2c/STEP/` from this guide. The installed CAD Viewer launcher was unavailable; these PNGs provide the checked visual review.

[Native Stock/Slim joint checks](../build/h2c/geometry_validation.json) · [Obiwan LM / regular UM and ND25FN-4 vase interface checks](../build/h2c/obiwan_interface_validation.json)

## Verification and limits

The audit checks sliced geometry and modifier identity, machine/material settings, actual nozzle assignments, deposition bounds including bead widths and arc extrema, support clearance from every duct contract, captive-wall extrusion and insertion timing, plus static G-code validation. Native CAD checks cover the new Stock/Slim pin joint, solid interference and preserved driver openings.

Passing those checks does not establish physical adhesion, magnet retention, loaded stand strength or acoustic performance. The first H2C build still needs the project’s hardware/material qualification. BMR and the Dayton ND25FN-4 vases retain their candidate status.

## Rebuild

On a fresh checkout, first follow the [retained-source setup](../docs/GENERATED_HISTORY.md#fresh-checkout-validation).

```sh
make                    # H2C geometry, projects, slicing, audit and generated guide
make h2c_prepare        # prepare editable projects from existing authorities
make h2c_validate       # slice/audit prepared projects; reuse hash-matched outputs
make h2c_review         # regenerate native assembly and mesh review images
make h2c_docs           # regenerate this guide
```

Policy sources: [H2C printer/material policy](../print_policy_h2c.json) and [shared role/hardware policy](../print_policy.json). The build does not connect to a printer.
