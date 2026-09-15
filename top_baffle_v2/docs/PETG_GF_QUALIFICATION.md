# Current PETG-GF qualification procedure

Status: **PENDING — no physical pass claimed**. Qualify floor and bridge
states independently, including the keyed LM form. Record results in
[results_template.csv](../qualification/results_template.csv) and the
[physical qualification record](obiwan_physical_qualification.md).

The current process uses TINMORRY PETG-GF and the P2S 0.6 mm high-flow nozzle;
PLA Basic is used only at supported interfaces. TINMORRY specifies a
wear-resistant steel nozzle, preferably 0.6 mm, and lists 240–270 °C nozzle
and 65–75 °C bed ranges. Those are vendor ranges, not permission to change
an audited project arbitrarily. [TINMORRY product guidance](https://tinmorry.net/en-us/products/tinmorry-glass-fiber-reinforced-petg-filament-high-impact-resistance-petg-gf-3d-printing-filament-durability-stiffness-perfect-compatible-with-bambu-lab-fdm-3d-printer-1-kg-1-spool-frosted-grey).

PLA/PETG's poor mutual adhesion is useful for removable interfaces. Bambu's
support introduction recommends drying both materials. Its PLA/PETG-HF
guidance is a starting point, not validation of this glass-filled formulation.
[Bambu support introduction](https://forum.bambulab.com/t/printing-guide-for-using-pla-basic-and-petg-hf-for-support/93273).

## Preserve the process identity

Record the selected GUI project SHA-256 from `to_print/delivery_manifest.json`,
exported slice SHA-256, Bambu version, actual installed nozzle, filament
brands/lots, drying settings/time, room conditions and material-slot map.
Use the embedded process: six walls; 100% zig-zag for core combos, LM bottoms,
standalone LM tops and UM; 30% gyroid for the regular crescent and lid. Supported parts
use normal/snug support, GF body, PLA interface and zero top Z gap.
The top needs support; crescent/lid do not. Preserve front-down orientation.

The current calibration uses **560 mm³ purge in each direction** and limits
**PLA flushing to 12 mm³/s**. It supersedes 280 mm³ and automatic PLA flushing
after a user-reported blockage during a GF-to-PLA change. This is a proposed
calibration, **not a physical pass or a confirmed diagnosis**. Printing
temperatures and printing flow limits retain their existing recipes.
Start with the [83-minute D6 body/wing test](../candidates/nd25fn4_crescent/print/qualification/README.md).
Observe both change directions for continuous extrusion and loading errors. Keep a
GF-only control and record purge volume in each direction, tower geometry,
interface removal force and failure location. Do not flush PLA into load-
carrying GF walls/infill. Accept the calibration only after repeatable
results; a color-clean transition alone does not prove bonding. The
[shared policy reference](PRINT_POLICIES.md) and native changeover audit
record these settings; the separate translucent lane retains its own recipe.

## Coupons and measurements

The [qualification folder](../qualification/README.md) contains STEP sources,
STLs and inspected snapshots. Coupons are unsliced test fixtures.

| Test | Method | Evidence required / rejection |
|---|---|---|
| Registration pair | Print `male_pin_pair.stl` and `female_socket_pair.stl` in the actual GF recipe. The native seam ends retain the real 217.841807 mm pitch; sacrificial braces hold the ends together. Dry-fit, separate and repeat, then test the full LM halves. | Both Ø1.60 pins and open Ø1.80 / X-relieved 1.84 × 1.80 sockets must have continuous preview paths. Record pitch, force, seating gap and damage after repeated assembly. Any missing socket wall, forced fit or broken pin fails. Braces do not reproduce full-carrier shrink or strength. |
| Native route cover | Print `native_route_cover.stl` with the matching production support/blocker treatment; its crop is from the current no-floor LM STEP. Fish the actual lead, then section a sacrificial copy across the cover. | Record wall/roof measurements and a section photo. Reject gaps, damaged covers, trapped support or cable obstruction. This one sector does not qualify every route; repeat the functional checks on complete parts. |
| Supported roof and wall steps | Print `supported_roof_and_wall_steps.stl` with GF model/body and PLA interface. Its accessible 0.8 mm roof and 0.8/1.2/1.6 mm walls compare path continuity and support removal. Keep a GF-only control. | Photograph every thickness in preview and after removal. Record removal force, surface damage and destructive bend/fracture location across swaps. Reject interlayer separation or fused support. This representative fixture is not a carrier strength test. |
| Captive magnet skins | Use a sacrificial production core/wing pair from the chosen project, keeping its exact pause and orientation; section after insertion. | Measure both skins and fit/pull. Current source skins are 0.52 mm; Obi-Wan's nominal paired face separation is 1.24 mm including the 0.15 mm burial and 0.05 mm construction standoff. Reject a protruding disc, wrong polarity, roof discontinuity or skin damage. |
| Inserts and screw preload | Install actual M2/M3/M5 SKUs in sacrificial production bores with recorded heat/time/depth. Test pullout and tightening with the real screw stack; repeat after conditioned dwell. | Record peak force, torque, displacement, failure mode and photos. Define acceptance loads before testing using the installed load case; no approved pullout value is inferred from a bore diameter. Reject spin, cracking, pullout or creep under the required load. |
| Coplanarity and assembly | Dry-fit complete parts on a flat reference with actual drivers, leads and joints. Test both wing profiles if both are to be qualified. | Record seam/front-plane gaps before tightening and after dwell. No forcing the driver flange to hide a proud seam; no pin or magnet structural-load credit. |
| Loaded behavior / creep | Use the real selected driver mass, bridge/foot, fasteners and operating-temperature range. Follow the state-specific record and record baseline, 1 h, 24 h and longer-term readings. | Record applied load, temperature, screw preload, deflection and permanent set. Predeclare acceptance limits and duration. Cracking, joint slip, growing deflection or cable contact fails. A short room-temperature pass does not prove long-term behavior. |

The review's mesh screen found carrier p05 thickness near 0.8 mm, consistent
with intentional thin covers. It is a sampling result, not a guaranteed
minimum or a strength prediction. Nested magnet voids received no thickness
ray hits, so their skins must be measured from sections/source geometry.
Generic 1.2/1.6 mm wall guidelines are screening values, not proof that these
parts fail. If a cover or socket cannot print continuously, thicken toward
those values only where it preserves lumens, driver recesses, flange and
wing clearances, then regenerate the CAD, fit audit and slice. Do not
indiscriminately grow a tightly constrained mating wall.

Keep front-face-down orientation. Lower calculated overhang on an edge-
standing carrier does not preserve its magnet roof/pause sequence; orientation
changes require new support, loading-aperture, roof and pause audits.

## Export and audit a GUI slice

After slicing and preview inspection in Studio, export the **sliced** 3MF:

```sh
../.venv/bin/python scripts/audit_gui_slice.py \
  --name obiwan_01_02_03_04_LM_UM_combo_no_floor_stand \
  --project /path/to/exported.gcode.3mf
```

The importer checks the current source meshes/placements and modifiers,
pinned settings, actual material extrusion, magnet cavities and pause order,
functional ducts, enclosed support and G-code machine bounds. It saves a
successful import under `review/gui_slices/<name>/<hash>/`; it never starts a
printer or promotes a release. Keep the preview screenshots beside that
record. An unsliced file, stale geometry or a missing/failed gate is rejected.
This command still requires the repository, Bambu's pinned profiles and the
G-code validator; the standalone print pack is not a slicer installation.

## Photos and acoustic claims

Capture the front, rear, seams, support removal, inserts, seated magnets,
loaded setup and finished speaker with a scale/reference in frame. Store real
images and paths in the worksheet. No physical photographs were supplied for
this update; CAD images are labeled illustrations.

For acoustic comparisons, record the installed drivers and wiring/polarity,
crossover/EQ, microphone/distance/height, room or outdoor setup, time window,
level, smoothing and angular sampling. Compare the same speaker and settings
with a stated baseline; retain raw measurements. Until then, describe geometry
and interchangeability, without claims of flatter response, wider dispersion
or reduced diffraction.
