# H2C migration

The pre-migration source checkpoint is recorded in the
[generated-file history guide](GENERATED_HISTORY.md). H2C outputs live
in `to_print/h2c/`; the older printer shelf remains a separate P2S release.

## Geometry and interfaces

- Millimetres; installed X is lateral, Y is up, and Z points forward.
- Keep all driver centres, seats, screw axes, ducts and magnetic attachment
  interfaces. Print front down, with rotation only about the bed normal.
- Stock and Slim: one complete LM without the stand, plus a matching
  upper vase. The stand version keeps the lower Y120 dovetail seam and
  combines the two former middle panels. Its one-piece trial failed actual
  PLA-nozzle reach in both 90-degree orientations; preserve their reports
  outside the print shelf and keep trial meshes local. Replace the broad upward seam-B keys with two smaller,
  rear-biased round registration pins integral with the upper vase. Their
  receiving holes stay inside the LM envelope. Retain the hidden M3 clamp
  screw at the original joint. Validate pin clearance against the driver
  envelope, cable ducts and magnets. Do not scale the baffle.
- Obiwan: use its canonical unsplit LM and full detachable left/right
  wings (flat and graded). Keep the regular UM and tweeter service joint.
- Dayton ND25FN-4 waveguide: retain the approved body, caps and retainers. Extend each matched
  Dayton ND25FN-4 waveguide upper wing continuously into the canonical lower wing, eliminating
  the old print split without changing the magnetic interfaces.
- Retain Stock/Slim shoulder and wing alternatives and the experimental
  BMR alternatives, clearly labelled as alternatives.

## Printer and process

- H2C with 0.6 mm High Flow hotends. Model material on the left nozzle;
  PLA interface material on the right. These are physical feed paths,
  not the old P2S AMS slot numbers.
- Tinmorry PETG-GF/PLA is the normal lane. Dayton ND25FN-4 waveguide also receives a separate
  PETG Translucent/PLA Translucent lane for the Engineering Plate with glue.
- Resolve actual installed H2C profiles. Preserve machine tool-change,
  heating and priming programs; do not transplant P2S machine G-code or
  same-nozzle purge volumes into a two-nozzle job.
- Preserve centralized role, insert, magnet, infill and support policies.
  Structural LM/UM 100%; wings 10% gyroid; Dayton ND25FN-4 waveguide tweeter region 15% gyroid.
- Stock/Slim complete no-floor LM uses a 2 mm outer brim; other parts start with
  5 mm. Check model, brim, supports and prime tower against each nozzle's
  reachable region. A bounding-box fit alone is insufficient.

## Validation and qualification

Hash-bind generated files to their inputs. Check native solids and mesh
topology, protected mounting datums, assembled joint clearances, and the
elimination of obsolete split seams. Inspect orthographic snapshots.
Resolve and audit material vectors and nozzle assignment. Slice with the
installed BambuStudio CLI, inspect actual support/material toolpaths, and
recalculate magnet insertion pauses for the H2C files. Publish separate
statuses for prepared projects, successful slices and physical qualification.
No printer connection or print start is part of this work.
