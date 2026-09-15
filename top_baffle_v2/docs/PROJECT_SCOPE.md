# Project scope and CAD brief

## Intent

This project adapts the LX521.4 top baffle for three tweeter families:
Dayton ND25FW-4, Tectonic TEBM35C10-4 BMR and Dayton ND25FN-4 printed
waveguide. It makes the oversized baffle printable as registered pieces,
with the current splits targeting H2C. It also explores two reductions in rear material: a front-flush slim
proud-routing plate and the more radical Obi-Wan carrier system. Optional perimeter
pieces are meant to make acoustic-boundary experiments repeatable without
reprinting the driver carriers.

![The three tweeter families and their five carrier arrangements](../images/generated/iso/rows/tweeter_row.png)

The repository is not a generic collection of every historical geometry idea.
Its human-facing inventory is three product families:

1. **Stock:** B2, nominally 18.3 mm deep, with mutually exclusive
   A-comp shoulders or B1 wings.
2. **Slim:** V1L lower/mids plus the V1 top. The acoustic field is
   11.5 mm deep and front-flush at z=18.3; the bottom structural strip remains
   full-depth. Use only the matching thin shoulder/wing set.
3. **Obiwan:** a common LM with either the regular UM and ND25FW-4/BMR
   crescent, or an integrated ND25FN-4 UM/waveguide body with two caps and
   two retainers. Each upper construction has matching flat or graded
   magnetic wings. Floor and stock-bridge/no-floor LM states share the
   upper interface but differ behind it.

Historical C7/V0 knife-edge work remains design research, not a top-level
product. The retired Obi-Wan W-series wing-concept generator and renders are
not retained. Current Obiwan wings have two upper interfaces (regular or
ND25FN-4) and two depth choices (flat or graded). Stock/Slim support
ND25FW-4 and BMR; ND25FN-4 currently fits Obiwan only.

## CAD brief

- **Task type:** inspection, repository curation, and generated-artifact
  catalog; no dimensional geometry change.
- **Units:** millimetres.
- **Coordinate convention:** acoustic baffle lies in XY; the shared acoustic
  front is z=18.3; rear material extends toward lower z.
- **Primary CAD:** STEP assemblies and parts from the existing build123d
  generators.
- **Secondary outputs:** front-face-down STL pieces with adjacent hash-bound
  `.print.json` orientation authorities; generated PNG design sheets and CAD
  snapshots.
- **Stock validation target:** one-piece B2 bounds 304.802 x 453.457 x
  18.3 mm; print assembly retains the same front plane.
- **Slim validation target:** 304.802 x 453.457 mm plan; main rear plane z=6.8
  (11.5 mm field); structural bottom strip may reach z=0.
- **Obiwan validation target:** both regular and integrated uppers preserve
  165.100 mm LM/UM axis spacing and the common LM joint/front at z=18.3,
  with separate floor/no-floor rear structure and matching wing interfaces.
- **Catalog paths:** current `to_print/h2c/` includes all three tweeter families;
  `artifacts/stock`, `artifacts/slim` and `artifacts/obiwan` retain the earlier CAD facade.
- **Assumptions:** “stock” names the full-depth B2 product family. The
  mounting state it ships in is the “stock-bridge (no-floor)” state, always
  written out in full so the product name is never read as a state.
  “Wings/shoulders” means alternative perimeter sets, never both installed at
  the same time. Flat is the constant-depth Obi-Wan wing; graded is the weighted-depth
  experiment.

## Release boundary

CAD/manifold checks are not physical authorization. The Obi-Wan state manifests
currently record `release_authorized: false`; the slim family also has an
explicit stiffness/physical-fit qualification burden. Real drivers, Fastons,
inserts, magnets, printed coupons, cable pull-through, and proof loading remain
outside purely geometric validation.
