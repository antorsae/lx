# Project scope and CAD brief

## Intent

This project adapts the LX521.4 top baffle for a face-to-face ND25FW-4
tweeter arrangement and makes the oversized baffle printable as registered
pieces. It also explores two reductions in rear material: a front-flush slim
R6P plate and the more radical Obi-Wan/R6F carrier system. Optional perimeter
pieces are meant to make acoustic-boundary experiments repeatable without
reprinting the driver carriers.

The repository is not a generic collection of every historical geometry idea.
Its human-facing inventory is three product families:

1. **Stock R6P:** B2, nominally 18.3 mm deep, with mutually exclusive
   A-comp shoulders or B1 wings.
2. **Slim R6P:** V1L lower/mids plus the V1 top. The acoustic field is
   11.5 mm deep and front-flush at z=18.3; the bottom structural strip remains
   full-depth. Use only the matching thin shoulder/wing set.
3. **Obi-Wan R6F:** mandatory LM and UM collars, optional tweeter crescent, and
   flat or graded magnetic acoustic wings. Floor and stock-bridge/no-floor mounting
   states share the wing-contact outline but differ behind it.

Historical C7/V0 knife-edge work remains design research, not a top-level
product. The retired Obi-Wan W-series wing-concept generator and renders are
not retained; flat and graded are the complete Obi-Wan wing inventory.

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
- **Obi-Wan validation target:** two-carrier core, 165.100 mm LM/UM axis spacing,
  shared z=18.3 front, separate floor/no-floor rear structure, flat/graded manifests.
- **Catalog paths:** `artifacts/stock`, `artifacts/slim`, and
  `artifacts/obiwan`.
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
