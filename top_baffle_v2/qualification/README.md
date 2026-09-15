# Qualification fixtures — unsliced, not production parts

These fixtures turn the current review findings into measurable checks.
Follow [PETG_GF_QUALIFICATION.md](../docs/PETG_GF_QUALIFICATION.md), record
values in `results_template.csv`, and leave the release status pending until
real evidence supports it.

- `lm_registration.step.py` / `.step`: two native seam ends at the current
  pin pitch, joined by sacrificial braces; the separate STLs retain the
  standalone bottom/top Z rotations (26°/45°); `male_pin_pair.stl` and
  `female_socket_pair.stl` print in the stored front-down orientation.
- `process_witnesses.step.py` / `.step`: `native_route_cover.stl` is a crop
  of current LM CAD. `supported_roof_and_wall_steps.stl` is an accessible
  support-removal/purge specimen with a 0.8 mm roof and 0.8/1.2/1.6 mm walls.
- JSON records identify source hashes, geometry dimensions and STL hashes.
  PNGs are inspected CAD snapshots, not photos of printed results.
- `mesh_review.json` preserves the sampled production mesh review and its
  limitations. No fixture result or physical pass has been recorded.

Use the actual 0.6 mm high-flow GF/PLA process for supported tests and keep a
GF-only control. These STLs contain no supports, material changes or pauses;
slice and inspect them as test fixtures. They contain no captive magnets.
The two registration braces change stiffness and shrink behavior: follow a
coupon pass with full-carrier fit and loaded tests.

CAD brief: millimetres, build plate XY, +Z away from bed. Preserve current
production pin/socket surfaces and pitch; no production mating geometry is
modified. Validate one positive, closed solid per STL, bed fit and visual
snapshots. The native route crop is local evidence; every complete route
still needs a cable-fishing test. The support specimen's wall thicknesses
are deliberate experimental comparisons, not structural design limits.

STEP assemblies are spaced for review; print the separate STLs, each at the
bed origin. These match standalone part rotations. Combo plates add their
own Z rotations; repeat qualification at those placements before transferring
a coupon result to a combo process.

Inspected views: [registration pair](lm_registration_20260905T041039Z.png)
and [route/support witnesses](process_witnesses_20260905T041041Z.png).
