# Obi-Wan R6F

For current H2C print files and the integrated **Dayton ND25FN-4 waveguide**
upper, use the [H2C catalog](../../to_print/h2c/README.md) and
[three-family selection guide](../../docs/TWEETER_OPTIONS.md).
This facade retains the regular carrier and earlier split geometry.


![Obi-Wan no-floor core](states/no_floor/images/iso.png)

Obi-Wan replaces the full plate with mandatory front-flush LM and UM carriers.
Select exactly one mounting state:

- `states/no_floor/`: shallow fused web for the stock four-hole bridge; or
- `states/floor/`: LM-owned integral W64 stem/foot and NL8 panel.

Use either the monolithic LM carrier on a sufficiently large bed or both keyed
split halves—never both forms together. The tweeter crescent is optional.

Then choose one complete acoustic-wing family:

- `wings/flat/`: constant 11.5 mm solid depth; or
- `wings/graded/`: LM/UM/T-weighted rear depth with the same plan and magnetic
  roots.

Flat and graded are the entire Obi-Wan wing inventory. Each side prints as two
keyed pieces (four per speaker); do not mix flat and graded segments.

Both state manifests currently record `release_authorized: false`. CAD,
manifold, analytical-strength, and snapshot checks do not replace the physical
qualification procedure linked as `qualification.md`.
