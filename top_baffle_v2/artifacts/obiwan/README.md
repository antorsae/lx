# Obiwan — regular and integrated waveguide uppers

For current H2C print files and the integrated **Dayton ND25FN-4 waveguide**
upper, use the [H2C catalog](../../to_print/h2c/README.md) and
[three-family selection guide](../../docs/TWEETER_OPTIONS.md).
This facade retains the regular carrier and earlier split geometry; current
print files for every upper are linked below.

![Obiwan ND25FW-4, coaxial BMR, opposed BMR and ND25FN-4 waveguide uppers](../../images/generated/iso/rows/obiwan_upper_row.png)

| Upper choice | Parts | Matching H2C wings | Files |
|---|---|---|---|
| ND25FW-4 | Regular UM + crescent | Regular flat or graded | [Catalog](../../to_print/h2c/README.md) |
| TEBM35C10-4 BMR | Regular UM + coaxial or opposed crescent | Regular flat or graded | [Catalog](../../to_print/h2c/README.md) |
| ND25FN-4 waveguide | Fused UM/body + two caps + two M3 retainers | ND25FN-4 flat or graded | [Body, accessories and wings](../../to_print/h2c/dayton_nd25fn4/) |

![Regular and ND25FN-4 waveguide assemblies with flat and graded wings](../../images/generated/iso/rows/obiwan_wing_row.png)

All use the same LM attachment. Regular UM/wing contacts use Ø5 × 2 N52
magnets; the ND25FN-4 curved upper uses four Ø6 × 3 N45 magnets. Each
matching ND25FN-4 wing contains two D6 upper contacts and two D5 LM
contacts. See [Obiwan geometry, magnets and assembly](../../docs/obiwan.md).

## Earlier regular-carrier facade

Select exactly one mounting state:

- `states/no_floor/`: shallow fused web for the stock four-hole bridge; or
- `states/floor/`: LM-owned integral W64 stem/foot and NL8 panel.

Use either the monolithic LM carrier on a sufficiently large bed or both keyed
split halves—never both forms together. The tweeter crescent is optional.

Then choose one complete acoustic-wing family:

- `wings/flat/`: constant 11.5 mm solid depth; or
- `wings/graded/`: LM/UM/T-weighted rear depth with the same plan and magnetic
  roots.

These regular flat/graded files are for the regular UM only. Each earlier
P2S wing side prints as two keyed pieces (four per speaker); do not mix
flat and graded segments. Current H2C regular and ND25FN-4 wings each
print as one continuous piece per side.

Both state manifests currently record `release_authorized: false`. CAD,
manifold, analytical-strength, and snapshot checks do not replace the physical
qualification procedure linked as `qualification.md`.
