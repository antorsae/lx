# Five tweeter families

Generated from `src/lx521_baffle/tweeter_options.py` by `make h2c_docs`. Stock, Slim and Obiwan are baffle designs; the following are the five tweeter families.

![Five tweeter families, ten arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)

Left to right: ND25FW-4 crescent, BMR Stock/Slim vase, BMR Obiwan coaxial crescent, BMR Obiwan opposed crescent, ND25FN-4 Stock vase, ND25FN-4 Obiwan vase top, and the PT2522-4 and AMT17D2.1 dipoles (T above, upright and rotated; all four layouts of each below). Actual printed carriers share one scale; drivers and removable service caps are omitted.

| Tweeter family | Construction | Compatible baffles |
|---|---|---|
| Dayton ND25FW-4 face-to-face | Two domes with factory waveguide faceplates clamping the crescent | Stock, Slim, Obiwan |
| Tectonic TEBM35C10-4 BMR | Two BMR drivers in an opposed vase or a coaxial/opposed crescent | Stock, Slim, Obiwan |
| Dayton ND25FN-4 waveguide | Two faceplate-free domes in printed front/rear waveguides, fused with the MU10 upper | Stock, Obiwan |
| GRS PT2522-4 dipole planar | One open-back planar on a frame above (T above) or below (T below) the raised MU10, upright or rotated, M3 heat-set inserts | Obiwan |
| Mundorf AMT17D2.1 dipole AMT | One open-back AMT on a frame above (T above) or below (T below) the raised MU10, upright or rotated, M3 heat-set inserts | Obiwan |

## What to change

![Obiwan upper choices aligned at the common LM joint](../images/generated/iso/rows/obiwan_upper_row.png)

| Baffle | ND25FW-4 | TEBM35C10-4 BMR | ND25FN-4 waveguide | PT2522-4 / AMT17D2.1 dipole |
|---|---|---|---|---|
| Stock | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | ND25FN-4 vase replaces the whole standard upper; two twist-lock caps; no wings | Not provided |
| Slim | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | No compatible upper currently provided | Not provided |
| Obiwan | Regular UM + separate crescent + regular wings | Regular UM + coaxial or opposed BMR crescent + regular wings | Keyed LM bottom (either stand state) + vase top (LM top + UM + waveguides) + two caps + matching ND25FN-4 wings | T above: any LM + UM-with-tweeter piece. T below: keyed LM bottom + LM-top-with-tweeter-and-UM piece (PT2522 upright: two prints). Upright or rotated. No wings |

The dipole parts carry one open-back driver each, upright or rotated 90°, above or below the raised MU10, with hidden T and UM ducts (T below: on opposite sides). Drivers fix with M3 screws into M3 heat-set inserts. [Dipole tweeter guide](DIPOLE_TWEETERS.md).

![Dipole tweeters: PT2522-4 and AMT17D2.1, T above / T below, upright / rotated](../images/generated/iso/rows/dipole_row.png)

BMR coaxial and opposed mounts use the same driver family. On Obiwan they share the regular UM half-lap mount. The ND25FN-4 Obiwan vase top replaces the LM top and the UM together; it keys onto the released keyed LM bottom of either stand state, so it does not use the one-piece Obiwan LM. Its upper LM wing magnets sit at 9/171° and need the matching wings. The ND25FN-4 Stock vase uses the H2C Stock LM in either stand state. Choose one upper arrangement per speaker.

![Regular and ND25FN-4 uppers with their flat and graded H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)

| Obiwan upper | Buried upper contacts | Each matching H2C wing | Hardware / assembly |
|---|---|---|---|
| ND25FW-4 | 2 × Ø5 × 2 mm N52 in regular UM | 3 × Ø5 × 2 mm N52 | [Regular crescent](obiwan.md#dayton-nd25fw-4-crescent) |
| TEBM35C10-4 BMR | Same regular UM contacts | Same regular wings | [Coaxial](obiwan.md#candidate-coaxial-tebm35c10-4-bmr-crescent) / [opposed](obiwan.md#candidate-opposed-tebm35c10-4-bmr-crescent) |
| ND25FN-4 vase top | 2 × Ø5 × 2 mm N52 at 9/171° in the vase top + 2 in the keyed LM bottom | 2 × Ø5 × 2 mm N52 (LM lower + LM upper) | [Vase top, caps and wings](DAYTON_ND25FN4_WAVEGUIDE.md) |
| PT2522-4 / AMT17D2.1 dipole | None (T below: the LM top pockets are filled) | No wings | [Dipole tweeter guide](DIPOLE_TWEETERS.md) |

Stock/Slim standard shoulders and B1 wings are for ND25FW-4 only: BMR upper magnet seats differ and no matching BMR perimeter is supplied. The extra side magnets on the Obiwan BMR pods also have no supplied mate; regular Obiwan wings retain their existing LM/UM contacts.

## Current H2C deliverables

The [H2C print catalog](../to_print/README.md) includes every family and records compatibility on each job. Choose a stand state, an upper arrangement and one optional wing style. Stock/Slim use one LM without the stand or two LM pieces with the stand. Obiwan uses one LM and one continuous optional wing per side.

The [Dayton ND25FN-4 waveguide guide](DAYTON_ND25FN4_WAVEGUIDE.md) covers both vases, their twist-lock caps, the Obiwan keyed LM bottom and matching wings, and their PETG-GF/PLA jobs.

For detailed geometry and BMR arrangements, see [Stock](stock.md#tweeter-options), [Slim](slim.md#tweeter-options), [Obiwan](obiwan.md#tweeter-options) and [Variants](VARIANTS.md). H2C is the only supported printer. Archived small-bed files and CAD-only BMR-slim topology are not current print selections.

The ND25FN-4 vases replace the former fused body with M3 retainers; that package is archived. Historical revision strings survive only as source/provenance identifiers; use the current H2C catalog for printing.

## Qualification

All five families are selection options; that does not certify identical acoustics or interchangeable drivers. BMR, ND25FN-4 and the dipoles remain qualification candidates. CAD and slicer checks do not establish physical fit, loaded retention, directivity or crossover suitability. The current H2C files still require a hardware/material trial.
