# Three tweeter families

Generated from `src/lx521_baffle/tweeter_options.py` by `make h2c_docs`. Stock, Slim and Obiwan are baffle designs; the following are the three tweeter families.

| Tweeter family | Construction | Compatible baffles |
|---|---|---|
| Dayton ND25FW-4 face-to-face | Two domes with factory waveguide faceplates clamping the crescent | Stock, Slim, Obiwan |
| Tectonic TEBM35C10-4 BMR | Two BMR drivers in an opposed vase or a coaxial/opposed crescent | Stock, Slim, Obiwan |
| Dayton ND25FN-4 waveguide | Two faceplate-free domes in printed front/rear waveguides, fused with the MU10 UM carrier | Obiwan |

## What to change

| Baffle | ND25FW-4 | TEBM35C10-4 BMR | ND25FN-4 waveguide |
|---|---|---|---|
| Stock / Slim | Standard upper module, integral crescent | Opposed-BMR upper module replaces the whole standard upper | No compatible upper currently provided |
| Obiwan | Regular UM + separate crescent + regular wings | Regular UM + coaxial or opposed BMR crescent + regular wings | Fused UM/waveguide body + matching ND25FN-4 wings |

BMR coaxial and opposed mounts use the same driver family. On Obiwan they share the regular UM half-lap mount. The ND25FN-4 body instead includes the UM carrier: its new surround and buried upper magnets need the matching wings. All three Obiwan selections retain the common LM interface. Choose one upper arrangement per speaker.

Stock/Slim standard shoulders and B1 wings are for ND25FW-4 only: BMR upper magnet seats differ and no matching BMR perimeter is supplied. The extra side magnets on the Obiwan BMR pods also have no supplied mate; regular Obiwan wings retain their existing LM/UM contacts.

## Current H2C deliverables

The [H2C print catalog](../to_print/h2c/README.md) includes every family and records compatibility on each job. Choose a stand state, an upper arrangement and one optional wing style. Stock/Slim use one LM without the stand or two LM pieces with the stand. Obiwan uses one LM and one continuous optional wing per side.

The [Dayton ND25FN-4 waveguide guide](DAYTON_ND25FN4_WAVEGUIDE.md) covers its fused body, two caps, two M3 retainers, matching wings, magnet types and separate PETG-GF/PLA and PETG Translucent/PLA Translucent jobs.

For detailed geometry and BMR arrangements, see [Stock](stock.md#tweeter-options), [Slim](slim.md#tweeter-options), [Obiwan](obiwan.md#tweeter-options) and [Variants](VARIANTS.md). The earlier P2S shelf and CAD-only BMR-slim topology keep their separate qualification boundaries.

ND25FN-4 waveguide is the current product name for the former retained-package revision. Historical revision strings survive only as source/provenance identifiers; use the current H2C catalog for printing.

## Qualification

All three families are selection options; that does not certify identical acoustics or interchangeable drivers. BMR and ND25FN-4 remain qualification candidates. CAD and slicer checks do not establish physical fit, loaded retention, directivity or crossover suitability. The current H2C files still require a hardware/material trial.
