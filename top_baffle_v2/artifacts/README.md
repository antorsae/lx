# Product artifact catalog

This is the earlier product-oriented CAD facade. Use the
[current H2C manufacturing catalog](../to_print/README.md) for current
STLs, projects and slices. The [five-family tweeter guide](../docs/TWEETER_OPTIONS.md)
includes **Dayton ND25FN-4 waveguide**, an integrated Obiwan upper with
matching wings, and the **PT2522-4 / AMT17D2.1 dipoles**, alongside ND25FW-4
and TEBM35C10-4 BMR.

![Five tweeter families and eight carrier arrangements at one scale](../images/generated/iso/rows/tweeter_row.png)

This is the human-facing inventory for the three supported LX521.4 top-baffle
product choices. Relative symlinks point to validated generator outputs, so
the catalog does not duplicate large STEP or STL files.

| Product | Choose it for | Status |
|---|---|---|
| [`stock/`](stock/) | Full-depth B2 base with either A-comp shoulders or B1 wings | Canonical CAD; physical fit remains the builder's responsibility |
| [`slim/`](slim/) | V1L + V1 front-flush acoustic field with matching thin attachments | Experimental; structural and hardware qualification required |
| [`obiwan/`](obiwan/) | Common LM with regular UM/crescent or integrated ND25FN-4 UM/waveguide, and matching flat/graded wings | Candidate only; release authorization is false |

| Upper family | Printed construction | Current files and documentation |
|---|---|---|
| Dayton ND25FW-4 | Standard Stock/Slim upper or regular Obiwan UM + crescent | [H2C files](../to_print/README.md) · [Obiwan assembly](../docs/obiwan.md#dayton-nd25fw-4-crescent) |
| Tectonic BMR | Stock/Slim opposed vase or Obiwan coaxial/opposed crescent on regular UM | [H2C files](../to_print/README.md) · [Mount variants](../docs/TWEETER_OPTIONS.md) |
| Dayton ND25FN-4 waveguide | Stock vase, or Obiwan vase top on the keyed LM bottom; two twist-lock caps; matching Obiwan wings | [H2C files](../to_print/README.md) · [Assembly](../docs/DAYTON_ND25FN4_WAVEGUIDE.md) |

Each product directory has a generated `manifest.json` containing SHA-256,
byte size, source path, and role for every linked file. Rebuild or verify the
facade with:

```bash
python3 scripts/build_artifact_catalog.py
python3 scripts/build_artifact_catalog.py --check
```

For manufacturing, use the current H2C projects with their support settings
and magnet pauses. The older linked STLs use their adjacent `.print.json`
orientation authorities. STEP review assemblies are not print plates.
