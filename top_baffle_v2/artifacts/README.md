# Product artifact catalog

This is the human-facing inventory for the three supported LX521.4 top-baffle
product choices. Relative symlinks point to validated generator outputs, so
the catalog does not duplicate large STEP or STL files.

| Product | Choose it for | Status |
|---|---|---|
| [`stock/`](stock/) | Full-depth B2 base with either A-comp shoulders or B1 wings | Canonical CAD; physical fit remains the builder's responsibility |
| [`slim/`](slim/) | V1L + V1 front-flush acoustic field with matching thin attachments | Experimental; structural and hardware qualification required |
| [`obiwan/`](obiwan/) | Minimal R6F carriers with floor/no-floor mounting and Ac/Ae acoustic wings | Candidate only; release authorization is false |

Each product directory has a generated `manifest.json` containing SHA-256,
byte size, source path, and role for every linked file. Rebuild or verify the
facade with:

```bash
python3 scripts/build_artifact_catalog.py
python3 scripts/build_artifact_catalog.py --check
```

Do not print a STEP review assembly. Use each STL together with its adjacent
`.print.json` orientation authority and follow [`PRINTING.md`](../docs/PRINTING.md).
