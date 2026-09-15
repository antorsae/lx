# Repository and artifact structure

Current manufacturing files are in [to_print/h2c](../to_print/h2c/README.md).
The [three-family comparison](TWEETER_OPTIONS.md) includes regular
ND25FW-4, BMR and the integrated ND25FN-4 waveguide as peer selections.

| Family / shared data | Authoritative source | Current delivery |
|---|---|---|
| ND25FW-4 and regular UM | `src/lx521_baffle/{proud,obiwan}/` | `to_print/h2c/{stock,slim,obiwan}/` |
| TEBM35C10-4 BMR | BMR geometry and candidate sources named in [Variants](VARIANTS.md) | Family-compatible jobs in the [H2C catalog](../to_print/h2c/README.md) |
| ND25FN-4 integrated waveguide | `candidates/nd25fn4_crescent/v4_model.py` and `build_wings.py`; retained-source mapping in `src/lx521_baffle/h2c/dayton.py` | `to_print/h2c/dayton_nd25fn4/`; canonical meshes in `to_print/h2c/STL/` |
| All-family visual comparisons | `scripts/gen_product_iso_matrix.py`, checked against `src/lx521_baffle/tweeter_options.py` | `images/generated/iso/rows/{tweeter,obiwan_upper,obiwan_wing}_row.png` |

The ND25FN-4 source directory retains historical revision names for input
hashes and print evidence; its location does not exclude it from the current
family catalog. `make h2c_review` rebuilds the shared-scale comparisons;
`make h2c_docs` refreshes the H2C file guide and family-selection guide.

## Implemented layout

```text
top_baffle_v2/
├── README.md                 project entry point and canonical inventory
├── src/lx521_baffle/         parametric geometry package
│   ├── proud/                proud-routing geometry families (Stock, Slim)
│   ├── obiwan/               Obiwan carriers, routes, floor, split, and wings
│   └── h2c/                  current printer geometry, materials and ND25FN-4 mapping
├── candidates/nd25fn4_crescent/  retained ND25FN-4 geometry sources and evidence
├── scripts/                  CAD export/render/release CLIs
├── tests/                    geometry, metadata, transport, and release gates
├── docs/                     prose authorities
├── build/                    validated generated state
│   ├── floor_stand/
│   ├── no_floor_stand/
│   ├── wings/{flat,graded}/
│   └── common/               independently promoted shared outputs
├── artifacts/                product-oriented links + hash manifests
│   ├── stock/
│   ├── slim/
│   └── obiwan/
├── coupons/                  process/fit qualification pieces
├── review/                   release catalogs and review evidence
├── to_print/                 earlier P2S shelf, plus current H2C under h2c/
└── tools/                    untouched concept sketches
```

The distinction is deliberate:

- `build/floor_stand/`, `build/no_floor_stand/`, and `build/wings/` are
  independently promoted build-system state and validation authorities.
- `artifacts/` is the earlier product facade. It contains no duplicate CAD
  bytes; its guides link to all current H2C upper choices.
- `review/` contains release evidence. The earlier P2S shelf is protected;
  the current `to_print/h2c/` is regenerated and audited by the H2C pipeline.

## Artifact contract

Each product owns:

```text
<product>/
├── README.md
├── manifest.json             SHA-256, byte size, source, role, status
├── cad/                      design/review STEP links
├── stl/                      printable STL + adjacent .print.json
└── images/                   generated plan, routing, and CAD snapshots
```

Obi-Wan nests `states/{floor,no_floor}` and `wings/{flat,graded}` because those are
real, mutually exclusive configuration choices. The stock/slim shoulder
and wing files remain siblings because each set is an alternative add-on to
the same base.

## Source and generated-state boundary

The implemented source hierarchy is:

```text
src/lx521_baffle/             parametric geometry package
scripts/                      CAD export/render/release CLIs
tests/                        unit, geometry, metadata, and remote tests
docs/                         all prose authorities
build/                        generated and promoted state output
artifacts/                    curated release facade
```

Make prerequisites, remote snapshots, source attestations, native-stage
manifests, and release catalogs all use these canonical paths. The artifact
facade remains a generated view and does not duplicate CAD bytes.

## Cleanup policy

Keep source, manifests, current STEP/STL/PNG outputs, and physical
qualification evidence. Ignore or trash viewer GLBs, `__pycache__`, pytest
caches, remote jobs, Make stamps, G-code workspaces, failed slice runs, and
dated one-off review PNGs. Explicit review inputs used by the current H2C
release stay tracked; archived slicer workspaces do not. Current print files
use ordinary Git, with no new Git LFS types. The
[history guide](GENERATED_HISTORY.md) documents the unpublished-history cleanup,
source checkpoint mapping, and fresh-checkout setup.

The artifact facade also links oversized local CAD assemblies and the graded
wing plate, which are intentionally ignored by Git. Regenerate those before
running the full facade check; they are not required to validate the H2C print
shelf. Regenerate the product catalog after any promoted
artifact change:

```bash
python3 scripts/build_artifact_catalog.py
python3 scripts/build_artifact_catalog.py --check
```
