# Repository and artifact structure

## Implemented layout

```text
top_baffle_v2/
├── README.md                 project entry point and canonical inventory
├── docs/                     scope and documentation index
├── tools/                    repository-level, non-CAD curation tools
├── images/generated/         current CAD snapshot packets
├── artifacts/                product-oriented links + hash manifests
│   ├── standard/
│   ├── slim/
│   └── obiwan/
├── floor_stand/              validated generator output state
├── no_floor_stand/           validated generator output state
├── wings/{ac,ae}/            validated Obi-Wan wing output families
├── coupons/                  process/fit qualification pieces
├── top_baffle_*.py           current build123d source modules
├── export_*.py, gen_*.py     current build/export utilities
└── test_*.py                 current geometry and release gates
```

The distinction is deliberate:

- `floor_stand/`, `no_floor_stand/`, and `wings/` are build-system state and
  validation authorities.
- `artifacts/` is the product-selection interface for people. It contains no
  duplicate CAD bytes.
- `images/generated/` contains the small current snapshot packet used by the
  README and catalog. Historical one-off review renders are disposable.

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

Obi-Wan nests `states/{floor,no_floor}` and `wings/{ac,ae}` because those are
real, mutually exclusive configuration choices. The standard/slim shoulder
and wing files remain siblings because each set is an alternative add-on to
the same base.

## Recommended source migration

The flat Python source layout should eventually become:

```text
src/lx521_baffle/             parametric geometry package
scripts/                      CAD export/render/release CLIs
tests/                        unit, geometry, metadata, and remote tests
docs/                         all prose authorities
build/                        ignored intermediate/state output
artifacts/                    curated release facade
archive/                      source-only historical experiments, if retained
```

That migration should be a separate mechanical change. The present Makefile,
remote cache keys, source hashes, tests, and Obi-Wan native-stage manifests all
encode root-relative module names. Moving them during an inventory cleanup
would make geometry provenance harder to audit and could silently rebuild or
orphan state outputs. The product facade solves the immediate usability and
artifact-duplication problem without taking that risk.

## Cleanup policy

Keep source, manifests, current STEP/STL/PNG outputs, and physical
qualification evidence. Ignore or trash viewer GLBs, `__pycache__`, pytest
caches, remote jobs, Make stamps, G-code workspaces, failed slice runs, and
dated one-off review PNGs. Regenerate the product catalog after any promoted
artifact change:

```bash
python3 tools/build_artifact_catalog.py
python3 tools/build_artifact_catalog.py --check
```
