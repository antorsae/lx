# Documentation index

- [`PROJECT_SCOPE.md`](PROJECT_SCOPE.md) — concise intent, three-product
  inventory, CAD brief, assumptions, and release boundary.
- [`REPOSITORY_STRUCTURE.md`](REPOSITORY_STRUCTURE.md) — current layout,
  artifact contract, and safe source-migration target.
- [`../VARIANTS.md`](../VARIANTS.md) — detailed geometry and compatibility
  matrix.
- [`../PRINTING.md`](../PRINTING.md) — material, slicer, insert, magnet, and
  assembly instructions.
- [`../CAPTIVE_MAGNET_SLICING.md`](../CAPTIVE_MAGNET_SLICING.md) — pause and
  embed workflow.
- [`../obiwan_physical_qualification.md`](../obiwan_physical_qualification.md) —
  fail-closed physical qualification record.
- [`../obiwan_acoustic_wings_spec.md`](../obiwan_acoustic_wings_spec.md) — Ac/Ae
  wing design authority.

The older engineering authorities stay at repository root for now because the
Make DAG and source-attestation manifests hash those exact paths. Moving them
without a coordinated manifest migration would invalidate otherwise identical
candidate artifacts.
