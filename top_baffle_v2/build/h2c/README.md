# H2C build authorities

The user-facing file guide is [to_print](../../to_print/README.md).
This directory stores native geometry, frozen profiles, print provenance and
digital qualification reports.

- `STEP/` and `STL/`: explicit H2C exports. Stock/Slim uses
  `src/lx521_baffle/h2c/proud.py`; the full Obiwan LM and regular wings reuse the
  existing canonical native solids. Full Stock/Slim floor LM exports are
  intermediate geometry for the released lower/upper pair.
- `contracts/` and `support_blockers/`: native bore definitions and nonprinting
  cable/insert keepouts. `inputs/` contains their positioned print copies.
- `profiles/`: resolved H2C machine, process and material settings, plus the
  original Bambu profile inheritance sources. `print_policy_h2c.json` owns the
  printer-specific settings; `print_policy.json` owns shared hardware and infill.
- `jobs/`: current process inputs, exact CLI commands and per-job audits.
  Disposable slicer work files are excluded from Git. Validation can recover
  plain G-code from each hash-verified published `.gcode.3mf`.
- `plates/`: one audit per combined per-product file, listed in
  `to_print/plates.json`. Per-plate workspaces and G-code stay local.
- `review_models/` and `views/`: source-positioned native assemblies, inspection
  reports and orthographic review images.
- `experiments/`: development evidence outside the print shelf.

The migration also preserves both existing `build/<stand-state>/.obiwan_stage/`
transactions in Git. They are native input authorities for the continuous LM,
including the complete manifests and referenced BREP files. The exporter checks
the state, transaction and hashes before importing them. Changing shared Obiwan
CAD requires regenerating those canonical inputs before rebuilding H2C outputs.

The Dayton ND25FN-4 vases and the dipole tweeter parts are approved mesh
packages (`candidates/nd25fn4_vase/`, `candidates/dipole_tweeter/`) in the
installed frame. Like every other released mesh, they reach the shelf through
the shared front-down print sidecar contract. Every shelf STL must pass the
strict manifold check (`scripts/check_manifold.py`) as written. These
mesh-boolean parts carry sub-micron slivers that float32 bed coordinates would
collapse, so the shelf export collapses edges shorter than `SHELF_SLIVER_MM`
first, and their print authority records it. The retired V4 waveguide body,
its caps and retainers, and its wings are archived under
`experiments/retired_dayton_nd25fn4_v4/`.
