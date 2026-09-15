# Review changes — 2026-09-05

The delivery and documentation defects from the review have been repaired.
Physical qualification remains pending; fixtures and a measurement procedure
are now supplied. No acoustic performance or printed-strength pass is claimed.

| Review item | Action / evidence |
|---|---|
| Obsolete ready core jobs | Moved both single-material structural G-code projects out of the print shelf into `review/retired_delivery_2026-09-05/`, with original hashes. |
| Stale facade and counts | Rebuilt `artifacts/`: 144 current links, zero broken links; check now rejects obsolete links. Reconciled the 42 choices, 68 sliced jobs and eight GUI projects. |
| Confusing file selection and material names | Added `BUILD_GUIDE.md` and generated `to_print/FILE_GUIDE.md`, with configuration rules, combo substitutions, hardware quantities and slicer estimates. Renamed the two `petg-cf` folders to `petg-gf`. |
| Inconsistent README / rendering | Reordered README around the builder workflow, corrected current materials and registration dimensions, added bare/exploded CAD views, and replaced triangle sorting with depth-buffered rendering of hash-checked canonical meshes. |
| Cross-lane deletion | Shared `delivery_contract.py` defines delivery kind, paths and ownership. The default publisher preserves the independent 0.6 lane; the 0.6 publisher owns its own cleanup. |
| Mutating validation | `--validate-only` and `make to_print_validate` are read-only; publication uses an explicit separate option. A real shelf check confirmed 186 file/directory entries unchanged, including bytes, names and mtimes. |
| Missing driver reference | Centralized reference paths and vendored the existing MU10 reference assets; W22 resolves to its already-vendored STEP. The prior missing-reference failures are fixed. |
| GUI project blind spot | Added geometry/source/settings/hash binding for all eight projects. The audit found a floor/no-floor basename lookup bug: three standalone GUI projects were using the wrong/stale meshes. Fixed state lookup, re-exported the three projects and replaced seven stale shelf STL copies. |
| GUI slice handoff | Added `audit_gui_slice.py` for exported sliced 3MF, composing the existing machine, cavity, pause, duct and support gates plus actual filament-extrusion checks. It imports only into review storage. No current two-material GUI-exported G-code was supplied, so a successful complete GUI import has not been exercised on that output. Unsliced input rejection and material-assignment regressions were tested. |
| Standalone distribution | `make delivery_package` validates and creates `dist/lx521-print-pack.zip` with actual files, orientation/plate authorities, guides, fixtures and checksums; the archive is verified after writing. This is a print pack, not a complete source/CAD build archive. |
| Thin covers and pin/socket walls | Added current full-pitch registration and process fixtures, inspected STEP snapshots, and re-measured current carriers. The outer-body p05 values remain about 0.8 mm, below generic wall-screening values. Retained the functional mating/lumen geometry and made this an explicit, pending process qualification. |
| Magnet skins, purge, support removal, inserts, creep and load | Added a current PETG-GF procedure and worksheet for both stand states. These require actual slicing, printing and measurements. No results were invented. |
| Photos / acoustic evidence | Added the required capture and measurement protocol. Real printed photos and raw acoustic comparisons remain pending. |

Validation: 42 current production meshes plus four fixture meshes are
watertight, consistently wound, positive-volume and within a 256 mm envelope.
The current eight GUI mesh audits pass with their canonical sources and
placement/blocker/modifier inventories. The legacy shelf regression passes,
and the artifact freshness check passes. The six selected pytest modules pass 197 tests. Package checksums are
written beside the verified archive.

Existing CAD/slice-generation provenance is retained. This update does not
relabel historical geometry as a newly regenerated CAD release. A future
geometry build must regenerate its source fingerprints and slice evidence
through the normal candidate pipeline. Both Obi-Wan stand states remain
`release_authorized: false`; print-pack completeness is a separate check.
Future magnet metadata now derives skin/allowance prose from its specification
and refers to the closure's seating datum; the stale hard-coded 0.45 mm text
is removed from the generator. Existing historical CAD facts are retained.
