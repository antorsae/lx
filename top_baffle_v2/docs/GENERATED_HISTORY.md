# Generated-file history

The 2026-09-15 cleanup trims only the 18 unpublished commits after
`3ae20d9`. Published history stays unchanged. Every original source revision,
author, date and commit message is retained; the commit IDs change because
archived and superseded generated files have been removed from their trees.

The pre-H2C source checkpoint is **`e3962a4`**, formerly `c261f32`.
The original checkpoint labels in frozen catalogs and audit reports remain
unchanged so the recorded evidence is not silently rewritten. Use the mapping
below to find the corresponding published source revision. Intermediate
commits contain source history, not a complete collection of past binary
releases; rebuild old exports from the matching source when needed.

## What remains

- Current H2C and earlier printer-facing meshes, editable projects and sliced
  projects. Their geometry and process bytes are unchanged by this cleanup.
- Current CAD authorities, native inputs, profiles, print policies, reports,
  manifests, curated review images and qualification evidence.
- The original retained design package and manufacturer references. The ZIP
  is an immutable source input, not an additional release bundle.
- Explicit review inputs needed by the H2C build, including the Dayton support
  blockers, infill modifier, discovery project and accessory preparation.

Archived review slices, raw G-code, redundant workspaces, trial export meshes
and superseded generated versions are omitted from the unpublished history.
Their source/configuration and compact reports remain. Historical reports may
name local-only workspaces and record hashes of files no longer distributed.
The current H2C catalog references retained, hash-verified inputs.

The original 18 commits are recoverable on the original workstation through
`refs/backup/baffle-before-generated-history-trim-20260915`. Files removed from
tracking remain locally available and ignored. This recovery ref is not
pushed. Nothing is removed from already-published history.

## Storage policy

Current print artifacts use ordinary Git. No CAD, STL, 3MF, G-code or image
LFS rules are added; the repository's existing `.mdat` rule is unchanged.
Archive and slicer-cache exclusions are in the project `.gitignore`, with
explicit exceptions for live inputs. Keep one current reviewed export on the
release shelf and keep disposable slicing workspaces out of commits.

The retained outgoing blob storage estimate fell from 4.77 GB to 1.51 GB
(68% less). These are local stored-blob totals, not an exact network-pack size.
Already-published data and the local recovery copy are unaffected.

## Fresh-checkout validation

From `top_baffle_v2`, use the project's Python environment. The retained
source package is extracted locally and verified before importing the Dayton
geometry helpers; extraction does not rebuild the approved meshes:

```sh
PYTHONPATH=candidates/nd25fn4_crescent python3 -c 'from rebuild import verify_package; print(verify_package())'
python3 scripts/validate_h2c_catalog.py
```

The H2C validator restores disposable plain-G-code caches from the qualified
`.gcode.3mf` files after verifying their hashes. It checks all 45 print jobs,
including the 45 editable projects, 45 slices, 40 distinct meshes and their
retained input authorities. No printer connection is needed.

The earlier artifact facade also links ignored oversized CAD assemblies and
a graded wing plate. A fresh checkout needs to regenerate those optional
local exports before `scripts/build_artifact_catalog.py --check` can pass.
They are not required by the H2C print catalog. See
[repository structure](REPOSITORY_STRUCTURE.md#cleanup-policy).

## Commit mapping

| Original local ID | Rewritten ID | Source change |
|---|---|---|
| `d6ce1e1` | [`8b89640`](https://github.com/antorsae/lx/commit/8b896409f81cc6a8b751b2106a22c53f7b5eca5f) | Slow PETG-GF bridges to 12 mm/s so large duct roofs stop pitting |
| `b32db8d` | [`308f01e`](https://github.com/antorsae/lx/commit/308f01e4c7121c4da5113ece1b3d5df1afa5c059) | Close the two print openings on the keyed floor stand |
| `01dc042` | [`0ea2bf6`](https://github.com/antorsae/lx/commit/0ea2bf62e15d6f324666f6b42d719bb6155a7a31) | Extend the floor stand's flat foot to 130 mm so it cannot tip forward |
| `e03cbc2` | [`e6c9061`](https://github.com/antorsae/lx/commit/e6c9061feb381b1bb16a2dc7f7b1f4925c83ec70) | Move the NL8 service opening to the foot's underside and add an M5 floor anchor |
| `6b8b336` | [`f9f6fa4`](https://github.com/antorsae/lx/commit/f9f6fa4e1ca7790e3c2f7d6f154a5550fcadfeed) | Hollow the boss, anchor the stand with two M5 inserts, bury the LM exit |
| `40ad12b` | [`5c3ddce`](https://github.com/antorsae/lx/commit/5c3ddce812155291de61eede793583f541e6b1c9) | Cut the duct voids as 16-gons so their crowns print in GF-PETG |
| `c4922ec` | [`9dae75f`](https://github.com/antorsae/lx/commit/9dae75fe494c5a8e48bf82edd371246f6b70fa0c) | Remove stale NL8 concept scratch from review |
| `23b0533` | [`d6377a9`](https://github.com/antorsae/lx/commit/d6377a92d63bf660d74b933a1f2c5e5e7a70702e) | File the Purifi, SEAS, Peerless, Dayton and Tectonic driver references under vendor/ with provenance |
| `644839c` | [`d502444`](https://github.com/antorsae/lx/commit/d502444376f10f47943ea987c0e95ea091ac0e5b) | Add the candidate Purifi PTT1.3 dipole crescent generator and artifacts |
| `f6f8ac9` | [`c8d68ee`](https://github.com/antorsae/lx/commit/c8d68eeda07e0f3acf2edf6e4d9608acca2e0c87) | Fasten and duct the PTT1.3 dipole crescent: M3 inserts, minimal ducted foot |
| `77a0a1c` | [`51271ff`](https://github.com/antorsae/lx/commit/51271ff97a52f9f0bc30b9035b10c4e67c07578c) | Route the crescent cables as twin buried arcs and mate the released UM joint |
| `c4a25d8` | [`17a79a3`](https://github.com/antorsae/lx/commit/17a79a3f2b19453cccdc6c17269615897e7b9336) | Bore the driver inserts with the released M3 receiver recipe, mesh-verified |
| `e9e46b4` | [`8a2c66a`](https://github.com/antorsae/lx/commit/8a2c66aae0cb8acd8b1be91f914b6ad9839b1e80) | Thread-form the rim screws into printable pilots and fix the phantom probe |
| `0f0ce1f` | [`d5c7e8f`](https://github.com/antorsae/lx/commit/d5c7e8f406e5c0df63ca8de5713ce964e44d3e03) | Catalog every heat-set insert site and the three insert types |
| `c261f32` | [`e3962a4`](https://github.com/antorsae/lx/commit/e3962a4811035ac1802c3f43451cf9236337062f) | Checkpoint baffle experiments, V4 print revisions and H2 printer studies |
| `07206ed` | [`bda62b3`](https://github.com/antorsae/lx/commit/bda62b3bc20cc0a0e16e148d706a7672a83e43bd) | Generate H2C baffle parts and validated dual-nozzle print release |
| `f35010c` | [`ef44120`](https://github.com/antorsae/lx/commit/ef4412006894812f5aae7a5e22a1be361040f49f) | Document three tweeter families and name Dayton waveguide deliverables |
| `3f835e3` | [`e0bf1cc`](https://github.com/antorsae/lx/commit/e0bf1cc24f806127ffec6a771eefcf6e184bfd10) | Keep local credentials out of baffle release artifacts |
