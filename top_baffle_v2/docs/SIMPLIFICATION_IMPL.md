# Top-baffle staged simplification IMPL

- **Date:** 2026-07-23
- **Status:** Accepted — Stages 0–4 implemented and release-scale audit passed
- **Source:** `docs/SIMPLIFICATION.md`
- **Review mode:** Deep — five risk-selected reviewers
- **Scope:** Preserve the completed Stage 0–1 reuse work, then implement Stages
  2, 3, and 4 sequentially: compact Make without changing its execution graph;
  mechanically migrate source, scripts, tests, docs, and generated state into
  the proposed hierarchy; and split the four named monoliths without changing
  released geometry or behavior.

## Requirements and locked decisions

1. Keep GNU Make as the only scheduler. Stage 2 may factor repeated text, but
   every public target, prerequisite edge, stamp, selector, recipe expansion,
   environment value, remote allowlist entry, and local/remote distinction must
   remain equivalent. Stage 4 may retarget R6F selectors to stable case IDs only
   when the test registry and per-case stamp graph change together.
2. Preserve `LX_STAND_FOOT` and `LX_ROUTING_PROFILE` as process-scoped inputs,
   fail-closed Obi-Wan staged BREP exports, source/hash provenance, atomic
   promotion, front-face-down sidecars, and the exact `to_print/` mirror model.
3. Consolidate only behavior proven equivalent in the current tree. Keep
   domain rules, validation order, and special transaction semantics with
   their established owners.
4. Keep change-detection goldens independent: code may import an authority,
   but tests must compare it with one explicitly named expected value rather
   than reusing the production value on both sides of an assertion.
5. Treat the current dirty tree, including the untracked source/IMPL, shelf
   implementation, shelf files, and three untracked concept generators, as
   user-owned baseline state. Do not overwrite unrelated edits, alter index
   state, commit, deploy, contact a printer, or mutate `to_print/`.
6. The user accepts only the existing `UM route / LM pads` actual-BREP result:
   0.260 mm no-floor and 0.357 mm floor versus the general 0.370 mm gate.
   Encode this as a label-and-state-specific baseline with explicit tolerance,
   keep reporting the measurements, and fail on regression or any other
   clearance exception. Do not change route/pad geometry to obtain acceptance.
7. Stage 3 is a mechanical path migration. Preserve emitted artifact basenames,
   labels, schemas, transform contracts, and product inventory while updating
   every Make path, import, source fingerprint, manifest source path, remote
   snapshot rule, documentation link, and generated-state root in one work unit.
8. Stage 4 splits one monolith at a time behind its established public module or
   CLI facade. Move existing ownership; do not redesign geometry, validation,
   slicing policy, publication transactions, or process scheduling.

## Non-goals

- No manifest vocabulary/schema unification. The native-stage v7, state
  release v10, wing v2, captive-release v1, artifact v1, and print-shelf v1
  consumers remain unchanged.
- No C7/V0/standalone-V1 retirement, untracked sketch disposition, commit,
  deployment, protected-shelf refresh, or physical-release authorization.
- No attachment/taper unification, multi-tool boolean regrouping, worker-slot
  redesign, wing fan-out, or P0–P8 performance work.
- Do not make `remote_cad.py` depend on a new helper; it remains a standalone,
  stdlib-only transport. Moving it under `scripts/` must preserve that property.

## Completed Stage 0–1 design evidence

- `front_down_contract.py` is the closest proven shared analogue: it is pure
  Python, imported by exporters and release/slicer checks, and already owns
  sidecar validation. The missing piece is one pure constructor for the
  X180-plus-Z transform record now assembled in three exporters.
- `run_memory_guarded.py` already owns guard identity and policy. Creating a
  parallel `guards.py` would duplicate that authority; it needs small public
  helpers for fail-closed model entry and direct-CLI re-exec instead.
- `export_steps.py` already owns STEP transaction validation, and
  `export_obiwan_staged.py` imports its private validator. Making that owner
  public is smaller than adding a second STEP utility.
- `export_piece_stls.py`, `export_coupon.py`, and
  `export_obiwan_wings.py` contain related binary-STL validation,
  transform-zero canonicalization, and strict-topology implementations with
  material caller differences. Their publication order is explicitly asserted
  by `test_release_metadata.py`, so common primitives, caller adapters, and the
  order assertion must move together.
- `captive_magnets.py` is the geometry authority but imports build123d at
  module load. `slice_captive_magnets.py` is intentionally pure/local and
  currently retypes the default dimensions. A pure contract module is needed;
  directly importing the CAD module into the slicer is not acceptable.
- `export_obiwan_staged._source_fingerprint()`,
  `write_obiwan_release_manifest.generation_source_paths()`, wing
  `INTERFACE_SOURCES`, and captive-catalog provenance lists are separate from
  Make prerequisites. Every new transitive helper must enter the affected
  runtime attestation as well as the Make graph or stale staged BREPs can be
  reused under an incomplete source record.
- The source's deletion claim conflicts with one current README sentence that
  says `gen_lm_knife_draft.py` is retained for research. The source decision
  wins for this tranche, but that sentence and the two catalog/report
  exclusions must be corrected in the same cleanup.

## Completed Stage 0–1 reuse map

| Need | Existing owner to extend | Smallest verified gap |
|---|---|---|
| Deterministic file/byte hashes and JSON bytes | Root CLI call sites share the same stdlib idiom | Add `lx_io.py` for pure primitives; keep publisher transactions local |
| Front-down metadata construction | `front_down_contract.py` | Add a pure transform-record constructor; exporters still own CAD placement |
| Default captive-magnet dimensions | `captive_magnets.py` public constants/spec | Extract a pure `captive_magnet_contract.py`, then re-export for compatibility |
| STL write-time validation | `check_manifold.stl_diagnostics` plus three related exporter implementations | Add a common low-level `stl_export.py`; preserve caller adapters and differences |
| STEP transaction validation | `export_steps.py` | Publish and reuse `validate_step_transaction()` |
| Guard entry/re-exec | `run_memory_guarded.py` | Add two narrow helpers; keep host-profile detection local |
| Labeled compounds | Repeated `gen_step()` loops | Add `assembly.py` and convert only simple mapping-to-Compound wrappers |
| Scalar interpolation/distance | Byte-equivalent scalar implementations | Add only `smoothstep01()` and `point_segment_distance()` to `geom_primitives.py` |

### Exact I/O migration boundary

| Primitive | Migrate in this tranche | Retain locally / explicitly exclude |
|---|---|---|
| `sha256_file()` | Hash-only use in `front_down_contract.py`, `write_obiwan_release_manifest.py`, `slice_captive_magnets.py`, `check_manifold.py`, `generate_captive_magnet_catalog.py`, `export_obiwan_wings.py`, `export_obiwan_staged.py`, `export_piece_stls.py`, `build_to_print_shelf.py`, `top_baffle_nd25fw4_obiwan_floor_strength.py`, and `test_obiwan_wings.py` | Keep a compatibility alias only where another current module imports that public name |
| `sha256_bytes()` | `slice_captive_magnets.py` and `build_to_print_shelf.py` | `remote_cad.py`; `tools/build_artifact_catalog.py` hashes already-rendered manifest bytes |
| Pretty deterministic JSON bytes | Caller wrappers in piece export, staged export, slicing, shelf, and front-down sidecars may reuse serialization only, with explicit `allow_nan` behavior | Candidate rendering, readback, validation, temporary naming, `fsync`, replacement, cleanup, and multi-file publication stay with each producer |
| Publisher transaction | None in this tranche | Keep `generate_captive_magnet_catalog.py` (`mkstemp`/`fsync`/readback), `front_down_contract.py` (pre/post validation), wing transaction-directory writes, staged unique-temp publication, slice/shelf fixed-temp policy, `remote_cad.py`, and `tools/build_artifact_catalog.py` unchanged |

## Completed Stage 0–1 work units

### WU1 — Freeze the baseline and remove only approved obsolete sources

1. Record scoped `git status`, representative forced dry-run output, public
   target/prerequisite facts, current test/artifact inventory, and a byte/hash
   inventory of the user-owned `to_print/` tree before edits. Record the
   existing `tools/build_artifact_catalog.py --check` failure for stale
   `artifacts/standard/manifest.json` as baseline, not as a new regression.
2. Delete only `gen_c_variants.py`, `gen_lm_knife_draft.py`, and
   `gen_um_knife_draft.py`. Remove their stale narrative/exclusion entries from
   `README.md`, `CAPTIVE_MAGNET_MIGRATION_REPORT.md`, and
   `generate_captive_magnet_catalog.py`. Do not touch the three untracked
   `tools/gen_obiwan_*entry*.py` sketches or caches.
3. Preserve the untracked `build_to_print_shelf.py`,
   `test_to_print_shelf.py`, and `to_print/` content and index status.

### WU2 — Single-source pure I/O and print-transform behavior

1. Add dependency-free `lx_io.py` with `sha256_file()`, `sha256_bytes()`, and
   explicit pretty deterministic JSON-byte serialization. It must not own
   publication or choose temporary paths.
2. Apply only the migrations in the exact boundary table. Retain thin caller
   wrappers where names are imported or error/serialization behavior is part
   of a current contract. Explicitly exclude `remote_cad.py` and `tools/`.
3. Extend `front_down_contract.py` with a constructor from oriented bounding
   minimum plus in-bed Z rotation. Use it from piece, coupon, and wing export
   paths while retaining each path's bed-fit/orientation selection.
4. Add pure tests for file/byte digests, strict/permissive deterministic JSON
   bytes, transform matrices/translations, and malformed transform input.
   Existing producer tests continue to own last-known-good publication,
   candidate readback, and cleanup behavior.

### WU3 — Make captive-magnet dimensions pure and single-owned

1. Add dependency-free `captive_magnet_contract.py` for the purchased/default
   dimensions, validation error/spec, derived land, and nominal paired-face
   separation. Import and re-export those names from `captive_magnets.py` so
   existing CAD callers retain their API.
2. Derive `slice_captive_magnets.RELEASE_SITE_GEOMETRY_MM` from the pure spec.
   Keep its design-family pair-separation map local because the 1.09/1.10 mm
   values include caller-owned interface offsets, not only the default spec.
3. Retain one independent named golden mapping in tests and assert both the
   CAD and slicer views match it. Add an import gate proving that importing the
   slicer/contract does not load `build123d` or `OCP`.

### WU4 — Reuse export, guard, assembly, and scalar helpers

1. Add `stl_export.py` only for the common binary-length, coordinate rewrite,
   and topology-defect primitives. Keep thin adapters in all three exporters:
   piece/coupon retain the positive-epsilon rejection and their error text;
   wings retains its current tolerance behavior and JSON-normalized facts;
   piece retains its exact-apex sanitizer and support-blocker component policy.
   Tests must freeze each difference before deleting duplicated bodies and
   still assert `export -> byte validation -> zero canonicalization ->
   optional exact-apex removal -> strict topology -> atomic replace`.
2. Rename the private STEP validator in `export_steps.py` to a public symbol
   and import it from V1L staged, Obi-Wan staged, and wing exporters.
3. Extend `run_memory_guarded.py` with narrow require/re-exec helpers and
   convert only equivalent preambles/model guards. Do not merge
   `_large_host_execution()` or change profile/slot policy. Add subprocess
   tests for unguarded command/argument construction and exit propagation,
   guarded no-op behavior, recursive-reexec prevention, and a static assertion
   that eager CAD imports remain after the re-exec gate.
4. Add `assembly.py` for ordered labeled compounds and use it only where
   `gen_step()` is a direct mapping-to-Compound wrapper. Preserve explicit
   guards, part filtering, labels, and specialized proxy assemblies.
5. Add pure `geom_primitives.py` containing only clamped scalar smoothstep and
   point-to-segment distance. Leave NumPy-returning polar/cubic code and
   domain-specific coordinate helpers local.
6. Add every new owner to the applicable `SRCS`, `OBIWAN_SRCS`,
   `STL_EXPORT_SRCS`, wing input, release input, and test prerequisites. Also
   update every affected non-Make registry: staged `_source_fingerprint()`,
   release `generation_source_paths()`, wing `INTERFACE_SOURCES`, and captive
   catalog source provenance. Add `lx_io.py` and
   `captive_magnet_contract.py` to
   `slice_captive_magnets.AUDIT_SOURCE_FILES`. Because V1L, staged Obi-Wan,
   and wing exporters will import the public STEP validator, add
   `export_steps.py` to each affected Make prerequisite and source-attestation
   list, including `OBIWAN_WING_INPUTS`/wing `INTERFACE_SOURCES`. Add focused
   tests proving each dependency is recorded and a dependency-byte change
   alters the corresponding fingerprint.

### WU5 — Verify Stage 0–1 and record evidence

Run fast checks first, then the release-scale oracle:

1. New shared-infrastructure and guard-entry tests,
   `test_release_metadata.py`,
   `test_slice_captive_magnets.py`, `test_remote_cad.py`,
   `test_to_print_shelf.py`, and guarded `test_captive_magnets.py`.
2. Representative forced dry-runs to confirm the new prerequisites without
   recipe/DAG drift, followed by `git diff --check`, duplicate-definition and
   import-boundary scans, and a comparison with the initial `to_print/` byte
   inventory. Do not run `make to_print_validate`: its nominal validation mode
   relinks/prunes files and rewrites the protected shelf manifest.
3. **Before launching candidate**, resolve every destination returned/promoted
   for `candidate`, including `floor_stand/`, `no_floor_stand/`, `wings/`,
   review catalogs, top-level STEP/PNG outputs, and the separately rebuilt
   `artifacts/` facade. Any unexplained dirty or untracked overlap blocks
   promotion; do not launch first and check afterward.
4. If the destinations are safe, run one default remote `make candidate` to
   regenerate source-hash-bearing manifests and run geometry, staged-BREP,
   strict-mesh, metadata, and cross-state gates. Rebuild the artifact facade
   with `tools/build_artifact_catalog.py`, then run its `--check`. If promotion
   is blocked, retain the known baseline catalog failure as pending evidence
   and do not rewrite `artifacts/`.
5. After candidate, perform a genuinely read-only shelf-currentness audit:
   compare `to_print/release_manifest.json`'s release-catalog hash and source
   revision with the current catalog, and verify every recorded source and
   delivered hash. Reconfirm the original `to_print/` byte inventory and index
   status. If the binding is stale, record it as an acceptance blocker needing
   explicit shelf-refresh authorization; do not rewrite, relink, prune, or
   re-slice the protected shelf.
6. Update this IMPL with concise actual files, test results, candidate/catalog
   and read-only shelf evidence, or the exact promotion/currentness blocker,
   and acceptance disposition.

## Stage 0–1 acceptance record

- Apart from explicit new source prerequisites, all public targets, selector
  registries, stamps, prerequisite semantics, env values, remote allowlist
  entries, recipes, and emitted artifact inventories remain unchanged.
- Candidate geometry/mesh/clearance/metadata gates pass for both stand states,
  Stock, Slim, Obi-Wan, and flat/graded. Existing part names, labels, front datum,
  print transforms, bed limits, and `to_print` 39-file mapping are unchanged.
- A single pure owner supplies default captive-magnet dimensions to CAD and
  slicing, while an independent named golden catches unintended edits and the
  slicer import remains OCC-free.
- Shared STL tests retain the exact fail-closed publication sequence and cover
  malformed length, transform-zero bounds, topology failure, and prior-file
  preservation.
- Migrated duplicate definitions disappear; specialized variants remain local
  with a stated semantic difference. `remote_cad.py`, manifest schemas, root
  paths, and generated-view mechanisms are unchanged.
- Every transitive shared helper appears in the affected Make prerequisites
  and runtime source attestations, and focused tests prove it invalidates the
  relevant fingerprint rather than reusing stale staged geometry.
- Only the three approved obsolete generators and their references are
  removed. Untracked sketches, shelf content/status, unrelated worktree edits,
  and physical authorization records are preserved.
- The read-only shelf audit proves its manifest still binds the current release
  catalog and every delivered hash; otherwise implementation stops with the
  exact stale fields recorded and does not mutate `to_print/`.

## Stage 0–1 implementation and verification evidence

Stages 0–4 are implemented and the final release-scale audit passed.

- WU1 removed only `gen_c_variants.py`, `gen_lm_knife_draft.py`, and
  `gen_um_knife_draft.py`, plus their named README/report/catalog exclusions.
  The unrelated untracked `tools/gen_obiwan_d20_entry_concept.py`,
  `tools/gen_obiwan_entry_layout_alternatives.py`, and
  `tools/gen_obiwan_entry_reroute_concept.py` files, plus all unrelated
  worktree/index state, remain untouched.
- WU2 added dependency-free `lx_io.py`, extended
  `front_down_contract.py`, and migrated only the locked hash/JSON/transform
  call sites. Publication, temporary-file, readback, rollback, and strict versus
  permissive JSON policy remain with their callers. `remote_cad.py` and
  `tools/build_artifact_catalog.py` are unchanged.
- WU3 added dependency-free `captive_magnet_contract.py`; the CAD module
  re-exports its compatibility API and the slicer derives its release geometry
  from the pure spec. An independent literal golden and a subprocess import
  probe cover dimension drift and prove that `build123d`/`OCP` stay unloaded.
- WU4 added `stl_export.py`, `assembly.py`, and `geom_primitives.py`; published
  the existing STEP transaction validator; and extended
  `run_memory_guarded.py` with the shared fail-closed entry helpers. Only
  byte-equivalent low-level behavior and direct mapping-to-compound wrappers
  moved. Exporter-specific epsilon errors, exact-apex cleanup, support-blocker
  policy, wing normalization, transaction order, and specialized assemblies
  remain local. Every affected Make prerequisite and runtime fingerprint list
  now records the new transitive owners.
- A cold candidate exposed one missing dependency in the current Make graph:
  `check_to_print_shelf` consumed the release catalog and its meshes before
  they existed. It now depends on `$(CAPTIVE_MAGNET_CATALOG)`, with a pure
  regression assertion. This adds no recipe. The same cold run exposed a
  stale `test_margin_dashboard` expectation left by the current HEAD's R14 LM
  outlet change; its LM mask/radius now matches the already-authoritative
  `test_v1_field` and `test_route_smoothness` contracts. No CAD changed.

Focused evidence:

- `test_shared_contracts.py`: 13 pure gates pass.
- `test_release_metadata.py`: 38 pure gates pass.
- `test_slice_captive_magnets.py`: 52 gates pass.
- `test_remote_cad.py`: all remote transport checks pass.
- `test_to_print_shelf.py`: the 39-entry catalog contract passes (the stronger
  read-only hash audit below intentionally finds stale protected content).
- Guarded `test_captive_magnets.py`: 5 focused gates pass;
  `test_obiwan_lm_split_two_pin_static.py`: 5 gates pass.
- The focused forced dry-run remains byte-identical to baseline. The complete
  forced remote-worker candidate dry-run is 882 lines versus 881 initially;
  the sole added command is `test_shared_contracts.py`. `git diff --check`,
  duplicate-owner/import-boundary scans, and the scoped source audit pass.

Promotion and candidate evidence:

- Before every launch, `floor_stand/`, `no_floor_stand/`, `wings/`, `review/`,
  the top-level STEP/PNG, and `artifacts/` had no modified or unexplained
  untracked overlap. Ignored members were only documented stamps, staged
  BREPs/manifests, facts, and viewer caches; `.remote-cad/backups/` was empty.
- Final cold remote job
  `20260722T211656Z-00285303c4c2-71ec2b` (source
  `00285303c4c2f4ba454b12f7839918d799bfbcb6c3530b82f5b7852e42c4873b`)
  passed the shared, slicing, transport, captive-magnet, static LM split, full
  clearance, dashboard, Stock/Slim export, common STEP, and early Obi-Wan
  gates. It then failed the existing actual-BREP route gate in both states:
  `UM route / LM pads` measured 0.260 mm no-floor and 0.357 mm floor, below
  `INSERT_COVER_CLEAR - 0.03` (0.370 mm). The tranche changes neither route nor
  pad geometry; the only prior recorded `check_bump_brep` job also failed.
  Fixing or weakening this geometry-bearing contract is outside this IMPL.
- The client canceled only the already-failed remaining remote workers. No
  archive was fetched and no local artifact was promoted. Consequently the
  facade was not rebuilt; its read-only check retains the exact baseline
  failure: `stale or missing generated catalog file:
  artifacts/standard/manifest.json`.

Read-only shelf evidence:

- The shelf manifest self-hash, catalog hash
  `27f1b2bf465f0d0fbede1367e3d7f53beea2771a285c3a46b4f34a881d896371`,
  and catalog source revision
  `acb645884a02ff0ec0d9c3b2e1811f9f634b99fa60d4e56b37bc28b26cab0c23`
  all match the current on-disk catalog. Of 236 recorded source/delivery path
  hashes, five baseline-stale fields remain:
  - `obiwan_02_of_16_LM_top_keyed_2_of_2.source_stl` and
    `.delivered_stl`: expected `da9b573e905a358a52b04aa360881bbf4798e1a054b7e1e8993b03c379adb52d`,
    actual `9a92e441876a0623934e9b77012c45a5f3ebeea7b184e773d3975d7880e3a961`;
    its source sidecar expected
    `5782d6e73da1f6ff3426aff013162d7510f012d7b20400562e5af8f92e19f426`,
    actual `58b1f93c3fe1ca6c8e37ea64ca3bf2aa4b8ed161faf6e9e43bec8ebfc567caa2`.
  - `obiwan_03_of_16_UM_carrier_1_of_1.source_stl`: expected
    `e951f0aaeea591035b8713cfc00236fd3b604763665508f8ddab0149b4a715ee`,
    actual `073301c63e82ca318a264856d2898f4a8548f52d5741ec5fb484a5809ade8332`;
    its source sidecar expected
    `70b15f6d9e10a019d52eea2c66f05d3fdfec352407094a804b9c77bfb0825ea7`,
    actual `1fa0aac202c776c6d42deb764bb51c27a458609e706ef8701c5e549f81372b51`.
- No shelf operation ran. All 83 `to_print/` files exactly match the initial
  normalized byte/size/inode/link-count inventory (manifest SHA-256
  `46eb5df5640867dae43d7c33fe0cf71c771e44f8911550e7046f754ca82bd1d2`),
  and scoped index status is unchanged.

Stage 0–1 acceptance was intentionally blocked at the time, not silently
weakened. The later locked decisions supersede the BREP disposition and the
old requirement that this already-stale protected shelf become current. For
Stages 2–4, shelf bytes, paths, and index status are blocking invariants;
historical source-binding currentness and source-side hard-link counts are
reported but informational because the user forbids refreshing the shelf.

## Stage 0–1 risks and retained evidence

- **Dirty generated roots:** final candidate promotion can replace generated
  trees and top-level artifacts. The pre-launch overlap check is mandatory; an
  unexplained overlap blocks candidate/catalog regeneration but not completed
  local refactor evidence.
- **Stale facade at baseline:** the read-only artifact catalog check currently
  fails at `artifacts/standard/manifest.json`. Do not silently repair it before
  source work or treat it as a new regression; rebuild/check only after a safe
  candidate promotion.
- **Shelf currentness versus preservation:** candidate necessarily changes the
  source revision even when STL bytes remain stable. If the preserved shelf
  manifest then fails the read-only binding audit, refreshing it requires new
  user authority and is an implementation blocker, not permission to run the
  mutating `to_print_validate` path.
- **Source references are already stale in places:** the current Makefile is
  1,164 lines and contains ongoing edits beyond the source's cited snapshot.
  Implementation must address symbols and current behavior, not blindly apply
  recorded line numbers.
- **Generated hashes will change:** helper/source edits intentionally change
  source attestations and manifests even when geometry bytes do not. Regenerate
  through the normal pipeline; do not hand-edit hashes or require STEP/STL byte
  identity instead of the established gates.
- **No blocking product decision in this tranche.** Sketch retention, legacy
  product retirement, Makefile compaction, schema migration, package layout,
  geometry unification, performance experiments, commits, and physical release
  remain deferred.

## Stage 0–1 complexity inventory

- New shared modules: `lx_io.py`, `captive_magnet_contract.py`,
  `stl_export.py`, `assembly.py`, and `geom_primitives.py` (five). The I/O,
  magnet-contract, and scalar modules are dependency-free; STL/assembly reuse
  existing checker/CAD dependencies. Each replaces current repeated ownership,
  and the magnet contract keeps local slicing free of CAD imports.
- Extended existing owners: `front_down_contract.py`, `export_steps.py`, and
  `run_memory_guarded.py` (three).
- New services, schemas, queues, secrets, feature flags, migrations, workers,
  runtime dependencies, or schedulers: zero.

## Stage 0–1 review summary

Deep review completed in seven role-passes: five initial roles (simplicity/
reuse/scope, correctness/contracts, tests/operability, compatibility/
migration, and performance/reliability) plus the allowed two targeted
rechecks for scope and contracts. All substantiated High/Medium findings were
required and applied: split Stage 2, preserve I/O/STL caller contracts, cover
every runtime attestation and direct-CLI guard, preflight promotion, omit
mutating shelf validation, record the stale facade baseline, and add read-only
post-candidate shelf binding. No Low notes or unresolved plan blockers remain.

## Stage 2–4 evidence and reuse map

### Stage 2–3 implementation and verification evidence

- Stage 2 scopes the accepted actual-BREP baseline only to
  `UM route / LM pads`: 0.260 mm no-floor and 0.357 mm floor, each with a
  0.005 mm repeatability band. The measured value remains printed and every
  other label/state retains the 0.370 mm gate. Pure AST coverage rejects any
  broader exception.
- The seven Make compaction mechanics are implemented with GNU Make 3.81
  syntax. Normalized target/prerequisite and forced recipe-stream comparisons
  preserve the captured DAG; C7, V0, and V1 remain candidate members.
- Stage 2 cold job `20260723T003318Z-f31aa97c8bab-1b46ae` used source
  `f31aa97c8bab30f08fed0e746f943402fd485f1411bb27084f35df5aac75762c`,
  published a cold Make cache, promoted 348 verified files, and passed 92/92
  state STLs plus 12/12 wing STLs and every release join.
- Stage 3 moved CAD into `src/lx521_baffle/{proud,obiwan}`, CLIs into
  `scripts/`, tests into `tests/`, docs into `docs/`, and generated state,
  wing, and common outputs into `build/`. Direct CLIs bootstrap canonical
  import roots, remote protocol-3 jobs accept both the frozen legacy and new
  transport paths, and promotion treats each state root, wing root, and common
  file independently and atomically.
- The protected shelf is consumed through one read-only frozen legacy-root
  resolver. No obsolete root module or state directory remains authoritative;
  active old-root literals are limited to that resolver/test, legacy transport
  compatibility, invalid-path fixtures, and historical documentation.
- Stage 3 cold job `20260723T021552Z-5f622722957c-d14b3c` used source
  `5f622722957c4953d4f8e8d7ce38a805e9250ffe66a7bffc18a18e359c9cfb47`,
  published a cold Make cache, and promoted 348 verified files. It passed the
  full R6F and dense closure matrices, 92/92 state STLs, 12/12 wing STLs, both
  live flat/graded BREP contracts, release metadata, and the shelf binding gate.
- After promotion the curated artifact facade rebuilt and passed `--check`.
  The protected shelf remained byte-identical to its pre-work baseline: 83
  files, 214,997,136 bytes. Shared/static, slicing, remote transport, direct
  CLI, Make dry-run/database, and `git diff --check` gates pass.

### Stage 4 implementation and verification evidence

- The route checkpoint retains the public/core facade in `route.py` (1,949
  lines), with rear-entry/tube ownership in `rear_entry.py` (894) and
  bump/backfill/burial/cover/shell ownership in `bumps.py` (1,705). Public
  objects are re-exported by identity and all new owners participate in Make,
  staged, wing, and catalog provenance. Focused jobs
  `20260723T033314Z-0a5887a6a334-29f32f`,
  `20260723T034248Z-0a5887a6a334-3b5784`,
  `20260723T034329Z-0a5887a6a334-7987b2`, and
  `20260723T034756Z-0a5887a6a334-3e7ffe` used source
  `0a5887a6a334f715892cf0e70fd6684a7e9adff8cf2be4a20b81481547a70fbc`
  and passed route boundaries, exact BREP clearances, all four shells, and
  the route-facts contract.
- The carrier checkpoint retains construction/finalization in `carriers.py`
  (1,335 lines), with `closure_webs.py` (792), `joints.py` (667), and the
  Obi-Wan `magnets.py` owner (369). Boolean order and facade object identity
  are frozen. Closure job `20260723T035147Z-55c4ad7bdae7-33a561`, no-floor
  carrier job `20260723T042405Z-55c4ad7bdae7-478e88`, and wing job
  `20260723T042552Z-55c4ad7bdae7-2843ef` used source
  `55c4ad7bdae70f4f046540e238e00be3ce08a7a503828d98aa9b29866b2e8363`.
  They passed the complete dense closure/service matrix, exact final LM mesh,
  both live flat/graded BREP contracts, and 12/12 strict wing STLs; 45 verified wing
  artifacts were promoted.
- The slicer is now a 298-line CLI/API facade over `release_validation.py`
  (2,394 lines), `gcode_analysis.py` (1,859), and `artifact_emit.py` (2,088).
  The frozen shelf-consumer attributes retain their signatures and re-export
  owner objects by identity. All four files are Make prerequisites and
  `AUDIT_SOURCE_FILES`; a pure byte-change probe proves an extracted owner
  changes the audit fingerprint. All 52 slicer gates, the read-only shelf
  contract, and OCC-free import/API checks pass.
- `tests/test_harness.py` (185 lines) now single-owns generic guarded process
  dispatch, stable selector lookup, ordered/bounded execution, output capture,
  and failure aggregation. `test_obiwan_r6f.py` retains all geometry/service
  workers and goldens, but its 26 state-only wrappers are replaced by an exact
  ordered 37-record registry. Make stamps and selectors use stable case IDs via
  `LX_R6F_CASE_ID`; legacy `test_*` selectors remain record metadata only.
  Focused case-ID job `20260723T053134Z-8d2a8833cf2e-b085ab` (source
  `8d2a8833cf2ea25f34ab7bc5d3b043d3fb72251baa1c0251d9465122eadcd839`)
  re-proved 0.260/0.357 mm accepted BREP baselines and the general pairs.
  Service job `20260723T053430Z-5ffc9606e8b7-dafa3a` (source
  `5ffc9606e8b72d1a8ae629521adbde6a02fa25d6dcfbf8b90428af076690fd32`)
  passed both complete large-host service matrices.
- Stage 4 pure/static evidence is 42 release-metadata gates, 52 slicer gates,
  13 shared-contract gates, all remote-transport gates, Make registry/database
  checks, direct selected/unknown-case CLI checks, and a read-only exact shelf
  audit. The shelf remains 83 files and 214,997,136 bytes with identical paths,
  hashes, and index status. The temporary mechanical split tool was removed.

### Final Stage 4 release-scale acceptance evidence

- A new remote root proved the environment and Make cache cold before launch.
  Final candidate job `20260723T054819Z-5798519801cd-e6da84` used source
  `5798519801cd5267ad81ccb1fe91cfcc04098e391eef4f7a784248f92ab71e6a`,
  exited zero, published the cold Make cache, and verified/promoted all 348
  artifact files.
- The candidate passed the complete R6F, dense closure, terminal-service,
  staged-BREP, Stock/Slim/Obi-Wan, metadata, and cross-state joins. Both state
  inventories passed strict topology (92/92 STLs), as did flat/graded (12/12 STLs),
  including the live wing profile, mirror, receiver, split, and depth gates.
  C7, V0, and V1 remained in both state inventories.
- The only accepted BREP observations reproduced exactly: `UM route / LM pads`
  was 0.260 mm no-floor and 0.357 mm floor. The unexcepted pairs remained above
  the 0.370 mm general gate: `T route / LM pads` was 0.523/0.515 mm and
  `T route / UM inserts` was 1.131/1.127 mm (no-floor/floor).
- After promotion, `scripts/build_artifact_catalog.py` rebuilt the curated
  facade and its independent `--check` reported current. The protected shelf
  was not refreshed: its pre-work inventory still matches exactly at 83 files,
  214,997,136 bytes, identical paths and SHA-256 values, and zero scoped index
  changes. The read-only shelf contract passes; current hard-link counts are
  44 single-link files and 39 two-link files and remain informational.
- Final local evidence is 42 release-metadata gates, 52 slicer gates, 13 shared
  gates, all remote-transport checks, and the shelf contract. GNU Make 3.81
  expands the exact ordered 37-case registry to 37 targets and 37 stamps;
  route, BREP, and service dry runs use `LX_R6F_CASE_ID`. Active stale-authority
  and legacy-selector scans are empty, 82 Python files compile, 12 split-owner
  modules and the 14-attribute OCC-free slicer facade import, and
  `git diff --check` passes.

- The current 1,158-line Makefile uses `RULES`, `CLEARANCE_CHECK_RULE`,
  `R6F_CASE_RULE`, and nested `$(eval $(call ...))`.
  Those are the analogue for all Stage 2 factoring; no Python scheduler or new
  build generator is needed.
- Baseline forced remote-worker dry runs are captured for `check`, `candidate`,
  `floor_obiwan`, and `check_route_contract`, together with the parsed Make
  database and pre-change Makefile. Stage 2 must reproduce command streams and
  a normalized public target/prerequisite registry exactly.
- `docs/REPOSITORY_STRUCTURE.md` and source §7 define the Stage 3 boundary.
  Existing imports, Make prerequisites, `remote_cad.py` snapshot/allowlist
  behavior, source fingerprints, catalog source fields, tests, and docs all use
  root-relative names, so path changes must be generated from one explicit
  mapping and validated for stale old-path literals.
- The pre-split Stage 4 baselines were 3,532 lines (`obiwan_route`), 2,380
  lines (`obiwan`), 6,586 lines (`slice_captive_magnets`), and 7,070 lines
  (`test_obiwan_r6f`). Their established import/CLI surfaces are preserved by
  the completed cohesive-owner moves recorded above.

| Requirement | Existing owner/analogue | Smallest change |
|---|---|---|
| Scoped BREP acceptance | `_bump_brep_clearance()` | One state+label baseline table and regression tolerance; retain the general gate for every other pair |
| Make compaction | Existing `define`/`eval` families | Add state, focused-check, wing-slug, STL-variant, run-prefix, and recovery macros only |
| Package hierarchy | Current flat import graph + Make source registries | Mechanical path map, package-relative imports, explicit CLI/test `PYTHONPATH`, and regenerated provenance |
| Route split | Current `obiwan_route` public module | Keep `route` facade/core; extract existing bump/shell ownership and rear-entry ownership |
| Carrier split | Current `obiwan` public module | Keep `carriers` facade/core; extract closure webs, joints, and side magnets |
| Slicer split | Current `slice_captive_magnets` CLI/API | Re-export existing API while extracting G-code analysis, release validation, and artifact emission |
| Test split | Existing selector registry and Make stamps | Add generic `test_harness`, case registry/parameters, and stable case IDs without reducing coverage |

### Authoritative Stage 3 path map

| Current path(s) | Canonical destination | Contract |
|---|---|---|
| `top_baffle_nd25fw4.py` | `src/lx521_baffle/base.py` | package import `lx521_baffle.base` |
| `captive_magnets.py` | `src/lx521_baffle/magnets.py` | preserve public magnet API |
| `captive_magnet_contract.py` | `src/lx521_baffle/magnet_contract.py` | dependency-free owner |
| `assembly.py`, `geom_primitives.py`, `lx_io.py`, `front_down_contract.py`, `stl_export.py` | `src/lx521_baffle/{assembly,geom,io,print_contract,stl_export}.py` | shared package owners |
| `top_baffle_nd25fw4_{cables,flush,um_fit}.py` | `src/lx521_baffle/{cables,flush,um_fit}.py` | package imports |
| `top_baffle_nd25fw4_{a_comp,a_comp_assembled,attachments,b,b1,b1_assembled,b2,b2_split,c7,c7_split,v0,v0_split,v1,v1_attachments,v1_split,v1l,v1l_split}.py` | like-named files under `src/lx521_baffle/proud/` | no compatibility root modules |
| `top_baffle_nd25fw4_obiwan.py` | `src/lx521_baffle/obiwan/carriers.py` | canonical carrier facade before Stage 4 |
| `top_baffle_nd25fw4_obiwan_{assembled,attachments,bridge,floor,floor_strength,lm_split,route,split}.py` | like-named files under `src/lx521_baffle/obiwan/` | `route.py` remains the route facade |
| `obiwan_wings_cad.py` | `src/lx521_baffle/obiwan/wings.py` | preserve wing geometry API |
| `bambu_3mf_audit.py`, `build_to_print_shelf.py`, `check_manifold.py`, `export_*.py`, `gen_*.py`, `generate_captive_magnet_catalog.py`, `json_schema_subset.py`, `polar_index_base.py`, `remote_cad.py`, `run_memory_guarded.py`, `slice_captive_magnets.py`, `write_obiwan_release_manifest.py` | like-named files under `scripts/` | importable CLI modules via `PYTHONPATH=$(CURDIR)/src:$(CURDIR)/scripts`; each anchors project assets at its parent project root, not `scripts/` |
| `test_*.py` | like-named files under `tests/` | direct CLI plus package/script paths; Stage 4 adds `test_harness.py` |
| `PRINTING.md`, `VARIANTS.md`, `CAPTIVE_MAGNET_*.md`, `obiwan_*.md`, `SIMPLIFICATION*.md` | like-named files under `docs/` | rewrite internal links and code-path prose; root `README.md` remains |
| `floor_stand/`, `no_floor_stand/`, `wings/` | `build/floor_stand/`, `build/no_floor_stand/`, `build/wings/` | independently promoted generated roots |
| `top_baffle_nd25fw4_attachments.step`, `obiwan_wing_design_map.png` | like-named files under `build/common/` | independently promoted common files; basenames unchanged |
| `tools/build_artifact_catalog.py` | `scripts/build_artifact_catalog.py` | keep self-contained behavior; `tools/` remains for untouched untracked sketches |
| `README.md`, `Makefile`, `.gitignore`, `cad-remote-requirements.lock`, both JSON schemas/profiles | stay at project root | stable entry/configuration boundary |
| `coupons/`, `review/`, `artifacts/`, `to_print/`, `.remote-cad/`, untracked `tools/gen_obiwan_*entry*.py`, caches/transients | stay in place | no protected or unrelated migration |

`scripts/run_memory_guarded.py` remains the single dual-use guard owner; Stage 3
does not create a parallel package `guards.py`. Every moved executable computes
`PROJECT_ROOT = Path(__file__).resolve().parents[1]`. The stale-path oracle is
the table above plus the explicit protected-shelf compatibility literals.

### Authoritative Stage 4 ownership maps

- **Route checkpoint:** `route.py` keeps route constants, centerline/plan
  construction, route point APIs, `route_facts()`, and the compatibility export
  surface. `bumps.py` owns `CoveredBump`, `BumpBackfillSpec`, bump/backfill,
  cover ownership, burial webs, assembled-shell component builders, and their
  private helpers. `rear_entry.py` owns `RearEntryBore`,
  `RearEntryVestibule`, bores/vestibules, rear port/support blocker, tube
  section/global-suffix builders, and cap-relief/internal-cutter helpers.
  Imports flow route-core → rear-entry → bumps; the facade re-exports the
  frozen externally referenced symbols without copied definitions.
- **Carrier checkpoint:** `carriers.py` keeps constants, base ring/spoke
  primitives, carrier construction/finalization, `core_parts()`, `gen_step()`,
  and the compatibility facade. `closure_webs.py` owns closure bands,
  polygons, printable/partition lenses, webs, and plan ownership.
  `joints.py` owns joint-ear/tweeter plans and solids, load facts, receiver
  notches, and joint application. `magnets.py` owns side-magnet sites, pocket
  cuts, and land verification. Dependencies point from those owners toward the
  carrier core; carrier assembly imports their public operations only after
  core constants/primitives are defined.
- **Slicer checkpoint:** `release_validation.py` owns schema/profile/catalog,
  source binding/staging, release completeness, and manifest validation;
  `gcode_analysis.py` owns Segment/Layer/ParsedGcode, parsing, toolpath/profile
  metrics, and closure-layer discovery; `artifact_emit.py` owns evidence
  rendering, Bambu invocation, cache fingerprints, pause/project injection,
  archive validation, and transactional bundle emission. The CLI facade must
  retain the established shelf-consumer attributes `AuditError`,
  `READY_3MF_FILENAME`, `_artifact_profile_bundle`, `_bambu_command`,
  `_find_bambu_binary`, `_gcode_pause_events`, `_profile_value_equal`, `_slug`,
  `_validate_actual_gcode_profile`, `_validate_ready_project_archive`,
  `inspect_stl`, `normalize_catalog`, `parse_gcode`, and `prepare_profiles`,
  with frozen call signatures.
- **R6F-test checkpoint:** `test_harness.py` is initially suite-private and owns
  only generic guarded subprocess dispatch, selector lookup, ordered execution,
  concurrency, output collection, and failure aggregation. The test module
  retains geometry/service worker logic and independent goldens. Before edits,
  freeze the ordered 37-record mapping of case ID, callable, stand state,
  service-orchestrator class, Make stamp, and legacy selector; compare it
  exactly after replacing state-only wrappers with parametrized case records.

## Stage 2–4 work units

### WU6 — Formalize the accepted BREP baseline

1. Add a named mapping keyed only by stand state and `UM route / LM pads`,
   recording 0.260/0.357 mm plus a small documented numeric-repeatability
   tolerance. Continue printing the measured clearance on every run.
2. Apply the general 0.370 mm requirement to every unlisted pair. For the two
   listed observations, require the accepted baseline and fail on downward
   regression; add pure/static coverage proving the exception cannot match a
   different label or state.
3. Run both focused BREP selectors before Stage 2 candidate.

### WU7 — Stage 2 Makefile compaction and equivalence gate

1. Implement all seven source §6 mechanics with GNU Make 3.81 syntax: focused
   R6F aliases/rules, paired Obi-Wan states, `wings/.stamp_%`, the B2/C7/V0/V1
   STL loop, one state-aware run prefix, one recovery macro, and documented
   public iteration-only phony targets. Preserve genuinely distinct V1L,
   Obi-Wan, service, focused-state, and multi-command recipes.
2. Compare normalized pre/post public targets, ordinary/order-only
   prerequisites, and recipe command streams. Forced remote-worker dry runs for
   the four frozen goals must be byte-identical; any intentional textual path
   change belongs to Stage 3, not this unit.
3. Run static metadata/transport/shared tests, `git diff --check`, then one cold
   remote `candidate`. Immediately before candidate and before rebuilding the
   facade, preflight every promoted/deleted destination for unrelated overlap.
   Rebuild/check the artifact facade after safe promotion and record the
   candidate job/source hash before starting Stage 3.

### WU8 — Stage 3 mechanical hierarchy migration

1. Move geometry into `src/lx521_baffle/` (`base`, shared helpers, `proud/`,
   and `obiwan/`); CLIs into `scripts/`; tests into `tests/`; prose authorities
   other than the root README into `docs/`; state outputs into
   `build/{floor_stand,no_floor_stand,wings}`; and the two common generated
   outputs into `build/common/`. Leave `coupons/`, `review/`, `artifacts/`,
   `to_print/`, schemas/profiles, and the untracked concept generators in place.
2. Use package-relative geometry imports and explicit root/src/scripts paths in
   CLI entry points and Make. Update module arguments, target/prerequisite paths,
   clean rules, remote dispatch, cache/snapshot allowlists, source fingerprints,
   release/wing/catalog provenance, artifact-facade sources, tests, docs, and
   manifest path values from the same path map. Preserve emitted artifact
   basenames even though their state roots move under `build/`.
3. Keep `build/floor_stand`, `build/no_floor_stand`, and `build/wings` as three
   independently atomic remote promotion roots. The attachments STEP and wing
   design map under `build/common/` remain independently promoted files, not a
   complete-root transaction, so a focused target cannot delete its sibling.
   Replace second-path-component assumptions with one logical
   root-to-relative-prefix registry used by source exclusion, artifact scans,
   required-output selection, cache deletion coverage, bundling, rollback, and
   promotion. Focused remote tests must prove one state root and each common
   file never replaces a sibling.
4. Moving `remote_cad.py` must keep `.remote-cad/` and its lock rooted at the
   project directory. New snapshots record `top_baffle_v2/scripts/remote_cad.py`,
   while the loader accepts the recorded legacy protocol-3 transport path and
   hash for existing jobs. Test status/wait/resume/fetch/promotion/rollback of
   both metadata forms, including terminal-success jobs not yet promoted. This
   avoids canceling, abandoning, or eagerly promoting unrelated legacy jobs.
5. The protected shelf is a frozen historical delivery, not a current generated
   view during this migration. Do not edit it. Add one read-only compatibility
   resolver that maps its legacy `floor_stand/`, `no_floor_stand/`, and `wings/`
   source literals to the canonical `build/...` roots; use it only at shelf
   consumption/audit boundaries and test that new release catalogs still bind.
   Preserve shelf file bytes, paths, and index status; source-side promotion may
   naturally change hard-link counts, which is not a shelf-content mutation.
6. Prove old root module names and old state directories no longer act as
   authorities, while direct documented CLIs and remote snapshots still work.
   Preflight old/new promotion and deletion destinations, regenerate state trees
   remotely, run one cold `candidate`, preflight and rebuild/check `artifacts/`,
   and record the one-time cache/path migration. Do not touch `to_print/`.

### WU9 — Stage 4 monolith splits, one checkpoint per file

1. Split `lx521_baffle/obiwan/route.py` into route/core, bump/shell, and
   rear-entry modules. Preserve the `route` facade exports, constant identity,
   construction order, and route facts; run route, bump, mouth, burial,
   backfill, shell, and staged-fingerprint gates.
2. Split `lx521_baffle/obiwan/carriers.py` into carriers, closure-webs, joints,
   and magnets. Preserve carrier public exports and Boolean/validation order;
   run carrier, joint, closure, split, wing, and source-fingerprint gates.
3. Split `scripts/slice_captive_magnets.py` into `gcode_analysis`,
   `release_validation`, and `artifact_emit`, retaining a thin CLI/API facade,
   transaction ordering, cache fingerprints, schema vocabulary, and OCC-free
   imports. Run the full slicer and shelf-contract tests without shelf writes.
4. Add `tests/test_harness.py`, move only generic guard/process/cache/runner
   orchestration from `test_obiwan_r6f`, replace state-only wrapper functions
   with data-driven case records, and retarget Make selectors/stamps to stable
   case IDs. Preserve independent goldens, per-case process isolation, service
   worker behavior, direct-CLI coverage, and the exact case set.
5. Treat each of the four splits as its own reviewable checkpoint. Freeze its
   symbol/export/import-direction map first; update every affected Make source
   prerequisite and runtime fingerprint/attestation registry; prove a byte
   change in one newly extracted owner invalidates the relevant fingerprint;
   then run focused checks and the import/API inventory before continuing.
   After all four, compare artifact inventories and run full static/check suites.

### WU10 — Final release-scale audit

1. Preflight every candidate promotion destination for unrelated overlap, then
   run one cold remote `candidate` for completed Stage 4. The only permitted
   release-gate deviation is the two recorded BREP baseline observations; all
   other geometry, manifold, mesh, metadata, wing, and cross-state gates pass.
2. Preflight again, then rebuild and check the curated artifact facade. Perform
   a read-only audit of the frozen protected shelf and prove its file bytes,
   paths, and index status are unchanged; report source-side hard-link counts
   and historical currentness separately without refreshing it.
3. Run stale-import/path scans, normalized Make registry and selector/case
   inventory checks, `git diff --check`, and a requirement-by-requirement audit.
   Update this IMPL with actual moves, split ownership, line counts, candidate
   job/source hashes, tests, artifact evidence, and any retained baseline note.

## Stage 2–4 acceptance criteria

- All seven Make compaction mechanics are present and Stage 2 preserves the
  frozen DAG and recipe streams; GNU Make 3.81 remains supported.
- The §7 hierarchy is real, all executable/import/provenance paths resolve from
  their new owners, remote cache rebuild/promotion succeeds, and no obsolete
  root module or state directory remains authoritative.
- Each of the four monoliths is materially split into the named cohesive owners
  with its old public behavior available through the new canonical facade; no
  duplicate implementation remains behind.
- The R6F case inventory and per-case stamp granularity are unchanged after
  selector-ID migration, and all focused/static/full candidate checks pass.
- Only `UM route / LM pads` in the two recorded states uses the accepted BREP
  baseline; its values remain observable and cannot regress silently. Every
  other release criterion remains fail-closed.
- C7/V0/V1 stay in the candidate graph; schemas, artifact basenames, product
  inventory, physical authorization, unrelated changes, sketches, and all
  protected `to_print/` content remain unchanged.

## Stage 2–4 review summary

The five-role initial Deep review and two targeted rechecks completed. Its
substantiated findings were
required: freeze the shelf as historical and add a read-only legacy-path
resolver; preflight every promotion/facade rebuild; enumerate the path and
split maps; preserve nested promotion-root atomicity; reconcile legacy remote
jobs before transport relocation; freeze the 37-case and slicer-facade APIs;
and register every extracted owner in Make plus runtime provenance. Two
recheck findings were also applied without another review loop: common outputs
remain independent file promotions, legacy protocol-3 jobs remain resumable and
promotable after tool relocation, and historical shelf currentness is explicitly
informational. No unresolved review blocker or non-blocking Low note remains.

## Stage 2–4 complexity inventory

- New package/subpackage markers and the four source-required split owners are
  path/ownership changes, not new services or runtime systems.
- New services, schemas, queues, secrets, feature flags, schedulers, runtime
  dependencies, or deployment mechanisms: zero.
