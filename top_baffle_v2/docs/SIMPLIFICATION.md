# SIMPLIFICATION.md — code reuse, cleanup, and file-hierarchy proposal

Review scope: how this project builds its STEP / 3D-print (STL + `.gcode.3mf`)
artifacts and the `to_print/` shelf, across the three shipping products —
**Stock** (B2 proud family), **Slim** (V1L bottom+mids + V1 vase), and
**Obi-Wan** (R6F two-carrier system). Everything below is grounded in the
current tree (line references are as of this commit); nothing has been changed
yet. Companion context: `docs/REPOSITORY_STRUCTURE.md` already sketches a
`src/` migration; §7 turns that sketch into a staged plan.

---

## 1. The build today (orientation)

Sixty-odd Python files (~62 k lines) sit flat at the repo root next to a
1,115-line remote-first Makefile. Two import-time env vars select all
geometry state:

| Axis | Env var | Values | Read at |
|---|---|---|---|
| Stand state | `LX_STAND_FOOT` | `1` → `floor_stand/`, `0` → `no_floor_stand/` | `top_baffle_nd25fw4.py:98` |
| Routing family | `LX_ROUTING_PROFILE` | `proud` (R6P), `obiwan` (R6F) | `top_baffle_nd25fw4_cables.py:70` |

Build flow per product (each runs twice, once per stand state):

```
Stock   b2_split.pieces() ──► export_piece_stls.py --variant b2 ──► <state>/stl/lx521_top_base_*.stl
        attachments / um_fit grommet ──► addonA_*, addonB1_*, grommet STLs
Slim    v1l_split.pieces_v1l() + v1_split/v1_attachments ──► --variant v1l / v1
ObiWan  export_obiwan_staged.py stage ──► <state>/.obiwan_stage/manifest.json (hashed BREPs)
        ──► export_piece_stls.py --variant obiwan --obiwan-part {lm,lm_split,um,tweeter}
        ──► export_obiwan_staged.py step --kind {split,lm_split,attachments,assembled}
Wings   export_obiwan_wings.py --slug {ac,ae} ──► wings/{ac,ae}/
STEPs   generic rule ──► export_steps.py <module>.gen_step()
Shelf   make to_print: captive-magnet release audit ──► build_to_print_shelf.py
        ──► to_print/{stock,slim,obiwan}/{stl,3mf}/  (39 files, hard-linked, SHA-256-bound)
```

`make` dispatches to osado.lan via `remote_cad.py` by default; every heavy
recipe runs under `run_memory_guarded.py`. C7 / V0 / standalone-V1 are legacy
experiments: still built, checked, and manifold-swept on every release, but
excluded from `to_print/` (`build_to_print_shelf.py:52-55`).

## 2. What already works well — do not break

These are deliberate design decisions the cleanup must preserve:

- **Fail-closed Obi-Wan exports.** `export_steps.py` refuses `obiwan*`
  modules; STL export demands a hash-verified stage manifest. Keep.
- **`front_down_contract.py`** is the one properly shared module
  (11 importers). It is the template for how the rest should look.
- **`to_print/` as a view, not a source.** Catalog entries bind to the
  release audit by SHA-256 of STL + sidecar; hard links, prune-to-exact-
  mirror, atomic transactions. Keep the mechanism; §5 only deduplicates its
  helpers.
- **Make as the only scheduler** (stamp files + jobserver, no second Python
  scheduler). Keep.
- **Change-detection literals in tests.** The golden numbers in
  `test_obiwan_r6f.py` intentionally re-state model constants so an
  unintended model edit fails the gate. Refactors must keep the *pin*
  (assert against one named golden), not delete the literal.
- **Constraints:** GNU Make 3.81 compatibility (macOS default make); env-var
  state selection is load-bearing for per-process isolation; remote cache
  keys, stage manifests, and tests encode **root-relative module names** —
  which is why the hierarchy change in §7 must be one mechanical, standalone
  commit.

## 3. Duplication inventory (the evidence)

Roughly **1,000–1,200 lines of genuine copy-paste** across the Python layer,
plus ~40 collapsible Makefile targets. Grouped by layer:

### 3.1 Model layer (`top_baffle_nd25fw4*.py`)

| # | Duplicate | Where | ~Lines |
|---|---|---|---|
| M1 | `gen_step()` Compound-assembly boilerplate (`for label, solid in pieces(): …`) | 15 files: `_b2_split.py:315`, `_c7_split.py:21`, `_v0_split.py:24`, `_v1_split`, `_v1l_split`, `_v1l`, `_obiwan_split.py:36`, `_obiwan`, `_obiwan_assembled`, `_obiwan_lm_split`, `_obiwan_attachments`, `_attachments`, `_a_comp_assembled`, `_b1_assembled`, `_v1_attachments` | 150 |
| M2 | `_smoothstep` cubic (3u²−2u³), byte-identical | base `:235`, `_c7.py:52`, `_v0.py:138`, `_v1l.py:43`; aliases `_cables.py:257`, quintic `_obiwan_route.py:598` | 25 |
| M3 | Rear-taper cutter machinery (depth law → ruled loft → mirror) | base `_crescent_taper_cutters():269`, `_c7.py:122-184`, `_v0.py:187-227` (+ forked duct clamps `_c7:86` / `_v0:164`) | 120–150 |
| M4 | `_require_guarded_build()` + routing-profile guard raises | `_obiwan.py:72`, `_obiwan_route.py:92`, `_obiwan_floor.py`, `obiwan_wings_cad.py:143`; profile guards `_b2_split.py:56`, `_obiwan_split.py:24` | 60–80 |
| M5 | Forked low-level primitives: `_polar_xy`/`_polar`, `_cubic_point`/`_cubic`/`_cubic_points`, cylinder/prism/arc constructors | `_obiwan.py:433,856,439`, `_obiwan_route.py:486,494,1211,543`, `_um_fit.py:238,736`, `_cables.py:167`, `_obiwan_floor` | ~100 |
| M6 | `_d_seg` point-to-segment distance | `_c7.py:62`, `_v0.py:143` | 10 |
| M7 | Magnet constants re-derived outside `captive_magnets.py` (the authority, `:59-85`) | `_b`, `_v0`, `_obiwan.side_magnet_sites()`; worst: `slice_captive_magnets.py:171-181` re-hardcodes `cavity_diameter_mm=5.20` etc. without importing the authority | — |

### 3.2 Export / tooling layer

| # | Duplicate | Where | ~Lines |
|---|---|---|---|
| T1 | SHA-256 file digest | **~16 copies**: `front_down_contract.py:71`, `write_obiwan_release_manifest.py:36`, `slice_captive_magnets.py:243`, `check_manifold.py:412`, `generate_captive_magnet_catalog.py:99`, `export_obiwan_wings.py:115`, `export_obiwan_staged.py:304`, `export_piece_stls.py:240`, `build_to_print_shelf.py:62`, `remote_cad.py:124`, `tools/build_artifact_catalog.py:302`, `_obiwan_floor_strength.py:538`, … | 35 |
| T2 | Atomic-JSON write (`.tmp` → `os.replace`, `indent=2, sort_keys=True`) | 7 copies: `build_to_print_shelf.py:75`, `slice_captive_magnets.py:251`, `export_obiwan_wings.py:174`, `export_obiwan_staged.py:351`, `remote_cad.py:142`, `generate_captive_magnet_catalog.py:121`, `front_down_contract.py:323` | 40–50 |
| T3 | STL export helper trio (`_validate_binary_stl` / `_canonicalize_transform_zeros` / `_strict_mesh_facts`), byte-identical | `export_coupon.py:80,93,121`, `export_piece_stls.py:74,87,169`, `export_obiwan_wings.py:217,230,255` | 110–130 |
| T4 | `_validate_step_transaction` | `export_steps.py:34`, `export_v1l_staged_step.py:53`, `export_obiwan_wings.py:205` | 20 |
| T5 | Front-down transform **builder** inlined (the contract module only *validates*) | `export_coupon.py:288-309`, `export_obiwan_wings.py:336-352`, `export_piece_stls.py:321-343` (+ second copy `:263-307`) | 50–65 |
| T6 | Binary-STL header parsing (`unpack_from("<I", …, 80)`) | ~8 sites; only `bambu_3mf_audit.read_stl_triangles` is reused | 30 |
| T7 | Memory-guard self-re-exec `__main__` preamble | ~10 scripts, e.g. `export_steps.py:21-29`, `export_piece_stls.py:28-37`, `export_obiwan_staged.py:120-128` | 80 |
| T8 | matplotlib Agg setup + atomic-PNG save + vase-wall landmark literals (`(38.113, 315.947)`, slope `0.29752`, …) | 6 sheet generators; landmarks copy-pasted in `gen_c_variants.py:46-51`, `gen_um_knife_draft.py:27-29`, `gen_lm_knife_draft.py:67`, `gen_obiwan_wing_design_map.py:157` | 60 |

### 3.3 Makefile (1,115 lines; ≈90 hand-written + ~85 generated targets)

| # | Duplicate | Where |
|---|---|---|
| K1 | ~17 focused `check_*` targets, each `LX_R6F_SINGLE_CHECK=<name> $(RUN) test_obiwan_r6f.py` (22 occurrences) — duplicating the *generated* `_check_r6f_<name>` stamp nodes | `:857-958` |
| K2 | floor/no-floor pairs written twice verbatim: `validate_*_obiwan_stage` `:475-485`, `{floor,no_floor}_obiwan` `:530-541`, the whole mouth/burial/backfill/shell/split ladder `:837-925` | |
| K3 | `wings/.stamp_ac` / `.stamp_ae` — 11-line recipes differing only by slug | `:572-592` |
| K4 | 16 `export_piece_stls.py` recipe lines; `.stamp/_c7/_v0/_v1` are the same 4-line skeleton ×4; V1L (5 calls) and Obi-Wan (4 calls) are hand-unrolled loops | `:966-1013` |
| K5 | `LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=… $(RUN)` prefix repeated on 23 recipe lines | `:968-1079` |
| K6 | "Recover-if-missing" multi-output idiom duplicated (Make 3.81 grouped-target workaround) | `:1085-1087`, `:1105-1108` |
| K7 | `.PHONY` sprawl: ~90 names, several reachable from no public goal (`check_obiwan_t_shells`, `check_obiwan_lm_profile*`, `check_bump_brep`, `check_no_floor_lm_mesh`) | `:408-441` |

### 3.4 Tests (~1.06 MB, no `conftest.py`, no shared test utils)

- `test_obiwan_r6f.py` (6,949 lines) is a **structure** problem, not a data
  problem: 37 named wrapper functions exist only because the Makefile
  addresses each by name via `LX_R6F_SINGLE_CHECK`; ~1,100 lines are in-file
  subprocess/BREP-cache orchestration; ~1,400 inline goldens.
- Guard re-exec + ThreadPool `main()` near-verbatim between
  `test_clearances.py:1988-2040` and `test_obiwan_wings.py:1979-2030`;
  `_large_host_execution()` verbatim ×3 (`test_clearances.py:44`,
  `test_obiwan_wings.py:91`, `test_obiwan_r6f.py:37`).
- Only `test_bambu_3mf_audit.py` is a real pytest suite — and it is the one
  test *not* wired into the Makefile.

### 3.5 Manifest vocabulary

Six bespoke "files + hashes" JSON producers with four naming conventions
(`catalog.json`, `manifest.json`, `release_manifest.json`,
`obiwan_release_manifest.json`, `captive_magnet_release_catalog.json`,
`*_print_manifest.json`) and five different schema-version keys
(`schema_version`, `FORMAT_VERSION`, `SCHEMA_VERSION`, `AUDIT_SCHEMA_VERSION`,
`R6F_NATIVE_STAGE_SCHEMA_VERSION`). `artifacts/` (symlink view) and
`to_print/` (hard-link view) are two overlapping release-view mechanisms with
independent hash/`--check` logic.

## 4. Dead code and pruning

**Safe to delete now** (zero Makefile/test/import references):

- `gen_c_variants.py`, `gen_lm_knife_draft.py`, `gen_um_knife_draft.py` —
  only referenced as *string entries* in the exclusion list of
  `generate_captive_magnet_catalog.py:981-992`, whose own `reason` fields
  describe them as obsolete concepts.
- `tools/__pycache__/*.pyc` for deleted sources (`diagnose_obiwan_faces`,
  `diagnose_obiwan_mesh`, `gen_obiwan_tangential_entry_concept`,
  `probe_obiwan_entry_vestibule`) — dead bytecode.
- The three untracked `tools/gen_obiwan_*entry*.py` sketches: either commit
  them under a marked `sketches/` directory or delete; today they are
  invisible history.

**Product decision, not a delete** — C7 / V0 / standalone-V1: every release
builds, meshes, manifold-checks and STEP-exports them for both stand states,
yet nothing ships (`to_print` excludes them; `VARIANTS.md` documents them as
experiments). Retiring them from the default `all`/`candidate` DAG would cut
a large share of build time — but they are wired into
`captive_magnet_release_catalog.json` expected counts, release metadata
tests, and the documented mix-and-match compatibility matrix. If the
experiments are considered concluded, demote them to an opt-in target
(`make legacy_variants`) and update the catalog counts in the same change;
do not silently delete.

**Root clutter:** generated-but-tracked `obiwan_wing_design_map.png` (1 MB)
and `top_baffle_nd25fw4_attachments.step` (988 KB) sit at the repo root, as
do stray `result.json` and viewer GLB caches. §7 gives them a home. (Note:
`build_to_print_shelf.py`, `test_to_print_shelf.py`, and `to_print/` are
currently untracked — commit them; they are release infrastructure.)

## 5. Reuse plan — the shared modules to create

Small, dependency-free modules; each line names what it absorbs (IDs from §3).
Phase-1 versions live at the repo root (no import-path change); §7 later
moves them into the package.

| New module | Contents | Absorbs | ~Lines removed |
|---|---|---|---|
| `lx_io.py` | `sha256_file()`, `sha256_bytes()`, `atomic_write_json()`, `atomic_write_bytes()`, binary-STL header read | T1, T2, T6 | 100–150 |
| `stl_export.py` | validate/canonicalize/strict-facts trio, STEP-transaction validator, front-down transform **builder** (constructor next to the existing validator, or added to `front_down_contract.py`) | T3, T4, T5 | 180–220 |
| `guards.py` | `require_guarded_build()`, `require_routing_profile()`, the `__main__` self-re-exec preamble as `reexec_under_guard(main)` | M4, T7 | 140–160 |
| `geom_primitives.py` | `smoothstep()`/`smootherstep()`, `polar_xy()`, `cubic_point()`, segment distance, cylinder/prism/arc constructors | M2, M5, M6 | 130 |
| `taper.py` (or fold into `_b`) | one `rear_taper_cutters(depth_law, stations, duct_clamp)` used by base crescent, C7, V0 | M3 | 100+ |
| `assembly.py` | `gen_step_assembly(pieces: dict, label: str) -> Compound` | M1 | 140 |
| `manifest_schema.py` | one manifest builder: `{schema_version, entries:[{path, sha256, bytes, role, source}]}`; adopted producer-by-producer | §3.5 | drift risk |
| `test_harness.py` | guard re-exec, ThreadPool runner, `large_host_execution()`, solid-validity asserts, named-golden helper | §3.4 | 150+ |
| `plot_base.py` | Agg setup, atomic PNG save, shared colors, vase-wall landmark table | T8 | 60 |

Single-source-of-truth fixes that ride along:

- `slice_captive_magnets.py` must **import** its site geometry from
  `captive_magnets.py` and assert equality against one named golden
  (keeps the change-detection pin, removes the un-sourced literals).
- Parameterize `top_baffle_nd25fw4_attachments.py` by thickness
  (18.3 / 11.5) instead of maintaining the `_v1_attachments.py` fork; merge
  `_a_comp_assembled.py` / `_b1_assembled.py` into one wrapper taking the
  add-on subset.

Longer-term (separate efforts, biggest files):

- Split `top_baffle_nd25fw4_obiwan_route.py` (3,427 ln / 97 fns / 211
  consts) into route / bumps / rear-entry; split `top_baffle_nd25fw4_obiwan.py`
  (2,388 ln) into carriers / closure-webs / joints / magnets.
- Split `slice_captive_magnets.py` (6,594 ln, logic-dense, not a data blob)
  into `gcode_analysis`, `release_validation`, `artifact_emit`.
- Restructure `test_obiwan_r6f.py`: move orchestration into `test_harness.py`,
  collapse the 37 name-addressed wrappers into parametrized cases, retarget
  the Makefile from `LX_R6F_SINGLE_CHECK=<fn>` to case IDs. (Makefile and
  test change together; the per-check stamp granularity is preserved.)
- `remote_cad.py` (2,988 ln) is deliberately stdlib-only and self-shipping —
  leave it alone.

## 6. Makefile simplification (concrete mechanics)

All GNU Make 3.81-safe; the macro machinery (`RULES`, `*_CHECK_RULE`,
`$(eval $(call …))`) already in the file is the pattern to extend:

1. **Collapse the focused-check ladder (K1, K2).** One
   `define FOCUSED_R6F` macro parameterized over (public name, check name,
   state) generating the ~17 hand-written targets at `:857-958`; the
   `$(patsubst check_%,%,$@)` form at `:887-896` shows the compact shape.
   Where a focused target should *always* re-run (iteration), keep it phony;
   where it duplicates a generated `_check_r6f_*` stamp node, make it an
   alias prerequisite instead of a second recipe. ~40 targets → ~8
   definitions.
2. **State-pair macro (K2).** `validate_*_obiwan_stage` and
   `{floor,no_floor}_obiwan` become one two-line macro instantiated like
   `RULES` already is (`$(eval $(call …,floor_stand,1))`).
3. **Slug pattern rule (K3).** `wings/.stamp_%` with `$*`;
   `OBIWAN_WING_SLUGS` already exists at `:178`.
4. **Variant loop (K4).** Generate the `.stamp/_c7/_v0/_v1` 4-line skeletons
   with a `$(foreach v,b2 c7 v0 v1,…)`; keep V1L/Obi-Wan hand-written (they
   genuinely differ: per-piece local staging vs osado single-build).
5. **`RUN_STATE` variable (K5).**
   `RUN_STATE = LX_STAND_FOOT=$(2) LX_ROUTING_PROFILE=$(3) $(RUN)` removes
   the 23 repeated prefixes inside `RULES`.
6. **One `recover_stamp` macro (K6)** for the two recover-if-missing blocks.
7. **Prune `.PHONY` names with no public path (K7)** or document each as an
   iteration entry point in the header comment (several are; say so).

Net effect: same DAG, same stamps, same remote dispatch — the file drops
roughly a third of its length and every state/variant/check family has one
definition.

## 7. Proposed file hierarchy

### Target layout

```text
top_baffle_v2/
├── README.md                     entry point: product picker + quickstart only
├── Makefile
├── docs/                         ALL prose: VARIANTS, PRINTING, qualification,
│                                 briefs, this file, REPOSITORY_STRUCTURE
├── src/lx521_baffle/             parametric geometry package
│   ├── config.py                 LX_STAND_FOOT / LX_ROUTING_PROFILE parsing (one place)
│   ├── geom.py, assembly.py, guards.py, taper.py      (§5 shared modules)
│   ├── base.py                   ← top_baffle_nd25fw4.py
│   ├── cables.py  flush.py  magnets.py (← captive_magnets)  um_fit.py
│   ├── proud/                    R6P: b.py b1.py b2.py a_comp.py split.py
│   │                             attachments.py (thickness-param) c7.py v0.py v1.py v1l.py
│   └── obiwan/                   R6F: carriers.py route.py bridge.py floor.py
│                                 floor_strength.py lm_split.py attachments.py
│                                 assembled.py wings.py
├── scripts/                      CLIs: export_*, gen_*, slice_captive_magnets,
│                                 bambu_3mf_audit, check_manifold, build_to_print_shelf,
│                                 build_artifact_catalog, remote_cad, run_memory_guarded,
│                                 write_obiwan_release_manifest, lx_io/stl_export/
│                                 manifest_schema live here or in src/ as importable libs
├── tests/                        all test_*.py + test_harness.py + conftest.py
├── coupons/                      unchanged (qualification reference pieces)
├── sketches/                     dated one-off concept generators (or delete)
├── build/                        ─ generated, per-state build outputs ─
│   ├── floor_stand/  no_floor_stand/  wings/{ac,ae}/
│   └── common/                   attachments.step, obiwan_wing_design_map.png
├── review/                       audit outputs (unchanged role)
├── artifacts/                    curated product facade (symlinks + manifests)
└── to_print/                     printer-facing shelf (catalog tracked, files local)
```

Naming: inside the package the `top_baffle_nd25fw4_` prefix disappears —
`top_baffle_nd25fw4_obiwan_route.py` becomes `lx521_baffle/obiwan/route.py`.
The 26-file flat prefix family is the single biggest legibility cost of the
current tree.

### Why staged (and why the facade came first)

`docs/REPOSITORY_STRUCTURE.md` is right that a big-bang move is risky: the
Makefile, remote cache keys, stage manifests, tests, and release metadata all
encode root-relative module names, and `floor_stand/` (117 tracked files),
`no_floor_stand/` (114), `wings/` (38), `artifacts/` (160) are *committed*
build outputs whose provenance must stay auditable.

### Migration stages (each independently shippable, gated by `make check`)

- **Stage 0 — deletes and commits.** §4 safe deletes; commit the untracked
  shelf infrastructure. No geometry impact.
- **Stage 1 — shared modules in place.** Add the §5 modules at the repo
  root; convert callers file-by-file. Imports stay root-relative; every
  step is verifiable with `make check` (and `make candidate` at the end —
  geometry gates, manifold sweeps, and metadata tests are the oracle;
  STEP/STL bytes are not expected to change, but the gates, not byte
  equality, are the authority).
- **Stage 2 — Makefile compaction.** §6. Same DAG; verify with
  `make -n <goal>` diffs against the old file plus one full remote
  `candidate`.
- **Stage 3 — the move.** One mechanical commit: `git mv` into
  `src/`/`scripts/`/`tests/`/`build/`, update `PYTHONPATH`/imports/Makefile
  paths and the manifest path fields, regenerate state trees remotely, and
  re-promote so every hash-bearing manifest is rebuilt in the same commit.
  No behavior change may ride along. Expect remote caches to rebuild once.
- **Stage 4 — monolith splits.** §5 longer-term items (`route`, `obiwan`,
  `slice_captive_magnets`, `test_obiwan_r6f`), one file per effort.

## 8. Documentation

Same disease, milder form: `README.md` (1,159 lines) and `VARIANTS.md`
(434 lines — one table cell in it runs ~1,900 words) both try to be the
complete authority; `PRINTING.md` (52 KB) overlaps both. Suggested split:
README keeps the product picker, quickstart, and layout map; per-family
authority pages move to `docs/` (`docs/stock.md`, `docs/slim.md`,
`docs/obiwan.md`) each owning its dimensions/routing/hardware; printing and
qualification stay their own documents. Pure moves — the content is good;
its container is a monolith.

## 9. Prioritized roadmap

| # | Action | Effort | Risk | Payoff |
|---|---|---|---|---|
| 1 | Stage 0 deletes + commit untracked shelf files | minutes | none | tree tells the truth |
| 2 | `lx_io.py` + `stl_export.py` (+ front-down builder) | small | low | ~350 lines, removes hash-drift risk in release gates |
| 3 | `guards.py` + `assembly.py` + `geom_primitives.py` | small | low | ~400 lines across 25+ files |
| 4 | Makefile compaction (§6) | medium | low-med | −⅓ file, one definition per family |
| 5 | `slice_captive_magnets` imports magnet dims from `captive_magnets` | small | low | single source for the geometry all release tests pin |
| 6 | `manifest_schema.py` + adopt per producer | medium | low | one vocabulary, §3.5 gone |
| 7 | `test_harness.py`; then `test_obiwan_r6f.py` restructure with Makefile retarget | medium-large | medium | biggest single-file win (6,949 ln) |
| 8 | Attachments thickness parameterization; taper unification | medium | medium | removes forked-variant pattern at its root |
| 9 | Decide C7/V0/V1-standalone retirement (product call) | small code, big decision | medium | large build-time cut if taken |
| 10 | Stage 3 directory migration | large, mechanical | medium (one-time cache/hash rebuild) | the §7 tree; prefix noise gone |
| 11 | Stage 4 monolith splits | large | medium | maintainability of the two 2.4–3.4 k-line CAD cores |

Items 1–6 are safe wins deliverable this week and shrink the repo by
~1,000 duplicated lines plus a third of the Makefile, without moving a single
file or changing any released geometry.

## 10. Build performance — why a release takes an hour, and the path to ~10×

Measured from `.remote-cad/jobs/` history (153 jobs, 27.4 h cumulative over
four days, all on the 32c/64t osado host):

| Goal | n | warm min | median | max |
|---|---|---|---|---|
| `obiwan_release` | 10 | 37.6 m | **64.5 m** | 72.0 m |
| `candidate` | 1 | — | 68.0 m | |
| `all` | 2 | 44.8 m | 61.7 m | |
| `obiwan_wings` alone | 1 | — | **62.0 m** | |
| `check_obiwan_wings` | 1 | — | 62.9 m | |
| `check_obiwan_junction_closures` | 2 | 15.9 m | 22.4 m | |
| `validate_obiwan_stages` (both states) | 2 | 5.8 m | 6.0 m | |
| `check_route_contract` (single check) | 18 | 0.6 m | 0.6 m | 5.2 m |

### Why it is slow

1. **The wings are the critical path — and they run ~2-wide on a 64-thread
   machine.** `obiwan_wings` alone takes 62 m ≈ the whole 64.5 m release.
   Each slug (`ac`, `ae`) is ONE `export_obiwan_wings.py` process that
   serially: rebuilds the full Obi-Wan carriers via `core_parts()`
   (`obiwan_wings_cad.py:1559-1562` — even though hash-verified staged
   BREPs of exactly those carriers already exist), then booleans/meshes
   6 wing pieces, writes 2 STEPs, renders review images. No internal
   parallelism. During the last ~40 minutes of a release, ~2 of 64 threads
   are busy.
2. **The remote worker is capped at 16 slots** (`remote_cad.py:51`
   `DEFAULT_REMOTE_JOBS = 16`; `run_memory_guarded.py:37`
   `max_guard_slots: 16`; job header: `workers:16
   guard-per-worker:28672MiB floor:65536MiB` — 16 × 28 GiB + 64 GiB
   = exactly the 512 GiB cgroup). The cap is a worst-case RAM reservation,
   not a CPU decision: most tasks (STEP exports, manifold stamps, single
   checks) use a small fraction of 28 GiB, so the machine idles at ≤25 %
   of cores even in the fan-out phases.
3. **Per-task process tax.** Every artifact/check is a fresh
   Python + build123d/OCC import (~0.5–0.6 m floor per task — see
   `check_route_contract` median). ~67 checks + ~38 STEPs + ~30 STL
   invocations pay it every time.
4. **Legacy fan-out**: C7/V0/standalone-V1 STLs, STEPs, and checks build
   for both states on every release and ship nothing (§4).
5. OCC **booleans** are single-threaded per process (build123d never sets
   `BOPAlgo` parallel mode) — and they are the long pole in the carrier and
   wing builds. Meshing is *already* parallel (`export_stl` passes
   `isInParallel=True`), which is why htop shows multi-core bursts even in
   the 2-process wings tail; time-averaged, the tail still leaves most of
   the machine idle. (When reading htop on osado, note it is a shared host
   — redis/mysql/docker/agent sessions run alongside builds — and NLWP
   counts include large parked thread pools.)

### Path to ~10×

| Fix | Effort | Expected effect |
|---|---|---|
| P1. Fan the wings out per piece (add `--piece` like `export_piece_stls.py --obiwan-part`; one Make node per wing piece + a join for STEP/facts) | moderate | wings 62 m → ~8–12 m; release ≈ 25 m |
| P2. Wings load carrier cut-context from the staged BREPs instead of rebuilding `core_parts()` per slug | small | −3–6 m per slug; removes duplicate authority |
| P3. Tiered guard slots: keep ~6 × 28 GiB heavy slots, add ~40 light slots (2–4 GiB) for manifold stamps/checks/PNGs; raise `-j` accordingly. The guard already measures RSS — log peaks per task class and right-size from data | small | fan-out phases ~3×; release ≈ 12–15 m |
| P4. Retire legacy C7/V0/V1 from the default DAG (§4 product call) | small | −10–15 m of slot time |
| P5. Extend the existing warm make-cache to per-piece stamps so an untouched wing/carrier piece never rebuilds | moderate | warm iteration 37.6 m → ~5 m |
| P6. Batch compatible single-checks per process (or a forkserver that pre-imports build123d) to amortize the 0.5 m import tax | moderate | check phase ~2× |
| P7. Experiment: enable OCCT **boolean** parallelism (`BOPAlgo_Options` `SetRunParallel(True)`/`SetParallelMode`; meshing is already parallel via build123d `export_stl`) | small experiment | speeds the serial boolean chains that P1 cannot split |
| P0. Instrument first: log per-task CPU-seconds vs wall (getrusage in `run_memory_guarded.py`) so per-phase utilization is measured, not inferred | tiny | turns every later fix into a before/after number |

| P8. Replace pairwise `part -= cutter` loops with single multi-tool cuts (one BOPAlgo pass over N tools). ≥30 sites: `_obiwan.py:1547-1548, 2221-2222, 2308-2345` (26 `-=` in that file), `_c7.py:189-190`, `_v0.py:255-256`, `_v1.py:69-70`, `_obiwan_floor.py:542-543` | moderate | several-fold on long cutter chains; **geometry-affecting** — regroup changes exact BREP bytes, so regenerate stage manifests + full `make check` |

P1–P3 alone take a cold release from ~64 m to ~12–15 m (4–5×) and are
orthogonal. Adding P5–P7 brings typical (warm, iterative) builds to
~4–6 m — the 10× regime — and cold releases to ~8–12 m. The hard floor is
the longest single OCC boolean chain (one carrier, one wing piece); only
P7/P8 or geometry simplification moves that.

### Alternative stacks assessed (mid-2026)

There is no mature, faster open-source replacement for OCCT's B-rep + STEP
role. Fornjot (the Rust B-rep kernel) is officially discontinued with its
goals unmet; truck (Rust) still lists solid booleans and STEP I/O as future
work; opencascade-rs is the same OCCT kernel behind Rust bindings; the
commercial kernels (Parasolid/ACIS/C3D) and Zoo's cloud engine are
incompatible with this repo's licensing and local hash-pinned provenance.
The one genuinely fast open alternative — Manifold (`manifold3d`, TBB-parallel
guaranteed-manifold mesh CSG, orders of magnitude faster than CGAL-based
CSG) — is triangle-mesh only: no STEP, no exact analytic faces, so it cannot
carry the STEP-first release architecture. build123d has an open proposal for
optional manifold integration; if it lands, a bounded preview/STL-only side
lane could use it. Conclusion: stay on build123d/OCP and spend the effort on
P0–P8.
