# Remote build execution, caching, and memory profiles

H2C is the only supported printer. `make` builds the H2C manufacturing
shelf locally with the installed Bambu software, inside the 8 GiB guard.
Shared CAD and acoustic-study targets use the remote-first executor below:
it snapshots the working tree, runs on `osado.lan`, and promotes only
hash-verified outputs. This document is the authority for its memory and
cache policies. P2S manufacturing targets are deprecated.

## Public remote targets

```sh
make waveguide_study_preflight  # Linux dependencies and solver parity
make waveguide_study_screen     # coupled waveguide geometry comparison
make waveguide_study_broadband  # broadband LR4 study
make floor_obiwan               # shared native geometry and strict QA
make obiwan_wings               # shared flat/graded native wing sources
make vase_tebm35c10_4_cad        # shared Stock/Slim BMR geometry
```

These source-geometry targets do not publish a P2S printer shelf. H2C
manufacturing selection remains in `to_print/`.

## Job control and snapshot semantics

A remote target snapshots the exact current working-tree inputs, runs in an isolated
job on `osado.lan`, and promotes only hash-verified artifacts back into this
directory. The pinned Python 3.12.12 environment is content-addressed and
reused remotely. Jobs survive a lost SSH connection; the printed job id can
be inspected or resumed with:

    python3 scripts/remote_cad.py status JOB_ID
    python3 scripts/remote_cad.py resume JOB_ID
    python3 scripts/remote_cad.py cancel JOB_ID

For a deliberately detached launch use
`python3 scripts/remote_cad.py run --detach TARGET`; ordinary remote Make invocations wait and fetch
automatically, while Ctrl-C detaches without killing the remote cgroup.

The client accepts only documented public Make targets; Make-variable
assignments and private file targets are rejected. `make manifold` includes
the current local candidate in its content-addressed snapshot so osado checks
those exact files. `make clean` mirrors the local target and preserves
`review/` snapshots.

The remote job has a hard 512 GiB aggregate systemd cgroup limit and a 64 GiB
host-free floor, and disables swap. Ordinary CAD uses twelve heavy slots
at 28 GiB plus eighteen light slots at 6 GiB by default: 444 GiB reserved
within the 448 GiB worker budget. Set `LX_CAD_REMOTE_JOBS=N` and
`LX_CAD_REMOTE_LIGHT_JOBS=N` to tune concurrency; the aggregate cap cannot be
relaxed. `LX_CAD_REMOTE_HOST` and `LX_CAD_REMOTE_ROOT` override `osado.lan`
and `~/temp/lx-cad` respectively.

## Lateral waveguide acoustic study

The lateral MU10/ND25FN study has dedicated remote targets:

    make waveguide_study_preflight
    make waveguide_study_screen
    make waveguide_study_broadband
    make waveguide_study_benchmark

They use **one 448 GiB guarded process tree**, a **512 GiB aggregate cgroup**,
**zero swap**, and the existing **64 GiB host-free floor**. No ordinary
CAD/light workers run alongside that tree. The solver checks its dense-matrix
allocation against this budget before allocating; the live RSS guard remains
authoritative. This larger study allocation was authorized on 2026-09-19.
It does not raise workstation limits or ordinary CAD workers' limits.

Acoustic jobs also have a **30-core aggregate CPU quota**. Frequency workers
use disjoint physical cores (SMT siblings are not counted as extra cores), and
Numba, OpenMP and BLAS pools are explicitly bounded. The parent alone writes
frequency caches and the combined manifest. Workers remain descendants of
the same 448 GiB process-tree guard; concurrency never multiplies that limit.

The pinned Gmsh wheel needs GLU even for headless meshing. The study's
`gmsh_runtime.py` resolves system GLU or explicitly loads the already-installed
MeshLab wheel's GLU, verifies its wheel RECORD hash and records the binary
digest. Preflight initializes Gmsh in a fresh child process, so an earlier
MeshLab import cannot conceal an import-order dependency. No system package
installation or root access is required.

`make waveguide_study_source_calibration` rebuilds the spatial study’s bare MU10 with the owner-confirmed 5.4 mm flange. `spatial_source_calibration.json` declares the decreasing mesh edges and common 1.5–14 kHz frequency grid. The current refinement uses 2.0/1.6/1.3 mm; the earlier 4/3/2.5/2-mm campaign remains archived separately. Each campaign uses fixed common CHIEF points across its meshes, records unchanged numerical acceptance checks and identifies a provisional radial-motion hypothesis. It uses the same guarded Osado execution and solver parity/concurrency policy. Historical 4-mm-flange calibration is preserved; new spatial geometry may not silently reuse it. Numerical mesh checks and agreement with measured source polars are separate gates.

`make waveguide_study_spatial_numerics` consumes an explicitly selected,
admitted native case, its assembly record and actual coupled mesh in
`spatial_numerical_campaign.json`. It compares successive fixed-boundary
refinements, near-quadrature depth, nested/independent CHIEF points and angular
integration. Inputs and immutable field views are bound by hashes. Two
successive passing, improving mesh comparisons are required at the same
checkpoints; resource rejection or failed tests remain explicit. This target
does not certify a frequency band from one checkpoint or use the old planar
fields for a new carrier. No selection is supplied before assembly admission.

`waveguide_study_benchmark` uses the completed wide-layout boundary and actual
4/7 kHz BEM solves to compare one 16-thread solver, one 30-thread solver, and
two concurrent 15-thread solvers. It checks each source's complex boundary and
far fields against the saved reference before selecting a policy. A measured
gain of at least 5% is required to change the serial baseline. The report is
`build/side_waveguide_study/validation/solver_concurrency.json`; it records wall
time, CPU use, RSS, thread pools, affinity, code/runtime hashes and field errors.
Missing or stale evidence falls back to one 16-thread solver.

Admission reserves `(dense matrix + 2 GiB) / 0.60` per worker plus 2 GiB for
the coordinator. Larger meshes reduce the worker count automatically; the
best measured serial configuration is used when only one worker fits or one
frequency remains. The benchmark qualifies execution speed and numerical
parity on the tested mesh, not acoustic convergence or physical performance.

The 2026-09-19 Osado benchmark used the actual 20,848-panel assembled boundary
at 4 and 7 kHz. Including worker startup, 1×16 took 116.03 s, 1×30 took
99.32 s, and 2×15 took 74.25 s. The selected default is **two 15-thread
workers**, with **one 30-thread worker** for a single pending frequency or
when memory admission permits only one. This was 1.56× the baseline throughput
(36% less elapsed time) on that workload. The maximum per-source relative
complex-field difference was 3.81e-14; cgroup memory peaked at 15.49 GiB, with
zero OOM events. Retest when the solver/runtime changes or a substantially
different mesh changes the performance balance.

The separate `acoustic-remote-requirements.lock` pins the Linux environment.
The preflight compares the actual Linux solver against saved reference fields
before the campaign can run. Explicit hash-bound inputs are listed in
`candidates/obiwan_side_waveguide_study/remote_inputs.json`; refresh that list
with `python3 candidates/obiwan_side_waveguide_study/src/prepare_remote_inputs.py`
after intentionally changing CAD or calibration inputs. Unrelated slicer jobs,
old archives and generated output trees are excluded from this study snapshot.

Outputs are verified and promoted only to `build/side_waveguide_study/`.
Existing release and printer shelves are not targets of this workflow.
The same detach/status/resume/cancel commands apply. `remote_execution.json`
records the applied limits; the job's `profile.json` records measured usage.

## Incremental cache seeding

Successful remote builds are incremental across isolated jobs.  After an
artifact archive has passed hash verification, local promoted-root QA and the
atomic local promotion transaction, its complete remote worktree is published
as a verified Make-cache seed keyed by both the environment lock identity and
the measured Python/runtime attestation.  A
fresh job clones that seed (using a filesystem reflink when available), checks
every cached byte, mode and mtime, then overlays the exact immutable source
snapshot.  Unchanged source mtimes are retained; changed/new sources are made
newer than every cached target; a removed source rejects the seed and leaves
the job's source-only cold tree intact.  GNU Make remains the sole dependency
engine—checksums only establish a trustworthy input tree.

The first build for an environment is cold.  Later no-change builds reuse
generated CAD and the remote-only selector `.ok` targets; a source edit
invalidates only the Make prerequisite groups it reaches.  Obi-Wan-only carrier,
attachment, bridge and split edits no longer invalidate legacy-family CAD.
Focused jobs return generated files whose bytes changed plus the explicit
public result of the requested target.  In particular, `floor_obiwan` and
`no_floor_obiwan` always return every file named by their hash-bound release
manifest, `common` always returns its STEP, and `obiwan_wings` always returns
the flat/graded design map and both wing artifact trees; a warm-cache Make no-op
therefore also reconstructs a fresh or locally deleted output tree. Complete
targets return their declared
output roots in full and always include their top-level PNG/STEP/catalog
outputs.  Cache corruption is deleted and falls back to the source-only cold
snapshot.  For the exact same immutable source hash, a newer successful
focused job publishes a coverage union: its own files, modes and mtimes win,
while files absent from it are retained from the previous verified seed.  A
narrow concurrent job therefore cannot evict richer CAD or check coverage.
Complete-root targets remain authoritative for their roots, `clean` never
unions, and source-hash changes replace rather than combine cache worktrees, so
represented deletions are not resurrected.  Any source-file deletion also
forces a true cold build, because generic Make prerequisites cannot prove which
inherited generated files belonged only to that source.  A focused job that
removes an inherited generated artifact fails closed instead of publishing an
unrepresentable per-file deletion, and cache publication recomputes the exact
artifact delta.  Failed, canceled, unfetched or locally rejected jobs never
publish a cache, and completion ordering prevents a late fetch from replacing
a newer seed.  This does not change job IDs, snapshot immutability,
status/resume/cancel behavior, artifact hashing or promotion rollback
semantics.

After verified download, promoted STL topology is checked again locally. That
defense-in-depth pass uses eight Make-jobserver workers inside the existing
single 8 GiB aggregate promotion guard. This private mesh-only path cannot run
CAD/OCC targets; ordinary `LX_CAD_EXECUTION=local` remains strictly serial.

The osado profile also removes workstation-only computational fragmentation:
the clearance and Obi-Wan selectors are ordinary Make prerequisites, so the same
jobserver bounds checks and artifact writers together. The Obi-Wan suite imports
LM, UM and tweeter BREPs only from each state's Make-owned, hash-validated native
stage; its private cache contains only test-only shell witnesses absent from that
stage, so assertion-only edits cannot rebuild carriers. It then evaluates
complete shell/cable/service witnesses. The exact assembled-junction sweep
still checks every Bambu 0.20/0.16-mm layer in both stand states, but Make
owns sixteen complementary state/layer-shard stamps. Each shard checks both
junctions and sections their shared UM owner/route only once, while the
sixteen guarded slots remain full instead of running serially. The two
state-specific base BREP gates are independent Make stamps as well;
V1L builds its shared split geometry once; and each state's V1L/Obi-Wan STLs are
meshed in one guarded process. Physical print-bed splits and mating interfaces
remain unchanged. Explicit local mode keeps the segmented, one-heavy-part-at-
a-time implementations needed by the 8 GiB workstation process-tree cap.

Strict STL topology QA is likewise one GNU Make stamp target per mesh, so
osado checks independent meshes concurrently through that same jobserver.
Each state's sweep starts as soon as that state's last artifact lands, so its
47 floor or 45 no-floor mesh checks can overlap remaining work in the other
state. The final combined node does metadata only: it still validates exact
inventories, print sidecars, review artifacts, manifests, and cross-state
differences every time without repeating any of the 92 state mesh checks.
Focused `floor_obiwan`/`no_floor_obiwan` sweeps expand only `*obiwan*.stl` Make
nodes, so verified legacy meshes inherited by a warm cache are not rechecked.

Before launch, the executor measures and hashes the actual Linux x86_64
Python ABI, interpreter binary, uv version and complete installed-package
freeze. The running job attests its real cgroup-v2 `memory.max` and zero-swap
limit. Download, source-drift rechecks, both state-directory swaps and the
local portable manifold check form one locked promotion transaction; any
failure restores the prior floor/no-floor pair.

## Local execution

Running OCC on the current machine requires an explicit opt-in:

    LX_CAD_EXECUTION=local make PYTHON=<venv>/bin/python

## Opposed-BMR vase targets

The opposed-BMR alternative is a complete two-profile family rather than a
Stock-only one-off. Its public CAD target emits independent BREP/STEP/STL,
print sidecar, facts, catalog, slicing-profile, and manifest roots under
`build/vase_TEBM35C10-4/{stock,slim}/`. Explicit profile targets are
`vase_tebm35c10_4_stock_cad` and `vase_tebm35c10_4_slim_cad`. Local Bambu
packaging is likewise available per profile through
`vase_tebm35c10_4_{stock,slim}_3mf`, or together through
`vase_tebm35c10_4_3mf`; those slicing targets never dispatch to osado. Each
validated project is promoted to the stable path
`build/vase_TEBM35C10-4/{stock,slim}/vase_TEBM35C10-4.gcode.3mf`.

## Candidate, release, and provenance boundary

The generated directories are **candidate packages**, not physical-release
authorization. Even the Make target named `release` performs CAD, artifact
and manifold checks only; while the state manifests say
`release_authorized: false`, neither state may be put into service. Complete
and sign the state-specific physical qualification record separately.
Each candidate also retains `.obiwan_stage/`: the hash-validated native BREP
transaction behind its Obi-Wan STEP/STL exports. Its manifest records the exact
Python/package identity, selected memory profile, per-tree cap, free-memory
floor, guard-slot count and any remote aggregate cgroup cap. Those records are
portable, so the local checker can verify a promoted osado build without
pretending it was generated under the macOS profile. It is provenance
evidence, not a printable part, and the strict checker intentionally requires
it. A recorded free-memory floor of `0` means that host-free monitoring was
disabled; RSS monitoring remains active.

## Memory guard

Do not bypass the guard with direct bulk Python invocations. The local profile
stops a CAD process tree above 8 GiB RSS, has no host-free-memory floor, and
keeps local builds serialized. Set a positive `LX_CAD_MIN_FREE_MB` to opt a
local invocation into a stricter floor. The high-memory profile refuses to
load on non-Linux or undersized hosts, retains its mandatory 64 GiB host-free
floor, and is selected by the remote executor only. Limit variables can
tighten their selected profile but cannot relax it.

Self-contained — the direct pip dependencies are
`build123d`, `shapely`, `matplotlib`, `numpy`, and `Pillow` (no external CAD
tooling). `make check` runs `tests/test_clearances.py` for proud-family
regressions and `tests/test_obiwan_r6f.py` for guarded Obi-Wan BREP contracts
(that file keeps its internal name).
Together they cover duct/pilot separation, flange-seat containment,
service envelopes, proud grommet regressions, complete-route smoothness, exact
normal/eroded-outline containment, closed assembled shells, hardware
clearance, and structural/bed screens. `make manifold` (also run at the
end of `make all`) proves candidate-mesh topology; final Obi-Wan BREP shell
tests, rather than manifoldness alone, prove that no rear cable window
is present.
