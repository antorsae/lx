# LX521.4 top baffle — ND25FW-4 face-to-face mod (V2)

3D-printable version of the modified top baffle from
`plano top baffle con anidados V2.pdf` (exact 1:1 vector geometry extracted
from the PDF, not redrawn). Overall 304.8 × 468.31 × 18.3 mm.
That envelope describes the R6P proud family; the R6F Obi-Wan experiment
deliberately removes the full outline and retains only two collars.

## Canonical product inventory

The project now has one human-facing artifact catalog: [`artifacts/`](artifacts/README.md).
The exact design depth is **18.3 mm**, not 18.6 mm.

| Product | Geometry | Optional perimeter | Status |
|---|---|---|---|
| [Standard R6P](artifacts/standard/) | B2, 304.802 x 453.457 x 18.3 mm | A-comp shoulders **or** B1 wings | Canonical CAD |
| [Slim R6P](artifacts/slim/) | V1L + V1; 11.5 mm front-flush acoustic field, full-depth bottom strip | matching V1 shoulders **or** V1 wings | Experimental |
| [Obi-Wan R6F](artifacts/obiwan/) | separate LM/UM collars; floor and stock-bridge states | Ac constant-depth or Ae weighted-depth wings | Candidate; not release-authorized |

![Standard B2 CAD snapshot](artifacts/standard/images/iso.png)

The original state-oriented build outputs remain in `floor_stand/`,
`no_floor_stand/`, and `wings/` because the validation pipeline depends on
them. `artifacts/` adds stable names, hashes, and product grouping through
relative links without duplicating large CAD files. See
[`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md) for the intent, assumptions,
and release boundary; [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md)
documents the implemented layout and the safe future source-package migration.

## Files

| File | What |
|---|---|
| `top_baffle_nd25fw4.py` | Geometry library (drawing outline, holes, pilots); its own gen_step is the un-compromised aligned drawing (no artifacts kept) |
| `top_baffle_nd25fw4_a_comp.py` | Variant A-comp: straight-sided tower — vertical flanks at ±60.65 (tangent to B2's flare crest) from the extended top edge down to the LM chamfer-extension; tweeter section at the B2 drop. Buildable as B2 pieces + 4 shoulder pieces |
| `top_baffle_nd25fw4_attachments.py` / `.step` | The 6 attachment pieces (exact boolean complements): 2+2 A-comp shoulders (top/bottom per side, split at the crest tangent), 2 B1 wings |
| `top_baffle_nd25fw4_b.py` | Shared B-family builder: mini-LM upper-mid vase (no shelf corners) + tweeter section lowered 9.0 mm. Governing clearance is on the FRONT face: the lower tweeter faces forward (stock LX521.4 arrangement), so its D104 faceplate shares the front plane with the 10F's D97.5 flange -- axis spacing 102.84 mm vs 100.75 mm contact leaves a 2.1 mm edge gap (drawing spacing allows an 11.1 mm drop max). Scallop-to-flange 14.1 mm; scallop-to-D82 web 21.9 mm. Total height 459.3 mm. Below the y=306 seam identical to A. |
| `top_baffle_nd25fw4_b1.py` | B1: flank is ONE straight line from the crescent horn corner (36.8, 432.9) through the max-width point (83.8, 399.6) to the V-waist at (+/-56.12, 306.5) -- extended to the horn so the top magnet site lands in the B1 wing |
| `top_baffle_nd25fw4_b2.py` | B2: constant wall around the 10F -- flare and chamfer keep the LM tilts but are both tangent to the r=50.83 circle about the UM center (9.8 mm wall at the D82, 2.1 mm to the D97.5 flange at both tangential points). Chamfer runs from the flare corner (+/-60.65, 391.71; max width 121.3 mm) to the crescent's D102.11 arc extended to (+/-10.08, 418.18); waist (+/-38.1, 315.95). |
| `top_baffle_nd25fw4_b2_split.py` | 4-piece print split of variant B2 (the universal **R6P proud-family** base set), shown assembled |
| `top_baffle_nd25fw4_c7.py` | Variant C7: B2 with the LM section rear-tapered to a ~0.5 knife edge over the last 19 mm inside the flank/chamfer outline (front face stays a full plane). Full-depth land kept at the bottom strip (foot/bridge) and before seam B; half-round ribs (r=5.4) carry the T ducts across the band. See "Variant C7" below |
| `top_baffle_nd25fw4_c7_split.py` | C7 print split: same seams/dovetails/ducts as B2 -- the three LM pieces are drop-in replacements, piece_top and all attachments are shared |
| `top_baffle_nd25fw4_cables.py` | Proud-family **R6P** subtractive routing and routing-profile dispatch: standard B2/C7/V0/V1 UM tail plus the keyed V1L-only 283-degree alternate |
| `top_baffle_nd25fw4_v1l.py` / `_split.py` | Thin R6P bottom+mids; its alternate UM tail and rear-face exit remain wholly in `piece_mid_right`, so the shared top/vase is unchanged |
| `top_baffle_nd25fw4_obiwan.py` / `_split.py` | Extreme Obi-Wan core: structural LM/UM flush-driver collars at R113.0/R51.7 with smooth exposed R113.8/R52.5 side fairings clipped only inside the existing LM--UM and T--UM cusp/service regions, with the 0.40 mm LM--UM inter-carrier gap preserved; rounded LM-to-UM M3 half-laps whose closure-web/base teardrops remain nominal Ø9 while each complete Z-owned cylindrical functional boss is locally Ø9.8, with standalone rear Ø3.4 LM clearance bores and standalone rear-opening blind Ø4.6 x 4.0 UM heat-set receivers; six pause-and-bury captive magnet stations (two upper LM ring-radial, two lower LM base-side, and two UM ring-radial), with ring cavity datums hidden 0.15 mm beneath the continuous fairing and no local pad/boss/flat/cue; buried UM/T route spans; and free rear cable continuations. Floor and no-floor share one exact lower-LM front/wing-contact outline from Y=0 through the broad ring shoulder; only rear/deep structure differs (integral W64 stand/NL8 panel versus shallow four-insert bridge). |
| `top_baffle_nd25fw4_obiwan_lm_split.py` | Optional, mutually exclusive two-print form of the finalized Obi-Wan LM carrier: exact zero-gap world-Y butt seam plus two symmetric Ø1.60 cylindrical pins normal to the seam (world +Y). The pins engage 2.40 mm; the right blind socket is round Ø1.84 and the left is X-relieved to 1.96 × 1.84 mm so the 218.374 mm pitch cannot bind like two tight round fits. Two tiny exterior lands outside the LM recess retain 0.12 mm radial and 0.25 mm end clearance, at least 0.50 mm local radial/end wall, at least 0.05 mm recess plan clearance and 0.13 mm conservative W22-flange clearance. Their worst-case reach is R114.4036: 1.4036 mm beyond the structural R113.0 ring and 0.6036 mm beyond the finalized R113.8 visible fairing. They add no extra fastener or standalone retention/load credit; the monolithic LM remains canonical. |
| `top_baffle_nd25fw4_obiwan_route.py` | Exact R6F printed-owner segments and physical cable continuations: 0.8 mm minimum walls and 0.85 mm seat roof on the surviving buried UM/T spans, full-width longitudinal burial webs plus solid roof-to-bore saddles at every named insert-bypass Z bump, free UM cable behind the UM carrier, free T cable behind the tweeter crescent, and the 82.67° physical crown crossing |
| `top_baffle_nd25fw4_obiwan_bridge.py` | Universal lower-LM front profile (filled exterior union of the historical floor stem and no-floor bridge), immutable no-floor four-hole datum, fused 62 mm insert core with soft cubic shoulders and two centered rear cable entries at the deepest existing LM-pad depth (no separate keel or rear ribs), hardware proxies, and an opening-aware biaxial 4 kg sustained-1g/3g/5g structural screen |
| `top_baffle_nd25fw4_obiwan_floor.py` / `_floor_strength.py` | Floor-only integral W64 full-depth stem/foot, R12 root, rear NL8 panel/service cavity and three buried cable continuations; closed-form five-material net-section screen. This is part of the LM carrier, not an add-on, and the analysis is not FEA or physical qualification. |
| `top_baffle_nd25fw4_obiwan_attachments.py` | Optional tweeter crescent with complete standalone blind-M3 receiver ears; any cable retention is external/non-modeled, and magnets receive zero structural load credit |
| `top_baffle_nd25fw4_obiwan_assembled.py` | Review assembly containing the R6F core, selected add-ons, and the explicitly non-manufacturing terminal/Faston proxy |
| `obiwan_wings_cad.py` | STEP-first Ac/Ae Obi-Wan acoustic attachments: one canonical monolith per side, three exact surface-normal captive D5 × 2 magnet receivers per side (LM lower, LM upper, and UM), one saddle compatible with the shared floor/no-floor lower-LM front profile, the approved constant-depth Ac or monotonic LM/UM/T-weighted Ae rear, and three exact-mask print intersections per side. Each physical side has one V1L-style through-local-thickness XY dovetail at the lower→middle interface (lower male, neck/head/depth 7/9/4 mm) and one at the middle→UM interface (middle male, 7/8.5/4 mm); female clearance is 0.05 mm, the exposed split clearance closes over the final 2 mm at both endpoints, and the keys add no envelope growth. They register/interlock in XY but provide no independent Z retention. Ae’s complete internal protected-land perimeter is accepted only when paired actual-BREP probes show a C0 jump ≤0.03 mm |
| `export_obiwan_wings.py` | Transactional Ac/Ae exporter: canonical/assembled STEP, six strict front-face-down STLs with six exact adjacent `.print.json` authorities, facts, hash manifest, and CAD-derived QA renders under `wings/ac/` or `wings/ae/`; every review PNG uses hash-validated staged BREPs for a neutral no-floor LM-upper/UM/tweeter reference plus the two coincident LM-lower outlines—blue dash-dot for no-floor and green dotted for floor stand; the side view keeps its useful acoustic-depth scale and includes a complete-depth floor inset |
| `test_obiwan_wings.py` | Remote-only Ac/Ae BREP, print-inventory, STEP, STL, mirror, depth, receiver, dovetail/clearance, endpoint-closure, bed-fit, provenance, render, and exact dual-state lower-LM front-profile gates |
| `top_baffle_nd25fw4_um_fit.py` | 283-degree MU terminal service model: terminal-less MU body, hash-pinned W22 reference and declared-placement conservative rear keepout, independent low-profile flag-Faston pull states, physical OD8/OD4 Y-breakout harness, and the proud/V1L split strain reliefs; `PHYSICAL_MEASURE_REQUIRED` remains true |
| `export_piece_stls.py` | Exports the print-ready proud-family or Obi-Wan core/add-on STLs (`--variant`, `--outdir`) and one exact adjacent, hash-bound `.print.json` authority for every STL |
| `export_steps.py` | Exports a module's `gen_step()` to STEP via build123d's native exporter (`<module.py> --output <path>`) — no CAD-skill dependency |
| `Makefile` | Generates STEPs/STLs/PNGs for both stand states into `floor_stand/` and `no_floor_stand/` (see "Generated artifact layout"). Local OCC jobs are serial; the remote executor uses bounded parallel slots, and every CAD subprocess runs through `run_memory_guarded.py`. |
| `remote_cad.py` / `cad-remote-requirements.lock` | Content-addressed SSH executor, resumable job control, verified artifact return, and exact remote Python environment |
| `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md` | Authoritative per-STL front-face-down orientation, actual sliced open/closing layers, Bambu pause markers, grouped magnet counts, and local-axis polarity |
| `<variant>/stl/*.stl` and `wings/{ac,ae}/stl/*.stl` | The enforced acoustic-print inventory is 45 nonpolar front-face-down STL/sidecar pairs in each stand state plus six Ac and six Ae pairs: 102 exact pairs total. Every acoustic piece is source-X180 with only an optional in-bed Z rotation and its front datum at STL Z=0. A missing, orphaned, stale-hash, tilted, or translation-inconsistent `<stem>.print.json` fails release validation. The two floor polar-index jigs are the sole orientation-sidecar exclusions because they are fixtures with no acoustic front-face datum. |

Regenerate through the remote executor (the default):

    make
    make floor_stand
    make floor_obiwan  # focused integral-floor Obi-Wan release and strict QA
    make obiwan_release  # both Obi-Wan states + Ac/Ae, concurrent on osado
    make obiwan_wings  # Ac + Ae STEP/STL families, built concurrently

`make` snapshots the exact current working-tree inputs, runs in an isolated
job on `osado.lan`, and promotes only hash-verified artifacts back into this
directory. The pinned Python 3.12.12 environment is content-addressed and
reused remotely. Jobs survive a lost SSH connection; the printed job id can
be inspected or resumed with:

    python3 remote_cad.py status JOB_ID
    python3 remote_cad.py resume JOB_ID
    python3 remote_cad.py cancel JOB_ID

For a deliberately detached launch use
`python3 remote_cad.py run --detach TARGET`; ordinary `make` waits and fetches
automatically, while Ctrl-C detaches without killing the remote cgroup.

The client accepts only documented public Make targets; Make-variable
assignments and private file targets are rejected. `make manifold` includes
the current local candidate in its content-addressed snapshot so osado checks
those exact files. `make clean` mirrors the local target and preserves
`review/` snapshots.

The remote job has a hard 512 GiB aggregate systemd cgroup limit and a 64 GiB
host-free floor. It uses sixteen parallel guarded recipe slots by default,
with the remaining 448 GiB divided equally (28 GiB each). Set
`LX_CAD_REMOTE_JOBS=N` to tune the slot count; the aggregate cap cannot be
relaxed. `LX_CAD_REMOTE_HOST` and `LX_CAD_REMOTE_ROOT` override `osado.lan`
and `~/temp/lx-cad` respectively.

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
the Ac/Ae design map and both wing artifact trees; a warm-cache Make no-op
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
the clearance and R6F selectors are ordinary Make prerequisites, so the same
jobserver bounds checks and artifact writers together. R6F imports LM, UM and
tweeter BREPs only from each state's Make-owned, hash-validated native stage;
its private cache contains only test-only shell witnesses absent from that
stage, so assertion-only edits cannot rebuild carriers. R6F then evaluates
complete shell/cable/service witnesses. The exact assembled-junction sweep
still checks every Bambu 0.20/0.16-mm layer in both stand states, but Make
owns sixteen complementary state/junction/layer-shard stamps so those
expensive BREP sections keep the same sixteen guarded slots full instead of
running serially;
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

Running OCC on the current machine requires an explicit opt-in:

    LX_CAD_EXECUTION=local make PYTHON=<venv>/bin/python

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

Do not bypass the guard with direct bulk Python invocations. The local profile
stops a CAD process tree above 8 GiB RSS, has no host-free-memory floor, and
keeps local builds serialized. Set a positive `LX_CAD_MIN_FREE_MB` to opt a
local invocation into a stricter floor. The high-memory profile refuses to
load on non-Linux or undersized hosts, retains its mandatory 64 GiB host-free
floor, and is selected by the remote executor only. Limit variables can
tighten their selected profile but cannot relax it.

Self-contained — the direct pip dependencies are
`build123d`, `shapely`, `matplotlib`, `numpy`, and `Pillow` (no external CAD
tooling). `make check` runs `test_clearances.py` for proud-family
regressions and `test_obiwan_r6f.py` for guarded R6F BREP contracts.
Together they cover duct/pilot separation, flange-seat containment,
service envelopes, R6P grommet regressions, complete-route smoothness, exact
normal/eroded-outline containment, closed assembled shells, hardware
clearance, and structural/bed screens. `make manifold` (also run at the
end of `make all`) proves candidate-mesh topology; final R6F BREP shell
tests, rather than manifoldness alone, prove that no rear cable window
is present.

## Key dimensions (from the drawing, verified against printed dims)

- Outline: bottom 152.4 → ±152.4 @ y≈256.1 → neck 114.3 (y 306–409) →
  121.84 across the tweeter prongs; top scallop cut from Ø78.50,
  corner arcs from Ø102.11 (both centered ≈ (0, 483.05) = rear tweeter axis).
- Lower-mid cutout Ø190 @ (0, 200.98); upper-mid Ø82 @ (0, 366.08) — the
  drawing had it at 371.94; all variants now align the UM (and tweeter
  section, and the perimeter above the neck) to the stock LX521.4 baffle
  (`lx521 baffle metric.dxf`, UM at 368.3 with LM at 203.2, LM-aligned).
- 4 bridge mounting points @ (±20.0, 20.0 / 70.0) — measured on the
  actual bridge (40.0 × 50.0 pattern; the V2 plano's positions were
  wrong). BRASS HEAT-SET inserts identical to the W22/LM (bore Ø6.4 ×
  6.8, M5 × 5.8 × Ø6.3), but bored BLIND from the REAR face (opposite
  the front-mounted driver inserts): the stock bridge screws in from
  behind with M5 machine screws. no-stand only. Front face stays solid
  (no through-hole, no countersink).
- 2 corner holes Ø4.5 @ (±66.2, 10.0) — OPTIONAL, disabled by default
  (set CORNER_HOLES_ENABLED = True in top_baffle_nd25fw4.py to cut them).
  When enabled: M5 machine screws thread-form through the full 18.3 mm
  (pre-run the screw once to cut the threads).
- Blind driver mounts, front face only:
  - Upper mid (production SEAS MU10RB-SL H1658-04, historically called
    “10F” in this project; four-hole flange on pitch D89.5): 4 x Ø4.6
    bores at 58/148/238/328 deg (a square clocked
    +13 deg from 45 -- 45/90 grids are geometrically impossible, see
    VARIANTS.md), 4.0 mm deep, for BRASS HEAT-SET inserts M3 x 3 long
    x Ø5 OD (soldering-iron set). The ring sits 3.75 from the D82 cutout
    wall; the slim bore keeps 1.45 mm on its inboard side. The shared TS duct
    (z=11.5) clears the rotated pattern IN PLAN (>=6.8 to every bore;
    pilot floor z=14.3).
  - Lower mid (production SEAS U22REX/P-SL H1659-08; the hash-pinned
    W22EX001 shrinkwrap supplies the reference mounting template): 6 x
    D5.0 flange holes with D8.8 head recesses on pitch D209.5 and 6 x
    Ø6.4 × 6.8 bores, aligned VERTICALLY (30/90/...330 deg), for BRASS
    HEAT-SET inserts M5 x 5.8 long x Ø6.3 OD. The structural screen assumes
    600 N pull-out per correctly installed insert; it does not qualify the
    actual print process, creep, reuse cycles or unlimited removal/refit.
    M5 screws pass the D5.0 flange holes natively and seat in the D8.8 recesses.
    Floor z=11.5; the ring is plan-clear of every front-half duct
    (seam C clears the 90-deg bore by 2.25 mm, the LM duct keeps
    3.05 mm to the 270-deg bore).
- 2 tweeter-clamp holes @ (±32.56, 451.24) — drawing Ø4.0, printed Ø4.4
  for M4 clearance. The face-to-face ND25FW-4 pair bolts through these,
  sandwiching the baffle crescent between the two faceplates; the pair's
  upper two holes are joined by standoffs above the baffle.
- CRESCENT REAR TAPER: the horseshoe that carries the tweeter pair
  thins from the REAR (the front face stays a full plane): 18.3 at the
  bottom of the scallop, 4.0 at the clamp pass-throughs, feathering to
  ~0.4 at the horn tips. Thickness follows the arc angle about the
  scallop center through two C1 smoothstep segments (zero slope at the
  bottom blend AND across the clamp ring, so the rear faceplate gets a
  locally flat 4 mm seat); cut as a loft of radial sections. Each
  section holds full cut depth from r=36 out to a knee r=51.5 (covering
  the D102.11 arc joint at r≈51.05), then SMOOTHSTEP-FADES the cut back
  to 0 by r=62 (just inside the flank's top corner at r≈62.4). The fade
  carries the SAME taper across the arc joint
  into the crescent's outboard neighbours — the A-comp TOP SHOULDERS
  and B1 WINGS — so when magnetically installed, their rear faces are FLUSH
  with the tapered crescent (no proud step), then ramp back to full
  18.3 depth before their outboard vertical flank/top edges (which stay
  full for the silhouette) and before the crest (y=391.71, where the
  top shoulder meets the full-depth bottom shoulder). Beside the horn
  tips, where the crescent feathers to ~0.4 mm, the shoulder feathers
  with it; the chamfer/flare walls at larger r keep full depth.
  Consequences handled: the T duct tails keep a >=1.3 mm floor where
  the taper starts (cut only ~0.5 deep there); the upper magnet site
  sits where the taper, the T ducts, and the shoulder's chamfer mating
  face balance (see the magnet section); the bottom shoulders are
  untouched (full depth).

## Print split (256×256×256 bed)

- Seam A: y=120 (two ~58 mm lands beside the Ø190 cutout), 4 dovetails:
  shallow inner teeth at x=±66 (neck 7 / head 9 / depth 3) and main
  outer teeth at x=±103 (neck 6 / head 7 / depth 5). They straddle the
  R6P duct crossings and remain in full-depth material outside the C7
  taper.
- piece_bottom carries a FUSED stand foot (STAND_FOOT flag in
  top_baffle_nd25fw4.py). The foot starts as the baffle's own bottom
  strip (18.3 tall, side faces continuing the flank slopes: ±76.2 at
  the floor widening to ±81.6, sharing the floor plane with the plate
  -- no step), runs 150 mm rearward, and TAPERS in plan continuously
  (one straight line per side, from the strip corners to 38 wide at
  the panel inner face, z=-146). The plate/foot inner corner is a
  plain 90-deg joint (no rib -- printed front-face-down the joint is
  continuous perimeter walls, but it still requires the solid-infill
  modifier and proof procedure in PRINTING.md). The dressed baffle's CG
  sits ~52 mm behind the front face, so it stands upright with no
  front toe.
  The foot's far end carries a minimal 38 × 44 × 4 panel wall for a
  Neutrik NL8MPXX-BAG speakON: Ø31 cutout centered at (0, 20.5) plus
  4 × Ø3.2 screw pass-throughs on the 29.2 × 29.2 pattern (flange is
  38.7 sq -- 0.35 mm/side overhang past the 38 panel, cosmetic). The
  tongue center is channeled to a 4.0 floor between 2.0-thick side
  rails (interior 34 wide: >=1.75 around the Ø30.5 body, which reaches
  ~z=-113); the channel's step face sits at z=-99.
  With the flag ON: the four bridge pass-throughs (and countersinks)
  are omitted, and the cable ducts no longer break the rear face --
  each continues down the plate, drifts to its packed foot lane, takes
  a 90-deg vertical-plane elbow (R14 -- the largest radius that wraps
  the plate/foot inner corner with >=1.4 mm clearance), and runs
  rearward at y=10.5 (LM x=-5.45, UM x=+5.4) / y=5.5 (T1 x=+13.9,
  T2 x=-13.9), exiting through FOUR holes in the channel's step face
  -- ~40 mm of open channel between the cable outs and the connector
  tabs for dressing/Faston access. (Lanes are packed by Δx alone --
  8.45 + 10.85 + 8.5 webs -- because each pair of descent curves
  crosses in the (y,z) plane.) At the driver end, LM retains its rear
  outlet below the Ø190 opening and TS pierces the open tweeter scallop. The proud-family UM
  route instead stays Ø8.2 through one continuous G1 R14 turn and
  leaves the rear face at (33.446, 301.492); there is no separate UM bore
  into the Ø82 opening. With the flag OFF: the original flat piece,
  bridge holes, and rear-face breakouts aimed at the SUPPORT
  WINDOW -- the stock support plate has a Ø20 hole (center (0, 60):
  horizontally centered, top edge tangent to the upper screw line
  y=70) that all four cables must pass. Packing: LM/UM breakouts side
  by side (steep ramps crossing z=0 at (∓5.2, 60.5), tips lancing
  their mains at (−8, 68.5, 12.55) / (+8, 60, 12.55)); twin Ø4.6 T
  ramps at the window's lower edge, breakouts (+3.8, 52.2) /
  (−3.1, 52.7) with far lips up to ~1.4 past the rim (the floppy
  AWG24 pairs duck in), lancing the strip feeders (t1f z=3.7 /
  t2f z=9.5) that merge into the Ø6.8 z-step west of the LM column.
  Fish the LM/UM steep ramps with the plate off the support.
  Print orientation: plate flat FRONT FACE DOWN (rotate 180 deg about X;
  foot rises as an 18.3-thick wall, the panel just widens its top --
  no supports: the step face looks upward and the NL8 holes print as
  vertical-axis circles). Standing the part on its foot is not a released
  orientation because it would give the acoustic front a different texture.
- Seam B: y=315.95, exactly through B2's waist kinks, 2 dovetails
  (left -19, right +30; neck 10 / head 14 / depth 6). Both pieces get
  OBTUSE corners at
  this seam (top foot ≈107° against the flare, mids ≈152° against the
  chamfer) — no brittle knife-tips — and the glue line hides in the crease.
- Seam C: x=-5.6 between A and B (~20 mm land above the cutout; offset
  left so its dovetail pocket clears the 90-deg W22 insert bore),
  1 dovetail (neck 7 / head 8.5 / depth 4 at y=305.0).
- Dovetails are through-thickness, 0.10 mm clearance on female sides.

These seams and the foot/bridge behavior apply only to the R6P proud
family. Obi-Wan R6F is not a thinned four-piece shell: its mandatory core
print set is the two collars described below. In floor state the complete
stand is fused into the LM carrier itself; there is no separate
`lx521_top_obiwan_addon_mount_floor_support` artifact. The no-floor bridge
remains fused into the LM core exactly as before.

### Generated artifact layout

The default remote `make` builds BOTH stand-foot states. Use
`LX_CAD_EXECUTION=local make -j1` only for an intentional local build:

    floor_stand/      LX_STAND_FOOT=1: R6P fused foot + NL8 panel;
      stl/  *.step  *.png     R6F integral LM-owned W64 floor stem/foot + NL8 panel
    no_floor_stand/   LX_STAND_FOOT=0: R6P flat piece_bottom + bridge;
      stl/  *.step  *.png     R6F solid bridge web fused into the LM core

Each folder contains all proud-family variants and the matching Obi-Wan
core/add-on set. In R6P, `piece_bottom` is the only functionally
different base piece; the other base STLs can differ by <0.05 mm as
the foot-entry knots move. In R6F, the UM core is state-independent. The
floor LM owns the integral stand; the no-floor LM owns the fused bridge web,
but their front wing-contact outlines are identical through z=6.8..18.3.
`top_baffle_nd25fw4_attachments.step` is flag-independent and stays at
the top level. The `LX_STAND_FOOT` environment flag defaults to 1.

| STL in `<variant>/stl/` | Footprint (mm) | Used by |
|---|---|---|
| lx521_top_base_1of4_bottom | 223.8 × 125.0, 168.3 tall (fused stand foot; front-face-down only — see above) | R6P proud family |
| lx521_top_base_2of4_mid_left | 146.7 × 201.9 | R6P proud family |
| lx521_top_base_3of4_mid_right | 162.0 × 201.9 | R6P proud family |
| lx521_top_base_4of4_vase_b2 | 121.3 × 137.4 | R6P proud family |
| lx521_top_addonA_1..2of4_shoulder_top_l/r | 50.6 × 61.8 | A-comp only |
| lx521_top_addonA_3..4of4_shoulder_bottom_l/r | 22.5 × 85.9 | A-comp only |
| lx521_top_addonB1_1..2of2_wing_l/r | 73.7 × 125.1 | B1 only |
| lx521_top_c7base_1of4_bottom | as base 1of4 | C7 (LM knife taper) |
| lx521_top_c7base_2..3of4_mid_l/r | as base 2/3of4 | C7 (LM knife taper) |
| lx521_top_c7base_4of4_vase_b2 | same part as base 4of4 (re-tessellated file) | C7 = same vase |
| lx521_top_v0_4of4_vase / v1_4of4_vase | as base 4of4 | V0 / V1 vase experiments |
| `lx521_top_v1l_1..3of4_*` + re-exported 4of4 vase | as proud base 1..4of4 | keyed V1L bottom/mids; its 283-degree alternate is confined to 3of4 `mid_right`, while 4of4 is the unchanged V1 vase |
| `lx521_top_obiwan_core_1of2_lm_carrier.stl` | Structural Ø226 (R113.0) collar with a smooth exposed R113.8 side fairing, clipped only inside the LM--UM cusp to retain the 0.40 mm gap; six ordinary blind LM insert bores at 0/60/120/180/240/300°; two complete rear LM-to-UM ears with locally Ø9.8 cylindrical functional bosses and standalone Ø3.4 rear-driven screw-clearance passages at x=±32/y=315.770; two captive upper ring-magnet stations hidden 0.15 mm beneath the fairing plus two captive lower base-side stations at `(x,y,z)=(±32,18,12.55)`; and the LM-owned buried UM/T route segments. The D7.8 LM lead is free inside a minimum-radius 3.96 mm rear-open subtractive clearance, not a printed duct. Both states have the same broad lower shoulder and Y=0 front tongue, so the lower interface faces and outward ±X normals coincide. Behind it, floor state owns the full-height W64 stem/foot and NL8 panel while no-floor owns the shallow four-insert bridge. | canonical large-format release form of the mandatory R6F LM carrier; approximately 236.41 x 313.75 mm front-face-down, so **not P2S-printable**. Use this on a verified larger bed **or** both optional keyed halves, never both forms. |
| `lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` | front-face-down; in-plane bed rotation only; verified within 220 mm in both states | optional replacement print form for the canonical LM; in floor state it inherits the **entire** stem/foot/NL8 panel but remains the bed-checked alternative to the oversized monolith; requires the matching top half |
| `lx521_top_obiwan_optional_lm_keyed_2of2_top.stl` | front-face-down; in-plane bed rotation only; inherits both complete LM-to-UM ears, their local Ø9.8 cylindrical functional bosses, and their standalone Ø3.4 rear clearance passages | optional replacement print form for the canonical LM; requires the matching bottom half |
| `lx521_top_obiwan_core_2of2_um_carrier.stl` | Structural Ø103.4 (R51.7) collar with a smooth exposed R52.5 side fairing, clipped only inside the LM--UM and T--UM cusp/service regions while retaining the 0.40 mm LM--UM gap; two complete front LM-to-UM ears with standalone rear-opening blind Ø4.6 x 4.0 M3 heat-set receivers and 1.9 mm acoustic-front floors; two complete rear UM-to-tweeter ears with standalone Ø3.4 screw-clearance passages; locally Ø9.8 cylindrical functional bosses at both interfaces; two captive ring-magnet stations hidden 0.15 mm beneath the fairing; and the buried T continuation with fully solid-webbed 328°/58° insert bypasses. The UM cable is free behind this carrier and has no printed rear duct. | mandatory R6F UM core; install both LM-to-UM inserts in this individual print before assembly |
| `lx521_top_obiwan_addon_tweeter_crescent.stl` | cropped V1 crescent plus two complete front UM-to-tweeter ears with locally Ø9.8 functional bosses, standalone rear-opening blind Ø4.6 x 4.0 M3 heat-set receivers, complete 360° walls, and 1.9 mm acoustic-front floors; no printed T-cable arc or conduit | optional R6F face-to-face tweeter carrier; install both inserts in this individual print before assembly, then attach at x=±24, y=421.5 with the T cable free behind it |
| `lx521_top_proud_addon_um_grommet_half_{a,b}.stl` | split TPU insert with short curved shank | standard B2/C7/V0/V1 R14-bore strain relief; not V1L |
| `lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` | keyed split TPU D8 curved shank, D7.1 bore, D13 × 2 flange | V1L-only strain relief; seats at Q on the z=6.8 rear face and follows the alternate R14 |
| `lx521_coupon_*` | small blocks/gauges | calibration, routing, and clocking checks (PRINTING.md) |
| lx521_polar_base_1..2of2 | Ø216 / 169×185 | polar-measurement turntable under the stand foot (floor_stand only) |

Stable routing/fit review files in each state folder are
`top_baffle_nd25fw4_obiwan_split.step` (mandatory two-carrier core),
`top_baffle_nd25fw4_obiwan_lm_split.step` (the optional two-print LM form;
mutually exclusive with the monolithic LM carrier),
`top_baffle_nd25fw4_obiwan_attachments.step` (optional tweeter add-on only),
`top_baffle_nd25fw4_obiwan_assembled.step` (review assembly), and
`top_baffle_nd25fw4_um_fit.step` (non-manufacturing Faston proxy,
standard/V1L/Obi-Wan UM Ø7 cable references, and the proud/V1L profile-fitted
split inserts). Obi-Wan has no printed grommet. The V1L grommet halves are also exported as the stable
STLs listed above.
The assembled R6F STEP also shows the independent LM Ø7.8 reference.

Building the variants: B2 = the four base pieces. A-comp = B2 + the four
shoulder pieces. B1 = B2 + the two wings. Captive magnets retain the
attachments against piece_top_b2's flanks; the 0.05 mm local air gap and
outline kinks, notch corner, and crescent arc register them. Magnets receive
no structural-load credit. The A bottom shoulders and B1 wings extend below
seam B and register against the mids.

## Cable routing: R6P proud (standard + V1L) vs R6F Obi-Wan

Routing is now deliberately split into two physically incompatible
profiles. Generate and review both sheets; the generic routing image no
longer exists:

- `baffle_cable_routing_proud.png` documents **R6P**. It shows the
  normal B2/C7/V0/V1 UM path and, on the same sheet, the clearly
  labeled V1L-only 283-degree alternate tail.
- `baffle_cable_routing_obiwan.png` documents **R6F**, the surviving buried
  UM/T owner segments, the free rear UM and tweeter spans, the short
  un-ducted LM free span, solid-backed insert-bypass bumps, the physical
  T-over-UM crown crossing, state-specific support, and optional tweeter
  carrier. In addition to
  plan routing, it contains the true longitudinal side profiles and local
  nominal diametric u-z sections with authoritative vertical limits through
  representative UM and T bump/pilot axes.

R6P keeps the UM cable space **Ø8.2 end-to-end**. For B2/C7/V0/V1, the
planar main follows the outer U22 arc, returns through the broad lower
neck, and joins an analytic R14 three-dimensional quarter-turn with
constrained G1 tangency. The same sweep reaches a vertical rear tangent
and leaves the rear face at **(33.446, 301.492)**; it is not assembled
from intersecting planar and vertical cylinders. For the estimated Ø7
cable this provides 0.6 mm nominal radial slack. Its 297.376-degree
bearing remains between mounting screws 238 and 328 degrees but is
14.376 degrees away from the 283-degree Faston pull axis. The
conservative D7 rear continuation and profile-fitted curved grommet are
collision-checked against the full 32 × 40 × 10 mm outboard service envelope.

V1L is the keyed R6P exception. Its complete UM cutter substitutes an
alternate tail wholly inside `piece_mid_right`; it does not branch from
or retain the normal R14 outlet. The physical aperture is centered at
**Q = (13.497063, 307.618796, 6.8)**, where the V1L rear face intersects
the exact 283-degree terminal axis at radius 60.0 mm. The nominal cutter
continuation ends outside the part at **(11.080158, 308.797599, −2.0)**,
2.689 mm farther in XY along the tail; that nominal endpoint is not the
aperture center. The route stays below seam B and never enters the
top/vase, so B2/C7/V0/V1 geometry and every top-piece route remain
unchanged. The reference MU mesh still omits its terminal tabs, so the
real driver, Fastons, boots, pull-off stroke, and the supplied
`lx521_top_v1l_addon_um_grommet_half_{a,b}.stl` strain relief require a
physical dry fit before release.

The V1L grommet has a Ø8 curved body around a Ø7.1 nominal cable bore,
inserts 2.5 mm into the keyed R14, and seats a Ø13 × 2 mm flange against
the z=6.8 rear face. Its printed solid clears the conservative Faston
motion box; the installed cable intentionally enters that box because it
is the functional terminal handoff. Do not treat cable/envelope overlap
as a collision or grommet/envelope clearance as proof of real hardware
fit.

R6F rotates only its six LM inserts to **0/60/120/180/240/300°** on the
unchanged Ø209.5 PCD, leaving the 90° crown clear. The physical UM and T
cable envelopes are Ø7.0/Ø5.2. The UM cable uses an Ø8.2 buried passage only
inside the LM carrier, then runs free behind the UM carrier. The T cable uses
an Ø6.0 buried passage through the LM and UM carriers, then runs free behind
the tweeter crescent; the crescent owns no printed cable arc. The
independent D7.8 LM lead instead floats over a short 20.15 mm radial span at
269.5° behind the carrier: there is no printed micro-duct or cover. The LM
carrier owns only a minimum-radius 3.96 mm rear-open subtractive clearance
around the unchanged centerline.
Its center rises smoothly from z=0.40 to 3.80, beginning at R103 before the
R95/D190 mouth; its outer station
retains 1.00 mm clearance to the deepest z=5.3 pad/web rear datum. The floor
state continues the cable inside its buried integral-stem lane. The UM route
rises in the right LM arc and exits the LM-owned
passage before continuing as free cable behind the UM carrier. The tweeter
route rises in the left LM arc, stays buried through the UM carrier, passes
the 328° and 58° pilots on shallow covered Z bumps, and exits before the
tweeter crescent. At the crown the physical routes cross at **82.67°**: T is
the higher +Z cable and UM the lower cable, with **1.85 mm** between their
physical envelopes. This is no longer a two-printed-duct crossover and has
no separator-web claim.

The free UM cable follows the modeled R15 terminal approach to the immutable
283° service axis with a clockwise circumferential **193° tangent** at z=2.7,
then continues with exact G1 continuity through R20, clearing
the known Ø60 motor and terminal-carrier proxy before reaching the named
Y breakout. That breakout has a 4 mm-long OD8 collar with two OD4 branch
sleeves. Two provisional Ø3.2 conductors use R8-minimum slack paths into
separate provisional low-profile flag Fastons (8.5 mm receptacle / 9.5 mm
boot at 11 mm pitch). The review states move one connector at a time
through **0/3/6/9/12 mm** while the other remains installed.

Every surviving printed UM/T owner segment is continuously covered and has no
cable window. The
non-load-bearing wall is two complete 0.4 mm extrusion widths (**0.8 mm
minimum**); the seat roof is 0.85 mm to avoid a tangent BREP union. Insert
bypasses move only in Z and retain at least 0.4 mm to the complete
pad/bore envelope. Each of the eight named bypasses has a local full-width
solid saddle from the conduit roof to the applicable blind-bore floor; there
is no hollow trapped between the duct and bore, and the saddle never extends
behind the existing conduit bump. The LM-owned UM/T low runs and the
UM-owned T low run also have continuous full-width longitudinal webs from
the rear half of the conduit to the seat membrane. Those webs close both
shoulder cavities on either side of the 328°/58° UM bypasses while retaining
only the functional D6 lumen, blind insert bores, captive-magnet cavity voids and
half-lap mating clearances. In floor mode the 300/240/180° saddles retain the
same ordinary blind carrier insert floors as the other three LM axes; all
surrounding saddle volume is solid. The routing
PNG's nominal diametric u-z sections show the authoritative
vertical saddle limits without pretending to be exact octagonal BREP slices.
Obi-Wan deliberately exports no printed grommet or tunnel clip. Keep any
external cable retention outside the modeled buried-route, free-cable,
driver and Faston
service envelopes, and qualify it with the measured cable.

The conservative W22 keepout records the placement of the hash-pinned
manufacturer reference shrinkwrap `E0022_W22EX001.stp`, SHA-256
`7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6`.
Its declared native-to-world transform rotates +90° about X (native +Y
to world +Z and native +Z to world -Y), then translates by
`(0, 200.981, -47.498931)`. Native bounds
`(-110.5,-37,-110.5)..(110.5,65.798931,110.5)` therefore map to world
`(-110.5,90.481,-84.498931)..(110.5,311.481,18.3)`, with the LM centre at
`(0,200.981)` and native max-Y on the front datum z=18.3. These are cached,
hash-bound placement facts for the conservative proxy, not a runtime proof
that it contains every surface of the STEP or the installed U22. The
physical U22 and service harness still require the recorded fit check.

`PHYSICAL_MEASURE_REQUIRED = True`; qualification remains pending. The
MU reference omits both terminals and the datasheet leaves their carrier
and withdrawal geometry un-dimensioned. The maximum modeled pull is
12 mm, exactly the provisional exposed-tab length, so it has **zero
positive release overtravel margin**. The real MU, both Fastons and boots,
one-at-a-time withdrawal, cable, Y breakout, and selected external retention
must pass and record a physical dry fit before release. The completed record
must also bind each state to its exact candidate artifacts and document the
required 1g/3g/5g structural proof. Record all evidence and per-state signoff
in `obiwan_physical_qualification.md`; its current pending record and checksum
are bound into every R6F candidate manifest.

The proud-family route set is:

| Driver | Cable | Duct | R6P route |
|---|---|---|---|
| LM (U22/W22) | 2 × 2.5 mm² twisted | Ø8.2 | z=12.55 main past the 270-degree insert, then the retained rear outlet below the Ø190 opening |
| UM (MU10/10F) | estimated Ø7 twisted pair | Ø8.2 | B2/C7/V0/V1: outer U22 arc, broad-neck return, continuous G1 R14 handoff to (33.446, 301.492). V1L only: keyed alternate tail to the 283-degree rear-face aperture Q=(13.497063, 307.618796, 6.8) in `piece_mid_right` |
| T1+T2 (both ND25) | 2 × (2 × AWG24) | shared Ø6.0, flattened to W6.6 × H4.4 under the MU seat | strip feeders merge through the Ø6.8 step, rise along the left vase flank, and make one head-on scallop pierce near (−3.3, 430) |

The route suites sample the complete physical centerlines, including the
standard proud R14 handoff, the V1L alternate tail and rear-face handoff,
every R6F covered Z bump, physical crown crossing, printed-to-free owner
handoffs, independent LM lead, free rear UM/tweeter spans, and the R6F
cable's review-only G1 R20 turn to its Y breakout. Two separate Ø3.2 conductors then
retain R8-minimum slack paths into non-overlapping low-profile flag-Faston
boots and one-at-a-time 0/3/6/9/12 mm pull states. The printed terminal
approach is R15 and the exact G1 free continuation is R20; Ø3.2/R8 remains
provisional until the physical lead and manufacturer bend requirement are
measured.
For R6P, `test_um_eroded_outline_containment` erodes the exact outline by
duct radius plus the 1.6 mm proud-family skin, tests the complete
interpolated route `LineString` for containment (not just sampled
vertices), and reports true normal distance to the boundary. R6F instead
uses its state-specific 0.8 mm wall/0.85 mm roof checks plus final assembled
BREP shell subtraction and an independent 0.76..0.90 mm manufactured-BREP
normal-wall bracket. The former horizontal-gap approximation is not used
in either family.
Pilot and duct-pair checks retain the 1.5 mm separation rule. R6P ducts
cross its glue seams, so fish a cable or pull string through each open
segment during assembly. R6F UM/T cables must be dry-fished through their
buried owner segments and rehearsed across their free rear spans, while the
LM cable must be rehearsed over its free rear span and, in floor state,
through the integral stem continuation before driver installation.

## Variant C7 — LM knife-edge taper

An experimental replacement for the three LM-section pieces: full
18.3 mm around the W22, then the REAR face tapers (smoothstep over the
last 19 mm inside the flank/chamfer outline) down to a ~0.5 mm knife at
the edge -- the front face stays a full plane, exactly like the
crescent rear taper. It tests SL's "ideally the baffle would be even
thinner" in the band where the LM section's edges act (upper LM /
lower UM octaves), removing ~70 cm³ net (taper minus the T ribs).
The ducts sit at FIXED z from the rear face, so the binding rule is
z-interval containment: the rear cut over a duct must stay above
z_duct - r - skin (3.25 for the mid-plane mains; ~0 for the
rear-skinned T ducts).

- The cut fades in above the bottom strip (y 52..70: foot/bridge
  interface keeps full depth) and fades out toward seam B (y 270..~304:
  full-depth land, flush joint to the shared vase piece, dovetails at
  full section). The four seam-A teeth at ±66/±103 stay in full-depth
  lands.
- The standard R6P routing is shared by B2 and C7, so those pieces mix
  freely across the seams. Every duct remains inside the protected
  full-depth corridor; the tapered rear face carries no ribs or marks.
  This is asserted by
  test_c7_duct_corridor (`make check`) and verified with duct-envelope
  probes on the built piece solids.
- Print: same bed footprints as the B2 pieces; the taper prints
  front-face-down with layers shrinking as they rise (support-free).
- The historical `gen_lm_knife_draft.py` source is retained for research; its
  one-off concept raster is generated on demand and is not a release artifact.

## Variant V0 — minimalist UM vase (front slide)

An alternate piece_top for the low-crossover (3-4 kHz) experiments:
a REAR-side knife bevel (same side and philosophy as the C7 LM taper;
front plane fully intact) -- 18.3 -> ~0.5 over the last 2.8 mm inside
the flare/chamfer outline, fading out at the seam-B land and blending
into the crescent's rear taper above y~400. The band is capped at
2.8 mm by the shared O6.0 T duct (z=11.5) hugging the left vase
walls at ~1.6.
The standard top-piece routing remains identical for B2/C7/V0/V1; V0
mixes with B2 or C7 bottom/mids freely. One D5 x 2 captive station per side
uses the common Ø5.20 x 2.10 pause-and-bury cavity. The old orphan centres at
`(±46.000, 324.000)` were 5.263 mm outside the exact B2 flare: even the
Ø5.20 cavity was detached by 2.663 mm. Because no matching V0 scarf
attachment is released, the first correction moved each centre 8.663 mm
along the flare inward normal to `(±37.697, 326.470)`, leaving the R3.20
required land 0.20 mm inside the unchanged outline. That connected geometry
was not release-safe: the mirrored-left station was only 2.605 mm from the T
centerline, versus the required 8.000 mm (R3.20 land + 3.30-mm T half-width +
1.50-mm web). The all-constraint optimizer found a shorter qualified point at
`(-7.500, 322.300)`, but it had only 0.219 mm beyond the D82-cutout rule and
0.241 mm beyond the seam-B rule. The robust final adaptation keeps the clear
right station at `(37.697, 326.470)` and moves only the orphan left station to
`(-7.250, 321.200)`. This is only 0.416 mm farther from the rejected station
(30.899 mm total), while leaving 1.263 mm beyond the cutout rule, 12.363 mm
beyond the nearest-pilot rule, 0.549 mm beyond the grown-seam rule, and 18.014
mm beyond the T-route rule. Contained full-depth circular keeps make both
lands continuous and front-supported while printing front-face-down; the rear
axes, 45° conical closures, two 0.45 mm skins, driver seat, inserts, and
provisional rearward marked-pole directions remain unchanged. These stations
still have no released mate or pairing polarity.
The B2-family shoulders/wings do NOT fit V0. Guarded by
test_v0_duct_corridor and test_v0_captive_geometry (`make check`); STL:
lx521_top_v0_4of4_vase
(--variant v0).

## Variant V1 — 11.5 mm UM vase (minimum-thickness field)

The vase field is thinned to t=11.5 while retaining the standard
B2/C7/V0/V1 R6P route. The UM main hands off below seam B, so it does
not enter the vase; the
binding buried passage here is the shared tweeter duct, flattened to
W6.6 × H4.4 under the MU seat. The rear plane and all ducts are
unchanged, with a sharp step exactly at seam B (keys auto-trim to 11.5
on both sides). The whole top is flush at 11.5: the crescent taper is
re-derived on the 6.8..18.3 slab (same 4.0 clamp seat / 0.4 tips), and
the tweeter pair clamps an 11.5 septum (shorter standoffs; pair spacing
−6.8). The front-datum geometry keeps V1 front-flush with the LM
section. Pair with V1L for the complete thin proud-family baffle. 10F
mounting: 4 × Ø4.6 × 4.0 bores from
the new front for M3 x 3 x O5 brass heat-sets (floor z=7.5 stays 1.9
above the T-lane roofs at the ring crossings). Two D5 x 2 magnets per side
are fully buried in the common Ø5.20 x 2.10 captive flank-wall cavities
(zc 12.5/14.4);
B2 wall
pockets are skipped (B2 attachments do not fit V1). Guarded by
test_v1_field (`make check`); STL: lx521_top_v1_4of4_vase
(--variant v1). Thinner is possible only by externalizing cables to
rear-face grooves (~7) or through-bolting the 10F (~5-6) -- see the
constraint ladder in the V0/V1 discussion.

## Variant V1L — 11.5 mm LM section (front-flush)

The bottom + both mids thinned to t=11.5 (material z 6.8..18.3 above
the foot strip): the ENTIRE baffle then shares one front plane (use
with the V1 vase -- same rear plane, NO step at seam B). Binding
constraint: the Ø8.2 LM/UM z-window. The bottom strip keeps full 18.3 (smoothstep ramp
y=78 -> 96: full past the top pass-through seats +5, thin 10 short
of the D190 edge) for the fused foot / bridge hardware / cable
feeders; W22 heat-sets unchanged (floor keeps a 4.5 wall). It preserves
the common R6P entries, LM route, and tweeter route, but its UM outlet is
a keyed V1L-only alternate:

* LM Ø8.2 at z=12.55 follows the established plan and retained rear
  outlet below the LM opening.
* UM Ø8.2 at z=12.55 follows the r=119.5 outer U22 arc and broad-neck
  return, then substitutes the V1L alternate tail for the normal R14
  outlet. Its physical exit is centered at Q=(13.497063, 307.618796,
  6.8), radius 60.0 mm on the 283-degree terminal axis. The nominal
  outside continuation ends at (11.080158, 308.797599, −2.0). The
  entire alternate stays in `piece_mid_right`, below seam B; neither
  seam B nor the top/vase changes.
* T1+T2 SHARE one O6.0 duct ("ts") at z=11.5 up the LEFT flank -- the
  largest bore the notch corridor (D82 rim vs vase chamfer) admits --
  with a SINGLE scallop exit at (-3.3, 430); both pairs dress to their
  tweeters through the open scallop void. Pair feeders (O3.8, t1f
  z=3.7 / t2f z=9.5) cross the full-depth strip under the LM/UM
  columns and merge into a O6.8 z-step west of the LM column. 10F pilot pattern rotated to
  (58/148/238/328) so its left pair clears the lane and dive.
* Seam-A teeth at ±66/±103 stay in full-depth material and clear the
  crossings. Both seam-B male teeth project 6 mm into the V1 vase and
  are therefore trimmed by the vase-side rear slab as well: their rear
  plane is z=6.8, never the stock z=0 depth. The alternate tail never
  reaches seam B, and the right vase flank still carries no duct.

STLs: lx521_top_v1l_{1of4_bottom,2of4_mid_left,3of4_mid_right}
(--variant v1l) + lx521_top_v1_4of4_vase. Structural note: ~30% of
stock bending stiffness -- measure assembly modes before trusting the
W22 on it. The standard proud R14 coupon/grommet does not validate this
exit: dry-fish the printed V1L `mid_right` with the real cable and prove
the physical terminals, boots, measured withdrawal, and the dedicated V1L
split TPU grommet before final assembly.

## Variant Obi-Wan R6F — extreme two-collar barebone

Obi-Wan is no longer a flush-recessed copy of the full outline. Its
mandatory geometry is only:

- an LM flush carrier with Ø190 opening, Ø221.2 seat, **R113.0 structural
  radius**, and a smooth **R113.8 exposed side radius**;
- an UM flush carrier with Ø82 opening, Ø98.6 seat, **R51.7 structural
  radius**, and a smooth **R52.5 exposed side radius**;
- two compact half-lap pairs at x=±32.0, y=315.770 that establish the
  165.100 mm driver-center spacing without entering either flange seat. Their
  closure-web/base teardrops remain nominal **Ø9**, while each complete
  Z-owned cylindrical functional boss is locally **Ø9.8** to preserve the
  joint screen with the Ø4.6 receiver. Each LM rear Z-half owns a complete
  standalone Ø3.4
  rear-driven screw-clearance passage; each UM front Z-half owns a complete
  standalone rear-opening blind Ø4.6 x 4.0 receiver for an M3 x 3 heat-set.
  The receiver retains a **1.9 mm solid acoustic-front floor**, and the LM and
  UM ear halves retain a **0.20 mm axial gap**. Install the inserts in the
  individual UM print before assembly, then drive the screws from the LM rear;
  this interface has no washer, nut, or front bolt head;
- exactly six surface-normal D5×2 alignment/anti-rattle interfaces using
  captive Ø5.20 × 2.10 cavities: four LM and two UM. Each magnet is enclosed
  between 0.45 mm axial skins and a self-supporting 45° closing roof, with no
  glue or external access opening. The upper LM pair retains the world polar
  64°/116° axes (±26° from top), has no proud ear, and retains at least 2.2 mm cavity-edge to the nearest
  insert-pad edge and 0.86 mm to its route covers. The lower LM pair is
  captive in the two straight base side faces at
  **`(x,y,z)=(±32,18,12.55)`**, with outward normals `(-1,0)` on the left
  and `(1,0)` on the right. These base-side stations are identical in the
  floor and no-floor carriers and clear the lower inserts, buried routes,
  and bridge/integral-stand load path. The UM pair retains its
  50.5°/129.5° axes and z=15.1 datum. The LM and UM structural ring radii stay
  R113.0/R51.7, while their exposed sides are continuous cylindrical
  R113.8/R52.5 fairings. The fairings stop only inside the existing LM--UM and
  T--UM cusp/service regions; the LM--UM stop keeps the 0.40 mm inter-carrier
  gap open. At each ring-radial station the
  cavity construction datum is structural radius **+0.65 mm**, or **0.15 mm
  beneath the exposed surface**. The D5×2 cavity and 0.45 mm skin remain
  unchanged, and there is no local pad, boss, flat, or visible pocket cue. All
  six have
  **zero structural load credit**;
- Ac and Ae provide three coaxial captive receivers on each physical
  side—one at LM lower, one at LM upper, and one at UM—so all six carrier
  magnet axes have matching wing cavities. With the preserved 0.05 mm mating
  air gap, nominal paired magnet-face separation is **1.10 mm** at LM-upper
  and UM (`0.45 + 0.15 + 0.05 + 0.45`) and remains **0.95 mm** at the
  LM-lower base-side pair (`0.45 + 0.05 + 0.45`);
- six Obi-Wan-only LM axes at 0/60/120/180/240/300° on radius 104.75 mm,
  leaving the crown clear; both states own six ordinary blind carrier
  inserts;
- two compact direct UM-to-tweeter half-lap ears at **x=±24,
  y=421.5**. UM owns complete rear ears with Ø3.4 passages; the crescent owns
  complete front ears with rear-opening blind Ø4.6 x 4.0 receivers, 360°
  walls, and 1.9 mm acoustic-front floors, so each part is independently
  printable and no fastener breaks the acoustic front; and
- complementary, tangent-blended **full-depth** closure webs at both
  LM–UM and T–UM junctions. LM owns the lower LM–UM web, UM owns the upper
  LM–UM and lower T–UM webs, and the tweeter crescent owns the upper T–UM
  webs. Every owner spans z=6.8..18.3 and overlaps its own ring/crescent by
  0.40 mm; the local anti-void lens fills retain a separate 0.45 mm
  Classic-wall fusion land. Sub-resolution Boolean shards are discarded,
  while the independently printed owners retain the normal 0.05 mm plan
  seam. The functional bosses at both LM-to-UM and UM-to-tweeter are a
  Z-owned exception: each base closure teardrop remains nominal Ø9, but every
  complete cylindrical boss is locally Ø9.8. Each ear remains wholly in its
  assigned axial half, and the opposing print is fully notched over that half
  so the plan seam cannot split a bore, receiver wall, or front floor. The
  separate 0.20 mm axial gaps remain open. These are
  solid members behind the common z=18.3 front plane,
  not front skins over cavities; the only non-functional opening between the
  upper rings is the central ±6 mm T free-cable mouth; and
- an Ø8.2 UM passage buried only in the LM carrier and an Ø6.0 T passage
  buried in the LM and UM carriers, each with 0.8 mm minimum walls and a
  0.85 mm seat roof on its printed span. The UM cable exits the LM passage
  and remains free behind the UM carrier; there is no printed UM-carrier rear
  duct or D82 mouth. T exits the UM passage and remains free behind the
  tweeter crescent; the crescent has no printed cable arc. Their physical
  centerlines still cross at 82.67° with T above UM. The D7.8 LM lead is a
  modeled free span behind the carrier with no micro-duct or cover; only its
  minimum-radius 3.96 mm rear-open subtractive clearance is cut.

The load-bearing outer lips extend 2.4 mm past the flange-seat radii. Smooth
0.8 mm side fairings cover those structural lips at exposed radii R113.8 and
R52.5; they are clipped only inside the existing LM--UM and T--UM cusp/service
regions, and the LM--UM stop keeps the nominal 0.4 mm gap between the
structural collar envelopes open. The LM's six
insert-pad buttons, both pilot patterns, and flush seats remain; the old
5.5/7.5 mm annular floors and perimeter skin have been removed. Each seat
keeps only a 0.85 mm two-extrusion membrane. Narrow outer lips, local
blind-insert floors/bosses, calculated spokes, surviving buried-route covers, and
the explicit mechanical interfaces are the retained material.
The guarded closure acceptance clips the actual independently printable
LM/UM/crescent BREPs through fixed physical windows, not a window generated
from the closure target. It checks the actual front-face-down Bambu schedule
(0.20 mm first layer, then 0.16 mm layers) plus both sides of
each half-lap transition against frozen conservative front silhouettes,
proves the standalone LM clearance passages and UM blind receivers retain
their complete local Ø9.8 functional bosses, 360° walls, and the 1.9 mm UM
front floor; rejects exact 3-D owner overlap and proud material above z=18.3;
and rejects
any bounded residual void component beyond the declared fit seams, fastener
interfaces, route lumen, and T cable mouth. Thus a self-shrunk target, an open
cusp connected to a driver aperture, or a thin front skin over a rear cavity
cannot satisfy the release gate.

The canonical LM carrier remains one monolithic large-format release part.
Its mandatory front-face-down footprint is approximately 236.41 x 313.75 mm
in both states, so it is **not P2S-printable**. On a P2S it must instead be
printed as the mutually exclusive pair
`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` and
`lx521_top_obiwan_optional_lm_keyed_2of2_top.stl`; do not install either half
with the monolithic LM. The pair is cut from the finalized state-specific LM
at world **Y=172.481 mm** with an exact **zero-gap planar butt**, so both buried
route lumens cross the seam without being redrawn. The bottom owns two
symmetric Ø1.60 cylindrical pins at `x=±109.187`, `z=14.30`; each points world
+Y normal to the seam, has 0.50 mm root overlap, and engages the top by
2.40 mm (2.90 mm total male length). The top owns two 2.65 mm-deep blind
sockets with 0.12 mm radial and 0.25 mm end clearance: right is round Ø1.84,
while left is X-relieved to 1.96 × 1.84 mm. This round-plus-relieved constraint
tolerates ±0.30 mm relative pitch error across the 218.374 mm spacing instead
of binding like two tight round sockets. Two small exterior support lands
grow outward from the R113 lip, outside the LM recess. They retain at least
0.50 mm local radial and blind-end wall, 0.05 mm recess plan clearance, and
0.13 mm conservative W22-flange plan clearance. Their worst-case reach is
R114.4036: 1.4036 mm beyond the structural R113.0 ring and 0.6036 mm beyond
the finalized R113.8 visible fairing. Ac and Ae include a hidden 0.25 mm
clearance pocket around each land at the carrier interface, wholly between the
front and rear faces. CAD compatibility is gated; physical printed fit remains
coupon-qualified. With the monolithic LM these pockets are only small hidden
local reliefs; the three magnetic datums and primary wing retention are unchanged.
The pins create no extra screw or standalone retention/load credit.
Print both halves front-face-down. Assemble them front-face-down on one flat
datum, bring the top toward the bottom along world -Y so both pins enter
together without flexing, and confirm full seating, coplanarity, and
route-seam continuity. Then install the LM driver:
its flange and all normal LM fasteners are the installed structural splice
across the seam. Both keyed halves now print front-face-down with only
in-plane bed rotation. The former Z26°/Z45° and floor-bottom X=−90° footprint
figures are obsolete because those out-of-plane orientations cannot support
the captive-magnet pause. Revalidate the generated front-down footprint on the
selected printer. Each horizontal Ø1.60 pin is four nominal 0.4 mm nozzle
widths: release requires a process-matched coupon and sliced preview proving
both complete pin paths, both blind mouths, the exterior lands, and continuous
minimum-wall paths.
This option is still **PENDING** until two-pin/socket fit, full-seat and
coplanarity evidence, route-seam inspection, cable pull-through, and
driver-installed 1g/3g/5g proof are recorded; monolithic-LM evidence does not
qualify the split form.

The captive-magnet release audit does not create monolith G-code or a fake
monolith pause. Instead, every monolith station is source-contract matched to
the corresponding same-state keyed half, and coverage is accepted only after
that actual half passes the normal P2S cavity/toolpath gates. The pause
manifest distinguishes `not_p2s_printable__cavity_covered_by_exact_split`
coverage from actual keyed-half pause groups. Scaling, clipping, tilting, and
virtual-bed overrides are prohibited because all pieces must retain the same
front-face-down texture and insertion geometry.

The floor stand is not an add-on. In `floor_stand/`, both the canonical LM
carrier and the optional keyed bottom own the complete floor structure: a
full-depth W64 stem softly blended into the lower LM cap, a W64 × 18.3 mm
rectangular foot extending from `z=-150..18.3`, a true R12 internal root,
and the rear NL8 panel/service cavity. The floor plane is world `Y=0`, so the
LM-axis-to-floor distance is exactly **200.981 mm**. The rear panel is
`z=-150..-146`, 44 mm high, with an Ø31 NL8 cutout centered at
`(x=0,y=22)` and four Ø3.2 holes on a 29.2 mm square. The outer stem/foot
is solid except for the necessary connector service cavity and three buried
continuation lumens (LM Ø9, UM Ø8.2, shared T Ø6) through R14 turns.
There is no yoke, open rail, support fastener, or
`lx521_top_obiwan_addon_mount_floor_support` file. All six LM driver insert
bores are ordinary blind carrier bores in both states.

The floor and no-floor state geometry intentionally diverges only behind the
wing-contact face. Their lower-LM front exterior is one common profile: the
union of the former broad no-floor cubic shoulder and the former floor stem,
continued to world Y=0. This removes the old 25.57 mm width discrepancy near
Y=94.665 and the old 14 mm height discrepancy. Ac and Ae therefore use one
unchanged saddle and sit flush on either state; the floor's deep load path and
the no-floor bridge's four blind inserts remain independent.

The optional V1 face-to-face tweeter crescent remains a separate add-on with
complete local-Ø9.8 blind-M3 receiver half-laps, 360° walls, 1.9 mm front
floors, and no printed T-cable arc.
Obi-Wan has no printed grommet; selected external cable retention remains a
physical-fit item, and cable load must never reach the MU tabs. No-floor
support is not an add-on: a 62 mm insert-bearing plate with soft cubic
shoulders is fused into the LM carrier around the unchanged holes at
(±20,20)/(±20,70). Two rear-facing cable mouths at x=±8, y=82 enter flush at
z=5.3 and rise internally; the acoustic front stays solid. The plate occupies z=5.3..18.3, flush
with the acoustic front and no deeper than the six existing LM insert-pad rear
faces. It has no X-frame, acoustic-front window, rear rib, or other depth structure. Its four
Ø6.4 × 6.8 bores open from the rear and leave a 6.2 mm solid front floor.
No bridge geometry extends behind the existing LM pad envelope. No-floor
geometry is otherwise unchanged. Select the tweeter module and an
independently qualified external retention method required by the
installation.

The conservative room-temperature PLA Tough+ screen assumes a 4.0 kg
installed mass, y=230 mm center of mass, and 70 mm rear offset. The 62 mm
insert core is reduced by the complete Ø8.2 and Ø6.0 entry lumens, with no
thin-skin credit, to a conservative **47.8 mm** design section. Exact 0.01 mm
sampled cuts through the soft outline retain at least 53.5 mm. At 13.0 mm
depth the credited section's in-plane/rear moduli are **4950.5/1346.4 mm³**.
Conservatively summing in-plane and rear bending gives approximately
**3.28/9.84/16.41 MPa** and safety factors **2.44/1.83/1.10** at sustained
1g/8 MPa, transient 3g/18 MPa, and transient 5g/18 MPa. Its 68° lower-ring fusion cradle physically follows the plate
to z=5.3, but the existing ring lip begins at z=6.8, so the load screen
credits only the actual 11.5 mm-deep monolithic interface. That interface
retains 118.5 mm effective width after deducting one Ø8.2 UM tunnel plus the
complete Ø6.0 tweeter tunnel. Its in-plane/rear section moduli are
**26908.4/2611.6 mm³**, with biaxial factors about **6.37/4.78/2.87**. The
combined normal/rear 5g insert reaction is 434.2 N, giving 1.38 pull-out
safety factor under the same assumed 600 N per insert; magnets contribute
exactly 0 N. These are design calculations, not certification. The two-ear
upper joints use the actual 0.43 kg MU + 0.20 kg tweeters plus carrier,
crescent, wire, and hardware allowance: 0.85 kg total over conservative
120 mm plan and 70 mm rear levers. Both receiver interfaces co-govern with
contact factors about 2.85/2.14/1.28 and M3 screw-tension factor about 1.28 at
5g. Those screens do not independently qualify either heat-set process,
receiver wall, or 1.9 mm front floor; the 5g pullout demand is approximately
393.9 N per insert. Magnets receive no credit in any case. The
finished print, inserts, screws, stock bridge, and installation substrate
remain
inside the physical proof-test boundary. The factors apply only near
room temperature; direct sun, a closed vehicle, or any service
approaching Bambu's published 61 °C heat-deflection temperature
invalidates them.

The integral floor stem has its own conservative closed-form
rectangle-minus-lumens screen. It is explicitly **not FEA, certification, or
physical release evidence**. The 4.0 kg load model uses `y=230 mm`, a 70 mm
rear eccentricity, and 1g/3g/5g load cases. The net W64 × 18.3 root deducts
the complete Ø9/Ø8.2/Ø6 lane sections; magnets and both optional concealed
split pins/sockets receive 0 N credit. Current project-allowable results are:

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g diagnostic deflection | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 4.22 / 2.73 / 1.64 | 1.18 mm | analytical pass |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 6.09 / 3.85 / 2.31 | 1.05 mm | analytical pass |
| Bambu PLA Lite | 2.69 / 1.73 / **1.04** | 3.73 / 2.40 / 1.44 | 1.40 mm | **FAIL at vertical 5g; provisional data** |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.85 / 2.49 / 1.49 | 1.49 mm | analytical pass |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.47 / 2.90 / 1.74 | 1.17 mm | analytical pass |

PLA Lite is provisional because no product-specific official TDS was
available; its comparison-sheet input is fail-closed. Every reported stress
includes an explicit **1.25 root geometry/model factor**. PLA Lite fails the
1.05 minimum at vertical 5g and is not an accepted material under this screen;
the other four pass only the analytical thresholds (2.0/1.5/1.05 and ≤2.0
mm), not the physical gate. The result is valid only with a **100% local-solid
modifier through the complete stem/root**; sparse infill receives no structural
credit. The W64 footprint also tips before material failure under lateral
acceleration: the calculated free-standing thresholds are only 0.139g
lateral, 0.348g rearward, and 0.384g forward. Therefore the installed speaker
**must use a positively attached anti-tip tether or anchor**; the rectangular
foot alone is not a safety restraint. Qualification remains pending a
2×-service-load 24 h proof at 35 °C with no cracking/whitening and residual
set ≤0.5 mm or ≤10% of loaded deflection, followed by at least 168 h at
1.5× service load for creep. Service above 35 °C, direct sun, radiator hot
soak, changed material/process, or the optional keyed LM form requires its own
recorded qualification.

The terminal service reference lies on the **283-degree axis**, exactly
midway between mounting screws 238 and 328 degrees; coupon 9 is the
physical clocking witness—there is no cosmetic collar engraving. Printed
UM conduit stops at the LM owner boundary; the Ø7.0 cable is free behind the
UM carrier, follows the modeled R15 approach to the 283° reference, and
continues through the R20 service turn to
a Y breakout comprising a 4 mm-long OD8 collar and two OD4 branch sleeves.
Two provisional Ø3.2/R8 slack leads enter separate provisional low-profile
flag Fastons. Service states pull one connector at a time through
0/3/6/9/12 mm while the other remains installed. The STEP fit model adds
closed Ø98/Ø80/Ø60 MU and conservative stepped W22 rear-body keep-outs.
The W22 source and transform are hash-pinned and recorded above.

`PHYSICAL_MEASURE_REQUIRED = True`, so terminal qualification remains
pending. The raw MU reference is an open acoustic surface and omits the
terminals; the datasheet also does not dimension them. The 12 mm maximum
pull exactly equals the provisional exposed-tab length and has zero
positive release overtravel margin. Measure and record carrier radius,
tab pitch/projection, 8.5/9.5 mm proxy body widths, flag orientation,
polarity, real withdrawal, cable/Y fit, and the selected external cable
retention before committing a full print. The proxy is a keep-clear aid,
not manufacturer
geometry or release proof.

See VARIANTS.md for the variant/add-on catalog and the
compatibility matrix, and PRINTING.md for filament choice, print settings, fastener
torques, and insert installation.

## Magnet attachment (swappable shoulders/wings)

Attachments mount with neodymium N52 D5 x 2 disc magnets (superimanes
ref D-05-02-N52; supplier figure 0.68 kg/pair; 12 needed + spares) so B2 <-> A-comp <->
B1 are interchangeable without magnet adhesive. TWO sites per flank side
(4 magnets in the base total). Every released base and receiver uses the same
pause-and-bury captive cavity derived from
`coupons/obiwan_ae_embed/obiwan_ae_embed_coupon.py`: actual magnet D5.0 × 2.0,
internal cavity Ø5.20 × 2.10, 0.45 mm plastic skin at each axial face, an
upward-open printable cradle during insertion, and a self-supporting 45°
closing roof. The finished magnet is completely buried, has no glue and has
no external access opening. The supplier's 0.68 kg/pair figure is not proof
of achieved pull through the production pair's two 0.45 mm skins plus 0.05 mm
interface gap (0.95 mm nominal magnet-face separation); qualify that retention
with a physical pull test. The outline kinks/corners and saddles provide
registration; magnets receive no shear or structural-load credit.

The released upper site is a rounded tangent datum on the true D102.11 arc.
Keeping the coupon-qualified 6.4 mm-wide retaining land therefore creates a
strictly local planar base boss: 0.134666 mm maximum and 0.430824 mm2 in plan
(3.812789 mm3 over the standard 8.85 mm land height). The mating receiver is
relieved to its +0.05 mm face plane, at most 0.184666 mm from the old curve.
V1/V1L use the same plan adaptation over their own Z span. At the lower
straight site, a 0.031572 mm maximum base trim makes the nominal 0.05 mm gap
physical rather than metadata. These adaptations do not move any magnet
centre or axis and receive no structural-load credit.

The existing approximately 0.05 mm piece-to-piece air gap remains. Because
both opposing magnets are now behind 0.45 mm skins, nominal magnet-to-magnet
separation is **0.45 + 0.05 + 0.45 = 0.95 mm**. This replaces every old
face-flush/adhesive-pocket assumption. Magnet axes, site positions, and
polarity are unchanged. Neo stacks ship uniformly oriented; sharpie-mark one
pole before separating the stack and follow the site-by-site polarity table in
[`CAPTIVE_MAGNET_PAUSE_MANIFEST.md`](review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md), including mirrored
parts. Never infer polarity from left/right appearance:

| Site (right; left mirrored) | Wall | Serves | Placement rationale |
|---|---|---|---|
| (40.0, 322.4) | flare, waist-kink end | A bottom shoulder, B1 wing lower end | the flank's farthest point from the UM driver (59.2 mm); captive cavity and local backing clear the T duct |
| (17.88, 420.37) | crescent arc, theta=-69.5 deg | A top shoulder, B1 wing top end | as far down-arc as the receiver permits; the captive envelope and local backing are rechecked against the chamfer, rear taper, front face, and TS duct by `make check` |

For the legacy R6P B2/A/B1 arrangement specifically, magnet count per baffle
is 4 base + 4 per attachment set (12 with both sets; 24 for a stereo pair).
Obi-Wan, its keyed alternative, Ac/Ae, and the calibration coupon use their own
per-part counts in the authoritative pause manifest; do not apply this R6P
count to them.

Polarity discipline: use the "MARCADO NORTE" batch and the manifest's local
axis convention; verify every magnet against a marked master before its
insertion pause. Once the roof is printed the polarity cannot be corrected.
Insertion direction is independent of polarity: every released site must be
loaded vertically downward from above the paused part (its +Z side) along
print -Z, exactly `print_insertion_direction_xyz = [0, 0, -1]`. The catalog
consumer fails closed if the front-face-down transform produces any other
loading direction.

The other inventory magnets are not suitable here:
D18 exceeds the 18.3 mm wall, adhesive tape magnets are too weak for a
structural joint, D10x5 only fits the receiver side.

Print every released baffle/acoustic part front-face-down, including
non-magnet pieces such as the Obi-Wan tweeter crescent. For magnet-bearing parts,
pause at the listed Bambu Studio marker, which is the first layer whose
toolpath begins closing the roof after the last fully open layer. At the pause, insert the exact
count vertically downward from the +Z side along print -Z, with the polarity
shown in the manifest; ensure every disc is fully seated below the completed
layer, clear the toolpath, and resume. Do not use the
coupon's **UM 5.96 mm / LM 8.52 mm** regression markers for unrelated parts;
those values apply only to matching Obi-Wan coupon geometry on the tested P2S
0.4 mm / 0.16 mm profile.

The sibling `.print.json` beside each released nonpolar STL is mandatory
machine-readable print authority, not optional documentation. It binds the
exact STL hash and size to X180 plus an optional bed-normal Z rotation and the
origin translation. Keep each pair together and rerun `check_manifold.py`
after copying artifacts; do not print a missing, orphaned, hash-stale, tilted,
or translation-inconsistent pair. The only release STLs intentionally without
this acoustic/front-down sidecar are the floor-state polar base and rotor,
which are measurement jigs with their own flat orientation.

Retired concept-only drawings, historical diagnostic renders, and non-release
fit coupons are excluded from the released part migration because they are not
printable release outputs. The
`coupons/obiwan_ae_embed/` coupon is not an assembly part; it remains the
physical reference implementation and regression evidence. V0's hypothetical
scarf mate is also excluded because no printable mate has been released.

## Printing

- Follow PRINTING.md and run the applicable coupons first.
- R6P baffle pieces print front-face-down; use the documented support
  blockers for internal ducts and the floor-foot strength modifier.
- The canonical floor R6F LM with its integral stand and the no-floor monolith
  are not P2S-printable at their approximately 236.41 x 313.75 mm front-down
  footprint. On a verified larger-format machine, print them front-face-down
  and keep support out of buried route mouths, connector cavities, and
  free-cable paths. Preview
  every closed bump and optional add-on separately.
- If the optional Obi-Wan LM keyed split is selected instead, print both halves
  front-face-down with in-plane rotation only. Recheck each generated
  footprint against the actual printer; this is the required P2S form. It replaces,
  rather than accompanies, the monolithic LM. Its two concealed Ø1.60 +Y
  pins and right-round/left-X-relieved blind sockets sit on small exterior
  lands outside the LM recess. Their worst-case reach is R114.4036: 1.4036 mm
  beyond the structural R113.0 ring and 0.6036 mm beyond the finalized R113.8
  visible fairing.
  Ac/Ae include matching 0.25 mm interface pockets around those lands;
  physical fit remains coupon-qualified.
  Preview the four-nozzle-width horizontal pins, ≥0.50 mm
  socket/end walls and both lands, then qualify their simultaneous straight-
  pull fit and actual U22 clearance with a process-matched coupon/print.
- Ac and Ae wing sides each print as lower, middle, and UM segments cut from
  the finalized monolith. The lower segment owns the 7/9/4 mm male dovetail
  into the middle segment; the middle segment owns the 7/8.5/4 mm male
  dovetail into the UM segment. Both female complements use 0.05 mm clearance.
  The clearance collapses to exact closure over the final 2 mm at each exposed
  split endpoint, and neither key may grow the installed plan or depth
  envelope. Both complete keys retain at least 2.0 mm measured exterior plan
  ligament. Qualify the fit on a process-matched coupon before a complete wing.

## Assembly

**R6P:** dry-fit and tune the coupon before gluing. Assemble seam C,
then A, then B on a flat front-face datum; fish each cable segment as
its seam closes. Set the driver and rear bridge inserts square, mount
the drivers at the low torques in PRINTING.md, and re-torque after the
first preload-settling interval. For V1L, dry-fish the actual keyed
`mid_right` alternate outlet before glue-up; confirm that its aperture
is centered at the 283-degree rear-face witness and physically rehearse
the real Fastons, boots, service loop, dedicated split TPU grommet, and
measured withdrawal before installing the MU.

**R6F:** first prove the real MU terminal/Faston fit with coupon 9 and
the review STEP. If the optional LM print split is selected, use both halves
and omit the monolithic LM. With both front faces down on one flat datum, seat
the bottom half's two symmetric Ø1.60 +Y pins simultaneously in the top
half's right round and left X-relieved blind sockets by bringing the top along
world -Y without flexing or twisting. Verify full seating, coplanarity, the
closed route seam, and unobstructed UM/T cable pull-through. Hold that
registration while lifting the LM for driver fit-up. The pins/sockets have no
standalone retention or load credit; only the installed LM flange and its
normal fasteners splice the seam.
Install both LM-to-UM M3 inserts through the individual UM carrier's rear
receiver openings before assembly. On a flat front-face datum, engage the
rounded x=±32.0, y=315.770 half-lap ears while preserving their 0.20 mm axial
gap, then drive two M3 screws from the LM rear through its Ø3.4 clearance
passages into the UM's blind Ø4.6 x 4.0 receivers. Use no washer/nut and do not
drill through the 1.9 mm UM front floor. Verify the
165.100 mm axis spacing. Place the LM cable in its short free span, dry-fish
the UM and shared tweeter cables through their buried owner segments, and
rehearse the free UM span behind the UM carrier and free T span behind the
tweeter crescent. Confirm the physical T-over-UM crown crossing and covered
328°/58° T-route bumps are unobstructed. All six LM screws use the same
ordinary blind carrier inserts in floor and no-floor states. In floor mode,
verify that the integral W64 stem/foot, R12 root, three buried continuations,
connector service cavity, and NL8 panel are unobstructed, then install a
positive anti-tip tether or anchor before loading the assembly. In no-floor
mode, bolt the stock bridge directly to the four rear-entry inserts in the
fused front-flush LM web. Fit only the selected add-ons;
fasten the crescent at x=±24, y=421.5 with rear-driven M3 screws into
its blind inserts. Obi-Wan has no TPU tunnel clip; keep the selected external
cable retention clear of the buried-route mouths, free cable, and service
envelope. Clock the physical terminals between screws 238/328 on the 283°
coupon-9 service axis. Confirm each measured flag Faston fits separately,
its lead follows the polarity-specific slack path, and each one-at-a-time
0/3/6/9/12 mm review state clears the installed opposite connector and
both drivers before final driver installation. This is still not release
proof: the 12 mm state has zero positive overtravel beyond the provisional
12 mm exposed tab. The numbered procedure and hardware
cautions are in PRINTING.md.

**Ac/Ae wing segments:** slide each through-local-thickness dovetail along
local Z while holding the acoustic front faces on a common datum. The two keys
provide XY registration and in-plane interlock only; they do not independently
retain the segments against Z separation and carry no structural-retention
claim. If handling or the experiment requires Z retention, use the same
documented rear tape or light-bond method on every compared wing. This Ac/Ae
contract supersedes their former wavy butt-glue/epoxy seams only; it does not
change the adhesive instructions for legacy R6P attachments or other splits.
