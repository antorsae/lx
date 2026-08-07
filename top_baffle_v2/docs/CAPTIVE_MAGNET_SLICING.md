# Captive-magnet slicing and pause audit

`scripts/slice_captive_magnets.py` is the offline release pipeline for the buried
D5 x 2 magnet system. It resolves the installed Bambu presets, slices every
catalogued STL that fits the P2S bed in its mandatory front-face-down
orientation, reads the actual G-code layer schedule, checks the open cradle
and closing roof toolpaths, and writes both the pause audit and ready-to-open
`.gcode.3mf` projects. The catalog contains exactly 64 magnet-bearing release
STLs, of which 62 are actual P2S slice jobs. The two canonical Obi-Wan LM
monoliths are catalogued release artifacts but are explicitly classified as
not P2S-printable; their cavity audits are supplied by exact same-state keyed
halves as described below. The number 114 is the count of individual magnet
stations, not a claim that this pipeline slices 114 STLs. Separately, the full
repository release contains 110 STL/`.print.json` pairs. This magnet
catalog excludes every nonmagnet-bearing print, including the Obi-Wan tweeter
crescent and coupons, so these targets are not yet a complete-print-set slicer.

It has no printer transport code. It does not use MQTT, FTPS, LAN discovery,
or a printer address, and cannot upload or start a print.

## Fixed process contract

- Printer: Bambu Lab P2S, 0.4 mm nozzle
- Process: 0.16 mm High Quality
- Wall generator: Arachne
- Nominal wall paths: 0.42 mm outer / 0.45 mm inner
- Detect thin wall: enabled for the captive-magnet retaining skins
- Filament used for the general release: Bambu PLA Tough+. Both standalone
  Obi-Wan keyed LM bottoms (`01a` no-floor and `01b` floor) and both combined
  `01+02+03+04` core plates use the separately pinned saved TINMORRY PETG-GF
  profile.
- Support: disabled by default; enabled only for both keyed LM halves and the
  UM carrier in both stand states. Every supported project pins **Enable
  support**, **On build plate only**, **Support critical regions only**, and
  **Remove small overhangs** globally and on every object.
- Duct safety: every support-enabled carrier carries a hash-bound modifier
  over every functional lumen it owns. The final G-code must pass the
  independent deposited-support-bead-versus-duct collision gate with zero
  intersections.
- A Bambu Studio `floating cantilever` diagnostic is release-blocking even
  when the slicer labels it non-critical and exits successfully.
- Orientation: **front face down for every piece**

The production overrides retain the general project's structural profile: six wall
loops as the requested **maximum** where the local geometry has room, six top
and five bottom shell layers, and gyroid infill at the artifact-specific 30 or
40 percent requirement. Six loops is not a promise of six parallel paths in a
feature that is physically thinner. In particular, every 0.45 mm captive-
magnet retaining skin must resolve to exactly one bounded Arachne variable-
width bead or traversal; `detect_thin_wall=1` remains pinned and the actual
toolpaths are audited independently. The configured nominal outer-wall
width is not an exact-width promise for this thin feature. Transverse skins
have a nominal 0.42--0.67 mm bound. The audit allows only a 0.005 mm
lower-side Arachne tolerance (an effective 0.415 mm floor); the pinned full
release reaches 0.415656 mm at its narrowest path. The orthogonal 0.45 mm
coupon spacing emits a deterministic 0.484336 mm bead at 0.16 mm layers;
angled Obi-Wan LM/UM skins reached 0.586 mm and the legacy V1 adaptive inner
bead reached the full-run maximum of 0.661027 mm. Every transverse station must independently retain at
least 2.0 mm of path-width-aware free loading slot (observed full-run range
2.047--2.083 mm). Axial skins remain nominally within 0.42--0.65 mm; only a
0.000005 mm serialization tolerance is allowed below the lower bound. The
observed maximum is 0.631 mm, and the actual sliced
loading aperture must still be at least D5.0. The toolpath audit hard-fails if
that single path is absent, doubled, broken, outside its width bound, or moved
into the loading aperture.

X-Y hole compensation is fixed at 0.00 mm for this pipeline. An empirical
slice with +0.05 mm deleted the 0.45 mm retaining skin, so the general fit-
tuning starting point does not apply to captive-magnet artifacts.

The no-floor LM bridge/root requires local solid material. Its structural
PETG-GF job embeds a deterministic parameter-modifier mesh, hash-bound to the
released `01a` STL and print sidecar, with exactly 100 percent zig-zag infill
through the complete bridge/root and both lower LM boss axes. Global infill
remains 40 percent gyroid. The integral-floor keyed LM uses global 100 percent
zig-zag because Bambu Studio rejects gyroid at 100 percent. Both standalone
keyed LM bottoms and both combined core plates use eight walls and the exact
saved TINMORRY PETG-GF preset. The archive audit rejects a missing, extra,
misbound, or settings-drifted modifier.

The ready-project archive is not accepted merely because its profile flags
look correct. Its object overrides, embedded blocker mesh, source/STL/3MF
equivalence, and final support toolpaths are all checked before shelf
promotion. Any support bead entering a functional duct is a release failure.

Both keyed LM halves and the UM carrier in both floor states carry the support
override:
`enable_support=1`, `support_on_build_plate_only=1`,
`support_critical_regions_only=1`, and
`support_remove_small_overhang=1`. The four settings are pinned in the global
project profile and on every model object. Each supported carrier also embeds
its state-specific duct blocker. Ordinary buried-magnet jobs pin the same four
fields to `0` globally and per object, and are rejected if they emit any
support feature. The
cavity audit still rejects support or any other extrusion that blocks the
last-open D5 loading aperture, while the separate duct gate rejects every
support bead that intersects a functional cable lumen.

The general process selector lives in `captive_magnet_slicing_profile.json`;
the core-only structural selector lives in
`captive_magnet_slicing_profile_petg_gf.json`. The latter has an explicit
artifact allowlist that excludes every Ac/Ae wing and shoulder. The
pipeline recursively flattens the installed system presets' `inherits` and
`include` chains into complete machine/process/filament JSON snapshots. Every
source preset, dependency, flattened profile, STL, G-code file, result file,
and evidence image receives a SHA-256 digest. A profile drift therefore cannot
silently reuse an older slice.

## CAD-to-slicer catalog boundary

The remote CAD build writes
`review/captive_magnet_release_catalog.json`. Its schema is
`captive_magnet_release_catalog.schema.json`.

The catalog records the 64-hex content-addressed osado source snapshot and a
SHA-256 for every listed artifact source. Obi-Wan records additionally bind the
exact state `.obiwan_stage/manifest.json`; Ac/Ae records bind their facts and
transaction manifests. The catalog itself is rendered beside the current
authority, normalized against the checked-in schema, and checked against all
artifact bindings before one atomic replacement. A failed generation cannot
destroy or expose a partially validated catalog.

Each artifact identifies its released STL, source files, state/variant, and
one or more magnet stations. CAD station facts remain in the installed/source
frame, while the exporter's exact X=180 plus in-bed-Z transform and canonical
translation are recorded as a 4x4 `source_to_stl_matrix`. The slicer consumer
applies that matrix to points and vectors without importing OCC:

Every production transverse family uses one source magnet plane at
**Z = 15.10 mm**. In particular, Obi-Wan LM-lower, LM-upper, UM, and their
matching Ac/Ae receivers may not drift onto separate source-Z or roof planes.
The interface contract records a 0.05 mm **solid receiver construction
standoff** and a 0.00 mm physical mating gap; the standoff must never be
reported as an air gap. Nominal opposing magnet-face spacing is 0.95 mm for
straight pairs, 1.09 mm for standard curved pairs with the 0.14 mm base inset,
and 1.10 mm for Obi-Wan ring pairs with the 0.15 mm carrier inset. The cavity
contract is wholly internal: a local exterior box, cap, flat, dent, boss, or
other magnet-location cue is a release failure.

The current fail-closed inventory is 64 magnet-bearing released STLs and 114
individual magnet stations: 22 STLs / 45 stations in each floor state plus
20 shared Ac/Ae segments / 24 stations. Each Ac/Ae side exposes the unchanged
three-piece A set and the two-piece B alternative; the B lower is identical to
the A lower, while the fused B upper carries both LM-upper and UM stations. A
count change is a source-level release change, not something the generator
silently accepts. Non-print compound STEP review packages are not separate
prints; their constituent STLs are present in this inventory.

### Oversized canonical Obi-Wan LM policy

The floor and no-floor canonical
`lx521_top_obiwan_core_1of2_lm_carrier.stl` monoliths each have an approximately
236.41 x 313.75 mm mandatory front-face-down footprint. They therefore do not
fit the P2S 256 x 256 mm bed. They remain valid large-format release CAD/STL,
but the P2S audit does **not** slice them, generate G-code for them, or publish
a pause marker for them. It never grants a virtual larger bed, scales, clips,
or tilts a monolith to manufacture a passing result.

Each of the monolith's four cavity stations maps one-to-one to the matching
station in the **same floor state** on
`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` or
`lx521_top_obiwan_optional_lm_keyed_2of2_top.stl`. Before accepting that mapping,
the consumer requires an identical source-space cavity contract: station and
closure kind, cavity and seated centers, installed/marked axes, actual face
and material-inward direction where applicable, magnet/cavity dimensions,
both skins, roof angle, bury/apex planes, retaining geometry, interface gap,
and polarity. Here `interface gap` is the catalog's legacy field name for the
0.05 mm solid receiver construction standoff; the physical mating gap is
zero. The keyed proxy artifact must then pass the ordinary P2S slice
and every per-site cradle, aperture, clearance, closure, and sealed-roof gate.
Any missing, duplicate, cross-state, or contract-mismatched proxy fails the
release.

The current 64-STL/114-station catalog consequently produces 62 actual P2S
slices with 106 actual pause-station rows, plus eight canonical monolith site
contracts covered by those exact split slices. This is coverage of duplicate
geometry, not a printable monolith pause. The authoritative manifests label
each monolith `not_p2s_printable__cavity_covered_by_exact_split`, list its
hashed proxy evidence separately, and never place it in a pause group.

The following is an intentionally abridged field-layout illustration, not a
standalone schema-valid catalog instance (the real generated catalog carries
the full 64-hex source revision, closed root contract, hashes, inventories,
exclusions, and complete artifact IDs):

```json
{
  "schema_version": 1,
  "source_revision": "0000000000000000000000000000000000000000000000000000000000000000",
  "artifacts": [{
    "id": "no_floor_stand:Obi-Wan:lx521_top_obiwan_core_1of2_lm_carrier",
    "state": "no_floor_stand",
    "variant": "Obi-Wan",
    "part": "lx521_top_obiwan_core_1of2_lm_carrier",
    "stl": "../no_floor_stand/stl/lx521_top_obiwan_core_1of2_lm_carrier.stl",
    "source_files": ["../src/lx521_baffle/obiwan/carriers.py"],
    "print_orientation": "front_face_down",
    "source_to_stl_matrix": [
      [1, 0, 0, 0],
      [0, -1, 0, 453.457],
      [0, 0, -1, 18.3],
      [0, 0, 0, 1]
    ],
    "sites": [{
      "name": "lm_upper_right",
      "closure_kind": "transverse_gable_45deg",
      "cavity_bury_roof_start_print_z_mm": 5.80,
      "roof_apex_print_z_mm": 8.40,
      "expected_pause_marker_z_mm": 5.96,
      "cavity_diameter_mm": 5.20,
      "cavity_depth_mm": 2.10,
      "face_skin_mm": 0.45,
      "inner_skin_mm": 0.45,
      "polarity_instruction": "marked/N pole points along the listed axis",
      "installed_marked_pole_axis_xyz": [0.438371, 0.898794, 0],
      "cavity_center_xyz_mm": [48.878, 301.196, 15.10],
      "seated_magnet_center_xyz_mm": [48.900, 301.242, 15.10],
      "actual_face_xyz_mm": [49.536, 302.545, 15.10],
      "material_inward_xyz": [-0.438371, -0.898794, 0],
      "marked_pole_axis_xyz": [0.438371, 0.898794, 0],
      "insertion_direction_xyz": [0, 0, 1]
    }]
  }]
}
```

As a schema-compatible alternative, a producer may provide the station
centres and vectors already transformed beneath a nested `print_space`
object. The release generator uses the source-facts-plus-matrix form above.
`insertion_direction_xyz` describes the physical motion used to lower the
magnet, not its polarity. Source +Z becomes print `[0, 0, -1]` after X180:
insert vertically downward from the +Z side of the paused part. Any released
site whose transformed insertion direction is not exactly print -Z fails
closed.

The consumer rejects any orientation other than `front_face_down`. The
release exporter may rotate a part by X=180 degrees and then rotate in the bed
plane about Z, but it may not tilt or auto-orient it. Bambu's slicer is run
with arrangement enabled and auto-orientation/rotation disabled; the pipeline
also compares the sliced bounding-box dimensions to the STL dimensions.

## Running the local Bambu chain

CAD/STEP/STL generation remains remote on `osado.lan`; Bambu Studio remains on
the local Mac. If the promoted catalog is missing or stale, first run this as a
separate command:

```bash
make obiwan_release
```

Then use one of the local-only targets:

```bash
# Resolve and verify the installed Bambu profiles without slicing.
make bambu_profiles

# Write the resolved profiles and exact per-STL commands without slicing.
make bambu_dry_run

# Generate ready projects for the Obi-Wan-filtered diagnostic subset.
make bambu_slice_obiwan

# Slice/audit all 62 P2S jobs and generate the authoritative ready projects.
make bambu_slice_release

# Serial is the conservative default. Increase only when host memory permits.
make bambu_slice_release BAMBU_JOBS=2

# Redirect all local audit/project products when desired.
make bambu_slice_release BAMBU_AUDIT_OUTPUT=/absolute/output/directory
```

`BAMBU_JOBS` defaults to `1`, and `BAMBU_AUDIT_OUTPUT` defaults to
`review/captive_magnet_slice_audit`. Every target first runs the static slicing
metadata tests and requires the existing
`review/captive_magnet_release_catalog.json` at runtime. The catalog is
deliberately not a Make prerequisite: these targets never regenerate CAD.

The Makefile detects these goals before its remote dispatcher, forces them to
the local execution path, and requires a Darwin/macOS host. It rejects a mixed
invocation such as `make bambu_slice_release check`; run remote CAD and local
slicing as separate commands. The targets invoke only the local Bambu Studio
CLI and write files under `BAMBU_AUDIT_OUTPUT`. They contain no MQTT, FTPS,
upload, start-print, or other printer contact.

`bambu_slice_obiwan` is intentionally diagnostic. Its `--only '*Obi-Wan*'`
run writes subset results and ready projects for matching artifacts, but it
is not a complete Obi-Wan print set and cannot replace the release-wide
canonical JSON/CSV/Markdown pause manifests. Only the unfiltered
`bambu_slice_release` can publish that authority after all 64 catalog entries,
62 real P2S slices, proxy coverage, and toolpath gates pass.

### Why the output is a sliced 3MF, not an STL sidecar

An STL stores only triangle geometry. Bambu Studio does not automatically
discover a neighboring project-specific JSON file when an STL is opened, so a
raw STL cannot reliably auto-load the required printer, process, filament,
orientation, infill, or pause settings. That is the brittle GUI path this
pipeline avoids.

With `--emit-ready-projects`, the pipeline instead exports one sliced
`.gcode.3mf` project for each actual P2S job. It embeds the resolved settings,
the exact transformed mesh, generated G-code, and custom per-layer pause
events. Open that file in Bambu Studio to obtain the already-sliced Preview and
layer slider with the magnet pauses present; no manual profile editing or pause
insertion is part of the release chain.

Canonical publication is fail-closed and release-wide. The slicer first
copies the exact catalog, schema, STL, adjacent print sidecar, source files,
Obi-Wan stage manifest, and Ac/Ae facts/transaction manifest into a read-only
staging tree. It slices those frozen STL bytes, then rechecks the live and
staged hashes, resolved profiles, and Bambu Studio binary. Only exact coverage
of all 64 artifacts / 114 stations with zero failures may transactionally
replace the three canonical manifests. A subset, dry run, failed slice,
missing evidence render, provenance drift, or interrupted publication leaves
the previous authority untouched.

No OpenCascade code is imported by this command. No local CAD execution is
performed.

### Floor/no-floor 01+02+03+04 composite shelf plates

The canonical 64-artifact audit remains unchanged. The printer shelf adds two
local packaging alternatives,
`obiwan_01a_02_03_04_LM_UM_1_of_1_no_floor_stand` and
`obiwan_01b_02_03_04_LM_UM_1_of_1_floor_stand`, by translation-only
concatenation of the four already released, same-state 01/02/03/04 STLs. Each
Bambu project is one printable object with four normal volumes and the three
state-specific canonical duct blockers. Each plate repeats the six
authoritative LM/UM sites in one Z=5.96 mm pause; neither creates a new CAD
artifact or magnet station. Both plates use the hash-pinned saved TINMORRY
PETG-GF preset and eight walls. The no-floor plate uses 40% gyroid globally
with one audited 100% zig-zag parameter modifier through the `01a`
bridge/root; the integral-floor plate uses global 100% zig-zag.

`scripts/build_obiwan_combo_plate.py` independently requires exact
four-volume/project/STL equivalence, all four support fields globally and per
object, actual support under each carrier footprint, no support under the
tweeter footprint, and zero support-bead collisions against each of the three
functional duct contracts. `make obiwan_no_floor_combo_plate` and `make
obiwan_floor_combo_plate` each perform a dry run before the local macOS Bambu
slice and refuse remote-worker/osado execution.
The source STL/manifest, ready project/audit, and promoted shelf pair are
ordinary Make artifacts with recovery stamps; the corresponding
`_to_print` targets refresh only their delivery pair after a slice-disabled
complete 51/51 shelf validation. `make to_print` consumes
existing authoritative captive-magnet ready projects and never implicitly
launches `bambu_slice_release`.

### Ac/Ae B four-piece wing shelf plates

The shelf packages the released Ac B-split 05b/06b/08b/09b meshes as
`obiwan_05b_06b_08b_09b_Ac_wings_1_of_1` and the Ae B-split
11b/12b/14b/15b meshes as `obiwan_11b_12b_14b_15b_Ae_wings_1_of_1`.
The shared builder applies only
deterministic Rz and XY rigid transforms to the exact front-face-down STL
triangle records. Its analytic footprint and mesh-witness gates require at
least 3.5 mm between parts and at every bed edge; the locked result achieves
3.587 mm and 3.592 mm respectively.

`scripts/build_obiwan_wing_plate.py` slices only on local macOS with Bambu
arranging/orienting/rotations disabled. It pins all four support settings off
globally and per object, requires zero support feature blocks, preserves one
six-site Z=5.96 mm magnet pause, and checks every cavity against the
authoritative release audit. Project promotion additionally requires identity
placement and exact equivalence to all 15,692 Ac or 958,546 Ae source
triangles.
`make obiwan_ac_wing_plate` and
`make obiwan_ae_wing_plate` build and audit the local ready projects;
their corresponding `_to_print` targets refresh the friendly shelf pairs
after the complete 51/51 slice-disabled shelf gate.
All Ac/Ae wing and shoulder projects remain on Bambu PLA Tough+ with six
walls. They are explicitly outside the PETG-GF profile's artifact scope.

## Layer and toolpath evidence

For every cavity the audit selects and records five actual G-code layers:

1. the lowest sliced layer at which the cavity begins;
2. a representative fully open layer;
3. the last completely open layer at or below the CAD bury plane;
4. the following first-closing layer, which is the Bambu pause-marker Z;
5. the first inspection layer above the fully closed roof apex.

At the first emerging cradle layer, the audit requires continuous retaining
material but does not misclassify its sub-line-width circular chord as a
mature wall. At the representative and last-open layers, both transverse
skins must each form exactly one connected medial traversal within the nominal
0.42--0.67 mm width bound, with only the documented 0.005 mm lower-side
Arachne tolerance. Candidates are classified by the physical bead edge
that forms the cavity boundary, within 0.06 mm of nominal, before the exact-one
test. This excludes a surrounding-body path that merely passes through the
broad skin-centre band, without weakening rejection of a second cavity-wall
traversal: every overlapping path in the broader skin band is checked as a
secondary guard. The only permitted return is the measured continuous
same-path V1 hairpin, limited to three contiguous edge scan bins; independent,
central, or longer duplicate paths fail. An axial skin must form one geometric annular
traversal within the nominal 0.42--0.65 mm bound, with only the 0.000005 mm
lower serialization tolerance; Bambu may split that ring into no more than two
complementary G-code components at its seam. Each split component must be one
cyclic interval with at least 18 exclusive rays, the components may not overlap,
and their union must occupy at least 70 of 72 rays. At most two missing/double
rays are allowed and each must be endpoint-local; the largest uncovered arc
also remains bounded. The two component junctions are measured from annular
near-endpoint geometry and must have bead-footprint contact while remaining
within the 0.52 mm connectivity cap. This rejects disconnected complementary
arcs, a full ring plus a stray second path, and the former two-concentric-path
slice. The audit also
checks that the roof boundary moves inward on the first-closing layer and that
sealed-layer toolpaths occupy the former opening.
At the last-open layer it additionally requires a path-width-aware free
loading diameter of at least D5.0, at least 2.0 mm of axial slot length for
transverse stations, and no material path crossing the loading aperture.
G2/G3 arcs are expanded with bounded chord spacing before these connectivity,
multiplicity, width, and aperture tests; unsupported arc modes fail closed
instead of being silently treated as straight travel.

The audit also transforms the exact seated-magnet centre and axis into print
space, proves the D5 x 2 disc fits the Ø5.20 x 2.10 cradle, and fails if any
point of the seated cylinder is not below the completed last-open layer and
clear of the resumed first-closing nozzle height. It renders all five local
toolpath views to SVG and PNG in each actually sliced artifact's `slices/`
directory.

The tested coupon-equivalent regression is fail-closed when a catalog site
declares it:

- Every common-plane Obi-Wan LM/UM transverse first-closing/pause marker:
  **Z = 5.96 mm**

These values are not global constants. Every unrelated site is bracketed
against the actual G-code schedule generated for that STL.

## Outputs

`review/captive_magnet_slice_audit/` contains:

- `captive_magnet_pause_manifest.json` — machine-readable authority;
- `captive_magnet_pause_manifest.csv` — one row per part and pause group;
- `CAPTIVE_MAGNET_PAUSE_MANIFEST.md` — insertion instructions and human table;
- `profiles/` — flattened presets and their provenance/hashes;
- `slices/<artifact>/` — plain G-code, Bambu `result.json`, validation JSON,
  five-layer toolpath evidence, and the ready `.gcode.3mf` project for each
  actually sliced artifact.

Sites at the same actual first-closing Z are grouped into one pause. The JSON,
CSV, and Markdown record the exact print insertion vector `[0, 0, -1]` and
human instruction: lower every magnet vertically from above the paused part
(the +Z side) along print -Z. They also preserve per-site
installed/print-coordinate polarity vectors, the complete human polarity
instruction (including the provisional unpaired V0 and unpaired coupon
warnings), and the minimum measured seated-disc margin below both the
last-open and first-closing layers. Their summaries separately count actual
P2S slices/pause magnets and oversized exact-split coverage. An oversized
monolith has proxy hashes but no monolith G-code and no fake pause row. Always
mark one pole before insertion, fully seat each magnet below the completed
layer, and verify polarity before resuming: it cannot be corrected after
burial.
