# Captive-magnet slicing and pause audit

`slice_captive_magnets.py` is the offline release pipeline for the buried
D5 x 2 magnet system. It resolves the installed Bambu presets, slices every
catalogued STL that fits the P2S bed in its mandatory front-face-down
orientation, reads the actual G-code layer schedule, checks the open cradle
and closing roof toolpaths, and writes both the pause audit and ready-to-open
`.gcode.3mf` projects. The catalog contains exactly 56 magnet-bearing release
STLs, of which 54 are actual P2S slice jobs. The two canonical Obi-Wan LM
monoliths are catalogued release artifacts but are explicitly classified as
not P2S-printable; their cavity audits are supplied by exact same-state keyed
halves as described below. The number 102 is the count of individual magnet
stations, not a claim that this pipeline slices 102 STLs. Separately, the full
repository release happens to contain 102 STL/`.print.json` pairs. This magnet
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
- Filament used for slicing: Bambu PLA Tough+
- Support: disabled by default; enabled only for both
  `lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` jobs with **On build
  plate only** and **Support critical regions only**
- Orientation: **front face down for every piece**

The production overrides retain the project's structural profile: six wall
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

The integral floor LM requires a 100 percent local-solid stem/root. Bambu's
CLI cannot yet generate and bind the required modifier volume through this
pipeline without a brittle GUI edit, so only the floor-state
`lx521_top_obiwan_optional_lm_keyed_1of2_bottom.stl` uses global 100 percent
infill as the safe automated fallback. Its artifact profile uses Bambu's
`zig-zag` solid pattern because Bambu Studio rejects gyroid at 100 percent;
both discovery and ready G-code are checked for that exact combination. This
costs material and increases warp risk, but it preserves the structural
requirement until modifier-volume `.3mf` automation is implemented. Do not
reduce that piece to sparse infill after loading the generated project.

Both floor states of that keyed LM bottom also carry the only support override
in the release catalog: `enable_support=1`,
`support_on_build_plate_only=1`, and
`support_critical_regions_only=1`. This supports the floating cantilever from
the plate without turning ordinary buried-magnet jobs into supported prints.
The cavity audit still rejects support or any other extrusion that blocks the
last-open D5 loading aperture, and both the final G-code CONFIG_BLOCK and the
ready `.gcode.3mf` project settings must contain the exact three flags.

The process selector lives in `captive_magnet_slicing_profile.json`. The
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

The current fail-closed inventory is 56 magnet-bearing released STLs and 102
individual magnet stations: 22 STLs in each floor state plus 12 shared Ac/Ae
segments. A count change is a source-level release change, not something the
generator silently accepts. Non-print compound STEP review packages are not
separate prints; their constituent STLs are present in this inventory.

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

The current 56-STL/102-station catalog consequently produces 54 actual P2S
slices with 94 actual pause-station rows, plus eight canonical monolith site
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
    "source_files": ["../top_baffle_nd25fw4_obiwan.py"],
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

# Slice/audit all 54 P2S jobs and generate the authoritative ready projects.
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
`bambu_slice_release` can publish that authority after all 56 catalog entries,
54 real P2S slices, proxy coverage, and toolpath gates pass.

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
of all 56 artifacts / 102 stations with zero failures may transactionally
replace the three canonical manifests. A subset, dry run, failed slice,
missing evidence render, provenance drift, or interrupted publication leaves
the previous authority untouched.

No OpenCascade code is imported by this command. No local CAD execution is
performed.

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
