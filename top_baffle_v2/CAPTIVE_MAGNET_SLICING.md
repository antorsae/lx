# Captive-magnet slicing and pause audit

`slice_captive_magnets.py` is the offline release pipeline for the buried
D5 x 2 magnet system. It resolves the installed Bambu presets, slices every
catalogued STL that fits the P2S bed in its mandatory front-face-down
orientation, reads the actual G-code layer schedule, checks the open cradle
and closing roof toolpaths, and writes the authoritative pause manifest. The
two canonical Obi-Wan LM monoliths are catalogued release artifacts but are
explicitly classified as not P2S-printable; their cavity audits are supplied
by exact same-state keyed halves as described below.

It has no printer transport code. It does not use MQTT, FTPS, LAN discovery,
or a printer address, and cannot upload or start a print.

## Fixed process contract

- Printer: Bambu Lab P2S, 0.4 mm nozzle
- Process: 0.16 mm High Quality
- Wall generator: Classic
- Nominal wall paths: 0.42 mm outer / 0.45 mm inner
- Filament used for slicing: Bambu PLA Tough+
- Support inside the cavity: disabled
- Orientation: **front face down for every piece**

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
and polarity. The keyed proxy artifact must then pass the ordinary P2S slice
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
      "cavity_bury_roof_start_print_z_mm": 8.40,
      "roof_apex_print_z_mm": 11.00,
      "expected_pause_marker_z_mm": 8.52,
      "cavity_diameter_mm": 5.20,
      "cavity_depth_mm": 2.10,
      "face_skin_mm": 0.45,
      "inner_skin_mm": 0.45,
      "polarity_instruction": "marked/N pole points along the listed axis",
      "installed_marked_pole_axis_xyz": [0.438371, 0.898794, 0],
      "cavity_center_xyz_mm": [48.878, 301.196, 12.55],
      "seated_magnet_center_xyz_mm": [48.900, 301.242, 12.55],
      "actual_face_xyz_mm": [49.536, 302.545, 12.55],
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

## Running the audit

CAD/STEP/STL generation remains remote on `osado.lan`. After the promoted
artifacts and catalog are present locally, run only the slicer audit on macOS:

```bash
cd top_baffle_v2

# Resolve and verify the exact installed profiles without slicing.
python3 slice_captive_magnets.py --prepare-profiles-only

# Show the exact per-STL Bambu commands without executing them.
python3 slice_captive_magnets.py --dry-run

# Complete release audit. Serial is the conservative default; independent
# Bambu processes can be requested explicitly when host memory permits.
python3 slice_captive_magnets.py --jobs 2

# Resume a complete run with the content-addressed cache. A filtered run is
# diagnostic only: it writes subset_slice_results.json and can never replace
# the authoritative release-wide JSON/CSV/Markdown manifests.
python3 slice_captive_magnets.py --only '*Obi-Wan*' --jobs 2
```

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

The audit checks that both axial retaining skin paths form connected spans at
the lowest, representative, and last-open layers (or that an axial cavity has
one connected annular path with complete angular coverage), that the roof
boundary moves inward on the first-closing layer, and that sealed-layer
toolpaths occupy the former opening. At the last-open layer it additionally
requires a path-width-aware free loading diameter of at least D5.0, at least
2.0 mm of axial slot length for transverse stations, and no material path
crossing the loading aperture. G2/G3 arcs are expanded with bounded chord
spacing before these connectivity and aperture tests; unsupported arc modes
fail closed instead of being silently treated as straight travel.

The audit also transforms the exact seated-magnet centre and axis into print
space, proves the D5 x 2 disc fits the Ø5.20 x 2.10 cradle, and fails if any
point of the seated cylinder is not below the completed last-open layer and
clear of the resumed first-closing nozzle height. It renders all five local
toolpath views to SVG and PNG in each actually sliced artifact's `slices/`
directory.

The tested coupon-equivalent regressions are fail-closed when their catalog
sites declare them:

- UM first-closing/pause marker: **Z = 5.96 mm**
- LM first-closing/pause marker: **Z = 8.52 mm**

These values are not global constants. Every unrelated site is bracketed
against the actual G-code schedule generated for that STL.

## Outputs

`review/captive_magnet_slice_audit/` contains:

- `captive_magnet_pause_manifest.json` — machine-readable authority;
- `captive_magnet_pause_manifest.csv` — one row per part and pause group;
- `CAPTIVE_MAGNET_PAUSE_MANIFEST.md` — insertion instructions and human table;
- `profiles/` — flattened presets and their provenance/hashes;
- `slices/<artifact>/` — plain G-code, Bambu `result.json`, validation JSON,
  and five-layer toolpath evidence for actually sliced artifacts.

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
