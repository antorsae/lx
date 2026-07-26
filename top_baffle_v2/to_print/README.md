# P2S print shelf

This is the short, printer-facing release shelf for an LX521.4 top baffle.
It contains only printable parts for the Bambu Lab P2S: friendly-named STL
sources and matching, already-sliced `.gcode.3mf` projects.  The oversized
Obi-Wan LM monolith, C7/V0/V1 legacy variants, coupons, and diagnostics are
intentionally not here.

```text
to_print/
├── catalog.json                 # tracked friendly-name/source map
├── README.md
├── stock/
│   ├── stl/                     # 11 printable Stock pieces
│   └── 3mf/                     # matching ready P2S projects
├── slim/
│   ├── stl/                     # 11 printable Slim pieces
│   └── 3mf/                     # matching ready P2S projects
└── obiwan/
    ├── stl/                     # 26 entries, including one core-plate alternative
    └── 3mf/                     # matching ready P2S projects
```

Open the `.gcode.3mf` from `3mf/` directly in Bambu Studio.  Do not reorient
or re-slice it.  The project already has the front face on the build plate,
the P2S 0.4 mm / 0.16 mm Arachne profile, six general walls, 30% gyroid
infill (40% for the no-floor keyed LM bottom and both UM carriers; 100%
zig-zag for the floor-stand keyed LM bottom), and the verified magnet events.
Each magnetic event makes no XY move: it raises the nozzle to Z=250 mm at
20 mm/s (lowering the P2S bed to within 6 mm of its bottom), executes the
Bambu pause `M400 U1`, then restores the exact closing-layer Z when you press
**Continue**.  Every support-enabled Obi-Wan carrier project carries all four
required support fields globally and on every object, plus hash-bound duct
blockers and a zero-collision support-toolpath gate, so an ordinary Bambu
Studio re-slice cannot silently obstruct a functional cable lumen.

`obiwan_01a_02_03_04_LM_UM_1_of_1` is the no-floor single-plate
alternative for 01a+02+03+04. It is one Bambu printable object containing
four normal volumes and three aligned duct-blocker volumes at locked,
translation-only positions. It uses 40% gyroid, pins all four support fields
globally and on the object, pauses once at Z=5.96 mm for the six LM/UM
magnets, and has no support toolpath under the tweeter footprint. Its
promotion audit requires exact 3MF/STL equivalence and zero support-bead
collisions in all three duct-bearing parts.

The six-wall setting is deliberately general-body strength, not six walls in
the captive-magnet skin. Arachne is allowed to emit the one bounded
0.42–0.45 mm retaining path required by each 0.45 mm magnet wall.

`of_10` and `of_16` are catalogue positions, not a literal one-build bill of
materials.  State and attachment choices are alternatives:

- **Stock:** choose either `01a_no_floor_stand` or `01b_floor_stand`. Core
  positions 02–04 are canonical shared parts. Print either the four
  `A_shoulder` pieces or the two `B1` wings, never both.
- **Slim:** the same choice model as Stock.
- **Obi-Wan:** choose the individual core (`01a_no_floor_stand` or
  `01b_floor_stand`, plus 02–04), or use
  `obiwan_01a_02_03_04_LM_UM_1_of_1` instead of the four individual files.
  The combined plate is only the no-floor 01a form; never print it together
  with 01a/02/03/04. Choose Ac (05–10) or Ae (11–16), never both wing
  families. Within the selected family, choose either every `a` file for the
  original three-piece-per-side split or every `b` file for the fused
  two-piece-per-side split; never mix the A and B wing sets.

The tracked [catalog.json](catalog.json) is the authoritative friendly-name
map.  `release_manifest.json` is generated locally by the build and records
the exact source STL, source and output SHA-256 hashes, P2S project source,
settings, and magnet-pause count for every delivered file.

## Refreshing the shelf

Run this on the Mac that has the pinned Bambu Studio version installed:

```sh
make to_print
```

It consumes the existing authoritative captive-magnet ready projects, builds
the combined plate incrementally, then creates or refreshes all 48
STL/project pairs. It does **not** implicitly launch the heavyweight
64-artifact release slicer; a missing canonical ready project fails closed and
must be refreshed explicitly with `make bambu_slice_release`. Of the 39
pause-bearing projects, 38 are hard-linked from that audited release and the
combined plate is built and sliced locally from those same released inputs.
The 9 magnet-free pieces are sliced locally under the same pinned profile.
Hard links make the visible files ordinary, directly-openable files without
duplicating the large project payloads on disk.

The combined plate is a first-class Make artifact. To build its deterministic
source STL/manifest, validated ready project, or promoted shelf pair, run:

```sh
make obiwan_combo_plate_source
make obiwan_combo_plate
make obiwan_combo_plate_to_print
```

The corresponding concrete STL, ready `.gcode.3mf`, audit, and promoted files
are backed by ordinary dependency stamps with missing-member recovery. The
ready target dry-runs before any required local slice; the promotion target
disables slicing and crosses the complete 48/48 shelf-equivalence gate.
Every alias and concrete artifact path rejects remote-worker or osado
execution.

To validate an existing shelf without re-slicing, run:

```sh
make to_print_validate
```

The materialized STLs and `.gcode.3mf` projects are ignored by Git on purpose.
The private slicer cache is kept outside this delivery tree at
`review/to_print_slice_workspace/`, so `to_print/` remains an exact 48-STL /
48-project printer shelf. Re-run `make to_print` after any intentional
canonical release-slice change.
