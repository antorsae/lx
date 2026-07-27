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
    ├── stl/                     # 29 entries, including four plate alternatives
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

`obiwan_01a_02_03_04_LM_UM_1_of_1_no_floor_stand` and
`obiwan_01b_02_03_04_LM_UM_1_of_1_floor_stand` are the no-floor and
floor-stand single-plate alternatives for 01+02+03+04. Each is one Bambu
printable object containing four normal volumes and three aligned
state-specific duct-blocker volumes at locked, translation-only positions.
The no-floor plate uses 40% gyroid; the floor plate preserves the integral
floor bottom's authoritative 100% zig-zag profile. Both pin all four support
fields globally and on the object, pause once at Z=5.96 mm for the six LM/UM
magnets, and emit no support under the tweeter footprint. Promotion requires
exact 3MF/STL equivalence and zero support-bead collisions in all three
duct-bearing parts.

`obiwan_05b_06b_08b_09b_Ac_wings_1_of_1` and
`obiwan_11b_12b_14b_15b_Ae_wings_1_of_1` are the single-plate alternatives
for the four Ac and Ae B-split wing pieces respectively. Each preserves the released
front-face-down orientation, applies only locked Z rotations and XY
translations, and uses 30% gyroid with all four support fields pinned off
globally and on the object. The audited packing has 3.587 mm minimum
part-to-part clearance and 3.592 mm minimum bed-edge clearance. Each pauses
once at Z=5.96 mm for all six wing magnets, emits zero support feature blocks,
and passes exact project/STL equivalence: 15,692 triangles for Ac and 958,546
for Ae.

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
  `01b_floor_stand`, plus 02–04), or use the matching
  `obiwan_01a_02_03_04_LM_UM_1_of_1_no_floor_stand` or
  `obiwan_01b_02_03_04_LM_UM_1_of_1_floor_stand` plate instead of those four
  individual files. Never mix stand states or print a combined core plate
  together with its individual 01/02/03/04 pieces. Choose Ac (05–10) or Ae
  (11–16), never both wing families. Within the selected family, choose either
  every `a` file for the original three-piece-per-side split or every `b` file
  for the fused two-piece-per-side split; never mix the A and B wing sets. For
  Ac B or Ae B, use either the four individual projects or the matching
  single-plate alternative; never print both forms.

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
the four combined plates incrementally, then creates or refreshes all 51
STL/project pairs. It does **not** implicitly launch the heavyweight
64-artifact release slicer; a missing canonical ready project fails closed and
must be refreshed explicitly with `make bambu_slice_release`. Of the 42
pause-bearing projects, 38 are hard-linked from that audited release and the
four combined plates are built and sliced locally from those same released inputs.
The 9 magnet-free pieces are sliced locally under the same pinned profile.
Hard links make the visible files ordinary, directly-openable files without
duplicating the large project payloads on disk.

All four combined plates are first-class Make artifacts. To build their
deterministic source STL/manifests, validated ready projects, or promoted
shelf pairs, run:

```sh
make obiwan_no_floor_combo_plate_source
make obiwan_no_floor_combo_plate
make obiwan_no_floor_combo_plate_to_print
make obiwan_floor_combo_plate_source
make obiwan_floor_combo_plate
make obiwan_floor_combo_plate_to_print
make obiwan_ac_wing_plate_source
make obiwan_ac_wing_plate
make obiwan_ac_wing_plate_to_print
make obiwan_ae_wing_plate_source
make obiwan_ae_wing_plate
make obiwan_ae_wing_plate_to_print
```

The corresponding concrete STL, ready `.gcode.3mf`, audit, and promoted files
are backed by ordinary dependency stamps with missing-member recovery. The
ready targets dry-run before any required local slice; the promotion targets
disable slicing and cross the complete 51/51 shelf-equivalence gate.
Every alias and concrete artifact path rejects remote-worker or osado
execution.

To validate an existing shelf without re-slicing, run:

```sh
make to_print_validate
```

The materialized STLs and `.gcode.3mf` projects are ignored by Git on purpose.
The private slicer cache is kept outside this delivery tree at
`review/to_print_slice_workspace/`, so `to_print/` remains an exact 51-STL /
51-project printer shelf. Re-run `make to_print` after any intentional
canonical release-slice change.
