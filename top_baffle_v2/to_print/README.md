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
    ├── stl/                     # 17 printable Obi-Wan pieces
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
**Continue**.  The Obi-Wan keyed LM-bottom projects additionally carry the
required build-plate-only critical support configuration both globally and on
the object, so that support survives an ordinary Bambu Studio re-slice.

The six-wall setting is deliberately general-body strength, not six walls in
the captive-magnet skin. Arachne is allowed to emit the one bounded
0.42–0.45 mm retaining path required by each 0.45 mm magnet wall.

`of_10` and `of_16` are catalogue positions, not a literal one-build bill of
materials.  State and attachment choices are alternatives:

- **Stock:** choose either `01a_no_floor_stand` or `01b_floor_stand`. Core
  positions 02–04 are canonical shared parts. Print either the four
  `A_shoulder` pieces or the two `B1` wings, never both.
- **Slim:** the same choice model as Stock.
- **Obi-Wan:** choose `01a_no_floor_stand` or `01b_floor_stand`; 02 (keyed LM
  top), 03 (UM carrier), and 04 (tweeter crescent) are canonical shared
  parts. Choose either all Ac wings (05–10) or all Ae wings (11–16), never
  both wing families.

The tracked [catalog.json](catalog.json) is the authoritative friendly-name
map.  `release_manifest.json` is generated locally by the build and records
the exact source STL, source and output SHA-256 hashes, P2S project source,
settings, and magnet-pause count for every delivered file.

## Refreshing the shelf

Run this on the Mac that has the pinned Bambu Studio version installed:

```sh
make to_print
```

It first runs the local authoritative captive-magnet release audit, then
creates or refreshes all 39 delivery files.  The 30 magnet-bearing projects
are hard-linked from that audited release; the 9 magnet-free pieces are
sliced locally under the same pinned profile.  Hard links make the visible
files ordinary, directly-openable files without duplicating the large project
payloads on disk.

To validate an existing shelf without re-slicing, run:

```sh
make to_print_validate
```

The materialized STLs and `.gcode.3mf` projects are ignored by Git on purpose.
The private slicer cache is kept outside this delivery tree at
`review/to_print_slice_workspace/`, so `to_print/` remains an exact 39-STL /
39-project printer shelf.  Re-run `make to_print` after any canonical CAD or
release-slice change.
