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
├── catalog_06hf.json            # 0.6-mm-lane hash manifest
├── stock/
│   ├── stl/                     # 11 printable Stock pieces (shared by both lanes)
│   ├── 3mf_04/                  # ready P2S projects, 0.4 mm nozzle lane
│   └── 3mf_06hf/                # ready projects, 0.6 mm high-flow lane (magnet parts)
├── slim/
│   ├── stl/                     # 11 printable Slim pieces
│   ├── 3mf_04/                  # 0.4 mm lane
│   └── 3mf_06hf/                # 0.6 mm high-flow lane
└── obiwan/
    ├── stl/                     # 31: pieces + 4 plate alternatives + 2 candidates
    ├── 3mf_04/                  # 0.4 mm lane
    └── 3mf_06hf/                # 0.6 mm high-flow lane
```

Open the `.gcode.3mf` from the lane folder matching the installed nozzle
(`3mf_04/` for the 0.4 mm nozzle, `3mf_06hf/` for the 0.6 mm high-flow
nozzle) directly in Bambu Studio. The 0.6-mm lane appends `_06hf` to
every filename, so a Studio tab is never ambiguous about its lane; the
same base name in both folders is the same part with the same pause; combo plates, candidates, and
non-magnet pieces currently ship in `3mf_04/` only.  Do not reorient
or re-slice it.  The project already has the front face on the build plate,
the P2S 0.4 mm / 0.16 mm Arachne profile and the verified magnet events.
Ordinary parts use Bambu PLA Basic (PLA Tough+ is deprecated) with six
general walls on the 0.4 lane and four on the 0.6 lane and 30% gyroid
infill (40% for both UM carriers). Both standalone keyed LM `01` bottoms,
plus both combined `01+02+03+04` core plates, are the structural
exceptions: they use the hash-pinned saved
**TINMORRY PETG-GF Profile @BBL P2S** preset with six 0.62 mm walls on the
**0.6 mm high-flow nozzle only** — PETG-GF projects ship exclusively in
`3mf_06hf/`. The no-floor-stand `01`
uses 40% gyroid globally plus its 100%-solid bridge/root modifier; the
floor-stand `01` uses global 100% zig-zag. Flat/graded wings and shoulders remain PLA Basic;
do not print them in PETG-GF.
Each magnetic event makes no XY move: it raises the nozzle to Z=250 mm at
20 mm/s (lowering the P2S bed to within 6 mm of its bottom), executes the
Bambu pause `M400 U1`, then restores the exact closing-layer Z when you press
**Continue**.  Every support-enabled Obi-Wan carrier project carries all four
required support fields globally and on every object, plus hash-bound duct
blockers and a zero-collision support-toolpath gate, so an ordinary Bambu
Studio re-slice cannot silently obstruct a functional cable lumen.

## Combined plates

A **combo** is one pre-arranged Bambu plate that holds several parts at locked
positions and prints them in a single job.  It is not a different part: a
combo carries exactly the same meshes as the individual files it replaces, in
the same released front-face-down orientation, and promotion requires exact
project/STL equivalence against them.  There are four:

| Combo | Contains |
|---|---|
| `obiwan_01_02_03_04_LM_UM_combo_no_floor_stand` | `01` keyed LM bottom (no-floor) + `02` keyed LM top + `03` UM carrier + `04` tweeter crescent |
| `obiwan_01_02_03_04_LM_UM_combo_floor_stand` | `01` keyed LM bottom (floor stand) + `02` keyed LM top + `03` UM carrier + `04` tweeter crescent |
| `obiwan_flat_wings_split2_combo` | all four flat split2 wing pieces: `05`/`08` LM-lower left and right, `06`/`09` fused LM/UM-upper left and right |
| `obiwan_graded_wings_split2_combo` | all four graded split2 wing pieces: `11`/`14` LM-lower left and right, `12`/`15` fused LM/UM-upper left and right |

**The rule is exclusive-or.**  Print a combo *or* its individual pieces, never
both — running a combo plate and the same parts as separate jobs prints every
piece twice.  And never mix stand states: `01` is the only state-specific
piece, so a floor-stand plate and any no-floor part do not belong in the same
speaker.

`obiwan_01_02_03_04_LM_UM_combo_no_floor_stand` and
`obiwan_01_02_03_04_LM_UM_combo_floor_stand` are the no-floor and
floor-stand combo plates for 01+02+03+04. Each is one Bambu
printable object containing four normal volumes and three aligned
state-specific duct-blocker volumes at locked, translation-only positions.
Both use TINMORRY PETG-GF with six 0.62 mm walls on the 0.6 mm high-flow
nozzle only. The no-floor plate uses 40% gyroid
globally plus a 100% zig-zag parameter modifier through the complete no-floor-stand `01`
bridge/root; the floor plate preserves the integral-floor bottom's global
100% zig-zag profile. Both pin all four support fields globally and on the
object, pause once at Z=5.96 mm for the six LM/UM magnets, and emit no support
under the tweeter footprint. Promotion requires exact 3MF/STL equivalence,
the required modifier inventory, and zero support-bead collisions in all
three duct-bearing parts.

`obiwan_flat_wings_split2_combo` and
`obiwan_graded_wings_split2_combo` are the combo plates
for the four flat and graded split2 wing pieces respectively. Each preserves the released
front-face-down orientation, applies only locked Z rotations and XY
translations, and uses 30% gyroid with all four support fields pinned off
globally and on the object. These non-load-bearing pieces retain the standard
Bambu PLA Tough+ profile; they are not PETG-GF jobs. The audited packing has
3.587 mm minimum
part-to-part clearance and 3.592 mm minimum bed-edge clearance. Each pauses
once at Z=5.96 mm for all six wing magnets, emits zero support feature blocks,
and passes exact project/STL equivalence: 15,692 triangles for flat and 958,546
for graded.

The six-wall setting is deliberately general-body strength, not six walls in
the captive-magnet skin. Arachne is allowed to emit the one bounded
0.42–0.45 mm retaining path required by each 0.45 mm magnet wall.

## Choosing what to print

The two-digit slot numbers are catalogue positions, not a literal one-build
bill of materials.  State and attachment choices are alternatives:

- **Stock:** choose either the `01` `no_floor_stand` or `floor_stand` piece. Core
  positions 02–04 are canonical shared parts. Print either the four
  `A_shoulder` pieces or the two `B1` wings, never both.
- **Slim:** the same choice model as Stock.
- **Obi-Wan:** choose the individual core (the `01` `no_floor_stand` or
  `floor_stand` piece, plus 02–04), or use the matching
  `obiwan_01_02_03_04_LM_UM_combo_no_floor_stand` or
  `obiwan_01_02_03_04_LM_UM_combo_floor_stand` plate instead of those four
  individual files. Never mix stand states or print a combined core plate
  together with its individual 01/02/03/04 pieces. Choose flat (05–10) or graded
  (11–16), never both wing families. Within the selected family, choose either
  every `split3` file for the original three-piece-per-side split or every
  `split2` file for the fused two-piece-per-side split; never mix them. For
  a split2 wing set, use either the four individual projects or the matching
  combo plate; never print both forms. Finally, the crescent slot takes
  exactly one choice: the released `04` tweeter crescent (which either core
  combo plate already carries), the **candidate** `17` coaxial BMR crescent,
  the **candidate** `18` opposed BMR crescent, or nothing at all. `17` and
  `18` exclude each other and `04`; if you take one of them together with a
  core combo plate, the `04` that plate carries is the piece you set aside.

The tracked [catalog.json](catalog.json) is the authoritative friendly-name
map.  `release_manifest.json` is generated locally by the build and records
the exact source STL, source and output SHA-256 hashes, P2S project source,
settings, and magnet-pause count for every delivered file.

## The opposed-BMR vase, delivered separately

`vase_TEBM35C10-4` is the alternative to the Dayton ND25FW-4 crescent: one
vase piece carrying two opposed Tectonic TEBM35C10-4 BMRs, the lower facing
front and the upper facing rear. It is released in Stock and Slim envelope
profiles only — never for Obi-Wan — and it replaces the `04` vase in the
matching product's set. It is **not** part of this shelf, so nothing above
changes if you use it.

Its ready projects are delivered on their own parallel path, alongside the CAD
rather than under `to_print/`:

```text
build/vase_TEBM35C10-4/stock/vase_TEBM35C10-4.gcode.3mf
build/vase_TEBM35C10-4/slim/vase_TEBM35C10-4.gcode.3mf
```

Build them on this Mac with `make vase_tebm35c10_4_stock_3mf` and
`make vase_tebm35c10_4_slim_3mf`, or both with `make vase_tebm35c10_4_3mf`.
Each requires the promoted CAD from `make vase_tebm35c10_4_cad` and fails
closed if it is missing. Like everything here, these targets never dispatch to
osado, and each project is opened directly in Bambu Studio without
reorienting or re-slicing.

## The candidate BMR crescents, on the shelf as candidates

The two Obi-Wan BMR pods — `obiwan_17_BMR_crescent_coaxial_1_of_1` and
`obiwan_18_BMR_crescent_opposed_1_of_1` — are on this shelf so they can
actually be printed and physically qualified. They are still **candidates**:
`release_authorized` is false on both, neither is in the release inventory or
the released captive-magnet catalog, and printing one is a qualification
exercise rather than building a finished speaker. See
[`docs/obiwan.md`](../docs/obiwan.md) for what each still owes.

Being on the shelf changes only where the files appear. Both are sliced out of
their own isolated one-artifact catalog and their own pinned profile beside
their CAD, and the shelf hard-links that finished delivery rather than
re-slicing it:

```text
build/bmr_crescent_TEBM35C10-4/obiwan_bmr_crescent_TEBM35C10-4.gcode.3mf
build/bmr_crescent_TEBM35C10-4/obiwan_bmr_crescent_opposed_TEBM35C10-4.gcode.3mf
```

Build that delivery with `make obiwan_bmr_crescent_coaxial_3mf` and
`make obiwan_bmr_crescent_opposed_3mf`, or both with
`make obiwan_bmr_crescent_3mf`; `make obiwan_bmr_crescent_3mf_validate`
rechecks existing projects without slicing. Each requires the promoted CAD
from `make obiwan_bmr_crescent_cad` and fails closed if it is missing, and so
does `make to_print`: the shelf reports which candidate input is missing and
tells you to run `make obiwan_bmr_crescent_3mf` first rather than slicing a
candidate itself.

Each project carries one real magnet pause at **Z = 5.96 mm**, burying two
captive D5 × 2 magnets in the coaxial pod and all four in the opposed one —
every cavity in both parts closes on the same layer, so one pause covers them
all. The event parks at Z=250 mm, pauses with `M400 U1` and restores the exact
layer Z, exactly like the released projects above. Insert the magnets
vertically downward with the marked pole as the delivery record states, and do
not add or move pauses by hand.

## Refreshing the shelf

Run this on the Mac that has the pinned Bambu Studio version installed:

```sh
make to_print
```

It consumes the existing authoritative captive-magnet ready projects, builds
the four combined plates incrementally, then creates or refreshes all 53
STL/project pairs. It does **not** implicitly launch the heavyweight
58-artifact release slicer; a missing canonical ready project fails closed and
must be refreshed explicitly with `make bambu_slice_release`. Of the 44
pause-bearing projects, 38 are hard-linked from that audited release, the
four combined plates are built and sliced locally from those same released
inputs, and the two candidate BMR crescents are hard-linked from their own
parallel delivery, which fails closed with a `make obiwan_bmr_crescent_3mf`
message rather than being sliced here.
The 9 magnet-free pieces are sliced locally under the same pinned profile.
Hard links make the visible files ordinary, directly-openable files without
duplicating the large project payloads on disk.

All four combined plates are first-class Make artifacts. To build their
deterministic source STL/manifests, validated ready projects, or promoted
shelf pairs, run:

```sh
make obiwan_no_floor_combo_plate_source
make obiwan_no_floor_petg_gf_01a
make obiwan_no_floor_petg_gf_01a_to_print
make obiwan_floor_petg_gf_01b
make obiwan_floor_petg_gf_01b_to_print
make obiwan_no_floor_combo_plate
make obiwan_no_floor_combo_plate_to_print
make obiwan_floor_combo_plate_source
make obiwan_floor_combo_plate
make obiwan_floor_combo_plate_to_print
make obiwan_flat_wing_plate_source
make obiwan_flat_wing_plate
make obiwan_flat_wing_plate_to_print
make obiwan_graded_wing_plate_source
make obiwan_graded_wing_plate
make obiwan_graded_wing_plate_to_print
```

The corresponding concrete STL, ready `.gcode.3mf`, audit, and promoted files
are backed by ordinary dependency stamps with missing-member recovery. The
ready targets dry-run before any required local slice; the promotion targets
disable slicing and cross the complete 53/53 shelf-equivalence gate.
Every alias and concrete artifact path rejects remote-worker or osado
execution.

To validate an existing shelf without re-slicing, run:

```sh
make to_print_validate
```

The materialized STLs and `.gcode.3mf` projects are ignored by Git on purpose.
The private slicer cache is kept outside this delivery tree at
`review/to_print_slice_workspace/`, so `to_print/` remains an exact 53-STL /
53-project printer shelf. Re-run `make to_print` after any intentional
canonical release-slice change.
