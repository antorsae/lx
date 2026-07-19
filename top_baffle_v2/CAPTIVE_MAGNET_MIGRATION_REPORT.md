# Captive-magnet production migration report

## Release status and outcome caveat

The production source migration is implemented: every discovered released,
printable magnet station is represented by the shared pause-and-bury captive
system instead of an externally accessible glue pocket. The fail-closed
inventory contains **56 magnet-bearing STLs and 102 per-STL magnet stations**.
This count includes both stand-state copies and the mutually exclusive Obi-Wan LM
monolith/split alternatives.

This document is not the final release authority until the content-addressed
CAD candidate has been atomically promoted and the complete Bambu slice audit
has published its manifests. The geometry, inventory, exclusions, orientation
contract, and remaining physical risks below are final static facts. The only
intentionally unfinished fields are labeled `FINALIZE AFTER PROMOTION`.

- Promoted source snapshot SHA-256: **`[FINALIZE AFTER PROMOTION: 64-hex source hash]`**
- Actual complete-slice totals and Obi-Wan regression observations:
  **`[FINALIZE AFTER PROMOTION: passed/failed STLs, actual pause stations,
  exact-split proxy coverage, observed UM 5.96 / LM 8.52 results]`**
- Authoritative catalog/pause/polarity manifest paths:
  **`[FINALIZE AFTER PROMOTION: release catalog plus JSON, CSV, and Markdown
  pause-manifest paths]`**
- Representative CAD Viewer URLs:
  **`[FINALIZE AFTER PROMOTION: URLs listed in the review table below]`**

No magnet receives structural-load credit. The release contract remains
`structural_load_credit_n = 0.0` at every site.

## Before and after

| Property | Superseded glue-in interface | Captive production interface |
|---|---:|---:|
| Purchased magnet | D5.0 x 2.0 mm disc | D5.0 x 2.0 mm disc |
| Pocket/cavity | Externally open D5.20 x 2.20 mm pocket | Internal D5.20 x 2.10 mm cavity |
| Axial clearance | 0.20 mm, including adhesive allowance | 0.10 mm, no adhesive allowance |
| Interface-side skin | None; magnet remained externally accessible | 0.45 mm |
| Inner skin | Not captive | 0.45 mm |
| Qualified captive land | Not applicable | 3.00 mm = 0.45 + 2.10 + 0.45 |
| Loading | Post-print glue-in | Insert through the open chimney at the slicer pause |
| Closure | External opening remained | Self-supporting 45-degree roof, 2.60 mm nominal rise |
| Cavity support | Not applicable | No internal support material |
| Finished access | Exposed/accessible | Completely buried; no glue and no access opening |
| Mating plastic gap | Approximately 0.05 mm | Preserved at approximately 0.05 mm |
| Nominal opposing magnet-face separation | Old documentation sometimes conflated this with the plastic gap | **0.95 mm = 0.45 + 0.05 + 0.45** for standard and Obi-Wan LM-lower base-side pairs; **1.10 mm = 0.45 + 0.15 + 0.05 + 0.45** for Obi-Wan LM-upper/UM ring pairs |
| Classic-wall retention | Not applicable | 0.45 mm skins preserve approximately 0.42 mm wall paths |

The cavity bottom follows the circular D5 magnet profile. Its upper loading
region remains vertically open until insertion, with continuous retaining
extrusion on both axial sides. The production helper is derived from the
physically tested `coupons/obiwan_ae_embed` implementation; the earlier 0.30 mm
skin was rejected because Bambu Studio Classic omitted it with the qualified
0.4 mm nozzle process.

## Exact released magnet inventory

The family counts below are enforced in code and must not change silently.

| Family | Magnet-bearing STLs | Captive stations | Converted printable outputs |
|---|---:|---:|---|
| B2 | 2 | 8 | `lx521_top_base_4of4_vase_b2`, one per stand state |
| C7 | 2 | 8 | `lx521_top_c7base_4of4_vase_b2`, one per stand state |
| A | 8 | 8 | Four `lx521_top_addonA_*_shoulder_*` receivers per stand state |
| B1 | 4 | 8 | Two `lx521_top_addonB1_*_wing_*` receivers per stand state |
| V0 | 2 | 4 | `lx521_top_v0_4of4_vase`, one per stand state |
| V1 | 2 | 8 | `lx521_top_v1_4of4_vase`, one per stand state |
| V1-A | 8 | 8 | Four `lx521_top_v1addonA_shoulder_*` receivers per stand state |
| V1-B1 | 4 | 8 | Two `lx521_top_v1addonB1_wing_*` receivers per stand state |
| V1L | 2 | 8 | `lx521_top_v1l_4of4_vase_b2`, one per stand state |
| Obi-Wan | 4 | 12 | LM monolith and UM carrier in both stand states |
| Obi-Wan-split | 4 | 8 | Keyed LM lower and upper alternatives in both stand states |
| Obi-Wan-Ac | 6 | 6 | Left/right LM-lower, LM-upper, and UM wing segments |
| Obi-Wan-Ae | 6 | 6 | Left/right LM-lower, LM-upper, and UM wing segments |
| coupon1 | 2 | 2 | `lx521_coupon_1_fit_plate`, one per stand state |
| **Total** | **56** | **102** | Exact fail-closed catalog |

State accounting is 22 magnet-bearing STLs / 45 stations in
`floor_stand`, 22 / 45 in `no_floor_stand`, and 12 / 12 shared Ac/Ae wing
segments. A Obi-Wan LM monolith and its two keyed substitutes are alternative
ways to print one installed carrier; both forms are release artifacts and are
therefore deliberately counted in the catalog.

The two canonical Obi-Wan LM monoliths have an approximately
236.41 x 313.75 mm mandatory front-down footprint and do not fit a P2S
256 x 256 mm bed. They remain valid large-format release outputs. Their eight
site contracts are audited on the exact, same-state keyed lower/upper
geometry; that proxy coverage is not a fabricated monolith G-code pause.

## Variant-specific adaptations

- Standard, C7, A, B1, V1, V1-A, V1-B1, and V1L retain their established
  in-plane site positions, axes, mating datums, and approximately 0.05 mm
  interface clearance. Only the local material needed for the 3.00 mm captive
  land and roof is added.
- Obi-Wan upper/lower LM and UM sites retain their radial/base-side axes.
  Ring-radial cavity datums sit at structural radius +0.65 mm, 0.15 mm beneath
  continuous exposed R113.8/R52.5 side fairings. The fairings stop only inside
  existing LM--UM and T--UM cusp/service regions, preserve the 0.40 mm LM--UM
  gap, and have no local pad, boss,
  flat, protrusion, or other visible magnet-location cue.
- Obi-Wan Ac/Ae receivers use the matching pair axes and protected local lands.
  Existing outlines, acoustic depth laws, three-piece splits, and dovetail
  geometry remain authoritative.
- V0 has two rear-axis stations and no released mate. Its front-down axial
  treatment uses a 0.45 mm rear skin, 45-degree conical closure, 2.10 mm
  cavity, and 0.45 mm inner skin rather than pretending the transverse coupon
  topology fits an axial site.
- Coupon 1 is retained as a released fit/regression plate with one unpaired
  station per stand state. It is not a polarity-pair qualification coupon.

The conversion preserves established part splits, driver and insert
locations, dovetails, mating faces, route/duct geometry, and intended magnet
polarity. The Obi-Wan ring exterior is the deliberate exception: its former
station-local backing cues are replaced by continuous R113.8/R52.5 fairings.
Keepout tests treat each magnet cavity, both skins, and the actual surrounding
carrier or wing wall as real geometry.

## Front-face-down texture contract

**Every released baffle, carrier, attachment, wing segment, and production
coupon prints with its acoustic front face on the build plate.** This applies
to magnet-bearing and non-magnet pieces alike so their visible texture is
consistent.

The broader orientation inventory is **102 acoustic STL/sidecar pairs**:
45 in each stand state plus six Ac and six Ae wing segments. Do not confuse
that orientation count with the separate 102 captive-station count above.
Each acoustic STL has an adjacent, hash-bound `<stem>.print.json` which
records:

- exact source X rotation of 180 degrees;
- only an optional in-bed rotation about Z;
- no tilt or auto-orientation;
- translation placing the source front datum at STL Z = 0; and
- the exact STL hash and size.

A missing, orphaned, stale-hash, tilted, or translation-inconsistent sidecar
is a release failure. The large Obi-Wan monoliths must also remain front-face-down
on a sufficiently large machine; tilting one to force it onto a P2S is not an
approved workaround.

There are exactly two orientation-sidecar exclusions:

- `floor_stand/stl/lx521_polar_base_1of2_base.stl`
- `floor_stand/stl/lx521_polar_base_2of2_rotor.stl`

These are non-acoustic polar-index measurement jigs with no acoustic front
datum. They print in their documented functional flat orientations; applying
X180 would put the spigot, detent noses, or rotor fence into the build plate.
They are not exceptions for any baffle or attachment.

## Pause and polarity authority

An STL cannot store a pause. The final manifest is authoritative for each STL
and groups only sites whose first actual closing layer has the same Z. It must
provide the part/variant, sites, orientation, last fully open layer, CAD
bury/roof-start plane, first closing layer, Bambu pause-marker Z, count, and
site-specific polarity vector.

Required Obi-Wan coupon-equivalent regressions for the tested P2S 0.4 mm /
0.16 mm High Quality / Classic profile are UM **Z = 5.96 mm** and LM
**Z = 8.52 mm**. These are regression requirements, not global pause values.
Every unrelated station gets its pause from the actual generated G-code layer
schedule.

| Site role | Marked/N-pole convention |
|---|---|
| Base or carrier | Marked/N pole points out of the carrier toward its mate, along the listed installed pair-axis vector |
| Receiver, attachment, or Ac/Ae wing | Marked/N pole points along the same installed pair-axis vector as the carrier magnet; the face toward the carrier is therefore the opposite pole |
| V0 | Provisional and unpaired: marked/N pole points rearward along the listed installed axis; verify any future mate before burial |
| Coupon 1 | Unpaired regression convention: marked/N pole points installed -Y; no mating or attraction claim |

Mirrored parts must follow their listed local vectors. Do not infer polarity
from a visually similar left/right pocket and do not insert every disc with
the same visible face merely because the parts are mirrored.

For every released station, insertion is independent of polarity: lower the
magnet vertically from above the paused part, from its +Z side along
**print -Z**, exactly `print_insertion_direction_xyz = [0, 0, -1]`.

## User print procedure

1. Keep the exact STL and adjacent `.print.json` together. Confirm the part is
   front-face-down and has only its recorded in-bed Z rotation.
2. Use the pinned Bambu P2S 0.4 mm / 0.16 mm High Quality / Classic-wall
   process. Internal cavity support must remain disabled. Do not let Bambu
   auto-orient or tilt the part.
3. Mark and verify one pole on every D5 x 2 mm magnet against a retained
   master magnet before slicing or printing.
4. Use the authoritative per-part pause manifest. Do not copy 5.96 or 8.52 mm
   to an unrelated part.
5. At each pause, confirm the previous layer is the final fully open cavity
   layer and that the next toolpath begins the roof. Insert the manifest's
   exact count from above along print -Z, using every site's listed polarity.
6. Press every disc fully into its circular cradle. Its highest point must be
   below the completed layer and clear of the resumed nozzle path. Remove no
   retaining wall and add no glue.
7. Check count, seating, and polarity a second time, clear the build plate,
   and resume. **Polarity cannot be corrected after the roof buries the
   magnet.**

No file in this migration uploads to a printer or starts a print.

## Converted source and release-artifact categories

### Source authorities and infrastructure

| Category | Changed/new authorities |
|---|---|
| Reusable geometry | `captive_magnets.py` |
| Standard and variant CAD | `top_baffle_nd25fw4.py`, `top_baffle_nd25fw4_b.py`, `top_baffle_nd25fw4_{b2,c7,v0,v1,v1l}_split.py`, `top_baffle_nd25fw4_{v0,v1,obiwan}.py` |
| Attachments and Obi-Wan alternatives | `top_baffle_nd25fw4_attachments.py`, `top_baffle_nd25fw4_v1_attachments.py`, `top_baffle_nd25fw4_obiwan_{attachments,bridge,lm_split,route}.py`, `obiwan_wings_cad.py` |
| Front-down/STL exporters | `front_down_contract.py`, `export_piece_stls.py`, `export_coupon.py`, `export_obiwan_wings.py` |
| Catalog and slicing | `generate_captive_magnet_catalog.py`, `captive_magnet_release_catalog.schema.json`, `slice_captive_magnets.py`, `captive_magnet_slicing_profile.json`, `json_schema_subset.py` |
| Build and promotion | `Makefile`, `remote_cad.py`, `write_obiwan_release_manifest.py`, `check_manifold.py` |
| Tests | `test_captive_magnets.py`, `test_release_metadata.py`, `test_slice_captive_magnets.py`, plus updated clearance, Obi-Wan, wing, and remote-CAD tests |
| Documentation/diagrams | `README.md`, `PRINTING.md`, `VARIANTS.md`, `obiwan_acoustic_wings_spec.md`, `obiwan_physical_qualification.md`, `obiwan_r6f_cad_brief.md`, and updated routing/overlay generators |

### Generated release categories

- Floor and no-floor STEP-first variant masters, split packages, attachment
  packages, and assembled review STEPs.
- All 45 released non-polar-index STL/`.print.json` pairs in each stand state,
  including the 22 magnet-bearing pairs in each state.
- Six Ac and six Ae wing STL/`.print.json` pairs, their STEP masters and
  assemblies, facts, transactional manifests, and review images.
- Obi-Wan staged-build manifests, release manifests, cable-routing PNGs, and
  driver/variant overlays whose magnet depictions now show the captive skins,
  0.95 mm base-side paired separation, and 1.10 mm Obi-Wan ring-pair
  separation.
- The hash-bound captive-magnet release catalog, Bambu profile provenance,
  per-STL G-code validation records, five-layer SVG/PNG evidence, and the
  authoritative JSON/CSV/Markdown pause outputs after final publication.

## Explicitly excluded artifacts

| Excluded category | Reason |
|---|---|
| Oversized one-piece STEP masters, `*_assembled.step`, `*_attachments.step`, and other STEP review packages | Geometry authorities or review containers, not additional prints; their released constituent STLs are inventoried |
| `coupons/obiwan_ae_embed` | Physically validated geometry/process reference, not a production baffle print; retained as regression authority |
| `lx521_coupon_7_recess_seat.stl` in each stand state | Diagnostic driver-seat crop whose current XY region contains no captive site |
| `gen_c_variants.py` | C1-C6 concept-only magnet-boss study generator; its one-off raster is not retained |
| `gen_um_knife_draft.py` | Obsolete pin-magnet slide drawing; no released solid |
| Hypothetical V0 mating attachment | No released printable mate exists; V0 remains explicitly unpaired |
| Legacy exposed-pocket generated artifacts | Obsolete bytes superseded in place by the 56 hash-bound captive STLs |

The two polar-index jigs listed in the orientation section are also excluded
from the acoustic front-down sidecar inventory, but not because they are
legacy or concept artifacts: they are valid non-acoustic measurement tools.

## Validation completed before final slice publication

- Pure release-metadata/front-down tests: **26 passed**.
- Slicer/catalog/toolpath unit tests: **35 passed**.
- Remote-CAD transport, guard, and promotion tests passed, including expected
  simulated guard failures.
- Remote native/source suites, standard clearances, Obi-Wan R6F geometry,
  terminal/Faston/Y-boot service matrices, exact cable envelopes, buried
  backs, route contracts, floor datum, and analytical structure gates passed
  before artifact publication.
- Every production STL export is subject to the strict manifold gate. The V0
  sharp axial roof's exact repeated-vertex apex records are removed
  losslessly before that unchanged gate; ordinary collinear or malformed
  facets remain hard failures.
- Profile preflight resolves Bambu Studio, P2S 0.4 mm, 0.20 mm first layer,
  0.16 mm subsequent layers, Classic walls, 0.42/0.45 mm nominal paths,
  support disabled, and Bambu PLA Tough+ without contacting a printer.

The expected complete-audit policy is 54 actual P2S slices covering 94 pause
stations plus eight exact same-state keyed-split proxy contracts for the two
oversized LM monoliths. Only the finalized observed totals in the release
manifest are authoritative.

## Representative CAD review

| Review target | CAD Viewer URL |
|---|---|
| No-floor Obi-Wan assembled carrier/attachments | `[FINALIZE AFTER PROMOTION: viewer URL]` |
| Floor Obi-Wan assembled carrier/integral stand | `[FINALIZE AFTER PROMOTION: viewer URL]` |
| V0 axial captive stations | `[FINALIZE AFTER PROMOTION: viewer URL]` |
| Ac assembled wing and receivers | `[FINALIZE AFTER PROMOTION: viewer URL]` |
| Ae assembled wing and receivers | `[FINALIZE AFTER PROMOTION: viewer URL]` |
| Representative sliced captive-station G-code | `[FINALIZE AFTER PROMOTION: viewer URL]` |

## Remaining physical risks and required qualification

- The reference coupon physically validates the topology and 0.45 mm Classic
  retaining walls at 0.90 mm opposing-magnet separation. Production pairs
  retain the 0.05 mm plastic gap: standard and Obi-Wan LM-lower base-side
  pairs separate magnet faces by 0.95 mm, while Obi-Wan LM-upper/UM ring pairs
  include the 0.15 mm buried carrier datum and separate them by 1.10 mm.
  Measure pull/retention through each production stack; do not rely on a
  supplier bare-magnet pull figure.
- V0 has no mating production print. Its marked-pole direction is deliberately
  provisional until a real mate is designed and tested.
- The raw MU10RB reference mesh omits the electrical terminal tabs. The chosen
  between-screw terminal clocking, Faston bodies, lead slack, and removal
  envelope remain a physical-fit check; the current 12 mm proxy pull equals
  the provisional exposed-tab length and has no positive overtravel margin.
- The optional Obi-Wan keyed LM split still requires a process-matched physical
  coupon/full print proving the thin socket walls, two-pin/socket seating,
  coplanar front faces, route-seam continuity, and repeated assembly fit.
- The canonical Obi-Wan LM monoliths cannot be printed on a P2S in the mandatory
  orientation. Use the exact keyed alternative or a verified larger-format
  printer; do not scale, crop, or tilt them.
- The integral floor version has analytical strength results, not completed
  physical proof. A positively attached anti-tip tether/anchor and the
  documented distributed proof-load/creep program remain mandatory. Bambu
  PLA Lite is provisional and fails the vertical-5g analytical screen; it is
  not a qualified substitute.
- Ac/Ae dovetails register and interlock in plane but provide no independent
  Z retention. Their documented equal-treatment external retention method and
  a physical pull test remain required.
- A changed nozzle, layer schedule, wall generator, first-layer height,
  filament profile, STL hash, or Bambu preset invalidates the published pause
  evidence. Re-slice the complete catalog rather than editing pause heights by
  hand.
- Magnet polarity is irreversible after closure. A mirrored-site loading
  error cannot be repaired without destroying the buried roof or reprinting
  the part.
