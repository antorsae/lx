# LX521.4 top baffle — ND25FW-4 face-to-face mod (V2)

Three printable top-baffle designs for an LX521.4 modification: Stock's
full-depth outline, Slim's thinner acoustic field, and Obi-Wan's separate
driver collars with optional wings. Choose one mounting state and one
complete configuration per speaker.

**Start with the [build guide](docs/BUILD_GUIDE.md)** for file selection,
per-speaker/stereo quantities, hardware and assembly. The generated
[file guide](to_print/FILE_GUIDE.md) lists all 42 choices and actual sliced
job estimates. There are 68 sliced projects and 8 projects requiring GUI
slicing across alternate nozzle/material lanes.

The [complete parts and magnet catalog (PNG)](images/generated/catalog/ALL_ITEMS_MAGNET_CATALOG.png)
shows all 78 current part variants, their buried magnet locations and types,
shared parts, alternate tweeters and separate test fixtures. The accompanying
[file map](images/generated/catalog/README.md) identifies every source.

For the 0.6 mm PETG-GF/PLA setup, use the generated [print and hardware
policies](docs/PRINT_POLICIES.md): structural LM/UM infill is 100%, wings
are 10%, and the retained ND25FN V4 uses M3 fasteners. The [current update
report](review/print_policy_update_20260912/SUMMARY.md) links the regenerated
print files, support exceptions and D6 magnet test pair.

Stock is the canonical CAD baseline. Slim is experimental. Obi-Wan and BMR
options are qualification candidates; printed fit, long-term loaded behavior
and acoustic performance are not established by CAD renders or passing code
checks. See the [current test procedure](docs/PETG_GF_QUALIFICATION.md).

## Product comparison

One row per product, each showing both stand states, then one row for the
tweeter options. Every render uses one camera and one declared frame per scale
group, so the panels within a row and the three product rows against each
other are directly comparable. Regenerate them with `make iso_matrix` after
any CAD change.

![Stock, both stand states](images/generated/iso/rows/stock_row.png)

Stock is the canonical product: the complete 18.3 mm outline printed as four
registered pieces — [`docs/stock.md`](docs/stock.md).

![Slim, both stand states](images/generated/iso/rows/slim_row.png)

Slim is the same outline with the acoustic field thinned to 11.5 mm, sharing
one front plane end to end — [`docs/slim.md`](docs/slim.md).

![Obi-Wan, both stand states](images/generated/iso/rows/obiwan_row.png)

Obi-Wan keeps only the two driver collars; both panels add its optional
crescent and flat wings, because the mandatory geometry alone is two bare
rings — [`docs/obiwan.md`](docs/obiwan.md).

![The four tweeter carriers](images/generated/iso/rows/tweeter_row.png)

The four tweeter carriers share their own larger scale, so they are
comparable with each other but not with the product rows above.

## Tweeter options

The established shelf offers two driver families:

- **Dayton ND25FW-4 face-to-face pair** — two dome tweeters with waveguide,
  bolted through the baffle so their faceplates clamp the crescent between
  them. This is the default on every product.
- **Tectonic TEBM35C10-4 BMR pair** — balanced-mode radiators fitted in place
  of the domes.

What you actually swap to take the BMR option depends on the product, because
each one carries its tweeters on a different part:

| Product | Interchangeable part | ND25FW-4 (default) | TEBM35C10-4 |
|---|---|---|---|
| Stock | the **vase**, piece `04` | the standard B2 vase: it carries both the dome pair on its integral crescent and the MU10 upper-mid seat | the opposed-BMR vase replaces that whole vase; `make vase_tebm35c10_4_cad` |
| Slim | the **vase**, piece `04` | the standard V1 vase — the same arrangement thinned to 11.5 mm | the Slim-profile opposed-BMR vase, the same one-piece swap |
| Obi-Wan | the **crescent** on the UM collar | the released tweeter crescent add-on | two candidate BMR crescents, coaxial or opposed, on the identical half-lap mount; `make obiwan_bmr_crescent_cad` |

The three BMR implementations arrange the drivers differently. On Stock and
Slim they are **opposed**: the lower BMR faces front and the upper one faces
rear. Obi-Wan offers both arrangements on one mount, with the lower acoustic
axis fixed at `(0, 452.494193)` rather than recomputed from the land radius —
**coaxial**, the two stacked back to back, 50.2 mm deep; or **opposed**, the
vase's own layout on a second land 49.3 mm above the first, 25.1 mm deep but
49.3 mm taller. Both put the lower driver 86.413 mm from the MU10 axis.

The default BMR land is now a conservative clipped **Ø63** full-circle
prototype. Its two side-magnet faces are at `x=±31.326666`, so its actual
maximum width is **62.653 mm**; each face moved inward by about **1.508 mm**
from the former land. The alternate, explicitly unqualified
**BMR-slim** topology keeps those same side magnets and maximum width while
removing the unused circular field: it uses a **Ø56 driver-following core**,
four local M2 pads and two discrete magnet lobes. Both topologies require
physical driver-fit, insert, cable and magnet-pull qualification. The Obi-Wan
BMR crescents remain **candidates**: they are not release-authorized and are
deliberately absent from the release inventory, the stage manifests and the
released captive-magnet catalog.
[`docs/VARIANTS.md`](docs/VARIANTS.md#candidate-tebm35c10-4-bmr-crescents-obi-wan)
puts the three BMR parts side by side.

The [UM finish revision](docs/UM_FINISH.md) closes the LM handoff cover, fills the UM seat underside and rounds M2 access in the standalone UM and fused ND25FN candidate.
The fused candidate also blends the lower rear band into its organic surround, removing the rectangular thickness steps between the LM ears. [Updated rear detail](candidates/nd25fn4_crescent/views/UM_lower_band_oblique.png).
Its latest UM follows the supplied concept outline, with a tighter lower waist and flowing shoulders. The lower edge now meets the existing LM front at Z18.30 mm without covering it. The front is wider than the rear, with an inclined outer wall and broad shallow bowl. Four Ø6×3 mm magnets remain buried, and the upper wings and enclosed gallery are reshaped to fit. The driver seat and LM mating faces retain their datums. [Reference overlay](candidates/nd25fn4_crescent/views/UM_reference_overlay.png) · [Outline validation](candidates/nd25fn4_crescent/reference_validation.json) · [Magnet selection](candidates/nd25fn4_crescent/MAGNET_SELECTION.md).

Separate CAD candidates include the [Dayton ND25FN-4 retained V4
crescent](candidates/nd25fn4_crescent/README.md), recreated from the supplied
`MU10_ND25FN_V4_Retained_Package.zip` and fused with the actual Obi-Wan UM carrier.
The compact revision lowers the tweeters 20.13 mm, steepens the lower forward
flare, and gives the UM a broad, curved surround with a smooth waist. The tweeter retains
its 35.8 mm depth, 62 mm pitch, service caps and screw retainers; the new UM bowl
makes the complete body approximately 38.20 mm deep.
One shared body fits both stand configurations, retaining their LM mounting
interfaces and an enclosed cable gallery buried in the UM surround. A broad rear
thickness loft removes the abrupt UM-to-tweeter arc junction. Flat and graded upper wings
match its outline and four concealed shoulder magnets. The changed acoustic
surfaces need acoustic validation.
Its [prepared P2S 0.6 HF PETG-GF + PLA jobs](candidates/nd25fn4_crescent/print/README.md)
use 15% gyroid in the tweeters and the regular UM's 100% zig-zag infill.
The shared body, caps/retainers and matching upper wings have passed slice checks,
including buried-magnet pauses. The user reports printing the earlier body with
PETG Translucent and PLA; physical finish, fit and retention qualification remain pending.
Separate [PETG Translucent + PLA Translucent jobs](candidates/nd25fn4_crescent/print_translucent/README.md)
use the same 0.6 mm HF nozzle with PETG in AMS slot 4 and PLA in slot 2. Their revised
body keeps the exterior magnet-cover paths at a constant 0.52 mm width; a small
surface test is included for physical confirmation.
An [alternate translucent body and caps/retainers set](candidates/nd25fn4_crescent/print_translucent_changeover/README.md)
uses the Engineering Plate with glue at 70 °C, a 5 mm outer brim, no raft, 560 mm³ purge
each way and an explicit 12 mm³/s PLA flush following the reported changeover blockage.
The file guide records the support checks and exceptions: Bambu's mutual-support guide
excludes PETG Translucent and only covers PLA Basic with PETG Basic/HF. This custom
material/plate calibration has not yet been physically qualified.
These files and their mounting checks remain outside the regular 42-choice shelf.
The existing Purifi PTT1.3 crescent remains in
`build/ptt_crescent_PTT1.3T04-HAG-01/`.

## Products

The project has one human-facing artifact catalog:
[`artifacts/`](artifacts/README.md).

| Product | Geometry | Optional perimeter | Tweeter options | Status | Doc |
|---|---|---|---|---|---|
| [Stock](artifacts/stock/) | B2, 304.802 x 453.457 x 18.3 mm | A-comp shoulders **or** B1 wings | ND25FW-4 crescent (integral) or TEBM35C10-4 BMR vase | Canonical CAD | [`docs/stock.md`](docs/stock.md) |
| [Slim](artifacts/slim/) | V1L + V1; 11.5 mm front-flush acoustic field, full-depth bottom strip | matching V1 shoulders **or** V1 wings | ND25FW-4 crescent (integral) or TEBM35C10-4 BMR vase | Experimental | [`docs/slim.md`](docs/slim.md) |
| [Obi-Wan](artifacts/obiwan/) | separate LM/UM collars; floor and stock-bridge states | flat constant-depth or graded weighted-depth wings | ND25FW-4 crescent add-on, or a candidate coaxial or opposed BMR crescent | Candidate; not release-authorized | [`docs/obiwan.md`](docs/obiwan.md) |

The original state-oriented build outputs remain in `build/floor_stand/`,
`build/no_floor_stand/`, and `build/wings/` because the validation pipeline depends on
them. `artifacts/` adds stable names, hashes, and product grouping through
relative links without duplicating large CAD files. See
[`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md) for the intent, assumptions,
and release boundary; [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md)
documents the implemented source/package and generated-state boundary.

## Use the delivered files

Open [to_print/](to_print/README.md) and select the correct material/nozzle
lane. Sliced `.gcode.3mf` jobs retain their audited orientation and magnet
pauses. PETG-GF core `_GUI.3mf` projects must be sliced in Bambu Studio and
the exported result audited. Do not print both a combo and its contents.

```sh
make to_print_validate  # read-only validation of every delivery and the CAD facade
make delivery_package  # verified local dist/lx521-print-pack.zip
```

The print pack contains actual files and checksums, so it works independently
of the source checkout's relative symlinks. Physical evidence remains pending.

## Develop and regenerate

Use Python with the dependencies in `cad-remote-requirements.lock`. The
vendored driver references are self-contained under `vendor/SEAS/`; no
sibling checkout is needed for those inputs. VTK renders CAD with an actual
depth buffer, and Matplotlib/Pillow compose the comparison panels.

```sh
LX_CAD_EXECUTION=local make PYTHON=<venv>/bin/python -j1
make artifacts          # relink and rehash the CAD facade
make iso_matrix         # regenerate the CAD comparison images
make delivery_refresh   # bind already-published files and regenerate the file guide
```

CAD builds default to the original maintainer's remote host `osado.lan`.
Set `LX_CAD_EXECUTION=local` on another workstation. Remote execution,
resource limits and promotion are documented in [REMOTE_BUILD.md](docs/REMOTE_BUILD.md).
A source change requires current CAD/slice provenance before republishing;
validation does not silently regenerate or re-certify old geometry.

## Generated artifact layout

The default remote `make` builds BOTH stand-foot states. Use
`LX_CAD_EXECUTION=local make -j1` only for an intentional local build:

    build/floor_stand/      LX_STAND_FOOT=1: Stock/Slim fused foot + NL8 panel;
      stl/  *.step  *.png     Obi-Wan integral LM-owned W64 floor stem/foot + NL8 panel
    build/no_floor_stand/   LX_STAND_FOOT=0: Stock/Slim flat piece_bottom + bridge;
      stl/  *.step  *.png     Obi-Wan solid bridge web fused into the LM core
    build/wings/{flat,graded}/    Obi-Wan acoustic wing families
    build/vase_TEBM35C10-4/{stock,slim}/   opposed-BMR vase, full Ø63 lands
    build/bmr_slim_TEBM35C10-4/proud/{stock,slim}/  Ø56-core/lobed vase alternatives
    build/bmr_crescent_TEBM35C10-4/       candidate BMR crescents, both variants
    build/bmr_slim_TEBM35C10-4/           lobed Obi-Wan CAD-only candidates
    build/common/           flag-independent shared outputs
    images/generated/iso/   the product-comparison cells and rows/ images

Each folder contains all proud-family variants and the matching Obi-Wan
core/add-on set. In Stock and Slim, `piece_bottom` is the only functionally
different base piece; the other base STLs can differ by <0.05 mm as
the foot-entry knots move. In Obi-Wan, the UM core is state-independent. The
floor LM owns the integral stand and only the shared upper shoulder; the
no-floor LM owns the complete fused bridge web. Their lower magnet axes are
coincident on that shoulder, but floor mode has no shallow skirt or rail below
it.
`build/common/attachments.step` is flag-independent and is
promoted as one shared file. The `LX_STAND_FOOT` environment flag defaults to 1.

Per-product printable-piece tables live in the product docs. The two
product-independent STL groups are:

| STL in `build/<state>/stl/` | Footprint (mm) | Used by |
|---|---|---|
| `lx521_coupon_*` | small blocks/gauges | calibration, routing, and clocking checks ([`docs/PRINTING.md`](docs/PRINTING.md)) |
| lx521_polar_base_1..2of2 | Ø216 / 169×185 | polar-measurement turntable under the stand foot (floor_stand only) |

## Shared source and evidence map

Geometry modules are listed in each product's doc. These are the shared
entry points:

| File | What |
|---|---|
| `scripts/export_piece_stls.py` | Exports the print-ready proud-family or Obi-Wan core/add-on STLs (`--variant`, `--outdir`) and one exact adjacent, hash-bound `.print.json` authority for every STL |
| `scripts/export_steps.py` | Exports a module's `gen_step()` to STEP via build123d's native exporter (`<module.py> --output <path>`) — no CAD-skill dependency |
| `scripts/gen_product_iso_matrix.py` | Renders the standardized product-comparison ISO set into `images/generated/iso/` from the promoted STEP files; one shared camera, one declared frame per scale group |
| `Makefile` | Generates STEPs/STLs/PNGs for both stand states into `build/floor_stand/` and `build/no_floor_stand/` (see "Generated artifact layout"). Local OCC jobs are serial; the remote executor uses bounded parallel slots, and every CAD subprocess runs through `scripts/run_memory_guarded.py`. |
| `scripts/remote_cad.py` / `cad-remote-requirements.lock` | Content-addressed SSH executor, resumable job control, verified artifact return, and exact remote Python environment |
| `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md` | Authoritative per-STL front-face-down orientation, actual sliced open/closing layers, Bambu Custom park/pause/restore events, grouped magnet counts, and local-axis polarity |
| `review/CAPTIVE_MAGNET_ARTIFACT_INVENTORY.md` | Clickable inventory of all 58 magnet-bearing STL records, their 56 locally generated/directly loadable G-code-bearing Bambu 3MF projects, descriptive piece names, and exact magnet counts; also reconciles the 86 transverse and eight exact split-proxy stations |
| `build/<state>/stl/*.stl` and `build/wings/{flat,graded}/stl/*.stl` | The enforced acoustic-print inventory is 39 nonpolar front-face-down STL/sidecar pairs in each stand state plus ten flat and ten graded pairs: 98 exact pairs total. Every acoustic piece is source-X180 with only an optional in-bed Z rotation and its front datum at STL Z=0. A missing, orphaned, stale-hash, tilted, or translation-inconsistent `<stem>.print.json` fails release validation. The two floor polar-index jigs are the sole orientation-sidecar exclusions because they are fixtures with no acoustic front-face datum. |

The generated directories are **candidate packages**, not physical-release
authorization: even `make release` performs CAD, artifact and manifold checks
only, and the Obi-Wan state manifests record `release_authorized: false`.

## Documentation

Products:

- [`docs/stock.md`](docs/stock.md) — Stock: key dimensions, print split,
  proud cable routing, magnet attachment, assembly.
- [`docs/slim.md`](docs/slim.md) — Slim: V1 vase, V1L LM section, the
  keyed 283° UM outlet.
- [`docs/obiwan.md`](docs/obiwan.md) — Obi-Wan: two-collar geometry,
  buried routes, floor/no-floor structure, structural screens, assembly.

Cross-cutting authorities:

- [`docs/PROJECT_SCOPE.md`](docs/PROJECT_SCOPE.md) — intent, three-product
  inventory, CAD brief, assumptions, release boundary.
- [`docs/REPOSITORY_STRUCTURE.md`](docs/REPOSITORY_STRUCTURE.md) — layout and
  the source/generated-state boundary.
- [`docs/VARIANTS.md`](docs/VARIANTS.md) — variant/add-on catalog, envelope
  compatibility matrix, hardware, and retired C7/V0 design history.
- [`docs/PRINTING.md`](docs/PRINTING.md) — filament, slicer profile, coupons,
  fastener torques, insert installation, magnet pauses, print constraints.
- [`docs/REMOTE_BUILD.md`](docs/REMOTE_BUILD.md) — remote executor, cache
  seeding, promotion transaction, memory profiles.
- [`docs/CAPTIVE_MAGNET_SLICING.md`](docs/CAPTIVE_MAGNET_SLICING.md) — pause
  and embed workflow.
- [`docs/obiwan_acoustic_wings_spec.md`](docs/obiwan_acoustic_wings_spec.md) —
  flat/graded wing design authority.
- [`docs/obiwan_physical_qualification.md`](docs/obiwan_physical_qualification.md)
  — fail-closed physical qualification record.
- [`docs/README.md`](docs/README.md) — the complete documentation index.
