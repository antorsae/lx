# P2S print shelf

**Current printer: [H2C](h2c/README.md).** Its catalog includes all three
tweeter families: ND25FW-4, TEBM35C10-4 BMR and ND25FN-4 printed waveguide.
This page retains the earlier P2S files and split arrangement.


Start with the [build guide](../docs/BUILD_GUIDE.md), then use the generated
[file guide and estimates](FILE_GUIDE.md) to choose exact files.

For the fused **Dayton ND25FN-4 waveguide + UM**, use its separate [sliced 0.6 HF Tinmorry
PETG-GF + PLA print guide](../candidates/nd25fn4_crescent/print/README.md).
It includes the shared body with 15% tweeter gyroid / 100% UM infill,
caps/retainers, matching upper wings and native magnet pauses. Those six jobs
are outside the regular shelf counts below.

The shelf contains **42 choices**: 11 Stock, 11 Slim and 20 Obi-Wan. Across
alternative material/nozzle lanes there are **68 sliced projects and
8 GUI projects**. These are alternatives, not the number to print.
`delivery_manifest.json` binds every project to its source STL and authority,
records hashes and settings, and includes geometry audits for GUI projects.
Existing slice provenance is retained in the lane manifests.

| Directory | Delivery |
|---|---|
| `<family>/stl` | Shared source meshes, one per choice |
| `<family>/3mf_04` | Audited PLA Basic jobs for a 0.4 mm nozzle |
| `<family>/3mf_06hf` | Audited PLA Basic jobs for a 0.6 mm high-flow nozzle |
| `obiwan/3mf_06hf_petg-gf` | Audited PETG-GF wing combos for 0.6 mm high flow |
| `obiwan/3mf_06hf_petg-gf_pla` | PETG-GF core projects requiring GUI slicing; PLA interfaces where support is enabled |

Open sliced `.gcode.3mf` jobs with the matching nozzle and preserve the
audited orientation/pauses. Open `_GUI.3mf` projects, assign material slots,
slice and inspect/export the result. A geometry-audited GUI project has no
audited toolpaths until this last step is completed. Core combos contain
four parts without a floor stand and five with one, including its NL8 lid.
They replace the corresponding individual files. Wing combos replace four
split2 pieces; flat and graded are mutually exclusive.

Obi-Wan remains a physical-qualification candidate. See the
[PETG-GF test procedure](../docs/PETG_GF_QUALIFICATION.md). No printer is
contacted by these commands.

```sh
make to_print_validate  # read-only: inventory, hashes, source and GUI geometry
make delivery_refresh  # bind an audited shelf and update the file guide
make delivery_package  # validate, zip actual files and verify checksums
```

Publishers remain available through `make to_print`, `make to_print_06hf`,
`make obiwan_petg_gui_projects` and `make obiwan_petg_wing_plates`. They may
regenerate files and require current CAD/slice provenance. Run
`make delivery_refresh` after publishing. Validation never repairs files,
creates slice workspaces, or touches promotion stamps.
