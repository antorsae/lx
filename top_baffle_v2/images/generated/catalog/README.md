# Complete parts and magnet catalog

This poster covers the earlier **P2S split arrangement** and diagnostic
parts. Current H2C quantities, continuous wings and all three tweeter
families are in the [H2C catalog](../../../to_print/h2c/README.md) and
[selection guide](../../../docs/TWEETER_OPTIONS.md). The naming update does
not change the 78 depicted part variants.


![Regular and integrated-waveguide Obiwan upper choices at one scale](../iso/rows/obiwan_upper_row.png)

This overview puts ND25FW-4, coaxial BMR, opposed BMR and ND25FN-4 waveguide
on the same LM datum. The detailed poster below shows their individual
parts and hidden magnet locations.

[Open the full-resolution PNG](ALL_ITEMS_MAGNET_CATALOG.png).

The sheet covers 78 individual part variants: Stock, Slim, regular Obi-Wan,
the fused Dayton ND25FN-4 waveguide, full-land and slim-land BMR alternatives,
Purifi, monolithic LM options, grommets, service parts and test fixtures.
It follows the print shelf's declared shared sources and includes both offered
stand states. Unselected raw-build copies, combo plate layouts and review
assemblies are not counted as additional product choices. The four STEP-only
BMR-slim candidates are clearly labeled. Raw stand directories can contain
different state-specific builds of parts the shelf shares; use the source
identified in this file map rather than selecting an arbitrary counterpart.

Filled dots mark individual buried magnets. Outlined dots in the assembled
examples mark mating pairs. Markers reveal the hidden locations through the
plastic and are enlarged for readability; they are not exposed holes. The
part views are independently fitted to their frames, so their displayed sizes
are not comparable.

| Magnet | Selected stock | Used at |
|---|---|---|
| D5 | Ø5×2 mm N52, Superimanes D-05-02-N52 | Regular Stock/Slim/Obi-Wan, preserved ND25FN-4 LM interfaces, BMR side stations and D5 fixtures |
| D6 | Ø6×3 mm N45, Superimanes D-06-03 | ND25FN-4 UM body and matching upper-wing stations, plus D6 test pair |

Counts for one complete speaker with one perimeter set: Stock/Slim 8 D5;
regular Obi-Wan 12 D5; ND25FN-4 8 D5 + 8 D6. These totals exclude alternate styles,
alternate stand states and test fixtures. BMR side stations have no delivered
matching perimeter. Magnet counts describe the CAD pockets, not measured
holding force or proof of physical qualification.

The [machine-readable file map](parts_catalog.json) contains every displayed
part's source path, source hash, placement transform, magnet seat coordinates,
type and authority, plus the grouping and assembled mating pairs used on the PNG.
[Validation](validation.json) accounts for all 42 shelf choices: 38 individual
part choices and four combo layouts of those same pieces. CAD/print files are
read as inputs and are not modified by this catalog.

ND25FN-4 views use the approved design STLs. Its prepared print variants add internal
magnet-loading relief; use the [ND25FN-4 print guide](../../../candidates/nd25fn4_crescent/print/README.md)
for the earlier P2S jobs, or the [H2C catalog](../../../to_print/h2c/README.md) for current files.

Regenerate from the project root:

```sh
../.venv/bin/python scripts/generate_parts_magnet_poster.py
```

The generator checks current source and orientation hashes, renders existing
STLs, and tessellates the four STEP-only candidates in memory. It does not
export those candidates as printable meshes. Tile caching is keyed by the
source geometry, placement, marker data and render settings. Inspect the PNG
after regeneration before updating its visual-review record.

Source authorities: `to_print/catalog.json`,
`review/captive_magnet_release_catalog.json`, candidate facts/catalogs and
current `.print.json` orientation records. ND25FN-4 seats are recovered from actual
closed cavities using its print-pause geometry routine. Frozen diagnostic
coupon markers use enclosed cavity centroids rather than production seat
datums. Source-only history and retired C7/V0/three-piece-wing layouts are
excluded from the current product inventory.
