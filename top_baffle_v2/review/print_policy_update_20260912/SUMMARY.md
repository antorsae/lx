# Print policy update — 2026-09-12

The PETG-GF files now use 100% structural LM/UM infill and 10% wing infill. The V4 tweeter retention uses the shared M3 bore convention. All six regenerated crescent jobs pass their geometry, material, support, pause and retaining-wall checks. D6 physical qualification remains pending the supplied test print.

## Deliverables

- [Current print files, quantities and slicer estimates](../../candidates/nd25fn4_crescent/print/README.md).
- [Fused body — sliced 0.6 HF PETG-GF/PLA](../../candidates/nd25fn4_crescent/print/01_UM_Crescent_V4_06HF_PETG_GF_PLA.gcode.3mf).
- [Two caps and two M3 retainers — sliced](../../candidates/nd25fn4_crescent/print/02_Caps_TWO_Retainers_TWO_06HF_PETG_GF_PLA.gcode.3mf).
- [D6 body/wing test pair — sliced](../../candidates/nd25fn4_crescent/print/qualification/D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf), with [physical test instructions](../../candidates/nd25fn4_crescent/print/qualification/README.md).
- [Standalone LM top — updated 100% GUI project](../../to_print/obiwan/3mf_06hf_petg-gf_pla/obiwan_02_LM_top_keyed_2_of_2_GUI.3mf). This lane retains unsliced GUI projects; a fresh native review slice independently passed the material/process audit.
- [Generated common policies and exceptions](../../docs/PRINT_POLICIES.md), [insert inventory](../../docs/INSERT_CATALOG.md), and [current LM interface render](../../candidates/nd25fn4_crescent/views/LM_interface_orthographic_3D.png).

Use the regenerated body and M3 retainers together. One shared body remains compatible with both floor-stand and no-floor-stand LM configurations. The caps and their O-ring interface retain their geometry. No ZIP was produced and no printer operation was initiated.

## Changes and resolved findings

| Area | Result |
|---|---|
| Structural infill | Standalone LM top changed from 30% to 100% zig-zag; LM bottom, UM and core combo remain 100%. |
| Wing infill | Both regular PETG-GF wing plates and all four crescent upper-wing alternatives are regenerated at 10% gyroid. Fixed the regular builder's hidden 30% object override, which previously overrode its 10% global setting. |
| Lightweight regions | The fused body's tweeter modifier remains 15% gyroid above installed Y421. UM remains 100%. Caps and retainers remain 15%. Six walls and shell layers still apply. |
| V4 fasteners | Six M3×8 screws; Ø4.6 × 4 mm blind insert bores and Ø3.4 clearance holes. All 12 body insert sites now use M3. Regular M2 auxiliary ties retain their separate function. |
| M3 geometry | Moved the screw circle to Ø50.2 and enlarged the retainer to Ø55.9. The driver-side ligament is 1.60 mm; retainer-hole edge ligament is 1.15 mm. Retainer-to-cap entry clearance is 0.25 mm radially. Actual pilot/head gauges have zero collision. |
| Cable support | The enlarged M3 service chamber exposed an inadequate short entry blocker. Extended the exclusion along the complete projected cable entry. The final body has at least 0.712 mm sampled support-bead clearance outside the cable lumen. |
| Inserts | All 12 complete blind bores are excluded from support, including their floors; obsolete narrow tip cavities are removed. Actual support paths are checked independently of blocker appearance. |
| Caps | Both flat ceilings receive three full PLA contact planes at Z8.68, 8.84 and 9.00 before the PETG-GF ceiling at Z9.16. Critical-only and small-overhang filtering remain disabled for this plate. |
| Material mapping | Shared normalization completes all four mapping vectors, pins High Flow and sizes purge arrays to the loaded materials. Actual extrusion must use T0 for model/support and T1 for interfaces. Metadata alone is insufficient. |
| Material recipe | Frozen, hash-checked Tinmorry recipe: PETG-GF 260°C, flow 0.93, volumetric limit 12 mm³/s. PLA interface remains 220°C. Purge remains 280 mm³ each direction; flushing into model, infill and supports is now consistently disabled. |
| Native export | Corrected native XML newline encoding so reopening a project retains the magnet pause/park sequence. Material auditing now handles scalar and Standard/High Flow variant columns for both filaments. |
| Fit and appearance | Current mesh checks and 17 reviewed renders preserve the approved outline, waist and buried magnets. LM front faces remain flush at Z18.30, with zero interference or projected LM-face coverage in both configurations. |

## Magnet models and qualification

The magnets are **not all the same model**:

| Interface | Magnet selection | Quantity in the crescent assembly |
|---|---|---|
| Regular Obi-Wan / preserved LM | Ø5×2 mm N52; Superimanes D-05-02-N52 candidate SKU | Existing LM pockets remain D5. |
| Crescent UM to matching upper wings | Ø6×3 mm N45; Superimanes D-06-03 | Four body discs plus two in each selected upper wing: eight D6 discs per speaker. |

Both families retain zero structural load credit. Neither size is a substitute for the other's pocket.

The new D6 gate checks actual model extrusion around the entire cavity boundary at every intersecting layer. The body passes 244 checks (61 at each of four pockets); the four alternative wing files add 488 checks. Maximum sampled boundary gap across those jobs is below 0.059 mm. In the seated-disc working region, the wall must also form one connected footprint and overlap the prior model layer over at least 50% of its area. All 432 working-region checks pass: measured minimum overlap is 96.1% on the body and 92.6% across the wing alternatives. The width envelope accepts the inclined D6 design explicitly; it does not borrow the thinner D5 single-traversal rule. Negative controls reject a removed layer, a disconnected wall and inadequate prior-layer contact.

The test pair crops the actual body pocket and mating wing pocket, preserving their angles, retaining skins and loading geometry. It uses 100%/10% infill and independently audited toolpaths and pauses. It must still demonstrate continuous extrusion, full magnet seating, correct polarity, sound bonding after the pause, closed roofs and retention with the actual printer/material batch. The user's photographed missing material remains unexplained without the exact printed file and layer; a digital pass cannot establish what the printer deposited.

## Exceptions and remaining issues

- **Bottom support contact differs from top contact:** three layers but a 0.18 mm gap and 0.5 mm spacing. This inherited exception matters if support begins on a model surface. Top contact is dense with zero gap.
- **One rear tweeter insert-mouth boundary contact:** 0.0601 mm nominal vertical overlap, below the 0.08 mm half-layer limit. It lies outside the sliced bore interior and remains explicitly reported; all actual bore interiors are clear.
- **Auxiliary NL8/M2 holes have no dedicated blocker:** fresh floor, LM-top and regular-UM slices contain no support in these holes. Their orientation-dependent exception requires a new check after changes to placement or support scope. See [actual bead collision checks](auxiliary_bore_support.json).
- **Support scope differs by geometry:** regular wings are support-free and load only PETG-GF. Curved crescent wings use PETG-GF supports with PLA contact. Cap ceilings need the full-overhang exception.
- **No universal 100% insert/magnet modifier exists:** structural parts are already solid; local shells govern the lightweight tweeter and wing regions. The no-floor bridge-root solid modifier remains explicit, although currently redundant.
- **Physical fit is still untested:** insert installation, cap removal, PLA separation, LM assembly and D6 retention need physical checks. The user has confirmed the actual M3 stock: Hanglife HLTI-M3-001, M3 internal thread, Ø5 mm outside and 4 mm long. The shared printed bores are Ø4.6 × 4 mm; previous 3 mm insert-length references are superseded. [Stock and screw-engagement check](M3_stock_confirmation.json).

## Centralization, catalog and evidence

`print_policy.json` and `src/lx521_baffle/print_policy.py` now own role settings, material mapping and declared exceptions. The regular exporter, wing builder, crescent preparation and audits consume them. Geometry-specific blockers and the D5/D6 loading algorithms remain specialized; their ownership is listed in the generated policy reference.

`scripts/generate_print_policy_docs.py` derives the policy tables and process/hardware values from policy, profile and measurement data, and derives the print estimates from the current native manifest. The old [inspection report](../print_policy_audit_20260912/SUMMARY.md) remains historical.

The [D5 release catalog](../captive_magnet_release_catalog.json) now binds 46 existing local STLs and 82 stations, including the current UM meshes and blockers. Its provenance explicitly describes local artifact revalidation; no remote rebuild is claimed. The [regular shelf manifest](../../to_print/delivery_manifest.json) validates 42 choices and 76 projects. The [candidate delivery manifest](../../candidates/nd25fn4_crescent/delivery_manifest.json) binds seven direct STLs, M3 hardware, six native jobs and the D6 test report.

Validation: 70 focused Python tests passed; both regular PETG-GF wing plates were rebuilt and audited; all six final crescent jobs passed current source/geometry/material/toolpath checks; all four GLB review links loaded in the browser with matching served bytes. [Final print verification](verify_print.log), [regular shelf validation](delivery_validate.log), [candidate STL validation](candidate_catalog.log), [tests](profile_tests.log), [M3 measurements](../../candidates/nd25fn4_crescent/hardware_validation.json), [D6 test evidence](../../candidates/nd25fn4_crescent/print/qualification/qualification.json).
