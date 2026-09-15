# Print-policy audit — Obi-Wan and ND25FN-4 crescent

Audit date: 2026-09-12. Scope: current delivered 0.6 mm High Flow Tinmorry PETG-GF/PLA projects, their source policies, and actual sliced extrusion. This is an inspection report. No production STL, preset, or print project was changed; no printer was contacted.

**The supplied crescent body G-code contains retaining walls at all four magnet pockets. The current regular Obi-Wan projects specify three PLA interface layers, and fresh review slices reproduce three beneath their main supported surfaces.** The user's photograph documents a physical surface concern, but the actual file/version and printing layer have not been identified, so the cause of the photographed missing material remains unconfirmed.

## What was checked

- All 14 native core/candidate projects currently in `to_print/obiwan/3mf_06hf_petg-gf_pla/` and `candidates/nd25fn4_crescent/print/`: project settings, per-object/per-part overrides, and hashes. [Complete inventory](delivered_project_settings.json).
- Additionally inspected both regular PETG-GF wing combo projects, including their conflicting global/object infill settings.
- Exact G-code embedded in the delivered crescent body and accessories. The body evidence below matches the embedded G-code hash, not just the working STL.
- Fresh slices of the current standalone Obi-Wan floor LM bottom and UM GUI projects, using Bambu Studio **02.07.01.62** and the same frozen Tinmorry/PLA filament profiles as the crescent.
- Those regular GUI projects contain no G-code. The original floor project crashed in the CLI. The review copies completed after normalizing the two-material mapping vectors, nozzle statistics, colour arrays and purge-matrix dimensions. Geometry, layer/wall/infill/support settings and High Flow selection were retained. These reproductions are **not evidence of exactly what the user printed after GUI edits**. Every normalization is recorded in [floor](floor_normalization.json) and [UM](um_normalization.json) records.

## A. How the multi-material supports work

The project policy assigns **filament 1 / G-code T0 to Tinmorry PETG-GF** for the model and support structure, and **filament 2 / T1 to PLA** for the contact interfaces. These are two filament selections feeding the same physical nozzle. PLA is not an infill material for the structural part.

| Setting | Current regular Obi-Wan and crescent projects |
|---|---|
| Process | 0.6 mm High Flow; 0.16 mm layers; 0.20 mm first layer |
| Support form | Normal / snug; build plate only; 30° threshold; 0.7 mm XY clearance |
| Top interface | **3 layers**, rectilinear, **0 mm spacing**, **0 mm top Z gap** |
| Bottom interface | **3 layers**, but **0.5 mm spacing and 0.18 mm bottom Z gap** are inherited |
| Independent support layer height | Off |
| Temperatures | PETG-GF 260°C; PLA 220°C; project bed policy uses the first filament's textured-PEI 80°C setting |
| Model flushing | Into model and infill disabled |
| Support flushing | **Regular Obi-Wan permits it; crescent disables it** |
| Purge | 280 mm³ in each material-change direction; prime tower enabled |

The bottom-interface fields matter when support starts on a model surface. Most jobs here restrict support to the plate, so the top interface is the main relevant contact. The settings should not be described as universally “zero Z gap” or “dense interfaces on both sides.”

**The floor is not currently configured for one PLA interface layer.** Actual review toolpaths include three consecutive PLA layers in the same contact area:

| Job / contact area | PLA interface Z heights (mm) | Evidence |
|---|---|---|
| Standalone floor LM bottom, main supported region | **5.64, 5.80, 5.96** | Same-area overlap 3,838.68 mm² across all three layers |
| Standalone regular UM, main supported region | **3.72, 3.88, 4.04** | Same-area overlap 2,012.62 mm² across all three layers |
| Current supplied caps, flat cavity ceilings | **8.68, 8.84, 9.00** | Existing exact-job cap ceiling audit; an additional partial interface occurs at 8.52 around earlier features |

[Actual three-layer PLA views](obiwan_PLA_three_layers.png) · [Coverage witnesses](support_layer_witnesses.json) · [Material extrusion counts by layer](support_materials.json).

Curved supports produce PLA at many different overall plate heights; “three interface layers” refers to contact stacks, not three PLA layers in the entire job. Very shallow/local regions can have a truncated stack. The single layer the user saw cannot be attributed to a project-wide one-layer rule from the files checked here. It needs the actual GUI-exported slice or its settings/layer location to resolve.

In the reproduced floor/UM and the supplied body/caps, the material scan found model/support extrusion on T0 and interface extrusion on T1, with no other material assignments for those features. This checks the emitted instructions, not which physical spool was loaded into an AMS slot.

## The linked Bambu Studio issue and workaround

The original [issue #11893](https://github.com/bambulab/BambuStudio/issues/11893) reported a CLI crash associated with two filaments, supports and a large model. The [linked comment by Snail3D](https://github.com/bambulab/BambuStudio/issues/11893#issuecomment-5575014503) later attributes a crash path to a single-nozzle preset listing both Standard and High Flow variants. It suggests collapsing the variant and correcting flattened-profile compatibility metadata. This is a software configuration issue; the comment does not prescribe one PLA interface layer.

The current upstream [`support_different_extruders()` implementation](https://github.com/bambulab/BambuStudio/blob/master/src/libslic3r/PrintConfig.cpp) splits comma-separated variant names and returns true when their distinct count exceeds one. This supports the comment's explanation of branch selection, without proving every crash has that cause.

Both current project families still carry:

```json
"extruder_variant_list": ["Direct Drive Standard,Direct Drive High Flow"]
```

The regular GUI builder fills `filament_map` and selects `High Flow`, but leaves other mapping vectors incomplete. The crescent's separate `settings()` function fills those vectors and uses an explicit 2×2 purge matrix. It does **not** implement the comment's single-variant normalization.

I also tested a review-only UM copy with only `extruder_variant_list` and `printer_extruder_variant` collapsed to **Direct Drive High Flow**, preserving the requested hardware. That run exited successfully but emitted **zero PLA-interface extrusion**, while its metadata still claimed PLA and three interface layers. Its mapping vectors remained incomplete and the log reported missing nozzle information. The complete-mapping reproduction emitted **2,296 PLA interface moves**. [Variant-only test](um_variant_materials.json) · [Metadata changes](um_variant_only_normalization.json).

Therefore the two-field change alone is insufficient for this native-project path. This does not invalidate the comment's complete flattened-profile workflow. A shared adapter needs to normalize variant selection **and** every filament/nozzle mapping, then verify actual T0/T1 extrusion. Do not copy its Standard-nozzle example verbatim over the requested High Flow hardware.

## B. Magnet walls

| Item | Regular Obi-Wan | Current fused crescent |
|---|---|---|
| Magnet | Ø5×2 mm | Four Ø6×3 in body; two Ø6×3 per upper wing; the lower LM wing pocket retains Ø5×2 |
| Cavity | Ø5.20×2.10 mm | Ø6.20×3.10 mm |
| Retaining geometry | Shared circular cradle, loading chimney, 45° closing roof; nominal 0.52 mm axial skins | Scaled/transformed cavity under the inclined organic exterior; nominal minimum cover 0.82 mm; print-only loading-roof relief |
| Measured crescent cover | Not applicable | Minimum original CAD cover 0.809 mm; print-relieved cover minimum approximately 0.786 mm |
| Closing pause | Normally before Z5.96 for regular stations | Body before **Z9.80**; upper wings retain Z5.96 for LM and Z9.80 for new UM magnets |
| Wall generation | Arachne, one bounded retaining traversal where the skin is thin; ordinary six-wall setting cannot create six paths in that skin | Arachne; some exposed pocket sections become **one 1.06–1.09 mm wide bead** |
| Validation | Shared aperture, bounded bead/traversal, connectivity and roof-progression checks | Separate CAD-cover and inclined loading-sweep checks; no shared D5 bounded-width qualification for the D6 design |

**The magnet cavities intentionally remain empty of support and infill for loading. Their retaining walls are model plastic and must remain present.** Being invisible from outside after closure is intended; an actual breakthrough through a completed retaining wall is not.

For the exact supplied body G-code, I checked **61 cavity-intersecting layer midplanes per pocket: 244 section checks total**, sampling the entire cavity boundary at at most 0.06 mm intervals. Every boundary sample is within 0.059 mm of a deposited model-bead footprint. Four representative layers are shown in [the body toolpath image](crescent_magnet_toolpaths.png); all measurements are in [the full-height check](crescent_magnet_fullheight.json).

This finds **no omitted retaining wall in that supplied slice**. It does not prove the actual extruder deposited those beads correctly, that they adhered, or that the currently printing file is this version. The wide variable bead is an explicit difference from regular Obi-Wan's narrower retained-wall contract. It is a process feature to qualify, not a confirmed explanation of the photograph. Increasing infill will not address a missing skin: the UM is already 100%, and the affected skin is a perimeter feature.

[Regular UM current-mesh toolpaths](regular_UM_magnet_toolpaths.png) provide the direct comparison: sampled nearby outer-wall widths are **0.520–0.635 mm**, versus the approximately 1.06–1.09 mm single-bead sections in the crescent. This independent inspection is separate from the failed provenance gate below.

The D6 pocket needs its own inclined-pocket physical coupon in the same material/profile, including pause, insertion, resume and closure. The older D5 coupon does not qualify its cover, loading direction or larger bead. If a completed pocket side is visibly open, inspect that failure before burying magnets and resuming; do not treat it as the intended loading opening.

## Shared policy, implementation and exceptions

| Element | Current behaviour | Central code / local exceptions |
|---|---|---|
| Printer/material recipe | 0.6 HF; 6 walls; 10 top/5 bottom; Arachne; outer width 0.52, inner 0.62; 85% minimum bead; XY hole compensation 0 | [Central profile](../../captive_magnet_slicing_profile_petg_gf_06hf.json); [profile resolution and artifact overrides](../../scripts/release_validation.py). Crescent instead copies a delivered regular UM project plus frozen resolved profiles in [prepare_print.py](../../candidates/nd25fn4_crescent/prepare_print.py). |
| Structural infill | LM bottoms and regular UM: **100% zig-zag**. Standalone LM top: **30% gyroid**. Combined core plates: **100% globally**, including their LM top and tweeter components. | Central per-artifact JSON rules plus [combo builder](../../scripts/build_obiwan_combo_plate.py); not one universal value per named part across plate layouts. |
| Light-part infill | Regular standalone tweeter crescent/lid: 30% gyroid. Current PETG-GF wing combos conflict: **10% globally but 30% at object level**, support-free. The object override takes precedence when reslicing. New crescent: UM 100%; tweeters **15% gyroid** above installed Y421; caps/retainers 15%; new upper wings 30%. | The separate [wing builder](../../scripts/build_obiwan_wing_plate.py) declares `WING_SPARSE_INFILL_PERCENT=10` but still writes a hardcoded 30% assembled-object override. [Both delivered wing projects](regular_wing_settings.json) contain that conflict; the global 10% label alone is not a reliable effective-density description. Candidate split is a local modifier in `prepare_print.py`; the cut boundary does not change walls or top/bottom shells. |
| Local solid reinforcement | A 100% no-floor bridge-root modifier is retained, although the current entire LM bottom is already 100%. | [Bridge-root modifier builder](../../scripts/build_obiwan_bridge_root_modifier.py). There is **no universal automatic 100% modifier around every insert, screw or magnet**. Ordinary bores rely on surrounding shells and their region's infill. |
| Heat-set inserts | LM M5 stepped bores; UM and joint M3 Ø4.6 bores. Plain pilot/clearance geometry receives metal hardware; these are not generally printed helical threads. | Shared stepped M5 cutter in [base.py](../../src/lx521_baffle/base.py); seating dimensions in [flush.py](../../src/lx521_baffle/flush.py); joints/ties in [carriers.py](../../src/lx521_baffle/obiwan/carriers.py); floor/NL8 sites in [floor.py](../../src/lx521_baffle/obiwan/floor.py). Several size/location constants remain distributed. |
| New tweeter fasteners | Six **M2×8 screws with M2 inserts 4 mm long**, Ø3.2 pilots; stepped narrow blind-tip relief. Caps are O-ring friction covers; retaining rings clamp the drivers. | [Imported V4 mechanical model](../../design_inputs/MU10_ND25FN_V4_Retained/source/mechanical_v4.py). This is a distinct insert-length exception to the regular M2×2.5 tie insert. The global insert catalog has not incorporated this candidate. |
| Other screw exceptions | Regular ND25FW clamps use M4 clearance passages; some other candidates use thread-forming pilots rather than inserts. | [Insert catalog](../../docs/INSERT_CATALOG.md), with candidate-specific implementations. A screw-size label alone does not establish bore type. |
| Insert/joint support exclusion | Keep support out of full blind-bores, including their mouths. “Build plate only” alone does not protect a front-down bore. | [obiwan_support_blocker.py](../../scripts/obiwan_support_blocker.py) covers driver, joint and other owned fastener passages. Candidate independently enumerates 12 insert sites / 18 axial intervals, copying the 0.25+0.02 mm margin convention. Its bore/duct G-code audit is local. |
| Magnet geometry | Shared D5 contract and reusable cavity generators. Empty captive cavity; solid skins; insert at pause and close roof. | [magnet_contract.py](../../src/lx521_baffle/magnet_contract.py), [magnets.py](../../src/lx521_baffle/magnets.py), [Obi-Wan sites](../../src/lx521_baffle/obiwan/magnets.py). Candidate D6 placement/scaling/burial is in [v4_model.py](../../candidates/nd25fn4_crescent/v4_model.py), with separate [print loading relief](../../candidates/nd25fn4_crescent/prepare_magnet_loading.py). |
| Magnet pause and retaining checks | Pause location must come from the closing toolpaths; raise to Z250, pause, then restore. | Regular [gcode_analysis.py](../../scripts/gcode_analysis.py) and [artifact_emit.py](../../scripts/artifact_emit.py); candidate [print_magnets.py](../../candidates/nd25fn4_crescent/print_magnets.py) uses inclined insertion sweeps. The latter lacks the regular bounded-retaining-bead gate. |
| Support scope | Regular LM halves/UM enable support with critical-regions-only and remove-small-overhangs on. Regular standalone crescent/lid and standard wing jobs are support-free. | Central artifact overrides and regular export code. Candidate body inherits those filters; candidate caps and curved upper wings explicitly turn both filters **off** and enable support. Retainers happen to receive no generated support on the mixed accessories plate. |
| Cable/support exclusions | Prevent unremovable support in buried cable passages and magnet cavities. | Regular route-aware blocker ownership is central; candidate routes, lumen meshes and bore checks are separate implementations in `prepare_print.py` / [audit_print.py](../../candidates/nd25fn4_crescent/audit_print.py). Candidate removable interface contact at an insert mouth is recorded separately from support inside the bore. |
| Interface coverage validation | Configured PLA is checked against actual material extrusion. | Shared [audit_gui_slice.py](../../scripts/audit_gui_slice.py); candidate imports its material scanner. Exact cap-ceiling three-layer coverage is a newer **caps-only** check in [cap_support_check.py](../../candidates/nd25fn4_crescent/cap_support_check.py), not a project-wide contact-surface requirement. |
| Delivery and provenance | Regular PLA-interface files are unsliced GUI projects; candidate files contain prepared G-code. | Shared delivery contract/geometry/source hashes versus separate candidate manifests, finalizer and verifier. These are two related pipelines, not one. |

## Confirmed gaps and recommended priorities

1. **Close the magnet qualification gap.** The supplied D6 paths are present, but the wider single-bead sections and inclined loading/closure do not share the regular coupon's qualification. Add per-layer retaining-bead width, connectivity and interlayer-overlap checks to the D6 pipeline, then qualify the actual pocket physically. Keep the current photograph as unresolved physical evidence until the printed file and layer are known.
2. **Centralize two-material/nozzle normalization and material assertions.** Regular and candidate exporters currently handle mapping differently. A process metadata claim is insufficient: the variant-only test emitted no PLA despite claiming it. Preserve High Flow, resolve all per-filament arrays, and reject missing T1 interfaces.
3. **Refresh the regular release catalog and blocker provenance.** `normalize_catalog()` currently fails with `floor_stand:Obi-Wan:obiwan_core_2_of_2_um_carrier: support-blocker binding hashes differ from the release meshes`. The current UM STL hash is `8ed399801dd38e770114223e7c6d30e5ea9e3e312483d74522e74bf6abd2d2bc`; its older magnet catalog records a different mesh. Geometry-only GUI checks do not repair this. [Failure record](regular_um_magnet_checks.json).
4. **Make support and infill exceptions declarative.** The LM top changes from 30% alone to 100% in a combo; regular PETG-GF wing combos conflict between global 10% and per-object 30%, while new crescent upper wings explicitly use 30%; support filters differ for new caps/wings; flushing into support differs between families; the bottom-interface defaults differ from the top. These should be explicit rows in one policy source rather than inherited surprises.
5. **Generate the print and hardware documentation from current policy.** `docs/PRINTING.md` still contains older 8-wall / 6-top-shell / 40% passages, while delivered projects use 6 walls / 10 top shells and the densities above. It also retains an overly broad “PLA interface blocked upstream” statement despite the current working candidate path. The insert catalog omits the new M2×4-length candidate insert, and an older magnet catalog description still says 0.45 mm despite numeric 0.52 mm fields.

The independent images and JSON records here document the audit findings. The review reproduction slices are diagnostic artifacts, not newly promoted print jobs.
