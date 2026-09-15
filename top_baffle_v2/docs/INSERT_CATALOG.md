# Heat-set insert catalog

Base inventory compiled 2026-08-21; Dayton ND25FN-4 waveguide M3 addition 2026-09-12. Based on the source constants (every bore cited to its
constant block).  Counts are **per speaker**; double everything for a
stereo pair.  Candidate parts (BMR crescents 17/18, PTT1.3 dipole
crescent) are listed separately from the released inventory.

## The three insert SKUs

| # | Insert | Body Ø | Length | Bore recipe | Defined in |
|---|--------|--------|--------|-------------|-----------|
| 1 | **M5 brass heat-set** | 6.4 | 5.8 | stepped: Ø6.5 × 2.0 entry, then Ø6.4; total depth 6.2–6.8 | `base.py` `M5_INSERT_*`, `m5_insert_bore_cutter`; `flush.py` `LM_INSERT_L_MM` |
| 2 | **Hanglife M3 × 5 × 4, HLTI-M3-001** | **5.0** | **4.0** | Ø4.6 × 4.0 (+0.2 overshoot where blind) | `print_policy.json` `hardware_stock.M3_insert`; `base.py` `UM_PILOT_*`; `carriers.py` `JOINT_INSERT_*`, `TWEETER_JOINT_INSERT_*`; `b2_split.py` `SEAM_B_M3_*`; `floor.py` `NL8_*` |
| 3 | **M2 heat-set** | 3.2 | 2.5 | Ø3.2 × 2.5 (ties) / Ø3.2 × 4.0 (TEBM lands — same insert, deeper relief) | `carriers.py` `LM_UM_TIE_INSERT_*`, `T_UM_TIE_INSERT_*`; `tebm35c10_4_land.py` `M2_INSERT_*` |

The M3 row is the user's existing stock, confirmed by the [package drawing](../vendor/HANGLIFE/M3x5x4_reference.png). **Ø5 mm is the insert outside diameter; Ø4.6 mm is the printed heat-set pilot.** The previous table incorrectly put the pilot dimension in the insert-diameter column. M3×8 describes a screw with 8 mm length under its head; it does not change the insert length.

The M2 tie bores are cut for the L2.5 insert explicitly; the TEBM land
bores are 4.0 deep.  If M2 × 4 inserts are preferred for the TEBM lands
that would be a fourth SKU — the L2.5 insert works in both (the extra
bore depth is relief).

## Insert sites — released inventory

| Site | Host part | Profiles | State | Insert | Count | Constants |
|------|-----------|----------|-------|--------|-------|-----------|
| LM driver pilots (L22MG / W22EX001), PCD 209.5, 6 × 60° | LM carrier / shoulder pieces | stock, slim, Obi-Wan | both | M5 | **6** | `L22_PILOT_*` |
| Bridge mounts (40 × 50 pattern), rear-opening blind | baffle rear | stock, slim, Obi-Wan | no-floor only | M5 | **4** | `BRIDGE_INSERT_*`, `BRIDGE_HOLE_XY` |
| Floor anchors, foot underside, flush with the floor | Obi-Wan floor foot | Obi-Wan | floor only | M5 | **2** | `floor.py` `FLOOR_M5_ANCHOR_Z_MM` |
| UM driver pilots (10F/8424G00), PCD 89.5, 4 × 90° | UM carrier | stock, slim, Obi-Wan | both | M3 | **4** | `UM_PILOT_*` |
| NL8 receptacle flange, 29.2 square pattern | Obi-Wan foot underside bay | Obi-Wan | floor only | M3 | **4** | `floor.py` `NL8_SCREW_D_MM`, `NL8_INSERT_L_MM` |
| LM–UM joint ears (x ±32), rear-opening | UM carrier front ears | Obi-Wan | both | M3 | **2** | `JOINT_EAR_X`, `JOINT_INSERT_*` |
| UM–tweeter joint (x ±24), rear-opening | the crescent (whichever occupies the slot) | Obi-Wan | both | M3 | **2** | `TWEETER_JOINT_X`, `TWEETER_JOINT_INSERT_*` |
| B2 vase seam, seam-face blind | vase piece | stock (B2), slim (V1L) | both | M3 | **1** | `b2_split.py` `SEAM_B_M3_INSERT_*` |
| LM–UM M2 tie (x −17), LM seam face | LM carrier | Obi-Wan | both | M2 | **1** | `LM_UM_TIE_*` |
| T–UM M2 ties (x ±13) | UM/T seam | Obi-Wan | both | M2 | **2** | `T_UM_TIE_*` |

## Insert sites — candidate parts (print-to-qualify)

| Site | Host part | Insert | Count | Constants |
|------|-----------|--------|-------|-----------|
| Dayton ND25FN-4 waveguide retainers (two drivers, three screws each) | Fused Dayton ND25FN-4 waveguide UM/crescent | M3, Ø4.6 × 4 mm bore | 6 | `print_policy.json`; `v4_model.py`; exact final cylinders |
| TEBM35C10-4 mounts, PCD 48.26, 4 × 90° | BMR vases (2 drivers) | M2 | 8 | `tebm35c10_4_land.py` `TEBM_MOUNT_*`, `M2_INSERT_*` |
| TEBM35C10-4 mounts | BMR crescents 17/18 (2 drivers) | M2 | 8 | same |
| UM–tweeter joint receivers | BMR crescents 17/18, PTT1.3 dipole crescent | M3 | 2 | `TWEETER_JOINT_INSERT_*` (inherited contract) |

## Fastener bores that are deliberately NOT inserts

| Bore | Why | Where |
|------|-----|-------|
| ND25FW-4 clamp passages: 4 × Ø4.4 M4 through | the face-to-face tweeter pair clamps with through M4 machine screws | `base.py` (Ø4.4 "so an M4 machine screw passes an FDM part without reaming"); inherited by the released crescent |
| PTT1.3 rim screws: 12 × Ø2.5 M3 thread-forming pilots | at the vendor's Ø98 pattern a Ø4.6 insert pocket leaves 0.35 mm to the Ø92.7 skirt bore — unprintable (0.6-nozzle slicing opens the pocket sideways); the Ø2.5 pilot leaves 1.4 mm walls and ~5.8 mm M3 engagement | `tools/gen_ptt13_dipole_crescent.py` `DRIVER_PILOT_*` |
| Optional corner holes: 2 × Ø4.5 M5 thread-form | disabled by default | `base.py` `CORNER_HOLES_ENABLED` |

**Design rule this catalog encodes:** a heat-set insert pocket needs a
printable wall (≥ ~1.0 mm, ideally ≥ 1.6) on every side.  Where a driver
skirt or cavity leaves less, use thread-forming pilots (Ø2.5 for M3) or
through-clamping, never a thin-walled insert pocket.

## Per-speaker totals by configuration

| Configuration | M5 | M3 | M2 |
|---------------|----|----|----|
| Obi-Wan, floor stand, ND25 crescent | 8 (6 LM + 2 anchor) | 12 (4 UM + 4 NL8 + 2 ear + 2 T-joint) | 3 (1 tie + 2 T-ties) |
| Obi-Wan, no floor stand, ND25 crescent | 10 (6 LM + 4 bridge) | 8 (4 UM + 2 ear + 2 T-joint) | 3 |
| … with a BMR crescent (17/18) instead | +0 | +0 | +8 |
| … with the PTT1.3 dipole crescent instead | +0 | +0 (joint pair already counted; rim screws are pilots) | +0 |
| stock A/B1/B2 or slim V1-A/B1/V1L, bridge | 10 (6 LM + 4 bridge) | 5 (4 UM + 1 vase seam) | 0 (+8 if the BMR vase variant) |

Dayton ND25FN-4 waveguide uses 12 M3 insert sites on its shared body: 4 UM driver, 2 LM receivers, 6 tweeter retainers. The former separate UM/T joint is fused away. All use the existing Hanglife M3 × 5 × 4 insert stock and the shared Ø4.6 × 4 mm bore convention. Older 3 mm insert-length references are superseded. See [generated current policies](PRINT_POLICIES.md).
