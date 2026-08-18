# Obi-Wan integral floor-stand analytical screen

This is a conservative closed-form net-section screen, not FEA or physical qualification. All reported stresses include the explicit 1.25 geometry/model factor.

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g deflection (mm) | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 2.20 / 1.42 / 0.85 | 2.05 / 1.33 / 0.80 | 1.78 | FAIL |
| Bambu PLA Basic | 3.17 / 2.00 / 1.20 | 2.96 / 1.87 / 1.12 | 1.58 | PASS (analytical) |
| Bambu PLA Lite | 1.94 / 1.25 / 0.75 | 1.81 / 1.17 / 0.70 | 2.11 | FAIL; provisional data |
| Bambu PLA Matte | 2.00 / 1.29 / 0.78 | 1.87 / 1.21 / 0.72 | 2.25 | FAIL |
| Bambu PLA Silk+ | 2.33 / 1.51 / 0.91 | 2.17 / 1.41 / 0.84 | 1.76 | FAIL |

## Bound production geometry

- `build/floor_stand/obiwan_split.step` — SHA-256 `3518f93f8701cb0f256da1c870cd93722107c2b776100d18932d86ac626f28a3`
- `build/floor_stand/obiwan_lm_split.step` — SHA-256 `ca90f93bbfdb7b7af73acda7bd2ecc7e39e28e331dd7a6a2c79911920dff554f`

## Shoulder-to-LM-ring diagnostic

This deliberately conservative lower bound credits only the two uninterrupted printed outer-lip ligaments at the lower D190 tangent. It gives no credit to the seat membrane, integrated shoulder below the tangent, route covers, insert bosses, magnets or installed metal LM flange. It therefore does not redefine the root analytical result and is not a complete-assembly failure prediction.

| Material | 1g sustained SF | 3g transient SF | 5g transient SF | Lower-bound threshold |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 0.19 | 0.13 | 0.08 | BELOW |
| Bambu PLA Basic | 0.28 | 0.18 | 0.11 | BELOW |
| Bambu PLA Lite | 0.17 | 0.11 | 0.07 | BELOW; provisional data |
| Bambu PLA Matte | 0.18 | 0.11 | 0.07 | BELOW |
| Bambu PLA Silk+ | 0.20 | 0.13 | 0.08 | BELOW |

The lip-only lower bound is below the project thresholds. The installed LM flange and fasteners are therefore required parallel load paths, and the documented assembled proof/creep gate remains mandatory.

## Governing limitations

- Exact nominal root section: 734.8 mm²; governing section modulus 2319.8 mm³ after subtracting D9, D8.2 and D6 lumens.
- The section result is valid only with the required 100% local solid modifier through the complete stem/root; sparse infill gets no structural credit.
- Free-standing lateral tip threshold: 0.139 g. This is a stability limit, not a PLA strength limit.
- The optional hidden split key receives 0 N structural credit; the installed LM driver flange must bridge the seam.
- Every material/process remains **PENDING** until the documented proof and creep tests pass.
