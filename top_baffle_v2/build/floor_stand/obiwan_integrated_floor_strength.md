# Obi-Wan integral floor-stand analytical screen

This is a conservative closed-form net-section screen, not FEA or physical qualification. All reported stresses include the explicit 1.25 geometry/model factor.

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g deflection (mm) | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 2.99 / 1.94 / 1.16 | 3.68 / 2.38 / 1.43 | 1.32 | PASS (analytical) |
| Bambu PLA Basic | 4.32 / 2.73 / 1.64 | 5.30 / 3.35 / 2.01 | 1.17 | PASS (analytical) |
| Bambu PLA Lite | 2.64 / 1.70 / 1.02 | 3.25 / 2.09 / 1.25 | 1.57 | FAIL; provisional data |
| Bambu PLA Matte | 2.73 / 1.76 / 1.06 | 3.35 / 2.16 / 1.30 | 1.67 | PASS (analytical) |
| Bambu PLA Silk+ | 3.17 / 2.06 / 1.23 | 3.89 / 2.52 / 1.51 | 1.31 | PASS (analytical) |

## Bound production geometry

- `build/floor_stand/obiwan_split.step` — SHA-256 `d4021eed0fa8df19e153b5981c661c7d12c7e2595e58e3f61ee40d2b37240ac9`
- `build/floor_stand/obiwan_lm_split.step` — SHA-256 `1309ad64d132b7b7f4d1ebe84e1c69f8d2dbeaa932be2e7e59f02c0c70b7a6bc`

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

- Exact nominal root section: 1008.2 mm²; governing section modulus 3158.4 mm³ after subtracting D9, D8.2 and D6 lumens.
- The section result is valid only with the required 100% local solid modifier through the complete stem/root; sparse infill gets no structural credit.
- Free-standing lateral tip threshold: 0.139 g. This is a stability limit, not a PLA strength limit.
- The optional hidden split key receives 0 N structural credit; the installed LM driver flange must bridge the seam.
- Every material/process remains **PENDING** until the documented proof and creep tests pass.
