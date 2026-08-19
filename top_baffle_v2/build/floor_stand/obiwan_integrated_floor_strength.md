# Obi-Wan integral floor-stand analytical screen

This is a conservative closed-form net-section screen, not FEA or physical qualification. All reported stresses include the explicit 1.25 geometry/model factor.

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g deflection (mm) | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 2.62 / 1.69 / 1.02 | 2.85 / 1.85 / 1.11 | 1.50 | FAIL |
| Bambu PLA Basic | 3.77 / 2.39 / 1.43 | 4.11 / 2.60 / 1.56 | 1.34 | PASS (analytical) |
| Bambu PLA Lite | 2.31 / 1.49 / 0.89 | 2.52 / 1.62 / 0.97 | 1.78 | FAIL; provisional data |
| Bambu PLA Matte | 2.39 / 1.54 / 0.92 | 2.60 / 1.68 / 1.01 | 1.91 | FAIL |
| Bambu PLA Silk+ | 2.77 / 1.80 / 1.08 | 3.02 / 1.96 / 1.18 | 1.49 | PASS (analytical) |

## Bound production geometry

- `build/floor_stand/obiwan_split.step` — SHA-256 `53f3c9f7fc790db94a66bc2b310897362903a9a4a4bc1f3929bf55103280942d`
- `build/floor_stand/obiwan_lm_split.step` — SHA-256 `02570b3178e97ded355f1b73d7040320079ba49ca2ecab6d6be9773962c89916`

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

- Exact nominal root section: 878.3 mm²; governing section modulus 2760.2 mm³ after subtracting D9, D8.2 and D6 lumens.
- The section result is valid only with the required 100% local solid modifier through the complete stem/root; sparse infill gets no structural credit.
- Free-standing lateral tip threshold: 0.139 g. This is a stability limit, not a PLA strength limit.
- The optional hidden split key receives 0 N structural credit; the installed LM driver flange must bridge the seam.
- Every material/process remains **PENDING** until the documented proof and creep tests pass.
