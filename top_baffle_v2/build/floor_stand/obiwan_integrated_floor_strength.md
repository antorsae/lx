# Obi-Wan integral floor-stand analytical screen

This is a conservative closed-form net-section screen, not FEA or physical qualification. All reported stresses include the explicit 1.25 geometry/model factor.

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g deflection (mm) | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 3.80 / 2.46 / 1.48 | 1.30 | PASS (analytical) |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 5.48 / 3.47 / 2.08 | 1.15 | PASS (analytical) |
| Bambu PLA Lite | 2.69 / 1.73 / 1.04 | 3.35 / 2.16 / 1.30 | 1.54 | FAIL; provisional data |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.47 / 2.24 / 1.34 | 1.65 | PASS (analytical) |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.03 / 2.61 / 1.57 | 1.29 | PASS (analytical) |

## Bound production geometry

- `build/floor_stand/obiwan_split.step` — SHA-256 `bf6cdf40534b6f766ea02a6d12927aba74f8b6b5b51bfc9aa456d291081a61bb`
- `build/floor_stand/obiwan_lm_split.step` — SHA-256 `3e68483a72aace5a06732534bc2040b2972aa8681b2c2e308f425ba403457796`

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

- Exact nominal root section: 1026.5 mm²; governing section modulus 3214.4 mm³ after subtracting D9, D8.2 and D6 lumens.
- The section result is valid only with the required 100% local solid modifier through the complete stem/root; sparse infill gets no structural credit.
- Free-standing lateral tip threshold: 0.139 g. This is a stability limit, not a PLA strength limit.
- The optional hidden split key receives 0 N structural credit; the installed LM driver flange must bridge the seam.
- Every material/process remains **PENDING** until the documented proof and creep tests pass.
