# V1LF integral floor-stand analytical screen

This is a conservative closed-form net-section screen, not FEA or physical qualification. All reported stresses include the explicit 1.25 geometry/model factor.

| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | 1g deflection (mm) | Result |
|---|---:|---:|---:|---|
| Bambu PLA Tough+ | 3.05 / 1.97 / 1.18 | 4.22 / 2.73 / 1.64 | 1.18 | PASS (analytical) |
| Bambu PLA Basic | 4.39 / 2.78 / 1.67 | 6.09 / 3.85 / 2.31 | 1.05 | PASS (analytical) |
| Bambu PLA Lite | 2.69 / 1.73 / 1.04 | 3.73 / 2.40 / 1.44 | 1.40 | FAIL; provisional data |
| Bambu PLA Matte | 2.78 / 1.79 / 1.08 | 3.85 / 2.49 / 1.49 | 1.49 | PASS (analytical) |
| Bambu PLA Silk+ | 3.23 / 2.09 / 1.25 | 4.47 / 2.90 / 1.74 | 1.17 | PASS (analytical) |

## Bound production geometry

- `floor_stand/top_baffle_nd25fw4_v1lf_split.step` — SHA-256 `1739ebbfb87409cac10aed0761bbbd877ed277bae97d58dd4664b33d1fe22053`
- `floor_stand/top_baffle_nd25fw4_v1lf_lm_split.step` — SHA-256 `499c1b2711a7aef3162330557bafb7068aaff0618f7463e5f11ce2176e70ec8f`

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
