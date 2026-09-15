# Buried UM magnets — selection and limits

Use **eight Ø6×3 mm, axially magnetized neodymium disks per body plus upper-wing pair**: four in the shared UM/V4 body, two in each upper wing. The two existing LM magnets in the upper wings remain Ø5×2 mm at their original datums. The body and its matching wings must use the same new revision.

## Why the crescent uses a different size

The D6 choice was intended to recover holding margin across the crescent's thicker buried covers and curved mating interface. The current measured pocket-face separation is about 2.19–2.21 mm, compared with approximately 1.1–1.24 mm at the regular D5 interfaces. Keeping D5 at the LM preserves those existing pockets.

Checked again on 2026-09-12: Superimanes lists [D6×3 N45](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-6x3-mm) at approximately 0.99 kgf and [D5×2 N52](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-5x2-mm-n52) at 0.68 kgf. That is about **46% greater listed pull**, despite the lower grade; the D6 disc has **2.16 times the nominal magnetic volume**. Those catalog ratings do not predict force through this assembly's plastic separation.

The expected benefit is additional holding margin. It has not been measured in the assembled wings, and no test has established that D5 is inadequate. D6 also requires a second inventory item, larger pockets and its own print qualification. Standardizing on D5 would require resized crescent pockets and an assembly holding test; simply placing D5 discs in the current D6 cavities is not a compatible substitution. [Current D6 test pair](print/qualification/README.md).

Recommended part: [Superimanes D-06-03, Ø6×3 mm N45](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-6x3-mm). The vendor lists approximately 0.99 kg pull (9.71 N), axial magnetization, nickel coating, ±0.1 mm dimensional tolerance and an 80°C service limit. The listed entry price checked on 11 September 2026 is €0.37 each, minimum 10. No order has been placed.

| Option | Vendor pull rating | Assessment for this revision |
|---|---:|---|
| [Ø5×2 N52 / D-05-02-N52](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-5x2-mm-n52) | 0.68 kg | Stronger same-size option, but gives less margin for the new curved cover. Does not fill the new pockets. |
| [Ø6×3 N45 / D-06-03](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-6x3-mm) | 0.99 kg | Selected: larger magnetic volume while retaining room around the ducts and for continuous wing material. |
| [Ø8×3 N45 / D-08-03](https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-8x3-mm) | 1.5 kg | A possible future escalation; its larger loading chimney/roof requires a different surround and fresh fit checks. Does not fit these pockets. |

These are **vendor pull ratings, not measured holding forces through this printed assembly**. The selected grade and size are an engineering recommendation, not proof of retention. The mating plastic is curved; the pocket-face gap includes two covers plus the wing gap, with further allowance for a magnet seating against the back of its pocket. Exact measured cover, pocket-face separation and locations are recorded in [depth_magnet_validation.json](depth_magnet_validation.json). Magnet strength does not justify flattening the outside again.

This front-wide revision inclines the pocket axes with the actual wall normals, about 27° out of the XY plane, with surface contact datums at Z13.4 mm. The four stations are now 12°/168°/−12°/192°, away from the narrowed diagonal waists. Aligning the pockets to the slope brings their nominal face separation to about 2.2 mm. Check the exact exported geometry in the linked validation report; vendor pull ratings cannot be applied directly to this printed gap.

The new cavities are Ø6.20×3.10 mm and retain the original circular cradle, loading chimney and closure-roof topology at the new scale. The entire cradle is inclined with the magnet axis, so its roof angles in the print pose differ from the former horizontal-axis version. They are closed, fully buried pockets: insert magnets during a slicer-verified pause before the first closing layer. The original Ø5×2 coupon qualification does not qualify this enlarged pocket or its new print height. Check the real magnet dimensions, complete insertion, polarity, pocket closure and assembled wing holding before printing a full set. Layer numbers and pull-force acceptance cannot be established from the STL alone.
