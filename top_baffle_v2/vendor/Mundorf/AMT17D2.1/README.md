# Mundorf hifiAMT17D2.1 — dipole AMT, 4 Ω (no faceplate, free-standing)

- **Datasheet:** [`pdf_Mundorf_AMT17D2.1_1.pdf`](pdf_Mundorf_AMT17D2.1_1.pdf), from
  [doc.soundimports.nl](https://doc.soundimports.nl/pdf/brands/Mundorf/AMT17D2.1/pdf_Mundorf_AMT17D2.1_1.pdf).
- **Dimensioned drawing:** [`AMT17D2.1-drawing.jpg`](AMT17D2.1-drawing.jpg) (dealer image: depth 10, hole pitch
  47 × 69) and a front photo, [`AMT17D2.1-front.jpg`](AMT17D2.1-front.jpg).
- **Successor, for reference:** [`mundorf-amt17d2.2-specification.pdf`](mundorf-amt17d2.2-specification.pdf).
- **Model:** Mundorf publishes no CAD. [`amt17d2_1.py`](amt17d2_1.py) builds an envelope model from the drawings
  and photos: `AMT17D2.1.step`, `AMT17D2.1.stl`, and `AMT17D2.1.json` with every value and where it came from.

| Parameter | Value |
|---|---|
| Impedance: nominal / Zo / Zmin / Re | 4 Ω / 4.2 Ω / 3.9 Ω / 3.9 Ω |
| Sensitivity | 91 dB @ 2.83 V |
| Response (−6 dB) | 1.5 – 35 kHz |
| Power | 60 W |
| Fs | 2000 Hz |
| Recommended crossover | 5 kHz (6 dB/oct), 4 kHz (12 dB/oct) |
| Horizontal dispersion (15 kHz, −6 dB) | 50° |
| Diaphragm area | 1260 mm² |
| Weight | 160 g |

## Construction

- **Front plate:** flat and black, 55 × 60 mm; it is the slotted grille itself.
- **Ears:** four, two at the top and two at the bottom, with Ø4.5 holes on a 47 × 69 mm pattern (77 mm over the ears).
- **Body:** the AMT motor sits behind the plate. Both faces radiate (dipole).
- **Leads:** flying leads (with grommet) leave one short edge near a corner; there are no terminals.

From the drawings:

- **Plate:** 55 × 60 mm.
- **Over the ears:** 77 mm.
- **Holes:** Ø4.5 on a 47 × 69 mm pattern.
- **Depth:** 10 mm, front face to rear.

Estimated from the photos (measure a physical driver before release):

- **Plate:** 1.5 mm thick.
- **Ears:** 8 mm wide.
- **Body:** taken as the plate footprint (conservative).
- **Lead exit:** x = 14 mm on the −y edge, mid-depth.

The **AMT17D2.2** successor keeps the ear plate and hole pattern, but it is 12 mm deep and 60 mm wide behind the plate.
It does **not** fit this envelope.

Model frame: x across the plate, y along it (ears at ±y), z normal to it. z = 0 is the plate's front face; the body
lies behind it, to z = −10. The leads leave the −y edge.
