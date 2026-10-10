# GRS PT2522-4 — 3-1/2" planar tweeter, 4 Ω (open-back dipole)

- **Datasheet:** [`272-122--grs-pt2522-4-spec-sheet.pdf`](272-122--grs-pt2522-4-spec-sheet.pdf), from
  [Parts Express 272-122](https://www.parts-express.com/GRS-PT2522-4-3-1-2-Planar-Tweeter-4-Ohm-272-122).
- **Model:** GRS publishes no CAD. [`pt2522_4.py`](pt2522_4.py) builds an envelope model from the drawing and
  the product photos: `PT2522-4.step`, `PT2522-4.stl`, and `PT2522-4.json` with every value and where it came from.

| Parameter | Value |
|---|---|
| Impedance / Re | 4 Ω / 3.4 Ω |
| Le | 0.01 mH @ 10 kHz |
| Sd | 20.2 cm² |
| Sensitivity | 93 dB @ 2.83 V / 1 m |
| Power | 20 W RMS |
| Usable range | 800 Hz – 20 kHz |
| Qms / Qts | 4.60 / 4.46 |

## Construction

The driver is two-sided: a thin steel flange carries a grille housing on both faces.

- **Thin face:** grille only, with screw heads.
- **Thick face:** the motor housing, a black terminal insulator block below the grille, and two L-shaped
  solder lugs. The lugs stand proud of the flange, further than the housing.
- **Screws:** six assembly screws on the flange (two top, two mid-sides, two bottom), with heads on both faces.
- **Mounting:** the four corner-notch holes.

From the drawing:

- **Flange:** 67.5 × 89.3 mm.
- **Mounting holes:** four Ø3.5 mm holes in R5.5 corner notches, on a 62.0 × 79.0 mm pattern.
- **Grille housing:** 57.7 × 65.5 mm.
- **Overall depth:** 9.4 mm, housing to housing.
- **Solder lugs:** 5.2 mm wide.

Estimated from the photos, scaled from the plate size (measure a physical driver before release):

- **Flange:** 1.0 mm thick.
- **Housings:** 2.4 mm proud on the thin face and 6.0 mm on the thick face (2.4 + 1.0 + 6.0 = 9.4).
- **Housing position:** centre 2.35 mm above the plate centre, corners R4.
- **Terminal block:** about 40 × 9 × 3 mm, centred 35.35 mm below the plate centre.
- **Solder lugs:** at x ±16.6 mm, standing **13.5 mm** proud of the flange.
- **Screw heads:** Ø5 × 1.8 mm; positions in `PT2522-4.json`.

Model frame: x across, y along the plate (terminals at −y), z normal to it. z = 0 is the thin-face side of
the flange; the thin face is +z and the thick face with the lugs is −z. The Obiwan candidates mount the thin face
toward the listener, so the flange sits flush and the housing and lugs sit in an open pocket behind it.
