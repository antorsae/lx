# Purifi PTT1.3T04-HAG-01, vendor 3D CAD

`PTT1.3T04-HAG-01 - 3D CAD.stp` is Purifi Audio's own STEP model of the
PTT1.3T04-HAG-01 1.3" dome tweeter (HAG waveguide variant), added here on
2026-08-21 (downloaded from the vendor's published product CAD on
2026-08-18).

- sha256 `3aadc3cabe257e186ca4bdbd9e83985a61fcd77c460513b71387a89ba75386ea`

It is kept as the authoritative mechanical reference for the tweeter
position: flange diameter and thickness, mounting-hole pattern, waveguide
profile, rear-can depth and any protrusions are read from this model, not
from datasheet drawings, when validating the printed baffle's recess,
pilot holes and rear clearance.

## Data sheet

`PTT1.3T04-HAG-01 - Data Sheet.pdf` is Purifi's data sheet (rev 1.00,
Jan 2026), added 2026-08-21 (retrieved 2026-08-17).

- sha256 `66e4c1b48011f366229c163066292f43b411f9ac8a54d4f8a95a32fd65db5918`

Key mechanicals (Table 4): faceplate 104, mounting hole pattern D98 with
D3.5 holes (six, verified on the STEP at 30 deg + k*60 from the terminal
azimuth), recommended cutout D93/81, outer flange 4.0 thick, build-in
depth 39, magnet D80 (the "80.0" in the drawing is the magnet, not the
rear can -- the STEP's can measures D76.6), 1.5 gasket not shown.
Electrical: 4 ohm nominal (3.5 DC), fs 680 Hz, 95.3 dB@2.83V/1m.
