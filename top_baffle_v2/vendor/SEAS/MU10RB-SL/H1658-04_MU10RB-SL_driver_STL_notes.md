# H1658-04 / MU10RB-SL STL notes

Generated from `H1658-04_MU10RB-SL_Datasheet.pdf`.

Coordinate system: millimeters. Driver axis is `Y`; mounting plane is at `Y=0`;
front proud geometry extends to `Y=+5.4`; rear/motor geometry
extends to `Y=-38.2`. Total front-to-rear envelope is
43.6 mm.

Dimensions encoded from the drawing:
- Front flange/frame outside diameter: 98.0 mm.
- Front proud section: 5.4 +/- 0.2 mm.
- Total depth: 43.6 mm.
- Rear basket/frame envelope diameter: 80.0 +/- 0.4 mm.
- Motor/rear cylinder diameter: 60.0 mm.
- Mounting holes per drawing: 4 x diameter 5.0 mm with diameter 8.3 mm pockets on 89.0 mm PCD.
- Effective piston area from datasheet: 38.5 cm2, equivalent diameter 70.0 mm.

Solid-cap policy:
- The front diaphragm/dust cap is a solid surface with no center hole or phase-plug opening.
- The solver-facing generated STL also omits screw-hole cutouts (`INCLUDE_FLANGE_HOLES=False`) so this asset is a solid acoustic obstruction rather than a screw-hole leakage model.

Approximations:
- This is not manufacturer CAD.
- Cone, surround, dust cap, basket supports, and motor step geometry are simplified from the side-view envelope and public photos.
- Basket strut width is estimated at 12 degrees with four tapered supports.
- The screw-hole dimensions remain documented above but are not cut into this solid-cap STL.

Output:
- `H1658-04_MU10RB-SL_driver.stl`
- Facets: 1616
