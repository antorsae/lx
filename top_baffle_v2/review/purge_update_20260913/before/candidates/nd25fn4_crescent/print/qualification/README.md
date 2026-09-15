# D6 body / wing test pair

Print `D6_body_wing_fit_06HF_PETG_GF_PLA.gcode.3mf` with the same 0.6 mm High Flow nozzle and materials as the body. This contains one cropped body station (100% infill) and its corresponding wing station (10%). Both pockets, side skins and inclined loading directions come from the actual print STLs. Cropping and bed translation make this a separate print; its toolpaths and pauses are checked independently.

Use two Ø6×3 mm N45 discs. At each embedded pause, check that the retaining wall is continuous, seat the indicated disc fully, and check the pole orientation against its mate before continuing. After cooling, confirm the roof is closed, the wall and resumed layers are bonded, the discs cannot escape, and the two original mating faces fit without rocking. Record filament lot, drying, printer/nozzle and observations in `physical_result.json` alongside this file. A split skin, loose disc, delamination or blocked insertion is a failure. Do not treat the digital pass as a measured holding force.

No printer operation has been initiated. Physical qualification remains pending your print.
