# TINMORRY PETG-GF, vendor profile

`PETG-GF(P2S-Bambu-TINMORRY).json` is TINMORRY's own Bambu Studio filament
profile for the P2S, retrieved verbatim on 2026-08-15 from the vendor's
repository:

- <https://github.com/TINMORRY/TINMORRY-filament-profile-for-Bambu-printers/blob/main/PETG-GF(P2S-Bambu-TINMORRY).json>
- sha256 `90f9bb69f3f043996ec0bdc443d6d4165d425d070232817a173efea9acb381b1`

It is kept here because it is the only authoritative statement of this
material's print parameters, and because reading it settles a question the
product page cannot: TINMORRY's store lists 240-270 C, 65-75 C bed, and
"< 250 mm/s", but no volumetric ceiling at all.

## What it actually specifies

    inherits                          Generic PETG-CF @BBL P2S
    filament_max_volumetric_speed     ["12",   "nil"]
    nozzle_temperature                ["260",  "nil"]
    nozzle_temperature_initial_layer  ["260",  "nil"]
    filament_flow_ratio               ["0.93", "nil"]
    textured_plate_temp               ["80"]

Every value is a two-element vector: **Standard variant, then High Flow**.
TINMORRY characterised only the Standard column and left High Flow as
`nil`, which resolves to whatever the inherited Bambu `Generic PETG-CF
@BBL P2S` preset says -- 11.5 mm3/s, 255 C and a 0.95 flow ratio, none of
it measured on this material.

That matters because this lane pins `default_nozzle_volume_type` to
**High Flow** to match the installed hotend, which selects exactly the
column the vendor never filled in.  So
`captive_magnet_slicing_profile_petg_gf_06hf.json` mirrors TINMORRY's
Standard-column figures into both slots -- see its `repo_overrides.filament`
-- and the lane prints the vendor's numbers whichever variant Studio
resolves.

## Consequence for print time

12 mm3/s is the authoritative ceiling, and it is only 4% above the 11.5 the
lane was running at.  Nothing about reading the real spec makes these plates
meaningfully faster: at 0.62 x 0.16 mm it lifts the speed ceiling from 116 to
121 mm/s.  Wall count, infill density and support density are the levers,
not the flow cap.
