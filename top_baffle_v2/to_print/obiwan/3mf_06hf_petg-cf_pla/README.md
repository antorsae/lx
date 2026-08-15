# PETG-GF structural plates for GUI slicing

These are Bambu Studio **projects**, not sliced deliveries: they carry no
G-code, and they are not part of the audited `3mf_06hf` shelf. Same 0.6-mm
high-flow lane, kept separate because they still have to be sliced.

Bambu's CLI cannot slice them. Every structural plate loads an assemble list
(the duct blockers and the bridge/root modifier need one) and on that path
Studio maps the second filament to nozzle 0, which does not exist, then
prints the support interface in the model filament and reports success. The
repo now refuses that G-code, so these plates are handed over as projects
instead. See `docs/PRINTING.md` for the full finding.

## What is already set

* filament 1 = `TINMORRY PETG-GF Profile @BBL P2S`, filament 2 =
  `Bambu PLA Basic @BBL P2S 0.6 nozzle`, both mapped to the single nozzle
* supports on, printed in filament 1, interface in filament 2, zero top Z
  gap, snug, `support_on_build_plate_only`
* 10 top shell layers and ironing on top surfaces
* the four core pieces at their locked placements, all three duct blockers,
  the 100%-solid bridge/root modifier
* the six-magnet pause at Z = 5.96 mm, with its park/restore program

## What to do

1. Open the project in Bambu Studio 02.07.01.62.
2. Assign filament 1 to the **AMS-HT** slot holding TINMORRY PETG-GF, and
   filament 2 to the **AMS** slot holding PLA (slot 2 or 3).
3. Slice. Confirm before printing:
   * the support **interface** is filament 2 and the support **body** is
     filament 1 -- if both come out filament 1 the mapping did not take;
   * the pause still sits at Z = 5.96 mm and announces six magnets;
   * a prime tower is placed clear of all four parts.
4. Insert the six magnets at the pause, then resume.

Do not print these alongside the individual 01/02/03/04 files, and do not
mix the two stand states.
