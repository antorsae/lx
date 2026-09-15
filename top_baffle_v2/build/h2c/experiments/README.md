# H2C development trials

These are diagnostic inputs and reports, outside the manufacturing shelf.
Use the current projects in `../../../to_print/h2c/` for printing.

- `rejected_one_piece_floor/`: unchanged Stock/Slim full floor-stand LM
  proposals. Actual PLA support deposition exceeded the right nozzle's reach.
  The released floor version uses the established lower LM joint instead.
- `classic_wing/`: changing the full V4 right wing to Classic fixed the D6
  insertion path but obstructed a D5 pocket too early. This setting was rejected.
- `wing_arachne_43.5_inner/`: a small orientation change still obstructed a D6
  insertion path. `wing_arachne_224.0_inner/` established the successful opposite
  orientation subsequently qualified for both right-wing styles and materials.
- `floor_support_witness.*` and `h2c_slim_lm_lower_floor_stand_shadow_trial/`:
  support seeded beneath the rear service panel passed down the cable entries.
  Extending the entry blockers through that panel removed those columns. The
  current floor jobs include this correction and have their own release audits.

Trial hashes describe the inputs at the time of each experiment. They do not
qualify current print files. Plain G-code and duplicate trial slice archives
remain disposable local caches; release G-code is stored in the published
`.gcode.3mf` files.
