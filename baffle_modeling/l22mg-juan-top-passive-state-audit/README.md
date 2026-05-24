# Juan LX521 Top-Baffle Passive-State Audit

Source HDF5: `output/data/polar_data_juan_lx521_top_raw.h5`.

This audit reads only the stored HDF5 notes and attributes. It does not infer passive geometry from the model result.
When run with `--write-attrs`, it records the same provenance as structured HDF5 attrs; unknown passive geometry remains unknown.

## Conclusion

- The current L22MG validation target proves distance, mic height, top-baffle mounting, and raw/no-crossover processing, but it does not record the physical state of the unused L10NEO/tweeter positions.
- The separate multi-driver capture explicitly records L22MG + L10NEO + tweeters on the LX521 top baffle, but it is a multi-source no-crossover measurement, not an L22-alone passive-geometry target.
- Therefore the passive-geometry acceptance gate should remain `not_proven` for the Juan L22-only target until notes or HDF5 metadata state whether the unused positions were open, covered, or mounted inactive.

## Summary

| driver | role | distance | height | front angles | rear angles | passive-state status | HDF5 attr | acceptance use |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- |
| `L22MG (LX521 top raw)` | `current_l22_validation_target` | 0.5 | `l22mg` | 0/15/30/45/60/75/90 | 0/15/30/45/60/75/90 | `unused_um_tweeter_state_unrecorded` | `unused_um_tweeter_state_unrecorded` | `current_l22_target_but_passive_geometry_not_proven` |
| `L10NEO (LX521 top raw)` | `single_driver_context` | 0.5 | `l22mg` | 0/15/30/45/60/75/90 | 0/15/30/45/60/75/90 | `unused_l22_tweeter_state_unrecorded` | `unused_l22_tweeter_state_unrecorded` | `context_only_not_l22_target` |
| `L22MG+L10NEO+Tweeters (LX521 top raw)` | `multi_driver_context` | 0.5 | `l22mg` | 0/15/30/45/60/75/90 | 0/15/30/45/60/75/90 | `all_drivers_active_no_crossover_recorded` | `all_drivers_active_no_crossover_recorded` | `context_only_not_l22_alone_passive_target` |

## Stored Notes

### L22MG (LX521 top raw)

- Measurement note: Measurement distance: 50 cm. Mic height: L22MG/LM. LX521 top baffle mounted; raw/no crossover/no EQ.
- Sample titles: SEAS L22MG A 0 F | SEAS L22MG A 15 F
- Sample notes: PB -40 DB Measurement distance: 50 cm. Mic height: L22MG/LM. LX521 top baffle mounted; raw/no crossover/no EQ. | Measurement distance: 50 cm. Mic height: L22MG/LM. LX521 top baffle mounted; raw/no crossover/no EQ.
- Passive-state evidence: notes prove 50 cm, LM/L22MG height, LX521 top-baffle mounting, and raw/no crossover/no EQ; they do not state whether the unused L10NEO/tweeter positions were open holes, covered patches, or mounted inactive drivers.

### L22MG+L10NEO+Tweeters (LX521 top raw)

- Measurement note: Measurement distance: 50 cm. Mic height: L22MG/LM. L22MG + L10NEO + tweeters on LX521 top baffle; raw/no crossover/no EQ.
- Sample titles: SEAS L22MG A 0 F | SEAS L22MG A 15 F
- Sample notes: PB -40 dB L22MG + L10NEO + TWEETERS SIN XOVER EN TOP BAFFLE LX521 NO EQ Measurement distance: 50 cm. Mic height: L22MG/LM. L22MG + L10NEO + tweeters on LX521 top baffle; raw/no crossover/no EQ. | Measurement distance: 50 cm. Mic height: L22MG/LM. L22MG + L10NEO + tweeters on LX521 top baffle; raw/no crossover/no EQ.
- Passive-state evidence: notes explicitly identify L22MG + L10NEO + tweeters on the LX521 top baffle, raw/no crossover/no EQ.

## Structured HDF5 Attrs

The audit writes these attrs when invoked with `--write-attrs`:

- `passive_state_status`
- `passive_state_evidence`
- `passive_state_acceptance_use`
- `passive_state_metadata_policy`

These attrs are attached to the driver group, `angles`, `rear_angles`, and per-angle groups.
For the L22-only target, `passive_state_status=unused_um_tweeter_state_unrecorded` is an explicit unknown, not evidence for open holes, solid patches, or mounted inactive drivers.

Machine-readable summary: `target_passive_state_summary.csv`.
