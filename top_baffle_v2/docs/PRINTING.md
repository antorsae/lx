# Printing on H2C

**H2C is the only supported printer, using two 0.6 mm High Flow nozzles.**
Use the [generated H2C file catalog](../to_print/README.md) for current
projects, material assignments, quantities and audit results. The authority
is [print_policy_h2c.json](../print_policy_h2c.json).

| Item | Current policy |
|---|---|
| Materials | TINMORRY PETG-GF model / PLA interface (the only material lane) |
| Nozzles | Left model, right support interface; native H2C priming and standby |
| Plate | Engineering Plate with glue, 70 °C; no raft |
| Layers / walls | 0.16 mm layers, 0.20 mm first layer, six walls |
| Structural LM / UM | 100% infill |
| Wings | 10% gyroid |
| ND25FN-4 vases | 100% zig-zag, except the T and t waveguides: 15% gyroid (modifier volume; MU10 seat and LM/seam joints stay 100%) |
| ND25FN-4 caps | 15% gyroid |
| Regular crescent | Existing 30% policy |
| Supports | Model-material support bases; three dense zero-gap PLA interface layers above and below |
| Caps / retainers | PLA beneath cap ceilings; retainers without supports |
| Brim | 2 mm for complete Stock/Slim no-floor LM; 5 mm for other parts |

Preserve the selected project's support blockers, infill modifier, material
assignment and magnet pauses. Never infer these from the STL. Dedicated
nozzles use the H2C purge policy; flushing into the model, infill or supports
is disabled.

```sh
make h2c_prepare    # editable H2C projects from current geometry
make h2c_validate   # slice/audit; reuse only hash-matched results
make h2c_docs       # regenerate the file guide
```

Local preparation and slicing run inside the 8 GiB process-tree memory
guard. The waveguide acoustic study runs on Osado; see
[execution and memory policy](REMOTE_BUILD.md). These commands do not send a
job to the printer.

Before a full build, check actual plate adhesion, insert fit, magnet-wall
closure, support removal and loaded structural behavior. Use the selected
job's audit and the [qualification procedure](PETG_GF_QUALIFICATION.md).

[Archived process notes](archive/P2S_PRINTING.md) retain earlier small-bed
and PLA experiments. Those presets and printer commands are deprecated.
