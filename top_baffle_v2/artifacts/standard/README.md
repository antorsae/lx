# Standard R6P

![B2, A-comp, and B1 plan](images/plan.png)

This is the stock-bridge, full-depth product family. The B2 base is nominally
18.3 mm deep. Print the four `lx521_top_base_*` pieces, then choose one
perimeter treatment:

- no add-on: compact B2 outline;
- four `addonA` pieces: straight A-comp shoulders; or
- two `addonB1` pieces: flared B1 wings.

The shoulder and wing sets overlap and are mutually exclusive. `cad/base.step`
is the unsplit design; `cad/base_print_assembly.step` and the optional assembly
STEPs show the installed print pieces. Keep every STL paired with its
`.print.json` file.

The catalog uses the no-floor/stock-bridge state. The generator also retains a
separate integrated-floor state for engineering comparison.
