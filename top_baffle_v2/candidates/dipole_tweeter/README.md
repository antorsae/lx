# Dipole tweeters (GRS PT2522-4, Mundorf AMT17D2.1) — approved H2C inputs

The shelf's `grs_pt2522_4` and `mundorf_amt17d2_1` families (`to_print/obiwan/dipole/`); see
[the dipole tweeter guide](../../docs/DIPOLE_TWEETERS.md) for what to print and how the parts fit.

| Part (`STL/`) | Layout | Replaces | Mates with |
|---|---|---|---|
| `<family>_obiwan_t_above_{upright,rotated}` | UM + tweeter frame, T above the MU10 | UM carrier + crescent | Obiwan LM carrier or keyed LM top (released LM–UM joint) |
| `<family>_obiwan_t_below_{upright,rotated}` (not PT2522-4 upright) | keyed LM top + tweeter + raised MU10, one piece | keyed LM top + UM + crescent | Keyed LM bottom, either stand state (released keyed seam) |
| `grs_pt2522_4_obiwan_t_below_upright_{lm_top,um_t}` | the same as a custom keyed LM top + its T + UM piece (they only fit each other) | keyed LM top + UM + crescent | Keyed LM bottom; the two prints meet at the released LM–UM joint |

**Do not print from this folder.** These STLs are source meshes in the installed frame (front towards +z), so a
slicer shows them front face up. The print files are in `to_print/obiwan/dipole/`: posed front face down, with the
duct support blockers, PLA interface and brim settings in the `.petg-gf.3mf` / `.gcode.3mf`.

## Layout

```
STL/*.stl + *.print.json            approved meshes in the installed frame (identity print authority)
support_blockers/*.json|.stl        duct contracts + blockers (the shelf dilates them by 0.8 mm)
build_package.py                    packages the approved meshes and writes the blockers
source/                             the duct searches and part checks (run on Osado)
source/checks/*.log                 latest check output per print
```

The meshes are built by `scripts/build_dipole_tweeter.py` (`um_t_above`, `lm_um_t`, `lm_um_t --pair`) from the
audited Obiwan shelf parts and the searched routes; the heavy mesh booleans and duct searches run on Osado
([searches and checks](source/README.md)). `build_package.py` then copies the meshes from `build/planar_pt2522/` and
`build/amt17d2/` and writes every blocker:
- **Released blocker:** the released part's own blocker, the UM carrier's for T above and the keyed LM top's for T below. The old duct tails past each branch are cut out of it.
- **New ducts:** the new duct stretches each print holds.
- **Insert bores:** the MU10 and driver M3 bores.
- **Lead channel:** on the AMT, its lead channel.

Contracts keep only the polyline duct regions, because the shelf audit reads capsules. The AMT lead channel stays in
the blocker geometry.

Status: digital checks pass ([latest results](source/README.md#latest-results)); nothing has been printed yet.
