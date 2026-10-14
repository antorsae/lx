# Dipole tweeter duct searches and checks

The scripts that searched the dipole parts' ducts and checked the built parts. They ran on Osado and are kept here
as run, unmodified. The parts themselves are built by `scripts/build_dipole_tweeter.py` from the routes stored in
`src/lx521_baffle/obiwan/dipole_tweeter*_routes.json`; nothing here is needed to rebuild a part from those routes.

## Running them

From `top_baffle_v2/`, in an environment with build123d, manifold3d, trimesh, shapely and scipy:

```
ln -s candidates/dipole_tweeter/source dtw     # the scripts refer to each other as dtw/...
export SC=/path/to/scratch/                    # search solids and results (trailing slash)
LX_DIPOLE_T_DRIVER=amt17d2_1 LX_DIPOLE_T_ORIENTATION=landscape python dtw/int_check.py
```

`LX_DIPOLE_T_DRIVER` is `pt2522_4` or `amt17d2_1`; `LX_DIPOLE_T_ORIENTATION` is `portrait` (upright) or `landscape`
(rotated). The searches are differential-evolution runs over one clamped cubic B-spline per duct (`SEED`, `POP`,
`MAXITER`/`IT` select the run). They take minutes to hours each; run them on Osado. The checks read the built STLs in
`build/planar_pt2522/` and `build/amt17d2/`. For the comparisons against the released parts, they can also read
`dtw/um_carrier_installed.stl` and `dtw/lm_keyed_top_installed.stl`: the released UM carrier and keyed LM top in
installed coordinates (`released_installed()` in `scripts/build_dipole_tweeter.py`).

## Shipped layouts

| Layout | Stage solid | Search | Into the routes file | Check |
|---|---|---|---|---|
| T above (4 prints) | `export_above.py` → `above_solid_<key>.stl` | `above_opt.py` (T duct) | `above/<driver>/<orientation>` in `dipole_tweeter_routes.json` (no import script kept) | `final_check.py` (`VARIANT=um_t_above`), `taper_check.py` (AMT) |
| T below, one piece (3 prints) | `export_isolid.py` (`STAGE=pre_ducts`, then `pre_t`) | `int_opt.py` (`DUCT=um`, then `DUCT=t`) | `import_int.py` → `dipole_tweeter_lm_routes.json` | `int_check.py` |
| PT2522-4 upright T below, pair (2 prints) | as above, with `LX_DIPOLE_T_PAIR=1` | `int_opt.py` with `INTERIOR=1` (keeps clear of the M2 tie screw and of any bore inside the duct) | `import_int.py` → `dipole_tweeter_pair_routes.json` | `pair_check.py` |

`int_opt_eval.py` and `guide_opt_eval.py` are earlier variants of the searches with diagnostic output (`EVAL=`), and
`pair_dbg.py` lists the sealed cavities left at each pair build stage.

## Retired split T-below parts

The first T-below layout was a separate T + UM piece above the regular LM. These scripts searched and checked it;
its routes are the `<driver>/<orientation>` keys of `dipole_tweeter_routes.json`. None of it ships.
- **Stage solids:** `export_solids.py` and `export_gsolid.py`.
- **Searches:** `route_opt3.py` (T duct) and `guide_opt.py` (UM guide).
- **Imports:** `pick_best.py`, `import_routes.py`, `import_t_one.py`, `import_guide.py`, `import_guide_one.py` and `merge_routes.py`.
- **Check:** `final_check.py` with `VARIANT=um_t`.
- **Diagnostics:** `diag_above.py` and `diag_pt_rot.py`.
- **One-off patches:** `pilot_patch.py`, `pilot_patch2.py` and `tdiag_patch.py` edit `route_opt3.py` in place. Their edits are already in the copy kept here.

## Latest results

[`checks/`](checks/) has the latest check output for each shipped print, one file per layout. Each starts with the date and the
Osado log it came from. Every check passes against the meshes in `STL/`:
- **Ducts:** every new duct stays inside material. On the T-below prints, the LM's own ducts stay enclosed from the keyed seam to each branch.
- **Bends:** shipped bends are R15.8 or more.
- **Insert bores:** the M3 bores are 4.0 mm deep with walls of 1.2 mm or more.
- **Front plane:** nothing stands proud of it.
- **Seam zone:** on T below, the keyed seam zone matches the released keyed LM top to within 0.1 mm³.
- **Pair:** the two prints don't overlap, and the T duct passes 5.69 mm from the M2 tie screw path (5.40 needed).

One output line needs reading with care. On the AMT T-above parts, `final_check.py` samples the T duct at a constant
D4. It therefore reports the first 12 mm of the duct as exposed: that is the deliberate D6 → D4 taper at the branch
(`T_TAPER_MM` 15, see `t_taper_d`). `taper_check.py`, in the same log, checks that stretch at its real diameter and
finds no exposed skin.
