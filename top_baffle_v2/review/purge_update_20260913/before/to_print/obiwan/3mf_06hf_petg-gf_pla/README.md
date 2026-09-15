# PETG-GF projects for GUI slicing

These eight projects contain no G-code. Open them in Bambu Studio with the
0.6 mm high-flow nozzle. Supported parts use TINMORRY PETG-GF for the model
and support body, PLA Basic for the interface, normal/snug support and a
zero top Z gap. Assign both filaments to the actual single-nozzle material
slots, slice, inspect and export the result for `scripts/audit_gui_slice.py`.

The no-floor core combo contains four parts; the floor combo adds the NL8
service lid (five). Either replaces the matching individual parts. Both
combos pause at Z=5.96 mm for six magnets. Singles pause only for their own
sites, and the support-free crescent/lid have no magnets or pauses.

Six walls are used throughout this lane. Core combos, LM bottoms and UM
carriers and the standalone LM top use 100% zig-zag with support. The crescent/lid use 30% gyroid with support off.

See `../../../docs/BUILD_GUIDE.md` and `../../FILE_GUIDE.md` for selection,
assembly, estimates and qualification. `../../delivery_manifest.json` binds
project hashes and audits the actual normal, blocker and modifier meshes.
No GUI project or partial slice audit authorizes a physical release.
