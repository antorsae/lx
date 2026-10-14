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

## Changeover calibration — 2026-09-13

The prepared PETG-GF/PLA projects use **560 mm³ purge in both directions** and a **12 mm³/s PLA flush limit**. This is a changeover setting; printing temperatures and printing flow limits retain their existing recipes. Open as a project and slice in Bambu Studio to generate new G-code. Older exported G-code does not acquire these changes automatically.

Use the [updated D6 body/wing test](../../../candidates/nd25fn4_crescent/print/qualification/README.md) to check actual changes between the two materials before a full part. [Shared policy](../../../docs/PRINT_POLICIES.md).
