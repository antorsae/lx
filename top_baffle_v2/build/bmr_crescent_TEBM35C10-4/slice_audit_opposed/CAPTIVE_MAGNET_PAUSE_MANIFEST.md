# Captive-magnet pause manifest

Authoritative for the exact STL and profile hashes below. This run used Bambu Lab P2S 0.4 nozzle, 0.16 mm Arachne layers, and Bambu PLA Basic @BBL P2S. All parts are front-face-down.

This audit did not contact a printer and did not upload or start a print.

## Insertion procedure

1. Open the hash-listed ready-to-print 3MF; do not auto-orient it.
2. The Bambu Custom park/pause/restore events at the exact **first-closing** Z values below are already embedded and were verified in both project XML and G-code; do not add or move them manually. Each raises the nozzle to Z=250 mm (lowering the bed), pauses with `M400 U1`, then restores the exact layer Z on Continue.
3. At each pause, insert the listed number of D5 x 2 mm magnets vertically downward from above (+Z side) along print `-Z` (`print_insertion_direction_xyz = [0, 0, -1]`), with the marked pole oriented exactly as specified.
4. Ensure every magnet is fully seated below the completed layer and cannot rise into the toolhead path.
5. Resume printing. Polarity cannot be corrected after the roof buries the magnet.

## Exact pauses

| State | Variant / part | Pause Z | Last open | Seated margin | Magnets / sites | Insertion | Polarity |
|---|---|---:|---:|---:|---|---|---|
| shared | Obiwan-TEBM35C10-4-BMR-crescent-opposed / `obiwan_bmr_crescent_opposed_TEBM35C10-4` | **5.96 mm** | 5.80 mm | 0.10 mm | 4 / tebm_lower_left_base, tebm_lower_right_base, tebm_upper_left_base, tebm_upper_right_base | `[0.0, 0.0, -1.0]`: insert vertically downward from above the paused part (+Z side) through the open loading chimney along print -Z | `tebm_lower_left_base`: marked pole → `(-1.0, 0.0, 0.0)` in print coordinates; marked/N pole points OUT from the pod along installed_marked_pole_axis_xyz; verify the future mating piece uses the opposite interface-facing pole before burial<br>`tebm_lower_right_base`: marked pole → `(1.0, 0.0, 0.0)` in print coordinates; marked/N pole points OUT from the pod along installed_marked_pole_axis_xyz; verify the future mating piece uses the opposite interface-facing pole before burial<br>`tebm_upper_left_base`: marked pole → `(-1.0, 0.0, 0.0)` in print coordinates; marked/N pole points OUT from the pod along installed_marked_pole_axis_xyz; verify the future mating piece uses the opposite interface-facing pole before burial<br>`tebm_upper_right_base`: marked pole → `(1.0, 0.0, 0.0)` in print coordinates; marked/N pole points OUT from the pod along installed_marked_pole_axis_xyz; verify the future mating piece uses the opposite interface-facing pole before burial |

## Audited Bambu arrangements

Every listed 3MF was exported by the same Bambu slice invocation, hash-bound to the staged STL, and audited as an exact mesh with only a proper unit-scale rotation about print Z plus XY placement.

| State | Variant / part | Arrange Rz | Ready-to-print 3MF | SHA-256 | Ready fingerprint |
|---|---|---:|---|---|---|
| shared | Obiwan-TEBM35C10-4-BMR-crescent-opposed / `obiwan_bmr_crescent_opposed_TEBM35C10-4` | 0.000000 deg | `/Users/antor/gh/lx/top_baffle_v2/build/bmr_crescent_TEBM35C10-4/slice_audit_opposed/slices/shared_Obiwan-TEBM35C10-4-BMR-crescent-opposed_obiwan_bmr_crescent_opposed_TEBM35C10-4/ready/ready_to_print.gcode.3mf` | `201d4035999f7ab16d86aea81c63c78f10bc7b85df5bf848f0a020273c4ebabc` | `551cd0d9d6a8504d021c5bbbd03f198f67eb579ccb198ca5e8565df4f8ccf119` |

## Profile and evidence

- Catalog SHA-256: `411a829c4508c1b9d05d31e6236b9d33a7be002db3b7f31e923df2ed2afe7f6c`
- Resolved profile-set SHA-256: `83151d56c495b6964e835fdb0a79797dfd6caace5157052e00494101d77aaaeb`
- Bambu Studio binary SHA-256: `b022be6750898454803e9e07178b7c7446c0e5b4d148c593b4b56efde09ba281`
- Artifacts: 1 passed, 0 failed
- Each printable artifact directory under `slices/` contains the hash-bound arranged Bambu 3MF, plain G-code, Bambu `result.json`, static validator output, and five-layer SVG/PNG toolpath evidence for every cavity.

The JSON file is the machine-readable authority; this Markdown and the CSV are derived views.
