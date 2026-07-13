# V1LF R6F final CAD brief

- Model: source modification of the two-piece V1LF LM/UM carrier core,
  its state-specific support interface, required floor-support add-on,
  optional tweeter/retention add-ons,
  and their review assembly.
- Inputs: `review/v1lf_routing_concept_preview_v3.png` is the plan-routing
  authority. Numeric requirements in the active goal override image
  proportions.
- Units and frame: millimetres; existing baffle XY frame is preserved;
  the LM centre is `(0, 200.981)` and the UM centre is `(0, 366.081)`;
  +Z points toward the acoustic/front face.
- Mandatory core: minimum LM and UM load-bearing rings, two pairs of tiny
  rounded bolted ears, six magnet sites used only for alignment, an UM route
  buried only in LM, and a T route buried only in LM/UM. Actual Ø5 x 2
  magnets use the global
  Ø5.2 x 2.2 pockets; their extra 0.2 mm depth is adhesive allowance and the
  magnets must be held flush rather than bottomed during bonding. Preserve the
  upper LM pair buried flush directly in the R113 lip at world polar 64/116
  deg (±26 deg from top), with no ear and at least 2.2 mm pocket-edge to
  nearest insert-pad edge. Add a lower LM pair face-flush at 224/316 deg with
  at least 23.0 mm nearest-insert edge clearance. Keep 224 deg at z=12.55;
  use the Z-preferred z=15.40 position at 316 deg, retaining a closed 0.30 mm
  front skin. Both clear buried routes and the bridge/support load path. The
  UM pair is also earless and flush at 50.5/129.5 deg, z=15.1, with
  1.1 mm T-cover clearance, 0.2 mm radial floor and 0.6 mm front skin. The
  D7.8 LM lead is a modeled free span with no printed micro-duct. The UM cable
  is free behind the UM carrier with no printed rear duct; T is free behind
  the tweeter crescent, which owns no printed cable arc.
  No-floor mode additionally
  owns one monolithic fused bridge-interface tail; there is no separate
  no-floor keel. Everything else is a separate add-on. The floor support
  add-on is mandatory in the floor state, and the floor LM has no tail.
- Minimal carrier section: non-load annular slabs are deleted. Each driver
  seat retains only a 0.85 mm two-extrusion membrane; narrow outer lips,
  local blind-insert bosses/floors and calculated radial spokes carry load.
- V1LF LM axes: six sites at `0/60/120/180/240/300 deg` on the unchanged
  209.5 mm PCD. No-floor owns six carrier heat-sets. Floor owns upper
  carrier heat-sets at 0/60/120 plus lower Ø5.5 clearances leading to
  rear-installed support heat-sets at 180/240/300. Proud/V1L families
  retain their existing `30/90/.../330 deg` pattern.
- Bridge datum: global hole centres `(-20,20)`, `(20,20)`, `(-20,70)`,
  `(20,70)` are immutable. They preserve the 40 x 50 mm pattern and,
  relative to the LM centre, the existing 182.083 mm lower-row and
  132.499 mm upper-row radii. No-floor mode fuses a 62 mm-wide rounded,
  opening-free solid web around these holes into the LM carrier. The web is
  flush with the front and occupies `z=5.3..18.3`, exactly the deepest existing
  LM insert-pad envelope; it has no X, hollow opening, rear rib, or additional
  rear-depth structure. Four rear-opening Ø6.4 x 6.8 bores leave a 6.2 mm solid front
  floor, and no bridge geometry extends behind the existing LM-pad envelope.
  Floor mode has no bridge web and requires the independent floor/NL8
  add-on to complete its lower three threaded W22 axes and qualified load
  path. The no-floor web extends through a 68° lower-ring cradle; because the
  existing annular ring lip starts at z=6.8, only the actual z=6.8..18.3
  monolithic interface is credited structurally.
  The 4 kg biaxial PLA Tough+ screen uses a conservative 47.8 × 13.0 mm
  route-net member: the 62 mm insert core minus the complete Ø8.2 and Ø6.0
  entry lumens, with no credit for their thin skins. Exact sampled soft-outline
  sections retain at least 53.5 mm. The design section has in-plane/rear
  section moduli 4950.5/1346.4 mm³ and 1g/3g/5g factors
  2.44/1.83/1.10. The cradle calculation deducts one
  Ø8.2 UM tunnel plus the complete Ø6.0 tweeter tunnel; its effective width
  is 118.5 mm, its 11.5 mm-deep in-plane/rear section moduli are
  26908.4/2611.6 mm³, and its factors are 6.37/4.78/2.87. The combined 5g
  insert reaction is 434.2 N, for 1.38
  assumed pull-out safety factor at 600 N per insert. Magnets receive 0 N.
- Cable voids: the printed UM passage is nominal Ø8.2 and exists only in the
  LM carrier. The printed T passage is nominal Ø6.0 and exists in the LM and
  UM carriers only. Their complete physical validation solids remain UM Ø7.0
  and T Ø5.2 across both buried and free spans. The LM physical envelope is
  Ø7.8, but it has no printed duct.
- Routing intent: LM uses a short 20.15 mm free radial span at 269.5 deg behind the
  carrier with no printed micro-duct, cover, or cutter. Its centerline rises
  from z=0.40 to 3.80, beginning at R103 before the R95/D190 mouth, and
  retains 1.00 mm clearance to the deepest z=5.3 pad/web rear datum at the
  outer station.
  Floor support removes only the physical LM cable plus 0.4 mm clearance. At
  its three threaded axes, Ø11.6 connected printed bosses sit inside Ø12.4
  carrier clearances and retain 2.6 mm radial wall around Ø6.4 heat-set
  cavities. UM rises inside the right LM arc, exits the LM-owned buried
  passage, and continues free behind the UM carrier. T rises inside the left
  LM arc, remains buried through the UM carrier, then exits and continues
  free behind the tweeter crescent. Their physical centerlines cross at
  82.67 deg with T higher in +Z, UM lower, and a 1.85 mm physical-envelope
  gap. There is no printed UM-owner arc at the crossing, no two-duct
  separator web, and no crescent-owned T arc.
  All eight named insert bypasses are smooth local Z dips with continuous
  closed cover and a full-width solid saddle from conduit roof to the
  applicable blind-bore floor. The saddle does not extend behind its conduit
  bump. Continuous full-width longitudinal burial webs back the LM-owned
  UM/T low runs and the UM-owned T low run to their seat membranes; in
  particular, neither longitudinal shoulder at the UM 328°/58° bypasses may
  contain a trapped cavity outside the exact D6 lumen, blind-bore,
  flush-magnet and half-lap interface voids. At the floor-state 300/240/180° axes only the exact grown support
  insert/shank hardware clearances remain void; all surrounding saddle
  material is solid. Every surviving buried span retains a 0.8 mm minimum
  wall and 0.85 mm seat roof; no trapped roof-to-bore cavity, bore-jump, or
  unintended rear cable window is permitted. Printed ownership ends in
  plain flush mouths: UM becomes free after its LM-owned passage, and T
  becomes free after its UM-owned passage. No telescoping printed handoff,
  UM-carrier rear duct, or crescent cable arc is permitted.
- Terminal intent: MU terminal clock remains 283 deg, midway between the
  238/328 deg screws. With no printed UM-carrier rear duct or D82 mouth, the
  free Ø7 jacket follows the modeled R15 terminal approach to that immutable
  service axis with a clockwise circumferential 193 deg tangent at z=2.7,
  then continues with exact G1
  continuity through R20 to a named Y breakout, not inward through the
  known Ø60 motor. The breakout has a 4 mm-long OD8 collar and two OD4
  branch sleeves. Its two provisional Ø3.2 conductors then follow explicit
  R8-minimum slack paths into provisional
  non-overlapping low-profile flag Fastons (8.5 mm receptacle, 9.5 mm boot,
  11 mm pitch). Service review moves one connector at a time through the
  declared 0/3/6/9/12 mm pull states while the other remains installed.
  The dedicated `top_baffle_nd25fw4_um_fit.step` review model includes a
  closed Ø98/Ø80/Ø60 body keep-out derived from the terminal-less reference
  mesh, the conservative stepped W22 rear-body proxy, cable and service
  envelopes. The assembled carrier STEP is reviewed
  together with this fit STEP; it does not independently contain the W22
  proxy.
  The W22 envelope is provenance-linked to hash-pinned
  `E0022_W22EX001.stp` SHA-256
  `7fc2be551c86006e11c32a570b046772987cb86dcf65350f77c6e34709aa5ab6`:
  native +Y maps to world +Z through +90 deg about X, native +Z maps to
  world -Y, translation is `(0, 200.981, -47.498931)`, and native max-Y
  is placed at front datum z=18.3. Cached native bounds
  `(-110.5,-37,-110.5)..(110.5,65.798931,110.5)` therefore map to world
  `(-110.5,90.481,-84.498931)..(110.5,311.481,18.3)`. The guarded
  `test_w22_reference_step_geometry` phase imports that exact STEP, verifies
  the transform/bounds, and proves the transformed reference is contained by
  the stepped service proxy. This qualifies only the pinned W22 reference,
  not the installed custom U22. The final
  V1LF emits no printed grommet or relief; the lead is already free behind
  the UM carrier. Cable retention and physical terminal fit remain physical
  checks.
- Qualification status: `PHYSICAL_MEASURE_REQUIRED = True`. The MU
  reference omits both terminals and the datasheet leaves their carrier
  and withdrawal geometry un-dimensioned. The modeled 12 mm pull equals
  the provisional 12 mm exposed tab length, so it has zero positive
  release overtravel margin. Terminal, boot, cable, Y breakout, strain relief,
  and one-at-a-time removal qualification therefore remain
  pending a recorded real-driver dry fit. Floor and no-floor candidates also
  require separate artifact/process identity, coupon evidence, documented
  1g/3g/5g structural proof and signed release decisions; neither state may
  inherit the other's evidence.
- Manufacturing assumptions: Bambu PLA Tough+; 0.4 mm nozzle. A
  non-load-bearing skin on every surviving buried route starts at two full extrusion widths
  (0.8 mm). Structural rails/bosses use separately calculated sections.
- Primary paths: `top_baffle_nd25fw4_v1lf*.py`; generated STEP files in
  `floor_stand/` and `no_floor_stand/`; print meshes in each `stl/` tree;
  routing sheets at `*/baffle_cable_routing_v1lf.png`, including plan,
  longitudinal side profiles, and nominal diametric u-z sections with exact
  vertical backfill limits through
  representative surviving UM/T conduit-plus-pilot bumps and explicit free
  rear UM/tweeter spans; candidate provenance
  in each `v1lf_release_manifest.json`; physical evidence and per-state
  signoff in `V1LF_PHYSICAL_QUALIFICATION.md`.
- Validation: exact insert and bridge coordinates; opening-free front web,
  rear-entry bore/front-floor depth, and zero rear protrusion; state isolation;
  six global Ø5.2 x 2.2 magnet pockets, four LM/two UM polar positions and
  face-flush bonding, preservation of the upper 64/116 deg LM pair, addition
  of the lower 224/316 deg LM pair, including at least 23.0 mm nearest-insert
  edge clearance and the 316 deg z=15.40 route bypass, and at least 2.2 mm
  upper-LM nearest-insert edge gap; absence of LM proud ears and floor-support LM magnet cups/arms;
  free D7.8 LM span clearance with zero printed micro-duct and cable-only floor
  support clearance;
  physical-cable containment; insert/head/terminal/service clearance;
  minimum normal wall and eroded-outline containment on printed-owner spans;
  positive absence of a printed UM-carrier rear duct and crescent-owned T
  arc; free-cable clearance, crossover separation and angle; G1 continuity
  and bend radii across printed-to-free handoffs;
  final-BREP solid saddle continuity from conduit roof to bore floor at all
  eight named bumps, with only the exact floor hardware exceptions;
  combined-axis 4 kg sustained-1g/3g/5g bridge/support screen, plus the
  actual 0.85 kg upper-joint load case; 256 mm bed fit; one valid
  solid per print part; and zero open or over-shared STL edges.
- Review handoff: state-specific snapshots of the split core, attachments,
  assembled carrier and terminal-fit STEP, plus live CAD Viewer links. Routing
  diagrams are not substitutes for rendered STEP review.
- Generation safety: ordinary Make targets execute on `osado.lan` in one
  512 GiB/no-swap systemd cgroup with a 64 GiB host-available floor. The
  default four guarded recipe slots are capped at 112 GiB per process tree.
  That profile uses concurrent check processes, direct final-part builds and
  full route witnesses; it does not inherit macOS-only cutter/route tiling.
  Explicit `LX_CAD_EXECUTION=local` remains serial with an 8 GiB process-tree
  RSS ceiling and a 0.5 GiB macOS immediately reclaimable-memory
  launch/runtime floor (free + speculative + purgeable; inactive/compressed
  excluded).
