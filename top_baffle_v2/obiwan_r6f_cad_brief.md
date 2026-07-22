# Obi-Wan R6F final CAD brief

- Model: source modification of the two-carrier Obi-Wan LM/UM core,
  its state-specific integral bridge/floor geometry, optional
  tweeter/retention add-ons,
  and their review assembly.
- Inputs: `review/obiwan_routing_concept_preview_v3.png` is the plan-routing
  authority. Numeric requirements in the active goal override image
  proportions.
- Units and frame: millimetres; existing baffle XY frame is preserved;
  the LM centre is `(0, 200.981)` and the UM centre is `(0, 366.081)`;
  +Z points toward the acoustic/front face.
- Print orientation: every released printable R6F piece, magnet-bearing or
  not, prints front-face-down, with only in-plane bed rotation. Pause heights
  and polarity for magnet-bearing pieces come from
  `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md`, not
  nominal CAD Z arithmetic.
- Mandatory core: minimum LM and UM load-bearing rings, two pairs of tiny
  rounded insert-fastened ears, six magnet sites used only for alignment, an
  UM route buried only in LM, and a T route buried only in LM/UM. At
  x=±32.0/y=315.770 and x=±24/y=421.5, each closure-web/base teardrop remains
  nominal Ø9 while every complete Z-owned cylindrical functional boss is
  locally Ø9.8. LM and UM respectively own complete standalone rear Ø3.4
  screw-clearance passages; UM and the crescent respectively own complete
  standalone rear-opening blind Ø4.6 x 4.0 receivers for M3 x 3 heat-set
  inserts, with 360° walls and 1.9 mm acoustic-front floors. Both interfaces
  retain a 0.20 mm axial gap. Install inserts in the individual UM and
  crescent prints before assembly; no washer, nut, front bolt head, or
  cross-owner receiver wall is part of either interface. Actual Ø5.0 x 2.0
  magnets use the global pause-and-bury captive system: Ø5.20 x 2.10 internal
  cavity, 0.45 mm plastic skin at each axial face, vertically open loading
  cradle, and self-supporting 45-degree roof. The finished magnet has no glue
  or external access opening. Preserve the upper LM pair at world polar 64/116
  deg (±26 deg from top), with no ear and at least 2.2 mm cavity-edge to
  nearest insert-pad edge. Preserve the lower LM pair axes in the common
  straight base side faces at `(x,y,z)=(±32,18,15.10)`, with left/right
  outward normals `(-1,0)`/`(1,0)`. Both stand states expose those exact
  base-side datums, and both cavities clear buried routes and the
  bridge/integral-stand load path. The
  UM pair is also earless at 50.5/129.5 deg. All LM-lower, LM-upper, and UM
  magnet axes share the common source plane **Z = 15.10 mm**. The structural LM and UM
  ring radii remain 113.0 and 51.7 mm. Their exposed sides are continuous
  cylindrical fairings at radii 113.8 and 52.5 mm, clipped only inside the
  existing LM--UM and T--UM cusp/service regions. The LM--UM stop keeps the
  0.40 mm inter-carrier gap open. At each LM-upper
  and UM ring station, the cavity construction datum is structural radius
  +0.65 mm, 0.15 mm beneath the exposed surface. There is no local pad, boss,
  flat, or silhouette cue. The D5 x 2 cavity and 0.45 mm skins are unchanged.
  Matching Ac/Ae solids contact with zero physical mating gap; their receiver
  construction datum has a 0.05 mm solid standoff, not an air gap. Nominal
  paired magnet-face separation is therefore 1.10 mm at the ring stations and
  remains 0.95 mm at the straight LM-lower base-side stations. Every station
  is wholly internal and leaves no front, rear, or side location cue. Magnets receive zero
  structural-load credit. In no-floor state the D7.8 LM lead enters through
  the upper Ø9 bore in the D20 cluster and follows a buried Ø9 path to the
  common R14 rear handoff; floor state reaches that handoff through its
  integral Ø9 lane. The UM cable
  is free behind the UM carrier with no printed rear duct; T is free behind
  the tweeter crescent, which owns no printed cable arc.
  No-floor mode additionally owns one monolithic fused bridge-interface tail;
  there is no separate no-floor keel. Floor mode instead owns a full-height
  integral W64 stem/foot, R12 root, three buried floor lanes and rear NL8
  panel. There is no separate floor-support add-on or support fastener.
  Both modes expose one identical LM-lower front/wing-contact outline: the
  opening-free union of the W64 floor stem and the broad cubic no-floor bridge
  shoulder. It begins at world `Y=0`, reaches the same R113 tangencies in both
  states, and is identical through the complete wing depth `z=6.8..18.3`.
  Only geometry behind that common contact depth remains state-specific.
- Optional LM print form: the authoritative LM carrier remains one monolithic
  solid. A mutually exclusive two-print option is derived from that finalized,
  state-specific solid at world `Y=172.481 mm` with an exact zero-gap planar
  butt; use both optional halves or the monolithic LM, never a mixture. The
  bottom owns two symmetric Ø1.60 cylindrical pins at `x=±109.187`,
  `z=14.30`; both point world +Y normal to the seam, overlap the root by
  0.50 mm, and engage the top by 2.40 mm (2.90 mm total length). The top owns
  two 2.65 mm-deep blind sockets with 0.12 mm radial and 0.25 mm end clearance:
  right is round Ø1.84, while left is X-relieved to 1.96 × 1.84 mm. This
  round-plus-relieved constraint accepts ±0.30 mm relative pitch error across
  the 218.374 mm spacing instead of binding like two round sockets. Tiny
  exterior lands grow outward from the carrier lip, outside the LM recess, and
  retain ≥0.50 mm local radial/end wall, ≥0.05 mm recess plan clearance, and
  ≥0.13 mm conservative W22-flange plan clearance. Their worst-case reach is
  R114.4036: 1.4036 mm beyond structural R113.0 and 0.6036 mm beyond the
  finalized R113.8 visible fairing. Ac/Ae are geometrically compatible through 0.25 mm
  hidden carrier-interface pockets, with physical fit still coupon-qualified.
  With the monolithic LM those pockets are small hidden local reliefs; the
  three magnetic datums and primary retention geometry remain unchanged.
  The keys add no extra screw or standalone
  retention/load credit. Print and assemble both front faces down on one flat
  datum, move the top straight along world -Y so both pins enter together, and
  verify two-pin/socket fit, full seating, coplanarity and route-seam continuity.
  Each horizontal pin is four nominal 0.4 mm nozzle widths; require a
  process-matched coupon, actual U22 fit and slicer proof of both pins, both
  support lands and all minimum walls.
  The installed LM driver flange and all normal LM fasteners are the service
  splice across the seam. Both optional halves in both stand states print
  front-face-down with in-plane bed rotation only. Former Z26°/Z45° and
  floor-bottom X=−90° footprint qualifications are superseded because
  out-of-plane orientation cannot support the captive-magnet pause. Validate
  each generated front-down footprint against the selected printer. Both buried
  cable lumens cross
  the seam and retain their final open sections. The optional LM top inherits
  both complete LM-to-UM rear ears, their local Ø9.8 cylindrical functional
  bosses, and their standalone Ø3.4 clearance passages; neither bore may
  depend on the bottom half or the assembled UM.
- Minimal carrier section: non-load annular slabs are deleted. Each driver
  seat retains only a 0.85 mm two-extrusion membrane; narrow outer lips,
  local blind-insert bosses/floors and calculated radial spokes carry load.
- Obi-Wan LM axes: six sites at `0/60/120/180/240/300 deg` on the unchanged
  209.5 mm PCD. Both states own six ordinary blind carrier heat-sets; floor
  mode has no secondary support inserts or through-clearance sites. Proud/V1L families
  retain their existing `30/90/.../330 deg` pattern.
- Bridge datum: global hole centres `(-20,20)`, `(20,20)`, `(-20,70)`,
  `(20,70)` are immutable. They preserve the 40 x 50 mm pattern and,
  relative to the LM centre, the existing 182.083 mm lower-row and
  132.499 mm upper-row radii. No-floor mode retains a 62 mm-wide rounded
  insert-bearing core around these holes, but its opening-free front web grows
  to the universal LM-lower contact outline shared with floor mode. The web is
  flush with the front and occupies `z=5.3..18.3`, exactly the deepest existing
  LM insert-pad envelope; it has no X, hollow opening, rear rib, or additional
  rear-depth structure. Four rear-opening Ø6.4 x 6.8 bores leave a 6.2 mm solid front
  floor, and no bridge geometry extends behind the existing LM-pad envelope.
  Floor mode owns the monolithic stand described below plus only the shallow
  missing shoulder delta needed to reach the same universal contact outline;
  it does not acquire the four bridge bores. The no-floor web extends through
  a 68° lower-ring cradle; because the
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
- Integral floor datum and envelope: floor world `Y=0`; LM axis world
  `Y=200.981`, therefore exact LM-axis-to-floor **200.981 mm**. The LM owns a
  full-depth W64 stem softly integrated into the lower cap, a W64 × 18.3 mm
  rectangular foot spanning `z=-150..18.3`, and a true R12 internal root.
  A W64 × 44 × 4 mm rear panel spans `z=-150..-146`; its Ø31 NL8 cutout is
  centered at `(0,22)` and four Ø3.2 holes use a 29.2 mm square. A necessary
  connector service cavity occupies x=±18, y=4..48, z=-146.2..-104. The
  only other outer-foot subtractions are three buried continuation lumens:
  LM Ø9, UM Ø8.2, and shared T Ø6, each with a minimum R14 turn. There is
  no yoke, open rail, secondary floor-support print, support screw, or support
  insert. The canonical floor LM is intentionally large-format; the optional
  keyed bottom inherits the entire stand. Its broad universal front shoulder
  is limited to `z=6.8..18.3`; the W64 stem remains the sole deep load path.
- Integral floor strength screen: closed-form exact rectangle-minus-circles
  root section, 4.0 kg mass at y=230 mm and 70 mm rear eccentricity, checked
  at 1g/3g/5g. This is **not FEA, certification, or physical qualification**.
  All stresses include an explicit 1.25 root geometry/model factor.
  Project-allowable vertical 1g/3g/5g safety factors and 1g diagnostic
  deflections are Bambu PLA Tough+ 3.05/1.97/1.18 and 1.18 mm; PLA Basic
  4.39/2.78/1.67 and 1.05 mm; PLA Lite 2.69/1.73/**1.04** and 1.40 mm; PLA
  Matte 2.78/1.79/1.08 and 1.49 mm; PLA Silk+ 3.23/2.09/1.25 and 1.17 mm.
  Lite is provisional pending a product-specific official TDS and fails the
  vertical-5g threshold, so it is not accepted by this screen. The section
  result requires a 100% local-solid modifier through the complete stem/root;
  sparse infill gets no structural credit. Magnets and the concealed split
  key receive 0 N credit. Analytical pass does not authorize service:
  physical gates are
  2× service load for 24 h at 35 °C with no crack/whitening and residual set
  ≤0.5 mm or ≤10% of loaded deflection, plus 1.5× service load for at
  least 168 h. Free-standing tip thresholds are only 0.139g lateral, 0.348g
  rearward, and 0.384g forward, so a positively attached anti-tip tether or
  anchor is mandatory.
- Cable voids: the printed UM passage is nominal Ø8.2 and exists only in the
  LM carrier. The printed T passage is nominal Ø6.0 and exists in the LM and
  UM carriers only. Their complete physical validation solids remain UM Ø7.0
  and T Ø5.2 across both buried and free spans. The LM physical envelope is
  Ø7.8 inside the nominal Ø9 route.
- Routing intent: no-floor LM/T/UM enter wholly inside the D20 support opening
  in an LM-above, T-lower-left, UM-lower-right layout. LM follows the buried
  Ø9 branch to the common R14 outlet.
  Floor state continues LM/UM/T into the three buried integral-stem lanes.
  UM rises inside the right LM arc, exits the LM-owned buried
  passage, and continues free behind the UM carrier. T rises inside the left
  LM arc, remains buried through the UM carrier, then exits and continues
  free behind the tweeter crescent. The LM-owned UM/T lumens finish at R112.95
  and their covers at R113.75 beneath the continuous visible R113.8 carrier
  exterior. The 0.05 mm solid owner land yields a 0.85 mm outside skin with
  no groove. Their
  physical centerlines cross at 82.95 deg with T higher in +Z, UM lower, and
  a 2.00 mm physical-envelope gap. There is no printed UM-owner arc at the crossing, no two-duct
  separator web, and no crescent-owned T arc.
  All eight named insert bypasses are smooth local Z dips with continuous
  closed cover and a full-width solid saddle from conduit roof to the
  applicable blind-bore floor. The saddle does not extend behind its conduit
  bump. Continuous full-width longitudinal burial webs back the LM-owned
  UM/T low runs and the UM-owned T low run to their seat membranes; in
  particular, neither longitudinal shoulder at the UM 328°/58° bypasses may
  contain a trapped cavity outside the exact D6 lumen, blind-bore,
  captive-magnet and half-lap interface voids. Every surviving buried span retains a 0.8 mm minimum
  wall and 0.85 mm seat roof; no trapped roof-to-bore cavity, bore-jump, or
  unintended rear cable window is permitted. Printed ownership ends in
  plain flush mouths: UM becomes free after its LM-owned passage, and T
  becomes free after its UM-owned passage. No telescoping printed handoff,
  UM-carrier rear duct, or crescent cable arc is permitted.
- Junction closure intent: the LM–UM and T–UM cusp regions are filled by
  complementary plan-split solids spanning the complete z=6.8..18.3 depth.
  LM owns lower LM–UM, UM owns upper LM–UM plus lower T–UM, and the tweeter
  crescent owns upper T–UM. Owners overlap their own ring/crescent by 0.40 mm
  while every local anti-void lens has a separate 0.45 mm slicable fusion
  land; Boolean shards below 0.05 mm² are rejected. The complementary owners
  preserve the 0.05 mm assembly seam. Their front faces are exactly coplanar
  at z=18.3; no shallow patch or rear cavity is allowed. Fixed-window/frozen-
  silhouette acceptance runs at every 0.16 mm print layer and both sides of
  every half-lap transition. The central ±6 mm T free-cable mouth is the sole
  non-functional open span. Functional bosses at both LM-to-UM and
  UM-to-tweeter are explicitly Z-owned rather than plan-split: each base
  closure teardrop remains nominal Ø9 and every complete cylindrical
  functional boss is locally Ø9.8. LM owns complete rear Ø3.4 ears and UM
  complete front Ø4.6 receiver ears at the lower joint; UM owns complete rear
  Ø3.4 ears and the crescent complete front Ø4.6 receiver ears at the upper
  joint. Each opposing print is fully notched over the other Z-half. The plan
  seam must not bisect a cylindrical wall or receiver floor, and both separate
  0.20 mm axial gaps must remain open.
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
  Obi-Wan emits no printed grommet or positive strain-relief component; the UM
  lead is already free behind the UM carrier. That absence is distinct from
  the short LM lead's minimal subtractive rear-open cable clearance. Cable
  retention and physical terminal fit remain physical checks.
- Qualification status: `PHYSICAL_MEASURE_REQUIRED = True`. The MU
  reference omits both terminals and the datasheet leaves their carrier
  and withdrawal geometry un-dimensioned. The modeled 12 mm pull equals
  the provisional 12 mm exposed tab length, so it has zero positive
  release overtravel margin. Terminal, boot, cable, Y breakout, strain relief,
  and one-at-a-time removal qualification therefore remain
  pending a recorded real-driver dry fit. Floor and no-floor candidates also
  require separate artifact/process identity, coupon evidence, documented
  1g/3g/5g structural proof and signed release decisions; neither state may
  inherit the other's evidence. The optional LM split additionally remains
  fail-closed pending two-pin/socket fit, slicer-path proof, full-seat,
  coplanarity and route-seam
  evidence, physical cable pull-through, and driver-installed 1g/3g/5g proof;
  monolithic-LM evidence does not qualify it by inheritance.
- Manufacturing assumptions: Bambu PLA Tough+, PLA Basic, PLA Lite, PLA
  Matte, or PLA Silk+; 0.4 mm nozzle; at least six walls and a **100% local-
  solid modifier through the complete integral floor stem/root**.
  Material/process selection remains subject to its own
  qualification record. A
  non-load-bearing skin on every surviving buried route starts at two full extrusion widths
  (0.8 mm). Structural rails/bosses use separately calculated sections.
- Primary paths: `top_baffle_nd25fw4_obiwan*.py`; generated STEP files in
  `floor_stand/` and `no_floor_stand/`; print meshes in each `stl/` tree;
  optional split review at `*/top_baffle_nd25fw4_obiwan_lm_split.step` and its
  two `lx521_top_obiwan_optional_lm_keyed_*` meshes;
  routing sheets at `*/baffle_cable_routing_obiwan.png`, including plan,
  longitudinal side profiles, and nominal diametric u-z sections with exact
  vertical backfill limits through
  representative surviving UM/T conduit-plus-pilot bumps and explicit free
  rear UM/tweeter spans; candidate provenance
  in each `obiwan_release_manifest.json`; authoritative print pauses and
  polarity in `review/captive_magnet_slice_audit/CAPTIVE_MAGNET_PAUSE_MANIFEST.md`; physical evidence and per-state
  signoff in `obiwan_physical_qualification.md`.
- Validation: exact insert and bridge coordinates; opening-free front web,
  rear-entry bore/front-floor depth, and zero rear protrusion; exact
  x=±32.0/y=315.770 LM-to-UM axes; two complete standalone LM rear Ø3.4
  clearance passages, including the optional keyed LM top; two complete
  standalone UM rear-opening blind Ø4.6 x 4.0 insert receivers; nominal Ø9
  closure-web/base teardrops and complete local Ø9.8 Z-owned cylindrical
  functional bosses; 1.9 mm solid UM acoustic-front floors; 0.20 mm axial ear
  gaps; continuous rear-driven
  screw access; independent insert installation in the UM print; exact
  x=±24.0/y=421.5 UM-to-tweeter axes; two complete standalone UM rear Ø3.4
  passages and two complete standalone crescent rear-opening blind Ø4.6 x 4.0
  insert receivers; complete local Ø9.8 functional bosses, 1.9 mm crescent
  acoustic-front floors, and a 0.20 mm axial gap; independent insert
  installation in the crescent print; complete 360° receiver and
  clearance-bore walls at both interfaces with no cross-owner dependency; no
  washer/nut or front breakthrough; state isolation;
  exact floor/no-floor LM-lower exterior equality at the front and throughout
  `z=6.8..18.3`, common world `Y=0` lower extent, coincident station widths,
  and equal 0.20 mm wing saddle clearance in both states;
  six global Ø5.20 x 2.10 surface-normal captive magnet cavities with 0.45 mm
  axial skins, continuous printable cradles and 45-degree roofs; preservation
  of the upper 64/116 deg LM ring pair on the R113.0 structural carrier and
  the 50.5/129.5 deg UM pair on the R51.7 structural carrier; continuous
  smooth exposed side radii R113.8/R52.5, clipped only inside the existing
  LM--UM and T--UM cusp/service regions, with the 0.40 mm LM--UM
  inter-carrier gap preserved; ring cavity construction datums at
  structural radius +0.65 mm, 0.15 mm beneath the exposed surface, with no
  local pad, boss, flat, or visible cue;
  exact lower-LM
  base-side faces at `(x,y,z)=(±32,18,15.10)` with outward ±X normals in
  both stand states; at least 2.2 mm upper-LM nearest-insert edge gap; absence
  of LM proud ears; three matching Ac/Ae receivers per physical side at
  LM lower, LM upper, and UM; common source Z=15.10 for every LM/UM station;
  0.05 mm solid receiver construction standoff with zero physical mating gap;
  1.10 mm nominal paired magnet-face separation at LM-upper/UM and 0.95 mm at
  the straight LM-lower base-side pair; no exterior magnet-location cue;
  manifest-derived pauses and mirrored polarity;
  unobstructed D7.8 LM travel through the no-floor Ø9 D20 branch/R14 handoff
  and the corresponding floor-lane continuation;
  physical-cable containment; insert/head/terminal/service clearance;
  minimum normal wall and eroded-outline containment on printed-owner spans;
  positive absence of a printed UM-carrier rear duct and crescent-owned T
  arc; free-cable clearance, crossover separation and angle; G1 continuity
  and bend radii across printed-to-free handoffs;
  final-BREP solid saddle continuity from conduit roof to ordinary blind-bore
  floor at all eight named bumps;
  optional-LM seam Y/zero-gap butt, final-lumen preservation and non-overlapping
  halves; two symmetric concealed Ø1.60 world-+Y cylindrical pins, 2.40 mm
  engagement, right Ø1.84 round and left 1.96 × 1.84 X-relieved blind sockets,
  0.12 mm radial/0.25 mm end clearances, ≥0.50 mm local radial/end walls,
  ≥0.05 mm recess plan clearance, ≥0.13 mm conservative W22-flange plan
  clearance, R114.4036 worst-case land reach (1.4036 mm beyond structural R113.0;
  0.6036 mm beyond the finalized R113.8 visible fairing), round-plus-relieved pitch tolerance, complete four-nozzle-width
  sliced pin/land/wall paths, process-matched fit, actual U22 fit and full seating,
  explicit exterior support lands and zero standalone
  retention/load credit; front-face-down orientation and selected-printer
  footprint clearance for all optional split parts;
  exact floor Y=0 and LM-axis-to-floor 200.981 mm; integral W64 stem/foot,
  R12 root, NL8 panel/service cavity and three buried continuations; positive
  absence of a separate floor-support artifact; combined-axis 4 kg
  sustained-1g/3g/5g bridge/integral-stand screen for all five named Bambu
  materials, plus the
  actual 0.85 kg upper-joint load case; 256 mm bed fit; one valid
  solid per print part; and zero open or over-shared STL edges.
- Review handoff: state-specific snapshots of the two-carrier core, optional
  LM keyed split, attachments, assembled carrier and terminal-fit STEP, plus
  live CAD Viewer links. Routing
  diagrams are not substitutes for rendered STEP review.
- Generation safety: ordinary Make targets execute on `osado.lan` in one
  512 GiB/no-swap systemd cgroup with a 64 GiB host-available floor. The
  default eight guarded recipe slots are capped at 56 GiB per process tree.
  That profile uses concurrent check processes, direct final-part builds and
  full route witnesses; it does not inherit macOS-only cutter/route tiling.
  Explicit `LX_CAD_EXECUTION=local` remains serial with an 8 GiB process-tree
  RSS ceiling and no host-free-memory floor. A positive
  `LX_CAD_MIN_FREE_MB` may opt a local invocation into a stricter floor.
