# Follow-up: what H2D adds, and smaller changes that remove more pieces

14 September 2026. This extends the [initial study](README.md), which deliberately assumed a 5 mm brim and kept the complete job within the area shared by the two nozzles. Production CAD and printer files remain unchanged.

**H2D does not unlock an additional recommended split over H2C.** Its useful size advantage is 20 mm more right-nozzle reach. H2C already has the same 325 × 320 mm area on the left, and both offer the same 300 × 320 mm shared area. H2D therefore gives more freedom to put the model material on either side when using the larger area. It does not give a larger shared PETG/PLA footprint. See the [official H2D machine profile](https://raw.githubusercontent.com/bambulab/BambuStudio/master/resources/profiles/BBL/machine/Bambu%20Lab%20H2D%200.4%20nozzle.json) and the H2C area diagrams in [Bambu's manual, pages 90–91](https://csm.bblcdn.com/hub/eff78da43720461787dc8bbe5fa0372d.pdf).

The H2C can also assign materials to separate interchangeable right-side hotends. That bank has 305 × 320 mm reach. It is a different arrangement from using the fixed left and right nozzles together; Bambu describes left/right switching as faster than exchanging induction hotends because the latter still involves AMS loading and unloading. [H2C manual, pages 77–86](https://csm.bblcdn.com/hub/eff78da43720461787dc8bbe5fa0372d.pdf).

For your alternating PETG-GF, translucent PETG and PLA jobs, separate assigned H2C hotends could reduce cleaning between those materials across jobs as well as within a job. That is an inference from the dedicated-hotend mechanism, not a measured waste or reliability comparison. H2D can already dedicate one nozzle to the model material and the other to PLA. I would prefer H2D if its price saving matters more than the additional hotends; I would not prefer it for a claimed reduction in this project's piece count.

## Smallest promising change: Stock/Slim LM in one piece

The current complete LM reaches **Y321.95**, partly because its two top registration keys project **6 mm above the actual Y315.95 joint**. Replace those projecting keys with registration contained inside the existing part envelope and the complete lower body becomes approximately **304.8 × 315.95 mm**. Driver positions and the exterior outline can remain unchanged.

Rotated 90 degrees on the **325 × 320 mm area**, with a **2 mm brim**, its footprint is **319.95 × 308.80 mm**. That leaves **2.53 mm edge clearance on the limiting axis**, and space for a separate 50 × 50 mm prime-tower reserve in the shared area. The envelope passes for Stock and Slim, with and without the floor stand.

This could take Stock/Slim from **3 LM pieces to 1**, and the core baffle from **4 pieces to 2: complete LM plus upper vase**.

The qualifications matter:

- The 2 mm brim is a proposed adhesion experiment for the glued plate, not a claim that it is equivalent to the current 5 mm brim.
- On H2C, use the left nozzle for this model envelope. On H2D, either nozzle has the necessary reach. PLA support and priming must remain reachable by their assigned nozzle.
- Registration needs a proper contained/rear joint design and a matching upper module. Simply deleting or flipping the existing dovetails is not a mounting solution: flipping them downward can intrude into the driver-seat region.
- Keeping the full 5 mm brim and forcing a rectangular top limit at Y311 would cut the protected driver-seat envelope, which reaches **Y311.581**. That shortcut is rejected.

This is the most promising small adjustment. It avoids scaling the entire model, changing driver cutouts, or moving the driver centers. It is more conditional than the original shared-area two-piece LM proposal.

## Obiwan: fewer pieces without making the drivers smaller

The initial recommendation already produces one complete LM and one full detachable wing per side. There is no need to shrink those parts.

**A simple additional consolidation is to fuse the regular UM and regular crescent**, as already done in V4. Their existing combined envelope fits even the present 256 mm printer. With detachable wings retained, regular Obiwan would then have **4 main pieces**, matching V4: LM, fused upper, left wing and right wing.

If minimum piece count takes priority over removable wings, a more ambitious configuration is possible:

| Arrangement | Main pieces per speaker | Envelope finding |
|---|---:|---|
| Regular Obiwan, permanent wings | **2** | LM + lower wing regions; fused UM/crescent + upper wing regions. Both fit the shared 300 × 320 area with 5 mm brims. |
| V4 Obiwan, permanent wings | **Potentially 2** | The corresponding two envelopes fit the 325 × 320 nozzle area with 2 mm brims. They fail the original shared-area/5 mm-brim rule. |

A common wing cut at approximately **Y222** works for both stand states of regular Obiwan. On the no-floor version, the two rotated body envelopes are approximately **273.7 × 293.6 mm** and **274.0 × 293.9 mm**, before brims. The floor version also fits. The proposed division leaves the whole LM ring on one piece; it divides the lateral wing material separately.

For V4, a common wing cut around **Y257** gives the more conditional larger-nozzle-area option. Using one cut for both stand states preserves an identical upper-body envelope across those states. Both variants retain the current driver positions.

These permanent-wing options are a product redesign, not merely a slicer merge. Wing-to-carrier clearances must become solid material; obsolete magnetic joints and split features need removal; the new side joints and upper-module installation path need design. A 2 mm brim and a prime-tower reserve fitting on the plate do not qualify all generated supports. Caps and retainers remain separate for driver access. Counts above exclude them and all hardware.

## What should stay fixed

Keep the driver seats and center spacing. Current LM/UM centers are **165.1 mm apart**; the clearance between their protected circular seat envelopes is only **5.2 mm** before surrounding wall material. A tens-of-millimetres reduction in spacing is therefore not a small cosmetic adjustment.

The full current baffles still do not become front-face-down monoliths through a few millimetres of trimming. The useful opportunities are contained registration, an appropriate brim, and choosing which surfaces belong to each printed part.

## Evidence

[Adjustment results](adjustments.json) include the unchanged-driver envelope checks, brim comparisons, local side-trim thresholds, wing seam sweeps and tower reserves. [Reproduction script](adjustment_study.py) reads the same hash-checked STLs as the initial study. Convex projected outlines are conservative envelope screens, not CAD solids or sliced support footprints.

The side-trim sweep held the top joint and base fixed; it did not find a width-only clipping solution inside the shared-area rules without exceeding the protected-seat clipping limit. This does not rule out a more extensive curved-outline redesign. The regular two-piece permanent-wing option, and the conditional Stock/Slim one-piece LM option, are the useful next design targets from this follow-up.
