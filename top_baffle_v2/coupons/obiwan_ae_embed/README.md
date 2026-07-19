# Obi-Wan Ae embedded-magnet coupon

This is a deliberately small two-piece process coupon: one carrier-side strip
and one matching Ae-wing-side strip. It combines the released upper-LM and UM
magnet stations into each strip while preserving their R113/R51.7 interface
curvatures, coupon-specific zero modeled plastic-interface gap, D5.2 diametral
clearance, root-envelope widths, and installed magnet heights. “Zero” here
does not mean exposed magnets touch: the two 0.45 mm skins still separate the
coupon magnets by 0.90 mm. Released production interfaces preserve their
approximately 0.05 mm mating air gap. The Obi-Wan LM-upper/UM ring cavity
datum is also buried 0.15 mm beneath its smooth carrier surface, giving
**1.10 mm nominal magnet-to-magnet separation**; the LM-lower base-side pair
has no datum inset and remains **0.95 mm**. This coupon qualifies the captive
printing process, not production pull force.

The coupon targets D5 x 2 mm magnets. Print both coupon strips
front-face-down, matching the production texture/process direction. At each
pause, drop the magnets vertically through the open loading chimneys and use
the polarity directions below. For a constant 0.20 mm profile:

1. Let the fully open UM cavity layer at Z = 5.80 mm finish. Put the pause on
   the following, first-closing layer at Z = 6.00 mm. Insert the UM magnet in
   both pieces.
2. Let the fully open LM cavity layer at Z = 8.40 mm finish. Put the pause on
   the following, first-closing layer at Z = 8.60 mm. Insert the LM magnet in
   both pieces.
3. Resume after checking that every magnet is fully seated below the current
   layer and cannot rise toward the toolhead.

With Bambu's 0.16 mm profile and its 0.20 mm initial layer, the corresponding
first-closing layers are Z = 5.96 mm (UM) and Z = 8.52 mm (LM). Use the slicer
preview to place each pause immediately before the first inward roof line,
rather than trusting a nominal layer number. The cavity roof closes at 45
degrees and leaves no access opening in the finished part.

For attraction, mark one pole on every magnet before printing. Insert all four
magnets with that marked pole pointing in the same assembled +X direction:
toward the functional curved face in the carrier piece, and away from the
functional concave face in the Ae piece.

Calibration parameters live at the top of `obiwan_ae_embed_coupon.py`. This
coupon uses a 0.45 mm nominal axial retaining skin on both sides of the magnet
and a 2.10 mm radial cavity. The finished disc is fully buried without glue or
an external access opening. The 0.45 mm skin is deliberate: the earlier
0.30 mm version existed in the STL but was removed by Bambu Studio's Classic
wall generator with a 0.4 mm nozzle. Always confirm that the Preview shows one
continuous extrusion line down each long side of every open cavity before
printing.

The released carrier uses a cavity datum at structural radius +0.65 mm,
0.15 mm beneath a continuous exposed +0.80 mm ring fairing. The fairing is
clipped only inside the existing LM--UM and T--UM cusp/service regions and
contains no local pad, boss,
flat, protrusion, or visible magnet-location cue. This full-carrier clearance
and surface contract is validated in CAD rather than by the strip coupon. A
thinner 0.30 mm
skin remains possible as an Arachne-only calibration variant, but it must not
be printed from a Preview in which those retaining paths disappear.
