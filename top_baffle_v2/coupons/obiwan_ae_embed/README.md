# Obi-Wan graded embedded-magnet coupon

The directory keeps its frozen `obiwan_ae_embed` name from the wing vocabulary
that spelled the graded family `Ae`; like every other coupon it is a
diagnostic identity and is never renamed with the product words.

This is a deliberately small two-piece process coupon: one carrier-side strip
and one matching graded-wing-side strip. It combines the released upper-LM and UM
magnet stations into each strip while preserving their R113/R51.7 interface
curvatures, coupon-specific zero modeled plastic-interface gap, D5.2 diametral
clearance, root-envelope widths, and installed magnet heights. “Zero” here
does not mean exposed magnets touch: the two 0.45 mm skins still separate the
coupon magnets by 0.90 mm. Released production mating surfaces are flush with
zero physical air gap: their 0.05 mm receiver construction offset is solid
plastic, not a local exterior cut. The Obi-Wan LM-upper/UM ring cavity datum
is also buried 0.15 mm beneath its smooth carrier surface, giving
**1.10 mm nominal magnet-to-magnet separation**. The current LM-lower shoulder
pair uses the same 0.15 mm inset and **1.10 mm** stack, although this
two-radius strip coupon does not reproduce that shoulder shape. This coupon
qualifies the captive printing process, not production pull force.

The coupon targets D5 x 2 mm magnets. Print both coupon strips
front-face-down, matching the production texture/process direction. At each
pause, drop the magnets vertically through the open loading chimneys and use
the polarity directions below. For a constant 0.20 mm profile:

1. Let the fully open LM and UM cavity layer at Z = 5.80 mm finish. Put the
   pause on the following, first-closing layer at Z = 6.00 mm. Insert all four
   magnets.
2. Resume after checking that every magnet is fully seated below the current
   layer and cannot rise toward the toolhead.

With Bambu's 0.16 mm profile and its 0.20 mm initial layer, the corresponding
common first-closing layer is Z = 5.96 mm for LM and UM. Use the generated
manifest pause rather than trusting a nominal layer number; the pipeline
derives it from the actual sliced roof onset. The cavity roof closes at 45
degrees and leaves no access opening in the finished part.

For attraction, mark one pole on every magnet before printing. Insert all four
magnets with that marked pole pointing in the same assembled +X direction:
toward the functional curved face in the carrier piece, and away from the
functional concave face in the graded piece.

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
