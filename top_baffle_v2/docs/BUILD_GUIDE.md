# Build your baffle on H2C

**H2C is the only supported printer: two 0.6 mm High Flow nozzles.**
Start with the [H2C file catalog](../to_print/README.md). Choose a baffle,
stand state, upper arrangement and matching optional wings for each speaker.
P2S builds are deprecated.

![All five tweeter families](../images/generated/iso/rows/tweeter_row.png)

| Baffle | Without floor stand | With integrated floor stand | Upper choices |
|---|---|---|---|
| Stock | One LM + one upper | Two LM sections + one upper | ND25FW-4, opposed TEBM35C10-4 BMR or ND25FN-4 vase |
| Slim | One LM + one upper | Two LM sections + one upper | ND25FW-4 or opposed TEBM35C10-4 BMR |
| Obiwan | One LM + selected upper | One LM/stand + selected upper; NL8 service lid | Regular UM + ND25FW-4 or BMR crescent; ND25FN-4 vase top on the keyed LM bottom; or a PT2522-4 / AMT17D2.1 dipole piece |

The ND25FN-4 vase top also needs its two twist-lock caps; it is common to both
stand states and keys onto the keyed LM bottom. Select the matching ND25FN-4
wings; regular-UM wings fit the ND25FW-4 and BMR arrangements instead. The
dipole parts take no wings ([dipole guide](DIPOLE_TWEETERS.md)).
H2C provides one continuous Obiwan wing per side in flat or graded form.

![Upper choices sharing the LM interface](../images/generated/iso/rows/obiwan_upper_row.png)

![Matching H2C wings](../images/generated/iso/rows/obiwan_wing_row.png)

Use the catalog's material-specific editable 3MF or audited sliced project.
An STL has no supports, material assignments, infill modifiers or insertion
pauses. Follow each project's measured magnet pauses; do not transfer pause
heights between different orientations or materials.

Regular and ND25FN-4 interfaces use buried Ø5×2 mm N52 magnets (the ND25FN-4
vase top's two LM-ring magnets sit at 9/171°). Standard M3
inserts are Hanglife M3×5×4 in Ø4.6×4 mm printed pilots. BMR driver mounts
retain their documented M2 exception. Match polarity before closing pockets.

Use the [printing guide](PRINTING.md), [five-family selection](TWEETER_OPTIONS.md),
[ND25FN-4 assembly guide](DAYTON_ND25FN4_WAVEGUIDE.md) and [dipole guide](DIPOLE_TWEETERS.md) for the current
process and hardware sequence. CAD and slice checks do not replace physical
fit, retention, adhesion or loaded-stand qualification.

The [archived build notes](archive/P2S_BUILD_GUIDE.md) preserve earlier joint
and hardware history. Their split counts, print files and commands are obsolete.
