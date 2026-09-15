# Measured print-quality screening

Measurements use the current delivered STL bytes, 2,000 ray samples per mesh and a 45° FDM support-angle screen. Raw measurements and six-axis orientation comparisons are in this folder.

| Current carrier | Body | Sample minimum (mm) | p05 (mm) | Down-facing area below 45° |
|---|---|---|---|---|
| obiwan_01_LM_bottom_keyed_1_of_2_no_floor_stand | outer material body 0 | 0.762 | 0.8 | 13.2% |
| obiwan_02_LM_top_keyed_2_of_2 | outer material body 0 | 0.543 | 0.797 | 20.6% |
| obiwan_03_UM_carrier_1_of_1 | outer material body 0 | 0.039 | 0.798 | 16.0% |

The outer-body p05 values violate the generic FDM screening recommendations of 1.2 mm supported / 1.6 mm unsupported walls (DfAM skill `references/process-limits.md`, FDM minimum-wall rows). This is a documented thin-cover exception awaiting process evidence, not a demonstrated physical failure of the complete carrier. The smallest sampled values can occur at edges; they do not locate a reliable global minimum.

The two nested magnet-void bodies in each carrier have zero valid ray samples. No measured magnet-skin minimum is claimed. Section sacrificial production parts to verify the source 0.52 mm skins and actual magnet seating.

Concrete remediation if preview/physical coupons fail: thicken the offending non-mating cover toward 1.2 mm supported / 1.6 mm unsupported while retaining its lumen. At pin sockets, grow material only where driver/recess and wing clearance allow; retain the exact pin pitch and repeat the fit and interference audits. Do not consume functional cable clearance to achieve a wall-number target.

For the LM top, rotating onto an edge reduces the measured support-area proxy from 20.6% to 10.8%, but raises build height from 23.64 mm to 212.10 mm. It invalidates the qualified front-down magnet-loading/roof/pause workflow and is not a drop-in orientation change.

See [PETG_GF_QUALIFICATION.md](../../docs/PETG_GF_QUALIFICATION.md) for the test plan. No physical results have been entered.
