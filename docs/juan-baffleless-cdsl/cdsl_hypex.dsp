# Preliminary Hypex-style filter listing generated from the synthetic CDSL model
# Topology: cascaded mixed-order LR split tree; later drivers include upstream high-pass stages in this per-channel export.
# LR2 splits invert the next/downstream branch; polarity is exported per driver.
# Diagram:
# Input
#   +-- LR4 HP 70 Hz (2 biquads, global boundary)
#     +-- LR4 split @ 120 Hz
#         +-- LR4 LP 120 Hz (2 biquads) -> L26RO4Y
#         +-- LR4 HP 120 Hz (2 biquads) -> next split
#       +-- LR4 split @ 650 Hz
#           +-- LR4 LP 650 Hz (2 biquads) -> L22MG (nude)
#           +-- LR4 HP 650 Hz (2 biquads) -> next split
#         +-- LR2 split @ 2000 Hz
#             +-- LR2 LP 2000 Hz (1 biquad) -> SS10F8414G10
#             +-- LR2 HP 2000 Hz (1 biquad, invert downstream branch) -> next split
#           +-- LR2 split @ 12000 Hz
#               +-- LR2 LP 12000 Hz (1 biquad) -> GRS PT6816
#               +-- LR2 HP 12000 Hz (1 biquad) -> ND25FW4 (nude 18mm)

[L26RO4Y]
Gain=-7.787 dB
Delay=0.0000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter4=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass

[L22MG (nude)]
Gain=16.927 dB
Delay=-1.3750 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter5=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter6=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter7=peaking, Fc=500.000, Q=1.0000, Gain=-5.960, Source=flat-EQ

[SS10F8414G10]
Gain=-3.297 dB
Delay=-0.6900 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter7=lowpass, Fc=2000.000, Q=0.5000, Gain=0.000, Source=LR2 branch low-pass
Filter8=peaking, Fc=1450.000, Q=1.0000, Gain=4.592, Source=flat-EQ
Filter9=peaking, Fc=1800.000, Q=1.0000, Gain=-2.842, Source=flat-EQ
Filter10=peaking, Fc=700.000, Q=1.0000, Gain=-5.788, Source=flat-EQ

[GRS PT6816]
Gain=-6.102 dB
Delay=-0.7300 ms
Polarity=-1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2000.000, Q=0.5000, Gain=0.000, Source=LR2 branch high-pass
Filter8=lowpass, Fc=12000.000, Q=0.5000, Gain=0.000, Source=LR2 branch low-pass
Filter9=peaking, Fc=4500.000, Q=1.0000, Gain=7.062, Source=flat-EQ
Filter10=peaking, Fc=3500.000, Q=1.0000, Gain=-7.008, Source=flat-EQ
Filter11=peaking, Fc=11000.000, Q=1.0000, Gain=8.000, Source=flat-EQ
Filter12=peaking, Fc=2800.000, Q=1.0000, Gain=6.499, Source=flat-EQ

[ND25FW4 (nude 18mm)]
Gain=8.704 dB
Delay=-0.7000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2000.000, Q=0.5000, Gain=0.000, Source=LR2 cascaded upstream high-pass
Filter8=highpass, Fc=12000.000, Q=0.5000, Gain=0.000, Source=LR2 branch high-pass
Filter9=peaking, Fc=14000.000, Q=1.0000, Gain=-3.562, Source=flat-EQ
