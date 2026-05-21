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
#           +-- LR2 split @ 8000 Hz
#               +-- LR2 LP 8000 Hz (1 biquad) -> GRS PT6816
#               +-- LR2 HP 8000 Hz (1 biquad) -> ND25FW4 (nude 18mm)

[L26RO4Y]
Gain=-7.843 dB
Delay=0.0000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter4=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass

[L22MG (nude)]
Gain=14.951 dB
Delay=-1.3750 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter5=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter6=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter7=peaking, Fc=500.000, Q=1.0000, Gain=-5.446, Source=flat-EQ
Filter8=peaking, Fc=620.000, Q=1.0000, Gain=1.496, Source=flat-EQ
Filter9=peaking, Fc=150.000, Q=1.0000, Gain=1.377, Source=flat-EQ
Filter10=peaking, Fc=220.000, Q=1.0000, Gain=0.964, Source=flat-EQ
Filter11=peaking, Fc=330.000, Q=1.0000, Gain=0.293, Source=flat-EQ

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
Filter8=peaking, Fc=1450.000, Q=1.0000, Gain=8.000, Source=flat-EQ
Filter9=peaking, Fc=1800.000, Q=1.0000, Gain=-5.184, Source=flat-EQ
Filter10=peaking, Fc=1150.000, Q=1.0000, Gain=-2.634, Source=flat-EQ
Filter11=peaking, Fc=900.000, Q=1.0000, Gain=-1.761, Source=flat-EQ
Filter12=peaking, Fc=700.000, Q=1.0000, Gain=1.701, Source=flat-EQ

[GRS PT6816]
Gain=-5.466 dB
Delay=-0.7450 ms
Polarity=-1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2000.000, Q=0.5000, Gain=0.000, Source=LR2 branch high-pass
Filter8=lowpass, Fc=8000.000, Q=0.5000, Gain=0.000, Source=LR2 branch low-pass
Filter9=peaking, Fc=4500.000, Q=1.0000, Gain=7.874, Source=flat-EQ
Filter10=peaking, Fc=3500.000, Q=1.0000, Gain=-6.819, Source=flat-EQ
Filter11=peaking, Fc=2800.000, Q=1.0000, Gain=4.598, Source=flat-EQ
Filter12=peaking, Fc=10000.000, Q=2.0000, Gain=-2.492, Source=flat-EQ
Filter13=peaking, Fc=9000.000, Q=2.0000, Gain=-1.933, Source=flat-EQ
Filter14=peaking, Fc=8500.000, Q=2.0000, Gain=0.717, Source=flat-EQ
Filter15=peaking, Fc=9000.000, Q=1.0000, Gain=2.500, Source=flat-EQ

[ND25FW4 (nude 18mm)]
Gain=4.261 dB
Delay=-0.7150 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2000.000, Q=0.5000, Gain=0.000, Source=LR2 cascaded upstream high-pass
Filter8=highpass, Fc=8000.000, Q=0.5000, Gain=0.000, Source=LR2 branch high-pass
Filter9=peaking, Fc=14000.000, Q=1.0000, Gain=-6.413, Source=flat-EQ
Filter10=peaking, Fc=16500.000, Q=1.0000, Gain=5.217, Source=flat-EQ
Filter11=peaking, Fc=10000.000, Q=2.0000, Gain=4.538, Source=flat-EQ
Filter12=peaking, Fc=9000.000, Q=2.0000, Gain=-2.298, Source=flat-EQ
Filter13=highshelf, Fc=18000.000, Q=1.0000, Gain=1.652, Source=flat-EQ
Filter14=peaking, Fc=12000.000, Q=1.0000, Gain=0.263, Source=flat-EQ
