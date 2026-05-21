# Preliminary Hypex-style filter listing generated from the synthetic CDSL model
# Topology: cascaded mixed-order LR split tree; later drivers include upstream high-pass stages in this per-channel export.
# LR2 splits invert the next/downstream branch; polarity is exported per driver.
# Diagram:
# Input
#   +-- LR4 HP 70 Hz (2 biquads, global boundary)
#     +-- LR4 split @ 200 Hz
#         +-- LR4 LP 200 Hz (2 biquads) -> L26RO4Y
#         +-- LR4 HP 200 Hz (2 biquads) -> next split
#       +-- LR4 split @ 800 Hz
#           +-- LR4 LP 800 Hz (2 biquads) -> L22MG (nude)
#           +-- LR4 HP 800 Hz (2 biquads) -> next split
#         +-- LR4 split @ 2500 Hz
#             +-- LR4 LP 2500 Hz (2 biquads) -> GRS PT6816
#             +-- LR4 HP 2500 Hz (2 biquads) -> ND25FW4 (nude 18mm)

[L26RO4Y]
Gain=-13.295 dB
Delay=0.0000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=lowpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter4=lowpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter5=peaking, Fc=110.000, Q=1.0000, Gain=-1.200, Source=flat-EQ
Filter6=lowshelf, Fc=95.000, Q=1.0000, Gain=-0.534, Source=flat-EQ

[L22MG (nude)]
Gain=12.533 dB
Delay=-0.4800 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter5=lowpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter6=lowpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter7=peaking, Fc=500.000, Q=1.0000, Gain=-4.535, Source=flat-EQ
Filter8=peaking, Fc=330.000, Q=1.0000, Gain=3.239, Source=flat-EQ
Filter9=peaking, Fc=620.000, Q=1.0000, Gain=-0.497, Source=flat-EQ

[GRS PT6816]
Gain=4.024 dB
Delay=0.0250 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter6=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter7=lowpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter8=lowpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter9=peaking, Fc=2800.000, Q=1.0000, Gain=-0.373, Source=flat-EQ

[ND25FW4 (nude 18mm)]
Gain=1.160 dB
Delay=-0.1250 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter8=highpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter9=highshelf, Fc=18000.000, Q=1.0000, Gain=8.000, Source=flat-EQ
Filter10=peaking, Fc=10000.000, Q=2.0000, Gain=6.310, Source=flat-EQ
Filter11=peaking, Fc=12000.000, Q=1.0000, Gain=-4.590, Source=flat-EQ
Filter12=peaking, Fc=16500.000, Q=1.0000, Gain=3.830, Source=flat-EQ
Filter13=peaking, Fc=9000.000, Q=2.0000, Gain=-3.108, Source=flat-EQ
Filter14=peaking, Fc=14000.000, Q=1.0000, Gain=-1.310, Source=flat-EQ
