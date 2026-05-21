# Preliminary Hypex-style filter listing generated from the synthetic CDSL model
# Topology: cascaded LR4 split tree; later drivers include upstream high-pass stages in this per-channel export.
# Diagram:
# Input
#   +-- LR4 HP 70 Hz (2 biquads, global boundary)
#     +-- split @ 200 Hz
#         +-- LR4 LP 200 Hz (2 biquads) -> L26RO4Y
#         +-- LR4 HP 200 Hz (2 biquads) -> next split
#       +-- split @ 800 Hz
#           +-- LR4 LP 800 Hz (2 biquads) -> L22MG (nude)
#           +-- LR4 HP 800 Hz (2 biquads) -> next split
#         +-- split @ 2500 Hz
#             +-- LR4 LP 2500 Hz (2 biquads) -> GRS PT6816
#             +-- LR4 HP 2500 Hz (2 biquads) -> ND25FW4 (nude 18mm)

[L26RO4Y]
Gain=-7.467 dB
Delay=0.0000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=lowpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter4=lowpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter5=peaking, Fc=110.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter6=lowshelf, Fc=95.000, Q=1.0000, Gain=-1.464, Source=flat-EQ

[L22MG (nude)]
Gain=11.244 dB
Delay=-1.3600 ms
Polarity=-1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter5=lowpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter6=lowpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter7=peaking, Fc=150.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter8=peaking, Fc=330.000, Q=1.0000, Gain=7.426, Source=flat-EQ
Filter9=peaking, Fc=500.000, Q=1.0000, Gain=-5.635, Source=flat-EQ
Filter10=peaking, Fc=220.000, Q=1.0000, Gain=-4.504, Source=flat-EQ
Filter11=peaking, Fc=620.000, Q=1.0000, Gain=1.406, Source=flat-EQ

[GRS PT6816]
Gain=1.746 dB
Delay=-0.8550 ms
Polarity=-1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter6=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter7=lowpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter8=lowpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter9=peaking, Fc=2200.000, Q=1.0000, Gain=8.000, Source=flat-EQ
Filter10=peaking, Fc=2800.000, Q=1.0000, Gain=-8.000, Source=flat-EQ

[ND25FW4 (nude 18mm)]
Gain=-0.719 dB
Delay=-0.9350 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=200.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=800.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter8=highpass, Fc=2500.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter9=peaking, Fc=14000.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter10=peaking, Fc=16500.000, Q=1.0000, Gain=7.377, Source=flat-EQ
Filter11=highshelf, Fc=18000.000, Q=1.0000, Gain=6.642, Source=flat-EQ
Filter12=peaking, Fc=12000.000, Q=1.0000, Gain=4.504, Source=flat-EQ
