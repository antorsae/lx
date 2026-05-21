# Preliminary Hypex-style filter listing generated from the synthetic CDSL model
# Topology: cascaded LR4 split tree; later drivers include upstream high-pass stages in this per-channel export.
# Diagram:
# Input
#   +-- LR4 HP 70 Hz (2 biquads, global boundary)
#     +-- split @ 120 Hz
#         +-- LR4 LP 120 Hz (2 biquads) -> L26RO4Y
#         +-- LR4 HP 120 Hz (2 biquads) -> next split
#       +-- split @ 650 Hz
#           +-- LR4 LP 650 Hz (2 biquads) -> L22MG (nude)
#           +-- LR4 HP 650 Hz (2 biquads) -> next split
#         +-- split @ 3200 Hz
#             +-- LR4 LP 3200 Hz (2 biquads) -> SS10F8414G10
#             +-- LR4 HP 3200 Hz (2 biquads) -> GRS PT6816

[L26RO4Y]
Gain=-9.856 dB
Delay=0.0000 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter4=lowpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass

[L22MG (nude)]
Gain=17.785 dB
Delay=-1.3750 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter5=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter6=lowpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter7=peaking, Fc=500.000, Q=1.0000, Gain=-3.698, Source=flat-EQ
Filter8=peaking, Fc=620.000, Q=1.0000, Gain=-2.723, Source=flat-EQ
Filter9=peaking, Fc=330.000, Q=1.0000, Gain=-1.214, Source=flat-EQ

[SS10F8414G10]
Gain=-4.576 dB
Delay=-0.6900 ms
Polarity=1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter7=lowpass, Fc=3200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter8=lowpass, Fc=3200.000, Q=0.7071, Gain=0.000, Source=LR4 branch low-pass
Filter9=peaking, Fc=1450.000, Q=1.0000, Gain=8.000, Source=flat-EQ
Filter10=peaking, Fc=1800.000, Q=1.0000, Gain=-5.020, Source=flat-EQ
Filter11=peaking, Fc=700.000, Q=1.0000, Gain=-5.595, Source=flat-EQ
Filter12=peaking, Fc=1150.000, Q=1.0000, Gain=-1.882, Source=flat-EQ
Filter13=peaking, Fc=900.000, Q=1.0000, Gain=-0.721, Source=flat-EQ

[GRS PT6816]
Gain=5.039 dB
Delay=-0.6000 ms
Polarity=-1
Filter1=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter2=highpass, Fc=70.000, Q=0.7071, Gain=0.000, Source=LR4 global boundary high-pass
Filter3=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter4=highpass, Fc=120.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter5=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter6=highpass, Fc=650.000, Q=0.7071, Gain=0.000, Source=LR4 cascaded upstream high-pass
Filter7=highpass, Fc=3200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter8=highpass, Fc=3200.000, Q=0.7071, Gain=0.000, Source=LR4 branch high-pass
Filter9=peaking, Fc=3500.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter10=peaking, Fc=7200.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter11=peaking, Fc=11000.000, Q=1.0000, Gain=-8.000, Source=flat-EQ
Filter12=peaking, Fc=9000.000, Q=1.0000, Gain=3.690, Source=flat-EQ
Filter13=peaking, Fc=2800.000, Q=1.0000, Gain=3.710, Source=flat-EQ
Filter14=peaking, Fc=5700.000, Q=1.0000, Gain=2.482, Source=flat-EQ
