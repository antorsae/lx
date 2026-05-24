# L22MG Off-Plane Source Ambiguity Audit

This audit uses Juan's naked L22MG measurements only. Andres' mounted-baffle data are not read, and no source or rear gain correction is introduced.

## Inputs

- Juan HDF5: `output/data/polar_data_juan_baffleless.h5`.
- Juan front notes: distance=0.5 m; height=reference l22mg; first note=Measurement distance: 50 cm from driver. Mic height: L22MG/LM..
- Juan rear notes: distance=0.5 m; height=reference l22mg; first note=Measurement distance: 50 cm from driver. Mic height: L22MG/LM..
- Source-fit/evaluation band: 300-1200 Hz at 24 points/octave.
- Equivalent-source fit radius: 0.5 m.
- Observer radius: 1 m.
- Off-plane observer z offset: 165.0 mm.

## Z-Offset Sensitivity

Rows show how much each Juan-fitted source's normalized polar changes when moving the observer from the Juan horizontal plane to Andres' UM-height plane.
The source-fit columns report normalized SPL RMS, absolute SPL RMS, and phase RMS against Juan front/rear data; normalized fit alone is not sufficient evidence for an acceptance-grade physical source.

| case | model/rear phase | fit norm F/R | fit abs F/R | fit phase F/R | cond max | z-offset RMS 70-90 | z-offset max loc |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| stable | split-discrete / dipole | 2.541 / 2.642 | 2.507 / 2.598 | 16.2 / 16.3 | 5.64e+05 | 0.616 | 1039 Hz / 90 deg |
| wide-svd | split-discrete / dipole | 2.404 / 2.500 | 2.381 / 2.469 | 14.9 / 15.0 | 4.21e+09 | 0.546 | 1039 Hz / 90 deg |
| axisymmetric-directivity | axisymmetric-directivity / dipole | 0.000 / 0.569 | 0.000 / 0.569 | 0.0 / 68.0 | nan | 0.147 | 1200 Hz / 80 deg |
| profile-ring-compact | split-profile-ring / dipole | 0.865 / 1.093 | 0.856 / 1.033 | 12.3 / 12.3 | 1.23e+05 | 0.141 | 980 Hz / 90 deg |
| profile-ring-compact-svd | split-profile-ring / dipole | 0.865 / 1.093 | 0.856 / 1.033 | 12.3 / 12.3 | 1.23e+05 | 0.141 | 980 Hz / 90 deg |
| profile-ring-full | split-profile-ring / dipole | 0.855 / 1.090 | 0.845 / 1.030 | 12.3 / 12.3 | 1.69e+10 | 0.139 | 980 Hz / 90 deg |
| profile-ring-full-svd | split-profile-ring / dipole | 0.855 / 1.091 | 0.845 / 1.030 | 12.3 / 12.3 | 1.69e+10 | 0.139 | 980 Hz / 90 deg |
| modal-compact2 | split-profile-ring / dipole | 2.631 / 1.226 | 2.578 / 0.974 | 15.0 / 13.7 | 53.5 | 0.150 | 1166 Hz / 90 deg |
| modal-compact3 | split-profile-ring / dipole | 2.215 / 1.171 | 2.184 / 0.934 | 14.0 / 13.5 | 9.94e+03 | 0.137 | 1133 Hz / 90 deg |
| modal-full2 | split-profile-ring / dipole | 2.682 / 1.261 | 2.629 / 1.007 | 15.1 / 13.9 | 57.6 | 0.153 | 1166 Hz / 90 deg |
| modal-full3 | split-profile-ring / dipole | 2.184 / 1.204 | 2.156 / 0.965 | 13.9 / 13.6 | 9.33e+03 | 0.135 | 1133 Hz / 90 deg |
| modal-full4-svd | split-profile-ring / dipole | 1.050 / 0.952 | 1.039 / 0.868 | 13.0 / 12.9 | 2.87e+05 | 0.103 | 1100 Hz / 90 deg |
| measured-stable | split-discrete / measured | 2.541 / 2.862 | 2.507 / 2.667 | 16.2 / 27.8 | 5.64e+05 | 1.735 | 1200 Hz / 70 deg |
| measured-profile-ring-compact | split-profile-ring / measured | 0.865 / 4.595 | 0.856 / 4.333 | 12.3 / 24.6 | 1.23e+05 | 0.197 | 1009 Hz / 80 deg |
| measured-profile-ring-full | split-profile-ring / measured | 0.855 / 4.653 | 0.845 / 4.383 | 12.3 / 24.6 | 1.69e+10 | 0.195 | 1009 Hz / 80 deg |
| measured-modal-full2 | split-profile-ring / measured | 2.682 / 5.967 | 2.629 / 6.200 | 15.1 / 62.8 | 57.6 | 0.229 | 1200 Hz / 70 deg |
| measured-modal-full4-svd | split-profile-ring / measured | 1.050 / 5.119 | 1.039 / 5.207 | 13.0 / 38.9 | 2.87e+05 | 0.173 | 1200 Hz / 90 deg |
| d55-r25-95-svd | split-discrete / dipole | 2.537 / 2.639 | 2.518 / 2.612 | 15.3 / 15.3 | 4.44e+08 | 0.568 | 1039 Hz / 90 deg |
| asym-f45-r55-r35-svd | split-discrete / dipole | 2.404 / 2.656 | 2.381 / 2.627 | 14.9 / 15.5 | 4.21e+09 | 0.473 | 1200 Hz / 90 deg |
| asym-f45-r55-front35-rear25-svd | split-discrete / dipole | 2.404 / 2.639 | 2.381 / 2.612 | 14.9 / 15.3 | 4.21e+09 | 0.890 | 1039 Hz / 90 deg |
| h1659-profile-compact-split-discrete-svd | split-discrete / dipole | 1.387 / 1.161 | 1.372 / 1.149 | 11.2 / 11.0 | 1.22e+07 | 1.211 | 873 Hz / 80 deg |
| h1659-profile-full-split-discrete-svd | split-discrete / dipole | 1.397 / 1.169 | 1.383 / 1.159 | 11.1 / 11.0 | 8.3e+06 | 1.320 | 873 Hz / 80 deg |
| split-ring-r35-az24 | split-ring / dipole | 2.797 / 2.944 | 2.833 / 2.958 | 17.5 / 17.5 | 5.56e+06 | 0.248 | 300 Hz / 80 deg |
| split-ring-r25-95-az48-svd | split-ring / dipole | 2.790 / 2.936 | 2.827 / 2.952 | 17.5 / 17.4 | 1.45e+09 | 0.248 | 300 Hz / 80 deg |
| active-surface-dipole-m3 | active-surface-modal / dipole | 2.146 / 6.111 | 2.223 / 5.473 | 67.3 / 32.4 | 2.73e+03 | 0.232 | 1200 Hz / 90 deg |
| split-active-dipole-m4g16-svd | split-active-surface-modal / dipole | 7.807 / 7.624 | 7.741 / 7.510 | 29.5 / 54.6 | 3.97e+08 | 0.112 | 1200 Hz / 90 deg |
| split-active-measured-m4g30 | split-active-surface-modal / measured | 2.518 / 3.766 | 2.553 / 4.046 | 42.6 / 37.2 | 6.55e+08 | 0.151 | 1100 Hz / 70 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg1e-06 | active-surface-modal / measured | 2.864 / 7.064 | 2.815 / 8.138 | 63.5 / 90.1 | 113 | 0.114 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg3e-06 | active-surface-modal / measured | 2.865 / 7.136 | 2.816 / 8.218 | 63.5 / 90.1 | 113 | 0.114 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg1e-06 | active-surface-modal / measured | 2.937 / 6.993 | 2.860 / 8.052 | 63.9 / 90.0 | 122 | 0.114 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg3e-06 | active-surface-modal / measured | 2.938 / 7.080 | 2.861 / 8.148 | 63.8 / 90.1 | 122 | 0.114 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg1e-06 | active-surface-modal / dipole | 5.086 / 4.243 | 4.179 / 4.261 | 69.4 / 29.2 | 113 | 0.159 | 1200 Hz / 70 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg3e-06 | active-surface-modal / dipole | 5.087 / 4.245 | 4.174 / 4.256 | 69.4 / 29.2 | 113 | 0.159 | 1200 Hz / 70 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg1e-06 | active-surface-modal / dipole | 5.210 / 4.318 | 4.272 / 4.329 | 69.5 / 29.6 | 122 | 0.159 | 1200 Hz / 70 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg3e-06 | active-surface-modal / dipole | 5.211 / 4.320 | 4.267 / 4.322 | 69.5 / 29.6 | 122 | 0.159 | 1200 Hz / 70 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06 | active-surface-modal / measured | 1.927 / 4.422 | 2.096 / 6.374 | 64.9 / 78.9 | 5.7e+03 | 0.136 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg3e-06 | active-surface-modal / measured | 2.248 / 5.921 | 2.334 / 7.568 | 63.5 / 80.0 | 5.7e+03 | 0.131 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg1e-06 | active-surface-modal / measured | 1.914 / 4.300 | 2.085 / 6.278 | 65.1 / 78.9 | 5.38e+03 | 0.137 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg3e-06 | active-surface-modal / measured | 2.250 / 5.782 | 2.331 / 7.442 | 63.8 / 79.9 | 5.38e+03 | 0.133 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg1e-06 | active-surface-modal / dipole | 2.757 / 5.786 | 2.633 / 5.351 | 67.0 / 26.9 | 5.7e+03 | 0.191 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg3e-06 | active-surface-modal / dipole | 3.183 / 4.854 | 2.907 / 4.637 | 68.1 / 26.6 | 5.7e+03 | 0.189 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg1e-06 | active-surface-modal / dipole | 2.752 / 5.899 | 2.636 / 5.449 | 66.9 / 27.0 | 5.38e+03 | 0.191 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg3e-06 | active-surface-modal / dipole | 3.178 / 4.961 | 2.909 / 4.725 | 68.1 / 26.7 | 5.38e+03 | 0.189 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06 | active-surface-modal / measured | 2.144 / 4.286 | 2.325 / 5.754 | 56.0 / 69.5 | 3.76e+04 | 0.103 | 1009 Hz / 80 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06 | active-surface-modal / measured | 2.817 / 5.689 | 3.041 / 7.182 | 60.8 / 74.1 | 3.76e+04 | 0.118 | 1166 Hz / 80 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06 | active-surface-modal / measured | 2.064 / 4.216 | 2.218 / 5.612 | 55.4 / 68.8 | 3.31e+04 | 0.089 | 980 Hz / 80 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06 | active-surface-modal / measured | 2.780 / 5.545 | 2.986 / 6.990 | 59.2 / 73.3 | 3.31e+04 | 0.132 | 1133 Hz / 80 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06 | active-surface-modal / dipole | 3.419 / 4.636 | 3.340 / 4.336 | 61.9 / 23.5 | 3.76e+04 | 0.164 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06 | active-surface-modal / dipole | 3.301 / 4.299 | 3.124 / 4.100 | 66.3 / 24.5 | 3.76e+04 | 0.181 | 1200 Hz / 80 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06 | active-surface-modal / dipole | 3.541 / 4.679 | 3.465 / 4.389 | 61.1 / 23.7 | 3.31e+04 | 0.162 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06 | active-surface-modal / dipole | 3.354 / 4.307 | 3.188 / 4.103 | 65.7 / 24.4 | 3.31e+04 | 0.179 | 1200 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m2-az16-q3-reg1e-06 | active-surface-modal / measured | 3.217 / 6.615 | 3.223 / 7.607 | 66.0 / 91.8 | 132 | 0.136 | 1200 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-m2-az16-q3-reg1e-06 | active-surface-modal / dipole | 5.732 / 4.764 | 4.786 / 4.806 | 70.9 / 31.9 | 132 | 0.187 | 1200 Hz / 70 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06 | active-surface-modal / measured | 3.758 / 2.922 | 3.496 / 5.262 | 56.9 / 69.5 | 1.88e+03 | 0.317 | 1009 Hz / 70 deg |
| physical-diaphragm-full-coupled-dipole-m3-az16-q3-reg1e-06 | active-surface-modal / dipole | 7.050 / 7.419 | 7.104 / 7.232 | 55.6 / 34.4 | 1.88e+03 | 0.125 | 1166 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06 | active-surface-modal / measured | 2.556 / 3.119 | 2.437 / 4.712 | 49.7 / 62.1 | 3.07e+04 | 0.224 | 734 Hz / 80 deg |
| physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06 | active-surface-modal / dipole | 6.323 / 6.521 | 6.314 / 6.347 | 45.0 / 38.3 | 3.07e+04 | 0.126 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | active-surface-modal / measured | 1.927 / 4.422 | 2.096 / 6.374 | 64.9 / 78.9 | 5.7e+03 | 0.136 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05 | active-surface-modal / measured | 2.144 / 4.286 | 2.325 / 5.754 | 56.0 / 69.5 | 3.76e+04 | 0.103 | 1009 Hz / 80 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | active-surface-modal / measured | 3.758 / 2.922 | 3.496 / 5.262 | 56.9 / 69.5 | 1.88e+03 | 0.317 | 1009 Hz / 70 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05 | active-surface-modal / measured | 2.556 / 3.119 | 2.437 / 4.712 | 49.7 / 62.1 | 3.07e+04 | 0.224 | 734 Hz / 80 deg |
| physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.276 / 2.635 | 1.330 / 4.543 | 71.6 / 72.7 | 326 | 0.083 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.769 / 3.432 | 1.532 / 4.510 | 60.0 / 64.3 | 1.31e+04 | 0.204 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.795 / 3.685 | 1.463 / 4.400 | 55.1 / 60.2 | 1.99e+05 | 0.218 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.582 / 3.796 | 1.518 / 4.323 | 50.5 / 52.3 | 3.43e+03 | 0.125 | 734 Hz / 80 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.463 / 4.431 | 1.395 / 4.852 | 49.1 / 48.8 | 8.5e+04 | 0.138 | 693 Hz / 80 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.431 / 4.693 | 1.370 / 5.069 | 48.5 / 46.2 | 1.83e+07 | 0.142 | 673 Hz / 80 deg |
| physical-rear-basket-full-measured-m2-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.238 / 2.683 | 1.336 / 4.555 | 70.8 / 71.5 | 264 | 0.093 | 1200 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.797 / 3.540 | 1.568 / 4.502 | 58.6 / 62.9 | 1.24e+04 | 0.208 | 1200 Hz / 70 deg |
| physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.706 / 3.829 | 1.384 / 4.406 | 53.0 / 57.4 | 1.3e+05 | 0.202 | 801 Hz / 80 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.585 / 3.813 | 1.519 / 4.323 | 50.3 / 52.0 | 3.14e+03 | 0.128 | 714 Hz / 80 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.454 / 4.607 | 1.388 / 5.011 | 48.8 / 47.7 | 7.62e+04 | 0.138 | 673 Hz / 80 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.417 / 4.460 | 1.357 / 4.818 | 48.1 / 45.1 | 1.22e+07 | 0.143 | 654 Hz / 80 deg |
| physical-rear-basket-compact-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 3.572 / 4.282 | 3.642 / 3.952 | 44.8 / 39.4 | 326 | 0.103 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 5.270 / 5.417 | 5.308 / 5.124 | 47.4 / 36.0 | 1.31e+04 | 0.077 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 6.617 / 6.632 | 6.622 / 6.396 | 47.1 / 35.6 | 1.99e+05 | 0.094 | 925 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 8.010 / 7.836 | 7.946 / 7.708 | 31.5 / 50.3 | 3.43e+03 | 0.120 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.150 / 9.000 | 9.092 / 8.868 | 29.5 / 52.4 | 8.5e+04 | 0.121 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.307 / 9.159 | 9.251 / 9.027 | 28.3 / 54.0 | 1.83e+07 | 0.123 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 3.770 / 4.351 | 3.836 / 4.045 | 44.4 / 40.5 | 264 | 0.102 | 1200 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 5.670 / 5.762 | 5.698 / 5.488 | 47.1 / 36.4 | 1.24e+04 | 0.075 | 1200 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 7.684 / 7.637 | 7.672 / 7.435 | 46.5 / 36.7 | 1.3e+05 | 0.135 | 778 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 8.106 / 7.931 | 8.042 / 7.804 | 31.2 / 50.8 | 3.14e+03 | 0.119 | 1200 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.221 / 9.073 | 9.164 / 8.941 | 28.5 / 53.4 | 7.62e+04 | 0.122 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.439 / 9.293 | 9.383 / 9.161 | 27.3 / 55.0 | 1.22e+07 | 0.124 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.573 / 3.767 | 1.513 / 4.316 | 50.8 / 52.8 | 3.65e+03 | 0.122 | 734 Hz / 80 deg |
| physical-rear-basket-compact-measured-m3-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.458 / 4.377 | 1.385 / 4.808 | 49.3 / 49.4 | 8.27e+04 | 0.138 | 693 Hz / 80 deg |
| physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.424 / 4.719 | 1.353 / 5.100 | 48.7 / 46.7 | 1.29e+07 | 0.142 | 673 Hz / 80 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 7.922 / 7.750 | 7.858 / 7.620 | 31.6 / 49.8 | 3.65e+03 | 0.120 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.143 / 8.996 | 9.086 / 8.863 | 29.6 / 52.1 | 8.27e+04 | 0.121 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / dipole | 9.370 / 9.225 | 9.315 / 9.091 | 28.2 / 53.8 | 1.29e+07 | 0.126 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.734 / 3.824 | 1.743 / 4.356 | 50.4 / 52.7 | 2.65e+03 | 0.117 | 693 Hz / 80 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.641 / 4.270 | 1.642 / 4.657 | 48.5 / 46.1 | 5.52e+04 | 0.131 | 636 Hz / 80 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.634 / 4.013 | 1.641 / 4.354 | 47.7 / 44.3 | 8.94e+06 | 0.135 | 583 Hz / 80 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.751 / 3.835 | 1.758 / 4.359 | 50.3 / 52.3 | 2.46e+03 | 0.119 | 693 Hz / 80 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.646 / 4.136 | 1.648 / 4.506 | 48.1 / 45.2 | 4.92e+04 | 0.131 | 618 Hz / 80 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.632 / 3.913 | 1.641 / 4.232 | 47.2 / 43.4 | 4.52e+06 | 0.136 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.606 / 4.055 | 1.632 / 4.376 | 46.3 / 41.6 | 2.08e+07 | 0.145 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.707 / 3.971 | 1.743 / 4.275 | 45.5 / 40.4 | 1.2e+09 | 0.150 | 1200 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb3-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.636 / 3.953 | 1.668 / 4.251 | 45.7 / 40.6 | 1.52e+07 | 0.148 | 1200 Hz / 70 deg |
| physical-rear-basket-full-measured-m4-rb4-az16-gap16-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.768 / 3.890 | 1.809 / 4.172 | 44.6 / 39.2 | 6.7e+08 | 0.155 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 1.948 / 3.807 | 1.988 / 4.107 | 45.3 / 40.4 | 1.14e+07 | 0.139 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap30-q3-reg1e-06 | coupled-rear-basket-active-surface / measured | 2.079 / 3.784 | 2.123 / 4.069 | 44.3 / 39.0 | 2.03e+09 | 0.144 | 1200 Hz / 70 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06 | coupled-rear-basket-active-surface / measured | 1.461 / 4.111 | 1.364 / 4.597 | 50.1 / 51.5 | 1.83e+07 | 0.135 | 734 Hz / 80 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05 | coupled-rear-basket-active-surface / measured | 1.431 / 4.693 | 1.370 / 5.069 | 48.5 / 46.2 | 1.83e+07 | 0.142 | 673 Hz / 80 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06 | split-active-surface-modal / measured | 2.295 / 3.792 | 2.325 / 4.134 | 44.8 / 40.1 | 8.82e+05 | 0.144 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-06 | split-active-surface-modal / measured | 2.295 / 3.792 | 2.325 / 4.134 | 44.8 / 40.1 | 8.82e+05 | 0.144 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-05 | split-active-surface-modal / measured | 2.295 / 3.792 | 2.325 / 4.134 | 44.8 / 40.1 | 8.82e+05 | 0.144 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06 | split-active-surface-modal / measured | 2.437 / 3.759 | 2.467 / 4.088 | 44.3 / 39.8 | 9.42e+05 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-06 | split-active-surface-modal / measured | 2.437 / 3.759 | 2.467 / 4.088 | 44.3 / 39.8 | 9.42e+05 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-05 | split-active-surface-modal / measured | 2.437 / 3.759 | 2.467 / 4.088 | 44.3 / 39.8 | 9.42e+05 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06 | split-active-surface-modal / measured | 2.197 / 3.728 | 2.228 / 4.042 | 44.2 / 39.0 | 8.66e+11 | 0.147 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-06 | split-active-surface-modal / measured | 2.197 / 3.728 | 2.228 / 4.042 | 44.2 / 39.0 | 8.66e+11 | 0.147 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-05 | split-active-surface-modal / measured | 2.197 / 3.728 | 2.228 / 4.042 | 44.2 / 39.0 | 8.66e+11 | 0.147 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06 | split-active-surface-modal / measured | 2.321 / 3.694 | 2.353 / 3.995 | 43.8 / 38.8 | 7.31e+11 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-06 | split-active-surface-modal / measured | 2.321 / 3.694 | 2.353 / 3.995 | 43.8 / 38.8 | 7.31e+11 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-05 | split-active-surface-modal / measured | 2.321 / 3.694 | 2.353 / 3.995 | 43.8 / 38.8 | 7.31e+11 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap24-q3-reg1e-06 | split-active-surface-modal / measured | 2.168 / 3.854 | 2.196 / 4.211 | 45.4 / 40.4 | 8.66e+05 | 0.148 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az24-gap24-q3-reg1e-06 | split-active-surface-modal / measured | 2.175 / 3.849 | 2.203 / 4.204 | 45.3 / 40.3 | 8.39e+05 | 0.149 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg1e-06 | split-active-surface-modal / measured | 2.251 / 3.810 | 2.281 / 4.156 | 45.0 / 40.2 | 8.72e+05 | 0.145 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az24-gap28-q3-reg1e-06 | split-active-surface-modal / measured | 2.259 / 3.805 | 2.289 / 4.149 | 44.9 / 40.1 | 8.43e+05 | 0.146 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg1e-06 | split-active-surface-modal / measured | 2.341 / 3.778 | 2.371 / 4.115 | 44.7 / 40.0 | 8.97e+05 | 0.142 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az24-gap32-q3-reg1e-06 | split-active-surface-modal / measured | 2.350 / 3.775 | 2.380 / 4.110 | 44.6 / 39.9 | 8.64e+05 | 0.142 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap42-q3-reg1e-06 | split-active-surface-modal / measured | 2.594 / 3.751 | 2.622 / 4.071 | 43.7 / 39.7 | 1.06e+06 | 0.132 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az24-gap42-q3-reg1e-06 | split-active-surface-modal / measured | 2.607 / 3.751 | 2.634 / 4.069 | 43.6 / 39.6 | 1.01e+06 | 0.132 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-07 | split-active-surface-modal / measured | 2.568 / 3.802 | 2.606 / 4.063 | 41.7 / 36.1 | 8.72e+05 | 0.157 | 1069 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-06 | split-active-surface-modal / measured | 2.020 / 3.906 | 2.024 / 4.364 | 47.5 / 44.1 | 8.72e+05 | 0.130 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07 | split-active-surface-modal / measured | 2.624 / 3.801 | 2.662 / 4.059 | 41.5 / 35.9 | 8.82e+05 | 0.155 | 1039 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06 | split-active-surface-modal / measured | 2.049 / 3.871 | 2.056 / 4.322 | 47.4 / 44.0 | 8.82e+05 | 0.129 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-07 | split-active-surface-modal / measured | 2.682 / 3.804 | 2.720 / 4.059 | 41.2 / 35.7 | 8.97e+05 | 0.153 | 1039 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-06 | split-active-surface-modal / measured | 2.081 / 3.839 | 2.090 / 4.284 | 47.3 / 44.0 | 8.97e+05 | 0.127 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07-svd1e-05 | split-active-surface-modal / measured | 2.624 / 3.801 | 2.662 / 4.059 | 41.5 / 35.9 | 8.82e+05 | 0.155 | 1039 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06-svd1e-05 | split-active-surface-modal / measured | 2.049 / 3.871 | 2.056 / 4.322 | 47.4 / 44.0 | 8.82e+05 | 0.129 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06 | split-active-surface-modal / measured | 2.090 / 3.785 | 2.118 / 4.116 | 44.7 / 39.4 | 1.15e+12 | 0.151 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06 | split-active-surface-modal / dipole | 7.212 / 7.020 | 7.144 / 6.904 | 24.5 / 55.9 | 8.82e+05 | 0.117 | 1200 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06-svd1e-05 | split-active-surface-modal / measured | 2.090 / 3.785 | 2.118 / 4.116 | 44.7 / 39.4 | 1.15e+12 | 0.151 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06-svd1e-05 | split-active-surface-modal / dipole | 7.212 / 7.020 | 7.144 / 6.904 | 24.5 / 55.9 | 8.82e+05 | 0.117 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.153 / 3.814 | 2.188 / 4.139 | 44.9 / 39.7 | 5.47e+08 | 0.151 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m4 | split-active-surface-modal / measured | 1.941 / 3.949 | 1.954 / 4.374 | 47.3 / 43.4 | 5.47e+08 | 0.135 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.159 / 3.808 | 2.194 / 4.130 | 44.8 / 39.6 | 5.18e+08 | 0.151 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m4 | split-active-surface-modal / measured | 1.945 / 3.938 | 1.959 / 4.361 | 47.2 / 43.3 | 5.18e+08 | 0.135 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.203 / 3.799 | 2.241 / 4.106 | 44.4 / 39.0 | 7.23e+35 | 0.153 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m5 | split-active-surface-modal / measured | 1.978 / 3.911 | 1.997 / 4.313 | 46.8 / 42.6 | 7.23e+35 | 0.138 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.210 / 3.792 | 2.247 / 4.098 | 44.3 / 38.9 | 6.95e+35 | 0.153 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m5 | split-active-surface-modal / measured | 1.982 / 3.902 | 2.002 / 4.301 | 46.8 / 42.5 | 6.95e+35 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.246 / 3.787 | 2.284 / 4.081 | 43.9 / 38.4 | 1.57e+51 | 0.155 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.010 / 3.885 | 2.035 / 4.269 | 46.4 / 42.0 | 1.57e+51 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.252 / 3.781 | 2.291 / 4.073 | 43.8 / 38.3 | 2.81e+51 | 0.155 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.015 / 3.876 | 2.040 / 4.258 | 46.4 / 41.9 | 2.81e+51 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.282 / 3.778 | 2.321 / 4.062 | 43.5 / 37.9 | 4.76e+67 | 0.156 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.040 / 3.866 | 2.067 / 4.236 | 46.1 / 41.5 | 4.76e+67 | 0.143 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.289 / 3.772 | 2.328 / 4.054 | 43.4 / 37.8 | 7.03e+66 | 0.156 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.045 / 3.858 | 2.073 / 4.226 | 46.0 / 41.4 | 7.03e+66 | 0.143 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.248 / 3.779 | 2.285 / 4.095 | 44.6 / 39.4 | 6.64e+08 | 0.147 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.010 / 3.878 | 2.028 / 4.292 | 47.0 / 43.2 | 6.64e+08 | 0.133 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.256 / 3.773 | 2.292 / 4.088 | 44.5 / 39.3 | 6.34e+08 | 0.147 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.016 / 3.869 | 2.034 / 4.280 | 47.0 / 43.1 | 6.34e+08 | 0.133 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.303 / 3.769 | 2.341 / 4.069 | 44.0 / 38.7 | 5.37e+35 | 0.150 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.053 / 3.848 | 2.076 / 4.240 | 46.6 / 42.4 | 5.37e+35 | 0.136 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.311 / 3.763 | 2.349 / 4.062 | 43.9 / 38.6 | 9.98e+35 | 0.150 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.059 / 3.840 | 2.083 / 4.229 | 46.5 / 42.3 | 9.98e+35 | 0.136 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.349 / 3.762 | 2.387 / 4.050 | 43.5 / 38.1 | 8.37e+50 | 0.151 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.090 / 3.828 | 2.117 / 4.203 | 46.2 / 41.8 | 8.37e+50 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.357 / 3.757 | 2.395 / 4.043 | 43.5 / 38.0 | 1.42e+51 | 0.151 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.097 / 3.821 | 2.124 / 4.193 | 46.1 / 41.7 | 1.42e+51 | 0.139 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.388 / 3.757 | 2.427 / 4.035 | 43.1 / 37.6 | 3.34e+67 | 0.153 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.124 / 3.814 | 2.153 / 4.175 | 45.8 / 41.3 | 3.34e+67 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.396 / 3.752 | 2.435 / 4.028 | 43.0 / 37.5 | 5.23e+67 | 0.152 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.130 / 3.808 | 2.160 / 4.166 | 45.7 / 41.2 | 5.23e+67 | 0.141 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.351 / 3.755 | 2.386 / 4.066 | 44.2 / 39.2 | 8.24e+08 | 0.144 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.086 / 3.821 | 2.107 / 4.225 | 46.8 / 43.1 | 8.24e+08 | 0.130 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.359 / 3.751 | 2.396 / 4.059 | 44.1 / 39.1 | 8.05e+08 | 0.144 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.093 / 3.814 | 2.115 / 4.215 | 46.7 / 43.0 | 8.05e+08 | 0.130 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.410 / 3.752 | 2.447 / 4.047 | 43.6 / 38.5 | 5.97e+35 | 0.146 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.135 / 3.799 | 2.160 / 4.182 | 46.3 / 42.3 | 5.97e+35 | 0.133 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.419 / 3.747 | 2.456 / 4.040 | 43.6 / 38.4 | 8.84e+35 | 0.146 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.143 / 3.793 | 2.168 / 4.173 | 46.2 / 42.2 | 8.84e+35 | 0.133 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.458 / 3.750 | 2.497 / 4.033 | 43.2 / 37.8 | 1.32e+51 | 0.148 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.177 / 3.785 | 2.205 / 4.151 | 45.9 / 41.6 | 1.32e+51 | 0.136 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.468 / 3.746 | 2.506 / 4.027 | 43.1 / 37.7 | 1.44e+51 | 0.147 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.185 / 3.779 | 2.214 / 4.143 | 45.8 / 41.5 | 1.44e+51 | 0.136 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.500 / 3.750 | 2.539 / 4.023 | 42.7 / 37.3 | 8.29e+66 | 0.149 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.214 / 3.775 | 2.245 / 4.128 | 45.5 / 41.1 | 8.29e+66 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.509 / 3.745 | 2.548 / 4.017 | 42.6 / 37.2 | 4.26e+67 | 0.148 | 1069 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.222 / 3.770 | 2.253 / 4.120 | 45.4 / 41.0 | 4.26e+67 | 0.138 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.459 / 3.744 | 2.494 / 4.049 | 43.8 / 39.1 | 9.67e+08 | 0.140 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.169 / 3.777 | 2.191 / 4.172 | 46.5 / 43.0 | 9.67e+08 | 0.127 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m4 | split-active-surface-modal / measured | 2.469 / 3.741 | 2.504 / 4.044 | 43.8 / 38.9 | 9.91e+08 | 0.140 | 1166 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m4 | split-active-surface-modal / measured | 2.177 / 3.771 | 2.200 / 4.164 | 46.4 / 42.9 | 9.91e+08 | 0.127 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.523 / 3.747 | 2.559 / 4.037 | 43.3 / 38.3 | 5.93e+35 | 0.142 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.223 / 3.762 | 2.250 / 4.137 | 46.0 / 42.2 | 5.93e+35 | 0.130 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.533 / 3.744 | 2.569 / 4.032 | 43.2 / 38.2 | 5.25e+35 | 0.142 | 1133 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m5 | split-active-surface-modal / measured | 2.232 / 3.757 | 2.259 / 4.129 | 45.9 / 42.1 | 5.25e+35 | 0.130 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.575 / 3.750 | 2.612 / 4.029 | 42.8 / 37.6 | 2.25e+51 | 0.144 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.270 / 3.753 | 2.299 / 4.112 | 45.5 / 41.5 | 2.25e+51 | 0.132 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m6 | split-active-surface-modal / measured | 2.586 / 3.748 | 2.622 / 4.025 | 42.7 / 37.5 | 1.09e+51 | 0.143 | 1100 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m6 | split-active-surface-modal / measured | 2.279 / 3.749 | 2.308 / 4.105 | 45.5 / 41.4 | 1.09e+51 | 0.132 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.620 / 3.754 | 2.657 / 4.024 | 42.3 / 37.1 | 6.6e+67 | 0.145 | 1069 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.310 / 3.748 | 2.341 / 4.094 | 45.2 / 41.0 | 6.6e+67 | 0.134 | 1200 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.630 / 3.752 | 2.667 / 4.020 | 42.2 / 37.0 | 2.67e+67 | 0.144 | 1069 Hz / 70 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.320 / 3.744 | 2.351 / 4.088 | 45.1 / 40.9 | 2.67e+67 | 0.134 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.047 / 3.800 | 2.080 / 4.126 | 45.0 / 39.8 | 9.68e+35 | 0.150 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m5 | split-active-surface-modal / measured | 1.869 / 3.943 | 1.873 / 4.377 | 47.4 / 43.3 | 9.68e+35 | 0.134 | 1200 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.117 / 3.776 | 2.154 / 4.074 | 44.2 / 38.7 | 2e+67 | 0.155 | 1166 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m7 | split-active-surface-modal / measured | 1.914 / 3.885 | 1.929 / 4.283 | 46.7 / 42.2 | 2e+67 | 0.139 | 1200 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m5 | split-active-surface-modal / measured | 2.215 / 3.728 | 2.250 / 4.037 | 44.4 / 39.3 | 6.01e+35 | 0.144 | 1166 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m5 | split-active-surface-modal / measured | 1.982 / 3.807 | 1.998 / 4.216 | 46.9 / 43.1 | 6.01e+35 | 0.130 | 1200 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m7 | split-active-surface-modal / measured | 2.298 / 3.720 | 2.336 / 4.005 | 43.6 / 38.2 | 1.24e+67 | 0.148 | 1133 Hz / 70 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m7 | split-active-surface-modal / measured | 2.045 / 3.772 | 2.069 / 4.146 | 46.2 / 41.9 | 1.24e+67 | 0.135 | 1200 Hz / 70 deg |

## Source-Model Spread At UM Height

Rows compare each off-plane normalized source field to `wide-svd` before any baffle scattering.

| case | reference | RMS 70-90 | max loc |
| --- | --- | ---: | --- |
| stable | wide-svd | 2.418 | 1200 Hz / 90 deg |
| wide-svd | wide-svd | 0.000 | 300 Hz / 70 deg |
| axisymmetric-directivity | wide-svd | 10.032 | 1009 Hz / 90 deg |
| profile-ring-compact | wide-svd | 10.875 | 1200 Hz / 90 deg |
| profile-ring-compact-svd | wide-svd | 10.876 | 1200 Hz / 90 deg |
| profile-ring-full | wide-svd | 10.949 | 1200 Hz / 90 deg |
| profile-ring-full-svd | wide-svd | 10.948 | 1200 Hz / 90 deg |
| modal-compact2 | wide-svd | 6.900 | 1009 Hz / 90 deg |
| modal-compact3 | wide-svd | 5.755 | 1009 Hz / 90 deg |
| modal-full2 | wide-svd | 6.951 | 1009 Hz / 90 deg |
| modal-full3 | wide-svd | 5.567 | 1009 Hz / 90 deg |
| modal-full4-svd | wide-svd | 7.484 | 1133 Hz / 90 deg |
| measured-stable | wide-svd | 6.827 | 1009 Hz / 90 deg |
| measured-profile-ring-compact | wide-svd | 7.981 | 1039 Hz / 80 deg |
| measured-profile-ring-full | wide-svd | 7.929 | 1039 Hz / 80 deg |
| measured-modal-full2 | wide-svd | 9.176 | 1009 Hz / 90 deg |
| measured-modal-full4-svd | wide-svd | 7.330 | 1009 Hz / 90 deg |
| d55-r25-95-svd | wide-svd | 1.293 | 1200 Hz / 70 deg |
| asym-f45-r55-r35-svd | wide-svd | 1.519 | 1069 Hz / 90 deg |
| asym-f45-r55-front35-rear25-svd | wide-svd | 1.679 | 1039 Hz / 90 deg |
| h1659-profile-compact-split-discrete-svd | wide-svd | 9.864 | 437 Hz / 90 deg |
| h1659-profile-full-split-discrete-svd | wide-svd | 9.572 | 899 Hz / 70 deg |
| split-ring-r35-az24 | wide-svd | 4.445 | 980 Hz / 90 deg |
| split-ring-r25-95-az48-svd | wide-svd | 4.480 | 980 Hz / 90 deg |
| active-surface-dipole-m3 | wide-svd | 8.544 | 1009 Hz / 90 deg |
| split-active-dipole-m4g16-svd | wide-svd | 8.346 | 1200 Hz / 90 deg |
| split-active-measured-m4g30 | wide-svd | 9.929 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg1e-06 | wide-svd | 11.476 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg3e-06 | wide-svd | 11.475 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg1e-06 | wide-svd | 11.590 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg3e-06 | wide-svd | 11.588 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg1e-06 | wide-svd | 16.366 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg3e-06 | wide-svd | 16.367 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg1e-06 | wide-svd | 16.551 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg3e-06 | wide-svd | 16.552 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06 | wide-svd | 8.497 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg3e-06 | wide-svd | 9.297 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg1e-06 | wide-svd | 8.440 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg3e-06 | wide-svd | 9.249 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg1e-06 | wide-svd | 9.651 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg3e-06 | wide-svd | 11.730 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg1e-06 | wide-svd | 9.466 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg3e-06 | wide-svd | 11.578 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06 | wide-svd | 8.141 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06 | wide-svd | 8.967 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06 | wide-svd | 8.121 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06 | wide-svd | 8.957 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06 | wide-svd | 8.761 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06 | wide-svd | 11.064 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06 | wide-svd | 8.541 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06 | wide-svd | 10.864 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m2-az16-q3-reg1e-06 | wide-svd | 11.728 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-m2-az16-q3-reg1e-06 | wide-svd | 17.293 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06 | wide-svd | 8.790 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-m3-az16-q3-reg1e-06 | wide-svd | 6.519 | 1200 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06 | wide-svd | 8.711 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06 | wide-svd | 6.194 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | wide-svd | 8.497 | 1009 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05 | wide-svd | 8.141 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | wide-svd | 8.790 | 1009 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05 | wide-svd | 8.711 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 7.519 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 8.373 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 9.431 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.222 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.360 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.342 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 7.138 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 8.512 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 9.878 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.313 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.348 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 10.319 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 8.423 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 7.611 | 300 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 8.872 | 952 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 8.595 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 9.219 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 9.376 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 8.318 | 1009 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 7.712 | 300 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | wide-svd | 9.523 | 778 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 8.648 | 1200 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 9.278 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | wide-svd | 9.417 | 1100 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 10.095 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 10.332 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 10.334 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 8.541 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 9.268 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06 | wide-svd | 9.501 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.265 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.300 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.226 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.334 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.272 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06 | wide-svd | 10.188 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap16-q3-reg1e-06 | wide-svd | 10.011 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap16-q3-reg1e-06 | wide-svd | 9.907 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb3-az16-gap16-q3-reg1e-06 | wide-svd | 9.941 | 1009 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb4-az16-gap16-q3-reg1e-06 | wide-svd | 9.834 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap30-q3-reg1e-06 | wide-svd | 10.016 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap30-q3-reg1e-06 | wide-svd | 9.969 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06 | wide-svd | 10.166 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05 | wide-svd | 10.342 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06 | wide-svd | 9.762 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-06 | wide-svd | 9.762 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-05 | wide-svd | 9.762 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06 | wide-svd | 9.845 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-06 | wide-svd | 9.845 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-05 | wide-svd | 9.845 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06 | wide-svd | 9.771 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-06 | wide-svd | 9.771 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-05 | wide-svd | 9.771 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06 | wide-svd | 9.871 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-06 | wide-svd | 9.871 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-05 | wide-svd | 9.871 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap24-q3-reg1e-06 | wide-svd | 9.702 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap24-q3-reg1e-06 | wide-svd | 9.702 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg1e-06 | wide-svd | 9.739 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap28-q3-reg1e-06 | wide-svd | 9.741 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg1e-06 | wide-svd | 9.787 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap32-q3-reg1e-06 | wide-svd | 9.791 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap42-q3-reg1e-06 | wide-svd | 9.948 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap42-q3-reg1e-06 | wide-svd | 9.956 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-07 | wide-svd | 9.986 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-06 | wide-svd | 9.573 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07 | wide-svd | 10.039 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06 | wide-svd | 9.576 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-07 | wide-svd | 10.094 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-06 | wide-svd | 9.582 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07-svd1e-05 | wide-svd | 10.039 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06-svd1e-05 | wide-svd | 9.576 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06 | wide-svd | 9.691 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06 | wide-svd | 8.243 | 1200 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06-svd1e-05 | wide-svd | 9.691 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06-svd1e-05 | wide-svd | 8.243 | 1200 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m4 | wide-svd | 9.775 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m4 | wide-svd | 9.687 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m4 | wide-svd | 9.775 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m4 | wide-svd | 9.687 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m5 | wide-svd | 9.794 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m5 | wide-svd | 9.708 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m5 | wide-svd | 9.795 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m5 | wide-svd | 9.707 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m6 | wide-svd | 9.812 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m6 | wide-svd | 9.723 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m6 | wide-svd | 9.814 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m6 | wide-svd | 9.723 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m7 | wide-svd | 9.831 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m7 | wide-svd | 9.734 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m7 | wide-svd | 9.833 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m7 | wide-svd | 9.735 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m4 | wide-svd | 9.821 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m4 | wide-svd | 9.696 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m4 | wide-svd | 9.823 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m4 | wide-svd | 9.697 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m5 | wide-svd | 9.850 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m5 | wide-svd | 9.723 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m5 | wide-svd | 9.853 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m5 | wide-svd | 9.724 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m6 | wide-svd | 9.877 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m6 | wide-svd | 9.743 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m6 | wide-svd | 9.880 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m6 | wide-svd | 9.745 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m7 | wide-svd | 9.903 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m7 | wide-svd | 9.760 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m7 | wide-svd | 9.906 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m7 | wide-svd | 9.762 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m4 | wide-svd | 9.876 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m4 | wide-svd | 9.711 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m4 | wide-svd | 9.880 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m4 | wide-svd | 9.714 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m5 | wide-svd | 9.915 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m5 | wide-svd | 9.745 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m5 | wide-svd | 9.919 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m5 | wide-svd | 9.749 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m6 | wide-svd | 9.950 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m6 | wide-svd | 9.772 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m6 | wide-svd | 9.955 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m6 | wide-svd | 9.775 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m7 | wide-svd | 9.984 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m7 | wide-svd | 9.794 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m7 | wide-svd | 9.989 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m7 | wide-svd | 9.798 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m4 | wide-svd | 9.940 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m4 | wide-svd | 9.734 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m4 | wide-svd | 9.946 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m4 | wide-svd | 9.739 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m5 | wide-svd | 9.988 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m5 | wide-svd | 9.775 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m5 | wide-svd | 9.995 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m5 | wide-svd | 9.781 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m6 | wide-svd | 10.032 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m6 | wide-svd | 9.809 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m6 | wide-svd | 10.039 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m6 | wide-svd | 9.814 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m7 | wide-svd | 10.073 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m7 | wide-svd | 9.837 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m7 | wide-svd | 10.080 | 1009 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m7 | wide-svd | 9.842 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m5 | wide-svd | 9.741 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m5 | wide-svd | 9.649 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m7 | wide-svd | 9.770 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m7 | wide-svd | 9.682 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m5 | wide-svd | 9.837 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m5 | wide-svd | 9.679 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m7 | wide-svd | 9.892 | 1009 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m7 | wide-svd | 9.728 | 1009 Hz / 90 deg |

## Acceptance-Eligible Source Spread

Rows compare each source to the worst matching member of the Juan-CV `recommended_juan_only` finite-source set. This avoids using a Juan-rejected source as the only spread anchor.

| case | worst recommended reference | RMS 70-90 | max loc |
| --- | --- | ---: | --- |
| stable | profile-ring-full | 10.862 | 412 Hz / 90 deg |
| wide-svd | profile-ring-full | 10.949 | 1200 Hz / 90 deg |
| axisymmetric-directivity | profile-ring-full | 13.940 | 1133 Hz / 90 deg |
| profile-ring-compact | modal-full2 | 13.358 | 734 Hz / 90 deg |
| profile-ring-compact-svd | modal-full2 | 13.359 | 734 Hz / 90 deg |
| profile-ring-full | modal-full2 | 13.406 | 734 Hz / 90 deg |
| profile-ring-full-svd | modal-full2 | 13.406 | 734 Hz / 90 deg |
| modal-compact2 | profile-ring-full | 13.249 | 734 Hz / 90 deg |
| modal-compact3 | profile-ring-full-svd | 11.570 | 412 Hz / 90 deg |
| modal-full2 | profile-ring-full-svd | 13.406 | 734 Hz / 90 deg |
| modal-full3 | profile-ring-full-svd | 11.242 | 412 Hz / 90 deg |
| modal-full4-svd | modal-full2 | 8.946 | 1100 Hz / 90 deg |
| measured-stable | profile-ring-full | 14.486 | 734 Hz / 90 deg |
| measured-profile-ring-compact | profile-ring-full-svd | 11.958 | 756 Hz / 90 deg |
| measured-profile-ring-full | profile-ring-full-svd | 11.934 | 756 Hz / 90 deg |
| measured-modal-full2 | profile-ring-full | 16.001 | 1133 Hz / 90 deg |
| measured-modal-full4-svd | profile-ring-full | 12.040 | 1166 Hz / 90 deg |
| d55-r25-95-svd | profile-ring-full | 10.387 | 1200 Hz / 90 deg |
| asym-f45-r55-r35-svd | profile-ring-full | 10.577 | 1200 Hz / 90 deg |
| asym-f45-r55-front35-rear25-svd | profile-ring-full | 10.500 | 1200 Hz / 90 deg |
| h1659-profile-compact-split-discrete-svd | profile-ring-full | 12.062 | 714 Hz / 90 deg |
| h1659-profile-full-split-discrete-svd | profile-ring-full | 12.646 | 899 Hz / 70 deg |
| split-ring-r35-az24 | profile-ring-full | 11.509 | 400 Hz / 90 deg |
| split-ring-r25-95-az48-svd | profile-ring-full | 11.544 | 400 Hz / 90 deg |
| active-surface-dipole-m3 | profile-ring-full | 14.013 | 1133 Hz / 90 deg |
| split-active-dipole-m4g16-svd | modal-full2 | 10.515 | 1039 Hz / 90 deg |
| split-active-measured-m4g30 | profile-ring-full | 11.577 | 1133 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg1e-06 | profile-ring-full-svd | 15.872 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg3e-06 | profile-ring-full-svd | 15.871 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg1e-06 | profile-ring-full-svd | 15.994 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg3e-06 | profile-ring-full-svd | 15.992 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg1e-06 | profile-ring-full-svd | 20.556 | 1166 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg3e-06 | profile-ring-full-svd | 20.557 | 1166 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg1e-06 | profile-ring-full-svd | 20.730 | 1166 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg3e-06 | profile-ring-full-svd | 20.731 | 1166 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06 | profile-ring-full | 13.512 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg3e-06 | profile-ring-full | 14.236 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg1e-06 | profile-ring-full | 13.473 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg3e-06 | profile-ring-full | 14.222 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg1e-06 | profile-ring-full-svd | 15.036 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg3e-06 | profile-ring-full-svd | 16.652 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg1e-06 | profile-ring-full-svd | 14.894 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg3e-06 | profile-ring-full-svd | 16.546 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06 | profile-ring-full | 13.445 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06 | profile-ring-full | 14.025 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06 | profile-ring-full | 13.413 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06 | profile-ring-full | 14.054 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06 | profile-ring-full-svd | 14.091 | 412 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06 | profile-ring-full-svd | 16.099 | 734 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06 | profile-ring-full-svd | 13.865 | 412 Hz / 90 deg |
| physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06 | profile-ring-full-svd | 15.949 | 734 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m2-az16-q3-reg1e-06 | profile-ring-full-svd | 16.350 | 734 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-m2-az16-q3-reg1e-06 | profile-ring-full-svd | 21.471 | 1166 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06 | profile-ring-full | 13.536 | 1166 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-m3-az16-q3-reg1e-06 | modal-full2 | 8.781 | 1100 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06 | profile-ring-full | 13.239 | 1166 Hz / 90 deg |
| physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06 | modal-full2 | 8.155 | 778 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | profile-ring-full | 13.512 | 1200 Hz / 90 deg |
| physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05 | profile-ring-full | 13.445 | 1200 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05 | profile-ring-full | 13.536 | 1166 Hz / 90 deg |
| physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05 | profile-ring-full | 13.239 | 1166 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 11.307 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.636 | 1200 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.383 | 1166 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.586 | 1166 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.532 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.451 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 11.301 | 1200 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.987 | 1166 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.692 | 1166 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.645 | 1166 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.502 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06 | profile-ring-full | 13.408 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 8.603 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | modal-full2 | 8.743 | 1009 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | modal-full2 | 11.794 | 952 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 10.655 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 11.795 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 12.006 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb1-az16-gap16-q3-reg1e-06 | profile-ring-full | 8.303 | 1133 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb1-az16-gap16-q3-reg1e-06 | modal-full2 | 9.596 | 980 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06 | modal-full2 | 12.725 | 778 Hz / 90 deg |
| physical-rear-basket-full-dipole-m2-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 10.759 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m3-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 11.841 | 1069 Hz / 90 deg |
| physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06 | modal-full2 | 12.032 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az24-gap16-q3-reg1e-06 | profile-ring-full | 13.521 | 1166 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az24-gap16-q3-reg1e-06 | profile-ring-full | 13.536 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06 | profile-ring-full | 13.471 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m2-rb2-az24-gap16-q3-reg1e-06 | modal-full2 | 10.549 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m3-rb2-az24-gap16-q3-reg1e-06 | modal-full2 | 11.865 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06 | modal-full2 | 12.167 | 1069 Hz / 90 deg |
| physical-rear-basket-compact-measured-m2-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 13.263 | 1166 Hz / 90 deg |
| physical-rear-basket-compact-measured-m3-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 13.079 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 12.932 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m2-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 13.308 | 1166 Hz / 90 deg |
| physical-rear-basket-full-measured-m3-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 13.029 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06 | profile-ring-full | 12.867 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.741 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.482 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb3-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.581 | 1133 Hz / 90 deg |
| physical-rear-basket-full-measured-m4-rb4-az16-gap16-q3-reg1e-06 | profile-ring-full | 12.272 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb3-az16-gap30-q3-reg1e-06 | profile-ring-full | 12.329 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb4-az16-gap30-q3-reg1e-06 | profile-ring-full | 12.098 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06 | profile-ring-full | 13.553 | 1133 Hz / 90 deg |
| physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05 | profile-ring-full | 13.451 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06 | profile-ring-full | 11.826 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-06 | profile-ring-full | 11.826 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg1e-06-svd1e-05 | profile-ring-full | 11.826 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06 | profile-ring-full | 11.747 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-06 | profile-ring-full | 11.747 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap36-q3-reg1e-06-svd1e-05 | profile-ring-full | 11.747 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06 | profile-ring-full | 11.860 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-06 | profile-ring-full | 11.860 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap30-q3-reg1e-06-svd1e-05 | profile-ring-full | 11.860 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06 | profile-ring-full | 11.807 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-06 | profile-ring-full | 11.807 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap36-q3-reg1e-06-svd1e-05 | profile-ring-full | 11.807 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap24-q3-reg1e-06 | profile-ring-full | 11.932 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap24-q3-reg1e-06 | profile-ring-full | 11.922 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg1e-06 | profile-ring-full | 11.858 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap28-q3-reg1e-06 | profile-ring-full | 11.848 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg1e-06 | profile-ring-full | 11.797 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap32-q3-reg1e-06 | profile-ring-full | 11.788 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap42-q3-reg1e-06 | profile-ring-full | 11.689 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az24-gap42-q3-reg1e-06 | profile-ring-full | 11.683 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-07 | profile-ring-full | 11.494 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap28-q3-reg3e-06 | profile-ring-full | 12.190 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07 | profile-ring-full | 11.488 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06 | profile-ring-full | 12.144 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-07 | profile-ring-full | 11.485 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap32-q3-reg3e-06 | profile-ring-full | 12.101 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-07-svd1e-05 | profile-ring-full | 11.488 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-az16-gap30-q3-reg3e-06-svd1e-05 | profile-ring-full | 12.144 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06 | profile-ring-full | 11.939 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06 | modal-full2 | 9.836 | 1069 Hz / 90 deg |
| sweep-split-active-surface-full-annular-az16-gap24-q3-reg1e-06-svd1e-05 | profile-ring-full | 11.938 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-annular-dipole-az16-gap30-q3-reg1e-06-svd1e-05 | modal-full2 | 9.836 | 1069 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m4 | profile-ring-full | 11.942 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m4 | profile-ring-full | 12.316 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m4 | profile-ring-full | 11.934 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m4 | profile-ring-full | 12.307 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m5 | profile-ring-full | 11.861 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m5 | profile-ring-full | 12.245 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m5 | profile-ring-full | 11.853 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m5 | profile-ring-full | 12.237 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m6 | profile-ring-full | 11.796 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m6 | profile-ring-full | 12.186 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m6 | profile-ring-full | 11.787 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m6 | profile-ring-full | 12.177 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg1e-06-m7 | profile-ring-full | 11.741 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap24-q3-reg3e-06-m7 | profile-ring-full | 12.133 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg1e-06-m7 | profile-ring-full | 11.733 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap24-q3-reg3e-06-m7 | profile-ring-full | 12.125 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m4 | profile-ring-full | 11.877 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m4 | profile-ring-full | 12.223 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m4 | profile-ring-full | 11.868 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m4 | profile-ring-full | 12.214 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m5 | profile-ring-full | 11.804 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m5 | profile-ring-full | 12.157 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m5 | profile-ring-full | 11.795 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m5 | profile-ring-full | 12.148 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m6 | profile-ring-full | 11.745 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m6 | profile-ring-full | 12.101 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m6 | profile-ring-full | 11.736 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m6 | profile-ring-full | 12.092 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg1e-06-m7 | profile-ring-full | 11.697 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap28-q3-reg3e-06-m7 | profile-ring-full | 12.053 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg1e-06-m7 | profile-ring-full | 11.688 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap28-q3-reg3e-06-m7 | profile-ring-full | 12.044 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m4 | profile-ring-full | 11.822 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m4 | profile-ring-full | 12.138 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m4 | profile-ring-full | 11.814 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m4 | profile-ring-full | 12.130 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m5 | profile-ring-full | 11.757 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m5 | profile-ring-full | 12.078 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m5 | profile-ring-full | 11.749 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m5 | profile-ring-full | 12.069 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m6 | profile-ring-full | 11.705 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m6 | profile-ring-full | 12.026 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m6 | profile-ring-full | 11.697 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m6 | profile-ring-full | 12.018 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg1e-06-m7 | profile-ring-full | 11.664 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap32-q3-reg3e-06-m7 | profile-ring-full | 11.982 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg1e-06-m7 | profile-ring-full | 11.656 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap32-q3-reg3e-06-m7 | profile-ring-full | 11.973 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m4 | profile-ring-full | 11.776 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m4 | profile-ring-full | 12.063 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m4 | profile-ring-full | 11.769 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m4 | profile-ring-full | 12.055 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m5 | profile-ring-full | 11.719 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m5 | profile-ring-full | 12.007 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m5 | profile-ring-full | 11.712 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m5 | profile-ring-full | 11.999 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m6 | profile-ring-full | 11.675 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m6 | profile-ring-full | 11.960 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m6 | profile-ring-full | 11.669 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m6 | profile-ring-full | 11.952 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg1e-06-m7 | profile-ring-full | 11.640 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az16-gap36-q3-reg3e-06-m7 | profile-ring-full | 11.919 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg1e-06-m7 | profile-ring-full | 11.634 | 1133 Hz / 90 deg |
| sweep-split-active-surface-compact-uniform-annular-az24-gap36-q3-reg3e-06-m7 | profile-ring-full | 11.912 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m5 | profile-ring-full | 12.030 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m5 | profile-ring-full | 12.399 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg1e-06-m7 | profile-ring-full | 11.908 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap24-q3-reg3e-06-m7 | profile-ring-full | 12.294 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m5 | profile-ring-full | 11.914 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m5 | profile-ring-full | 12.231 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg1e-06-m7 | profile-ring-full | 11.814 | 1133 Hz / 90 deg |
| sweep-split-active-surface-full-uniform-annular-az16-gap32-q3-reg3e-06-m7 | profile-ring-full | 12.140 | 1133 Hz / 90 deg |

## Worst Eligible Spread Surface Points

Rows locate the largest pointwise normalized-polar spread across the Juan-CV recommended finite-source ensemble at Andres' UM-height observer plane.

| frequency | angle | max pairwise spread | low/high cases |
| ---: | ---: | ---: | --- |
| 734 Hz | 90 deg | 26.655 dB | `profile-ring-full` / `modal-full2` |
| 714 Hz | 90 deg | 26.458 dB | `profile-ring-full` / `modal-full2` |
| 756 Hz | 90 deg | 26.268 dB | `profile-ring-full` / `modal-full2` |
| 693 Hz | 90 deg | 25.892 dB | `profile-ring-full` / `modal-full2` |
| 778 Hz | 90 deg | 25.283 dB | `profile-ring-full` / `modal-full2` |
| 673 Hz | 90 deg | 25.194 dB | `profile-ring-full-svd` / `modal-full2` |
| 654 Hz | 90 deg | 24.516 dB | `profile-ring-full-svd` / `modal-full2` |
| 636 Hz | 90 deg | 23.928 dB | `profile-ring-full-svd` / `modal-full2` |
| 801 Hz | 90 deg | 23.907 dB | `profile-ring-full` / `modal-full2` |
| 424 Hz | 90 deg | 23.830 dB | `profile-ring-full-svd` / `modal-full2` |

## Interpretation

Large source-model spread here means the 3D incident field at Andres' mic height is not uniquely constrained by Juan's horizontal naked polars. A BEM result can therefore pass a Juan horizontal source-fit audit while still using a questionable off-plane source field.
This audit does not replace BEM validation or Andres shape metrics. It is now used by the strict gate summary as a source-generalization gate: accepted artifacts must keep both z=0-to-UM-height movement and audited source-family spread within the 1.5 dB normalized-polar target over 70-90 deg before width sweeps can be treated quantitatively. When available, the gate uses the Juan-CV recommended finite-source ensemble spread instead of the legacy wide-SVD-relative spread.

## Files

- `source_offplane_summary.csv` contains z=0 to UM-height changes by source and angle group.
- `source_pairwise_offplane_summary.csv` contains source-model differences at UM height.
- `source_eligible_pairwise_offplane_summary.csv` contains source-model differences against the Juan-CV recommended finite-source ensemble.
- `source_eligible_pairwise_spread_surface.csv` contains the pointwise max pairwise spread across the Juan-CV recommended finite-source ensemble.
- `plots/source_offplane_delta_by_angle.png`, `plots/source_pairwise_offplane_70_90.png`, `plots/source_eligible_pairwise_offplane_70_90.png`, and `plots/source_eligible_pairwise_spread_contour.png` summarize the spread.
