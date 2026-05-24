# L22MG Source Candidate Decision

This joins Juan-only naked-source cross-validation with stored mounted-baffle BEM validation evidence. It does not fit to the baffle target and does not rerun BEM.

Decision policy:

- A source that passes Juan-only CV is not accepted unless a mapped BEM row also passes the canonical validation gates.
- Mapped BEM evidence must use the current canonical Juan top-baffle target `polar_data_juan_lx521_top_raw.h5` / `L22MG (LX521 top raw)`; explicit Andres published-parity rows remain legacy evidence, but first-lobe, direct-gate, or strongest-lobe diagnostics are never accepted here.
- Mapped BEM evidence must also use the current Juan naked-source reference radius, 0.50 m; old 0.75 m artifacts are treated as stale and are not joined as current BEM evidence.
- A stored hotband BEM row cannot prove full 300-1200 Hz acceptance, but a large hotband failure is enough to reject an immediate replacement claim.
- Juan-only source-CV recommendation requires held-out normalized shape plus in-sample normalized, absolute SPL, and phase fit gates; rows without phase cells are called out explicitly and are not treated as current source-phase evidence.
- Off-plane source ambiguity is reported separately because Juan's horizontal naked polars do not by themselves prove the 3D incident field at Andres' UM-height mic.
- When available, off-plane spread uses only Juan-CV `recommended_juan_only` finite sources as the reference ensemble; legacy reference spread is retained only for rows outside that eligible ensemble.
- The normalized polar shape target remains <= 1.5 dB over 300-1200 Hz; source CV is a modeling diagnostic, not a substitute for validation.

Summary:

- Source-CV sets included: l22mg-source-model-cross-validation, l22mg-source-cv-active-surface-annular-sweep, l22mg-source-cv-active-surface-annular-extended, l22mg-source-cv-active-surface-uniform-annular-local, l22mg-source-cv-active-surface-annular-fine, l22mg-source-cv-physical-diaphragm, l22mg-source-cv-physical-diaphragm-fsmooth, l22mg-source-cv-physical-rear-basket.
- BEM gate summary inputs: `docs/l22mg-validation-gate-summary/validation_gate_summary.csv`, `docs/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-juan-top-h1659-modal-compact2-current-width-sweep-h42-h28-q7near-targetavg7-gates/validation_gate_summary.csv`, `docs/l22mg-coupled-rear-basket-source-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-coupled-rear-basket-m2rb1-source-smoke-gates/validation_gate_summary.csv`.
- Andres published-parity gate summary inputs: `docs/l22mg-validation-gate-andres-published-parity/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-physical-rear-basket-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-physical-rear-basket-m2rb1-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-coupled-rear-basket-feature-proxy-smoke-gates/validation_gate_summary.csv`.
- Andres published-parity BEM rows are reported as a separate legacy comparison channel and do not change the canonical Juan decision status.
- Joined source-candidate evidence scope: 129 current-radius rows.
- Accepted source candidates in this joined evidence set: 0.
- Current five-width BEM sweep evidence: `docs/l22mg-bem-juan-top-h1659-modal-compact2-current-width-sweep-h42-h28-q7near-targetavg7-gates/validation_gate_summary.csv`. It is used for current-radius width reporting and remains subject to the same gate ordering as any other stored BEM row; it cannot make a failing baseline accepted by itself.
- Physical rear-filter diagnostic: `docs/l22mg-physical-rear-filter/rear_filter_summary.csv`; delay-only fit -0.222 ms (-76.197 mm path), delay phase residual 26.600 deg, bounded-filter rear normalized/absolute/phase RMS 3.180 / 6.500 / 62.586. This is Juan-only context and cannot make a rejected source acceptable.
- Rear shared-sign diagnostic: `docs/l22mg-rear-sign-convention/rear_sign_summary.csv`; best normalized Juan front/rear fit keeps sign `-1` (-1 gives 2.635 dB, +1 gives 3.584 dB). Best absolute fit is -1 4.323 dB vs +1 4.364 dB; best phase is -1 48.099 deg vs +1 48.265 deg. This is Juan-only sign-convention context and cannot make a rejected source acceptable.
- Physical source path: 100 physical diaphragm/rear-basket rows are joined; 0 are Juan-only recommended, 100 are rejected by Juan source-CV, 2 have Andres published-parity side-channel BEM rows, and 0 are accepted by current evidence.
- Best physical Juan-CV row: `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05` from `l22mg-source-cv-physical-diaphragm`; recommendation `rejected_bad_insample_fit`; held-out side-worst 3.448 dB; in-sample normalized/absolute/phase 3.119 / 4.712 dB / 62.053 deg; decision `rejected_by_source_cv`.
- Best physical Andres side-channel BEM row: `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06` maps to `physical-rear-basket-andres-smoke` (available_overlap_300_301_hz); all-angle RMS 6.869 dB, through-80 1.218 dB, source off-plane gate `fail`. This remains side-channel evidence and does not override Juan source-CV rejection or the canonical target gates.
- Physical source path status: no physical source row is an acceptance candidate yet; the next path is better Juan-only physical-source fit and mapped BEM evidence, not a scalar dB correction or Andres-tuned source knob.
- Current wide/SVD source: rejected_by_source_cv; Juan held-out worst-side CV 8.810 dB; mapped BEM nan dB; off-plane spread 10.949 dB (source_cv_recommended_juan_only vs `profile-ring-full`).
- Juan-only recommended sources contradicted by mapped BEM evidence: h1659-modal-compact2-az24-reg1e-06, h1659-modal-full2-az24-reg1e-06, h1659-modal-compact3-az24-reg1e-06, h1659-modal-full4-az24-reg1e-06-svd1e-05, h1659-profile-ring-full-az24-reg1e-06-svd1e-05, h1659-profile-ring-full-az24-reg1e-06. Their stored BEM checks fail the canonical validation target.

| set | rank | model | Juan status | Juan CV dB | in-sample norm | in-sample abs | phase deg | off-plane z | off-plane spread | spread basis | surface max | BEM evidence | BEM RMS dB | through 80 dB | worst error | decision | Andres parity | Andres RMS dB | Andres through 80 dB |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| `l22mg-source-model-cross-validation` | 1 | `axisymmetric-directivity-table-front-duplicate` | diagnostic_only | 1.229 | 0.569 | 0.569 | 68.034 | 0.147 | 13.940 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | `axisymmetric-directivity-juan-current` (available_overlap_1000_1200_hz) | 5.768 | 3.344 | +13.8 @ 1000 Hz/90 deg | diagnostic_only | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 2 | `diagnostic-legendre-o8` | diagnostic_only | 2.201 | 9.203 | 9.145 | 62.252 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | diagnostic_only | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 3 | `h1659-modal-compact2-az24-reg1e-06` | recommended_juan_only | 3.298 | 2.631 | 2.578 | 15.048 | 0.150 | 13.249 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | `compact2-h32-h21-h28-h18-targetavg7-meshtarget-conv-smoke` (full_300_1200) | 3.920 | 3.491 | +19.1 @ 300 Hz/75 deg | rejected_by_full_band_bem | `andres-published-parity-q7near-h50-h42` (available_overlap_300_360_hz) | 8.852 | 0.989 |
| `l22mg-source-model-cross-validation` | 4 | `h1659-modal-full2-az24-reg1e-06` | recommended_juan_only | 3.300 | 2.682 | 2.629 | 15.134 | 0.153 | 13.406 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | `q7near-solid-full2-h42-h36` (full_300_1200) | 4.270 | 3.558 | +19.0 @ 300 Hz/75 deg | rejected_by_full_band_bem | `andres-published-parity-h1659-modal-full2-current` (full_300_1200) | 4.428 | 1.377 |
| `l22mg-source-model-cross-validation` | 5 | `measured-phase-h1659-modal-full2-az24-reg1e-06` | rejected_bad_insample_fit | 3.494 | 5.967 | 6.200 | 62.847 | 0.229 | 16.001 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 6 | `h1659-modal-compact3-az24-reg1e-06` | recommended_juan_only | 3.768 | 2.215 | 2.184 | 14.029 | 0.137 | 11.570 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | `q7near-solid-compact3-h42-h36` (full_300_1200) | 5.410 | 4.308 | +18.2 @ 300 Hz/75 deg | rejected_by_full_band_bem | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 7 | `h1659-modal-full4-az24-reg1e-06-svd1e-05` | recommended_juan_only | 3.800 | 1.050 | 1.039 | 12.999 | 0.103 | 8.946 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | `q7near-solid-full4-svd-h42-h36` (full_300_1200) | 10.911 | 8.705 | +35.0 @ 1166 Hz/90 deg | rejected_by_full_band_bem | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 8 | `h1659-profile-ring-full-az24-reg1e-06-svd1e-05` | recommended_juan_only | 3.842 | 1.091 | 1.030 | 12.298 | 0.139 | 13.406 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | `q7near-solid-profile-ring-full-svd-h42-h36` (full_300_1200) | 9.254 | 6.696 | +34.6 @ 939 Hz/90 deg | rejected_by_full_band_bem | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 9 | `h1659-profile-ring-full-az24-reg1e-06` | recommended_juan_only | 3.844 | 1.090 | 1.030 | 12.298 | 0.139 | 13.406 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | `q7near-solid-profile-ring-full-h42-h36` (full_300_1200) | 9.265 | 6.707 | +34.6 @ 940 Hz/90 deg | rejected_by_full_band_bem | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 10 | `h1659-profile-ring-compact-az24-reg1e-06` | rejected_weak_cv | 4.090 | 1.093 | 1.033 | 12.283 | 0.141 | 13.358 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 11 | `h1659-modal-full3-az24-reg1e-06` | rejected_weak_cv | 4.090 | 2.184 | 2.156 | 13.920 | 0.135 | 11.242 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 12 | `h1659-profile-ring-compact-az24-reg1e-06-svd1e-05` | rejected_weak_cv | 4.092 | 1.093 | 1.033 | 12.283 | 0.141 | 13.359 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 13 | `split-active-surface-compact-m4-az16-gap30-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 4.531 | 3.766 | 4.046 | 42.570 | 0.151 | 11.577 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 14 | `measured-phase-h1659-modal-full4-az24-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 5.011 | 5.119 | 5.207 | 38.861 | 0.173 | 12.040 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 15 | `split-active-surface-compact-m4-az16-gap16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 5.468 | 7.807 | 7.741 | 54.561 | 0.112 | 10.515 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 16 | `h1659-profile-compact-split-discrete-d45-r0_40_75-reg1e-06-svd1e-05` | rejected_weak_cv | 6.649 | 1.387 | 1.372 | 11.159 | 1.211 | 12.062 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 17 | `asym-split-discrete-f45-r55-r0_35_70_95-reg1e-06-svd1e-05` | rejected_weak_cv | 6.793 | 2.656 | 2.627 | 15.474 | 0.473 | 10.577 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 18 | `asym-split-discrete-f45-r55-front35-rear25-reg1e-06-svd1e-05` | rejected_weak_cv | 6.815 | 2.639 | 2.612 | 15.338 | 0.890 | 10.500 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 19 | `d55-split-discrete-r0_25_50_75_95-reg1e-06-svd1e-05` | rejected_weak_cv | 6.874 | 2.639 | 2.612 | 15.338 | 0.568 | 10.387 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 20 | `h1659-profile-full-split-discrete-d45-r0_40_75-reg1e-06-svd1e-05` | rejected_weak_cv | 6.893 | 1.397 | 1.383 | 11.103 | 1.320 | 12.646 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 21 | `active-surface-compact-m3-az16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 7.696 | 6.111 | 5.473 | 67.330 | 0.232 | 14.013 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 22 | `measured-phase-split-discrete-d45-r0_40_75-reg1e-06` | rejected_weak_cv | 7.707 | 2.862 | 2.667 | 27.759 | 1.735 | 14.486 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 23 | `measured-phase-h1659-profile-ring-compact-az24-reg1e-06` | rejected_bad_insample_fit | 8.415 | 4.595 | 4.333 | 24.572 | 0.197 | 11.958 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 24 | `measured-phase-h1659-profile-ring-full-az24-reg1e-06` | rejected_bad_insample_fit | 8.557 | 4.653 | 4.383 | 24.630 | 0.195 | 11.934 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 25 | `stable-split-discrete-d45-r0_40_75-reg1e-06` | rejected_weak_cv | 8.622 | 2.642 | 2.598 | 16.308 | 0.616 | 10.862 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 26 | `current-wide-split-discrete-d45-r0_35_70_95-reg1e-06-svd1e-05` | rejected_weak_cv | 8.810 | 2.500 | 2.469 | 14.961 | 0.546 | 10.949 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 27 | `wide-split-discrete-d45-r0_35_70_95-reg1e-06` | rejected_weak_cv | 8.810 | 2.500 | 2.469 | 14.960 | 0.546 | 10.949 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 28 | `split-ring-d45-r0_25_50_75_95-az48-reg1e-06-svd1e-05` | rejected_weak_cv | 9.274 | 2.936 | 2.952 | 17.497 | 0.248 | 11.544 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-model-cross-validation` | 29 | `split-ring-d45-r0_35_70_95-az24-reg1e-06` | rejected_weak_cv | 9.279 | 2.944 | 2.958 | 17.545 | 0.248 | 11.509 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 1 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 3.448 | 3.119 | 4.712 | 62.053 | 0.224 | 13.239 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 2 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06` | rejected_bad_insample_fit | 3.448 | 3.119 | 4.712 | 62.053 | 0.224 | 13.239 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 3 | `physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06` | rejected_bad_insample_fit | 3.545 | 3.758 | 5.262 | 69.518 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 4 | `physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 3.545 | 3.758 | 5.262 | 69.518 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 5 | `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06` | rejected_bad_insample_fit | 4.486 | 6.521 | 6.347 | 44.957 | 0.126 | 8.155 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 6 | `physical-diaphragm-full-coupled-dipole-m3-az16-q3-reg1e-06` | rejected_bad_insample_fit | 5.117 | 7.419 | 7.232 | 55.566 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 7 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06` | rejected_bad_insample_fit | 5.287 | 4.299 | 4.100 | 66.340 | 0.181 | 16.099 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 8 | `physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06` | rejected_bad_insample_fit | 5.364 | 4.307 | 4.103 | 65.744 | 0.179 | 15.949 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 9 | `physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06` | rejected_bad_insample_fit | 5.407 | 4.216 | 5.612 | 68.839 | 0.089 | 13.413 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 10 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06` | rejected_bad_insample_fit | 5.569 | 4.286 | 5.754 | 69.514 | 0.103 | 13.445 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 11 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 5.569 | 4.286 | 5.754 | 69.514 | 0.103 | 13.445 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 12 | `physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg3e-06` | rejected_bad_insample_fit | 5.691 | 4.854 | 4.637 | 68.097 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 13 | `physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg3e-06` | rejected_bad_insample_fit | 5.850 | 4.961 | 4.725 | 68.069 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 14 | `physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg1e-06` | rejected_bad_insample_fit | 5.858 | 5.086 | 4.261 | 69.406 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 15 | `physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg3e-06` | rejected_bad_insample_fit | 5.859 | 5.087 | 4.256 | 69.402 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 16 | `physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg1e-06` | rejected_bad_insample_fit | 5.993 | 5.210 | 4.329 | 69.507 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 17 | `physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg3e-06` | rejected_bad_insample_fit | 5.995 | 5.211 | 4.322 | 69.503 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 18 | `physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg1e-06` | rejected_bad_insample_fit | 6.094 | 4.300 | 6.278 | 78.906 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 19 | `physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06` | rejected_bad_insample_fit | 6.135 | 4.679 | 4.389 | 61.127 | 0.162 | 13.865 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 20 | `physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06` | rejected_bad_insample_fit | 6.282 | 4.422 | 6.374 | 78.889 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 21 | `physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 6.282 | 4.422 | 6.374 | 78.889 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 22 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06` | rejected_bad_insample_fit | 6.348 | 4.636 | 4.336 | 61.881 | 0.164 | 14.091 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 23 | `physical-diaphragm-full-coupled-dipole-m2-az16-q3-reg1e-06` | rejected_bad_insample_fit | 6.588 | 5.732 | 4.806 | 70.948 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 24 | `physical-diaphragm-full-coupled-measured-m2-az16-q3-reg1e-06` | rejected_bad_insample_fit | 6.686 | 6.615 | 7.607 | 91.805 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 25 | `physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06` | rejected_bad_insample_fit | 7.383 | 5.545 | 6.990 | 73.252 | 0.132 | 14.054 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 26 | `physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg1e-06` | rejected_bad_insample_fit | 7.452 | 5.786 | 5.351 | 67.013 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 27 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06` | rejected_bad_insample_fit | 7.553 | 5.689 | 7.182 | 74.091 | 0.118 | 14.025 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 28 | `physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg1e-06` | rejected_bad_insample_fit | 7.690 | 5.899 | 5.449 | 66.941 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 29 | `physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg3e-06` | rejected_bad_insample_fit | 7.787 | 7.080 | 8.148 | 90.069 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 30 | `physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg1e-06` | rejected_bad_insample_fit | 7.897 | 6.993 | 8.052 | 90.015 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 31 | `physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg3e-06` | rejected_bad_insample_fit | 8.157 | 7.136 | 8.218 | 90.113 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 32 | `physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg3e-06` | rejected_bad_insample_fit | 8.173 | 5.782 | 7.442 | 79.890 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 33 | `physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg1e-06` | rejected_bad_insample_fit | 8.318 | 7.064 | 8.138 | 90.065 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm` | 34 | `physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg3e-06` | rejected_bad_insample_fit | 8.380 | 5.921 | 7.568 | 79.983 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 1 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06` | rejected_bad_insample_fit | 3.481 | 3.159 | 4.726 | 61.985 | 0.224 | 13.239 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 2 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_insample_fit | 3.482 | 3.159 | 4.726 | 61.986 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 3 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_insample_fit | 3.483 | 3.162 | 4.730 | 61.991 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 4 | `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_insample_fit | 3.494 | 3.192 | 4.764 | 62.039 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 5 | `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06` | rejected_bad_insample_fit | 4.486 | 6.543 | 6.368 | 44.864 | 0.126 | 8.155 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 6 | `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_insample_fit | 4.486 | 6.543 | 6.369 | 44.864 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 7 | `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_insample_fit | 4.487 | 6.545 | 6.371 | 44.871 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 8 | `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_insample_fit | 4.494 | 6.564 | 6.392 | 44.941 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 9 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06` | rejected_bad_insample_fit | 5.612 | 4.328 | 5.775 | 69.327 | 0.103 | 13.445 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 10 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_insample_fit | 5.612 | 4.328 | 5.775 | 69.327 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 11 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_insample_fit | 5.613 | 4.329 | 5.777 | 69.333 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 12 | `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_insample_fit | 5.618 | 4.337 | 5.793 | 69.392 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 13 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06` | rejected_bad_insample_fit | 6.404 | 4.696 | 4.399 | 61.893 | 0.164 | 14.091 | source_cv_recommended_juan_only vs `profile-ring-full-svd` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 14 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_insample_fit | 6.404 | 4.696 | 4.399 | 61.893 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 15 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_insample_fit | 6.407 | 4.695 | 4.399 | 61.895 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-diaphragm-fsmooth` | 16 | `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_insample_fit | 6.439 | 4.692 | 4.398 | 61.910 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 1 | `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_source_level_fit | 3.592 | 2.635 | 4.543 | 72.737 | nan | nan | missing | n/a | `physical-rear-basket-m2rb1-juan-smoke` (available_overlap_300_301_hz) | 6.447 | 6.959 | +17.0 @ 300 Hz/75 deg | rejected_by_source_cv_with_failed_bem_overlap | `physical-rear-basket-m2rb1-andres-smoke` (available_overlap_300_301_hz) | 10.021 | 1.719 |
| `l22mg-source-cv-physical-rear-basket` | 2 | `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_source_level_fit | 3.592 | 2.635 | 4.543 | 72.737 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 3 | `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_source_level_fit | 3.593 | 2.635 | 4.544 | 72.738 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 4 | `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_source_level_fit | 3.596 | 2.640 | 4.557 | 72.743 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 5 | `physical-rear-basket-full-measured-m2-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_source_level_fit | 3.623 | 2.683 | 4.555 | 71.471 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 6 | `physical-rear-basket-full-dipole-m2-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 3.767 | 4.351 | 4.045 | 44.366 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 7 | `physical-rear-basket-compact-dipole-m2-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 3.789 | 4.282 | 3.952 | 44.781 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 8 | `physical-rear-basket-compact-measured-m2-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.076 | 3.824 | 4.356 | 52.698 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 9 | `physical-rear-basket-full-measured-m2-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.087 | 3.835 | 4.359 | 52.295 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 10 | `physical-rear-basket-compact-measured-m2-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.131 | 3.767 | 4.316 | 52.820 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 11 | `physical-rear-basket-compact-measured-m3-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.144 | 4.270 | 4.657 | 48.468 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 12 | `physical-rear-basket-compact-measured-m2-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.157 | 3.796 | 4.323 | 52.323 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 13 | `physical-rear-basket-full-measured-m3-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.158 | 4.136 | 4.506 | 48.127 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 14 | `physical-rear-basket-full-measured-m2-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.180 | 3.813 | 4.323 | 51.958 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 15 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06` | rejected_bad_insample_fit | 4.189 | 4.111 | 4.597 | 51.542 | 0.135 | 13.553 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 16 | `physical-rear-basket-full-measured-m3-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.195 | 4.607 | 5.011 | 48.779 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 17 | `physical-rear-basket-compact-measured-m3-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.197 | 4.377 | 4.808 | 49.410 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 18 | `physical-rear-basket-compact-measured-m3-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.199 | 4.431 | 4.852 | 49.055 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 19 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.201 | 4.013 | 4.354 | 47.665 | 0.135 | 12.932 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 20 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.215 | 4.693 | 5.069 | 48.476 | 0.142 | 13.451 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | `coupled-rear-basket-source-smoke` (available_overlap_300_301_hz) | 7.608 | 8.189 | +19.8 @ 300 Hz/75 deg | rejected_by_source_cv_with_failed_bem_overlap | `physical-rear-basket-andres-smoke` (available_overlap_300_301_hz) | 6.869 | 1.218 |
| `l22mg-source-cv-physical-rear-basket` | 21 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05` | rejected_bad_insample_fit | 4.215 | 4.693 | 5.069 | 48.476 | 0.142 | 13.451 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 22 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-08` | rejected_bad_insample_fit | 4.216 | 4.693 | 5.069 | 48.476 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 23 | `physical-rear-basket-compact-measured-m4-rb3-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.216 | 4.055 | 4.376 | 46.303 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 24 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-07` | rejected_bad_insample_fit | 4.217 | 4.695 | 5.072 | 48.478 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 25 | `physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.218 | 4.719 | 5.100 | 48.667 | 0.142 | 13.471 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 26 | `physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.225 | 4.460 | 4.818 | 48.099 | 0.143 | 13.408 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 27 | `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-06` | rejected_bad_insample_fit | 4.227 | 4.715 | 5.094 | 48.499 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 28 | `physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.234 | 3.913 | 4.232 | 47.194 | 0.136 | 12.867 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 29 | `physical-rear-basket-compact-measured-m4-rb4-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.252 | 3.971 | 4.275 | 45.464 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 30 | `physical-rear-basket-full-measured-m4-rb3-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.254 | 3.953 | 4.251 | 45.659 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 31 | `physical-rear-basket-compact-measured-m4-rb3-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.299 | 3.807 | 4.107 | 45.292 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 32 | `physical-rear-basket-full-measured-m4-rb4-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.303 | 3.890 | 4.172 | 44.643 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 33 | `physical-rear-basket-compact-measured-m4-rb4-az16-gap30-q3-reg1e-06` | rejected_bad_insample_fit | 4.361 | 3.784 | 4.069 | 44.257 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 34 | `physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.500 | 7.684 | 7.672 | 46.522 | 0.135 | 12.725 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 35 | `physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.548 | 3.829 | 4.406 | 57.445 | 0.202 | 13.692 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 36 | `physical-rear-basket-compact-measured-m3-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.585 | 3.432 | 4.510 | 64.301 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 37 | `physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.624 | 3.685 | 4.400 | 60.218 | 0.218 | 13.383 | source_cv_recommended_juan_only vs `profile-ring-full` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 38 | `physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.741 | 6.632 | 6.622 | 47.115 | 0.094 | 11.794 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 39 | `physical-rear-basket-full-measured-m3-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.795 | 3.540 | 4.502 | 62.871 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 40 | `physical-rear-basket-full-dipole-m3-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.858 | 5.762 | 5.698 | 47.077 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 41 | `physical-rear-basket-compact-dipole-m3-rb1-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 4.871 | 5.417 | 5.308 | 47.400 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 42 | `physical-rear-basket-compact-dipole-m3-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.456 | 9.143 | 9.086 | 52.112 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 43 | `physical-rear-basket-compact-dipole-m2-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.460 | 7.922 | 7.858 | 49.757 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 44 | `physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.466 | 9.370 | 9.315 | 53.761 | 0.126 | 12.167 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 45 | `physical-rear-basket-compact-dipole-m3-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.507 | 9.150 | 9.092 | 52.445 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 46 | `physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.510 | 9.307 | 9.251 | 53.952 | 0.123 | 12.006 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 47 | `physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.532 | 9.439 | 9.383 | 54.974 | 0.124 | 12.032 | source_cv_recommended_juan_only vs `modal-full2` | 26.655 | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 48 | `physical-rear-basket-compact-dipole-m2-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.536 | 8.010 | 7.946 | 50.314 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 49 | `physical-rear-basket-full-dipole-m3-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.540 | 9.221 | 9.164 | 53.402 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |
| `l22mg-source-cv-physical-rear-basket` | 50 | `physical-rear-basket-full-dipole-m2-rb2-az16-gap16-q3-reg1e-06` | rejected_bad_insample_fit | 5.598 | 8.106 | 8.042 | 50.768 | n/a | n/a | n/a | n/a | missing | nan | nan | nan | rejected_by_source_cv | missing | nan | nan |

## Detail

### axisymmetric-directivity-table-front-duplicate

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 1.
- Source: `axisymmetric-directivity`; profile `measured_axisymmetric_directivity`; sources/bases 13/13.
- Juan-only recommendation: `diagnostic_only`; worst held-out side RMS 1.229 dB; in-sample side-worst normalized/absolute 0.569 / 0.569 dB; phase 68.034 deg.
- Source off-plane ambiguity: case `axisymmetric-directivity`; z=0 to UM-height RMS 70-90 deg 0.147 dB; source-family spread at UM height 13.940 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `axisymmetric-directivity`; z=0 to UM-height normalized-polar change is 0.147 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.940 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `axisymmetric-directivity-juan-current` (available_overlap_1000_1200_hz); all-angle RMS 5.768 dB; through-60/through-80 RMS 2.987 / 3.344 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `diagnostic_only`. This source family is diagnostic-only and is not a finite source support for BEM source selection.

### diagnostic-legendre-o8

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 2.
- Source: `legendre`; profile `symmetric-depth-rings`; sources/bases 0/0.
- Juan-only recommendation: `diagnostic_only`; worst held-out side RMS 2.201 dB; in-sample side-worst normalized/absolute 9.203 / 9.145 dB; phase 62.252 deg.
- Source off-plane ambiguity: case `not_applicable_diagnostic_only`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Diagnostic-only source family is not eligible for acceptance; off-plane source ambiguity metrics are not required for source selection.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `diagnostic_only`. This source family is diagnostic-only and is not a finite source support for BEM source selection.

### h1659-modal-compact2-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 3.
- Source: `split-profile-ring`; profile `h1659-modal-compact-2`; sources/bases 292/4.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.298 dB; in-sample side-worst normalized/absolute 2.631 / 2.578 dB; phase 15.048 deg.
- Source off-plane ambiguity: case `modal-compact2`; z=0 to UM-height RMS 70-90 deg 0.150 dB; source-family spread at UM height 13.249 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `modal-compact2`; z=0 to UM-height normalized-polar change is 0.150 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.249 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `compact2-h32-h21-h28-h18-targetavg7-meshtarget-conv-smoke` (full_300_1200); all-angle RMS 3.920 dB; through-60/through-80 RMS 0.875 / 3.491 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: `andres-published-parity-q7near-h50-h42` (available_overlap_300_360_hz); all-angle RMS 8.852 dB; through-80 RMS 0.989 dB; target gate `pass`; published explorer match `True`; source off-plane gate `fail`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.298 dB) but the mapped full-band validation/BEM row fails shape at 3.920 dB.

### h1659-modal-full2-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 4.
- Source: `split-profile-ring`; profile `h1659-modal-full-2`; sources/bases 484/4.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.300 dB; in-sample side-worst normalized/absolute 2.682 / 2.629 dB; phase 15.134 deg.
- Source off-plane ambiguity: case `modal-full2`; z=0 to UM-height RMS 70-90 deg 0.153 dB; source-family spread at UM height 13.406 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `modal-full2`; z=0 to UM-height normalized-polar change is 0.153 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 13.406 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `q7near-solid-full2-h42-h36` (full_300_1200); all-angle RMS 4.270 dB; through-60/through-80 RMS 0.970 / 3.558 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: `andres-published-parity-h1659-modal-full2-current` (full_300_1200); all-angle RMS 4.428 dB; through-80 RMS 1.377 dB; target gate `pass`; published explorer match `True`; source off-plane gate `fail`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.300 dB) but the mapped full-band validation/BEM row fails shape at 4.270 dB.

### measured-phase-h1659-modal-full2-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 5.
- Source: `split-profile-ring`; profile `h1659-modal-full-2`; sources/bases 484/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.494 dB; in-sample side-worst normalized/absolute 5.967 / 6.200 dB; phase 62.847 deg.
- Source off-plane ambiguity: case `measured-modal-full2`; z=0 to UM-height RMS 70-90 deg 0.229 dB; source-family spread at UM height 16.001 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `measured-modal-full2`; z=0 to UM-height normalized-polar change is 0.229 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 16.001 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.967 dB exceeds 3.000 dB.

### h1659-modal-compact3-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 6.
- Source: `split-profile-ring`; profile `h1659-modal-compact-3`; sources/bases 438/6.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.768 dB; in-sample side-worst normalized/absolute 2.215 / 2.184 dB; phase 14.029 deg.
- Source off-plane ambiguity: case `modal-compact3`; z=0 to UM-height RMS 70-90 deg 0.137 dB; source-family spread at UM height 11.570 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `modal-compact3`; z=0 to UM-height normalized-polar change is 0.137 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 11.570 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `q7near-solid-compact3-h42-h36` (full_300_1200); all-angle RMS 5.410 dB; through-60/through-80 RMS 2.040 / 4.308 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.768 dB) but the mapped full-band validation/BEM row fails shape at 5.410 dB.

### h1659-modal-full4-az24-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 7.
- Source: `split-profile-ring`; profile `h1659-modal-full-4`; sources/bases 968/8.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.800 dB; in-sample side-worst normalized/absolute 1.050 / 1.039 dB; phase 12.999 deg.
- Source off-plane ambiguity: case `modal-full4-svd`; z=0 to UM-height RMS 70-90 deg 0.103 dB; source-family spread at UM height 8.946 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `modal-full4-svd`; z=0 to UM-height normalized-polar change is 0.103 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 8.946 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `q7near-solid-full4-svd-h42-h36` (full_300_1200); all-angle RMS 10.911 dB; through-60/through-80 RMS 5.685 / 8.705 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.800 dB) but the mapped full-band validation/BEM row fails shape at 10.911 dB.

### h1659-profile-ring-full-az24-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 8.
- Source: `split-profile-ring`; profile `h1659-acoustic-full`; sources/bases 242/12.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.842 dB; in-sample side-worst normalized/absolute 1.091 / 1.030 dB; phase 12.298 deg.
- Source off-plane ambiguity: case `profile-ring-full-svd`; z=0 to UM-height RMS 70-90 deg 0.139 dB; source-family spread at UM height 13.406 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `profile-ring-full-svd`; z=0 to UM-height normalized-polar change is 0.139 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 13.406 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `q7near-solid-profile-ring-full-svd-h42-h36` (full_300_1200); all-angle RMS 9.254 dB; through-60/through-80 RMS 3.996 / 6.696 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.842 dB) but the mapped full-band validation/BEM row fails shape at 9.254 dB.

### h1659-profile-ring-full-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 9.
- Source: `split-profile-ring`; profile `h1659-acoustic-full`; sources/bases 242/12.
- Juan-only recommendation: `recommended_juan_only`; worst held-out side RMS 3.844 dB; in-sample side-worst normalized/absolute 1.090 / 1.030 dB; phase 12.298 deg.
- Source off-plane ambiguity: case `profile-ring-full`; z=0 to UM-height RMS 70-90 deg 0.139 dB; source-family spread at UM height 13.406 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `profile-ring-full`; z=0 to UM-height normalized-polar change is 0.139 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 13.406 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `q7near-solid-profile-ring-full-h42-h36` (full_300_1200); all-angle RMS 9.265 dB; through-60/through-80 RMS 4.005 / 6.707 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_full_band_bem`. Passes Juan-only CV (3.844 dB) but the mapped full-band validation/BEM row fails shape at 9.265 dB.

### h1659-profile-ring-compact-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 10.
- Source: `split-profile-ring`; profile `h1659-acoustic-compact`; sources/bases 146/8.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 4.090 dB; in-sample side-worst normalized/absolute 1.093 / 1.033 dB; phase 12.283 deg.
- Source off-plane ambiguity: case `profile-ring-compact`; z=0 to UM-height RMS 70-90 deg 0.141 dB; source-family spread at UM height 13.358 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `profile-ring-compact`; z=0 to UM-height normalized-polar change is 0.141 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 13.358 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 4.090 dB exceeds 4.000 dB.

### h1659-modal-full3-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 11.
- Source: `split-profile-ring`; profile `h1659-modal-full-3`; sources/bases 726/6.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 4.090 dB; in-sample side-worst normalized/absolute 2.184 / 2.156 dB; phase 13.920 deg.
- Source off-plane ambiguity: case `modal-full3`; z=0 to UM-height RMS 70-90 deg 0.135 dB; source-family spread at UM height 11.242 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `modal-full3`; z=0 to UM-height normalized-polar change is 0.135 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 11.242 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 4.090 dB exceeds 4.000 dB.

### h1659-profile-ring-compact-az24-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 12.
- Source: `split-profile-ring`; profile `h1659-acoustic-compact`; sources/bases 146/8.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 4.092 dB; in-sample side-worst normalized/absolute 1.093 / 1.033 dB; phase 12.283 deg.
- Source off-plane ambiguity: case `profile-ring-compact-svd`; z=0 to UM-height RMS 70-90 deg 0.141 dB; source-family spread at UM height 13.359 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `profile-ring-compact-svd`; z=0 to UM-height normalized-polar change is 0.141 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 13.359 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 4.092 dB exceeds 4.000 dB.

### split-active-surface-compact-m4-az16-gap30-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 13.
- Source: `split-active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 8/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.531 dB; in-sample side-worst normalized/absolute 3.766 / 4.046 dB; phase 42.570 deg.
- Source off-plane ambiguity: case `split-active-measured-m4g30`; z=0 to UM-height RMS 70-90 deg 0.151 dB; source-family spread at UM height 11.577 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `split-active-measured-m4g30`; z=0 to UM-height normalized-polar change is 0.151 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 11.577 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.766 dB exceeds 3.000 dB.

### measured-phase-h1659-modal-full4-az24-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 14.
- Source: `split-profile-ring`; profile `h1659-modal-full-4`; sources/bases 968/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.011 dB; in-sample side-worst normalized/absolute 5.119 / 5.207 dB; phase 38.861 deg.
- Source off-plane ambiguity: case `measured-modal-full4-svd`; z=0 to UM-height RMS 70-90 deg 0.173 dB; source-family spread at UM height 12.040 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `measured-modal-full4-svd`; z=0 to UM-height normalized-polar change is 0.173 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 12.040 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.119 dB exceeds 3.000 dB.

### split-active-surface-compact-m4-az16-gap16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 15.
- Source: `split-active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 8/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.468 dB; in-sample side-worst normalized/absolute 7.807 / 7.741 dB; phase 54.561 deg.
- Source off-plane ambiguity: case `split-active-dipole-m4g16-svd`; z=0 to UM-height RMS 70-90 deg 0.112 dB; source-family spread at UM height 10.515 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `split-active-dipole-m4g16-svd`; z=0 to UM-height normalized-polar change is 0.112 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 10.515 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.807 dB exceeds 3.000 dB.

### h1659-profile-compact-split-discrete-d45-r0_40_75-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 16.
- Source: `split-discrete`; profile `h1659-acoustic-compact`; sources/bases 14/14.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 6.649 dB; in-sample side-worst normalized/absolute 1.387 / 1.372 dB; phase 11.159 deg.
- Source off-plane ambiguity: case `h1659-profile-compact-split-discrete-svd`; z=0 to UM-height RMS 70-90 deg 1.211 dB; source-family spread at UM height 12.062 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `h1659-profile-compact-split-discrete-svd`; z=0 to UM-height normalized-polar change is 1.211 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 12.062 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 6.649 dB exceeds 4.000 dB.

### asym-split-discrete-f45-r55-r0_35_70_95-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 17.
- Source: `split-discrete`; profile `asymmetric_depth_rings`; sources/bases 14/14.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 6.793 dB; in-sample side-worst normalized/absolute 2.656 / 2.627 dB; phase 15.474 deg.
- Source off-plane ambiguity: case `asym-f45-r55-r35-svd`; z=0 to UM-height RMS 70-90 deg 0.473 dB; source-family spread at UM height 10.577 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `asym-f45-r55-r35-svd`; z=0 to UM-height normalized-polar change is 0.473 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.577 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 6.793 dB exceeds 4.000 dB.

### asym-split-discrete-f45-r55-front35-rear25-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 18.
- Source: `split-discrete`; profile `asymmetric_depth_rings`; sources/bases 16/16.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 6.815 dB; in-sample side-worst normalized/absolute 2.639 / 2.612 dB; phase 15.338 deg.
- Source off-plane ambiguity: case `asym-f45-r55-front35-rear25-svd`; z=0 to UM-height RMS 70-90 deg 0.890 dB; source-family spread at UM height 10.500 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `asym-f45-r55-front35-rear25-svd`; z=0 to UM-height normalized-polar change is 0.890 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.500 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 6.815 dB exceeds 4.000 dB.

### d55-split-discrete-r0_25_50_75_95-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 19.
- Source: `split-discrete`; profile `symmetric-depth-rings`; sources/bases 18/18.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 6.874 dB; in-sample side-worst normalized/absolute 2.639 / 2.612 dB; phase 15.338 deg.
- Source off-plane ambiguity: case `d55-r25-95-svd`; z=0 to UM-height RMS 70-90 deg 0.568 dB; source-family spread at UM height 10.387 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `d55-r25-95-svd`; z=0 to UM-height normalized-polar change is 0.568 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.387 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 6.874 dB exceeds 4.000 dB.

### h1659-profile-full-split-discrete-d45-r0_40_75-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 20.
- Source: `split-discrete`; profile `h1659-acoustic-full`; sources/bases 22/22.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 6.893 dB; in-sample side-worst normalized/absolute 1.397 / 1.383 dB; phase 11.103 deg.
- Source off-plane ambiguity: case `h1659-profile-full-split-discrete-svd`; z=0 to UM-height RMS 70-90 deg 1.320 dB; source-family spread at UM height 12.646 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `h1659-profile-full-split-discrete-svd`; z=0 to UM-height normalized-polar change is 1.320 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 12.646 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 6.893 dB exceeds 4.000 dB.

### active-surface-compact-m3-az16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 21.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.696 dB; in-sample side-worst normalized/absolute 6.111 / 5.473 dB; phase 67.330 deg.
- Source off-plane ambiguity: case `active-surface-dipole-m3`; z=0 to UM-height RMS 70-90 deg 0.232 dB; source-family spread at UM height 14.013 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `active-surface-dipole-m3`; z=0 to UM-height normalized-polar change is 0.232 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 14.013 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.111 dB exceeds 3.000 dB.

### measured-phase-split-discrete-d45-r0_40_75-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 22.
- Source: `split-discrete`; profile `symmetric-depth-rings`; sources/bases 10/10.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 7.707 dB; in-sample side-worst normalized/absolute 2.862 / 2.667 dB; phase 27.759 deg.
- Source off-plane ambiguity: case `measured-stable`; z=0 to UM-height RMS 70-90 deg 1.735 dB; source-family spread at UM height 14.486 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `measured-stable`; z=0 to UM-height normalized-polar change is 1.735 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 14.486 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 7.707 dB exceeds 4.000 dB.

### measured-phase-h1659-profile-ring-compact-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 23.
- Source: `split-profile-ring`; profile `h1659-acoustic-compact`; sources/bases 146/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.415 dB; in-sample side-worst normalized/absolute 4.595 / 4.333 dB; phase 24.572 deg.
- Source off-plane ambiguity: case `measured-profile-ring-compact`; z=0 to UM-height RMS 70-90 deg 0.197 dB; source-family spread at UM height 11.958 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `measured-profile-ring-compact`; z=0 to UM-height normalized-polar change is 0.197 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 11.958 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.595 dB exceeds 3.000 dB.

### measured-phase-h1659-profile-ring-full-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 24.
- Source: `split-profile-ring`; profile `h1659-acoustic-full`; sources/bases 242/12.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.557 dB; in-sample side-worst normalized/absolute 4.653 / 4.383 dB; phase 24.630 deg.
- Source off-plane ambiguity: case `measured-profile-ring-full`; z=0 to UM-height RMS 70-90 deg 0.195 dB; source-family spread at UM height 11.934 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `measured-profile-ring-full`; z=0 to UM-height normalized-polar change is 0.195 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 11.934 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.653 dB exceeds 3.000 dB.

### stable-split-discrete-d45-r0_40_75-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 25.
- Source: `split-discrete`; profile `symmetric-depth-rings`; sources/bases 10/10.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 8.622 dB; in-sample side-worst normalized/absolute 2.642 / 2.598 dB; phase 16.308 deg.
- Source off-plane ambiguity: case `stable`; z=0 to UM-height RMS 70-90 deg 0.616 dB; source-family spread at UM height 10.862 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `stable`; z=0 to UM-height normalized-polar change is 0.616 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.862 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 8.622 dB exceeds 4.000 dB.

### current-wide-split-discrete-d45-r0_35_70_95-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 26.
- Source: `split-discrete`; profile `symmetric-depth-rings`; sources/bases 14/14.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 8.810 dB; in-sample side-worst normalized/absolute 2.500 / 2.469 dB; phase 14.961 deg.
- Source off-plane ambiguity: case `wide-svd`; z=0 to UM-height RMS 70-90 deg 0.546 dB; source-family spread at UM height 10.949 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `wide-svd`; z=0 to UM-height normalized-polar change is 0.546 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.949 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 8.810 dB exceeds 4.000 dB.

### wide-split-discrete-d45-r0_35_70_95-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 27.
- Source: `split-discrete`; profile `symmetric-depth-rings`; sources/bases 14/14.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 8.810 dB; in-sample side-worst normalized/absolute 2.500 / 2.469 dB; phase 14.960 deg.
- Source off-plane ambiguity: case `wide-svd`; z=0 to UM-height RMS 70-90 deg 0.546 dB; source-family spread at UM height 10.949 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `wide-svd`; z=0 to UM-height normalized-polar change is 0.546 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 10.949 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 8.810 dB exceeds 4.000 dB.

### split-ring-d45-r0_25_50_75_95-az48-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 28.
- Source: `split-ring`; profile `symmetric-depth-rings`; sources/bases 386/10.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 9.274 dB; in-sample side-worst normalized/absolute 2.936 / 2.952 dB; phase 17.497 deg.
- Source off-plane ambiguity: case `split-ring-r25-95-az48-svd`; z=0 to UM-height RMS 70-90 deg 0.248 dB; source-family spread at UM height 11.544 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `split-ring-r25-95-az48-svd`; z=0 to UM-height normalized-polar change is 0.248 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 11.544 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 9.274 dB exceeds 4.000 dB.

### split-ring-d45-r0_35_70_95-az24-reg1e-06

- Source-CV set: `l22mg-source-model-cross-validation`; set-local rank 29.
- Source: `split-ring`; profile `symmetric-depth-rings`; sources/bases 146/8.
- Juan-only recommendation: `rejected_weak_cv`; worst held-out side RMS 9.279 dB; in-sample side-worst normalized/absolute 2.944 / 2.958 dB; phase 17.545 deg.
- Source off-plane ambiguity: case `split-ring-r35-az24`; z=0 to UM-height RMS 70-90 deg 0.248 dB; source-family spread at UM height 11.509 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `split-ring-r35-az24`; z=0 to UM-height normalized-polar change is 0.248 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 11.509 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan held-out side-worst normalized RMS 9.279 dB exceeds 4.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 1.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.448 dB; in-sample side-worst normalized/absolute 3.119 / 4.712 dB; phase 62.053 deg.
- Source off-plane ambiguity: case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05`; z=0 to UM-height RMS 70-90 deg 0.224 dB; source-family spread at UM height 13.239 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-svd1e-05`; z=0 to UM-height normalized-polar change is 0.224 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.239 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.119 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 2.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.448 dB; in-sample side-worst normalized/absolute 3.119 / 4.712 dB; phase 62.053 deg.
- Source off-plane ambiguity: case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.224 dB; source-family spread at UM height 13.239 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.224 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.239 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.119 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 3.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.545 dB; in-sample side-worst normalized/absolute 3.758 / 5.262 dB; phase 69.518 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.758 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 4.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.545 dB; in-sample side-worst normalized/absolute 3.758 / 5.262 dB; phase 69.518 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.758 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 5.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.486 dB; in-sample side-worst normalized/absolute 6.521 / 6.347 dB; phase 44.957 deg.
- Source off-plane ambiguity: case `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.126 dB; source-family spread at UM height 8.155 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.126 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 8.155 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.521 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-m3-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 6.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.117 dB; in-sample side-worst normalized/absolute 7.419 / 7.232 dB; phase 55.566 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.419 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 7.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.287 dB; in-sample side-worst normalized/absolute 4.299 / 4.100 dB; phase 66.340 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06`; z=0 to UM-height RMS 70-90 deg 0.181 dB; source-family spread at UM height 16.099 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg3e-06`; z=0 to UM-height normalized-polar change is 0.181 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 16.099 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.299 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 8.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.364 dB; in-sample side-worst normalized/absolute 4.307 / 4.103 dB; phase 65.744 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06`; z=0 to UM-height RMS 70-90 deg 0.179 dB; source-family spread at UM height 15.949 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-dipole-az24-q3-reg3e-06`; z=0 to UM-height normalized-polar change is 0.179 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 15.949 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.307 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 9.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.407 dB; in-sample side-worst normalized/absolute 4.216 / 5.612 dB; phase 68.839 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.089 dB; source-family spread at UM height 13.413 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az24-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.089 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.413 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.216 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 10.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.569 dB; in-sample side-worst normalized/absolute 4.286 / 5.754 dB; phase 69.514 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.103 dB; source-family spread at UM height 13.445 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.103 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.445 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.286 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 11.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.569 dB; in-sample side-worst normalized/absolute 4.286 / 5.754 dB; phase 69.514 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05`; z=0 to UM-height RMS 70-90 deg 0.103 dB; source-family spread at UM height 13.445 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-svd1e-05`; z=0 to UM-height normalized-polar change is 0.103 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.445 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.286 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 12.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.691 dB; in-sample side-worst normalized/absolute 4.854 / 4.637 dB; phase 68.097 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.854 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 13.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.850 dB; in-sample side-worst normalized/absolute 4.961 / 4.725 dB; phase 68.069 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.961 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 14.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.858 dB; in-sample side-worst normalized/absolute 5.086 / 4.261 dB; phase 69.406 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.086 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m2-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 15.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.859 dB; in-sample side-worst normalized/absolute 5.087 / 4.256 dB; phase 69.402 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.087 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 16.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.993 dB; in-sample side-worst normalized/absolute 5.210 / 4.329 dB; phase 69.507 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.210 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m2-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 17.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.995 dB; in-sample side-worst normalized/absolute 5.211 / 4.322 dB; phase 69.503 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.211 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 18.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.094 dB; in-sample side-worst normalized/absolute 4.300 / 6.278 dB; phase 78.906 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.300 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 19.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.135 dB; in-sample side-worst normalized/absolute 4.679 / 4.389 dB; phase 61.127 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.162 dB; source-family spread at UM height 13.865 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-dipole-az24-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.162 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 13.865 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.679 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 20.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.282 dB; in-sample side-worst normalized/absolute 4.422 / 6.374 dB; phase 78.889 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.422 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 21.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.282 dB; in-sample side-worst normalized/absolute 4.422 / 6.374 dB; phase 78.889 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.422 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 22.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.348 dB; in-sample side-worst normalized/absolute 4.636 / 4.336 dB; phase 61.881 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.164 dB; source-family spread at UM height 14.091 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.164 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 14.091 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.636 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-m2-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 23.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.588 dB; in-sample side-worst normalized/absolute 5.732 / 4.806 dB; phase 70.948 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.732 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-m2-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 24.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.686 dB; in-sample side-worst normalized/absolute 6.615 / 7.607 dB; phase 91.805 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.615 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 25.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.383 dB; in-sample side-worst normalized/absolute 5.545 / 6.990 dB; phase 73.252 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06`; z=0 to UM-height RMS 70-90 deg 0.132 dB; source-family spread at UM height 14.054 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az24-q3-reg3e-06`; z=0 to UM-height normalized-polar change is 0.132 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 14.054 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.545 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m3-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 26.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.452 dB; in-sample side-worst normalized/absolute 5.786 / 5.351 dB; phase 67.013 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.786 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 27.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.553 dB; in-sample side-worst normalized/absolute 5.689 / 7.182 dB; phase 74.091 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06`; z=0 to UM-height RMS 70-90 deg 0.118 dB; source-family spread at UM height 14.025 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az16-q3-reg3e-06`; z=0 to UM-height normalized-polar change is 0.118 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 14.025 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.689 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-m3-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 28.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.690 dB; in-sample side-worst normalized/absolute 5.899 / 5.449 dB; phase 66.941 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.899 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 29.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.787 dB; in-sample side-worst normalized/absolute 7.080 / 8.148 dB; phase 90.069 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.080 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m2-az24-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 30.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 7.897 dB; in-sample side-worst normalized/absolute 6.993 / 8.052 dB; phase 90.015 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.993 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 31.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.157 dB; in-sample side-worst normalized/absolute 7.136 / 8.218 dB; phase 90.113 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.136 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m3-az24-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 32.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.173 dB; in-sample side-worst normalized/absolute 5.782 / 7.442 dB; phase 79.890 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.782 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m2-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 33.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 2/2.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.318 dB; in-sample side-worst normalized/absolute 7.064 / 8.138 dB; phase 90.065 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.064 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-m3-az16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm`; set-local rank 34.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 8.380 dB; in-sample side-worst normalized/absolute 5.921 / 7.568 dB; phase 79.983 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.921 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 1.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.481 dB; in-sample side-worst normalized/absolute 3.159 / 4.726 dB; phase 61.985 deg.
- Source off-plane ambiguity: case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.224 dB; source-family spread at UM height 13.239 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.224 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.239 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.159 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 2.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.482 dB; in-sample side-worst normalized/absolute 3.159 / 4.726 dB; phase 61.986 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.159 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 3.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.483 dB; in-sample side-worst normalized/absolute 3.162 / 4.730 dB; phase 61.991 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.162 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-measured-az16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 4.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.494 dB; in-sample side-worst normalized/absolute 3.192 / 4.764 dB; phase 62.039 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.192 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 5.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.486 dB; in-sample side-worst normalized/absolute 6.543 / 6.368 dB; phase 44.864 deg.
- Source off-plane ambiguity: case `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.126 dB; source-family spread at UM height 8.155 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.126 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 8.155 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.543 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 6.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.486 dB; in-sample side-worst normalized/absolute 6.543 / 6.369 dB; phase 44.864 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.543 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 7.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.487 dB; in-sample side-worst normalized/absolute 6.545 / 6.371 dB; phase 44.871 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.545 dB exceeds 3.000 dB.

### physical-diaphragm-full-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 8.
- Source: `active-surface-modal`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.494 dB; in-sample side-worst normalized/absolute 6.564 / 6.392 dB; phase 44.941 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.564 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 9.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.612 dB; in-sample side-worst normalized/absolute 4.328 / 5.775 dB; phase 69.327 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.103 dB; source-family spread at UM height 13.445 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.103 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.445 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.328 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 10.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.612 dB; in-sample side-worst normalized/absolute 4.328 / 5.775 dB; phase 69.327 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.328 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 11.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.613 dB; in-sample side-worst normalized/absolute 4.329 / 5.777 dB; phase 69.333 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.329 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-measured-az16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 12.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.618 dB; in-sample side-worst normalized/absolute 4.337 / 5.793 dB; phase 69.392 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.337 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 13.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.404 dB; in-sample side-worst normalized/absolute 4.696 / 4.399 dB; phase 61.893 deg.
- Source off-plane ambiguity: case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.164 dB; source-family spread at UM height 14.091 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full-svd`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.164 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full-svd`) is 14.091 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.696 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 14.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.404 dB; in-sample side-worst normalized/absolute 4.696 / 4.399 dB; phase 61.893 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.696 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 15.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.407 dB; in-sample side-worst normalized/absolute 4.695 / 4.399 dB; phase 61.895 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.695 dB exceeds 3.000 dB.

### physical-diaphragm-compact-coupled-dipole-az16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-diaphragm-fsmooth`; set-local rank 16.
- Source: `active-surface-modal`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 6.439 dB; in-sample side-worst normalized/absolute 4.692 / 4.398 dB; phase 61.910 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.692 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 1.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_source_level_fit`; worst held-out side RMS 3.592 dB; in-sample side-worst normalized/absolute 2.635 / 4.543 dB; phase 72.737 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg nan dB; source-family spread at UM height nan dB; spread basis missing; eligible surface max n/a. off-plane source ambiguity audit did not include mapped case `physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06`
- Stored BEM evidence: `physical-rear-basket-m2rb1-juan-smoke` (available_overlap_300_301_hz); all-angle RMS 6.447 dB; through-60/through-80 RMS 0.629 / 6.959 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: `physical-rear-basket-m2rb1-andres-smoke` (available_overlap_300_301_hz); all-angle RMS 10.021 dB; through-80 RMS 1.719 dB; target gate `pass`; published explorer match `True`; source off-plane gate `not_proven`.
- Decision: `rejected_by_source_cv_with_failed_bem_overlap`. Juan source evidence is rejected: Juan in-sample side-worst absolute SPL RMS 4.543 dB exceeds 3.000 dB; normalized shape alone would not catch uniform source level errors. Stored BEM overlap also exceeds 1.5 dB (6.447 dB).

### physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 2.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_source_level_fit`; worst held-out side RMS 3.592 dB; in-sample side-worst normalized/absolute 2.635 / 4.543 dB; phase 72.737 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst absolute SPL RMS 4.543 dB exceeds 3.000 dB; normalized shape alone would not catch uniform source level errors.

### physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 3.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_source_level_fit`; worst held-out side RMS 3.593 dB; in-sample side-worst normalized/absolute 2.635 / 4.544 dB; phase 72.738 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst absolute SPL RMS 4.544 dB exceeds 3.000 dB; normalized shape alone would not catch uniform source level errors.

### physical-rear-basket-compact-measured-m2-rb1-az16-gap16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 4.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_source_level_fit`; worst held-out side RMS 3.596 dB; in-sample side-worst normalized/absolute 2.640 / 4.557 dB; phase 72.743 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst absolute SPL RMS 4.557 dB exceeds 3.000 dB; normalized shape alone would not catch uniform source level errors.

### physical-rear-basket-full-measured-m2-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 5.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_source_level_fit`; worst held-out side RMS 3.623 dB; in-sample side-worst normalized/absolute 2.683 / 4.555 dB; phase 71.471 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst absolute SPL RMS 4.555 dB exceeds 3.000 dB; normalized shape alone would not catch uniform source level errors.

### physical-rear-basket-full-dipole-m2-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 6.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.767 dB; in-sample side-worst normalized/absolute 4.351 / 4.045 dB; phase 44.366 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.351 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m2-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 7.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 3/3.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 3.789 dB; in-sample side-worst normalized/absolute 4.282 / 3.952 dB; phase 44.781 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.282 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m2-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 8.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.076 dB; in-sample side-worst normalized/absolute 3.824 / 4.356 dB; phase 52.698 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.824 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m2-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 9.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.087 dB; in-sample side-worst normalized/absolute 3.835 / 4.359 dB; phase 52.295 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.835 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m2-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 10.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.131 dB; in-sample side-worst normalized/absolute 3.767 / 4.316 dB; phase 52.820 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.767 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m3-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 11.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.144 dB; in-sample side-worst normalized/absolute 4.270 / 4.657 dB; phase 48.468 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.270 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m2-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 12.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.157 dB; in-sample side-worst normalized/absolute 3.796 / 4.323 dB; phase 52.323 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.796 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m3-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 13.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.158 dB; in-sample side-worst normalized/absolute 4.136 / 4.506 dB; phase 48.127 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.136 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m2-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 14.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.180 dB; in-sample side-worst normalized/absolute 3.813 / 4.323 dB; phase 51.958 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.813 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 15.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.189 dB; in-sample side-worst normalized/absolute 4.111 / 4.597 dB; phase 51.542 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06`; z=0 to UM-height RMS 70-90 deg 0.135 dB; source-family spread at UM height 13.553 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg3e-06`; z=0 to UM-height normalized-polar change is 0.135 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.553 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.111 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m3-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 16.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.195 dB; in-sample side-worst normalized/absolute 4.607 / 5.011 dB; phase 48.779 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.607 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m3-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 17.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.197 dB; in-sample side-worst normalized/absolute 4.377 / 4.808 dB; phase 49.410 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.377 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m3-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 18.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.199 dB; in-sample side-worst normalized/absolute 4.431 / 4.852 dB; phase 49.055 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.431 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 19.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.201 dB; in-sample side-worst normalized/absolute 4.013 / 4.354 dB; phase 47.665 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.135 dB; source-family spread at UM height 12.932 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb2-az16-gap30-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.135 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 12.932 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.013 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 20.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.215 dB; in-sample side-worst normalized/absolute 4.693 / 5.069 dB; phase 48.476 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.142 dB; source-family spread at UM height 13.451 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.142 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.451 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: `coupled-rear-basket-source-smoke` (available_overlap_300_301_hz); all-angle RMS 7.608 dB; through-60/through-80 RMS 1.344 / 8.189 dB.
- Stored BEM target: target gate `pass`; hypothesis `juan_baffleless_to_juan_top_baffle_l22mg_raw`; target kind `juan_lx521_top_raw`; HDF5 `output/data/polar_data_juan_lx521_top_raw.h5`; driver `L22MG (LX521 top raw)`; published explorer match `False`.
- Andres published-parity side evidence: `physical-rear-basket-andres-smoke` (available_overlap_300_301_hz); all-angle RMS 6.869 dB; through-80 RMS 1.218 dB; target gate `pass`; published explorer match `True`; source off-plane gate `fail`.
- Decision: `rejected_by_source_cv_with_failed_bem_overlap`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.693 dB exceeds 3.000 dB. Stored BEM overlap also exceeds 1.5 dB (7.608 dB).

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 21.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.215 dB; in-sample side-worst normalized/absolute 4.693 / 5.069 dB; phase 48.476 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05`; z=0 to UM-height RMS 70-90 deg 0.142 dB; source-family spread at UM height 13.451 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-svd1e-05`; z=0 to UM-height normalized-polar change is 0.142 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.451 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.693 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-08

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 22.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.216 dB; in-sample side-worst normalized/absolute 4.693 / 5.069 dB; phase 48.476 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.693 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb3-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 23.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 7/7.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.216 dB; in-sample side-worst normalized/absolute 4.055 / 4.376 dB; phase 46.303 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.055 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-07

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 24.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.217 dB; in-sample side-worst normalized/absolute 4.695 / 5.072 dB; phase 48.478 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.695 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 25.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.218 dB; in-sample side-worst normalized/absolute 4.719 / 5.100 dB; phase 48.667 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.142 dB; source-family spread at UM height 13.471 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb2-az24-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.142 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.471 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.719 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 26.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.225 dB; in-sample side-worst normalized/absolute 4.460 / 4.818 dB; phase 48.099 deg.
- Source off-plane ambiguity: case `physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.143 dB; source-family spread at UM height 13.408 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-full-measured-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.143 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.408 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.460 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb2-az16-gap16-q3-reg1e-06-fsmooth1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 27.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.227 dB; in-sample side-worst normalized/absolute 4.715 / 5.094 dB; phase 48.499 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 4.715 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 28.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.234 dB; in-sample side-worst normalized/absolute 3.913 / 4.232 dB; phase 47.194 deg.
- Source off-plane ambiguity: case `physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.136 dB; source-family spread at UM height 12.867 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-full-measured-m4-rb2-az16-gap30-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.136 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 12.867 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.913 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb4-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 29.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 8/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.252 dB; in-sample side-worst normalized/absolute 3.971 / 4.275 dB; phase 45.464 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.971 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m4-rb3-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 30.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 7/7.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.254 dB; in-sample side-worst normalized/absolute 3.953 / 4.251 dB; phase 45.659 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.953 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb3-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 31.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 7/7.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.299 dB; in-sample side-worst normalized/absolute 3.807 / 4.107 dB; phase 45.292 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.807 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m4-rb4-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 32.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 8/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.303 dB; in-sample side-worst normalized/absolute 3.890 / 4.172 dB; phase 44.643 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.890 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb4-az16-gap30-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 33.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 8/8.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.361 dB; in-sample side-worst normalized/absolute 3.784 / 4.069 dB; phase 44.257 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.784 dB exceeds 3.000 dB.

### physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 34.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.500 dB; in-sample side-worst normalized/absolute 7.684 / 7.672 dB; phase 46.522 deg.
- Source off-plane ambiguity: case `physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.135 dB; source-family spread at UM height 12.725 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-full-dipole-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.135 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 12.725 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.684 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 35.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.548 dB; in-sample side-worst normalized/absolute 3.829 / 4.406 dB; phase 57.445 deg.
- Source off-plane ambiguity: case `physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.202 dB; source-family spread at UM height 13.692 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-full-measured-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.202 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.692 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.829 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m3-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 36.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.585 dB; in-sample side-worst normalized/absolute 3.432 / 4.510 dB; phase 64.301 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.432 dB exceeds 3.000 dB.

### physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 37.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.624 dB; in-sample side-worst normalized/absolute 3.685 / 4.400 dB; phase 60.218 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.218 dB; source-family spread at UM height 13.383 dB; spread basis source_cv_recommended_juan_only vs `profile-ring-full`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-measured-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.218 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `profile-ring-full`) is 13.383 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.685 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 38.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.741 dB; in-sample side-worst normalized/absolute 6.632 / 6.622 dB; phase 47.115 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.094 dB; source-family spread at UM height 11.794 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-dipole-m4-rb1-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.094 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 11.794 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 6.632 dB exceeds 3.000 dB.

### physical-rear-basket-full-measured-m3-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 39.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.795 dB; in-sample side-worst normalized/absolute 3.540 / 4.502 dB; phase 62.871 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 3.540 dB exceeds 3.000 dB.

### physical-rear-basket-full-dipole-m3-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 40.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.858 dB; in-sample side-worst normalized/absolute 5.762 / 5.698 dB; phase 47.077 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.762 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m3-rb1-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 41.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 4.871 dB; in-sample side-worst normalized/absolute 5.417 / 5.308 dB; phase 47.400 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 5.417 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m3-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 42.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.456 dB; in-sample side-worst normalized/absolute 9.143 / 9.086 dB; phase 52.112 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.143 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m2-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 43.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.460 dB; in-sample side-worst normalized/absolute 7.922 / 7.858 dB; phase 49.757 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 7.922 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 44.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.466 dB; in-sample side-worst normalized/absolute 9.370 / 9.315 dB; phase 53.761 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.126 dB; source-family spread at UM height 12.167 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-dipole-m4-rb2-az24-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.126 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 12.167 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.370 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m3-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 45.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.507 dB; in-sample side-worst normalized/absolute 9.150 / 9.092 dB; phase 52.445 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.150 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 46.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.510 dB; in-sample side-worst normalized/absolute 9.307 / 9.251 dB; phase 53.952 deg.
- Source off-plane ambiguity: case `physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.123 dB; source-family spread at UM height 12.006 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-compact-dipole-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.123 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 12.006 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.307 dB exceeds 3.000 dB.

### physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 47.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 6/6.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.532 dB; in-sample side-worst normalized/absolute 9.439 / 9.383 dB; phase 54.974 deg.
- Source off-plane ambiguity: case `physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height RMS 70-90 deg 0.124 dB; source-family spread at UM height 12.032 dB; spread basis source_cv_recommended_juan_only vs `modal-full2`; eligible surface max 26.655 dB at 734 Hz / 90 deg between `profile-ring-full` and `modal-full2`. Mapped off-plane source case `physical-rear-basket-full-dipole-m4-rb2-az16-gap16-q3-reg1e-06`; z=0 to UM-height normalized-polar change is 0.124 dB RMS over 70-90 deg; source-family spread against the Juan-CV recommended finite-source ensemble (worst reference `modal-full2`) is 12.032 dB RMS, above the 1.5 dB shape target; Juan horizontal polars do not uniquely constrain this off-plane incident field
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.439 dB exceeds 3.000 dB.

### physical-rear-basket-compact-dipole-m2-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 48.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-compact`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.536 dB; in-sample side-worst normalized/absolute 8.010 / 7.946 dB; phase 50.314 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 8.010 dB exceeds 3.000 dB.

### physical-rear-basket-full-dipole-m3-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 49.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 5/5.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.540 dB; in-sample side-worst normalized/absolute 9.221 / 9.164 dB; phase 53.402 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 9.221 dB exceeds 3.000 dB.

### physical-rear-basket-full-dipole-m2-rb2-az16-gap16-q3-reg1e-06

- Source-CV set: `l22mg-source-cv-physical-rear-basket`; set-local rank 50.
- Source: `coupled-rear-basket-active-surface`; profile `h1659-acoustic-full`; sources/bases 4/4.
- Juan-only recommendation: `rejected_bad_insample_fit`; worst held-out side RMS 5.598 dB; in-sample side-worst normalized/absolute 8.106 / 8.042 dB; phase 50.768 deg.
- Source off-plane ambiguity: case `not_applicable_rejected_source_cv`; z=0 to UM-height RMS 70-90 deg n/a; source-family spread at UM height n/a; spread basis n/a; eligible surface max n/a. Juan source-CV already rejects this source, so off-plane source ambiguity metrics are not required before rejecting it as an acceptance candidate.
- Stored BEM evidence: missing; all-angle RMS nan dB; through-60/through-80 RMS nan / nan dB.
- Stored BEM target: target gate `missing`; hypothesis `missing`; target kind `missing`; HDF5 `missing`; driver `missing`; published explorer match `missing`.
- Andres published-parity side evidence: missing; all-angle RMS nan dB; through-80 RMS nan dB; target gate `missing`; published explorer match `missing`; source off-plane gate `missing`.
- Decision: `rejected_by_source_cv`. Juan source evidence is rejected: Juan in-sample side-worst normalized RMS 8.106 dB exceeds 3.000 dB.

CSV: `docs/l22mg-source-candidate-decision/source_candidate_decision.csv`.
Inputs: `docs/l22mg-source-model-cross-validation/source_cv_summary.csv`, `docs/l22mg-source-cv-active-surface-annular-sweep/source_cv_summary.csv`, `docs/l22mg-source-cv-active-surface-annular-extended/source_cv_summary.csv`, `docs/l22mg-source-cv-active-surface-uniform-annular-local/source_cv_summary.csv`, `docs/l22mg-source-cv-active-surface-annular-fine/source_cv_summary.csv`, `docs/l22mg-source-cv-physical-diaphragm/source_cv_summary.csv`, `docs/l22mg-source-cv-physical-diaphragm-fsmooth/source_cv_summary.csv`, `docs/l22mg-source-cv-physical-rear-basket/source_cv_summary.csv`, `docs/l22mg-validation-gate-summary/validation_gate_summary.csv`, `docs/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-juan-top-h1659-modal-compact2-current-width-sweep-h42-h28-q7near-targetavg7-gates/validation_gate_summary.csv`, `docs/l22mg-coupled-rear-basket-source-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-coupled-rear-basket-m2rb1-source-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-validation-gate-andres-published-parity/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-physical-rear-basket-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-physical-rear-basket-m2rb1-smoke-gates/validation_gate_summary.csv`, `docs/l22mg-bem-andres-published-parity-coupled-rear-basket-feature-proxy-smoke-gates/validation_gate_summary.csv`.
