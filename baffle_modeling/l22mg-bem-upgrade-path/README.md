# L22MG BEM solver upgrade path

This note records the non-ad-hoc path for moving beyond the current dense constant-panel BEM validation backend. It is generated from the current driver-STL audit and acceptance scorecards so source radius, target geometry, and blocker status do not drift from the validation artifacts.

## Current Boundary

- The current automated high-fidelity backend is dense exterior Neumann BEM on a finite-thickness LX521.4 baffle mesh, driven by finite or prescribed-Neumann sources fitted from Juan's nude L22MG measurements.
- Juan's naked L22MG front/rear measurements are now treated at the corrected 0.50 m measurement radius. Runs or reports carrying 0.75 m source-radius metadata are stale diagnostics, not current acceptance evidence.
- Andres published-parity validation is explicit: 1 m horizontal radius at the upper-mid height, +165 mm relative to the L22 center, using the published-explorer-matching HDF5 and the Andres target angle grid.
- The nude measurements are treated as the measured field of the real driver on the measurement radius. They are not treated as a point-source directivity table.
- Validation alignment policy: Gain 5.955 dB and delay 0.348 ms; scalar gain count 1, delay count 1, angle-dependent gain count 0, band-specific gain count 0, rear/source level corrections 0.000 / 0.000 dB. one scalar gain and one delay only; no source or rear level correction
- Current Andres selected row: Selected artifact `andres-published-parity-h1659-modal-full2-current` from `output/l22mg-bem-andres-published-parity-h1659-modal-full2-300-1200-current`; normalized polar RMS 4.428 dB over 300-1200 Hz.
- Current Juan same-room row, used as separate same-driver context: Selected artifact `compact2-h32-h21-h28-h18-targetavg7-meshtarget-conv-smoke` from `output/l22mg-bem-juan-top-h1659-modal-compact2-solid-passive-h32-h21-h28-h18-q7near-targetavg7-meshtarget-conv-smoke`; normalized polar RMS 3.920 dB over 300-1200 Hz.
- Andres selected artifact driver-STL geometry provenance (`driver_stl_geometry_provenance`): `pass`; STL `linkwitz/H1659-08_U22REX_P-SL_driver.stl` status `present`; OD 221.000 mm, depth 90.600 mm, front proud height 0.600 mm, rear depth 90.000 mm, flange overlap past L22 cutout radius 15.500 mm. STL dimensions are recorded for provenance of H1659-derived face/frame/source geometry; recording these dimensions is not a raw-STL solve, feature-preserving remesh, or acceptance proof
- Juan selected artifact driver-STL geometry provenance (`driver_stl_geometry_provenance`): `pass`; STL `linkwitz/H1659-08_U22REX_P-SL_driver.stl` status `present`; OD 221.000 mm, depth 90.600 mm, front proud height 0.600 mm, rear depth 90.000 mm, flange overlap past L22 cutout radius 15.500 mm. STL dimensions are recorded for provenance of H1659-derived face/frame/source geometry; recording these dimensions is not a raw-STL solve, feature-preserving remesh, or acceptance proof
- No rear-source, source-level, angle-dependent, or band-specific scalar correction is part of the accepted workflow.

## Current Blockers

- Andres normalized-polar shape: `fail`; All-angle RMS 4.428 dB vs 1.5 dB target; through-60 0.686 dB; through-80 1.377 dB; worst +24.924 dB at 335 Hz / 90 deg.
- Andres mesh convergence: `pass`; Max normalized-polar movement 0.111 dB vs 0.5 dB gate.
- Andres source off-plane generalization: `fail`; Mapped case `modal-full2`; z=0-to-UM-height movement for this source is 0.153 dB RMS / 0.264 dB max over 70-90 deg, but the Juan-CV-eligible source-family spread at UM height is 13.406 dB RMS / 26.652 dB max over 70-90 deg against 6 reference sources. Worst reference `profile-ring-full-svd`; surface extremes `profile-ring-full` to `modal-full2` reach 26.655 dB at 734.4 Hz / 90 deg. CSVs: `docs/l22mg-offplane-source-ambiguity/source_offplane_summary.csv`, `docs/l22mg-offplane-source-ambiguity/source_eligible_pairwise_offplane_summary.csv`, `docs/l22mg-offplane-source-ambiguity/source_eligible_pairwise_spread_surface.csv`. `modal-full2` off-plane source ambiguity: source-family spread versus Juan-CV recommended source ensemble (worst reference `profile-ring-full-svd`) 13.406 dB exceeds 1.5 dB
- Andres target measurement quality: `not_proven`; late-window target quality: late in-gate peaks can contaminate the gated response at 40,50,60,70,80,90; right 0.80 ms minus right 3.00 ms: raw-IR short-vs-3 ms gate movement is 0.822 dB RMS through 60 deg and 6.424 dB RMS over 70-90 deg; max +34.774 dB at 1074.1 Hz / 80 deg; target polar quality: measured target has -42.736 dB normalized null at 330.322 Hz / 90 deg; cluster width 293.7 Hz; adjacent-angle contrast 26.234 dB; this is target-quality context, not the artifact's worst residual
- Width sweeps remain qualitative until the 305 mm baseline passes shape and mesh-convergence gates.

## Serious Upgrade Path

1. Remesh the driver STL into a feature-preserving, high-quality exterior surface mesh.
2. Filter contact/internal surfaces so only acoustically exposed driver, flange, basket, motor, and baffle surfaces enter the exterior solve.
3. Keep local refinement at the baffle cutout lip, driver flange, sharp baffle edges, basket features, and high-curvature regions.
4. Use larger higher-order elements on smooth broad surfaces where the geometry and field are smooth.
5. Replace the dense O(N^3) backend with FMM-BEM, H-matrix BEM, or another accelerated exterior Helmholtz solver.
6. Add higher-order curved BEM elements before spending effort on uniformly tiny flat triangles.
7. Consider FEM-BEM coupling only when modeling flexible cone, suspension, porous, cavity, or structural transmission effects becomes part of the target.

## Mesh Scale

At 2000 Hz with c = 343 m/s, wavelength is about 171.5 mm. The driver-STL audit reports lambda/6 = 28.6 mm and lambda/10 = 17.1 mm. A 15-30 mm acoustic mesh is not inherently coarse for the passband. The problem is a uniform flat-triangle mesh on detailed geometry: it wastes panels on smooth areas while still representing lips, edges, and curvature poorly.

The right mesh question is therefore adaptive and higher-order sizing, not sub-mm geometry everywhere.

The current dependency-free mesh generator exposes this explicitly:

- `delaunay-local` keeps broad baffle regions at coarse spacing while adding a finer point field near the outside edge, L22 cutout, and passive holes.
- `--bem-boundary-mesh-h-mm`, `--bem-local-mesh-h-mm`, and `--bem-local-refinement-width-mm` make the local refinement auditable.
- `--bem-auto-mesh` derives broad spacing from lambda/6 and local spacing from lambda/10 at the requested target frequency, then selects `delaunay-local`.
- For prescribed-Neumann H1659 active-surface source models, including the coupled rear-basket surface source, `--bem-auto-mesh` raises the source-surface azimuth count from the H1659 outer acoustic diameter and lambda/10 chord length.
- `--bem-convergence-h-mm` records broad, boundary, local, and local-region-width values in the convergence CSV/report.
- Dense-BEM reports include mesh edge-length and triangle-quality metadata.

This still does not replace FMM/H-matrix or higher-order BEM; it is a better dense-backend mesh policy for bounded convergence experiments.

## STL Feature Graph

`scripts/inspect_l22mg_driver_stl.py` extracts a welded topology and feature-edge graph from the generated Linkwitz-style H1659/U22REX driver STL.

- STL: `linkwitz/H1659-08_U22REX_P-SL_driver.stl`.
- Raw facets: 9888; welded vertices: 5138; welded triangles: 9888.
- Outer diameter: 221.0 mm; depth span: 90.6 mm; flange extends 15.5 mm beyond the L22 cutout radius.
- Raw-STL triangle quality is not good enough to treat as a solver-quality mesh: minimum angle 1.18 deg and max edge ratio 48.7.
- Extracted feature graph: 4026 feature edges, 4008 feature vertices, and 9520.5 mm total feature-edge length.
- Adding the raw STL directly to an h18 baffle would create about 11688 panels before contact/internal-surface filtering.
- A uniform lambda/10 driver surface would be roughly 941 triangles before baffle panels, so the target is feature-preserving coarsening and exposed-surface filtering, not retaining every raw STL facet.
- Current bounded feature proxy: 1287 vertices, 1620 panels, minimum angle 2.82 deg, max edge ratio 9.3, and 0.164x the raw STL panel count.

This feature graph is a remeshing constraint and audit artifact. It is not yet an acceptance-grade BEM mesh, does not include contact/internal-surface filtering, and is not validation evidence.

## Diagnostic STL Paths

- `--include-driver-stl-envelope` uses only a coarse revolved radius envelope from the STL for bounded obstruction-sensitivity smoke runs. It is not a raw-STL solve, not a feature-preserving remesh, and not an acceptance check.
- `--driver-face-from-stl` derives a thin surface-mounted flange annulus from the STL outer radius and front proud height. It is useful for checking gross flange scattering, but it is still not a feature-preserving driver remesh.
- `--active-driver-feature-proxy` combines a rigid STL-dimensioned frame/flange/motor scatterer with a Juan-fitted prescribed-Neumann H1659 acoustic source surface. It is not fitted to Andres, not a scalar correction, not a raw-STL dense solve, and not elastic FEM-BEM.

## Acceptance Rule

Any future report should keep these separate:

- source-fit diagnostics against Juan nude measurements;
- source off-plane generalization from Juan's horizontal measurement plane to Andres' UM-height mic geometry;
- BEM mesh or solver convergence against a finer/accelerated reference;
- Andres validation at the UM-height mic geometry;
- full 300-1200 Hz acceptance metrics.

Short smoke bands, scalar source corrections, diagnostic STL-envelope runs, and raw-STL panel drops cannot be presented as full-band acceptance proof.
