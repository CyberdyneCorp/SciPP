# Changelog

## 1.1.0 — 2026-06-28 — special functions, `odr`, GPU acceleration, `just`, and the SciPP rename

The first published release. Builds on the v1.0.0 12-phase foundation with a large
additive feature pass, real GPU offload, a task-runner workflow, and the project
rename. Built on **NumPP 1.5.0** and validated against **SciPy 1.15** as the
numerical oracle: **140 cases / 7503 checks, 0 divergences**.

### Project rename — `ScyPP` → `SciPP`
- The project, namespace (`scypp::` → `scipp::`), include tree (`include/scipp/`),
  CMake project/targets, backend flags (`SCIPP_WITH_*`), and docs were renamed to
  **SciPP**, matching the GitHub repository `CyberdyneCorp/SciPP`. Pure mechanical
  change, no behaviour difference.

### GPU acceleration (now real, via NumPP 1.5.0)
- The deferred device-kernel paths are wired to NumPP's new acceleration
  primitives — each auto-selects a device backend above the size threshold and
  always falls back to the portable CPU path, reporting the choice via
  `last_backend()`:
  - **sparse** — `spmv`/`spmm` (CSR) → `numpp::csr_spmv`.
  - **spatial** — `cdist`/`pdist` euclidean/sqeuclidean → `numpp::cdist_euclidean`.
  - **ndimage** — `correlate1d` and the gaussian/uniform/derivative separable
    passes → `numpp::correlate1d`.
- Pins the NumPP dependency floor to **1.5.0**.

### `scipp::special` — the deferred function tail
- **Airy**: `airy`, `airye`.
- **Elliptic**: `ellipk`, `ellipkm1`, `ellipe`, `ellipkinc`, `ellipeinc`, `ellipj`.
- **Error-function relatives**: `erfcx`, `dawsn`, `wofz`/`voigt_profile`, `fresnel`.
- **Spherical Bessel**: `spherical_jn`/`yn`/`in`/`kn`; **integrals** `sici`, `shichi`.
- **Misc**: `lambertw`, `zeta`/`zetac`, `struve`/`modstruve`, `spence`.
- **Hypergeometric**: `hyp0f1`, `hyp1f1`, `hyp2f1`, `hyperu`.

### New `scipp::odr` capability
- Orthogonal distance regression (`Model`/`Data`/`ODR`) — total-least-squares fit
  with estimated parameters, standard errors, and residual variance.

### Capability drawdowns
- **stats** — discrete distributions (`binom`/`poisson`/`geom`/`bernoulli`/`nbinom`/
  `hypergeom`), nonparametric rank tests (`mannwhitneyu`/`wilcoxon`/`kruskal`/
  `kendalltau`), and normality tests (`shapiro`/`anderson`).
- **optimize** — `linprog` (two-phase simplex), `nnls` (Lawson–Hanson), and the
  `Powell`/`CG`/`L-BFGS-B` `minimize` methods (L-BFGS-B with box bounds).
- **integrate** — stiff solvers `solve_ivp` `Radau` and `BDF`, `solve_bvp`
  (collocation), and nested/extended quadrature (`romberg`, `quad_vec`, `dblquad`,
  `tplquad`, `nquad`).
- **interpolate** — `griddata` (nearest/linear) and B-splines (`BSpline`,
  `make_interp_spline`, `splev`).
- **fft** — DCT/DST `ortho`/`forward` norms (types I–IV) and N-D `dctn`/`idctn`/
  `dstn`/`idstn`.
- **signal** — discrete-time LTI (`cont2discrete`, `dstep`/`dimpulse`/`dlsim`,
  `dbode`/`dfreqresp`).
- **sparse** — `csgraph` traversals and flow (`breadth_first_order`,
  `depth_first_order`, `johnson`, `maximum_flow`, `maximum_bipartite_matching`).

### Tooling & process
- **`justfile`** task runner (`bootstrap`/`build`/`test`/`ctest`/`debug`/`gcc`/
  `asan`/`oracle`/`spec`/`ci`/`clean`), mirroring NumPP/SymPP.
- **Phase 0 foundation closed out**: added `scipp::linalg_error`, a foundation
  regression test, and a CI workflow (`openspec validate` + CPU build/test).
- **Deferred-work tracking**: every open backlog is mirrored as a labeled GitHub
  issue (`deferred`/`openspec`/`module:*`), cross-linked from its OpenSpec change;
  the convention (and the bug → regression-test rule) is documented in
  `openspec/project.md`.

## 1.0.0 — 2026-06-27 — initial 12-phase SciPy parity

First clean-room C++20 port of SciPy's commonly-used surface, built on NumPP and
validated against SciPy as the oracle. Ships all 12 phases:

- **special** + **constants**, **linalg**, **fft**/**fftpack**, **optimize**,
  **integrate** + **differentiate**, **interpolate**, **stats**, **signal**,
  **sparse**, **spatial**, **ndimage**, **cluster** + **io**.
- Foundation: CMake/C++20 project, pinned NumPP `find_package` dependency, the
  `scipp::error` model, the tiered-acceleration dispatch shim over NumPP's
  `CapabilityRegistry`/`last_backend()`, and the frozen-golden SciPy oracle
  harness for Python-free CI.
