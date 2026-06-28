# Tasks — Bootstrap ScyPP foundation

## Phase 0 — Foundation (this change)

### Project skeleton & build
- [x] CMake ≥ 3.25 / C++20 project; `include/scypp/` + `src/<subpackage>/` layout with reserved subpackage dirs
- [x] Generated `config.hpp` with `SCYPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}` flags (default OFF)
- [x] NumPP wired as a **pinned Conan/vcpkg release dependency** resolved via `find_package` (not vendored), umbrella `scypp/scypp.hpp`, `version.hpp.in`
- [x] Configure-time check: requested GPU flag requires a NumPP package variant built with the matching `NUMPP_WITH_<GPU>`; fail fast otherwise
- [x] GPU flag propagation: `SCYPP_WITH_<GPU>` enables the matching `NUMPP_WITH_<GPU>`
- [x] CPU-only build verified green (all backend flags OFF); iOS/Android cross-compile contract documented

### NumPP integration
- [x] Confirm public-API contract: ScyPP routines take/return `numpp::ndarray`, no ScyPP array type
- [x] `scypp::error` aligned with `numpp::error`; `value_error`/`linalg_error`/`not_implemented_error`; status-struct convention for SciPy result objects

### Backend acceleration
- [x] Dispatch shim layer over NumPP's `CapabilityRegistry` / `last_backend()` / `NUMPP_GPU_TARGET` / device pool
- [x] Define the SciPy kernel vtable slots (fft, gemm-backed linalg step, CSR SpMV, separable convolution, pairwise distance) — interface only, no subpackage impl
- [x] Size-threshold + dtype-gating policy and CPU-fallback guarantee documented; cross-backend equivalence + `last_backend()` test pattern established

### SciPy oracle harness
- [x] Catch2 test scaffold + SciPy oracle runner (real Python SciPy → `allclose`)
- [x] Frozen/`tests/golden/` mode for Python-free CI; regeneration + divergence-review flow
- [x] Stochastic-routine convention: seed via NumPP `random`; distributional-equivalence fallback documented
- [x] `openspec validate --all --strict` green; CI gate added

## Roadmap — each item graduates into its own OpenSpec change

> Not implemented in this change. Ported against the SciPy oracle and its OpenSpec
> baseline; GPU-accelerable kernels reuse the backend-acceleration substrate.

- [x] **Phase 1** — `special` (gamma/erf/Bessel/orthogonal polys/logsumexp/softmax) + `constants` (CODATA, unit conversion) — delivered via `add-special-constants`
- [~] **Phase 2** — `linalg` core delivered via `add-linalg` (inv/det/solve/lstsq/pinv/pinvh/norm, lu/qr/svd/cholesky, eig/eigh, expm/polar, special matrices; GEMM via NumPP). Deferred to a follow-up: ldl, sqrtm/logm/funm, schur/qz/cossin, banded, matrix-equation solvers
- [x] **Phase 3** — `fft` + `fftpack` delivered via `add-fft` (fft/rfft/hfft, fftn/rfftn delegated to NumPP; dct/dst I–IV + next_fast_len + fftpack alias; ortho/forward DCT norm, N-D DCT, fht/czt deferred)
- [x] **Phase 4** — `optimize` core delivered via `add-optimize` (brentq/bisect/newton, minimize_scalar brent/bounded, minimize Nelder-Mead/BFGS, least_squares/curve_fit/fsolve). Deferred: linprog/milp, constrained (SLSQP/trust-constr), global optimizers, nnls
- [x] **Phase 5** — `integrate` core + `differentiate` delivered via `add-integrate` (trapezoid/simpson/cumulative_trapezoid, quad adaptive Gauss-Kronrod, fixed_quad, solve_ivp RK45/RK23; derivative/jacobian/hessian). Deferred work tracked in the open `add-integrate-stiff-bvp` backlog change (stiff solvers Radau/BDF/LSODA + odeint, solve_bvp, dblquad/tplquad/nquad, romberg)
- [x] **Phase 6** — `interpolate` delivered via `add-interpolate` (Interp1d, CubicSpline/Pchip/Akima + derivatives, RegularGridInterpolator/interpn, RBFInterpolator). Deferred: FITPACK smoothing splines (splrep/BSpline/UnivariateSpline), Delaunay griddata/LinearND/NearestND
- [x] **Phase 7** — `stats` core delivered via `add-stats` (continuous distributions norm/expon/uniform/gamma/chi2/beta/t/f, summary statistics, pearsonr/spearmanr/linregress, ttest family/f_oneway/ks_2samp/chi2_contingency/normaltest, gaussian_kde). Deferred: rvs/fit sampling, QMC, discrete distributions, rank-based tests (mannwhitneyu/wilcoxon/kruskal/shapiro)
- [x] **Phase 8** — `signal` core delivered via `add-signal` (convolve/correlate/fftconvolve, windows, waveforms, lfilter/filtfilt/sosfilt/hilbert/detrend/freqz, butter/cheby1/cheby2/firwin + zpk conversions, welch/periodogram). Deferred: LTI systems, resampling, find_peaks, spectrogram/stft/csd, ellip/bessel, 2-D filtering
- [x] **Phase 9** — `sparse` core delivered via `add-sparse` (COO/CSR/CSC formats + conversions, eye/diags, spmv/spmm with capability-registry dispatch, spsolve/cg/gmres/norm, csgraph dijkstra/bellman_ford/floyd_warshall/connected_components/minimum_spanning_tree). Deferred: actual CUDA/OpenCL CSR device kernel (needs NumPP sparse backend), eigsh/svds, DIA/LIL/BSR, splu
- [x] **Phase 10** — `spatial` core delivered via `add-spatial` (pdist/cdist/squareform + metrics with backend dispatch, KDTree, 2-D ConvexHull/Delaunay, transform.Rotation + Slerp). Deferred: N-D Qhull geometry, Voronoi, GPU distance device kernel
- [x] **Phase 11** — `ndimage` core delivered via `add-ndimage` (correlate/convolve + gaussian/uniform/median/min/max/sobel/laplace with boundary modes + dispatch, binary/grey morphology, distance_transform_edt, label/center_of_mass/measurements, order-0/1 shift/zoom/rotate/affine/map_coordinates). Deferred: cubic-spline (order>=2), Fourier filters, GPU separable-convolution device kernel
- [x] **Phase 12** — `cluster` + `io` delivered via `add-cluster-io` (whiten/vq/kmeans2, linkage/fcluster/cophenet; mmread/mmwrite, wavread/wavwrite, loadarff). Deferred: stochastic kmeans, dendrogram, scipy.datasets, MATLAB .mat. v1.0 tagged.
