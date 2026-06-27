# Tasks — Bootstrap ScyPP foundation

## Phase 0 — Foundation (this change)

### Project skeleton & build
- [ ] CMake ≥ 3.25 / C++20 project; `include/scypp/` + `src/<subpackage>/` layout with reserved subpackage dirs
- [ ] Generated `config.hpp` with `SCYPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}` flags (default OFF)
- [ ] NumPP wired as a **pinned Conan/vcpkg release dependency** resolved via `find_package` (not vendored), umbrella `scypp/scypp.hpp`, `version.hpp.in`
- [ ] Configure-time check: requested GPU flag requires a NumPP package variant built with the matching `NUMPP_WITH_<GPU>`; fail fast otherwise
- [ ] GPU flag propagation: `SCYPP_WITH_<GPU>` enables the matching `NUMPP_WITH_<GPU>`
- [ ] CPU-only build verified green (all backend flags OFF); iOS/Android cross-compile contract documented

### NumPP integration
- [ ] Confirm public-API contract: ScyPP routines take/return `numpp::ndarray`, no ScyPP array type
- [ ] `scypp::error` aligned with `numpp::error`; `value_error`/`linalg_error`/`not_implemented_error`; status-struct convention for SciPy result objects

### Backend acceleration
- [ ] Dispatch shim layer over NumPP's `CapabilityRegistry` / `last_backend()` / `NUMPP_GPU_TARGET` / device pool
- [ ] Define the SciPy kernel vtable slots (fft, gemm-backed linalg step, CSR SpMV, separable convolution, pairwise distance) — interface only, no subpackage impl
- [ ] Size-threshold + dtype-gating policy and CPU-fallback guarantee documented; cross-backend equivalence + `last_backend()` test pattern established

### SciPy oracle harness
- [ ] Catch2 test scaffold + SciPy oracle runner (real Python SciPy → `allclose`)
- [ ] Frozen/`tests/golden/` mode for Python-free CI; regeneration + divergence-review flow
- [ ] Stochastic-routine convention: seed via NumPP `random`; distributional-equivalence fallback documented
- [ ] `openspec validate --all --strict` green; CI gate added

## Roadmap — each item graduates into its own OpenSpec change

> Not implemented in this change. Ported against the SciPy oracle and its OpenSpec
> baseline; GPU-accelerable kernels reuse the backend-acceleration substrate.

- [x] **Phase 1** — `special` (gamma/erf/Bessel/orthogonal polys/logsumexp/softmax) + `constants` (CODATA, unit conversion) — delivered via `add-special-constants`
- [~] **Phase 2** — `linalg` core delivered via `add-linalg` (inv/det/solve/lstsq/pinv/pinvh/norm, lu/qr/svd/cholesky, eig/eigh, expm/polar, special matrices; GEMM via NumPP). Deferred to a follow-up: ldl, sqrtm/logm/funm, schur/qz/cossin, banded, matrix-equation solvers
- [x] **Phase 3** — `fft` + `fftpack` delivered via `add-fft` (fft/rfft/hfft, fftn/rfftn delegated to NumPP; dct/dst I–IV + next_fast_len + fftpack alias; ortho/forward DCT norm, N-D DCT, fht/czt deferred)
- [x] **Phase 4** — `optimize` core delivered via `add-optimize` (brentq/bisect/newton, minimize_scalar brent/bounded, minimize Nelder-Mead/BFGS, least_squares/curve_fit/fsolve). Deferred: linprog/milp, constrained (SLSQP/trust-constr), global optimizers, nnls
- [ ] **Phase 5** — `integrate` (quad/simpson/trapezoid, solve_ivp RK/Radau/BDF/LSODA, solve_bvp) + `differentiate`
- [ ] **Phase 6** — `interpolate` (interp1d, CubicSpline/Pchip/Akima, B-splines, griddata/RegularGrid/RBF)
- [ ] **Phase 7** — `stats` (distributions, summary stats, hypothesis tests, qmc, gaussian_kde)
- [ ] **Phase 8** — `signal` (convolve/fftconvolve, lfilter/filtfilt/sosfilt, filter design, welch/spectrogram/stft, LTI)
- [ ] **Phase 9** — `sparse` (CSR/CSC/COO/DIA/LIL/BSR, sparse.linalg spsolve/cg/gmres/eigsh/svds, csgraph; GPU SpMV)
- [ ] **Phase 10** — `spatial` (KDTree, cdist/pdist/squareform, ConvexHull/Delaunay/Voronoi, transform.Rotation; GPU distances)
- [ ] **Phase 11** — `ndimage` (gaussian/uniform/median filters, morphology, label/measurements, zoom/rotate/affine; GPU separable convolution)
- [ ] **Phase 12** — `cluster` (kmeans/vq, hierarchical linkage), `io` (mmread/wavfile/arff; MATLAB `.mat` out of scope), `datasets`; hardening, docs, v1.0 packaging
