# ScyPP — Project Context

## What this is
ScyPP is a clean-room **C++20 port of [SciPy](https://github.com/scipy/scipy)** —
the scientific-computing layer that sits on top of NumPy. It is the SciPy sibling
of two existing ports:

- **[NumPP](https://github.com/CyberdyneCorp/NumPP)** — the NumPy port (N-D
  `ndarray`, dtypes, ufuncs, linalg, fft, random) with a tiered acceleration
  layer (portable CPU kernel + optional weak-linked BLAS/LAPACK and **CUDA /
  OpenCL / Metal** GPU backends selected at runtime). ScyPP is **built on top of
  NumPP**: NumPP is the array engine and the backend-dispatch substrate.
- **[SymPP](https://github.com/leonardoaraujosantos/SymPP)** — the SymPy port
  (symbolic math). Available for closed-form/special-function needs.

Just as SciPy is validated against textbook math, ScyPP uses **real Python SciPy
as the numerical oracle**: tests assert `allclose` against values produced by the
reference implementation, exactly as NumPP validates against NumPy and SymPP
against SymPy.

## Relationship to the reference SciPy
The upstream SciPy checkout at `/home/leonardo/work/scipy` is the **oracle and
reference implementation**. Its own OpenSpec baseline
(`/home/leonardo/work/scipy/openspec/specs/`) documents the observed behavior of
each public subpackage and is the behavioral contract ScyPP ports against. ScyPP
re-implements the *algorithms* in portable C++ (no CPython C-API, no `PyObject`)
and adds GPU acceleration where SciPy has none.

## Capability map
Each capability corresponds to a public SciPy subpackage (`scipy/<name>/`),
ported into the `scypp::<name>` namespace:

| Capability | SciPy subpackage | Scope |
|------------|------------------|-------|
| cluster | `scipy/cluster` | Vector quantization / k-means and hierarchical clustering |
| constants | `scipy/constants` | Physical/mathematical constants, CODATA, unit conversion |
| datasets | `scipy/datasets` | Sample datasets (ascent, face, ECG) |
| differentiate | `scipy/differentiate` | Finite-difference derivative / jacobian / hessian |
| fft | `scipy/fft` | Discrete Fourier transforms (DCT/DST, real, N-D), GPU-accelerated |
| fftpack | `scipy/fftpack` | Legacy FFT API |
| integrate | `scipy/integrate` | Quadrature, ODE (`solve_ivp`), BVP |
| interpolate | `scipy/interpolate` | 1-D/N-D interpolation and splines |
| io | `scipy/io` | File I/O (MATLAB, WAV, Matrix Market, ARFF, NetCDF) |
| linalg | `scipy/linalg` | Dense linear algebra, decompositions, matrix functions (BLAS/LAPACK + GPU) |
| ndimage | `scipy/ndimage` | N-dimensional image processing (filters, morphology, measurements) |
| optimize | `scipy/optimize` | Minimization, root finding, least squares, linear/MILP |
| signal | `scipy/signal` | Filtering, filter design, spectral analysis, LTI systems |
| sparse | `scipy/sparse` | Sparse arrays/matrices, `sparse.linalg`, `sparse.csgraph` |
| spatial | `scipy/spatial` | KD-trees, computational geometry, distances, rotations |
| special | `scipy/special` | Special functions (gamma, Bessel, erf, orthogonal polynomials) |
| stats | `scipy/stats` | Probability distributions, statistical tests, QMC, KDE |

## What ScyPP does differently from SciPy
- **One portable source tree, three build profiles** — mobile (CPU-only, zero
  extra deps; builds on iOS/Android), desktop (+BLAS/LAPACK), workstation/server
  (+GPU). Acceleration is strictly additive; a portable CPU kernel is always
  present and is the only thing required to build.
- **CUDA / OpenCL / Metal acceleration** for the hot kernels SciPy leaves on the
  CPU (FFT, dense linalg, sparse SpMV, ndimage filters, distance matrices),
  routed through NumPP's weak-linked `CapabilityRegistry` + `GpuVTable` so the
  result always equals the CPU path within documented tolerance.

## Conventions worth knowing
- **Build system**: CMake ≥ 3.25, C++20, Conan + vcpkg packaging — mirrors
  NumPP/SymPP. Backend feature flags follow NumPP: `SCYPP_WITH_BLAS`,
  `SCYPP_WITH_LAPACK`, `SCYPP_WITH_CUDA`, `SCYPP_WITH_OPENCL`, `SCYPP_WITH_METAL`,
  all default OFF.
- **Namespacing**: implementation in `src/<subpackage>/`, public headers in
  `include/scypp/<subpackage>/`, umbrella header `scypp/scypp.hpp`.
- **Oracle breadcrumbs**: spec requirements cite the SciPy source they port,
  e.g. `(oracle: scipy/linalg/_decomp_lu.py)`.
- **Backend reuse**: ScyPP does not re-invent device management — it consumes
  NumPP's `CapabilityRegistry`, `last_backend()`, `NUMPP_GPU_TARGET`, and device
  buffer pool, and registers SciPy-specific kernels into the same vtable shape.

## Working with these specs
- This bootstrap change specifies the **foundation** (build, NumPP integration,
  backend acceleration, SciPy oracle) in detail and records the full
  **scipy-parity** roadmap as the scope guardrail. Each subpackage graduates into
  its own OpenSpec change when picked up.
- New work flows through `propose → apply → validate → archive`.
- Validate with `openspec validate --all --strict`.
