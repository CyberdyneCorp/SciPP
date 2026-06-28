# Bootstrap SciPP foundation

## Why

SciPP is a clean-room C++20 port of SciPy — the SciPy sibling of NumPP (NumPy)
and SymPP (SymPy). SciPy is enormous (19 public subpackages spanning optimization,
integration, interpolation, linear algebra, signal/image processing, statistics,
sparse matrices, spatial structures, and special functions), so we commit to a
**phased roadmap** and detail only the foundation now.

The foundation locks in the four cross-cutting decisions every later subpackage
depends on:

1. **Built on NumPP.** SciPy is built on NumPy; SciPP is built on NumPP. NumPP's
   `ndarray`, dtype system, broadcasting, ufuncs, linalg, fft and random are the
   array engine. SciPP does not re-implement arrays — it ports SciPy's
   *algorithms* on top of them.
2. **SciPy is the oracle.** Like NumPP↔NumPy and SymPP↔SymPy, every numeric
   result is validated `allclose` against real Python SciPy
   (`/home/leonardo/work/scipy`), whose OpenSpec baseline is the behavioral
   contract we port against.
3. **Tiered acceleration with CUDA / OpenCL / Metal.** The one thing we
   deliberately do differently from SciPy: hot kernels (FFT, dense linalg, sparse
   SpMV, ndimage filters, distance matrices) gain optional GPU backends, routed
   through **NumPP's existing weak-linked `CapabilityRegistry` + `GpuVTable`**, so
   the library still compiles and runs CPU-only on iOS/Android and the GPU result
   always equals the CPU path within tolerance.
4. **One source tree, three build profiles** — mobile (CPU-only), desktop
   (+BLAS/LAPACK), workstation (+GPU) — selected at runtime.

This change establishes Phase 0 (skeleton, build/packaging, NumPP integration,
SciPy oracle harness, backend-acceleration architecture) and records the full
SciPy parity backlog. Everything else (optimize, integrate, signal, stats, …)
follows as separate OpenSpec changes against this baseline.

## What changes

This change introduces the following capabilities (spec deltas):

- **scipp-foundation** — project skeleton, the NumPP dependency contract (SciPP
  builds on `numpp::ndarray` and the NumPP backend substrate), CMake ≥ 3.25 /
  C++20 layout, feature flags, Conan + vcpkg packaging, the iOS/Android
  CPU-only build contract, and the ported exception model
  (`scipp::error` hierarchy mirroring SciPy/NumPP error categories).
- **backend-acceleration** — how SciPP-specific kernels (FFT, GEMM-backed
  linalg, sparse SpMV, separable ndimage convolution, pairwise distances) reuse
  NumPP's `CapabilityRegistry`, `last_backend()`, `NUMPP_GPU_TARGET` override and
  device buffer pool; registering SciPy kernels into the same `GpuVTable` shape
  for **CUDA, OpenCL and Metal**, always with a portable CPU fallback and
  documented cross-backend result equivalence.
- **scipy-oracle** — the SciPy validation harness: tests express expected results
  by running real Python SciPy and asserting `allclose`, with a checked/frozen
  generation mode so CI can run without Python, plus the random-seed and
  tolerance conventions for stochastic routines (stats, optimize, QMC).
- **scipy-parity** — the prioritized, tiered backlog of the 19 SciPy subpackages
  with a suggested implementation order, so gap-closing work flows through
  OpenSpec as discrete changes.

## Phased roadmap (scope guardrail)

Only Phase 0 (foundation) is specified in detail here. Later phases are the
agreed roadmap and each arrives as its own OpenSpec change. **Do not implement
Phase 1+ subpackages in this change.**

| Phase | Title | This change? |
|------:|-------|:---:|
| 0 | Foundation: build/packaging, NumPP integration, SciPy oracle, backend-acceleration architecture | ✅ |
| 1 | `special` (gamma/erf/Bessel/orthogonal polynomials) + `constants` — broadly depended on by other modules | ⬜ later |
| 2 | `linalg` (decompositions, matrix functions, solvers; BLAS/LAPACK + GPU GEMM) | ⬜ later |
| 3 | `fft` / `fftpack` (DCT/DST, real & N-D transforms; GPU-accelerated) | ⬜ later |
| 4 | `optimize` (minimize, root, least_squares, linprog/milp) | ⬜ later |
| 5 | `integrate` (quad, `solve_ivp`, BVP) + `differentiate` | ⬜ later |
| 6 | `interpolate` (1-D/N-D, splines) | ⬜ later |
| 7 | `stats` (distributions, tests, QMC, KDE) | ⬜ later |
| 8 | `signal` (filtering, filter design, spectral, LTI) | ⬜ later |
| 9 | `sparse` (+ `sparse.linalg`, `sparse.csgraph`; GPU SpMV) | ⬜ later |
| 10 | `spatial` (KD-tree, distances, ConvexHull/Delaunay/Voronoi, transform/Rotation) | ⬜ later |
| 11 | `ndimage` (filters, morphology, measurements; GPU separable convolution) | ⬜ later |
| 12 | `cluster`, `io`, `datasets`, hardening, docs, v1.0 packaging | ⬜ later |

## Reuse vs rewrite

Per project policy, prefer adapting proven code over reinventing — after testing:

- **Reuse from NumPP** (`/home/leonardo/work/NumPP`): the entire array engine
  (`numpp::ndarray`, dtypes, broadcasting, ufuncs, `linalg`, `fft`, `random`),
  the backend-dispatch substrate (`CapabilityRegistry`, weak-linked `GpuVTable`,
  `last_backend()`, `NUMPP_GPU_TARGET`, the CUDA/OpenCL device pool + tiled GEMM),
  and the build/packaging skeleton (CMake options, Conan, vcpkg, config.hpp).
- **Port from SciPy** (`/home/leonardo/work/scipy`): the algorithms themselves,
  re-expressed in C++ without the CPython C-API; the documented behavior, error
  model, and edge cases captured in its OpenSpec baseline.
- **Reuse from SymPP** when a closed form helps (special-function asymptotics,
  symbolic Jacobians); otherwise numeric.
- **Rewrite**: anything bound to CPython (`PyObject*`, Cython `cdef`, `f2py`
  wrappers) — re-implemented as pure C++ calling BLAS/LAPACK or device kernels
  directly.

## Non-goals

- **Not** a Python extension module — no CPython C-API, no `PyObject`. (Optional
  Python bindings may be a separate change much later.)
- **No** Phase 1+ subpackage implementation here — this change is foundation +
  roadmap only.
- **Not** re-implementing arrays/dtypes/ufuncs/fft/random — those come from NumPP.
- **No** new GPU device-management layer — SciPP reuses NumPP's. This change
  specifies the *integration*, not new vtables.
- **No** drop-in ABI/source compatibility with SciPy's C/Cython API or `.so`
  layout, and no pickling of SciPy objects.
