# Design — ScyPP foundation

## Context

ScyPP ports SciPy to C++20. SciPy is *the layer on top of NumPy*, so the single
most important architectural decision is that **ScyPP is the layer on top of
NumPP**: it consumes NumPP's `ndarray`, dtype system, broadcasting, ufuncs,
`linalg`, `fft` and `random` rather than re-implementing any of them. This mirrors
the upstream dependency (`scipy` → `numpy`) exactly and means ScyPP inherits, for
free, NumPP's tiered acceleration substrate — the portable CPU kernel plus the
weak-linked BLAS/LAPACK and CUDA/OpenCL/Metal GPU backends selected at runtime.

The same portability constraint that shapes NumPP shapes ScyPP: the library must
build and run CPU-only on iOS/Android with zero extra dependencies. Acceleration
is strictly additive.

## Goals / non-goals

- **Goal**: one source tree, three build profiles (mobile CPU-only, desktop
  +BLAS/LAPACK, workstation +GPU) from the same code, selecting the fastest
  available path at runtime.
- **Goal**: every numeric result validatable `allclose` against real SciPy.
- **Goal**: GPU acceleration for SciPy's CPU-bound hot kernels, reusing NumPP's
  device layer — never a second device-management stack.
- **Non-goal (this change)**: any Phase 1+ subpackage. Only the skeleton, the
  NumPP integration contract, the backend-acceleration architecture, and the
  oracle harness are in scope.

## Dependency model: ScyPP on NumPP

NumPP is consumed as a **pinned Conan/vcpkg release dependency** (decided — see
below), resolved via `find_package`, not vendored. The contract:

- ScyPP public APIs take and return `numpp::ndarray`. There is **no** ScyPP array
  type. `scypp::linalg::lu(A)` accepts a `numpp::ndarray` and returns
  `numpp::ndarray` factors.
- ScyPP reuses `numpp::DType`, broadcasting, `numpp::error` categories, and the
  ufunc machinery for any element-wise step inside an algorithm.
- ScyPP's CMake **propagates NumPP's backend feature flags**: enabling
  `SCYPP_WITH_CUDA` requires/imples `NUMPP_WITH_CUDA`, so both layers share one
  device runtime. ScyPP never links a CUDA/OpenCL/Metal SDK directly for device
  management — it goes through NumPP.

```
            scypp::optimize / linalg / fft / signal / stats / sparse / ...
                                   │  (ndarray in, ndarray out)
                                   ▼
   numpp::ndarray · dtypes · ufuncs · linalg · fft · random   ← array engine
                                   │
              CapabilityRegistry · weak GpuVTable · last_backend()
                                   │
   portable CPU kernel ──► BLAS/LAPACK ──► CUDA / OpenCL / Metal (runtime select)
```

## Module / repo layout (mirrors NumPP / SymPP)

```
include/scypp/
  scypp.hpp              # umbrella header
  fwd.hpp                # forward decls
  version.hpp(.in)
  backend/               # SciPy-kernel dispatch shims over NumPP's registry
  special/  linalg/  fft/  optimize/  integrate/  interpolate/
  signal/   stats/   sparse/  spatial/  ndimage/  cluster/
  constants/  io/   datasets/  differentiate/  fftpack/
src/<subpackage>/        # implementation .cpp mirroring include/
tests/                   # Catch2 + SciPy oracle harness, golden/ frozen data
cmake/  conanfile.py  vcpkg.json  CMakeLists.txt
```

Reserved-but-empty subpackage dirs make the phased roadmap visible in the tree;
each is filled by its own later change.

## Backend acceleration — reusing NumPP, not rebuilding it

SciPy leaves most kernels on the CPU. ScyPP adds GPU paths for the ones that
matter, but **registers them into NumPP's existing dispatch shape** rather than
creating a parallel device stack:

- **What we reuse unchanged**: `CapabilityRegistry` (compiled-backend + present-
  device probing), `last_backend()` introspection, the `NUMPP_GPU_TARGET`
  override (`cpu|cuda|opencl|metal|auto`), the bounded device buffer **reuse
  pool**, and the tiled GEMM kernel.
- **What ScyPP adds**: SciPy-shaped kernels behind the same weak-linked vtable
  idea — e.g. `fft` (Cooley-Tukey/Bluestein on device), `sparse` SpMV (CSR), a
  separable-convolution kernel for `ndimage`, and a pairwise-distance kernel for
  `spatial.distance`. Each is gated by the relevant `SCYPP_WITH_<BACKEND>` flag,
  built as a separate weak-linked TU, and is **null/absent** when not compiled or
  when no device is present.
- **Dispatch rule** (inherited): choose from `(op, dtype, problem size, available
  backends)`; below a per-op size threshold use the portable CPU kernel; a CPU
  fallback always exists; the accelerated result equals the CPU result within a
  documented tolerance (IEEE-exact for element-wise where NumPP guarantees it,
  within reduction/FMA tolerance for GEMM/FFT/SpMV).
- **Metal** is the Apple path NumPP already flags (`*_WITH_METAL`); ScyPP wires
  its kernels through the same slot so `auto` prefers Metal on Apple, else CUDA →
  OpenCL → CPU.

Algorithms that are inherently sequential or branchy (e.g. `solve_ivp` adaptive
stepping, simplex/MILP, hierarchical clustering) stay CPU-only by design; only
their inner dense/array kernels benefit from NumPP's accelerated linalg.

## SciPy oracle harness

Mirrors NumPP's NumPy oracle and SymPP's SymPy oracle:

- A test states inputs and the SciPy call; the harness runs real Python SciPy
  (`/home/leonardo/work/scipy`) to produce reference values and asserts the ScyPP
  result is `allclose` (with per-domain tolerances).
- A **frozen/checked mode** serializes the reference values into
  `tests/golden/` so CI runs without a Python/SciPy install.
- **Stochastic routines** (`stats` sampling, QMC, stochastic optimizers) pin
  seeds via NumPP's bit-exact `random` (PCG64) so comparisons are reproducible;
  where SciPy's own RNG stream cannot be matched bit-for-bit, the harness asserts
  distributional/statistical equivalence instead, documented per case.
- Breadcrumbs in specs cite the SciPy source ported, e.g.
  `(oracle: scipy/optimize/_minimize.py)`.

## Error model

`scypp::error` derives from / aligns with `numpp::error` so a single catch site
works across both layers. Ported categories follow SciPy: `value_error`,
`linalg_error` (singular/non-convergent factorizations), `not_implemented_error`,
plus result-object conventions where SciPy returns status objects rather than
throwing (e.g. `OptimizeResult.success`, integration `ier` codes) — these are
ported as returned status structs, not exceptions, matching SciPy behavior.

## Phasing rationale

`special` + `constants` go first because `stats`, `signal`, `integrate` and
`optimize` depend on special functions and physical constants. `linalg` and `fft`
come next as the most acceleration-sensitive and most widely depended-upon
numerical kernels. The remaining subpackages follow roughly in dependency order,
with `io`/`datasets`/`cluster` last as the least entangled.

## Resolved decisions

- **NumPP packaging** → **pinned Conan/vcpkg release dependency.** NumPP is
  consumed as a versioned package (not a vendored `add_subdirectory`); ScyPP pins
  an exact NumPP version and resolves it via `find_package`. GPU flag propagation
  (`SCYPP_WITH_<GPU>` → `NUMPP_WITH_<GPU>`) is honored by requiring a NumPP
  package variant/option built with the matching backend; configuration fails
  fast if the resolved NumPP package lacks the requested backend.
- **Sparse on GPU** → **CSR SpMV only** in the first `sparse` change. SpMM,
  sparse factorizations, and cuSPARSE/clSPARSE-class coverage are deferred to a
  dedicated later change.
- **`io.matlab`** → **out of scope.** The Phase 12 `io` change ports Matrix
  Market, WAV, and ARFF only. MATLAB `.mat` (v4/v5/v7.3) read/write is dropped
  from the roadmap; if wanted later it is a separate, standalone change.
