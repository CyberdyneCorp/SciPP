# ScyPP

[![C++20](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](https://en.cppreference.com/w/cpp/20)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20OpenCL%20%7C%20Metal-76B900?logo=nvidia&logoColor=white)](#tiered-acceleration)
[![Spec](https://img.shields.io/badge/spec-OpenSpec-3B5526)](openspec/)

Modern **C++20 port of [SciPy](https://github.com/scipy/scipy)** — the scientific
computing layer that sits on top of NumPy. ScyPP is a clean-room implementation
validated against real SciPy as a numerical oracle, and is the SciPy sibling of:

- **[NumPP](https://github.com/CyberdyneCorp/NumPP)** — the NumPy port (N-D
  `ndarray`, dtypes, ufuncs, linalg, fft, random) with a tiered CPU / BLAS / GPU
  acceleration layer. **ScyPP is built on NumPP** — it is the array engine.
- **[SymPP](https://github.com/leonardoaraujosantos/SymPP)** — the SymPy port
  (symbolic math).

```cpp
#include "scypp/scypp.hpp"
#include "numpp/numpp.hpp"
using numpp::ndarray;

ndarray A = /* ... */;
auto [P, L, U] = scypp::linalg::lu(A);          // SciPy-shaped decomposition
ndarray y      = scypp::signal::filtfilt(b, a, x);
auto   spec    = scypp::fft::dct(x, /*type=*/2); // DCT-II (GPU-accelerated)
auto   res     = scypp::optimize::minimize(f, x0, "BFGS");
double p       = scypp::stats::ttest_ind(a, b).pvalue;
```

## Why ScyPP

SciPy is the de-facto scientific toolbox, but it is Python + Cython + Fortran and
assumes a desktop/server with BLAS/LAPACK. ScyPP brings the same algorithms to a
single portable C++20 source tree that:

- **Builds and runs everywhere, including iOS and Android** — a portable,
  dependency-free C++ CPU kernel is always present and is the only thing required
  to build (only NumPP + the standard library).
- **Accelerates SciPy's CPU-bound hot kernels on the GPU** — FFT, dense linalg,
  sparse SpMV, `ndimage` separable convolution and pairwise distances gain
  optional **CUDA / OpenCL / Metal** backends, routed through NumPP's weak-linked
  device dispatch and always falling back to the CPU.
- **Is validated against real SciPy** — every numeric result is asserted
  `allclose` to the reference implementation.

## Tiered acceleration

ScyPP does not build a second device stack. It reuses NumPP's runtime dispatch —
`CapabilityRegistry`, `last_backend()`, the `NUMPP_GPU_TARGET` override
(`cpu|cuda|opencl|metal|auto`) and the device buffer pool — and registers
SciPy-specific kernels into the same weak-linked vtable.

```
scypp::linalg / fft / signal / optimize / stats / sparse / spatial / ndimage / ...
                         │  (numpp::ndarray in, numpp::ndarray out)
                         ▼
   numpp::ndarray · dtypes · ufuncs · linalg · fft · random          ← NumPP engine
                         │
         CapabilityRegistry · weak GpuVTable · last_backend()
                         │
   portable CPU kernel ─► BLAS/LAPACK ─► CUDA / OpenCL / Metal  (runtime select)
```

Below a per-operation size threshold, or for unsupported dtypes, or when no device
is present, the portable CPU kernel is used. An accelerated result always equals
the CPU result within documented floating-point tolerance.

## Target capability map

Each capability ports a public SciPy subpackage into the `scypp::<name>`
namespace. See [`openspec/project.md`](openspec/project.md) for the full map.

| Subpackage | Scope |
|------------|-------|
| `special` · `constants` | Special functions (gamma/erf/Bessel/orthogonal polys), CODATA constants |
| `linalg` | Decompositions, matrix functions, solvers (BLAS/LAPACK + GPU GEMM) |
| `fft` · `fftpack` | FFT, real/N-D transforms, DCT/DST (GPU-accelerated) |
| `optimize` | `minimize`, `root`, `least_squares`/`curve_fit`, `linprog`/`milp` |
| `integrate` · `differentiate` | `quad`, `solve_ivp`, `solve_bvp`, finite differences |
| `interpolate` | `interp1d`, splines, `griddata`/`RBFInterpolator` |
| `stats` | Distributions, hypothesis tests, QMC, `gaussian_kde` |
| `signal` | Filtering, filter design, spectral analysis, LTI systems |
| `sparse` | CSR/CSC/COO formats, `sparse.linalg`, `csgraph` (GPU SpMV) |
| `spatial` | KD-trees, distances, ConvexHull/Delaunay/Voronoi, rotations |
| `ndimage` | Filters, morphology, measurements (GPU separable convolution) |
| `cluster` · `io` · `datasets` | k-means/hierarchical, Matrix Market/WAV/ARFF I/O, sample datasets |

## Project status

ScyPP is in **Phase 0 — foundation**. The architecture, build/packaging, the NumPP
integration contract, the CUDA/OpenCL/Metal backend strategy, the SciPy oracle
harness, and the full parity roadmap are specified with **OpenSpec** under
[`openspec/`](openspec/). Each subpackage graduates into its own OpenSpec change as
it is implemented, ported against the SciPy oracle.

```bash
openspec list                              # active changes
openspec show bootstrap-scypp-foundation   # the foundation + parity roadmap
openspec validate --all --strict           # validate the specs
```

## Building (planned)

```bash
# CPU-only (mobile-friendly, zero extra deps beyond NumPP)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# With GPU acceleration (propagates to NumPP's backend)
cmake -S . -B build -DSCYPP_WITH_CUDA=ON     # or -DSCYPP_WITH_OPENCL=ON / -DSCYPP_WITH_METAL=ON
```

Backend feature flags (`SCYPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}`) all default
**OFF**; enabling one enables the matching NumPP backend so both layers share a
single device runtime.

## Relationship to the reference SciPy

The upstream SciPy checkout is the **oracle and reference implementation**; ScyPP
re-implements its algorithms in portable C++ (no CPython C-API, no `PyObject`) and
adds GPU acceleration where SciPy has none. Spec requirements cite the SciPy source
they port as breadcrumbs, e.g. `(oracle: scipy/linalg/_decomp_lu.py)`.

## License

MIT — see [LICENSE](LICENSE).
