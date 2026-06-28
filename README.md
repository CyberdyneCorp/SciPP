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
#include "numpp/core/ndarray.hpp"
using numpp::ndarray;
namespace sp = scypp::special;
namespace cst = scypp::constants;

// Special functions — scalar or element-wise over a numpp::ndarray.
double g   = sp::gamma(0.5);                  // 1.7724538509...
double b   = sp::jv(0.0, 2.0);                // Bessel J0(2)
ndarray ls = sp::logsumexp(x, /*axis=*/1);    // numerically stable

// Physical constants & conversions (CODATA, matches scipy.constants).
double c   = cst::c;                          // speed of light
double m_e = cst::value("electron mass");     // CODATA table lookup
double f   = cst::convert_temperature(100.0, "Celsius", "Fahrenheit");  // 212

// Linear algebra (scipy.linalg conventions, GEMM via NumPP).
auto [P, L, U] = scypp::linalg::lu(Asq);      // explicit P·L·U
ndarray E      = scypp::linalg::expm(Asq);    // matrix exponential
```

> **All 12 subpackages are implemented** (`special`, `constants`, `linalg`, `fft`,
> `optimize`, `integrate`, `differentiate`, `interpolate`, `stats`, `signal`,
> `sparse`, `spatial`, `ndimage`, `cluster`, `io`). Deferred long-tail features are
> tracked as open OpenSpec backlog changes under `openspec/changes/`.

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

**v1.0 — all 12 phases shipped.** Every public SciPy subpackage's commonly-used
surface is ported, built on NumPP and validated against SciPy 1.15 — **5182 oracle
checks, 0 divergences**:

- **Phase 1** — `scypp::special` (gamma/erf/Bessel/exponential integrals/
  orthogonal evaluators/combinatorics/`logsumexp`/`softmax`) and `scypp::constants`
  (CODATA table + scale constants + unit conversions). This phase also stood up the
  foundation: CMake/C++20 skeleton, pinned NumPP `find_package` dependency, the
  `scypp::error` model, and the frozen-golden SciPy oracle harness.
- **Phase 2** — `scypp::linalg`: `inv`/`det`/`solve`/`lstsq`/`pinv`/`pinvh`/`norm`,
  `lu`/`qr`/`svd`/`cholesky` (+ factor/solve helpers), `eig`/`eigh`, `expm` (Padé)
  and `polar`, and special-matrix constructors. Standard decompositions delegate to
  NumPP; GEMM-heavy paths route through NumPP's BLAS/GPU `matmul`.
- **Phase 3** — `scypp::fft` (+ legacy `scypp::fftpack`): the FFT family
  (`fft`/`rfft`/`hfft`, N-D variants, `fftfreq`/`fftshift`) delegated to NumPP, plus
  SciPy's **DCT/DST** types I–IV, `next_fast_len`, and axis-wise transforms.
- **Phase 4** — `scypp::optimize`: scalar root finders (`brentq`/`bisect`/`newton`),
  `minimize_scalar` (brent/bounded), `minimize` (Nelder–Mead/BFGS), and nonlinear
  least squares (`least_squares`/`curve_fit`/`fsolve`).
- **Phase 5** — `scypp::integrate` + `scypp::differentiate`: fixed-sample and
  adaptive quadrature (`trapezoid`/`simpson`/`quad`/`fixed_quad`), explicit ODE
  solvers (`solve_ivp` RK45/RK23), and `derivative`/`jacobian`/`hessian`.
- **Phase 6** — `scypp::interpolate`: `Interp1d`, `CubicSpline`/`PchipInterpolator`/
  `Akima1DInterpolator` (with derivatives), `RegularGridInterpolator`/`interpn`, and
  `RBFInterpolator` (radial basis functions).
- **Phase 7** — `scypp::stats`: continuous distributions (`norm`/`gamma`/`beta`/`t`/
  `f`/`chi2`/…) with `pdf`/`cdf`/`ppf`, summary statistics, correlation/regression
  (`pearsonr`/`spearmanr`/`linregress`), hypothesis tests (`ttest_*`/`f_oneway`/
  `ks_2samp`/`chi2_contingency`/`normaltest`), and `gaussian_kde`.
- **Phase 8** — `scypp::signal`: convolution, windows, waveforms, time-domain
  filtering (`lfilter`/`filtfilt`/`sosfilt`/`hilbert`/`freqz`), filter design
  (`butter`/`cheby1`/`cheby2`/`ellip`/`bessel`/`firwin` + zpk conversions), spectral
  estimation (`welch`/`periodogram`/`csd`/`coherence`/`spectrogram`/`stft`), peak
  analysis (`find_peaks`/`peak_prominences`/`peak_widths`), LTI systems
  (`bode`/`step`/`impulse`/`lsim`), resampling (`resample`/`resample_poly`/
  `decimate`/`upfirdn`), and `savgol_filter`/`medfilt`/`wiener`/`convolve2d`.
- **Phase 9** — `scypp::sparse`: COO/CSR/CSC formats + conversions, `eye`/`diags`,
  `spmv`/`spmm` with NumPP capability-registry **backend dispatch** (CPU kernel +
  GPU-ready architecture), `sparse.linalg` (`spsolve`/`cg`/`gmres`/`norm`), and
  `sparse.csgraph` (`dijkstra`/`bellman_ford`/`floyd_warshall`/
  `connected_components`/`minimum_spanning_tree`).
- **Phase 10** — `scypp::spatial`: distances (`pdist`/`cdist`/`squareform` + metrics
  with backend dispatch), `KDTree`, 2-D `ConvexHull`/`Delaunay`, and 3-D rotations
  (`transform::Rotation` quat/matrix/euler/rotvec + `apply`/`inv`/compose/`Slerp`).
- **Phase 11** — `scypp::ndimage`: filters (`gaussian`/`uniform`/`median`/`sobel`/…
  with boundary modes + dispatch), morphology (binary/grey + `distance_transform_edt`),
  measurements (`label`/`center_of_mass`), and order-0/1 geometric transforms
  (`shift`/`zoom`/`rotate`/`affine_transform`/`map_coordinates`).
- **Phase 12** — `scypp::cluster` (`whiten`/`vq`/`kmeans2`, `linkage`/`fcluster`/
  `cophenet`) and `scypp::io` (Matrix Market `mmread`/`mmwrite`, WAV `wavread`/
  `wavwrite`, ARFF `loadarff`).

The architecture, CUDA/OpenCL/Metal backend strategy, and the full parity roadmap
are specified with **OpenSpec** under [`openspec/`](openspec/). Each remaining
subpackage graduates into its own OpenSpec change, ported against the SciPy oracle.

```bash
openspec list                              # active changes
openspec show bootstrap-scypp-foundation   # the foundation + parity roadmap
openspec validate --all --strict           # validate the specs
```

## Building

ScyPP depends on NumPP via `find_package(NumPP)`. For local development, build and
install the pinned NumPP into `.deps/` first:

```bash
# 1. Build + install the pinned NumPP dependency (expects ../NumPP source)
./scripts/bootstrap_numpp.sh            # or: ./scripts/bootstrap_numpp.sh /path/to/NumPP

# 2. Configure + build ScyPP (CPU-only, mobile-friendly, zero extra deps)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 3. Run the SciPy-oracle test suite
ctest --test-dir build --output-on-failure

# With GPU acceleration (requires a NumPP package built with the matching backend)
cmake -S . -B build -DSCYPP_WITH_CUDA=ON   # or -DSCYPP_WITH_OPENCL=ON / -DSCYPP_WITH_METAL=ON
```

Refresh the frozen oracle data after changing the test set:
`python3 tests/oracle/generate.py` (requires Python + SciPy; CI runs against the
committed golden data without Python).

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
