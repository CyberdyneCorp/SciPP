# Design — special + constants (Phase 1)

## Context

First implemented slice of SciPP. It realizes two SciPy subpackages and, because
the repo is greenfield, the minimal Phase 0 foundation they need to compile. The
foundation architecture is already specified in `bootstrap-scipp-foundation`; this
change *implements the subset* that `special`/`constants` exercise and defers the
rest (GPU dispatch shims, BLAS/LAPACK wiring) to the phases that need it.

## Numerical strategy: SciPy is the oracle, but match its algorithms

SciPy's `special` is largely the Cephes library (plus Boost.Math for a few
functions). To hit `allclose` against the SciPy oracle we port the **same
algorithm families** Cephes uses rather than naive series:

- **gamma/gammaln** — Lanczos / Stirling split as in Cephes `gamma.c`/`lgam.c`;
  reflection formula for negative arguments; `loggamma` is the complex/relative
  log-gamma with the principal branch.
- **digamma/polygamma** — asymptotic series with recurrence shift; `polygamma(n,x)`
  via the Hurwitz zeta derivative for `n ≥ 1`.
- **erf/erfc** — rational Chebyshev approximations (Cephes `ndtr.c`); `erfinv`/
  `erfcinv` via Newton refinement on `erf` from a rational initial guess.
- **Bessel** — Cephes `jv.c`/`yv.c`/`iv.c`/`kv.c`: power series for small
  argument, asymptotic expansions for large, recurrence for integer order; `i0`/
  `i1`/`k0`/`k1` use the dedicated Cephes polynomials.
- **expn/exp1/expi/exprel** — Cephes `expn.c`; `exprel(x) = (e^x − 1)/x` computed
  stably near 0.
- **orthogonal evaluators** — stable three-term recurrences (Clenshaw) matching
  SciPy's `eval_*` (which delegate to the recurrence, not the class objects).
- **logsumexp/softmax** — max-shift for numerical stability, matching SciPy's
  implementation exactly (including the `axis` and `keepdims` handling).

Tolerances are per-function (most `rtol ≤ 1e-12` for double; relaxed in the wings
where Cephes itself is only ~1e-10). Each tolerance is documented in the test.

### Reuse note
Cephes is the reference SciPy itself vendors (`scipy/special/cephes/`), public
domain. Where helpful we adapt those routines into C++ (no global error state;
re-entrant), citing the source file. SymPP is available if a closed form helps a
test, but the kernels here are numeric.

## API shape

Element-wise functions accept and return `numpp::ndarray` and also offer a scalar
`double` overload (the common call site):

```cpp
namespace scipp::special {
  double  gamma(double x);
  ndarray gamma(const ndarray& x);          // elementwise via numpp broadcast
  double  jv(double v, double x);
  ndarray jv(double v, const ndarray& x);
  ndarray logsumexp(const ndarray& a, std::optional<int> axis = {},
                    bool keepdims = false);
  // ...
}
```

- Element-wise kernels are written once as a scalar `double(double)` (or
  `double(double,double)`) and lifted over arrays by a small
  `elementwise(fn, x)` helper that uses NumPP broadcasting/iteration — no
  hand-written loops per function.
- `comb`/`perm`/`factorial` take integer args with an `exact` flag (exact uses
  `__int128`/big accumulation up to overflow, else the `gamma`-based float path),
  matching SciPy.
- Domain handling matches SciPy: out-of-domain returns `nan`/`inf` (e.g.
  `gamma` at non-positive integers → `inf`/`nan`), **not** exceptions — SciPy's
  special functions do not throw. `scipp::error` is reserved for genuine misuse
  (e.g. mismatched shapes from NumPP).

## constants design

- A single generated header table `physical_constants` — `name → {value, unit,
  uncertainty}` — transcribed from SciPy's CODATA-2022 dataset, with
  `value(name)`, `unit(name)`, `precision(name)` accessors throwing
  `scipp::value_error` on unknown keys (matching SciPy's `KeyError`).
- Top-level `constexpr double` scale constants (`pi`, `c`, `h`, `hbar`, `G`, `e`,
  `k`, `N_A`, `R`, `g`, `atm`, …) and named unit factors (`kilo`, `milli`,
  `inch`, `bar`, `hour`, `eV`, …) as `constexpr`.
- `convert_temperature(arr, from, to)` over `Celsius/Kelvin/Fahrenheit/Rankine`;
  `lambda2nu`/`nu2lambda` using `c`. These operate element-wise over `ndarray`.

## Foundation subset implemented here

```
CMakeLists.txt           # C++20, SCIPP_WITH_* flags (OFF), find_package(numpp)
cmake/                   # config.hpp.in, helpers
conanfile.py vcpkg.json  # pinned numpp/<ver> dependency
include/scipp/
  scipp.hpp  fwd.hpp  version.hpp.in  error.hpp
  special/*.hpp  constants/*.hpp
src/special/  src/constants/
tests/                   # Catch2 + oracle harness, golden/ frozen data
```

- **NumPP** is pinned (Conan/vcpkg) and resolved with `find_package(numpp
  CONFIG REQUIRED)`; configure fails fast if a requested `SCIPP_WITH_<GPU>` has no
  matching NumPP variant (no GPU is requested this phase).
- **Oracle harness**: a Python generator script runs real SciPy
  (`/home/leonardo/work/scipy`) over the test inputs and writes
  `tests/golden/*.json`; the C++ tests load the frozen values and assert
  `allclose`. CI runs Python-free against the committed golden data; a `just
  oracle-refresh`-style target regenerates and surfaces diffs.

## Risks / tradeoffs

- **Cephes accuracy ceiling**: a few functions (large-order Bessel, deep wings)
  are only ~1e-10 in SciPy itself; tests use matching per-function tolerances
  rather than a global `1e-12`.
- **`factorial(exact=True)` overflow**: SciPy returns Python bigints; SciPP caps
  exact mode at the `unsigned __int128` range and switches to the float/`gamma`
  path beyond, documented in the spec.
- **CODATA version drift**: pin to the same CODATA release the vendored SciPy uses
  so values match the oracle bit-for-bit; record the version in the table header.

## Open questions

- Adapt vendored Cephes C directly vs. re-derive in idiomatic C++? Default: adapt
  Cephes (public domain, proven, matches the oracle), modernized to be re-entrant.
