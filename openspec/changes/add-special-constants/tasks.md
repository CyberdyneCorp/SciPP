# Tasks — special + constants (Phase 1)

## 1. Foundation (minimal Phase 0 subset)
- [ ] Root `CMakeLists.txt`: C++20, `SCYPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}` options (default OFF), `cmake/config.hpp.in` generated to `scypp/config.hpp`
- [ ] `find_package(numpp CONFIG REQUIRED)`; `conanfile.py` + `vcpkg.json` pin an exact NumPP version; configure fails fast if a requested GPU flag has no matching NumPP variant
- [ ] `include/scypp/scypp.hpp` umbrella, `fwd.hpp`, `version.hpp.in`
- [ ] `include/scypp/error.hpp`: `scypp::error` aligned with `numpp::error`; `value_error`, `not_implemented_error` (linalg_error reserved for later phases)
- [ ] `elementwise()` helper lifting scalar `double(double)` / `double(double,double)` kernels over `numpp::ndarray` via NumPP broadcasting (no per-function loops)
- [ ] CPU-only build green with all backend flags OFF

## 2. SciPy oracle harness
- [ ] Catch2 wired via CMake/CTest
- [ ] Python generator (`tests/oracle/generate.py`) runs real SciPy at `/home/leonardo/work/scipy` over declared inputs → `tests/golden/*.json`
- [ ] C++ `allclose` assertion helper with per-function `rtol`/`atol`; loads frozen golden data (runs Python-free in CI)
- [ ] `oracle-refresh` build target regenerates golden data and surfaces diffs for review

## 3. special — gamma family
- [ ] `gamma`, `gammaln`, `loggamma` (Lanczos/Stirling + reflection); poles → inf/nan, no throw
- [ ] `digamma`, `polygamma`; `beta`, `betaln`
- [ ] Oracle tests incl. recurrence `gamma(x+1)≈x·gamma(x)`, betaln identity, non-positive-integer poles

## 4. special — error & exponential integrals
- [ ] `erf`, `erfc` (Cephes rational approx); `erfinv`, `erfcinv` (Newton refine)
- [ ] `expn`, `exp1`, `expi`, `exprel` (stable near 0)
- [ ] Oracle tests incl. `erf+erfc≈1`, inverse round-trips, `exprel→1` at origin

## 5. special — Bessel
- [ ] `jv`, `yv`, `iv`, `kv`; integer-order `jn`/`yn`; `i0`/`i1`/`k0`/`k1`
- [ ] Oracle tests across small/large arguments and several orders; specialized vs general agreement; per-function tolerances documented

## 6. special — orthogonal evaluators, combinatorics, reductions
- [ ] `eval_legendre`, `eval_chebyt`, `eval_hermite`, `eval_laguerre`, `eval_genlaguerre` (Clenshaw recurrence)
- [ ] `comb`, `perm`, `factorial` with `exact` mode (int128 → gamma fallback)
- [ ] `logsumexp`, `softmax`, `log_softmax` with `axis`/`keepdims` (max-shift)
- [ ] Oracle tests incl. high-degree stability, exact vs inexact, overflow-safe logsumexp, softmax sums to 1

## 7. constants
- [ ] `constexpr` scale constants (`pi`, `c`, `h`, `hbar`, `G`, `e`, `k`, `N_A`, `R`, `g`, `atm`, …) and named unit factors
- [ ] Generated `physical_constants` CODATA table (pinned release recorded) with `value`/`unit`/`precision`; unknown name → `value_error`
- [ ] `convert_temperature`, `lambda2nu`, `nu2lambda` (element-wise)
- [ ] Oracle tests: scale constants & unit factors equal SciPy; table lookup triple; temperature & wavelength round-trips

## 8. Wire-up & validation
- [ ] Export `special` + `constants` headers from `scypp/scypp.hpp`
- [ ] `openspec validate add-special-constants --strict` green
- [ ] Full CPU test suite green against frozen oracle data; regression test added for any divergence found & fixed
- [ ] Check off Phase 1 in `bootstrap-scypp-foundation/tasks.md`; update README status (Phase 1 ✅)
