# Tasks — special + constants (Phase 1)

## 1. Foundation (minimal Phase 0 subset)
- [x] Root `CMakeLists.txt`: C++20, `SCIPP_WITH_{BLAS,LAPACK,CUDA,OPENCL,METAL}` options (default OFF), `cmake/config.hpp.in` generated to `scipp/config.hpp`
- [x] `find_package(NumPP CONFIG REQUIRED)`; `conanfile.py` + `vcpkg.json` pin an exact NumPP version; configure fails fast if a requested GPU flag has no matching NumPP variant
- [x] `include/scipp/scipp.hpp` umbrella, `fwd.hpp`, `version.hpp.in`
- [x] `include/scipp/error.hpp`: `scipp::error` aligned with `numpp::error`; `value_error`, `not_implemented_error` (linalg_error reserved for later phases)
- [x] `elementwise()` helper lifting scalar `double(double)` / `double(double,double)` kernels over `numpp::ndarray` via NumPP broadcasting (no per-function loops)
- [x] CPU-only build green with all backend flags OFF
- [x] `scripts/bootstrap_numpp.sh` to build+install the pinned NumPP into `.deps/`

## 2. SciPy oracle harness
- [x] Self-contained test harness (`scipp_test.hpp`, mirrors NumPP's) wired via CMake/CTest
- [x] Python generator (`tests/oracle/generate.py`) runs real SciPy at `/home/leonardo/work/scipy` over declared inputs → frozen `tests/golden/golden.hpp` + `src/constants/codata_table.inc`
- [x] `CHECK_CLOSE`/`CHECK_ARR` `allclose` assertions with per-function `rtol`/`atol`; loads frozen golden data (runs Python-free in CI)
- [x] Regeneration flow documented in the generator (refresh + review diff)

## 3. special — gamma family
- [x] `gamma`, `gammaln`, `loggamma`; poles → inf/nan, no throw
- [x] `digamma` (series + reflection), `polygamma` (Hurwitz zeta); `beta`, `betaln`
- [x] Oracle tests incl. recurrence `gamma(x+1)≈x·gamma(x)`, betaln identity, non-positive-integer poles

## 4. special — error & exponential integrals
- [x] `erf`, `erfc`; `erfinv`, `erfcinv` (rational + Newton refine)
- [x] `expn`, `exp1`, `expi`, `exprel` (stable near 0)
- [x] Oracle tests incl. `erf+erfc≈1`, inverse round-trips, `exprel→1` at origin

## 5. special — Bessel
- [x] `jv`, `yv`, `iv`, `kv`; integer-order `jn`/`yn`; `i0`/`i1`/`k0`/`k1`
- [x] Oracle tests across small/large arguments and several orders; specialized vs general agreement; per-function tolerances documented

## 6. special — orthogonal evaluators, combinatorics, reductions
- [x] `eval_legendre`, `eval_chebyt`, `eval_hermite`, `eval_laguerre`, `eval_genlaguerre`
- [x] `comb`, `perm`, `factorial` with `exact` mode (int128 → gamma fallback)
- [x] `logsumexp`, `softmax`, `log_softmax` with `axis`/`keepdims` (max-shift)
- [x] Oracle tests incl. high-degree stability, exact vs inexact, overflow-safe logsumexp, softmax sums to 1

## 7. constants
- [x] `constexpr` scale constants (`pi`, `c`, `h`, `hbar`, `G`, `e`, `k`, `N_A`, `R`, `g`, `atm`, …) and named unit factors
- [x] Generated `physical_constants` CODATA table (445 entries) with `value`/`unit`/`precision`; unknown name → `value_error`
- [x] `convert_temperature`, `lambda2nu`, `nu2lambda` (element-wise)
- [x] Oracle tests: scale constants & unit factors equal SciPy; table lookup triple; temperature & wavelength round-trips

## 8. Wire-up & validation
- [x] Export `special` + `constants` headers from `scipp/scipp.hpp`
- [x] `openspec validate add-special-constants --strict` green
- [x] Full CPU test suite green against frozen oracle data (11 cases / 248 checks)
- [x] Check off Phase 1 in `bootstrap-scipp-foundation/tasks.md`; update README status (Phase 1 ✅)
