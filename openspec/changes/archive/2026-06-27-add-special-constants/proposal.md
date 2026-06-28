# Add special + constants (Phase 1)

## Why

Phase 1 of the SciPP roadmap (see `bootstrap-scipp-foundation`). `scipy.special`
and `scipy.constants` go first because the rest of SciPy depends on them:
`stats` distributions need gamma/beta/erf and `logsumexp`; `signal`/`integrate`/
`optimize` need Bessel functions, orthogonal polynomials, and physical constants.
Delivering them first unblocks every later phase.

Both are also an ideal first vertical slice: most special functions are scalar,
element-wise CPU kernels with crisp numerical oracles, and constants are
compile-time tables — so this change can stand up the **minimal Phase 0
foundation** (build skeleton, NumPP dependency, error model, SciPy oracle
harness) against a small, well-understood surface before the heavier numerical
subpackages (`linalg`, `fft`) arrive.

## What changes

This change introduces two capability specs plus the foundation needed to build
and test them:

- **special** — `scipp::special`, ported element-wise over `numpp::ndarray`:
  - gamma family: `gamma`, `gammaln`, `loggamma`, `digamma`, `polygamma`,
    `beta`, `betaln`
  - error functions: `erf`, `erfc`, `erfinv`, `erfcinv`
  - exponential/log integrals: `expn`, `exp1`, `expi`, `exprel`
  - Bessel: `jv`, `yv`, `iv`, `kv`, integer-order `jn`/`yn`, and the common
    `i0`/`i1`/`k0`/`k1`
  - orthogonal-polynomial evaluators: `eval_legendre`, `eval_chebyt`,
    `eval_hermite`, `eval_laguerre`, `eval_genlaguerre`
  - combinatorics: `comb`, `perm`, `factorial`
  - reductions: `logsumexp`, `softmax`, `log_softmax` (with `axis`)
- **constants** — `scipp::constants`: the mathematical/SI scale constants
  (`pi`, `c`, `h`, `hbar`, `G`, `e`, `k`, `N_A`, `R`, …), the CODATA
  `physical_constants` table with `value`/`unit`/`precision` lookup, and
  conversion helpers `convert_temperature`, `lambda2nu`/`nu2lambda`, plus the
  named unit scale factors.
- **Phase 0 foundation (minimal, to satisfy the foundation specs for this slice)**:
  CMake ≥ 3.25 / C++20 skeleton with `SCIPP_WITH_*` flags (all OFF), a pinned
  NumPP Conan/vcpkg dependency resolved via `find_package`, the `scipp/scipp.hpp`
  umbrella + `version.hpp.in`, the `scipp::error` model aligned with
  `numpp::error`, and the Catch2 + SciPy oracle harness with frozen `tests/golden/`
  data.

## Impact

- Affected specs: **adds** `special` and `constants` capabilities; **implements**
  (first concrete slice of) the `scipp-foundation` and `scipy-oracle` capabilities
  from the bootstrap change.
- Affected code: new `include/scipp/{special,constants}/`, `src/{special,constants}/`,
  `tests/`, and the root build files (`CMakeLists.txt`, `conanfile.py`,
  `vcpkg.json`, `cmake/`). No existing code changes (repo is greenfield).
- Roadmap: checks off Phase 1 in `bootstrap-scipp-foundation/tasks.md`.

## Non-goals

- **No GPU acceleration** of special functions — these are scalar/element-wise CPU
  kernels; the `backend-acceleration` substrate is not exercised this phase.
- **No** hypergeometric (`hyp2f1`/`hyp1f1`), elliptic, spheroidal, Mathieu, or
  Struve function families — deferred.
- **No** orthogonal-polynomial *class* objects (`scipy.special.legendre(n)`
  returning a `poly1d`) — only the `eval_*` evaluators are in scope here.
- **No** `constants` astronomical/obscure unit long-tail beyond the standard SI
  scale factors and the CODATA table lookup.
- **No** array-API/Python bindings.
