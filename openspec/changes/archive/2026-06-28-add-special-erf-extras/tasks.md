# Tasks — error-function relatives

- [x] Faddeeva core in `src/special/erf_extras.cpp` (real-axis anchor + Taylor-in-y near the axis, Maclaurin series for small `|z|`, continued fraction for large `|z|`)
- [x] `erfcx` (direct product + asymptotic continued fraction + negative reflection) and `dawsn` (series + Rybicki Gaussian sum + asymptotic)
- [x] `wofz`, `voigt_profile` (with `gamma = 0` Gaussian limit and `sigma <= 0` -> nan) and `fresnel` (via the complex-erf relation)
- [x] declarations in `include/scipp/special/special.hpp` (`fresnel_t`, scalar entry points, `ndarray` overloads of `erfcx`/`dawsn`); wire `erf_extras.cpp` into `src/CMakeLists.txt`
- [x] extend oracle generator (erfcx, dawsn, fresnel S/C, voigt grids incl. gamma=0, wofz complex points); regenerate `golden.hpp`
- [x] tests in `tests/test_special_erf_extras.cpp` (oracle match + identities: erfcx(0), dawsn oddness, fresnel oddness, voigt symmetry/domain, wofz at 0 and on the imaginary axis)
- [x] full suite green; `openspec validate add-special-erf-extras --strict`
- [x] trim the "Error-fn relatives" item from the `add-special-extras` backlog
