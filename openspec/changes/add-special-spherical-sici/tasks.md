# Tasks — spherical Bessel functions and sine/cosine integrals

- [x] `sici_t`/`shichi_t` structs + `spherical_jn`/`yn`/`in`/`kn`/`sici`/`shichi` declarations in `include/scypp/special/special.hpp`
- [x] `src/special/spherical_bessel.cpp`: half-integer `jv`/`yv`/`iv`/`kv` relation for `x>0`, SciPy `x=0` limits, parity/closed-form recurrence for `x<0`; wired into `src/CMakeLists.txt`
- [x] `src/special/sici.cpp`: convergent series + asymptotic auxiliary functions for `sici`; `shichi` via `Ei`/`E1` (large `|x|`) and series (small `|x|`)
- [x] extend oracle generator (spherical grids for `n=0..4`, sici/shichi grids); regenerate `golden.hpp`
- [x] `tests/test_special_spherical_sici.cpp` (oracle match + `x=0` limits + closed-form/parity identities); registered in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-special-spherical-sici --strict`
- [x] trim the spherical part of "More Bessel" and the "Integrals" line from the `add-special-extras` backlog
