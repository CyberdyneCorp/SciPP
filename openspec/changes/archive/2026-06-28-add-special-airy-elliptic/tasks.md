# Tasks — Airy functions and elliptic integrals

- [x] `airy_t`/`ellipj_t` structs + `airy`/`airye`/`ellipk`/`ellipkm1`/`ellipe`/`ellipkinc`/`ellipeinc`/`ellipj` declarations in `include/scipp/special/special.hpp`
- [x] `src/special/airy.cpp`: Bessel-relation kernels for `x>0` (K/I) and `x<0` (J/Y), exact constants at 0, `airye` scaling with `nan` Ai/Aip for `x<0`; wired into `src/CMakeLists.txt`
- [x] `src/special/elliptic.cpp`: AGM `ellipk`/`ellipkm1`/`ellipe`, Carlson `R_F`/`R_D` incomplete integrals with phi reduction, Cephes descending-Landen `ellipj`
- [x] extend oracle generator (airy/airye grids, complete/incomplete/jacobi grids over (phi/u, m)); regenerate `golden.hpp`
- [x] `tests/test_special_airy_elliptic.cpp` (oracle match + Wronskian + sn^2+cn^2 / dn^2+m sn^2 identities + domain edges); registered in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-special-airy-elliptic --strict`
- [x] trim the Airy and Elliptic lines from the `add-special-extras` backlog
