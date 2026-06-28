# Tasks — misc special functions (lambertw, zeta, struve, spence)

- [x] `lambertw`/`zeta`/`zetac`/`struve`/`modstruve`/`spence` declarations in `include/scypp/special/special.hpp`
- [x] `src/special/misc.cpp`: Halley `lambertw` (k=0/-1), Euler-Maclaurin `zeta`/`zetac`, power-series + Bessel asymptotic `struve`/`modstruve`, `Li_2` region-reduction `spence`; wired into `src/CMakeLists.txt`
- [x] extend oracle generator (lambertw/zeta/struve/spence grids); regenerate `golden.hpp`
- [x] `tests/test_special_misc.cpp` (oracle match + identities + domain edges); registered in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-special-misc --strict`
- [x] trim the "Misc" line from the `add-special-extras` backlog
