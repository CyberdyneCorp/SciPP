# Tasks — hypergeometric functions (hyp0f1, hyp1f1, hyp2f1, hyperu)

- [x] `hyp0f1`/`hyp1f1`/`hyp2f1`/`hyperu` declarations in `include/scipp/special/special.hpp`
- [x] `src/special/hypergeometric.cpp`: 0F1/1F1/2F1 power series, Kummer transform for `hyp1f1(x<0)`, Pfaff + `1-z` reflection + Gauss theorem for `hyp2f1`, two-`1F1` connection formula for `hyperu`; wired into `src/CMakeLists.txt`
- [x] extend oracle generator (hyp0f1/hyp1f1/hyp2f1/hyperu grids); regenerate `golden.hpp`
- [x] `tests/test_special_hypergeometric.cpp` (oracle match + identities + domain edges); registered in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-special-hypergeometric --strict`
- [x] trim the "Hypergeometric" line from the `add-special-extras` backlog
