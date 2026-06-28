# Tasks — Shapiro-Wilk and Anderson-Darling normality tests

- [x] `ShapiroResult` / `AndersonResult` structs + declarations in `include/scypp/stats/stats.hpp`
- [x] Royston AS R94 Shapiro-Wilk (weights, W, polynomial p-value, n==3 exact branch) in `src/stats/normality.cpp`
- [x] Anderson-Darling A^2 for the normal case (mean/std ddof=1, SciPy critical values) in `src/stats/normality.cpp`
- [x] wire `stats/normality.cpp` into `src/CMakeLists.txt`
- [x] extend oracle generator (normal, skewed, n==3 samples); regenerate golden
- [x] `tests/test_stats_normality.cpp` vs SciPy; register in `tests/CMakeLists.txt`
- [x] full suite green; `openspec validate add-stats-normality-tests --strict`
- [x] trim shapiro/anderson item from `add-stats-extras` backlog
