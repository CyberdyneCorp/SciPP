# Tasks — discrete distributions + rank tests

- [x] `binom`, `poisson`, `geom`, `bernoulli`, `nbinom`, `hypergeom` (pmf/logpmf/cdf/sf/ppf + mean/var) in `src/stats/discrete.cpp`
- [x] `mannwhitneyu`, `wilcoxon`, `kruskal`, `kendalltau` (asymptotic + tie correction) in `src/stats/ranktests.cpp`
- [x] header decls; wire into build
- [x] extend oracle generator; regenerate
- [x] `tests/test_stats_discrete.cpp` vs SciPy
- [x] CPU build + full suite green; `openspec validate add-stats-discrete-tests --strict`
- [x] trim implemented items from `add-stats-extras` backlog
