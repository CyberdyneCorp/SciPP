# Tasks — stats (Phase 7)

## 1. Module scaffold + special enablers
- [x] `include/scypp/stats/stats.hpp` (distribution structs, result structs, function decls); `src/stats/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [x] Internal `gammainc`/`gammaincc`/`betainc` (series + continued fraction) and `gammaincinv`/`betaincinv` (Newton + bisection)

## 2. Continuous distributions
- [x] `norm`, `expon`, `uniform` (closed form; norm via erf/erfinv)
- [x] `gamma`, `chi2` (gammainc), `beta` (betainc), `t`, `f` (betainc)
- [x] each: `pdf`/`logpdf`/`cdf`/`sf`/`ppf` + `mean`/`var`/`std`/`median`; ndarray `pdf`/`cdf`

## 3. Summary statistics
- [x] `gmean`, `hmean`, `moment`, `skew`, `kurtosis`, `sem`, `variation`, `zscore`, `iqr`, `rankdata`, `mode`, `describe`

## 4. Correlation, regression, tests
- [x] `pearsonr`, `spearmanr`, `linregress`
- [x] `ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`, `ks_2samp`, `chi2_contingency`, `normaltest`

## 5. Density estimation
- [x] `gaussian_kde` (Scott/Silverman bandwidth, covariance via numpp, N-D evaluate)

## 6. Oracle + validation
- [x] Extend `tests/oracle/generate.py` with stats golden values; regenerate
- [x] `tests/test_stats.cpp`: distributions (pdf/cdf/ppf round-trip) vs SciPy; summary stats; tests; KDE
- [x] CPU build green; full suite green
- [x] `openspec validate add-stats --strict` green
- [x] Check off Phase 7 in `bootstrap-scypp-foundation/tasks.md`; update README
