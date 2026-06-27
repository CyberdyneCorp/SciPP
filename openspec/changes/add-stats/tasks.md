# Tasks — stats (Phase 7)

## 1. Module scaffold + special enablers
- [ ] `include/scypp/stats/stats.hpp` (distribution structs, result structs, function decls); `src/stats/*.cpp` in `src/CMakeLists.txt`; export from `scypp/scypp.hpp`
- [ ] Internal `gammainc`/`gammaincc`/`betainc` (series + continued fraction) and `gammaincinv`/`betaincinv` (Newton + bisection)

## 2. Continuous distributions
- [ ] `norm`, `expon`, `uniform` (closed form; norm via erf/erfinv)
- [ ] `gamma`, `chi2` (gammainc), `beta` (betainc), `t`, `f` (betainc)
- [ ] each: `pdf`/`logpdf`/`cdf`/`sf`/`ppf` + `mean`/`var`/`std`/`median`; ndarray `pdf`/`cdf`

## 3. Summary statistics
- [ ] `gmean`, `hmean`, `moment`, `skew`, `kurtosis`, `sem`, `variation`, `zscore`, `iqr`, `rankdata`, `mode`, `describe`

## 4. Correlation, regression, tests
- [ ] `pearsonr`, `spearmanr`, `linregress`
- [ ] `ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`, `ks_2samp`, `chi2_contingency`, `normaltest`

## 5. Density estimation
- [ ] `gaussian_kde` (Scott/Silverman bandwidth, covariance via numpp, N-D evaluate)

## 6. Oracle + validation
- [ ] Extend `tests/oracle/generate.py` with stats golden values; regenerate
- [ ] `tests/test_stats.cpp`: distributions (pdf/cdf/ppf round-trip) vs SciPy; summary stats; tests; KDE
- [ ] CPU build green; full suite green
- [ ] `openspec validate add-stats --strict` green
- [ ] Check off Phase 7 in `bootstrap-scypp-foundation/tasks.md`; update README
