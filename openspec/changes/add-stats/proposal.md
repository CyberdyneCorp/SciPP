# Add stats (Phase 7)

## Why

Phase 7 of the ScyPP roadmap. `scipy.stats` is the statistics workhorse —
probability distributions, summary statistics, correlation/regression, hypothesis
tests, and density estimation. It builds directly on the special functions from
Phase 1 (the distribution CDFs are incomplete gamma/beta integrals) and on
`numpp` reductions.

`scipy.stats` is enormous, so this change delivers the **deterministic** core
(distribution `pdf`/`cdf`/`ppf`, summary statistics, the common parametric tests,
and Gaussian KDE) and defers the stochastic and specialized surface (sampling,
fitting, QMC, discrete distributions, and rank-based tests).

## What changes

Adds the **stats** capability — `scypp::stats`, validated against the SciPy oracle:

- **Special-function enablers** (internal): regularized incomplete gamma
  `gammainc`/`gammaincc` and incomplete beta `betainc`, plus their inverses —
  needed for the distribution CDFs/PPFs.
- **Continuous distributions**: `norm`, `expon`, `uniform`, `gamma`, `chi2`,
  `beta`, `t`, `f`, each exposing `pdf`/`logpdf`/`cdf`/`sf`/`ppf` and
  `mean`/`var`/`std`/`median`, with `loc`/`scale` (and shape parameters).
- **Summary statistics**: `gmean`, `hmean`, `describe`, `moment`, `skew`,
  `kurtosis`, `sem`, `variation`, `zscore`, `iqr`, `rankdata`, `mode`.
- **Correlation & regression**: `pearsonr`, `spearmanr`, `linregress`.
- **Hypothesis tests**: `ttest_1samp`, `ttest_ind`, `ttest_rel`, `f_oneway`,
  `ks_2samp`, `chi2_contingency`, `normaltest`.
- **Density estimation**: `gaussian_kde` (Scott/Silverman bandwidth, N-D).

## Impact

- Affected specs: **adds** the `stats` capability.
- Affected code: new `include/scypp/stats/`, `src/stats/`, `tests/test_stats.cpp`,
  extended oracle generator. Reuses Phase 1 special functions and Phase 2 linalg
  (KDE covariance).
- Roadmap: checks off Phase 7 in `bootstrap-scypp-foundation/tasks.md`.

## Non-goals (deferred to a later stats change)

- **Sampling and fitting**: distribution `rvs` and `fit` (need the RNG stream and
  MLE optimization) — a follow-up once a stats-RNG story is settled.
- **Discrete distributions**: `binom`, `poisson`, `geom`, `nbinom`,
  `hypergeom`, `bernoulli`.
- **Quasi-Monte-Carlo**: `qmc.Sobol`, `Halton`, `LatinHypercube`.
- **Rank-based / nonparametric tests**: `mannwhitneyu`, `wilcoxon`, `kruskal`,
  `kendalltau`, and the special null-distribution tests `shapiro`, `anderson`.
- **`binned_statistic`**, `theilslopes`, circular statistics, and the long tail of
  distributions beyond the eight above.
